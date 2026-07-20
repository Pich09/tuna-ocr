"""Training loop for the Recognizer (Conformer encoder + blockwise-AR
decoder), over real_data-only sources (see recognizer/config.py /
recognizer/data/manifest.py -- data_gen synthetic output is out of scope).

CLI:
    python -m recognizer.train --real-data-dirs real_data/samples/deepcopy_khmer_text_recognition \\
        real_data/samples/chanrith_ocr_image_line real_data/samples/darayut_scene_text \\
        real_data/samples/sokheng_synthetic_v1 \\
        --tokenizer-dir recognizer/tokenizer/assets --run-name v1

Runs on CPU, CUDA, or TPU -- the device is auto-detected (see env_utils.py)
unless explicitly passed. TPU note: PyTorch/XLA's CTC op support has
historically been spotty across versions; if `compute_loss`'s CTCLoss call
errors or silently falls back to a slow CPU path on your TPU runtime,
that's a torch_xla limitation, not a bug in this file -- report the exact
error rather than assuming it's this code.

Also exposes `run_training(...)` as a plain callable so the Colab/Kaggle/
local notebook (notebooks/train_recognizer.ipynb) can drive training
directly instead of shelling out to this CLI.
"""
import argparse
import csv
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import env_utils
from .config import ModelConfig, TrainConfig, DEFAULT_CHECKPOINT_ROOT, TOKENIZER_ASSETS_DIR
from .data.dataset import BucketBatchSampler, OCRLineDataset, make_collate_fn
from .data.manifest import build_combined_index
from .hf_push import push_checkpoint
from .modules.model import Recognizer
from .tokenizer.khmer_ocr_tokenizer import KhmerOcrTokenizer


def find_max_batch_size(model, dataset, collate_fn, device, start: int = 64, min_batch: int = 1) -> int:
    """OOM-probing auto-tune: try `start`, halve on CUDA OOM until a batch
    fits (forward + backward). CPU/TPU just use the configured default --
    on TPU, memory errors don't surface as a catchable Python OOM the same
    way (XLA compiles lazily), so this probe isn't meaningful there."""
    if device.type != "cuda":
        return start
    bs = start
    while bs >= min_batch:
        try:
            idxs = [i % len(dataset) for i in range(bs)]
            batch = collate_fn([dataset[i] for i in idxs])
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            ctc_logits, ar_logits, _ = model(batch["chunks"], batch["chunks_per_line"], batch["ar_targets"])
            (ctc_logits.float().sum() + ar_logits.float().sum()).backward()
            model.zero_grad(set_to_none=True)
            del ctc_logits, ar_logits
            torch.cuda.empty_cache()
            return bs
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            bs //= 2
    return max(min_batch, 1)


def lr_lambda(step: int, warmup_steps: int, max_steps: int) -> float:
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def compute_loss(ctc_logits, ar_logits, batch, ctc_weight: float, pad_id: int, ctc_blank_id: int):
    b, t_max, _ = ctc_logits.shape
    log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)  # (T,N,C)
    ctc_input_lengths = torch.full((b,), t_max, dtype=torch.long, device=ctc_logits.device)
    # CTCLoss expects the blank id to match the logits' extra class; ctc_head
    # appends it at index vocab_size, so remap that here explicitly.
    ctc_loss = F.ctc_loss(
        log_probs, batch["ctc_targets"], ctc_input_lengths, batch["ctc_lengths"],
        blank=ctc_blank_id, zero_infinity=True,
    )
    ce_loss = F.cross_entropy(
        ar_logits.reshape(-1, ar_logits.shape[-1]), batch["ar_targets"].reshape(-1),
        ignore_index=pad_id,
    )
    loss = ctc_weight * ctc_loss + (1 - ctc_weight) * ce_loss
    return loss, ctc_loss, ce_loss


def run_training(model_cfg: ModelConfig, train_cfg: TrainConfig, real_data_roots, tokenizer_dir=TOKENIZER_ASSETS_DIR,
                  checkpoint_root=DEFAULT_CHECKPOINT_ROOT, run_name: str = "v1", push_to_hub: bool = False,
                  repo_id=None, hf_token=None, hub_private: bool = True, auto_batch_size: bool = False,
                  resume_path=None, device=None):
    device = device or env_utils.get_torch_device()
    is_xla = device.type == "xla"
    xm = None
    if is_xla:
        import torch_xla.core.xla_model as xm  # noqa: PLC0415
    torch.manual_seed(train_cfg.seed)

    tokenizer = KhmerOcrTokenizer(tokenizer_dir)
    samples = build_combined_index(real_data_roots)
    if not samples:
        raise RuntimeError(f"No is_full_line samples found under {real_data_roots} -- generate them first with "
                            f"`python -m real_data.generate_external_chunks`.")
    n_val = max(1, int(len(samples) * train_cfg.val_frac))
    train_samples, val_samples = samples[n_val:], samples[:n_val]

    train_ds = OCRLineDataset(train_samples, tokenizer, model_cfg)
    collate_fn = make_collate_fn(tokenizer, model_cfg.chunk_width)

    model = Recognizer(model_cfg, tokenizer.vocab_size, bos_id=tokenizer.bos_id, pad_id=tokenizer.pad_id).to(device)

    batch_size = train_cfg.batch_size
    if auto_batch_size:
        batch_size = find_max_batch_size(model, train_ds, collate_fn, device, start=max(64, train_cfg.batch_size))
        print(f"auto batch size: {batch_size}")

    sampler = BucketBatchSampler(train_ds, batch_size=batch_size, shuffle=True)
    loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_lambda(step, train_cfg.warmup_steps, train_cfg.max_steps),
    )

    ckpt_dir = Path(checkpoint_root) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "train_log.csv"
    if not log_path.exists():
        log_path.write_text("step,loss,ctc_loss,ce_loss,lr\n")

    step = 0
    if resume_path:
        # weights_only=False: this checkpoint is self-produced (stores our own
        # ModelConfig dataclass alongside the state dicts), not an untrusted
        # third-party file -- PyTorch 2.6+'s weights_only=True default would
        # otherwise reject the ModelConfig global. map_location="cpu": XLA
        # devices aren't reliably understood by map_location -- load_state_dict
        # onto the already-.to(device)'d model handles the actual placement.
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        step = state["step"]
        print(f"resumed from {resume_path} at step {step}")

    ctc_blank_id = tokenizer.vocab_size
    model.train()
    t0 = time.time()
    while step < train_cfg.max_steps:
        for batch in loader:
            if step >= train_cfg.max_steps:
                break
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            ctc_logits, ar_logits, _ = model(batch["chunks"], batch["chunks_per_line"], batch["ar_targets"])
            loss, ctc_loss, ce_loss = compute_loss(
                ctc_logits, ar_logits, batch, train_cfg.ctc_weight, tokenizer.pad_id, ctc_blank_id,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if is_xla:
                xm.optimizer_step(optimizer)  # reduces gradients across cores + steps + marks the XLA graph
            else:
                optimizer.step()
            scheduler.step()
            step += 1

            if step % train_cfg.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                print(f"step {step} loss {loss.item():.4f} ctc {ctc_loss.item():.4f} "
                      f"ce {ce_loss.item():.4f} lr {lr:.2e} ({elapsed:.1f}s)")
                with log_path.open("a", newline="") as f:
                    csv.writer(f).writerow([step, loss.item(), ctc_loss.item(), ce_loss.item(), lr])

            if step % train_cfg.ckpt_every == 0:
                ckpt_path = ckpt_dir / f"step_{step:07d}.pt"
                state = {
                    "step": step, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(), "model_cfg": model_cfg,
                }
                save_fn = xm.save if is_xla else torch.save  # xm.save moves XLA tensors to CPU before writing
                save_fn(state, ckpt_path)
                save_fn(state, ckpt_dir / "last.pt")
                print(f"saved checkpoint {ckpt_path}")
                if push_to_hub:
                    kwargs = {"token": hf_token, "private": hub_private}
                    if repo_id:
                        kwargs["repo_id"] = repo_id
                    url = push_checkpoint(ckpt_path, **kwargs)
                    print(f"pushed checkpoint to {url}")

    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-data-dirs", nargs="+", required=True)
    parser.add_argument("--tokenizer-dir", type=Path, default=TOKENIZER_ASSETS_DIR)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--run-name", default="v1")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--auto-batch-size", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--ckpt-every", type=int, default=None)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-public", action="store_true",
                         help="Make the pushed HF checkpoint repo public (default: private).")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model_cfg = ModelConfig()
    train_cfg = TrainConfig(seed=args.seed)
    if args.batch_size:
        train_cfg.batch_size = args.batch_size
    if args.max_steps:
        train_cfg.max_steps = args.max_steps
    if args.log_every:
        train_cfg.log_every = args.log_every
    if args.ckpt_every:
        train_cfg.ckpt_every = args.ckpt_every

    run_training(
        model_cfg, train_cfg, args.real_data_dirs, tokenizer_dir=args.tokenizer_dir,
        checkpoint_root=args.checkpoint_root, run_name=args.run_name, push_to_hub=args.push_to_hub,
        hf_token=args.hf_token, hub_private=not args.hub_public, auto_batch_size=args.auto_batch_size,
        resume_path=args.resume,
    )


if __name__ == "__main__":
    main()
