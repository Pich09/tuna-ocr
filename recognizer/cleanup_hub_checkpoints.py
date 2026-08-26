"""One-off maintenance: prune old step_*.pt checkpoints from a Hub run repo,
keeping only the ones actually worth keeping -- run this from wherever
HF_TOKEN already lives (Colab/Kaggle secrets, or an HF_TOKEN env var), NOT
from an automated training cell. See env_utils.get_hf_token for the token
resolution chain.

Keeps up to three checkpoints, matched against real eval numbers in the
local eval_log.csv (written next to the checkpoints by train.py's
run_training) so "best" isn't a guess:
  - the single latest step_*.pt (needed to resume training)
  - the best (lowest val_ar_cer) checkpoint from the sequential-AR phase
    (step < --sequential-ar-steps), so sequential mode can still be tested
  - the best (lowest val_ar_cer) checkpoint from the blockwise phase
    (step >= --sequential-ar-steps)
A step only counts as a candidate if a matching step_*.pt actually exists
on the Hub (eval_every is usually finer-grained than ckpt_every, so most
eval rows have no checkpoint to match).

Defaults to a dry run -- prints what WOULD be deleted. Pass --confirm to
actually delete.

Usage (from the Kaggle/Colab session that already has the checkpoints + HF_TOKEN):
    python -m recognizer.cleanup_hub_checkpoints \
        --repo-id Panhapich/tuna-ocr-v2-scratch-k16 \
        --eval-log /kaggle/working/tuna-ocr/checkpoints/v2_scratch_k16/eval_log.csv \
        --sequential-ar-steps 60000
    # review the printed plan, then re-run with --confirm to actually delete
"""
import argparse
import csv
import re
from pathlib import Path

from .env_utils import get_hf_token

_STEP_RE = re.compile(r"^step_(\d+)\.pt$")


def _read_eval_log(path: Path) -> list[tuple[int, float]]:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append((int(row["step"]), float(row["val_ar_cer"])))
    return rows


def plan_deletion(hub_steps: set[int], eval_rows: list[tuple[int, float]], sequential_ar_steps: int):
    """Returns (keep_steps, keep_reasons) -- keep_reasons maps step -> why it's kept."""
    candidates = [(step, cer) for step, cer in eval_rows if step in hub_steps]
    seq = [(s, c) for s, c in candidates if s < sequential_ar_steps]
    block = [(s, c) for s, c in candidates if s >= sequential_ar_steps]

    keep_reasons = {}
    if hub_steps:
        latest = max(hub_steps)
        keep_reasons[latest] = keep_reasons.get(latest, []) + ["latest (resume point)"]
    if seq:
        best_seq_step, best_seq_cer = min(seq, key=lambda r: r[1])
        keep_reasons[best_seq_step] = keep_reasons.get(best_seq_step, []) + [f"best sequential (val_ar_cer={best_seq_cer:.4f})"]
    if block:
        best_block_step, best_block_cer = min(block, key=lambda r: r[1])
        keep_reasons[best_block_step] = keep_reasons.get(best_block_step, []) + [f"best blockwise (val_ar_cer={best_block_cer:.4f})"]

    return keep_reasons


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--eval-log", required=True, type=Path, help="local eval_log.csv next to the checkpoints")
    parser.add_argument("--sequential-ar-steps", type=int, default=60_000)
    parser.add_argument("--confirm", action="store_true", help="actually delete; omit for a dry run")
    args = parser.parse_args()

    from huggingface_hub import HfApi

    token = get_hf_token()
    api = HfApi(token=token)
    files = api.list_repo_files(args.repo_id)
    hub_steps = {int(m.group(1)): f for f in files if (m := _STEP_RE.match(f))}
    if not hub_steps:
        print(f"no step_*.pt files found in {args.repo_id} -- nothing to do")
        return

    eval_rows = _read_eval_log(args.eval_log)
    keep_reasons = plan_deletion(set(hub_steps), eval_rows, args.sequential_ar_steps)

    to_delete = sorted(s for s in hub_steps if s not in keep_reasons)
    to_keep = sorted(keep_reasons)

    print(f"{args.repo_id}: {len(hub_steps)} checkpoints on the Hub")
    print(f"\nKEEP ({len(to_keep)}):")
    for s in to_keep:
        print(f"  step_{s:07d}.pt -- {', '.join(keep_reasons[s])}")
    print(f"\nDELETE ({len(to_delete)}):")
    for s in to_delete:
        print(f"  step_{s:07d}.pt")

    if not args.confirm:
        print("\nDry run -- nothing deleted. Re-run with --confirm to actually delete the DELETE list above.")
        return

    for s in to_delete:
        api.delete_file(path_in_repo=hub_steps[s], repo_id=args.repo_id, token=token)
        print(f"deleted step_{s:07d}.pt")
    print(f"\ndone -- {len(to_delete)} checkpoints removed, {len(to_keep)} kept.")


if __name__ == "__main__":
    main()
