"""Central config for the OCR recognizer (Conformer encoder + blockwise-AR decoder)."""
from dataclasses import dataclass
from pathlib import Path

RECOGNIZER_ROOT = Path(__file__).resolve().parent
TOKENIZER_ASSETS_DIR = RECOGNIZER_ROOT / "tokenizer" / "assets"
DEFAULT_CHECKPOINT_ROOT = RECOGNIZER_ROOT / "checkpoints"

TOKENIZER_HUB_ID = "Panhapich/khmer-sp-8k"
HF_MODEL_REPO_ID = "Panhapich/tuna-ocr"

# Language-tag ids appended past the base 8000-token SentencePiece vocab.
# <blk> is intentionally not here -- see KhmerOcrTokenizer's <eob>.
LANG_TOKENS = ["<km>", "<en>", "<fr>"]
EXTRA_CONTROL_TOKENS = ["<eob>"]  # end-of-block; base vocab already has PAD/BOS/EOS/UNK/MASK


@dataclass
class ModelConfig:
    img_height: int = 64          # matches data_gen.config.LineRenderConfig.img_height
    chunk_width: int = 128        # encoder's literal input unit width, in pixels
    chunk_overlap: int = 16       # backward overlap/left-context per chunk, in pixels
    d_model: int = 256
    num_encoder_layers: int = 8
    encoder_attn_heads: int = 4
    encoder_conv_kernel: int = 15
    encoder_ff_expansion: int = 4
    encoder_dropout: float = 0.1
    num_decoder_layers: int = 4
    decoder_attn_heads: int = 4
    max_tokens_per_block: int = 8


@dataclass
class TrainConfig:
    batch_size: int = 32
    lr: float = 5e-4
    min_lr: float = 1e-6
    warmup_steps: int = 4000
    max_steps: int = 200_000
    ctc_weight: float = 0.3
    val_frac: float = 0.02
    log_every: int = 100
    ckpt_every: int = 10_000
    sample_every: int = 500
    eval_every: int = 500  # held-out val CER (AR + CTC) every N steps; 0 disables
    # Cap on how many val samples the PERIODIC eval decodes. evaluate_val_cer
    # decodes one sample at a time, and in sequential-AR mode emits one token
    # per forward pass -- measured at 1072 ms/sample on CUDA, i.e. 153 minutes
    # for the full 8578-sample val set of the production dataset, every
    # eval_every steps. That is not a metric, it is a stall (and it is what an
    # earlier run's "stuck for hours at step 1000" turned out to be). A few
    # hundred samples give a perfectly usable CER estimate; set 0 for no cap.
    max_eval_samples: int = 512
    # Separate, much smaller cap on the AR greedy decode inside that eval. The
    # CTC read-out is an argmax over an encoder pass the AR decode already
    # needed, so it is nearly free across all max_eval_samples; the AR decode
    # is not, and in sequential mode is ~40x slower still. Measured on a real
    # run: 512 AR decodes = 639s (10.7 min) every 500 steps, against 265s of
    # actual training -- 71% of wall clock. 64 keeps the AR signal without
    # letting it own the run. 0 disables the AR decode during periodic eval
    # entirely (CTC CER only); None means no cap.
    max_ar_eval_samples: int = 64
    # Drop samples whose CTC target is longer than the encoder frames their
    # image can produce. CTC cannot emit more labels than it has frames, so the
    # loss is +inf, and compute_loss's zero_infinity=True silently turns that
    # into 0 -- the sample contributes no gradient at all, forever, while still
    # costing a forward pass. Measured on the production dataset: 5.3% overall,
    # 23.5% of sokheng_synthetic_v1 and 0.0% of every other source. Set False
    # (--keep-unlearnable) to train on them anyway.
    filter_unlearnable: bool = True
    num_samples: int = 3
    num_workers: int = 8
    seed: int = 0
    # Initial steps trained with the decoder's plain sequential-AR mode (full
    # teacher forcing over the true token sequence, unrestricted cross-attention)
    # before switching to blockwise mode for the rest of the run -- see
    # modules/decoder.py's "Two-stage training" docstring section. 0 (default)
    # disables this and trains blockwise from step 0, matching prior behavior.
    sequential_ar_steps: int = 0
