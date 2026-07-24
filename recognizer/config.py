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
    num_samples: int = 3
    num_workers: int = 8
    seed: int = 0
