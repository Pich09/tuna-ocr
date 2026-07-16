"""Detects which platform training is running on (Colab / Kaggle / local) and
resolves the checkpoint root + HF token accordingly, so the same train.py
code/notebook works unmodified in all three places.
"""
import os
from pathlib import Path

from .config import DEFAULT_CHECKPOINT_ROOT

DRIVE_CHECKPOINT_ROOT = Path("/content/drive/My Drive/tuna-ocr/checkpoints")
KAGGLE_CHECKPOINT_ROOT = Path("/kaggle/working/tuna-ocr/checkpoints")


def detect_environment() -> str:
    """Returns 'colab', 'kaggle', or 'local'."""
    if "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ:
        return "colab"
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or Path("/kaggle/working").is_dir():
        return "kaggle"
    return "local"


def get_checkpoint_root(env: str = None) -> Path:
    """Mounts Google Drive on Colab (no-op if already mounted) and returns the
    environment-appropriate checkpoint directory."""
    env = env or detect_environment()
    if env == "colab":
        if not Path("/content/drive").is_dir():
            from google.colab import drive  # noqa: PLC0415

            drive.mount("/content/drive")
        DRIVE_CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
        return DRIVE_CHECKPOINT_ROOT
    if env == "kaggle":
        KAGGLE_CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
        return KAGGLE_CHECKPOINT_ROOT
    DEFAULT_CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    return DEFAULT_CHECKPOINT_ROOT


def get_hf_token(env: str = None) -> str:
    """Reads the HF_TOKEN secret from the current platform's secret store.
    Never hardcode a token in the notebook -- raises a clear error naming
    where to add it if not found, rather than silently training without push."""
    env = env or detect_environment()
    if env == "colab":
        from google.colab import userdata  # noqa: PLC0415

        try:
            return userdata.get("HF_TOKEN")
        except Exception as exc:  # userdata.SecretNotFoundError, notebook-access errors
            raise RuntimeError(
                "HF_TOKEN not found in Colab secrets. Add it via the key icon in the "
                "left sidebar (Secrets) and grant this notebook access."
            ) from exc
    if env == "kaggle":
        from kaggle_secrets import UserSecretsClient  # noqa: PLC0415

        try:
            return UserSecretsClient().get_secret("HF_TOKEN")
        except Exception as exc:
            raise RuntimeError(
                "HF_TOKEN not found in Kaggle secrets. Add it via Add-ons > Secrets "
                "in the notebook editor."
            ) from exc
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN environment variable not set. Export it before running training, "
            "e.g. `export HF_TOKEN=hf_...`."
        )
    return token
