"""Detects which platform training is running on (Colab / Kaggle / local) and
which accelerator is available (TPU / GPU / CPU), and resolves the
checkpoint root + HF token accordingly, so the same train.py code/notebook
works unmodified everywhere.
"""
import os
from pathlib import Path

from .config import DEFAULT_CHECKPOINT_ROOT

DRIVE_CHECKPOINT_ROOT = Path("/content/drive/My Drive/tuna-ocr/checkpoints")
KAGGLE_CHECKPOINT_ROOT = Path("/kaggle/working/tuna-ocr/checkpoints")

# Any of these being set is how Colab/Kaggle/GCP advertise a TPU runtime --
# checked before ever importing torch_xla, since importing it with no TPU
# present is either an error or an expensive no-op.
_TPU_ENV_VARS = ("COLAB_TPU_ADDR", "TPU_NAME", "XRT_TPU_CONFIG")


def detect_environment() -> str:
    """Returns 'colab', 'kaggle', or 'local'.

    Kaggle is checked first: some Kaggle kernels leak a stray COLAB_GPU/
    COLAB_RELEASE_TAG env var (likely shared base-image heritage with
    Colab's TPU tooling), which would otherwise cause a Kaggle notebook to
    misdetect as Colab and crash trying to mount Google Drive. The
    "/kaggle/working" directory is a much harder signal to spoof than an
    env var, so it takes priority.
    """
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or Path("/kaggle/working").is_dir():
        return "kaggle"
    if "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ or "COLAB_TPU_ADDR" in os.environ:
        return "colab"
    return "local"


def detect_accelerator() -> str:
    """Returns 'tpu', 'cuda', or 'cpu'. TPU is checked first since a TPU
    runtime has no CUDA device to report."""
    if any(v in os.environ for v in _TPU_ENV_VARS):
        return "tpu"
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def get_torch_device():
    """Returns the torch.device to train on, importing torch_xla only when a
    TPU was actually detected. Raises with an actionable message if a TPU is
    advertised but torch_xla isn't installed -- Colab/Kaggle TPU runtimes
    ship it preinstalled, but a custom environment might not."""
    accel = detect_accelerator()
    if accel == "tpu":
        try:
            import torch_xla.core.xla_model as xm  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "A TPU runtime was detected (TPU env var present) but torch_xla isn't "
                "installed. On Colab/Kaggle, select the TPU runtime/accelerator (which "
                "ships torch_xla preinstalled) rather than pip-installing it manually."
            ) from exc
        return xm.xla_device()
    import torch

    return torch.device("cuda" if accel == "cuda" else "cpu")


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
