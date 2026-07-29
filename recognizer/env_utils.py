"""Detects which platform training is running on (Colab / Kaggle / local) and
which accelerator is available (TPU / GPU / CPU), and resolves the
checkpoint root + HF token accordingly, so the same train.py code/notebook
works unmodified everywhere.
"""
import glob
import importlib.util
import os
from pathlib import Path

from .config import DEFAULT_CHECKPOINT_ROOT

DRIVE_CHECKPOINT_ROOT = Path("/content/drive/My Drive/tuna-ocr/checkpoints")
KAGGLE_CHECKPOINT_ROOT = Path("/kaggle/working/tuna-ocr/checkpoints")

# How Colab/Kaggle/GCP advertise a TPU runtime, checked before ever importing
# torch_xla (importing it with no TPU present is either an error or an
# expensive no-op). Two generations are covered on purpose:
#   * XRT era (older Colab TPU runtimes): COLAB_TPU_ADDR / XRT_TPU_CONFIG.
#   * PJRT era (every current Colab/Kaggle TPU runtime, torch_xla >= 2.0):
#     PJRT_DEVICE=TPU plus the TPU_* topology vars the libtpu client reads.
# Checking only the XRT-era vars -- as this did originally -- silently
# misdetects a modern TPU runtime as CPU: none of them are set anymore, so
# detection fell through to torch.cuda.is_available() (False on a TPU VM) and
# training ran on CPU at full "it works, just 100x too slow" plausibility.
_TPU_ENV_VARS = (
    "COLAB_TPU_ADDR",
    "XRT_TPU_CONFIG",
    "TPU_NAME",
    "TPU_ACCELERATOR_TYPE",
    "TPU_WORKER_ID",
    "TPU_CHIPS_PER_HOST_BOUNDS",
    "TPU_PROCESS_ADDRESSES",
)


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


def _tpu_present() -> bool:
    """True if this machine looks like a TPU VM, using progressively more
    expensive checks and stopping at the first positive.

    1. `PJRT_DEVICE` -- compared by *value*, not mere presence: PJRT_DEVICE is
       also legitimately set to "CUDA"/"CPU", so `"PJRT_DEVICE" in os.environ`
       would report a TPU on a GPU runtime.
    2. Any of the env vars above.
    3. `/dev/accel*` -- the TPU chips' device nodes. A filesystem fact rather
       than an env var, so it still holds in a subprocess/venv that inherited
       a scrubbed environment.
    4. Ask torch_xla itself, but only if it's actually installed (find_spec,
       so this never raises ImportError on the common no-torch_xla machine).
       Kept last because importing torch_xla initializes its runtime, which is
       far from free.
    """
    if os.environ.get("PJRT_DEVICE", "").upper() == "TPU":
        return True
    if any(v in os.environ for v in _TPU_ENV_VARS):
        return True
    if glob.glob("/dev/accel*"):
        return True
    if importlib.util.find_spec("torch_xla") is None:
        return False
    try:
        import torch_xla.runtime as xr  # noqa: PLC0415

        return xr.device_type() == "TPU"
    except Exception:
        # torch_xla installed but its runtime won't come up (no TPU attached,
        # version too old to have torch_xla.runtime, libtpu mismatch). Not a
        # TPU as far as this process is concerned -- fall through to CUDA/CPU
        # rather than dying here on a machine that trains fine without it.
        return False


def detect_accelerator() -> str:
    """Returns 'tpu', 'cuda', or 'cpu'.

    CUDA is checked first because it's a cheap, unambiguous query and no
    Colab/Kaggle runtime offers both -- so a positive here means the user
    picked the GPU accelerator and there is nothing to disambiguate. Only
    when there's no CUDA device do we pay for the TPU probe, whose last
    resort imports torch_xla.
    """
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "tpu" if _tpu_present() else "cpu"


def get_torch_device():
    """Returns the torch.device to train on, importing torch_xla only when a
    TPU was actually detected. Raises with an actionable message if a TPU is
    advertised but torch_xla isn't installed -- Colab/Kaggle TPU runtimes
    ship it preinstalled, but a custom environment might not."""
    accel = detect_accelerator()
    if accel == "tpu":
        # torch_xla reads PJRT_DEVICE at import time to pick its backend; a TPU
        # detected via /dev/accel* or a topology var alone may not have it set
        # (e.g. a subprocess with a trimmed environment), in which case
        # torch_xla would quietly initialize a CPU backend instead of the TPU.
        os.environ.setdefault("PJRT_DEVICE", "TPU")
        try:
            import torch_xla  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "A TPU was detected but torch_xla isn't installed. On Colab/Kaggle, "
                "select the TPU runtime/accelerator (which ships torch_xla "
                "preinstalled) rather than pip-installing it manually."
            ) from exc
        # torch_xla.device() is the current entry point; xm.xla_device() is the
        # pre-2.5 spelling, deprecated but still what older preinstalled
        # runtimes ship. Try the new one, fall back rather than assume either.
        if hasattr(torch_xla, "device"):
            return torch_xla.device()
        import torch_xla.core.xla_model as xm  # noqa: PLC0415

        return xm.xla_device()
    import torch

    return torch.device("cuda" if accel == "cuda" else "cpu")


def describe_accelerator() -> str:
    """One human-readable line naming the device actually selected -- printed
    by the notebook before a multi-hour run starts, so "it silently fell back
    to CPU" is visible in the first seconds instead of inferred later from
    step timings."""
    accel = detect_accelerator()
    if accel == "cuda":
        import torch  # noqa: PLC0415

        name = torch.cuda.get_device_name(0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"cuda: {name} ({total_gb:.1f} GB)"
    if accel == "tpu":
        try:
            import torch_xla.runtime as xr  # noqa: PLC0415

            return f"tpu: {xr.device_type()}, {xr.global_runtime_device_count()} device(s) visible"
        except Exception:
            return "tpu (torch_xla runtime details unavailable)"
    return "cpu (no GPU/TPU detected -- training will be impractically slow at this scale)"


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
