"""Pushes/pulls checkpoint files to/from the shared Hugging Face model repo
-- Colab/Kaggle sessions are ephemeral (a disconnect wipes local disk, or
Kaggle's /kaggle/working doesn't persist across a fresh interactive
session), so the pushed checkpoints on the Hub are the only copy that
reliably survives a session restart. See pull_latest_checkpoint for
resuming from them."""
import re
from pathlib import Path

from .config import HF_MODEL_REPO_ID

_STEP_RE = re.compile(r"^step_(\d+)\.pt$")


def push_checkpoint(local_ckpt_path: Path, token: str, repo_id: str = HF_MODEL_REPO_ID, private: bool = True) -> str:
    """Uploads `local_ckpt_path` to `repo_id` under the same filename, creating
    the repo on first use (private by default -- this pushes trained model
    weights, so visibility is opt-in, not a default; pass private=False only
    if you've deliberately decided the repo should be public). Returns the
    resulting hub URL. Raises with the underlying huggingface_hub error
    message on failure rather than swallowing it -- a silently-failed
    checkpoint push is worse than a loud one."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, exist_ok=True, private=private)
    return api.upload_file(
        path_or_fileobj=str(local_ckpt_path),
        path_in_repo=Path(local_ckpt_path).name,
        repo_id=repo_id,
        token=token,
    )


def pull_latest_checkpoint(dest_dir: Path, token: str = None, repo_id: str = HF_MODEL_REPO_ID):
    """Downloads the highest-step `step_*.pt` checkpoint from `repo_id` into
    `dest_dir` and returns its local Path -- or None if the repo doesn't
    exist yet or has no checkpoints (a fresh run, nothing to resume from;
    not an error). Only `step_*.pt` files are considered (not `last.pt`,
    which push_checkpoint never uploads -- see run_training's ckpt_every
    block) so this always finds a real, fully-uploaded checkpoint rather
    than depending on filename conventions that aren't actually pushed."""
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import RepositoryNotFoundError

    api = HfApi(token=token)
    try:
        files = api.list_repo_files(repo_id)
    except RepositoryNotFoundError:
        return None

    steps = [(int(m.group(1)), f) for f in files if (m := _STEP_RE.match(f))]
    if not steps:
        return None
    _, latest_file = max(steps)

    downloaded = hf_hub_download(repo_id=repo_id, filename=latest_file, token=token)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / latest_file
    if str(Path(downloaded).resolve()) != str(dest_path.resolve()):
        import shutil
        shutil.copy(downloaded, dest_path)
    return dest_path
