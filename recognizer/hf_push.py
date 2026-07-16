"""Pushes a local checkpoint file to the shared Hugging Face model repo."""
from pathlib import Path

from .config import HF_MODEL_REPO_ID


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
