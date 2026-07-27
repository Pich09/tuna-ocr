"""Pushes/pulls the packed+deduplicated dataset (see pack_arrow.py,
deduplicate.py) to/from a shared Hugging Face dataset repo, so the
pull -> pack -> dedup pipeline (a real network + CPU cost, potentially
hours at production scale) only has to run once -- later sessions/notebook
runs can just download the prebuilt result instead of repeating it."""
from pathlib import Path

from .config import HF_DATA_REPO_ID


def dataset_exists_on_hub(repo_id: str = HF_DATA_REPO_ID, token: str = None) -> bool:
    """Returns False for a missing/inaccessible repo rather than raising --
    this is used as a plain yes/no gate before deciding whether to pull
    from source or download from the hub."""
    from huggingface_hub import HfApi
    from huggingface_hub.utils import RepositoryNotFoundError

    try:
        HfApi(token=token).dataset_info(repo_id)
        return True
    except RepositoryNotFoundError:
        return False


def push_dataset(local_arrow_path: Path, token: str, repo_id: str = HF_DATA_REPO_ID, private: bool = True) -> str:
    """Uploads `local_arrow_path` (the deduplicated dataset, see
    deduplicate.py) to `repo_id` as dedup.arrow, creating the repo on first
    use (private by default -- opt-in, not a default, matching
    recognizer/hf_push.py's checkpoint-push convention). Also uploads the
    accompanying dedup.report.json if present, for provenance (dup counts,
    per-source kept counts). Returns the resulting hub URL."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=private)
    url = api.upload_file(
        path_or_fileobj=str(local_arrow_path),
        path_in_repo="dedup.arrow",
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
    )
    report_path = Path(local_arrow_path).with_suffix(".report.json")
    if report_path.exists():
        api.upload_file(
            path_or_fileobj=str(report_path),
            path_in_repo="dedup.report.json",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )
    return url


def pull_dataset(dest_path: Path, token: str = None, repo_id: str = HF_DATA_REPO_ID) -> Path:
    """Downloads dedup.arrow from `repo_id` to `dest_path`. Raises with the
    underlying huggingface_hub error on failure rather than swallowing it."""
    from huggingface_hub import hf_hub_download
    import shutil

    downloaded = hf_hub_download(repo_id=repo_id, filename="dedup.arrow", repo_type="dataset", token=token)
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if str(Path(downloaded).resolve()) != str(dest_path.resolve()):
        shutil.copy(downloaded, dest_path)
    return dest_path
