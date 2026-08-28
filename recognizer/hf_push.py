"""Pushes/pulls checkpoint files to/from the shared Hugging Face model repo
-- Colab/Kaggle sessions are ephemeral (a disconnect wipes local disk, or
Kaggle's /kaggle/working doesn't persist across a fresh interactive
session), so the pushed checkpoints on the Hub are the only copy that
reliably survives a session restart. See pull_latest_checkpoint for
resuming from them.

Checkpoint scheme: three small files per run (optionally under a
`path_prefix` folder -- see push_checkpoint), not one step-numbered file
per ckpt_every -- the old scheme swamped the Hub repo with hundreds of
step_*.pt files over a long run.
    latest.pt      -- full training state as of the most recent checkpoint
                       (always overwritten in place; what a resume loads).
    best.pt        -- full training state as of the checkpoint with the
                       lowest tracked eval metric so far (overwritten only
                       when a new checkpoint beats it -- see run_training).
    metadata.json   -- tiny JSON with latest_step/best_step/best_metric/
                       last eval numbers; lets a resume recover best-so-far
                       bookkeeping (and anyone browsing the repo see
                       progress) without downloading a multi-hundred-MB .pt.

`pull_checkpoint_at_step`/the legacy `step_*.pt` scan in
`pull_latest_checkpoint` remain only so a repo that already has old
per-step checkpoints (pushed before this scheme existed) keeps resuming
correctly -- new pushes never create another one."""
import re
from pathlib import Path

from .config import HF_MODEL_REPO_ID

_STEP_RE = re.compile(r"^step_(\d+)\.pt$")


def push_checkpoint(local_ckpt_path: Path, token: str, repo_id: str = HF_MODEL_REPO_ID, private: bool = True,
                    path_prefix: str = "") -> str:
    """Uploads `local_ckpt_path` to `repo_id` under the same filename, creating
    the repo on first use (private by default -- this pushes trained model
    weights, so visibility is opt-in, not a default; pass private=False only
    if you've deliberately decided the repo should be public). Returns the
    resulting hub URL. Raises with the underlying huggingface_hub error
    message on failure rather than swallowing it -- a silently-failed
    checkpoint push is worse than a loud one.

    `path_prefix`: when set (e.g. "ctc_only"), uploads under that folder
    (`path_prefix/step_0002000.pt`) instead of the repo root -- lets several
    independent runs share ONE Hub repo without their step-numbered
    filenames colliding (see pull_latest_checkpoint's matching `path_prefix`).
    Empty string (default) uploads at the repo root, unchanged from before."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, exist_ok=True, private=private)
    filename = Path(local_ckpt_path).name
    path_in_repo = f"{path_prefix}/{filename}" if path_prefix else filename
    return api.upload_file(
        path_or_fileobj=str(local_ckpt_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        token=token,
    )


def pull_latest_checkpoint(dest_dir: Path, token: str = None, repo_id: str = HF_MODEL_REPO_ID, path_prefix: str = ""):
    """Downloads the checkpoint to resume from into `dest_dir` and returns
    its local Path -- or None if the repo doesn't exist yet or has nothing
    to resume under this prefix (a fresh run, not an error).

    Prefers `latest.pt` (the only file run_training's ckpt_every block
    pushes under this name going forward -- always the most recent training
    state). Falls back to scanning for the highest-numbered legacy
    `step_*.pt` file if `latest.pt` isn't there, so a repo that was last
    checkpointed under the old per-step scheme still resumes from its real
    latest progress instead of silently starting over."""
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    prefix = f"{path_prefix}/" if path_prefix else ""
    try:
        downloaded = hf_hub_download(repo_id=repo_id, filename=f"{prefix}latest.pt", token=token)
        out_name = "latest.pt"
    except (RepositoryNotFoundError, EntryNotFoundError):
        api = HfApi(token=token)
        try:
            files = api.list_repo_files(repo_id)
        except RepositoryNotFoundError:
            return None
        steps = [(int(m.group(1)), f) for f in files
                 if f.startswith(prefix) and (m := _STEP_RE.match(f[len(prefix):]))]
        if not steps:
            return None
        _, latest_file = max(steps)
        downloaded = hf_hub_download(repo_id=repo_id, filename=latest_file, token=token)
        out_name = Path(latest_file).name

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / out_name
    if str(Path(downloaded).resolve()) != str(dest_path.resolve()):
        import shutil
        shutil.copy(downloaded, dest_path)
    return dest_path


def pull_best_checkpoint(dest_dir: Path, token: str = None, repo_id: str = HF_MODEL_REPO_ID, path_prefix: str = ""):
    """Downloads `best.pt` (the checkpoint with the lowest tracked eval
    metric so far -- see run_training's ckpt_every block) into `dest_dir`
    and returns its local Path, or None if this repo/prefix has no best.pt
    yet (no eval-informed checkpoint has been pushed -- e.g. eval_every=0,
    or training hasn't reached its first eval yet)."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    filename = "best.pt"
    repo_filename = f"{path_prefix}/{filename}" if path_prefix else filename
    try:
        downloaded = hf_hub_download(repo_id=repo_id, filename=repo_filename, token=token)
    except (RepositoryNotFoundError, EntryNotFoundError):
        return None

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename
    if str(Path(downloaded).resolve()) != str(dest_path.resolve()):
        import shutil
        shutil.copy(downloaded, dest_path)
    return dest_path


def pull_metadata(token: str = None, repo_id: str = HF_MODEL_REPO_ID, path_prefix: str = ""):
    """Downloads and parses `metadata.json` (pushed alongside every
    checkpoint -- see run_training's ckpt_every block) if one exists, so a
    resumed run can recover its best_metric/best_step and keep comparing
    new checkpoints against the real best instead of resetting the
    comparison to "no best yet" and overwriting a genuinely better best.pt.
    Returns None if missing -- a fresh run, or a repo still on the legacy
    per-step scheme that predates metadata.json -- in which case the caller
    should just start best-tracking from scratch."""
    import json

    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    filename = "metadata.json"
    repo_filename = f"{path_prefix}/{filename}" if path_prefix else filename
    try:
        downloaded = hf_hub_download(repo_id=repo_id, filename=repo_filename, token=token)
    except (RepositoryNotFoundError, EntryNotFoundError):
        return None
    return json.loads(Path(downloaded).read_text(encoding="utf-8"))


def pull_checkpoint_at_step(dest_dir: Path, step: int, token: str = None, repo_id: str = HF_MODEL_REPO_ID,
                            path_prefix: str = ""):
    """Downloads the EXACT `step_{step:07d}.pt` checkpoint from `repo_id`
    (optionally scoped to `path_prefix`'s folder -- see push_checkpoint) into
    `dest_dir` and returns its local Path -- or None if that specific file
    doesn't exist. Unlike pull_latest_checkpoint (always the newest, for
    resuming a session that got interrupted), this is for deliberately
    rewinding: e.g. resuming from an earlier point in a training curriculum
    under a changed TrainConfig (a different sequential_ar_steps, say) rather
    than continuing forward from wherever a prior run happened to stop."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    filename = f"step_{step:07d}.pt"
    repo_filename = f"{path_prefix}/{filename}" if path_prefix else filename
    try:
        downloaded = hf_hub_download(repo_id=repo_id, filename=repo_filename, token=token)
    except (RepositoryNotFoundError, EntryNotFoundError):
        return None

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename
    if str(Path(downloaded).resolve()) != str(dest_path.resolve()):
        import shutil
        shutil.copy(downloaded, dest_path)
    return dest_path
