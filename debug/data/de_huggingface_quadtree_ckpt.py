import os
import zipfile
from pathlib import Path

from huggingface_hub import HfApi


# ``SEARCH_ROOT`` is the directory that contains the local ``outputs`` folder.
# Change it if the quadtree checkout is stored elsewhere.
SEARCH_ROOT = Path.home() / "code/dp23rss_fork"
OUTPUTS_ROOT = SEARCH_ROOT / "outputs"
DEFAULT_REPO_ID = "ygtxr1997/tcl_models"
DEFAULT_REPO_TYPE = "model"

CHECKPOINT_NAMES = (
    # "epoch=0750-val_diffusion_loss=0.005532.ckpt",
    # "epoch=0800-val_diffusion_loss=0.005964.ckpt",
    # "epoch=0850-val_diffusion_loss=0.006221.ckpt",
    # "epoch=0900-val_diffusion_loss=0.005732.ckpt",
    # "epoch=0950-val_diffusion_loss=0.005053.ckpt",
    "latest.ckpt",
)


def main() -> None:
    """Create and upload the configured quadtree checkpoint archives."""
    log_times = [
        # "2026-11-02-23-35-12",
        "2026-11-09-10-03-53",
    ]

    for log_time in log_times:
        compress_and_upload_checkpoint(
            log_time,
            repo_id=DEFAULT_REPO_ID,
            repo_type=DEFAULT_REPO_TYPE,
            force=True,
        )


def parse_log_time(log_time: str) -> tuple[str, str]:
    """Parse a timestamp into its date and time components."""
    value = log_time.strip().strip("/")
    if value.startswith("outputs/"):
        value = value[len("outputs/") :]
    parts = value.split("-")
    if len(parts) != 6 or len(parts[0]) != 4 or len(parts[1]) != 2 or len(parts[2]) != 2:
        raise ValueError(
            f"Invalid log_time {log_time!r}; expected YYYY-MM-DD-HH-MM-SS"
        )
    date_part = "-".join(parts[:3])
    time_part = "-".join(parts[3:])
    return date_part, time_part


def _iter_run_dirs(root: Path, max_depth: int = 2):
    """Yield directories at most ``max_depth`` levels below ``root``."""
    if not root.is_dir():
        return

    def visit(parent: Path, depth: int):
        if depth > max_depth:
            return
        for path in sorted(parent.iterdir()):
            if not path.is_dir():
                continue
            yield path
            if depth < max_depth:
                yield from visit(path, depth + 1)

    yield from visit(root, 1)


def resolve_run_dir(log_time: str, outputs_root: Path = OUTPUTS_ROOT) -> Path:
    """Find a run below any outputs project directory, at most two levels deep."""
    date_part, time_part = parse_log_time(log_time)
    exact_names = {log_time.strip().strip("/"), f"{date_part}-{time_part}"}
    candidates = []
    # Search the outputs root itself and every project directory (dp_tcl,
    # dp_pretrain, dp_ccc, ...). A date directory is only inspected one level
    # further, so a matching first-level date never causes an unbounded walk.
    search_roots = [outputs_root]
    if outputs_root.is_dir():
        search_roots.extend(path for path in sorted(outputs_root.iterdir()) if path.is_dir())
    for search_root in search_roots:
        for path in _iter_run_dirs(search_root, max_depth=2):
            if path.name not in exact_names and path.name != date_part and not path.name.startswith(time_part):
                continue
            if (path / "dataset_metadata.json").is_file():
                candidates.append(path)

    candidates = sorted(set(candidates))
    if not candidates:
        raise FileNotFoundError(
            f"No run matching {log_time!r} found below {outputs_root}; "
            "searched at most two directory levels"
        )
    if len(candidates) > 1:
        match_list = "\n".join(f"  - {path}" for path in candidates)
        raise RuntimeError(f"Multiple runs match {log_time!r}:\n{match_list}")
    return candidates[0]


def validate_required_paths(run_dir: Path) -> dict[str, Path]:
    """Validate and return all files included in a quadtree archive."""
    paths = {
        "dataset_metadata.json": run_dir / "dataset_metadata.json",
        "logs.json.txt": run_dir / "logs.json.txt",
        ".hydra": run_dir / ".hydra",
        "checkpoints": run_dir / "checkpoints",
    }
    missing = [path for key, path in paths.items() if not path.exists()]
    missing.extend(
        paths["checkpoints"] / name
        for name in CHECKPOINT_NAMES
        if not (paths["checkpoints"] / name).is_file()
    )
    if not paths[".hydra"].is_dir():
        missing.append(paths[".hydra"])
    if missing:
        missing_list = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Required quadtree files are missing:\n{missing_list}")
    return paths


def add_directory_to_zip(zipf: zipfile.ZipFile, source_dir: Path, arc_dir: Path) -> None:
    """Add a directory tree while retaining empty directories."""
    zipf.write(source_dir, arc_dir)
    for path in sorted(source_dir.rglob("*")):
        zipf.write(path, arc_dir / path.relative_to(source_dir))


def create_checkpoint_zip(log_time: str, force: bool = True) -> Path:
    """Create a zip whose extraction root is ``outputs/pretrain_dp``."""
    run_dir = resolve_run_dir(log_time)
    paths = validate_required_paths(run_dir)
    date_part, time_part = parse_log_time(log_time)
    # Preserve the layout found below ``outputs`` inside the archive.
    archive_run = Path("outputs") / run_dir.relative_to(OUTPUTS_ROOT)
    # Keep the local archive beside the resolved run tree, under its outputs root.
    OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
    zip_path = OUTPUTS_ROOT / f"{date_part}-{time_part}.zip"
    if zip_path.exists() and not force:
        return zip_path

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zipf:
        for name in ("dataset_metadata.json", "logs.json.txt"):
            zipf.write(paths[name], archive_run / name)
        for name in CHECKPOINT_NAMES:
            zipf.write(paths["checkpoints"] / name, archive_run / "checkpoints" / name)
        add_directory_to_zip(zipf, paths[".hydra"], archive_run / ".hydra")
    print(f"Zip finished: {zip_path} ({zip_path.stat().st_size / (1024 * 1024):.2f} MB)")
    return zip_path


def hf_resolve_url(repo_id: str, repo_type: str, filename: str) -> str:
    """Return the public Hugging Face URL for an uploaded file."""
    prefix = "datasets/" if repo_type == "dataset" else ""
    return f"https://huggingface.co/{prefix}{repo_id}/resolve/main/{filename}"


def compress_and_upload_checkpoint(
    log_time: str, repo_id: str, repo_type: str = "model", force: bool = True
) -> None:
    """Create a quadtree checkpoint archive and upload it to Hugging Face."""
    zip_path = create_checkpoint_zip(log_time, force=force)
    api = HfApi(token=os.getenv("HF_TOKEN"))
    api.upload_file(
        path_or_fileobj=str(zip_path),
        path_in_repo=zip_path.name,
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print(f"Uploaded: {hf_resolve_url(repo_id, repo_type, zip_path.name)}")


if __name__ == "__main__":
    main()
