import os
import textwrap
import zipfile
from pathlib import Path

from huggingface_hub import HfApi


OUTPUTS_ROOT = Path.home() / "code/dp23rss_fork/data/outputs"
LOCAL_ZIP_DIR = Path.home() / "pretrained/dp23rss_fork"
DEFAULT_REPO_ID = "ygtxr1997/tcl_models"
DEFAULT_REPO_TYPE = "model"


def main():
    # Use YYYY.MM.DD-HH.MM.SS. The time part is matched as a unique prefix
    # under ~/code/dp23rss_fork/data/outputs/YYYY.MM.DD/.
    upload_dirs = [
        # "2026.03.18-21.22.07",  # tower_boby_easy
        # "2026.04.29-01.20.15",  # tower_boby_easy_reversed, iter1/2/3
        # "2026.04.18-00.45.25",  # put_mouse_easy
        # "2026.04.29-01.26.47",  # put_mouse_easy_reversed
        # "2026.04.24-18.38.41",  # test_tube
        # "2026.04.29-14.23.51",  # test_tube_reversed
        # "2026.05.01-01.51.17",  # brush
        # "2026.05.01-14.35.36",  # brush_reversed

        # "2026.05.14-21.32.50",  # ethernet, cfgrl

        "2026.09.09-11.12.55",  # tower_boby_hard
        "2026.09.09-11.13.27",  # plastic_egg
        "2026.09.09-01.18.17",  # put_mouse_hard
    ]

    for upload_dir in upload_dirs:
        compress_and_upload_checkpoint(
            upload_dir,
            repo_id=DEFAULT_REPO_ID,
            repo_type=DEFAULT_REPO_TYPE,
            force=True,
        )

    if upload_dirs:
        download_and_extract_example(
            upload_dirs[0],
            repo_id=DEFAULT_REPO_ID,
            repo_type=DEFAULT_REPO_TYPE,
        )


def parse_upload_dir(upload_dir: str) -> tuple[str, str]:
    """Parse YYYY.MM.DD-HH.MM.SS into date and time prefix."""
    upload_dir = upload_dir.strip()
    if not upload_dir or "-" not in upload_dir:
        raise ValueError(
            f"Invalid upload_dir {upload_dir!r}. Expected format: YYYY.MM.DD-HH.MM.SS"
        )

    date_part, time_prefix = upload_dir.split("-", 1)
    if not date_part or not time_prefix:
        raise ValueError(
            f"Invalid upload_dir {upload_dir!r}. Expected format: YYYY.MM.DD-HH.MM.SS"
        )
    return date_part, time_prefix


def resolve_run_dir(upload_dir: str, outputs_root: Path = OUTPUTS_ROOT) -> Path:
    """Resolve a short upload_dir to exactly one completed training output directory."""
    date_part, time_prefix = parse_upload_dir(upload_dir)
    date_dir = outputs_root / date_part

    if not date_dir.is_dir():
        raise FileNotFoundError(f"Date directory not found: {date_dir}")

    prefix_matches = sorted(
        path for path in date_dir.iterdir() if path.is_dir() and path.name.startswith(time_prefix)
    )
    matches = [path for path in prefix_matches if (path / "statistics.json").is_file()]
    if not matches:
        if prefix_matches:
            match_list = "\n".join(f"  - {path}" for path in prefix_matches)
            raise FileNotFoundError(
                f"Run directories matched {upload_dir!r}, but none contain statistics.json:\n"
                f"{match_list}"
            )
        raise FileNotFoundError(
            f"No run directory found under {date_dir} with prefix {time_prefix!r} "
            "and statistics.json"
        )
    if len(matches) > 1:
        match_list = "\n".join(f"  - {path}" for path in matches)
        raise RuntimeError(
            f"Multiple run directories found for {upload_dir!r}; expected exactly one:\n{match_list}"
        )

    return matches[0]


def validate_required_paths(run_dir: Path) -> dict[str, Path]:
    paths = {
        "statistics.json": run_dir / "statistics.json",
        "checkpoints/latest.ckpt": run_dir / "checkpoints/latest.ckpt",
        ".hydra": run_dir / ".hydra",
    }

    missing = []
    if not paths["statistics.json"].is_file():
        missing.append(paths["statistics.json"])
    if not paths["checkpoints/latest.ckpt"].is_file():
        missing.append(paths["checkpoints/latest.ckpt"])
    if not paths[".hydra"].is_dir():
        missing.append(paths[".hydra"])

    if missing:
        missing_list = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Required checkpoint files are missing:\n{missing_list}")

    return paths


def add_directory_to_zip(zipf: zipfile.ZipFile, source_dir: Path, arc_dir: Path) -> None:
    zipf.write(source_dir, arc_dir)
    print(f"Adding dir: {arc_dir}")

    for path in sorted(source_dir.rglob("*")):
        arc_path = arc_dir / path.relative_to(source_dir)
        zipf.write(path, arc_path)
        if path.is_dir():
            print(f"Adding dir: {arc_path}")
        else:
            print(f"Adding file: {arc_path}")


def create_checkpoint_zip(upload_dir: str, force: bool = True) -> Path:
    run_dir = resolve_run_dir(upload_dir)
    paths = validate_required_paths(run_dir)

    LOCAL_ZIP_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = LOCAL_ZIP_DIR / f"{upload_dir}.zip"

    print(f"Resolved run dir: {run_dir}")
    print(f"Target zip file: {zip_path}")

    if zip_path.exists() and not force:
        print(f"Zip file already exists: {zip_path}")
        return zip_path

    root_in_zip = Path(upload_dir)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zipf:
        stats_arc = root_in_zip / "statistics.json"
        ckpt_arc = root_in_zip / "checkpoints/latest.ckpt"
        hydra_arc = root_in_zip / ".hydra"

        zipf.write(paths["statistics.json"], stats_arc)
        print(f"Adding file: {stats_arc}")

        zipf.write(paths["checkpoints/latest.ckpt"], ckpt_arc)
        print(f"Adding file: {ckpt_arc}")

        add_directory_to_zip(zipf, paths[".hydra"], hydra_arc)

    print(f"Zip finished: {zip_path}")
    print(f"File size: {zip_path.stat().st_size / (1024 * 1024):.2f} MB")
    return zip_path


def hf_resolve_url(repo_id: str, repo_type: str, filename: str) -> str:
    if repo_type == "dataset":
        return f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"
    return f"https://huggingface.co/{repo_id}/resolve/main/{filename}"


def compress_and_upload_checkpoint(
    upload_dir: str,
    repo_id: str,
    repo_type: str = "model",
    force: bool = True,
) -> None:
    """Zip statistics.json, checkpoints/latest.ckpt, and the full .hydra directory."""
    zip_path = create_checkpoint_zip(upload_dir, force=force)
    zip_filename = zip_path.name

    api = HfApi(token=os.getenv("HF_TOKEN"))

    try:
        print(f"Uploading to {repo_id} ({repo_type})...")
        api.upload_file(
            path_or_fileobj=str(zip_path),
            path_in_repo=zip_filename,
            repo_id=repo_id,
            repo_type=repo_type,
        )
        print(f"Uploaded! File: {zip_filename}")
        print(f"Download URL: {hf_resolve_url(repo_id, repo_type, zip_filename)}")
    except Exception as exc:
        print(f"Uploading failed: {exc}")
        raise


def download_and_extract_example(
    upload_dir: str = "2026.03.18-21.22.07",
    repo_id: str = DEFAULT_REPO_ID,
    repo_type: str = DEFAULT_REPO_TYPE,
) -> None:
    """Print example code for downloading and extracting the uploaded zip."""
    zip_filename = f"{upload_dir}.zip"
    example_code = f'''
    # Run this on the target machine.
    import zipfile
    from pathlib import Path
    from huggingface_hub import hf_hub_download

    repo_id = "{repo_id}"
    repo_type = "{repo_type}"
    zip_filename = "{zip_filename}"
    extract_to = Path.home() / "downloads"
    extract_to.mkdir(parents=True, exist_ok=True)

    zip_path = hf_hub_download(
        repo_id=repo_id,
        repo_type=repo_type,
        filename=zip_filename,
        local_dir=extract_to,
    )

    with zipfile.ZipFile(zip_path, "r") as zipf:
        zipf.extractall(extract_to)

    run_dir = extract_to / "{upload_dir}"
    print("Unzip finished")
    print(f"statistics.json: {{run_dir / 'statistics.json'}}")
    print(f"latest.ckpt: {{run_dir / 'checkpoints/latest.ckpt'}}")
    print(f".hydra: {{run_dir / '.hydra'}}")
    '''
    print("=" * 50)
    print("Downloading example:")
    print("=" * 50)
    print(textwrap.dedent(example_code).strip())


if __name__ == "__main__":
    main()
