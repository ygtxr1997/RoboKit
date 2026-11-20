import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = "ygtxr1997/tcl_models"
REPO_TYPE = "model"
DOWNLOAD_ROOT = Path.home() / "pretrained/quadtree"


def parse_log_time(log_time: str) -> tuple[str, str]:
    """Parse ``YYYY-MM-DD-HH-MM-SS`` into date and time components."""
    value = log_time.strip().strip("/")
    parts = value.split("-")
    if len(parts) != 6 or len(parts[0]) != 4 or len(parts[1]) != 2 or len(parts[2]) != 2:
        raise ValueError(
            f"Invalid log_time {log_time!r}; expected YYYY-MM-DD-HH-MM-SS"
        )
    return "-".join(parts[:3]), "-".join(parts[3:])


def download_and_extract_checkpoint(
    log_time: str,
    repo_id: str = REPO_ID,
    repo_type: str = REPO_TYPE,
    extract_to: Path = DOWNLOAD_ROOT,
) -> Path:
    """Download one archive and extract it below ``extract_to``."""
    date_part, time_part = parse_log_time(log_time)
    zip_filename = f"{date_part}-{time_part}.zip"
    extract_to.mkdir(parents=True, exist_ok=True)

    zip_path = hf_hub_download(
        repo_id=repo_id,
        repo_type=repo_type,
        filename=zip_filename,
        local_dir=extract_to,
    )
    with zipfile.ZipFile(zip_path, "r") as zipf:
        zipf.extractall(extract_to)

    run_dir = extract_to / "outputs" / "pretrain_dp" / date_part / time_part
    print(f"Unzip finished: {run_dir}")
    print(f"checkpoints: {run_dir / 'checkpoints'}")
    print(f"dataset_metadata.json: {run_dir / 'dataset_metadata.json'}")
    print(f"logs.json.txt: {run_dir / 'logs.json.txt'}")
    print(f".hydra: {run_dir / '.hydra'}")
    return run_dir


def main() -> None:
    """Download the configured quadtree checkpoint archives."""
    log_times = [
        # "2026-11-02-23-35-12",
    ]
    for log_time in log_times:
        download_and_extract_checkpoint(log_time)


if __name__ == "__main__":
    main()
