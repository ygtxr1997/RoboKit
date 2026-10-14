import zipfile
from pathlib import Path
from huggingface_hub import hf_hub_download

repo_id = "ygtxr1997/tcl_models"
repo_type = "model"
log_times = [
    # "2026.03.18-21.22.07",  # tower_boby
    # "2026.04.29-01.20.15",
    # "2026.04.18-00.45.25",  # put_mouse
    # "2026.04.29-01.26.47",
    # "2026.04.24-18.38.41",  # test_tube
    # "2026.04.29-14.23.51",
    # "2026.05.01-01.51.17",  # brush
    # "2026.05.01-14.35.36",

    # "2026.05.14-21.32.50",  # cfgrl, ethernet

    "2026.09.09-11.12.55",  # tower_boby_hard
    "2026.09.09-11.13.27",  # plastic_egg
    "2026.09.09-01.18.17",  # put_mouse_hard
]

for log_time in log_times:
    zip_filename = f"{log_time}.zip"
    extract_to = Path.home() / "code/dp23rss_fork/data/outputs/"
    extract_to.mkdir(parents=True, exist_ok=True)

    zip_path = hf_hub_download(
        repo_id=repo_id,
        repo_type=repo_type,
        filename=zip_filename,
        local_dir=extract_to,
    )

    with zipfile.ZipFile(zip_path, "r") as zipf:
        zipf.extractall(extract_to)

    run_dir = extract_to / f"{log_time}"
    print("Unzip finished")
    print(f"statistics.json: {run_dir / 'statistics.json'}")
    print(f"latest.ckpt: {run_dir / 'checkpoints/latest.ckpt'}")
    print(f".hydra: {run_dir / '.hydra'}")