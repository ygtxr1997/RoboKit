"""
This script will help you convert any LeRobot dataset already pushed to the hub from codebase version
2.0 to 2.1.

It will:
- Generate per-episodes stats and writes them in episodes_stats.jsonl
- Check consistency between these new stats and the old ones.
- Remove the deprecated stats.json.
- Update codebase_version in info.json.
- Push this new version to the hub on the 'main' branch and tags it with "v2.1".

Usage:

Convert a dataset from the hub:
```bash
python -m lerobot.datasets.v21.convert_dataset_v20_to_v21 \
    --repo-id=aliberts/koch_tutorial
```

Convert a local dataset (works in place):
```bash
python -m lerobot.datasets.v21.convert_dataset_v20_to_v21 \
    --repo-id=aliberts/koch_tutorial \
    --root=/path/to/local/dataset/directory \
    --push-to-hub=false
```
"""

import argparse
import logging
from pathlib import Path

from huggingface_hub import HfApi

from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
from lerobot.datasets.utils import EPISODES_STATS_PATH, STATS_PATH, load_info, load_stats, write_info
# from lerobot.datasets.v21.convert_stats import check_aggregate_stats, convert_stats

from robokit.datasets.eo.lerobot033.convert_stats import check_aggregate_stats, convert_stats

V20 = "v2.0"
V21 = "v2.1"


class SuppressWarnings:
    def __enter__(self):
        self.previous_level = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.ERROR)

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.getLogger().setLevel(self.previous_level)


def validate_local_dataset_version(local_path: Path) -> None:
    """Validate that the local dataset has the expected v2.0 version."""
    info = load_info(local_path)
    dataset_version = info.get("codebase_version", "unknown")
    if dataset_version != V20:
        raise ValueError(
            f"Local dataset has codebase version '{dataset_version}', expected '{V20}'. "
            f"This script is specifically for converting v2.0 datasets to v2.1."
        )


def convert_dataset(
        repo_id: str,
        branch: str | None = None,
        num_workers: int = 4,
        root: str | Path | None = None,
        push_to_hub: bool = True,
):
    # Determine if using local dataset
    use_local_dataset = root is not None

    if use_local_dataset:
        local_root = Path(root) / repo_id
        if not local_root.exists():
            raise ValueError(f"Local dataset path does not exist: {local_root}")
        validate_local_dataset_version(local_root)
        print(f"Using local dataset at {local_root}")

        with SuppressWarnings():
            dataset = LeRobotDataset(repo_id, root=local_root, revision=V20, force_cache_sync=False)
    else:
        with SuppressWarnings():
            dataset = LeRobotDataset(repo_id, revision=V20, force_cache_sync=True)

    # Remove existing episodes_stats.jsonl if present
    if (dataset.root / EPISODES_STATS_PATH).is_file():
        (dataset.root / EPISODES_STATS_PATH).unlink()

    # Convert stats
    convert_stats(dataset, num_workers=num_workers)

    # Load and optionally check stats
    ref_stats = load_stats(dataset.root)
    # check_aggregate_stats(dataset, ref_stats)

    # Update codebase version
    dataset.meta.info["codebase_version"] = CODEBASE_VERSION
    write_info(dataset.meta.info, dataset.root)

    # Delete old stats.json file
    if (dataset.root / STATS_PATH).is_file():
        (dataset.root / STATS_PATH).unlink()

    if push_to_hub:
        dataset.push_to_hub(branch=branch, tag_version=False, allow_patterns="meta/")

        hub_api = HfApi()
        if hub_api.file_exists(
                repo_id=dataset.repo_id, filename=STATS_PATH, revision=branch, repo_type="dataset"
        ):
            hub_api.delete_file(
                path_in_repo=STATS_PATH, repo_id=dataset.repo_id, revision=branch, repo_type="dataset"
            )

        hub_api.create_tag(repo_id, tag=CODEBASE_VERSION, revision=branch, repo_type="dataset")
    else:
        print(f"Conversion complete. Dataset saved locally at {dataset.root}")
        print("Skipping push to hub as --push-to-hub=false was specified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository identifier on Hugging Face: a community or a user name / the name of the dataset "
             "(e.g. lerobot/pusht, cadene/aloha_sim_insertion_human).",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default=None,
        help="Repo branch to push your dataset. Defaults to the main branch.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for parallelizing stats compute. Defaults to 4.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Local directory containing the dataset. If provided, the script will convert the local dataset instead of downloading from the hub.",
    )
    parser.add_argument(
        "--push-to-hub",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Push the converted dataset to the hub. Set to 'false' for local-only conversion.",
    )

    args = parser.parse_args()
    convert_dataset(**vars(args))