import argparse
from robokit.data.tcl_datasets import TCLDataset


def main(opts):
    data_root = opts.root  # e.g. "/home/geyuan/local_soft/TCL/collected_data_0507_light_random"

    # 1. Load data
    dataset = TCLDataset(data_root, use_extracted=False)
    dataset.__getitem__(0)
    # dataset.total_length = 100  # For debug

    # 2. Save and load statistics info
    print("Save and load statistics info")
    statistics_json_path = "statistics.json"
    _ = dataset.get_statistics_and_save(save_json_path=statistics_json_path)
    meta_info = dataset.load_statistics_from_json(json_path=statistics_json_path)
    print(meta_info)

    # 3. Extract data by key
    print("Extract data by key")
    dataset.save_to_npy_by_key("rel_actions")
    dataset.load_npy_by_key("rel_actions")
    print("save:", dataset.extracted_data['rel_actions'][opts.check_index])

    # 4. Reload dataset using extracted key
    dataset_2 = TCLDataset(data_root, use_extracted=True)
    print("load:", dataset.extracted_data['rel_actions'][opts.check_index])

    print("Preprocessing finished.")


if __name__ == "__main__":
    """
    Example usage:
    python scripts/01_preprocess_data.py -R "/home/geyuan/local_soft/TCL/collected_data_0507_light_random"
    """
    args = argparse.ArgumentParser("Test data format")
    args.add_argument("-R", "--root", required=True, type=str, help="Root folder path")
    args.add_argument("-i", "--check_index", default=23, type=int, help="Index for checking")
    args = args.parse_args()
    main(args)
