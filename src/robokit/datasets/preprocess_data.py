import argparse
from robokit.debug_utils.printer import print_batch, beautiful_print
from robokit.datasets.tcl_datasets import TCLDataset, TCLDatasetHDF5


def main(opts):
    if opts.as_hdf5 != "":
        convert_to_hdf5(opts)
    else:
        preprocess(opts)


def preprocess(opts):
    data_root = opts.root  # e.g. "/home/geyuan/local_soft/TCL/collected_data_0507_light_random"
    batch_size = opts.batch_size
    num_workers = opts.num_workers

    # 1. Load data_manager
    dataset = TCLDataset(data_root, use_extracted=False)
    dataset.__getitem__(0)
    # dataset.total_length = 100  # For debug

    # 2. Save and load statistics info
    print("Save and load statistics info")
    statistics_json_path = "statistics.json"

    _ = dataset.extract_data_and_compute_statistics(
        keys_to_extract=["rel_actions"],
        save_json_path=statistics_json_path,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # _ = dataset.get_statistics_and_save(
    #     save_json_path=statistics_json_path, batch_size=batch_size, num_workers=num_workers)
    meta_info = dataset.load_statistics_from_json(json_path=statistics_json_path)
    print(meta_info)

    # 3. Extract data_manager by key
    # print("Extract data_manager by key")
    # dataset.save_to_npy_by_key(
    #     "rel_actions", batch_size=batch_size, num_workers=num_workers)
    dataset.load_npy_by_key("rel_actions")
    print("save:", dataset.extracted_data['rel_actions'][opts.check_index])

    # 4. Reload dataset using extracted key
    dataset_2 = TCLDataset(data_root, use_extracted=True)
    print("load:", dataset.extracted_data['rel_actions'][opts.check_index])

    print("Preprocessing finished.")


def convert_to_hdf5(opts):
    num_workers = opts.num_workers
    data_root = opts.root
    batch_size = opts.batch_size
    num_workers = opts.num_workers

    if opts.resize == "240p":
        resize_wh = (320, 240)
    elif opts.resize == "480p":
        resize_wh = None
    else:
        raise NotImplementedError(f"Unknown resize {opts.resize}")

    # 2. Save and load statistics info
    converter = TCLDatasetHDF5(
        data_root,
        opts.as_hdf5,
        use_h5=False,
    )

    print("Save and load statistics info")
    statistics_json_path = "statistics.json"

    _ = converter.extract_data_and_compute_statistics(
        keys_to_extract=["rel_actions", "robot_obs", "force_torque"],  # in [`rel_actions`, `robot_obs`, `force_torque`]
        save_json_path=statistics_json_path,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    meta_info = converter.load_meta_from_json(json_path=statistics_json_path)
    print("[DEBUG] meta_info:")
    beautiful_print(meta_info)

    # 5. Convert to HDF5
    converter.convert_to_hdf5(
        num_workers=num_workers,
        pin_memory=True,
        resize_wh=resize_wh,
    )

    h5_dataset = TCLDatasetHDF5(
        data_root,
        opts.as_hdf5,
        load_keys=["primary_rgb", "gripper_rgb", "language_text", "rel_actions", "robot_obs", "force_torque"],
        use_h5=True,
        is_img_decoded_in_h5=False,
    )
    sample = h5_dataset.__getitem__(0)
    print_batch('h5_data', sample)

    print("HDF5 conversion finished.")


if __name__ == "__main__":
    """
    Example usage:
    Op1. Extract keys and statistics
    python scripts/01_preprocess_data.py  \
        -R "/home/geyuan/local_soft/TCL/1009_spoon_pick_place"

    Op2. Convert to HDF5
    python scripts/01_preprocess_data.py  \
        -R "/home/geyuan/local_soft/TCL/1009_spoon_pick_place"  \
        --as_hdf5 "/home/geyuan/local_soft/TCL/hdf5/1009_spoon_pick_place_240p.h5"
    """
    args = argparse.ArgumentParser("Test data_manager format")
    args.add_argument("-R", "--root", required=True, type=str, help="Root folder path")
    args.add_argument("-i", "--check_index", default=23, type=int, help="Index for checking")
    args.add_argument("--as_hdf5", default="", type=str, help="(Optional) saving HDF5 file path")
    args.add_argument("--resize", default="240p", type=str,
                      choices=["240p", "480p"], help="(Optional) resize the image, default 240p:(wh=320x240)")
    args.add_argument("-b", "--batch_size", default=32, type=int, help="batch_size")
    args.add_argument("-j", "--num_workers", default=8, type=int, help="num_workers")
    args = args.parse_args()
    main(args)
