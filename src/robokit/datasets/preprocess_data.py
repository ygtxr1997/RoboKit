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

    keys_to_extract = ["rel_actions"]
    if "gripper_max_effort" in dataset.load_keys:
        keys_to_extract.append("gripper_max_effort")
    _ = dataset.extract_data_and_compute_statistics(
        keys_to_extract=keys_to_extract,
        save_json_path=statistics_json_path,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # _ = dataset.get_statistics_and_save(
    #     save_json_path=statistics_json_path, batch_size=batch_size, num_workers=num_workers)
    meta_info = dataset.load_meta_from_json(json_path=statistics_json_path)
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
    """  Used when `as_hdf5` is not None """
    num_workers = opts.num_workers
    data_root = opts.root
    batch_size = opts.batch_size
    num_workers = opts.num_workers

    if opts.resize == "240p":
        resize_wh = (320, 240)
    elif opts.resize == "256x320":
        resize_wh = (320, 256)
    elif opts.resize == "384x512":
        resize_wh = (512, 384)
    elif opts.resize == "480p":
        resize_wh = (640, 480)
    else:
        raise NotImplementedError(f"Unknown resize {opts.resize}")

    # 2. Save and load statistics info
    converter = TCLDatasetHDF5(
        data_root,
        opts.as_hdf5,
        use_h5=False,
        load_episode_success_fn=opts.load_suc_fn or None,
    )
    if opts.no_depth:
        # Keep automatic effort-field detection and mixed-schema validation.
        converter.load_keys = [
            key for key in converter.load_keys
            if key not in ("primary_depth", "gripper_depth")
        ]

    print("Save and load statistics info")
    statistics_json_path = "statistics.json"

    keys_to_extract = ["rel_actions", "robot_obs", "force_torque"]
    if "gripper_max_effort" in converter.load_keys:
        keys_to_extract.append("gripper_max_effort")
    _ = converter.extract_data_and_compute_statistics(
        keys_to_extract=keys_to_extract,
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
        load_keys=["primary_rgb", "gripper_rgb", "language_text"] + keys_to_extract,
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

    Op2. Convert to HDF5 with resize
    python scripts/01_preprocess_data.py  \
        -R "/home/geyuan/local_soft/TCL/1009_spoon_pick_place"  \
        --as_hdf5 "/home/geyuan/local_soft/TCL/hdf5/1009_spoon_pick_place_240p.h5"
    python -m robokit.datasets.preprocess_data  \
        -R "/home/geyuan/local_soft/TCL/1014_lh_tower_three_blocks/"  \
        --resize 256x320 \
        --as_hdf5 "/home/geyuan/local_soft/TCL/hdf5/1014_lh_tower_three_blocks_256x320.h5"
        
    Op3. Convert to HDF5 at 640x480
    python -m robokit.datasets.preprocess_data  \
        -R "/home/geyuan/local_soft/TCL/260514_ethernet"  \
        --resize 480p  \
        --as_hdf5 "/home/geyuan/local_soft/TCL/hdf5/260514_ethernet_480p.h5"
    
    export REV_DIR="0417_put_mouse_reversed"
    python -m robokit.datasets.preprocess_data  \
        -R "/home/geyuan/datasets/reverse/${REV_DIR}"  \
        --as_hdf5 "/home/geyuan/datasets/reverse/hdf5/${REV_DIR}_240p.h5"
    export REV_DIR="0417_ethernet_reversed"
    python -m robokit.datasets.preprocess_data  \
        -R "/home/geyuan/datasets/reverse/${REV_DIR}"  \
        --as_hdf5 "/home/geyuan/datasets/reverse/hdf5/${REV_DIR}_240p.h5"
    
    # export REV_DIR="0417_french_press_reversed"
    export REV_DIR="0417_brush_reversed"
    mkdir ~/datasets/reverse/${REV_DIR}
    cp -r /mnt/dongxu-fs1/data-hdd/qiyuanqiao/datasets/TCL_reversed/${REV_DIR}/2026* ~/datasets/reverse/${REV_DIR}/
    python -m robokit.datasets.preprocess_data  \
        -R "/home/geyuan/datasets/reverse/${REV_DIR}"  \
        --as_hdf5 "/home/geyuan/datasets/reverse/hdf5/${REV_DIR}_240p.h5"
        
    export REV_DIR="tower_boby_easy_reversed_iter2"
    mkdir ~/datasets/reverse/${REV_DIR}
    cp -r /mnt/dongxu-fs1/data-hdd/qiyuanqiao/datasets/TCL_reversed/${REV_DIR}/ep* ~/datasets/reverse/${REV_DIR}/
    python -m robokit.datasets.preprocess_data  \
        -R "/home/geyuan/datasets/reverse/${REV_DIR}"  \
        --as_hdf5 "/home/geyuan/datasets/reverse/hdf5/${REV_DIR}_240p.h5"  \
        --no_depth
    """
    args = argparse.ArgumentParser("Test data_manager format")
    args.add_argument("-R", "--root", required=True, type=str, help="Root folder path")
    args.add_argument("-i", "--check_index", default=23, type=int, help="Index for checking")
    args.add_argument("--as_hdf5", default="", type=str, help="(Optional) saving HDF5 file path")
    args.add_argument("--resize", default="240p", type=str,
                      choices=["240p", "256x320", "384x512", "480p"], help="(Optional) resize the image, default 240p:(wh=320x240)")
    args.add_argument("-b", "--batch_size", default=32, type=int, help="batch_size")
    args.add_argument("-j", "--num_workers", default=8, type=int, help="num_workers")

    args.add_argument("--no_depth", action="store_true", default=False, help="True: do not use depth image")
    args.add_argument("--load_suc_fn", default="", type=str, help="Load `episode_index.json` for success/fail trajectory state")
    args = args.parse_args()
    main(args)
