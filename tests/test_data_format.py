import argparse

from robokit.data.tcl_datasets import TCLDataset
from robokit.debug_utils.printer import print_batch


def main(opts):
    dataset = TCLDataset(opts.root)
    for i in range(len(dataset)):
        episode_data = dataset[i]
        print_batch(f'{i}-th data:', episode_data)
    print(f"[Checked] Passed {len(dataset)} samples.")


if __name__ == '__main__':
    args = argparse.ArgumentParser("Test data format")
    args.add_argument("-R", "--root", required=True, type=str, help="Root folder path")
    args = args.parse_args()
    main(args)
