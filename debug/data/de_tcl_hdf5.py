import time
import mediapy
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader

from robokit.datasets.tcl_datasets import TCLDatasetHDF5
from robokit.debug_utils import print_batch
from robokit.debug_utils import concatenate_rgb_images, plot_action_wrt_time, plot_force_sensor_wrt_time


""" Load TCL dataset and visualize some samples"""
dataset = TCLDatasetHDF5(
    root="/mnt/dongxu-fs1/data-hdd/geyuan/datasets/TCL/0209_tower_boby",
    h5_path="/mnt/dongxu-fs1/data-hdd/geyuan/datasets/TCL/hdf5/0209_tower_boby_240p.h5",
    use_extracted=False,
    load_keys=["rel_actions", "primary_rgb", "gripper_rgb", "robot_obs", "language_text", "force_torque"],
)
print("Total tasks:", len(dataset.tasks))
print_batch('TCLDatasetHDF5', dataset[0])


""" Dataloader speed test """
train_loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=8,
)

max_iter = 100
start_time = time.time()
for idx, batch in enumerate(tqdm(train_loader)):
    if idx == 0:
        print_batch('TCLDatasetHDF5 Dataloader', batch)
    if idx >= max_iter:
        break
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken for {max_iter} iterations: {elapsed_time:.2f} seconds, avg_speed={max_iter / elapsed_time:.2f} it/s")
