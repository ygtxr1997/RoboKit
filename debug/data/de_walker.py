import os
import numpy as np
from torch.utils.data import DataLoader, Dataset

from robokit.data.tcl_datasets import TCLDataset
from robokit.data.data_handler import DataHandler
from robokit.debug_utils.printer import print_batch
from robokit.debug_utils.io import dataloader_speed_test


class DeTCLDataset(Dataset):
    def __init__(self, root):
        super(DeTCLDataset, self).__init__()
        self.root = root

        self.tasks = os.listdir(self.root)
        self.tasks.sort()
        self.task_lengths = []
        self.ep_fns = []
        self.map_index_to_task_id = []
        for task_id, task in enumerate(self.tasks):
            task_ep_fns = os.listdir(os.path.join(self.root, task))
            task_ep_fns.sort()
            self.ep_fns.extend([os.path.join(task, ep_fn) for ep_fn in task_ep_fns])

            self.task_lengths.append(len(task_ep_fns))
            self.map_index_to_task_id.extend([task_id] * self.task_lengths[task_id])

        self.total_length = sum(self.task_lengths)
        assert(len(self.ep_fns) == len(self.map_index_to_task_id))

    def __getitem__(self, index):
        task_id = self.map_index_to_task_id[index]
        ep_path = os.path.join(self.root, self.ep_fns[index])

        ep_data = self.load_single_episode(str(ep_path))
        return ep_data

    def load_single_episode(self, ep_path: str):
        # with np.load(ep_path, allow_pickle=True) as f:
        #     data = dict(f)

        data = DataHandler.load(file_path=ep_path)
        primary_rgb = data.data_dict['primary_rgb']
        print_batch('data', data.data_dict)

    def __len__(self):
        return self.total_length


from PIL import Image
from io import BytesIO
if __name__ == "__main__":
    from robokit.data.tcl_datasets import TCLDatasetHDF5
    h5_path = "/home/geyuan/local_soft/TCL/hdf5/collected_data_0507.h5"
    keys_config = {
        "primary_rgb": "rgb",
        "gripper_rgb": "rgb",
        "primary_depth": "depth",
        "gripper_depth": "depth",
        "language_text": "string",
        "actions": "float",
        "rel_actions": "float",
        "robot_obs": "float"
    }

    # dataset = TCLDataset(
    #     "/home/geyuan/local_soft/TCL/memory/collected_data_0507",
    #     load_keys=None,  #["rel_actions", "primary_rgb", "gripper_rgb", "robot_obs", "language_text"],
    #     use_extracted=True,
    # )
    # sample = dataset.__getitem__(0)
    # print_batch('data', sample)
    #
    # dataset = TCLDataset(root="/home/geyuan/local_soft/TCL/memory/collected_data_0507")

    # converter = TCLDatasetHDF5(
    #     "/home/geyuan/local_soft/TCL/collected_data_0507_light_random",
    #     "/home/geyuan/local_soft/TCL/hdf5/collected_data_0507_light_random.h5",
    #     use_h5=False,
    # )
    # converter.convert_to_hdf5(
    #     num_workers=48,
    #     pin_memory=True,
    # )

    import h5py

    def _binary_to_image(binary_data, decode_image: bool = True):
        """将二进制数据转换回 PIL 图像"""
        buffer = BytesIO(binary_data)  # 保持 buffer 在内存中
        if decode_image:
            pil_image = Image.open(buffer)
            pil_image.load()  # 强制加载图像数据
            return pil_image
        else:
            return np.array(buffer)

    import cv2
    def map_depth_with_color(depth_image: np.ndarray) -> Image.Image:
        depth_colored = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGRA2RGBA)
        depth_colored_image = Image.fromarray(depth_colored)
        return depth_colored_image

    dsets = {}
    with h5py.File("/home/geyuan/local_soft/TCL/hdf5/collected_data_0507_light_random.h5", 'r') as f:
        print(list(f.keys()))  # 查看文件顶级内容
        print(f['primary_rgb'].shape)  # 查看一个数据集的形状
        for key in keys_config:
            dsets[key] = f[key]
        img = _binary_to_image(dsets['primary_rgb'][10])
        print(img.size)
        # img = map_depth_with_color(np.array(img))
        img.save("tmp_h5.png", "PNG")

    # h5_dataset = TCLDatasetHDF5(
    #     "/home/geyuan/local_soft/TCL/memory/collected_data_0507",
    #     "/home/geyuan/local_soft/TCL/hdf5/collected_data_0507.h5",
    #     load_keys=["primary_rgb", "gripper_rgb", "language_text", "rel_actions", "robot_obs"],
    #     use_h5=True,
    #     is_img_decoded_in_h5=False,
    # )
    # sample = h5_dataset.__getitem__(0)
    # print_batch('h5_data', sample)
    #
    # dataloader_speed_test(h5_dataset, num_workers=24)

