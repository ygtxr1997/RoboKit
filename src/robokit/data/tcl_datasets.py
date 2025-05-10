import os
import json
from typing import List
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from robokit.data.data_handler import DataHandler


class TCLDataset(Dataset):
    def __init__(self, root, use_extracted: bool = False, load_keys: List[str] = None):
        super(TCLDataset, self).__init__()
        self.root = root

        self.tasks = self.get_tasks(root)
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
        assert (len(self.ep_fns) == len(self.map_index_to_task_id))

        self.extracted_data = {}
        if use_extracted:
            self.load_npy_by_key("rel_actions")

        if load_keys is None:
            load_keys = ["primary_rgb", "gripper_rgb", "primary_depth", "gripper_depth",
                         "language_text", "actions", "rel_actions", "robot_obs"]
        self.load_keys = load_keys

        print("[TCLDataset] total length:", self.total_length)

    def get_tasks(self, root):
        self.root = root
        tasks = []
        for task in os.listdir(root):
            if not os.path.isdir(os.path.join(root, task)):
                continue
            elif "extracted" in task:  # skip some meta info folders
                continue
            tasks.append(task)
        tasks.sort()
        return tasks

    def __getitem__(self, index):
        task_id = self.map_index_to_task_id[index]
        npz_path = os.path.join(self.root, self.ep_fns[index])

        npz_data = self.load_single_frame(str(npz_path))
        if npz_data is None:  # file broken or other error
            npz_data = self.__getitem__(np.random.randint(0, len(self.ep_fns)))

        return npz_data

    def load_single_frame(self, npz_path: str):
        data_handler = DataHandler.load(file_path=npz_path, load_keys=self.load_keys)
        data = {
            k: data_handler.get(k) for k in self.load_keys
        }
        if "language_text" in data:
            data["language_text"] = str(data["language_text"])
        return data

    def __len__(self):
        return self.total_length

    def get_statistics_and_save(
            self,
            save_json_path: str = None,
            batch_size: int = 256,
            num_workers: int = 16,
            pin_memory: bool = True,
    ) -> dict:
        """
        Calculate min, max, mean, std for each component of 'rel_actions' across the entire dataset,
        using a DataLoader for batch loading.
        """
        # 1. 初始化统计量
        n_components = 7
        min_vals = np.inf * np.ones(n_components, dtype=np.float64)
        max_vals = -np.inf * np.ones(n_components, dtype=np.float64)
        sum_vals = np.zeros(n_components, dtype=np.float64)
        squared_sum = np.zeros(n_components, dtype=np.float64)
        total_count = 0

        # 2. 构造 DataLoader
        loader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=lambda batch: np.stack([item['rel_actions'] for item in batch], axis=0)
        )

        # 3. 遍历 DataLoader，批量更新统计量
        for batch_rel_actions in tqdm(loader, desc="Computing stats"):
            # batch_rel_actions: shape (B, 7)
            # 转成 float64 numpy（若已经是 np.ndarray 则直接使用）
            if not isinstance(batch_rel_actions, np.ndarray):
                batch_rel_actions = batch_rel_actions.numpy()
            B = batch_rel_actions.shape[0]

            # 更新 min/max
            min_vals = np.minimum(min_vals, batch_rel_actions.min(axis=0))
            max_vals = np.maximum(max_vals, batch_rel_actions.max(axis=0))

            # 更新 sum 和 squared sum
            sum_vals += batch_rel_actions.sum(axis=0)
            squared_sum += (batch_rel_actions ** 2).sum(axis=0)

            total_count += B

        # 4. 计算最终的 mean 和 std
        mean_vals = sum_vals / total_count
        var_vals = (squared_sum / total_count) - (mean_vals ** 2)
        std_vals = np.sqrt(np.maximum(var_vals, 0.0))

        statistics = {
            "min": min_vals.tolist(),
            "max": max_vals.tolist(),
            "mean": mean_vals.tolist(),
            "std": std_vals.tolist(),
            "total_len": getattr(self, "total_length", total_count)
        }

        # 5. 可选：保存到 JSON
        if save_json_path:
            save_path = os.path.join(getattr(self, 'root', ''), save_json_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as fp:
                json.dump(statistics, fp, indent=4)
            print(f"[TCLDataset] statistics saved to: {save_path}")

        return statistics

    def load_statistics_from_json(self, json_path: str) -> dict:
        json_path = os.path.join(self.root, json_path)
        print("[TCLDataset] loading dataset statistics from:", json_path)
        with open(json_path, 'r') as json_file:
            statistics = json.load(json_file)
        return statistics

    def save_to_npy_by_key(
            self,
            key: str,
            path: str = None,
            batch_size: int = 256,
            num_workers: int = 4,
            pin_memory: bool = False,
    ):
        """
        将整个数据集里某个字段 key 对应的内容，批量读取后保存为一个 .npy 文件。
        """
        if path is None:
            path = os.path.join(self.root, f"extracted/{key}.npy")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # 1. 定义 DataLoader，collate_fn 只抽出 key 对应的数据，并堆叠成 (B, ...) 形状
        loader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=lambda batch: np.stack([item[key] for item in batch], axis=0)
        )

        # 2. 按批次读取并收集到 list
        all_data = []
        for batch_data in tqdm(loader, desc=f"Loading key='{key}'"):
            # batch_data 已经是一个 numpy 数组，形状 (B, ...)
            all_data.append(batch_data)

        # 3. 拼接所有批次，保存到 .npy
        all_data = np.concatenate(all_data, axis=0)
        np.save(path, all_data)
        print(f"[TCLDataset] key='{key}' shape={all_data.shape} → saved to '{path}'")

    def load_npy_by_key(self, key: str, path: str = None):
        if path is None:
            path = os.path.join(self.root, f"extracted/{key}.npy")
        self.extracted_data[key] = np.load(path)
        print(f"[TCLDataset] loaded key={key} shape={self.extracted_data[key].shape} from {path}")


if __name__ == "__main__":
    data_root = "/home/geyuan/local_soft/TCL/collected_data_0507"

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
    print(dataset.extracted_data['rel_actions'][23])

    # 4. Reload dataset using extracted key
    dataset_2 = TCLDataset(data_root, use_extracted=True)
    print(dataset.extracted_data['rel_actions'][23])
