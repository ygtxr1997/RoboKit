import os
import json
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from robokit.data.data_handler import DataHandler


class TCLDataset(Dataset):
    def __init__(self, root, use_extracted: bool = False):
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
        data_handler = DataHandler.load(file_path=npz_path)
        data = data_handler.data_dict
        return data

    def __len__(self):
        return self.total_length

    def get_statistics_and_save(self, save_json_path: str = None) -> dict:
        """
        Calculate min, max, mean, std for each component of 'rel_actions' across the entire dataset.
        """
        # Initialize stats for each component
        min_vals = np.inf * np.ones(7)
        max_vals = -np.inf * np.ones(7)
        sum_vals = np.zeros(7)
        squared_sum = np.zeros(7)
        count = 0

        # Iterate over the dataset
        for index in tqdm(range(len(self))):
            rel_actions_data = self[index]['rel_actions']  # Load the data for the current index

            # Update min, max, sum, squared_sum for each component (7 components in total)
            min_vals = np.minimum(min_vals, rel_actions_data)
            max_vals = np.maximum(max_vals, rel_actions_data)
            sum_vals += rel_actions_data
            squared_sum += rel_actions_data ** 2
            count += rel_actions_data.size

        # Calculate mean and std for each component
        mean_vals = sum_vals / count
        variance_vals = (squared_sum / count) - (mean_vals ** 2)
        std_vals = np.sqrt(variance_vals)

        statistics = {
            "min": min_vals.tolist(),
            "max": max_vals.tolist(),
            "mean": mean_vals.tolist(),
            "std": std_vals.tolist(),
            "total_len": self.total_length,
        }

        # Save statistics to a JSON file
        if save_json_path is not None:
            save_json_path = os.path.join(self.root, save_json_path)
            with open(save_json_path, 'w') as json_file:
                json.dump(statistics, json_file, indent=4)
            print("[TCLDataset] statistics info saved to:", save_json_path)

        return statistics

    def load_statistics_from_json(self, json_path: str) -> dict:
        json_path = os.path.join(self.root, json_path)
        print("[TCLDataset] loading dataset statistics from:", json_path)
        with open(json_path, 'r') as json_file:
            statistics = json.load(json_file)
        return statistics

    def save_to_npy_by_key(self, key: str, path: str = None):
        if path is None:
            path = os.path.join(self.root, f"extracted/{key}.npy")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        all_key_data = []
        for idx in tqdm(range(self.__len__()), desc=f"Loading key={key}"):
            data_dict = self.__getitem__(idx)
            key_data = data_dict[key]
            all_key_data.append(key_data)
        all_key_data = np.stack(all_key_data, axis=0)
        np.save(path, all_key_data)
        print(f"[TCLDataset] key={key} shape={all_key_data.shape} saved to {path}.")

    def load_npy_by_key(self, key: str, path: str = None):
        if path is None:
            path = os.path.join(self.root, f"extracted/{key}.npy")
        self.extracted_data[key] = np.load(path)
        print(f"[TCLDataset] loaded key={key} shape={self.extracted_data[key].shape} from {path}")


if __name__ == "__main__":
    data_root = "/home/geyuan/local_soft/TCL/collected_data_0425"

    # 1. Load data
    dataset = TCLDataset(data_root, use_extracted=False)
    dataset.__getitem__(0)
    # dataset.total_length = 100  # For debug

    # 2. Save and load statistics info
    statistics_json_path = "statistics.json"
    _ = dataset.get_statistics_and_save(save_json_path=statistics_json_path)
    meta_info = dataset.load_statistics_from_json(json_path=statistics_json_path)
    print(meta_info)

    # 3. Extract data by key
    dataset.save_to_npy_by_key("rel_actions")
    dataset.load_npy_by_key("rel_actions")
    print(dataset.extracted_data['rel_actions'][23])

    # 4. Reload dataset using extracted key
    dataset_2 = TCLDataset(data_root, use_extracted=True)
    print(dataset.extracted_data['rel_actions'][23])
