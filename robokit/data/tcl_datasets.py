import os
import numpy as np
from torch.utils.data import DataLoader, Dataset

from robokit.data.data_handler import DataHandler


class TCLDataset(Dataset):
    def __init__(self, root):
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

        print("[TCLDataset] total length:", self.total_length)

    def get_tasks(self, root):
        self.root = root
        tasks = []
        for task in os.listdir(root):
            if not os.path.isdir(os.path.join(root, task)):
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


if __name__ == "__main__":
    dataset = TCLDataset("F:\datasets\collected_data")
    dataset.__getitem__(0)
