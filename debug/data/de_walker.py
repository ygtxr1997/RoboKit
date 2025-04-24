import os
import numpy as np
from torch.utils.data import DataLoader, Dataset

from robokit.data import DataHandler
from robokit.debug_utils import print_batch


class TCLDataset(Dataset):
    def __init__(self, root):
        super(TCLDataset, self).__init__()
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


if __name__ == "__main__":
    dataset = TCLDataset("F:\datasets\collected_data")
    dataset.__getitem__(0)
