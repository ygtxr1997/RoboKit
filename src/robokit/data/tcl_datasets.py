import os
import json
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm
import h5py
from PIL import Image
from io import BytesIO
from torch.utils.data import DataLoader, Dataset

from robokit.data.data_handler import DataHandler


class TCLDataset(Dataset):
    def __init__(self, root,
                 use_extracted: bool = False,
                 load_keys: List[str] = None,
                 needs_decode_image: bool = True
                 ):
        super(TCLDataset, self).__init__()
        self.root = root
        self.needs_decode_image = needs_decode_image

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
        data_handler = DataHandler.load(file_path=npz_path, load_keys=self.load_keys,
                                        decode_image=self.needs_decode_image)
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


class TCLDatasetHDF5(TCLDataset):
    def __init__(self, root: str, h5_path: str, keys_config: Dict[str, str] = None, use_h5: bool = True,
                 use_extracted: bool = True, load_keys: List[str] = None, is_img_decoded_in_h5: bool = False):
        """
        初始化 TCLDatasetHDF5 数据集对象
        :param root: `.npz` 数据集的根目录
        :param h5_path: 输出的 HDF5 文件路径
        :param keys_config: 键配置，指定每个字段的类型（如 'rgb', 'depth', 'string' 等）
        """
        super().__init__(root, use_extracted, load_keys, is_img_decoded_in_h5)
        self.h5_path = h5_path
        self.is_img_decoded_in_h5 = is_img_decoded_in_h5

        keys_config = keys_config or {
            "primary_rgb": "rgb",
            "gripper_rgb": "rgb",
            "primary_depth": "depth",
            "gripper_depth": "depth",
            "language_text": "string",
            "actions": "float",
            "rel_actions": "float",
            "robot_obs": "float"
        }
        self.keys_config = keys_config  # 例如 {"primary_rgb": "rgb", "language_text": "string", ...}

        # 如果 h5_path 存在，提前加载 HDF5 文件
        self.use_h5 = use_h5
        self.dsets = {}
        if os.path.exists(self.h5_path) and use_h5:
            print("[TCLDatasetHDF5] using h5 data:", self.h5_path)
            self._load_hdf5_file()

    @staticmethod
    def _extract_image_bytes(raw):
        """
        提取图像字节流（如 BytesIO 或 ndarray）
        :param raw: 输入的数据，可以是 BytesIO 对象或者 ndarray
        :return: 二进制字节流
        """
        if isinstance(raw, np.ndarray):
            raw = raw.item() if raw.shape == () else raw

        if isinstance(raw, BytesIO):
            return raw.getvalue()
        elif isinstance(raw, bytes):
            return raw
        elif isinstance(raw, np.ndarray):
            return raw.tobytes()
        else:
            raise ValueError(f"Unexpected raw type: {type(raw)}")

    @staticmethod
    def encode_fixed_string(s: str, max_len: int = 300, end_token: bytes = b'#') -> bytes:
        """
        将字符串编码为定长的 S300 字符串，填充空格并添加结束符号
        :param s: 输入字符串
        :param max_len: 最大长度
        :param end_token: 结束符
        :return: 定长编码的字节流
        """
        s = s.replace('\0', '')  # 清理 NULL 字符
        s_trim = s[: max_len - 1]
        return (s_trim + end_token.decode('utf-8')).ljust(max_len, ' ').encode('utf-8')

    @staticmethod
    def binary_to_image(binary_data, decode_image: bool = True):
        """将二进制数据转换回 PIL 图像"""
        buffer = BytesIO(binary_data)  # 保持 buffer 在内存中
        if decode_image:
            pil_image = Image.open(buffer)
            pil_image.load()  # 强制加载图像数据
            return pil_image
        else:
            return np.array(binary_data)

    def convert_to_hdf5(self, batch_size: int = 16, num_workers: int = 0, pin_memory: bool = False,
                        resize_wh: Tuple[int, int] = None):
        """
        使用 DataLoader 批量读取并将整个数据集写入 HDF5 文件。
        :param batch_size: 每个批次的大小
        :param num_workers: 使用的工作进程数量
        :param pin_memory: 是否启用内存锁定
        """
        print(f"[TCLDatasetHDF5] Saving data to HDF5 at {self.h5_path}")
        os.makedirs(os.path.dirname(self.h5_path), exist_ok=True)

        # 1. 从第一个样本推断每个字段的 dtype 和 shape
        first = self.load_single_frame(os.path.join(self.root, self.ep_fns[0]))
        dtypes, shapes = {}, {}
        for k in self.load_keys:
            val = first[k]
            kind = self.keys_config.get(k, 'float')

            if kind == 'string':
                dtypes[k] = 'S300'
                shapes[k] = (self.total_length,)
            elif kind in ('rgb', 'depth'):
                if not self.is_img_decoded_in_h5:
                    b = self._extract_image_bytes(val)
                    dtypes[k] = h5py.vlen_dtype(np.uint8)
                    shapes[k] = (self.total_length,)
                else:
                    resize_wh = resize_wh or Image.fromarray(val).size
                    val = Image.fromarray(val).resize(size=resize_wh)
                    arr = np.array(val)
                    dtypes[k] = arr.dtype  # 注意此时是 uint8，不是 vlen
                    shapes[k] = (self.total_length, *arr.shape)  # 例如 (N, 480, 848, 3)
            else:
                arr = np.asarray(val)
                dtypes[k] = arr.dtype
                shapes[k] = (self.total_length,) + arr.shape

        # 2. 创建 HDF5 文件和数据集
        hf = h5py.File(self.h5_path, 'w')
        datasets = {
            k: hf.create_dataset(k, shape=shapes[k], dtype=dtypes[k], compression='lzf')
            for k in self.load_keys
        }

        # 3. 定义 DataLoader 和 collate_fn
        def collate_fn(batch):
            out = {}
            for k in self.load_keys:
                kind = self.keys_config.get(k, 'float')
                items = [b[k] for b in batch]

                if kind == 'string':
                    out[k] = np.array([
                        self.encode_fixed_string(str(x)) for x in items
                    ], dtype='S300')
                elif kind in ('rgb', 'depth'):
                    if not self.is_img_decoded_in_h5:
                        out[k] = np.array([
                            np.frombuffer(self._extract_image_bytes(x), dtype=np.uint8)
                            for x in items
                        ], dtype=object)
                    else:
                        # 解码后的 RGB 应该是 (480, 848, 3)，堆叠为 (B, 480, 848, 3)
                        out[k] = np.stack([np.array(Image.fromarray(x).resize(resize_wh)) for x in items], axis=0)
                else:
                    out[k] = np.stack(items, axis=0)
            return out

        loader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn
        )

        # 4. 批量写入 HDF5
        idx = 0
        for batch in tqdm(loader, desc="Writing HDF5 batches"):
            bsize = next(iter(batch.values())).shape[0]
            for k, data in batch.items():
                datasets[k][idx:idx + bsize] = data
            idx += bsize

        hf.close()
        print(f"[TCLDatasetHDF5] Data successfully saved to {self.h5_path}")

    def _load_hdf5_file(self):
        """加载 HDF5 文件并将每个字段的 dataset 存储到 dsets 字典中"""
        self.hf = h5py.File(self.h5_path, 'r')  # 这里打开文件并保留文件句柄
        for key in self.keys_config:
            self.dsets[key] = self.hf[key]

    def __getitem__(self, index: int):
        """根据索引从 HDF5 文件中读取数据"""
        sample = {}

        # 如果有 h5_path，且已经加载过数据，则从 HDF5 读取
        if hasattr(self, 'h5_path') and self.h5_path and hasattr(self, 'dsets') and self.dsets:
            # 使用加载的 HDF5 数据集 dsets
            for key in self.load_keys:
                kind = self.keys_config[key]
                data = self.dsets[key][index]  # 从已加载的 HDF5 数据集读取数据

                if kind == "string":
                    sample[key] = data.decode('utf-8').strip()  # 还原为原始字符串，去掉填充的空格和结束符
                elif kind in ('rgb', 'depth'):
                    if not self.is_img_decoded_in_h5:
                        sample[key] = np.array(self.binary_to_image(data))
                    else:
                        sample[key] = np.array(data)
                else:
                    sample[key] = data
        else:
            # 如果没有加载 HDF5 文件或没有指定 h5_path，则回退到从 .npz 读取数据
            sample = super().__getitem__(index)

        return sample


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
