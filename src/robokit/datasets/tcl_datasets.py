import os
import json
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm
import h5py
from PIL import Image
from io import BytesIO
from torch.utils.data import DataLoader, Dataset

from robokit.data_manager.data_handler import DataHandler


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
        if use_extracted:  # NOTE: needs to be updated
            self.load_npy_by_key("rel_actions")
            # self.load_npy_by_key("robot_obs")
            # self.load_npy_by_key("force_torque")

        if load_keys is None:
            load_keys = ["primary_rgb", "gripper_rgb", "primary_depth", "gripper_depth",
                         "language_text", "actions", "rel_actions", "robot_obs", "force_torque"]
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
            print("[Warning] file broken at:", npz_path, ", load a random one instead.")
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

    # Not used, replaced by `extract_data_and_compute_statistics`
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

    # Not used, replaced by `extract_data_and_compute_statistics`
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

    def extract_data_and_compute_statistics(
            self,
            keys_to_extract: List[str],
            stats_keys: str = None,  # in [`rel_actions`, `robot_obs`, `force_torque`]
            save_json_path: str = None,
            batch_size: int = 256,
            num_workers: int = 16,
            pin_memory: bool = True,
    ) -> dict:
        """
        在一次数据集遍历中同时完成：
        1. 提取指定 keys 的数据并保存为 .npy 文件
        2. 对 stats_keys 中的每个 key 计算统计信息（逐元素/逐通道）

        :param keys_to_extract: 需要提取并保存为 .npy 的 keys 列表
        :param stats_keys: 需要计算统计信息的 keys 列表（如 ['rel_actions', 'robot_obs', 'force_torque']）
        :param save_json_path: 统计信息保存路径（rel path to self.root）or abs path
        """
        if keys_to_extract is None:
            keys_to_extract = ["rel_actions"]
        if stats_keys is None:
            stats_keys = keys_to_extract

        # 校验：所有统计/提取的 key 都必须在 load_keys 中
        for k in set(keys_to_extract) | set(stats_keys):
            if k not in self.load_keys:
                raise ValueError(f"key '{k}' not in load_keys: {self.load_keys}")

        print(f"[TCLDataset] Starting combined extraction and statistics computation...")
        print(f"[TCLDataset] Keys to extract: {keys_to_extract}")
        print(f"[TCLDataset] Stats keys: {stats_keys}")

        # 统计量容器（为每个 stats_key 单独维护）
        # 每个 key 的条目在第一次见到 batch 时按维度初始化
        stats_aggr = {}  # k -> dict(min, max, sum, sqsum, count, n_components)

        # 提取数据的收集容器
        collected_data = {key: [] for key in keys_to_extract}

        # collate：联合需要提取与需要统计的 key，避免二次遍历
        union_keys = list(set(keys_to_extract) | set(stats_keys))

        def collate_fn(batch):
            result = {}
            for key in union_keys:
                result[key] = np.stack([item[key] for item in batch], axis=0)
            return result

        loader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn
        )

        total_count = 0
        for batch_data in tqdm(loader, desc="Processing batches"):
            # 1) 统计
            for key in stats_keys:
                x = batch_data[key]
                if not isinstance(x, np.ndarray):
                    x = x.numpy()
                # 将除 batch 维以外全部展平，统一成 (B, D)
                x = x.reshape(x.shape[0], -1)
                B, D = x.shape

                # 初始化该 key 的聚合器
                if key not in stats_aggr:
                    stats_aggr[key] = {
                        "min": np.full(D, np.inf, dtype=np.float64),
                        "max": np.full(D, -np.inf, dtype=np.float64),
                        "sum": np.zeros(D, dtype=np.float64),
                        "sqsum": np.zeros(D, dtype=np.float64),
                        "count": 0,
                        "n_components": D,
                    }

                ag = stats_aggr[key]
                ag["min"] = np.minimum(ag["min"], x.min(axis=0))
                ag["max"] = np.maximum(ag["max"], x.max(axis=0))
                ag["sum"] += x.sum(axis=0)
                ag["sqsum"] += (x ** 2).sum(axis=0)
                ag["count"] += B

            # 2) 收集需要提取的数据
            for key in keys_to_extract:
                xi = batch_data[key]
                if not isinstance(xi, np.ndarray):
                    xi = xi.numpy()
                collected_data[key].append(xi)

            total_count += next(iter(batch_data.values())).shape[0]

        # 汇总每个 key 的统计结果
        out_stats = {}
        eps = 1e-6  # 定义一个小的 epsilon
        for key, ag in stats_aggr.items():
            # 预处理 min 和 max，处理 min == max 的情况
            is_equal = (ag["min"] == ag["max"])
            ag["max"][is_equal] += eps
            print("[TCLDataset] is_equal sum for key", key, ":", is_equal)

            mean = ag["sum"] / ag["count"]
            var = (ag["sqsum"] / ag["count"]) - (mean ** 2)
            std = np.sqrt(np.maximum(var, 1e-8))
            out_stats[key] = {
                "min": ag["min"].tolist(),
                "max": ag["max"].tolist(),
                "mean": mean.tolist(),
                "std": std.tolist(),
                "n_components": ag["n_components"],
                "total_count": ag["count"],
            }

        statistics = {
            "total_len": total_count,
            "stats": out_stats
        }
        self.meta_statistics = statistics

        # 保存统计信息
        if save_json_path:
            self.save_meta_as_json(save_json_path, statistics)

        # 拼接并保存提取的数据
        for key in keys_to_extract:
            all_data = np.concatenate(collected_data[key], axis=0)
            save_path = os.path.join(self.root, f"extracted/{key}.npy")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, all_data)
            print(f"[TCLDataset] Key '{key}' shape={all_data.shape} saved to '{save_path}'")
            self.extracted_data[key] = all_data
            print(f"[TCLDataset] Key '{key}' loaded to extracted_data")

        print(f"[TCLDataset] Combined extraction and statistics computation completed!")
        return statistics

    def save_meta_as_json(self, json_path: str, meta_statistics: dict = None, force_path: bool = False):
        if os.path.isabs(json_path) or force_path:
            save_path = json_path
        else:
            save_path = os.path.join(getattr(self, 'root', ''), json_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if meta_statistics is None:
            meta_statistics = self.meta_statistics
        with open(save_path, 'w') as fp:
            json.dump(meta_statistics, fp, indent=4)
        print(f"[TCLDataset] Statistics saved to: {save_path}")

    def load_meta_from_json(self, json_path: str) -> dict:
        if os.path.isabs(json_path):
            pass
        else:
            json_path = os.path.join(self.root, json_path)
        print("[TCLDataset] loading dataset statistics from:", json_path)
        with open(json_path, 'r') as json_file:
            statistics = json.load(json_file)
        return statistics


class TCLDatasetHDF5(TCLDataset):
    def __init__(self, root: str, h5_path: str, keys_config: Dict[str, str] = None, use_h5: bool = True,
                 use_extracted: bool = False, load_keys: List[str] = None, is_img_decoded_in_h5: bool = False):
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
            "robot_obs": "float",
            "force_torque": "float",
        }
        self.keys_config = keys_config  # 例如 {"primary_rgb": "rgb", "language_text": "string", ...}

        # 如果 h5_path 存在，提前加载 HDF5 文件
        self.use_h5 = use_h5
        self.dsets = {}
        if os.path.exists(self.h5_path) and use_h5:
            self._load_hdf5_file()
            print("[TCLDatasetHDF5] using h5 data_manager:", self.h5_path, ". total_len:", self.total_length)

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
    def binary_to_image(binary_data: bytes, decode_image: bool = True):
        """将二进制数据转换回 PIL 图像"""
        buffer = BytesIO(binary_data)  # 保持 buffer 在内存中
        if decode_image:
            pil_image = Image.open(buffer)
            pil_image.load()  # 强制加载图像数据
            return pil_image
        else:
            return np.array(binary_data)

    @staticmethod
    def image_to_binary(pil_image: Image.Image, fn_format='JPEG') -> bytes:
        """将 PIL 图像转换为二进制数据"""
        with BytesIO() as buffer:
            pil_image.save(buffer, format=fn_format)  # 或使用其他格式如 PNG
            binary_data = buffer.getvalue()
        return binary_data

    def convert_to_hdf5(self, batch_size: int = 16, num_workers: int = 0, pin_memory: bool = False,
                        resize_wh: Tuple[int, int] = None):
        """
        使用 DataLoader 批量读取并将整个数据集写入 HDF5 文件。
        :param batch_size: 每个批次的大小
        :param num_workers: 使用的工作进程数量
        :param pin_memory: 是否启用内存锁定
        """
        print(f"[TCLDatasetHDF5] Saving data_manager to HDF5 at {self.h5_path}")
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
                        if not resize_wh:  # Op1. no need to resize
                            out[k] = np.array([
                                np.frombuffer(self._extract_image_bytes(x), dtype=np.uint8)
                                for x in items
                            ], dtype=object)
                        else:  # Op2. needs resize
                            out[k] = np.array([
                                np.frombuffer(
                                    self.image_to_binary(
                                        self.binary_to_image(self._extract_image_bytes(x)).resize(resize_wh),
                                        fn_format='JPEG' if kind == 'rgb' else 'png'
                                    ),
                                    dtype=np.uint8
                                )
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
        for batch in tqdm(loader, desc=f"Writing HDF5, wxh={resize_wh}"):
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
            self.total_length = max(len(self.dsets[key]), self.total_length)  # modify total length

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

    # 1. Load data_manager
    dataset = TCLDataset(data_root, use_extracted=False)
    dataset.__getitem__(0)
    # dataset.total_length = 100  # For debug

    # 2. Save and load statistics info
    print("Save and load statistics info")
    statistics_json_path = "statistics.json"
    _ = dataset.get_statistics_and_save(save_json_path=statistics_json_path)
    meta_info = dataset.load_statistics_from_json(json_path=statistics_json_path)
    print(meta_info)

    # 3. Extract data_manager by key
    print("Extract data_manager by key")
    dataset.save_to_npy_by_key("rel_actions")
    dataset.load_npy_by_key("rel_actions")
    print(dataset.extracted_data['rel_actions'][23])

    # 4. Reload dataset using extracted key
    dataset_2 = TCLDataset(data_root, use_extracted=True)
    print(dataset.extracted_data['rel_actions'][23])
