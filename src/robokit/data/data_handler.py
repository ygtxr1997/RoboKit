import warnings
import numpy as np
import pickle
from PIL import Image
from io import BytesIO
import os
import multiprocessing as mp


class DataHandler:
    def __init__(self, data_dict: dict):
        """初始化 DataHandler
        Example:
        data_dict = {
            "primary_rgb": (np.random.randn(img_h, img_w, 3) * 255).astype(np.uint8),
            "gripper_rgb": (np.random.randn(img_h, img_w, 3) * 255).astype(np.uint8),
            "primary_depth": (np.random.randn(img_h, img_w) * 255).astype(np.float32),
            "gripper_depth": (np.random.randn(img_h, img_w) * 255).astype(np.float32),
            "language_text": np.array("None"),
            "actions": np.random.randn(7),  # (x,y,z,row,pitch,yaw,g)
            "rel_actions": np.random.randn(7),  # (j_x,j_y,j_z,j_ax,j_ay,j_az,g)
            "robot_obs": np.random.randn(14),
            # (tcp pos (3), tcp ori (3), gripper width (1), joint_states (6) in rad, gripper_action (1)
        }
        """
        self.data_dict = data_dict

        if not self._validate_data(data_dict):
            raise ValueError("数据字典格式无效！")

    def update(self, data_dict: dict):
        self.data_dict = data_dict

    def save(self, file_path: str):
        """保存数据字典为 .npz 格式"""
        # 数据验证：确保字典符合预期格式
        data_dict = self.data_dict

        # 使用 pickle 对数据进行序列化并保存
        pickled_data = {}

        for key, value in data_dict.items():
            if key in ['primary_rgb', 'gripper_rgb']:  # uint8
                pil_data = Image.fromarray(value)
                pickled_data[key] = self._image_to_binary(pil_data, fn_format='JPEG')
            elif key in ['primary_depth', 'gripper_depth']:  # float32
                pil_data = Image.fromarray(value.astype(np.uint16))
                pickled_data[key] = self._image_to_binary(pil_data, fn_format='PNG')
            else:
                # 对其他数据进行 pickle 序列化
                pickled_data[key] = pickle.dumps(value)

        # 使用 numpy 保存数据为 .npz 文件，存储字节流数据
        np.savez_compressed(file_path, **pickled_data)
        print(f"数据已保存为 {file_path}")

    @classmethod
    def load(cls, file_path: str):
        """加载 .npz 文件并恢复数据"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"{file_path} 文件未找到！")

            # 使用 numpy 加载压缩的 .npz 文件
            npzfile = np.load(file_path)

            data_dict = {}
            for key in npzfile.files:
                # 读取并反序列化每个数据项
                if key in ['primary_rgb', 'gripper_rgb',
                           'primary_depth', 'gripper_depth',
                           ]:
                    data_dict[key] = np.array(cls._binary_to_image(npzfile[key]))
                else:
                    data_dict[key] = pickle.loads(npzfile[key])

            handler_instance = cls(data_dict)
            return handler_instance
        except FileNotFoundError as e:
            warnings.warn(f"FileNotFoundError: {e} for {file_path}")
            return None
        except Exception as e:
            warnings.warn(f"File broken or other error: {e} for {file_path}")
            return None

    def get(self, key):
        """获取 .npz 文件中指定项的数据"""
        data_dict = self.data_dict
        if key in data_dict:
            return data_dict[key]
        else:
            raise KeyError(f"字典中没有 {key} 这一项！")

    def _validate_data(self, data_dict):
        """验证数据字典的格式和类型"""
        expected_keys = {
            "primary_rgb": np.ndarray,
            "gripper_rgb": np.ndarray,
            "primary_depth": np.ndarray,
            "gripper_depth": np.ndarray,
            "language_text": np.ndarray,
            "actions": np.ndarray,
            "rel_actions": np.ndarray,
            "robot_obs": np.ndarray,
        }
        if isinstance(data_dict['language_text'], str):
            data_dict['language_text'] = np.array(data_dict['language_text'])

        for key, expected_type in expected_keys.items():
            if key not in data_dict:
                print(f"[Warning] 缺少关键项：{key}")
                continue
            if not isinstance(data_dict[key], expected_type):
                print(f"项 {key} 的类型不匹配，预期类型 {expected_type}，实际类型 {type(data_dict[key])}")
                return False
            if isinstance(data_dict[key], np.ndarray):
                if data_dict[key].ndim == 3 and key in ["primary_rgb", "gripper_rgb"]:
                    pass  # 对应 rgb 数据
                elif data_dict[key].ndim == 2 and key in ["primary_depth", "gripper_depth"]:
                    pass  # 对应深度数据
                elif data_dict[key].shape == (7,) and key in ["actions", "rel_actions"]:
                    pass  # 对应动作数据
                elif data_dict[key].shape == (14,) and key == "robot_obs":
                    pass  # 对应机器人观测数据
                elif data_dict[key].shape == () and key in ["language_text"]:
                    pass  # Language task description
                else:
                    print(f"项 {key} 的形状不匹配！")
                    return False
        return True

    @staticmethod
    def _image_to_binary(pil_image: Image.Image, fn_format='PNG'):
        """将 PIL 图像转换为二进制数据"""
        with BytesIO() as buffer:
            pil_image.save(buffer, format=fn_format)  # 或使用其他格式如 PNG
            binary_data = buffer.getvalue()
        return binary_data

    @staticmethod
    def _binary_to_image(binary_data):
        """将二进制数据转换回 PIL 图像"""
        buffer = BytesIO(binary_data)  # 保持 buffer 在内存中
        pil_image = Image.open(buffer)
        pil_image.load()  # 强制加载图像数据
        return pil_image


class MultiDataHandler:
    def __init__(self):
        self.data_queue = []
        self.save_path_queue = []
        self.max_data_cnt = 10

    def add_data(self, data_dict: dict, save_path: str):
        self.data_queue.append(DataHandler(data_dict))
        self.save_path_queue.append(save_path)

    def save_data(self, is_last=False):
        if len(self.data_queue) >= self.max_data_cnt or is_last:
            for i in range(len(self.save_path_queue)):
                self.data_queue[i].save(self.save_path_queue[i])
            self.reset_data()
        else:
            return

    def reset_data(self):
        self.data_queue = []
        self.save_path_queue = []


class ForkedDataSaver:
    def __init__(self, save_dir="tmp_data", num_workers=None, max_queue_size=5000):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.max_queue_size = max_queue_size
        self.queue = mp.Queue(maxsize=max_queue_size)

        self.num_workers = num_workers or max(1, mp.cpu_count() // 2)
        self.workers = [
            mp.Process(target=self._worker, args=(self.queue,))
            for _ in range(self.num_workers)
        ]
        for w in self.workers:
            w.start()
        print(f"[ForkedDataSaver] Started. num_workers={self.num_workers}.")

    def _worker(self, queue):
        while True:
            item = queue.get()
            if item is None:
                break
            data_dict, file_path = item
            try:
                handler = DataHandler(data_dict)
                handler.save(file_path)
            except Exception as e:
                print(f"[ForkedDataSaver Worker Error] Failed to save {file_path}: {e}")

    def submit(self, data_dict, file_path=None):
        if file_path is None:
            timestamp = datetime.now().strftime("%m%d_%H%M%S_%f")
            file_path = os.path.join(self.save_dir, f"data_{timestamp}.npz")
        try:
            self.queue.put_nowait((data_dict, file_path))
        except mp.queues.Full:
            print(f"[Warning] Save queue full (>{self.max_queue_size}). Data dropped.")
        return file_path

    def close(self):
        for _ in self.workers:
            self.queue.put(None)
        for w in self.workers:
            w.join()
