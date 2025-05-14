import warnings
from typing import List
import numpy as np
import pickle
from PIL import Image
from io import BytesIO
import os
import multiprocessing as mp
from datetime import datetime
import time
import cv2
from collections import deque


class DataHandler:
    def __init__(self, data_dict: dict, verbose=False):
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

        if not self._validate_data(data_dict, verbose=verbose):
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
        print(f"Data saved as: {file_path}")

    @classmethod
    def load(cls, file_path: str, load_keys: List[str], decode_image: bool = True):
        """加载 .npz 文件并恢复数据"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"{file_path} 文件未找到！")

            # 使用 numpy 加载压缩的 .npz 文件
            npzfile = np.load(file_path)

            data_dict = {}
            for key in npzfile.files:
                if key not in load_keys:
                    continue  # skip some unused keys
                # 读取并反序列化每个数据项
                if key in ['primary_rgb', 'gripper_rgb',
                           'primary_depth', 'gripper_depth',
                           ]:
                    data_dict[key] = np.array(cls._binary_to_image(npzfile[key], decode_image=decode_image))
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

    def _validate_data(self, data_dict, verbose=False):
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
        if 'language_text' in data_dict.keys() and isinstance(data_dict['language_text'], str):
            data_dict['language_text'] = np.array(data_dict['language_text'])

        for key, expected_type in expected_keys.items():
            if key not in data_dict:
                if verbose:
                    print(f"[Warning] missing：{key}")
                continue
            if not isinstance(data_dict[key], expected_type):
                print(f"项 {key} 的类型不匹配，预期类型 {expected_type}，实际类型 {type(data_dict[key])}")
                return False
            if isinstance(data_dict[key], np.ndarray):
                if data_dict[key].ndim <= 3 and key in ["primary_rgb", "gripper_rgb"]:
                    pass  # 对应 rgb 数据
                elif data_dict[key].ndim <= 2 and key in ["primary_depth", "gripper_depth"]:
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
    def _binary_to_image(binary_data, decode_image: bool = True):
        """将二进制数据转换回 PIL 图像"""
        buffer = BytesIO(binary_data)  # 保持 buffer 在内存中
        if decode_image:
            pil_image = Image.open(buffer)
            pil_image.load()  # 强制加载图像数据
            return pil_image
        else:
            return np.array(binary_data)


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
    def __init__(self, num_workers=None, max_queue_size=5000):
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
                print("[ForkedDataSaver] item is None, existing loop!")
                break
            data_dict, file_path = item
            try:
                handler = DataHandler(data_dict)
                handler.save(file_path)
            except Exception as e:
                print(f"[ForkedDataSaver Worker Error] Failed to save {file_path}: {e}")

    def submit(self, data_dict, saving_dir, file_path=None):
        if file_path is None:
            timestamp = datetime.now().strftime("%m%d_%H%M%S_%f")
            file_path = os.path.join(saving_dir, f"{timestamp}.npz")
        try:
            self.queue.put_nowait((data_dict, file_path))
        except mp.queues.Full:
            print(f"[Warning] Save queue full (>{self.max_queue_size}). Data dropped.")
        return file_path

    def save_remaining(self, check_interval=0.1, verbose=True):
        while True:
            remaining = self.queue.qsize()
            if remaining == 0:
                if verbose:
                    print("[ForkedDataSaver] Queue fully processed. No remaining data.")
                break
            if verbose:
                print(f"[ForkedDataSaver] Waiting... {remaining} item(s) remaining in queue.")
            time.sleep(check_interval)

    def close(self):
        remaining = self.queue.qsize()
        print(f"[ForkedDataSaver] Closing. Remaining: {remaining}")

        for _ in self.workers:
            self.queue.put(None)
        for w in self.workers:
            w.join()

        print("[ForkedDataSaver] All saver workers exited.")


class ImageAsVideoSaver:
    def __init__(self, buffer_size=10, frame_rate=30, width=640, height=480):
        # 初始化类，设置最大缓存大小、帧率、视频分辨率等
        self.buffer_size = buffer_size
        self.frame_rate = frame_rate
        self.width = width
        self.height = height

        # 使用队列来存储图像
        self.image_queue = deque(maxlen=self.buffer_size)

    def add_image(self, image: np.ndarray):
        """
        向队列中添加图像
        :param image: 要添加到视频队列中的图像，应该是 numpy 数组形式
        """
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # 确保图像大小为 (height, width, channels)，并且图像是 BGR 格式
        if image.shape[0] != self.height or image.shape[1] != self.width:
            image = cv2.resize(image, (self.width, self.height))

        self.image_queue.append(image)

    def save_to_video(self, path: str):
        """
        将队列中的图像保存为视频
        :param path: 保存视频的路径（例如: 'output.avi'）
        """
        if len(self.image_queue) == 0:
            print("No images in the queue to save.")
            return

        # 获取视频的编码器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # 创建 VideoWriter 对象
        out = cv2.VideoWriter(path, fourcc, self.frame_rate, (self.width, self.height))

        # 写入图像队列中的每一帧
        for img in self.image_queue:
            out.write(img)

        # 释放 VideoWriter
        out.release()
        print(f"Video saved at {path}")
