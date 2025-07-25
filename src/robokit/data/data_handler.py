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
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter



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


class ActionAsVideoSaver:
    def __init__(self, buffer_size=200, frame_rate=30, width=12, height=8,
                 action_names=None, window_size=100):
        """
        初始化ActionAsVideoSaver - 延迟创建图形以节省内存

        动作可视化约定：
        - 前3维 (XYZ): 用实线显示，代表位置信息
        - 后4维 (RPY + Gripper): 用虚线显示，代表旋转和夹爪信息

        Args:
            buffer_size: 最大缓存的动作序列数量
            frame_rate: 视频帧率
            width: 图形宽度 (英寸)
            height: 图形高度 (英寸)
            action_names: 7个动作维度的名称列表
            window_size: 时间序列图显示的时间窗口大小
        """
        self.buffer_size = buffer_size
        self.frame_rate = frame_rate
        self.window_size = window_size
        self.width = width
        self.height = height

        # 7维动作的默认名称
        if action_names is None:
            self.action_names = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw', 'Gripper']
        else:
            assert len(action_names) == 7, "action_names must have exactly 7 elements"
            self.action_names = action_names

        # 使用队列存储动作序列
        self.action_queue = deque(maxlen=self.buffer_size)

        # 设置颜色和线型
        self.colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
        self.linestyles = ['-', '-', '-', '--', '--', '--', '--']  # 前3个实线，后4个虚线
        self.linewidths = [2.5, 2.5, 2.5, 2.0, 2.0, 2.0, 2.0]  # 前3个稍粗

        # 所有7个维度在一个图中显示
        self.all_dimensions = list(range(7))  # [0, 1, 2, 3, 4, 5, 6]

        # Y轴范围设置
        self.y_range = [-0.5, 0.5]  # 统一的Y轴范围

        # 延迟创建matplotlib对象
        self.fig = None
        self.ax = None
        self.lines = {}
        self.current_frame = 0

    def add_action(self, action: np.ndarray):
        """向队列中添加一个7维动作"""
        action = np.array(action)
        assert action.shape == (7,), f"Action must be 7-dimensional, got shape {action.shape}"
        self.action_queue.append(action.copy())

    def add_action_sequence(self, actions: np.ndarray):
        """批量添加动作序列"""
        actions = np.array(actions)
        assert actions.shape[1] == 7, f"Actions must have 7 dimensions, got shape {actions.shape}"

        for action in actions:
            self.add_action(action)

    def _setup_figure(self):
        """延迟创建matplotlib图形"""
        if self.fig is not None:
            return

        # 设置非交互式backend避免内存问题
        matplotlib.use('Agg')

        # 创建图形和子图 - 只有一个子图显示所有7个维度
        self.fig, self.ax = plt.subplots(1, 1, figsize=(self.width, self.height))
        self.lines = {}

    def _init_animation(self):
        """初始化动画图表"""
        self._setup_figure()

        self.ax.clear()

        # 为所有7个动作维度创建线条
        for action_idx in self.all_dimensions:
            line, = self.ax.plot([], [],
                                 label=self.action_names[action_idx],
                                 color=self.colors[action_idx],
                                 linestyle=self.linestyles[action_idx],
                                 linewidth=self.linewidths[action_idx])
            self.lines[action_idx] = line

        # 设置图表属性
        self.ax.set_xlim(0, self.window_size)
        self.ax.set_ylim(self.y_range)
        self.ax.set_title('7D Robot Action Sequence (XYZ: solid, RPY&G: dashed)',
                          fontsize=12, fontweight='bold')
        self.ax.set_xlabel('Time Steps')
        self.ax.set_ylabel('Action Value')
        self.ax.legend(loc='lower left', ncol=2)
        self.ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return list(self.lines.values())

    def _update_animation(self, frame):
        """更新动画帧"""
        if len(self.action_queue) == 0:
            return list(self.lines.values())

        # 确定当前显示的数据范围
        end_idx = min(frame + 1, len(self.action_queue))
        start_idx = max(0, end_idx - self.window_size)

        if end_idx > start_idx:
            # 获取当前窗口的动作数据
            window_actions = np.array(list(self.action_queue)[start_idx:end_idx])
            time_steps = np.arange(start_idx, end_idx)

            # 更新每条线的数据
            for action_idx in range(7):
                if action_idx in self.lines:
                    self.lines[action_idx].set_data(time_steps, window_actions[:, action_idx])

            # 动态调整X轴范围
            if end_idx > self.window_size:
                self.ax.set_xlim(start_idx, end_idx)
            else:
                self.ax.set_xlim(0, self.window_size)

        return list(self.lines.values())

    def save_to_video(self, path: str, show_progress=True):
        """
        将动作序列保存为视频

        Args:
            path: 保存视频的路径（例如: 'actions.mp4'）
            show_progress: 是否显示进度信息
        """
        if len(self.action_queue) == 0:
            print("No actions in the queue to save.")
            return

        # Save action .npy
        action_npy = np.array(list(self.action_queue))
        npy_save_path = path.replace('.mp4', '.npy')
        np.save(npy_save_path, action_npy)

        if show_progress:
            print(f"Creating video with {len(self.action_queue)} frames...")

        # 确保图形已创建
        self._setup_figure()

        # 创建动画
        anim = FuncAnimation(
            self.fig,
            self._update_animation,
            init_func=self._init_animation,
            frames=len(self.action_queue),
            interval=1000 // self.frame_rate,  # 转换为毫秒
            blit=False,
            repeat=False
        )

        # 保存为视频
        writer = FFMpegWriter(
            fps=self.frame_rate,
            metadata=dict(artist='ActionAsVideoSaver'),
            bitrate=1250
        )



        try:
            anim.save(path, writer=writer, progress_callback=lambda i, n:
            print(f"Progress: {i + 1}/{n} frames") if show_progress and (i + 1) % 20 == 0 else None)
            if show_progress:
                print(f"Video saved successfully at {path}")
        except Exception as e:
            print(f"Error saving video: {e}")
            print("Make sure you have ffmpeg installed: conda install ffmpeg")
        finally:
            # 清理资源
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self.lines = {}

    def preview_animation(self):
        """预览动画（在jupyter notebook中很有用）"""
        if len(self.action_queue) == 0:
            print("No actions in the queue to preview.")
            return None

        # 确保图形已创建
        self._setup_figure()

        anim = FuncAnimation(
            self.fig,
            self._update_animation,
            init_func=self._init_animation,
            frames=len(self.action_queue),
            interval=1000 // self.frame_rate,
            blit=False,
            repeat=True
        )

        plt.tight_layout()
        plt.show()
        return anim

    def clear_buffer(self):
        """清空动作缓存"""
        self.action_queue.clear()

    def get_buffer_size(self):
        """获取当前缓存中的动作数量"""
        return len(self.action_queue)

    def set_y_range(self, y_min, y_max):
        """
        设置Y轴范围
        Args:
            y_min: Y轴最小值
            y_max: Y轴最大值
        """
        self.y_range = [y_min, y_max]

    def __del__(self):
        """析构函数，清理matplotlib资源"""
        if self.fig is not None:
            plt.close(self.fig)


class ActionAsVideoSaverV1:
    def __init__(self, buffer_size=100, frame_rate=30, width=800, height=600,
                 action_names=None, window_size=50):
        """
        初始化ActionAsVideoSaver

        动作可视化约定：
        - 前3维 (XYZ): 用实线显示，代表位置信息
        - 后4维 (RPY + Gripper): 用虚线显示，代表旋转和夹爪信息

        Args:
            buffer_size: 最大缓存的动作序列数量
            frame_rate: 视频帧率
            width: 视频宽度
            height: 视频高度
            action_names: 7个动作维度的名称列表，如果为None则使用默认名称
            window_size: 时间序列图显示的时间窗口大小
        """
        self.buffer_size = buffer_size
        self.frame_rate = frame_rate
        self.width = width
        self.height = height
        self.window_size = window_size

        # 7维动作的默认名称
        if action_names is None:
            self.action_names = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw', 'Gripper']
        else:
            assert len(action_names) == 7, "action_names must have exactly 7 elements"
            self.action_names = action_names

        # 使用队列存储动作序列
        self.action_queue = deque(maxlen=self.buffer_size)

        # 设置matplotlib参数
        plt.style.use('default')
        self.colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']

    def add_action(self, action: np.ndarray):
        """
        向队列中添加一个7维动作

        Args:
            action: 7维numpy数组，表示一个时刻的动作
        """
        action = np.array(action)
        assert action.shape == (7,), f"Action must be 7-dimensional, got shape {action.shape}"

        self.action_queue.append(action.copy())

    def add_action_sequence(self, actions: np.ndarray):
        """
        批量添加动作序列

        Args:
            actions: shape为(sequence_length, 7)的numpy数组
        """
        actions = np.array(actions)
        assert actions.shape[1] == 7, f"Actions must have 7 dimensions, got shape {actions.shape}"

        for action in actions:
            self.add_action(action)

    def _create_frame(self, current_step):
        """
        创建当前时刻的可视化帧

        Args:
            current_step: 当前步数

        Returns:
            numpy数组形式的图像帧
        """
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.width / 100, self.height / 100), dpi=100)

        # 获取当前窗口的数据
        start_idx = max(0, current_step - self.window_size)
        end_idx = current_step + 1

        if end_idx <= len(self.action_queue):
            window_actions = list(self.action_queue)[start_idx:end_idx]
            window_actions = np.array(window_actions)
            time_steps = np.arange(start_idx, end_idx)

            # 上半部分：时间序列图
            ax1.set_title('Action Time Series (XYZ: solid lines, RPY&Gripper: dashed lines)', fontsize=12,
                          fontweight='bold')
            for i in range(7):
                # 前3维(XYZ)用实线，后4维(RPY和Gripper)用虚线
                linestyle = '-' if i < 3 else '--'
                linewidth = 2.5 if i < 3 else 2.0

                ax1.plot(time_steps, window_actions[:, i],
                         color=self.colors[i], label=self.action_names[i],
                         linewidth=linewidth, linestyle=linestyle)

            # 标记当前时刻
            if len(window_actions) > 0:
                current_action = window_actions[-1]
                for i in range(7):
                    ax1.scatter(current_step, current_action[i],
                                color=self.colors[i], s=50, zorder=5)

            ax1.set_xlabel('Time Step')
            ax1.set_ylabel('Action Value')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)

            # 下半部分：当前动作的条形图
            ax2.set_title('Current Action Values', fontsize=14, fontweight='bold')
            if len(window_actions) > 0:
                current_action = window_actions[-1]
                bars = ax2.bar(range(7), current_action, color=self.colors, alpha=0.7)

                # 添加数值标签
                for i, (bar, value) in enumerate(zip(bars, current_action)):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                             f'{value:.3f}', ha='center', va='bottom', fontsize=10)

            ax2.set_xlabel('Action Dimension')
            ax2.set_ylabel('Value')
            ax2.set_xticks(range(7))
            ax2.set_xticklabels(self.action_names, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3, axis='y')

        # 调整布局
        plt.tight_layout()

        # 将图像转换为numpy数组
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)

        # 读取图像数据
        img_arr = plt.imread(buf)
        plt.close(fig)

        # 转换为BGR格式并调整大小
        if img_arr.dtype == np.float32 or img_arr.dtype == np.float64:
            img_arr = (img_arr * 255).astype(np.uint8)

        img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
        img_bgr = cv2.resize(img_bgr, (self.width, self.height))

        return img_bgr

    def save_to_video(self, path: str):
        """
        将动作序列保存为视频

        Args:
            path: 保存视频的路径（例如: 'actions.mp4'）
        """
        if len(self.action_queue) == 0:
            print("No actions in the queue to save.")
            return

        # 获取视频编码器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # 创建VideoWriter对象
        out = cv2.VideoWriter(path, fourcc, self.frame_rate, (self.width, self.height))

        print(f"Creating video with {len(self.action_queue)} frames...")

        # 为每个时间步创建帧
        for step in range(len(self.action_queue)):
            frame = self._create_frame(step)
            out.write(frame)

            # 显示进度
            if (step + 1) % 10 == 0:
                print(f"Processed {step + 1}/{len(self.action_queue)} frames")

        # 释放VideoWriter
        out.release()
        print(f"[ActionAsVideoSaver] Video saved at {path}")

    def clear_buffer(self):
        """清空动作缓存"""
        self.action_queue.clear()

    def get_buffer_size(self):
        """获取当前缓存中的动作数量"""
        return len(self.action_queue)
