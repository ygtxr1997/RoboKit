from abc import abstractmethod, ABC
import math
from typing import List, Dict, Tuple, Iterable

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.animation import FuncAnimation


def concatenate_rgb_images(img1, img2, vertical=False, resize_ratio=1.):
    """
    拼接两个RGB图像（左右拼接）

    参数：
    img1 (np.ndarray): 第一个RGB图像，形状为 (height, width, 3)
    img2 (np.ndarray): 第二个RGB图像，形状为 (height, width, 3)

    返回：
    np.ndarray: 拼接后的图像，形状为 (height, width1 + width2, 3)
    """
    height = img1.shape[0]
    width = img1.shape[1]
    img1 = np.array(Image.fromarray(img1).resize((int(width * resize_ratio), int(height * resize_ratio))))
    hw_ratio2 = float(img2.shape[0]) / float(img2.shape[1])

    if not vertical:
        # 确保两个图像的高度相同
        img2 = np.array(Image.fromarray(img2).resize((int(int(height / hw_ratio2) * resize_ratio),
                                                      int(height * resize_ratio))))
        # 使用numpy的hstack来拼接两个图像
        return np.hstack((img1, img2))
    else:
        # 确保两个图像的宽度相同
        img2 = np.array(Image.fromarray(img2).resize((int(width * resize_ratio),
                                                      int(int(width * hw_ratio2) * resize_ratio))))
        # 使用numpy的vstack来拼接两个图像
        return np.vstack((img1, img2))


def plot_action_wrt_time(action_data: np.ndarray, only_last_frame: bool = False):
    """
    Plot action with respect to time

    :param action_data: (T, 7) array with format (x, y, z, a, b, c, g)
    :return: (frames, fig, (ax1, ax2))
        frames: A list of RGB images for animation
        fig: matplotlib figure object
        (ax1, ax2): tuple of axes objects
    """
    frames_cnt = action_data.shape[0]
    action_dim = action_data.shape[1]
    assert action_dim >= 3, "Action data_manager must have at least 3 dimensions"

    # 创建一个图和两个子图（2行1列）
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # --- 子图1：x, y, z ---
    labels_xyz = ['x', 'y', 'z']
    colors_xyz = ['red', 'green', 'blue']
    styles_xyz = ['solid', 'solid', 'solid']
    lines_xyz = []
    for label, color, style in zip(labels_xyz, colors_xyz, styles_xyz):
        line, = ax1.plot([], [], label=label, color=color, linestyle=style, linewidth=2)
        lines_xyz.append(line)

    ax1.set_xlim(0, frames_cnt)
    # 设置y轴范围，留出10%的边距
    xyz_min = np.min(action_data[:, :3])
    xyz_max = np.max(action_data[:, :3])
    xyz_margin = 0.1 * (xyz_max - xyz_min) if xyz_max != xyz_min else 0.1
    ax1.set_ylim(xyz_min - xyz_margin, xyz_max + xyz_margin)
    ax1.set_xlabel('Time (Frames)')
    ax1.set_ylabel('Position')
    ax1.set_title('XYZ Position')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # --- 子图2：a, b, c, g ---
    labels_abcg = ['a', 'b', 'c', 'g'][:(action_dim - 3)]
    colors_abcg = ['darkorange', 'purple', 'brown', 'black'][:(action_dim - 3)]
    styles_abcg = ['dashed', 'dashed', 'dashed', 'dotted'][:(action_dim - 3)]
    lines_abcg = []
    for label, color, style in zip(labels_abcg, colors_abcg, styles_abcg):
        line, = ax2.plot([], [], label=label, color=color, linestyle=style, linewidth=2)
        lines_abcg.append(line)

    ax2.set_xlim(0, frames_cnt)
    # 设置y轴范围，留出10%的边距
    abcg_min = np.min(action_data[:, 3:])
    abcg_max = np.max(action_data[:, 3:])
    abcg_margin = 0.1 * (abcg_max - abcg_min) if abcg_max != abcg_min else 0.1
    ax2.set_ylim(abcg_min - abcg_margin, abcg_max + abcg_margin)
    ax2.set_xlabel('Time (Frames)')
    ax2.set_ylabel('Orientation/Gripper')
    ax2.set_title('ABCG Parameters')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # 帧图列表
    frames = []

    if not only_last_frame:
        print("Plotting action dynamic figures...")
    frame_indices = [frames_cnt - 1] if only_last_frame else range(frames_cnt)
    for frame_num in frame_indices:
    # for frame_num in range(frames_cnt):
        # 更新 xyz 曲线
        for i, line in enumerate(lines_xyz):
            line.set_data(np.arange(frame_num + 1), action_data[:frame_num + 1, i])

        # 更新 abcg 曲线
        for i, line in enumerate(lines_abcg):
            line.set_data(np.arange(frame_num + 1), action_data[:frame_num + 1, i + 3])

        # 显示当前值
        if frame_num > 0:
            # 清除之前的文本
            for txt in ax1.texts:
                txt.remove()
            for txt in ax2.texts:
                txt.remove()

            # 显示当前xyz值
            current_xyz_text = f'Current: x={action_data[frame_num, 0]:.3f}, y={action_data[frame_num, 1]:.3f}, z={action_data[frame_num, 2]:.3f}'
            ax1.text(0.02, 0.95, current_xyz_text, transform=ax1.transAxes,
                     fontsize=9, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # 显示当前abcg值
            current_abcg_values = [f'{labels_abcg[i]}={action_data[frame_num, i + 3]:.3f}'
                                   for i in range(len(labels_abcg))]
            current_abcg_text = f'Current: {", ".join(current_abcg_values)}'
            ax2.text(0.02, 0.95, current_abcg_text, transform=ax2.transAxes,
                     fontsize=9, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(img)

    plt.close(fig)
    return frames, fig, (ax1, ax2)


def save_frames_as_video(
    frames: List[np.ndarray],
    save_path: str,
    fps: int = 30,
    codec: str = None,                 # 手动指定编码器，例如 "mp4v" / "avc1" / "XVID"
    target_size: Tuple[int, int] = None,  # (width, height)，不传则用首帧尺寸
):
    """
    将一组 RGB 帧写成视频文件（支持帧尺寸不一致、float/uint8、灰度/RGBA）。
    - 自动把 float[0,1] 或其他数值类型规范化成 uint8
    - 灰度扩成 3 通道，RGBA 丢弃 alpha
    - 若帧尺寸与 target_size 不一致会统一 resize
    - 根据文件后缀选择合适的编码器并尝试回退
    """
    import os
    import cv2
    import numpy as np

    if not frames:
        raise ValueError("save_frames_as_video: frames 为空")

    # 1) 确保保存目录存在
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    # 2) 规范化到 uint8 RGB(3 通道)
    def _to_uint8_rgb(img: np.ndarray) -> np.ndarray:
        arr = np.asarray(img)
        if arr.ndim == 2:            # 灰度 -> 3 通道
            arr = np.repeat(arr[..., None], 3, axis=2)
        elif arr.ndim == 3 and arr.shape[2] == 4:  # 丢 alpha
            arr = arr[..., :3]
        elif arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"非法帧形状: {arr.shape}, 期望(H,W,3)/(H,W)/(H,W,4)")

        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating):
                arr = np.clip(arr, 0.0, 1.0)
                arr = np.rint(arr * 255.0).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    frames = [_to_uint8_rgb(f) for f in frames]

    # 3) 统一分辨率
    if target_size is None:
        h0, w0 = frames[0].shape[:2]
        target_size = (w0, h0)  # (width, height)

    if any(f.shape[:2] != (target_size[1], target_size[0]) for f in frames):
        frames = [cv2.resize(f, target_size, interpolation=cv2.INTER_AREA) for f in frames]

    # 4) 选择编码器（可手动传 codec，否则按扩展名尝试回退）
    ext = os.path.splitext(save_path)[1].lower()
    if codec:
        candidates = [codec]
    elif ext in (".mp4", ".m4v", ".mov"):
        candidates = ["mp4v", "avc1", "H264"]   # 按常见可用性排序
    elif ext == ".avi":
        candidates = ["XVID", "MJPG"]
    else:
        candidates = ["mp4v"]

    if fps <= 0:
        fps = 30

    writer = None
    for c in candidates:
        fourcc = cv2.VideoWriter_fourcc(*c)
        vw = cv2.VideoWriter(save_path, fourcc, float(fps), target_size, True)
        if vw is not None and vw.isOpened():
            writer = vw
            break
    if writer is None:
        raise RuntimeError(f"无法打开 VideoWriter，尝试的编码器：{candidates}，路径：{save_path}")

    # 5) 写入（RGB -> BGR）
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    return save_path


def plot_force_sensor_wrt_time(sensor_data: np.ndarray, only_last_frame: bool = False):
    """
    Plot force sensor data_manager (forces and moments) with respect to time

    :param sensor_data: (T, 6) array with format (Fx[N], Fy[N], Fz[N], Mx[Nm], My[Nm], Mz[Nm])
    :return: (frames, fig, (ax1, ax2))
        frames: A list of RGB images for animation
        fig: matplotlib figure object
        (ax1, ax2): tuple of axes objects
    """
    frames_cnt = sensor_data.shape[0]
    sensor_dim = sensor_data.shape[1]
    assert sensor_dim == 6, "Force sensor data_manager must have 6 dimensions"

    # 创建一个图和两个子图（2行1列）
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # --- 子图1：三维力 Fx, Fy, Fz ---
    labels_force = ['Fx', 'Fy', 'Fz']
    colors_force = ['red', 'green', 'blue']
    styles_force = ['solid', 'solid', 'solid']
    lines_force = []
    for label, color, style in zip(labels_force, colors_force, styles_force):
        line, = ax1.plot([], [], label=label, color=color, linestyle=style, linewidth=2)
        lines_force.append(line)

    ax1.set_xlim(0, frames_cnt)
    # 设置y轴范围，留出10%的边距
    force_min = np.min(sensor_data[:, :3])
    force_max = np.max(sensor_data[:, :3])
    force_margin = 0.1 * (force_max - force_min) if force_max != force_min else 1
    ax1.set_ylim(force_min - force_margin, force_max + force_margin)
    ax1.set_xlabel('Time (Frames)')
    ax1.set_ylabel('Force (N)')
    ax1.set_title('3D Forces')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # --- 子图2：三维力矩 Mx, My, Mz ---
    labels_moment = ['Mx', 'My', 'Mz']
    colors_moment = ['darkred', 'darkgreen', 'darkblue']
    styles_moment = ['dashed', 'dashed', 'dashed']
    lines_moment = []
    for label, color, style in zip(labels_moment, colors_moment, styles_moment):
        line, = ax2.plot([], [], label=label, color=color, linestyle=style, linewidth=2)
        lines_moment.append(line)

    ax2.set_xlim(0, frames_cnt)
    # 设置y轴范围，留出10%的边距
    moment_min = np.min(sensor_data[:, 3:])
    moment_max = np.max(sensor_data[:, 3:])
    moment_margin = 0.1 * (moment_max - moment_min) if moment_max != moment_min else 0.1
    ax2.set_ylim(moment_min - moment_margin, moment_max + moment_margin)
    ax2.set_xlabel('Time (Frames)')
    ax2.set_ylabel('Moment (Nm)')
    ax2.set_title('3D Moments')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # 帧图列表
    frames = []

    if not only_last_frame:
        print("Plotting force sensor dynamic figures...")
    frame_indices = [frames_cnt - 1] if only_last_frame else range(frames_cnt)
    for frame_num in frame_indices:
        # 更新力曲线 (Fx, Fy, Fz)
        for i, line in enumerate(lines_force):
            line.set_data(np.arange(frame_num + 1), sensor_data[:frame_num + 1, i])

        # 更新力矩曲线 (Mx, My, Mz)
        for i, line in enumerate(lines_moment):
            line.set_data(np.arange(frame_num + 1), sensor_data[:frame_num + 1, i + 3])

        # 清除之前的文本（稳妥）
        for txt in list(ax1.texts):
            txt.remove()
        for txt in list(ax2.texts):
            txt.remove()

        # 显示当前力值/力矩值（当前帧）
        current_force_text = f'Current: Fx={sensor_data[frame_num, 0]:.2f}N, Fy={sensor_data[frame_num, 1]:.2f}N, Fz={sensor_data[frame_num, 2]:.2f}N'
        ax1.text(0.02, 0.95, current_force_text, transform=ax1.transAxes,
                 fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        current_moment_text = f'Current: Mx={sensor_data[frame_num, 3]:.3f}Nm, My={sensor_data[frame_num, 4]:.3f}Nm, Mz={sensor_data[frame_num, 5]:.3f}Nm'
        ax2.text(0.02, 0.95, current_moment_text, transform=ax2.transAxes,
                 fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(img)

    plt.close(fig)
    return frames, fig, (ax1, ax2)


class DynamicDataDrawer(ABC):
    def __init__(self, data_provider, data_keys: List[List[str]], max_points: int = 1000,
                 y_minmax_values: List[Iterable[float]] = None):
        self.data_provider = data_provider
        self.data_keys_grouped = data_keys  # 二维列表
        self.max_points = max_points

        # 展平所有 key，便于初始化数据缓存
        flat_keys = [key for group in data_keys for key in group]
        self.data_dict = {k: [] for k in flat_keys}
        self.data_max = {k: -np.inf for k in flat_keys}

        self.colors = ['r', 'g', 'b', 'c', 'm', 'y']
        self.linestyles = ['solid'] * 3 + ['dashed'] * 3
        if y_minmax_values is None:
            y_minmax_values = [[-10, 10], [-20, 20], [-200, 200], [-2.4, 2.4]]  # 一组子图一个范围
        self.y_minmax_values = y_minmax_values

        # 设置子图行列
        self.n_subplots = len(data_keys)
        self.ncols = 2
        self.nrows = math.ceil(self.n_subplots / self.ncols)

        # 创建子图
        self.fig, self.axes = plt.subplots(self.nrows, self.ncols, figsize=(24, 6 * self.nrows))
        self.axes = self.axes.flatten()  # 展平，方便用索引访问
        self.lines = {}

    def init(self):
        for idx, key_group in enumerate(self.data_keys_grouped):
            ax = self.axes[idx]
            for j, key in enumerate(key_group):
                self.lines[key], = ax.plot(
                    [], [], label=key,
                    color=self.colors[j % len(self.colors)],
                    linestyle=self.linestyles[j % len(self.linestyles)],
                )
            ax.set_xlim(0, self.max_points)
            y_min, y_max = self.y_minmax_values[idx]
            ax.set_ylim(y_min, y_max)
            ax.set_title(f"Live Data Group {idx+1}")
            ax.legend(loc='upper left')
        return list(self.lines.values())

    @abstractmethod
    def get_new_data(self) -> dict:
        pass

    def update(self, frame):
        new_data = self.get_new_data()
        for key in new_data:
            self.data_dict[key].append(new_data[key])
            self.data_max[key] = max(self.data_max[key], new_data[key])
            if len(self.data_dict[key]) > self.max_points:
                self.data_dict[key].pop(0)
            x = list(range(len(self.data_dict[key])))
            self.lines[key].set_data(x, self.data_dict[key])
        # new_data = dict(sorted(new_data.items()))
        # print("[DynamicDataDrawer] NOW:", new_data)
        # print("[DynamicDataDrawer] MAX:", self.data_max)
        return list(self.lines.values())

    def run(self):
        self.ani = FuncAnimation(
            self.fig,
            self.update,
            init_func=self.init,
            interval=100,
            blit=False,
            save_count=200,
        )
        plt.tight_layout()
        plt.show()



