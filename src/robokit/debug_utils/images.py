from abc import abstractmethod, ABC
import math
from typing import List, Dict

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def concatenate_rgb_images(img1, img2, vertical=False, smaller_size=2):
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
    img1 = np.array(Image.fromarray(img1).resize((width // smaller_size, height // smaller_size)))
    hw_ratio2 = float(img2.shape[0]) / float(img2.shape[1])

    if not vertical:
        # 确保两个图像的高度相同
        img2 = np.array(Image.fromarray(img2).resize((int(height / hw_ratio2) // smaller_size, height // smaller_size)))
        # 使用numpy的hstack来拼接两个图像
        return np.hstack((img1, img2))
    else:
        # 确保两个图像的宽度相同
        img2 = np.array(Image.fromarray(img2).resize((width // smaller_size, int(width * hw_ratio2) // smaller_size)))
        # 使用numpy的vstack来拼接两个图像
        return np.vstack((img1, img2))


def plot_action_wrt_time(action_data: np.ndarray):
    frames_cnt = action_data.shape[0]

    # 创建一个图和两个子图（1行2列）
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    # --- 子图1：x, y, z ---
    labels_xyz = ['x', 'y', 'z']
    styles_xyz = ['solid', 'solid', 'solid']
    lines_xyz = [ax1.plot([], [], label=label, linestyle=style)[0] for label, style in zip(labels_xyz, styles_xyz)]
    ax1.set_xlim(0, frames_cnt)
    ax1.set_ylim(np.min(action_data[:, :3])-0.02, np.max(action_data[:, :3])+0.02)
    ax1.set_xlabel('Time (Frames)')
    ax1.set_ylabel('XYZ')
    ax1.legend()

    # --- 子图2：a, b, c, g ---
    labels_abcg = ['a', 'b', 'c', 'g']
    styles_abcg = ['dashed', 'dashed', 'dashed', 'dotted']
    lines_abcg = [ax2.plot([], [], label=label, linestyle=style)[0] for label, style in zip(labels_abcg, styles_abcg)]
    ax2.set_xlim(0, frames_cnt)
    ax2.set_ylim(np.min(action_data[:, 3:])-0.02, np.max(action_data[:, 3:])+0.02)
    ax2.set_xlabel('Time (Frames)')
    ax2.set_ylabel('ABCG')
    ax2.legend()

    # 帧图列表
    frames = []

    for frame_num in range(frames_cnt):
        # 更新 xyz 曲线
        for i, line in enumerate(lines_xyz):
            line.set_data(np.arange(frame_num), action_data[:frame_num, i])

        # 更新 abcg 曲线
        for i, line in enumerate(lines_abcg):
            line.set_data(np.arange(frame_num), action_data[:frame_num, i + 3])

        plt.tight_layout()
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(img)

    plt.close(fig)
    return frames, fig, (ax1, ax2)


class DynamicDataDrawer(ABC):
    def __init__(self, data_provider, data_keys: List[List[str]], max_points: int = 1000):
        self.data_provider = data_provider
        self.data_keys_grouped = data_keys  # 二维列表
        self.max_points = max_points

        # 展平所有 key，便于初始化数据缓存
        flat_keys = [key for group in data_keys for key in group]
        self.data_dict = {k: [] for k in flat_keys}
        self.data_max = {k: -np.inf for k in flat_keys}

        self.colors = ['r', 'g', 'b', 'c', 'm', 'y']
        self.linestyles = ['solid'] * 3 + ['dashed'] * 3
        self.y_minmax_values = [[-10, 10], [-20, 20], [-200, 200], [-2.4, 2.4]]  # 一组子图一个范围

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



