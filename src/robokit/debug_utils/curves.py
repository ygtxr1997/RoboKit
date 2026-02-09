import os
import numpy as np
import h5py
from typing import Dict, Optional, List
from filelock import FileLock
import matplotlib.pyplot as plt


class H5CurveStorage:
    """用于保存和读取测试过程中的成功率等指标到 HDF5 文件（支持多进程）"""

    def __init__(self, filepath: str):
        """
        Args:
            filepath: h5 文件路径
        """
        self.filepath = filepath
        self.lockfile = filepath + '.lock'
        self._lock = FileLock(self.lockfile, timeout=30)  # 增加超时时间

        # 如果文件不存在，创建空文件
        if not os.path.exists(filepath):
            with self._lock:
                # 使用 libver='latest' 支持 SWMR
                with h5py.File(filepath, 'w', libver='latest') as f:
                    pass

    def save(self, key: str, array: np.ndarray, overwrite: bool = True):
        """
        保存或更新某个 key 对应的数组

        Args:
            key: 数据的键名
            array: 要保存的 numpy 数组
            overwrite: 是否允许覆盖已存在的 key，默认 True
        """
        with self._lock:
            # 禁用 HDF5 内部文件锁
            with h5py.File(self.filepath, 'a', libver='latest', locking=False) as f:
                if key in f:
                    if not overwrite:
                        raise KeyError(f"Key '{key}' already exists. Set overwrite=True to replace.")
                    del f[key]
                f.create_dataset(key, data=array)

    def update_or_append(self, key: str, index: int, value: float):
        """如果索引超出范围，自动扩展数组"""
        with self._lock:
            # 禁用 HDF5 内部文件锁
            with h5py.File(self.filepath, 'a', libver='latest', locking=False) as f:
                if key not in f:
                    f.create_dataset(key, data=np.array([value]), maxshape=(None,))
                    return

                dataset = f[key]
                current_data = dataset[:]

                if index >= len(current_data):
                    new_data = np.append(current_data, [value])
                    del f[key]
                    f.create_dataset(key, data=new_data, maxshape=(None,))
                else:
                    dataset[index] = value

    def get(self, key: str) -> Optional[np.ndarray]:
        """
        获取指定 key 的数组

        Args:
            key: 数据的键名

        Returns:
            对应的 numpy 数组，如果不存在返回 None
        """
        with self._lock:
            with h5py.File(self.filepath, 'r', locking=False) as f:
                if key in f:
                    return f[key][:]
                return None

    def keys(self) -> List[str]:
        """返回所有已保存的 key"""
        with self._lock:
            with h5py.File(self.filepath, 'r', locking=False) as f:
                return list(f.keys())

    def delete(self, key: str):
        """
        删除指定 key 的数据

        Args:
            key: 要删除的键名
        """
        with self._lock:
            with h5py.File(self.filepath, 'a', locking=False) as f:
                if key in f:
                    del f[key]

    def load(self) -> Dict[str, np.ndarray]:
        """
        从文件加载所有数据

        Returns:
            包含所有 key-array 对的字典
        """
        with self._lock:
            with h5py.File(self.filepath, 'r', locking=False) as f:
                return {key: f[key][:] for key in f.keys()}



def plot_curves(storage: H5CurveStorage,
                keys: List[str],
                title: str = "Curves Comparison",
                xlabel: str = "Index",
                ylabel: str = "Value",
                figsize: tuple = (12, 7),
                colors: List[str] = None,
                markers: List[str] = None,
                grid: bool = True,
                legend_loc: str = 'best',
                save_path: str = None,
                show_mean: bool = True,
                show_max_len: int = None,
                vis_ol_step: int = 10,
                ):
    """
    绘制多条曲线在同一张图上，可选显示各自的累积均值线

    Args:
        storage: H5CurveStorage 实例
        keys: 要绘制的数据键名列表
        title: 图表标题
        xlabel: 横轴标签
        ylabel: 纵轴标签
        figsize: 图表大小
        colors: 各曲线颜色列表，默认使用预设配色
        markers: 各曲线标记样式列表
        grid: 是否显示网格
        legend_loc: 图例位置
        save_path: 保存路径，为 None 则不保存
        show_mean: 是否显示累积均值线
    """
    # 默认配色方案
    default_colors = ['#2E86AB', '#E63946', '#06A77D', '#F77F00', '#8338EC', '#FB5607']
    default_markers = ['o', 's', '^', 'D', 'v', 'p']

    colors = colors or default_colors
    markers = markers or default_markers

    plt.figure(figsize=figsize)

    for i, key in enumerate(keys):
        vis_step = 1
        if "ol=True" in key or "vis" in key:
            vis_step = vis_ol_step
        values = storage.get(key)[::vis_step]  # None: use full length
        values = values[:show_max_len]
        if values is None:
            print(f"Warning: Key '{key}' not found, skipping.")
            continue

        indices = range(len(values))
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        # 绘制原始曲线
        plt.plot(indices, values,
                 color=color,
                 linewidth=2.5,
                 linestyle='--',
                 marker=marker,
                 markersize=7,
                 markerfacecolor='white',
                 markeredgewidth=2,
                 markeredgecolor=color,
                 alpha=0.85,
                 label=key)

        # 绘制累积均值线
        if show_mean:
            cumsum = np.cumsum(values)
            cumcount = np.arange(1, len(values) + 1)
            running_mean = cumsum / cumcount

            plt.plot(indices, running_mean,
                     color=color,
                     linewidth=2,
                     alpha=0.6,
                     label=f'{key} (Mean)')

            print(f"last mean of '{key}': {running_mean[-1]:.4f}")

    plt.title(title, fontsize=15, fontweight='bold', pad=20)
    plt.xlabel(xlabel, fontsize=13)
    plt.ylabel(ylabel, fontsize=13)
    plt.legend(loc=legend_loc, fontsize=11, framealpha=0.9)

    if grid:
        plt.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_vis_curves(storage: H5CurveStorage,
                keys: List[str],
                title: str = "Curves Comparison",
                xlabel: str = "Index",
                ylabel: str = "Value",
                figsize: tuple = (12, 7),
                colors: List[str] = None,
                markers: List[str] = None,
                grid: bool = True,
                legend_loc: str = 'best',
                save_path: str = None,
                show_mean: bool = True,
                show_start_idx: int = 0,
                show_max_len: int = None,
                vis_ol_step: int = 10,
                ):
    """
    绘制多条曲线在同一张图上，可选显示各自的累积均值线

    Args:
        storage: H5CurveStorage 实例
        keys: 要绘制的数据键名列表
        title: 图表标题
        xlabel: 横轴标签
        ylabel: 纵轴标签
        figsize: 图表大小
        colors: 各曲线颜色列表，默认使用预设配色
        markers: 各曲线标记样式列表
        grid: 是否显示网格
        legend_loc: 图例位置
        save_path: 保存路径，为 None 则不保存
        show_mean: 是否显示累积均值线
    """
    # 默认配色方案
    default_colors = ['#2E86AB', '#E63946', '#06A77D', '#F77F00', '#8338EC', '#FB5607']
    default_markers = ['o', 's', '^', 'D', 'v', 'p']

    colors = colors or default_colors
    markers = markers or default_markers

    plt.figure(figsize=figsize)

    for i, key in enumerate(keys):
        vis_step = 1
        if "ol=True" in key or "vis" in key:
            vis_step = vis_ol_step
        values = storage.get(key)
        if "ol=True" in key:
            values = values[show_start_idx: show_start_idx + show_max_len]  # 从 show_start_idx 开始采样

            # 将 (T*D) 转为 (T, D) 然后转置为 (D, T) 再展平为 (D*T), T:seed个数， D:每个seed update steps总数
            TD = len(values)
            D = 50  # 每个 setting 测试次数
            T = TD // D  # seed 个数
            values = values.reshape(T, D).T  # (D,T)

            mean_wrt_steps = values.mean(axis=1)  # (D,)
            values = mean_wrt_steps  # (D,)
        elif "ol=False" in key:
            # TD = len(values)
            # values = values.reshape(vis_ol_step, TD // vis_ol_step)  # (num_seeds, num_updates)
            # mean_wrt_steps = values.mean(axis=0)  # (num_updates,)
            # values = mean_wrt_steps  # (num_updates,)
            ## Only show the first num_seeds
            values = values[show_start_idx: show_start_idx + show_max_len]

            TD = show_max_len
            D = 50  # 每个 setting 测试次数
            T = TD // D  # seed 个数

            values = values.reshape(T, D).T  # (D,T)
            values = values.mean(axis=1)  # (D,)

        if values is None:
            print(f"Warning: Key '{key}' not found, skipping.")
            continue

        indices = range(len(values))
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        # 绘制原始曲线
        plt.plot(indices, values,
                 color=color,
                 linewidth=2.5,
                 linestyle='--',
                 marker=marker,
                 markersize=7,
                 markerfacecolor='white',
                 markeredgewidth=2,
                 markeredgecolor=color,
                 alpha=0.85,
                 label=key)

        # 绘制累积均值线或平行线
        if show_mean:
            cumsum = np.cumsum(values)
            cumcount = np.arange(1, len(values) + 1)
            running_mean = cumsum / cumcount

            # 如果是 ol=False，绘制平行线
            if "ol=False" in key:
                final_mean = running_mean[-1]
                plt.axhline(y=final_mean,
                           color=color,
                           linewidth=2,
                           linestyle='-',
                           alpha=0.6,
                           label=f'{key} (Mean={final_mean:.3f})')
            else:
                # 正常绘制累积均值线
                plt.plot(indices, running_mean,
                        color=color,
                        linewidth=2,
                        alpha=0.6,
                        label=f'{key} (Mean)')

    plt.title(title, fontsize=15, fontweight='bold', pad=20)
    plt.xlabel(xlabel, fontsize=13)
    plt.ylabel(ylabel, fontsize=13)
    plt.legend(loc=legend_loc, fontsize=11, framealpha=0.9)

    if grid:
        plt.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

