import time
import numpy as np
from typing import Dict, List
from contextlib import contextmanager


class TimeStatistics:
    """时间统计工具类"""

    def __init__(self, print_every_calls: int = 10):
        self._records: Dict[str, List[float]] = {}
        self.called_times = 0

        self.print_every_calls = print_every_calls

    @contextmanager
    def __call__(self, label: str, enabled: bool = True):
        """上下文管理器，用于统计代码块执行时间

        Args:
            label: 统计标签名称
        """
        if not enabled:
            yield
            return
        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_time
            if label not in self._records:
                self._records[label] = []
            self._records[label].append(elapsed)

    def print_stats(self):
        self.called_times += 1
        if self.called_times % self.print_every_calls != 0:
            return

        """打印所有标签的时间统计信息"""
        if not self._records:
            print("No timing data recorded.")
            return

        print("\n" + "=" * 70)
        print("Time Statistics Report")
        print("=" * 70)

        for label, times in self._records.items():
            times_array = np.array(times) * 1000.0  # convert to milliseconds
            print(f"\n[{label}]")
            print(f"  Count:   {len(times)}")
            print(f"  Mean:    {np.mean(times_array):.2f} ms")
            print(f"  Std:     {np.std(times_array):.2f} ms")
            print(f"  Min:     {np.min(times_array):.2f} ms")
            print(f"  Max:     {np.max(times_array):.2f} ms")

        print("=" * 70 + "\n")

    def get_stats(self, label: str) -> Dict[str, float]:
        """获取指定标签的统计数据

        Args:
            label: 统计标签名称

        Returns:
            包含统计信息的字典
        """
        if label not in self._records:
            return {}

        times_array = np.array(self._records[label])
        return {
            "mean": float(np.mean(times_array)),
            "std": float(np.std(times_array)),
            "min": float(np.min(times_array)),
            "max": float(np.max(times_array)),
        }

    def reset(self, label: str = None):
        """重置统计数据

        Args:
            label: 如果指定则只重置该标签，否则重置所有
        """
        if label is None:
            self._records.clear()
        elif label in self._records:
            del self._records[label]


# 创建全局实例
time_stat = TimeStatistics()
