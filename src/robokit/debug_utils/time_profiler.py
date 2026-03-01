from __future__ import annotations
import time
from typing import Dict, List, Optional


def _stats(ts: List[float]):
    n = len(ts)
    tot = sum(ts)
    mean = tot / n
    mn, mx = min(ts), max(ts)
    std = (sum((x - mean) ** 2 for x in ts) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return n, tot, mean, std, mn, mx


class TimeProfiler:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.item: Dict[str, List[float]] = {}   # name  -> [per_step_sum_dt...]
        self.group: Dict[str, List[float]] = {}  # group -> [per_step_sum_dt...]
        self._icur: Dict[str, float] = {}        # name  -> current step sum
        self._gcur: Dict[str, float] = {}        # group -> current step sum

    def __call__(self, name: str, group: Optional[str] = None):
        prof = self

        class _Ctx:
            __slots__ = ("t0",)
            def __enter__(self_):
                if prof.enabled:
                    self_.t0 = time.perf_counter()
                return self_
            def __exit__(self_, exc_type, exc, tb):
                if not prof.enabled:
                    return False
                dt = time.perf_counter() - self_.t0
                prof._icur[name] = prof._icur.get(name, 0.0) + dt
                if group is not None:
                    prof._gcur[group] = prof._gcur.get(group, 0.0) + dt
                return False

        return _Ctx()

    def step(self):
        """每个 iteration/forward 结束调用一次：把本 step 累加写入统计（每 step 记 1 个样本）。"""
        if not self.enabled:
            return
        for name, s in self._icur.items():
            self.item.setdefault(name, []).append(s)
        for g, s in self._gcur.items():
            self.group.setdefault(g, []).append(s)
        self._icur.clear()
        self._gcur.clear()

    def reset(self):
        self.item.clear()
        self.group.clear()
        self._icur.clear()
        self._gcur.clear()

    def report(self, sort_by: str = "total_ms", top_k: Optional[int] = None):
        # rows: (label, n, total_ms, mean_ms, std_ms, min_ms, max_ms)
        def mk_rows(d: Dict[str, List[float]]):
            rows = []
            for k, ts in d.items():
                n, tot, mean, std, mn, mx = _stats(ts)
                rows.append((k, n, tot*1e3, mean*1e3, std*1e3, mn*1e3, mx*1e3))
            key_i = {"cnt": 1, "total_ms": 2, "mean_ms": 3, "std_ms": 4, "min_ms": 5, "max_ms": 6}[sort_by]
            rows.sort(key=lambda r: r[key_i], reverse=True)
            return rows[:top_k] if top_k else rows

        def pr(title: str, rows):
            if not rows:
                print(f"\n[{title}] (empty)")
                return
            print(f"\n[{title}]")
            print(f"{'name':34} {'cnt':>5} {'total':>10} {'mean':>10} {'std':>10} {'min':>10} {'max':>10}")
            print("-" * 97)
            for k, n, tot, mean, std, mn, mx in rows:
                print(f"{k[:34]:34} {n:5d} {tot:10.3f} {mean:10.3f} {std:10.3f} {mn:10.3f} {mx:10.3f}")

        pr("Items (per step sum)", mk_rows(self.item))
        pr("Groups (per step sum)", mk_rows(self.group))


global_time_profiler = TimeProfiler()
