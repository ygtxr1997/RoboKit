import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


_STEP_RE = re.compile(r"\]\s*(\d+)\s*:\s*iter_speed\b")
_KV_RE = re.compile(r"\|\s*([^|:]+?)\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*")


def parse_log(log_path: str) -> Tuple[List[int], Dict[str, List[Optional[float]]]]:
    """Parse steps + all `Key: Value` metrics from the log."""
    steps: List[int] = []
    rows: List[Dict[str, float]] = []

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = _STEP_RE.search(line)
            if not m:
                continue
            step = int(m.group(1))
            kvs = _KV_RE.findall(line)
            if not kvs:
                continue

            d = {k.strip(): float(v) for k, v in kvs}
            steps.append(step)
            rows.append(d)

    keys = sorted({k for r in rows for k in r.keys()})
    metrics: Dict[str, List[Optional[float]]] = {k: [] for k in keys}
    for r in rows:
        for k in keys:
            metrics[k].append(r.get(k))  # missing -> None

    return steps, metrics


def _safe_png(dir_path: Path, name: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")
    return dir_path / f"{safe}.png"


def _downsample(
    steps: List[int],
    metrics: Dict[str, List[Optional[float]]],
    every_n_steps: int,
) -> Tuple[List[int], Dict[str, List[Optional[float]]]]:
    """
    Keep points whose step is a multiple of `every_n_steps`.
    If every_n_steps <= 1: keep all points.
    """
    if every_n_steps <= 1:
        return steps, metrics

    idx = [i for i, s in enumerate(steps) if s % every_n_steps == 0]
    if not idx:  # fallback: keep all to avoid empty plots
        return steps, metrics

    ds_steps = [steps[i] for i in idx]
    ds_metrics = {k: [v[i] for i in idx] for k, v in metrics.items()}
    return ds_steps, ds_metrics


def plot_lines(
    steps: List[int],
    series: Dict[str, List[Optional[float]]],
    title: str,
    save_path: Optional[str] = None,
    y_max: Optional[float] = None,
):
    """Plot 1+ lines; None will be plotted as gaps."""
    to_nan = lambda xs: [float("nan") if v is None else float(v) for v in xs]

    plt.figure(figsize=(10, 4.5))
    y_mean = 0.0
    for name, ys in series.items():
        xs_f, ys_f = zip(*[
            (x, y) for x, y in zip(steps, ys)
            if (y is not None and y != 0)
        ]) if any((y is not None and y != 0) for y in ys) else ([], [])
        plt.plot(xs_f, ys_f, label=name)
    for name, ys in series.items():
        y_mean += sum(y for y in ys if y is not None) / max(1, sum(1 for y in ys if y is not None))
    y_mean /= len(series)
    plt.xlabel("step")
    plt.ylabel("value")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    if y_mean <= 0.03 or y_max is not None:
        plt.ylim(top=0.035) if y_max is None else plt.ylim(top=y_max)
    plt.ylim(bottom=0)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=160)
        plt.close()
    else:
        plt.show()


def plot_groups(
    log_path: str,
    save_dir: Optional[str] = None,
    loss_key: str = "Loss",
    prefixes: Tuple[str, ...] = ("Action", "Video"),
    every_n_steps: int = 100,  # <- 新增：每隔多少 step 采样一个点
    y_max: Optional[float] = None,
):
    """
    - loss_key: 单独画一张
    - prefixes: 每个 prefix 下的指标，每两个指标画一张（不足两个则单独画）
    - every_n_steps: 只画 step % every_n_steps == 0 的点（<=1 表示不过滤）
    """
    steps, metrics = parse_log(log_path)
    if not steps:
        raise ValueError(f"No valid metric lines found in: {log_path}")

    steps, metrics = _downsample(steps, metrics, every_n_steps)

    out_dir = Path(save_dir) if save_dir else None
    out = (lambda name: str(_safe_png(out_dir, name))) if out_dir else (lambda name: None)

    # 1) Loss alone
    if loss_key in metrics:
        plot_lines(steps, {loss_key: metrics[loss_key]}, title=loss_key, save_path=out(loss_key))

    # 2) Prefix groups: every 2 metrics per figure
    for p in prefixes:
        keys = sorted([k for k in metrics.keys() if k.startswith(p)])
        for i in range(0, len(keys), 2):
            chunk = keys[i : i + 2]
            series = {k: metrics[k] for k in chunk}
            title = f"{p} metrics" if len(chunk) == 2 else chunk[0]
            name = f"{p}_{i//2:02d}_" + "_vs_".join(chunk)
            plot_lines(steps, series, title=title, save_path=out(name), y_max=y_max)


if __name__ == "__main__":
    # 显示（每 100 step 一个点）
    # plot_groups("train.log")

    # 保存（比如每 500 step 一个点）
    plot_groups("train.log", save_dir="./log_plots", every_n_steps=500)
