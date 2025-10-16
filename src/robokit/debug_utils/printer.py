import torch
import numpy as np


def print_leaf(prefix, x):
    if isinstance(x, str):
        # 为字符串添加引号以便清晰地看到其边界
        return f"{prefix}: {type(x)}, len={len(x)}, value='{x}'"
    elif isinstance(x, (torch.Tensor, np.ndarray)):
        if x.ndim == 0:
            return f'{prefix}, {type(x)}, shape={x.shape}, value={x.item()}'
        else:
            # 确保在计算 min/max 之前张量不为空
            num_elements = x.numel() if isinstance(x, torch.Tensor) else x.size
            if num_elements > 0:
                return (f'{prefix}, {type(x)}, shape={x.shape}, min={float(x.min()):.4f}, max={float(x.max()):.4f}, '
                        f'dtype={x.dtype}')
            else:
                return f'{prefix}, {type(x)}, shape={x.shape} (empty)'
    elif isinstance(x, (bool, int, float)):
        return f'{prefix}: {type(x)}, value={x}'
    elif x is None:
        return f'{prefix}: {type(x)}'
    else:
        # 对于未知类型，只打印其类型
        return f'{prefix}: {type(x)}'


def _print_batch_recursive(prefix, x, depth, lines):
    """递归地构建输出行列表。"""
    if isinstance(x, (str, bool, int, float, torch.Tensor, np.ndarray, type(None))):
        lines.append(print_leaf(prefix, x))
    elif isinstance(x, dict):
        lines.append(f"{prefix}: Dict, keys={list(x.keys())}")
        for k, v in x.items():
            child_prefix = ('--' * (depth + 1)) + k
            _print_batch_recursive(child_prefix, v, depth + 1, lines)
    elif isinstance(x, list):
        if not x:
            lines.append(f'{prefix}: List, len=0')
            return
        # 仅展示第一个元素的类型作为代表
        lines.append(f'{prefix}: List, len={len(x)}, elem_type={type(x[0])}')
        # 递归打印第一个元素以展示列表内容结构
        child_prefix = ('--' * (depth + 1)) + '[0]'
        _print_batch_recursive(child_prefix, x[0], depth + 1, lines)
    else:
        # 对于其他可迭代但未明确处理的类型
        try:
            lines.append(f"{prefix}: {type(x)}, value={str(x)}")
        except Exception:
            lines.append(f"{prefix}: {type(x)} (un-stringable)")


def print_batch(prefix, x, depth=0):
    """
    递归地遍历一个批次数据结构，构建一个完整的字符串表示，然后一次性打印。
    """
    lines = []
    _print_batch_recursive(prefix, x, depth, lines)
    print('\n'.join(lines))