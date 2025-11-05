import json
from typing import Any, Union
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


def beautiful_print(
    obj: Any,
    decimals: int = 4,
    indent: int = 2,
    sort_keys: bool = False,
    return_str: bool = False,
    json_safe_nan: bool = False,
    tuple_as_paren: bool = True,   # True: 元组用 (...)；False: 用 [...]（更接近 JSON）
):
    """
    - list/tuple 内部元素不换行，始终单行输出
    - dict 按缩进多行输出
    - 所有浮点数最多保留 `decimals` 位（默认 4），不补 0
    - 可选：把 NaN/±Inf 转为 None(null) 以保证 JSON 合法
    - 支持 numpy 标量/数组
    """

    # ---------- 1) 先把对象递归规整 & 四舍五入 ----------
    def _round_float(x: float):
        if isinstance(x, (float, np.floating)):
            if json_safe_nan and (np.isnan(x) or np.isposinf(x) or np.isneginf(x)):
                return None
            return round(float(x), decimals)
        return x

    def _normalize(x: Any):
        # 标量
        if isinstance(x, (bool, str, bytes)):
            return x
        if isinstance(x, (int, np.integer)):
            return int(x)
        if isinstance(x, (float, np.floating)):
            return _round_float(x)

        # 容器
        if isinstance(x, dict):
            return {k: _normalize(v) for k, v in x.items()}
        if isinstance(x, (list, tuple, set)):
            seq = list(x) if not isinstance(x, set) else sorted(x, key=lambda v: str(v))
            return [ _normalize(v) for v in seq ]

        # numpy
        if isinstance(x, np.ndarray):
            return _normalize(x.tolist())
        if isinstance(x, np.generic):
            # 其他 numpy 标量
            if isinstance(x, np.floating):
                return _round_float(x)
            if isinstance(x, np.integer):
                return int(x)
            return x.item()

        # 兜底
        return str(x)

    clean = _normalize(obj)

    # ---------- 2) 自定义格式化：dict 多行；list/tuple 单行 ----------
    def _fmt_scalar(x):
        return json.dumps(x, ensure_ascii=False)

    def _fmt_compact_json(x):
        # 单行紧凑 JSON 表达，用于出现在 list/tuple 里的 dict/子结构
        return json.dumps(x, ensure_ascii=False, separators=(",", ": "), sort_keys=sort_keys)

    def _fmt(x, level: int = 0, in_array: bool = False):
        pad  = " " * (indent * level)
        pad2 = " " * (indent * (level + 1))

        if isinstance(x, dict):
            # 在数组中：压成单行，避免换行
            if in_array:
                return _fmt_compact_json(x)
            # 否则多行打印
            items = x.items()
            if sort_keys:
                items = sorted(items, key=lambda kv: kv[0])
            if not items:
                return "{}"
            parts = []
            for k, v in items:
                k_str = json.dumps(k, ensure_ascii=False)
                v_str = _fmt(v, level + 1, in_array=False)
                parts.append(f"{pad2}{k_str}: {v_str}")
            return "{\n" + ",\n".join(parts) + f"\n{pad}" + "}"

        if isinstance(x, list):
            # 单行打印 list，内部元素也单行（嵌套 dict 会被压成单行）
            elems = [_fmt(v, level=level, in_array=True) for v in x]
            return "[" + ", ".join(elems) + "]"

        if isinstance(x, tuple):
            # 同 list；是否用 () 由 tuple_as_paren 控制
            elems = [_fmt(v, level=level, in_array=True) for v in x]
            if tuple_as_paren:
                return "(" + ", ".join(elems) + ")"
            else:
                return "[" + ", ".join(elems) + "]"

        # 其他（标量）
        return _fmt_scalar(x)

    s = _fmt(clean, level=0, in_array=False)
    if return_str:
        return s
    print(s)
