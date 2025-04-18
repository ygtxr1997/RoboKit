import torch
import numpy as np
from numpy.dtypes import ByteDType


def print_leaf(prefix, x):
    if isinstance(x, str):
        print(f'{prefix}:{type(x)},len={len(x)}')
    elif isinstance(x, torch.Tensor) or isinstance(x, np.ndarray):
        if x.ndim == 0:
            print(f'{prefix},{type(x)},shape={x.shape}, {x}')
        else:
            print(f'{prefix},{type(x)},shape={x.shape}')
    elif isinstance(x, bool) or isinstance(x, int):
        print(f'{prefix}:{type(x)},{x}')
    else:
        raise TypeError(f'Unexpected type {type(x)}')


def print_batch(prefix, x, depth=0):
    if isinstance(x, str) or isinstance(x, bool) or isinstance(x, int):
        print_leaf(prefix, x)
        return
    elif isinstance(x, torch.Tensor) or isinstance(x, np.ndarray):
        print_leaf(prefix, x)
        return
    elif isinstance(x, dict):
        print(f'{prefix}: Dict,keys={x.keys()}')
        for k, v in x.items():
            if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray):
                print_leaf(('-' * depth) + k, v)
            else:
                print_batch(('-' * depth) + k, v, depth + 1)
    elif isinstance(x, list):
        print(f'{prefix}: List,len={len(x)},elem:{type(x[0])}')
        if isinstance(x[0], torch.Tensor) or isinstance(x[0], np.ndarray):
            print_batch(('-' * depth) + '[0]', x[0], depth + 1)
    else:
        raise TypeError(f'type {type(x)} not supported. x must be torch.Tensor or list or dict')