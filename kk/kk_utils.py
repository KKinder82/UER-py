import json
import os
import torch
import datetime
import time


def kk_num101(num_, p1_, p0, p1):
    if num_ < 0:
        return p1_
    if num_ == 0:
        return p0
    return p1


def kk_env_int(name: str, default=0):
    if name in os.environ:
        return int(os.environ[name])
    return default


def kk_env_value(name: str, default: str = ""):
    if name in os.environ:
        return os.environ[name]
    return default


def kk_norm_min_max(x, min_, max_, min_range=0, max_range=1):
    return (x - min_) / (max_ - min_) * (max_range - min_range) + min_range


def kk_norm_std(x, axis=-1):
    mean = x.mean(-1, keepdim=True)
    std = x.std(-1, keepdim=True)
    return (x - mean) / std

def kk_load_argsconfig(file: str):
    if not os.path.exists(file):
        return []

    args = []
    with open(file, 'r', encoding='utf-8') as io :
        for line in io.readlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                continue
            items = line.split()
            args += items
    return args


def kk_onehot(index, bits=10):
    onehot = [0] * bits
    onehot[index] = 1
    return onehot


def kk_tensor_fill(tensor, index_tensor, value):
    _index = index_tensor.view(-1, index_tensor.size(-1))
    _tensor = tensor.view(-1, tensor.size(-1))
    for i in range(_index.size(0)):
        for j in _index[i]:
            _tensor[i, j] = value
    return tensor


def kk_percent_weight(*size: int):
    return torch.randint(1, 100000, size).to(torch.float32)


def kk_init_softmax(parameter):
    # param = parameter # .pow(1.9)
    summ = parameter.sum(dim=-1, keepdim=True) + 1e-7
    parameter.data = parameter / summ


def kk_init_normalization(parameter):
    std = parameter.std(dim=-1, unbiased=False, keepdim=True) + 1e-7
    mean = parameter.mean(dim=-1, keepdim=True)
    parameter.data = (parameter - mean) / std
    exit(0)


def kk_split(str_: str, sep: str = " \\n\\t"):
    out = []
    pos1 = 0
    len_ = len(str_)
    for i in range(1, len_):
        if str_[i] in sep:
            out.append(str_[pos1:i])
            pos1 = i + 1
    if pos1 < len_:
        out.append(str_[pos1:])
    return out


if __name__ == "__main__":
    print(100000.0 * 2000)
    a = torch.randint(1, 100000, (100, 200)).to(torch.float32)
    kk_init_softmax(a)
    print(a.std())
    b = a < 1e-15
    print(a.min())
    print(b.count_nonzero())
    print(a)
