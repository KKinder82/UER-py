import json
import os
import torch


def env_int(name: str, default=0):
    if name in os.environ:
        return int(os.environ[name])
    return default


def env_value(name: str, default: str = ""):
    if name in os.environ:
        return os.environ[name]
    return default


def load_argsconfig(file: str):
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


def onehot(index, bits=10):
    onehot = [0] * bits
    onehot[index] = 1
    return onehot


def tensor_fill(tensor, index_tensor, value):
    _index = index_tensor.view(-1, index_tensor.size(-1))
    _tensor = tensor.view(-1, tensor.size(-1))
    for i in range(_index.size(0)):
        for j in _index[i]:
            _tensor[i, j] = value
    return tensor


def new_percent_weight(*size: int):
    return torch.randint(1, 100000, size).to(torch.float32)


def init_softmax(parameter):
    # param = parameter # .pow(1.9)
    summ = parameter.sum(dim=-1, keepdim=True) + 1e-7
    parameter.data = parameter / summ


def init_normalization(parameter):
    std = parameter.std(dim=-1, unbiased=False, keepdim=True) + 1e-7
    mean = parameter.mean(dim=-1, keepdim=True)
    parameter.data = (parameter - mean) / std


if __name__ == "__main__":
    print(100000.0 * 2000)
    a = torch.randint(1, 100000, (100, 200)).to(torch.float32)
    init_softmax(a)
    print(a.std())
    b = a < 1e-15
    print(a.min())
    print(b.count_nonzero())
    print(a)


