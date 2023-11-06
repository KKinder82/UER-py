import json
import os


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