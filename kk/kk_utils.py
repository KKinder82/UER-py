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