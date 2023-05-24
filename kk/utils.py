import json
import os


def load_argsconfig(file: str):
    if not os.path.exists(file):
        return None

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
