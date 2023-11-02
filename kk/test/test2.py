import numpy as np

path = "../../datasets/c3/_.json"
with open(path, mode="r", encoding="utf-8") as f:
    data = json.load(f)