import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kk_Finetune(nn.Module):
    def __init__(self):
        super(Kk_Finetune, self).__init__()
