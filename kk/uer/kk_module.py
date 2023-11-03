import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class KKM_Config(object):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


class Kk_Module(nn.Module):
    def __init__(self, config: KKM_Config):
        super(Kk_Module, self).__init__()
        self.config = config

    def init_weights(self):
        pass

