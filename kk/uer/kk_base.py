import os
import pathlib as path
import sys

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F

import torch.distributed as dist
import torch.utils.data.distributed as dist_data
import torch.nn.parallel.distributed as dist_nn
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
import math
import logging as log
import tqdm
import kk.kk_utils as kku
import random
import time
import kk.uer.layers.kk_Normalization as kkn
import kk.uer.kk_config as kkc



class KkModule(nn.Module):
    def __init__(self, config: kkc.KkmConfig):
        super(KkModule, self).__init__()
        self.config = config

    def reset_epoch(self, **args):
        # iepoch = args["iepoch"]
        pass

    def get_normalization(self, normalization: str, groups: int = 2):
        if normalization == "batch":
            return kkn.KkBatchNormal(self.config)
        elif normalization == "layer":
            return kkn.KkLayerNormal(self.config)
        elif normalization == "instance":
            return kkn.KkInstanceNormal(self.config)
        elif normalization == "group":
            return kkn.KkGroupNormal(self.config, groups)
        else:
            return None

    def _init_weights(self, model, mean=0.0, std: (str, float)=0.01):
        if isinstance(std, str):
            if std == "relu":
                std = math.sqrt(2.0 / model.weight.size(-1))
            elif std == "tanh":
                std = 5.0 / 3.0 * math.sqrt(3.0 / model.weight.size(-1))
            elif std == "sigmoid":
                std = math.sqrt(1.0 / model.weight.size(-1))
            elif std == "normal":
                std = 0.01
            else:
                std = random.randint(1, 10) / 100
        else:
            if std <= 0:
                std = random.randint(1, 10)/100

        if isinstance(model, nn.Linear):
            nn.init.normal_(model.weight, mean=mean, std=std)
            nn.init.constant_(model.bias, 0)
        elif isinstance(model, nn.Embedding):
            nn.init.normal_(model.weight, mean=mean, std=std)
        elif isinstance(model, nn.LayerNorm):
            nn.init.constant_(model.bias, mean)
            nn.init.constant_(model.weight, 1.0 + std)
        elif isinstance(model, nn.Conv2d):
            nn.init.normal_(model.weight, mean=mean, std=std)
            nn.init.constant_(model.bias, mean)
        elif isinstance(model, nn.Conv1d):
            nn.init.normal_(model.weight, mean=mean, std=std)
            nn.init.constant_(model.bias, mean)
        elif isinstance(model, nn.BatchNorm1d):
            nn.init.constant_(model.bias, mean)
            nn.init.constant_(model.weight, 1.0 + std)
        elif isinstance(model, nn.BatchNorm2d):
            nn.init.constant_(model.bias, mean)
            nn.init.constant_(model.weight, 1.0 + std)
        elif isinstance(model, nn.GRU):
            for name, param in model.named_parameters():
                if 'weight' in name or 'bias' in name:
                    nn.init.normal_(param, mean=mean, std=std)
        elif isinstance(model, nn.LSTM):
            for name, param in model.named_parameters():
                if 'weight' in name or 'bias' in name:
                    nn.init.normal_(param, mean=mean, std=std)
        elif isinstance(model, nn.RNN):
            for name, param in model.named_parameters():
                if 'weight' in name or 'bias' in name:
                    nn.init.normal_(param, mean=mean, std=std)
        else:
            pass

    def before_forward(self, **args):
        # x = args["x"]
        # y = args["y"]
        pass

    def after_loss(self, **args):
        # x = args["x"]
        # y = args["y"]
        # loss = args["y"]
        pass

