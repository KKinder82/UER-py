import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import kk.uer.kk_module as kkm
import math


class Kk_FFNLayer(kkm.Kk_Module):
    def __init__(self, config: kkm.KKM_Config, in_feathers: int, out_feathers: int,
                 norm: bool = False, inner_feathers: int = 128):
        super(Kk_FFNLayer, self).__init__(config)
        self.FFN = nn.Sequential(nn.Linear(in_feathers, inner_feathers),
                                 nn.ReLU(),
                                 nn.Linear(inner_feathers, out_feathers))
        self.Norm = None
        if norm:
            self.Norm = nn.LayerNorm(out_feathers)

    def forward(self, x):
        o = self.FFN(x)
        if self.Norm is not None:
            o = self.Norm(o)
        return o


class Kk_FeatherLayer(kkm.Kk_Module):
    def __init__(self, config: kkm.KKM_Config, sentence_length: int, out_feathers: int, norm: bool = False):
        super(Kk_FeatherLayer, self).__init__(config)
        self.Linear = nn.Linear(sentence_length, sentence_length * out_feathers)
        self.Norm = None
        if norm:
            self.Norm = nn.LayerNorm(out_feathers)

    def forward(self, x):
        o = self.Linear(x)
        o = o.view(*x.shape, -1)
        if self.Norm is not None:
            o = self.Norm(o)
        return o


class Kk_ChannelLayer(kkm.Kk_Module):
    def __init__(self, config: kkm.KKM_Config, sentence_length: int, out_feathers: int, norm: bool = False):
        super(Kk_ChannelLayer, self).__init__(config)
        self.Linear = nn.Linear(sentence_length, sentence_length * out_feathers)
        self.Norm = None
        if norm:
            self.Norm = nn.LayerNorm(out_feathers)

    def forward(self, x):
        o = self.Linear(x)
        o = o.view(*x.shape, -1)
        if self.Norm is not None:
            o = self.Norm(o)
        return o
