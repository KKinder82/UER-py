import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import kk.uer.kk_module as kkm
import math


class Kk_FFNLayer(kkm.Kk_Module):
    def __init__(self, config: kkm.KKM_Config, in_feather: int, out_feathers: int,
                 norm: bool = False, inner_feathers: int = 128):
        super(Kk_FFNLayer, self).__init__(config)
        self.FFN = nn.Sequential(nn.Linear(in_feather, inner_feathers),
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
        self.Linear = nn.Linear(sentence_length, sentence_length * out_feathers, bias=False if norm else True)
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
    def __init__(self, config: kkm.KKM_Config, in_feather: int, channel_size: int, norm: bool = False):
        super(Kk_ChannelLayer, self).__init__(config)
        self.channel_size = channel_size
        self.Linear = nn.Linear(in_feather, in_feather * channel_size, bias=False if norm else True)
        self.Norm = None
        if norm:
            self.Norm = nn.LayerNorm(in_feather * channel_size)

    def forward(self, x):
        o = self.Linear(x)
        if self.Norm is not None:
            o = self.Norm(o)
        o = o.view(*x.shape[0:-1], self.channel_size, -1)
        return o


if __name__ == "__main__":
    x = torch.arange(2 * 3 * 4, dtype=torch.float32).view(2, 3, 4)
    channel_layer = Kk_ChannelLayer(None, 4, 10)
    y = channel_layer(x)
    print(y.shape)
