import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import kk.apps.kk_app as kka
import math


class KkFFNLayer(kka.KkModule):
    def __init__(self, config: kka.KkmConfig, in_feather: int, out_feathers: int,
                 norm: bool = False, inner_feathers: int = 128):
        super(KkFFNLayer, self).__init__(config)
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


class KkClassifierLayer(kka.KkModule):
    def __init__(self, config: kka.KkmConfig, in_feather: int, classes: int, classifierMode:int = 1,
                 norm: bool = False, loops: int = 0, inner_feathers: int = 128):
        super(KkClassifierLayer, self).__init__(config)
        self.FFN0 = nn.Sequential(nn.Linear(in_feather, inner_feathers),
                                  nn.ReLU())
        self.FFNx= nn.ModuleList([nn.Sequential(nn.Linear(inner_feathers, inner_feathers), nn.ReLU())
                                  for _ in range(loops)])
        self.FFNe = nn.Linear(inner_feathers, classes)

        self.norm = norm
        if classifierMode == 1:
            self.Norm = nn.Softmax(-1)
        else:
            self.Norm = nn.Sigmoid()

    def forward(self, x):
        o = self.FFN0(x)
        for ffn in self.FFNx:
            o = ffn(o)
        o = self.FFNe(o)
        return self.Norm(o)

class KkExtendlayer(kka.KkModule):
    def __init__(self, config: kka.KkmConfig, pos: int, in_feather, extend_feather: int, norm: bool = False):
        super(KkExtendlayer, self).__init__(config)
        self.pos = pos
        self.norm = norm
        self.Linear = nn.Linear(in_feather, in_feather * extend_feather, bias=False if norm else True)
        if self.norm:
            self.Norm = nn.LayerNorm(in_feather * extend_feather)

    def forward(self, x):
        o = self.Linear(x)
        if self.norm:
            o = self.Norm(o)
        o = o.view(*x.shape, -1)
        if self.pos is None:
            return o
        return o.reshape(*x.shape[:self.pos], -1, *x.shape[self.pos:])


if __name__ == "__main__":
    x = torch.arange(2 * 3 * 4, dtype=torch.float32).view(2, 3, 4)
    channel_layer = KkExtendlayer(None, -2, 4, 6)
    y = channel_layer(x)
    print(y.shape)
