import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import kk.apps.kk_app as kka
import kk.uer.kk_base as kkb
import kk.uer.kk_config as kkc
import kk.kk_utils as kku
import kk.uer.layers.kk_Normalization as kkn
import math


class KkLinear(kkb.KkModule):
    def __init__(self, config: kkc.KkmConfig, in_feather: int, out_feather: int,
                 *,
                 tradition: bool = False,
                 std: (str, float) = "normal",
                 normalization: str = "none"):
        super(KkLinear, self).__init__(config)
        inner_feather = math.ceil(100/in_feather)
        self.tradition = tradition
        if self.tradition:
            pass
        else:
            weights = torch.randn(in_feather, inner_feather, dtype=torch.float32) * 10
            kku.init_softmax(weights)
            self.register_buffer("weights", weights)
        self.Linear = nn.Linear(inner_feather, out_feather)
        if normalization == "batch":
            self.Norm = kkn.KkBatchNormal(config)
        elif normalization == "layer":
            self.Norm = kkn.KkLayerNormal(config)
        elif normalization == "instance":
            self.Norm = kkn.KkInstanceNormal(config)
        else:
            self.Norm = None
        super(KkLinear, self)._init_weights(self.Linear, 0, std)

    def forward(self, x):
        if self.tradition:
            o = self.Linear(x)
        else:
            o = torch.matmul(x, self.weights)
            o = self.Linear(o)
        if self.Norm is None:
            pass
        else:
            o = self.Norm(o)
        return o


class KkFFNLayer(kkb.KkModule):
    def __init__(self, config: kkc.KkmConfig, in_feather: int, out_feathers: int, inner_feathers: int = 128):
        super(KkFFNLayer, self).__init__(config)
        self.FFN = nn.Sequential(KkLinear(config, in_feather, inner_feathers),
                                 nn.ReLU(),
                                 KkLinear(config, inner_feathers, out_feathers))

    def forward(self, x):
        o = self.FFN(x)
        return o


class KkClassifierLayer(kkb.KkModule):
    def __init__(self, config: kkc.KkmConfig, in_feather: int, classes: int,
                 *,
                 one_formal: str = "softmax",
                 loops: int = 0, inner_feathers: int = 128):
        super(KkClassifierLayer, self).__init__(config)
        self.FFN0 = nn.Sequential(KkLinear(config, in_feather, inner_feathers),
                                  nn.ReLU())
        self.FFNx = nn.ModuleList([nn.Sequential(KkLinear(config, inner_feathers, inner_feathers), nn.ReLU())
                                  for _ in range(loops)])
        self.FFNe = KkLinear(config, inner_feathers, classes)

        if one_formal == "sigmoid":
            self.Norm = nn.Sigmoid()
        else:
            self.Norm = nn.Softmax(-1)

    def forward(self, x):
        o = self.FFN0(x)
        for ffn in self.FFNx:
            o = ffn(o)
        o = self.FFNe(o)
        return self.Norm(o)


class KkExtendlayer(kkb.KkModule):
    def __init__(self, config: kkc.KkmConfig, pos: int, in_feather, extend_feather: int, norm: bool = False):
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
