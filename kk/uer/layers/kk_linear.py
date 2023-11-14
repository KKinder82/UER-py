import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import kk.apps.kk_app as kka
import kk.uer.kk_base as kkb
import kk.uer.kk_config as kkc
import kk.kk_utils as kku
import kk.uer.layers.kk_normalization as kkn
import math
import copy


class KkLinear(kkb.KkModule):
    def __init__(self, in_feathers: int, out_feathers: int, *,
                 tradition: bool = True,
                 init_std: (str, float) = "normal",
                 normalization: str = "none"):
        super(KkLinear, self).__init__()
        self.tradition = tradition
        if self.tradition:
            self.Linear = nn.Linear(in_feathers, out_feathers, bias=True)
            kkb.init_weights(self.Linear, std=init_std)
        else:
            inner_feather = math.ceil(100 / in_feathers)
            # perc_weights = torch.randn(in_feathers, inner_feather, dtype=torch.float32) * 10
            perc_weights = kkb.get_randn_parameter(in_feathers, inner_feather, std="kk")
            self.register_buffer("perc_weights", perc_weights)

            self.Linear = nn.Linear(inner_feather, out_feathers, bias=True)
            kkb.init_weights(self.Linear, std=init_std)

        self.Norm = kkn.get_normalization(normalization)

    def forward(self, x):
        if self.tradition:
            o = self.Linear(x)
        else:
            o = torch.matmul(x, self.perc_weights)
            o = self.Linear(o)
        if self.Norm is not None:
            o = self.Norm(o)
        return o
#
#
# class KkLinear(kkb.KkModule):
#     def __init__(self, in_feathers: int, out_feathers: int,
#                  *,
#                  tradition: bool = False,
#                  init_std: (str, float) = "normal",
#                  normalization: str = "none"):
#         super(KkLinear, self).__init__()
#         self.tradition = tradition
#         if self.tradition:
#             weight = kkb.get_randn_parameter(in_feathers, out_feathers, std=init_std)
#             self.weight = nn.Parameter(weight)
#             bias = kkb.get_randn_parameter(1, out_feathers, std=init_std)
#             self.bias = nn.Parameter(bias)
#         else:
#             inner_feather = math.ceil(100 / in_feathers)
#             # perc_weights = torch.randn(in_feathers, inner_feather, dtype=torch.float32) * 10
#             perc_weights = kkb.get_randn_parameter(in_feathers, inner_feather, std="kk")
#             self.register_buffer("perc_weights", perc_weights)
#
#             weight = kkb.get_randn_parameter(inner_feather, out_feathers, std=init_std)
#             self.weight = nn.Parameter(weight)
#             bias = kkb.get_randn_parameter(1, out_feathers, std=init_std)
#             self.bias = nn.Parameter(bias)
#
#         self.Norm = kkn.get_normalization(normalization)
#
#     def forward(self, x):
#         if self.tradition:
#             o = torch.matmul(x, self.weight) + self.bias
#         else:
#             o = torch.matmul(x, self.perc_weights)
#             o = torch.matmul(o, self.weight) + self.bias
#             if self.Norm is not None:
#                 o = self.Norm(o)
#         return o


# class KkLinear(kkb.KkModule):
#     def __init__(self, in_feathers: int, out_feathers: int,
#                  *,
#                  tradition: bool = False,
#                  init_std: (str, float) = "normal",
#                  normalization: str = "none"):
#         super(KkLinear, self).__init__()
#         self.tradition = tradition
#         if self.tradition:
#             self.Linear = nn.Linear(in_feathers, out_feathers, bias=True)
#             kkb.init_weights(self.Linear, std=init_std)
#         else:
#             inner_feather = math.ceil(100 / in_feathers)
#             # perc_weights = torch.randn(in_feathers, inner_feather, dtype=torch.float32) * 10
#             perc_weights = kkb.get_randn_parameter(in_feathers, inner_feather, std="kk")
#             self.register_buffer("perc_weights", perc_weights)
#
#             self.Linear = nn.Linear(inner_feather, out_feathers, bias=True)
#             kkb.init_weights(self.Linear, std=init_std)
#
#         self.Norm = kkn.get_normalization(normalization)
#
#     def forward(self, x):
#         o = torch.matmul(x, self.perc_weights)
#         o = self.Linear(o)
#         if self.Norm is not None:
#             o = self.Norm(o)
#         return o


class KkFFNLayer(kkb.KkModule):
    def __init__(self, in_feathers: int, out_feathers: int, inner_feathers: int = 128):
        super(KkFFNLayer, self).__init__()
        self.FFN = nn.Sequential(KkLinear(in_feathers, inner_feathers),
                                 nn.ReLU(),
                                 KkLinear(inner_feathers, out_feathers))

    def forward(self, x):
        o = self.FFN(x)
        return o


class KkClassifierLayer(kkb.KkModule):
    def __init__(self, in_feathers: int, classes: int, *,
                 one_formal: str = "softmax", inner_feathers: int = 128, loops: int = 0):
        super(KkClassifierLayer, self).__init__()
        self.FFN0 = nn.Sequential(KkLinear(in_feathers, inner_feathers), nn.ReLU())
        self.FFNx = nn.ModuleList([nn.Sequential(KkLinear(inner_feathers, inner_feathers),
                                                 nn.ReLU()) for _ in range(loops)])
        self.FFN9 = KkLinear(inner_feathers, classes)

        if one_formal == "sigmoid":
            self.Norm = nn.Sigmoid()
        else:
            self.Norm = nn.Softmax(-1)

    def forward(self, x):
        o = self.FFN0(x)
        last_o = o
        for ffn in self.FFNx:
            _o = ffn(o)
            o += _o + last_o
            last_o = _o
        o = self.FFN9(o)
        return self.Norm(o)


class KkExtendlayer(kkb.KkModule):
    def __init__(self, in_feathers, extend_feather: int, *, norm: bool = True):
        super(KkExtendlayer, self).__init__()
        self.norm = norm
        self.Linear = nn.Linear(in_feathers, in_feathers * extend_feather,
                                bias=False if norm else True)
        if self.norm:
            self.Norm = kkn.get_normalization("instance")

    def forward(self, x):
        o = self.Linear(x)
        if self.norm:
            o = self.Norm(o)
        o = o.view(*x.shape, -1)
        return o


if __name__ == "__main__":
    x = torch.arange(2 * 3 * 4, dtype=torch.float32).view(2, 3, 4)
    channel_layer = KkExtendlayer(None, -2, 4, 6)
    y = channel_layer(x)
    print(y.shape)
