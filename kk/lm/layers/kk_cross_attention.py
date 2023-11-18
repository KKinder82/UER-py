import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import kk.lm.kk_app as kka
import kk.lm.kk_base as kkb
import kk.lm.kk_config as kkc
import kk.lm.layers.kk_linear as kkl
import kk.lm.layers.kk_normalization as kkn
import math


class KkCrossAttention(kkb.KkModule):
    def __init__(self, content_feathers: int, in_feathers: int, *, out_length: int,
                 inner_feathers: int = 256, normalization: str = "none"):
        super(KkCrossAttention, self).__init__()
        self.a_feathers = content_feathers + in_feathers
        self.x_net = kkl.KkExtendlayer(self.a_feathers, inner_feathers)
        # perc_metric = kkb.get_randn_parameter(inner_feathers, inner_feathers, std="kk")
        # self.register_buffer("perc_metric", perc_metric)
        self.o_net = kkl.KkLinear(self.a_feathers, out_length, init_std="normal", tradition=True)

    def forward(self, content, x):
        o = torch.concatenate((content.expand(x.size(0), -1), x), dim=-1)
        o = self.x_net(o)
        # o = o @ self.perc_metric @ o.transpose(-1, -2)
        o = o @ o.transpose(-1, -2)
        o = self.o_net(o).transpose(-1, -2)
        return o
