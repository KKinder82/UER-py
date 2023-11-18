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


class KkSelfAttentionItem(kkb.KkModule):
    def __init__(self, in_feathers: int, out_feathers: int, *,
                 inner_feathers: int = 256, normalization: str = "none"):
        super(KkSelfAttentionItem, self).__init__()
        self.inner_feathers = inner_feathers
        self.qnet = kkl.KkLinear(in_feathers, inner_feathers)
        self.knet = kkl.KkLinear(in_feathers, inner_feathers)
        self.vnet = kkl.KkLinear(in_feathers, out_feathers)
        self.softmax = nn.Softmax(-1)

    def forward(self, q, k, v, *, postion_encoding: bool = False):
        if postion_encoding:
            pass
        q = self.qnet(q)
        k = self.knet(k)
        v = self.vnet(v)
        o = torch.matmul(q, k.transpose(-2, -1))
        o = o / math.sqrt(self.inner_feathers)
        o = self.softmax(o)
        o = torch.matmul(o, v)
        return o

#
# class KkSelfAttationItem(kkb.KkModule):
#     def __init__(self, config: kkc.KkmConfig, qk_feathers: int, out_feathers: int, v_feathers: int,
#                  inner_feathers: int = 256, *, normalization: str = "none"):
#         super(KkSelfAttationItem, self).__init__(config)
#         self.inner_feathers = inner_feathers
#         self.QNet = kkl.KkLinear(config, qk_feathers, inner_feathers)
#         self.KNet = kkl.KkLinear(config, qk_feathers, inner_feathers)
#         self.VNet = kkl.KkLinear(config, v_feathers, out_feathers)
#         self.Softmax = nn.Softmax(-1)
#         # self.Norm = kkn.get_normalization(self.config, normalization)
#
#
#     def forward(self, q, k, v, *, postion_encoding: bool = False):
#         if postion_encoding:
#             pass
#         q = self.QNet(q)
#         k = self.KNet(k)
#         v = self.VNet(v)
#         o = torch.matmul(q, k.transpose(-2, -1))
#         o = o / math.sqrt(self.inner_feathers)
#         o = self.Softmax(o)
#         o = torch.matmul(o, v)
#         # if self.Norm is None:
#         #     pass
#         # else:
#         #     o = self.Norm(o)
#         return o


class KkSelfAttention(kkb.KkModule):
    def __init__(self, *, in_feathers: int, out_feathers: int, inner_feathers: int = 256,
                 loops: int = 6, normalization: str = "none"):
        super(KkSelfAttention, self).__init__()
        self.SA1 = KkSelfAttentionItem(in_feathers, inner_feathers, v_feathers, inner_feathers,
                                       normalization="layer")
        self.SAs = [KkSelfAttentionItem(in_feathers, inner_feathers, inner_feathers, inner_feathers,
                                        normalization="layer") for _ in range(loops)]
        self.SAl = KkSelfAttentionItem(in_feathers, out_feathers, inner_feathers, inner_feathers,
                                       normalization=normalization)

    def forward(self, q, k, v):
        o = self.SA1(q, k, v)
        for isa in self.SAs:
            o = isa(o, o, o)
        o = self.SAl(o, o, o)
        return o


class KkMultiSelfAttentionItem(kkb.KkModule):
    def __init__(self, in_feathers: int, out_feathers: int, *,
                 head_feathers: int = 512, head_count: int = 8, normalization: str = "none"):
        super(KkMultiSelfAttentionItem, self).__init__()
        self.head_count = head_count
        self.head_feathers = head_feathers
        self.QNets = nn.ModuleList([kkl.KkLinear(in_feathers, head_feathers) for _ in range(head_count)])
        self.KNets = nn.ModuleList([kkl.KkLinear(in_feathers, head_feathers) for _ in range(head_count)])
        self.VNets = nn.ModuleList([kkl.KkLinear(in_feathers, head_feathers) for _ in range(head_count)])
        self.Softmax = nn.Softmax(-1)
        self.Linear = kkl.KkLinear(head_feathers * head_count, out_feathers)
        self.Norm = kkn.get_normalization(normalization)

    def forward(self, q, k, v):
        outs = []
        for i in range(self.head_count):
            _q = self.QNets[i](q)
            _k = self.KNets[i](k)
            _v = self.VNets[i](v)
            o = torch.matmul(_q, _k.transpose(-2, -1))
            o = o / math.sqrt(self.head_feathers)
            o = self.Softmax(o)
            o = torch.matmul(o, _v)
            outs.append(o)
        o = torch.cat(outs, dim=-1)
        o = self.Linear(o)
        if self.Norm is None:
            pass
        else:
            o = self.Norm(o)
        return o


class KkMultiSelfAttention(kkb.KkModule):
    def __init__(self, in_feathers: int, out_feathers: int, *,
                 head_feathers: int = 128, head_count: int = 8, loops: int = 6,
                 normalization: str = "none"):
        super(KkMultiSelfAttention, self).__init__()
        self.out_feathers = out_feathers
        inner_feathers = head_feathers * head_count
        self.SA1 = KkMultiSelfAttentionItem(in_feathers, inner_feathers,
                                            head_feathers=head_feathers, head_count=head_count,
                                            normalization="layer")
        self.SAs = nn.ModuleList([
            KkMultiSelfAttentionItem(inner_feathers, inner_feathers,
                                     head_feathers=head_feathers, head_count=head_count,
                                     normalization="layer") for _ in range(loops)])
        self.SAl = KkMultiSelfAttentionItem(inner_feathers, out_feathers,
                                            head_feathers=head_feathers, head_count=head_count)

    def forward(self, q, k=None, v=None):
        if k is None:
            k = q
        if v is None:
            v = q
        o = self.SA1(q, k, v)
        for isa in self.SAs:
            o = isa(o, o, o)
        o = self.SAl(o, o, o)
        return o


