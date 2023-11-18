import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import kk.lm.kk_app as kka
import kk.lm.kk_base as kkb
import kk.lm.kk_config as kkc
import kk.lm.layers.kk_selfattention as kksa
import kk.lm.layers.kk_linear as kkl
import math


class KkTransformerEncode(kkb.KkModule):
    def __init__(self, in_feathers: int,
                 head_feathers: int = 128, head_size: int = 6, loops: int = 6):
        super(KkTransformerEncode, self).__init__()
        self.MSA = kksa.KkMultiSelfAttention(in_feathers, in_feathers, loops=loops)
        self.MSANorm = nn.LayerNorm(in_feathers)
        self.FFN = kkl.KkFFNLayer(in_feathers, in_feathers)
        self.FFNNorm = nn.LayerNorm(in_feathers)
        pass

    def forward(self, qkv, out_format: int = 0):
        o = self.MSA(qkv, qkv, qkv)
        o1 = o
        o = self.MSANorm(o + qkv)
        shortcut = o
        o = self.FFN(o)
        o2 = o
        o = self.FFNNorm(o + shortcut)
        if out_format == 0:
            return o
        return o, o1, o2


class KkTransformerDecode(kkb.KkModule):
    def __init__(self, in_feathers: int,
                 head_feathers: int = 128, head_size: int = 6, loops: int = 6):
        super(KkTransformerDecode, self).__init__()
        self.MSA = kksa.KkMultiSelfAttention(in_feathers, in_feathers, loops=loops)
        self.MSANorm = nn.LayerNorm(in_feathers)
        self.MSA2 = kksa.KkMultiSelfAttention(in_feathers, in_feathers, loops=loops)
        self.MSA2Norm = nn.LayerNorm(in_feathers)
        self.FFN = kkl.KkFFNLayer(in_feathers, in_feathers)
        self.FFNNorm = nn.LayerNorm(in_feathers)
        pass

    def forward(self, q, kv, out_format: int = 0):
        o = self.MSA(q, kv, kv)
        o1 = o
        o = self.MSANorm(o + kv)
        o = self.MSA2(q, o, o)
        o2 = o
        o = self.MSA2Norm(q + o)
        shortcut = o
        o = self.FFN(o)
        o3 = o
        o = self.FFNNorm(o + shortcut)
        if out_format == 0:
            return o
        return o, o1, o2, o3


class KkTransformer(kkb.KkModule):
    def __init__(self, in_feathers: int, *,
                 head_feathers: int = 128, head_size: int = 8, loops: int = 6):
        super(KkTransformer, self).__init__()
        self.encoder = KkTransformerEncode(in_feathers,
                                           head_feathers=head_feathers, head_size=head_size, loops=loops)
        self.decoder = KkTransformerDecode(in_feathers,
                                           head_feathers=head_feathers, head_size=head_size, loops=loops)

    def forward(self, context, x):
        o_context = self.encoder(context)
        o = self.decoder(x, o_context)
        return o


if __name__ == "__main__":
    context = torch.arange(88, dtype=torch.float32).view(1, 1, 88)
    x = torch.arange(88, dtype=torch.float32).view(1, 1, 88)
    net = KkTransformer(None, 88, 49, 128, 6)
    o = net(context, x, activationFn="sigmoid")
    print(o)

