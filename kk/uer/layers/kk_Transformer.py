import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import kk.apps.kk_app as kka
import kk.uer.layers.kk_Selfattation as kksa
import kk.uer.layers.kk_Linear as kkl
import math


class KkTransformerEncode(kka.KkModule):
    def __init__(self, config: kka.KkmConfig, in_feather: int,
                 head_feathers: int = 128, head_size: int = 6, loops: int = 6):
        super(KkTransformerEncode, self).__init__(config)
        self.MSA = kksa.KkMultiSelfAttation(config, in_feather, in_feather, loops=loops, out_feathers=in_feather)
        self.MSANorm = nn.LayerNorm(in_feather)
        self.FFN = kkl.KkFFNLayer(config, in_feather, in_feather)
        self.FFNNorm = nn.LayerNorm(in_feather)
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


class KkTransformerDecode(kka.KkModule):
    def __init__(self, config: kka.KkmConfig, in_feather: int,
                 head_feathers: int = 128, head_size: int = 6, loops: int = 6):
        super(KkTransformerDecode, self).__init__(config)
        self.MSA = kksa.KkMultiSelfAttation(config, in_feather, in_feather, loops=loops, out_feathers=in_feather)
        self.MSANorm = nn.LayerNorm(in_feather)
        self.MSA2 = kksa.KkMultiSelfAttation(config, in_feather, in_feather, loops=loops, out_feathers=in_feather)
        self.MSA2Norm = nn.LayerNorm(in_feather)
        self.FFN = kkl.KkFFNLayer(config, in_feather, in_feather)
        self.FFNNorm = nn.LayerNorm(in_feather)
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


class KkTransformer(kka.KkModule):
    def __init__(self, config: kka.KkmConfig, in_feather: int,
                 head_feathers: int = 128, head_size: int = 8, loops: int = 6):
        super(KkTransformer, self).__init__(config)
        self.encoder = KkTransformerEncode(config, in_feather,
                                           head_feathers=head_feathers, head_size=head_size, loops=loops)
        self.decoder = KkTransformerDecode(config, in_feather,
                                           head_feathers=head_feathers, head_size=head_size, loops=loops)

    def forward(self, context, x, last_o=None):
        if last_o is None:
            context = self.encoder(context)
        else:
            context = last_o
        o = self.decoder(x, context)
        return o


if __name__ == "__main__":
    context = torch.arange(88, dtype=torch.float32).view(1, 1, 88)
    x = torch.arange(88, dtype=torch.float32).view(1, 1, 88)
    net = KkTransformer(None, 88, 49, 128, 6)
    o = net(context, x, activationFn="sigmoid")
    print(o)

