import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import kk.uer.kk_module as kkm
import kk.uer.layers.kk_SelfAttation as kksa
import kk.uer.layers.kk_Linear as kkl
import math


class Kk_TransformerEncode(kkm.Kk_Module):
    def __init__(self, config: kkm.KKM_Config, in_feather: int,
                 head_feathers: int = 128, head_size: int = 6, loops: int = 6):
        super(Kk_TransformerEncode, self).__init__(config)
        self.MSA = kksa.Kk_MultiSelfAttation(config, in_feather, in_feather, loops=loops, out_feathers=in_feather)
        self.MSANorm = nn.LayerNorm(in_feather)
        self.FFN = kkl.Kk_FFNLayer(config, in_feather, in_feather)
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


class Kk_TransformerDecode(kkm.Kk_Module):
    def __init__(self, config: kkm.KKM_Config, in_feather: int,
                 head_feathers: int = 128, head_size: int = 6, loops: int = 6):
        super(Kk_TransformerDecode, self).__init__(config)
        self.MSA = kksa.Kk_MultiSelfAttation(config, in_feather, in_feather, loops=loops, out_feathers=in_feather)
        self.MSANorm = nn.LayerNorm(in_feather)
        self.MSA2 = kksa.Kk_MultiSelfAttation(config, in_feather, in_feather, loops=loops, out_feathers=in_feather)
        self.MSA2Norm = nn.LayerNorm(in_feather)
        self.FFN = kkl.Kk_FFNLayer(config, in_feather, in_feather)
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


class Kk_Transformer(kkm.Kk_Module):
    def __init__(self, config: kkm.KKM_Config, in_feathers: int, out_feather: int,
                 head_feathers: int = 128, head_size: int = 8, loops: int = 6):
        super(Kk_Transformer, self).__init__(config)
        self.encoder = Kk_TransformerEncode(config, in_feathers,
                                            head_feathers=head_feathers, head_size=head_size, loops=loops)
        self.decoder = Kk_TransformerDecode(config, in_feathers,
                                            head_feathers=head_feathers, head_size=head_size, loops=loops)
        self.FFN = kkl.Kk_FFNLayer(config, in_feathers, out_feather)
        self.FFNSoftMax = nn.Softmax(-1)

    def forward(self, in_context, in_x, last=None, out_format: int = 0):
        if last is None:
            in_context = self.encoder(in_context)
        else:
            in_context = last
        o = self.decoder(in_x, in_context)
        o = self.FFN(o)
        o1 = o
        o = self.FFNSoftMax(o)
        if out_format == 0:
            return o
        return o, o1


if __name__ == "__main__":
    context = torch.arange(88, dtype=torch.float32).view(1, 1, 88)
    x = torch.arange(88, dtype=torch.float32).view(1, 1, 88)
    net = Kk_Transformer(None, 88, 88, 128, 6)
    o = net(context, x)
    print(o)

