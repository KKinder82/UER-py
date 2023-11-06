import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import kk.apps.kk_app as kka
import kk.uer.layers.kk_Selfattation as kksa
import kk.uer.layers.kk_Linear as kkl
import math


class KkNormalization(kka.KkModule):
    def __init__(self, config: kka.KkmConfig, eps: float = 1e-5):
        super(KkNormalization, self).__init__(config)
        self.eps = eps

    def forward(self, x):
        # x = x.type(torch.float64)
        _mean = x.mean(dim=-1, keepdim=True)
        _var = torch.var(x, dim=-1, keepdim=True, unbiased=False)
        return (x - _mean) / (_var + self.eps).sqrt()


class KkBatchNormal(KkNormalization):
    def __init__(self, config: kka.KkmConfig, eps: float = 1e-5):
        super(KkBatchNormal, self).__init__(config, eps)

    # 批次, 通道, 数据
    def forward(self, x):
        _s = x.shape
        o = x.transpose(0, 1)
        o = o.reshape(_s[1], -1)
        o = super(KkBatchNormal, self).forward(o)
        return o.view(_s[1], _s[0], *_s[2:]).transpose(0, 1).contiguous()


class KkLayerNormal(KkNormalization):
    def __init__(self, config: kka.KkmConfig, eps: float = 1e-5):
        super(KkLayerNormal, self).__init__(config, eps)

    # 批次, 通道, 数据
    def forward(self, x):
        _s = x.shape
        o = x.view(_s[0], -1)
        o = super(KkLayerNormal, self).forward(o)
        return o.reshape(*_s)


class KkInstanceNormal(KkNormalization):
    def __init__(self, config: kka.KkmConfig, eps: float = 1e-5):
        super(KkInstanceNormal, self).__init__(config, eps)

    # 批次, 通道, 数据
    def forward(self, x):
        _s = x.shape
        o = x.view(*_s[0:2], -1)
        o = super(KkInstanceNormal, self).forward(o)
        return o.reshape(*_s)


class KkGroupNormal(KkNormalization):
    def __init__(self, config: kka.KkmConfig, groups: int = 2, eps: float = 1e-5):
        super(KkGroupNormal, self).__init__(config, eps)
        self.groups = groups

    # 批次, 通道, 数据
    def forward(self, x):
        _s = x.shape
        if _s[1] % self.groups != 0:
            raise ValueError("分组不匹配")
        o = x.view(*_s[0:1], self.groups, -1)
        o = super(KkGroupNormal, self).forward(o)
        return o.reshape(*_s)


class KkGroupNormalEx(KkNormalization):
    def __init__(self, config: kka.KkmConfig, groups: int, dim: int = 1, eps: float = 1e-5):
        super(KkGroupNormalEx, self).__init__(config, eps)
        self.groups = groups
        self.dim = dim

    # 批次, 通道, 数据
    def forward(self, x):
        _s = x.shape
        if _s[self.dim] % self.groups != 0:
            raise ValueError("分组不匹配")
        o = x.view(*_s[0:self.dim], self.groups, -1)
        o = super(KkGroupNormalEx, self).forward(o)
        return o.reshape(*_s)


if __name__ == "__main__":
    x = torch.arange(2 * 6 * 4, dtype=torch.float32).view(2, 6, 4)
    # x = torch.tensor([[[1, 2, 3], [5, 5, 6], [1, 8, 9]]]).float()
    bn = KkBatchNormal(None)
    print(bn(x))
    bn = nn.GroupNorm(2, 6)
    print(bn(x))