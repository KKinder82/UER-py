import torch
import torch.nn as nn
import kk.lm.kk_base as kkb
import kk.lm.kk_config as kkc


def get_normalization(normalization: str, groups: int = 2):
    if normalization == "batch":
        return KkBatchNormal()
    elif normalization == "layer":
        return KkLayerNormal()
    elif normalization == "instance":
        return KkInstanceNormal()
    elif normalization == "group":
        return KkGroupNormal(groups)
    else:
        return None


class KkNormalization(kkb.KkModule):
    def __init__(self, *,
                 mean: float = 0, std: float = 1, eps: float = 1e-5):
        super(KkNormalization, self).__init__()
        self.eps = eps
        self.normalization_mean = mean
        self.normalization_std = std

    def forward(self, x):
        # x = x.type(torch.float64)
        _mean = x.mean(dim=-1, keepdim=True)
        _var = torch.var(x, dim=-1, keepdim=True, unbiased=False)
        o = (x - _mean) / (_var + self.eps).sqrt()  # 标准
        o = o * self.normalization_std + self.normalization_mean                # 自定义
        return o


class KkBatchNormal(KkNormalization):
    def __init__(self, *, eps: float = 1e-5):
        super(KkBatchNormal, self).__init__(eps=eps)

    # 批次, 通道, 数据
    def forward(self, x):
        _s = x.shape
        o = x.transpose(0, 1)
        o = o.reshape(_s[1], -1)
        o = super(KkBatchNormal, self).forward(o)
        return o.view(_s[1], _s[0], *_s[2:]).transpose(0, 1).contiguous()


class KkLayerNormal(KkNormalization):
    def __init__(self, *, eps: float = 1e-5):
        super(KkLayerNormal, self).__init__(eps=eps)

    # 批次, 通道, 数据
    def forward(self, x):
        _s = x.shape
        o = x.view(_s[0], -1)
        o = super(KkLayerNormal, self).forward(o)
        return o.reshape(*_s)


class KkInstanceNormal(KkNormalization):
    def __init__(self, *, eps: float = 1e-5):
        super(KkInstanceNormal, self).__init__(eps=eps)

    # 批次, 通道, 数据
    def forward(self, x):
        _s = x.shape
        o = x.view(*_s[0:2], -1)
        o = super(KkInstanceNormal, self).forward(o)
        return o.reshape(*_s)


class KkGroupNormal(KkNormalization):
    def __init__(self, groups: int = 2, *, eps: float = 1e-5):
        super(KkGroupNormal, self).__init__(eps=eps)
        self.groups = groups

    # 批次, 通道, 数据
    def forward(self, x):
        _s = x.shape
        if _s[1] % self.groups != 0:
            raise ValueError("分组不匹配")
        o = x.view(*_s[0:1], self.groups, -1)
        o = super(KkGroupNormal, self).forward(o)
        return o.reshape(*_s)


class KkGroupNormalDim(KkNormalization):
    def __init__(self, config: kkc.KkmConfig, groups: int, dim: int = 1, *, eps: float = 1e-5):
        super(KkGroupNormalDim, self).__init__(config, eps=eps)
        self.groups = groups
        self.dim = dim

    # 批次, 通道, 数据
    def forward(self, x):
        _s = x.shape
        if _s[self.dim] % self.groups != 0:
            raise ValueError("分组不匹配")
        o = x.view(*_s[0:self.dim], self.groups, -1)
        o = super(KkGroupNormalDim, self).forward(o)
        return o.reshape(*_s)


if __name__ == "__main__":
    x = torch.arange(2 * 6 * 4, dtype=torch.float32).view(2, 6, 4)
    # x = torch.tensor([[[1, 2, 3], [5, 5, 6], [1, 8, 9]]]).float()
    bn = KkBatchNormal(None)
    print(bn(x))
    bn = nn.GroupNorm(2, 6)
    print(bn(x))
