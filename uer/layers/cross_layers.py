import torch
import torch.nn as nn
from uer.layers.layer_norm import LayerNorm

class CrossMatrix(nn.Module):
    def __init__(self, in_feather_size, out_feather_size=None):
        super(CrossMatrix, self).__init__()
        self.feather_size = in_feather_size if isinstance(in_feather_size, (tuple, list)) else (in_feather_size, in_feather_size)
        if out_feather_size is None:
            self.out_feather_size = self.feather_size
        else:
            self.out_feather_size = out_feather_size if isinstance(out_feather_size, (tuple, list)) else (out_feather_size, out_feather_size)

        self.cross_Vector = CrossVector(self.feather_size[0] * self.feather_size[1],
                                        self.out_feather_size[0] * self.out_feather_size[1])

    # x : (batch, width, height)
    def forward(self, x):
        x = x.view(x.size(0), *self.feather_size)
        x = self.cross_Vector(x)
        x = x.view(x.size(0), *self.out_feather_size)
        return x


class CrossVector(nn.Module):
    """
    Layer Normalization.
    https://arxiv.org/abs/1607.06450
    """
    def __init__(self, in_feather_size, out_feather_size=None):
        super(CrossVector, self).__init__()
        self.feather_size = in_feather_size
        self.out_feather_size = in_feather_size if out_feather_size is None else out_feather_size
        self.inner_feather_size = (self.feather_size + 1 ) ** 2
        # self.useGate = True if useGate else False
        self.out_layer = nn.Linear(self.inner_feather_size, self.out_feather_size)

    # x : (batch, feather)
    def forward(self, x):
        src_shape = list(x.shape)
        x = x.view(-1, self.feather_size)
        _one = x.new_ones((1, 1)).expand(x.size(0), 1)
        x_1 = torch.cat((_one, x), dim=-1)
        x_1 = x_1[..., None]
        x_t = x_1.transpose(-1, -2)
        x_out = torch.bmm(x_1, x_t)
        x_out = x_out.flatten(start_dim=-2)
        x_out = self.out_layer(x_out)
        _shape = src_shape[:-1]+[self.out_feather_size]
        print(_shape)
        x_out = x_out.view(_shape)
        return x_out