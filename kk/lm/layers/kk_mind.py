import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import kk.lm.kk_app as kka
import kk.lm.kk_base as kkb
import kk.lm.kk_config as kkc
import kk.kk_utils as kku
import kk.lm.layers.kk_normalization as kkn
import math
import copy


class KkLinearEx(kkb.KkModule):
    def __init__(self, in_feathers: int, out_feathers: int, *, inner_feathers: int = 32):
        super(KkLinearEx, self).__init__()
        _size = out_feathers * inner_feathers * 3
        self.out_feathers = out_feathers
        self.Linear1 = nn.Linear(in_feathers, _size, bias=True)
        self.Relu = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.Linear2 = nn.Linear(inner_feathers * 3, 1, bias=True)
        pass

    def forward(self, x):
        o = self.Linear1(x)
        bsize = x.size(0)
        o = o.view(x.size(0), self.out_feathers, 3, -1)
        o1 = self.Relu(o[:, :, 0:1])
        o2 = self.Sigmoid(o[:, :, 1:2])
        o3 = self.tanh(o[:, :, 2:3])
        o = torch.cat((o1, o2, o3), dim=-2)
        o = o.view(bsize, self.out_feathers, -1)
        o = self.Linear2(o)
        o.squeeze_(dim=-1)
        return o


class KkParabola(kkb.KkModule):
    def __init__(self, in_feathers, out_feathers):
        super(KkParabola, self).__init__()
        _weight2 = kkb.get_randn_parameter(in_feathers, out_feathers, mean=0.0, std=0.01)
        self.weight2 = nn.Parameter(_weight2)
        _weight1 = kkb.get_randn_parameter(in_feathers, out_feathers, mean=0.0, std=0.01)
        self.weight = nn.Parameter(_weight1)
        _bias = kkb.get_constant_parameter(1, out_feathers)
        self.bias = nn.Parameter(_bias)
        pass

    def forward(self, x):
        _x = torch.pow(x, 2)
        o2 = torch.matmul(_x, self.weight2)
        o1 = torch.matmul(x, self.weight)
        o = o2 + o1 + self.bias
        return o


class KkSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        result = 1 / (1 + torch.exp(-x))
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        return result * (1 - result) * grad_output


if __name__ == "__main__":
    x = torch.arange(2 * 3, dtype=torch.float32).view(2, 3)
    model = KkLinearEx(3, 4)
    o = model(x)
    print(o)


