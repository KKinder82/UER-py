import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import kk.uer.kk_module as kkm
import kk.uer.layers.kk_Linear as kkl
import math


class Kk_SelfAttationItem(kkm.Kk_Module):
    def __init__(self, config: kkm.KKM_Config, qk_feathers: int, out_feathers: int, v_feathers: int = 0, inner_feathers: int = 256) -> np.array:
        super(Kk_SelfAttationItem, self).__init__(config)
        if v_feathers == 0:
            v_feathers = qk_feathers
        self.inner_feathers = inner_feathers
        self.QNet = nn.Linear(qk_feathers, inner_feathers)
        self.KNet = nn.Linear(qk_feathers, inner_feathers)
        self.VNet = nn.Linear(v_feathers, out_feathers)
        self.Softmax = nn.Softmax(-1)

        self.Dropout1 = nn.Dropout(0.3)

    def forward(self, q, k, v, postion_encoding: bool = False):
        if postion_encoding:
            pass
        q = self.QNet(q)
        k = self.KNet(k)
        v = self.VNet(v)
        o = torch.matmul(q, k.transpose(-2, -1))
        o = o / math.sqrt(self.inner_feathers)
        o = self.Softmax(o)
        o = torch.matmul(o, v)
        o = self.Dropout1(o)
        return o

class Kk_SelfAttation(kkm.Kk_Module):
    def __init__(self, config: kkm.KKM_Config, qk_feathers: int, out_feathers: int, v_feathers: int = 0,
                 inner_feathers: int = 256, loops: int = 6) -> np.array:
        super(Kk_SelfAttation, self).__init__(config)
        self.SA1 = Kk_SelfAttationItem(config, qk_feathers, inner_feathers, v_feathers, inner_feathers)
        self.SAs = [Kk_SelfAttationItem(config, inner_feathers, inner_feathers, inner_feathers, inner_feathers) for _ in range(loops)]
        self.SAl = Kk_SelfAttationItem(config, inner_feathers, out_feathers, inner_feathers, inner_feathers)

    def forward(self, q, k, v):
        o = self.SA1(q, k, v)
        for isa in self.SAs:
            o = isa(o, o, o)
        o = self.SAl(o, o, o)
        return o


class Kk_MultiSelfAttationItem(kkm.Kk_Module):
    def __init__(self, config: kkm.KKM_Config, qk_feathers: int, v_feathers: int = 0,
                 head_feathers: int = 512, head_count: int = 8, out_feathers: int = 0) -> np.array:
        super(Kk_MultiSelfAttationItem, self).__init__(config)
        self.head_count = head_count
        self.head_feathers = head_feathers
        if v_feathers == 0:
            v_feathers = qk_feathers

        self.QNet = nn.Linear(qk_feathers, head_feathers * head_count)
        self.KNet = nn.Linear(qk_feathers, head_feathers * head_count)
        self.VNet = nn.Linear(v_feathers, head_feathers * head_count)
        self.Softmax = nn.Softmax(-1)
        self.Linear = None
        if out_feathers > 0:
            self.Dropout1 = nn.Dropout(0.3)
            self.Linear = nn.Linear(head_feathers * head_count, out_feathers)

    def forward(self, q, k, v):
        q = self.QNet(q)
        q = q.view(*q.shape[0:-1], self.head_count, -1)
        q = q.transpose(-3, -2)
        k = self.KNet(k)
        k = k.view(*k.shape[0:-1], self.head_count, -1)
        k = k.transpose(-3, -2)
        v = self.VNet(v)
        v = v.view(*v.shape[0:-1], self.head_count, -1)
        v = v.transpose(-3, -2)
        o = torch.matmul(q, k.transpose(-2, -1))
        o = o / math.sqrt(self.head_feathers)
        o = self.Softmax(o)
        o = torch.matmul(o, v)
        o = o.transpose(-3, -2)
        # o = o.reshape(*o.shape[0:-2], -1)
        o = o.contiguous().view(*o.shape[0:-2], -1)
        if self.Linear is not None:
            o = self.Dropout1(o)
            o = self.Linear(o)
        return o


class Kk_MultiSelfAttation(kkm.Kk_Module):
    def __init__(self, config: kkm.KKM_Config, qk_feathers: int, v_feathers: int,
                 head_feathers: int = 512, head_count: int = 8, loops: int = 6, out_feathers: int = 0,) -> np.array:
        super(Kk_MultiSelfAttation, self).__init__(config)
        self.out_feathers = out_feathers
        self.inner_feathers = head_feathers * head_count
        self.SA1 = Kk_MultiSelfAttationItem(config, qk_feathers, v_feathers, head_feathers, head_count)
        self.SAs = nn.ModuleList([Kk_MultiSelfAttationItem(config, self.inner_feathers, self.inner_feathers,
                                                           head_feathers, head_count)
                    for _ in range(loops)])
        self.SAl = Kk_MultiSelfAttationItem(config, self.inner_feathers, self.inner_feathers,
                                            head_feathers, head_count, out_feathers)

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


if __name__ == "__main__":
    x = torch.randn((1, 88), dtype=torch.float32).view(1, 88, 1)
    # x = kkl.Kk_FeatherLayer(None, 88, 128)(x)
    SA = Kk_MultiSelfAttation(None, 1, 1, out_feathers=128)
    x = SA(x, x, x)
    print("**A**")
    print(x)
    print(x.shape)