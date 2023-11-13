import os
import torch
import torch.nn as nn
import torch.utils.data as torchdata
import numpy as np
import kk.kk_utils as kku
import kk.apps.kk_app as kka
import kk.uer.kk_base as kkb
from torch.nn.parallel import DistributedDataParallel as DDP
import kk.uer.layers.kk_linear as kkl
import kk.uer.layers.kk_transformer as kkt
import kk.uer.layers.kk_selfattention as kksa
import kk.uer.kk_config as kkc


class KkTestModel(kka.KkAppModel):
    def __init__(self, *, in_feathers: int = 2):
        super(KkTestModel, self).__init__()
        self.Linear = kkl.KkLinear(in_feathers=in_feathers, out_feathers=1)

    def forward(self, x):
        o = self.Linear(x)
        return o


class KkTestSelfAttationModel(kka.KkAppModel):
    def __init__(self, *, in_feathers: int = 10):
        super(KkTestSelfAttationModel, self).__init__()

        self.net = kksa.KkSelfAttentionItem(in_feathers=in_feathers, out_feathers=in_feathers)
        self.Linear = kkl.KkLinear(in_feathers=in_feathers, out_feathers=1)
        # _context = torch.zeros(1, in_feathers, dtype=torch.float32)
        # self.register_buffer("context", _context)

    def forward(self, x):
        o = self.net(x, x, x)
        o = self.Linear(o)
        return o


class KkTestTransformModel(kka.KkAppModel):
    def __init__(self, *, in_feathers: int = 10):
        super(KkTestTransformModel, self).__init__()

        self.net = kkt.KkTransformer(in_feathers=in_feathers)
        self.Linear = kkl.KkLinear(in_feathers=in_feathers, out_feathers=1)

    def forward(self, x):
        o = self.net(x, x)
        o = self.Linear(o)
        return o
