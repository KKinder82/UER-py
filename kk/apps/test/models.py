import os
import torch
import torch.nn as nn
import torch.utils.data as torchdata
import numpy as np
import kk.kk_utils as kku
import kk.apps.kk_app as kka
import kk.uer.kk_base as kkb
from torch.nn.parallel import DistributedDataParallel as DDP
import kk.uer.layers.kk_Linear as kkl
import kk.uer.layers.kk_Transformer as kkt
import kk.uer.layers.kk_Selfattation as kksa
import kk.uer.kk_config as kkc


class KkTestModel(kka.KkAppModel):
    def __init__(self, config: kkc.KkmConfig, *, in_feather: int = 2):
        super(KkTestModel, self).__init__(config)
        self.Linear = kkl.KkLinear(config, in_feather=in_feather, out_feather=1)

    def forward(self, x):
        o = self.Linear(x)
        return o


class KkTestSelfAttationModel(kka.KkAppModel):
    def __init__(self, config: kkc.KkmConfig, *, in_feather: int = 10):
        super(KkTestSelfAttationModel, self).__init__(config)

        self.net = kksa.KkSelfAttationItem(config, qk_feathers=in_feather, v_feathers=in_feather,
                                           out_feathers=in_feather)
        self.Linear = kkl.KkLinear(config, in_feather=in_feather, out_feather=1)
        # _context = torch.zeros(1, in_feather, dtype=torch.float32)
        # self.register_buffer("context", _context)

    def forward(self, x):
        o = self.net(x, x, x)
        o = self.Linear(o)
        return o
