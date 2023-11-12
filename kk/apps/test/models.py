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
import kk.uer.kk_config as kkc


class KkTestModel(kka.KkAppModel):
    def __init__(self, config: kkc.KkmConfig, *, in_feather: int = 2):
        super(KkTestModel, self).__init__(config)
        self.Linear = kkl.KkLinear(config, in_feather=in_feather, out_feather=1)

    def forward(self, x):
        o = self.Linear(x)
        return o


class KkTestTransformerModel(kka.KkAppModel):
    def __init__(self, config: kkc.KkmConfig, *, in_feather: int = 10):
        super(KkTestTransformerModel, self).__init__(config)
        self.net = kkt.KkTransformer(config, in_feather=in_feather)
        self.Linear = kkl.KkLinear(config, in_feather=in_feather, out_feather=1)
        _context = torch.zeros(1, in_feather, dtype=torch.float32)
        self.register_buffer("context", _context)

    def forward(self, x):
        o = self.net(self.context, x)
        o = self.Linear(o)
        return o
