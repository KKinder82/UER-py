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
