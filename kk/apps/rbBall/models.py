import torch
import torch.nn as nn
import torch.utils.data as torchdata
import numpy as np
import kk.kk_utils as kku
import kk.apps.kk_app as kka
import kk.uer.layers.kk_Linear as kkl
import kk.uer.layers.kk_Transformer as kkt


class RBModel(kka.KkModule):
    def __init__(self, config: kka.KkmConfig, context=None):
        super(RBModel, self).__init__(config)
        self.backbone = kkt.KkTransformer(config=config, in_feather=88, loops=6)
        self.classifier = kkl.KkClassifierLayer(config, 88, 49, classifierMode=7)
        if context is None:
            context = torch.zeros((1, 88), dtype=torch.float32)
        self.register_buffer("context", context)
        self.last_o = None

    def forward(self, x):
        o = self.backbone(self.context, x, last_o=None)
        o = self.classifier(o)
        return o

