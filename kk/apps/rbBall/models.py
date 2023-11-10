import torch
import torch.nn as nn
import torch.utils.data as torchdata
import numpy as np
import kk.kk_utils as kku
import kk.apps.kk_app as kka
import kk.uer.layers.kk_Linear as kkl
import kk.uer.layers.kk_Transformer as kkt
import kk.uer.kk_base as kkb
import kk.uer.kk_config as kkc


class RBModel(kkb.KkModule):
    def __init__(self, config: kkc.KkmConfig, context=None):
        super(RBModel, self).__init__(config)
        self.context_linner = kkl.KkFFNLayer(config, 33 + 16, 128)
        self.x_linner = kkl.KkFFNLayer(config, 88, 128)
        self.backbone = kkt.KkTransformer(config=config, in_feather=128, loops=3)
        self.classifier = kkl.KkClassifierLayer(config, 128, 49, one_formal="sigmoid")
        if context is None:
            context = torch.zeros((1, 33 + 16), dtype=torch.float32)
        self.register_buffer("context", context)
        self.last_o = None

    def forward(self, x):
        _context = self.context_linner(self.context)
        _x = self.x_linner(x)
        o = self.backbone(_context, _x, last_o=None)
        o = self.classifier(o)
        return o

    def before_forward(self, **args):
        x = args["x"]
        y = args["y"]
        pass

    def after_forward(self, **args):
        x = args["x"]
        y = args["y"]
        _y = torch.sum(y, dim=0)
        self.context = self.context + _y
        pass

    def reset_epoch(self, **args):
        iepoch = args["iepoch"]
        self.context -= self.context  # torch.zeros((1, 33 + 16), dtype=torch.float32)
        pass