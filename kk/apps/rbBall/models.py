import torch
import torch.nn as nn
import kk.uer.kk_module as kkm
import kk.apps.kk_app as kkf
import kk.uer.layers.kk_Transformer as kkt


class RBModel(kkf.Kk_Finetune):
    def __init__(self, config: kkm.KKM_Config):
        super(RBModel, self).__init__()
        self.net = kkt.Kk_Transformer(config=config, in_feather=88, out_feather=49, loops=6)

    def forward(self, in_context, in_x, in_last=None):
        o, lasto = self.net(in_context, in_x, in_last=in_last, activationFn="sigmoid", out_format=1)
        return o, lasto
