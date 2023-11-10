import torch
import torch.nn as nn
import torch.utils.data as torchdata
import numpy as np
import kk.kk_utils as kku
import kk.apps.kk_app as kka
import kk.uer.layers.kk_Linear as kkl
import kk.uer.layers.kk_Transformer as kkt
import kk.uer.kk_config as kkc


class NumInferModel(kka.KkModule):
    def __init__(self, config: kkc.KkmConfig):
        super(NumInferModel, self).__init__(config)
        self.L1 = nn.Linear(4, 2)
        self.Relu1 = nn.ReLU()
        self.L2 = nn.Linear(2, 2)
        self.Relu2 = nn.ReLU()
        self.L3 = nn.Linear(2, 4)
        self.Softmax = nn.Softmax(dim=1)

    def forward(self, x):
        o = self.L1(x)
        o = self.Relu1(o)
        o = self.L2(o)
        o = self.Relu2(o)
        o = self.L3(o)
        o = self.Softmax(o)
        return o
