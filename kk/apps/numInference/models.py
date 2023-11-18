
import torch.nn as nn
import kk.lm.kk_base as kkb


class NumInferModel(kkb.KkModule):
    def __init__(self,):
        super(NumInferModel, self).__init__()
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
