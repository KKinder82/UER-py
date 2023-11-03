import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import kk.finetune.kk_finetune as kkf
import kk.uer.kk_module as kkm
import kk.uer.layers.kk_Transformer as kkt

class RBModel(kkf.Kk_Finetune):
    def __init__(self, config:kkm.KKM_Config):
        super(RBModel, self).__init__()
        self.net = kkt.Kk_Transformer(config=config, in_feather=88, out_feather=88, loops=6)

    def forward(self, x, context):
        o = self.net(context, x)
        return o


def train():
    train = np.load("../datasets/rbBall/rbBall_train.npy")
    val = np.load("../datasets/rbBall/rbBall_train.npy")
    # test = np.load("../datasets/rbBall/rbBall_train.npy")
    x = train[:, :88]
    y = train[:, 88:]

    # 编码

def main():



    print(y.shape)

    pass


if __name__ == "__main__":
    main()