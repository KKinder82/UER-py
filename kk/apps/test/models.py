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


class TestModel(kkb.KkModule):
    def __init__(self, config: kkc.KkmConfig, context=None):
        super(TestModel, self).__init__(config)
        self.net = nn.Sequential(nn.Linear(88, 10),
                                 nn.Linear(10, 10),
                                 nn.Linear(10, 10),
                                 nn.Linear(10, 49))

    def forward(self, x):
        o = self.net(x)
        return o

def test():
    config = kkc.KkmConfig(__file__)
    config.sys_init()


    dataset = kka.KkDataset(config, path_np="../rbBall/data/rbBall_train.npy", x_len=88)
    # _data = torch.arange(10 * (88+49)).reshape(10, 88+49).float()
    # dataset = kka.KkDataset(config, data=_data, x_len=88)

    dataset_val = kka.KkDataset(config, path_np="../rbBall/data/rbBall_val.npy", x_len=88)
    # _data = torch.arange(10 * (88+49)).reshape(10, 88+49).float()
    # dataset_val = kka.KkDataset(config, data=_data, x_len=88)

    model = TestModel(config)
    loss_fn = kka.KkClassfierLoss(config, blocks=[33], counts=[6, 1])
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = kka.KkTrain(config, model=model, dataset=dataset, dataset_val=dataset_val,
                          loss_fn=loss_fn, optim=optim)
    trainer.train()


if __name__ == "__main__":
    test()
