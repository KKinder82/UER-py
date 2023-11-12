import numpy as np
import os
import torch
import torch.distributed as dist
import torch.utils.data as data
import torch.nn as nn
import logging as log
import tqdm
import models as models
import kk.apps.kk_app as kka
import kk.uer.kk_config as kkc


def test():
    _path = os.path.dirname(os.path.abspath(__file__))
    config = kkc.KkmConfig(_path)
    config.sys_init()


    dataset = kka.KkDataset(config, path_np="data/rbBall_train.npy", x_len=88)
    # _data = torch.arange(10 * (88+49)).reshape(10, 88+49).float()
    # dataset = kka.KkDataset(config, data=_data, x_len=88)

    dataset_val = kka.KkDataset(config, path_np="data/rbBall_val.npy", x_len=88)
    # _data = torch.arange(10 * (88+49)).reshape(10, 88+49).float()
    # dataset_val = kka.KkDataset(config, data=_data, x_len=88)

    model = models.RBModel(config)
    loss_fn = kka.KkExtendLoss(config, lossFn=nn.MSELoss())
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = kka.KkTrain(config, model=model, dataset=dataset, dataset_val=dataset_val,
                          loss_fn=loss_fn, optim=optim)
    trainer.train()


if __name__ == "__main__":
    test()

