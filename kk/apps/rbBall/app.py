import numpy as np
import os
import torch
import torch.nn as nn
import kk.apps.rbBall.models as models
import kk.apps.kk_app as kka
import kk.uer.kk_config as kkc
import torch.utils.data as data
import torch.utils.data.distributed as dist_data


def train():
    _path = os.path.dirname(os.path.abspath(__file__))
    config = kkc.KkmConfig(_path)

    # config.batch_size = 1
    model = models.RBModel(config)
    dataset = kka.KkDataset(config, path_np="data/rbBall_train.npy", x_len=88)
    dataset_val = kka.KkDataset(config, path_np="data/rbBall_val.npy", x_len=88)
    loss_fun = kka.KkClassfierLoss(config, blocks=[33], counts=[6, 1])
    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    trainer = kka.KkTrain(config, model=model, dataset=dataset, dataset_val=dataset_val,
                          loss_fn=loss_fun, optim=optim)
    trainer.train()


def prediction():
    _path = ""


def test():
    _path = os.path.dirname(os.path.abspath(__file__))
    config = kkc.KkmConfig(_path)
    config.sys_init()

    dataset = kka.KkDataset(config, path_np="data/rbBall_train.npy", x_len=88)
    # _data = torch.arange(10 * (88+49)).reshape(10, 88+49).float()
    # dataset = kka.KkDataset(config, data=_data, x_len=88)

    dataset_val = kka.KkDataset(config, path_np="data/rbBall_val.npy", x_len=88)
    # _data = torch.arange(10 * (88+49)).reshape(10, 88+49).float()
    # dataset = kka.KkDataset(config, data=_data, x_len=88)

    model = kka.KkDemoModel(config, in_feather=99)
    loss_fn = kka.KkExtendLoss(config, lossFn=nn.MSELoss())
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = kka.KkTrain(config, model=model, dataset=dataset, dataset_val=dataset_val,
                          loss_fn=loss_fn, optim=optim)
    trainer.train()


if __name__ == "__main__":
    # os.environ['RANK'] = "0"
    # os.environ['LOCAL_RANK'] = "0"
    # os.environ['WORLD_SIZE'] = "2"
    # os.environ['MASTER_ADDR'] = "127.0.0.1"
    # os.environ['MASTER_PORT'] = "16666"

    test()
