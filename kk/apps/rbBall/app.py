import numpy as np
import os
import torch
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

    sampler = dist_data.DistributedSampler(dataset, rank=0,
                                           num_replicas=1, shuffle=False)
    dataLoader = data.DataLoader(dataset, batch_size=config.batch_size,
                                 shuffle=False, sampler=sampler, num_workers=2,
                                 pin_memory=config.pin_memory)

    for i, (x, y) in enumerate(dataLoader):
        print(i, " : ", x.shape, y.shape)


if __name__ == "__main__":
    test()
