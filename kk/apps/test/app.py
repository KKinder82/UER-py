import numpy as np
import os
import torch
import torch.distributed as dist
import torch.utils.data as data
import torch.nn as nn
import logging as log
import tqdm
import kk.apps.test.models as models
import kk.apps.kk_app as kka
import kk.uer.kk_config as kkc


def test_simple():
    # 运行指令 torchrun --nperc-per-node 1 .\kk_app.py
    config = kkc.KkmConfig(__file__)
    datas = torch.randn(1000, 3)
    # print(datas[:, 0:88].sum(dim=1) / 3.1415926)
    datas[:, 2] = datas[:, 0:2].sum(dim=1) / 3.1415926
    dataset = kka.KkDataset(config, datas)
    datas_val = torch.randn(100, 3)
    datas_val[:, 2] = datas_val[:, 0:2].sum(dim=1) / 3.1415926
    dataset_val = kka.KkDataset(config, datas_val)
    model = models.KkTestModel(config)
    loss_fn = kka.KkExtendLoss(config, lossFn=nn.MSELoss())
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = kka.KkTrain(config, model=model, dataset=dataset, dataset_val=dataset_val,
                      loss_fn=loss_fn, optim=optim)
    trainer.train()


def test_transform():
    # 运行指令 torchrun --nperc-per-node 1 .\kk_app.py
    config = kkc.KkmConfig(__file__)
    feather_size = 11

    datas = torch.randn(1000, feather_size)
    datas[:, feather_size-1] = datas[:, 0:feather_size-2].sum(dim=1) / 3.1415926
    dataset = kka.KkDataset(config, datas)

    datas_val = torch.randn(100, feather_size)
    datas_val[:, feather_size-1] = datas_val[:, 0:feather_size-2].sum(dim=1) / 3.1415926
    dataset_val = kka.KkDataset(config, datas_val)

    model = models.KkTestTransformerModel(config)
    loss_fn = kka.KkExtendLoss(config, lossFn=nn.MSELoss())
    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    trainer = kka.KkTrain(config, model=model,
                          dataset=dataset, dataset_val=dataset_val,
                          loss_fn=loss_fn, optim=optim)
    trainer.train()


if __name__ == "__main__":
    test_transform()

