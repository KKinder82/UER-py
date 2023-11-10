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


if __name__ == "__main__":
    # Test
    # _path = os.path.dirname(os.path.abspath(__file__))
    # config = kkc.KkmConfig(_path)
    # model = models.NumInferModel(config)
    # params = model.named_modules()
    # for name, value in params:
    #     print(name + "\t\t" + str(type(value)))
    # exit()

    # main
    _path = os.path.dirname(os.path.abspath(__file__))
    config = kkc.KkmConfig(_path)
    config.batch_size = 1

    # config.batch_size =
    model = models.TestModel(config)
    x = np.random.randint(1, 100, (100, 11))
    dataset = kka.KkDataset(config, data=x, x_len=10)
    val = np.random.randint(1, 100, (20, 11))
    dataset_val = kka.KkDataset(config, data=val, x_len=10)
    loss_fun = kka.KkClassfierLoss(config, blocks=[], counts=1)
    optim = torch.optim.RMSprop(model.parameters(), lr=0.001)

    trainer = kka.KkTrain(config, model, dataset=dataset, dataset_val=dataset_val, loss_fn=loss_fun, optim=optim)
    trainer.train()
