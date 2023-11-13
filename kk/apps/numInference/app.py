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
    config = kkc.KkmConfig(__file__)
    config.batch_size = 1

    # config.batch_size =
    model = models.NumInferModel()
    dataset = kka.KkDataset(path_np="data/numInfer_train.npy", x_len=4)
    dataset_val = kka.KkDataset(path_np="data/numInfer_val.npy", x_len=4)
    loss_fun = kka.KkClassfierLoss(lossFn=nn.BCELoss())
    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    trainer = kka.KkTrain(model=model, dataset=dataset, dataset_val=dataset_val, loss_fn=loss_fun, optim=optim)
    trainer.train()
