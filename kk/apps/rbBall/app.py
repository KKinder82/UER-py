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

if __name__ == "__main__":
    _path = os.path.dirname(os.path.abspath(__file__))
    config = kka.KkmConfig(_path)
    config.pt_load = False

    # config.batch_size =
    model = models.RBModel(config)
    dataset = kka.KkDataset(config, path_np="data/rbBall_train.npy", x_len=88)
    dataset_val = kka.KkDataset(config, path_np="data/rbBall_val.npy", x_len=88)
    loss_fun = kka.KkWrongLoss(config, blocks=[88], counts=[6, 1])
    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    trainer = kka.KkTrain(config, model, dataset=dataset, dataset_val=dataset_val, lossFn=loss_fun, optim=optim)
    trainer.train()
