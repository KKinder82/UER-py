import numpy as np
import os
import torch
import kk.apps.rbBall.models as models
import kk.apps.kk_app as kka
import kk.uer.kk_config as kkc


def train():
    _path = os.path.dirname(os.path.abspath(__file__))
    config = kkc.KkmConfig(_path)

    # config.batch_size = 1
    model = models.RBModel(config)
    dataset = kka.KkDataset(config, path_np="data/rbBall_train.npy", x_len=88)
    dataset_val = kka.KkDataset(config, path_np="data/rbBall_val.npy", x_len=88)
    loss_fun = kka.KkClassfierLoss(config, blocks=[88], counts=[6, 1])
    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    trainer = kka.KkTrain(config, model=model, dataset=dataset, dataset_val=dataset_val,
                          loss_fn=loss_fun, optim=optim)
    trainer.train()


def prediction():
    _path = ""


if __name__ == "__main__":
    # print(os.path.abspath("./../../../"))
    train()
