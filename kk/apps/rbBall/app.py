import numpy as np
import os
import torch
import torch.nn as nn
import kk.apps.rbBall.models as models
import kk.apps.kk_app as kka
import kk.uer.kk_config as kkc
import kk.kk_utils as kku
import kk.kk_datetime as kkdt
import torch.utils.data as data
import torch.utils.data.distributed as dist_data


def train():
    _path = os.path.dirname(os.path.abspath(__file__))
    config = kkc.KkmConfig(_path)
    config.batch_size = 2
    model = models.RBModel()
    dataset = kka.KkDataset(path_np="data/rbBall_train.npy", x_len=88)
    dataset_val = kka.KkDataset(path_np="data/rbBall_val.npy", x_len=88)
    loss_fun = kka.KkClassfierLoss(blocks=[33], counts=[6, 1])
    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    trainer = kka.KkTrain(model=model, dataset=dataset, dataset_val=dataset_val,
                          loss_fn=loss_fun, optim=optim)
    trainer.train()


def _date_onehot(date_):
    date_ = kkdt.kk_datetime(date_, out_format=5)
    zh_date = kkdt.kk_date_ganzhi(date_[0], date_[1], date_[2], 19, 1)
    tdata = kku.kk_onehot(int(zh_date[0]), 10)
    tdata += kku.kk_onehot(int(zh_date[1]), 12)
    tdata += kku.kk_onehot(int(zh_date[2]), 10)
    tdata += kku.kk_onehot(int(zh_date[3]), 12)
    tdata += kku.kk_onehot(int(zh_date[4]), 10)
    tdata += kku.kk_onehot(int(zh_date[5]), 12)
    tdata += kku.kk_onehot(int(zh_date[6]), 10)
    tdata += kku.kk_onehot(int(zh_date[7]), 12)
    o = torch.tensor(tdata).view(1, -1)
    return o


infer_object = None
infer_date = "2014-1-12"
infer_config = None


def infer():
    global infer_object
    global infer_date
    global infer_config
    if infer_object is None:
        _path = os.path.dirname(os.path.abspath(__file__))
        infer_config = kkc.KkmConfig(_path)

        # config.batch_size = 1
        model = models.RBModel()
        infer_object = kka.KkInference(model)

    x = _date_onehot(infer_date)
    x = x.float().to(infer_config.config.device)
    o = infer_object(x)
    torch.top
    print(o)


def test():
    x = "2023-11-14"
    # x = _date_onehot(infer_date)
    # o = infer_object(x)
    o = _date_onehot(infer_date)
    print(o)
    exit()

    _path = os.path.dirname(os.path.abspath(__file__))
    config = kkc.KkmConfig(_path)
    config.sys_init()

    dataset = kka.KkDataset(path_np="data/rbBall_train.npy", x_len=88)
    # _data = torch.arange(10 * (88+49)).reshape(10, 88+49).float()
    # dataset = kka.KkDataset(config, data=_data, x_len=88)

    dataset_val = kka.KkDataset(path_np="data/rbBall_val.npy", x_len=88)
    # _data = torch.arange(10 * (88+49)).reshape(10, 88+49).float()
    # dataset_val = kka.KkDataset(config, data=_data, x_len=88)

    model = models.RBModel()
    loss_fn = kka.KkExtendLoss(lossFn=nn.MSELoss())
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = kka.KkTrain(model=model, dataset=dataset, dataset_val=dataset_val,
                          loss_fn=loss_fn, optim=optim)
    trainer.train()


if __name__ == "__main__":
    train()
