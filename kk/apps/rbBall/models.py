import torch
import torch.nn as nn
import torch.utils.data as torchdata
import numpy as np
import kk.uer.kk_module as kkm
import kk.uer.layers.kk_Transformer as kkt


class RBModel(kkm.KkInference):
    def __init__(self, config: kkm.KkmConfig):
        super(RBModel, self).__init__()
        self.net = kkt.KkTransformer(config=config, in_feather=88, out_feather=49, loops=6)

    def forward(self, in_context, in_x, in_last=None):
        o, lasto = self.net(in_context, in_x, in_last=in_last, activationFn="sigmoid", out_format=1)
        return o, lasto


class RBDataset(kkm.KkDataset):
    def __init__(self, config: kkm.KkmConfig, path: str):
        super(RBDataset, self).__init__(config)
        self.config = config
        _data = np.load(path)
        # 转为 tensor 格式
        _data = torch.tensor(_data, dtype=torch.float32)
        _x = _data[..., :88]
        _y = _data[..., 88:]
        self.data = list(zip(_x, _y))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    path = "../../datasets/rbBall/rbBall_train.npy"
    train = RBDataset(kkm.KkmConfig(), path)
    for item in train:
        print(item)
        break