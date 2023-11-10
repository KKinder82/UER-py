import os
import torch
import torch.nn as nn
import torch.utils.data as torchdata
import numpy as np
import kk.kk_utils as kku
import kk.apps.kk_app as kka
from torch.nn.parallel import DistributedDataParallel as DDP
import kk.uer.layers.kk_Linear as kkl
import kk.uer.layers.kk_Transformer as kkt
import kk.uer.kk_config as kkc


class TestModel(kka.KkModule):
    def __init__(self, config: kkc.KkmConfig, context=None):
        super(TestModel, self).__init__(config)
        self.net = nn.Sequential(nn.Linear(10, 10),
                                 nn.Linear(10, 10),
                                 nn.Linear(10, 10),
                                 nn.Linear(10, 1))

    def forward(self, x):
        o = self.net(x)
        return o


if __name__ == "__main__":
    os.environ['RANK'] = "0"
    os.environ['LOCAL_RANK'] = "0"
    os.environ['WORLD_SIZE'] = "1"
    os.environ['MASTER_ADDR'] = "127.0.0.1"
    os.environ['MASTER_PORT'] = "16666"

    config = kkc.KkmConfig(__file__)
    x = torch.arange(10, dtype=torch.float32).reshape((1, 10))
    y = torch.tensor([1], dtype=torch.float32).reshape((1, 1))
    model = TestModel(config)
    model.cuda()
    torch.cuda.set_device(0)
    x = x.cuda()
    y = y.cuda()
    lossFn = nn.MSELoss()
    optim = torch.optim.RMSprop(model.parameters(), lr=0.001)
    for ibatch in range(10):
        print("  >> ibatch <<  " + str(ibatch))
        i = -1
        modules = model.named_modules()
        for iname, imodule in modules:
            # 找是否有参数
            params = list(imodule.parameters(recurse=False))
            if len(params) > 0:
                i += 1
                print("  >>  " + str(i))
                sign = i in [0, ibatch]
            for iparam in params:
                iparam.requires_grad = sign

        o = model(x)
        loss = lossFn(o, y)
        loss.backward()
        optim.step()
        optim.zero_grad()
