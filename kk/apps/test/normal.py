import numpy as np
import os
import torch
import torch.distributed as dist
import torch.utils.data as data
import torch.nn as nn
import logging as log
import tqdm
import kk.lm.kk_config as kkc

class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.linear = nn.Linear(2, 1)
        self.cc = nn.Parameter(torch.randn((2, 3), dtype=torch.float32))

    def forward(self, x):
        o = self.linear(x)
        return o


class TestDataset(data.Dataset):
    def __init__(self):
        self.data = torch.arange(200000, dtype=torch.float32).reshape(100000, 2)
        print(self.data)

    def __getitem__(self, index):
        # 根据索引获取样本
        return self.data[index]

    def __len__(self):
        # 返回数据集大小
        return len(self.data)


if __name__ == "__main__":
    config = kkc.KkmConfig(__file__)
    dataset = TestDataset()
    dataloader = data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    dataloader2 = data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

    model = TestModel()
    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    y = torch.arange(2, dtype=torch.float32).reshape(2, 1)
    print(y)
    for i, x in enumerate(dataloader):
        optim.zero_grad()
        o = model(x)
        loss = loss_fn(o, y)
        loss.backward()
        optim.step()
        print(loss.item())

