import numpy as np
import torch
import torch.nn as nn
import logging as log
import tqdm
import models as rbModels


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train = np.load("../../datasets/rbBall/rbBall_train.npy")
    val = np.load("../../datasets/rbBall/rbBall_train.npy")
    # test = np.load("../datasets/rbBall/rbBall_train.npy")
    train = torch.tensor(train, dtype=torch.float32)
    val = torch.tensor(val, dtype=torch.float32)
    x = train[:, :88]
    y = train[:, 88:]
    log.info("开始训练")

    model = rbModels.RBModel(None)
    lossFn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    #移动
    model.to(device)
    lossFn.to(device)
    x = x.to(device)
    y = y.to(device)

    # 训练
    model.train()
    epoch = 100
    min_loss = 100000
    for iepoch in tqdm.tqdm(range(epoch)):
        log.info(">> 开始训练第{}轮".format(iepoch))
        context = torch.zeros((1, 88), dtype=torch.float32)
        last = None
        loops = len(x) // 20
        optim.zero_grad()
        need_optim = False
        for i in range(loops):
            need_optim = True
            ix = x[i:i+10, :]
            iy = y[i:i+10, :]
            o, context = model(context, ix, last)
            context = context.detach()
            loss = lossFn(o, iy)
            loss.backward()
            if i % 10 == 0:
                optim.step()
                optim.zero_grad()
                need_optim = False
            log.info("  >> 第 {}/{} 优化 loss:{}".format(i+1, loops, loss.item()))

        if need_optim:
            optim.step()
            optim.zero_grad()
            need_optim = False

        torch.save(model, "rbBall_last.pth".format(iepoch))
        if min_loss > loss.item():
            min_loss = loss.item()
            torch.save(model, "rbBall_best.pth".format(iepoch))


if __name__ == "__main__":
    log.basicConfig(level=log.INFO)
    train()