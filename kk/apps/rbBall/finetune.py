import numpy as np
import os
import torch
import torch.distributed as dist
import torch.utils.data as data
import torch.nn as nn
import logging as log
import tqdm
import models as models
import kk.uer.kk_module as kkm


def ddp_setup(config: kkm.Kk_config, rank: int, world_size: int = 2):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '11166'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes 1 --nproc-per-node=2
#   py_filename.py --arguments
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --use-env --nnodes 1 --nproc_per_node=2
#   py_filename.py --arguments
# torchrun == python -m torch.distributed.launch --use-env
#  NPROC_PER_NODE
def ddp_setup_torchrun(rank: int, world_size: int = 2):
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '11166'
    dist.init_process_group("nccl")
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)

def train():
    config = kkm.KKM_Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train = models.RBDataset(config, "../../datasets/rbBall/rbBall_train.npy")
    val = models.RBDataset(config, "../../datasets/rbBall/rbBall_val.npy")
    # test = models.RBDataset(config, "../../datasets/rbBall/rbBall_test.npy")
    tloader = data.DataLoader(train, batch_size=10, shuffle=True)

    x = train[:, :88]
    y = train[:, 88:]
    log.info("开始训练")

    model = models.RBModel(None)
    lossFn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    if device == "cuda":
        ddp_setup(0, 2)
        #移动到GPU
        model.to(device)
        lossFn.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[0], output_device=0)

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

    if device == "cuda":
        torch.distributed.destroy_process_group()
        # torch.cuda.empty_cache()


if __name__ == "__main__":
    log.basicConfig(level=log.INFO)
    train()