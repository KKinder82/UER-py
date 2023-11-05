import os

import torch
import torch.distributed as dist
import torch.utils.data.distributed as dist_data
import torch.nn.parallel.distributed as dist_nn

import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
import math
import logging as log
import tqdm


class KKM_Config(object):
    def __init__(self):
        # GPU 训练
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_count = torch.cuda.device_count()
        # 分布式训练
        self.rank = int(os.environ['RANK'])
        self.local_rank = 0                 # int(os.environ['LOCAL_RANK'])
        self.world_size = 2                 # int(os.environ['WORLD_SIZE'])
        self.MASTER_ADDR = "127.0.0.1"      # os.environ['MASTER_ADDR']
        self.Master_PORT = "16666"          # os.environ['MASTER_PORT']
        self.backend = "nccl"
        # 数据集
        self.shuffle = True
        self.num_workers = 2 * self.world_size
        self.pin_memory = True
        # 训练
        self.batch_size = 10
        self.epoch = 100
        self.accumulation_steps = 5
        self.save_checkpoint_steps = 20
        self.report_steps = 1
        self.Checkpoint_Last = "rbBall_last.pth"
        self.Checkpoint_Best= "rbBall_best.pth"



class Kk_Module(nn.Module):
    def __init__(self, config: KKM_Config):
        super(Kk_Module, self).__init__()
        self.config = config

    def init_weights(self):
        pass


class Kk_Finetune(Kk_Module):
    def __init__(self, config: KKM_Config):
        super(Kk_Finetune, self).__init__(config)


class Kk_Dataset(data.Dataset):
    def __init__(self, config: KKM_Config, in_data=[]):
        super(Kk_Dataset, self).__init__()
        self.data = in_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class Kk_train(object):
    def __init__(self, config: KKM_Config, model: Kk_Finetune, dataset: Kk_Dataset, dataset_val: Kk_Dataset,
                 lossFn, optim):
        super(Kk_train, self).__init__()
        self.loss_value = 0.0
        self.last_loss = None
        self.config = config
        self.gpu_count = config.gpu_count
        self.dataset = dataset
        self.dataset_val = dataset_val
        self.sampler_val = None
        self.dataLoader_val = None
        self.model_src = model
        self.model = model
        self.lossFn = lossFn
        self.optim = optim

    def _val(self):
        pass

    def _epoch(self, iepoch: int):
        self.optim.zero_grad()
        need_optim = False
        for ibatch, batch in enumerate(self.dataLoader):
            x, y = batch
            loss = self._batch(iepoch, ibatch, x, y)
            need_optim = True
            if ibatch % self.config.report_steps == 0:
                log.info("  >> Epoch {}/{}, batch {}/{}, loss:{}".format(iepoch, self.config.epoch,
                                                                         ibatch, len(self.dataLoader),
                                                                         loss))
            if ibatch % self.config.accumulation_steps == 0:
                self.optim.step()
                self.optim.zero_grad()
                need_optim = False

            if ibatch % self.config.save_checkpoint_steps == 0:
                torch.save(self.model, "rbBall_last.pth".format(iepoch))

            if self.last_loss is None or self.last_loss > loss:
                self.last_loss = loss
                torch.save(self.model, "rbBall_best.pth".format(iepoch))
        if need_optim:
            self.optim.step()
            self.optim.zero_grad()
            # need_optim = False
        # 进行

    def _batch(self, iepoch, ibatch, x, y):
        x.to(self.config.device)
        x.to(self.config.device)
        o = self.model(x)
        loss = self.lossFn(o, y)
        loss.backward()
        return loss.item()


    def train(self):
        if self.config.gpu_count > 1:
            # 分布式处理
            dist.init_process_group(backend=self.config.backend, init_method="env://",  # init_method="store" 手工
                                    world_size=self.config.world_size, rank=self.config.rank)
            self.sampler = torch.utils.data.distributed.DistributedSampler(self.dataset,
                                                                           rank=self.config.rank,
                                                                           num_replicas=self.config.world_size)
            self.dataLoader = data.DataLoader(self.dataset, batch_size=self.config.batch_size,
                                              shuffle=self.config.shuffle,
                                              sampler=self.sampler, num_workers=self.config.num_workers,
                                              pin_memory=self.config.pin_memory)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model_src, device_ids=[self.config.local_rank],
                                                                   output_device=self.config.local_rank)
            torch.cuda.set_device(self.config.local_rank)
        elif self.config.device == "cuda":
            torch.cuda.set_device(self.config.local_rank)
            # self.sampler = None
            self.dataLoader = data.DataLoader(self.dataset, batch_size=self.config.batch_size,
                                              shuffle=self.config.shuffle,
                                              sampler=None, num_workers=self.config.num_workers,
                                              pin_memory=self.config.pin_memory)
            self.model = self.model_src.to(self.config.device)
        else:
            # cpu
            # self.sampler = None
            self.dataLoader = data.DataLoader(self.dataset, batch_size=self.config.batch_size,
                                              shuffle=self.config.shuffle,
                                              sampler=None, num_workers=self.config.num_workers,
                                              pin_memory=self.config.pin_memory)
            self.model = self.model_src.to(self.config.device)
        self.loss_value = 0.0
        for iepoch in tqdm(self.config.epoch, desc="Epoch"):
            self._epoch(iepoch)
            pass
        pass


if __name__ == "__main__":
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(x.shape)
    print(x[1:2, :])
