import os

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F

import torch.distributed as dist
import torch.utils.data.distributed as dist_data
import torch.nn.parallel.distributed as dist_nn

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
        self.rank = int(os.environ['RANK'])                   # int(os.environ['RANK'])
        self.local_rank = int(os.environ['LOCAL_RANK'])       # int(os.environ['LOCAL_RANK'])
        self.world_size = int(os.environ['WORLD_SIZE'])       # int(os.environ['WORLD_SIZE'])
        self.MASTER_ADDR = os.environ['MASTER_ADDR']          # os.environ['MASTER_ADDR']
        self.Master_PORT = os.environ['MASTER_PORT']          # os.environ['MASTER_PORT']
        print("  >>> 分布式训练 <<< MASTER_ADDR:{}, Master_PORT:{} ,world_size:{}, rank:{}, local_rank:{}"
              .format(self.MASTER_ADDR, self.Master_PORT, self.world_size, self.rand, self.local_rank))
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
    def __init__(self, config: KKM_Config, in_data=[], x_len: int = -1):
        super(Kk_Dataset, self).__init__()
        self.x_len = x_len
        self.data = in_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def dataFn(self, idata):
        return idata[..., 0:self.x_len], idata[..., self.x_len:]


class Kk_train(object):
    def __init__(self, config: KKM_Config,
                 model: Kk_Finetune, dataset: Kk_Dataset, dataset_val: Kk_Dataset,
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
        all_loss = 0
        for ibatch, idata in enumerate(self.dataLoader_val):
            loss = self._batch_val(0, ibatch, idata)
            all_loss += loss
        if ibatch % self.config.report_steps == 0:
            log.info("  >> 验证, batch {}/{}, loss:{}".format(ibatch, len(self.dataLoader_val),
                                                                 loss))
        pass

    def _batch_val(self, iepoch, ibatch, idata):
        x, y = self.dataset_val.dataFn(idata)
        if "cuda" == self.config.device:
            x = x.to(self.config.device)
        o = self.model(x)
        loss = self.lossFn(o, y)
        # loss.backward()
        return loss.item()

    def _batch(self, iepoch, ibatch, idata):
        x, y = self.dataset.dataFn(idata)
        if "cuda" == self.config.device:
            x = x.to(self.config.device)
        o = self.model(x)
        loss = self.lossFn(o, y)
        loss.backward()
        return loss.item()

    def _epoch(self, iepoch: int):
        self.optim.zero_grad()
        need_optim = False
        for ibatch, idata in enumerate(self.dataLoader):

            loss = self._batch(iepoch, ibatch, idata)
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

    def train(self):
        if self.config.gpu_count > 1:
            # 单机多卡 处理
            dist.init_process_group(backend=self.config.backend)
            torch.cuda.set_device(self.config.local_rank)
            self.model = self.model_src.to(self.config.local_rank)  # 先将模放到GPU
            self.model = torch.nn.parallel.DistributedDataParallel(self.model_src, device_ids=[self.config.local_rank],
                                                                   output_device=self.config.local_rank)
            # , init_method="env://",  # init_method="store" 手工
            # world_size=self.config.world_size, rank=self.config.local_rank)
            self.sampler = dist_data.DistributedSampler(self.dataset, rank=self.config.local_rank,
                                                        num_replicas=self.config.world_size)
            self.dataLoader = data.DataLoader(self.dataset, batch_size=self.config.batch_size,
                                              shuffle=False,
                                              sampler=self.sampler, num_workers=self.config.num_workers,
                                              pin_memory=self.config.pin_memory)
            self.sampler_val = dist_data.DistributedSampler(self.dataset_val, rank=self.config.rank,
                                                            num_replicas=self.config.world_size)
            self.dataLoader_val = data.DataLoader(self.dataset_val, batch_size=self.config.batch_size,
                                                  shuffle=False,
                                                  sampler=self.sampler_val, num_workers=self.config.num_workers,
                                                  pin_memory=self.config.pin_memory)

        elif self.config.device == "cuda":
            # 单机单卡 处理
            torch.cuda.set_device(0)
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
            self.model = self.model_src     #.to(self.config.device)
        for iepoch in tqdm.tqdm(range(self.config.epoch), desc="Epoch"):
            self._epoch(iepoch)
            # 进行验证
            self._val()


        if self.config.gpu_count > 1:
            # 分布式处理
            dist.destroy_process_group()
            torch.cuda.empty_cache()
        elif self.config.device == "cuda":
            torch.cuda.empty_cache()


class DatasetTest(Kk_Dataset):
    def dataFn(self, idata):
        return data[..., 0:88], data[..., 88:]

class ModuleTest(Kk_Module):
    def __init__(self, config: KKM_Config):
        super(ModuleTest, self).__init__(config)
        self.net = nn.Sequential(
            nn.Linear(88, 880),
            nn.BatchNorm1d(880),
            nn.Linear(880, 88),
            nn.BatchNorm1d(88),
            nn.Linear(88, 880),
            nn.BatchNorm1d(880),
            nn.Linear(880, 88),
            nn.Linear(88, 1))

    def forward(self, x):
        o = self.net(x)
        return o


if __name__ == "__main__":
    config = KKM_Config()
    datas = torch.randn(1000, 89)
    dataset = DatasetTest(config, datas)
    datas_val = torch.randn(80, 89)
    dataset_val = DatasetTest(config, datas_val)
    model = ModuleTest(config)
    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(ModuleTest(config).parameters(), lr=0.001)
    trainer = Kk_train(config, model, dataset, dataset_val, loss_fn, optim)
    trainer()

