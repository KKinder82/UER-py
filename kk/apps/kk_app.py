import os
import pathlib as path
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F

import torch.distributed as dist
import torch.utils.data.distributed as dist_data
import torch.nn.parallel.distributed as dist_nn
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
import math
import logging as log
import tqdm
import kk.kk_utils as kku


def env_int(name: str, default=0):
    if name in os.environ:
        return int(os.environ[name])
    return default


def env_value(name: str, default: str = ""):
    if name in os.environ:
        return os.environ[name]
    return default


class KkmConfig(object):
    def __init__(self, *args, **kwargs): # args:可变数量参数； kwargs：关键字参数（可变数量）
        # 日志配置
        # %(name)s        logger 名称, 即调用logging.getLogger函数传入的参数
        # %(levelno)s     数字形式的日志记录级别
        # %(levelname)s   日志级别文本描述, 即"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
        # %(pathname)s    执行日志记录调用的源文件路径
        # %(filename)s    执行日志记录调用的源文件名
        # %(module)s      执行日志记录调用的模块名称
        # %(lineno)d      执行日志记录调用的行号
        # %(funcName)s    执行日志记录调用的函数路径
        # %(created)d     执行日志记录调用的时间, 它是time.time()返回的数字
        # %(asctime)s     执行日志记录调用的ascii格式的时间, 其格式由datefmt指定
        # %(msecs)d       执行日志记录调用的时间中的毫秒部分
        # %(thread)d      线程id( if available)
        # %(threadName)s  线程名称( if available)
        # %(process)d     进程ID( if available)
        # %(message)s     记录的消息, 如logging.getLogger().debug(msg)指定的msg

        log.basicConfig(level=log.INFO, format="%(created)f %(asctime)s %(levelname)s %(message)s \r\n", datefmt="%Y-%m-%d %H:%M:%S")
        # 应用配置
        self.app_name = "DLApp"
        # 分布式训练
        self.rank = env_int('RANK', 0)                   # int(os.environ['RANK'])
        self.local_rank = env_int('LOCAL_RANK', 0)       # int(os.environ['LOCAL_RANK'])
        self.world_size = env_int('WORLD_SIZE', 1)       # int(os.environ['WORLD_SIZE'])
        self.master_addr = env_value('MASTER_ADDR', "127.0.0.1")  # os.environ['MASTER_ADDR']
        self.master_port = env_value('MASTER_PORT', "16666")        # os.environ['MASTER_PORT']
        print("  >>> 分布式训练 <<< MASTER_ADDR:{}, Master_PORT:{} ,world_size:{}, rank:{}, local_rank:{}"
              .format(self.master_addr, self.master_port, self.world_size, self.rank, self.local_rank))
        self.backend = "nccl"
        # GPU 训练
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_count = min(torch.cuda.device_count(), self.world_size)
        # 数据集
        self.shuffle = True
        self.batch_size = 10
        self.num_workers = self.world_size
        self.pin_memory = True
        self.batch_ceil = False      # 当数据不能整除时，是否补齐
        # 模型加载
        self.ptload_mode = "model"   # 文件类型 None: 不加载, dict: 参数文件, model : 模型与参数
        self.pt = "model_best.pth"
        # 训练
        self.epoch = 100
        self.accumulation_steps = 5
        self.save_checkpoint_steps = 20
        self.report_steps = 1
        self.checkpoint_mode = "model"           # None: 不保存, dict: 参数文件, model : 模型与参数
        self.checkpoint_last = "model_last.pth"
        self.checkpoint_best = "model_best.pth"


class KkModule(nn.Module):
    def __init__(self, config: KkmConfig):
        super(KkModule, self).__init__()
        self.config = config

    def init_weights(self):
        pass

class KkSigmoid(KkModule):
    def __init__(self, config: KkmConfig, factor: float = 1.0):
        super(KkLoss, self).__init__(config)
        # self.factor = self.register_buffer("kk_factor", torch.tensor(factor))
        self.factor = self.register_parameter("kk_factor", torch.tensor(factor))

    def forward(self, x):
        o = 1 / ((0 - x / self.factor).exp() + 1)
        return 0


class KkLoss(KkModule):
    def __init__(self, config: KkmConfig):
        super(KkLoss, self).__init__(config)

    def forward(self):
        raise NotImplemented(" KkLoss.forward() 未实现。")
        # 返回值 非 Tuple 时： loss
        # 返回值 Tuple 时：    loss, 0, right_perc
        # 返回值 Tuple 时：    loss, 1, right_perc, right_count, all_count

class KkWrongLoss(KkLoss):
    def __init__(self, config: KkmConfig, *, blocks=[], counts=1):
        super(KkLoss, self).__init__(config)
        if isinstance(counts, (int,)):
            self.blocks = []
            self.counts = [counts]
        else:
            self.blocks = blocks
            self.counts = counts

    def forward(self, o, y):
        o_y = torch.zeros_like(o)
        _last = 0
        for i in range(len(self.counts) - 1):
            _max, _index = torch.topk(o[..., _last:self.blocks[i]], self.counts[i], dim=-1)
            kku.tensor_fill(o_y[..., _last:self.blocks[i]], _index, 1)
            _last += self.blocks[i]
        if _last < y.size(-1):
            _max, _index = torch.topk(o[..., _last:], self.counts[-1], dim=-1)
            kku.tensor_fill(o_y[..., _last:], _index, 1)
        _f = (o_y - y) != 0
        loss = torch.sum(torch.pow(o[_f] - y[_f], 2))
        all_count = y.count_nonzero()
        wrong_count2 = _f.count_nonzero()
        wrong_perc = wrong_count2 / (all_count * 2)
        return loss, 0, wrong_perc



class KkApp(object):
    def __init__(self, config: KkmConfig, model):
        super(KkApp, self).__init__()
        self.config = config
        self.model_src = model
        if self.config.ptload_mode is not None:
            self.model = self.model_src
        elif self.config.ptload_mode == "dict":
            if os.path.exists(self.config.pt):
                print("  >> 正在加载模型参数文件：{} ... ".format(self.config.pt))
                self.model_src.load_state_dict(torch.load(self.config.pt))
                self.model = self.model_src
            else:
                raise FileNotFoundError(" KkmConfig配置文件中配置的模型文件不存在。")
        elif self.config.ptload_mode == "model":
            if os.path.exists(self.config.pt):
                print("  >> 正在加载模型：{} ... ".format(self.config.pt))
                self.model = torch.load(self.config.pt)
            else:
                raise FileNotFoundError(" KkmConfig配置文件中配置的模型文件不存在。")
        else:
            raise Exception(" KkmConfig 配置文件中的 ptload_mode 配置错误。")

    def device_init(self):
        if self.config.device == "cuda":
            if self.config.gpu_count > 1:
                # 单机多卡 处理
                dist.init_process_group(backend=self.config.backend, init_method="env://",
                                        world_size=self.config.world_size, rank=self.config.rank)
                torch.cuda.set_device(self.config.local_rank)
                self.model = self.model_src.to(self.config.local_rank)  # 先将模放到GPU
                self.model = DDP(self.model_src, device_ids=[self.config.local_rank],
                                 output_device=self.config.local_rank)
            else:
                # 单机单卡 处理
                torch.cuda.set_device(self.config.local_rank)
                self.model = self.model_src.to(self.config.local_rank)
        else:
            # cpu
            self.model = self.model_src  # .to(self.config.device)
            # self.lossFn.to(self.config.local_rank)

    def device_uninit(self):
        if self.config.device == "cuda":
            if self.config.gpu_count > 1:
                # 分布式处理
                dist.destroy_process_group()
                torch.cuda.empty_cache()
            elif self.config.device == "cuda":
                torch.cuda.empty_cache()


class KkInference(KkApp):
    def __init__(self, config: KkmConfig, model):
        super(KkInference, self).__init__(config, model)

    def forward(self, datas):
        # 模型处理
        self.device_init()
        with torch.no_grad():
            datas = datas.view(-1, datas.size(-1))
            batch = math.ceil(datas.size(0) / self.config.batch_size)
            out = []
            for ibatch in range(batch):
                x = datas[ibatch * self.config.batch_size:(ibatch + 1) * self.config.batch_size, :]
                if self.config.device == "cuda":
                    x = x.to(self.config.device)
                    o = self.model(x)
                    o = o.cpu()
                else:
                    o = self.model(x)
                out.append(o)
            print(out)
        self.device_uninit()


class KkSampler(data.Sampler):
    def __init__(self, config: KkmConfig, data_source):
        self.config = config
        super(KkSampler, self).__init__()
        self.data_source = data_source
        self.batch_size = self.config.batch_size
        self.shuffle = self.config.shuffle
        # 处理
        self.data_count = len(self.data_source)
        self.indexes = np.arange(self.data_count)
        if self.config.batch_ceil:
            _diff = self.batch_size - self.data_count % self.batch_size
            _v = np.random.randint(0, self.data_count, _diff)
            self.indexes = np.concatenate((self.indexes, self.indexes[_v]))
        else:
            _diff = self.data_count % self.batch_size
            _v = np.random.randint(0, self.data_count, _diff)
            self.indexes = np.delete(self.indexes, _v)
        self.batch_count = len(self.indexes) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
        for i in range(len(self.indexes)):
            yield self.indexes[i]

    def __len__(self):
        return len(self.indexes)

class KkDataset(data.Dataset):
    def __init__(self, config: KkmConfig, data=[], path_np: str = None, x_len: int = -1):
        super(KkDataset, self).__init__()
        self.config = config
        if path_np is not None:
            _data = np.load(path_np)
            self.data = torch.tensor(_data, dtype=torch.float32)
        else:
            self.data = torch.tensor(data, dtype=torch.float32)
        self.x_len = x_len

    def __getitem__(self, index):
        return self.data[index, ...]

    def __len__(self):
        return len(self.data)

    def dataFn(self, idata):
        if self.x_len >= idata.size(-1):
            y = torch.zeros(idata.shape[:-1, 1], dtype=idata.dtype)
            return idata, y
        return idata[..., 0:self.x_len], idata[..., self.x_len:]


class KkTrain(KkApp):
    def __init__(self, config: KkmConfig,
                 model: KkInference, dataset: KkDataset, dataset_val: KkDataset,
                 lossFn, optim=None):
        super(KkTrain, self).__init__(config, model)
        self.loss_value = 0.0
        self.last_loss = None
        self.config = config
        self.gpu_count = config.gpu_count
        self.dataset = dataset
        self.sampler = None
        self.dataLoader = None
        self.dataset_val = dataset_val
        self.sampler_val = None
        self.dataLoader_val = None
        self.lossFn = lossFn
        if optim is None:
            self.optim = torch.optim.Adam(model.parameters(), lr=0.001)
        else:
            self.optim = optim

    def _val(self, iepoch: int):
        all_loss = 0
        for ibatch, idata in enumerate(self.dataLoader_val):
            loss = self._batch_val(0, ibatch, idata)
            if iepoch == 0:
                all_loss = loss
            else:
                # all_loss = all_loss * iepoch / (iepoch + 1) + loss / (iepoch + 1)
                all_loss = all_loss + (loss - all_loss) / (iepoch + 1)
        print("\n  >> 验证, epoch {}/{}, loss:{}".format(iepoch + 1, self.config.epoch, all_loss))
        pass

    def _batch_val(self, iepoch, ibatch, idata):
        with torch.no_grad():
            if "cuda" == self.config.device:
                idata = idata.to(self.config.device)
            x, y = self.dataset_val.dataFn(idata)
            o = self.model(x)
            loss, loss_value = self._loss(o, y)
            # loss.backward()
            return loss_value

    def _loss(self, o, y):
        _loss_value = 0
        loss = self.lossFn(o, y)
        if isinstance(loss, tuple):
            if isinstance(self.lossFn, KkLoss):
                _loss_mode = loss[1]
                if _loss_mode == 0:
                    _loss_value = loss[2]
                    loss = loss[0]
                else:
                    _loss_value = loss[2]
        else:
            _loss_value = loss.item()
        return loss, _loss_value

    def _batch(self, iepoch, ibatch, idata):
        if "cuda" == self.config.device:
            idata = idata.to(self.config.local_rank)
        x, y = self.dataset.dataFn(idata)
        o = self.model(x)
        loss, loss_value = self._loss(o, y)
        loss.backward()
        return loss_value

    def _epoch(self, iepoch: int):
        self.optim.zero_grad()
        need_optim = False
        for ibatch, idata in enumerate(self.dataLoader):
            loss = self._batch(iepoch, ibatch, idata)
            need_optim = True
            # if ibatch % self.config.report_steps == 0:
            #     log.info("  >> Epoch {}/{}, batch {}/{}, loss:{}".format(iepoch, self.config.epoch,
            #                                                              ibatch, len(self.dataLoader),
            #                                                              loss))
            if ibatch % self.config.accumulation_steps == 0:
                self.optim.step()
                self.optim.zero_grad()
                need_optim = False
            if self.config.checkpoint_mode is not None:
                if self.config.checkpoint_mode == "model":
                    if ibatch % self.config.save_checkpoint_steps == 0:
                        torch.save(self.model, self.config.checkpoint_last)

                    if self.last_loss is None or self.last_loss > loss:
                        self.last_loss = loss
                        torch.save(self.model, self.config.checkpoint_best)
                elif self.config.checkpoint_mode == "dict":
                    if ibatch % self.config.save_checkpoint_steps == 0:
                        torch.save(self.model.state_dict(), self.config.checkpoint_last)

                    if self.last_loss is None or self.last_loss > loss:
                        self.last_loss = loss
                        torch.save(self.model.state_dict(), self.config.checkpoint_best)
                else:
                    print(" KkmConfig 配置文件中的 checkpoint_mode 配置错误。")
        if need_optim:
            self.optim.step()
            self.optim.zero_grad()
            need_optim = False

    def __call__(self, *args, **kwargs):
        self.train()

    def train(self):
        self.device_init()
        need_dataloader_init = True
        if self.config.device == "cuda":
            if self.config.gpu_count > 1:
                # 单机多卡、多机多卡
                if isinstance(self.lossFn, nn.Module):
                    self.lossFn.to(self.config.local_rank)
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
                need_dataloader_init = False

            else:
                # 单机单卡 处理
                if isinstance(self.lossFn, nn.Module):
                    self.lossFn.to(self.config.local_rank)
        else:
            # CPU
            # self.lossFn.to(self.config.local_rank)
            pass
        if need_dataloader_init:
            # self.sampler = data.SequentialSampler(self.dataset)
            self.sampler = KkSampler(self.config, self.dataset)
            self.dataLoader = data.DataLoader(self.dataset, batch_size=self.config.batch_size,
                                              shuffle=False,
                                              sampler=self.sampler, num_workers=self.config.num_workers,
                                              pin_memory=self.config.pin_memory)

            self.sampler_val = KkSampler(self.config, self.dataset_val)
            self.dataLoader_val = data.DataLoader(self.dataset_val, batch_size=self.config.batch_size,
                                                  shuffle=False,
                                                  sampler=self.sampler_val, num_workers=self.config.num_workers,
                                                  pin_memory=self.config.pin_memory)

        for iepoch in tqdm.tqdm(range(self.config.epoch), desc=f" Epoch"):
            self.model.train()
            self._epoch(iepoch)

            # 进行验证
            self.model.eval()
            self._val(iepoch)

        self.device_uninit()


class ModuleTest(KkModule):
    def __init__(self, config: KkmConfig):
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


def torchrun():
    # 运行指令 torchrun --nperc-per-node 1 .\kk_app.py
    config = KkmConfig()
    datas = torch.randn(1000, 89)
    # print(datas[:, 0:88].sum(dim=1) / 3.1415926)
    datas[:, 88] = datas[:, 0:88].sum(dim=1) / 3.1415926
    dataset = KkDataset(config, datas)
    datas_val = torch.randn(80, 89)
    datas_val[:, 88] = datas_val[:, 0:88].sum(dim=1) / 3.1415926
    dataset_val = KkDataset(config, datas_val)
    model = ModuleTest(config)
    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(ModuleTest(config).parameters(), lr=0.001)
    trainer = KkTrain(config, model, dataset, dataset_val, loss_fn, optim)
    trainer.train()


def python_spawn_fn(rank: int, world_size: int):
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = "127.0.0.1"
    os.environ['MASTER_PORT'] = "16666"
    torchrun()


def python_spawn():
    world_size = torch.cuda.device_count()
    # python_spawn_fn 的第1个参数是rank，由mp.spawn 函数内容自动传入，范围为 0-（world_size-1）
    mp.spawn(python_spawn_fn, args=(world_size,), nprocs=world_size)


def python():
    os.environ['RANK'] = "0"
    os.environ['LOCAL_RANK'] = "0"
    os.environ['WORLD_SIZE'] = "1"
    os.environ['MASTER_ADDR'] = "127.0.0.1"
    os.environ['MASTER_PORT'] = "16666"

    torchrun()


if __name__ == "__main__":
    python()
    # python_spawn()
    # torchrun()

