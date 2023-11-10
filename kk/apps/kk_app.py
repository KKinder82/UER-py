import os
import pathlib as path
import sys

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
import random
import time
import kk.uer.layers.kk_Normalization as kkn
import kk.uer.kk_config as kkc
import kk.uer.kk_base as kkb
import kk.uer.layers.kk_Linear as kkl


class KkSigmoid(kkb.KkModule):
    def __init__(self, config: kkc.KkmConfig, factor: float = 1.0):
        super(KkLoss, self).__init__(config)
        # self.factor = self.register_buffer("kk_factor", torch.tensor(factor))
        self.factor = self.register_parameter("kk_factor", torch.tensor(factor))

    def forward(self, x):
        o = 1 / ((0 - x / self.factor).exp() + 1)
        return 0


class KkLoss(kkb.KkModule):
    def __init__(self, config: kkc.KkmConfig):
        super(KkLoss, self).__init__(config)

    def forward(self):
        raise NotImplemented(" KkLoss.forward() 未实现。")
        # 返回值 非 Tuple 时： loss
        # 返回值 Tuple 时：    loss, 0, right_perc
        # 返回值 Tuple 时：    loss, 1, right_perc, right_count, all_count


class KkClassfierLoss(KkLoss):
    def __init__(self, config: kkc.KkmConfig, *, blocks=[], counts=1, lossFn=nn.BCELoss(), diffOnly=False):
        super(KkClassfierLoss, self).__init__(config)
        if isinstance(counts, (int,)):
            self.blocks = []
            self.counts = [counts]
        else:
            self.blocks = blocks
            self.counts = counts
        self.lossFn = lossFn
        self.diffOnly = diffOnly

    def forward(self, o, y):  # 分类损失
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
        if self.diffOnly:
            loss = self.lossFn(o[_f], y[_f])
        else:
            loss = self.lossFn(o, y)
        all_count = y.count_nonzero()
        wrong_count2 = _f.count_nonzero()
        prec = 1 - (wrong_count2 / (all_count * 2))
        return loss, 0, prec.item()


class KkExtendLoss(KkLoss):   # 回归损失
    def __init__(self, config: kkc.KkmConfig, *, lossFn):
        super(KkExtendLoss, self).__init__(config)
        self.lossFn = lossFn

    def forward(self, o, y):
        loss = self.lossFn(o, y)
        prec = ((o - y)/ (y + 1e5)).abs().mean()
        return loss, 0, prec.item()


class KkApp(object):
    def __init__(self, config: kkc.KkmConfig, model):
        super(KkApp, self).__init__()
        self.config = config
        self.model_src = model
        self.last_loss = None
        if self.config.pt_load:
            if not os.path.exists(self.config.pt):
                print("  >> 找不到 KkmConfig中，pt 参数文件{}。".format(self.config.pt));
            else:
                print("  >> 加载模型参数文件：{} ... ".format(self.config.pt));
                _save = torch.load(self.config.pt)
                if isinstance(_save, tuple):
                    if _save[0] == "model":
                        self.model = _save[2]
                        self.last_loss = _save[1]
                    elif _save[0] == "dict":
                        self.model = self.model_src
                        self.model.load_state_dict(_save[2])
                        self.last_loss = _save[1]
                    else:
                        raise Exception("  >> KkmConfig中，pt 参数文件错误。")
                else:
                    self.model = _save
        else:
            self.model = self.model_src

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
    def __init__(self, config: kkc.KkmConfig, model):
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
    def __init__(self, config: kkc.KkmConfig, data_source):
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
    def __init__(self, config: kkc.KkmConfig, data=[], path_np: str = None, x_len: int = -1):
        super(KkDataset, self).__init__()
        self.config = config
        if path_np is not None:
            _data = np.load(path_np)
            self.data = torch.from_numpy(_data)
        else:
            if isinstance(data, np.ndarray):
                self.data = torch.from_numpy(data)
            elif isinstance(data, torch.Tensor):
                self.data = data
            else:
                self.data = torch.as_tensor(data)
        self.data = self.data.float()
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
    def __init__(self, config: kkc.KkmConfig, *,
                 model: KkInference, dataset: KkDataset, dataset_val: KkDataset,
                 loss_fn, optim=None):
        super(KkTrain, self).__init__(config, model)
        self.loss_value = 0.0
        self.config = config
        self.gpu_count = config.gpu_count
        self.dataset = dataset
        self.sampler = None
        self.dataLoader = None
        self.dataset_val = dataset_val
        self.sampler_val = None
        self.dataLoader_val = None
        self.loss_fn = loss_fn
        if optim is None:
            self.optim = torch.optim.Adam(model.parameters(), lr=0.001)
        else:
            self.optim = optim

    def _val(self, iepoch: int):
        all_loss = 0
        all_prec = 0
        for ibatch, idata in enumerate(self.dataLoader_val):
            loss, _prec = self._batch_val(0, ibatch, idata)
            # print("  >> val{} : loss {}".format(ibatch + 1, loss))
            if ibatch == 0:
                all_loss = loss.item()
                all_prec = _prec
            else:
                # all_loss = all_loss * iepoch / (iepoch + 1) + loss / (iepoch + 1)
                all_loss = all_loss + (loss.item() - all_loss) / (ibatch + 1)
                all_prec = all_prec + (_prec - all_prec) / (ibatch + 1)
        print("\n  >> 验证: RANK:{}, epoch {}/{}, loss:{}, prec:{}".format(self.config.rank,
                                                                           iepoch + 1, self.config.epoch,
                                                                           all_loss, all_prec))
        return all_loss

    def _batch_val(self, iepoch, ibatch, idata):
        with torch.no_grad():
            if "cuda" == self.config.device:
                idata = idata.to(self.config.device)
            x, y = self.dataset_val.dataFn(idata)
            o = self.model(x)
            loss, _prec = self._loss(o, y)
            # loss.backward()
            return loss, _prec

    def _loss(self, o, y):
        _loss_value = 0
        loss = self.loss_fn(o, y)
        if isinstance(loss, tuple):
            if isinstance(self.loss_fn, KkLoss):
                if loss[1] == 0:
                    _prec = loss[2]
                    loss = loss[0]
                else:
                    raise RuntimeError(" KkLoss 返回值错误。")
        else:
            _prec = loss.item()
        return loss, _prec

    def _batch(self, iepoch, ibatch, idata):
        if "cuda" == self.config.device:
            idata = idata.to(self.config.local_rank)
        x, y = self.dataset.dataFn(idata)
        if hasattr(self.model, "before_forward"):
            self.model.before_forward(x=x, y=y)
        o = self.model(x)
        loss, _prec = self._loss(o, y)
        if hasattr(self.model, "after_loss"):
            self.model.after_loss(x=x, y=y, loss=loss)
        loss.backward()
        return loss, _prec

    def _save(self, ibatch, loss, focus: bool = False):
        if self.config.rank != 0:
            return
        _model = self.model
        if isinstance(self.model, DDP):
            _model = _model.module
        if focus:
            if self.config.checkpoint_mode is None:
                pass
            elif self.config.checkpoint_mode == "model":
                _save_obj = (self.config.checkpoint_mode, loss, _model)
                torch.save(_save_obj, self.config.checkpoint_last)
                if self.last_loss is None or self.last_loss > loss:
                    self.last_loss = loss
                    torch.save(_save_obj, self.config.checkpoint_best)
            elif self.config.checkpoint_mode == "dict":
                _save_obj = (self.config.checkpoint_mode, loss, _model.state_dict())
                torch.save(_save_obj, self.config.checkpoint_last)
                if self.last_loss is None or self.last_loss > loss:
                    self.last_loss = loss
                    torch.save(_save_obj, self.config.checkpoint_best)
            else:
                print(" KkmConfig 配置文件中的 checkpoint_mode 配置错误。")
        else:
            if self.config.checkpoint_mode is None:
                pass
            elif self.config.checkpoint_mode == "model":
                if ibatch % self.config.save_checkpoint_steps == 0:
                    _save_obj = (self.config.checkpoint_mode, loss, _model)
                    torch.save(_save_obj, self.config.checkpoint_last)
                if self.last_loss is None or self.last_loss > loss:
                    self.last_loss = loss
                    _save_obj = (self.config.checkpoint_mode, loss, _model)
                    torch.save(_save_obj, self.config.checkpoint_best)
            elif self.config.checkpoint_mode == "dict":
                if ibatch % self.config.save_checkpoint_steps == 0:
                    _save_obj = (self.config.checkpoint_mode, loss, _model.state_dict())
                    torch.save(_save_obj, self.config.checkpoint_last)

                if self.last_loss is None or self.last_loss > loss:
                    self.last_loss = loss
                    _save_obj = (self.config.checkpoint_mode, loss, _model.state_dict())
                    torch.save(_save_obj, self.config.checkpoint_best)
            else:
                print(" KkmConfig 配置文件中的 checkpoint_mode 配置错误。")
        _save_obj = None

    def _epoch(self, iepoch: int):
        self.optim.zero_grad()
        if hasattr(self.model_src, "reset_epoch"):
            self.model_src.reset_epoch(iepoch=iepoch)
        if self.config.use_layer_optim and self.config.use_layer_optim_by_batch:
            for ibatch, idata in enumerate(self.dataLoader):
                self.config.auto_ibatch = ibatch
                self._layer_optim(self.model, batch_sign=True)
                loss = self._batch(iepoch, ibatch, idata)
                self.optim.step()
                self.optim.zero_grad()
                self._save(ibatch, loss, focus=False)
        else:
            need_optim = False
            for ibatch, idata in enumerate(self.dataLoader):
                # 按层优化
                self.config.auto_ibatch = ibatch
                loss, prec = self._batch(iepoch, ibatch, idata)
                # print("  >> Epoch {}/{}, batch {}/{}, rank:{}, loss:{}".format(iepoch, self.config.epoch,
                #                                              ibatch, len(self.dataLoader), self.config.rank, loss))
                need_optim = True
                if ibatch % self.config.accumulation_steps == 0:
                    self.optim.step()
                    self.optim.zero_grad()
                    need_optim = False
                    self._save(ibatch, loss, focus=False)
            if need_optim:
                self.optim.step()
                self.optim.zero_grad()
                need_optim = False
                self._save(ibatch, loss, focus=False)


    def _layer_optim_init(self, model):
        cfg = self.config
        m_cfg = {}
        if model in cfg.auto_layer_optim_models:
            m_cfg = cfg.auto_layer_optim_models.get(model)
        else:
            cfg.auto_layer_optim_models[model] = m_cfg
        m_cfg["parameters_layers"] = 0
        m_cfg["layer_optim_loops"] = 0
        m_cfg["layer_optim_times"] = 0
        m_cfg["layer_optim_index"] = -1
        m_cfg["layer_optim_finished"] = False

        _layers = model.modules()
        _i = 0
        for _ilayer in _layers:
            _parameters = list(_ilayer.parameters(recurse=False))
            if len(_parameters) > 0:
                _i += 1
        m_cfg["parameters_layers"] = _i

    def _layer_optim_finished(self, model):
        cfg = self.config
        m_cfg = cfg.auto_layer_optim_models.get(model)
        m_cfg["layer_optim_finished"] = True
        _stoped = []
        if cfg.use_layer_optim_finished_rand > 0:
            _stoped = [random.randint(0, m_cfg["parameters_layers"] - 1)\
                       for _ in range(cfg.use_layer_optim_finished_rand)]
        _layers = model.modules()
        _i = -1
        for _ilayer in _layers:
            _parameters = list(_ilayer.parameters(recurse=False))
            if len(_parameters) > 0:
                _i += 1
                _sign = _i not in _stoped
                for _iparameter in _parameters:
                    _iparameter.requires_grad = _sign

        # _parameters = model.parameters()
        # for _iparam in _parameters:
        #     _iparam.requires_grad = True


    def _layer_optim(self, model, batch_sign: bool = False):
        cfg = self.config
        m_cfg = cfg.auto_layer_optim_models.get(model)
        if ((not cfg.use_layer_optim)
                or (batch_sign != cfg.use_layer_optim_by_batch)):
            return
        if m_cfg["layer_optim_finished"]:
            self._layer_optim_finished(model)
            return

        _need_optim = False
        if cfg.use_layer_optim_by_batch:
            _all_batch = cfg.auto_iepoch * cfg.batch_count + cfg.auto_ibatch
            if _all_batch % cfg.use_layer_optim_by_count == 0:
                _need_optim = True
        else:
            # 一个 epoch 的开始
            if cfg.auto_iepoch % cfg.use_layer_optim_by_count == 0:
                _need_optim = True
        if _need_optim:
            # 具体优化
            m_cfg["layer_optim_times"] += 1
            if m_cfg["layer_optim_times"] > cfg.use_layer_optim_loops:
                # 已经达到最大次数
                self._layer_optim_finished(model)
                return
            else:
                m_cfg["layer_optim_index"] = (m_cfg["layer_optim_index"] + 1) % m_cfg["parameters_layers"]
                if cfg.use_layer_optim_param_group_size < 1:
                    layer_optim_param_group_size = random.randint(0, m_cfg["parameters_layers"]-1)
                else:
                    layer_optim_param_group_size = cfg.use_layer_optim_param_group_size
                if cfg.use_layer_optim_random:
                    _min = random.randint(0, m_cfg["parameters_layers"]-1)
                    _max = _min + layer_optim_param_group_size
                else:
                    _min = cfg.use_layer_optim_param_group_size * m_cfg["layer_optim_index"]
                    _min = _min % m_cfg["parameters_layers"]
                    _max = _min + layer_optim_param_group_size
                    if cfg.use_layer_optim_from_zero:
                        if _max >= m_cfg["parameters_layers"]:
                            self._layer_optim_finished(model)
                            return
                        _min = 0

                _max = min(_max, m_cfg["parameters_layers"])
                _trained = [i for i in range(_min, _max, 1)]
                if len(_trained) < 1:
                    print("  >> 优化层：未找到优化层。")
                    _trained.append(0)
                # print("  >> 优化层：" + str(_min) + "-" + str(_max-1))
                _layers = model.modules()
                _i = -1
                for _ilayer in _layers:
                    _parameters = list(_ilayer.parameters(recurse=False))
                    if len(_parameters) > 0:
                        _i += 1
                        _sign = _i in _trained
                        for _iparameter in _parameters:
                            _iparameter.requires_grad = _sign

    def __call__(self, *args, **kwargs):
        self.train()

    def train(self):
        self.device_init()
        self.config.auto_init()

        need_dataloader_init = True
        if self.config.device == "cuda":
            if self.config.gpu_count > 1:
                # 单机多卡、多机多卡
                if isinstance(self.loss_fn, nn.Module):
                    self.loss_fn.to(self.config.local_rank)
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
                if isinstance(self.loss_fn, nn.Module):
                    self.loss_fn.to(self.config.local_rank)
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

        # 开始显示
        if self.config.rank == 0:
            print("  >> 开始训练，epoch:{}, batch_size:{}, world_size:{}, gpu_count:{}, loss:{}"
                  .format(self.config.epoch, self.config.batch_size,
                          self.config.world_size, self.config.gpu_count, self.last_loss))
        self._layer_optim_init(self.model)
        for iepoch in tqdm.tqdm(range(self.config.epoch), desc=f" Epoch"):
            # 训练
            self.config.auto_iepoch = iepoch
            self.config.auto_ibatch = -1
            self.config.auto_training = True
            self.model.train()
            if isinstance(self.sampler, dist_data.DistributedSampler):
                self.sampler.set_epoch(iepoch)
            self._layer_optim(self.model, batch_sign=False)
            self._epoch(iepoch)

            # 进行验证
            self.config.auto_training = False
            self.model.eval()
            if isinstance(self.sampler_val, dist_data.DistributedSampler):
                self.sampler_val.set_epoch(iepoch)
            loss_value = self._val(iepoch)
            if loss_value < self.config.stop_train_loss:
                print("  >> 当前损失已满足系统要求，训练结束。")
                break
        self.device_uninit()


class DemoModel(kkb.KkModule):
    def __init__(self, config: kkc.KkmConfig):
        super(DemoModel, self).__init__(config)
        self.backbone = nn.Sequential(
            kkl.KkLinear(config, 88, 880, normalization="batch"),
            kkl.KkLinear(config, 880, 88),
            kkl.KkLinear(config, 88, 1))

    def forward(self, x):
        o = self.backbone(x)
        return o


def torchrun():
    # 运行指令 torchrun --nperc-per-node 1 .\kk_app.py
    config = kkc.KkmConfig(__file__)
    datas = torch.randn(1000, 89)
    # print(datas[:, 0:88].sum(dim=1) / 3.1415926)
    datas[:, 88] = datas[:, 0:88].sum(dim=1) / 3.1415926
    dataset = KkDataset(config, datas)
    datas_val = torch.randn(80, 89)
    datas_val[:, 88] = datas_val[:, 0:88].sum(dim=1) / 3.1415926
    dataset_val = KkDataset(config, datas_val)
    model = DemoModel(config)
    loss_fn = KkExtendLoss(config, lossFn=nn.MSELoss())
    optim = torch.optim.Adam(DemoModel(config).parameters(), lr=0.001)
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

class TestModel(kkb.KkModule):
    def __init__(self, config: kkc.KkmConfig):
        super(TestModel, self).__init__(config)
        self.Linear = kkl.KkLinear(config, 2, 1)

    def forward(self, x):
        o = self.Linear(x)
        return o


def test():
    os.environ['RANK'] = "0"
    os.environ['LOCAL_RANK'] = "0"
    os.environ['WORLD_SIZE'] = "1"
    os.environ['MASTER_ADDR'] = "127.0.0.1"
    os.environ['MASTER_PORT'] = "16666"

    config = kkc.KkmConfig(__file__)
    datas = torch.randn(1000, 3)
    # print(datas[:, 0:88].sum(dim=1) / 3.1415926)
    datas[:, 2] = datas[:, 0:2].sum(dim=-1) / 3.1415926
    dataset = KkDataset(config, datas)
    datas_val = torch.randn(80, 3)
    datas_val[:, 2] = datas_val[:, 0:2].sum(dim=-1) / 3.1415926
    dataset_val = KkDataset(config, datas_val)
    model = TestModel(config)
    loss_fn = KkExtendLoss(config, lossFn=nn.MSELoss())
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = KkTrain(config, model=model, dataset=dataset, dataset_val=dataset_val,
                      loss_fn=loss_fn, optim=optim)
    trainer.train()

if __name__ == "__main__":
    test()
    # python()
    # python_spawn()
    # torchrun()


    # config = kkc.KkmConfig(__file__)
    # loss = KkClassfierLoss(config, counts=1, lossFn=nn.BCELoss())
    # o = torch.Tensor([[0.4, 0.5, 0.1], [0.1, 0.1, 0.1]]).to(torch.float32)
    # y = torch.Tensor([[0, 1, 0], [0, 1, 0]]).to(torch.float32)
    # l = nn.BCELoss()(o, y)
    # print(l)
    #
    # l = loss(o, y)
    # print(l)
