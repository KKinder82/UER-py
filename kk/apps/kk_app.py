import os

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.utils.data as data

import torch.distributed as dist
import torch.utils.data.distributed as dist_data

from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
import math

import tqdm
import kk.kk_utils as kku
import random
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
    def __init__(self, config: kkc.KkmConfig, *, lossFn=nn.BCELoss(), blocks=[], counts=1, diffOnly=False):
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


class KkExtendLoss(KkLoss):  # 回归损失
    def __init__(self, config: kkc.KkmConfig, *, lossFn):
        super(KkExtendLoss, self).__init__(config)
        self.lossFn = lossFn

    def forward(self, o, y):
        loss = self.lossFn(o, y)
        prec = ((o - y) / (y + 1e-7)).abs().mean()
        if prec > 1:
            prec = 0
        else:
            prec = 1 - prec
            prec = prec.item()
        return loss, 0, prec


class KkApp(object):
    config = None

    def __init__(self, config: kkc.KkmConfig, model):
        super(KkApp, self).__init__()
        KkApp.config = config
        self.model_src = model
        self.last_loss = (None, None)
        # 加载参数文件
        self._model_load()

    def _device_init(self):
        if self.config.device == "cuda":
            if self.config.world_size > 1:
                # 单机多卡 处理
                print("  >> KkApp._device_init << Pytorch:GPU 多机多卡初始化 ")
                print("  >>> 分布式训练参数 <<<  MASTER_ADDR:{}, Master_PORT:{} ,world_size:{}, rank:{}, local_rank:{}"
                      .format(self.config.master_addr, self.config.master_port,
                              self.config.world_size, self.config.rank, self.config.local_rank))
                dist.init_process_group(backend=self.config.backend, init_method="env://",
                                        world_size=self.config.world_size, rank=self.config.rank)
                torch.cuda.set_device(self.config.local_rank)
                self.model_src = self.model_src.to(self.config.local_rank)  # 先将模放到GPU
                self.model = DDP(self.model_src,
                                 device_ids=[self.config.local_rank])   # , find_unused_parameters=True)
            else:
                # 单机单卡 处理
                print("  >> KkApp._device_init << Pytorch:GPU 单机单卡初始化 ")
                torch.cuda.set_device(self.config.local_rank)
                self.model_src = self.model_src.to(self.config.device)
                self.model = self.model_src
        else:
            # cpu
            print("  >> KkApp._device_init << Pytorch:CPU 初始化 ")
            # self.model_src.to(self.config.device)
            self.model = self.model_src  # .to(self.config.device)

    def _device_uninit(self):
        if self.config.device == "cuda":
            if self.config.gpu_count > 1:
                # 分布式处理
                dist.destroy_process_group()
                torch.cuda.empty_cache()
            elif self.config.device == "cuda":
                torch.cuda.empty_cache()

    def _model_load(self):
        if self.config.pt_load:
            if not os.path.exists(self.config.pt):
                print("  >> KkmConfig << 找不到 KkmConfig中，pt 参数文件{}。".format(self.config.pt))
            else:
                print("  >> KkmConfig << 加载模型参数文件：{} ... ".format(self.config.pt))
                _load_obj = torch.load(self.config.pt)
                if isinstance(_load_obj, tuple):
                    if len(_load_obj) == 3:
                        print(_load_obj[1])
                        if _load_obj[0] == "model":
                            self.model_src = _load_obj[2]
                            self.last_loss = _load_obj[1]
                            return True
                        elif _load_obj[0] == "dict":
                            self.model_src.load_state_dict(_load_obj[2])
                            self.last_loss = _load_obj[1]
                            return True
                        else:
                            pass
                    else:
                        pass
                else:
                    pass
                raise Exception("  >> KkmConfig << KkmConfig.pt 参数文件格式错误。")

    def _model_save(self, *, ibatch, loss, is_force: bool = False):
        if self.config.rank != 0:
            # 只在 rank = 0 的进程保存模型参数
            return False

        loss_tuple = loss

        def _save(best_only: bool = False):
            _model = self.model_src
            if self.config.checkpoint_mode is None:
                return
            elif self.config.checkpoint_mode == "model":
                _save_obj = (self.config.checkpoint_mode, loss_tuple, _model)
                if not best_only:
                    torch.save(_save_obj, self.config.checkpoint_last)
                if self.last_loss[0] is None or self.last_loss[0] > loss_tuple[0]:
                    torch.save(_save_obj, self.config.checkpoint_best)
                _save_obj = None
            elif self.config.checkpoint_mode == "dict":
                _save_obj = (self.config.checkpoint_mode, loss_tuple, _model.state_dict())
                if not best_only:
                    torch.save(_save_obj, self.config.checkpoint_last)
                if self.last_loss[0] is None or self.last_loss[0] > loss_tuple[0]:
                    torch.save(_save_obj, self.config.checkpoint_best)
                _save_obj = None
            else:
                raise Exception(" KkmConfig 配置文件中的 checkpoint_mode 配置错误。")


        if is_force:
            _save()
        else:
            if ibatch % self.config.save_checkpoint_steps == 0:
                _save()
            else:
                _save(True)


class KkInference(KkApp):
    def __init__(self, config: kkc.KkmConfig, model):
        super(KkInference, self).__init__(config, model)

    def forward(self, datas):
        # 模型处理
        self._device_init()
        with torch.no_grad():
            datas = datas.view(-1, datas.size(-1))
            batch = math.ceil(datas.size(0) / self.config.batch_size)
            out = []
            for ibatch in range(batch):
                x = datas[ibatch * self.config.batch_size:(ibatch + 1) * self.config.batch_size, :]
                if self.config.device == "cuda":
                    o = self.model(x)
                    o = o.cpu()
                else:
                    o = self.model(x)
                out.append(o)
            print(out)
        self._device_uninit()


class KkSampler(data.Sampler):
    def __init__(self, config: kkc.KkmConfig, data_source):
        self.config = config
        super(KkSampler, self).__init__()
        # self.data_source = data_source
        self.batch_size = self.config.batch_size
        self.shuffle = self.config.shuffle
        # 处理
        self.data_count = len(data_source)
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
        self.data = self.data.to(torch.float32)
        self.x_len = x_len

    def __getitem__(self, index):
        _x = self.data[index]
        x = _x[..., 0:self.x_len]
        y = _x[..., self.x_len:]
        return x, y

    def __len__(self):
        return len(self.data)


class KkAppModel(kkb.KkModule):
    def __init__(self, config: kkc.KkmConfig):
        super(KkAppModel, self).__init__(config)

    def epoch_reset(self, **args):
        pass

    def before_forward(self, **args):
        # print("  >> KkDemoModel.before_forward <<  -------------------------------")
        # x = args["x"]
        # o = args["o"]
        pass

    def forward(self, x):
        raise NotImplemented(" KkAppModel.forward() 未实现。")

    def after_loss(self, **args):
        # x = args["x"]
        # o = args["o"]
        # y = args["y"]
        # loss = args["loss"]
        # print("  >> KkDemoModel.after_loss <<  -------------------------------")
        pass


class KkTrain(KkApp):
    def __init__(self, config: kkc.KkmConfig, *,
                 model: kkb.KkModule, dataset: KkDataset, dataset_val: KkDataset,
                 loss_fn, optim=None):
        super(KkTrain, self).__init__(config, model)
        kkb.KkModule.model = model
        self.loss_value = 0.0
        self.gpu_count = config.gpu_count
        self.dataset = dataset
        self.sampler = None
        self.dataLoader = None
        self.dataset_val = dataset_val
        self.sampler_val = None
        self.dataLoader_val = None
        self.loss_fn = loss_fn
        self.val_loss = None
        self.val_perc = None
        if optim is None:
            self.optim = torch.optim.Adam(model.parameters(), lr=0.001)
        else:
            self.optim = optim

    def _loss(self, o, y):
        _loss_value = 0
        loss = self.loss_fn(o, y)
        if isinstance(loss, tuple):
            if isinstance(self.loss_fn, KkLoss):
                if loss[1] == 0:
                    _perc = loss[2]
                    loss = loss[0]
                else:
                    raise RuntimeError(" KkLoss 返回值错误。")
        else:
            _perc = loss.item()
        return loss, _perc * 100

    def _optim(self, *, ibatch: int, loss):
        if self.config.sys_ibatch == 0 and self.config.sys_iepoch == 0:
            # 检查Weight权重是否存在全为0的情况，pytorch有时在启动时
            _params = self.model_src.named_parameters()
            for ipname, iparam in _params:
                if ipname.endswith("mean") or ipname.endswith("bias"):
                    continue
                if not torch.nonzero(iparam).size(0) > 0:
                    print("  >> KkTrain._optim <<  模型参数 {} 权重为 0 ，请确认该参数是否需要初始化。".format(ipname))
        self.optim.step()
        self.optim.zero_grad()
        self._model_save(ibatch=ibatch, loss=loss, is_force=False)
        self.last_loss = loss

    def _layer_optim_init(self, model):
        cfg = self.config
        m_cfg = {}
        if model in cfg.sys_layer_optim_models:
            m_cfg = cfg.sys_layer_optim_models.get(model)
        else:
            cfg.sys_layer_optim_models[model] = m_cfg
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
        m_cfg = cfg.sys_layer_optim_models.get(model)
        m_cfg["layer_optim_finished"] = True
        _stoped = []
        if cfg.use_layer_optim_finished_rand > 0:
            _stoped = [random.randint(0, m_cfg["parameters_layers"] - 1)
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

    def _layer_optim(self, *, batch_epoch: bool = True):
        cfg = self.config
        model = self.model
        m_cfg = cfg.sys_layer_optim_models.get(model)
        if (cfg.use_layer_optim
                and (batch_epoch == cfg.use_layer_optim_by_batch)):
            pass
        else:
            return
        if m_cfg["layer_optim_finished"]:
            self._layer_optim_finished(model)
            return

        _need_optim = False
        if cfg.use_layer_optim_by_batch:
            _all_batch = cfg.sys_iepoch * cfg.batch_count + cfg.sys_ibatch
            if _all_batch % cfg.use_layer_optim_by_count == 0:
                _need_optim = True
        else:
            # 一个 epoch 的开始
            if cfg.sys_iepoch % cfg.use_layer_optim_by_count == 0:
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
                    layer_optim_param_group_size = random.randint(0, m_cfg["parameters_layers"] - 1)
                else:
                    layer_optim_param_group_size = cfg.use_layer_optim_param_group_size
                if cfg.use_layer_optim_random:
                    _min = random.randint(0, m_cfg["parameters_layers"] - 1)
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

    def _data_init(self):
        need_dataloader_init = True
        if self.config.device == "cuda":
            if self.config.gpu_count > 1:
                # 单机多卡、多机多卡
                if isinstance(self.loss_fn, nn.Module):
                    self.loss_fn.to(self.config.local_rank)
                # , init_method="env://",  # init_method="store" 手工
                # world_size=self.config.world_size, rank=self.config.local_rank)
                self.sampler = dist_data.DistributedSampler(self.dataset, rank=self.config.local_rank,
                                                            shuffle=False)
                self.dataLoader = data.DataLoader(self.dataset, batch_size=self.config.batch_size,
                                                  shuffle=False,
                                                  sampler=self.sampler, num_workers=self.config.num_workers,
                                                  pin_memory=self.config.pin_memory)

                self.sampler_val = dist_data.DistributedSampler(self.dataset_val, rank=self.config.rank,
                                                                shuffle=False)
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
            # self.dataLoader = data.DataLoader(self.dataset, batch_size=self.config.batch_size,
            #                                   shuffle=False, num_workers=0)
            self.dataLoader = data.DataLoader(self.dataset, batch_size=2, shuffle=False, num_workers=0,
                                              pin_memory=self.config.pin_memory, sampler=self.sampler)

            self.sampler_val = KkSampler(self.config, self.dataset_val)
            self.dataLoader_val = data.DataLoader(self.dataset_val, batch_size=self.config.batch_size,
                                                  shuffle=False, sampler=self.sampler_val, num_workers=0,
                                                  pin_memory=self.config.pin_memory)

    def _val(self, iepoch: int):
        val_loss = 0
        val_perc = 0
        for ibatch, (x, y) in enumerate(self.dataLoader_val):
            x = x.to(self.config.device)
            y = y.to(self.config.device)
            loss, _perc = self._val_batch(iepoch=iepoch, ibatch=ibatch, x=x, y=y)
            # print("  >> val{} : loss {}".format(ibatch + 1, loss))
            if ibatch == 0:
                val_loss = loss
                val_perc = _perc
            else:
                # val_loss = val_loss * iepoch / (iepoch + 1) + loss / (iepoch + 1)
                val_loss = val_loss + (loss - val_loss) / (ibatch + 1)
                val_perc = val_perc + (_perc - val_perc) / (ibatch + 1)

        # 显示结果
        _loss_sign = "(*)"
        _perc_sign = "(*)"
        if self.val_loss is not None:
            _loss_diff = self.val_loss - val_loss
            _loss_sign = "(-" if _loss_diff > 0 else "(+"
            _loss_sign += str(abs(_loss_diff)) + ")"

            _perc_diff = self.val_perc - val_perc
            _perc_sign = "(-" if _perc_diff > 0 else "(+"
            _perc_sign += str(abs(_perc_diff)) + "%)"

        print("\n  >> 验证信息: RANK:{}/{}, epoch {}/{} "
               .format(self.config.rank, self.config.gpu_count, iepoch + 1, self.config.epoch))
        print("      train_loss : {:<26} |  val_loss {:<22} : {}%".format(self.last_loss[0], _loss_sign, val_loss))
        print("      train_perc : {:<26} |  val_perc {:<22} : {}%".format(self.last_loss[1], _perc_sign, val_perc))

        self.val_loss = val_loss
        self.val_perc = val_perc
        return val_loss, val_perc

    def _val_batch(self, *, iepoch, ibatch, x, y):
        with torch.no_grad():
            o = self.model(x)
            loss, _perc = self._loss(o, y)
            # loss.backward()
            return round(loss.item(), 2), _perc

    def _batch(self, *, iepoch, ibatch, x, y):
        if hasattr(self.model, "before_forward"):
            self.model.before_forward(x=x, y=y)
        o = self.model(x)
        loss, _perc = self._loss(o, y)
        if hasattr(self.model, "after_loss"):
            self.model.after_loss(x=x, o=o, y=y, loss=loss)
        loss.backward()
        return loss.item(), _perc

    def _epoch(self, iepoch: int):
        # self.optim.zero_grad()
        if hasattr(self.model_src, "epoch_reset"):
            self.model_src.epoch_reset(iepoch=iepoch)
        if self.config.use_layer_optim and self.config.use_layer_optim_by_batch:
            for ibatch, (x, y) in enumerate(self.dataLoader):
                x = x.to(self.config.device)
                y = y.to(self.config.device)
                self.config.sys_ibatch = ibatch
                self._layer_optim(batch_epoch=True)
                loss = self._batch(iepoch=iepoch, ibatch=ibatch, x=x, y=y)
                self._optim(ibatch=ibatch, loss=loss)
        else:
            need_optim = False
            for ibatch, (x, y) in enumerate(self.dataLoader):
                x = x.to(self.config.device)
                y = y.to(self.config.device)
                # 按层优化
                self.config.sys_ibatch = ibatch
                loss = self._batch(iepoch=iepoch, ibatch=ibatch, x=x, y=y)
                # print("  >> Epoch {}/{}, batch {}/{}, rank:{}, loss:{}".format(iepoch, self.config.epoch,
                #                                              ibatch, len(self.dataLoader), self.config.rank, loss))
                need_optim = True
                if ibatch % self.config.accumulation_steps == 0:
                    self._optim(ibatch=ibatch, loss=loss)
                    need_optim = False

            if need_optim:
                self._optim(ibatch=ibatch, loss=loss)

    def train(self):
        self.config.sys_init()
        self._device_init()
        self._data_init()
        self._layer_optim_init(self.model)
        try:
            # 开始训练
            if self.config.rank == 0:
                print("  >> 开始训练 << epoch : {}, batch_size : {}, world_size : {}, gpu_count:{},"
                      .format(self.config.epoch, self.config.batch_size,
                              self.config.world_size, self.config.gpu_count))
                _loss_sign = "(*)"
                print("      loss : {0:<26}  |   perc : {1:<26}"
                      .format(("None" if self.last_loss[0] is None else self.last_loss[0]),
                              ("None" if self.last_loss[1] is None else self.last_loss[0])))

            for iepoch in range(self.config.epoch):
                print("\n|------ [ Epoch ] : {} / {}  |  rank : {}  |  world_size : {} -----------------------------"
                      .format(iepoch + 1, self.config.epoch, self.config.rank, self.config.world_size))
                # 训练
                self.config.sys_iepoch = iepoch
                self.config.sys_ibatch = -1
                self.config.sys_training = True  # 标记状态
                self.model.train()
                if isinstance(self.sampler, dist_data.DistributedSampler):
                    self.sampler.set_epoch(iepoch)
                self._layer_optim(batch_epoch=False)

                # 进入 一个 epoch
                self._epoch(iepoch)

                # 进行验证
                self.config.sys_training = False
                self.model.eval()
                if isinstance(self.sampler_val, dist_data.DistributedSampler):
                    self.sampler_val.set_epoch(iepoch)
                val_loss, val_perc = self._val(iepoch)
                if val_loss < self.config.stop_train_loss:
                    print("\n\n  >> KkTrain.train << Rank {} : 当前预测精度已满足系统设计要求，训练结束。".format(self.config.rank))
                    break
            # if self.config.world_size > 1:
                # dist.barrier()
                # 在训练结束时进行全局归约
                # if self.config.rank == 0:
                #     for param in self.model.parameters():
                #         dist.all_reduce(param.data, op=dist.reduce_op.SUM)
                #         param.data /= dist.get_world_size()
                # 强制保存
                # self._model_save(ibatch=ibatch, loss=loss, is_force=True)
                # 在分布式训练结束时进行同步
        finally:
            self._device_uninit()


class KkDemoModel(kkb.KkModule):
    def __init__(self, config: kkc.KkmConfig, *, in_feather: int = 2):
        super(KkDemoModel, self).__init__(config)
        self.Linear = nn.Linear(in_feather, 1)

    def forward(self, x):
        o = self.Linear(x)
        return o


def torchrun():
    # 运行指令 torchrun --nperc-per-node 1 .\kk_app.py
    config = kkc.KkmConfig(__file__)
    datas = torch.randn(1000, 3)
    # print(datas[:, 0:88].sum(dim=1) / 3.1415926)
    datas[:, 2] = datas[:, 0:2].sum(dim=1) / 3.1415926
    dataset = KkDataset(config, datas)
    datas_val = torch.randn(100, 3)
    datas_val[:, 2] = datas_val[:, 0:2].sum(dim=1) / 3.1415926
    dataset_val = KkDataset(config, datas_val)
    model = KkDemoModel(config, in_feather=2)
    loss_fn = KkExtendLoss(config, lossFn=nn.MSELoss())
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = KkTrain(config, model=model, dataset=dataset, dataset_val=dataset_val,
                      loss_fn=loss_fn, optim=optim)
    trainer.train()


def python_spawn():
    def _python_spawn_fn(rank: int, world_size: int):
        os.environ['RANK'] = str(rank)
        os.environ['LOCAL_RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['MASTER_ADDR'] = "127.0.0.1"
        os.environ['MASTER_PORT'] = "16666"
        torchrun()

    world_size = torch.cuda.device_count()
    # python_spawn_fn 的第1个参数是rank，由mp.spawn 函数内容自动传入，范围为 0-（world_size-1）
    mp.spawn(_python_spawn_fn, args=(world_size,), nprocs=world_size)


def python():
    # os.environ['RANK'] = "0"
    # os.environ['LOCAL_RANK'] = "0"
    # os.environ['WORLD_SIZE'] = "1"
    # os.environ['MASTER_ADDR'] = "127.0.0.1"
    # os.environ['MASTER_PORT'] = "16666"

    torchrun()


if __name__ == "__main__":
    # 系统环境变量中 添加 "PYTHONPATH"， 指向项目
    # 在moudle 文件中 增加 __init__.py 文件
    # 端口也可能 影响 torchrun 的启动
    python()
    # python_spawn()
    # torchrun()
