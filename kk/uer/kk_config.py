import os
import torch

import numpy as np
import logging as log
import kk.kk_utils as kku
import random
import time


class KkmConfig(object):
    def __init__(self, app_path: str, *args, **kwargs):  # args:可变数量参数； kwargs：关键字参数（可变数量）
        log.basicConfig(level=log.INFO, format="%(created)f %(asctime)s %(levelname)s %(message)s \r\n", datefmt="%Y-%m-%d %H:%M:%S")
        # 应用配置
        self.app_name = "DLApp"
        if not os.path.exists(app_path):
            raise FileNotFoundError(" 应用路径不存在。")
        self.app_path = app_path if os.path.isdir(app_path) else os.path.dirname(app_path)

        # 分布式训练
        self.rank = kku.env_int('RANK', 0)                   # int(os.environ['RANK'])
        self.local_rank = kku.env_int('LOCAL_RANK', 0)       # int(os.environ['LOCAL_RANK'])
        self.world_size = kku.env_int('WORLD_SIZE', 1)       # int(os.environ['WORLD_SIZE'])
        self.master_addr = kku.env_value('MASTER_ADDR', "127.0.0.1")  # os.environ['MASTER_ADDR']
        self.master_port = kku.env_value('MASTER_PORT', "16666")        # os.environ['MASTER_PORT']
        self.master_port = "16666"
        self.backend = "nccl"
        # GPU 训练
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_count = min(torch.cuda.device_count(), self.world_size)
        # 数据集
        self.shuffle = False
        self.batch_size = 1
        self.batch_count = 0
        self.num_workers = self.world_size
        self.pin_memory = False
        self.batch_ceil = False      # 当数据不能整除时，是否补齐
        # 训练
        self.sys_iepoch = 0
        self.sys_ibatch = 0
        self.epoch = 10000
        self.accumulation_steps = 1
        self.save_checkpoint_steps = 20
        self.report_steps = 1
        self.checkpoint_mode = "dict"               # None: 不保存, dict: 参数文件, model : 模型与参数
        self.checkpoint_last = "model_last.pth"
        self.checkpoint_best = "model_best.pth"
        self.stop_train_loss = 0.005
        self.use_layer_optim = False                # 是否启用分层优化,  加载 模型后，自动设备为 False
        self.use_layer_optim_random = False         # 选择选层优化，（顺序)
        self.use_layer_optim_from_zero = True       # True 从0层到 sys_parameters_locked_index 优化， False: 单层
        self.use_layer_optim_param_group_size = 2    # 分层优化增量  < 1 则随机
        self.use_layer_optim_by_batch = True        # True 按批次优化, False 按epoch
        self.use_layer_optim_by_count = 20           # 按批次优化时，每多少个批次(Epoch)优化一次
        self.use_layer_optim_loops = 1000              # 优化次数
        self.use_layer_optim_finished_rand = 0       # >0 ：layer_optim 结束后，随机停止优化的层数
        # 训练（系统）
        self.sys_training = False                 # 检查是否在验证阶段
        self.sys_layer_optim_models = {}          # 存储分层优化的模型  parameters_layers: 含有参数的层数
                                                   # layer_optim_loops    : 第几轮
                                                   # layer_optim_times    : 当前锁定参数的层索引
                                                   # layer_optim_index    : 上次锁定层数
                                                   # layer_optim_finished : 优化结束标志
        self.sys_param_check_loops = 0             # 网络参数检查 次数

        # 模型加载
        self.pt_load = True                     # 是否加载： True: 加载， False: 不加载
        self.pt = "model_best.pth"

        # 初始化目录
        if self.pt and not os.path.isabs(self.pt):
            self.pt = os.path.join(self.app_path, self.pt)
        if self.checkpoint_last and not os.path.isabs(self.checkpoint_last):
            self.checkpoint_last = os.path.join(self.app_path, self.checkpoint_last)
        if self.checkpoint_best and not os.path.isabs(self.checkpoint_best):
            self.checkpoint_best = os.path.join(self.app_path, self.checkpoint_best)

        # 初始化随机值
        _seed = (self.rank + int(time.time() * 1000)) % (2 ** 32 - 1)
        random.seed(_seed)
        np.random.seed(_seed)
        torch.manual_seed(self.rank + time.time())
        torch.cuda.manual_seed(self.rank + time.time())
        torch.cuda.manual_seed_all(self.rank + time.time())
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def sys_init(self):
        self.sys_ibatch = -1
        self.sys_iepoch = -1
        self.sys_training = False
        self.sys_layer_optim_models = {}

# #################################  配置帮助  ##########################################
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
