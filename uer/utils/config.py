import json
import sys
from argparse import Namespace

# 加载超参数，然后用命令行参数覆盖配置文件中的参数
def load_hyperparam(default_args):
    """
    Load arguments form argparse and config file
    Priority: default options < config file < command line args
    """
    with open(default_args.config_path, mode="r", encoding="utf-8") as f:
        config_args_dict = json.load(f)

    default_args_dict = vars(default_args)      # 返回一个 dict 对象

    command_line_args_dict = {k: default_args_dict[k] for k in [
        a[2:] for a in sys.argv if (a[:2] == "--" and "local_rank" not in a)
    ]}
    default_args_dict.update(config_args_dict)          # 更新配置参数
    default_args_dict.update(command_line_args_dict)    # 更新命令行参数 级别更高
    args = Namespace(**default_args_dict)               # 转换为 Namespace 对象

    return args
