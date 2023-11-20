import torch
import torch.nn as nn
import math
import random
import kk.lm.kk_config as kkc


def init_weights(model, *, mean=0.0, std: (str, float) = 0.01):
    if isinstance(std, str):
        if std == "relu":
            std = math.sqrt(2.0 / model.weight.size(-1))
        elif std == "tanh":
            std = 5.0 / 3.0 * math.sqrt(3.0 / model.weight.size(-1))
        elif std == "sigmoid":
            std = math.sqrt(1.0 / model.weight.size(-1))
        elif std == "rand":
            std = random.randint(1, 10) / 100
        else:
            std = 0.01
    else:
        if std <= 0:
            std = random.randint(1, 10) / 100

    if isinstance(model, nn.Linear):
        nn.init.normal_(model.weight, mean=mean, std=std)
        nn.init.constant_(model.bias, 0)
    elif isinstance(model, nn.Embedding):
        nn.init.normal_(model.weight, mean=mean, std=std)
    elif isinstance(model, nn.LayerNorm):
        nn.init.constant_(model.bias, mean)
        nn.init.constant_(model.weight, 1.0 + std)
    elif isinstance(model, nn.Conv2d):
        nn.init.normal_(model.weight, mean=mean, std=std)
        nn.init.constant_(model.bias, mean)
    elif isinstance(model, nn.Conv1d):
        nn.init.normal_(model.weight, mean=mean, std=std)
        nn.init.constant_(model.bias, mean)
    elif isinstance(model, nn.BatchNorm1d):
        nn.init.constant_(model.bias, mean)
        nn.init.constant_(model.weight, 1.0 + std)
    elif isinstance(model, nn.BatchNorm2d):
        nn.init.constant_(model.bias, mean)
        nn.init.constant_(model.weight, 1.0 + std)
    elif isinstance(model, nn.GRU):
        for name, param in model.named_parameters():
            if 'weight' in name or 'bias' in name:
                nn.init.normal_(param, mean=mean, std=std)
    elif isinstance(model, nn.LSTM):
        for name, param in model.named_parameters():
            if 'weight' in name or 'bias' in name:
                nn.init.normal_(param, mean=mean, std=std)
    elif isinstance(model, nn.RNN):
        for name, param in model.named_parameters():
            if 'weight' in name or 'bias' in name:
                nn.init.normal_(param, mean=mean, std=std)
    else:
        pass


def get_randn_parameter(*shape, mean=0.0, std: (str, float) = 0.01):
    if isinstance(std, str):
        if std == "kk":
            weight = torch.randint(1, 100000, shape).to(torch.float32)
            weight = weight.pow(1.9) / weight.sum()
            return weight
        elif std == "relu":
            std = math.sqrt(2.0 / shape[-1])
        elif std == "tanh":
            std = 5.0 / 3.0 * math.sqrt(3.0 / shape[-1])
        elif std == "sigmoid":
            std = math.sqrt(1.0 / shape[-1])
        elif std == "normal":
            std = 0.01
        else:
            std = random.randint(1, 10) / 100
    else:
        if std <= 0:
            std = random.randint(1, 10) / 100
    return torch.randn(shape) * std + mean


def get_constant_parameter(*shape, value: int = 1):
    return torch.full(shape, value, dtype=torch.float32, requires_grad=True)


class KkModule(nn.Module):
    def __init__(self):
        super(KkModule, self).__init__()

    def epoch_reset(self, **args):
        # iepoch = args["iepoch"]
        pass

    def before_forward(self, **args):
        # x = args["x"]
        # o = args["o"]
        # y = args["y"]
        # loss = args["loss"]
        # print("  >> --------------------- after_loss --------------------")
        # print(args["o"])
        # print(args["y"])
        pass

    def after_loss(self, **args):
        # x = args["x"]
        # o = args["o"]
        # y = args["y"]
        # loss = args["loss"]
        pass
