import torch.nn as nn


class Target(nn.Module):
    def __init__(self):
        self.target_list = []   # 存储函数，存储模型的 target函数
        self.target_name_list = []
        self.loss_info = {}

    # target: target 对象
    # target_name : target 名称
    def update(self, target, target_name):
        self.target_list.append(target.forward)
        self.target_name_list.append(target_name)
        if "_modules" in self.__dict__:
            self.__dict__["_modules"].update(target.__dict__["_modules"])
        else:
            self.__dict__.update(target.__dict__)

    def forward(self, memory_bank, tgt, seg):
        self.loss_info = {}
        for i, target in enumerate(self.target_list):
            if len(self.target_list) > 1:
                self.loss_info[self.target_name_list[i]] = target(memory_bank, tgt[self.target_name_list[i]], seg)
            else:
                # 只有一个
                self.loss_info = target(memory_bank, tgt, seg)

        return self.loss_info
