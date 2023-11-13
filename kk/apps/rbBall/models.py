import torch
import torch.nn as nn
import kk.apps.kk_app as kka
import kk.uer.kk_base as kkb
import kk.uer.kk_config as kkc
import kk.uer.layers.kk_linear as kkl
import kk.uer.layers.kk_transformer as kkt
import kk.uer.layers.kk_cross_attention as kkca
import kk.uer.layers.kk_selfattention as kksa


class RBModel(kka.KkAppModel):
    def __init__(self):
        super(RBModel, self).__init__()
        self.o_feathers = 49
        self.c_feathers = self.o_feathers * 11
        self.in_feathers = 88
        self.a_feathers = self.c_feathers + self.in_feathers
        context = torch.zeros((1, self.c_feathers), dtype=torch.float32)
        self.register_buffer("context", context)

        self.c_net = kkca.KkCrossAttention(self.c_feathers, self.in_feathers,
                                           out_length=self.o_feathers)
        self.classifier = kkl.KkClassifierLayer(self.a_feathers, 1, one_formal="sigmoid")

    def forward(self, x):
        o = self.c_net(self.context, x)
        o = self.classifier(o)
        o = torch.squeeze(o, dim=-1)
        return o

    def after_loss(self, **args):
        # x = args["x"]
        # o = args["o"]
        y = args["y"]
        # loss = args["loss"]
        self.context[0:, 0:49] += torch.sum(y, dim=0)                # 前49个
        _y = y.view(1, -1)
        _yun_count = 0 - _y.size(-1)
        yun_count = 49 + _y.size(-1)
        _content = self.context.clone()
        self.context[0:, 49:_yun_count] = _content[0:, yun_count:]      # 移动
        self.context[0:, _yun_count:] = _y
        pass

    def epoch_reset(self, **args):
        iepoch = args["iepoch"]
        self.context -= self.context  # 清空内容
        pass