import torch
import torch.nn as nn
import kk.lm.kk_app as kka
import kk.lm.kk_base as kkb
import kk.lm.kk_config as kkc
import kk.lm.layers.kk_linear as kkl
import kk.lm.layers.kk_mind as kkm
import kk.lm.layers.kk_transformer as kkt
import kk.lm.layers.kk_cross_attention as kkca
import kk.lm.layers.kk_selfattention as kksa


class RBModel(kka.KkAppModel):
    def __init__(self):
        super(RBModel, self).__init__()
        self.out_feathers = 7
        self.c_feathers = 49 + self.out_feathers * 10
        self.in_feathers = 88
        self.all_feathers = self.c_feathers + self.in_feathers
        context = torch.zeros((1, self.c_feathers), dtype=torch.float32)
        self.register_buffer("context", context)

        self.cast_net = kkm.KkLinearEx(self.all_feathers, self.all_feathers, )

        # self.c_net = kkca.KkCrossAttention(self.c_feathers, self.in_feathers,
        #                                    out_length=self.o_feathers)
        self.c_net = kksa.KkMultiSelfAttention(self.all_feathers, self.all_feathers,
                                               head_count=16)
        self.classifier = kkl.KkClassifierLayer(self.all_feathers, self.out_feathers,
                                                one_formal="sigmoid", inner_feathers=1024)

    def forward(self, x):
        _context = self.context.expand(x.size(0), -1)
        o = torch.concatenate((_context, x), dim=-1)
        o = self.cast_net(o)
        o = self.c_net(o, o, o)
        o = self.classifier(o)
        o = torch.squeeze(o, dim=-1)
        o = o[:, 0:] * 32 + 1
        return o[..., 0:1]

    def after_loss(self, **args):
        # x = args["x"]
        # o = args["o"]
        y = args["y"]
        # loss = args["loss"]
        for i in range(y.size(0)):
            self.context[0:, y[i, :-1].long()-1] += 1
            self.context[0:, 33 + y[i, -1:].long() - 1] += 1
        # self.context[0:, 0:49] += torch.sum(y, dim=0)                # 前49个
        _y = y.view(1, -1)
        _yu_count = 0 - _y.size(-1)
        yu_count = 49 + _y.size(-1)
        _content = self.context.clone()
        self.context[0:, 49:_yu_count] = _content[0:, yu_count:]      # 移动
        self.context[0:, _yu_count:] = _y
        pass

    def epoch_reset(self, **args):
        iepoch = args["iepoch"]
        self.context -= self.context  # 清空内容
        pass


class RbClassfierLoss(kkb.KkModule):
    def __init__(self):
        super(RbClassfierLoss, self).__init__()
        self.loos_fn = nn.MSELoss()

    def forward(self, o, y):
        _y = y[..., 0:1]
        _s = o - _y
        perc = 1 - (torch.count_nonzero(_s) / _s.numel())
        loss = self.loos_fn(o, _y)
        return loss, 0, perc * 100

    def forward_after(self, o, y, loss):
        # loss = loss - torch.std(o) * 2000.0
        return loss
