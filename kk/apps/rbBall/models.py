import torch
import kk.apps.kk_app as kka
import kk.uer.layers.kk_Linear as kkl
import kk.uer.layers.kk_Transformer as kkt
import kk.uer.kk_config as kkc


class RBModel(kka.KkAppModel):
    def __init__(self):
        super(RBModel, self).__init__()
        c_feather = (33 + 16) * (1 + 10)
        self.context_net = kkl.KkFFNLayer(c_feather, 128)
        self.x_net = kkl.KkFFNLayer(88, 128)
        self.backbone = kkt.KkTransformer(in_feathers=128, loops=6)
        self.classifier = kkl.KkClassifierLayer(128, 49, one_formal="sigmoid")
        context = torch.zeros((1, c_feather), dtype=torch.float32)
        self.register_buffer("context", context)
        self.last_o = None

    def forward(self, x):
        _context = self.context_net(self.context)
        _x = self.x_net(x)
        o = self.backbone(_context, _x)
        o = self.classifier(o)
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