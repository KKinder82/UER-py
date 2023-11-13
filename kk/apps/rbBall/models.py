import torch
import kk.apps.kk_app as kka
import kk.uer.layers.kk_Linear as kkl
import kk.uer.layers.kk_Transformer as kkt
import kk.uer.kk_config as kkc


class RBModel(kka.KkAppModel):
    def __init__(self, context=None):
        super(RBModel, self).__init__()
        self.context_net = kkl.KkFFNLayer(33 + 16, 128)
        self.x_net = kkl.KkFFNLayer(88, 128)
        self.backbone = kkt.KkTransformer(in_feather=128, loops=6)
        self.classifier = kkl.KkClassifierLayer(128, 49, one_formal="sigmoid")
        if context is None:
            context = torch.zeros((1, 33 + 16), dtype=torch.float32)
        self.register_buffer("context", context)
        self.last_o = None

    def forward(self, x):
        _context = self.context_net(self.context)
        _x = self.x_net(x)
        o = self.backbone(_context, _x, last_o=None)
        o = self.classifier(o)
        return o

    def after_loss(self, **args):
        # x = args["x"]
        # o = args["o"]
        y = args["y"]
        # loss = args["loss"]
        self.context = self.context + torch.sum(y, dim=0)
        pass


    def epoch_reset(self, **args):
        iepoch = args["iepoch"]
        self.context -= self.context  # torch.zeros((1, 33 + 16), dtype=torch.float32)
        pass