import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class RBModel(nn.Module):
    def __init__(self):
        super(RBModel, self).__init__()
        self.net1 = nn.Sequential(
            nn.Linear(88, 88 * 3, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(88 * 3),
            nn.Linear(88 * 3, 88, bias=False),
        )

        self.net2 = nn.Sequential(
            nn.Linear(88, 88 * 3, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(88 * 3),
            nn.Linear(88 * 3, 88, bias=False),
        )

        self.net_Q = nn.Sequential(
            nn.Linear(88*2, 88, bias=False),
        )

        self.net_K = nn.Sequential(
            nn.Linear(88*2, 88, bias=False),
        )

        self.net_V = nn.Sequential(
            nn.Linear(88*2, 88, bias=False),
        )

        self.TransLayer = nn.Sequential(
            nn.Transformer(88 * 6, 6),
            nn.MultiheadAttention(88 * 6, 6),
            nn.MultiheadAttention(88 * 6, 6),
        )

        self.t = nn.Transformer()
        self.t()

        self.MLP = nn.Sequential(
            nn.Linear(88 * )

    def forward(self, x, last):
        x = self.net1(x)
        bs = x.shape[0]
        last = self.net2(last)
        x_all = torch.cat([x, last], dim=0)
        Q = self.net_Q(x_all)
        K = self.net_K(x_all)
        V = self.net_V(x_all)
        x = self.MSA(Q, K, V)
        x = x.view(88)
        return x




def main():
    train = np.load("../datasets/rbBall/rbBall_train.npy")
    val = np.load("../datasets/rbBall/rbBall_train.npy")
    # test = np.load("../datasets/rbBall/rbBall_train.npy")
    x = train[:, :88]
    y = train[:, 88:]

    # 编码


    print(y.shape)

    pass


if __name__ == "__main__":
    main()