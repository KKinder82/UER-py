import torch
import torch.nn as nn


class GatedcnnEncoder(nn.Module):
    """
    Gated CNN encoder.
    """
    def __init__(self, args):
        super(GatedcnnEncoder, self).__init__()
        self.layers_num = args.layers_num       # 层数
        self.kernel_size = args.kernel_size     # 卷积核个数
        self.block_size = args.block_size       # 块大小
        self.emb_size = args.emb_size           # 词向量维度大小
        self.hidden_size = args.hidden_size     # 隐藏层大小

        self.conv_1 = nn.Conv2d(1, args.hidden_size, (args.kernel_size, args.emb_size))
        self.gate_1 = nn.Conv2d(1, args.hidden_size, (args.kernel_size, args.emb_size))

        # conv_b1 [1, hidden_size, 1, 1]
        self.conv_b1 = nn.Parameter(torch.randn(1, args.hidden_size, 1, 1))
        self.gate_b1 = nn.Parameter(torch.randn(1, args.hidden_size, 1, 1))

        self.conv = nn.ModuleList(
            [
                nn.Conv2d(args.hidden_size, args.hidden_size, (args.kernel_size, 1))
                for _ in range(args.layers_num - 1)
            ]
        )
        self.gate = nn.ModuleList(
            [
                nn.Conv2d(args.hidden_size, args.hidden_size, (args.kernel_size, 1))
                for _ in range(args.layers_num - 1)
            ]
        )

        self.conv_b = nn.ParameterList(
            nn.Parameter(torch.randn(1, args.hidden_size, 1, 1))
            for _ in range(args.layers_num - 1)
        )
        self.gate_b = nn.ParameterList(
            nn.Parameter(torch.randn(1, args.hidden_size, 1, 1))
            for _ in range(args.layers_num - 1)
        )

    # emb: batch_size, seq_length, emb_size
    # <- output: batch_size, seq_length, hidden_size
    def forward(self, emb, seg):
        batch_size, seq_length, _ = emb.size()

        padding = torch.zeros([batch_size, self.kernel_size-1, self.emb_size]).to(emb.device)
        emb = torch.cat([padding, emb], dim=1).unsqueeze(1)  # batch_size, 1, seq_length+ self.kernel_size-1, emb_size
        # hidden: batch_size, hidden_size, seq_length, 1
        hidden = self.conv_1(emb)
        hidden += self.conv_b1.repeat(1, 1, seq_length, 1)
        # gate: batch_size, hidden_size, seq_length, 1
        gate = self.gate_1(emb)
        gate += self.gate_b1.repeat(1, 1, seq_length, 1)
        hidden = hidden * torch.sigmoid(gate)

        res_input = hidden

        padding = torch.zeros([batch_size, self.hidden_size, self.kernel_size-1, 1]).to(emb.device)
        # hidden: batch_size, hidden_size, seq_length + self.kernel_size-1, 1
        hidden = torch.cat([padding, hidden], dim=2)

        for i, (conv_i, gate_i) in enumerate(zip(self.conv, self.gate)):
            # hidden: batch_size, hidden_size, seq_length, 1
            hidden, gate = conv_i(hidden), gate_i(hidden)
            hidden += self.conv_b[i].repeat(1, 1, seq_length, 1)
            gate += self.gate_b[i].repeat(1, 1, seq_length, 1)

            # 门控的卷积
            hidden = hidden * torch.sigmoid(gate)
            if (i + 1) % self.block_size == 0:
                hidden = hidden + res_input
                res_input = hidden
            hidden = torch.cat([padding, hidden], dim=2)

        # hidden: batch_size, hidden_size, seq_length, 1
        hidden = hidden[:, :, self.kernel_size - 1:, :]
        # output: batch_size, seq_length, hidden_size
        output = hidden.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)

        return output
