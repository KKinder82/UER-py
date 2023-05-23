import math
import torch
import torch.nn as nn


class MultiHeadedAttention(nn.Module):
    """
    Each head is a self-attention operation.
    self-attention refers to https://arxiv.org/pdf/1706.03762.pdf
    """
    # hidden_size 隐层 特征大小
    # heads_num 头数
    # attention_head_size 每个头的 特征大小
    def __init__(self, hidden_size, heads_num, attention_head_size, dropout, has_bias=True, with_scale = True):
        super(MultiHeadedAttention, self).__init__()
        self.heads_num = heads_num

        self.per_head_size = attention_head_size
        self.with_scale = with_scale
        self.inner_hidden_size = heads_num * attention_head_size

        # 三个线性层 | 分别 对 QKV 进行 线性变换
        self.linear_layers = nn.ModuleList(
                [nn.Linear(hidden_size, self.inner_hidden_size, bias=has_bias) for _ in range(3)]
            )
        
        self.dropout = nn.Dropout(dropout)
        # 最后的线程性
        self.final_linear = nn.Linear(self.inner_hidden_size, hidden_size, bias=has_bias)

    # has_residual_attention 是否支持残差
    # prev_attn 上一次（层）的注意力 | 本次结果 与 上一层 结果 进行 残差

    def forward(self, key, value, query, mask, position_bias=None, has_residual_attention=False, prev_attn=None):
        """
        Args:
            key: [batch_size x seq_length x hidden_size]
            value: [batch_size x seq_length x hidden_size]
            query: [batch_size x seq_length x hidden_size]
            mask: [batch_size x 1|多头 x seq_length x seq_length]
            position_bias: [1 x heads_num x seq_length x seq_length]  # 位置 偏移
        Returns:
            output: [batch_size x seq_length x hidden_size]
        """
        batch_size, seq_length, _ = query.size()
        heads_num = self.heads_num
        per_head_size = self.per_head_size

        def shape(x):
            return x. \
                   contiguous(). \
                   view(batch_size, seq_length, heads_num, per_head_size). \
                   transpose(1, 2)

        def unshape(x):
            return x. \
                   transpose(1, 2). \
                   contiguous(). \
                   view(batch_size, seq_length, self.inner_hidden_size)


        # 对 QKV 进行 线性变换|影射
        #     (batch_size, seq_length, hidden_size)
        #  -> (batch_size, seq_length, inner_hidden_size)
        #  -> (batch_size, seq_length, inner_hidden_size)

        query, key, value = [l(x). \
                             view(batch_size, -1, heads_num, per_head_size). \
                             transpose(1, 2) \
                             for l, x in zip(self.linear_layers, (query, key, value))
                            ]

        scores = torch.matmul(query, key.transpose(-2, -1))
        if position_bias is not None:
            scores = scores + position_bias
        if self.with_scale:
            scores = scores / math.sqrt(float(per_head_size))

        # Mask 是 加 吗？ | 为什么不是乘法 | 原因是 结果 要进行 Softmax 归一化时 -oo 会更有效果
        scores = scores + mask.type_as(scores)
        prev_attn_out = None
        if has_residual_attention:          # residual 残差 attention
            if prev_attn is not None:
                scores += prev_attn
            prev_attn_out = scores
        probs = nn.Softmax(dim=-1)(scores)
        probs = self.dropout(probs)
        #    (batch_size, heads_num, seq_length, per_head_size)
        # -> (batch_size, seq_length, heads_num, per_head_size) -> (batch_size, seq_length, heads_num * per_head_size)
        # -> (batch_size, seq_length, inner_hidden_size)
        output = unshape(torch.matmul(probs, value))

        # 对后进行处理
        output = self.final_linear(output)
        return output, prev_attn_out
