import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.function as Function

import numpy as np
import matplotlib.pyplot as plt


class my_activation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input

class testLayer(torch.nn.Module):
    def __init__(self):
        super(testLayer, self).__init__()
        self.a = torch.nn.Parameter(torch.randn(3))
        self.b = torch.nn.Parameter(torch.randn(3))

    def forward(self, input):
        out = input * self.a
        out = torch.sum(out)
        y = input * self.b
        return out


class MyLinear(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MyLinear, self).__init__()
        self.linear = torch.nn.Linear(input_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, 1)
        # self.weight = torch.nn.Parameter(torch.randn(input_size,hidden_size))
        # self.bias = torch.nn.Parameter(torch.randn(hidden_size))
        self.a = torch.nn.Parameter(torch.randn(hidden_size))
        self.b = torch.nn.Parameter(torch.randn(hidden_size))

    def forward(self, input):
        out = self.linear(input)
        # '''
        for b in range(out.shape[0]):
            out[b][out[b] < self.a] = self.a[out[b] < self.a]
            out[b][out[b] > self.b] = self.b[out[b] > self.b]
        # '''
        out = self.linear2(out)
        return out

def plot(x, y, predict):
    plt.scatter(x, y, c='r')
    plt.plot(x, predict, color='g', label='预测值')
    plt.show()

def main():
    data = torch.arange(0, 30).view(-1, 3).float()
    loss_function = nn.MSELoss()
    model = MyLinear(1, 10)

    for i in model.parameters():
        print(i)

    exit()

    optim = torch.optim.AdamW(model.parameters(), lr=0.001)
    x = data[..., 0:1]
    y = data[..., 1]
    for i in range(1000):
        out = model(x)
        loss = loss_function(out, y)
        loss.backward()
        optim.step()
        optim.zero_grad()
        if i % 50 == 0:
            plot(x.detach().numpy(), y.detach().numpy(), out.detach().numpy())

def print_model(model):
    for i in model.parameters():
        print(i)
        print(i.grad)
