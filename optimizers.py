"""Optimizer implementations from scratch built on top of PyTorch."""

import numpy as np
from sklearn import datasets
import torch
from torch import nn
from torch import optim


class SGD(optim.Optimizer):
    """Implementation of Stochastic Gradient Descent (SGD) with momentum."""

    def __init__(self, params, lr, momentum=0):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            for param in group['params']:
                if param.grad is None:
                    continue
                grad = param.grad.data
                if momentum:
                    state = self.state[param]
                    if 'prev_grad' not in state:
                        grad = grad.clone().detach()
                    else:
                        prev_grad = state['prev_grad']
                        grad += momentum * prev_grad
                    state['prev_grad'] = grad
                step = lr * grad
                param.data.add_(-step)


class Adam(optim.Optimizer):
    """Naive implementation of the Adam optimizer.

    This implementation basically follows the math in the paper and
    does not account for things like stability issues.
    """

    def __init__(self, params, lr, beta1=.9, beta2=.99, eps=1e-8):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['beta1']
            beta2 = group['beta2']
            eps = group['eps']
            for param in group['params']:
                if param.grad is None:
                    continue
                grad = param.grad.data
                state = self.state[param]
                if not state:
                    state['t'] = 0
                    state['mov_avg'] = torch.zeros_like(param.grad.data)
                    state['mov_var'] = torch.zeros_like(param.grad.data)
                t = state['t']
                t += 1
                state['t'] = t
                mov_avg, mov_var = state['mov_avg'], state['mov_var']
                mov_avg = beta1 * mov_avg + (1 - beta1) * grad
                mov_var = beta2 * mov_var + (1 - beta2) * grad**2
                state['mov_avg'] = mov_avg
                state['mov_var'] = mov_var
                # Bias correct the moment estimates.
                mov_avg = mov_avg / (1 - beta1**t)
                mov_var = mov_var / (1 - beta2**t)
                step = lr * mov_avg / (torch.sqrt(mov_var) + eps)
                param.data.add_(-step)


class LinearRegressor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self._linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self._linear(x)


if __name__ == '__main__':
    input_size = 10
    X, y, coef = datasets.make_regression(
        n_samples=100, n_features=input_size, n_informative=input_size,
        n_targets=1, bias=0., coef=True, random_state=42)
    X = torch.from_numpy(X).type(torch.float32)
    y = torch.from_numpy(y).type(torch.float32).unsqueeze(1)

    model = LinearRegressor(input_size)
    optimizer = Adam(model.parameters(), lr=.1)
    criterion = torch.nn.MSELoss()
    for i in range(1000):
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

    w, b = [p.data.numpy() for p in model.parameters()]
    print("actual_bias=0\nlearned_bias={}".format(b))
    print("actual_weights={}\nlearned_weights={}".format(coef, w))
