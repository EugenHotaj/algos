"""Backpropagation from scratch by hand."""

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

def pytorch_model(W1, b1, W2, b2, W3, b3):
    l1 = nn.Linear(in_features=64, out_features=32)
    l1.weight.data.copy_(torch.from_numpy(W1))
    l1.bias.data.copy_(torch.from_numpy(b1))

    l2 = nn.Linear(in_features=32, out_features=16)
    l2.weight.data.copy_(torch.from_numpy(W2))
    l2.bias.data.copy_(torch.from_numpy(b2))

    l3 = nn.Linear(in_features=16, out_features=2)
    l3.weight.data.copy_(torch.from_numpy(W3))
    l3.bias.data.copy_(torch.from_numpy(b3))

    return nn.Sequential(
            l1, nn.Sigmoid(), l2, nn.Sigmoid(), l3)


if __name__ == '__main__':
    X = np.random.normal(size=(100, 64)).astype(np.float32)  # 100 rows, 64 features.
    y = np.random.randint(0, 2, size=(100, 2)).astype(np.float32)

    # Two hidden layer neural network.
    W1 = np.random.normal(size=(32, 64)).astype(np.float32) # 32 hidden units
    b1 = np.random.normal(size=(32,)).astype(np.float32)
    W2 = np.random.normal(size=(16, 32)).astype(np.float32) # 16 hidden units
    b2 = np.random.normal(size=(16,)).astype(np.float32)
    W3 = np.random.normal(size=(2, 16)).astype(np.float32) # 1 output
    b3 = np.random.normal(size=(2,)).astype(np.float32)

    # Forward pass. 
    z1 = np.dot(X, W1.T) + b1
    h1 = 1 / (1 + np.exp(-z1))
    z2 = np.dot(h1, W2.T) + b2
    h2 = 1 / (1 + np.exp(-z2))
    yhat = np.dot(h2, W3.T) + b3
    loss = (.5 * (yhat - y)**2).mean()

    # Backward pass.
    dyhat = yhat - y
    dW3 = np.dot(dyhat.T, h2)
    db3 = dyhat.mean(axis=0)
    dh2 = np.dot(dyhat, W3)

    dz2 = dh2 * (z2 * (1 - z2))
    dW2 = np.dot(dz2.T, h1)
    db2 = dz2.mean(axis=0)
    dh1 = np.dot(dz2, W2)

    dz1 = dh1 * (z1 * (1 - z1))
    dW1 = np.dot(dz1.T, X)
    db1 = dz1.mean(axis=0)

    net = pytorch_model(W1, b1, W2, b2, W3, b3)
    torch_loss = F.mse_loss(net(torch.from_numpy(X)), torch.from_numpy(y))
    torch_loss.backward()

    np_params = [W1, b1, W2, b2, W3, b3]
    for torch_param, np_param in zip(net.parameters(), np_params):
        torch_param = torch_param.data.numpy()
        print(((torch_param - np_param)**2).mean())
        # print(torch_param)
        # print(np_param)
        print(), print()
