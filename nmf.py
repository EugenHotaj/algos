"""Non-negative Matrix Factorization (NMF).

The NMF problem is described as follows:
    For some matrix M, find non-negative matricies (i.e. all elements >= 0) W 
    and H s.t. M = WH. 

This module provides 2 different algorithms for arriving at a solution:
    1. Basic gradient descent (GD) implemented by the NMF_GD function
    2. Projected gradient descent (PGD) implemented by the NMF_PGD function

To run:
    python3 nmf.py
"""

import numpy as np


def NMF_GD(M, inner_dim=3, steps=10000, lr=.001, lam=1e3, eps=1e-6):
    """Finds approximate non-negative factorization of M via constrained GD.

    Let M be an (m x n) matrix. Then M is factorized into W and H where W is an
    (m x inner_dim) matrix and H is an (inner_dim, n) matrix. 

    Args:
        M: The matrix to factorize.
        inner_dim: The height and width of W and H respectivley.
        steps: The number of gradient descent steps to run.
        lr: The learning rate, i.e. gradient descent step size. The learning
            rate is linearly annealed to 0 over the course of training.
        lam: The lagrangian multiplier term, i.e. by how much to penalize W, H
            for not being non-negative. 
        eps: The epsilon precison below which to consider things equal.
    Returns:
        The factorization matricies W, H. 
    """
    m, n = M.shape
    W = np.random.uniform(size=(m, inner_dim))
    H = np.random.uniform(size=(inner_dim, n))
    for i in range(steps):
        lr_ = lr - (i / steps * lr)
        dW, dH = _penalized_gradient(M, W, H, lam, eps)
        W -= lr_ * dW
        H -= lr_ * dH
    return W, H


def _penalized_gradient(M, W, H, lam, eps):
    """The gradient of the penalized loss with respect to W and H.
    
    The penalized loss is defined as:
        l = .5*norm(M - WH)^2 + \lamba * (.5*max(W, 0)^2 + .5*max(H, 0)^2)
    where we define norm to be the Frobenius norm and max(A, 0) is the element
    wise max operator on A.
    """
    dW, dH = _gradient(M, W, H)
    dW += lam * _penalty(W, eps)
    dH += lam * _penalty(H, eps)
    return dW, dH


def _penalty(A, eps):
    return np.where(A >= eps, 0, np.min(A - eps, 0))


def NMF_PGD(M, inner_dim=3, steps=10000, eps=1e-6):
    """Finds approximate non-negative factorization of M via constrained PGD.

    Let M be an (m x n) matrix. Then M is factorized into W and H where W is an
    (m x inner_dim) matrix and H is an (inner_dim, n) matrix. 

    The PGD algorithm computes the step size to be 1/beta where beta is the 
    Lipschitz constant of the gradient of the loss function.

    Args:
        M: The matrix to factorize.
        inner_dim: The height and width of W and H respectivley.
        steps: The number of gradient descent steps to run.
    Returns:
        The factorization matricies W, H. 
    """

    m, n = M.shape
    W = np.random.uniform(size=(m, inner_dim))
    H = np.random.uniform(size=(inner_dim, n))
    for _ in range(steps):
        # Update W.
        dW, _ = _gradient(M, W, H)
        lr = 1 / np.linalg.norm(np.dot(H, H.T))
        W -= lr * dW
        W[W < eps] = eps

        # Update H using updated W.
        _, dH = _gradient(M, W, H)
        lr = 1 / np.linalg.norm(np.dot(W.T, W))
        H -= lr * H
        H[H < eps] = eps
    return W, H


def _gradient(M, W, H):
    """The gradient of the loss with respect to W and H.
    
    The loss is defined as:
        l = .5*norm(M - WH)^2
    where we define norm to be the Frobenius norm.
    """
    residual = np.dot(W, H) - M
    dW = np.dot(residual, H.T)
    dH = np.dot(W.T, residual)
    return dW, dH


if __name__ == '__main__':
    M = np.random.uniform(0, 1, size=(10, 10))

    W, H = NMF_GD(M, inner_dim=5, steps=10000, lr=.02)
    print("|M-F|_f using GD: %.5f" % np.linalg.norm(M - np.dot(W, H)))

    W, H = NMF_PGD(M, inner_dim=5, steps=10000)
    print("|M-F|_f using PGD: %.5f" % np.linalg.norm(M - np.dot(W, H)))
