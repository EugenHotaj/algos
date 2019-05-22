"""Linear Regression from scratch."""

import matplotlib.pyplot as plt
import numpy as np

def generate_data(m, b, num=1000):
    """Generates data in [0, 100] of the form y= m*x +b +\eps."""
    X = np.random.uniform(low=0, high=100, size=(num,))
    eps = np.random.normal(scale=35, size=(num,))
    y = m * X + b + eps
    return X, y


def mse(weights, X, y):
    """Calculates the MSE and the gradient with respect to the weights."""
    # TODO(eugenhotaj): handle multidimensional regression.
    # Prepend 1 to every feature since weights[0] = bias.
    X_ = np.ones((X.shape[0], 2))
    X_[:, 1] = X

    residual = y - np.sum(weights * X_, axis=1)
    N = len(y)
    loss = np.sum(residual**2) / N
    grad = (-2/N) * np.sum(X_ * np.reshape(residual, (-1, 1)), axis=0)
    return loss, grad
    

if __name__ == '__main__':
    X, y = generate_data(m=2, b=5)

    # Run gradient descent.
    weights = np.random.uniform(size=(2,)) # [bias, parameter]
    losses = []
    for i in range(10):
        loss, grad = mse(weights, X, y)
        weights -= .0001 * grad 
        losses.append(loss)
    b, m = weights

    def line(x):
        return m * x + b

    plt.figure(figsize=(10,5))

    plt.subplot(121)
    plt.title("Linear Regression")
    plt.scatter(X, y, 8)
    plt.plot([0, 100], [line(0), line(100)], color='red')

    plt.subplot(122)
    plt.title("Loss")
    plt.plot(list(range(len(losses))), losses)

    plt.show()

