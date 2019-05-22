"""Simple Gaussian noise generator using Box-Muller Transform."""

import matplotlib.pyplot as plt
import numpy as np

def normal_sample(mean=0, std_dev=1):
    """Generates Gaussian noise using the Box-Muller Transform."""
    u = np.random.uniform()
    v = np.random.uniform()

    z = np.sqrt(-2*np.log(u))*np.cos(2*np.pi*v)

    return z * std_dev + mean


if __name__ == '__main__':
    x = []
    for i in range(10000):
        x.append(normal_sample(mean=-1, std_dev=3))
    plt.hist(x, bins=100)
    plt.show()

    
