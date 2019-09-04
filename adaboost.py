"""Implementation of the AdaBoost algorithm.

The algorithm is tested on the XOR dataset using as weak learners 8 linear
classifiers.

Example from Ensemble Methods (Zhou 2012).
"""

import math
import random

import numpy as np

xor = [((1, 0), 1), ((-1, 0), 1), ((0, 1), -1), ((0, -1), -1)]

# The hypothesis space of the weak learners.
basis_functions = [
    lambda x1, _: 1 if x1 > -0.5 else -1,
    lambda x1, _: -1 if x1 > -0.5 else 1,
    lambda x1, _: 1 if x1 > 0.5 else -1,
    lambda x1, _: -1 if x1 > 0.5 else 1,
    lambda _, x2: 1 if x2 > -0.5 else -1,
    lambda _, x2: -1 if x2 > -0.5 else 1,
    lambda _, x2: 1 if x2 > 0.5 else -1,
    lambda _, x2: -1 if x2 > 0.5 else 1,
]


def error(fn, data, dist):
    """Returns the error of fn on the weighted Xs."""
    # Weights sum up to 1.
    return 1 - sum([w for (x, y), w in zip(data, dist) if fn(*x) * y > 0])


def weak_learner(data, dist=None):
    """A weak learner which is not able to separate the XOR dataset.

    This weak learner is simply demonstrative and anything could be swaped out
    here (e.g. trees, DNNs, SVMs, etc.).

    Args:
      data: The training data.
      dist: A distribution for the training data, i.e. the weights per example.

    Returns:
      The basis function with the lowest eror. If multiple functions have the
      same error, one is randomly selected.
    """
    dist = dist or [1 / len(data) for _ in range(len(data))]
    fns_and_errs = [(fn, error(fn, data, dist)) for fn in basis_functions]
    fns_and_errs = sorted(fns_and_errs, key=lambda x: x[1])
    min_err = fns_and_errs[0][1]
    result = [fn for fn, err in fns_and_errs if err <= min_err]
    return random.choice(result), min_err


def adaboost(data, learner, iterations):
    """The AdaBoost algorithm.

    Args:
      data: The training examples.
      learner: A model to be fit on the data.
      iterations: Number of boosting iterations.

    Returns:
      The ensemble of boosted learners.
    """
    dist = [1 / len(data) for _ in range(len(data))]
    hs = []
    for _ in range(iterations):
        h, err = learner(data, dist)
        if err > 0.5:
            break
        hw = 0.5 * math.log((1 - err) / err)
        hs.append((h, hw))
        new_dist = []
        for (x, y), w in zip(data, dist):
            new_dist.append(w * math.exp(-hw * y * h(*x)))
        total = sum(new_dist)
        new_dist = [w / total for w in new_dist]
        dist = new_dist

    return lambda x1, x2: np.sign(sum([w * h(x1, x2) for h, w in hs]))


if __name__ == '__main__':
    boosted_learner = adaboost(xor, weak_learner, 5)
    acc = sum([1 for x, y in xor if boosted_learner(*x) * y > 0]) / len(xor)
    print('After boosting, the accuracy is:', acc)
