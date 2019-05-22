"""Implementation of a simple perceptron classifier.

To run:
    python3 perceptron.py
"""

import matplotlib.pyplot as plt
import numpy as np

class PerceptronClassifier(object):
    """A perceptron classifier.
    
    The perceptron update rule is extremely simple,
        if sign(w * x) != y:
            w += x * y
    where w is the weight vector, x is the feature vector, and y is the label.
    """

    def __init__(self):
        self._max_epochs = 100
        self._classified = None
        self._w = None

    def fit(self, X, y):
        self._classified = False
        X = np.insert(X, 0, 1, axis=1) # Prepend bias to examples.
        self._w = np.zeros_like(X[0])
        for _ in range(self._max_epochs):
            for x_, y_ in zip(X, y):
                if self.predict(x_) != y_:
                    self._w += x_ * y_
            if self._check_classified(X, y):
                self._classified = True
                return
        print("Could not learn a separator after 100 epochs. "
              "Is the data linearly seperable?")

    def _check_classified(self, X, y):
        for x_, y_ in zip(X, y):
            if self.predict(x_) != y_:
                return False
        return True

    def predict(self, x):
        return np.sign(np.dot(self._w, x))

    @property
    def weights(self):
        return self._w


def _generate_linearly_seperable_data():
    X = []
    y = []
    for i in range(100):
        X.append([np.random.normal(-1, .3), np.random.uniform(0, 10)])
        y.append(-1)
        X.append([np.random.normal(1, .3), np.random.uniform(0, 10)])
        y.append(1)
    return np.array(X), np.array(y)


if __name__ == '__main__':
    X, y = _generate_linearly_seperable_data()

    clf = PerceptronClassifier()
    clf.fit(X, y)

    # Plot the data points and decision boundary.
    plt.scatter(X[:,0], X[:,1], c=y)
    w = clf.weights
    w[0] += .000001  # In case bias == 0.
    a = - (w[0] / w[2]) / (w[0] / w[1])
    b = - w[0] / w[2]
    def line(x):
        return a * x + b  
    plt.plot([-1, 1], [line(-1), line(1)], c='red')
    plt.show()
