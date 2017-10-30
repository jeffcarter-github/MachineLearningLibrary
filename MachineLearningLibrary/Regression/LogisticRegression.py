import numpy as np


def logistic(X, W, b):
    Z = np.dot(X, W) + b
    return 1 / (1 + np.exp(-Z))


class LogisticRegression(object):
    def __init__(self, learning_rate, n_initerations=1000):
        self._b = None
        self._coeff = None
        self.learning_rate = learning_rate
        self.n_initerations = n_initerations
        self._cost = []
        self.bs = []

    def fit(self, X, y):
        y = np.copy(y)
        X = np.copy(X)

        self._b = np.zeros((1, 1))
        self._coeff = np.random.randn(X.shape[1], 1)
        for i in range(self.n_initerations):
            self.calculate_cost(X, y)
            dJdW, dJdb = self.calculate_gradient(X, y)
            self._b -= self.learning_rate * dJdb
            self._coeff -= self.learning_rate * dJdW

    def calculate_cost(self, X, y):
        '''calculates the negative log likelihood...'''
        A = logistic(X, self._coeff, self._b)

        self._cost.append(
            -1.0 * np.average(
                np.multiply(y, A) + np.multiply(1 - y, 1 - A)))

    def calculate_gradient(self, X, y):
        A = logistic(X, self._coeff, self._b)
        n_samples = X.shape[1]
        dJdW = np.dot(X.T, A - y) / n_samples
        dJdb = np.sum(A-y) / n_samples
        return dJdW, dJdb

    def predict(self, X):
        p = logistic(X, self._coeff, self._b)
        return p > 0.5

    def error(self, X, y):
        _y = self.predict(X)
        return np.sum(np.abs(_y - y)) / float(len(_y))


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    n_samples = 5000
    n_features = 2

    np.random.seed(0)
    X = np.random.randn(n_samples, n_features)
    W = np.random.randn(n_features, 1)
    b = 0.2

    y = 1 / (1 + np.exp(-(np.dot(X, W) + b)))
    y = y > 0.5 + 0.01 * np.random.randn(n_samples, 1)

    lr = LogisticRegression(0.01)
    lr.fit(X, y)

    print lr.error(X, y)