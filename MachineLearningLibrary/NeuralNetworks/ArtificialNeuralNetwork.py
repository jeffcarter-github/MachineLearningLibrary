import numpy as np
import NeuralNetworkUtilities as utilities


class BaseLayer(object):
    def __init__(self, n_nodes, activation, knockout_survial=None):
        # the number of nodes for this layer...
        self.n_nodes = n_nodes
        self.n_input_edges = None
        # the upstream layer, used for backpropagation...
        self.upstream_layer = None

        # non-linear activation function for each node...
        if activation in ['logistic', 'tanh', 'ReLU', 'LReLU']:
            if activation == 'logistic':
                self.func = utilities.logistic
                self.dfdz_func = utilities.dfdz_logistic

            elif activation == 'tanh':
                self.func = utilities.tanh
                self.dfdz_func = utilities.dfdz_tanh

            elif activation == 'ReLU':
                self.func = utilities.ReLU
                self.dfdz_func = utilities.dfdz_ReLU

            elif activation == 'LReLU':
                self.func = utilities.LReLU
                self.dfdz_func = utilities.dfdz_LReLU
        else:
            raise RuntimeError('%s not supported activation func' % activation)

        # survial rate for node values...
        self.knockout_survial = knockout_survial
        # learning rate for gradient decent...
        self.learning_rate = None

        # number of data sets in the training set...
        self.n_data_sets = None

        # parameters for each layer, i.e. vectorized for each node in layer
        self.W = None
        self.b = None
        self.A = None
        self.Z = None

        # filter for knockout regularization...
        self.F = None

        self.dW = None
        self.db = None
        self.dA = None
        self.dZ = None

    def initialize_layer(self, n_input_edges, n_data_sets, learning_rate, scalar=0.01):
        self.n_input_edges = n_input_edges
        self.n_data_sets = n_data_sets
        self.learning_rate = learning_rate

        self.W = np.random.randn(self.n_nodes, self.n_input_edges)
        self.b = np.zeros((self.n_nodes, 1))

    def set_upstream_layer(self, layer):
        self.upstream_layer = layer

    def calculate_Z(self):
        pass

    def calculate_A(self, prediction):
        self.A = self.func(self.Z)
        if self.knockout_survial and not prediction:
            self.F = np.random.random(size=(self.n_nodes, self.n_data_sets))
            self.F = self.F < self.knockout_survial
            self.A = np.multiply(self.A, self.F) / self.knockout_survial

    def calculate_gradients(self):
        pass

    def update_parameters(self, eta=None):
        self.W -= self.learning_rate * self.dW
        if eta:
            self.W = (1.0 - eta) * self.W
        self.b -= self.learning_rate * self.db


class InputLayer(BaseLayer):
    def __init__(self, n_nodes, activation, knockout_survial=None):
        super(InputLayer, self).__init__(n_nodes, activation, knockout_survial)

    def calculate_Z(self, X):
        self.Z = np.dot(self.W, X) + self.b

    def calculate_gradients(self, X):
        self.dZ = np.multiply(self.dA, self.dfdz_func(self.Z))
        self.dW = np.dot(self.dZ, X.transpose()) / self.n_data_sets
        # inner product with unity matrix is a sum over the rows (n,m) to (n,1)
        self.db = np.sum(self.dZ, axis=1, keepdims=True) / self.n_data_sets


class HiddenLayer(BaseLayer):
    def __init__(self, n_nodes, activation, knockout_survial=None):
        super(HiddenLayer, self).__init__(n_nodes, activation)

    def calculate_Z(self):
        self.Z = np.dot(self.W, self.upstream_layer.A) + self.b

    def calculate_gradients(self):
        self.dZ = np.multiply(self.dA, self.dfdz_func(self.Z))
        self.dW = np.dot(self.dZ, self.upstream_layer.A.transpose()) / self.n_data_sets
        # inner product with unity matrix is a sum over the rows (n,m) to (n,1)
        self.db = np.sum(self.dZ, axis=1, keepdims=True) / self.n_data_sets
        # inject the dA parameter into the upstream layer...
        self.upstream_layer.dA = np.dot(self.W.transpose(), self.dZ)


class LogisticOutputLayer(BaseLayer):
    def __init__(self):
        super(LogisticOutputLayer, self).__init__(1, 'logistic')

    def calculate_Z(self):
        self.Z = np.dot(self.W, self.upstream_layer.A) + self.b

    def calculate_gradients(self, Y):
        self.dA = -1.0 * (np.divide(Y, self.A) - np.divide((1 - Y), (1 - self.A)))
        self.dZ = np.multiply(self.dA, self.dfdz_func(self.Z))
        self.dW = np.dot(self.dZ, self.upstream_layer.A.transpose()) / self.n_data_sets
        # inner product with unity matrix is a sum over the rows (n,m) to (n,1)
        self.db = np.sum(self.dZ, axis=1, keepdims=True) / self.n_data_sets
        # inject the dA parameter into the upstream layer...
        self.upstream_layer.dA = np.dot(self.W.transpose(), self.dZ)


class ArtificialNeuralNetwork(object):
    def __init__(self, L2=None):
        self.L2 = L2
        self.learning_rate = None
        self.layers = []
        self.history = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def fit(self, X, Y, n_iter=1000, learning_rate=0.01):
        self.learning_rate = learning_rate
        self._configure_layers(X)
        for i in range(n_iter):
            self._feed_forward(X, prediction=False)
            self._calculate_cost(Y)
            self._backprop(X, Y)

    def predict(self, X):
        n, m = X.shape
        self._feed_forward(X, prediction=True)
        return (self.layers[-1].A > 0.5).reshape(1, m)

    def error(self, X, Y):
        P = self.predict(X)
        return np.mean(P - Y)

    def _configure_layers(self, X):
        n, m = X.shape
        for i, layer in enumerate(self.layers):
            if isinstance(layer, InputLayer):
                layer.initialize_layer(n, m, self.learning_rate)
            else:
                upstream = self.layers[i-1]
                layer.initialize_layer(upstream.n_nodes, m, self.learning_rate)
                layer.set_upstream_layer(upstream)

    def _feed_forward(self, X, prediction):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, InputLayer):
                layer.calculate_Z(X)
            else:
                layer.calculate_Z()
            layer.calculate_A(prediction)

    def _calculate_cost(self, Y):
        A = self.layers[-1].A
        n, m = Y.shape
        cost = np.sum(np.multiply(Y, np.log(A)))\
            + np.sum(np.multiply((1-Y), (np.log(1-A))))
        self.history.append(-1.0 * cost / m)

    def _backprop(self, X, Y):
        for i, layer in enumerate(reversed(self.layers)):
            if isinstance(layer, LogisticOutputLayer):
                layer.calculate_gradients(Y)
            elif isinstance(layer, InputLayer):
                layer.calculate_gradients(X)
            elif isinstance(layer, HiddenLayer):
                layer.calculate_gradients()
        for layer in self.layers:
            layer.update_parameters(self.L2)


if __name__ == '__main__':

    import classifier_utilities as clf_utils
    import matplotlib.pyplot as plt

    ## generate data
    d, m = 2, 1000
    X = np.random.randn(d, m)
    Y = (np.sqrt(np.square(X[0]) + np.square(X[1])) > 1.0).reshape(1, m)
    i = np.squeeze(Y)

    ## show data
    plt.figure()
    plt.plot(X[0][i], X[1][i],  'ob')
    plt.plot(X[0][~i], X[1][~i],  'or')
    # plt.show()

    ## create ANN...
    ann = ArtificialNeuralNetwork(L2=0.0)
    ann.add_layer(InputLayer(n_nodes=3, activation='tanh', knockout_survial=1.0))
    ann.add_layer(LogisticOutputLayer())

    ann.fit(X, Y, learning_rate=0.05, n_iter=9000)
    print 'training error: ', ann.error(X, Y)

    ## show results
    clf_utils.plot_gradient_descent_history(ann, display=False)
    clf_utils.plot_2D_decision_boundary(ann, X, Y, display=True)
