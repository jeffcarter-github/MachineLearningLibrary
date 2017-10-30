import matplotlib.pyplot as plt
import numpy as np
from ArtificialNeuralNetwork import ArtificialNeuralNetwork


def plot_gradient_descent_history(model, display=False):
    try:
        x = model.history
        lr = model.learning_rate

        plt.figure()
        plt.plot(x, 'r')
        plt.title('Gradient Descent, Learning Rate={0}'.format(lr))
        plt.xlabel('Optimization Step')
        plt.ylabel('Cross Entropy Cost')
        if display:
            plt.show()
    except AttributeError:
        return None


def plot_2D_decision_boundary(model, X, Y, display=False):
    x_min, x_max = X[0].min(), X[0].max()
    y_min, y_max = X[1].min(), X[1].max()

    n_points_per_dimension = 100
    step_x = (x_max - x_min) / n_points_per_dimension
    step_y = (x_max - x_min) / n_points_per_dimension

    xx, yy = np.meshgrid(np.arange(x_min, x_max, step_x),
                         np.arange(y_min, y_max, step_y))

    _X = np.array([xx.ravel(), yy.ravel()]).reshape(X.shape[0], -1)

    plt.figure()
    plt.contourf(xx, yy, model.predict(_X).reshape(xx.shape),
                 cmap=plt.cm.spectral, alpha=1.0)

    plt.scatter(X[0], X[1], c=Y, marker='o', alpha=500.0/X.shape[1])
    if display:
        plt.show()
