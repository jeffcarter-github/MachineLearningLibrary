import numpy as np


def identity(z):
    """Identity function...
    Args:
        z (np.array)

    Returns:
        f(z) = z (np.array)
    """
    return z


def dfdz_identity(z):
    """Derivative of the Identity function...
    Args:
        z (np.array)

    Returns:
        df(z)/dz = 1.0 (np.array)
    """
    return np.ones_like(z)


def sigmoid(z):
    """Sigmoid function...
    Args:
        z (np.array)

    Returns:
        f(z) = 1 / (1 + exp(-z)) (np.array)
    """
    return 1.0 / (1.0 + np.exp(-z))


def dfdz_sigmoid(z):
    """Derivative of the Sigmoid function...
    Args:
        z (np.array)

    Returns:
        df(z)/dz = f(z) * (1 - f(z)) (np.array)
    """
    return sigmoid(z) * (1.0 - sigmoid(z))


def logistic(z):
    """Logistic function...
    Args:
        z (np.array)

    Returns:
        f(z) = 1 / (1 + exp(-z)) (np.array)
    """
    return sigmoid(z)


def dfdz_logistic(z):
    """Derivative of the Logistic function...
    Args:
        z (np.array)

    Returns:
        df(z)/dz = f(z) * (1 - f(z)) (np.array)
    """
    return sigmoid(z) * (1.0 - sigmoid(z))


def tanh(z):
    """Hyperbolic tangent function...
    Args:
        z (np.array)

    Returns:
        f(z) = 2.0 / (1.0 + np.exp(-2.0 * z)) - 1.0 (np.array)
    """
    return 2.0 / (1.0 + np.exp(-2.0 * z)) - 1.0


def dfdz_tanh(z):
    """Derivative of the hyperbolic tangent function...
    Args:
        z (np.array)

    Returns:
        df(z)/dz = 1.0 - np.square(tanh(z)) (np.array)
    """
    return 1.0 - np.square(tanh(z))


def softsign(z):
    """Softsign function...
    Args:
        z (np.array)

    Returns:
        f(z) = z / (1.0 + np.abs(z)) (np.array)
    """
    return z / (1.0 + np.abs(z))


def dfdz_softsign(z):
    """Derivative of the softsign function...
    Args:
        z (np.array)

    Returns:
        df(z)/dz = None (np.array)
    """
    raise RuntimeError('not implemented...')


def ReLU(z):
    """Rectified linear unit function...
    Args:
        z (np.array)

    Returns:
        f(z) = np.max(0, z) (np.array)
    """
    return z * (z > 0)


def dfdz_ReLU(z):
    """Derivative of the rectified linear unit function...
    Args:
        z (np.array)

    Returns:
        df(z)/dz = 1 if x > 0 else 0 (np.array)
    """
    return (z > 0)


def LReLU(z):
    """Leaky rectified linear unit function...
    Args:
        z (np.array)

    Returns:
        f(z) = z if z > 0 else 0.01 * z (np.array)
    """
    return PReLU(z, 0.01)


def dfdz_LReLU(z):
    """Derivative of the leaky rectified linear unit function...
    Args:
        z (np.array)

    Returns:
        df(z)/dz = 1 if x > 0 else 0.01 (np.array)
    """
    return dfdz_PReLU(z, 0.01)


def PReLU(z, alpha):
    """Parametric rectified linear unit function...
    Args:
        z (np.array)

    Returns:
        f(z) = z if z > 0 else alpha * z (np.array)
    """
    return z * (z > 0) + alpha * z * (z <= 0)


def dfdz_PReLU(z, alpha):
    """Derivative of the parametric rectified linear unit function...
    Args:
        z (np.array)

    Returns:
        df(z)/dz = 1 if x > 0 else alpha (np.array)
    """
    return 1.0 * (z > 0) + alpha * (z <= 0)
