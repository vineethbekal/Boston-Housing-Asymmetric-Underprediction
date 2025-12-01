import numpy as np

def add_intercept(X):
    """
    Ensures X has a leading column of 1s as a bias column.
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    bias = np.ones((X.shape[0], 1))
    return np.hstack((bias, X))

def compute_Phi(x, p):
    """
    Builds polynomial feature matrix of degree p.
    """
    Phi = [np.power(x, i) for i in range(p)]
    return np.vstack(Phi).T


def compute_yhat_ridge(X, w):
    """
    Linear prediction y = Xw.
    If X is missing intercept but w has one, it is added automatically.
    """
    X = np.asarray(X)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # If X has one fewer column than w → add bias
    if X.shape[1] + 1 == w.shape[0]:
        X = add_intercept(X)

    return X @ w


def compute_msue(yhat, y):
    u = np.maximum(y - yhat, 0.0)
    return 0.5 * np.mean(u ** 2)


def compute_dL_dw(y, yhat, X, w, lambda_ridge):
    """
    Gradient of MSUE + Ridge penalty.
    MSUE applies only when yhat < y.
    Ridge does NOT apply to the bias term.
    """

    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    errors = (yhat - y) * (yhat < y)
    grad = (errors @ X) / X.shape[0]
    ridge_grad = np.concatenate([[0.0], lambda_ridge * w[1:]])

    return grad + ridge_grad


def update_w(w, grad, alpha):
    return w - alpha * grad


def train_ridge(X, Y, alpha=0.001, n_epoch=100, lambda_ridge=0.0):
    """
    Trains a ridge-regularized linear regression model
    using MSUE (asymmetric) loss.

    Args:
        X : input features
        Y : targets
        alpha : learning rate
        n_epoch : number of epochs
        lambda_ridge : ridge strength (L2)
    """

    # Always add the intercept
    X = add_intercept(X)

    # Initializing weights
    w = np.zeros(X.shape[1])

    for _ in range(n_epoch):
        yhat = compute_yhat_ridge(X, w)
        grad = compute_dL_dw(Y, yhat, X, w, lambda_ridge)
        w = update_w(w, grad, alpha)

    return w
