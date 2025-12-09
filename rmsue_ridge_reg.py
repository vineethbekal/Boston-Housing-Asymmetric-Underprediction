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
    If X is missing intercept but w has one, it is added.
    """
    X = np.asarray(X)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # If X has one fewer column than w → add bias
    if X.shape[1] + 1 == w.shape[0]:
        X = add_intercept(X)

    return X @ w


def compute_asymmetric_loss(yhat, y, gamma=1.0):
    """
    Loss:
        if y > yhat:  gamma * (y - yhat)^2
        else:         1 * (y - yhat)^2
    """
    err = y - yhat
    loss = err**2
    loss[err > 0] *= gamma
    return 0.5 * np.mean(loss)


def compute_dL_dw(y, yhat, X, w, lambda_ridge, gamma):
    """
    Gradient of asymmetric squared error + ridge penalty.
    Ridge DOES NOT apply to bias term.
    """

    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n = X.shape[0]
    err = yhat - y

    # asymmetric penalty
    weights = np.ones_like(err)
    weights[y > yhat] = gamma

    grad = (weights * err) @ X / n

    # Ridge term
    ridge_grad = np.concatenate([[0.0], lambda_ridge * w[1:]])

    return grad + ridge_grad


def update_w(w, grad, alpha):
    return w - alpha * grad


def train_ridge(X, Y, alpha=0.001, n_epoch=100, lambda_ridge=0.0, gamma=1.0):
    """
    Trains a ridge-regularized regression model
    with asymmetric (MSUE-style) loss.

    Args:
        X : feature matrix
        Y : targets
        alpha : learning rate
        n_epoch : epochs
        lambda_ridge : L2 penalty strength
        gamma : under-prediction penalty multiplier
    """

    # Always add the intercept
    X = add_intercept(X)

    # Initialize weights
    w = np.zeros(X.shape[1])

    for _ in range(n_epoch):
        yhat = compute_yhat_ridge(X, w)
        grad = compute_dL_dw(Y, yhat, X, w, lambda_ridge, gamma)
        w = update_w(w, grad, alpha)

    return w
