import math
import numpy as np
# Note: please don't add any new package, you should solve this problem using only the packages above.

'''
    Problem 1: Linear Regression
    In this problem, you will implement the linear regression method based upon gradient descent.
    Xw  = y
    You could test the correctness of your code by typing `pytest -v test.py` in the terminal.
    Note: please don't use any existing package for linear regression problem, implement your own version.
'''

# --------------------------
def compute_Phi(x, p):
    '''
        Compute the feature matrix Phi of x. We will construct p polynomials.
        The features of each sample are x^0, x^1, ..., x^(p-1)

        Input:
            x : numpy vector of shape (n,)
            p : number of polynomial features
        Output:
            Phi: numpy array of shape (n,p)
    '''
    Phi = []
    for i in range(p):
        Phi.append(np.power(x, i))

    return np.array(Phi).T


# --------------------------
def compute_yhat(Phi, w):
    # If shapes mismatched by 1, add bias column
    if Phi.shape[1] + 1 == w.shape[0]:
        Phi = np.hstack((np.ones((Phi.shape[0], 1)), Phi))
    return Phi @ w


# --------------------------
def compute_L(yhat, y, gamma=1.0):
    """
    Asymmetric squared error:
      - Under-prediction (y > yhat): gamma * (y - yhat)^2
      - Over-prediction:             1 * (y - yhat)^2
    """
    err = y - yhat
    under = (err > 0)               # boolean mask

    weighted_sq = err**2
    weighted_sq[under] *= gamma     # apply penalty only to under-prediction

    return 0.5 * np.mean(weighted_sq)



# --------------------------
def compute_dL_dw(y, yhat, Phi, gamma=1.0):
    n = y.shape[0]
    err = yhat - y

    weights = np.ones_like(err)
    weights[y > yhat] = gamma   # under-prediction penalty

    grad = (weights * err) @ Phi / n
    return grad


# --------------------------
def update_w(w, dL_dw, alpha=0.001):
    '''
        Performs one gradient-descent update:
            w := w - alpha * grad
    '''
    w = w - alpha * dL_dw
    return w


# --------------------------
def train(X, Y, alpha=0.001, n_epoch=100, gamma=1.0):
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    bias = np.ones((X.shape[0], 1))
    X = np.hstack((bias, X))

    w = np.zeros(X.shape[1])

    for _ in range(n_epoch):
        yhat = compute_yhat(X, w)
        dL_dw = compute_dL_dw(Y, yhat, X, gamma=gamma)
        w = update_w(w, dL_dw, alpha)

    return w
# --------------------------
