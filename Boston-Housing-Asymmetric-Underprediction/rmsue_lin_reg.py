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
def compute_L(yhat, y):
    '''
        Compute the (modified) mean squared error:
            L = (1/2) * mean( max(y - yhat, 0)^2 )
    '''
    u = np.maximum(y - yhat, 0.0)   # under-prediction per sample
    L = np.mean(u**2) / 2.0
    return L


# --------------------------
def compute_dL_dw(y, yhat, Phi):
    '''
        Compute gradient of L w.r.t. weights w.
    '''
    n = y.shape[0]

    # errors only when yhat < y
    errors = (yhat - y) * (yhat < y)

    dL_dw = np.dot(errors, Phi) / n
    return dL_dw


# --------------------------
def update_w(w, dL_dw, alpha=0.001):
    '''
        Performs one gradient-descent update:
            w := w - alpha * grad
    '''
    w = w - alpha * dL_dw
    return w


# --------------------------
def train(X, Y, alpha=0.001, n_epoch=100):
    '''
        Train the linear regression model using gradient descent.
        Automatically adds an intercept column so the model can shift predictions.
    '''
    # ---- ADD INTERCEPT COLUMN ----
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    bias = np.ones((X.shape[0], 1))
    X = np.hstack((bias, X))   # <---- ADD THIS

    # initialize weights (now p+1 parameters)
    w = np.zeros(X.shape[1])

    for _ in range(n_epoch):
        yhat = compute_yhat(X, w)
        dL_dw = compute_dL_dw(Y, yhat, X)
        w = update_w(w, dL_dw, alpha)

    return w
# --------------------------
