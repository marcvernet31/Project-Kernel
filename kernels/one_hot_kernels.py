import numpy as np

def smc(X, Y):
    xm, xn = X.shape
    ym, yn = Y.shape
    # Compute the kernel matrix:
    G = np.zeros((xm, ym))
    for i in range(xm):
        Xi = np.tile(X[i], (ym, 1))
        Xi = Xi == Y
        G[i, :] = np.sum(Xi, axis=1) / xn
    return G

def jaccard(X, Y):
    xm, xn = X.shape
    ym, yn = Y.shape
    # Compute the kernel matrix:
    G = np.zeros((xm, ym))
    ytraits = np.sum(Y, axis=1)
    for i in range(xm):
        Xi = np.tile(X[i], (ym, 1))
        xtraits = np.sum(Xi, axis = 1)
        Xi = Xi == Y
        eq = np.sum(Xi, axis = 1)
        G[i, :] = eq / (xtraits+ytraits-eq)
    return G

def k0prime(X, Y):
    xm, xn = X.shape
    ym, yn = Y.shape
    G = np.zeros((xm, ym))
    for i in range(xm):
        Xi = np.tile(X[i], (ym, 1))
        Xi = Xi == Y
        G[i, :] = np.mean(Xi, axis=1)
    gamma = 1
    return np.exp(gamma * G)
