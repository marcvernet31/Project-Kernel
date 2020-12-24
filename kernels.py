import numpy as np

def overlap(X, Y):
    xm, xn = X.shape
    ym, yn = Y.shape
    # Compute the kernel matrix:
    G = np.zeros((xm, ym))
    for i in range(xm):
        Xi = np.tile(X[i], (ym, 1))
        Xi = Xi == Y
        G[i, :] = np.sum(Xi, axis=1)
    return G

def smc(X, Y):
    x1, x2, x3 = X.shape
    y1, y2, y3 = Y.shape
    G = np.zeros((x1, y1))
    for i in range(x1):
        Xi = np.tile(X[i], (y1, 1, 1))
        Xi = Xi == Y
        #  k([a1, a2], [b1, b2]) = a1==b1 and a2==b2
        # compares the whole microsatellite
        X1 = np.all(Xi, axis = 2)
        G[i, :] = np.sum(X1, axis=1) / x2
    return G


def jaccard(X, Y):
    xm, xn = X.shape
    ym, yn = Y.shape
    # Compute the kernel matrix:
    G = np.zeros((xm, ym))
    for i in range(xm):
        Xi = np.tile(X[i], (ym, 1))
        Xi = Xi == Y
        eq = np.sum(Xi, axis = 1)
        G[i, :] = eq / (2*xn-eq)
    return G # postf f(G)

def jaccard2(X, Y):
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

def k0(X, Y): # what is this?
    xm, xn = X.shape
    ym, yn = Y.shape
    # Compute the kernel matrix:
    G = np.zeros((xm, ym))
    for i in range(xm):
        Xi = np.tile(X[i], (ym, 1))
        Xi = Xi == Y
        G[i, :] = np.mean(Xi, axis=1)
    return G

def k0prime(X, Y):
    x1, x2, x3 = X.shape
    y1, y2, y3 = Y.shape
    G = np.zeros((x1, y1))
    for i in range(x1):
        Xi = np.tile(X[i], (y1, 1, 1))
        Xi = Xi == Y
        #  k([a1, a2], [b1, b2]) = a1==b1 + a2==b2
        X1 = np.all(Xi, axis = 2)
        G[i, :] = np.sum(X1, axis=1) / x2
    gamma = 1/4 # the hyper parameter shouldn't be written here
    return np.exp(gamma * G)

def count(X, Y):
    x1, x2, x3 = X.shape
    y1, y2, y3 = Y.shape
    G = np.zeros((x1, y1))
    for i in range(x1):
        Xi = np.tile(X[i], (y1, 1, 1))
        Xi = Xi == Y
        #  k([a1, a2], [b1, b2]) = a1==b1 + a2==b2
        X1 = np.sum(Xi, axis = 2)
        G[i, :] = np.sum(X1, axis=1) / x2
    return G
