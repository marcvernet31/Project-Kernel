import numpy as np

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

def compare(array):
    return array[0] == array[1]


def combined(X, Y):
    gamma = 1
    x1, x2, x3 = X.shape
    y1, y2, y3 = Y.shape
    # Compute the kernel matrix:
    G = np.zeros((x1, y1))
    Ysame = np.apply_along_axis(compare, 2, Y)
    Yrev = np.apply_along_axis(np.flip, 2, Y)
    for i in range(x1):
        Xi = np.tile(X[i], (y1, 1, 1))
        # equality kernel
        Xisame = np.apply_along_axis(compare, 2, Xi)
        equality = 2 * np.sum(Xisame == Ysame, axis = 1) / x2
        # crossed SMC
        Xirev = np.apply_along_axis(np.flip, 2, Xi)
        crossedSMC = np.sum(np.all(Xirev==Yrev, axis = 2), axis = 1) / x2
        # Normal kernel or something
        Xi = np.all(Xi == Y, axis = 2)
        k1 = np.sum(Xi, axis=1)

        G[i, :] = 0.8 * k1 / x2 + 0.1 * equality + 0.1 * crossedSMC
    return G # postf f(G)
