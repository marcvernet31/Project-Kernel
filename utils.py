""" Utility functions to validate models. """
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
# from sklearn import datasets

import numpy as np

@timeit
def timed_k_fold_CV(model, X, y, folds):
    acc = np.mean(cross_val_score(model, X, y, cv=folds))
    # print("model with accuracy:", acc)
    return acc

def k_fold_CV(model, X, y, folds):
    acc = np.mean(cross_val_score(model, X, y, cv=folds))
    # print("model with accuracy:", acc)
    return acc

def test_small_datasets(dataset, my_kernel, n_folds = 5):
    Xtr, Xte, ytr, yte = dataset.generate_small(n_folds)
    model = svm.SVC(kernel = my_kernel, C=1)
    acc = []
    for i in range(n_folds):
        # model = svm.SVC(kernel = my_kernel, C=1) isok
        acc.append(model.fit(Xtr[i], ytr[i]).score(Xte[i], yte[i]))
    return np.mean(acc)

def test_small_datasets2(dataset, my_kernel, n_folds = 5):
    # the test sets will not be used
    Xtr, Xte, ytr, yte = dataset.generate_small(n_folds)
    model = svm.SVC(kernel = my_kernel, C=1)
    acc = []
    for i in range(n_folds):
        # model = svm.SVC(kernel = my_kernel, C=1) isok
        X_train, X_test, y_train, y_test = train_test_split(Xtr[i], ytr[i])
        acc.append(model.fit(X_train, y_train).score(X_test, y_test))
    return np.mean(acc)

def cross_val_small_datasets(dataset, my_kernel, n_folds=5):
    Xtr, Xte, ytr, yte = dataset.generate_small(n_folds)
    model = svm.SVC(kernel = my_kernel, C=1)
    acc = []
    for i in range(n_folds):
        # model = svm.SVC(kernel = my_kernel, C=1) isok
        acc.append(k_fold_CV(model, X, y, 3))
    return np.mean(acc)
