import numpy as np

from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import KFold

class Dataset:
    def __init__(self, *args, **kwargs):
        self.data_arrays = self.generate()

    def generate(self):
        raise NotImplementedError

    def train_test_split(self, testSize = 0.3, randomState = 0):
        return tts(*self.data_arrays, test_size=testSize, random_state=randomState)

    def generate_small(self, folds):
        X_tr_sets, X_te_sets = [], []
        y_tr_sets, y_te_sets = [], []
        kf = KFold(n_splits = folds, shuffle = True, random_state = 0)
        for train, test in kf.split(self.data_arrays[0]):
            X_tr_sets.append(self.data_arrays[0][train])
            X_te_sets.append(self.data_arrays[0][test])
            y_tr_sets.append(self.data_arrays[1][train])
            y_te_sets.append(self.data_arrays[1][test])
        return X_tr_sets, X_te_sets, y_tr_sets, y_te_sets

class CannabisGenotype2(Dataset):
    def generate(self):
        X, y = [], []
        with open("data/labeledGenotype.csv", "r") as data:
            head = True
            for line in data:
                if not head:
                    seq = line.split(',')
                    y.append(int(seq[3]))
                    X.append([int(x.strip()) for x in seq[4:]])
                else:
                    head = False
        X, y = np.array(X), np.array(y)
        return X, y

class CannabisGenotype(Dataset):
    def generate(self):
        X, y = [], []
        with open("data/labeledGenotype.csv", "r") as data:
            head = True
            for line in data:
                if not head:
                    seq = line.split(',')
                    y.append(int(seq[3]))
                    x = []
                    for i in range(4, len(seq), 2):
                        x.append((seq[i].strip(), seq[i+1].strip()))
                    X.append(x)
                else:
                    head = False
        X, y = np.array(X), np.array(y)
        return X, y

class CannabisOneHot(Dataset):
    def generate(self):
        X, y = [], []
        with open("data/cannabis_one_hot.csv", "r") as data:
            head = True
            for line in data:
                if not head:
                    seq = line.split(',')
                    y.append(int(seq[-2]))
                    X.append([float(x.strip()) for x in seq[1:-3]])
                else:
                    head = False
        X, y = np.array(X), np.array(y)
        return X, y

class CannabisDummies(Dataset):
    def generate(self):
        X, y = [], []
        with open("data/cannabis_genotype_dummies.csv", "r") as data:
            head = True
            for line in data:
                if not head:
                    seq = line.split(',')
                    y.append(int(seq[1]))
                    X.append([float(x.strip()) for x in seq[3:]])
                else:
                    head = False
        X, y = np.array(X), np.array(y)
        return X, y
