import numpy as np

from sklearn.model_selection import train_test_split as tts


class Dataset:
    def __init__(self, *args, **kwargs):
        self.data_arrays = self.generate()

    def generate(self):
        raise NotImplementedError

    def train_test_split(self, testSize = 0.3, randomState = 0):
        return tts(*self.data_arrays, test_size=testSize, random_state=randomState)

class CannabisGenotype(Dataset):
    def generate(self):
        X, y = [], []
        with open("data/cannabis_gen.csv", "r") as data:
            head = True
            for line in data:
                if not head:
                    seq = line.split(',')
                    y.append(int(seq[-1][1:-2]))
                    X.append([int(x.strip()) for x in seq[1:-1]])
                else:
                    head = False
        X, y = np.array(X), np.array(y)
        return X, y
