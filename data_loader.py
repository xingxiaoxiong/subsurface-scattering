import numpy as np
import math
import random
import os

from bin_viewer import read_bin

class Loader:

    def __init__(self, batch_size, split_fractions = [0.7, 0.3]):
        self.batch_size = batch_size

        self.ydata = np.load('./data/y.npy')
        self.nbatches = int(math.floor(self.ydata.shape[0] / batch_size))

        self.indices = [i for i in range(self.nbatches * batch_size)]
        random.shuffle(self.indices)

        self.indices = np.array(self.indices).reshape((-1, batch_size))

        self.ntrain = int(self.nbatches * split_fractions[0])
        self.nval = self.nbatches - self.ntrain

        self.split_sizes = [self.ntrain, self.nval]
        self.batch_ix = [0, 0]

        self.front_lit = np.load(os.path.join('./data', 'front_irradiance.npy'))
        self.back_lit = np.load(os.path.join('./data', 'back_irradiance.npy'))
        self.height = self.front_lit.shape[0]
        self.width = self.front_lit.shape[1]
        self.depth = 12

    def next_batch(self, split_index):
        index = self.batch_ix[split_index]
        if split_index == 1:
            index += self.ntrain

        indices = self.indices[index]
        y = self.ydata[indices, :]

        X = np.zeros(shape=(self.batch_size, self.height, self.width, self.depth), dtype=np.float32)
        for order, i in enumerate(indices):
            file_path = os.path.join('./data', '%05d.npy' % i)
            data = np.load(file_path)
            data = np.concatenate((data, self.front_lit, self.back_lit), axis=2)
            X[order] = data

        self.batch_ix[split_index] = (self.batch_ix[split_index] + 1) % self.split_sizes[split_index]
        return X, y


if __name__ == '__main__':
    loader = Loader(10)
    loader.next_batch(0)
    loader.next_batch(0)


