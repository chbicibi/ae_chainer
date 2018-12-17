
import argparse
import os
import shutil
import sys
from datetime import date
from functools import reduce

import chainer
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as plc
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition

import dataset


################################################################################

class TestDataset(object):
    def __init__(self, it):
        self.it = it
        self._len = len(it) // 4
        self._data = [None] * self._len

    def __len__(self):
        return self._len

    def __getitem__(self, ix):
        if ix >= self._len:
            raise IndexError
        i = ix // 4
        if self._data[i] is None:
            self._data[i] = self.make_data(self.it[i])
        return self._data[i]

    def make_data(self, data_org):
        ''' data_org => (u, v, _, _, _)
        '''
        a = data_org[::4, ::4, 0:2] # 間引き, uのみ
        return a
        # return np.transpose(a, (2, 0, 1))


################################################################################

def plot3d_scatter(X, Y, Z):
  fig = plt.figure()
  fig.subplots_adjust(left=0.15, bottom=0.1, right=0.7, top=0.95, wspace=0.1, hspace=0.2)
  ax = Axes3D(fig)
  # ax.set_xlabel(lx, size=14)
  # ax.set_ylabel(ly, size=14)
  # ax.set_zlabel(lz, labelpad=30, size=14)
  ax.tick_params(labelsize=14)
  ax.tick_params(axis='z', pad=20)

  # plt.gca().zaxis.set_tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True)
  # ax.set_aspect(0.2)

  ax.scatter(X, Y, Z, cmap='bwr', vmin=Z.min(), vmax=Z.max())
  # ax.contour(X, Y, Z, zdir='z', offset=np.min(Z))
  plt.show()
  # ax.imwrite("out.png")


################################################################################

def __test__():
    data = dataset.CFDBase('wing_00', size=2000) # (128, 256)
    train_val = TestDataset(data)
    data_size = len(train_val)

    n_components = 3

    # X = np.random.rand(100, 1000)
    # for i in range(X.shape[0]):
    # for j in range(X.shape[1]):
    #   X[i, j] = X[i, j] + np.cos(0.1 * i) + np.cos(0.1 * j + 1) + (i ** 2 + j ** 2) / 1000000
    # pca = decomposition.PCA(n_components=5)
    # pca.fit(X)
    # components = pca.components_         # => (n_components, n_features)
    # code = pca.transform(X)              # => (n_samples, n_components)
    # output = pca.inverse_transform(code) # => (n_samples, n_features)

    X = np.array(train_val) # => (500, 128, 256)
    Xs = X.reshape((data_size, -1)) # => (500, 128*256)
    print(X.shape)

    pca = decomposition.PCA(n_components=n_components)
    codes = pca.fit_transform(Xs)
    var_ratio = pca.explained_variance_ratio_
    components = pca.components_

    print(codes.shape)
    print(codes[0])

    if n_components == 2:
        plt.scatter(*codes.T)
        plt.show()
    else:
        plot3d_scatter(*codes.T)


    # a = np.array(memo)

    # print(a.shape)

    # flow = a[0, :, :, 0:2] # (u, v)

    # fig, ax = plt.subplots()
    # ax.imshow(flow[:, :, 0])
    # plt.show()


def get_args():
    '''
    docstring for get_args.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('out', nargs='?', default='new_script',
                        help='Filename of the new script')
    parser.add_argument('--force', '-f', action='store_true',
                        help='Force')
    parser.add_argument('--test', '-t', action='store_true',
                        help='Run as test mode')
    args = parser.parse_args()
    return args


def main():
    '''
    docstring for main.
    '''
    args = get_args()

    if args.test:
        __test__()
        return

    file = args.out

    if not os.path.splitext(file)[1] == '.py':
        file = file + '.py'

    if not args.force and os.path.exists(file):
        return

    shutil.copy(__file__, file)
    print('create:', file)


if __name__ == '__main__':
    main()
