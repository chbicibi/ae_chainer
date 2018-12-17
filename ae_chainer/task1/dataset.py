#! /usr/bin/env python3

'''
Abstruct
'''

import argparse
import configparser
import os
import shutil

import numpy as np
import matplotlib.pyplot as plt

import myutils as ut


SRC_DIR = os.path.dirname(__file__)
LOCAL_DIR = os.path.join(SRC_DIR, '../../__local__')
# PATH_INI = os.path.join(LOCAL_DIR, 'path.ini')
PATH_INI = os.path.join(SRC_DIR, 'path1.ini')


def get_datapath(key=None):
    config = configparser.ConfigParser()
    config.read(PATH_INI)
    if key:
        if key not in config['DATASET1']:
            raise KeyError
        return config['DATASET1'][key]
    keys = list(config['DATASET1'])
    for i, k in enumerate(keys):
        print(f'[{i}] {k}')
    inp = input()
    if not inp:
        return
    k = keys[int(inp)]
    return config['DATASET1'][k]


def get_filenames(path):
    with ut.chdir(path):
        a = ut.iglobm('out_*.npy')
        a = ut.fsort(a)
        a = [os.path.abspath(f) for f in a]
    return a


class CFDBase(object):
    def __init__(self, key=None, grid_path=None, cache=False, size=None):
        datapath = get_datapath(key)
        files = get_filenames(datapath)
        self.files = files
        self.len = len(files)
        if size and size < self.len:
            self.len = size
        if cache:
            self.data = [None] * self.len
        else:
            self.data = None

        if not grid_path:
            grid_path = os.path.join(datapath, '../grid.csv')
        self.grid_path = grid_path
        self.grid = None

    def __len__(self):
        return self.len

    def __getitem__(self, key):
        if key >= self.len:
            raise IndexError
        if not self.data:
            return np.load(self.files[key])
        if self.data[key] is None:
            self.data[key] = np.load(self.files[key])
        return self.data[key] # (u, v, p, f, w)

    def get_grid(self, dtype=np.float32):
        if not self.grid:
            self.grid = np.loadtxt(self.grid_path, delimiter=',', dtype=dtype)
        return self.grid # (height, width), data = 0 or 1


class MemoList(object):
    def __init__(self, it, fn, cache_path=None):
        self.it = it
        self.fn = fn
        self.data = [None] * len(it)
        if cache_path:
            os.makedirs(cache_path, exist_ok=True)
            self.cache_path = cache_path

    def __len__(self):
        return len(self.it)

    def __getitem__(self, key):
        if self.data[key] is None:
            self.data[key] = self.fn(self.it[key])
        return self.data[key]


################################################################################

def __test__():
    cfd_data = CFDBase('plate_00')
    memo = MemoList(cfd_data, fn=lambda x: x)

    a = np.array(memo)

    print(a.shape)

    flow = a[0, :, :, 0:2] # (u, v)

    fig, ax = plt.subplots()
    ax.imshow(flow[:, :, 0])
    plt.show()


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
