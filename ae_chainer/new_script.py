#! /usr/bin/env python3

'''
Abstruct
'''

import argparse
import configparser
import os
import shutil

import numpy as np

import myutils as ut


SRC_DIR = os.path.dirname(__file__)
LOCAL_DIR = os.path.join(SRC_DIR, '../../__local__')
PATH_INI = os.path.join(LOCAL_DIR, 'path.ini')


def get_datapath(key=None):
    config = configparser.ConfigParser()
    config.read(PATH_INI)
    if key:
        if key not in config['DATASET']:
            raise Error
        return config['DATASET'][key]
    keys = list(config['DATASET'])
    for i, k in enumerate(keys):
        print(f'[{i}] {k}')
    inp = input()
    if not inp:
        return
    k = keys[int(inp)]
    return config['DATASET'][k]


def get_filenames(path):
    with ut.chdir(path):
        a = ut.iglobm('out_*.npy')
        a = ut.fsort(a)
        a = [os.path.abspath(f) for f in a]
    return a


class Grids(object):
    def __init__(self, key=None):
        datapath = get_datapath(key)
        files = get_filenames(datapath)
        self.files = files
        self.data = [None] * len(files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, key):
        if self.data[key] is None:
            self.data[key] = np.load(self.files[key])
        return self.data[key]


################################################################################

def __test__():
    with open('test.txt', 'w') as f:
        print(f.encoding)
    grids = Grids()
    grid = grids[0]
    print(grid.shape)


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
