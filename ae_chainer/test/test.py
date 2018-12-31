#! /usr/bin/env python3

'''
Abstruct
'''

import argparse
import os
import shutil

import numpy as np
import chainer


class Chain(chainer.Chain):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        print('forward')
        return x


################################################################################

def __main__():
    chain = Chain()
    x = np.array([1, 2, 3])
    print(chain.__dict__)
    print(chain(x))


################################################################################

def __test__():
    pass


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

    __main__()


if __name__ == '__main__':
    main()
