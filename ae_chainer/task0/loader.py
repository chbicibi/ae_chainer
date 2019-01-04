#! /usr/bin/env python3

'''
学習に使うデータを取得する
'''

import argparse


################################################################################

def get_args():
    parser = argparse.ArgumentParser()
    '''
    Add your args.
    '''
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    '''
    Add main programs.
    '''


if __name__ == '__main__':
    main()
