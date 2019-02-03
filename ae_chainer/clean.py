import argparse
import glob
import os
import re
import shutil
import sys

import myutils as ut


# パス
SRC_DIR = os.path.dirname(__file__)
SRC_FILE = os.path.basename(__file__)
SRC_FILENAME = os.path.splitext(SRC_FILE)[0]


################################################################################

def thin_snapshot(snapshots, dirname=''):
    size = 0
    total = 0
    for snapshot in snapshots:
        m = re.search(r'(?<=epoch-)\d+', snapshot)
        if not m:
            continue
        epoch = int(m[0])
        if epoch < 10 or epoch % 10 == 0:
            continue
        # print(snapshot, '=> Remove', end='\r')
        size += ut.filesize(snapshot)
        total += 1
        os.remove(snapshot)
    if total:
        print(dirname, 'total:', total)
    return size


def remove_samll_dirs():
    with ut.chdir(SRC_DIR):
        dirs = ut.globm('**/res_*')
        size = 0
        for d in dirs:
            dirname = os.path.abspath(d)
            dirsize = ut.filesize(dirname)
            snapshots = [f for f in os.listdir(dirname)
                         if f.startswith('snapshot_')]
            print(d, f'snapshots={len(snapshots)}', end='')

            if len(snapshots) < 1 and ut.filesize(d) < 100 * ut.MB1:
                print(' => Remove')
                shutil.rmtree(dirname)
                size += dirsize
            else:
                print('\r')
                with ut.chdir(dirname):
                    size += thin_snapshot(snapshots, d)
        print(f'free: {size/1048576:.2f}MB')


################################################################################

def get_args():
    parser = argparse.ArgumentParser()

    # test option
    parser.add_argument('--test', '-t', action='store_true',
                        help='Run as test mode')
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    if args.test:
        __test__()

    else:
        remove_samll_dirs()


if __name__ == '__main__':
    sys.exit(main())
