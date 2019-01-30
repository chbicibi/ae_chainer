import argparse
import glob
import os
import shutil
import sys

import myutils as ut


def remove_samll_dirs():
    dirs = ut.globm('**/res_*')
    for d in dirs:
        dirname = os.path.abspath(d)
        snapshots = [f for f in os.listdir(dirname) if 'snapshot_' in f]
        print(d, f'snapshots={len(snapshots)}', end='')

        if len(snapshots) < 10: # or ut.filesize(d) < 1048576:
            print('REMOVE')
            shutil.rmtree(dirname)
        else:
            print()


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
