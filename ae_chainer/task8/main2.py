import argparse
import glob
import itertools
import os
import shutil
import sys
import traceback
from operator import itemgetter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as plc

import chainer
import chainer.distributions as D
import chainer.functions as F
from chainer.iterators import SerialIterator

import myutils as ut

import common as C_
import dataset as D_
import net_vae as NV_
import model as M_
import vis as V_
import anim as A_
import main as ms


# パス
SRC_DIR = os.path.dirname(__file__)
SRC_FILE = os.path.basename(__file__)
SRC_FILENAME = os.path.splitext(SRC_FILE)[0]


################################################################################

def process5(modelname, out):
    ''' モデルの損失計算 '''

    file = ms.check_snapshot(out)
    N = 300

    # 学習データ作成
    train_data_00 = D_.MapChain(ms.crop_center_sq, ms.get_it(N)('wing_00'))
    train_data_10 = D_.MapChain(ms.crop_center_sq, ms.get_it(N)('wing_10'))
    train_data_20 = D_.MapChain(ms.crop_center_sq, ms.get_it(N)('wing_20'))
    train_data_30 = D_.MapChain(ms.crop_center_sq, ms.get_it(N)('wing_30'))
    train_data_05 = D_.MapChain(ms.crop_center_sq, ms.get_it(N)('wing_05'))
    train_data_15 = D_.MapChain(ms.crop_center_sq, ms.get_it(N)('wing_15'))
    train_data = train_data_10

    # モデル読み込み
    sample = train_data[:1]
    model = M_.get_model(modelname, sample=sample)
    chainer.serializers.load_npz(file, model, path='updater/model:main/')
    model.to_cpu()

    def calc_loss(x):
        xa = model.predictor.xp.asarray(x)
        y = model.predict(xa[None, ...])[0]
        loss = F.mean_squared_error(xa, y)
        return loss.array
        # return chainer.cuda.to_cpu(loss.array)

    loss_it = D_.MapChain(calc_loss, train_data_00, train_data_10,
                          train_data_20, train_data_30, train_data_05,
                          train_data_15)

    fig, ax = plt.subplots()
    ax.plot(np.array(loss_it))
    plt.show()


def task5(*args, **kwargs):
    ''' task5: モデルの損失計算 '''

    print(sys._getframe().f_code.co_name)

    modelname = kwargs.get('case') or 'case8_0'
    out = f'__result__/{modelname}'

    if kwargs.get('check_snapshot'):
        check_snapshot(out, show=True)
        return

    process5(modelname, out)


################################################################################

def __test__():
    pass


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs='?', default='',
                        choices=['', '5'],
                        help='Number of main procedure')
    parser.add_argument('--case', '-c', default='',
                        help='Training case name')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    # parser.add_argument('--clear', '-c', action='store_true',
    #                     help='Remove directory at the beginning')
    parser.add_argument('--no-progress', '-p', action='store_true',
                        help='Hide progress bar')

    # additional options
    parser.add_argument('--check-snapshot', '-s', action='store_true',
                        help='Print names in snapshot file')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='Resume with loading snapshot')

    # test option
    parser.add_argument('--test', '-t', action='store_true',
                        help='Run as test mode')
    args = parser.parse_args()
    return args


def main():
    global DEVICE, PROGRESSBAR

    args = get_args()

    # if args.gpu:
    #     C_.DEVICE = args.gpu

    C_.SHOW_PROGRESSBAR = not args.no_progress

    # out = args.out
    out = f'result/{SRC_FILENAME}'

    if args.test:
        # print(vars(args))
        __test__()

    elif args.mode in '0123456789':
        taskname = 'task' + args.mode
        if taskname in globals():
            f_ = globals().get(taskname)
            with ut.stopwatch(taskname) as sw:
                f_(**vars(args), sw=sw)


if __name__ == '__main__':
    sys.exit(main())
