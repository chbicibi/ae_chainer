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
import chainer.functions as F
from chainer.iterators import SerialIterator

import myutils as ut

import common as C_
import dataset as D_
import net_vae as NV_
import model as M_
import vis as V_
import main as ms


# パス
SRC_DIR = os.path.dirname(__file__)
SRC_FILE = os.path.basename(__file__)
SRC_FILENAME = os.path.splitext(SRC_FILE)[0]


################################################################################

def process3(keys, modelname, out):
    ''' モデル読み出し+可視化 '''

    file = ms.check_snapshot(out)
    N = 20

    # 学習データ作成
    # train_data_00 = D_.MapChain(ms.crop_center_sq, ms.get_it(10)('wing_00'))
    # train_data_10 = D_.MapChain(ms.crop_center_sq, ms.get_it(20)('wing_10'))
    # train_data_20 = D_.MapChain(ms.crop_center_sq, ms.get_it(N)('wing_20'))
    train_data_30 = D_.MapChain(ms.crop_center_sq, ms.get_it(N)('wing_30'))
    train_data = train_data_30

    # モデル読み込み
    sample = train_data[:1]
    model = M_.get_model(modelname, sample=sample)
    chainer.serializers.load_npz(file, model, path='updater/model:main/')
    model.to_cpu()

    if False:
        V_.show_frame(ms.vorticity(train_data[0]))
        return

    # エンコード
    def enc(x, n=None):
        z = chainer.Variable(model.xp.asarray(x))
        for i, chain in enumerate(model.predictor):
            if i == n:
                break
            z = chain.encode(z)
        return z

    x = train_data[:1]
    V_.show_frame(x[0], exf=ms.vorticity)
    return

    z = enc(x, 0)
    V_.show_frame(z.array[0])
    return

    # データセット
    X_00 = train_data_00[10:11]
    X_10 = train_data_10[15:16]
    X_20 = train_data_20[30:31]
    X_30 = train_data_30[0:1]
    X = X_10, X_30
    with chainer.using_config('train', False), chainer.no_backprop_mode():
        Z = [model.predictor.encode(model.xp.asarray(x), inference=True)
             for x in X]

    def print_z(z):
        print(*(f'{s:.3f}' for s in z.array.flatten()))
        return z

    fig, ax = plt.subplots()
    def plot_z(z=None, l=[]):
        if z is None:
            ax.plot(np.array(l))
            fig.savefig(f'z_log_man_{modelname}.png')
            return
        ax.cla()
        l.append(z.array.flatten())
        ax.plot(np.array(l))
        return z

    for z in Z:
        print_z(z)

    # モデル適用
    def apply_model(z):
        y = model.predictor.decode(z, inference=True)
        if isinstance(model, NV_.VAELoss):
            y = F.sigmoid(y)
        return y

    it_z = (Z[0]*(1-i)+Z[1]*i for i in np.arange(0, 1, 0.01))
    it_zp = map(plot_z, it_z)
    it_decode = map(apply_model, it_zp)

    colors = [(0, 'red'), (0.5, 'black'), (1, 'green')]
    cmap = plc.LinearSegmentedColormap.from_list('custom_cmap', colors)
    def plot_vor(frame):
        vor = vorticity(frame)
        def f_(ax):
            ax.imshow(vor, cmap=cmap, vmin=-0.07, vmax=0.07)
        return f_

    # 結果データ取得
    it_result = map(lambda v: v.array[0], it_decode)
    it_add_vor = map(lambda v: (*v, vorticity(v)), it_result)

    # # 入力データと復号データを合成
    it_zip = zip(loop(X[0][0]), loop(X[1][0]), it_result)
    it_zip_with_vor = map(lambda vs: [(*v, plot_vor(v)) for v in vs], it_zip)

    with chainer.using_config('train', False), chainer.no_backprop_mode():
        V_.show_chainer_NrNc(it_zip_with_vor, nrows=3, ncols=3, direction='tb')
    plot_z()


def task3(*args, **kwargs):
    ''' task2: VAEのz入力から復元の可視化 '''

    print(sys._getframe().f_code.co_name)

    # keys = 'plate_10', 'wing_00', 'plate_20', 'wing_15', 'plate_30', 'wing_05'
    keys = 'wing_00',
    name = kwargs.get('case', 'case4_0')
    out = f'__result__/{name}'

    if kwargs.get('check_snapshot'):
        check_snapshot(out, show=True)
        return

    process3(keys, name, out)


################################################################################

def __test__():
    with ut.chdir(f'{SRC_DIR}/__result__/case4_2'):
        path = ut.select_file('.', key=r'res_.*')
        with ut.chdir(path):
            log = ut.load('log.json', from_json=True)
            loss = [l['main/loss'] for l in log]
            plt.ylim((180000, 210000))
            plt.plot(np.array(loss))
            plt.show()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs='?', default='',
                        choices=['', '3', '4', '5'],
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
