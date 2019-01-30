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

def process4(casename, out):
    ''' モデル読み出し+可視化 '''

    file = ms.check_snapshot(out)
    N = 30

    # 学習データ作成
    # train_data_p10 = D_.MapChain(ms.crop_front_sq, ms.get_it(N)('plate_10'))
    # train_data_00 = D_.MapChain(ms.crop_center_sq, ms.get_it(10)('wing_00'))
    train_data_10 = D_.MapChain(ms.crop_front_sq, ms.get_it(N)('wing_10'))
    train_data_20 = D_.MapChain(ms.crop_front_sq, ms.get_it(N)('wing_20'))
    train_data_30 = D_.MapChain(ms.crop_front_sq, ms.get_it(N)('wing_30'))
    # train_data_30c = D_.MapChain(ms.crop_center_sq, ms.get_it(N)('wing_30'))
    train_data = train_data_10

    # モデル読み込み
    sample = train_data[:1]
    model = M_.get_model(casename, sample=sample)
    chainer.serializers.load_npz(file, model, path='updater/model:main/')
    model.to_cpu()

    if False:
        V_.show_frame(ms.vorticity(train_data[0]))
        return

    def make_plot_z():
        fig, ax = plt.subplots()
        l = []
        def plot_z(z=None):
            if z is None:
                if not l:
                    return
                p = ax.plot(np.array(l))
                fig.legend(p, list(map(str, range(64))))
                fig.savefig(f'z_test.png')
                return
            ax.cla()
            l.append(z.array.flatten())
            ax.plot(np.array(l))
            return z
        return fig, ax, plot_z

    def apply(x):
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            return model.predict(x, inference=True)

    def enc(x, n=None):
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            return model.encode(x, inference=True)

    def dec(z, n=None):
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            return model.decode(z, inference=True)

    x0 = model.xp.asarray(train_data_10[:1])
    x1 = model.xp.asarray(train_data_10[15:16]) # 移動
    x2 = model.xp.asarray(train_data_20[15:16]) # 回転, 拡大
    x3 = model.xp.asarray(train_data_30[15:16]) # 回転, 拡大
    # x2c = model.xp.asarray(train_data_30c[15:16]) # 回転, 拡大, 移動
    # x1 = model.xp.asarray(train_data_p10[:1])

    z0 = enc(x0)
    z1 = enc(x1)
    z2 = enc(x2)
    z3 = enc(x3)

    exf = ms.vorticity

    with ut.chdir('__img__'):
        V_.show_frame(x0[0], exf=exf, file=f'src_x0.png')
        V_.show_frame(x3[0], exf=exf, file=f'src_x1.png')
        # return

        fig, ax, plot_z = make_plot_z()
        # plot_z(z0)
        # plot_z(z2)
        # plot_z(z3)
        # plot_z(z2*2-z0)
        # plot_z(z3-z2*2+z0)
        # # plot_z(z3-z2+z1)
        # plot_z()
        # return

        for i in range(0, 21):
            t = i / 20
            print(f't={t}')
            # z = (1 - t) * z0 + t * z1 # 渦移動
            z = (1 - t) * z0 + t * z3 # 翼回転
            # z = 0.5 * (1 - t) * (z0 + z3) + t * z2 # 中間
            # if i == 41:
            #     z = z3
            # else:
            # z = z0 + (z2c - z2) * t
            # z.array[0, :48] = 0.0
            # z = 0 * z0 + np.random.rand(1, 64)
            plot_z(z)
            y = dec(z)
            # plot_data = y.array[0]
            plot_data = np.concatenate([x0, y.array, x3])
            V_.show_frame(plot_data, exf=exf,
                          file=f'recon_{casename}_{i:02d}.png')
            # if t < 0:
            #     V_.show_frame(y.array[0], exf=ms.vorticity,
            # file=f'recon_t=n_{1+t:.1f}.png')
            # else:
            #     V_.show_frame(y.array[0], exf=ms.vorticity,
            # file=f'recon_t=p_{t:.1f}.png')
            plt.pause(0.001)
        plot_z()


def process4_1(casename, out):
    ''' サンプルデータアニメーション '''

    # file = ms.check_snapshot(out)
    N = 300

    # 学習データ作成
    # train_data_p10 = D_.MapChain(ms.crop_front_sq, ms.get_it(N)('plate_10'))
    # train_data_00 = D_.MapChain(ms.crop_center_sq, ms.get_it(10)('wing_00'))
    # train_data_10 = D_.MapChain(ms.crop_front_sq, ms.get_it(N)('wing_10'))
    # train_data_20 = D_.MapChain(ms.crop_front_sq, ms.get_it(N)('wing_20'))
    train_data_30 = D_.MapChain(ms.crop_front_sq, ms.get_it(N)('wing_30'))
    # train_data_30c = D_.MapChain(ms.crop_center_sq, ms.get_it(N)('wing_30'))

    fig, axes = plt.subplots(nrows=1, ncols=train_data_30[0].shape[0]+1)
    anim = A_.Animation(fig, frames=N)

    def plot_(i):
        [ax.cla() for ax in axes]
        frame = train_data_30[i]
        data = map(lambda f, d: lambda ax: f(d, ax),
                   (V_.plot_vel, V_.plot_vel, V_.plot_gray, V_.plot_vor),
                   (*frame, ms.vorticity(frame)))
        V_.show_frame_m(data, fig, axes, file=False)

    anim(plot_, file='anim.mp4')


def task4(*args, **kwargs):
    casename = kwargs.get('case') or 'case9_0'
    out = f'__result__/{casename}'

    if kwargs.get('check_snapshot'):
        check_snapshot(out, show=True)
        return

    process4(casename, out)


################################################################################

def __test__():
    pass


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs='?', default='4',
                        choices=['', '4'],
                        help='Number of main procedure')
    parser.add_argument('--case', '-c', default='',
                        help='Training case name')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
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
        __test__()

    elif args.mode in '0123456789':
        taskname = 'task' + args.mode
        if taskname in globals():
            f_ = globals().get(taskname)
            with ut.stopwatch(taskname) as sw:
                f_(**vars(args), sw=sw)


if __name__ == '__main__':
    sys.exit(main())
