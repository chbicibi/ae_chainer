import argparse
import glob
import itertools
import os
import shutil
import subprocess
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
import main as MS_


# パス
SRC_DIR = os.path.dirname(__file__)
SRC_FILE = os.path.basename(__file__)
SRC_FILENAME = os.path.splitext(SRC_FILE)[0]


################################################################################

def process4(casename, out):
    ''' モデル読み出し+可視化 '''

    file = MS_.check_snapshot(out)
    N = 500

    # 学習データ作成
    train_data_p10 = D_.MapChain(MS_.crop_front_sq, MS_.get_it(N)('plate_10'))
    # train_data_00 = D_.MapChain(MS_.crop_center_sq, MS_.get_it(10)('wing_00'))

    train_data_10 = D_.MapChain(MS_.crop_center_sq, MS_.get_it(N)('wing_10'))
    train_data_20 = D_.MapChain(MS_.crop_center_sq, MS_.get_it(N)('wing_20'))
    train_data_30 = D_.MapChain(MS_.crop_center_sq, MS_.get_it(N)('wing_30'))

    train_data_10b = D_.MapChain(MS_.crop_back_sq, MS_.get_it(N)('wing_10'))
    train_data_20b = D_.MapChain(MS_.crop_back_sq, MS_.get_it(N)('wing_20'))
    train_data_30b = D_.MapChain(MS_.crop_back_sq, MS_.get_it(N)('wing_30'))

    train_data = train_data_10

    # モデル読み込み
    sample = train_data[:1]
    model = M_.get_model(casename, sample=sample)
    chainer.serializers.load_npz(file, model, path='updater/model:main/')
    model.to_cpu()

    if False:
        V_.show_frame(MS_.vorticity(train_data[0]))
        return

    def make_plot_z():
        fig, ax = plt.subplots()
        l = []
        def plot_z(z=None, plot=False):
            if z is None:
                if not l:
                    return
                p = ax.plot(np.array(l))
                # fig.legend(p, list(map(str, range(64))))
                fig.savefig(f'z_test.png')
                return
            ax.cla()
            l.append(z.array.flatten())
            if plot:
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

    exf = MS_.vorticity

    x0 = model.xp.asarray(train_data_10[:1])
    x1 = model.xp.asarray(train_data_10[15:16]) # 移動
    x2_1 = model.xp.asarray(train_data_20[0:1]) # 回転, 拡大
    x2_2 = model.xp.asarray(train_data_20[15:16]) # 回転, 拡大
    x3 = model.xp.asarray(train_data_30[0:1]) # 回転, 拡大
    # x2c = model.xp.asarray(train_data_30c[15:16]) # 回転, 拡大, 移動
    x0p = model.xp.asarray(train_data_p10[:1])

    z0 = enc(x0)
    z1 = enc(x1)
    z2_1 = enc(x2_1)
    z2_2 = enc(x2_2)
    z3 = enc(x3)
    z0p = enc(x0p)

    ### Plot Case ###
    x_s0, x_s1 = x0, x1 # 渦移動1
    # x_s0, x_s1 = x2_1, x2_2 # 渦移動2
    # x_s0, x_s1 = x0, x3 # 翼回転
    # x_s0, x_s1 = x0, x2 # 翼回転2
    # x_s0, x_s1 = x0, x0p # 翼変形

    # z_s0, z_s1 = z0, z1 # 渦移動1
    # z_s0, z_s1 = z2_1, z2_2 # 渦移動2
    z_s0, z_s1 = z0, z3 # 翼回転
    # z_s0, z_s1 = z0, z2 # 翼回転2
    # z_s0, z_s1 = z0, z0p # 翼変形
    # z_s0, z_s1 = 0.5 * (z0 + z3), z2 # 中間
    z_diff = z1 - z0

    with ut.chdir('__img__'):
        V_.show_frame(x_s0[0], exf=exf, file=f'src_x0.png')
        V_.show_frame(x_s1[0], exf=exf, file=f'src_x1.png')
        # return

        fig, ax, plot_z = make_plot_z()

        if True:
            plot_z(z0)
            plot_z(z1)
            plot_z(z_diff)
            # plot_z(z2*2-z0)
            # plot_z(z3-z2*2+z0)
            # plot_z(z3-z2+z1)
            plot_z()
            return

        for i in range(0, 21):
            t = i / 20
            print(f't={t}', ' ' * 40, end='\r')

            # x = model.xp.asarray(train_data_30[i:i+1])
            x = model.xp.asarray(x3)
            z_base = enc(x)
            z = z_base + z_diff * 2
            # y_base = dec(z_base)
            y_diff = dec(z_diff)

            # y1 = dec(z_base)
            # y2 = dec(z)
            # plot_data = np.concatenate([x, y1.array, y2.array])

            # z = (1 - t) * z_s0 + t * z_s1
            y = dec(z)
            print('*')
            # plot_data = np.concatenate([x_s0, y.array, x_s1])

            plot_data = np.concatenate([x, y.array])

            plot_z(z)
            V_.show_frame(plot_data, exf=exf,
                          file=f'recon_{casename}_{i:02d}.png')
            # plt.pause(0.001)
            break
        plot_z()


def process4_1(casename, out):
    ''' サンプルデータアニメーション '''

    # file = MS_.check_snapshot(out)
    N = 100

    def sigmoid(x, a=1):
        return 1 / (1 + np.exp(-a * x))

    def logit(x, a=1):
        if a < 0:
            return -float('inf')
        elif a >= 1:
            return float('inf')
        return np.log(x / (1 - x)) / a

    # x = sigmoid(5.5, 2)
    # print(x)
    # y = logit(x, 2)
    # print(y)
    # exit()

    # 学習データ作成
    train_data_p10 = D_.MapChain(MS_.crop_front_sq, MS_.get_it(N)('plate_10'))
    train_data_00 = D_.MapChain(MS_.crop_front_sq, MS_.get_it(N)('wing_00'))
    train_data_10 = D_.MapChain(MS_.crop_front_sq, MS_.get_it(N)('wing_10'))
    train_data_20 = D_.MapChain(MS_.crop_front_sq, MS_.get_it(N)('wing_20'))
    train_data_30 = D_.MapChain(MS_.crop_front_sq, MS_.get_it(N)('wing_30'))
    train_data_30c = D_.MapChain(MS_.crop_center_sq, MS_.get_it(N)('wing_30'))

    train_data = train_data_30

    fig, axes = plt.subplots(nrows=1, ncols=train_data[0].shape[0]+1)
    anim = A_.Animation(fig, frames=N)

    def plot_(i):
        [ax.cla() for ax in axes]
        frame = train_data[i]

        # frame = sigmoid(frame, 5)

        ### sigmoid
        a = 2
        frame = np.stack([sigmoid(frame[0]-1, a), sigmoid(frame[1], a), frame[2]])
        plotv = lambda frame: MS_.vorticity0_logit(frame, a)

        ### linear
        # f_ = lambda frame: np.clip(frame / 4 + 0.5, 0, 1)
        # frame = np.stack([f_(frame[0]-1), f_(frame[1]), frame[2]])
        # plotv = MS_.vorticity

        data = map(lambda f, d: lambda ax: f(d, ax),
                   (V_.plot_vel, V_.plot_vel, V_.plot_gray, V_.plot_vor),
                   (*frame, plotv(frame)))

        # モノクロ
        # data = map(lambda f, d: lambda ax: f(d, ax),
        #            (V_.plot_gray, V_.plot_gray, V_.plot_gray),
        #            frame)
        V_.show_frame_m(data, fig, axes, file=False)

    plot_(0)
    fig.savefig('mono.png')

    # anim(plot_, file='anim.mp4')


def plot_cg(casename, out):
    with ut.chdir(out):
        dot_path = ut.select_file(files=ut.globm('**/cg.dot'), idx=None)
        # dot = respath + 'cg.dot'
        png_path = os.path.splitext(dot_path)[0] + '.png'
        code = subprocess.call(['dot', '-Tpng', dot_path, '-o', png_path])
    print(code)
    return code


def task4(*args, **kwargs):
    casename = kwargs.get('case') or 'case9_0'
    out = f'__result__/{casename}'

    if kwargs.get('check_snapshot'):
        check_snapshot(out, show=True)
        return

    mode = kwargs.get('mode')

    if not mode:
        process4(casename, out)

    elif mode == '1':
        process4_1(casename, out)

    elif mode == '2':
        plot_cg(casename, out)


################################################################################

def __test__():
    from scipy import stats
    fig, ax = plt.subplots()
    x = np.arange(-5, 5, 0.01)
    y = stats.norm.pdf(x)
    ax.plot(x, y)
    plt.show()


def __test__():
    from scipy import stats
    fig, ax = plt.subplots()
    x = np.arange(0, 1, 0.01)

    colors = [(0, '#ff0000'), (0.5, '#000000'), (1, '#00ff00')]
    cmap = plc.LinearSegmentedColormap.from_list('custom_cmap', colors)

    y = stats.norm.pdf(x)
    ax.plot(x, vmin=0, vmax=1)
    fig.colorbar()
    plt.show()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs='?', default='',
                        choices=['', '1', '2'],
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

    else:
        taskname = 'task4'
        if taskname in globals():
            f_ = globals().get(taskname)
            with ut.stopwatch(taskname) as sw:
                f_(**vars(args), sw=sw)


if __name__ == '__main__':
    sys.exit(main())
