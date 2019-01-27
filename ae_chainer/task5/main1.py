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
import main as ms


# パス
SRC_DIR = os.path.dirname(__file__)
SRC_FILE = os.path.basename(__file__)
SRC_FILENAME = os.path.splitext(SRC_FILE)[0]


################################################################################

def process3(keys, modelname, out):
    ''' 中間層の出力の可視化 '''

    file = ms.check_snapshot(out)
    N = 100

    # 学習データ作成
    # train_data_00 = D_.MapChain(ms.crop_center_sq, ms.get_it(10)('wing_00'))
    train_data_10 = D_.MapChain(ms.crop_center_sq, ms.get_it(N)('wing_10'))
    # train_data_20 = D_.MapChain(ms.crop_center_sq, ms.get_it(N)('wing_20'))
    # train_data_30 = D_.MapChain(ms.crop_center_sq, ms.get_it(N)('wing_30'))
    train_data = train_data_10

    # モデル読み込み
    sample = train_data[:1]
    model = M_.get_model(modelname, sample=sample)
    chainer.serializers.load_npz(file, model, path='updater/model:main/')
    model.to_cpu()

    # 入力の渦度を可視化
    if False:
        V_.show_frame(ms.vorticity(train_data[0]))
        return

    # エンコード
    def enc(x, n=None, file=None):
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            h = chainer.Variable(model.xp.asarray(x))
            for i, chain in enumerate(model.predictor):
                h = chain.encode(h)
                if n is None and not isinstance(h, D.Normal) and h.ndim == 4:
                    V_.show_frame(h.array[0], file=file)
                if i == n:
                    break
            return h

    def dec(h, n=None, file=None):
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            if isinstance(h, D.Normal):
                h = h.mean
            for i, chain in enumerate(reversed(model.predictor)):
                h = chain.decode(h)
                if n is None and h.ndim == 4:
                    V_.show_frame(h.array[0], file=file)
                if i == n:
                    break
            return h

    x = train_data[:1]
    xa = chainer.Variable(model.xp.asarray(x))

    if False:
        h = enc(x, 0)
        V_.show_frame(h.array[0], file=0.01)
        return

    if True:
        fig0, axes0 = plt.subplots(nrows=1, ncols=xa.shape[1]+1)

        hbatch = enc(train_data, 3)
        fig1, axes1 = V_.show_frame_filter_env(hbatch[0])

        for x, y in zip(train_data, hbatch):
            # [ax.cla() for ax in axes0.flatten()]
            # [ax.cla() for ax in axes1.flatten()]
            x_data = list(map(lambda f, d: lambda ax: f(d, ax),
                              (V_.plot_vel, V_.plot_vel, V_.plot_gray, V_.plot_vor),
                              [*x, ms.vorticity(x)]))
            V_.show_frame_m(x_data, fig0, axes0, file=0.01)
            V_.show_frame_m(y.array, fig1, axes1, file=0.01)

        plt.show()
        return

    if True:
        V_.show_frame(x[0], exf=ms.vorticity, file=f'filter_i.png')
        V_.show_frame(z.array[0], exf=ms.vorticity, file=f'filter_o.png')
        return

    for fn in range(1, 4):
        z = enc(x, fn+1)
        V_.show_frame(z.array[0], file=f'filter_{fn}.png')


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

def process4(keys, modelname, out):
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
    model = M_.get_model(modelname, sample=sample)
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
                print(len(p))
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
            y = model.predictor(x)
            if isinstance(model, NV_.VAELoss):
                y = F.sigmoid(y)
            return y

    # エンコード
    def enc(x, n=None):
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            z = model.predictor.encode(x)
            if isinstance(z, D.Normal):
                z = z.mean
            return z

    def dec(z, n=None):
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            y = model.predictor.decode(z)
            if isinstance(model, NV_.VAELoss):
                y = F.sigmoid(y)
            return y

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

    if True:
        V_.show_frame(x0[0], exf=ms.vorticity, file=f'src_x0.png')
        V_.show_frame(x3[0], exf=ms.vorticity, file=f'src_x1.png')
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
    # for i in (-10, 20):
        t = i / 20
        print(t)
        # z = (1 - t) * z0 + t * z1 # 渦移動
        # z = (1 - t) * z0 + t * z3 # 翼回転
        z = 0.5 * (1 - t) * (z0 + z3) + t * z2 # 中間
        # if i == 41:
        #     z = z3
        # else:
        # z = z0 + (z2c - z2) * t
        # z.array[0, :48] = 0.0
        # z = 0 * z0 + np.random.rand(1, 64)
        plot_z(z)
        y = dec(z)
        V_.show_frame(y.array[0], exf=ms.vorticity, file=f'recon_r={i:02d}.png')
        # if t < 0:
        #     V_.show_frame(y.array[0], exf=ms.vorticity, file=f'recon_t=n_{1+t:.1f}.png')
        # else:
        #     V_.show_frame(y.array[0], exf=ms.vorticity, file=f'recon_t=p_{t:.1f}.png')
        plt.pause(0.001)
    plot_z()
    return

    xa = chainer.Variable(model.xp.asarray(x))

    if True:
        z = dec(x)
        z = F.sigmoid(z)

    print(z.shape)
    print(F.mean_squared_error(xa, z))

    if True:
        V_.show_frame(x[0], exf=ms.vorticity, file=f'filter_i.png')
        V_.show_frame(z.array[0], exf=ms.vorticity, file=f'filter_o.png')
        return

    for fn in range(1, 4):
        z = enc(x, fn+1)
        V_.show_frame(z.array[0], file=f'filter_{fn}.png')


def task4(*args, **kwargs):
    ''' task2: VAEのz入力から復元の可視化 '''

    print(sys._getframe().f_code.co_name)

    # keys = 'plate_10', 'wing_00', 'plate_20', 'wing_15', 'plate_30', 'wing_05'
    keys = 'wing_00',
    name = kwargs.get('case', 'case4_0')
    out = f'__result__/{name}'

    if kwargs.get('check_snapshot'):
        check_snapshot(out, show=True)
        return

    process4(keys, name, out)


################################################################################

def __test__():
    fig, ax = plt.subplots()
    for d in ['top', 'right']:
        ax.spines[d].set_visible(False)

    with ut.chdir(f'{SRC_DIR}/__result__/case4_2'):
        path = ut.select_file('.', key=r'res_.*')
        with ut.chdir(path):
            log = ut.load('log.json', from_json=True)
            loss_t = [l['main/loss'] for l in log]
            loss_v = [l['val/main/loss'] for l in log]
            # ax.plot(np.array(loss_t), label='training error')
            # ax.plot(np.array(loss_v), label='validation error')
            # ax.set_ylim((0, 0.05))

            # loss_t = [l['main/kl_penalty'] for l in log]
            # loss_v = [l['val/main/kl_penalty'] for l in log]
            # ax.set_ylim((0, 1000))
            # ax.plot(np.array(loss_t), label='training $D_{KL}$')
            # ax.plot(np.array(loss_v), label='validation $D_{KL}$')

            loss_t = [-l['main/reconstr'] for l in log]
            loss_v = [-l['val/main/reconstr'] for l in log]
            ax.set_ylim((180000, 200000))
            ax.plot(np.array(loss_t), label='training loss')
            ax.plot(np.array(loss_v), label='validation loss')

            fig.legend()

    fig.savefig('error.png')
            # plt.show()


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
