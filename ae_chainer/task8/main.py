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


# パス
SRC_DIR = os.path.dirname(__file__)
SRC_FILE = os.path.basename(__file__)
SRC_FILENAME = os.path.splitext(SRC_FILE)[0]


################################################################################

def identity(x):
    return x


def loop(data):
    while True:
        yield data


def tapp(data, fn=None):
    if fn:
        print(*fn(data))
    else:
        print(data)
    return data


def sigmoid(x, a=1):
    return 1 / (1 + np.exp(-a * x))


def logit(x, a=1):
    x_ = np.clip(x, 1e-5, 1-1e-5)
    return np.where((0 < x) * (x < 1), np.log(x_ / (1 - x_)) / a, np.nan)


################################################################################

def get_extract(key):
    raise
    # vmin, vmax = {
    #     'plate_00': (-0.668, 1.261),
    #     'plate_10': (-1.329, 1.963),
    #     'plate_20': (-2.096, 2.726),
    #     'plate_30': (-2.289, 3.797),
    #     'wing_00': (-0.759, 1.444),
    #     'wing_05': (-1.041, 1.558),
    #     'wing_10': (-1.218, 1.728),
    #     'wing_15': (-1.617, 2.181),
    #     'wing_20': (-1.847, 2.842),
    #     'wing_30': (-2.192, 3.760),
    # }[key]
    vmin, vmax = -3.8, 3.8
    return D_.extract_uvf_norm(vmin, vmax)


def get_it(size=None, cache=False):

    src = '..\\__cache__'
    dst = '__cache__'
    if not os.path.isdir(dst) and os.path.isdir(src):
        print('symlink:', src, '<<===>>', dst)
        os.symlink(src, dst)

    default_it = (os.path.join('__cache__', x) for x in 'FDCHX')
    cache_default = next(filter(os.path.exists, default_it), '__cache__')

    len_org = 2000

    def g_(key):
        print('create data:', key)

        cache_path = os.path.join(cache_default + key)
        cache_path = next(ut.iglobm(f'__cache__/**/{key}'), cache_path)

        original_data = D_.get_original_data(key, size=len_org) # (2000, 512, 1024, 5)

        train_data = D_.get_train_data(D_.extract_uvf_sigmoid(1.5),
                                       original_data,
                                       name='full_norm_uvf_sig15', cache=cache,
                                       cache_path=cache_path)
        if size:
            return train_data[len_org-size:len_org]
        return train_data
    return g_


################################################################################
# データを加工(学習)
################################################################################

class TrainDataset(chainer.dataset.DatasetMixin):
    def __init__(self, it, name=''):
        self.it = it
        self.name = name
        self.count = 0

    def __len__(self):
        return len(self.it)

    def get_example(self, i):
        self.count += 1
        print(f'TrainDataset({self.name}):', self.count, end=' \r')
        a = self.it[i]
        return a, a


################################################################################

def crop_random_sq(frame):
    ''' frame: (C, H, W)
    (:, 512, 1024) => (:, 384, 384)
    random_range = 0:128, 0:640
    '''
    size = 384
    p = [np.random.randint(r - size) for r in frame.shape[1:]]
    return frame[:, p[0]:p[0]+size, p[1]:p[1]+size]
    # if np.random.rand() < 0.5:
    #     s = slice(p[0], p[0]+size, 1)
    # else:
    #     s = slice(p[0]+size-1, p[0]-1 if p[0] else None, -1)
    # a = frame[:, s, p[1]:p[1]+size]
    # # print('I:', np.min(a), np.max(a))
    # return a


def crop_center_sq(frame):
    ''' frame: (C, H, W)
    (:, 512, 1024) => (:, 384, 384)
    '''
    size = 384
    p = [(r - size) // 2 for r in frame.shape[1:]]
    a = frame[:, p[0]:p[0]+size, p[1]:p[1]+size]
    # a[2] = a[2] * (1 - 1e-5) + np.random.uniform(0, 1e-5, (384, 384))
    return a


def crop_front_sq(frame):
    ''' frame: (C, H, W)
    (:, 512, 1024) => (:, 384, 384)
    '''
    size = 384
    p = [(frame.shape[1] - size) // 2, 128] # [(r - size) // 2 for r in frame.shape[1:]]
    a = frame[:, p[0]:p[0]+size, p[1]:p[1]+size]
    # a[2] = a[2] * (1 - 1e-5) + np.random.uniform(0, 1e-5, (384, 384))
    return a


def crop_back_sq(frame):
    size = 384
    p = [(frame.shape[1] - size) // 2, frame.shape[2] - size]
    return frame[:, p[0]:p[0]+size, p[1]:p[1]+size]


def velocity(frame):
    return np.linalg.norm(frame[:2], axis=0)


def vorticity0(frame):
    return frame[0, :-2, 1:-1] - frame[0, 2:, 1:-1] \
         - frame[1, 1:-1, :-2] + frame[1, 1:-1, 2:]


def vorticity0_logit(frame, a=1):
    return logit(frame[0, :-2, 1:-1], a) - logit(frame[0, 2:, 1:-1], a) \
         - logit(frame[1, 1:-1, :-2], a) + logit(frame[1, 1:-1, 2:], a)


def vorticity1(frame):
    return (frame[0, :-2, :-2] + frame[0, :-2, 1:-1] + frame[0, :-2, 2:] \
          - frame[0, 2:, 1:-1] - frame[0,  2:, 1:-1] - frame[0,  2:, 2:] \
          - frame[1, :-2, :-2] - frame[1, 1:-1, :-2] - frame[1, 2:, :-2] \
          + frame[1, :-2,  2:] + frame[1, 1:-1,  2:] + frame[1, 2:, 2:]) / 3


vorticity = vorticity0


################################################################################

def check_snapshot(out, show=False):
    # モデルのパスを取得
    respath = ut.select_file(out, key=r'res_.*', idx=None)
    print('path:', respath)
    file = ut.select_file(respath, key=r'snapshot_.*', idx=None)
    print('file:', file)

    if show:
        # npz保存名確認
        with np.load(file) as npzfile:
            for f in npzfile:
                print(f)
                continue
                if f[-1] == 'W':
                    print(f)
                    print(npzfile[f].shape)
                # print(npzfile[f].dtype, npzfile[f].shape, f)
            # print(npzfile['extensions/LogReport/_log'])
    return file


def process1(keys, modelname, out):
    ''' モデル読み出し+可視化 '''

    file = check_snapshot(out)

    # 学習データ作成
    # keys = 'wing_00', 'wing_10', 'wing_20', 'wing_30'
    train_data = D_.MapChain(crop_center_sq, *map(get_it(300), keys),
                             name='center_crop')
    # train = TrainDataset(train_data)
    # train_iter = SerialIterator(train, batchsize)

    # モデル読み込み
    sample = train_data[:1]
    model = M_.get_model(modelname, sample=sample)
    chainer.serializers.load_npz(file, model, path='updater/model:main/')
    model.to_cpu()

    # データセット
    it_data = map(lambda a: model.xp.asarray(a[None, ...]), train_data)

    def convert_z(z, t_=[0]):
        t = t_[0]
        a = np.zeros_like(z)
        a[:, 0] = t * np.cos(t)
        a[:, 1] = t * np.sin(t)
        t_[0] += 0.01 * np.pi
        # a = np.ones_like(z)
        return z * 0 + a

    # plot it_zp
    fig, ax = plt.subplots()
    def plot_z(z=None, l=[]):
        if z is None:
            if not l:
                return
            ax.plot(np.array(l))
            fig.savefig(f'z_log_{modelname}.png')
            return
        ax.cla()
        l.append(z.array.flatten())
        ax.plot(np.array(l))
        return z

    # モデル適用
    def apply_model(x):
        y = model.predictor(x, inference=True, show_z=False, convert_z=plot_z)
        if isinstance(model, NV_.VAELoss):
            y = F.sigmoid(y)
        return y
    it_forward = map(apply_model, it_data)

    # 結果データ取得
    it_result = map(lambda v: v.array[0], it_forward)

    # 入力データと復号データを合成
    it_zip = zip(train_data, it_result)
    mse_list = []
    def hook_(x0, x1, *rest):
        mse = F.mean_squared_error(x0, x1).array
        print('mse:', mse)
        mse_list.append(mse)
        return x0, x1
    it_zip_with_mse = map(hook_, train_data, it_result)

    # plot mse
    if False:
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            file = f'mse_{modelname}_{keys[0]}.npy'
            if os.path.isfile(file):
                mse_list = np.load(file)
            else:
                list(it_zip_with_mse)
                mse_list = np.array(mse_list)
                np.save(file, mse_list)
            fig, ax = plt.subplots()
            ax.plot(mse_list)
            plt.show()
            return

    colors = [(0, 'red'), (0.5, 'black'), (1, '#00ff00')]
    cmap = plc.LinearSegmentedColormap.from_list('custom_cmap', colors)
    def plot_vor(frame):
        vor = vorticity(frame)
        def f_(ax):
            ax.imshow(vor, cmap=cmap, vmin=-0.1, vmax=0.1)
        return f_

    it_zip_msehook_with_vor = map(lambda vs: [(*v, plot_vor(v)) for v in vs], it_zip_with_mse)

    with chainer.using_config('train', False), chainer.no_backprop_mode():
        with ut.chdir('__img__'):
            # V_.show_chainer_2c(it_result)
            V_.show_chainer_NrNc(it_zip_msehook_with_vor, nrows=3, ncols=2,
                                 direction='ub')
    plot_z()


def task1(*args, **kwargs):
    ''' task1: VAEの復元の可視化 '''

    print(sys._getframe().f_code.co_name)

    # keys = 'plate_10', 'wing_00', 'plate_20', 'wing_15', 'plate_30', 'wing_05'
    keys = 'plate_30',
    name = kwargs.get('case', 'case4_0')
    out = f'__result__/{name}'

    if kwargs.get('check_snapshot'):
        check_snapshot(out, show=True)
        return

    process1(keys, name, out)


def plot_mse_0():
    # case = 'case4_2'
    case = 'case5_1'
    model = 'VAE' if '4' in case else 'AE'
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.15, bottom=0.1, right=0.98, top=0.98,
                        wspace=0.1, hspace=0.2)

    for d in ['top', 'right']:
      ax.spines[d].set_visible(False)
      # ax.spines[d].set_linewidth(0.5)

    ax.set_ylim(0, 0.002)
    ax.set_xlabel('steps')
    ax.set_ylabel('mean squared error [-]')


    for a in ('00', '10', '20', '30'):
        file = f'mse_{case}_wing_{a}.npy'

        mse = np.load(file)
        label = model + ' $\\alpha=$' + str(int(a)) + '°'
        ax.plot(mse, label=label)
    ax.legend()
    fig.savefig(f'mse_{case}.png')
    plt.show()


def plot_mse_1():
    case = 'case4_2'
    # case = 'case5_1'
    grid = 'wing'
    alphas = ('05', '15')
    model = 'VAE' if '4' in case else 'AE'
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.15, bottom=0.1, right=0.98, top=0.98,
                        wspace=0.1, hspace=0.2)

    for d in ['top', 'right']:
      ax.spines[d].set_visible(False)
      # ax.spines[d].set_linewidth(0.5)

    ax.set_ylim(0, 0.004)
    ax.set_xlabel('steps')
    ax.set_ylabel('mean squared error [-]')


    for a in alphas:
        file = f'mse_{case}_{grid}_{a}.npy'

        mse = np.load(file)
        label = model + ' $\\alpha=$' + str(int(a)) + '°'
        ax.plot(mse, label=label)
    ax.legend()
    fig.savefig(f'mse_{grid}_{case}.png')
    plt.show()


plot_mse = plot_mse_1


################################################################################

def process2(keys, modelname, out):
    ''' モデル読み出し+可視化 '''

    file = check_snapshot(out)

    # 学習データ作成
    train_data_00 = D_.MapChain(crop_center_sq, get_it(300)('wing_00'))
    train_data_10 = D_.MapChain(crop_center_sq, get_it(300)('wing_10'))
    train_data_20 = D_.MapChain(crop_center_sq, get_it(300)('wing_20'))
    train_data_30 = D_.MapChain(crop_center_sq, get_it(300)('wing_30'))
    train_data = train_data_10

    # モデル読み込み
    sample = train_data[:1]
    model = M_.get_model(modelname, sample=sample)
    chainer.serializers.load_npz(file, model, path='updater/model:main/')
    model.to_cpu()

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

    # for z in Z:
    #     print_z(z)

    # モデル適用
    def apply_model(z):
        y = model.predictor.decode(z, inference=True)
        if isinstance(model, NV_.VAELoss):
            y = F.sigmoid(y)
        return y

    it_z = (Z[0]*(1-i)+Z[1]*i for i in np.arange(0, 1, 0.01))
    it_zp = map(plot_z, it_z)
    it_decode = map(apply_model, it_zp)

    colors = [(0, 'red'), (0.5, 'black'), (1, '#00ff00')]
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
        with ut.chdir(f'__img__/man_with_vor/{modelname}'):
            V_.show_chainer_NrNc(it_zip_with_vor, nrows=3, ncols=3,
                                 direction='tb')
    plot_z()


def task2(*args, **kwargs):
    ''' task2: VAEのz入力から復元の可視化 '''

    print(sys._getframe().f_code.co_name)

    # keys = 'plate_10', 'wing_00', 'plate_20', 'wing_15', 'plate_30', 'wing_05'
    keys = 'wing_00',
    name = kwargs.get('case', 'case4_0')
    out = f'__result__/{name}'

    if kwargs.get('check_snapshot'):
        check_snapshot(out, show=True)
        return

    process2(keys, name, out)


################################################################################

def check_data_range():
    keys = ('wing_00', 'wing_10', 'wing_20', 'wing_30',
            'plate_00', 'plate_10', 'plate_20', 'plate_30')
    train_data = D_.MapChain(identity, *map(get_it(size=None), keys),
                             name='random_crop')
    pixels = np.prod(train_data[0][0].shape)
    for d in range(2):
        tmin, tmax = 0, 0
        print('check:', 'uv'[d])
        for i in range(0, len(train_data), 200):
            a = train_data[i][d, ...] - (1 - d)
            # a = sigmoid(a, 5)
            amin = a.min()
            amax = a.max()
            amean = a.mean()
            astd = a.std()
            count = np.sum((-astd <= a) * (a <= astd))
            # count = np.sum((-0.8 <= a) * (a <= 0.8))
            ratio = count / pixels
            tmin = min(tmin, amin)
            tmax = max(tmax, amax)
            print(f'{i:5d} min: {amin:8.5f} max: {amax:8.5f} mean: {amean:8.5f} std: {astd:8.5f} ratio: {ratio:8.5f}')
        print(f'tmin: {tmin} tmax: {tmax}')


def check_data_hist():
    keys = ('wing_00', 'wing_10', 'wing_20', 'wing_30',
            'plate_00', 'plate_10', 'plate_20', 'plate_30')
    train_data = D_.MapChain(identity, *map(get_it(size=None), keys))
    fig, ax = plt.subplots()
    d = 1
    for i in range(0, len(train_data), 200):
        data = (train_data[i][d] - (1 - d)).flatten()
        data = sigmoid(data, 4)
        ax.cla()
        ax.hist(data, bins=100, range=(-2, 2))
        plt.pause(0.1)
    plt.show()


################################################################################

def __test__():
    key = 'wing_00'
    modelname = 'case2n'
    get_task_data_merge(key, modelname, batchsize=1)


def __test__():
    crop_random_sq(np.ones((2, 512, 1024), dtype=np.float32))


def __test__():
    frame = get_it(1)('wing_30')[0]
    vel = frame[0, :-2, 1:-1] - frame[0, 2:, 1:-1] \
        - frame[1, 1:-1, :-2] + frame[1, 1:-1, 2:]
        # + frame[1, :-2] + frame[1, 1:-1] + frame[1, 2:] \
        # - frame[1, :-2] + frame[1, 1:-1] + frame[1, 2:]

    colors = [(0, 'red'), (0.5, 'black'), (1, 'green')]
    cmap = plc.LinearSegmentedColormap.from_list('custom_cmap', colors)

    maxval = np.abs(frame).max()

    plt.imshow(vel, cmap=cmap, vmin=-0.1, vmax=0.1)
    plt.show()
    print(vel.shape, maxval)


def __test__():
    with ut.chdir(f'{SRC_DIR}/__result__/case4_2'):
        path = ut.select_file('.', key=r'res_.*')
        with ut.chdir(path):
            log = ut.load('log.json', from_json=True)
            loss = [l['main/loss'] for l in log]
            plt.ylim((180000, 210000))
            plt.plot(np.array(loss))
            plt.show()


def __test__():
    def printa(v):
        print(v.array)
    o = np.ones(5, dtype=np.float) * 0.01
    a = np.arange(0.01, 0.6, 0.1)[:5]
    x = chainer.Variable(o)
    t = chainer.Variable(a)
    # tb = chainer.Variable((t.array > 0.5).astype(np.float32)) * 0.999 + 0.0001
    # b = F.sigmoid(a)
    b = NV_.D.Bernoulli(p=x)
    # # d = NV_.D.Bernoulli(p=b)
    l0 = b.log_prob(t)
    l1 = t * F.log(x) + (1 - t) * F.log(1 - x)
    l2 = x * F.log(t) + (1 - x) * F.log(1 - t)
    print(a)
    printa(l0)
    printa(l1)
    printa(l2)
    # l1 = b.log_prob(tb)
    # print(x)
    # print(l0, F.sum(l0).array)
    # print(l1, F.sum(l1).array)
    # s = F.sum(l0)
    # s.backward()
    # print(x.grad)
    # print(d.mean)
    # a = np.arange(8).reshape(2, 2, 2)
    # print(a)
    # print(a.sum(axis=-1))


def __tes__():
    reporter = chainer.Reporter()
    observer = object()
    reporter.add_observer('my_observer', observer)
    observation = {}
    with reporter.scope(observation):
        reporter.report({'x': 1}, observer)
        reporter.report({'x': 10}, observer)
    # reporter.report({'loss': 0}, None)
    print(observation)
    print(dir(object))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs='?', default='',
                        choices=['', '0', '1', '2'],
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

    if args.gpu:
        C_.DEVICE = args.gpu

    C_.SHOW_PROGRESSBAR = not args.no_progress

    # out = args.out
    out = f'result/{SRC_FILENAME}'

    if args.test:
        # print(vars(args))
        __test__()

    elif args.mode  == '1':
        check_data_range()

    elif args.mode  == '2':
        check_data_hist()

    elif args.mode in '0123456789':
        taskname = 'task' + args.mode
        if taskname in globals():
            f_ = globals().get(taskname)
            with ut.stopwatch(taskname) as sw:
                f_(**vars(args), sw=sw)

    # if args.mode == '01':
    #     with ut.stopwatch('sample01'):
    #         sample01(out=out, clear=args.clear)
    # elif args.mode == '1':
    #     model_path = ut.fsort(glob.glob(f'{out}/all/*.model'))[-1]
    #     print(model_path)
    #     sample1(model_path=model_path)


if __name__ == '__main__':
    sys.exit(main())

'''
GTX 760
  828,407,808 bytes
1,973,098,496 bytes

GTX 1070
 ,828,407,808 bytes
7,564,598,272 bytes
'''