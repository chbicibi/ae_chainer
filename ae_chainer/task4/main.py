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
# import chainer.links as L
# from chainer.datasets import mnist, cifar, split_dataset_random
# from chainer.datasets.tuple_dataset import TupleDataset
from chainer.iterators import SerialIterator
# from chainer.optimizers import Adam
# from chainer.training import StandardUpdater, Trainer, extensions

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

def loop(data):
    while True:
        yield data


def tapp(data, fn=None):
    if fn:
        print(*fn(data))
    else:
        print(data)
    return data


################################################################################

def get_extract(key):
    vmin, vmax = {
        'plate_00': (-0.668, 1.261),
        'plate_10': (-1.329, 1.963),
        'plate_20': (-2.096, 2.726),
        'plate_30': (-2.289, 3.797),
        'wing_00': (-0.759, 1.444),
        'wing_05': (-1.041, 1.558),
        'wing_10': (-1.218, 1.728),
        'wing_15': (-1.617, 2.181),
        'wing_20': (-1.847, 2.842),
        'wing_30': (-2.192, 3.760),
    }[key]
    return D_.extract_uv_norm(vmin, vmax)


def get_it(size):
    def g_(key):
        print('create data:', key)
        cache_path = f'__cache__/{key}'
        original_data = D_.get_original_data(key, size=2000) # (2000, 512, 1024, 5)
        train_data = D_.get_train_data(get_extract(key), original_data,
                                       name='full_norm', cache=True,
                                       cache_path=cache_path)
        a = train_data[2000-size:2000]
        return a
    return g_


def get_task_data(_, modelname, batchsize=1):
    ''' 学習データ作成
    key == 'wing'
    '''

    # 学習データ作成
    keys = 'wing_00', 'wing_10', 'wing_20', 'wing_30'
    train_data = D_.MapChain(crop_random_sq, *map(get_it(300), keys),
                        name='random_crop')
    train = TrainDataset(train_data)
    train_iter = SerialIterator(train, batchsize)

    # 検証データ作成
    keys = 'wing_05', 'wing_15'
    valid_data = D_.MapChain(crop_random_sq, *map(get_it(100), keys),
                        name='random_crop')
    valid = TrainDataset(valid_data)
    valid_iter = SerialIterator(valid, batchsize, repeat=False, shuffle=False)

    # 学習モデル作成
    sample = train_data[:1]
    model = M_.get_model(modelname, sample=sample)

    return model, train_data, train_iter, valid_iter


################################################################################
# データを加工(学習)
################################################################################

def crop_random_sq(frame):
    ''' frame: (C, H, W)
    (2, 512, 1024) => (2, 382, 382)
    random_range = 0:128, 0:640
    '''
    size = 382
    p = [np.random.randint(r - size) for r in frame.shape[1:]]
    if np.random.rand() < 0.5:
        s = slice(p[0], p[0]+size, 1)
    else:
        s = slice(p[0]+size-1, p[0]-1 if p[0] else None, -1)
    return frame[:, s, p[1]:p[1]+size]


def crop_center_sq(frame):
    ''' frame: (C, H, W)
    (2, 512, 1024) => (2, 382, 382)
    '''
    size = 382
    p = [(r - size) // 2 for r in frame.shape[1:]]
    return frame[:, p[0]:p[0]+size, p[1]:p[1]+size]


class TrainDataset(chainer.dataset.DatasetMixin):
    def __init__(self, it):
        self.it = it

    def __len__(self):
        return len(self.it)

    def get_example(self, i):
        a = self.it[i]
        return a, a


def vorticity(frame):
    return frame[0, :-2, 1:-1] - frame[0, 2:, 1:-1] \
         - frame[1, 1:-1, :-2] + frame[1, 1:-1, 2:]


################################################################################

def process0(key, modelname, out):
    ''' オートエンコーダ学習 '''

    # 学習パラメータ定義
    epoch = 1000
    batchsize = 50
    logdir = f'{out}/res_{key}_{modelname}_{ut.snow}'

    model, _, train_iter, valid_iter = get_task_data(key, modelname, batchsize)

    M_.train_model(model, train_iter, valid_iter, epoch=epoch, out=logdir)


def process0_resume(key, modelname, out):
    ''' オートエンコーダ学習 '''

    # 学習パラメータ定義
    epoch = 3000
    batchsize = 50
    logdir = f'{out}/res_{key}_{modelname}_{ut.snow}'

    model, _, train_iter, valid_iter = get_task_data(key, modelname, batchsize)

    init_file = check_snapshot(out)

    M_.train_model(model, train_iter, valid_iter, epoch=epoch, out=logdir,
                   init_file=init_file)


def task0(*args, **kwargs):
    ''' task0: 学習メイン '''

    print(sys._getframe().f_code.co_name)

    # keys = 'plate_10', 'wing_00', 'plate_20', 'wing_15', 'plate_30', 'wing_05'
    # keys = 'wing_30',
    # keys = 'plate_30', 'plate_20', 'plate_10', 'plate_00', 'wing_30', 'wing_20', 'wing_10', 'wing_00'
    name = kwargs.get('case', 'case4_0')
    out = f'__result__/{name}'
    error = None

    try:
        for key in [0]:
            if kwargs.get('resume'):
                process0_resume(key, name, out)
            else:
                process0(key, name, out)

    except Exception as e:
        error = e
        tb = traceback.format_exc()
        print('Error:', error)
        print(tb)

    if 'sw' in kwargs and int(kwargs['sw']) < 3600:
        return

    with ut.EmailIO(None, 'ae_chainer: Task is Complete') as e:
        print(sys._getframe().f_code.co_name, file=e)
        print(ut.strnow(), file=e)
        if 'sw' in kwargs:
            print('Elapsed:', kwargs['sw'], file=e)
        if error:
            print('Error:', error)
            print(tb)


################################################################################

def check_snapshot(out, show=False):
    # モデルのパスを取得
    respath = ut.select_file(out, key=r'res_.*')
    print('path:', respath)
    file = ut.select_file(respath, key=r'snapshot_.*')
    print('file:', file)

    if show:
        # npz保存名確認
        with np.load(file) as npzfile:
            for f in npzfile:
                print(f)
                # print(npzfile[f].dtype, npzfile[f].shape, f)
            print(npzfile['extensions/LogReport/_log'])
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

    # plot z
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
        y = model.predictor(x, inference=True, show_z=True, convert_z=plot_z)
        if isinstance(model, NV_.VAELoss):
            y = F.sigmoid(y)
        return y
    it_forward = map(apply_model, it_data)

    # 結果データ取得
    it_result = map(lambda v: v.array[0], it_forward)

    # 入力データと復号データを合成
    it_zip = zip(train_data, it_result)
    def hook_(x0, x1, *rest):
        print('\nmse:', F.mean_squared_error(x0, x1).array)
        return x0, x1
    it_zip_with_mse = map(hook_, train_data, it_result)

    colors = [(0, 'red'), (0.5, 'black'), (1, 'green')]
    cmap = plc.LinearSegmentedColormap.from_list('custom_cmap', colors)
    def plot_vor(frame):
        vor = vorticity(frame)
        def f_(ax):
            ax.imshow(vor, cmap=cmap, vmin=-0.07, vmax=0.07)
        return f_

    it_zip_msehook_with_vor = map(lambda vs: [(*v, plot_vor(v)) for v in vs], it_zip_with_mse)

    with chainer.using_config('train', False), chainer.no_backprop_mode():
        # V_.show_chainer_2c(it_result)
        V_.show_chainer_NrNc(it_zip_msehook_with_vor, nrows=2, ncols=3, direction='lr')
    plot_z()


def task1(*args, **kwargs):
    ''' task1: VAEの復元の可視化 '''

    print(sys._getframe().f_code.co_name)

    # keys = 'plate_10', 'wing_00', 'plate_20', 'wing_15', 'plate_30', 'wing_05'
    keys = 'wing_30',
    name = kwargs.get('case', 'case4_0')
    out = f'__result__/{name}'

    if kwargs.get('check_snapshot'):
        check_snapshot(out, show=True)
        return

    process1(keys, name, out)


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