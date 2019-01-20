import argparse
import glob
import os
import shutil
import sys
import traceback
from operator import itemgetter

import numpy as np
# import matplotlib.pyplot as plt

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
import model as M_
import vis as V_


# パス
SRC_DIR = os.path.dirname(__file__)
SRC_FILE = os.path.basename(__file__)
SRC_FILENAME = os.path.splitext(SRC_FILE)[0]


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
    epoch = 1000
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


def process1(key, modelname, out):
    ''' モデル読み出し+可視化 '''

    file = check_snapshot(out)

    # 学習データ作成
    # keys = 'wing_00', 'wing_10', 'wing_20', 'wing_30'
    keys = 'wing_30',
    train_data = D_.MapChain(crop_center_sq, *map(get_it(300), keys),
                        name='random_crop')
    # train = TrainDataset(train_data)
    # train_iter = SerialIterator(train, batchsize)

    # モデル読み込み
    # model, train_data, train_iter, valid_iter = get_task_data(key, modelname)
    sample = train_data[:1]
    model = M_.get_model(modelname, sample=sample)
    chainer.serializers.load_npz(file, model, path='updater/model:main/')
    model.to_cpu()

    # データセット
    it_data = map(lambda a: model.xp.asarray(a[None, ...]), train_data)

    # モデル適用
    it_forward = map(lambda x: model.predictor(x, inference=True, show_z=True),
                     it_data)

    # 結果データ取得
    it_result = map(lambda v: v.array[0], it_forward)

    # 入力データと復号データを合成
    it_zip = zip(train_data, it_result)
    def hook_(x0, x1):
        print('\nmse:', F.mean_squared_error(x0, x1).array)
        return x0, x1
    it_zip_msehook = map(hook_, train_data, it_result)

    with chainer.using_config('train', False), chainer.no_backprop_mode():
        # V_.show_chainer_2c(it_result)
        V_.show_chainer_2r2c(it_zip_msehook)


def task1(*args, **kwargs):
    ''' task1: 可視化 '''

    print(sys._getframe().f_code.co_name)

    # keys = 'plate_10', 'wing_00', 'plate_20', 'wing_15', 'plate_30', 'wing_05'
    keys = 'wing_00',
    name = kwargs.get('case', 'case4_0')
    out = f'__result__/{name}'

    if kwargs.get('check_snapshot'):
        check_snapshot(out, show=True)
        return

    for key in keys:
        process1(key, name, out)


################################################################################

def __test__():
    key = 'wing_00'
    modelname = 'case2n'
    get_task_data_merge(key, modelname, batchsize=1)


def __test__():
    crop_random_sq(np.ones((2, 512, 1024), dtype=np.float32))


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
        print(vars(args))
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