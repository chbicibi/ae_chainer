import argparse
import glob
import os
import shutil
import sys

import numpy as np
# import matplotlib.pyplot as plt

import chainer
# import chainer.functions as F
# import chainer.links as L
# from chainer.datasets import mnist, cifar, split_dataset_random
# from chainer.datasets.tuple_dataset import TupleDataset
# from chainer.iterators import SerialIterator
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

def get_task_data(key, modelname, batchsize=1):
    # 学習データ作成
    cache_path = f'__cache__/{key}'
    original_data = D_.get_original_data(key, size=2000) # (2000, 512, 1024, 5)
    train_data = D_.get_train_data(D_.extract_uv_norm, original_data,
                                   name='full_norm', cache=True,
                                   cache_path=cache_path)

    # 学習モデル作成
    sample = train_data[:1]
    model = M_.get_model(modelname, sample=sample)
    train_iter, valid_iter, _ = M_.get_cfd_train_data(train_data, table=None,
                                                      batchsize=batchsize)

    return model, train_data, train_iter, valid_iter


################################################################################

def task0(key, modelname, out):
    ''' オートエンコーダ学習 '''

    # 学習パラメータ定義
    epoch = 200
    batchsize = 25
    out = f'{out}/res_{key}_{modelname}_{ut.snow}'

    model, _, train_iter, valid_iter = get_task_data(key, modelname, batchsize)

    M_.train_model(model, train_iter, valid_iter, epoch=epoch, out=out)


def case0(*args, **kwargs):
    print(sys._getframe().f_code.co_name)

    # keys = 'plate_10', 'wing_00', 'plate_20', 'wing_15', 'plate_30', 'wing_05'
    keys = 'plate_20',
    name = 'case2'
    out = f'__result__/case0'

    for key in keys:
        task0(key, name, out)


################################################################################

def task1(key, modelname, out):
    ''' モデル読み出し+可視化 '''

    model, train_data, train_iter, valid_iter = get_task_data(key, modelname)

    # モデルのパスを取得
    respath = ut.select_file(out, key=r'res_.*')
    print('path:', respath)
    file = ut.select_file(respath, key=r'snapshot_.*')
    print('file:', file)

    # npz保存名確認
    # with np.load(file) as npzfile:
    #     for f in npzfile:
    #         print(f)
    #     return

    # モデル読み込み
    chainer.serializers.load_npz(file, model,
                                 path='updater/model:main/')

    model.to_cpu()

    # データセット
    it_data = map(lambda a: model.xp.asarray(a[None, ...]), train_data)

    # モデル適用
    it_forward = map(lambda x: model.link(x, inference=True), it_data)

    # 結果データ取得
    it_result = map(lambda v: v.array[0], it_forward)

    # 入力データと復号データを合成
    it_zip = zip(train_data, it_result)

    with chainer.using_config('train', False), chainer.no_backprop_mode():
        # V_.show_chainer_2c(it_result)
        V_.show_chainer_2r2c(it_zip)


def case1(*args, **kwargs):
    print(sys._getframe().f_code.co_name)

    # keys = 'plate_10', 'wing_00', 'plate_20', 'wing_15', 'plate_30', 'wing_05'
    keys = 'wing_10',
    name = 'case1'
    out = f'__result__/case0'

    for key in keys:
        task1(key, name, out)


################################################################################

def __test__():
    pass


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs='?', default='',
                        choices=['', '0', '1', '2'],
                        help='Number of main procedure')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--clear', '-c', action='store_true',
                        help='Remove directory at the beginning')
    parser.add_argument('--no-progress', '-p', action='store_true',
                        help='Hide progress bar')
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
    clear = args.clear

    if args.test:
        __test__()

    elif args.mode == '0':
        with ut.stopwatch('case0'):
            case0(args)

    elif args.mode in '0123456789':
        testf_ = globals().get('case' + args.mode)
        if testf_:
            with ut.stopwatch('case'+args.mode):
                testf_(args)

    # if args.mode == '01':
    #     with ut.stopwatch('sample01'):
    #         sample01(out=out, clear=clear)
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