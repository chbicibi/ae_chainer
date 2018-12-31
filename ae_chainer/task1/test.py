import argparse
import glob
import os
import shutil
import sys
from functools import reduce

import numpy as np
import matplotlib.pyplot as plt

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.datasets import mnist, cifar, split_dataset_random
from chainer.datasets.tuple_dataset import TupleDataset
from chainer.iterators import SerialIterator
from chainer.optimizers import Adam
from chainer.training import StandardUpdater, Trainer, extensions

import myutils as ut

import common as C_
import dataset as D_
import model as M_
import vis as V_


################################################################################

def test0(args):
    ''' 一括学習のテスト(ダミーデータ) '''

    epoch = 100

    model = M_.get_model('case0') #
    train_iter, valid_iter, test_iter = M_.get_dummy_train_data()

    if args.clear:
        shutil.rmtree(args.out, ignore_errors=True)

    M_.train_model(model, train_iter, valid_iter,
                   epoch=epoch,
                   out=f'{args.out}/all',
                   fix_trained=False)


def test1(args):
    ''' データ作成のテスト '''

    print(sys._getframe().f_code.co_name)

    ''' 学習パラメータ定義 '''
    epoch = 200
    batchsize = 50
    out = f'{args.out}/res_{ut.snow}'

    ''' 学習データ作成 '''
    key = 'plate_00'
    cache_path = f'__cache__/{key}'
    original_data = D_.get_original_data(key, size=2000) # (128, 256)
    train_data = D_.get_train_data(D_.extract_uv_sq_norm, original_data,
                                   name='sq_norm', cache=True,
                                   cache_path=cache_path)

    if False:
        a = np.array(train_data[1500:2000], dtype=np.float32)
        print(a.shape)
        print(np.min(a), np.max(a))
        return
    elif False:
        V_.show_v(train_data)
        return

    ''' 学習モデル作成 '''
    sample = C_.xp.array(train_data[:1])
    model = M_.get_model('case0', sample=sample)
    train_iter, valid_iter, _ = M_.get_cfd_train_data(train_data, table=None,
                                                      batchsize=batchsize)

    if args.clear:
        shutil.rmtree(out, ignore_errors=True)

    M_.train_model(model, train_iter, valid_iter, epoch=epoch, out=out)


def test2(args):
    ''' モデル読み出しのテスト '''

    # モデルのパスを取得
    out = ut.select_file('result', key=r'res_.*')
    print('out:', out)
    file = ut.select_file(out, key=r'snapshot_.*')
    print('file:', file)

    # 学習データ作成
    key = 'wing_10'
    cache_path = f'__cache__/{key}'
    original_data = D_.get_original_data(key, size=2000) # (128, 256)
    train_data = D_.get_train_data(D_.extract_uv_sq_norm, original_data,
                                   name='sq_norm', cache=True,
                                   cache_path=cache_path)

    # 学習モデル作成
    sample = train_data[:1]
    model = M_.get_model('case0', sample=sample)

    # モデル読み込み
    chainer.serializers.load_npz(file, model,
                                 path='updater/model:main/predictor/')

    model.to_cpu()
    it_data = map(lambda a: model.xp.asarray(a[None, ...]), train_data)
    it_forward = map(model, it_data)
    it_result = map(lambda v: v.array[0], it_forward)
    it_zip = zip(train_data, it_result)
    # a = next(it)
    # print(a.shape)
    # a = model(a)
    # print(a.shape)
    with chainer.using_config('train', False), \
         chainer.using_config('enable_backprop', False):
        # V_.show_chainer_2c(it_result)
        V_.show_chainer_2r2c(it_zip)


################################################################################

def __test__():
    import _locale
    l = _locale._getdefaultlocale()
    import common as C_
    l = _locale._getdefaultlocale()
    print(l)
    print(C_.V)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('case', nargs='?', default='',
                        choices=['', '0', '1', '2'],
                        help='Number of main procedure')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--clear', '-c', action='store_true',
                        help='Remove directory at the beginning')
    parser.add_argument('--no-progress', '-np', action='store_true',
                        help='Hide progress bar')
    parser.add_argument('--test', '-t', action='store_true',
                        help='Run as test mode')
    args = parser.parse_args()
    return args


def main():
    # global DEVICE, PROGRESSBAR

    args = get_args()

    # # グローバル変数定義
    # DEVICE = args.gpu
    # # if DEVICE >= 0:
    # #     chainer.cuda.get_device_from_id(0).use()
    # PROGRESSBAR = not args.no_progress

    # # out = args.out
    # out = f'result/{FILENAME}'
    # clear = args.clear

    if args.test:
        __test__()
        return

    if args.case == '0':
        with ut.stopwatch('case'+args.case):
            test0(args)

    elif args.case == '1':
        with ut.stopwatch('case'+args.case):
            test1(args)

    elif args.case == '2':
        with ut.stopwatch('case'+args.case):
            test2(args)

    elif args.case == '01':
        with ut.stopwatch('case'+args.case):
            sample01(out=out, clear=clear)

    elif args.case == '1':
        model_path = ut.fsort(glob.glob(f'{out}/all/*.model'))[-1]
        print(model_path)
        sample1(model_path=model_path)


if __name__ == '__main__':
    sys.exit(main())
