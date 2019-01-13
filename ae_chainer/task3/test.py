import argparse
import glob
import os
import shutil
import sys
from functools import reduce

import numpy as np
import matplotlib.pyplot as plt

import chainer
import chainer.distributions as D
import chainer.functions as F
import chainer.links as L
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
import net_vae as NV_


################################################################################

def test0(args):
    ''' D.Normalのテスト '''
    n_latent = 10

    loc = np.zeros(n_latent, np.float32)
    scale = np.ones(n_latent, np.float32)
    normal = D.Normal(loc, scale=scale)
    samples = np.arange(0, 1, 0.1, dtype=np.float32)

    # prior = NV_.Prior(3)
    # normal = prior()
    print(normal.sample())
    print(normal.sample(1))
    print(normal.sample((1,1)))
    print(normal.sample_n(1))
    print(normal.prob(samples))
    print(normal.log_prob(samples))
    print(normal.mean)


def test1(args):
    ''' D.Bernoulliのテスト
    chainer.distributions.Bernoulli(p=None, logit=None, binary_check=False)
    p, logitの内どちらか一方を指定する
    '''
    n_latent = 10

    loc = np.zeros(n_latent, np.float32)
    scale = np.ones(n_latent, np.float32)
    normal = D.Normal(loc, scale=scale)

    p = np.arange(0.1, 1, 0.1, dtype=np.float32)
    logit = np.log(p/(1-p))
    samples = np.ones(9, dtype=np.float32) * 0.5

    brnoulli = D.Bernoulli(logit=logit)
    print(brnoulli.sample(10))
    print(brnoulli.prob(samples))
    print(brnoulli.prob(samples*2))
    # print(brnoulli.log_prob(samples))
    print(brnoulli.mean)

    # print(brnoulli.sample(1))
    # print(brnoulli.sample((1,1)))
    # print(brnoulli.sample_n(1))

    brnoulli = D.Bernoulli(p=p, binary_check=True)
    print(brnoulli.sample(10))
    print(brnoulli.prob(samples))
    print(brnoulli.prob(samples*2))
    # print(brnoulli.log_prob(samples))
    print(brnoulli.mean)


def test2(args):
    ''' 学習データ範囲を表示 '''

    keys = 'plate_00', 'plate_10', 'plate_20', 'plate_30', \
           'wing_00', 'wing_05', 'wing_10', 'wing_15'
    keys = 'wing_20', 'wing_30'
    for key in keys:
        print(key)
        original_data = D_.get_original_data(key, size=2000) # (128, 256)
        for i in range(20):
            u = original_data[100*i:100*(i+1), :, :, 0]
            v = original_data[100*i:100*(i+1), :, :, 1]
            print(f'[{i:2d}] u: [{u.min():.3f}, {u.max():.3f}] v: [{v.min():.3f}, {v.max():.3f}]')


def test3(args):
    ''' データ作成のテスト '''

    print(sys._getframe().f_code.co_name)

    ''' 学習パラメータ定義 '''
    epoch = 200
    batchsize = 50
    out = f'__result__/res_{ut.snow}'

    ''' 学習データ作成 '''
    key = 'wing_00'
    cache_path = f'__cache__/{key}'
    original_data = D_.get_original_data(key, size=2000) # (128, 256)
    train_data = D_.get_train_data(D_.extract_uv_norm, original_data,
                                   name='full_norm', cache=True,
                                   cache_path=cache_path)

    if False:
        a = np.array(train_data[1000:1500], dtype=np.float32)
        print(a.shape)
        print(np.min(a), np.max(a))
        return
    elif False:
        V_.show_v(train_data)
        return

    ''' 学習モデル作成 '''
    sample = train_data[:1]
    model = M_.get_model('case1', sample=sample)
    # train_iter, valid_iter, _ = M_.get_cfd_train_data(train_data, table=None,
    #                                                   batchsize=batchsize)

    # if args.clear:
    #     shutil.rmtree(out, ignore_errors=True)

    # M_.train_model(model, train_iter, valid_iter, epoch=epoch, out=out)


# def test1(args):
#     ''' データ作成のテスト '''

#     print(sys._getframe().f_code.co_name)

#     ''' 学習パラメータ定義 '''
#     epoch = 200
#     batchsize = 50
#     out = f'{args.out}/res_{ut.snow}'

#     ''' 学習データ作成 '''
#     key = 'plate_00'
#     cache_path = f'__cache__/{key}'
#     original_data = D_.get_original_data(key, size=2000) # (128, 256)
#     train_data = D_.get_train_data(D_.extract_uv_sq_norm, original_data,
#                                    name='sq_norm', cache=True,
#                                    cache_path=cache_path)

#     if False:
#         a = np.array(train_data[1500:2000], dtype=np.float32)
#         print(a.shape)
#         print(np.min(a), np.max(a))
#         return
#     elif False:
#         V_.show_v(train_data)
#         return

#     ''' 学習モデル作成 '''
#     sample = C_.xp.array(train_data[:1])
#     model = M_.get_model('case0', sample=sample)
#     train_iter, valid_iter, _ = M_.get_cfd_train_data(train_data, table=None,
#                                                       batchsize=batchsize)

#     if args.clear:
#         shutil.rmtree(out, ignore_errors=True)

#     M_.train_model(model, train_iter, valid_iter, epoch=epoch, out=out)


# def test2(args):
#     ''' モデル読み出しのテスト '''

#     # モデルのパスを取得
#     out = ut.select_file('__result__', key=r'res_.*')
#     print('out:', out)
#     file = ut.select_file(out, key=r'snapshot_.*')
#     print('file:', file)

#     # 学習データ作成
#     key = 'wing_10'
#     cache_path = f'__cache__/{key}'
#     original_data = D_.get_original_data(key, size=2000) # (128, 256)
#     train_data = D_.get_train_data(D_.extract_uv_sq_norm, original_data,
#                                    name='sq_norm', cache=True,
#                                    cache_path=cache_path)

#     # 学習モデル作成
#     sample = train_data[:1]
#     model = M_.get_model('case0', sample=sample)

#     # モデル読み込み
#     chainer.serializers.load_npz(file, model,
#                                  path='updater/model:main/predictor/')

#     model.to_cpu()
#     it_data = map(lambda a: model.xp.asarray(a[None, ...]), train_data)
#     it_forward = map(model, it_data)
#     it_result = map(lambda v: v.array[0], it_forward)
#     it_zip = zip(train_data, it_result)
#     # a = next(it)
#     # print(a.shape)
#     # a = model(a)
#     # print(a.shape)
#     with chainer.using_config('train', False), \
#          chainer.using_config('enable_backprop', False):
#         # V_.show_chainer_2c(it_result)
#         V_.show_chainer_2r2c(it_zip)


################################################################################

class Cls(object):
    def __getitem__(self, key):
        return key


def __test__():
    a = np.array([[[1, 1.1], [2, 2.2], [3, 3.3]], [[4, 4.4], [5, 5.5], [6, 6.6]]])
    print(np.sum(a, axis=-1))
    # print(a.reshape((-1, *a.shape[2:])).shape)
    # print(np.broadcast_to(a, (2, 2, 3, 2)))

def __test__():
    a = np.array([[1], [2]])
    b = np.repeat(a, 2, axis=0)
    print(a)
    print(a.shape)
    print(b)
    print(b.shape)


def __test__():
    a = D.Normal(loc=np.array([0, 0, 0], dtype=np.float32),
                 scale=np.array([1, 1, 1], dtype=np.float32))
    print(a.mean)
    print(a.sample())
    print(a.batch_shape)


def __test__():
    d0 = D.Normal(loc=np.array([0, 0, 0], dtype=np.float32),
                  scale=np.array([1, 1, 1], dtype=np.float32))
    d1 = D.Normal(loc=np.array([0, 0, 1], dtype=np.float32),
                  scale=np.array([1, 1.5, 1], dtype=np.float32))
    print(chainer.kl_divergence(d0, d1))


def __test__():
    a = np.random.random(3)
    d0 = D.Normal(loc=a, scale=np.array([1, 1, 1], dtype=np.float32))
    b0 = D.Bernoulli(logit=a)
    print(a)
    print(d0.mean)
    print(b0.mean)
    print(F.sigmoid(a))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('case', nargs='?', default='',
                        choices=['', '0', '1', '2', '3'],
                        help='Number of main procedure')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='__result__',
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
    args = get_args()

    if args.test:
        __test__()
        return

    if args.case in '0123456789':
        testf_ = globals().get('test' + args.case)
        if testf_:
            with ut.stopwatch('case'+args.case):
                testf_(args)

    elif args.case == '01':
        with ut.stopwatch('case'+args.case):
            sample01(out=out, clear=clear)

    elif args.case == '1':
        model_path = ut.fsort(glob.glob(f'{out}/all/*.model'))[-1]
        print(model_path)
        sample1(model_path=model_path)


if __name__ == '__main__':
    sys.exit(main())
