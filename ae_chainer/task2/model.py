import argparse
import glob
import os
import shutil
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
import net as N_
import net_vae as NV_


DEBUG0 = False


################################################################################
# データセット (ダミー)
################################################################################

class DummyDataset(chainer.dataset.DatasetMixin):
    def __init__(self):
        pass

    def __len__(self):
        return 2000

    def get_example(self, i):
        a = np.zeros((3, 320, 320), dtype=np.float32)
        return a, a


def get_dummy_train_data():
    # データセットの準備
    # train_val, test = cifar.get_cifar10() # => (50000, 2), (10000, 2)

    # データセットの修正 (教師データを学習データで置き換え)
    # testは使わない
    # train_val_data = np.zeros((1000, 3, 320, 320), dtype=np.float32)
    # train_val = TupleDataset(train_val_data, train_val_data)
    train_val = DummyDataset()

    # Validation用データセットを作る
    train_size = int(len(train_val) * 0.9)
    train, valid = split_dataset_random(train_val, train_size, seed=0)

    # Iteratorの作成
    # SerialIteratorはデータセットの中のデータを順番に取り出してくる
    batchsize = 64
    train_iter = SerialIterator(train, batchsize)
    valid_iter = SerialIterator(valid, batchsize, repeat=False, shuffle=False)
    # test_iter = SerialIterator(test, batchsize, repeat=False, shuffle=False)
    test_iter = None

    return train_iter, valid_iter, test_iter


################################################################################
# データセット (CFD)
################################################################################

class CFDDataset(chainer.dataset.DatasetMixin):
    def __init__(self, it, table=None):
        self.it = it
        self.table = table

    def __len__(self):
        return len(self.it)

    def get_example(self, i):
        if DEBUG0:
            print(f'get(ds) {i}/{len(self)}', end=' '*20+'\r')
        if self.table:
            a = self.it[self.table[i]]
        else:
            a = self.it[i]
        return a, a


def get_cfd_train_data(it, table=None, batchsize=64):
    # データセットの準備
    train_val_data = CFDDataset(it, table=table)

    # Validation用データセットを作る
    # testは使わない
    train_size = int(len(train_val_data) * 0.9)
    train, valid = split_dataset_random(train_val_data, train_size, seed=0)

    # Iteratorの作成
    # SerialIteratorはデータセットの中のデータを順番に取り出してくる
    train_iter = SerialIterator(train, batchsize)
    valid_iter = SerialIterator(valid, batchsize, repeat=False, shuffle=False)
    # test_iter = SerialIterator(test, batchsize, repeat=False, shuffle=False)
    test_iter = None

    return train_iter, valid_iter, test_iter


################################################################################
# モデル
################################################################################

def get_model_case0():
    ''' 畳み込みオートエンコーダ '''

    # 入出力チャンネル数を指定
    model = N_.CAEList(
        N_.CAEChain(2, 10, activation=(F.relu, None)), # in: 512
        N_.CAEChain(10, 20), # in: 256
        N_.CAEChain(20, 20), # in: 128
        N_.CAEChain(20, 20), # in: 64
        N_.CAEChain(20, 20), # in: 32
        N_.CAEChain(20, 20), # in: 16
        N_.CAEChain(20, 20), # in: 8
        N_.CAEChain(20, 20), # in: 3
        N_.LAEChain(None, 20), # in: 10*3*3
        N_.LAEChain(20, 10, activation=(None, F.relu)))
    return model


def get_model_case1():
    ''' VAE '''

    # 入出力チャンネル数を指定
    model = N_.CAEList(
        N_.CAEChain(2, 10, activation=(F.relu, None)), # in: 512, 1024
        N_.CAEChain(10, 20), # in: 256
        N_.CAEChain(20, 20), # in: 128
        N_.CAEChain(20, 20), # in: 64
        N_.CAEChain(20, 20), # in: 32
        N_.CAEChain(20, 20), # in: 16
        N_.CAEChain(20, 20), # in: 8
        N_.CAEChain(20, 20), # in: 4
        N_.LAEChain(None, 20), # in: 10*3*3
        NV_.VAEChain(20, 10)) # in: 10*3*3

    loss = NV_.VAELoss(model, 10)
    return loss


def get_model_case2():
    ''' VAE '''

    # 入出力チャンネル数を指定
    model = N_.CAEList(
        N_.CAEChain(2, 10, activation=(F.relu, None)), # in: 512, 1024
        N_.CAEChain(10, 20), # in: 256
        N_.CAEChain(20, 20), # in: 128
        N_.CAEChain(20, 20), # in: 64
        N_.CAEChain(20, 20), # in: 32
        N_.CAEChain(20, 20), # in: 16
        N_.CAEChain(20, 20), # in: 8
        N_.CAEChain(20, 20), # in: 4
        N_.LAEChain(None, 10), # in: 10*3*3
        NV_.VAEChain(10, 2)) # in: 10*3*3

    loss = NV_.VAELoss(model, 10)
    return loss


def get_model(name, sample=None):
    if name == 'case0':
        model = get_model_case0()
    elif name == 'case1':
        model = get_model_case1()
    elif name == 'case2':
        model = get_model_case2()
    else:
        raise NameError

    if sample is not None:
        # モデル初期化
        print('init model')
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            model(model.xp.asarray(sample), show_shape=True)
    return model


################################################################################
# 学習
################################################################################

def train_model(model, train_iter, valid_iter, epoch=10, out=f'result',
                fix_trained=False):
    # learner = L.Classifier(model, lossfun=F.mean_squared_error)
    learner = model
    learner.compute_accuracy = False

    # 最適化手法の選択
    optimizer = Adam().setup(learner)

    if fix_trained:
        for m in model[:-1]:
            m.disable_update()

    # Updaterの準備 (パラメータを更新)
    updater = StandardUpdater(train_iter, optimizer, device=C_.DEVICE)

    # Trainerの準備
    trainer = Trainer(updater, stop_trigger=(epoch, 'epoch'), out=out)

    # TrainerにExtensionを追加する
    ## 検証
    trainer.extend(extensions.Evaluator(valid_iter, learner, device=C_.DEVICE),
                   name='val')

    ## モデルパラメータの統計を記録する
    last_model_id = len(learner.predictor) - 1
    observe_names_std = ['enc/W/data/std', 'dec/W/data/std',
                         'enc/b/data/std', 'dec/b/data/std']
    observe_keys_std = [f'links/{last_model_id}/{s}' for s in observe_names_std]
    trainer.extend(extensions.ParameterStatistics(learner.predictor[-1],
                                                  {'std': np.std},
                                                  report_grads=False,
                                                  prefix='links'))

    ## 学習率を記録する
    trainer.extend(extensions.observe_lr())

    ## 学習経過を画面出力
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'val/main/loss', 'elapsed_time', 'lr']))

    ## プログレスバー
    if C_.SHOW_PROGRESSBAR:
        trainer.extend(extensions.ProgressBar())

    ## ログ記録 (他のextensionsの結果も含まれる)
    trainer.extend(extensions.LogReport())
    # trainer.extend(extensions.LogReport(['main/loss', 'val/main/loss'],
    #                                     log_name='loss.json'))
    # trainer.extend(extensions.LogReport(observe_keys_std, log_name='std.json'))

    ## 学習経過を画像出力
    if C_.OS_IS_WIN:
        trainer.extend(extensions.PlotReport(['main/loss', 'val/main/loss'],
                                             x_key='epoch', file_name='loss.png', marker=None))
        trainer.extend(extensions.PlotReport(observe_keys_std,
                                             x_key='epoch', file_name='std.png', marker=None))
        trainer.extend(extensions.PlotReport('lr',
                                             x_key='epoch', file_name='lr.png', marker=None))

    ## ネットワーク形状をdot言語で出力
    ## 可視化コード: ```dot -Tpng cg.dot -o [出力ファイル]```
    trainer.extend(extensions.dump_graph('main/loss'))

    ## トレーナーオブジェクトをシリアライズし、出力ディレクトリに保存
    trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}.model'))

    # 学習を開始する
    trainer.run()


################################################################################
### ?

def check_inputsize(model, data):
    # model = get_model()
    # data = model.xp.zeros((1, 3, 32, 32), dtype=model.xp.float32)
    print(f'in:', data.shape)
    for i, m in enumerate(model):
        data = model.encode(data)
        print(f'out({i}):', data.shape)
