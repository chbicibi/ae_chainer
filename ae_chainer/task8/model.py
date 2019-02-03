import argparse
import glob
import os
import shutil
from functools import reduce
from itertools import chain
from time import sleep

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

    loss = NV_.VAELoss(model, beta=1.0, k=1)
    return loss


def get_model_case1_z30():
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
        # N_.LAEChain(None, 40), # in: 10*3*3
        NV_.VAEChain(None, 30)) # in: 10*3*3

    loss = NV_.VAELoss(model, beta=1.0, k=1)
    return loss


def get_model_case2n():
    ''' AE '''

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
        N_.LAEChain(10, 2, activation=(None, F.relu)))

    loss = L.Classifier(model, lossfun=F.mean_squared_error)
    if C_.DEVICE >= 0:
        loss.to_gpu(C_.DEVICE)
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

    loss = NV_.VAELoss(model, beta=1.0, k=1)
    return loss


def get_model_case3():
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
        NV_.VAEChain(10, 3)) # in: 10*3*3

    loss = NV_.VAELoss(model, beta=1.0, k=1)
    return loss


def get_model_case4_0():
    ''' VAE
    1 4 10 22 46 94 190 382
    2 6 14 30 62 126 254 510
    3 8 18 38 78 158 318
    '''

    # 入出力チャンネル数を指定
    model = N_.CAEList(
        N_.CAEChain(2, 10, activation=(F.relu, None)),
        N_.CAEChain(10, 20),
        N_.CAEChain(20, 30),
        N_.CAEChain(30, 30),
        N_.CAEChain(30, 30),
        N_.CAEChain(30, 30),
        N_.CAEChain(30, 20),
        NV_.VAEChain(None, 10))

    loss = NV_.VAELoss(model, beta=1.0, k=1)
    return loss


def get_model_case4_1():
    ''' VAE
    1 4 10 22 46 94 190 382
    2 6 14 30 62 126 254 510
    3 8 18 38 78 158 318
    '''

    # 入出力チャンネル数を指定
    model = N_.CAEList(
        N_.CAEChain(2, 10, activation=(F.relu, None)),
        N_.CAEChain(10, 20),
        N_.CAEChain(20, 30),
        N_.CAEChain(30, 30),
        N_.CAEChain(30, 30),
        N_.CAEChain(30, 30),
        N_.CAEChain(30, 20),
        NV_.VAEChain(None, 10))

    loss = NV_.VAELoss(model, beta=1.0, k=1)
    return loss


def get_model_case4_2():
    ''' VAE
    1 4 10 22 46 94 190 382
    2 6 14 30 62 126 254 510
    3 8 18 38 78 158 318
    '''

    # 入出力チャンネル数を指定
    model = N_.CAEList(
        N_.CAEChain(2, 8, activation=(F.relu, None)),
        N_.CAEChain(8, 16),
        N_.CAEChain(16, 32),
        N_.CAEChain(32, 64),
        N_.CAEChain(64, 64),
        N_.CAEChain(64, 128),
        N_.CAEChain(128, 256),
        NV_.VAEChain(None, 64))

    loss = NV_.VAELoss(model, beta=1.0, k=1)
    return loss


def get_model_case5_0():
    ''' AE
    1 4 10 22 46 94 190 382
    2 6 14 30 62 126 254 510
    3 8 18 38 78 158 318
    '''

    # 入出力チャンネル数を指定
    model = N_.CAEList(
        N_.CAEChain(2, 10, activation=(F.relu, None)),
        N_.CAEChain(10, 20),
        N_.CAEChain(20, 30),
        N_.CAEChain(30, 30),
        N_.CAEChain(30, 30),
        N_.CAEChain(30, 30),
        N_.CAEChain(30, 20),
        N_.LAEChain(None, 10, activation=(None, F.relu)))

    loss = L.Classifier(model, lossfun=F.mean_squared_error)
    if C_.DEVICE >= 0:
        loss.to_gpu(C_.DEVICE)
    return loss


def get_model_case5_1():
    ''' AE
    1 4 10 22 46 94 190 382
    2 6 14 30 62 126 254 510
    3 8 18 38 78 158 318
    '''

    # 入出力チャンネル数を指定
    model = N_.CAEList(
        N_.CAEChain(2, 10, activation=(F.relu, None)),
        N_.CAEChain(10, 20),
        N_.CAEChain(20, 30),
        N_.CAEChain(30, 30),
        N_.CAEChain(30, 30),
        N_.CAEChain(30, 30),
        N_.CAEChain(30, 20),
        N_.LAEChain(None, 10, activation=(None, F.relu)))

    loss = L.Classifier(model, lossfun=F.mean_squared_error)
    if C_.DEVICE >= 0:
        loss.to_gpu(C_.DEVICE)
    return loss


def get_model_case6():
    ''' VAE
    1 4 10 22 46 94 190 382
    2 6 14 30 62 126 254 510
    3 8 18 38 78 158 318
    '''

    # 入出力チャンネル数を指定
    model = N_.CAEList(
        N_.CAEChain(3, 8, activation=(F.relu, None)),
        N_.CAEChain(8, 16),
        N_.CAEChain(16, 32),
        N_.CAEChain(32, 64),
        N_.CAEChain(64, 64),
        N_.CAEChain(64, 128),
        N_.CAEChain(128, 256),
        NV_.VAEChain(None, 64))

    loss = NV_.VAELoss(model, beta=1.0, k=1)
    return loss


def get_model_case6_1():
    ''' VAE
    '''

    # 入出力チャンネル数を指定
    model = N_.CAEList(
        N_.CAEChain(3, 8, activation=(F.relu, None), batch_norm=True),
        N_.CAEChain(8, 16, batch_norm=True),
        N_.CAEChain(16, 32, batch_norm=True),
        N_.CAEChain(32, 64, batch_norm=True),
        N_.CAEChain(64, 64, batch_norm=True),
        N_.CAEChain(64, 128, batch_norm=True),
        N_.CAEChain(128, 256, batch_norm=True),
        NV_.VAEChain(None, 64))

    loss = NV_.VAELoss(model, beta=1.0, k=1)
    return loss


def get_model_case6_2():
    ''' VAE
    '''

    # 入出力チャンネル数を指定
    model = N_.CAEList(
        N_.CAEChain(3, 8, activation=(F.relu, None), batch_norm='re'),
        N_.CAEChain(8, 16, batch_norm='re'),
        N_.CAEChain(16, 32, batch_norm='re'),
        N_.CAEChain(32, 64, batch_norm='re'),
        N_.CAEChain(64, 64, batch_norm='re'),
        N_.CAEChain(64, 128, batch_norm='re'),
        N_.CAEChain(128, 256, batch_norm='re'),
        NV_.VAEChain(None, 64))

    loss = NV_.VAELoss(model, beta=1.0, k=1)
    return loss


def get_model_case6_3():
    ''' VAE
    '''

    # 入出力チャンネル数を指定
    model = N_.CAEList(
        N_.CAEChain(3, 8, activation=(F.relu, None), batch_norm='re',
                    padding=True),
        N_.CAEChain(8, 16, batch_norm='re', padding=True),
        N_.CAEChain(16, 32, batch_norm='re', padding=True),
        N_.CAEChain(32, 64, batch_norm='re', padding=True),
        N_.CAEChain(64, 64, batch_norm='re', padding=True),
        N_.CAEChain(64, 128, batch_norm='re', padding=True),
        N_.CAEChain(128, 256, batch_norm='re', padding=True),
        NV_.VAEChain(None, 64))

    loss = NV_.VAELoss(model, beta=1.0, k=1)
    return loss


def get_model_case6_4():
    ''' VAE
    '''

    # 入出力チャンネル数を指定
    model = N_.CAEList(
        N_.CAEChain(3, 8, activation=(F.relu, None)),
        N_.CAEChain(8, 16, activation=F.relu),
        N_.CAEChain(16, 32, activation=F.relu),
        N_.CAEChain(32, 64, activation=F.relu),
        N_.CAEChain(64, 64, activation=F.relu),
        N_.CAEChain(64, 128, activation=F.relu),
        N_.CAEChain(128, 256, activation=F.relu),
        NV_.VAEChain(None, 64, activation=F.relu))

    loss = NV_.VAELoss(model, beta=1.0, k=1)
    return loss


def get_model_case7():
    ''' AE
    '''

    # 入出力チャンネル数を指定
    model = N_.CAEList(
        N_.CAEChain(3, 8, activation=(F.relu, None), batch_norm='re',
                    ksize=5, padding=True),
        N_.CAEChain(8, 16, batch_norm='re', ksize=5, padding=True),
        N_.CAEChain(16, 32, batch_norm='re', ksize=5, padding=True),
        N_.CAEChain(32, 64, batch_norm='re', ksize=5, padding=True),
        N_.CAEChain(64, 64, batch_norm='re', ksize=5, padding=True),
        N_.CAEChain(64, 128, batch_norm='re', ksize=5, padding=True),
        N_.CAEChain(128, 256, batch_norm='re', ksize=5, padding=True),
        N_.LAEChain(None, 64, activation=(None, F.relu)))

    loss = L.Classifier(model, lossfun=F.mean_squared_error)
    if C_.DEVICE >= 0:
        loss.to_gpu(C_.DEVICE)
    return loss


def get_model_case7_1():
    ''' AE
    '''

    # 入出力チャンネル数を指定
    model = N_.CAEList(
        N_.CAEChain(3, 8, activation=(F.sigmoid, None)),
        N_.CAEChain(8, 16, activation=F.sigmoid),
        N_.CAEChain(16, 32, activation=F.sigmoid),
        N_.CAEChain(32, 64, activation=F.sigmoid),
        N_.CAEChain(64, 64, activation=F.sigmoid),
        N_.CAEChain(64, 128, activation=F.sigmoid),
        N_.CAEChain(128, 256, activation=F.sigmoid),
        N_.LAEChain(None, 64, activation=(None, F.sigmoid)))

    loss = N_.AELoss(model)
    return loss


def get_model_case8_0():
    ''' VAE
    activation: F.relu => F.sigmoid
    '''

    # 入出力チャンネル数を指定
    model = N_.CAEList(
        N_.CAEChain(3, 8, activation=(F.sigmoid, None)),
        N_.CAEChain(8, 16, activation=F.sigmoid),
        N_.CAEChain(16, 32, activation=F.sigmoid),
        N_.CAEChain(32, 64, activation=F.sigmoid),
        N_.CAEChain(64, 64, activation=F.sigmoid),
        N_.CAEChain(64, 128, activation=F.sigmoid),
        N_.CAEChain(128, 256, activation=F.sigmoid), # => (256, 3, 3)
        NV_.VAEChain(None, 64, activation=(None, F.sigmoid))) # 2304 -> 64 (1 / 36)

    loss = NV_.VAELoss(model, beta=1.0, k=1)
    return loss


def get_model_case9_0():
    ''' VAE
    z: 2dim
    '''
    model = N_.CAEList(
        N_.CAEChain(3, 8, activation=(F.sigmoid, None)),
        N_.CAEChain(None, 16),
        N_.CAEChain(None, 32),
        N_.CAEChain(None, 64),
        N_.CAEChain(None, 64),
        N_.CAEChain(None, 64),
        N_.CAEChain(None, 128), # => (128, 3, 3)
        N_.LAEChain(None, 128), # 1152 -> 128 (1 / 9)
        N_.LAEChain(None, 16), # 128 -> 16 (1 / 8)
        NV_.VAEChain(None, 2)) # 16 -> 2 (1 / 8)

    loss = NV_.VAELoss(model)
    return loss


def get_model_case9_1():
    ''' VAE
    z: 32dim
    '''
    model = N_.CAEList(
        N_.CAEChain(3, 8, activation=(F.sigmoid, None)),
        N_.CAEChain(None, 16),
        N_.CAEChain(None, 32),
        N_.CAEChain(None, 64),
        N_.CAEChain(None, 64),
        N_.CAEChain(None, 64),
        N_.CAEChain(None, 128), # => (128, 3, 3)
        N_.LAEChain(None, 288), # 1152 -> 288 (1 / 4)
        N_.LAEChain(None, 96), # 288 -> 96 (1 / 3)
        NV_.VAEChain(None, 32)) # 96 -> 32 (1 / 3)

    loss = NV_.VAELoss(model)
    return loss


def get_model_case9_2():
    ''' VAE
    z: 64dim
    パラメータ数: ((3*8+8*8+8*16+16*16+16*32+32*32+32*64+64*64+64*64+64*64*128+128*128+128*128+128*128)*5*5+
                1152*384+384*128+128*64)*2=30283824
    '''
    model = N_.CAEList(
        N_.CAEChainM(3, 8, activation=(F.sigmoid, None), ksize=5, n_conv=2),
        N_.CAEChainM(None, 16, ksize=5, n_conv=2),
        N_.CAEChainM(None, 32, ksize=5, n_conv=2),
        N_.CAEChainM(None, 64, ksize=5, n_conv=2),
        N_.CAEChainM(None, 64, ksize=5, n_conv=2),
        N_.CAEChainM(None, 128, ksize=5, n_conv=2),
        N_.CAEChainM(None, 128, ksize=5, n_conv=2), # => (128, 3, 3)
        N_.LAEChain(None, 384), # 1152 -> 384 (1 / 3)
        N_.LAEChain(None, 128), # 384 -> 128/ (1 / 3)
        NV_.VAEChain(None, 64)) # 128 -> 64 (1 / 2)

    loss = NV_.VAELoss(model)
    return loss


def get_model_case9_3():
    ''' VAE
    z: 64dim
    パラメータ数: ((3*8+8*8+8*16+16*16+16*32+32*32+32*64+64*64+64*64+64*64*128+128*128+128*128+128*128)*3*3+
                1152*384+384*128+128*64)*2=11541808
    '''
    model = N_.CAEList(
        N_.CAEChainM(3, 8, activation=(F.sigmoid, None), ksize=3, n_conv=2),
        N_.CAEChainM(None, 16, ksize=3, n_conv=2),
        N_.CAEChainM(None, 32, ksize=3, n_conv=2),
        N_.CAEChainM(None, 64, ksize=3, n_conv=2),
        N_.CAEChainM(None, 64, ksize=3, n_conv=2),
        N_.CAEChainM(None, 128, ksize=3, n_conv=2),
        N_.CAEChainM(None, 128, ksize=3, n_conv=2), # => (128, 3, 3)
        N_.LAEChain(None, 384), # 1152 -> 384 (1 / 3)
        N_.LAEChain(None, 128), # 384 -> 128 (1 / 3)
        NV_.VAEChain(None, 64)) # 128 -> 64 (1 / 2)

    loss = NV_.VAELoss(model)
    return loss


def get_model_case9_4():
    ''' VAE
    z: 64dim
    パラメータ数: ((3*8+8*8+8*16+16*16+16*32+32*32+32*64+64*64+64*64+64*64*128+128*128+128*128+128*128)*3*3+
                1152*384+384*128+128*64)*2=11541808
    '''
    model = N_.CAEList(
        N_.CAEChain(3, 8, activation=(F.sigmoid, None), ksize=3),
        N_.CAEChain(None, 16, ksize=3),
        N_.CAEChain(None, 32, ksize=3),
        N_.CAEChain(None, 64, ksize=3),
        N_.CAEChain(None, 64, ksize=3),
        N_.CAEChain(None, 128, ksize=3),
        N_.CAEChain(None, 128, ksize=3), # => (128, 3, 3)
        N_.LAEChain(None, 384), # 1152 -> 384 (1 / 3)
        N_.LAEChain(None, 128), # 384 -> 128 (1 / 3)
        NV_.VAEChain(None, 64)) # 128 -> 64 (1 / 2)

    loss = NV_.VAELoss(model)
    return loss


def get_model(name, sample=None):
    # 関数名自動取得に変更

    function_name = 'get_model_' + name
    if function_name in globals():
        model = globals()[function_name]()
    else:
        raise NameError('Function Not Found:', function_name)

    if sample is not None:
        # モデル初期化
        print('init model')
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            model.predictor(model.xp.asarray(sample), show_shape=True)
    return model


################################################################################
# 学習
################################################################################

def plot_loss_ex(trainer):
    return
    fig, ax = plt.subplots()
    for d in ['top', 'right']:
        ax.spines[d].set_visible(False)
    ylim_low = float('inf')
    ylim_upp = 0

    try:
        with ut.chdir(trainer.out):
            if not os.path.isfile('log.json'):
                return

            log = ut.load('log.json', from_json=True)

            for key in ('main/loss', 'val/main/loss'):
                a = np.array([l[key] for l in log])
                a = np.clip(a, 0, 1e6)

                ax.plot(a, label=key)
                ylim_upp = max(np.ceil(np.max(a[min(len(a)-1,3):50]))/1000*1000,
                               ylim_upp)
                ylim_low = min(np.min(a)//1000*1000, ylim_low)

            ax.set_ylim((ylim_low, ylim_upp))
            ax.set_xlabel('epoch')
            ax.grid(True)
            fig.legend()
            fig.savefig('loss1.png')

    finally:
        plt.close(fig)


def lr_drop_ex(alpha, start=1000):
    def f_(trainer):
        epoch = trainer.updater.epoch
        if epoch < start:
            return
        # trainer.updater.get_optimizer('main').alpha *= 0.1
        alpha_new = alpha * max(0.8**max((epoch-start)//50+1, 0), 0.1)
        trainer.updater.get_optimizer('main').alpha = alpha_new
    return f_


def pause_ex(trainer):
    while os.path.isfile(os.path.join(trainer.out, 'pause')):
        print('pause', end='\r')
        sleep(10)


def train_model(model, train_iter, valid_iter, epoch=10, out='__result__',
                init_file=None, fix_trained=False, alpha=0.001, init_all=True):
    learner = model

    # 最適化手法の選択
    optimizer = Adam(alpha=alpha).setup(learner)

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
    trainer.extend(extensions.ParameterStatistics(learner.predictor,
                                                  {'std': np.std},
                                                  prefix='links'))

    ## 学習率を記録する
    trainer.extend(extensions.observe_lr())

    ## 学習経過を画面出力
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'val/main/loss', 'elapsed_time', 'lr']))

    ## ログ記録 (他のextensionsの結果も含まれる)
    trainer.extend(extensions.LogReport(log_name='log.json'))

    ## 学習経過を画像出力
    if C_.OS_IS_WIN:
        def ex_pname(link):
            ls = list(link.links())[1:]
            if not ls:
                names = (p.name for p in link.params())
            else:
                names = chain(*map(ex_pname, ls))
            return [f'{link.name}/{n}' for n in names]

        def register(keys, file_name):
            trainer.extend(extensions.PlotReport(keys,# x_key='epoch',
                                                 file_name=file_name,
                                                 marker=None))

        register('lr', file_name='lr.png')
        register(['main/loss', 'val/main/loss'], file_name='loss.png')

        if isinstance(learner, NV_.VAELoss):
            register(['main/reconstr', 'val/main/reconstr'],
                     file_name='reconstr.png')

            register(['main/kl_penalty', 'val/main/kl_penalty'],
                     file_name='kl_penalty.png')

            register(['main/mse_vel', 'val/main/mse_vel'],
                     file_name='mse_vel.png')

            register(['main/mse_vor', 'val/main/mse_vor'],
                     file_name='mse_vor.png')

        if False:
            for link in learner.predictor:
                observe_keys_std = [f'links/predictor/{key}/data/std'
                                    for key in ex_pname(link)]
                file_name = f'std_{link.name}.png'
                register(observe_keys_std, file_name=file_name)

        else:
            for link in learner.predictor:
                param_names = ex_pname(link)
                for d in ('data', 'grad'):
                    observe_keys_std = [f'links/predictor/{key}/{d}/std'
                                        for key in param_names]
                    for l in ('enc', 'dec'):
                        file_name = f'std_{d}_{link.name}_{l}.png'
                        f_ = lambda s: l in s or f'bn{l[0]}' in s
                        keys = list(filter(f_, observe_keys_std))
                        register(keys, file_name=file_name)

    ## ネットワーク形状をdot言語で出力
    ## 可視化コード: ```dot -Tpng cg.dot -o [出力ファイル]```
    trainer.extend(extensions.dump_graph('main/loss'))

    ## トレーナーオブジェクトをシリアライズし、出力ディレクトリに保存
    trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}.model'))

    ## プログレスバー
    if C_.SHOW_PROGRESSBAR:
        trainer.extend(extensions.ProgressBar())

    if init_file:
        print('loading snapshot:', init_file)
        try:
            if init_all:
                chainer.serializers.load_npz(init_file, trainer)

            else:
                chainer.serializers.load_npz(init_file, learner,
                                             path='updater/model:main/')
        except KeyError:
            raise
            # chainer.serializers.load_npz(init_file, trainer.updater, path='updater/')

    # 自作Extension
    trainer.extend(plot_loss_ex, trigger=(1, 'epoch'))
    trainer.extend(lr_drop_ex(alpha), trigger=(1, 'epoch'))
    trainer.extend(pause_ex, trigger=(1, 'epoch'))

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
