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


DEBUG0 = False
DEBUG1 = True # maybe_init


################################################################################
# ベースネットワーク
################################################################################

class AEBase(object):

    def adjust(self):
        if C_.DEVICE >= 0:
            self.to_gpu(C_.DEVICE)
        else:
            self.to_cpu()


class LAEChain(chainer.Chain, AEBase):
    ''' 単層エンコーダ+デコーダ(全結合ネットワーク)
    '''

    def __init__(self, in_size, out_size, activation=F.relu):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.n_latent = out_size
        self.in_shape = None
        self.init = False
        self.maybe_init(in_size)
        # with self.init_scope():
        #     self.enc = L.Linear(in_size, out_size)
        #     self.dec = L.Linear(out_size, in_size)
        # self.in_size = in_size

        if type(activation) is tuple:
            self.activation_e = activation[0]
            self.activation_d = activation[1]
        else:
            self.activation_e = activation
            self.activation_d = activation

    def __call__(self, x, **kwargs):
        h = self.encode(x, **kwargs)
        y = self.decode(h, **kwargs)
        return y

    def encode(self, x, **kwargs):
        if DEBUG0:
            print(x.shape)
            print(self.in_size)
        self.in_shape = x.shape[1:]
        self.maybe_init(self.in_shape)

        x_ = x.reshape(-1, self.in_size)
        try:
            y = self.enc(x_)
        except:
            print(x_.shape)
            raise

        if self.activation_e:
            y = self.activation_e(y)

        if kwargs.get('show_shape'):
            print(f'layer(E{self.name}): in: {x.shape} out: {y.shape}')
        return y

    def decode(self, x, **kwargs):
        y = self.dec(x)
        if self.activation_d:
            y = self.activation_d(y)
        y = y.reshape(-1, *self.in_shape)

        if kwargs.get('show_shape'):
            print(f'layer(D{self.name}): in: {x.shape} out: {y.shape}')
        return y

    def maybe_init(self, in_size_):
        if self.init:
            return
        elif in_size_:
            if type(in_size_) is tuple:
                in_size = np.prod(in_size_)
                if DEBUG1:
                    print('maybe_init', in_size_, '->', in_size)
            else:
                in_size = in_size_
                if DEBUG1:
                    print('maybe_init', in_size)

            with self.init_scope():
                self.enc = L.Linear(in_size, self.out_size)
                self.dec = L.Linear(self.out_size, in_size)

            self.in_size = in_size
            self.init = True
            self.adjust()


class CAEChain(chainer.Chain, AEBase):
    ''' 単層エンコーダ+デコーダ(畳み込みネットワーク)
    引数:
        in_channels
        out_channels
        use_indices(bool): maxpoolのインデックス情報をキャッシュ
    '''

    def __init__(self, in_channels, out_channels, activation=F.relu,
                 use_indices=True):
        super().__init__()
        with self.init_scope():
            self.enc = L.Convolution2D(in_channels, out_channels, ksize=3,
                                       stride=1, pad=0)
            self.dec = L.Deconvolution2D(out_channels, in_channels, ksize=3,
                                         stride=1, pad=0)

        if type(activation) is tuple:
            self.activation_e = activation[0]
            self.activation_d = activation[1]
        else:
            self.activation_e = activation
            self.activation_d = activation

        self.n_latent = None
        self.use_indices = use_indices
        self.adjust()

    def __call__(self, x, **kwargs):
        h = self.encode(x, **kwargs)
        y = self.decode(h, **kwargs)
        return y

    def encode(self, x, **kwargs):
        h = self.enc(x)
        if self.activation_e:
            h = self.activation_e(h)
        self.insize = h.shape[2:]
        if self.use_indices:
            y, self.indexes = F.max_pooling_2d(h, ksize=2, return_indices=True)
        else:
            y = F.max_pooling_2d(h, ksize=2)
        # print('encode in:', x.shape)
        # print('encode enc:', h.shape)
        # print('encode out:', y.shape)
        self.n_latent = y.shape
        if kwargs.get('show_shape'):
            print(f'layer(E{self.name}): in: {x.shape} out: {y.shape}')
        return y

    def decode(self, x, **kwargs):
        if not x.shape[0] == self.indexes.shape[0]:
            self.indexes = self.xp.repeat(self.indexes,
                                          x.shape[0]//self.indexes.shape[0],
                                          axis=0)

        if self.use_indices:
            h = F.upsampling_2d(x, self.indexes, ksize=2, outsize=self.insize)
        else:
            h = F.unpooling_2d(x, ksize=2, outsize=self.insize)
        y = self.dec(h)
        if self.activation_d:
            y = self.activation_d(y)
        # print('decode in:', x.shape)
        # print('decode enc:', h.shape)
        # print('decode out:', y.shape)
        if kwargs.get('show_shape'):
            print(f'layer(D{self.name}): in: {x.shape} out: {y.shape}')
        return y


class CAEList(chainer.ChainList, AEBase):
    ''' 単層エンコーダ+デコーダの直列リスト
    '''

    def __init__(self, *links):
        super().__init__(*links)
        self.n_latent = self[-1].out_size
        self.adjust()
        self.count = 0

    def __call__(self, x, **kwargs):
        if DEBUG0:
            self.count += 1
            print('call CAEList:', self.count, ' '*20) #, end='\r')
        h = self.encode(x, **kwargs)
        y = self.decode(h, **kwargs)
        return y

    def encode(self, x, **kwargs):
        y = reduce(lambda h, l: l.encode(h, **kwargs), self, x)
        return y

    def decode(self, x, **kwargs):
        y = reduce(lambda h, l: l.decode(h, **kwargs), reversed(self), x)
        return y
