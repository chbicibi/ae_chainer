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


################################################################################
# 損失計算
################################################################################

class AELoss(L.Classifier, AEBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, lossfun=F.mean_squared_error, **kwargs)
        # self.lossfun = lossfun
        # with self.init_scope():
        #     self.predictor = predictor

        self.compute_accuracy = False
        self.adjust()

    def __call__(self, x, x_=None, **kwargs):
        if x_ is None:
            x_ = x

        return self.forward(x, x_)
        # loss = self.lossfun(self.predictor(x), x_)
        # reporter.report({'loss': loss}, self)
        # return loss

    def encode(self, x, **kwargs):
        return self.predictor.encode(x, **kwargs)

    def decode(self, x, **kwargs):
        return self.predictor.decode(x, **kwargs)

    def predict(self, x, **kwargs):
        xa = self.predictor.xp.asarray(x)
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            return self.predictor(xa, inference=True, **kwargs)


################################################################################

class LAEChain(chainer.Chain, AEBase):
    ''' 単層エンコーダ+デコーダ(全結合ネットワーク)
    '''

    def __init__(self, in_size, out_size, activation=F.sigmoid, batch_norm=True,
                 **kwargs):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
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

        self.n_latent = out_size
        self.in_shape = None
        self.init = False
        self.batch_norm = batch_norm
        self.kwargs = kwargs
        self.maybe_init(in_size)

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
            if self.batch_norm:
                y = self.bne(y)
            y = self.activation_e(y)

        if kwargs.get('show_shape'):
            print(f'layer(E{self.name}): in: {x.shape} out: {y.shape}')
        return y

    def decode(self, x, **kwargs):
        y = self.dec(x)
        if self.activation_d:
            if self.batch_norm:
                y = self.bnd(y)
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
                if self.batch_norm:
                    self.bne = L.BatchRenormalization(self.out_size)
                    self.bnd = L.BatchRenormalization(in_size)
                else:
                    # self.bne = L.BatchNormalization(self.out_size)
                    # self.bnd = L.BatchNormalization(in_size)
                    self.bne = None
                    self.bnd = None

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

    def __init__(self, in_channels, out_channels, ksize=5, padding=True,
                 activation=F.sigmoid, use_indices=False, batch_norm=True, **kwargs):
        super().__init__()
        # ksize = kwargs.get('ksize', 3)
        pad = ksize // 2 if padding else 0
        with self.init_scope():
            self.enc = L.Convolution2D(in_channels, out_channels, ksize=ksize,
                                       stride=1, pad=pad)
            self.dec = L.Deconvolution2D(out_channels, in_channels, ksize=ksize,
                                         stride=1, pad=pad)
            if batch_norm:
                self.bne = L.BatchRenormalization(out_channels)
                self.bnd = L.BatchRenormalization(in_channels)
            else:
                # self.bne = L.BatchNormalization(out_channels)
                # self.bnd = L.BatchNormalization(in_channels)
                self.bne = None
                self.bnd = None

        if type(activation) is tuple:
            self.activation_e = activation[0]
            self.activation_d = activation[1]
        else:
            self.activation_e = activation
            self.activation_d = activation

        self.n_latent = None
        self.use_indices = use_indices
        self.batch_norm = batch_norm
        self.adjust()

    def __call__(self, x, **kwargs):
        h = self.encode(x, **kwargs)
        y = self.decode(h, **kwargs)
        return y

    def encode(self, x, **kwargs):
        if not kwargs.get('inference'):
            print(f'E{self.name} in:', F.min(x), F.max(x), ' '*10, end='\r')

        h = self.enc(x)
        if self.activation_e:
            if self.batch_norm:
                h = self.bne(h)
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
        if not kwargs.get('inference'):
            print(f'D{self.name} in:', F.min(x), F.max(x), ' '*10, end='\r')

        if self.use_indices:
            if not x.shape[0] == self.indexes.shape[0]:
                self.indexes = self.xp.repeat(self.indexes,
                                              x.shape[0]//self.indexes.shape[0],
                                              axis=0)
            h = F.upsampling_2d(x, self.indexes, ksize=2, outsize=self.insize)
        else:
            h = F.unpooling_2d(x, ksize=2, outsize=self.insize)
        y = self.dec(h)
        if self.activation_d:
            if self.batch_norm:
                y = self.bnd(y)
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

        convert_z = kwargs.get('convert_z')
        if convert_z:
            h = convert_z(h)

        if kwargs.get('show_z'):
            z = h.array.flatten()
            print(*(f'{s:.3f}' for s in z), end='\r')

        y = self.decode(h, **kwargs)
        return y

    def encode(self, x, **kwargs):
        y = reduce(lambda h, l: l.encode(h, **kwargs), self, x)
        return y

    def decode(self, x, **kwargs):
        y = reduce(lambda h, l: l.decode(h, **kwargs), reversed(self), x)
        return y
