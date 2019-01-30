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

    def adjust(self, device=C_.DEVICE):
        if device >= 0:
            self.to_gpu(device)
        else:
            self.to_cpu()

    def adapt(self, x):
        return x
        self.adjust(device=self.device)
        if isinstance(x, C_.NDARRAY_TYPES):
            return self.xp.asarray(x, dtype=self.xp.float32)
            # x = chainer.Variable(x)
        elif not isinstance(x, chainer.Variable):
            return x

        if self.device >= 0:
            x.to_gpu(self.device)
        else:
            x.to_cpu()
        return x
        # return self.xp.asarray(x, dtype=self.xp.float32)


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
        x_a = self.predictor.xp.asarray(x)
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            return self.predictor(x_a, inference=True, **kwargs)


################################################################################

class LAEChain(chainer.Chain, AEBase):
    ''' 単層エンコーダ+デコーダ(全結合ネットワーク)
    '''

    def __init__(self, in_size, out_size, activation=F.sigmoid, batch_norm=True,
                 device=C_.DEVICE, **kwargs):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size

        if type(activation) is tuple:
            self.activation_e = activation[0]
            self.activation_d = activation[1]
        else:
            self.activation_e = activation
            self.activation_d = activation

        self.batch_norm = batch_norm
        self.device = device
        # self.kwargs = kwargs
        self.n_latent = out_size
        self.in_shape = None
        self.init = False
        self.maybe_init(in_size)

    def __call__(self, x, **kwargs):
        h = self.encode(x, **kwargs)
        y = self.decode(h, **kwargs)
        return y

    def encode(self, x, **kwargs):
        x = self.adapt(x)

        if DEBUG0:
            print(x.shape)
            print(self.in_size)

        self.in_shape = x.shape[1:]
        self.maybe_init(self.in_shape)

        x_ = x.reshape(-1, self.in_size)
        y = self.enc(x_)

        if self.activation_e:
            if self.batch_norm:
                y = self.bne(y)
            y = self.activation_e(y)

        if kwargs.get('show_shape'):
            print(f'layer(E{self.name}): in: {x.shape} out: {y.shape}')
        return y

    def decode(self, x, **kwargs):
        x = self.adapt(x)
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
        elif in_size_ is None:
            return

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
        self.adjust(device=self.device)


class CAEChain(chainer.Chain, AEBase):
    ''' 単層エンコーダ+デコーダ(畳み込みネットワーク)
    引数:
        in_channels
        out_channels
        use_indices(bool): maxpoolのインデックス情報をキャッシュ
    '''

    def __init__(self, in_channels, out_channels, ksize=5, padding=True,
                 activation=F.sigmoid, use_indices=False, batch_norm=True,
                 device=C_.DEVICE, **kwargs):
        super().__init__()

        self.out_channels = out_channels
        self.ksize = ksize
        self.padding = padding

        if type(activation) is tuple:
            self.activation_e = activation[0]
            self.activation_d = activation[1]
        else:
            self.activation_e = activation
            self.activation_d = activation

        self.use_indices = use_indices
        self.batch_norm = batch_norm
        self.device = device
        self.n_latent = None
        self.init = False
        self.maybe_init(in_channels)

    def __call__(self, x, **kwargs):
        h = self.encode(x, **kwargs)
        y = self.decode(h, **kwargs)
        return y

    def encode(self, x, **kwargs):
        x = self.adapt(x)

        # if not kwargs.get('inference'):
        #     print(f'E{self.name} in:', F.min(x), F.max(x), ' '*10, end='\r')

        if not self.init:
            self.maybe_init(x.shape[1])

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

        self.n_latent = y.shape
        if kwargs.get('show_shape'):
            print(f'layer(E{self.name}): in: {x.shape} out: {y.shape}')
        return y

    def decode(self, x, **kwargs):
        x = self.adapt(x)

        # if not kwargs.get('inference'):
        #     print(f'D{self.name} in:', F.min(x), F.max(x), ' '*10, end='\r')

        if self.use_indices:
            if not x.shape[0] == self.indexes.shape[0]:
                self.indexes = self.xp.repeat(self.indexes,
                    x.shape[0]//self.indexes.shape[0], axis=0)
            h = F.upsampling_2d(x, self.indexes, ksize=2, outsize=self.insize)
        else:
            h = F.unpooling_2d(x, ksize=2, outsize=self.insize)

        y = self.dec(h)

        if self.activation_d:
            if self.batch_norm:
                y = self.bnd(y)
            y = self.activation_d(y)

        if kwargs.get('show_shape'):
            print(f'layer(D{self.name}): in: {x.shape} out: {y.shape}')
        return y

    def maybe_init(self, in_channels):
        if self.init:
            return
        elif in_channels is None:
            return

        if DEBUG1:
            print('maybe_init', in_channels)

        pad = self.ksize // 2 if self.padding else 0

        with self.init_scope():
            self.enc = L.Convolution2D(in_channels, self.out_channels,
                                       ksize=self.ksize, stride=1, pad=pad)
            self.dec = L.Deconvolution2D(self.out_channels, in_channels,
                                         ksize=self.ksize, stride=1, pad=pad)

            if self.batch_norm:
                self.bne = L.BatchRenormalization(self.out_channels)
                self.bnd = L.BatchRenormalization(in_channels)
            else:
                # self.bne = L.BatchNormalization(self.out_size)
                # self.bnd = L.BatchNormalization(in_size)
                self.bne = None
                self.bnd = None

        self.in_channels = in_channels
        self.init = True
        self.adjust()


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
