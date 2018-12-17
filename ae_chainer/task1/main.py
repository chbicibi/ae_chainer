#! /usr/bin/env python3

'''
chainer練習用スクリプト
3層畳み込みネットワークによる自作データ
大きな入力に対応できるか確認
2018.10.31 作成
'''

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
import dataset


FILENAME = os.path.splitext(os.path.basename(__file__))[0]

OS = os.environ.get('OS')
OS_WIN = OS == 'Windows_NT'
DEVICE = 0
PROGRESSBAR = True


################################################################################

class LAEChain(chainer.Chain):
    def __init__(self, in_size, hidden_size, activation=F.relu):
        super().__init__()
        with self.init_scope():
            self.enc = L.Linear(in_size, hidden_size)
            self.dec = L.Linear(hidden_size, in_size)
        self.in_size = in_size
        self.activation = activation or (lambda x: x)
        if DEVICE >= 0:
            self.to_gpu(DEVICE)

    def encode(self, x):
        self.insize = x.shape[1:]
        y = self.activation(self.enc(x.reshape(-1, self.in_size)))
        return y

    def decode(self, x):
        y = F.relu(self.dec(x))
        y = y.reshape(-1, *self.insize)
        return y


class CAEChain(chainer.Chain):
    def __init__(self, in_channels, hidden_channels, activation=F.relu,
                 use_indices=True):
        super().__init__()
        with self.init_scope():
            self.enc = L.Convolution2D(in_channels, hidden_channels,
                                        ksize=3, stride=1, pad=0)
            self.dec = L.Deconvolution2D(hidden_channels, in_channels,
                                         ksize=3, stride=1, pad=0)
        self.activation = activation or (lambda x: x)
        self.use_indices = use_indices
        if DEVICE >= 0:
            self.to_gpu(DEVICE)

    def __call__(self, x):
        h = self.encode(x)
        y = self.decode(h)
        return y

    def encode(self, x):
        h = F.relu(self.enc(x))
        self.insize = h.shape[2:]
        if self.use_indices:
            y, self.indexes = F.max_pooling_2d(h, ksize=2, return_indices=True)
        else:
            y = F.max_pooling_2d(h, ksize=2)
        # print('encode in:', x.shape)
        # print('encode enc:', h.shape)
        # print('encode out:', y.shape)
        return y

    def decode(self, x):
        if self.use_indices:
            h = F.upsampling_2d(x, self.indexes, ksize=2, outsize=self.insize)
        else:
            h = F.unpooling_2d(x, ksize=2, outsize=self.insize)
        y = self.activation(self.dec(h))
        # print('decode in:', x.shape)
        # print('decode enc:', h.shape)
        # print('decode out:', y.shape)
        return y


class CAEList(chainer.ChainList):
    def __init__(self, *links):
        super().__init__(*links)
        if DEVICE >= 0:
            self.to_gpu(DEVICE)

    def __call__(self, x):
        h = self.encode(x)
        y = self.decode(h)
        return y

    def encode(self, x):
        y = reduce(lambda h, l: l.encode(h), self, x)
        return y

    def decode(self, x):
        y = reduce(lambda h, l: l.decode(h), reversed(self), x)
        return y


################################################################################

class TestDataset(chainer.dataset.DatasetMixin):
    def __init__(self, it):
        self.it = it
        self._len = len(it) // 4
        self._data = [None] * self._len

    def __len__(self):
        return self._len

    def get_example(self, ix):
        i = ix // 4
        if self._data[i] is None:
            self._data[i] = self.make_data(self.it[i])
        return self._data[i], self._data[i]

    def make_data(self, data_org):
        ''' data_org => (u, v, _, _, _)
        '''
        a = data_org[::4, ::4, 0:2] # 間引き
        return np.transpose(a, (2, 0, 1))


################################################################################

def train_model(model, train_iter, valid_iter, epoch=10, out=f'result/{FILENAME}', fix_trained=False):
    # ネットワークをClassifierで包んで、ロスの計算などをモデルに含める
    # L.Classifierはpredictorというattributeに持ち、ロス計算を行う機能を追加する
    learner = L.Classifier(model, lossfun=F.mean_squared_error)
    learner.compute_accuracy = False

    # 最適化手法の選択
    optimizer = Adam().setup(learner)

    if fix_trained:
        for m in model[:-1]:
            m.disable_update()

    # Updaterの準備 (パラメータを更新)
    updater = StandardUpdater(train_iter, optimizer, device=DEVICE)

    # Trainerの準備
    trainer = Trainer(updater, stop_trigger=(epoch, 'epoch'), out=out)

    # TrainerにExtensionを追加する
    ## 検証
    trainer.extend(extensions.Evaluator(valid_iter, learner, device=DEVICE), name='val')

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
    if PROGRESSBAR:
        trainer.extend(extensions.ProgressBar())

    ## ログ記録 (他のextensionsの結果も含まれる)
    trainer.extend(extensions.LogReport())
    # trainer.extend(extensions.LogReport(['main/loss', 'val/main/loss'],
    #                                     log_name='loss.json'))
    # trainer.extend(extensions.LogReport(observe_keys_std, log_name='std.json'))

    ## 学習経過を画像出力
    if OS_WIN:
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


def get_train_data():
    # データセットの準備
    # train_val, test = cifar.get_cifar10() # => (50000, 2), (10000, 2)

    # データセットの修正 (教師データを学習データで置き換え)
    # testは使わない
    # train_val_data = np.zeros((1000, 3, 320, 320), dtype=np.float32)
    # train_val = TupleDataset(train_val_data, train_val_data)
    data = dataset.CFDBase('plate_00', size=2000) # (128, 256)
    train_val = TestDataset(data)

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


def get_model():
    # ネットワークの定義
    # 入出力チャンネル数を指定
    model = CAEList(CAEChain(2, 16, activation=None),
                    CAEChain(16, 32),
                    CAEChain(32, 64),
                    CAEChain(64, 128),
                    LAEChain(128, 64),
                    LAEChain(64, 10, activation=None))
    return model


def check_inputsize(model, data):
    # model = get_model()
    # data = model.xp.zeros((1, 3, 32, 32), dtype=model.xp.float32)
    print(f'in:', data.shape)
    for i, m in enumerate(model):
        data = model.encode(data)
        print(f'out({i}):', data.shape)


################################################################################

def sample0(out=f'result/{FILENAME}', clear=False):
    ''' 一括学習 '''
    epoch = 100

    model = get_model()
    train_iter, valid_iter, test_iter = get_train_data()

    if clear:
        shutil.rmtree(out, ignore_errors=True)

    train_model(model, train_iter, valid_iter,
                epoch=epoch,
                out=f'{out}/all',
                fix_trained=False)


def sample01(out=f'result/{FILENAME}', clear=False):
    ''' 逐次学習 '''
    train_iter, valid_iter, test_iter = get_train_data()
    model_list = get_model()
    model = CAEList()

    if clear:
        shutil.rmtree(out, ignore_errors=True)

    for i, m in enumerate(model_list):
        print('STEP:', i + 1)
        model.append(m)
        file = f'{out}/step{i+1}/snapshot_epoch-50.model'
        if os.path.isfile(file):
            print('Loading model')
            path = f'updater/model:main/predictor/{i}/'
            chainer.serializers.load_npz(file, m, path)
        else:
            train_model(model, train_iter, valid_iter,
                        epoch=50,
                        out=f'{out}/step{i+1}',
                        fix_trained=False)

        if i < len(model_list) - 1:
            train_iter.reset()
            valid_iter.reset()


def sample1(model_path=f'result/{FILENAME}/step5/snapshot_epoch-30.model'):
    # データセットの準備
    model = get_model()
    train_iter, valid_iter, test_iter = get_train_data()

    # モデル読み込み
    chainer.serializers.load_npz(model_path, model,
                                 path='updater/model:main/predictor/')

    test_batch = next(test_iter)

    fig, axes = plt.subplots(10, 2)

    for i in range(10):
        x, t = test_batch[i]

        # ネットワークと同じデバイス上にデータを送る
        x = model.xp.asarray(x[None, ...])

        with chainer.using_config('train', False), \
             chainer.using_config('enable_backprop', False):
            y = model(x)

        if DEVICE >= 0:
            # 結果をCPUに送る
            x_array = chainer.cuda.to_cpu(x)
            y_array = chainer.cuda.to_cpu(y.array)
        else:
            x_array = x
            y_array = y.array

        axes[i][0].imshow(x_array[0].transpose(1, 2, 0), vmin=0, vmax=1)
        axes[i][1].imshow(y_array[0].transpose(1, 2, 0), vmin=0, vmax=1)
    plt.show()


def predict(model, image_id):
    ''' 学習済みモデルで推論する '''
    _, test = cifar.get_cifar10()
    x, t = test[image_id]
    model.to_cpu()
    with chainer.using_config('train', False), \
         chainer.using_config('enable_backprop', False):
        y = model.predictor(x[None, ...]).data.argmax(axis=1)[0]
    print('predicted_label:', cls_names[y])
    print('answer:', cls_names[t])

    plt.imshow(x.transpose(1, 2, 0))
    plt.show()


################################################################################

def __test__():
    shutil.rmtree('nuimnum', ignore_errors=True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs='?', default='',
                        choices=['', '0', '01', '1'],
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
    DEVICE = args.gpu
    # if DEVICE >= 0:
    #     chainer.cuda.get_device_from_id(0).use()
    PROGRESSBAR = not args.no_progress

    # out = args.out
    out = f'result/{FILENAME}'
    clear = args.clear

    if args.test:
        __test__()
        return
    if args.mode == '0':
        with ut.stopwatch('sample0'):
            sample0(out=out, clear=clear)
    if args.mode == '01':
        with ut.stopwatch('sample01'):
            sample01(out=out, clear=clear)
    elif args.mode == '1':
        model_path = ut.fsort(glob.glob(f'{out}/all/*.model'))[-1]
        print(model_path)
        sample1(model_path=model_path)


if __name__ == '__main__':
    main()

'''
GTX 760
  828,407,808 bytes
1,973,098,496 bytes

GTX 1070
 ,828,407,808 bytes
7,564,598,272 bytes
'''