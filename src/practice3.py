#! /usr/bin/env python3

'''
chainer練習用スクリプト
3層畳み込みネットワークによるcifarの学習
事前学習を取り入れる
2018.10.29 作成
'''

import argparse
import os
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

FILENAME = os.path.splitext(os.path.basename(__file__))[0]

DEVICE = 0

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
        y = F.relu(self.enc(x.reshape(-1, self.in_size)))
        return y

    def decode(self, x):
        y = self.activation(self.dec(x))
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

def train_model(model, train_iter, valid_iter, epoch=10, out=f'result/{FILENAME}'):
    # ネットワークをClassifierで包んで、ロスの計算などをモデルに含める
    # L.Classifierはpredictorというattributeに持ち、ロス計算を行う機能を追加する
    learner = L.Classifier(model, lossfun=F.mean_squared_error)
    learner.compute_accuracy = False

    # 最適化手法の選択
    optimizer = Adam().setup(learner)

    # print(model.update_enabled, model.xp)
    for m in model[:-1]:
        m.disable_update()
    # for m in model:
    #     print(m.update_enabled, m.xp)
    #     print(m.conv.xp)
    # exit()

    # Updaterの準備 (パラメータを更新)
    updater = StandardUpdater(train_iter, optimizer, device=DEVICE)

    # Trainerの準備
    trainer = Trainer(updater, stop_trigger=(epoch, 'epoch'), out=out)

    # TrainerにExtensionを追加する
    ## 検証
    trainer.extend(extensions.Evaluator(valid_iter, learner, device=DEVICE), name='val')

    ## モデルパラメータの統計を記録する
    trainer.extend(extensions.ParameterStatistics(learner.predictor[-1], {'std': np.std}))
    # trainer.extend(extensions.ParameterStatistics(learner.predictor[-2].conv, {'std': np.std}))

    ## 学習率を記録する
    trainer.extend(extensions.observe_lr())

    ## 学習経過を画面出力
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'val/main/loss', 'elapsed_time', 'lr']))

    ## プログレスバー
    trainer.extend(extensions.ProgressBar())

    ## ログ記録 (他のextensionsの結果も含まれる)
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.LogReport(['main/loss', 'val/main/loss'], log_name='loss'))

    ## 学習経過を画像
    trainer.extend(extensions.PlotReport(['enc/W/data/std', 'enc/b/data/std',
                                          'dec/W/data/std', 'dec/b/data/std'],
                                         x_key='epoch', file_name='std.png'))
    trainer.extend(extensions.PlotReport(['main/loss', 'val/main/loss'],
                                         x_key='epoch', file_name='loss.png'))

    ## ネットワーク形状をdot言語で出力
    ## 可視化コード: ```dot -Tpng cg.dot -o [出力ファイル]```
    trainer.extend(extensions.dump_graph('main/loss'))

    ## トレーナーオブジェクトをシリアライズし、出力ディレクトリに保存
    trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}.model'))

    def f_(trainer):
        print(trainer.observation)
    trainer.extend(f_)

    # 学習を開始する
    trainer.run()


def get_train_data():
    # データセットの準備
    train_val, test = cifar.get_cifar10() # => (50000, 2), (10000, 2)

    # データセットの修正 (教師データを学習データで置き換え)
    # testは使わない
    train_val_data = [x[0] for x in train_val]
    train_val = TupleDataset(train_val_data, train_val_data)

    # Validation用データセットを作る
    train_size = int(len(train_val) * 0.9)
    train, valid = split_dataset_random(train_val, train_size, seed=0)

    # Iteratorの作成
    # SerialIteratorはデータセットの中のデータを順番に取り出してくる
    batchsize = 128
    train_iter = SerialIterator(train, batchsize)
    valid_iter = SerialIterator(valid, batchsize, repeat=False, shuffle=False)
    test_iter = SerialIterator(test, batchsize, repeat=False, shuffle=False)

    return train_iter, valid_iter, test_iter


def get_model():
    # ネットワークの定義
    model = CAEList(CAEChain(3, 16),
                    CAEChain(16, 32),
                    CAEChain(32, 64),
                    CAEChain(64, 128),
                    LAEChain(128, 10))
    return model


def check_inputsize(model, data):
    # model = get_model()
    # data = model.xp.zeros((1, 3, 32, 32), dtype=model.xp.float32)
    print(f'in:', data.shape)
    for i, m in enumerate(model):
        data = model.encode(data)
        print(f'out({i}):', data.shape)


################################################################################

def sample0():
    ''' 一括学習 '''
    model = get_model()
    train_iter, valid_iter, test_iter = get_train_data()
    train_model(model, train_iter, valid_iter, epoch=120, out=f'result/{FILENAME}/all')


def sample01():
    ''' 逐次学習 '''
    train_iter, valid_iter, test_iter = get_train_data()
    model_list = get_model()
    model = CAEList()
    for i, m in enumerate(model_list):
        print('STEP:', i+1)
        model.append(m)
        if i < 3:
            file = f'result/{FILENAME}/step3/snapshot_epoch-30.model'
            path = f'updater/model:main/predictor/{i}/'
            chainer.serializers.load_npz(file, m, path)
        else:
            train_model(model, train_iter, valid_iter, epoch=30, out=f'result/{FILENAME}/step{i+1}')

        if i < len(model_list) - 1:
            train_iter.reset()
            valid_iter.reset()


def sample1(model_path=f'result/{FILENAME}/step2/snapshot_epoch-10.model'):
    # データセットの準備
    model = get_model()
    train_iter, valid_iter, test_iter = get_train_data()

    # モデル読み込み
    chainer.serializers.load_npz(model_path, model,
                                 path='updater/model:main/predictor/')

    x, t = next(test_iter)[1]

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

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(x_array[0].transpose(1, 2, 0), vmin=0, vmax=1)
    axes[1].imshow(y_array[0].transpose(1, 2, 0), vmin=0, vmax=1)
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
    file = f'result/{FILENAME}/step3/snapshot_epoch-30.model'
    with np.load(file) as npz:
        for f in npz.files:
            print(f)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs='?', default='',
                        choices=['', '0', '01', '1'],
                        help='Number of main procedure')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--test', '-t', action='store_true',
                        help='Run as test mode')
    args = parser.parse_args()
    return args


def main():
    global DEVICE

    args = get_args()
    DEVICE = args.gpu
    if DEVICE >= 0:
        chainer.cuda.get_device_from_id(0).use()

    if args.test:
        __test__()
        return
    if args.mode == '0':
        sample0()
    if args.mode == '01':
        sample01()
    elif args.mode == '1':
        sample1()


if __name__ == '__main__':
    main()
