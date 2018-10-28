#! /usr/bin/env python3

'''
chainer練習用スクリプト
3層畳み込みネットワークによるcifarの学習
事前学習を取り入れる
2018.10.27 作成
'''

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.datasets import mnist, cifar, split_dataset_random
from chainer.datasets.tuple_dataset import TupleDataset
from chainer.iterators import SerialIterator
from chainer.optimizers import Adam
from chainer.training import extensions, StandardUpdater, Trainer

FILENAME = os.path.splitext(os.path.basename(__file__))[0]

################################################################################

class CAEChain(chainer.Chain):
    def __init__(self, in_channels, hidden_channels, activation=F.relu,
                 use_indices=True):
        super().__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(in_channels, hidden_channels,
                                        ksize=3, stride=1, pad=0)
            self.deconv = L.Deconvolution2D(hidden_channels, in_channels,
                                            ksize=3, stride=1, pad=0)
        self.activation = activation or (lambda x: x)
        self.use_indices = use_indices

    def __call__(self, x):
        h = self.encode(x)
        y = self.decode(h)
        return y

    def encode(self, x):
        h = F.relu(self.conv(x))
        self.insize = h.shape[2:]
        if self.use_indices:
            y, self.indexes = F.max_pooling_2d(h, ksize=2, return_indices=True)
        else:
            y = F.max_pooling_2d(h, ksize=2)
        # print('encode in:', x.shape)
        # print('encode conv:', h.shape)
        # print('encode out:', y.shape)
        return y

    def decode(self, x):
        if self.use_indices:
            h = F.upsampling_2d(x, self.indexes, ksize=2, outsize=self.insize)
        else:
            h = F.unpooling_2d(x, ksize=2, outsize=self.insize)
        y = self.activation(self.deconv(h))
        # print('decode in:', x.shape)
        # print('decode conv:', h.shape)
        # print('decode out:', y.shape)
        return y


class CAEList(chainer.ChainList):
    def __init__(self, *links):
        super().__init__(*links)

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

def sample0(gpu_id=-1):
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

    # ネットワークの定義
    model = CAEList(CAEChain(3, 16), CAEChain(16, 32), CAEChain(32, 64))
    if gpu_id >= 0:
        model.to_gpu(gpu_id)

    # ネットワークをClassifierで包んで、ロスの計算などをモデルに含める
    # L.Classifierはpredictorというattributeに持ち、ロス計算を行う機能を追加する
    learner = L.Classifier(model, lossfun=F.mean_squared_error)
    learner.compute_accuracy = False

    # 最適化手法の選択
    optimizer = Adam().setup(learner)

    # Updaterの準備 (パラメータを更新)
    updater = StandardUpdater(train_iter, optimizer, device=gpu_id)

    # Trainerの準備
    max_epoch = 30
    trainer = Trainer(updater, stop_trigger=(max_epoch, 'epoch'),
                      out=f'result/{FILENAME}')

    # TrainerにExtensionを追加する
    ## 検証
    trainer.extend(extensions.Evaluator(valid_iter, learner, device=gpu_id), name='val')

    ## モデルパラメータの統計を記録する
    trainer.extend(extensions.ParameterStatistics(learner.predictor[0].conv, {'std': np.std}))

    ## 学習率を記録する
    trainer.extend(extensions.observe_lr())

    ## 学習経過を画面出力
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'val/main/loss', 'elapsed_time', 'lr']))

    ## プログレスバー
    trainer.extend(extensions.ProgressBar())

    ## ログ記録 (他のextensionsの結果も含まれる)
    trainer.extend(extensions.LogReport())

    ## 学習経過を画像
    trainer.extend(extensions.PlotReport(['conv/W/data/std', 'conv/b/data/std'], x_key='epoch', file_name='std.png'))
    trainer.extend(extensions.PlotReport(['main/loss', 'val/main/loss'], x_key='epoch', file_name='loss.png'))

    ## ネットワーク形状をdot言語で出力
    ## 可視化コード: ```dot -Tpng cg.dot -o [出力ファイル]```
    trainer.extend(extensions.dump_graph('main/loss'))

    ## トレーナーオブジェクトをシリアライズし、出力ディレクトリに保存
    trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}.model'))

    # 学習を開始する
    trainer.run()


def sample1(gpu_id=-1, model_path=f'result/{FILENAME}/snapshot_epoch-10.model'):
    # データセットの準備
    train_val, test = cifar.get_cifar10()

    # モデル読み込み
    model = CAEList(CAEChain(3, 16), CAEChain(16, 32), CAEChain(32, 64))
    chainer.serializers.load_npz(model_path, model,
                                 path='updater/model:main/predictor/')
    if gpu_id >= 0:
        model.to_gpu(gpu_id)

    x, t = test[1]

    # ネットワークと同じデバイス上にデータを送る
    x = model.xp.asarray(x[None, ...])

    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        y = model(x)

    if gpu_id >= 0:
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
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        y = model.predictor(x[None, ...]).data.argmax(axis=1)[0]
    print('predicted_label:', cls_names[y])
    print('answer:', cls_names[t])

    plt.imshow(x.transpose(1, 2, 0))
    plt.show()


################################################################################

def __test__():
    pass


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs='?', default='', choices=['', '0', '1'],
                        help='Number of main procedure')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--test',  action='store_true',
                        help='Run as test mode')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    if args.test:
        __test__()
        return
    if args.mode == '0':
        sample0(gpu_id=args.gpu)
    elif args.mode == '1':
        sample1(gpu_id=args.gpu)


if __name__ == '__main__':
    main()
