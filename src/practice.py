#! /usr/bin/env python3

'''
学習に使うデータを取得する
参考: https://qiita.com/mitmul/items/1e35fba085eb07a92560
'''

import argparse
import numpy as np
import matplotlib.pyplot as plt

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.datasets import mnist


################################################################################

class AutoEncoder(chainer.Chain):
    def __init__(self, n_mid_units=100, n_out=10):
        super().__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_mid_units)
            self.l2 = L.Linear(n_mid_units, n_mid_units)
            self.l3 = L.Linear(n_mid_units, n_out)

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


    def forward(self, x):
        pass


################################################################################

def sample0():
    # データセットの準備
    train_val, test = mnist.get_mnist(withlabel=True, ndim=1)

    # Validation用データセットを作る
    train, valid = chainer.datasets.split_dataset_random(train_val, 50000, seed=0)

    # Iteratorの作成
    # SerialIteratorはデータセットの中のデータを順番に取り出してくる
    batchsize = 128
    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    valid_iter = chainer.iterators.SerialIterator(
        valid, batchsize, repeat=False, shuffle=False)
    test_iter = chainer.iterators.SerialIterator(
        test, batchsize, repeat=False, shuffle=False)

    # ネットワークの定義
    gpu_id = -1
    model = AutoEncoder()
    if gpu_id >= 0:
        model.to_gpu(gpu_id)

    # ネットワークをClassifierで包んで、ロスの計算などをモデルに含める
    # L.Classifierはpredictorというattributeに持ち、ロス計算を行う機能を追加する
    model = L.Classifier(model)

    # 最適化手法の選択
    optimizer = chainer.optimizers.Adam( # default
        alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-08, eta=1.0).setup(model)

    # Updaterの準備 (パラメータを更新)
    updater = chainer.training.StandardUpdater(train_iter, optimizer, device=gpu_id)

    # Trainerの準備
    max_epoch = 10
    trainer = chainer.training.Trainer(
        updater, (max_epoch, 'epoch'), out='mnist_result')

    # TrainerにExtensionを追加する
    trainer.extend(chainer.training.extensions.LogReport())
    trainer.extend(chainer.training.extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
    trainer.extend(chainer.training.extensions.Evaluator(valid_iter, model, device=gpu_id), name='val')
    trainer.extend(chainer.training.extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'val/main/loss', 'val/main/accuracy', 'l1/W/data/std', 'elapsed_time']))
    trainer.extend(chainer.training.extensions.ParameterStatistics(model.predictor.l1, {'std': np.std}))
    trainer.extend(chainer.training.extensions.PlotReport(['l1/W/data/std'], x_key='epoch', file_name='std.png'))
    trainer.extend(chainer.training.extensions.PlotReport(['main/loss', 'val/main/loss'], x_key='epoch', file_name='loss.png'))
    trainer.extend(chainer.training.extensions.PlotReport(['main/accuracy', 'val/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
    trainer.extend(chainer.training.extensions.dump_graph('main/loss'))

    # 学習を開始する
    trainer.run()

    # テストデータで評価する
    test_evaluator = chainer.training.extensions.Evaluator(
        test_iter, model, device=gpu_id)
    results = test_evaluator()
    print('Test accuracy:', results['main/accuracy'])


def sample1(gpu_id=-1):
    # データセットの準備
    train_val, test = mnist.get_mnist(withlabel=True, ndim=1)

    model = AutoEncoder()
    chainer.serializers.load_npz(
        'mnist_result/snapshot_epoch-10',
        model, path='updater/model:main/predictor/')

    if gpu_id >= 0:
        model.to_gpu(gpu_id)

    x, t = test[0]
    # plt.imshow(x.reshape(28, 28), cmap='gray')
    # plt.show()

    x = model.xp.asarray(x[None, ...])
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        y = model(x)

    # y_array = chainer.cuda.to_cpu(y.array)
    y_array = y.array

    print('予測ラベル:', y_array.argmax(axis=1)[0])


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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', default='0',
                        help='program mode')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    if args.mode == '0':
        sample0()
    elif args.mode == '1':
        sample1()


if __name__ == '__main__':
    main()
