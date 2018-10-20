#! /usr/bin/env python3

'''
chainer練習用スクリプト
3層畳み込みネットワークによるcifarの学習
2018.10.19 作成
参考: https://qiita.com/mitmul/items/1e35fba085eb07a92560
'''

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.datasets import mnist
from chainer.datasets import cifar

FILENAME = os.path.splitext(os.path.basename(__file__))[0]

################################################################################

class MyModel(chainer.Chain):
    def __init__(self, n_mid_units=100, n_out=10):
        super().__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 32, 3, 3, 1)
            self.conv2 = L.Convolution2D(32, 64, 3, 3, 1)
            self.conv3 = L.Convolution2D(64, 128, 3, 3, 1)
            self.fc4 = L.Linear(None, 1000)
            self.fc5 = L.Linear(1000, n_out)

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.fc4(h))
        h = self.fc5(h)
        return h


class WrapperModel(chainer.Chain):
    ''' Point!
    with self.init_scope()内にあるLink以外はパラメータが更新されない
    '''
    def __init__(self, model):
        super().__init__()
        self.submodel = model

    def __call__(self, x):
        return self.submodel(x)

################################################################################

def sample0(gpu_id=-1):
    # データセットの準備
    train_val, test = cifar.get_cifar10() # => (50000, 2), (10000, 2)

    # Validation用データセットを作る
    train_size = int(len(train_val) * 0.9)
    train, valid = chainer.datasets.split_dataset_random(
        train_val, train_size, seed=0)

    # Iteratorの作成
    # SerialIteratorはデータセットの中のデータを順番に取り出してくる
    batchsize = 128
    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    valid_iter = chainer.iterators.SerialIterator(
        valid, batchsize, repeat=False, shuffle=False)
    test_iter = chainer.iterators.SerialIterator(
        test, batchsize, repeat=False, shuffle=False)

    # ネットワークの定義
    model = MyModel()
    if gpu_id >= 0:
        model.to_gpu(gpu_id)

    # ネットワークをClassifierで包んで、ロスの計算などをモデルに含める
    # L.Classifierはpredictorというattributeに持ち、ロス計算を行う機能を追加する
    model = L.Classifier(model)

    # 最適化手法の選択
    optimizer = chainer.optimizers.Adam( # default
        alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-08, eta=1.0).setup(model)

    # Updaterの準備 (パラメータを更新)
    updater = chainer.training.StandardUpdater(
       train_iter, optimizer, device=gpu_id)

    # Trainerの準備
    max_epoch = 10
    trainer = chainer.training.Trainer(
        updater, stop_trigger=(max_epoch, 'epoch'), out=f'result/{FILENAME}')

    # TrainerにExtensionを追加する
    trainer.extend(chainer.training.extensions.LogReport())

    # トレーナーオブジェクトをシリアライズし、出力ディレクトリに保存
    trainer.extend(chainer.training.extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))

    # trainer.extend(chainer.training.extensions.Evaluator(valid_iter, model, device=gpu_id), name='val')
    # trainer.extend(chainer.training.extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'val/main/loss', 'val/main/accuracy', 'l1/W/data/std', 'elapsed_time']))
    # trainer.extend(chainer.training.extensions.ParameterStatistics(model.predictor.l1, {'std': np.std}))
    # trainer.extend(chainer.training.extensions.PlotReport(['l1/W/data/std'], x_key='epoch', file_name='std.png'))
    # trainer.extend(chainer.training.extensions.PlotReport(['main/loss', 'val/main/loss'], x_key='epoch', file_name='loss.png'))
    # trainer.extend(chainer.training.extensions.PlotReport(['main/accuracy', 'val/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
    # trainer.extend(chainer.training.extensions.dump_graph('main/loss'))

    trainer.extend(chainer.training.extensions.observe_lr())
    trainer.extend(chainer.training.extensions.Evaluator(valid_iter, model, device=gpu_id), name='val')
    trainer.extend(chainer.training.extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'val/main/loss', 'val/main/accuracy', 'elapsed_time', 'lr']))
    trainer.extend(chainer.training.extensions.PlotReport(['main/loss', 'val/main/loss'], x_key='epoch', file_name='loss.png'))
    trainer.extend(chainer.training.extensions.PlotReport(['main/accuracy', 'val/main/accuracy'], x_key='epoch', file_name='accuracy.png'))

    ''' extensionを自作 '''
    def myextension(trainer):
        print(trainer)
    myextension.trigger = (1, 'epoch')

    trainer.extend(myextension)


    # 学習を開始する
    trainer.run()

    # テストデータで評価する
    test_evaluator = chainer.training.extensions.Evaluator(
        test_iter, model, device=gpu_id)
    results = test_evaluator()
    print('Test accuracy:', results['main/accuracy'])


def sample1(gpu_id=-1):
    # データセットの準備
    train_val, test = cifar.get_cifar10(withlabel=True, ndim=1)

    model = MyModel()
    chainer.serializers.load_npz(
        f'result/{FILENAME}/snapshot_epoch-10',
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

def __test__():
    # ex = chainer.training.extensions.PrintReport([])
    # print(dir(ex))
    # print(ex.trigger)
    self.at = 'at'
    print(dir(__test__))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs='?', default='', choices=['', '0', '1'],
                        help='Number of main procedure')
    parser.add_argument('-test',  action='store_true',
                        help='Run as test mode')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    if args.test:
        __test__()
        return
    if args.mode == '0':
        sample0(gpu_id=0)
    elif args.mode == '1':
        sample1(gpu_id=0)


if __name__ == '__main__':
    main()
