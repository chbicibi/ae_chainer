import argparse
import glob
import os
import shutil
import sys

import numpy as np
# import matplotlib.pyplot as plt

import chainer
# import chainer.functions as F
# import chainer.links as L
# from chainer.datasets import mnist, cifar, split_dataset_random
# from chainer.datasets.tuple_dataset import TupleDataset
# from chainer.iterators import SerialIterator
# from chainer.optimizers import Adam
# from chainer.training import StandardUpdater, Trainer, extensions

import myutils as ut

import common as C_
import dataset as D_
import model as M_
import vis as V_


# パス
SRC_DIR = os.path.dirname(__file__)
SRC_FILE = os.path.basename(__file__)
SRC_FILENAME = os.path.splitext(SRC_FILE)[0]


################################################################################

def task0(key, out):
    ''' オートエンコーダ学習 '''

    # 学習パラメータ定義
    epoch = 200
    batchsize = 50
    out = f'{out}/res_{ut.snow}'

    # 学習データ作成
    # key = 'plate_00'
    cache_path = f'__cache__/{key}'
    original_data = D_.get_original_data(key, size=2000) # (128, 256)
    train_data = D_.get_train_data(D_.extract_uv_sq_norm, original_data,
                                   name='sq_norm', cache=True,
                                   cache_path=cache_path)

    # 学習モデル作成
    sample = C_.xp.array(train_data[:1])
    model = M_.get_model('case0', sample=sample)
    train_iter, valid_iter, _ = M_.get_cfd_train_data(train_data, table=None,
                                                      batchsize=batchsize)

    M_.train_model(model, train_iter, valid_iter, epoch=epoch, out=out)


def case0(*args, **kwargs):
    print(sys._getframe().f_code.co_name)

    keys = 'plate_10', 'wing_00', 'plate_20', 'wing_15', 'plate_30', 'wing_05'
    out = f'result/case0'

    for key in keys:
        task0(key, out)


################################################################################

def sample0(out=f'result', clear=False):
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


def sample01(out=f'result', clear=False):
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


def sample1(model_path=f'result/step5/snapshot_epoch-30.model'):
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
    pass


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs='?', default='',
                        choices=['', '0', '1', '2'],
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

    # if args.gpu:
    #     C_.DEVICE = args.gpu

    C_.SHOW_PROGRESSBAR = not args.no_progress

    # out = args.out
    out = f'result/{SRC_FILENAME}'
    clear = args.clear

    if args.test:
        __test__()

    elif args.mode == '0':
        with ut.stopwatch('case0'):
            case0(args)

    # if args.mode == '01':
    #     with ut.stopwatch('sample01'):
    #         sample01(out=out, clear=clear)
    # elif args.mode == '1':
    #     model_path = ut.fsort(glob.glob(f'{out}/all/*.model'))[-1]
    #     print(model_path)
    #     sample1(model_path=model_path)


if __name__ == '__main__':
    sys.exit(main())

'''
GTX 760
  828,407,808 bytes
1,973,098,496 bytes

GTX 1070
 ,828,407,808 bytes
7,564,598,272 bytes
'''