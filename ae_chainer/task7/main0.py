import argparse
import glob
import itertools
import os
import shutil
import sys
import traceback
from operator import itemgetter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as plc

import chainer
import chainer.functions as F
from chainer.iterators import SerialIterator

import myutils as ut

import common as C_
import dataset as D_
import net_vae as NV_
import model as M_
import vis as V_


# パス
SRC_DIR = os.path.dirname(__file__)
SRC_FILE = os.path.basename(__file__)
SRC_FILENAME = os.path.splitext(SRC_FILE)[0]


################################################################################

def loop(data):
    while True:
        yield data


def tapp(data, fn=None):
    if fn:
        print(*fn(data))
    else:
        print(data)
    return data


################################################################################

def get_extract(key):
    # vmin, vmax = {
    #     'plate_00': (-0.668, 1.261),
    #     'plate_10': (-1.329, 1.963),
    #     'plate_20': (-2.096, 2.726),
    #     'plate_30': (-2.289, 3.797),
    #     'wing_00': (-0.759, 1.444),
    #     'wing_05': (-1.041, 1.558),
    #     'wing_10': (-1.218, 1.728),
    #     'wing_15': (-1.617, 2.181),
    #     'wing_20': (-1.847, 2.842),
    #     'wing_30': (-2.192, 3.760),
    # }[key]
    vmin, vmax = -3.8, 3.8
    return D_.extract_uvf_norm(vmin, vmax)


def get_it(size):
    def g_(key):
        print('create data:', key)
        cache_path = f'__cache__/{key}'
        original_data = D_.get_original_data(key, size=2000) # (2000, 512, 1024, 5)

        train_data = D_.get_train_data(get_extract(key), original_data,
                                       name='full_norm_uvf_38', cache=True,
                                       cache_path=cache_path)
        a = train_data[2000-size:2000]
        return a

    src = '..\\__cache__'
    dst = '__cache__'
    if not os.path.isdir(dst) and os.path.isdir(src):
        print('symlink:', src, '<<===>>', dst)
        os.symlink(src, dst)
    return g_


def get_task_data(casename, batchsize=1):
    ''' 学習データ作成
    key == 'wing'
    '''

    # 学習データ作成
    keys = ('wing_00', 'wing_10', 'wing_20', 'wing_30',
            'plate_00', 'plate_10', 'plate_20', 'plate_30')
    # keys = ('wing_00',)
    train_data = D_.MapChain(crop_random_sq, *map(get_it(200), keys),
                             name='random_crop')
    train = TrainDataset(train_data)
    train_iter = SerialIterator(train, batchsize)

    # 検証データ作成
    keys = 'wing_05', 'wing_15'
    valid_data = D_.MapChain(crop_random_sq, *map(get_it(100), keys),
                             name='random_crop')
    valid = TrainDataset(valid_data)
    valid_iter = SerialIterator(valid, batchsize, repeat=False, shuffle=False)

    # 学習モデル作成
    sample = train_data[:1]
    model = M_.get_model(casename, sample=sample)

    return model, train_data, train_iter, valid_iter


def check_snapshot(out, show=False):
    # モデルのパスを取得

    respath = ut.select_file(out, key=r'res_.*', idx=None)
    print('path:', respath)
    file = ut.select_file(respath, key=r'snapshot_.*', idx=-1)
    print('file:', file)

    if show:
        # npz保存名確認
        with np.load(file) as npzfile:
            for f in npzfile:
                print(f)
                continue
                if f[-1] == 'W':
                    print(f)
                    print(npzfile[f].shape)
                # print(npzfile[f].dtype, npzfile[f].shape, f)
            # print(npzfile['extensions/LogReport/_log'])
    return file


################################################################################
# データを加工(学習)
################################################################################

def crop_random_sq(frame):
    ''' frame: (C, H, W)
    (:, 512, 1024) => (:, 384, 384)
    random_range = 0:128, 0:640
    '''
    size = 384
    p = [np.random.randint(r - size) for r in frame.shape[1:]]
    return frame[:, p[0]:p[0]+size, p[1]:p[1]+size]
    # if np.random.rand() < 0.5:
    #     s = slice(p[0], p[0]+size, 1)
    # else:
    #     s = slice(p[0]+size-1, p[0]-1 if p[0] else None, -1)
    # a = frame[:, s, p[1]:p[1]+size]
    # # print('I:', np.min(a), np.max(a))
    # return a


def crop_center_sq(frame):
    ''' frame: (C, H, W)
    (:, 512, 1024) => (:, 384, 384)
    '''
    size = 384
    p = [(r - size) // 2 for r in frame.shape[1:]]
    return frame[:, p[0]:p[0]+size, p[1]:p[1]+size]


def crop_front_sq(frame):
    ''' frame: (C, H, W)
    (:, 512, 1024) => (:, 384, 384)
    '''
    size = 384
    p = [(r - size) // 2 for r in frame.shape[1:]]
    p[1] = 128
    return frame[:, p[0]:p[0]+size, p[1]:p[1]+size]


class TrainDataset(chainer.dataset.DatasetMixin):
    def __init__(self, it):
        self.it = it

    def __len__(self):
        return len(self.it)

    def get_example(self, i):
        a = self.it[i]
        return a, a


################################################################################

def process0(casename, out):
    ''' オートエンコーダ学習 '''

    # 学習パラメータ定義
    epoch = 500
    batchsize = 50
    logdir = f'{out}/res_{casename}_{ut.snow}'
    model, _, train_iter, valid_iter = get_task_data(casename, batchsize)
    M_.train_model(model, train_iter, valid_iter, epoch=epoch, out=logdir)


def process0_resume(casename, out):
    ''' オートエンコーダ学習 '''

    # 学習パラメータ定義
    epoch = 500
    batchsize = 50
    logdir = f'{out}/res_{casename}_{ut.snow}'
    model, _, train_iter, valid_iter = get_task_data(casename, batchsize)
    init_file = check_snapshot(out)
    M_.train_model(model, train_iter, valid_iter, epoch=epoch, out=logdir,
                   init_file=init_file)


def task0(*args, **kwargs):
    ''' task0: 学習メイン '''

    casename = kwargs.get('case', None) or 'case9_0'
    out = f'__result__/{casename}'
    error = None

    try:
        if kwargs.get('resume'):
            process0_resume(casename, out)
        else:
            process0(casename, out)

    except Exception as e:
        error = e
        tb = traceback.format_exc()
        print('Error:', error)
        print(tb)

    if kwargs.get('sw', 0) < 3600:
        return

    with ut.EmailIO(None, 'ae_chainer: Task is Complete') as e:
        print(sys._getframe().f_code.co_name, file=e)
        print(ut.strnow(), file=e)
        if 'sw' in kwargs:
            print('Elapsed:', kwargs['sw'], file=e)
        if error:
            print('Error:', error)
            print(tb)


################################################################################

def __test__():
    with ut.chdir(f'{SRC_DIR}/__result__/case4_2'):
        path = ut.select_file('.', key=r'res_.*')
        with ut.chdir(path):
            log = ut.load('log.json', from_json=True)
            loss = [l['main/loss'] for l in log]
            plt.ylim((180000, 210000))
            plt.plot(np.array(loss))
            plt.show()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs='?', default='0',
                        choices=['', '0'],
                        help='Number of main procedure')
    parser.add_argument('--case', '-c', default='',
                        help='Training case name')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--no-progress', '-p', action='store_true',
                        help='Hide progress bar')

    # additional options
    parser.add_argument('--check-snapshot', '-s', action='store_true',
                        help='Print names in snapshot file')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='Resume with loading snapshot')

    # test option
    parser.add_argument('--test', '-t', action='store_true',
                        help='Run as test mode')
    args = parser.parse_args()
    return args


def main():
    global DEVICE, PROGRESSBAR

    args = get_args()

    if args.gpu:
        C_.DEVICE = args.gpu

    C_.SHOW_PROGRESSBAR = not args.no_progress

    # out = args.out
    out = f'result/{SRC_FILENAME}'

    if args.test:
        # print(vars(args))
        __test__()

    elif args.mode in '0123456789':
        taskname = 'task' + args.mode
        if taskname in globals():
            f_ = globals().get(taskname)
            with ut.stopwatch(taskname) as sw:
                f_(**vars(args), sw=sw)


if __name__ == '__main__':
    sys.exit(main())

'''
GTX 760
  828,407,808 bytes
1,973,098,496 bytes

GTX 1070
  828,407,808 bytes
7,564,598,272 bytes
'''
