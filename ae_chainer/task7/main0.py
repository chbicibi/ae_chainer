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
import main as MS_


# パス
SRC_DIR = os.path.dirname(__file__)
SRC_FILE = os.path.basename(__file__)
SRC_FILENAME = os.path.splitext(SRC_FILE)[0]


################################################################################

def get_task_data(casename, batchsize=1):
    ''' 学習データ作成
    key == 'wing'
    '''

    # 学習データ作成
    keys = ('wing_00', 'wing_10', 'wing_20', 'wing_30',
            'plate_00', 'plate_10', 'plate_20', 'plate_30')
    # keys = ('wing_00',)
    train_data = D_.MapChain(crop_random_sq, *map(MS_.get_it(200), keys),
                             name='random_crop')
    train = TrainDataset(train_data)
    train_iter = SerialIterator(train, batchsize)

    # 検証データ作成
    keys = 'wing_05', 'wing_15'
    valid_data = D_.MapChain(crop_random_sq, *map(MS_.get_it(100), keys),
                             name='random_crop')
    valid = TrainDataset(valid_data)
    valid_iter = SerialIterator(valid, batchsize, repeat=False, shuffle=False)

    # 学習モデル作成
    sample = train_data[:1]
    model = M_.get_model(casename, sample=sample)

    return model, train_data, train_iter, valid_iter


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
    epoch = 3000
    batchsize = 50
    logdir = f'{out}/res_{casename}_{ut.snow}'
    model, _, train_iter, valid_iter = get_task_data(casename, batchsize)
    M_.train_model(model, train_iter, valid_iter, epoch=epoch, out=logdir,
                   alpha=0.01)


def process0_resume(casename, out):
    ''' オートエンコーダ学習 '''

    # 学習パラメータ定義
    epoch = 3000
    batchsize = 50
    logdir = f'{out}/res_{casename}_{ut.snow}'
    init_file = MS_.check_snapshot(out)
    model, _, train_iter, valid_iter = get_task_data(casename, batchsize)
    M_.train_model(model, train_iter, valid_iter, epoch=epoch, out=logdir,
                   init_file=init_file, alpha=0.01)


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
    mu = np.zeros(3, dtype=np.float32)
    sigma = np.ones(3, dtype=np.float32)

    p_z = chainer.distributions.Normal(loc=mu, scale=sigma)
    q_z = chainer.distributions.Normal(loc=mu+10, scale=sigma)
    p_x = chainer.distributions.Bernoulli(logit=sigma-0.5)

    l_rec = p_x.log_prob(mu)
    d_kl = chainer.kl_divergence(q_z, p_z)
    print(l_rec)
    print(d_kl)


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
