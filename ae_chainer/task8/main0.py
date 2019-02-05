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
import chainer.iterators as I #import SerialIterator

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
    train_data = D_.MapChain(MS_.crop_random_sq,
                             *map(MS_.get_it(size=1250, cache=False), keys),
                             name='random_crop')
    train = MS_.TrainDataset(train_data, name='train')
    train_iter = I.MultithreadIterator(train, batchsize, n_threads=32)

    # 検証データ作成
    keys = 'wing_05', 'wing_15'
    valid_data = D_.MapChain(MS_.crop_random_sq,
                             *zip(*map(MS_.get_it(size=500, cache=True), keys)),
                             name='random_crop')
    valid = MS_.TrainDataset(valid_data, name='valid')
    valid_iter = I.MultithreadIterator(valid, batchsize, repeat=False, shuffle=False, n_threads=8)

    # 学習モデル作成
    sample = train_data[:1]
    model = M_.get_model(casename, sample=sample)

    return model, train_data, train_iter, valid_iter


################################################################################

def process0(casename, out):
    ''' オートエンコーダ学習 '''

    # 学習パラメータ定義
    epoch = 300
    batchsize = 50
    logdir = f'{out}/res_{casename}_{ut.snow}'
    model, _, train_iter, valid_iter = get_task_data(casename, batchsize)
    M_.train_model(model, train_iter, valid_iter, epoch=epoch, out=logdir,
                   alpha=0.01)


def process0_resume(casename, out, init_all=True, new_out=False):
    ''' オートエンコーダ学習 '''

    # 学習パラメータ定義
    epoch = 300
    batchsize = 50
    init_file = MS_.check_snapshot(out)
    if new_out:
        logdir = f'{out}/res_{casename}_{ut.snow}'
    else:
        logdir = os.path.dirname(init_file)
    model, _, train_iter, valid_iter = get_task_data(casename, batchsize)
    M_.train_model(model, train_iter, valid_iter, epoch=epoch, out=logdir,
                   init_file=init_file, alpha=0.01, init_all=init_all)


def task0(*args, **kwargs):
    ''' task0: 学習メイン '''

    casename = kwargs.get('case', None) or 'case9_0'
    out = f'__result__/{casename}'
    error = None

    try:
        resume = kwargs.get('resume', '')
        if resume:
            init_all = not resume.startswith('m')
            new_out = 'new' in resume
            process0_resume(casename, out, init_all=init_all, new_out=new_out)

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
    parser.add_argument('--resume', '-r', nargs='?', const='all', default=None,
                        choices=['', 'model', 'all', 'modelnew', 'allnew'],
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
