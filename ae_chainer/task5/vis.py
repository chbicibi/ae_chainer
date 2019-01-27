import argparse
import math
import os
import shutil
import sys
from contextlib import contextmanager
from itertools import chain

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as plc

import myutils as ut


################################################################################

def extract_u(frame):
    return frame[:, :, 0]


def extract_v(frame):
    return frame[:, :, 1]


################################################################################

def remove_border(ax):
    for d in ['top', 'bottom', 'left', 'right']:
        ax.spines[d].set_visible(False)


def show_frame_vor(frame, file=None):
    fig, ax = plt.subplots()
    ax.tick_params(left=False, labelleft=False,
                   bottom=None, labelbottom=False)

    colors = [(0, '#ff0000'), (0.5, '#000000'), (1, '#00ff00')]
    cmap = plc.LinearSegmentedColormap.from_list('custom_cmap', colors)

    ax.imshow(frame, cmap=cmap, vmin=-0.1, vmax=0.1)
    plt.show()


def show_frame_m(frames, fig, axes, file=None):
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = np.array([axes])

    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.05, hspace=0.05)
    for ax in axes:
        ax.tick_params(left=False, labelleft=False,
                       bottom=None, labelbottom=False)
    # for ax in axes[len(frames):]:
        remove_border(ax)

    colors = [(0, '#000000'), (1, '#ffffff')]
    cmap = plc.LinearSegmentedColormap.from_list('custom_cmap', colors)

    for frame, ax in zip(frames, axes):
        if callable(frame):
            frame(ax)
        else:
            ax.imshow(frame, cmap=cmap, vmin=0)

    if isinstance(file, str):
        fig.savefig(file, aspect='auto', bbox_inches='tight', pad_inches=0)
    elif isinstance(file, float):
        plt.pause(file)
    else:
        plt.show()


def show_frame_filter_env(frames, tr=False, file=None):
    nbatch = frames.shape[0]
    ncols = math.floor(math.sqrt(nbatch))
    nrows = math.ceil(nbatch / ncols)
    if tr or ncols < 3:
        nrows, ncols = ncols, nrows
    figsize = (ncols*frames.shape[2]+max(0.1*ncols, 1)*(frames.shape[2]-1)+2,
               nrows*frames.shape[1]+max(0.1*nrows, 1)*(frames.shape[1]-1)+2)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=1)
    return fig, axes


def show_frame_filter(frames, tr=False, file=None):
    fig, axes = show_frame_filter_env(frames, tr=tr, file=file)
    return show_frame_m(frames, fig, axes, file=file)


def plot_gray(frame, ax):
    # colors = [(0, '#ff0000'), (0.5, '#000000'), (1, '#0000ff')]
    # cmap = plc.LinearSegmentedColormap.from_list('custom_cmap', colors)
    ax.imshow(frame, cmap='gray')


def plot_vel(frame, ax):
    colors = [(0, '#ff0000'), (0.5, '#000000'), (1, '#0000ff')]
    cmap = plc.LinearSegmentedColormap.from_list('custom_cmap', colors)
    ax.imshow(frame, cmap=cmap, vmin=0.1, vmax=0.9)


def plot_vor(frame, ax):
    colors = [(0, '#ff0000'), (0.5, '#000000'), (1, '#00ff00')]
    cmap = plc.LinearSegmentedColormap.from_list('custom_cmap', colors)
    ax.imshow(frame, cmap=cmap, vmin=-0.1, vmax=0.1)


def show_frame_uvo(frames, file=None):
    fig, axes = plt.subplots(nrows=1, ncols=3)
    data = list(map(lambda f, d: lambda ax: f(d, ax), (plot_vel, plot_vel, plot_vor), frames))
    return show_frame_m(data, fig, axes, file=file)


def show_frame_uvfo(frames, file=None):
    fig, axes = plt.subplots(nrows=1, ncols=len(frames))
    data = list(map(lambda f, d: lambda ax: f(d, ax),
                    (plot_vel, plot_vel, plot_gray, plot_vor),
                    frames))
    show_frame_m(data, fig, axes, file=file)
    plt.close(fig)


def show_frame(frame, exf=None, file=None):
    if frame.ndim == 2:
        return show_frame_vor(frame, file=file)
    elif frame.ndim == 3:
        if exf:
            data = [*frame, exf(frame)]
            return show_frame_uvfo(data, file=file)
        else:
            return show_frame_filter(frame, file=file)


################################################################################

def show_it(fn, it, vmin=-0.8, vmax=1.6):
    ''' 画面表示 '''

    color_list = [(0, 'blue'), (0.5, 'black'), (1, 'red')]
    cmap = plc.LinearSegmentedColormap.from_list('custom_cmap', color_list)

    fig, ax = plt.subplots()
    if hasattr(it, '__len__'):
        s = len(it)
    else:
        s = '-'

    for i, data in enumerate(it):
        ax.cla()
        a = fn(data)
        print(np.min(a), np.max(a))
        p = ax.imshow(a, cmap=cmap, vmin=vmin, vmax=vmax)
        if not i:
            fig.colorbar(p, orientation='horizontal')#, ticks=[vmin, 1, vmax])
        ax.annotate(f'{i}/{s}', xy=(1, 0), xycoords='axes fraction',
                    horizontalalignment='right', verticalalignment='bottom')
        plt.pause(0.01)


def show_it_m(fn, it, nrows=1, ncols=2, vmin=-0.8, vmax=1.6):
    ''' 画面表示 '''

    color_list = [(0, 'blue'), (0.5, 'black'), (1, 'red')]
    cmap = plc.LinearSegmentedColormap.from_list('custom_cmap', color_list)

    fig, axes = plt.subplots(nrows, ncols)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    elif axes.ndim > 1:
        axes = axes.reshape(-1)

    fig.subplots_adjust(left=0, bottom=0, right=1, top=1,
                        wspace=0.1, hspace=0.2)
    for ax in axes:
        ax.tick_params(left=False, labelleft=False,
                       bottom=None, labelbottom=False)

    if hasattr(it, '__len__'):
        s = len(it)
    else:
        s = '?'

    for i, data in enumerate(it):
        for ax, d in zip(axes, fn(data)):
            ax.cla()
            # p = ax.imshow(d, cmap=cmap, vmin=vmin, vmax=vmax)
            if callable(d):
                p = d(ax)
            else:
                p = ax.imshow(d, cmap=cmap)
        # if not i:
        #     fig.colorbar(p, orientation='horizontal')#, ticks=[vmin, 1, vmax])
        axes[-1].annotate(f'{i}/{s}', xy=(1, -0.05), xycoords='axes fraction',
                          horizontalalignment='right',
                          verticalalignment='top')
        if i % 5 == 0:
            fig.savefig(f'step{i}.png')
        plt.pause(0.01)
    fig.savefig(f'step{i}.png')


################################################################################

def show_u(it):
    vmin = -0.8
    vmax = 1.6
    return show_it(extract_u, it, vmin=vmin, vmax=vmax)


def show_v(it):
    vmin = -0.8
    vmax = 0.8
    return show_it(extract_v, it, vmin=vmin, vmax=vmax)


def show_chainer(it, n):
    vmin = 0
    vmax = 1
    def ex_(frame):
        if frame.ndim == 3:
            return frame[n, :, :]
        if frame.ndim == 4:
            return frame[0, n, :, :]
        else:
            raise TypeError
    return show_it(ex_, it, vmin=vmin, vmax=vmax)


def show_chainer_2c(it):
    vmin = 0
    vmax = 1
    def ex_(frame):
        if frame.ndim == 3:
            return frame
        if frame.ndim == 4:
            return frame[0]
        else:
            raise TypeError
    return show_it_m(ex_, it, vmin=vmin, vmax=vmax)


def show_chainer_2r2c(it):
    vmin = 0
    vmax = 1
    def ex_(frames):
        return chain(*frames)
    return show_it_m(ex_, it, nrows=2, ncols=2, vmin=vmin, vmax=vmax)


def show_chainer_NrNc(it, nrows, ncols, direction='lr'):
    vmin = 0
    vmax = 1
    def ex_(frames):
        ''' fraes: ([frame00, frame01, ...], [frame10, frame11, ...], ...) '''
        if direction == 'lr':
            return chain(*frames)
        if direction == 'rl':
            return chain(*reversed(frames))
        if direction == 'tb':
            return chain(*zip(*frames))
        if direction == 'bt':
            return chain(*reversed(zip(*frames)))
    return show_it_m(ex_, it, nrows=nrows, ncols=ncols, vmin=vmin, vmax=vmax)
