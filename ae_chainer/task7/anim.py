import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as plc
import matplotlib.animation as pla

import myutils as ut


################################################################################

class Animation(object):

    def __init__(self, fig, frames=1000, interval=1, fps=10):
        self.fig = fig
        self.frames = frames
        self.interval = interval
        self.fps = fps

        self.writer = pla.writers['ffmpeg'](fps=fps)
        # plt.rcParams['animation.ffmpeg_path'] = ''
        # FFMpegWriter = pla.writers['ffmpeg']

    def __call__(self, *args, **kwargs):
        return self.export_anim(*args, **kwargs)

    def export_anim(self, update, file=None, init_func=lambda:None):
        if file:
            anim = pla.FuncAnimation(self.fig, update, frames=self.frames,
                                     init_func=init_func,
                                     interval=self.interval)
            anim.save(file, writer=self.writer)

        else:
            init_func()
            for i in range(self.frames):
                update(i)
                plt.pause(0.001*self.interval)
