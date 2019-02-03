import os
import numpy as np
import chainer


################################################################################
# locale書き換え
################################################################################

import _locale
_locale._getdefaultlocale = lambda *args: ('ja_JP', 'utf-8')


################################################################################
# 環境・共通パラメータ定義
################################################################################

# 定数
KB1 = 1024
MB1 = 1048576
GB1 = 1073741824

# パス
SRC_DIR = os.path.dirname(__file__)
SRC_FILE = os.path.basename(__file__)
SRC_FILENAME = os.path.splitext(SRC_FILE)[0]
PATH_INI = 'path1.ini'

# 環境
OS = os.environ.get('OS')
OS_IS_WIN = OS == 'Windows_NT'

# 環境(chainer)
try:
    import cupy
    xp = cupy

except ModuleNotFoundError:
    xp = np
    DEVICE = -1
    NDARRAY_TYPES = np.ndarray,

else:
    DEVICE = 0
    chainer.cuda.get_device_from_id(DEVICE).use()
    NDARRAY_TYPES = np.ndarray, xp.ndarray

# 学習オプション
SHOW_PROGRESSBAR = True


################################################################################
# 関数
################################################################################

def sigmoid(x, a=1):
    return 1 / (1 + np.exp(-a * x))


def logit(x, a=1):
    x_ = np.clip(x, 1e-8, 1 - 1e-8)
    return np.where((0 < x) * (x < 1), np.log(x_ / (1 - x_)) / a, np.nan)
