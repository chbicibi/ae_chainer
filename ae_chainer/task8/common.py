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
