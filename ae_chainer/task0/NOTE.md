# 作業記録
## 2018.10.28
script: practice2.py  
多層オートエンコーダ実装のための予備調査  

### Datasets
```
class chainer.datasets.TupleDataset(*datasets)
```
TupleDataset(学習データリスト, 教師データリスト)
* zip() 関数に類似している

### Link
```
class chainer.links.Convolution2D(self, in_channels, out_channels, ksize=None, stride=1, pad=0, nobias=False, initialW=None, initial_bias=None, *, dilate=1, groups=1)
class chainer.links.Deconvolution2D(self, in_channels, out_channels, ksize=None, stride=1, pad=0, nobias=False, outsize=None, initialW=None, initial_bias=None, *, groups=1)
```

### Functions
```
chainer.functions.max_pooling_2d(x, ksize, stride=None, pad=0, cover_all=True, return_indices=False)
chainer.functions.unpooling_2d(x, ksize, stride=None, pad=0, outsize=None, cover_all=True)
```

### Link関数バージョン
```
chainer.functions.convolution_2d(x, W, b=None, stride=1, pad=0, cover_all=False, *, dilate=1, groups=1)
chainer.functions.deconvolution_2d(x, W, b=None, stride=1, pad=0, outsize=None, *, dilate=1, groups=1)
```
