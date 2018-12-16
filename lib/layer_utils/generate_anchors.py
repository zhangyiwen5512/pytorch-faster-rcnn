# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

# array([[ -83.,  -39.,  100.,   56.],
#       [-175.,  -87.,  192.,  104.],
#       [-359., -183.,  376.,  200.],
#       [ -55.,  -55.,   72.,   72.],
#       [-119., -119.,  136.,  136.],
#       [-247., -247.,  264.,  264.],
#       [ -35.,  -79.,   52.,   96.],
#       [ -79., -167.,   96.,  184.],
#       [-167., -343.,  184.,  360.]])

def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2 ** np.arange(3, 6)):
  """
  Generate anchor (reference) windows by enumerating aspect ratios X
  scales wrt a reference (0, 0, 15, 15) window.
  """
  #[0,0,15,15], ratios = [0.5,1,2], scales=[8,16,32]
  base_anchor = np.array([1, 1, base_size, base_size]) - 1
  #枚举ratios
  ratio_anchors = _ratio_enum(base_anchor, ratios)
  #找到anchors
  anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                       for i in range(ratio_anchors.shape[0])])
  return anchors #一共9种 基本尺寸+其缩放的3个， 扁平的+其缩放， 拉伸的+其缩放


def _whctrs(anchor):
  """
  Return width, height, x center, and y center for an anchor (window).
  """

  w = anchor[2] - anchor[0] + 1
  h = anchor[3] - anchor[1] + 1
  x_ctr = anchor[0] + 0.5 * (w - 1)
  y_ctr = anchor[1] + 0.5 * (h - 1)
  return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
  """
  Given a vector of widths (ws) and heights (hs) around a center
  (x_ctr, y_ctr), output a set of anchors (windows).
  """

  ws = ws[:, np.newaxis]
  """
  [512,256,128]求根号变成
  [
   [23],     [12]
   [16],     [16]
   [11]      [22]
  ]
  """
  hs = hs[:, np.newaxis]
  anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                       y_ctr - 0.5 * (hs - 1),
                       x_ctr + 0.5 * (ws - 1),
                       y_ctr + 0.5 * (hs - 1)))
  #anchor[x1,y1,x2,y2]坐下和右上，一共三组比例,面积相同
 #      [[-3.5,  2. , 18.5, 13. ],
#       [ 0. ,  0. , 15. , 15. ],
 #      [ 2.5, -3. , 12.5, 18. ]]
  return anchors


def _ratio_enum(anchor, ratios):
  """
  Enumerate a set of anchors for each aspect ratio wrt an anchor.
  列出一系列anchor box 通过使用所有的比例
  weight，hight，x，y
  anchor=[0,0,15,15]
  """
  w, h, x_ctr, y_ctr = _whctrs(anchor)# [16,16,，7.5,7.5]
  size = w * h
  size_ratios = size / ratios# [16*16 / [0.5 ,1,2]] =[512,256,123]
  ws = np.round(np.sqrt(size_ratios))# [512,256,128]求根号[23,16,11]
  hs = np.round(ws * ratios)#[水平，正方，竖直][12,16,22]
  anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
  return anchors


def _scale_enum(anchor, scales):
  """
  Enumerate a set of anchors for each scale wrt an anchor.
  """

  w, h, x_ctr, y_ctr = _whctrs(anchor)#[x1,x2,y1,y2]变成[w,h,x,y]
  ws = w * scales#[8,16,32]*w
  hs = h * scales#[8,16,32]*h
  anchors = _mkanchors(ws, hs, x_ctr, y_ctr)#得到缩放[8,16,32]倍后的anchors
  return anchors


if __name__ == '__main__':
  import time

  t = time.time()
  a = generate_anchors()
  print(time.time() - t)
  print(a)
  from IPython import embed;

  embed()
