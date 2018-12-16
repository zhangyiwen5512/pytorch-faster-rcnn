# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from layer_utils.generate_anchors import generate_anchors

def generate_anchors_pre(height, width, feat_stride, anchor_scales=(8,16,32), anchor_ratios=(0.5,1,2)):
  """
    A wrapper function to generate anchors given different scales
    Also return the number of anchors in variable 'length'
  """
  #转到generate_anchors。py,得到一系列anchor[x1,y1,x2,y2]左下和右上坐标
  anchors = generate_anchors(ratios=np.array(anchor_ratios), scales=np.array(anchor_scales))

  A = anchors.shape[0]#anchor的数量9个
  #16格一平移
  shift_x = np.arange(0, width) * feat_stride# feat_stride=[16,]
  shift_y = np.arange(0, height) * feat_stride
  shift_x, shift_y = np.meshgrid(shift_x, shift_y)#生成网格图纵向复制和横向复制均为（w×h）（w×h）
  #shift=[shift_x, shift_y, shift_x, shift_y]     shift_x.ravel[1× W*W]    shift_y.ravel[1× H*H]
  #shift=[w*h, 4]
  shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
  K = shifts.shape[0] # =w*h

  # width changes faster, so here it is H, W, C
  # transpose做轴对换0轴和1轴对换
  # ---------------A=9 K=W*H----------------------------------
  #print("________________________________________________________",A,K,width,height,width*height)
  #[1,9,4]+[k,1,4]=[k,A,4]
  #一个anchor生成4K个坐标，共扫图k次
  anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
  #[K*A,4]，生成9K个anchor
  anchors = anchors.reshape((K * A, 4)).astype(np.float32, copy=False)
  #length=K*A
  length = np.int32(anchors.shape[0])

  return anchors, length
