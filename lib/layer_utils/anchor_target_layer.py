# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from model.config import cfg
import numpy as np
import numpy.random as npr
from utils.bbox import bbox_overlaps
from model.bbox_transform import bbox_transform
import torch

def anchor_target_layer(rpn_cls_score, gt_boxes, im_info, _feat_stride, all_anchors, num_anchors):
  """
  Same as the anchor target layer in original Fast/er RCNN
  1 筛选anchor
  2 计算IoU
  3 根据IoU标记正负
  4 留下256个标签
  5 做回归计算
  6 反映射到原来19494个anchor中
  """


  A = num_anchors #9
  total_anchors = all_anchors.shape[0] #19494个anchor
  K = total_anchors / num_anchors


  # allow boxes to sit over the edge by a small amount
  _allowed_border = 0

  # map of shape (..., H, W)
  height, width = rpn_cls_score.shape[1:3] #height=57,width=38

  # only keep anchors inside the image,得到序号满足（x>0,y<0,w<W, h<H）
  inds_inside = np.where(
    (all_anchors[:, 0] >= -_allowed_border) &
    (all_anchors[:, 1] >= -_allowed_border) &
    (all_anchors[:, 2] < im_info[1] + _allowed_border) &  # width
    (all_anchors[:, 3] < im_info[0] + _allowed_border)  # height
  )[0]

  # keep only inside anchors
  anchors = all_anchors[inds_inside, :]

  # label: 1 is positive, 0 is negative, -1 is dont care
  labels = np.empty((len(inds_inside),), dtype=np.float32)
  labels.fill(-1)

  # overlaps between the anchors and the gt boxes
  # overlaps (ex, gt)
  # overlaps[inds_inside,gt]
  overlaps = bbox_overlaps(
    np.ascontiguousarray(anchors, dtype=np.float),
    np.ascontiguousarray(gt_boxes, dtype=np.float))

  argmax_overlaps = overlaps.argmax(axis=1)#得到序号[inds_inside]
  #[inds_inside]
  max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]#得到最大值的值,每行的最大值,确定这个anchor属于哪个boxes


  gt_argmax_overlaps = overlaps.argmax(axis=0)#得到最大值的序号,按列方向搜索，[gt]
  gt_max_overlaps = overlaps[gt_argmax_overlaps,
                             np.arange(overlaps.shape[1])]#得到最大值的值，每列的最大值，哪个anchor [gt],确定是哪个anchor

  gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]#得到最大值的序号（行号）,满足overlap中等于gt_max_overlaps


  if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
    # assign bg labels first so that positive labels can clobber them
    # first set the negatives小于0.3,标负标签，表示这一行都小于0.3,即这个box和任何gt都不重叠
    labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

  # fg label: for each gt, anchor with highest overlap,最大值肯定是正的，这一行
  labels[gt_argmax_overlaps] = 1

  # fg label: above threshold IOU,，每一行大于0.7的也是正的
  labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

  if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
    # assign bg labels last so that negative labels can clobber positives
    labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

  # subsample positive labels if we have too many 前景数量不超过batchsize的一半
  num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
  fg_inds = np.where(labels == 1)[0]#得到前景的index
  if len(fg_inds) > num_fg:
    disable_inds = npr.choice(
      fg_inds, size=(len(fg_inds) - num_fg), replace=False)
    labels[disable_inds] = -1# 多余的前景转为不感兴趣

  # subsample negative labels if we have too many
  num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
  bg_inds = np.where(labels == 0)[0]
  if len(bg_inds) > num_bg:
    disable_inds = npr.choice(
      bg_inds, size=(len(bg_inds) - num_bg), replace=False)
    labels[disable_inds] = -1# 多余的背景转为不感兴趣

  # 留下RPN_BATCHSIZE个标签

  bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
  # 计算筛选后的anchor和有最大IoU的gt的bounding box regression
  bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])

  bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
  # only the positive ones have regression targets [1.0, 1.0, 1.0, 1.0] ,
  bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)

  bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
  if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
    # uniform weighting of examples (given non-uniform sampling)
    num_examples = np.sum(labels >= 0)#256
    positive_weights = np.ones((1, 4)) * 1.0 / num_examples
    negative_weights = np.ones((1, 4)) * 1.0 / num_examples

  else:
    assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
            (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
    positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                        np.sum(labels == 1))
    negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                        np.sum(labels == 0))


  #初始化weight用作训练其他网络 [1.0, 1.0, 1.0, 1.0]/num_examples
  bbox_outside_weights[labels == 1, :] = positive_weights
  bbox_outside_weights[labels == 0, :] = negative_weights

  # map up to original set of anchors, 映射到原始的anchor
  labels = _unmap(labels, total_anchors, inds_inside, fill=-1)#[label, 19494, 7061,,-1]
  bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
  bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
  bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)
  #映射到原来19494个anchor中

  # labels
  labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2) #[1,A,height,width]
  labels = labels.reshape((1, 1, A * height, width))#[1,1,A * height,width]
  rpn_labels = labels

  # bbox_targets
  bbox_targets = bbox_targets \
    .reshape((1, height, width, A * 4))# [1,height,width ,9*4]

  rpn_bbox_targets = bbox_targets
  # bbox_inside_weights
  bbox_inside_weights = bbox_inside_weights \
    .reshape((1, height, width, A * 4))# [1,height,width ,9*4]

  rpn_bbox_inside_weights = bbox_inside_weights

  # bbox_outside_weights
  bbox_outside_weights = bbox_outside_weights \
    .reshape((1, height, width, A * 4))# [1,height,width ,9*4]

  rpn_bbox_outside_weights = bbox_outside_weights

  ##[1,1,A * height,width]标签 [1,height,width ,9*4]回归 [1,height,width ,9*4] [1,height,width ,9*4]
  return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


def _unmap(data, count, inds, fill=0):
  """
  Unmap a subset of item (data) back to the original set of items (of
  size count)
  """
  if len(data.shape) == 1:
    # 如果是一维的将标签标到19494中对应位置，其余置空
    ret = np.empty((count,), dtype=np.float32)
    ret.fill(fill)
    ret[inds] = data
  else:
    ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
    ret.fill(fill)
    ret[inds, :] = data
  return ret


def _compute_targets(ex_rois, gt_rois):
  """Compute bounding-box regression targets for an image."""

  assert ex_rois.shape[0] == gt_rois.shape[0]
  assert ex_rois.shape[1] == 4
  assert gt_rois.shape[1] == 5

  return bbox_transform(torch.from_numpy(ex_rois), torch.from_numpy(gt_rois[:, :4])).numpy()
