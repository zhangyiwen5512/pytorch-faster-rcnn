# --------------------------------------------------------
# Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from model.config import cfg
from model.bbox_transform import bbox_transform_inv, clip_boxes
from model.nms_wrapper import nms

import torch

def proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchors, num_anchors):
  """
     A simplified version compared to fast/er RCNN
     For details please see the technical report
  """
  if type(cfg_key) == bytes:
      cfg_key = cfg_key.decode('utf-8')

  pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N#12000
  post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N#2000
  nms_thresh = cfg[cfg_key].RPN_NMS_THRESH#0.7

  # Get the scores and bounding boxes
  # [1,57,38,18]     [1,57,38,36]
  scores = rpn_cls_prob[:, :, :, num_anchors:] #[1,57,38,9]
  rpn_bbox_pred = rpn_bbox_pred.view((-1, 4))#[19494,4]
  scores = scores.contiguous().view(-1, 1)# [19494,1]

  #9个anchor，[19494,4]，做边框平移和缩放，得到和ground truth相近的结果
  proposals = bbox_transform_inv(anchors, rpn_bbox_pred)  #[x1,y1,x2,y2]
  proposals = clip_boxes(proposals, im_info[:2])#根据宽高缩放比，w，h，scale

  # Pick the top region proposals
  # scores是值，order是序列
  scores, order = scores.view(-1).sort(descending=True)

  #保留2000个最高的
  if pre_nms_topN > 0:
    order = order[:pre_nms_topN]
    scores = scores[:pre_nms_topN].view(-1, 1)
  proposals = proposals[order.data, :]

  # Non-maximal suppression
  keep = nms(torch.cat((proposals, scores), 1).data, nms_thresh)

  # Pick th top region proposals after NMS
  if post_nms_topN > 0:
    keep = keep[:post_nms_topN]
  proposals = proposals[keep, :]
  scores = scores[keep,]

  # Only support single image as input  #保留2000个最高的
  batch_inds = proposals.new_zeros(proposals.size(0), 1)#[2000,1]，全0
  blob = torch.cat((batch_inds, proposals), 1)#[2000,5] 多一维batch_inds

  return blob, scores#[2000,5] [x1,y1,x2,y2] [2000,1]
