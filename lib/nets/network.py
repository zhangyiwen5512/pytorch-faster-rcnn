# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import utils.timer

from layer_utils.snippets import generate_anchors_pre
from layer_utils.proposal_layer import proposal_layer
from layer_utils.proposal_top_layer import proposal_top_layer
from layer_utils.anchor_target_layer import anchor_target_layer
from layer_utils.proposal_target_layer import proposal_target_layer
from utils.visualization import draw_bounding_boxes

from layer_utils.roi_pooling.roi_pool import RoIPoolFunction
from layer_utils.roi_align.crop_and_resize import CropAndResizeFunction

from model.config import cfg

import tensorboardX as tb

from scipy.misc import imresize

class Network(nn.Module):
  def __init__(self):
    nn.Module.__init__(self)# 此处初始化resnet属性self。net是一个module的子类
    self._predictions = {}
    self._losses = {}
#    if cfg.TRAIN.IMS_PER_BATCH == 2:
 #     self._RPN_losses = {}
    self._anchor_targets = {}
    self._proposal_targets = {}
    self._layers = {}
    self._gt_image = None
    self._act_summaries = {}
    self._score_summaries = {}
    self._event_summaries = {}
    self._image_gt_summaries = {}
    self._variables_to_fix = {}
    self._device = 'cuda'

  def _add_gt_image(self):
    # add back mean
    image = self._image_gt_summaries['image'] + cfg.PIXEL_MEANS
    image = imresize(image[0], self._im_info[:2] / self._im_info[2])
    # BGR to RGB (opencv uses BGR)############################################################################
    self._gt_image = image[np.newaxis, :,:,::-1].copy(order='C')

  def _add_gt_image_summary(self):
    # use a customized visualization function to visualize the boxes
    self._add_gt_image()
    image = draw_bounding_boxes(\
                      self._gt_image, self._image_gt_summaries['gt_boxes'], self._image_gt_summaries['im_info'])
#############################################################################################################################################
#    print(image[0].astype('float64'))
    return tb.summary.image('GROUND_TRUTH', image[0].astype('float64')/255.0)

  def _add_act_summary(self, key, tensor):
    return tb.summary.histogram('ACT/' + key + '/activations', tensor.data.cpu().numpy(), bins='auto'),
    tb.summary.scalar('ACT/' + key + '/zero_fraction',
                      (tensor.data == 0).float().sum() / tensor.numel())

  def _add_score_summary(self, key, tensor):
    return tb.summary.histogram('SCORE/' + key + '/scores', tensor.data.cpu().numpy(), bins='auto')

  def _add_train_summary(self, key, var):
    return tb.summary.histogram('TRAIN/' + key, var.data.cpu().numpy(), bins='auto')

  def _proposal_top_layer(self, rpn_cls_prob, rpn_bbox_pred):
    rois, rpn_scores = proposal_top_layer(\
                                    rpn_cls_prob, rpn_bbox_pred, self._im_info,
                                     self._feat_stride, self._anchors, self._num_anchors)
    return rois, rpn_scores

  def _proposal_layer(self, rpn_cls_prob, rpn_bbox_pred):
    #[2000,5] [x1,y1,x2,y2] [2000,1]
    rois, rpn_scores = proposal_layer(\
                                    rpn_cls_prob, rpn_bbox_pred, self._im_info, self._mode,
                                     self._feat_stride, self._anchors, self._num_anchors)

    return rois, rpn_scores

  def _roi_pool_layer(self, bottom, rois):
    return RoIPoolFunction(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1. / 16.)(bottom, rois)

  def _crop_pool_layer(self, bottom, rois, max_pool=True):
    # implement it using stn
    # box to affine
    # input (x1,y1,x2,y2)
    # ROIpolinglayer ：bottom
    """
    [  x2-x1             x1 + x2 - W + 1  ]
    [  -----      0      ---------------  ]
    [  W - 1                  W - 1       ]
    [                                     ]
    [           y2-y1    y1 + y2 - H + 1  ]
    [    0      -----    ---------------  ]
    [           H - 1         H - 1      ]
    """
    rois = rois.detach()

    x1 = rois[:, 1::4] / 16.0#[256,1]
    y1 = rois[:, 2::4] / 16.0#
    x2 = rois[:, 3::4] / 16.0#
    y2 = rois[:, 4::4] / 16.0#

    height = bottom.size(2)
    width = bottom.size(3)

    # pre_pool_size=7
    pre_pool_size = cfg.POOLING_SIZE * 2 if max_pool else cfg.POOLING_SIZE
    #[256,1024,7,7]  将h×w的roi划分为h/H和w/W的网格，再池化roi到对应网格
    crops = CropAndResizeFunction(pre_pool_size, pre_pool_size)(bottom, torch.cat([y1/(height-1),x1/(width-1),y2/(height-1),x2/(width-1)], 1), rois[:, 0].int())
    if max_pool:
      crops = F.max_pool2d(crops, 2, 2)

    return crops

  def _anchor_target_layer(self, rpn_cls_score):
    rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
      anchor_target_layer(
      rpn_cls_score.data, self._gt_boxes.data.cpu().numpy(), self._im_info, self._feat_stride, self._anchors.data.cpu().numpy(), self._num_anchors)
    ##[1,1,A * height,width]标签 [1,height,width ,9*4]回归 [1,height,width ,9*4] [1,height,width ,9*4] 转化为numpy
    rpn_labels = torch.from_numpy(rpn_labels).float().to(self._device) #.set_shape([1, 1, None, None])
    rpn_bbox_targets = torch.from_numpy(rpn_bbox_targets).float().to(self._device)#.set_shape([1, None, None, self._num_anchors * 4])
    rpn_bbox_inside_weights = torch.from_numpy(rpn_bbox_inside_weights).float().to(self._device)#.set_shape([1, None, None, self._num_anchors * 4])
    rpn_bbox_outside_weights = torch.from_numpy(rpn_bbox_outside_weights).float().to(self._device)#.set_shape([1, None, None, self._num_anchors * 4])

    rpn_labels = rpn_labels.long()
    self._anchor_targets['rpn_labels'] = rpn_labels
    self._anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets
    self._anchor_targets['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
    self._anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights
##############################################################################################################
    if cfg.TRAIN.IMS_PER_BATCH == 2 :
 ###################################################################   ??????
      rpn_labels2, rpn_bbox_targets2, rpn_bbox_inside_weights2, rpn_bbox_outside_weights2 = \
      anchor_target_layer(
      rpn_cls_score.data, self._gt_boxes2.data.cpu().numpy(), self._im_info, self._feat_stride, self._anchors.data.cpu().numpy(), self._num_anchors)
      ##[1,1,A * height,width]标签 [1,height,width ,9*4]回归 [1,height,width ,9*4] [1,height,width ,9*4] 转化为numpy
#########################################################################???????????

      rpn_labels2 = torch.from_numpy(rpn_labels2).float().to(self._device) #.set_shape([1, 1, None, None])
      rpn_bbox_targets2 = torch.from_numpy(rpn_bbox_targets2).float().to(self._device)#.set_shape([1, None, None, self._num_anchors * 4])
      rpn_bbox_inside_weights2 = torch.from_numpy(rpn_bbox_inside_weights2).float().to(self._device)#.set_shape([1, None, None, self._num_anchors * 4])
      rpn_bbox_outside_weights2 = torch.from_numpy(rpn_bbox_outside_weights2).float().to(self._device)#.set_shape([1, None, None, self._num_anchors * 4])

      rpn_labels2 = rpn_labels2.long()
      self._anchor_targets['rpn_labels2'] = rpn_labels2
      self._anchor_targets['rpn_bbox_targets2'] = rpn_bbox_targets2
      self._anchor_targets['rpn_bbox_inside_weights2'] = rpn_bbox_inside_weights2
      self._anchor_targets['rpn_bbox_outside_weights2'] = rpn_bbox_outside_weights2

##############################################################################################################
    for k in self._anchor_targets.keys():
      self._score_summaries[k] = self._anchor_targets[k]

    return rpn_labels

  def _proposal_target_layer(self, rois, roi_scores):
    #[256, 5],[256],[256, 1],[256, 84],[256, 84],[256, 84]  #weights,前景为1,背景为0
    rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = \
      proposal_target_layer(
      rois, roi_scores, self._gt_boxes, self._num_classes)

    self._proposal_targets['rois'] = rois
    self._proposal_targets['labels'] = labels.long()
    self._proposal_targets['bbox_targets'] = bbox_targets
    self._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
    self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights

    for k in self._proposal_targets.keys():
      self._score_summaries[k] = self._proposal_targets[k]

    return rois, roi_scores

  def _anchor_component(self, height, width):
    # just to get the shape right
    # 找出anchorbox,是合成的读取的图片的height和width
    #height = int(math.ceil(self._im_info.data[0, 0] / self._feat_stride[0]))
    #width = int(math.ceil(self._im_info.data[0, 1] / self._feat_stride[0]))
    #转到snippets。py
    anchors, anchor_length = generate_anchors_pre(\
                                          height, width,
                                           self._feat_stride, self._anchor_scales, self._anchor_ratios)
    #得到9×k个anchor,送到GPU
    self._anchors = torch.from_numpy(anchors).to(self._device)
    self._anchor_length = anchor_length

  def _smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
    sigma_2 = sigma ** 2 # 9 sigma=3
    box_diff = bbox_pred - bbox_targets# 预测和gt的差值
    in_box_diff = bbox_inside_weights * box_diff# 前景则为box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()# <差值 1/9 为正号，大于为0
    #smoothL1_sign表示绝对值小于sigma_2   1-smoothL1_sign表示else
    in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box
    #减序列
    for i in sorted(dim, reverse=True):
      loss_box = loss_box.sum(i)

    loss_box = loss_box.mean()
    return loss_box

################################################################################################################loss计算
  def _add_losses(self, sigma_rpn=3.0):
    if  cfg.TRAIN.IMS_PER_BATCH == 1:
      # RPN, class loss
      rpn_cls_score = self._predictions['rpn_cls_score_reshape'].view(-1, 2)#[前景loss，背景loss][Anchorsize*width*height]个anchor
      rpn_label = self._anchor_targets['rpn_labels'].view(-1)
      rpn_select = (rpn_label.data != -1).nonzero().view(-1)#选取的前景及背景
      rpn_cls_score = rpn_cls_score.index_select(0, rpn_select).contiguous().view(-1, 2)#[256,gt]
      rpn_label = rpn_label.index_select(0, rpn_select).contiguous().view(-1)#[256]
      # 是rpn部分的loss
      rpn_cross_entropy = F.cross_entropy(rpn_cls_score, rpn_label)

      # RPN, bbox loss
      rpn_bbox_pred = self._predictions['rpn_bbox_pred']# batch * h * w * (num_anchors*4) 回归框预测的坐标
      rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets']# [1,height,width ,9*4] 回归框目标的坐标(和gt的回归值)
      rpn_bbox_inside_weights = self._anchor_targets['rpn_bbox_inside_weights']# [1,height,width ,9*4]
      rpn_bbox_outside_weights = self._anchor_targets['rpn_bbox_outside_weights']# [1,height,width ,9*4]
      # 是rpn部分的loss
      rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                          rpn_bbox_outside_weights, sigma=sigma_rpn, dim=[1, 2, 3])
    elif cfg.TRAIN.IMS_PER_BATCH == 2:

      ############ img1
      # RPN, class loss
      rpn_cls_score = self._predictions['rpn_cls_score_reshape'].view(-1, 2)#[前景loss，背景loss][Anchorsize*width*height]个anchor
      rpn_label = self._anchor_targets['rpn_labels'].view(-1)
      rpn_select = (rpn_label.data != -1).nonzero().view(-1)#选取的前景及背景
      rpn_cls_score = rpn_cls_score.index_select(0, rpn_select).contiguous().view(-1, 2)#[256,gt]
      rpn_label = rpn_label.index_select(0, rpn_select).contiguous().view(-1)#[256]
      # 是rpn部分的loss
      rpn_cross_entropy1 = F.cross_entropy(rpn_cls_score, rpn_label)

      # RPN, bbox loss
      rpn_bbox_pred = self._predictions['rpn_bbox_pred']# batch * h * w * (num_anchors*4) 回归框预测的坐标
      rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets']# [1,height,width ,9*4] 回归框目标的坐标(和gt的回归值)
      rpn_bbox_inside_weights = self._anchor_targets['rpn_bbox_inside_weights']# [1,height,width ,9*4]
      rpn_bbox_outside_weights = self._anchor_targets['rpn_bbox_outside_weights']# [1,height,width ,9*4]
      # 是rpn部分的loss
      rpn_loss_box1 = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                          rpn_bbox_outside_weights, sigma=sigma_rpn, dim=[1, 2, 3])

      ############img2
      # RPN, class loss
      rpn_label2 = self._anchor_targets['rpn_labels2'].view(-1)
      rpn_select2 = (rpn_label2.data != -1).nonzero().view(-1)#选取的前景及背景
      rpn_cls_score = self._predictions['rpn_cls_score_reshape'].view(-1, 2)#[前景loss，背景loss][Anchorsize*width*height]个anchor
      rpn_cls_score2 = rpn_cls_score.index_select(0, rpn_select2).contiguous().view(-1, 2)#[256,gt]
      rpn_label2 = rpn_label2.index_select(0, rpn_select2).contiguous().view(-1)#[256]
      # 是rpn部分的loss
      rpn_cross_entropy2 = F.cross_entropy(rpn_cls_score2, rpn_label2)

      # RPN, bbox loss
      rpn_bbox_targets2 = self._anchor_targets['rpn_bbox_targets2']# [1,height,width ,9*4] 回归框目标的坐标(和gt的回归值)
      rpn_bbox_inside_weights2 = self._anchor_targets['rpn_bbox_inside_weights2']# [1,height,width ,9*4]
      rpn_bbox_outside_weights2 = self._anchor_targets['rpn_bbox_outside_weights2']# [1,height,width ,9*4]

      # 是rpn部分的loss

      rpn_loss_box2 = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets2, rpn_bbox_inside_weights2,
                                          rpn_bbox_outside_weights2, sigma=sigma_rpn, dim=[1, 2, 3])
##############################################3
      lam = cfg.lamda
      rpn_cross_entropy = lam * rpn_cross_entropy1 + (1 - lam) * rpn_cross_entropy2
      rpn_loss_box = lam * rpn_loss_box1 + (1 - lam) * rpn_loss_box2
    else:
       raise Exception("check cfg.TRAIN.IMS_PER_BACTH in /lib/model/config.py or experiments/cfgs/*.yml")

    if cfg.loss_strategy == 'RCNN_ONLY' or cfg.loss_strategy == 'RCNN+RPN' or cfg.loss_strategy == 'NOCHANGE':
      # RCNN, class loss
      cls_score = self._predictions["cls_score"]# [256,21]
      label = self._proposal_targets["labels"].view(-1)#[256]
      # RCNN的loss
      cross_entropy = F.cross_entropy(cls_score.view(-1, self._num_classes), label)

      # RCNN, bbox loss
      bbox_pred = self._predictions['bbox_pred']# [256,84]
      bbox_targets = self._proposal_targets['bbox_targets']# [256,84]
      bbox_inside_weights = self._proposal_targets['bbox_inside_weights']# [256,84]
      bbox_outside_weights = self._proposal_targets['bbox_outside_weights']# [256,84]
      # RCNN box的loss

      loss_box = self._smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

    if cfg.loss_strategy == 'RCNN_ONLY' or cfg.loss_strategy == 'RCNN+RPN':
      lam = cfg.lamda
      label2 = self._proposal_targets['labels'][self.rcnn_mix_index, :].view(-1)
      cross_entropy2 = F.cross_entropy(cls_score.view(-1, self._num_classes), label2)
      cross_entropy = lam * cross_entropy + (1 - lam) * cross_entropy2

      bbox_targets2 = self._proposal_targets['bbox_targets'][self.rcnn_mix_index, :]
      bbox_inside_weights2 = self._proposal_targets['bbox_inside_weights'][self.rcnn_mix_index, :]
      bbox_outside_weights2 = self._proposal_targets['bbox_outside_weights'][self.rcnn_mix_index, :]
      loss_box2 = self._smooth_l1_loss(bbox_pred, bbox_targets2, bbox_inside_weights2, bbox_outside_weights2)
      loss_box = lam * loss_box + (1 - lam) * loss_box2

    if cfg.loss_strategy == 'RPN_ONLY':
      pass

    if cfg.loss_strategy == 'RCNN+RPN' or cfg.loss_strategy == 'NOCHANGE':
      self._losses['cross_entropy'] = cross_entropy
      self._losses['loss_box'] = loss_box
      self._losses['rpn_cross_entropy'] = rpn_cross_entropy
      self._losses['rpn_loss_box'] = rpn_loss_box

      loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box
    elif cfg.loss_stratrgy == 'RPN_ONLY':
      loss  = rpn_cross_entropy + rpn_loss_box
      self._losses['rpn_cross_entropy'] = rpn_cross_entropy
      self._losses['rpn_loss_box'] = rpn_loss_box

    elif cfg.loss_strategy == 'RCNN_ONLY':
      loss = cross_entropy + loss_box
      self._losses['cross_entropy'] = cross_entropy
      self._losses['loss_box'] = loss_box

    else:
      raise Exception("check cfg.TRAIN.loss_strategy in /lib/model/config.py or experiments/cfgs/*.yml")

##################################################################################################################
    self._losses['total_loss'] = loss

    for k in self._losses.keys():
      self._event_summaries[k] = self._losses[k]

    return loss

  def _region_proposal(self, net_conv):
    # 得到RPN网络的结果（做完relu）
    rpn = F.relu(self.rpn_net(net_conv))
    self._act_summaries['rpn'] = rpn
#---------------------做anchor分数预测--------------------------------

    rpn_cls_score = self.rpn_cls_score_net(rpn) # batch * (num_anchors * 2) * h * w  顺序[0,1,2,3][1.18,57,38]

    # change it so that the score has 2 as its channel size，（前景，背景）
    rpn_cls_score_reshape = rpn_cls_score.view(1, 2, -1, rpn_cls_score.size()[-1]) # batch * 2 * (num_anchors*h) * w [0,1,2,3] [1.2,513,38] 9*57=513
    #sofamax預測分数
    rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, dim=1)#[1,2,513,38]
    # Move channel to the last dimenstion, to fit the input of python functions
    rpn_cls_prob = rpn_cls_prob_reshape.view_as(rpn_cls_score).permute(0, 2, 3, 1) # batch * h * w * (num_anchors * 2) [1,57,38,18]
    rpn_cls_score = rpn_cls_score.permute(0, 2, 3, 1) # batch * h * w * (num_anchors * 2) [1,57,38,18]
    rpn_cls_score_reshape = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous()  # batch * (num_anchors*h) * w * 2 [1,513,38,2]

    #最终预测结果rpn_cls_pred
    rpn_cls_pred = torch.max(rpn_cls_score_reshape.view(-1, 2), 1)[1]# 9*57*38=19494,是index
#---------------------做anchor的bounding box预测--------------------------
    rpn_bbox_pred = self.rpn_bbox_pred_net(rpn)
    rpn_bbox_pred = rpn_bbox_pred.permute(0, 2, 3, 1).contiguous()  # batch * h * w * (num_anchors*4)

    if self._mode == 'TRAIN':
      #RPN(可能性，坐标)得到正式的region 和其得分，从候选的anchor中。[1,57,38,18]
      rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred) # rois, roi_scores are varible
      #rois [2000,5] [x1,y1,x2,y2] roi_scores[2000,1]
      #标记正负样本  [1,57,38,18]
      rpn_labels = self._anchor_target_layer(rpn_cls_score)
      #选出用于训练的region  #[256, 5],[256]
      rois, _ = self._proposal_target_layer(rois, roi_scores)
    else:
      if cfg.TEST.MODE == 'nms':
        rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred)
      elif cfg.TEST.MODE == 'top':
        rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred)
      else:
        raise NotImplementedError

    self._predictions["rpn_cls_score"] = rpn_cls_score
    self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
    self._predictions["rpn_cls_prob"] = rpn_cls_prob
    self._predictions["rpn_cls_pred"] = rpn_cls_pred
    self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
    self._predictions["rois"] = rois#[x1,y1,x2,y2]

    return rois

  def _region_classification(self, fc7):
    cls_score = self.cls_score_net(fc7)
    cls_pred = torch.max(cls_score, 1)[1]
    cls_prob = F.softmax(cls_score, dim=1)
    bbox_pred = self.bbox_pred_net(fc7)

    self._predictions["cls_score"] = cls_score
    self._predictions["cls_pred"] = cls_pred
    self._predictions["cls_prob"] = cls_prob
    self._predictions["bbox_pred"] = bbox_pred

    return cls_prob, bbox_pred

  def _image_to_head(self):
    raise NotImplementedError

  def _head_to_tail(self, pool5):
    raise NotImplementedError

  def create_architecture(self, num_classes, tag=None,
                          anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
    self._tag = tag

    self._num_classes = num_classes
    self._anchor_scales = anchor_scales
    self._num_scales = len(anchor_scales)

    self._anchor_ratios = anchor_ratios
    self._num_ratios = len(anchor_ratios)

    self._num_anchors = self._num_scales * self._num_ratios

    assert tag != None

    # Initialize layers
    self._init_modules()

  def _init_modules(self):
    self._init_head_tail()#c初始化惹101和vgg16网络
####################################################################################################################################################
    # rpn
    self.rpn_net = nn.Conv2d(self._net_conv_channels, cfg.RPN_CHANNELS, [3, 3], padding=1)

    self.rpn_cls_score_net = nn.Conv2d(cfg.RPN_CHANNELS, self._num_anchors * 2, [1, 1])
    
    self.rpn_bbox_pred_net = nn.Conv2d(cfg.RPN_CHANNELS, self._num_anchors * 4, [1, 1])
    #softmax
    self.cls_score_net = nn.Linear(self._fc7_channels, self._num_classes)
    #bounding box
    self.bbox_pred_net = nn.Linear(self._fc7_channels, self._num_classes * 4)
#####################################################################################################################################################
    self.init_weights()

  def _run_summary_op(self, val=False):
    """
    Run the summary operator: feed the placeholders with corresponding newtork outputs(activations)
    """
    summaries = []
    # Add image gt
    ######################################################################################################
    summaries.append(self._add_gt_image_summary())
    # Add event_summaries
    for key, var in self._event_summaries.items():
      summaries.append(tb.summary.scalar(key, var.item()))
    self._event_summaries = {}
    if not val:
      # Add score summaries
      for key, var in self._score_summaries.items():
        summaries.append(self._add_score_summary(key, var))
      self._score_summaries = {}
      # Add act summaries
      for key, var in self._act_summaries.items():
        summaries += self._add_act_summary(key, var)
      self._act_summaries = {}
      # Add train summaries
      for k, var in dict(self.named_parameters()).items():
        if var.requires_grad:
          summaries.append(self._add_train_summary(k, var))

      self._image_gt_summaries = {}
    
    return summaries

  def _predict(self):
    # This is just _build_network in tf-faster-rcnn
    torch.backends.cudnn.benchmark = False
    net_conv = self._image_to_head()
    # build the anchors for the image 特征图net_conv.size(2)height  net_conv.size(3)weight
    #一张图片得到9×K个anchor feature size=(width/stride)*(height/stride) == K
    self._anchor_component(net_conv.size(2), net_conv.size(3))

    #--------------------------------RPN---------------------------------------------------------------------
    #得到roi[256, 5][class,x1,y1,x2,y2]
    rois = self._region_proposal(net_conv)
    #--------------------------------POLING---------------------------------------------------------------------
    if cfg.loss_strategy == 'RPN_ONLY':##########
      for k in self._predictions.keys():
        self._score_summaries[k] = self._predictions[k]
      return rois, None, None


    if cfg.POOLING_MODE == 'crop':#[256,1024,7,7]
      pool5 = self._crop_pool_layer(net_conv, rois)
    else:
      pool5 = self._roi_pool_layer(net_conv, rois)

    if cfg.loss_strategy == 'RCNN_ONLY' or cfg.loss_strategy == 'RCNN+RPN':
      pool5 = pool5.detach()
      lam = cfg.lamda
      rcnn_index = np.arange(pool5.szie()[0])
      np.random.shuffle(rcnn_index)
      self.rcnn_mix_index = rcnn_index
      pool5 = lam * pool5 + (1 - lam) * pool5[rcnn_index, :]

    if self._mode == 'TRAIN':
      torch.backends.cudnn.benchmark = True # benchmark because now the input size are fixed
    # [256,2048]
    fc7 = self._head_to_tail(pool5)
    #--------------------------------softmax,bouding box --------------------------------------------------------------------
    cls_prob, bbox_pred = self._region_classification(fc7)
    
    for k in self._predictions.keys():
      self._score_summaries[k] = self._predictions[k]

    return rois, cls_prob, bbox_pred

  def forward(self, image, im_info, gt_boxes=None, gt_boxes2=None, mode='TRAIN'):
    """

    :param image:
    :param im_info:
    :param gt_boxes:
    :param mode:
    :return:
    """
    self._image_gt_summaries['image'] = image
    self._image_gt_summaries['gt_boxes'] = gt_boxes
    self._image_gt_summaries['im_info'] = im_info

    self._image = torch.from_numpy(image.transpose([0,3,1,2])).to(self._device)# 通道换位置
    self._im_info = im_info # No need to change; actually it can be an list
    self._gt_boxes = torch.from_numpy(gt_boxes).to(self._device) if gt_boxes is not None else None
######################################################################################################################
    if cfg.TRAIN.IMS_PER_BATCH == 2 :
      self._gt_boxes2 = torch.from_numpy(gt_boxes2).to(self._device) if gt_boxes2 is not None else None
######################################################################################################################

    self._mode = mode
    #得到roi[256, 5][class,x1,y1,x2,y2]
    #cls_prob[256，21],若是前景则得到其预测分数
    #bbox_pred[256,84]，得到其坐标
    rois, cls_prob, bbox_pred = self._predict()
    if mode == 'TEST':
      stds = bbox_pred.data.new(cfg.TRAIN.BBOX_NORMALIZE_STDS).repeat(self._num_classes).unsqueeze(0).expand_as(bbox_pred)
      means = bbox_pred.data.new(cfg.TRAIN.BBOX_NORMALIZE_MEANS).repeat(self._num_classes).unsqueeze(0).expand_as(bbox_pred)
      self._predictions["bbox_pred"] = bbox_pred.mul(stds).add(means)
    else:
      self._add_losses() # compute losses


  def init_weights(self):
    def normal_init(m, mean, stddev, truncated=False):
      """
      weight initalizer: truncated normal and random normal.
      """
      # x is a parameter
      if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
      else:
        m.weight.data.normal_(mean, stddev)
      m.bias.data.zero_()
      
    normal_init(self.rpn_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
    normal_init(self.rpn_cls_score_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
    normal_init(self.rpn_bbox_pred_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
    normal_init(self.cls_score_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
    normal_init(self.bbox_pred_net, 0, 0.001, cfg.TRAIN.TRUNCATED)

  # Extract the head feature maps, for example for vgg16 it is conv5_3
  # only useful during testing mode
  def extract_head(self, image):
    feat = self._layers["head"](torch.from_numpy(image.transpose([0,3,1,2])).to(self._device))
    return feat

  # only useful during testing mode
  def test_image(self, image, im_info):
    self.eval()
    with torch.no_grad():
      self.forward(image, im_info, None, None, mode='TEST')
    cls_score, cls_prob, bbox_pred, rois = self._predictions["cls_score"].data.cpu().numpy(), \
                                                     self._predictions['cls_prob'].data.cpu().numpy(), \
                                                     self._predictions['bbox_pred'].data.cpu().numpy(), \
                                                     self._predictions['rois'].data.cpu().numpy()


    return cls_score, cls_prob, bbox_pred, rois

  def delete_intermediate_states(self):
    # Delete intermediate result to save memory
    for d in [self._losses, self._predictions, self._anchor_targets, self._proposal_targets]:
      for k in list(d):
        del d[k]

  def get_summary(self, blobs):
    self.eval()
    self.forward(blobs['data'], blobs['im_info'], blobs['gt_boxes'], blobs['gt_boxes2'])
    self.train()
    summary = self._run_summary_op(True)

    return summary

  def train_step(self, blobs, train_op):
    if cfg.TRAIN.IMS_PER_BATCH == 1:
      self.forward(blobs['data'], blobs['im_info'], blobs['gt_boxes'], None)
    if cfg.TRAIN.IMS_PER_BATCH == 2:
      self.forward(blobs['data'], blobs['im_info'], blobs['gt_boxes'], blobs['gt_boxes2'])

    if cfg.loss_strategy == 'NOCHANGE' or cfg.loss_strategy == 'RCNN+RPN':
      rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss = self._losses["rpn_cross_entropy"].item(), \
                                                                        self._losses['rpn_loss_box'].item(), \
                                                                        self._losses['cross_entropy'].item(), \
                                                                        self._losses['loss_box'].item(), \
                                                                        self._losses['total_loss'].item()
    if cfg.loss_strategy == 'RCNN_ONLY':
      rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss = -1, -1, \
                                                                        self._losses['cross_entropy'].item(), \
                                                                        self._losses['loss_box'].item(), \
                                                                        self._losses['total_loss'].item()
    if cfg.loss_strategy == 'NOCHANGE' or cfg.loss_strategy == 'RCNN+RPN':
      rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss = self._losses["rpn_cross_entropy"].item(), \
                                                                        self._losses['rpn_loss_box'].item(), \
                                                                        -1, -1,\
                                                                        self._losses['total_loss'].item()
    #utils.timer.timer.tic('backward')
    train_op.zero_grad()
    self._losses['total_loss'].backward()
    #utils.timer.timer.toc('backward')
    train_op.step()

    self.delete_intermediate_states()


    return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss



  def train_step_with_summary(self, blobs, train_op):
    if cfg.TRAIN.IMS_PER_BATCH == 1:
      self.forward(blobs['data'], blobs['im_info'], blobs['gt_boxes'], None)
    if cfg.TRAIN.IMS_PER_BATCH == 2:
      self.forward(blobs['data'], blobs['im_info'], blobs['gt_boxes'], blobs['gt_boxes2'])
    if cfg.loss_strategy == 'NOCHANGE' or cfg.loss_strategy == 'RCNN+RPN':
      rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss = self._losses["rpn_cross_entropy"].item(), \
                                                                        self._losses['rpn_loss_box'].item(), \
                                                                        self._losses['cross_entropy'].item(), \
                                                                        self._losses['loss_box'].item(), \
                                                                        self._losses['total_loss'].item()
    if cfg.loss_strategy == 'RCNN_ONLY':
      rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss = -1, -1, \
                                                                        self._losses['cross_entropy'].item(), \
                                                                        self._losses['loss_box'].item(), \
                                                                        self._losses['total_loss'].item()
    if cfg.loss_strategy == 'NOCHANGE' or cfg.loss_strategy == 'RCNN+RPN':
      rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss = self._losses["rpn_cross_entropy"].item(), \
                                                                        self._losses['rpn_loss_box'].item(), \
                                                                        -1, -1,\
                                                                        self._losses['total_loss'].item()
    train_op.zero_grad()#
    self._losses['total_loss'].backward()
    train_op.step()
    summary = self._run_summary_op()

    self.delete_intermediate_states()

    return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, summary



  def train_step_no_return(self, blobs, train_op):
    if cfg.TRAIN.IMS_PER_BATCH == 1:
      self.forward(blobs['data'], blobs['im_info'], blobs['gt_boxes'], None)
    if cfg.TRAIN.IMS_PER_BATCH == 2:
      self.forward(blobs['data'], blobs['im_info'], blobs['gt_boxes'], blobs['gt_boxes2'])
    train_op.zero_grad()
    self._losses['total_loss'].backward()
    train_op.step()
    self.delete_intermediate_states()

  def load_state_dict(self, state_dict):
    """
    Because we remove the definition of fc layer in resnet now, it will fail when loading 
    the model trained before.
    To provide back compatibility, we overwrite the load_state_dict
    """
    nn.Module.load_state_dict(self, {k: state_dict[k] for k in list(self.state_dict())})

