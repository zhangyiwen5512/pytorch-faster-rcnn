# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
import cv2
from model.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob


def get_minibatch(roidb, num_classes):
  """Given a roidb, construct a minibatch sampled from it."""
  num_images = len(roidb)
  # Sample random scales to use for each image in this batch
  # 随机挑选一个尺度，作为这个batch的roi尺度，[0,n(尺度的数量)]，选取size个
  random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                  size=num_images)
  assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
    'num_images ({}) must divide BATCH_SIZE ({})'. \
    format(num_images, cfg.TRAIN.BATCH_SIZE)

###改
#######################################################################################################################
  if cfg.TRAIN.IMS_PER_BATCH == 1 :
    """
    一次处理一张图片
    """
    # Get the input image blob, formatted for caffe
    # 获取blob并调整格式
    im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

    blobs = {'data': im_blob}

    assert len(im_scales) == 1, "Single batch only"
    assert len(roidb) == 1, "Single batch only"

     # gt boxes: (x1, y1, x2, y2, cls)
    if cfg.TRAIN.USE_ALL_GT:
      # Include all ground truth boxes
      # 属于前景的图片
      gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
    else:
      # For the COCO ground truth boxes, exclude the ones that are ''iscrowd''
      gt_inds = np.where(roidb[0]['gt_classes'] != 0 & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
    #创建一个空boxes对象，将roidb的相应对象给他
    gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
    gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
    gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
    blobs['gt_boxes'] = gt_boxes
    blobs['im_info'] = np.array(
      [im_blob.shape[1], im_blob.shape[2], im_scales[0]],
      dtype=np.float32)

    return blobs

  elif cfg.TRAIN.IMS_PER_BATCH == 2 :
    """
    一次处理两张图片，做mixup
    """
#################################################################################################################
    im_blob, im_scales, mix_scale= _get_2_image_blob(roidb, random_scale_inds)
##################################################################################################################
    blobs = {'data': im_blob}

    assert len(im_scales) == 2, "Single batch only"
    assert len(roidb) == 2, "Single batch only"

    # gt boxes: (x1, y1, x2, y2, cls)
    if cfg.TRAIN.USE_ALL_GT:
      # Include all ground truth boxes
      gt_inds1 = np.where(roidb[0]['gt_classes'] != 0)[0]
      gt_inds2 = np.where(roidb[1]['gt_classes'] != 0)[0]

    else:
      # For the COCO ground truth boxes, exclude the ones that are ''iscrowd''
      gt_inds1 = np.where(roidb[0]['gt_classes'] != 0 & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
      gt_inds2 = np.where(roidb[1]['gt_classes'] != 0 & np.all(roidb[1]['gt_overlaps'].toarray() > -1.0, axis=1))[0]

    gt_boxes1 = np.empty((len(gt_inds1), 5), dtype=np.float32)
    gt_boxes1[:, 0:4] = roidb[0]['boxes'][gt_inds1, :] * im_scales[0]
    gt_boxes1[:, 4] = roidb[0]['gt_classes'][gt_inds1]

    gt_boxes2 = np.empty((len(gt_inds2), 5), dtype=np.float32)
    gt_boxes2[:, 0:4] = roidb[1]['boxes'][gt_inds2, :] * im_scales[1]
    gt_boxes2[:, 0] *= mix_scale[1]#x1
    gt_boxes2[:, 1] *= mix_scale[0]#y1
    gt_boxes2[:, 2] *= mix_scale[1]#x2
    gt_boxes2[:, 3] *= mix_scale[0]#y2
    gt_boxes2[:, 4] = roidb[1]['gt_classes'][gt_inds2]#cls

    blobs['gt_boxes'] = gt_boxes1
    blobs['gt_boxes2'] = gt_boxes2
    blobs['im_info'] = np.array(
      [im_blob.shape[1], im_blob.shape[2], im_scales[0]],
      dtype=np.float32)


    return blobs

  else:
    raise Exception("check cfg.TRAIN.IMS_PER_BACTH in /lib/model/config.py")

###########################################################################################################################

def _get_image_blob(roidb, scale_inds):
  """
  Builds an input blob from the images in the roidb at the specified
  scales.
  构建blob从roidb中，以scale_inds的size
  """
  num_images = len(roidb)
  processed_ims = []
  im_scales = []
  for i in range(num_images):
    im = cv2.imread(roidb[i]['image'])#读取图片
    if roidb[i]['flipped']:#水平翻转
      im = im[:, ::-1, :]
    target_size = cfg.TRAIN.SCALES[scale_inds[i]]#读取尺度
    """
    #在blob。py中,
    图像-设定好的均值
    im_scale = float(选取的尺度) / float(实际的长宽的小者)
    返回调整好的图片和缩放比,此时宽高比与原图一致
    """
    im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                    cfg.TRAIN.MAX_SIZE)
    im_scales.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images，
  blob = im_list_to_blob(processed_ims)

  return blob, im_scales


def _get_2_image_blob(roidb, scale_inds):
  """Builds an input blob from the images in the roidb at the specified
  scales.
  """
  num_images = len(roidb)

  processed_ims = []
  im_scales = []

  for i in range(num_images):
    im = cv2.imread(roidb[i]['image'])
    if roidb[i]['flipped']:
      im = im[:, ::-1, :]
    target_size = cfg.TRAIN.SCALES[scale_inds[i]]
    im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                    cfg.TRAIN.MAX_SIZE)
    im_scales.append(im_scale)
    processed_ims.append(im)


  lam = cfg.lamda
  im1 = processed_ims[0]
  im2 = processed_ims[1]

  #取得两者的shape
  shape1,shape2 = np.array(im1.shape, dtype=np.float32), np.array(im2.shape, dtype=np.float32)
  scale = shape1 / shape2
  #将im2调整至im1的大小
  im2 = cv2.resize(im2, None, None, fx=scale[1], fy=scale[0],
                  interpolation=cv2.INTER_LINEAR)

  assert im1.shape == im2.shape,"im1.shape:{}   im2.shape:{}   scale:{}".format(im1.shape, im2.shape, scale)

  #mixup
  im = lam * im1 + (1 - lam) * im2
  processed_ims = [im]
  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, im_scales, scale
