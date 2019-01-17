# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nets.network import Network
from model.config import cfg

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
#import torchvision.models as models
import nets.VGG16 as models
import numpy as np

class vgg16(Network):
  def __init__(self):
    Network.__init__(self)
    self._feat_stride = [16, ]
    self._feat_compress = [1. / float(self._feat_stride[0]), ]
    self._net_conv_channels = 512
    self._fc7_channels = 4096

  def _init_head_tail(self):
    self.vgg = models.vgg16()
    # Remove fc8
    self.vgg.classifier = nn.Sequential(*list(self.vgg.classifier._modules.values())[:-1])

    # Fix the layers before conv3:
    for layer in range(10):
      for p in self.vgg.features[layer].parameters(): p.requires_grad = False

    # not using the last maxpool layer
    self._layers['head'] = nn.Sequential(*list(self.vgg.features._modules.values())[:-1])

  def _image_to_head(self):
    net_conv = self._layers['head'](self._image)
    self._act_summaries['conv'] = net_conv
    
    return net_conv

  def _head_to_tail(self, pool5):
    lam = cfg.lamda
    pool5_flat = pool5.view(pool5.size(0), -1)
    if cfg.MIX_LOCATION != 0:
      cfg.layer4 = True

#    fc7 = self.vgg.classifier(pool5_flat)

    classifier = self.vgg.classifier._modules

    x = classifier['0'](pool5_flat)# linear1
    x = classifier['1'](x)# relu1
    x = classifier['2'](x)# dropout1
    if cfg.MIX_LOCATION == 1 and cfg.layer4 == True:
      rcnn_index = np.arange(x.size()[0])
      np.random.shuffle(rcnn_index)
      self.rcnn_mix_index = rcnn_index
      x = lam * x + (1 - lam) * x[rcnn_index, :]

    x = classifier['3'](x)# linear2
    x = classifier['4'](x)# relu2
    x = classifier['5'](x)# dropout2
    if cfg.MIX_LOCATION == 2 and cfg.layer4 == True:
      rcnn_index = np.arange(x.size()[0])
      np.random.shuffle(rcnn_index)
      self.rcnn_mix_index = rcnn_index
      x = lam * x + (1 - lam) * x[rcnn_index, :]


    cfg.layer4 = False
    fc7 = x
    return fc7

  def load_pretrained_cnn(self, state_dict):
    self.vgg.load_state_dict({k:v for k,v in state_dict.items() if k in self.vgg.state_dict()})
