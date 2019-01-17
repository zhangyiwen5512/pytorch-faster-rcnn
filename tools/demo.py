#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

from model.config import cfg
import torch
import time

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_%d.pth',),'res101': ('res101_faster_rcnn_iter_%d.pth',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),
           'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',),
          }

datasets = {
            'baseline':['highway', 'pedestrians', 'office', 'PETS2006'],
            'cameraJitter':['badminton', 'traffic', 'boulevard', 'sidewalk'],
            'badWeather':['skating', 'blizzard', 'snowFall', 'wetSnow'],
            'dynamicBackground':['boats', 'canoe', 'fall', 'fountain01', 'fountain02', 'overpass'],
            'intermittentObjectMotion':['abandonedBox', 'parking', 'sofa', 'streetLight', 'tramstop', 'winterDriveway'],
            'lowFramerate':['port_0_17fps', 'tramCrossroad_1fps', 'tunnelExit_0_35fps', 'turnpike_0_5fps'],
            'nightVideos':['bridgeEntry', 'busyBoulvard', 'fluidHighway', 'streetCornerAtNight', 'tramStation', 'winterStreet'],
            'PTZ':['continuousPan', 'intermittentPan', 'twoPositionPTZCam', 'zoomInZoomOut'],
            'shadow':['backdoor', 'bungalows', 'busStation', 'copyMachine', 'cubicle', 'peopleInShade'],
            'thermal':['corridor', 'diningRoom', 'lakeSide', 'library', 'park'],
            'turbulence':['turbulence0', 'turbulence1', 'turbulence2', 'turbulence3']
}



def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""

    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect="equal")


    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time(), boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(torch.from_numpy(dets), NMS_THRESH)
        dets = dets[keep.numpy(), :]
        vis_detections(im, cls, dets, thresh=CONF_THRESH)


def drawBoundingBox(im, class_name, dets, thresh=0.5):

    inds = np.where(dets[:, -1] >= thresh)[0] # 找出大于0.8的序号，可以画出bounding box
    if len(inds) == 0:
        return
#    im = im[:, :, (2, 1, 0)]
    for i in inds:
        if class_name == CLASSES[20] \
            or class_name == CLASSES[18]\
            or class_name == CLASSES[17]\
            or class_name == CLASSES[16]\
            or class_name == CLASSES[14]\
            or class_name == CLASSES[13]\
            or class_name == CLASSES[12]\
            or class_name == CLASSES[11]\
            or class_name == CLASSES[10]\
            or class_name == CLASSES[9]\
            or class_name == CLASSES[8]\
            or class_name == CLASSES[5]\
            or class_name == CLASSES[3]\
            or class_name == CLASSES[19]\
            or class_name == CLASSES[2]\
            or class_name == CLASSES[1]:
            continue

        if class_name == CLASSES[15]:
          color = (0,255,0)
#        elif class_name == CLASSES[2]:
  #          color = (0,250,250)
        elif class_name == CLASSES[4]:
            color = (50,250,100)
        elif class_name == CLASSES[6]:
            color = (255,0,0)
        elif class_name == CLASSES[7]:
            color = (0,0,255)
 #       elif class_name == CLASSES[14]:
  #          color = (255,50,70)
#        elif class_name == CLASSES[1]:
 #           color = (0,255,0)
#        elif class_name == CLASSES[19]:
 #           color = (0,100,255)

        bbox = dets[i, :4]
        score = dets[i, -1]
        x = round(bbox[0])
        y = round(bbox[1])
        w = round(bbox[2] - bbox[0])
        h = round(bbox[3] - bbox[1])
        cv2.rectangle(im, (x,y), (x + w, y + h), color, 1)

        cv2.putText(im, class_name + "%.2f" % (score), (x,int(y + 10)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color,1)
        

def CDnet(net, image_name):
    # Load the demo image
    im_file = '/media/zhangyiwen/zhangyiwen/CDW2014/dataset/' + image_name + '/' + 'input'
    imgs = os.listdir(im_file)
    lens = len(imgs)
    for i in range(lens):
        if i == 0 :
            continue

        l = str(i)
        if l != 6:
          number = '0' * (6 - len(l)) + l
        image = im_file + '/in' + number + '.jpg'
        now = time.time()
        im = cv2.imread(image)

        scores, boxes = im_detect(net, im)#[300,21][300,84]

        CONF_THRESH = 0.8
        NMS_THRESH = 0.6
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)] # [300,4]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)# [300 ,5] [x,y,w,h,score]

            keep = nms(torch.from_numpy(dets), NMS_THRESH)
            dets = dets[keep.numpy(), :]
            drawBoundingBox(im, cls, dets, thresh=CONF_THRESH)

        cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
        im = cv2.resize(im, (640, 480), interpolation=cv2.INTER_NEAREST)# 最临近速度最快
        total_time = time.time() - now

        cv2.putText(im, "FPS:%.2f" % (1.0 / total_time), (10,20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,255),1)
        cv2.imshow("image", im)
        cv2.waitKey(10)




def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()


    if cfg.loss_strategy == 'NOCHANGE':
      tag = 'NOCHANGE'
    elif  cfg.loss_strategy == 'RCNN+RPN':
      tag = 'RCNN+RPN'
    elif  cfg.loss_strategy == 'RCNN_ONLY':
      tag = 'RCNN_ONLY'
    elif cfg.loss_strategy == 'RPN_ONLY':
      tag = 'RPN_ONLY'

    if tag != 'NOCHANGE':
       tag += '_loc' + str(cfg.MIX_LOCATION)

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    saved_model = os.path.join('output', demonet, DATASETS[dataset][0], tag,
                              NETS[demonet][0] %(70000 if dataset == 'pascal_voc' else 110000))


    if not os.path.isfile(saved_model):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(saved_model))

    # load network
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    else:
        raise NotImplementedError


    net.create_architecture(21,
                          tag=tag, anchor_scales=[8, 16, 32])

    net.load_state_dict(torch.load(saved_model, map_location=lambda storage, loc: storage))

    net.eval()
    if not torch.cuda.is_available():
        net._device = 'cpu'
    net.to(net._device)

    print('Loaded network {:s}'.format(saved_model))

    isCDnet = True

    if isCDnet == False:
      im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
                '001763.jpg', '004545.jpg']
      for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/demo/{}'.format(im_name))
        demo(net, im_name)
    else:
        for category, scene_list in datasets.items():
           for scene in scene_list:
               print("demo in:" + category + '/' + scene)
               print(time.strftime('End Time: %Y.%m.%d %H:%M:%S', time.localtime(time.time())))
               CDnet(net, category + '/' + scene)


    plt.show()
