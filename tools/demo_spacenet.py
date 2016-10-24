#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
import sys
# sys.path.append('/usr/local/lib/python2.7/site-packages')

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from utils.cython_nms import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import dlib
from skimage import io

# CLASSES = ('__background__',
#            'aeroplane', 'bicycle', 'bird', 'boat',
#            'bottle', 'bus', 'car', 'cat', 'chair',
#            'cow', 'diningtable', 'dog', 'horse',
#            'motorbike', 'person', 'pottedplant',
#            'sheep', 'sofa', 'train', 'tvmonitor')
CLASSES = ('__background__', 'building')
# CLASSES = ('__background__','n02958343','n02769748')

NETS = {'vgg16': ('VGG16',
                  'vgg16_fast_rcnn_iter_40000.caffemodel'),
        'vgg_cnn_m_1024': ('VGG_CNN_M_1024',
                           'vgg_cnn_m_1024_fast_rcnn_iter_40000.caffemodel'),
        'caffenet': ('CaffeNet',
                     'caffenet_fast_rcnn_iter_40000.caffemodel')}


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
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

def run_dlib_selective_search(image_name):
    img = io.imread(image_name)
    rects = []
    # dlib.find_candidate_object_locations(img,rects,min_size=100, max_merging_iterations=50)
    dlib.find_candidate_object_locations(img,rects,min_size=400)
    proposals = []
    for k,d in enumerate(rects):
        templist = [d.left(),d.top(),d.right(),d.bottom()]
        proposals.append(templist)
    proposals = np.array(proposals)
    return proposals

def demo(net, im_file, classes):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load pre-computed Selected Search object proposals
    #box_file = os.path.join(cfg.ROOT_DIR, 'data', 'demo',image_name + '_boxes.mat')
    #obj_proposals = sio.loadmat(box_file)['boxes']
    # im_file = os.path.join(cfg.ROOT_DIR, 'data', 'demo', image_name + '.jpg')
    # im_file = os.path.join('/media/wxie/UNTITLED/vision_log/rgb', image_name + '.jpg')

    # Dummy bounding box list with only 1 bounding box the size of the image
    # im = cv2.imread(im_file)
    # img_size_box = np.array([[0,0,im.shape[1]-1,im.shape[0]-1]])

    timer2 = Timer()
    timer2.tic()
    obj_proposals = run_dlib_selective_search(im_file)
    timer2.toc()
    print ('Proposal selective search took {:.3f}s').format(timer2.total_time)

    # Load the demo image
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im, obj_proposals)
    # scores, boxes = im_detect(net, im, img_size_box)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls in classes:
        cls_ind = CLASSES.index(cls)
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        keep = np.where(cls_scores >= CONF_THRESH)[0]
        cls_boxes = cls_boxes[keep, :]
        cls_scores = cls_scores[keep]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        # Filter out the overlaps
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        print 'All {} detections with p({} | box) >= {:.1f}'.format(cls, cls,
                                                                    CONF_THRESH)
        vis_detections(im, cls, dets, thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='caffenet')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    prototxt   = "/home/ubuntu/fast-rcnn/models/CaffeNetSpacenet/test_demo.prototxt"
    caffemodel = "/home/ubuntu/fast-rcnn/output/default/train/caffenetspacenet_fast_rcnn_iter_40000_bak.caffemodel"

    if not os.path.isfile(caffemodel):
        raise IOError(('Could not find caffemodel').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
        print("CPU MODE")
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        print("GPU MODE")
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    dataset_path = '/home/ubuntu/fast-rcnn/spacenet/data/'
    image_path = dataset_path + 'Images/'
    name_file_path = dataset_path + 'ImageSets/val.txt'
    output_path = '/home/ubuntu/fast-rcnn/spacenet/results/test/'

    with open(name_file_path, 'rb') as f:
        for line in f:
            filename = image_path + line.rstrip('\n') + '.png'
            print('')
            print(filename)
            timer = Timer()
            timer.tic()
            # demo(net, filename, ( 'person', 'pottedplant', 'bottle', 'sofa', 'tvmonitor'))
            demo(net, filename, CLASSES)
            timer.toc()
            print ('The entire detection took {:.3f}s').format(timer.total_time)
            filename_output = output_path + line.rstrip('\n') + ".png"
            plt.savefig(filename_output)
            # plt.show()
