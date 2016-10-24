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

CLASSES = ('__background__', 'building')

# a and b are bounding box coordinates (x1, y1, x2, y2)
def bb_intersection_area(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))

def bb_area(a):
    return abs((a[2] - a[0]) * (a[3] - a[1]))

def bb_union_area(a, b):
    return bb_area(a) + bb_area(b) - bb_intersection_area(a, b)

def bb_iou_area(a, b):
    return bb_intersection_area(a, b) / bb_union_area(a, b)

def process_detections(im, class_name, dets, true_dets, thresh=0.5):
    inds = np.where(dets[:, -1] >= thresh)[0]
    num_dets = len(inds)
    if num_dets == 0:
        return (0, 0, 0)

    remaining_true_dets = true_dets[:]
    num_true_dets = len(true_dets)
    true_pos = 0
    false_pos = 0
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        # Find if the durrent detection box has 50% overlap with ground truth
        for true_det in remaining_true_dets:
            matched = False
            if bb_iou_area(tuple(bbox), true_det) > 0.5:
                true_pos += 1
                matched = True
                # Allow only 1 patch per true detection
                remaining_true_dets.remove(true_det)
                break

        if not matched:
            false_pos += 1

    false_neg = num_true_dets - true_pos
    return (true_pos, false_pos, false_neg)

def vis_detections(im, class_name, dets, true_dets, thresh=0.5):
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

    for true_det in true_dets:
       ax.add_patch(
            plt.Rectangle((true_det[0], true_det[1]),
                          true_det[2] - true_det[0],
                          true_det[3] - true_det[1], fill=False,
                          edgecolor='green', linewidth=3.5)
            )


    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def run_dlib_selective_search(image_name):
    MIN_BB_SIZE = 300
    img = io.imread(image_name)
    rects = []
    dlib.find_candidate_object_locations(img,rects,min_size=MIN_BB_SIZE)
    proposals = []
    for k,d in enumerate(rects):
        templist = [d.left(),d.top(),d.right(),d.bottom()]
        proposals.append(templist)
    proposals = np.array(proposals)
    return proposals

def demo(net, threshold_list, annotation_path, file_path, classes, all_results, record_visual):
    """Detect object classes in an image using pre-computed object proposals."""
    timer2 = Timer()
    timer2.tic()
    obj_proposals = run_dlib_selective_search(file_path)
    timer2.toc()
    print ('Proposal selective search took {:.3f}s').format(timer2.total_time)

    # Load the demo image
    im = cv2.imread(file_path)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im, obj_proposals)
    # scores, boxes = im_detect(net, im, img_size_box)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Load ground truth annotation bounding boxes
    true_dets = []
    with open(annotation_path, 'rb') as f:
        for line in f:
            # Annotation is in x1, y1, x2, y2 format
            coord_list = line.rstrip('\n').split(" ")
            assert(len(coord_list) == 4)
            # Store as a tuple
            true_dets.append(tuple([int(coor) for coor in coord_list]))

    # Visualize detections for each class
    CONF_THRESH = 0.6
    NMS_THRESH = 0.5
    for cls in classes:
        # Skip background class
        if cls == '__background__':
            continue
        print("Class: {}".format(cls))
        for threshold in threshold_list:
            cls_ind = CLASSES.index(cls)
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            keep = np.where(cls_scores >= threshold)[0]
            cls_boxes = cls_boxes[keep, :]
            cls_scores = cls_scores[keep]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            # Filter out the overlaps
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            results = process_detections(im, cls, dets, true_dets, thresh=threshold)
            print("Threshold {}: true_pos={}   false_pos={}   false_neg={}".format(
                threshold, results[0], results[1], results[2]))
            all_results.setdefault(threshold, []).append(results)
            # Optional draw bounding box to the original image
            if threshold == CONF_THRESH and record_visual:
                vis_detections(im, cls, dets, true_dets, thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    # TODO move these to argument parser
    prototxt   = "/home/ubuntu/fast-rcnn/models/CaffeNetSpacenet/test_demo.prototxt"
    caffemodel = "/home/ubuntu/fast-rcnn/output/default/train/caffenetspacenet_fast_rcnn_iter_90000.caffemodel"

    # Data paths
    dataset_path = '/home/ubuntu/fast-rcnn/spacenet/data/'
    image_dir_path = dataset_path + 'Images/'
    annotation_dir_path = dataset_path + "Annotations/"
    name_file_path = dataset_path + 'ImageSets/test.txt'

    output_path = '/home/ubuntu/fast-rcnn/spacenet/results/test/'

    record_visual = False

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

    threshold_list = [x * 0.1 for x in range(1, 10)]

    # { threshold: [(true_pos, false_pos, false_neg), ...], ...}
    all_results = {}
    image_count = 0
    with open(name_file_path, 'rb') as f:
        for line in f:
            # Validation and test set image names are stored in respective files
            filename = line.rstrip('\n')
            file_path = image_dir_path + filename + '.png'
            print("Image number " + str(image_count))
            print(file_path)
            timer = Timer()
            timer.tic()

            # Annotaiton has the same name as the image
            annotation_path = annotation_dir_path + filename + ".txt"

            # Run detection and evaluation
            demo(net, threshold_list, annotation_path, file_path, CLASSES, all_results, record_visual)
            timer.toc()
            print ('The entire detection took {:.3f}s').format(timer.total_time)

            # Record visual outputs
            if record_visual:
                filename_output = output_path + filename + ".png"
                # plt.show()
                plt.savefig(filename_output)
                plt.close('all')

            print('')
            image_count += 1

    print("\n\nTotal\n\n")
    print("(true_pos, false_pos, false_neg)")
    for threshold in threshold_list:
        result_list = all_results[threshold]
        result_zip = zip(*result_list)
        true_pos  = sum(result_zip[0])
        false_pos = sum(result_zip[1])
        false_neg = sum(result_zip[2])
        print("{}\t{}\t{}".format(true_pos, false_pos, false_neg))
        print("Threshold {}: recall = {}   precision = {}".format(threshold,
                                                                  float(true_pos) / (true_pos + false_neg),
                                                                  float(true_pos) / (true_pos + false_pos)))

if __name__ == '__main__':
    main()
