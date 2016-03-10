#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an image database."""



import _init_paths


from fast_rcnn.config import cfg, get_output_dir
import argparse
from utils.timer import Timer
import numpy as np
import cv2
import caffe
from utils.cython_nms import nms
import cPickle
import heapq
from utils.blob import im_list_to_blob
import os
import matplotlib.pyplot as plt


from fast_rcnn.test import test_net, im_detect
# import fast_rcnn.test
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import caffe
import argparse
import pprint
import time, os, sys

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='voc_2007_test', type=str)
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args



class Extractor(caffe.Net):
    def __init__(self, model_file, pretrained_file, mean_file):
        caffe.Net.__init__(self, model_file, pretrained_file, caffe.TEST)

        mean = np.load(mean_file)
        if mean.shape[1:] != (1, 1):
            mean = mean.mean(1).mean(1)

        in_ = self.inputs[0]
        self.transformer = caffe.io.Transformer({in_: self.blobs[in_].data.shape})
        self.transformer.set_transpose(in_, (2,0,1))
        self.transformer.set_mean(in_, mean)
        self.transformer.set_channel_swap(in_, (2,1,0))
        self.features = ['fc7']

    def set_features(self, feature_layer):
        self.features = feature_layer

    def extract_features(self, images):
        in_ = self.inputs[0]
        caffe_in = np.zeros((len(images),)+self.blobs[in_].data.shape[1:])
        for ix, image in enumerate(images):
            caffe_in[ix] = self.transformer.preprocess(in_, caffe.io.load_image(image))
        features = self.forward_all(**{in_: caffe_in, 'blobs': self.features})
        return features

    def extract_feature(self, image):
        in_ = self.inputs[0]
        self.blobs[in_].data[...] = self.transformer.preprocess(in_, caffe.io.load_image(image))
        feature = self.forward(**{'blobs': self.features})
        feature = {blob: vals[0] for blob, vals in feature.iteritems()}
        return feature

def extractor_factory():
    model_def = os.path.join(caffe_root, "models/bvlc_reference_caffenet/deploy.prototxt")
    pretrained_model = os.path.join(caffe_root, "models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel")
    mean_file = os.path.join(pycaffe_dir, 'caffe/imagenet/ilsvrc_2012_mean.npy')

    extractor = Extractor(model_def, pretrained_model, mean_file)
    return extractor

def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.imshow(data)
    # plt.savefig("visualize_conv.jpg")
    # plt.close()

def vis_fc(feature):
    plt.plot(feature)
    plt.xlim(xmax=feature.shape[0])
    plt.savefig("visualize_fc.jpg")
    plt.close()


def vis_detections(im, class_name, dets, ax, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    # if len(inds) == 0:
    #     return

    im = im[:, :, (2, 1, 0)]

    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.0)
            )
        # ax.text(bbox[0], bbox[1] - 2,
        #         '{:s} {:.3f}'.format(class_name, score),
        #         bbox=dict(facecolor='blue', alpha=0.5),
        #         fontsize=14, color='white')

    # ax.set_title(('{} detections with '
    #               'p({} | box) >= {:.1f}').format(class_name, class_name,
    #                                               thresh),
    #               fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    while not os.path.exists(args.caffemodel) and args.wait:
        print('Waiting for {} to exist...'.format(args.caffemodel))
        time.sleep(10)

    # caffe.set_mode_gpu()
    # caffe.set_device(args.gpu_id)
    caffe.set_mode_cpu()

    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]

    imdb = get_imdb(args.imdb_name)
    imdb.competition_mode(args.comp_mode)


#     test_net(net, imdb)
# def test_net(net, imdb):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    # heuristic: keep an average of 40 detections per class per images prior
    # to NMS
    max_per_set = 40 * num_images
    # heuristic: keep at most 100 detection per class per image prior to NMS
    max_per_image = 100
    # detection thresold for each class (this is adaptively set based on the
    # max_per_set constraint)
    thresh = -np.inf * np.ones(imdb.num_classes)
    # top_scores will hold one minheap of scores per class (used to enforce
    # the max_per_set constraint)
    top_scores = [[] for _ in xrange(imdb.num_classes)]
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, net)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    roidb = imdb.roidb


    # Load imagenet data instead!
    # imagenet_dir = "/var/local/wxie/cs381V.grauman/ImageNet/CLS_LOC2014/ILSVRC2012_img_test/"
    imagenet_dir = "/home/users/wxie/fast-rcnn/data/imagenet_demo/cat/"
    correct_count = 0
    target_class = 'dogg'
    file_name_list = os.listdir(imagenet_dir)

    #########################################################

    # for i in xrange(num_images):
    for i in xrange(100):

        # print(imdb.image_path_at(i))
        im = cv2.imread(imdb.image_path_at(i))
        # im = cv2.imread(imagenet_dir + file_name_list[i])

        _t['im_detect'].tic()
        img_size_box = np.array([[0,0,im.shape[1]-1,im.shape[0]-1]])

        scores, boxes = im_detect(net, im, roidb[i]['boxes'])
        # scores, boxes = im_detect(net, im, img_size_box)

        _t['im_detect'].toc()
        print(roidb[i]['boxes'].shape)
        print(scores.shape)
        print(boxes.shape)

        CLASSES = ('__background__',
                   'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor')

        classes = (
                   'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor')

# (96, 213, 150)
# (256, 107, 76)
# (384, 54, 39)
# (384, 54, 39)
# (256, 54, 39)

        # layer = 'conv1'
        # feat = net.blobs[layer].data[0]
        # vis_square(feat[:], padval = 0.5)
        # print(feat.shape)
        # plt.show()

        # layer = 'conv2'
        # feat = net.blobs[layer].data[0]
        # vis_square(feat[:], padval = 0.5)
        # print(feat.shape)
        # plt.show()


        # layer = 'conv3'
        # feat = net.blobs[layer].data[0]
        # vis_square(feat[:], padval = 0.5)
        # print(feat.shape)
        # plt.show()

        # layer = 'conv4'
        # feat = net.blobs[layer].data[0]
        # vis_square(feat[:], padval = 0.5)
        # print(feat.shape)
        # plt.show()


        # layer = 'conv5'
        # feat = net.blobs[layer].data[0]
        # vis_square(feat[:], padval = 0.5)
        # print(feat.shape)

        # raw_input("enter")
        # plt.close('all')

##################################################

# (96, 3, 11, 11)
# (256, 48, 5, 5)
# (384, 256, 3, 3)
# (384, 192, 3, 3)
# (256, 192, 3, 3)

        # layer = 'conv1'
        # filters = net.params[layer][0].data
        # print(filters.shape)
        # vis_square(filters.transpose(0,2,3,1))

        # plt.show()


        # layer = 'conv2'
        # filters = net.params[layer][0].data
        # print(filters.shape)

        # vis_square(filters[:20].reshape(20 * 48, 5, 5))
        # plt.show()

        # layer = 'conv3'
        # filters = net.params[layer][0].data
        # print(filters.shape)

        # vis_square(filters[:20].reshape(20 * 256, 3, 3))
        # plt.show()

        # layer = 'conv4'
        # filters = net.params[layer][0].data
        # print(filters.shape)

        # vis_square(filters[:20].reshape(20 * 192, 3, 3))
        # plt.show()

        # layer = 'conv5'
        # filters = net.params[layer][0].data
        # print(filters.shape)

        # vis_square(filters[:20].reshape(20 * 192, 3, 3))
        # plt.show()


        # raw_input("enter")
        # plt.close('all')


##################################################

        fig, ax = plt.subplots(figsize=(12, 12), sharey=True)

        # Visualize detections for each class
        CONF_THRESH = 0.8
        NMS_THRESH = 0.3

        # print(boxes.shape)   # num_boxes, coordinate of each boxes
        # print(boxes[0])
        # raw_input("Press Enter to continue...")
        class_score_list = []

        for cls in classes:
            cls_ind = CLASSES.index(cls)
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]

            # Track the score
            for score in cls_scores:
                class_score_list.append((cls, score))

            # # TODO remove this test code
            # cls_boxes = img_size_box
            # #######################

            keep = np.where(cls_scores >= CONF_THRESH)[0]
            cls_boxes = cls_boxes[keep, :]
            cls_scores = cls_scores[keep]

            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]

            # print 'All {} detections with p({} | box) >= {:.1f}'.format(cls, cls,
            #                                                             CONF_THRESH)
            cls = ""            # No print class for box
            vis_detections(im, cls, dets, ax, thresh=CONF_THRESH)

        # Optional regression box draw
        CONF_THRESH = 0.0
        NMS_THRESH = 0.3
        fig2, ax = plt.subplots(figsize=(12, 12), sharey=True)
        for cls in classes:
            cls_ind = CLASSES.index(cls)
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            skip_this = False
            # Hacky way to only show the max bounding box
            for score in scores[0, 1:]:
                if cls_scores[0] < score:
                    skip_this = True
            if skip_this:
                continue

            keep = np.where(cls_scores >= CONF_THRESH)[0]
            cls_boxes = cls_boxes[keep, :]
            cls_scores = cls_scores[keep]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]

            vis_detections(im, cls, dets, ax, thresh=CONF_THRESH)


        # Get top 5 prediction scores
        class_score_list = sorted(class_score_list, key=lambda x: x[1], reverse=True)
        class_score_list = class_score_list[:5]
        if class_score_list[0][0] == target_class:
            correct_count += 1

        # print(class_score_list)

        # Print class confidence bar graph
        fig3, ax = plt.subplots()
        ind = np.arange(len(class_score_list))
        width = 0.4
        ax.bar(ind, [x[1] for x in class_score_list], width, color='r')
        plt.xticks(ind + width/2, [x[0] + '\n' + str(x[1]) for x in class_score_list])

        # plt.show()
        dir_name = target_class
        # fig.savefig("/home/users/wxie/fast-rcnn/output/"+dir_name+"/{:06d}.jpg".format(i))
        # fig2.savefig("/home/users/wxie/fast-rcnn/output/"+dir_name+"/{:06d}_2.jpg".format(i))
        # fig3.savefig("/home/users/wxie/fast-rcnn/output/"+dir_name+"/{:06d}_3.jpg".format(i))
        plt.close('all')

        # _t['misc'].tic()
        # for j in xrange(1, imdb.num_classes):
        #     inds = np.where((scores[:, j] > thresh[j]) &
        #                     (roidb[i]['gt_classes'] == 0))[0]
        #     cls_scores = scores[inds, j]
        #     cls_boxes = boxes[inds, j*4:(j+1)*4]
        #     top_inds = np.argsort(-cls_scores)[:max_per_image]
        #     cls_scores = cls_scores[top_inds]
        #     cls_boxes = cls_boxes[top_inds, :]
        #     # push new scores onto the minheap
        #     for val in cls_scores:
        #         heapq.heappush(top_scores[j], val)
        #     # if we've collected more than the max number of detection,
        #     # then pop items off the minheap and update the class threshold
        #     if len(top_scores[j]) > max_per_set:
        #         while len(top_scores[j]) > max_per_set:
        #             heapq.heappop(top_scores[j])
        #         thresh[j] = top_scores[j][0]

        #     all_boxes[j][i] = \
        #             np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
        #             .astype(np.float32, copy=False)

        #     if 0:
        #         keep = nms(all_boxes[j][i], 0.3)
        #         vis_detections(im, imdb.classes[j], all_boxes[j][i][keep, :])
        # _t['misc'].toc()

        print 'im_detect: {:d}/{:d} {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time)

    print("correct count for class {}: {}".format(target_class,correct_count))
    # for j in xrange(1, imdb.num_classes):
    #     for i in xrange(num_images):
    #         inds = np.where(all_boxes[j][i][:, -1] > thresh[j])[0]
    #         all_boxes[j][i] = all_boxes[j][i][inds, :]

    # det_file = os.path.join(output_dir, 'detections.pkl')
    # with open(det_file, 'wb') as f:
    #     cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    # print 'Applying NMS to all detections'
    # nms_dets = apply_nms(all_boxes, cfg.TEST.NMS)

    # print 'Evaluating detections'
    # imdb.evaluate_detections(nms_dets, output_dir)
