#!/usr/bin/python

import sys
import os
import dlib
import scipy.io as sio
from skimage import io
import numpy as np

sys.path.append('/usr/local/lib/python2.7/site-packages')

min_building_size = 400 # 20 x 20

def run_dlib_selective_search(image_name):
    img = io.imread(image_name)
    rects = []
    dlib.find_candidate_object_locations(img,rects,min_size=min_building_size)
    proposals = []
    for k,d in enumerate(rects):
        # templist = [d.left(),d.top(),d.right(),d.bottom()]
        # Matlab's odd format [top, left, bottom, right], 1-based index
        templist = [d.top(),d.left(),d.bottom(),d.right()]
        proposals.append(templist)
    proposals = np.array(proposals)
    return proposals

dataset_path = '/home/ubuntu/fast-rcnn/spacenet/data/'
image_path = dataset_path + 'Images/'
name_file_path = dataset_path + 'ImageSets/train.txt'
mat_path = "train.mat"

count = 0
all_proposals = []
imagenms = []

with open(name_file_path, 'rb') as f:
    for line in f:
	filename = image_path + line.rstrip('\n') + '.png'
	single_proposal = run_dlib_selective_search(filename)
	print("{}:\t{}\t{}".format(count, filename, single_proposal.shape))
	all_proposals.append(single_proposal)
	count += 1

print "Saving to .mat to {}".format(mat_path)
sio.savemat(mat_path,mdict={'all_boxes':all_proposals,'images':imagenms})
print "Saved!"
print "Loading .mat"
obj_proposals = sio.loadmat(mat_path)
print obj_proposals
