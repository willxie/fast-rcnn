#!/bin/bash

# Convert all images from TIF to PNG
input_dir=/home/ubuntu/dataset/processedData/3band/
output_dir=/home/ubuntu/fast-rcnn/spacenet/data/Images/
for image in $input_dir*.tif; do
    file_name_w_ext="${image##*/}"
    file_name="${file_name_w_ext%%.*}"
    echo $file_name
    convert $image ${output_dir}/${file_name}.png
    # echo $file_name >> test.txt
done
