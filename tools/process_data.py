#!/usr/bin/python

import json
import numpy as np
import cv2
import os

def main():
    view_bounding_box = False

    # Train, val, test set index split (3:1:1)
    num_images  =  4388  # number of images with buildings in them
    total_range = range(num_images)
    train_range = total_range[:num_images // 5 * 3]
    val_range   = total_range[num_images // 5 * 3 : num_images // 5 * 4]
    test_range  = total_range[num_images // 5 * 4 :]
    print("num_train: {}     num_val: {}     num_test: {}".format(len(train_range),
                                                             len(val_range),
                                                             len(test_range)))
    print("num_sum: {}".format(len(train_range) + len(val_range) + len(test_range)))
    print("num_images: {}".format(num_images))

    # Image chips source
    image_clip_path = '/home/ubuntu/dataset/processedData/3band/'

    # Bounding box annotaiton source
    json_path = '/home/ubuntu/dataset/processedData/vectorData/summaryData/AOI_1_Rio_polygons_solution_3Band.geojson'

    # Output destinations
    dataset_output_path = '/home/ubuntu/fast-rcnn/spacenet/data/'
    annotation_path = dataset_output_path + 'Annotations/'
    image_set_path  = dataset_output_path + 'ImageSets/'
    image_path      = dataset_output_path + 'Images/'

    # Outputs for bounding box overlay original image chip
    chip_bb_path = 'chips/'

    # Band
    band_prefix = '3band_'

    with open(json_path, 'rb') as f:
        geo_json = json.load(f)

    assert(geo_json["type"] == "FeatureCollection")
    feature_collection = geo_json["features"]

    image_counter = -1
    image_name_dict = {}

    # Each iteration and entry contains 1 bounding box
    for feature_dict in feature_collection:
        assert(feature_dict["type"] == "Feature")

        # Skip features / clips with no buildings
        building_id = feature_dict["properties"]["BuildingId"]
        if building_id == -1:
            continue

        # Some features for some reason has no coordinate, skip them
        if not feature_dict["geometry"]["coordinates"]:
            continue

        # This shouldn't happen in our dataset.
        if len(feature_dict["geometry"]["coordinates"]) != 1:
            print("\tThis entry has odd polygon:")
            continue

        # Grab the associated image clip
        image_id = feature_dict["properties"]["ImageId"]
        print(image_id + "     building id: " + str(building_id) +
              "     image count:" + str(image_counter))
        img = cv2.imread(image_clip_path + "3band_" + image_id + ".tif")

        coordinate_list = []
        coordinate_list_np = []
        if feature_dict["geometry"]["type"] == "Polygon":
            # We are making the assumption here that the bounding box is a simple polygon
            for coordinate in feature_dict["geometry"]["coordinates"][0]:
                x = int(coordinate[0])
                y = int(coordinate[1])
                z = int(coordinate[2])
                coordinate_list.append((x, y))
                coordinate_list_np.append(np.array((x, y)))
                # print("{}, {}, {}".format(x, y, z))
        else:
            print("********** Coordinate type not found **********")

        for coordinate_index in range(len(coordinate_list) - 1):
            color = (0, 0, 255)
            thickness = 2
            # cv2.line(img, coordinate_list[coordinate_index], coordinate_list[coordinate_index + 1], color, thickness)

        # Find the rectangular bounding box for the target
        # Need to change into an odd format that boundingRect takes
        x, y, w, h = cv2.boundingRect(np.expand_dims(np.array(coordinate_list), axis=1))
        # cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 1)


        # Write trian.txt
        if image_id not in image_name_dict:
            # Tag the image_id so we don't write double
            image_name_dict[image_id] = 1
            image_counter += 1
            if image_counter in train_range:
                with open(image_set_path + "train.txt", 'ab') as f:
                    f.write(band_prefix + image_id + "\n")
            if image_counter in val_range:
                with open(image_set_path + "val.txt", 'ab') as f:
                    f.write(band_prefix + image_id + "\n")
            if image_counter in test_range:
                with open(image_set_path + "test.txt", 'ab') as f:
                    f.write(band_prefix + image_id + "\n")

        # Write x_min, y_min, x_max, y_max as annotation required for fast-rcnn
        # TODO write annotaiton for train and val in another dir
        if image_counter in train_range:
            with open(annotation_path + band_prefix + image_id + ".txt", "ab") as f:
                f.write("{} {} {} {}\n".format(x, y, x+w, y+h))

            # Record converted image type file if it doesn't exist already
            current_image_path = image_path + band_prefix + image_id + ".png"
            if not os.path.isfile(current_image_path):
                cv2.imwrite(current_image_path, img)

        if view_bounding_box:
            cv2.imshow('image',img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # cv2.imwrite(chip_bb_path + image_id + "_" + str(building_id) + ".jpg", img[y:y+h, x:x+w])




if __name__ == "__main__":
    main()
