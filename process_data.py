import json
from pprint import pprint

import numpy as np
import cv2

def main():
    json_path = '/home/ubuntu/dataset/processedData/vectorData/summaryData/AOI_1_Rio_polygons_solution_3Band.geojson'
    #json_path = '/home/ubuntu/dataset/processedData/geoJson/013022223130_Public_img4_Geo.geojson'

    image_clip_path = '/home/ubuntu/dataset/processedData/3band/'

    with open(json_path, 'rb') as f:
        geo_json = json.load(f)

    assert(geo_json["type"] == "FeatureCollection")
    feature_collection = geo_json["features"]

    output_counter = 0
    for feature_dict in feature_collection:
        assert(feature_dict["type"] == "Feature")

        # Skip features / clips with no buildings
        building_id = feature_dict["properties"]["BuildingId"]
        if building_id == -1:
            continue

        # Some features for some reason has no coordinate
        if not feature_dict["geometry"]["coordinates"]:
            continue

        if len(feature_dict["geometry"]["coordinates"]) != 1:
            print("\tThis entry has odd polygon:")

        # Grab the image clip
        image_id = feature_dict["properties"]["ImageId"]
        print(image_id + "---" + str(building_id))
        img = cv2.imread(image_clip_path + "3band_" + image_id + ".tif")


        coordinate_list = []
        coordinate_list_np = []
        if feature_dict["geometry"]["type"] == "Polygon":
            # We are making the assumption here that the bounding box is a simply polygon
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
            cv2.line(img, coordinate_list[coordinate_index], coordinate_list[coordinate_index + 1], color, thickness)

        # print(np.array(coordinate_list))
        # print(coordinate_list_np)
        # print(np.array([[1,1], [2,2]]))

        # img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # contours, hierarchy = cv2.findContours(img2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # for cnt in contours:
        #     print(type(cnt))
        #     print(cnt.shape)
        #     print(cnt.dtype)
        #     print(cnt)
        #     x, y, w, h = cv2.boundingRect(cnt)
        #     cv2.rectangle(img, (x, y),(x + w, y + h), (0, 255, 0), 2)


        x, y, w, h = cv2.boundingRect(np.expand_dims(np.array(coordinate_list), axis=1))
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)

        output_counter += 1

        cv2.imshow('image',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # cv2.imwrite("output_imgs/" + image_id + "_" + str(building_id) + ".jpg", img)

if __name__ == "__main__":
    main()
