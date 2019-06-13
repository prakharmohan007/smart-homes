import simplejson
import matplotlib.pyplot as plt
from random import shuffle
import colorsys
import numpy as np
import cv2

from data_processing import GenerateSyntheticCluster
from region_growth import RegionGrowth
from data_visualization import DataVisualization as dv

# from data_processing import TempDataProcessing, ClusterProcessing, DataPreparation
# from region_growth import RegionGrowth

SAVE_IMAGE = 1
SHOW_IMAGE = 0

VISUALIZE = 1


def data_scaling(data, interval):
    scaled_data = []
    cols = len(data[0])
    jump = int(interval / 5)
    for row in data:
        data_row = []
        jump_count = jump
        for col in row:
            if jump_count == jump:
                data_row.append(col)
                jump_count = 0
            jump_count += 1
        scaled_data.append(data_row)
    return scaled_data


def HSV_to_RGB(h, s, v):
    (r, g, b) = colorsys.hsv_to_rgb(h, s, v)
    return int(255 * r), int(255 * g), int(255 * b)


def get_distinct_colors(n):
    hue_partition = 1.0 / (n + 1)
    return (HSV_to_RGB(hue_partition * value, 1.0, 1.0) for value in range(0, n))


def plot_cluster(clusters, dims):
    num_clusters = len(clusters)
    r, c, ch = dims
    image = np.zeros((r * 50, c, ch), np.uint8)
    label = np.ones((r * 50, c), np.int) * (-1)

    # generate colors for clusters
    color_gen = get_distinct_colors(num_clusters)
    colors = []
    for c in color_gen:
        colors.append(c)
    shuffle(colors)

    for i in clusters:
        for point in clusters[i]:
            r, c = point
            image[r * 50:(r + 1) * 50, c] = colors[i - 1]
            label[r * 50:(r + 1) * 50, c] = i

    return image, label


if __name__ == "__main__":
    # create dataloader
    print("********************************************************************************")
    print("Hyperparameters")
    thresh = 0.9
    scale = 30

    print("********************************************************************************")
    print("prepare data")

    obj_data = GenerateSyntheticCluster(
        routine_type="ADL1",
        file_path="../data/synthetic_data/level2/parsed_data/synt_data_lvl2_days30_sd5_1.csv",
        scale=30)

    data = obj_data.data
    clusters_feat = obj_data.get_cluster_features()
    cluster_pixels = obj_data.get_cluster_pixels()
    num_days = obj_data.num_days
    num_intervals = int(24 * 60 * 60 / scale)
    print("number of days:", obj_data.num_days)
    # print("number of intervals: ", len(obj_data.image[0]))
    # data_dims = (len(obj_data.image), len(obj_data.image[0]), 3)
    # with open("../data/image.log", 'w') as image_log:
    #     simplejson.dump(obj_data.img, image_log)

    # del obj_data
    dims = (num_days, num_intervals, 3)
    print("preparing visual results for clusters.....")
    img_fp, label = plot_cluster(cluster_pixels, dims)
    if SHOW_IMAGE:
        cv2.namedWindow("First Pass Clusters", cv2.WINDOW_NORMAL)
        cv2.imshow("First Pass Clusters", img_fp)
        # cv2.waitKey(0)

    print("********************************************************************************")
    print("performing Hierarchical merging.....")
    obj_rg = RegionGrowth()
    success = True
    while success:
        print("number of clusters before merging: ", len(cluster_pixels))
        cluster_elements, cluster_pixels, success = obj_rg.region_growth(cluster_pixels, clusters_feat, thresh=0.6)
        print("number of clusters after merging: ", len(cluster_pixels))
        print("Preparing features for new clusters")
        clusters_feat = obj_data.merge_cluster_features(orig_clusters_features=clusters_feat,
                                                        new_cluster_info=cluster_elements)
        if len(cluster_pixels) != len(clusters_feat):
            print("number of clusters in cluster_pixel and cluster_feat are different")

        # input()

    print("preparing visual results for clusters.....")
    img_sp, label_sp = plot_cluster(cluster_pixels, dims)
    if SHOW_IMAGE:
        cv2.namedWindow("Second Pass Clusters", cv2.WINDOW_NORMAL)
        cv2.imshow("Second Pass Clusters", img_sp)
        cv2.waitKey(0)

    if SAVE_IMAGE:
        cv2.imwrite("time_nhi_lvl2.jpg", img_sp)

    if VISUALIZE:
        obj_dv = dv(img_sp, label_sp)
        obj_dv.feature_comparison(clusters_feat)

    exit(1)