import simplejson
import matplotlib.pyplot as plt
from random import shuffle
import colorsys
import numpy as np
import cv2

from data_processing import TempDataProcessing, ClusterProcessing, DataPreparation
from region_growth import RegionGrowth
from data_visualization import DataVisualization as dv

SAVE_IMAGE = 0
SHOW_IMAGE = 1

VISUALIZE = 0


def data_scaling(data, interval):
    scaled_data = []
    cols = len(data[0])
    jump = int(interval/5)
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
    label = np.ones((r * 50, c, ch)) * (-1)

    # generate colors for clusters
    color_gen = get_distinct_colors(num_clusters)
    colors = []
    for c in color_gen:
        colors.append(c)
    shuffle(colors)

    for i in clusters:
        for point in clusters[i]:
            r, c = point
            image[r * 50:(r + 1) * 50, c] = colors[i-1]
            label[r * 50:(r + 1) * 50, c] = i

    return image, label


if __name__ == "__main__":
    # create dalaloader
    print("********************************************************************************")
    print("Hyperparameters")
    thresh = 0.9
    scale = 60

    print("********************************************************************************")
    print("prepare data")
    # obj_data = TempDataProcessing("../data/180724_180810_mod.csv")
    # obj_data = TempDataProcessing("../data/toy_example.csv")
    obj_data = DataPreparation(subject_id=2,
                               num_days=15,
                               dir_name="../data/experimental_data/Subject_2/processed_data")
                               # file_name="xandem_2018-12-02.log")
    # print("number of days:", len(obj_data.image))
    # print("number of intervals: ", len(obj_data.image[0]))
    # data_dims = (len(obj_data.image), len(obj_data.image[0]), 3)
    # with open("../data/image.log", 'w') as image_log:
    #     simplejson.dump(obj_data.img, image_log)
    data = obj_data.image.copy()
    del obj_data

    print("Scaling the image by a factor of: ", int(scale/5))
    data = data_scaling(data, interval=scale)
    print("number of days:", len(data))
    print("number of intervals: ", len(data[0]))
    data_dims = (len(data), len(data[0]), 3)

    print("********************************************************************************")
    print("performing first pass.....")
    obj_rg = RegionGrowth()
    first_pass_clusters, label_fp = obj_rg.first_pass(data)
    print("number of clusters: ", len(first_pass_clusters))
    # print(first_pass_clusters[1])
    # with open("../data/image.log", 'w') as image_log:
    #     image_log.write("********************************************************************************\n")
    #     image_log.write("First Pass Clusters\n")
    #     simplejson.dump(first_pass_clusters, image_log)

    print("preparing visual results for first pass.....")
    img_fp, label = plot_cluster(first_pass_clusters, data_dims)

    if SHOW_IMAGE:
        cv2.namedWindow("First Pass Clusters", cv2.WINDOW_NORMAL)
        cv2.imshow("First Pass Clusters", img_fp)
        # cv2.waitKey(0)

    if SAVE_IMAGE:
        cv2.imwrite("../graphs/first_pass_cluster_"+str(scale)+".jpg", img_fp)

    if VISUALIZE:
        obj_dv = dv(img_fp, label_fp)
        obj_dv.feature_comparison()

    print("preparing features of first pass clusters")
    obj_fp_data = ClusterProcessing(interval=scale)
    # cluster_feat_fp = obj_sp_data.get_cluster_histograms(data, first_pass_clusters)
    cluster_feat_fp = obj_fp_data.get_cluster_features(data, first_pass_clusters)
    print(cluster_feat_fp.keys())

    print("********************************************************************************")
    print("merging extra short activities")
    clusters_sm, label_sm = obj_rg.merge_short_activities(first_pass_clusters, cluster_feat_fp, label_fp, 60)
    print("number of clusters: ", len(clusters_sm))
    print("preparing visual results for first pass.....")
    img_sm, label = plot_cluster(clusters_sm, data_dims)

    if SHOW_IMAGE:
        cv2.namedWindow("short activities merged", cv2.WINDOW_NORMAL)
        cv2.imshow("short activities merged", img_sm)
        cv2.waitKey(0)

    if SAVE_IMAGE:
        cv2.imwrite("../graphs/short_clusters_merged_" + str(scale) + ".jpg", img_sm)

    print("preparing features of Short-activity merged clusters")
    obj_sm_data = ClusterProcessing(interval=scale)
    # cluster_feat_fp = obj_sp_data.get_cluster_histograms(data, first_pass_clusters)
    cluster_feat_sm = obj_sm_data.get_cluster_features(data, first_pass_clusters)
    print(cluster_feat_sm.keys())

    print("********************************************************************************")
    print("performing second pass.....")
    second_pass_clusters = obj_rg.cluster_by_time_hist(first_pass_clusters, cluster_feat_fp, thresh)
    print("number of clusters: ", len(second_pass_clusters))
    print("preparing visual results for second pass.....")
    img_sp, label_sp = plot_cluster(second_pass_clusters, data_dims)

    if SHOW_IMAGE:
        cv2.imshow("Second Pass Clusters", img_sp)
        # cv2.waitKey(0)

    if SAVE_IMAGE:
        cv2.imwrite("../graphs/second_pass_cluster_"+str(scale)+"_"+str(thresh)+".jpg", img_sp)

    if SHOW_IMAGE:
        cv2.waitKey(0)
    exit(1)
