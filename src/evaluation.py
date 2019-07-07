from random import shuffle
import colorsys
import numpy as np
import cv2
import matplotlib.pyplot as plt

from data_processing import GenerateSyntheticCluster
from region_growth import RegionGrowth
from data_visualization import DataVisualization as dv

SAVE_IMAGE = 1
SHOW_IMAGE = 0
VISUALIZE = 0


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


def plot_graph(x, y, graph_name, i_name, y_axis):
    fig = plt.figure()
    plt.plot(y, x, 'b')
    plt.plot(y, x, 'b', marker='o')
    plt.ylim(ymin=0)
    plt.xlabel("standard deviation")
    plt.ylabel(y_axis)
    plt.title(graph_name)
    plt.savefig("../data/synthetic_data/graphs/" + i_name + "_" + graph_name + ".png")


def total_variance(cluster_feat, cluster_elements):
    variance = 0.0
    for region in cluster_elements:
        stime = []
        dur = []
        for c_id in cluster_elements[region]:
            stime.append(cluster_feat[c_id]["stime"])
            dur.append(cluster_feat[c_id]["duration"])
        stime = np.array(stime)
        dur = np.array(dur)

        reg_var = np.var(stime) + np.var(dur)
        variance += reg_var
    return variance


def clustering(lvl, f_name):
    filepath = "../data/synthetic_data/level" + str(lvl) + "/parsed_data/" + f_name + ".csv"
    i_name = f_name + ".jpg"

    obj_data = GenerateSyntheticCluster(
        routine_type="ADL1",
        file_path=filepath,
        scale=30)

    data = obj_data.data
    clusters_feat = obj_data.get_cluster_features()
    init_cluster_feat = clusters_feat.copy()
    cluster_pixels = obj_data.get_cluster_pixels()
    cluster_coarse = obj_data.get_cluster_coarse()
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

    print(" TIME-DURATION HISTOGRAM COSINE ")

    success = True
    while success:
        print("number of clusters before merging: ", len(cluster_pixels))
        cluster_elements, cluster_pixels, cluster_coarse, success = obj_rg.region_growth(cluster_pixels, cluster_coarse,
                                                                                         clusters_feat, thresh=0.7,
                                                                                         measure="timedur_hist_cosine_sim")
        print("number of clusters after merging: ", len(cluster_pixels))
        print("Preparing features for new clusters")
        clusters_feat = obj_data.merge_cluster_features(orig_clusters_features=clusters_feat,
                                                        new_cluster_info=cluster_elements)
        if len(cluster_pixels) != len(clusters_feat):
            print("number of clusters in cluster_pixel and cluster_feat are different")

        # img_sp, label_sp = plot_cluster(cluster_pixels, dims)
        # obj_dv = dv(img_sp, label_sp)
        # obj_dv.feature_comparison(clusters_feat)
        # input()

    # img_sp, label_sp = plot_cluster(cluster_pixels, dims)
    # obj_dv = dv(img_sp, label_sp)
    # obj_dv.feature_comparison(clusters_feat)

    success = True

    print(" START TIME - DURATION and PREV ACTIVITY HISTOGRAM COSINE ")

    while success:
        print("number of clusters before merging: ", len(cluster_pixels))
        cluster_elements, cluster_pixels, cluster_coarse, success = obj_rg.region_growth(cluster_pixels, cluster_coarse,
                                                                                         clusters_feat, thresh=0.8,
                                                                                         measure="durprevact_hist_cosine_sim")
        print("number of clusters after merging: ", len(cluster_pixels))
        print("Preparing features for new clusters")
        clusters_feat = obj_data.merge_cluster_features(orig_clusters_features=clusters_feat,
                                                        new_cluster_info=cluster_elements)
        if len(cluster_pixels) != len(clusters_feat):
            print("number of clusters in cluster_pixel and cluster_feat are different")

        # img_sp, label_sp = plot_cluster(cluster_pixels, dims)
        # obj_dv = dv(img_sp, label_sp)
        # obj_dv.feature_comparison(clusters_feat)

    print("preparing visual results for clusters.....")
    img_sp, label_sp = plot_cluster(cluster_pixels, dims)
    if SHOW_IMAGE:
        cv2.namedWindow("Second Pass Clusters", cv2.WINDOW_NORMAL)
        cv2.imshow("Second Pass Clusters", img_sp)
        cv2.waitKey(0)

    if SAVE_IMAGE:
        cv2.imwrite("../data/synthetic_data/clustered/" + i_name, img_sp)

    if VISUALIZE:
        obj_dv = dv(img_sp, label_sp)
        obj_dv.feature_comparison(clusters_feat)

    variance = total_variance(init_cluster_feat, cluster_coarse)

    return len(cluster_pixels), variance


if __name__ == "__main__":
    # create dataloader
    print("********************************************************************************")
    print("Hyperparameters")
    thresh = 0.9
    scale = 30

    print("********************************************************************************")
    print("prepare data")

    lvl = 3
    prob = [0.3, 0.5, 0.7, 0.9]
    sd = [5, 10, 15, 20, 25, 30]
    filenum = 5

    for p in prob:
        num_clusters = []
        variance = []
        for d in sd:
            avg_num_clusters = 0
            avg_var = 0
            for f_num in range(1, filenum+1):

                if p is None:
                    f_name = "synt_data_lvl" + str(lvl) + "_days30_sd" + str(d) + "_" + str(f_num)
                else:
                    f_name = "synt_data_lvl" + str(lvl) + "_days30_sd" + str(d) + "_prob" + str(p) + "_" + str(f_num)

                print(f_name)
                res, var = clustering(lvl, f_name)
                avg_num_clusters += res
                avg_var += var

            num_clusters.append(int(avg_num_clusters/filenum))
            variance.append(avg_var/filenum)

        print("Graph: Cluster-SD")
        print("Clusters: ", num_clusters)
        plot_graph(num_clusters, sd, "Cluster-SD", str(lvl)+"_"+str(p), y_axis="Number of CLusters")

        print("Graph: Variance-SD")
        print("Variance", variance)
        plot_graph(variance, sd, "Variance-SD", str(lvl)+"_"+str(p), y_axis="Variance")
