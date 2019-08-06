from random import shuffle
import colorsys
import numpy as np
import cv2
import matplotlib.pyplot as plt

from data_processing import GenerateSyntheticCluster
from region_growth import RegionGrowth
from data_visualization import DataVisualization as dv

SAVE_IMAGE = 0
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


def plot_subgraphs(x, y, plot_labels, graph_name, i_name, y_axis):
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    fig = plt.figure()
    marker = ['o', 'x', '*', 'p']
    line = ['-', '--', '-.', ':']
    cmap = plt.cm.get_cmap("hsv", len(plot_labels))
    for p in range(len(plot_labels)):
        # print(x[p])
        # plt.plot(y, x[p], c=cmap(p), label=str(plot_labels[p]))
        # plt.plot(y, x[p], c=cmap(p), marker='o')
        color = np.random.rand(3,)
        plt.plot(y, x[p], linestyle=line[p], color='k', marker=marker[p], label=str(plot_labels[p]))
        # plt.plot(y, x[p], color='k', marker=marker[p])

    plt.legend(prop={'size': 20})
    plt.ylim(ymin=0)
    plt.xlabel("standard deviation", fontsize=20)
    plt.ylabel(y_axis, fontsize=20)
    # plt.title(graph_name)
    # plt.pause(1)
    plt.savefig("../data/synthetic_data/graphs/" + i_name + "_" + graph_name + ".png")
    plt.show()


def plot_graph(x, y, graph_name, i_name, y_axis):
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    fig = plt.figure()
    plt.plot(y, x, 'b')
    plt.plot(y, x, 'b', marker='o')
    plt.ylim(ymin=0)
    plt.xlabel("standard deviation", fontsize=20)
    plt.ylabel(y_axis, fontsize=20)
    # plt.title(graph_name, fontsize=1)
    plt.savefig("../data/synthetic_data/graphs/" + i_name + "_" + graph_name + ".png")
    plt.show()


def variance_norm(arr):
    l_arr = np.max(arr)
    if l_arr == 0:
        l_arr = 1
    norm_arr = arr/l_arr
    # print(norm_arr)
    var_arr = np.var(norm_arr)
    return var_arr


def total_RMSE(cluster_feat, cluster_elements):
    variance = 0.0
    num_contributors = 0
    prev_var = 0
    for region in cluster_elements:
        if len(cluster_elements[region]) > 1:
            stime = []
            dur = []
            for c_id in cluster_elements[region]:
                stime.append(cluster_feat[c_id]["stime"])
                dur.append(cluster_feat[c_id]["duration"])
            stime = np.array(stime)
            dur = np.array(dur)

            reg_var = np.var(stime) + np.var(dur)
            # reg_var = variance_norm(stime) + variance_norm(dur)
            # print(reg_var, len(cluster_elements[region]))
            variance += np.sqrt(reg_var)
            num_contributors += 1
        # print(variance - prev_var, len(cluster_elements[region]))
        prev_var = variance
    # variance = variance/num_contributors
    return variance

def total_MAE(cluster_feat, cluster_elements):
    total_mae = 0.0
    num_contributors = 0
    for region in cluster_elements:
        if len(cluster_elements[region]) > 1:
            stime = []
            dur = []
            for c_id in cluster_elements[region]:
                stime.append(cluster_feat[c_id]["stime"])
                dur.append(cluster_feat[c_id]["duration"])
            stime = np.array(stime)
            dur = np.array(dur)

            stime_mean = np.mean(stime)
            dur_mean = np.mean(dur)

            stime_mae = np.sum(np.absolute(stime - stime_mean))
            dur_mae = np.sum(np.absolute(dur - dur_mean))

            reg_mae = stime_mae + dur_mae
            # reg_var = variance_norm(stime) + variance_norm(dur)
            # print(reg_var, len(cluster_elements[region]))
            total_mae += np.sqrt(reg_mae)
            num_contributors += 1
    # variance = variance/num_contributors
    return total_mae


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

    variance = total_RMSE(init_cluster_feat, cluster_coarse)
    mae = total_MAE(init_cluster_feat, cluster_coarse)

    return len(cluster_pixels), variance, mae


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
    sd = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    filenum = 5

    cluster_plot = []
    rmse_plot = []
    mae_plot = []
    for p in prob:
        num_clusters = []
        rmse_sd = []
        mae_sd = []
        for d in sd:
            avg_num_clusters = 0
            avg_var = 0
            avg_mae = 0
            for f_num in range(1, filenum+1):

                if p is None:
                    f_name = "synt_data_lvl" + str(lvl) + "_days30_sd" + str(d) + "_" + str(f_num)
                else:
                    f_name = "synt_data_lvl" + str(lvl) + "_days30_sd" + str(d) + "_prob" + str(p) + "_" + str(f_num)

                print(f_name)
                res, var, mae = clustering(lvl, f_name)
                avg_num_clusters += res
                avg_var += var
                avg_mae += mae

            num_clusters.append(int(avg_num_clusters/filenum))
            rmse_sd.append(avg_var/(filenum*1000))
            mae_sd.append(avg_mae/(filenum*1000))
        cluster_plot.append(num_clusters)
        rmse_plot.append(rmse_sd)
        mae_plot.append(mae_sd)

    if lvl != 3:
        print("Graph: Cluster-SD")
        print("Clusters: ", cluster_plot[0])
        plot_graph(cluster_plot[0], sd, "Level "+str(lvl)+":  Cluster-SD", str(lvl), y_axis="Number of CLusters")

        print("Graph: RMSE-SD")
        print("RMSE", rmse_plot[0])
        plot_graph(rmse_plot[0], sd, "Level "+str(lvl)+":  RMSE-SD", str(lvl), y_axis="RMSE (scale=1000)")

        print("Graph: MAE-SD")
        print("MAE", mae_plot[0])
        plot_graph(mae_plot[0], sd, "Level "+str(lvl)+":  MAE-SD", str(lvl), y_axis="MAE (scale=1000)")
    else:
        print("Graph: Cluster-SD")
        print("Clusters: ", cluster_plot)
        plot_subgraphs(cluster_plot, sd, prob, "Level "+str(lvl)+":  Cluster-SD", str(lvl), y_axis="Number of CLusters")

        print("Graph: RMSE-SD")
        print("RMSE", rmse_plot)
        plot_subgraphs(rmse_plot, sd, prob, "Level "+str(lvl)+": RMSE-SD", str(lvl), y_axis="RMSE (scale=1000)")

        print("Graph: MAE-SD")
        print("MAE", mae_plot)
        plot_subgraphs(mae_plot, sd, prob, "Level "+str(lvl)+": MAE-SD", str(lvl), y_axis="MAE (scale=1000)")
