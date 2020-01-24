import glob
import re
import numpy as np
import cv2
import colorsys
import random
import matplotlib.pyplot as plt

from data_processing import Features, GenerateRealDataCluster
from region_growth import RegionGrowth
from seeds import SEEDS
from data_visualization import DataVisualization as dv


DEBUG = 0
SHOW_IMAGE = 0
VISUALIZE = 0
SAVE_IMAGE = 1


def dataInformation():
    info = {
        'num_loc': 9,  # R1, R2, R3, T1, T2, L, K, O, E
        'num_loc_type': 6,  # room, toilet, living, kitchen, out, entrance
        'scale': 60,   # minutes
        'loc_idx': {
            'R1': 0,
            'R2': 1,
            'R3': 2,
            'T1': 3,
            'T2': 4,
            'L': 5,
            'K': 6,
            'O': 7,
            'E': 8
            },
        'type_idx': {
            'room': 0,
            'toilet': 1,
            'living': 2,
            'kitchen': 3,
            'out': 4,
            'entry': 5
        }
    }
    return info


def HSV_to_RGB(h, s, v):
    (r, g, b) = colorsys.hsv_to_rgb(h, s, v)
    return int(255 * r), int(255 * g), int(255 * b)


def get_distinct_colors(n):
    hue_partition = 1.0 / (n + 1)
    return (HSV_to_RGB(hue_partition * value, 1.0, 1.0) for value in range(0, n))


def plot_cluster(clusters, dims):
    num_clusters = len(clusters)
    r, c, ch = dims
    image = np.zeros((r * 30, c, ch), np.uint8)
    label = np.ones((r * 30, c), np.int) * (-1)

    # generate colors for clusters
    color_gen = get_distinct_colors(num_clusters)
    colors = []
    for c in color_gen:
        colors.append(c)
    random.shuffle(colors)

    for i in clusters:
        for point in clusters[i]:
            r, c = point
            image[r * 30:(r + 1) * 30, c] = colors[i - 1]
            label[r * 30:(r + 1) * 30, c] = i

    return image, label


def timeToSec(t):
    if type(t) == str:
        t_split = t.split(sep=':')
    elif type(t) == list:
        t_split = list(map(int, t))
    sec = int(t_split[0]) * 3600 + int(t_split[1]) * 60 + int(t_split[2])
    return sec


def timeToMin(t):
    if type(t) == str:
        t_split = t.split(sep=':')
    elif type(t) == list:
        t_split = list(map(int, t))
    minute = int(t_split[0]) * 60 + int(t_split[1])
    return minute


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


def plot_graph(x, y, graph_name, i_name, y_axis):
    plt.rc('xtick', labelsize=25)
    plt.rc('ytick', labelsize=25)
    fig = plt.figure()
    plt.plot(x, y, 'k')
    plt.plot(x, y, 'k', marker='o')
    plt.ylim(ymin=0)
    plt.xlabel("standard deviation", fontsize=25)
    plt.ylabel(y_axis, fontsize=25)
    # plt.title(graph_name, fontsize=1)
    plt.savefig("../../data/synthetic_data/new_synthetic_data/" + i_name + "_" + graph_name + ".png")
    plt.show()


def plot_subgraphs(x, y, plot_labels, graph_name, i_name, y_axis):
    plt.rc('xtick', labelsize=25)
    plt.rc('ytick', labelsize=25)
    fig = plt.figure()
    marker = ['o', 'x', '*', 'p']
    line = ['-', '--', '-.', ':']
    cmap = ['r', 'g', 'b', 'k']

    # cmap = plt.cm.get_cmap("hsv", len(plot_labels))
    for p in range(len(plot_labels)):
        # print(x[p])
        # plt.plot(x, y[p], c=cmap(p), label=str(plot_labels[p]))
        # plt.plot(y, x[p], c=cmap(p), marker='o')
        plt.plot(x, y[p], linestyle=line[p], color=cmap[p], marker=marker[p], label=str(plot_labels[p]))
        # plt.plot(y, x[p], color=cmap(p), marker=marker[p])

    plt.legend(prop={'size': 20})
    plt.ylim(ymin=0)
    plt.xlabel("standard deviation", fontsize=25)
    plt.ylabel(y_axis, fontsize=25)
    # plt.title(graph_name)
    # plt.pause(1)
    plt.savefig("../../data/synthetic_data/new_synthetic_data/" + i_name + "_" + graph_name + ".png")
    plt.show()


def getRoutineInfo(filename):
    print("Reading file:", filename)
    data_info = dataInformation()
    img = []
    activity_gt = []

    obj_feat = Features(time_bins=0,
                        act_bins=data_info['num_loc'],
                        type_bins=data_info['num_loc_type'],
                        scale=data_info['scale'])
    # open routine file
    with open(filename, 'r') as f_read:
        data = f_read.readlines()
    del data[0]
    prev_day = 1
    routine = []
    acts = []
    stime = 0
    dur = 0
    prev_act = "sleeping1"

    for line in data:
        # line: Day, Time(hh:mm), space_id(char), space_type(string), activity
        sample = re.split(',|\n', line)[0:-1]
        if prev_act != sample[-1]:
            d = {
                'act': prev_act,
                'stime': stime,
                'duration': dur
            }
            acts.append(d)
            stime = stime + dur
            dur = 0
            prev_act = sample[-1]

        if prev_day != int(sample[0]):
            img.append(routine)
            # print(len(routine))
            routine = []
            prev_day = int(sample[0])
            activity_gt.append(acts)
            acts = []
            stime = 0

        minute = timeToMin(sample[1])
        dur += 1
        sample_feat = obj_feat.init_null_features()
        sample_feat["num_clusters"] = 1
        sample_feat["loc"].add(sample[2])
        sample_feat["loc_type"].add(sample[3])
        sample_feat["loc_array"][data_info['loc_idx'][sample[2]]] = 1
        sample_feat["type_array"][data_info['type_idx'][sample[3]]] = 1
        sample_feat["room_idx"] = data_info['loc_idx'][sample[2]]
        # routine[minute] = sample_feat
        routine.append(sample_feat)
    return img, activity_gt


def evaluateSingleFile(base_addr, target_addr, filename):
    data_info = dataInformation()
    data, activity_gt = getRoutineInfo(base_addr+filename)
    num_activities = data_info['num_loc']
    num_types = data_info['num_loc_type']

    seed_accuracy = []

    filtered_data = data.copy()
    # for day in data:
    #     filtered = median_filtering(day)
    #     if scale_down != scale:
    #         filtered = scale_data(filtered, int(scale_down/scale))
    #     filtered_data.append(filtered)
    num_days = 0
    scale = data_info['scale']
    cluster_feat = dict()
    cluster_pixel = dict()
    c_id = 1
    rgobj = RegionGrowth()
    realgenobj = GenerateRealDataCluster(num_act=num_activities, num_type=num_types, scale=1)
    for routine in filtered_data:
        # day is a 1 day filtered routine
        # print(len(routine))
        num_days += 1
        seedroutine = [None] * len(routine)
        routine_loc = [None] * len(routine)
        for i in range(len(routine)):
            seedroutine[i] = routine[i]["room_idx"]
            routine_loc[i] = list(routine[i]["loc"])[0]

        seedsobj = SEEDS()
        seedsobj.initialize(width=len(routine), scale=scale, num_locs=num_activities)
        seedsobj.assign_labels()
        seedsobj.compute_histograms(seedroutine)
        seedsobj.iterate()

        seedlabels = seedsobj.labels[-1]
        # realgenobj = GenerateRealDataCluster(num_act=num_activities, scale=5)
        cf, cp = realgenobj.make_single_day_clusters(routine, seedlabels, num_days)

        if DEBUG:
            seeds_loc = dict()
            j = 0
            for i in range(len(seedlabels)):
                if i == len(seedlabels) - 1 or seedlabels[i] != seedlabels[i + 1]:
                    seeds_loc[seedlabels[i]] = routine_loc[j:i + 1]
                    print(seedlabels[i], routine_loc[j:i + 1])
                    # print(seeds_loc[seedlabels[i]])
                    j = i + 1

            for c in cf:
                print(c, cf[c]["loc_array"])

        # merge adjacent single day similar activities
        cluster_new_old = rgobj.cluster_sameday_activity(cf)
        cf, cp = realgenobj.merge_sameday_cluster_features(cluster_new_old, cf, cp)

        if DEBUG:

            for c in cf:
                print(c, cp[c])
                print(c, cf[c]["loc_array"])

            print(num_days, len(cf))
            # input()

        for c in cf:
            cluster_feat[c_id] = cf[c]
            cluster_pixel[c_id] = cp[c]
            c_id += 1

        # print("Evaluating SEEDS")
        ious = []
        for act in activity_gt[num_days-1]:
            stime1 = act['stime']
            etime1 = stime1 + act['duration']
            maxiou = 0
            for c in cf:
                stime2 = cf[c]['stime']
                etime2 = stime2 + cf[c]['duration']

                # print(stime1, etime1, stime2, etime2)
                intersect = max(min(etime1, etime2) - max(stime1, stime2), 0)
                union = max(etime1, etime2) - min(stime1, stime2)
                iou = intersect/union
                if iou > maxiou:
                    maxiou = iou
            ious.append(maxiou)
            seed_accuracy.append(maxiou)
        #print("Intersection/Union", ious)
        # input()

    print("num days: ", num_days)
    print("Initial number of clusters: ", len(cluster_feat))
    init_cluster_feat = cluster_feat.copy()
    # make cluster course dict
    cluster_coarse = dict()
    for c in cluster_feat:
        cluster_coarse[c] = [c]

    num_intervals = int((24 * 60 * 60) / scale)
    dims = (num_days, num_intervals, 3)
    if SHOW_IMAGE:
        # visual results
        print("preparing SEEDS visual results.....")
        img_fp, label = plot_cluster(cluster_pixel, dims)
        if VISUALIZE:
            obj_dv = dv(img_fp, label)
            obj_dv.feature_comparison(cluster_feat, init_cluster_feat, cluster_coarse)
        else:
            cv2.namedWindow("First Pass Clusters", cv2.WINDOW_NORMAL)
            cv2.imshow("First Pass Clusters", img_fp)
            cv2.waitKey(0)

    success = True
    while success:
        print("number of clusters before merging: ", len(cluster_pixel))
        cluster_new_old, cluster_pixel, cluster_coarse, success = rgobj.region_growth(cluster_pixel,
                                                                                      cluster_coarse,
                                                                                      cluster_feat,
                                                                                      thresh=0.7,
                                                                                      measure="timedur_hist_cosine_sim")
        print("number of clusters after merging: ", len(cluster_pixel))
        print("Preparing features for new clusters")
        cluster_feat = realgenobj.merge_cluster_features(orig_clusters_features=cluster_feat,
                                                         cluster_new_old=cluster_new_old)
        if len(cluster_pixel) != len(cluster_feat):
            print("number of clusters in cluster_pixel and cluster_feat are different")

    success = True
    print(" START TIME - DURATION and PREV ACTIVITY HISTOGRAM COSINE ")
    while success:
        print("number of clusters before merging: ", len(cluster_pixel))
        cluster_new_old, cluster_pixel, cluster_coarse, success = rgobj.region_growth(cluster_pixel,
                                                                                      cluster_coarse,
                                                                                      cluster_feat,
                                                                                      thresh=0.9,
                                                                                      measure="durprevact_hist_cosine_sim")
        print("number of clusters after merging: ", len(cluster_pixel))
        print("Preparing features for new clusters")
        cluster_feat = realgenobj.merge_cluster_features(orig_clusters_features=cluster_feat,
                                                         cluster_new_old=cluster_new_old)
        if len(cluster_pixel) != len(cluster_feat):
            print("number of clusters in cluster_pixel and cluster_feat are different")

    print("preparing visual results for clusters.....")
    img_sp, label = plot_cluster(cluster_pixel, dims)
    if SHOW_IMAGE:
        if VISUALIZE:
            obj_dv = dv(img_sp, label)
            obj_dv.feature_comparison(cluster_feat, init_cluster_feat, cluster_coarse)
        else:
            cv2.namedWindow("First Pass Clusters", cv2.WINDOW_NORMAL)
            cv2.imshow("First Pass Clusters", img_sp)
            cv2.waitKey(0)

    if SAVE_IMAGE:
        img_addr = target_addr + filename[:-3] + "png"
        cv2.imwrite(img_addr, img_sp)

    rmse = total_RMSE(init_cluster_feat, cluster_coarse)
    mae = total_MAE(init_cluster_feat, cluster_coarse)

    return seed_accuracy, len(cluster_pixel), rmse, mae


def main():
    level = 3
    sdp = [5, 10, 15, 20, 25, 30]
    num_noise = [20, 30, 40]
    probs = [0.3, 0.5, 0.7, 0.9]

    if level == 1:
        base_addr = "../../data/synthetic_data/new_synthetic_data/level1/"
        target_addr = "../../data/synthetic_data/new_synthetic_data/images/level1/"
        seeds_iou_sd = []
        clusters_sd = []
        mae_sd = []
        rmse_sd = []
        for sd in sdp:
            seed_iou = []
            avg_clusters = 0
            avg_mae = 0
            avg_rmse = 0
            for n in range(10):
                filename = "newsynt_level1_sd" + str(sd) + "_" + str(n+1) + ".csv"
                iou, num_clusters, rmse, mae = evaluateSingleFile(base_addr, target_addr, filename)
                seed_iou = seed_iou + iou
                avg_clusters += num_clusters
                avg_rmse += rmse
                avg_mae += mae
                # print(sum(iou)/len(iou), num_clusters, rmse, mae)
            avg_iou = sum(seed_iou)/len(seed_iou)
            clusters_sd.append(avg_clusters/5)
            rmse_sd.append(avg_rmse/5)
            mae_sd.append(avg_mae/5)
            seeds_iou_sd.append(avg_iou)
        print(seeds_iou_sd)
        print(clusters_sd)
        plot_graph(sdp, clusters_sd, "num_clusters", "level1", "Num. of Clusters")
        print(rmse_sd)
        plot_graph(sdp, rmse_sd, "rmse", "level1", "RMSE")
        print(mae_sd)
        plot_graph(sdp, mae_sd, "mae", "level1", "MAE")

    if level == 2:
        base_addr = "../../data/synthetic_data/new_synthetic_data/level2/"
        target_addr = "../../data/synthetic_data/new_synthetic_data/images/level2/"
        seed_iou_noise = []
        cluster_noise = []
        rmse_noise = []
        mae_noise = []
        for noise in num_noise:
            seeds_iou_sd = []
            clusters_sd = []
            rmse_sd = []
            mae_sd = []
            for sd in sdp:
                seed_iou = []
                avg_clusters = 0
                avg_mae = 0
                avg_rmse = 0
                for n in range(5):
                    filename = "newsynt_level2_sd" + str(sd) + "_noise" + str(noise) + "_" + str(n + 1) + ".csv"
                    iou, num_clusters, rmse, mae = evaluateSingleFile(base_addr, target_addr, filename)
                    seed_iou = seed_iou + iou
                    avg_clusters += num_clusters
                    avg_rmse += rmse
                    avg_mae += mae
                    print(sum(iou) / len(iou))
                avg_iou = sum(seed_iou) / len(seed_iou)
                clusters_sd.append(avg_clusters / 5)
                rmse_sd.append(avg_rmse / 5)
                mae_sd.append(avg_mae / 5)
                seeds_iou_sd.append(avg_iou)
            print(seeds_iou_sd)
            seed_iou_noise.append(seeds_iou_sd)
            cluster_noise.append(clusters_sd)
            rmse_noise.append(rmse_sd)
            mae_noise.append(mae_sd)
        plot_subgraphs(sdp, cluster_noise, num_noise, "num_cluster", "level2", "Num. of Clusters")
        plot_subgraphs(sdp, rmse_noise, num_noise, "rmse", "level2", "RMSE")
        plot_subgraphs(sdp, mae_noise, num_noise, "mae", "level2", "MAE")


        for noise, ious in zip(num_noise, seed_iou_noise):
            print(noise)
            print(ious)

    if level == 3:
        base_addr = "../../data/synthetic_data/new_synthetic_data/level3/"
        target_addr = "../../data/synthetic_data/new_synthetic_data/images/level3/"
        seed_iou_prob = []
        cluster_prob = []
        rmse_prob = []
        mae_prob = []
        for prob in probs:
            seeds_iou_sd = []
            clusters_sd = []
            rmse_sd = []
            mae_sd = []
            for sd in sdp:
                seed_iou = []
                avg_clusters = 0
                avg_mae = 0
                avg_rmse = 0
                for n in range(5):
                    filename = "newsynt_level3_sd" + str(sd) + "_prob" + str(prob) + "_" + str(n + 1) + ".csv"
                    iou, num_clusters, rmse, mae = evaluateSingleFile(base_addr, target_addr, filename)
                    seed_iou = seed_iou + iou
                    avg_clusters += num_clusters
                    avg_rmse += rmse
                    avg_mae += mae
                    # print(sum(iou) / len(iou))
                avg_iou = sum(seed_iou) / len(seed_iou)
                seeds_iou_sd.append(avg_iou)
                clusters_sd.append(avg_clusters / 5)
                rmse_sd.append(avg_rmse / 5)
                mae_sd.append(avg_mae / 5)
            print(seeds_iou_sd)
            seed_iou_prob.append(seeds_iou_sd)
            cluster_prob.append(clusters_sd)
            rmse_prob.append(rmse_sd)
            mae_prob.append(mae_sd)

        plot_subgraphs(sdp, cluster_prob, probs, "num_cluster", "level3", "Num. of Clusters")
        plot_subgraphs(sdp, rmse_prob, probs, "rmse", "level3", "RMSE")
        plot_subgraphs(sdp, mae_prob, probs, "mae", "level3", "MAE")

        for prob, ious in zip(probs, seed_iou_prob):
            print(prob)
            print(ious)


if __name__ == '__main__':
    main()