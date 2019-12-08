import glob
import re
import numpy as np
import cv2
import colorsys
import random

from data_processing import Features, GenerateRealDataCluster
from region_growth import RegionGrowth
from seeds import SEEDS
from data_visualization import DataVisualization as dv


DEBUG = 0
SHOW_IMAGE = 0
VISUALIZE = 0


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
    image = np.zeros((r * 50, c, ch), np.uint8)
    label = np.ones((r * 50, c), np.int) * (-1)

    # generate colors for clusters
    color_gen = get_distinct_colors(num_clusters)
    colors = []
    for c in color_gen:
        colors.append(c)
    random.shuffle(colors)

    for i in clusters:
        for point in clusters[i]:
            r, c = point
            image[r * 50:(r + 1) * 50, c] = colors[i - 1]
            label[r * 50:(r + 1) * 50, c] = i

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


def evaluateSingleFile(filename):
    data_info = dataInformation()
    data, activity_gt = getRoutineInfo(filename)
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

    if SHOW_IMAGE:
        # visual results
        num_intervals = int((24 * 60 * 60) / scale)
        dims = (num_days, num_intervals, 3)
        print("preparing visual results for clusters.....")
        img_fp, label = plot_cluster(cluster_pixel, dims)
        if VISUALIZE:
            obj_dv = dv(img_fp, label)
            obj_dv.feature_comparison(cluster_feat, init_cluster_feat, cluster_coarse)
        else:
            cv2.namedWindow("First Pass Clusters", cv2.WINDOW_NORMAL)
            cv2.imshow("First Pass Clusters", img_fp)
            cv2.waitKey(0)

    return seed_accuracy


def main():
    level = 3
    sdp = [5, 10, 15, 20, 25, 30]
    num_noise = [20]
    all_probs = [0.3, 0.5, 0.7, 0.9]
    probs = [0.9]

    if level == 1:
        base_addr = "../../data/synthetic_data/new_synthetic_data/level1/"
        seeds_iou_sd = []
        for sd in sdp:
            seed_iou = []
            for n in range(5):
                filename = "newsynt_level1_sd" + str(sd) + "_" + str(n+1) + ".csv"
                iou = evaluateSingleFile(base_addr+filename)
                seed_iou = seed_iou + iou
                print(sum(iou)/len(iou))
            avg_iou = sum(seed_iou)/len(seed_iou)
            seeds_iou_sd.append(avg_iou)
        print(seeds_iou_sd)

    if level == 2:
        base_addr = "../../data/synthetic_data/new_synthetic_data/level2/"
        seed_iou_noise = []
        for noise in num_noise:
            seeds_iou_sd = []
            for sd in sdp:
                seed_iou = []
                for n in range(5):
                    filename = "newsynt_level2_sd" + str(sd) + "_noise" + str(noise) + "_" + str(n + 1) + ".csv"
                    iou = evaluateSingleFile(base_addr+filename)
                    seed_iou = seed_iou + iou
                    print(sum(iou) / len(iou))
                avg_iou = sum(seed_iou) / len(seed_iou)
                seeds_iou_sd.append(avg_iou)
            print(seeds_iou_sd)
            seed_iou_noise.append(seeds_iou_sd)

        for noise, ious in zip(num_noise, seed_iou_noise):
            print(noise)
            print(ious)

    if level == 3:
        base_addr = "../../data/synthetic_data/new_synthetic_data/level3/"
        seed_iou_prob = []
        for prob in probs:
            seeds_iou_sd = []
            for sd in sdp:
                seed_iou = []
                for n in range(5):
                    filename = "newsynt_level3_sd" + str(sd) + "_prob" + str(prob) + "_" + str(n + 1) + ".csv"
                    iou = evaluateSingleFile(base_addr+filename)
                    seed_iou = seed_iou + iou
                    print(sum(iou) / len(iou))
                avg_iou = sum(seed_iou) / len(seed_iou)
                seeds_iou_sd.append(avg_iou)
            print(seeds_iou_sd)
            seed_iou_prob.append(seeds_iou_sd)

        for prob, ious in zip(probs, seed_iou_noise):
            print(prob)
            print(ious)


if __name__ == '__main__':
    main()