import simplejson
import matplotlib.pyplot as plt
from random import shuffle
import colorsys
import numpy as np
import cv2

from data_processing import GenerateSyntheticCluster, ReadData, GenerateRealDataCluster
from region_growth import RegionGrowth
from data_visualization import DataVisualization as dv
from seeds import SEEDS

# from data_processing import TempDataProcessing, ClusterProcessing, DataPreparation
# from region_growth import RegionGrowth

SAVE_IMAGE = 1
SHOW_IMAGE = 1
VISUALIZE = 1
DEBUG = 0


def median_filtering(list1d, window_size=12):
    container1 = dict()
    container2 = dict()
    filtered = list()
    for i in range(len(list1d) + int(window_size / 2)):
        if i < len(list1d):
            if list(list1d[i]["loc"])[0] not in container1:
                container1[list(list1d[i]["loc"])[0]] = 1
                container2[list(list1d[i]["loc"])[0]] = list1d[i]
            else:
                container1[list(list1d[i]["loc"])[0]] += 1
                container2[list(list1d[i]["loc"])[0]] = list1d[i]

        if i >= window_size:
            container1[list(list1d[i - window_size]["loc"])[0]] -= 1

        if i >= int(window_size / 2):
            max_act = max(container1, key=container1.get)
            filtered.append(container2[max_act])

    if len(filtered) != len(list1d):
        print("median filtering: lengths different")
        raise

    return filtered


def scale_data(list1d, scale=6):
    container1 = dict()
    container2 = dict()
    scaled = list()
    for i in range(len(list1d)):
        if list(list1d[i]["loc"])[0] not in container1:
            container1[list(list1d[i]["loc"])[0]] = 1
            container2[list(list1d[i]["loc"])[0]] = list1d[i]
        else:
            container1[list(list1d[i]["loc"])[0]] += 1
            container2[list(list1d[i]["loc"])[0]] = list1d[i]

        if (i + 1) % scale == 0:
            max_act = max(container1, key=container1.get)
            scaled.append(container2[max_act].copy())
            container1 = dict()
            container2 = dict()

    if len(scaled) != int(len(list1d) / scale):
        print("scale_data: lengths different")
        raise

    return scaled


def total_RMSE(cluster_feat, cluster_elements):
    variance = 0.0
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

            reg_var = np.var(stime) + np.var(dur)
            # reg_var = variance_norm(stime) + variance_norm(dur)
            # print(reg_var, len(cluster_elements[region]))
            variance += np.sqrt(reg_var)
            num_contributors += 1
        # print(variance - prev_var, len(cluster_elements[region]))
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


def synthetic_dataset():
    # create dataloader
    print("********************************************************************************")
    print("Synthetic Dataset")
    thresh = 0.9
    scale = 30

    print("********************************************************************************")
    print("prepare data")

    level = 2
    prob = None
    noise_array = [5, 10, 15, 20]
    noise = 5
    sd = 5
    filenum = 1

    if level == 1:
        filename = "synt_data_lvl" + str(level) + "_days30_sd" + str(sd) + "_" + str(filenum) + ".csv"
        imgname = "synt_data_lvl" + str(level) + "_days30_sd" + str(sd) + "_" + str(filenum) + ".jpg"
    elif level == 2:
        filename = "synt_data_lvl" + str(level) + "_days30_sd" + str(sd) + "_noise" + str(noise) + "_" + str(filenum) + ".csv"
        imgname = "synt_data_lvl" + str(level) + "_days30_sd" + str(sd) + "_noise" + str(noise) + "_" + str(filenum) + ".jpg"
    else:
        filename = "synt_data_lvl" + str(level) + "_days30_sd" + str(sd) + "_prob" + str(prob) + "_" + str(
            filenum) + ".csv"
        imgname = "synt_data_lvl" + str(level) + "_days30_sd" + str(sd) + "_prob" + str(prob) + "_" + str(
            filenum) + "jpg"

    filepath = "../data/synthetic_data/level" + str(level) + "/parsed_data/" + filename
    # filename = "uci_adl_B_orig"
    # filepath = "../data/synthetic_data/uci_adl/" + filename + ".csv"

    obj_data = GenerateSyntheticCluster(
        routine_type="ADL1",  # ADL1, UCI
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
        cluster_elements, cluster_pixels, cluster_coarse, success = obj_rg.synt_region_growth(cluster_pixels,
                                                                                              cluster_coarse,
                                                                                              clusters_feat, thresh=0.7,
                                                                                              measure="timedur_hist_cosine_sim")
        print("number of clusters after merging: ", len(cluster_pixels))
        print("Preparing features for new clusters")
        clusters_feat = obj_data.merge_cluster_features(orig_clusters_features=clusters_feat,
                                                        new_cluster_info=cluster_elements)
        if len(cluster_pixels) != len(clusters_feat):
            print("number of clusters in cluster_pixel and cluster_feat are different")

    success = True
    print(" START TIME - DURATION and PREV ACTIVITY HISTOGRAM COSINE ")
    while success:
        print("number of clusters before merging: ", len(cluster_pixels))
        cluster_elements, cluster_pixels, cluster_coarse, success = obj_rg.synt_region_growth(cluster_pixels,
                                                                                              cluster_coarse,
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
        cv2.imwrite("../data/synthetic_data/clustered/" + filename + ".jpg", img_sp)

    if VISUALIZE:
        obj_dv = dv(img_sp, label_sp)
        obj_dv.feature_comparison(clusters_feat, init_cluster_feat, cluster_coarse)

    rmse = total_RMSE(init_cluster_feat, cluster_coarse)
    mae = total_MAE(init_cluster_feat, cluster_coarse)
    print("number of clusters: ", len(cluster_pixels))
    print("RMSE: ", rmse)
    print("MAE: ", mae)


def real_dataset():
    # create dalaloader
    print("********************************************************************************")
    print("Real Data")
    thresh = 0.9
    scale = 60
    scale_down = 60

    print("********************************************************************************")
    print("prepare data")
    # obj_data = TempDataProcessing("../data/180724_180810_mod.csv")
    # obj_data = TempDataProcessing("../data/toy_example.csv")
    obj_data = ReadData(subject_id=2,
                        num_days=15,
                        dir_name="../data/real_data/Subject_2/")
    # file_name="xandem_2018-12-02.log")
    # print("number of days:", len(obj_data.image))
    # print("number of intervals: ", len(obj_data.image[0]))
    # data_dims = (len(obj_data.image), len(obj_data.image[0]), 3)
    # with open("../data/image.log", 'w') as image_log:
    #     simplejson.dump(obj_data.img, image_log)
    data = obj_data.image.copy()
    num_activities = obj_data.get_num_spaces()
    num_types = obj_data.get_num_space_types()
    del obj_data

    filtered_data = data.copy()
    # for day in data:
    #     filtered = median_filtering(day)
    #     if scale_down != scale:
    #         filtered = scale_data(filtered, int(scale_down/scale))
    #     filtered_data.append(filtered)

    # process individual days
    if DEBUG:
        print("Processing single day data")
    num_days = 0
    cluster_feat = dict()
    cluster_pixel = dict()
    c_id = 1
    rgobj = RegionGrowth()
    realgenobj = GenerateRealDataCluster(num_act=num_activities, num_type=num_types, scale=scale)
    for routine in filtered_data:
        # day is a 1 day filtered routine
        print(len(routine))
        num_days += 1
        seedroutine = [None] * len(routine)
        routine_loc = [None] * len(routine)
        for i in range(len(routine)):
            seedroutine[i] = routine[i]["room_idx"]
            routine_loc[i] = list(routine[i]["loc"])[0]

        seedsobj = SEEDS()
        seedsobj.initialize(width=len(routine), scale=scale_down, num_locs=num_activities)
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
            input()

        for c in cf:
            cluster_feat[c_id] = cf[c]
            cluster_pixel[c_id] = cp[c]
            c_id += 1

    # visual results
    num_intervals = int((24 * 60 * 60) / scale_down)
    dims = (num_days, num_intervals, 3)
    print("preparing visual results for clusters.....")
    img_fp, label = plot_cluster(cluster_pixel, dims)

    print("num days: ", num_days)
    print("Initial number of clusters: ", len(cluster_feat))
    init_cluster_feat = cluster_feat.copy()
    # make cluster course dict
    cluster_coarse = dict()
    for c in cluster_feat:
        cluster_coarse[c] = [c]

    if SHOW_IMAGE:
        if VISUALIZE:
            obj_dv = dv(img_fp, label)
            obj_dv.feature_comparison(cluster_feat, init_cluster_feat, cluster_coarse)
        else:
            cv2.namedWindow("First Pass Clusters", cv2.WINDOW_NORMAL)
            cv2.imshow("First Pass Clusters", img_fp)
            cv2.waitKey(0)

    print("********************************************************************************")
    print("performing Hierarchical merging.....")

    print(" TIME-DURATION HISTOGRAM COSINE ")

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

        # img_sp, label_sp = plot_cluster(cluster_pixels, dims)
        # obj_dv = dv(img_sp, label_sp)
        # obj_dv.feature_comparison(clusters_feat)

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


def real_data_complete():
    scale = 60
    scale_down = 60

    for subject_id in range(1, 5):
        target_dir = "../data/real_data/clustered_images/"
        source_dir = "../data/real_data/Subject_" + str(subject_id)
        obj_data = ReadData(subject_id=subject_id,
                            num_days=-1,
                            dir_name=source_dir)
        data = obj_data.image.copy()
        num_activities = obj_data.get_num_spaces()
        num_types = obj_data.get_num_space_types()
        del obj_data

        filtered_data = data.copy()
        # for day in data:
        #     filtered = median_filtering(day)
        #     if scale_down != scale:
        #         filtered = scale_data(filtered, int(scale_down / scale))
        #     filtered_data.append(filtered)

        # data for every 14 days
        start_day = 0
        end_day = 14
        while end_day <= len(filtered_data):
            print("Day:", start_day + 1, "to Day:", end_day)
            routine14days = filtered_data[start_day:end_day]
            num_days = 0
            cluster_feat = dict()
            cluster_pixel = dict()
            c_id = 1
            rgobj = RegionGrowth()
            realgenobj = GenerateRealDataCluster(num_act=num_activities, num_type=num_types, scale=scale)

            # single day processing
            for routine in routine14days:
                # day is a 1 day filtered routine
                num_days += 1
                seedroutine = [None] * len(routine)
                routine_loc = [None] * len(routine)
                for i in range(len(routine)):
                    seedroutine[i] = routine[i]["room_idx"]
                    routine_loc[i] = list(routine[i]["loc"])[0]

                seedsobj = SEEDS()
                seedsobj.initialize(width=len(routine), scale=scale_down, num_locs=num_activities)
                seedsobj.assign_labels()
                seedsobj.compute_histograms(seedroutine)
                seedsobj.iterate()

                seedlabels = seedsobj.labels[-1].copy()
                del seedsobj
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

                for c in cf:
                    cluster_feat[c_id] = cf[c]
                    cluster_pixel[c_id] = cp[c]
                    c_id += 1

            # visual results
            num_intervals = int((24 * 60 * 60) / scale_down)
            dims = (num_days, num_intervals, 3)

            print("num days: ", num_days)
            print("Initial number of clusters: ", len(cluster_feat))
            init_cluster_feat = cluster_feat.copy()
            # make cluster course dict
            cluster_coarse = dict()
            for c in cluster_feat:
                cluster_coarse[c] = [c]

            # if SHOW_IMAGE:
            #     if VISUALIZE:
            #         obj_dv = dv(img_fp, label)
            #         obj_dv.feature_comparison(cluster_feat, init_cluster_feat, cluster_coarse)
            #     else:
            #         cv2.namedWindow("First Pass Clusters", cv2.WINDOW_NORMAL)
            #         cv2.imshow("First Pass Clusters", img_fp)
            #         cv2.waitKey(0)

            print("********************************************************************************")
            print("performing Hierarchical merging.....")
            print(" TIME-DURATION HISTOGRAM COSINE ")

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

            # print("preparing visual results for clusters.....")
            # img_sp, label = plot_cluster(cluster_pixel, dims)
            # if SHOW_IMAGE:
            #     if VISUALIZE:
            #         obj_dv = dv(img_sp, label)
            #         obj_dv.feature_comparison(cluster_feat, init_cluster_feat, cluster_coarse)
            #     else:
            #         cv2.namedWindow("First Pass Clusters", cv2.WINDOW_NORMAL)
            #         cv2.imshow("First Pass Clusters", img_sp)
            #         cv2.waitKey(0)

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

                # img_sp, label_sp = plot_cluster(cluster_pixels, dims)
                # obj_dv = dv(img_sp, label_sp)
                # obj_dv.feature_comparison(clusters_feat)

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
                filename = "Subject" + str(subject_id) + "_day" + str(start_day + 1) + "-" + str(end_day) + ".png"
                cv2.imwrite(target_dir + filename, img_sp)

            start_day += 7
            end_day += 7


def test_real_data():
    # create dalaloader
    print("********************************************************************************")
    print("Real Data")
    thresh = 0.9
    scale = 5
    scale_down = 30

    print("********************************************************************************")
    print("prepare data")
    # obj_data = TempDataProcessing("../data/180724_180810_mod.csv")
    # obj_data = TempDataProcessing("../data/toy_example.csv")
    obj_data = ReadData(subject_id=1,
                        num_days=15,
                        dir_name="../data/real_data/Subject_2/processed_data")
    # file_name="xandem_2018-12-02.log")
    # print("number of days:", len(obj_data.image))
    # print("number of intervals: ", len(obj_data.image[0]))
    # data_dims = (len(obj_data.image), len(obj_data.image[0]), 3)
    # with open("../data/image.log", 'w') as image_log:
    #     simplejson.dump(obj_data.img, image_log)
    data = obj_data.image.copy()
    num_activities = obj_data.get_num_spaces()
    del obj_data

    f1 = median_filtering(data[0])
    f2 = scale_data(f1, int(scale_down / scale))

    l1 = []
    for i in range(len(f1)):
        l1.append(list(f1[i]["loc"])[0])

    l2 = []
    for i in range(len(f2)):
        l2.append(list(f2[i]["loc"])[0])

    print(l1)
    print(l2)

    input()

    filtered_data = []
    for d in data:
        filtered = median_filtering(d)
        if scale_down != scale:
            filtered = scale_data(filtered, int(scale_down / scale))
        filtered_data.append(filtered)

    day = 14

    routine = [None] * len(filtered_data[day])
    routine_loc = [None] * len(filtered_data[day])
    for i in range(len(filtered_data[day])):
        routine[i] = filtered_data[day][i]["room_idx"]
        routine_loc[i] = list(filtered_data[day][i]["loc"])[0]

    print(routine_loc)

    seedsobj = SEEDS()
    seedsobj.initialize(width=17280, scale=5, num_locs=num_activities)
    seedsobj.assign_labels()
    seedsobj.compute_histograms(routine)
    seedsobj.iterate()

    labels = seedsobj.labels[-1]

    seeds_loc = dict()
    j = 0
    for i in range(len(labels)):
        if i == len(labels) - 1 or labels[i] != labels[i + 1]:
            seeds_loc[labels[i]] = routine_loc[j:i + 1]
            print(labels[i], routine_loc[j:i + 1])
            j = i + 1

    realgenobj = GenerateRealDataCluster(num_act=num_activities, scale=5)
    cf, cp = realgenobj.make_single_day_clusters(filtered_data[day], labels, day=1)

    print("seeds cluster data")
    for c in cf:
        print(c, cf[c]["loc_array"])

    # merge adjacent single day similar activities
    rgobj = RegionGrowth()
    cluster_new_old = rgobj.cluster_sameday_activity(cf)
    cf, cp = realgenobj.merge_cluster_features(cluster_new_old, cf, cp)

    print("merged cluster data")
    for c in cf:
        print(c, cluster_new_old[c])
        print(c, cf[c]["loc_array"])


if __name__ == "__main__":
    synthetic_dataset()
    # real_dataset()
    # real_data_complete()
    # test_real_data()

    exit(1)
