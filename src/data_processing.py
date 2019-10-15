import numpy as np
import xml.etree.ElementTree as ET
import glob
import re
import math
from space_mapper import SpaceMapper


def time_to_sec(t):
    if type(t) == str:
        t_split = t.split(sep=':')
    elif type(t) == list:
        t_split = list(map(int, t))
    sec = int(t_split[0]) * 3600 + int(t_split[1]) * 60 + int(t_split[2])
    return sec


class SyntheticDataXML:
    def __init__(self, routine_type="ADL1"):
        self.routine_type = routine_type
        self.tree = ET.parse('../data/synthetic_data/synthetic_routine.xml')
        self.root = self.tree.getroot()

        print("[SyntheticDataXML] init: Get activity information for routine ", routine_type)
        self.space_id = {}
        self.id_space = {}

        print("[SyntheticDataXML] init: Start parsing XML")
        self.parse_xml()
        print("[SyntheticDataXML] init: parsing complete")

    def parse_xml(self):
        routine_info = None
        for child in self.root:
            if child.get("type") == self.routine_type:
                routine_info = child
                break

        if routine_info is None:
            print("[SyntheticDataXML] parse_xml: routine type ", self.routine_type, " not found")
            raise

        num_rooms = 0
        for child in routine_info.find("rooms"):
            self.space_id[child.tag] = num_rooms
            self.id_space[num_rooms] = child.tag
            num_rooms += 1


class ReadSyntheticData:
    def __init__(self, file_path):
        self.data = self.read_file(file_path)

    # parse_time converts time in str(hh:mm:ss) to
    # int [hh, mm, ss] and total sec
    @staticmethod
    def parse_time(time_str):
        time_split = list(map(int, time_str.split(sep=':')))
        sec = time_split[0] * 3600 + time_split[1] * 60 + time_split[2]
        return time_split, sec

    # read_file read the parsed data as it is and returns the list
    # each row = [day(int), time(string), duration(int), activity(string), location(string / character)]
    @staticmethod
    def read_file(file_path):
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
        except IOError as err:
            print("[ReadSyntheticData] read_file: Error reading file ", file_path, " Error: ", err)
            raise
        del lines[0]

        data = []
        for line in lines:
            split_line = line.splitlines()[0].split(sep=',')
            day = int(split_line[0])
            stime = split_line[1]
            dur = int(split_line[2])
            activity = split_line[3]
            loc = split_line[4]
            data.append([day, stime, dur, activity, loc])
        return data

    def get_data(self):
        return self.data


# feature convention: dictionary
# numpy arrays and integers and sets
class Features:
    def __init__(self, time_bins, act_bins, type_bins, scale):
        self.time_bins = time_bins
        self.act_bins = act_bins
        self.type_bins = type_bins
        self.scale = scale

    def init_null_features(self):
        features = dict()
        features["num_clusters"] = 0
        features["loc"] = set()
        features["loc_type"] = set()
        features["loc_array"] = np.array([0] * self.act_bins)
        features["type_array"] = np.array([0] * self.type_bins)
        features["rood_idx"] = 0
        features["stime"] = 0
        features["time_hist"] = np.zeros(self.time_bins, dtype=int)
        features["duration"] = 0
        features["dur_hist"] = np.zeros(self.time_bins, dtype=int)
        features["prev_act_bag"] = None
        features["prev_act_avg"] = None
        return features

    # Features for clusters
    def init_features(self, loc, loc_type, stime, duration, loc_array, type_array, prev_act):
        features = self.init_null_features()
        features["num_clusters"] = 1

        if type(loc) == type(set()):
            features["loc"] = features["loc"].union(loc)
        else:
            features["loc"].add(loc)

        if type(loc_type) == type(set()):
            features["loc_type"] = features["loc_type"].union(loc_type)
        else:
            features["loc_type"].add(loc_type)

        features["stime"] = stime
        features["duration"] = duration

        scell = int(stime / self.scale)
        ecell = int((stime + duration) / self.scale)
        features["time_hist"][scell:ecell] = 1
        features["dur_hist"][0:int(duration / self.scale)] = 1

        features["loc_array"] = loc_array.copy()
        features["type_array"] = type_array.copy()

        if prev_act is not None:
            features["prev_act_bag"] = prev_act.copy()
        else:
            features["prev_act_bag"] = np.zeros(self.act_bins, dtype=int)

        features["prev_act_avg"] = features["prev_act_bag"].copy()

        return features

    @staticmethod
    def merge_features(feat1, feat2):
        feat1["loc"] = feat1["loc"].union(feat2["loc"])
        feat1["loc_type"] = feat1["loc_type"].union(feat2["loc_type"])
        feat1["stime"] = (feat1["stime"] * feat1["num_clusters"] + feat2["stime"] * feat2["num_clusters"]) / (
                feat1["num_clusters"] + feat2["num_clusters"])
        feat1["duration"] = (feat1["duration"] * feat1["num_clusters"] + feat2["duration"] * feat2["num_clusters"]) / (
                feat1["num_clusters"] + feat2["num_clusters"])
        feat1["num_clusters"] = feat1["num_clusters"] + feat2["num_clusters"]

        feat1["time_hist"] = feat1["time_hist"] + feat2["time_hist"]
        feat1["dur_hist"] = feat1["dur_hist"] + feat2["dur_hist"]

        feat1["loc_array"] = feat1["loc_array"] + feat2["loc_array"]
        feat1["type_array"] = feat1["type_array"] + feat2["type_array"]

        if feat1["prev_act_bag"] is None:
            feat1["prev_act_bag"] = feat2["prev_act_bag"].copy()
        else:
            feat1["prev_act_bag"] = np.vstack([feat1["prev_act_bag"], feat2["prev_act_bag"]])

        if feat1["prev_act_bag"].ndim > 1:
            feat1["prev_act_avg"] = feat1["prev_act_bag"].sum(axis=0) / len(feat1["prev_act_bag"])
        else:
            feat1["prev_act_avg"] = feat1["prev_act_bag"].copy()
        return feat1


class GenerateSyntheticCluster:
    def __init__(self, routine_type, file_path, scale=30):
        self.scale = scale
        self.data = self.get_cluster_data(file_path)
        self.num_days = 0
        self.obj_xml = SyntheticDataXML(routine_type)

    @staticmethod
    def get_cluster_data(file_path):
        data_obj = ReadSyntheticData(file_path)
        data = data_obj.get_data()
        return data

    def get_cluster_features(self):
        clust_feat = dict()
        time_bins = int(24 * 60 * 60 / self.scale)
        obj_feat = Features(time_bins=time_bins, act_bins=len(self.obj_xml.space_id), scale=self.scale)
        prev_day = 0
        prev_act = None
        for c in range(1, len(self.data) + 1):
            day = int(self.data[c - 1][0])
            if day != prev_day:
                prev_day = day
                prev_act = None
                self.num_days += 1

            loc = self.data[c - 1][4]
            loc_type = self.data[c - 1][4]
            stime = time_to_sec(self.data[c - 1][1])
            duration = int(self.data[c - 1][2])
            clust_feat[c] = obj_feat.init_features(loc=loc, loc_type=loc_type, stime=stime, duration=duration,
                                                   prev_act=prev_act)
            # print(clust_feat[c])
            prev_act = clust_feat[c]["prev_act_bag"].copy()
            prev_act[self.obj_xml.space_id[loc]] += 1
        return clust_feat

    def get_cluster_pixels(self):
        clust_pixels = dict()

        for c in range(1, len(self.data) + 1):
            day = int(self.data[c - 1][0])
            stime = time_to_sec(self.data[c - 1][1])
            duration = int(self.data[c - 1][2])
            scell = int(stime / self.scale)
            ecell = int((stime + duration) / self.scale)
            clust_pixels[c] = list()
            for cell in range(scell, ecell):
                clust_pixels[c].append((day - 1, cell))
        return clust_pixels

    def get_cluster_coarse(self):
        cluster_coarse = dict()

        for c in range(1, len(self.data) + 1):
            cluster_coarse[c] = [c]
        return cluster_coarse

    def merge_cluster_features(self, orig_clusters_features, new_cluster_info):
        cluster_feat = dict()
        time_bins = int(24 * 60 * 60 / self.scale)
        obj_feat = Features(time_bins=time_bins, act_bins=len(self.obj_xml.space_id), scale=self.scale)
        for new_c in new_cluster_info:
            cluster_feat[new_c] = obj_feat.init_null_features()
            for orig_c in new_cluster_info[new_c]:
                cluster_feat[new_c] = obj_feat.merge_features(cluster_feat[new_c], orig_clusters_features[orig_c])
        return cluster_feat


class GenerateRealDataCluster:
    def __init__(self, num_act, num_type, scale=5):
        self.scale = scale
        self.num_act = num_act
        self.num_type = num_type

    # the function takes generates clusters from a list of consecutive activities
    # Parameters:
    # > activities -> consecutive activities (dictionary)
    # > cluster_labels -> list of same dimensions as activities where each element signifies the cluster of
    #                     corresponding activity
    # > returns a dictionary with labels : features
    def make_single_day_clusters(self, activity, cluster_labels, day):
        cluster_feat = dict()
        cluster_pixels = dict()

        c_list = list()
        c_id = 0
        start_cell = 0

        obj_feat = Features(time_bins=0, act_bins=self.num_act, type_bins=self.num_type, scale=self.scale)
        temp = dict()
        temp["loc"] = set()
        temp["loc_type"] = set()
        temp["loc_array"] = np.array([0] * self.num_act)
        temp["type_array"] = np.array([0] * self.num_type)

        for a in range(len(activity)):
            temp["loc"] = temp["loc"].union(activity[a]["loc"])
            temp["loc_type"] = temp["loc_type"].union(activity[a]["loc_type"])
            temp["loc_array"] = np.add(temp["loc_array"], activity[a]["loc_array"])
            temp["type_array"] = np.add(temp["type_array"], activity[a]["type_array"])
            c_list.append((day-1, a))
            if a == len(activity)-1 or cluster_labels[a] != cluster_labels[a+1]:
                # make a cluster

                features = obj_feat.init_null_features()
                features["stime"] = start_cell*self.scale
                features["duration"] = (a-start_cell)*self.scale
                features["loc"] = temp["loc"].copy()
                features["loc_type"] = temp["loc_type"].copy()
                features["loc_array"] = temp["loc_array"].copy()
                features["type_array"] = temp["type_array"].copy()
                cluster_feat[c_id] = features
                cluster_pixels[c_id] = c_list.copy()

                temp["loc"] = set()
                temp["loc_type"] = set()
                temp["loc_array"] = np.array([0] * self.num_act)
                temp["type_array"] = np.array([0] * self.num_type)
                c_list = []

                start_cell = a+1
                c_id += 1

        return cluster_feat, cluster_pixels

    def merge_sameday_cluster_features(self, cluster_new_old, old_cluster_feat, old_cluster_pixel):
        new_cluster_feat = dict()
        new_cluster_pixel = dict()

        time_bins = int(24*60*60/self.scale)
        feat_obj = Features(time_bins, self.num_act, self.num_type, self.scale)
        prev_act = None
        for new in cluster_new_old:
            temp = feat_obj.init_null_features()
            temp["stime"] = math.inf
            new_cluster_pixel[new] = []
            for subc in cluster_new_old[new]:
                temp["stime"] = min(temp["stime"], old_cluster_feat[subc]["stime"])
                temp["duration"] += old_cluster_feat[subc]["duration"]
                temp["loc"] = temp["loc"].union(old_cluster_feat[subc]["loc"])
                temp["loc_type"] = temp["loc_type"].union(old_cluster_feat[subc]["loc_type"])
                temp["loc_array"] = temp["loc_array"] + old_cluster_feat[subc]["loc_array"]
                temp["type_array"] = temp["type_array"] + old_cluster_feat[subc]["type_array"]

                new_cluster_pixel[new] += old_cluster_pixel[subc]

            new_cluster_feat[new] = feat_obj.init_features(temp["loc"], temp["loc_type"],
                                                           temp["stime"], temp["duration"],
                                                           temp["loc_array"], temp["type_array"], prev_act)
            prev_act = new_cluster_feat[new]["prev_act_bag"] + new_cluster_feat[new]["loc_array"]
            del temp
        return new_cluster_feat, new_cluster_pixel

    def merge_cluster_features(self, orig_clusters_features, cluster_new_old):
        cluster_feat = dict()
        time_bins = int(24 * 60 * 60 / self.scale)
        obj_feat = Features(time_bins=time_bins, act_bins=self.num_act, type_bins=self.num_type, scale=self.scale)
        for new_c in cluster_new_old:
            cluster_feat[new_c] = obj_feat.init_null_features()
            for orig_c in cluster_new_old[new_c]:
                cluster_feat[new_c] = obj_feat.merge_features(cluster_feat[new_c], orig_clusters_features[orig_c])
        return cluster_feat


class ReadData:
    def __init__(self, subject_id=2,
                 num_days=30,
                 dir_name="../data/experimental_data/Subject_2/processed_data",
                 file_name=None,
                 scale=5):
        self.dir_name = dir_name
        self.file_name = file_name
        # self.unprocessed_data_dict, self.unprocessed_data_list = self.loadCSV()
        self.num_days = num_days
        self.scale = scale
        self.subject_id = subject_id
        self.subject_info = SpaceMapper(subject_id=subject_id)
        print("[DataPreparation] init: Object initialized for subject ", subject_id)
        print("[DataPreparation] init: Creating data for subject ", subject_id)
        self.image = self.readFiles2()

    def get_num_spaces(self):
        return len(self.subject_info.space_ids)

    def get_num_space_types(self):
        return len(self.subject_info.type_space)

    def readFiles(self):
        print("[ReadData] readFiles: Preparing data for subject ", self.subject_id, ".....")
        rows = self.num_days
        cols = int(24 * 60 * 60 / self.scale)
        img = []
        print("[ReadData] readFiles: reading log files of data")
        if self.file_name is None:
            log_files = glob.glob(self.dir_name + "/*.log")
        else:
            log_files = [self.dir_name + "/" + self.file_name]

        if len(log_files) == 0:
            print("[ReadData] readFiles: Error - No log files found in the folder ", self.dir_name)
            exit(-1)
        log_files.sort()

        obj_feat = Features(time_bins=0, act_bins=len(self.subject_info.space_ids), scale=self.scale)

        if self.num_days == -1:
            self.num_days = len(log_files)

        try:
            for file in log_files[:self.num_days]:
                print("[ReadData] readFiles: file read -> ", file)

                # initialize routine for current day
                routine = [None] * cols

                # open routine file
                with open(file, 'r') as f_read:
                    data = f_read.readlines()

                # delete the first line (header line)
                del data[0]
                sample_num = -1
                curr_info = None  # curr_info: [space_id, stime, duration]

                # iterate through all the files, each file is a record of a day
                for line in data:
                    sample_num += 1

                    # line: Time stamp (MM-DD-YYYY hh:mm:ss), x_coord, y_coord, is_motion, motion_score, space_id(char)
                    sample = re.split(',|\n', line)
                    time_split = re.split(' |:|-', sample[0])
                    seconds = int(time_split[3]) * 60 * 60 + int(time_split[4]) * 60 + int(time_split[5])

                    # if first sample, make curr info to keep track of start time and duration
                    if curr_info is None:
                        curr_info = dict()  # [0, 0, 0]
                        curr_info["loc"] = sample[5]
                        curr_info["stime"] = seconds
                        curr_info["dur"] = 0

                    # if activity changes, record the previous activity, make it's features and reinitialize curr time
                    if curr_info["loc"] != sample[5] or sample_num == len(data) - 1:
                        curr_info["dur"] = seconds - curr_info["stime"]

                        sample_feat = obj_feat.init_null_features()
                        sample_feat["num_clusters"] = 1
                        sample_feat["loc"].add(curr_info["loc"])
                        sample_feat["loc_type"].add(self.subject_info.space_type[curr_info["loc"]])
                        sample_feat["loc_array"][self.subject_info.space_ids[curr_info["loc"]]] = 1
                        sample_feat["stime"] = curr_info["stime"]
                        sample_feat["duration"] = curr_info["dur"]
                        sample_feat["room_idx"] = self.subject_info.space_ids[curr_info["loc"]]

                        start_cell = int(curr_info["stime"] / self.scale)
                        end_cell = int((curr_info["stime"] + curr_info["dur"]) / self.scale)
                        routine[start_cell:end_cell] = [sample_feat] * (end_cell - start_cell)

                        curr_info["loc"] = sample[5]
                        curr_info["stime"] = seconds
                        curr_info["dur"] = 0

                # check for missing cells
                for c in range(cols):
                    if routine[c] is None:
                        routine[c] = routine[c - 1]

                img.append(routine)

        except IOError as err:
            print("[ReadData] read_files: error reading log files. Error ", err)
            raise
        except Exception as err:
            print("[ReadData] read_files: Error ", err)
            raise

        if len(img) == 1:
            return img[0]
        else:
            return img

    def readFiles2(self):
        print("[ReadData] readFiles: Preparing data for subject ", self.subject_id, ".....")
        img = []
        print("[ReadData] readFiles: reading log files of data")
        if self.file_name is None:
            log_files = glob.glob(self.dir_name + "/*.log")
        else:
            log_files = [self.dir_name + "/" + self.file_name]

        if len(log_files) == 0:
            print("[ReadData] readFiles: Error - No log files found in the folder ", self.dir_name)
            exit(-1)
        log_files.sort()

        obj_feat = Features(time_bins=0,
                            act_bins=len(self.subject_info.space_ids),
                            type_bins=len(self.subject_info.type_space),
                            scale=self.scale)

        if self.num_days == -1:
            self.num_days = len(log_files)

        try:
            for file in log_files[:self.num_days]:
                print("[ReadData] readFiles: file read -> ", file)

                # initialize routine for current day
                routine = [None]*1440

                # open routine file
                with open(file, 'r') as f_read:
                    data = f_read.readlines()

                # delete the first line (header line)
                del data[0]
                sample_num = -1
                prev_min = 0
                container = dict()

                # iterate through all the files, each file is a record of a day
                for line in data:
                    sample_num += 1

                    # line: Time stamp (MM-DD-YYYY hh:mm:ss), x_coord, y_coord, is_motion, motion_score, space_id(char)
                    sample = re.split(',|\n', line)
                    # print(sample)
                    time_split = re.split(' |:', sample[1])
                    minute = int(time_split[1]) * 60 + int(time_split[2])
                    # print(minute)
                    if prev_min != minute or sample_num == len(data) - 1:
                        curr_info = dict()
                        curr_info["loc"] = max(container, key=container.get)
                        # print(curr_info["loc"])
                        sample_feat = obj_feat.init_null_features()
                        sample_feat["num_clusters"] = 1
                        sample_feat["loc"].add(curr_info["loc"])
                        sample_feat["loc_type"].add(self.subject_info.space_type[curr_info["loc"]])
                        sample_feat["loc_array"][self.subject_info.space_ids[curr_info["loc"]]] = 1
                        sample_feat["type_array"][self.subject_info.space_type[curr_info["loc"]]] = 1
                        sample_feat["room_idx"] = self.subject_info.space_ids[curr_info["loc"]]
                        routine[prev_min] = sample_feat
                        prev_min = minute
                        container = dict()

                    container[sample[-2]] = container.get(sample[-2], 0) + 1
                    # print(container)

                # check for missing cells
                for c in range(1440):
                    if routine[c] is None:
                        routine[c] = routine[c - 1]

                img.append(routine)

        except IOError as err:
            print("[ReadData] read_files: error reading log files. Error ", err)
            raise
        except Exception as err:
            print("[ReadData] read_files: Error ", err)
            raise

        if len(img) == 1:
            return img[0]
        else:
            return img


if __name__ == "__main__":
    # obj = GenerateSyntheticCluster("ADL1", "../data/synthetic_data/level2/parsed_data/synt_data_lvl2_days30_sd5_4.csv")
    # obj.get_cluster_features()

    obj = ReadData(subject_id=2, num_days=-1, dir_name="../data/real_data/Subject_2/")
    print("num days:", len(obj.image))
    print("num time stamps:", len(obj.image[0]))

    print(obj.image[0])

    # obj_xml = SyntheticDataXML("ADL1")
    # print(obj_xml.space_id)
    # exit(1)
