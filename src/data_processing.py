import numpy as np
import xml.etree.ElementTree as ET
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
    def __init__(self, time_bins, act_bins, scale):
        self.time_bins = time_bins
        self.act_bins = act_bins
        self.scale = scale

    def init_null_features(self):
        features = dict()
        features["num_clusters"] = 0
        features["loc"] = set()
        features["loc_type"] = set()
        features["stime"] = 0
        features["time_hist"] = np.zeros(self.time_bins, dtype=int)
        features["duration"] = 0
        features["dur_hist"] = np.zeros(self.time_bins, dtype=int)
        features["prev_act_bag"] = None
        features["prev_act_avg"] = None
        return features

    def init_features(self, loc, loc_type, stime, duration, prev_act):
        features = self.init_null_features()
        features["num_clusters"] = 1
        features["loc"].add(loc)
        features["loc_type"].add(loc_type)
        features["stime"] = stime
        features["duration"] = duration

        scell = int(stime / self.scale)
        ecell = int((stime + duration) / self.scale)
        features["time_hist"][scell:ecell] = 1

        features["dur_hist"][0:int(duration / self.scale)] = 1

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


class ReadData:
    def __init__(self, subject_id=2,
                 num_days=30,
                 dir_name="../data/experimental_data/Subject_2/processed_data",
                 file_name=None,
                 time_interval=5):
        self.dir_name = dir_name
        self.file_name = file_name
        # self.unprocessed_data_dict, self.unprocessed_data_list = self.loadCSV()
        self.num_days = num_days
        self.time_interval = time_interval
        self.subject_id = subject_id
        self.subject_info = SpaceMapper(subject_id=subject_id)
        print("[DataPreparation] init: Object initialized for subject ", subject_id)
        print("[DataPreparation] init: Creating data for subject ", subject_id)
        self.image = self.read_files()


if __name__ == "__main__":
    obj = GenerateSyntheticCluster("ADL1", "../data/synthetic_data/level2/parsed_data/synt_data_lvl2_days30_sd5_4.csv")
    obj.get_cluster_features()

    # obj_xml = SyntheticDataXML("ADL1")
    # print(obj_xml.space_id)
    # exit(1)
