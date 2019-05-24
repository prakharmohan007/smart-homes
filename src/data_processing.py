import numpy as np
import csv
import re
import simplejson

# OS dependent libraries for reading and writing files
import glob
from os import listdir
from os.path import isfile, join

# local classes
from space_mapper import SpaceMapper

# spaces
# [A] : Bathroom 2
# [B] : Bed (in Room 3)
# [C] : Toilet 2
# [D] : Room 3
# [F] : Sofa (in Living room)
# [G] : Room 1
# [H] : Bathroom 1
# [K] : Kitchen
# [L] : Living room
# [M] : Room 2
# [N] : OUT
# [R] : Refrigerator (in Kitchen)
# [S] : Sink (in Kitchen)
# [T] : Dining table (in Kitchen)
# [Y] : Balcony

roomidx = {}
roomidx['A'] = 1
roomidx['B'] = 2
roomidx['C'] = 3
roomidx['D'] = 4
roomidx['F'] = 5
roomidx['G'] = 6
roomidx['H'] = 7
roomidx['K'] = 8
roomidx['L'] = 9
roomidx['M'] = 10
roomidx['O'] = 11
roomidx['R'] = 12
roomidx['S'] = 13
roomidx['T'] = 14
roomidx['Y'] = 15

# room types:
# 1 -> Bathroom (A, H)
# 2 -> room1 (G), room2(M), room3(D), Bed(B)
# 3 -> Toilet(C)
# 4 -> Living room(L), sofa(F)
# 5 -> kitchen(K), refrigerator(R), sink(S), table(T)
# 6 -> Out(N)
# 7 -> Balcony(Y)

room_type = {}
room_type['A'] = 1
room_type['B'] = 2
room_type['C'] = 3
room_type['D'] = 2
room_type['F'] = 4
room_type['G'] = 2
room_type['H'] = 1
room_type['K'] = 5
room_type['L'] = 4
room_type['M'] = 2
room_type['O'] = 6
room_type['R'] = 5
room_type['S'] = 5
room_type['T'] = 5
room_type['Y'] = 7


# feature class
class InitFeatures:
    def __init__(self, num_space=0, num_space_type=0, interval=5):
        self.interval = interval
        self.num_space = num_space
        self.num_space_types = num_space_type
        self.space = np.array([0] * num_space)
        self.space_id = set()
        self.space_type = set()
        self.stime = 0
        self.duration = 0

        num_dims = int(24 * 60 * 60 / interval)
        self.time_hist = np.array([0] * num_dims)
        self.space_hist = np.array([0] * num_dims)


# Class to convert raw data into standard data
# Standard Data: Time (MM-DD-YYYY hh:mm:ss), x_coord, y_coord, is_motion, motion_score, space_id
class PrepareData:
    def __init__(self, subject_id=2):
        self.subject_id = subject_id
        print("[PrepareData] init: starting space mapper for subject ", subject_id)
        self.obj_sm = SpaceMapper(subject_id=subject_id)
        print("[PrepareData] init: Space Mapped")
        # print(obj_sm.house_width, obj_sm.house_height)
        # print(obj_sm.space_ids)
        # print(obj_sm.house_center)
        # print(obj_sm.house_map)

    def parse_sample(self, sample):
        try:
            time_split = sample.split()

            split_list = re.split(' |:|\[|\]|\{|\}|\"|,', sample)
            split_list = ' '.join(split_list).split()
            # print(split_list)

            parse = dict()
            parse["time"] = time_split[0] + " " + time_split[1]

            # find information in sample
            index = 0
            length = len(split_list)
            # motion coordinates
            while split_list[index] != "motion_coordinates":
                index += 1

            if split_list[index + 1] == "null":
                parse["motion_coordinates"] = None
            else:
                parse["motion_coordinates"] = [split_list[index + 1], split_list[index + 2]]

            # is motion
            while split_list[index] != "is_motion":
                index += 1

            parse["is_motion"] = split_list[index + 1]

            # motion score
            while split_list[index] != "motion_score":
                index += 1

            parse["motion_score"] = split_list[index + 1]
        except Exception as err:
            print("[PrepareData] parse_sample: Error -> ", err)
            print("[PrepareData] parse_sample: Sample: ", sample)
            return parse["time"], False

        return parse, True

    def prepare_single_file_csv(self, file_name, target_dir):
        m_cord_prev = None
        m_cord = None
        print("[PrepareData] prepare_csv: file read -> ", file_name)

        target_file = target_dir + "/" + file_name.split('/')[-1]
        csv_writer = open(target_file, 'w')
        f_write = csv.writer(csv_writer, delimiter=',')
        f_write.writerow(["time", "motion_coordinates", "is_motion", "motion_score", "room_id"])
        with open(file_name, 'r') as f_read:
            data = f_read.readlines()
        for sample in data:
            parsed_sample = self.parse_sample(sample)
            if parsed_sample is None:
                continue
            # print(parsed_sample)

            if parsed_sample["motion_coordinates"] is None and m_cord_prev is None:
                m_cord = "null"
            elif parsed_sample["motion_coordinates"] is None and m_cord_prev is not None:
                m_cord = m_cord_prev
            elif parsed_sample["motion_coordinates"] is not None:
                m_cord = [parsed_sample["motion_coordinates"][0], parsed_sample["motion_coordinates"][1]]
                m_cord_prev = m_cord

            if m_cord != "null":
                x, room_id = self.obj_sm.mapCoordToSpace(float(m_cord[0]), float(m_cord[1]))
            else:
                room_id = "null"

            f_write.writerow(
                [parsed_sample["time"],
                 m_cord,
                 parsed_sample["is_motion"],
                 parsed_sample["motion_score"],
                 room_id]
            )
        csv_writer.close()
        # print(log_files)

    def prepare_csv(self, dir_name, target_dir):
        # open raw data log file
        # log_files = [f for f in listdir(dir_name) if isfile(join(dir_name, f))]
        print("[PrepareData] init: Preparing data for subject ", self.subject_id, ".....")

        print("[PrepareData] prepare_csv: reading log files of data")
        log_files = glob.glob(dir_name + "/*.log")
        log_files.sort()

        print("[PrepareData] prepare_csv: reading log files one by one")
        m_cord_prev = None
        m_cord = None
        prev_parsed_sample = {}
        parsed_sample = {}
        for file in log_files:
            print("[PrepareData] prepare_csv: file read -> ", file)
            target_file = target_dir + "/" + file.split('/')[-1]
            csv_writer = open(target_file, 'w')
            f_write = csv.writer(csv_writer, delimiter=',')
            f_write.writerow(["time", "motion_coordinates", "is_motion", "motion_score", "room_id"])
            with open(file, 'r') as f_read:
                data = f_read.readlines()
            for sample in data:
                received_sample, status = self.parse_sample(sample)
                if status is False:
                    parsed_sample = prev_parsed_sample
                    parsed_sample["time"] = received_sample
                else:
                    parsed_sample = received_sample
                # print(parsed_sample)

                if parsed_sample["motion_coordinates"] is None and m_cord_prev is None:
                    m_cord = "null"
                elif parsed_sample["motion_coordinates"] is None and m_cord_prev is not None:
                    m_cord = m_cord_prev
                elif parsed_sample["motion_coordinates"] is not None:
                    m_cord = [parsed_sample["motion_coordinates"][0], parsed_sample["motion_coordinates"][1]]
                    m_cord_prev = m_cord

                if m_cord != "null":
                    x, room_id = self.obj_sm.mapCoordToSpace(float(m_cord[0]), float(m_cord[1]))
                    if room_id == "null":
                        print("yes")
                else:
                    room_id = "E"

                f_write.writerow(
                    [parsed_sample["time"],
                     m_cord[0],
                     m_cord[1],
                     parsed_sample["is_motion"],
                     parsed_sample["motion_score"],
                     room_id]
                )
                prev_parsed_sample = parsed_sample
            csv_writer.close()
            # print(log_files)


# Class for preparing image structure / matrix from standard data format
# used InitFeature class
class DataPreparation:
    def __init__(self, subject_id=2,
                 dir_name="../data/experimental_data/Subject_2/processed_data",
                 num_days=30,
                 time_interval=5):
        self.dir_name = dir_name
        # self.unprocessed_data_dict, self.unprocessed_data_list = self.loadCSV()
        self.num_days = num_days
        self.time_interval = time_interval
        self.subject_id = subject_id
        self.subject_info = SpaceMapper(subject_id=subject_id)
        print("[DataPreparation] init: Object initialized for subject ", subject_id)
        print("[DataPreparation] init: Creating data for subject ", subject_id)
        self.image = self.read_files()

    # [current_space, start_time, duration, prev_space1, start_time_prev1, duration_prev1
    def create_features(self, prev_info=None, curr_info=None):
        feat_vec = InitFeatures(num_space=len(self.subject_info.space_ids),
                                num_space_type=len(self.subject_info.space_type),
                                interval=self.time_interval)

        # get index of space
        feat_vec.space[self.subject_info.space_ids[curr_info[0]]] = 1
        feat_vec.stime = curr_info[1]
        feat_vec.duration = curr_info[2]
        feat_vec.space_type.add(self.subject_info.space_type[curr_info[0]])
        return feat_vec

    def read_files(self):
        print("[DataPreparation] read_files: Preparing data for subject ", self.subject_id, ".....")
        rows = self.num_days
        cols = int(24 * 60 * 60 / self.time_interval)
        img = []
        print("[DataPreparation] read_files: reading log files of data")
        log_files = glob.glob(self.dir_name + "/*.log")
        if len(log_files) == 0:
            print("[DataPreparation] read_files: Error - No log files found in the folder ", self.dir_name)
            exit(-1)

        log_files.sort()
        print("[PrepareData] prepare_csv: reading log files one by one")

        try:
            for file in log_files[:self.num_days]:
                print("[DataPreparation] read_files: file read -> ", file)

                # initialize routine for current day
                routine = [None] * cols

                # open routine file
                with open(file, 'r') as f_read:
                    data = f_read.readlines()

                # delete the first line (header line)
                del data[0]
                sample_num = -1
                curr_info = None

                # iterate through all the files, each file is a record of a day
                for line in data:
                    sample_num += 1

                    # line: Time stamp (MM-DD-YYYY hh:mm:ss), x_coord, y_coord, is_motion, motion_score, space_id
                    sample = re.split(',|\n', line)
                    time_split = re.split(' |:|-', sample[0])
                    seconds = int(time_split[3]) * 60 * 60 + int(time_split[4]) * 60 + int(time_split[5])

                    # if first sample, make curr info to keep track of start time and duration
                    if curr_info is None:
                        curr_info = [0, 0, 0]
                        curr_info[0] = sample[5]
                        curr_info[1] = seconds
                        curr_info[2] = 0

                    # if activity changes, record the previous activity, make it's features and reinitialize curr time
                    if curr_info[0] != sample[5] or sample_num == len(data) - 1:
                        curr_info[2] = seconds - curr_info[1]
                        sample_feat = self.create_features(curr_info=curr_info)

                        start_cell = int(curr_info[1] / self.time_interval)

                        # print(curr_info[1] + curr_info[2])
                        end_cell = int((curr_info[1] + curr_info[2]) / self.time_interval)
                        routine[start_cell:end_cell] = [sample_feat] * (end_cell - start_cell)

                        curr_info[0] = sample[5]
                        curr_info[1] = seconds
                        curr_info[2] = 0

                # check for missing cells
                for c in range(cols):
                    if routine[c] is None:
                        routine[c] = routine[c - 1]

                img.append(routine)
        except IOError as err:
            print("[DataPreparation] read_files: error reading log files. Error ", err)
            raise
        except Exception as err:
            print("[DataPreparation] read_files: Error ", err)
            raise

        return img


# Class for organizing cluster features and operations
class ClusterProcessing:
    def __init__(self, interval):
        self.num_clusters = 0
        self.interval = interval
        # self.cluster_feat = {}
        # self.create_features(data, clusters)

    def get_cluster_features(self, data, clusters):
        self.num_clusters = len(clusters)
        cluster_feat = {}
        for i in range(self.num_clusters):
            # features = self.init_feature_vec()
            features = InitFeatures(num_space=data[0][0].num_space,
                                    num_space_type=data[0][0].num_space_types,
                                    interval=self.interval)

            for point in clusters[i]:
                r, c = point

                # vec_s = np.array(data[r][c]["space"])
                # space = space + vec_s
                features.space = features.space + data[r][c].space
                features.space[features.space > 0] = 1

                # features["stime"] += data[r][c]["stime"]
                # features["duration"] += data[r][c]["duration"]
                # features["type"] = data[r][c]["type"]
                features.stime += data[r][c].stime
                features.duration += data[r][c].duration
                features.space_type.union(data[r][c].space_type)

            # features["space"] = list(space)
            # features["stime"] = int(features["stime"] / len(clusters[i]))
            # features["duration"] = int(features["duration"] / len(clusters[i]))
            features.stime = int(features.stime / len(clusters[i]))
            features.duration = int(features.duration / len(clusters[i]))

            cluster_feat[i] = features
            # print(features)
            # print(self.cluster_feat[i + 1])
            return cluster_feat

    # space_id: Spaces where the person did some activity in the cluster
    # space_type: The type of space, toilet, room etc
    # time_hist: Time and duration of the activity in histogram
    # space_hist: duration in that space type
    def hist_init(self):
        secs = 24 * 60 * 60
        num_cells = int(secs / self.interval)
        histograms = {}
        histograms["space_id"] = np.array([0] * 15)
        histograms["space_type"] = set()
        histograms["time_hist"] = np.array([0] * num_cells)
        histograms["space_hist"] = np.array([0] * num_cells)
        return histograms

    # TODO: how to use prev and later activity histograms
    def get_hist(self, histograms, data):
        # if len(histograms["time_hist"]) == 0 or len(histograms["space_hist"]) == 0:
        #     print("[ClusterProcessing] get_hist: histograms are not initialized")
        #     raise

        if len(histograms.time_hist) == 0 or len(histograms.space_hist) == 0:
            print("[ClusterProcessing] get_hist: histograms are not initialized")
            raise

        # if len(histograms["time_hist"]) != len(histograms["space_hist"]):
        #     print("[ClusterProcessing] get_hist: histograms are of incompatible sizes")
        #     raise

        if len(histograms.time_hist) != len(histograms.space_hist):
            print("[ClusterProcessing] get_hist: histograms are of incompatible sizes")
            raise

        try:
            # histograms["space_id"] = np.logical_or(histograms["space_id"], np.array(data["space"]))
            # histograms["space_type"].add(data["type"])

            histograms.space = np.logical_or(histograms.space, np.array(data.space))
            histograms.space_type.union(data.space_type)

            # start_interval = int(data["stime"] / self.interval)
            # dur_intervals = int(data["duration"] / self.interval)
            start_interval = int(data.stime / self.interval)
            dur_intervals = int(data.duration / self.interval)
            # histograms["space_hist"][start_interval:start_interval + dur_intervals] = np.array([1] * dur_intervals)
            # histograms["space_hist"][0:dur_intervals] = np.array([1] * dur_intervals)
            # histograms["time_hist"][start_interval:start_interval + dur_intervals] = np.array([1] * dur_intervals)
            histograms.space_hist[0:dur_intervals] = np.array([1] * dur_intervals)
            histograms.time_hist[start_interval:start_interval + dur_intervals] = np.array([1] * dur_intervals)
        except Exception as err:
            print("[ClusterProcessing] get_hist: error in forming histograms. Error: ", err)
            raise

    def add_hist(self, histograms, data):
        # if len(histograms["time_hist"]) == 0 or len(histograms["space_hist"]) == 0:
        #     print("[ClusterProcessing] add_hist: histograms are not initialized")
        #     raise
        if len(histograms.time_hist) == 0 or len(histograms.space_hist) == 0:
            print("[ClusterProcessing] add_hist: histograms are not initialized")
            raise

        # if len(histograms["time_hist"]) != len(histograms["space_hist"]):
        #     print("[ClusterProcessing] add_hist: histograms are of incompatible sizes")
        #     raise

        if len(histograms.time_hist) != len(histograms.space_hist):
            print("[ClusterProcessing] add_hist: histograms are of incompatible sizes")
            raise

        # histograms["space_id"] = np.logical_or(histograms["space_id"], np.array(data["space"]))
        # histograms["space_type"].add(data["type"])
        histograms.space = np.logical_or(histograms.space, np.array(data.space))
        histograms.space_type.union(data.space_type)

        try:
            # start_interval = int(data["stime"] / self.interval)
            # dur_intervals = int(data["duration"] / self.interval)
            start_interval = int(data.stime / self.interval)
            dur_intervals = int(data.duration / self.interval)
            # histograms["space_hist"][start_interval:start_interval+dur_intervals] += np.array([1]*dur_intervals)
            # histograms["space_hist"][0:dur_intervals] += np.array([1] * dur_intervals)
            # histograms["time_hist"][start_interval:start_interval + dur_intervals] += np.array([1] * dur_intervals)
            histograms.space_hist[0:dur_intervals] += np.array([1] * dur_intervals)
            histograms.time_hist[start_interval:start_interval + dur_intervals] += np.array([1] * dur_intervals)
        except Exception as err:
            print("[ClusterProcessing] get_hist: error in forming histograms. Error: ", err)
            raise

    # After first Pass, the activities in one cluster will be exactly same
    # get_cluster_histogram prepares a simple histogram for each cluster
    # For each cluster, the histogram can be made using the first point in the cluster
    def get_cluster_histograms(self, data, clusters):
        self.num_clusters = len(clusters)
        cluster_feat = {}
        for key in clusters:
            # cluster_hist = self.hist_init()
            cluster_hist = InitFeatures(num_space=data[0][0].num_space,
                                        num_space_type=data[0][0].num_space_types,
                                        interval=self.interval)
            # for point in clusters[key]:
            #     r, c = point
            #     self.add_hist(cluster_hist, data[r][c])
            r, c = clusters[key][0]
            self.get_hist(cluster_hist, data[r][c])
            cluster_feat[key] = cluster_hist

        return cluster_feat


# class for working with toy data
class TempDataProcessing:
    def __init__(self, csv_file_name):
        print("Loading CSV File.....")
        data_list = self.load_csv(csv_file_name)
        print("Dividing data into days.....")
        every_day_record = self.divide_data_days(data_list)
        # print("number of days: ", len(every_day_record))
        self.img = self.create_image(every_day_record)

    def load_csv(self, filepath):
        with open(filepath) as csvfile:
            read_csv = csv.reader(csvfile, delimiter=',')
            data = list(read_csv)

        data_list = []
        data_dict = []
        del data[0]
        del data[-1]

        for row in data:
            dictionary = {}
            dictionary['space'] = row[1]
            dictionary['start time'] = row[2]
            dictionary['duration'] = row[3]
            dictionary['ms_avg'] = row[4]
            dictionary['ms_sd'] = row[5]

            data_list.append(row)
            data_dict.append(dictionary)
            # print (row[2])
        return data_list

    def divide_data_days(self, data):
        day = 0
        record_num = 0
        every_day_record = {}
        for record in data:
            # print(record[2])
            start_time = re.split(' |-|:', record[2])
            # print(start_time)
            if day != int(start_time[2]):
                day = int(start_time[2])
                record_num += 1
                every_day_record[record_num] = []
            every_day_record[record_num].append(record)

        return every_day_record

    def init_feature_vec(self):
        feat_vec = {}
        feat_vec["space"] = [0] * 15
        feat_vec["stime"] = 0
        feat_vec["duration"] = 0
        feat_vec["type"] = 0

        prev_feat = {}
        prev_feat["space"] = [0] * 15
        prev_feat["stime"] = 0
        prev_feat["duration"] = 0
        prev_feat["type"] = 0

        feat_vec["prev_act"] = [prev_feat]
        return feat_vec

    # [current_space, start_time, duration, prev_space1, start_time_prev1, duration_prev1
    def create_feature_vector(self, prev_info, curr_info):
        feat_vec = self.init_feature_vec()

        # get index of space
        curr_space = [0] * 15
        curr_space[roomidx[curr_info[0]] - 1] = 1

        feat_vec["space"] = curr_space.copy()
        feat_vec["stime"] = curr_info[1]
        feat_vec["duration"] = curr_info[2]
        feat_vec["type"] = room_type[curr_info[0]]
        # prev_space = [0]*15
        # prev_space[roomidx[prev_info[0]] - 1] = 1
        # feat_vec["prev_act"][0]["space"] = prev_space.copy()
        feat_vec["prev_act"][0]["space"] = prev_info["space"]
        feat_vec["prev_act"][0]["stime"] = prev_info["stime"]
        feat_vec["prev_act"][0]["duration"] = prev_info["duration"]
        feat_vec["prev_act"][0]["type"] = prev_info["type"]
        return feat_vec

    def create_image(self, every_day_record):
        num_days = len(every_day_record)
        cols = int(24 * 60)
        rows = num_days
        size = rows, cols
        # img = np.zeros(size, dtype=np.uint8)
        img = []
        i = 0
        for idx in every_day_record:
            day = every_day_record[idx]
            i += 1
            prev_feat_vec = self.init_feature_vec()
            # routine = [prev_feat_vec]*cols
            routine = [0] * cols

            for record in day:
                # print (record)
                start_time = re.split(' |-|:', record[2])
                start_sec = int(start_time[3]) * 60 * 60 + int(start_time[4]) * 60 + int(start_time[5])
                duration_sec = int(record[3])

                # get start and end minutes to fill corresponding cells
                start_min = int(start_time[3]) * 60 + int(start_time[4])
                end_min = start_min + int(int(record[3]) / 60)

                # prepare information to create feature vector
                curr_info = [record[1], start_sec, duration_sec]
                feat_vec = self.create_feature_vector(prev_feat_vec, curr_info)

                routine[start_min:end_min] = [feat_vec] * (end_min - start_min)
                prev_feat_vec = feat_vec
            img.append(routine)
        # img = np.array(img)
        return img


if __name__ == '__main__':
    # obj_data = TempDataProcessing("../data/180724_180810_mod.csv")
    # obj_data = TempDataProcessing("../data/toy_example.csv")
    # print("number of days:", len(obj_data.img))
    # print("number of intervals: ", len(obj_data.img[0]))

    # with open("../data/image.log", 'w') as image_log:
    #     simplejson.dump(obj_data.img, image_log)

    # obj_data = PrepareData(2)
    # obj_data.prepare_csv("../data/experimental_data/Subject_2/corrected_data",
    #                      "../data/experimental_data/Subject_2/processed_data")

    obj_data = DataPreparation(subject_id=2, num_days=5)
    print("number of days:", len(obj_data.image))
    print("number of intervals: ", len(obj_data.image[0]))
    print("A feature sample: ", obj_data.image[0][1].stime, obj_data.image[0][1].duration)

    # with open("../data/image.log", 'w') as image_log:
    #     simplejson.dump(obj_data.img, image_log)

    exit(1)
