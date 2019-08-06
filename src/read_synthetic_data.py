# Author: Prakhar Mohan
# Title: Reading Synthetic Routine of basic ADL

import csv
import cv2
import colorsys
import re
from random import shuffle
import numpy as np
from pathlib import Path

IMG_SAVE = 1


class ReadSyntheticData:
    def __init__(self, scale=30, num_days=30, num_files=5,
                 base_dir="../data/synthetic_data/"):
        self.scale = scale
        self.num_days = num_days
        self.num_files = num_files
        self.base_dir = base_dir

    @staticmethod
    def HSV_to_RGB(h, s, v):
        (r, g, b) = colorsys.hsv_to_rgb(h, s, v)
        return int(255 * r), int(255 * g), int(255 * b)

    @staticmethod
    def get_distinct_colors(n):
        hue_partition = 1.0 / (n + 1)
        return (ReadSyntheticData.HSV_to_RGB(hue_partition * value, 1.0, 1.0) for value in range(0, n))

    @staticmethod
    def get_image(routine):
        elements = set()
        rows = len(routine)
        cols = len(routine[0])

        for row in routine:
            for col in row:
                if col not in elements:
                    elements.add(col)
        elements = list(elements)
        num_elements = len(elements)
        color_gen = ReadSyntheticData.get_distinct_colors(num_elements)
        colors = []
        for c in color_gen:
            colors.append(c)
        shuffle(colors)
        color_comb = list()
        color_comb.append(elements)
        color_comb.append(colors)

        image = np.zeros((len(routine) * 50, len(routine[0]), 3), np.uint8)
        # print(len(routine[0]))
        # for row in range(rows):
        #     for col in range(cols):
        for row in range(len(routine)):
            for col in range(len(routine[row])):
                color = color_comb[1][color_comb[0].index(routine[row][col])]
                image[row * 50:(row + 1) * 50, col] = color

        return image

    @staticmethod
    def get_image2(routine):
        elements = set()
        labelset = dict()
        rows = len(routine)
        cols = len(routine[0])
        # print(rows, cols)
        label = 0
        p = 0
        r = 1
        c = 1
        while p < rows*cols:
            j = c-1
            for i in range(0, r):
                # print(i, j)
                if routine[i][j] not in elements:
                    elements.add(routine[i][j])
                    labelset[routine[i][j]] = label
                    label += 1
                p += 1
            i = r-1
            if r == c:
                for j in range(j-1, -1, -1):
                    if routine[i][j] not in elements:
                        elements.add(routine[i][j])
                        labelset[routine[i][j]] = label
                        label += 1
                    p += 1

            c = c+1
            if r < rows:
                r = r+1
            # print(p)

        elements = list(elements)
        # labelset = list(labelset)
        num_elements = len(elements)
        color_gen = ReadSyntheticData.get_distinct_colors(num_elements)
        colors = []
        for c in color_gen:
            colors.append(c)
        shuffle(colors)
        color_comb = list()
        color_comb.append(elements)
        color_comb.append(colors)
        # color_comb.append(labelset)

        # label_comb = list()
        # label_comb.append(elements)
        # label_comb.append(labelset)

        image = np.zeros((len(routine) * 50, len(routine[0]), 3), np.uint8)
        label_img = np.zeros((len(routine)*50, len(routine[0]), 1), np.uint8)
        # print(len(routine[0]))
        # for row in range(rows):
        #     for col in range(cols):
        for row in range(len(routine)):
            for col in range(len(routine[row])):
                color = color_comb[1][color_comb[0].index(routine[row][col])]
                image[row * 50:(row + 1) * 50, col] = color
                # label = color_comb[2][color_comb[0].index(routine[row][col])]
                label = labelset[routine[row][col]]
                label_img[row * 50:(row + 1) * 50, col] = label

        return image, label_img

    @staticmethod
    def get_sec_from_time(time_str):
        split_time = time_str.split(sep=':')
        sec = int(split_time[0]) * 60 * 60 + int(split_time[1]) * 60 + int(split_time[2])
        return sec

    def read_parsed_file(self, file_path):

        try:
            with open(file_path, 'r') as csv_file:
                data = csv_file.readlines()
        except IOError as err:
            print("[ReadSyntheticData] read_parsed_file: error reading file. Error: ", err)
            raise
        del data[0]

        routine = []
        curr_routine = []
        prev_day = None
        sec = 0
        for line in range(len(data)):
            split_line = data[line].splitlines()[0].split(sep=',')
            curr_day = int(split_line[0])
            curr_dur = int(split_line[2])

            if line != 0 and prev_day != curr_day:
                routine.append(curr_routine)
                curr_routine = []
                sec = 0

            start_sec = self.get_sec_from_time(split_line[1])
            end_sec = start_sec + curr_dur
            while sec < end_sec:
                curr_routine.append(split_line[-2])
                sec = sec + self.scale

            if line == len(data) - 1:
                routine.append(curr_routine)

            prev_day = curr_day
        return routine

    def read_level(self, level=1, controlled=False, sdp=10, prob=None, num_noise=5):
        file_dir = self.base_dir + "level" + str(level) + "/"
        print("[ReadSyntheticData] read_level: Reading files of level" + str(level) + " synthetic data")

        if level == 2:
            num_noise_str = "_noise" + str(num_noise)
        else:
            num_noise_str = ""

        if level == 3:
            prob_str = "_prob" + str(prob)
        else:
            prob_str = ""

        for f_no in range(1, self.num_files + 1):
            if controlled:
                file_name = "synt_data_lvl" + str(level) + "_days" + str(self.num_days) + "_sd" + str(sdp) + prob_str + num_noise_str + "_" + str(f_no) + ".csv"
            else:
                file_name = "synt_data_lvl" + str(level) + "_days" + str(self.num_days) + prob_str + num_noise_str +"_" + str(f_no) + ".csv"

            # read parsed file
            file_path = file_dir + "parsed_data/" + file_name
            print("[ReadSyntheticData] read_level: File being read is ", file_path)
            routine = self.read_parsed_file(file_path)
            img, label_img = self.get_image2(routine)

            if IMG_SAVE:
                if controlled:
                    img_name = "synt_data_lvl" + str(level) + "_days" + str(self.num_days) + "_sd" + str(
                        sdp) + prob_str + num_noise_str + "_" + str(f_no) + ".png"
                else:
                    img_name = "synt_data_lvl" + str(level) + "_days" + str(self.num_days) + str(prob_str) + num_noise_str + "_" + str(
                        f_no) + ".png"
                image_path = self.base_dir + "image_routine/" + img_name
                label_image_path = self.base_dir + "image_routine/label_img/" + img_name
                cv2.imwrite(image_path, img)
                cv2.imwrite(label_image_path, label_img)
                # cv2.imshow("image", img)
                # cv2.waitKey(0)

    def read_level2(self):
        file_dir = self.base_dir + "level2/"
        print("[ReadSyntheticData] read_level2: Reading files of level 2 synthetic data")
        for f_no in range(1, self.num_files + 1):
            file_name = "synt_data_lvl" + str(2) + "_days" + str(self.num_days) + "_" + str(f_no) + ".csv"

            # read parsed file
            file_path = file_dir + "parsed_data/" + file_name
            print("[ReadSyntheticData] read_level2: File being read is ", file_path)
            self.read_parsed_file(file_path)

            routine = self.read_parsed_file(file_path)
            img = self.get_image(routine)

            if IMG_SAVE:
                img_name = "synt_data_lvl" + str(2) + "_days" + str(self.num_days) + "_" + str(f_no) + ".jpg"
                image_path = self.base_dir + "image_routine/" + img_name
                cv2.imwrite(image_path, img)
                # cv2.imshow("image", img)
                # cv2.waitKey(0)

    def read_level1(self):
        file_dir = self.base_dir + "level1/"
        print("[ReadSyntheticData] read_level1: Reading files of level 1 synthetic data")
        for f_no in range(1, self.num_files + 1):
            file_name = "synt_data_lvl" + str(1) + "_days" + str(self.num_days) + "_" + str(f_no) + ".csv"

            # read parsed file
            file_path = file_dir + "parsed_data/" + file_name
            print("[ReadSyntheticData] read_level1: File being read is ", file_path)
            self.read_parsed_file(file_path)

            routine = self.read_parsed_file(file_path)
            img, label_img = self.get_image(routine)

            if IMG_SAVE:
                img_name = "synt_data_lvl" + str(1) + "_days" + str(self.num_days) + "_" + str(f_no) + ".png"
                image_path = self.base_dir + "image_routine/" + img_name
                label_img_path = self.base_dir + "image_routine/label_img/" + img_name
                cv2.imwrite(image_path, img)
                cv2.imwrite(label_img_path, label_img)
                # cv2.imshow("image", img)
                # cv2.waitKey(0)

    def read_uci_orig_data(self):
        uci_file = "../data/downloaded_datasets/uci_adl_dataset/OrdonezB_ADLs_edit.txt"
        with open(uci_file, 'r') as f:
            lines = f.readlines()
        del lines[0:2]
        # print(lines)
        processed_data = []
        processed_data.append(["day", "Start time", "Duration", "Activity", "Location"])
        day = None
        for line in lines:
            record = line.split()
            print(record)
            # start date, start time, end date, end time, activity, location?

            start_time = record[1]
            start_sec = self.get_sec_from_time(start_time)
            end_sec = self.get_sec_from_time(record[3])
            activity = record[4]

            if day is None:
                day = 1
            elif record[0] != record[2]:
                processed_record = [str(day), start_time, str(self.get_sec_from_time("23:59:59") - start_sec), activity,
                                    activity]
                processed_data.append(processed_record)
                day = day + 1
                start_time = "00:00:00"
                start_sec = 0

            processed_record = [str(day), start_time, str(end_sec - start_sec), activity, activity]
            processed_data.append(processed_record)

        f_csv = open("../data/synthetic_data/uci_adl/uci_adl_B_orig.csv", 'w')
        csv_writer = csv.writer(f_csv)
        for row in processed_data:
            csv_writer.writerow(row)
        f_csv.close()
        # print(processed_data)

    # level: What levels are required. 1: Lvl1, 2:lvl2, 3:lvl3, 12:lvl1&2, 13:lvl1&3, 23:lvl2&3, 123:lvl1,2&3
    def read_synthetic_data(self, level=1, scale=30, num_days=30,
                            num_files=5, controlled=False, sdp=10, prob=0.7, num_noise=5,
                            base_dir="../data/synthetic_data/"):
        self.scale = scale
        self.num_days = num_days
        self.num_files = num_files
        self.base_dir = base_dir
        if level == 1:
            self.read_level(level=1, controlled=controlled, sdp=sdp)
        elif level == 2:
            self.read_level(level=2, controlled=controlled, sdp=sdp, num_noise=num_noise)
        elif level == 3:
            self.read_level(level=3, controlled=controlled, sdp=sdp, prob=prob)
        elif level == 12:
            self.read_level(level=1, controlled=controlled, sdp=sdp)
            self.read_level(level=2, controlled=controlled, sdp=sdp, num_noise=num_noise)


if __name__ == '__main__':
    obj = ReadSyntheticData()
    num_noise = 10  # change this for controlling the number of noise each day in level2
    sdp = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    prob = [0.3, 0.5, 0.7, 0.9]
    for sd in sdp:
        # for p in prob:
        obj.read_synthetic_data(level=2, num_days=30, num_files=5, controlled=True, sdp=sd, prob=None, num_noise=num_noise)
    # obj.read_synthetic_data(level=3, controlled=True, sdp=10, prob=1.0)
    # obj.read_uci_orig_data()
    exit(1)
