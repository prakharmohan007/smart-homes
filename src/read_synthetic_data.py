# Author: Prakhar Mohan
# Title: Reading Synthetic Routine of basic ADL

import csv
import cv2
import colorsys
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
                curr_routine.append(split_line[-1])
                sec = sec + self.scale

            if line == len(data) - 1:
                routine.append(curr_routine)

            prev_day = curr_day
        return routine

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
            img = self.get_image(routine)

            if IMG_SAVE:
                img_name = "synt_data_lvl" + str(1) + "_days" + str(self.num_days) + "_" + str(f_no) + ".jpg"
                image_path = self.base_dir + "image_routine/" + img_name
                cv2.imwrite(image_path, img)
                # cv2.imshow("image", img)
                # cv2.waitKey(0)

    # level: What levels are required. 1: Lvl1, 2:lvl2, 3:lvl3, 12:lvl1&2, 13:lvl1&3, 23:lvl2&3, 123:lvl1,2&3
    def read_synthetic_data(self, level=1, scale=30, num_days=30, num_files=5,
                            base_dir="../data/synthetic_data/"):
        self.scale = scale
        self.num_days = num_days
        self.num_files = num_files
        self.base_dir = base_dir
        if level == 1:
            self.read_level1()
        elif level == 2:
            self.read_level2()
        elif level == 12:
            self.read_level1()
            self.read_level2()


if __name__ == '__main__':
    obj = ReadSyntheticData()
    obj.read_synthetic_data(level=12, num_days=30)
    exit(1)
