import csv
import matplotlib.pyplot as plt
from skimage.measure import EllipseModel
import numpy as np
from matplotlib.patches import Ellipse
import math
from difflib import SequenceMatcher
import glob
import re

IMSAVE = 1
IMSHOW = 1


def median_filtering(list1d, window_size=12):
    container1 = dict()
    filtered = list()
    for i in range(len(list1d) + int(window_size / 2)):
        if i < len(list1d):
            if list1d[i] not in container1:
                container1[list1d[i]] = 1
            else:
                container1[list1d[i]] += 1

        if i >= window_size:
            container1[list1d[i - window_size]] -= 1

        if i >= int(window_size / 2):
            max_act = max(container1, key=container1.get)
            filtered.append(max_act)

    if len(filtered) != len(list1d):
        print("median filtering: lengths different")
        raise

    return filtered


def getSecFromTime(time_str):
    split_time = time_str.split(sep=':')
    sec = int(split_time[0]) * 60 * 60 + int(split_time[1]) * 60 + int(split_time[2])
    return sec


def getMinuteFromTime(time_str):
    split_time = time_str.split(sep=':')
    minute = int(split_time[0]) * 60 + int(split_time[1])
    return minute


# Class for Poincare Plot
class PoincarePlot:
    def __init__(self):
        self.results_path = "../data/synthetic_data/graphs/poincare_results/"

    @staticmethod
    def getActivityTableLevel1(filepath):
        converted_table = []
        try:
            with open(filepath, 'r') as file:
                lines = file.readlines()
        except IOError as err:
            print("[ReadSyntheticData] read_file: Error reading file ", filepath, " Error: ", err)
            raise
        del lines[0]
        print(len(lines))
        prev_day = 0
        for line in lines:
            split_line = line.splitlines()[0].split(sep=',')
            if prev_day != int(split_line[0]):
                converted_table.append([])
                prev_day = int(split_line[0])
            converted_table[-1].append(getMinuteFromTime(split_line[1]))

        # print(converted_table)
        return converted_table

    # Get level1 table of activities
    def getPointsLevel1(self, file_path):
        time_table = self.getActivityTableLevel1(file_path)
        x = []
        y = []
        points = []

        for r in range(len(time_table)-1):
            for c in range(min(len(time_table[r]), len(time_table[r+1]))):
                x.append(time_table[r][c])
                y.append(time_table[r+1][c])
                points.append((time_table[r][c], time_table[r+1][c]))

        return x, y, points

    @staticmethod
    def getActivityTableLevel2(filepath):
        converted_table = []
        try:
            with open(filepath, 'r') as file:
                lines = file.readlines()
        except IOError as err:
            print("[ReadSyntheticData] read_file: Error reading file ", filepath, " Error: ", err)
            raise
        del lines[0]
        prev_day = 0
        print(len(lines))
        for line in lines:
            split_line = line.splitlines()[0].split(sep=',')
            if prev_day != int(split_line[0]):
                converted_table.append([])
                prev_day = int(split_line[0])
            d = dict()
            d["time"] = getMinuteFromTime(split_line[1])
            d["act"] = split_line[3]
            # print(split_line[3])
            d["ispaired"] = False
            converted_table[-1].append(d)

        # print(converted_table)
        return converted_table

    # Get level2 table of activities
    def getPointsLevel2(self, file_path):
        time_table = self.getActivityTableLevel2(file_path)
        x = []
        y = []
        points = []

        for day in range(len(time_table) - 1):
            for act1 in time_table[day]:
                for act2 in time_table[day+1]:
                    if act1["act"] == act2["act"] and act2["ispaired"] is False:
                        if act1["act"] == 'nonroutine' and abs(act1["time"] - act2["time"]) > 180:
                            continue
                        x.append(act1["time"])
                        y.append(act2["time"])
                        points.append((act1["time"], act2["time"]))
                        act2["ispaired"] = True
                        break

        return x, y, points

    @staticmethod
    def getActivityTableLevel3(filepath):
        converted_table = []
        try:
            with open(filepath, 'r') as file:
                lines = file.readlines()
        except IOError as err:
            print("[ReadSyntheticData] read_file: Error reading file ", filepath, " Error: ", err)
            raise
        del lines[0]
        # print(len(lines))
        prev_day = 0
        for line in lines:
            split_line = line.splitlines()[0].split(sep=',')
            # print(split_line)
            if prev_day != int(split_line[0]):
                converted_table.append([])
                prev_day = int(split_line[0])
            d = dict()
            d["time"] = getMinuteFromTime(split_line[1])
            d["dur"] = int(split_line[2])
            d["act"] = split_line[4]
            # print(split_line[3])
            d["diff"] = math.inf
            d["key"] = -1
            converted_table[-1].append(d)

        # print(converted_table)
        return converted_table

    # Get level3 table of activities
    def getPointsLevel3(self, file_path):
        time_table = self.getActivityTableLevel3(file_path)
        x = []
        y = []
        points = []

        for day in range(len(time_table) - 1):
            d1 = time_table[day].copy()
            d2 = time_table[day+1].copy()
            for idx1 in range(len(d1)):
                act1 = d1[idx1]
                for idx2 in range(len(d2)):
                    act2 = d2[idx2]
                    if act1["act"] == act2["act"] and abs(act1["time"] - act2["time"]) < act1["diff"] and \
                            abs(act1["time"] - act2["time"]) < act2["diff"] and \
                            abs(act1["time"] - act2["time"]) < 660 and \
                            abs(act1["dur"] - act2["dur"]) < min(act1["dur"], act2["dur"]):
                        act1["diff"] = abs(act1["dur"] - act2["dur"])
                        act2["diff"] = abs(act1["dur"] - act2["dur"])
                        act1["key"] = idx2
                        act2["key"] = idx1

            for idx1 in range(len(d1)):
                idx2 = d1[idx1]["key"]
                try:
                    if idx2 != -1 and d2[idx2]["key"] == idx1:
                        x.append(d1[idx1]["time"])
                        y.append(d2[idx2]["time"])
                        points.append((d1[idx1]["time"], d2[idx2]["time"]))
                        # print(day, d1[idx1]["act"], d1[idx1]["time"], day+1, d2[idx2]["act"], d2[idx2]["time"])
                except IndexError as err:
                    # print(idx1, idx2)
                    pass

        return x, y, points

    def plotPairs(self):
        pair_x, pair_y, points = self.getPointsLevel3("../data/synthetic_data/level3/parsed_data/synt_data_lvl3_days30_sd25_prob0.7_4.csv")
        # pair_x, pair_y, points = self.getPointsLevel1("../data/synthetic_data/level1/parsed_data/synt_data_lvl1_days30_sd5_4.csv")
        # pair_x, pair_y, points = self.getPointsLevel2("../data/synthetic_data/level2/parsed_data/synt_data_lvl2_days30_sd5_noise10_4.csv")

        print(len(points))
        np_points = np.array(points)
        ell = EllipseModel()
        ell.estimate(np_points)

        xc, yc, a, b, theta = ell.params
        print("Elipse Details:")
        print("Center:", (xc, yc))
        print("Angle:", math.degrees(theta))
        print("Axis Length:", a, b)
        ell_patch = Ellipse((xc, yc), 2 * a, 2 * b, theta * 180 / np.pi, edgecolor='red', facecolor='none')

        lax1 = xc + b*math.cos(math.pi/2 + theta)
        lay1 = yc + b*math.sin(math.pi/2 + theta)
        lax2 = xc - b * math.cos(math.pi/2 + theta)
        lay2 = yc - b * math.sin(math.pi/2 + theta)

        sax1 = xc + a * math.cos(theta)
        say1 = yc + a * math.sin(theta)
        sax2 = xc - a * math.cos(theta)
        say2 = yc - a * math.sin(theta)

        fig, axs = plt.subplots(1, 1)  # , sharex=True, sharey=True)
        plt.scatter(pair_x, pair_y, color='black', s=20)
        # plt.plot(pair_x, pair_y, 'k', marker='o')
        # plt.plot([], pair_y, 'k', marker='o')
        plt.plot([0, 1600], [0, 1600], 'k--')
        plt.plot([lax1, lax2], [lay1, lay2], 'b-', linewidth=3)
        plt.plot([sax1, sax2], [say1, say2], 'g-', linewidth=3)

        plt.xlim(0, 1600)
        plt.ylim(0, 1600)
        axs.add_patch(ell_patch)
        plt.show()

    @staticmethod
    def plotPoincarePlot(x, y, points, level, sd, results_path, prob=""):

        x_start = 0  # min(x)
        x_end = 1600  # max(x)

        y_start = 0  # min(y)
        y_end = 1600  # max(y)

        # define elipse
        ell = EllipseModel()
        ell.estimate(points)
        xc, yc, a, b, theta = ell.params
        print("Elipse Details:")
        print("Center:", (xc, yc))
        print("Angle:", theta * 180 / np.pi)
        print("Axis Length:", a, b)
        # ell_patch = Ellipse((xc, yc), 2 * a, 2 * b, theta * 180 / np.pi,
        #                     edgecolor='red', facecolor='none', linestyle='--')
        ell_patch = Ellipse((xc, yc), 2*a, 2*b, theta * 180 / np.pi,
                            edgecolor='red', facecolor='none', linestyle='--')

        short_axis = min(a, b)
        long_axis = max(a, b)

        if a > b:
            theta = math.pi / 2 + theta

        # long axis
        lax1 = xc + long_axis * math.cos(math.pi / 2 + theta)
        lay1 = yc + long_axis * math.sin(math.pi / 2 + theta)
        lax2 = xc - long_axis * math.cos(math.pi / 2 + theta)
        lay2 = yc - long_axis * math.sin(math.pi / 2 + theta)

        # short axis
        sax1 = xc + short_axis * math.cos(theta)
        say1 = yc + short_axis * math.sin(theta)
        sax2 = xc - short_axis * math.cos(theta)
        say2 = yc - short_axis * math.sin(theta)

        # plotting
        fig, axs = plt.subplots(1, 1)  # , sharex=True, sharey=True)
        axs.scatter(x, y, color='black', s=5)
        axs.plot([x_start, x_end], [y_start, y_end], 'k--')
        axs.plot([lax1, lax2], [lay1, lay2], 'b-', linewidth=4)
        axs.plot([sax1, sax2], [say1, say2], 'g-', linewidth=4)
        axs.set_xlabel("ADL(time)")
        axs.set_ylabel("ADL(time-delta)")
        axs.set_xlim(x_start, x_end)
        axs.set_ylim(y_start, y_end)
        axs.add_patch(ell_patch)

        if IMSHOW:
            # plt.show()
            pass
        if IMSAVE:
            plt.savefig(results_path + "Level"+str(level)+"_sd"+str(sd)+prob+".png")

        return (xc, yc), short_axis, long_axis

    @staticmethod
    def plotGraphs(self, x_axis, center, long_axis, short_axis, level):

        center_x = []
        center_y = []

        for (x, y) in center:
            center_x.append(x)
            center_y.append(y)

        plt.rc('xtick', labelsize=15)
        plt.rc('ytick', labelsize=15)
        fig, axs = plt.subplots(3, 1, sharex='none', sharey='none')
        axs[0].plot(x_axis, short_axis, 'k')
        axs[0].plot(x_axis, short_axis, 'k', marker='o')
        axs[0].set_xlabel("standard deviation", fontsize=15)
        axs[0].set_ylabel("Short Axis Length", fontsize=15)
        # axs[0].set_title("Short Axis Length for Synthetic Data")

        axs[1].plot(x_axis, long_axis, 'k')
        axs[1].plot(x_axis, long_axis, 'k', marker='o')
        axs[1].set_xlabel("standard deviation", fontsize=15)
        axs[1].set_ylabel("Long Axis Length", fontsize=15)
        # axs[1].set_title("Long Axis Length for Synthetic Data")

        axs[2].scatter(center_x, center_y, color='black', s=5)
        axs[2].set_xlabel("x coordinate", fontsize=15)
        axs[2].set_ylabel("y coordinate", fontsize=15)
        # axs[2].set_title("Center of ellipse for Synthetic Data")

        if IMSAVE:
            # plt.savefig(self.results_path + "PoincareRes_Level"+str(level)+".png")
            pass
        if IMSHOW:
            plt.show()

    def plotGraphs3(self, x_axis, center, long_axis, short_axis, prob_labels):

        cmap = plt.cm.get_cmap("hsv", len(prob_labels))
        fig, axs = plt.subplots(3, 1, sharex='none', sharey='none')
        for i in range(len(prob_labels)):
            center_x = []
            center_y = []

            for (x, y) in center[i]:
                center_x.append(x)
                center_y.append(y)

            plt.rc('xtick', labelsize=15)
            plt.rc('ytick', labelsize=15)
            axs[0].plot(x_axis, short_axis[i], c=cmap(i), label=str(prob_labels[i]))
            axs[0].plot(x_axis, short_axis[i], c=cmap(i), marker='o')
            axs[0].set_xlabel("standard deviation", fontsize=15)
            axs[0].set_ylabel("Short Axis Length", fontsize=15)
            # axs[0].set_title("Short Axis Length for Synthetic Data")

            axs[1].plot(x_axis, long_axis[i], c=cmap(i), label=str(prob_labels[i]))
            axs[1].plot(x_axis, long_axis[i], c=cmap(i), marker='o')
            axs[1].set_xlabel("standard deviation", fontsize=15)
            axs[1].set_ylabel("Long Axis Length", fontsize=15)
            # axs[1].set_title("Long Axis Length for Synthetic Data")

            axs[2].scatter(center_x, center_y, color='black', s=5)
            axs[2].set_xlabel("x coordinate", fontsize=15)
            axs[2].set_ylabel("y coordinate", fontsize=15)
            # axs[2].set_title("Center of ellipse for Synthetic Data")

        axs[0].legend(prop={'size': 15})
        axs[1].legend(prop={'size': 15})
        if IMSAVE:
            # plt.savefig(self.results_path + "PoincareRes_Level"+str(level)+".png")
            pass
        if IMSHOW:
            plt.show()

    def main(self):
        lvl = 3
        noise = [5, 10, 15, 20]
        prob = [0.3, 0.5, 0.7, 0.9]
        sd = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        filenum = 5

        short_axis_all = []
        long_axis_all = []
        center_all = []

        n = 0  # for n in noise:
        for p in prob:
            short_axis = []
            long_axis = []
            center = []
            for d in sd:
                point_x = []
                point_y = []
                points = []

                for f_num in range(1, filenum + 1):
                    if lvl == 1:
                        f_name = "synt_data_lvl" + str(lvl) + "_days30_sd" + str(d) + "_" + str(f_num)
                    elif lvl == 2:
                        f_name = "synt_data_lvl" + str(lvl) + "_days30_sd" + str(d) + "_noise" + str(n) + "_" + str(f_num)
                    else:
                        f_name = "synt_data_lvl" + str(lvl) + "_days30_sd" + str(d) + "_prob" + str(p) + "_" + str(
                            f_num)

                    filepath = "../data/synthetic_data/level" + str(lvl) + "/parsed_data/" + f_name + ".csv"
                    print(filepath)

                    if lvl == 1:
                        x, y, pp = self.getPointsLevel1(filepath)
                    elif lvl == 2:
                        x, y, pp = self.getPointsLevel2(filepath)
                    elif lvl == 3:
                        x, y, pp = self.getPointsLevel3(filepath)

                    point_x = point_x + x
                    point_y = point_y + y
                    points = points + pp

                points = np.array(points)
                str_prob = "_prob"+str(p)
                c, sa, la = self.plotPoincarePlot(point_x, point_y, points, lvl, d, self.results_path, str_prob)

                center.append(c)
                short_axis.append(sa)
                long_axis.append(la)

            center_all.append(center)
            short_axis_all.append(short_axis)
            long_axis_all.append(long_axis)

        if lvl != 3:
            self.plotGraphs(sd, center_all[0], long_axis_all[0], short_axis_all[0], lvl)
        else:
            self.plotGraphs3(sd, center_all, long_axis_all, short_axis_all, prob)


class SequenceMatching:
    def __init__(self):
        pass

    @staticmethod
    def getSyntheticSequence(filepath):
        try:
            with open(filepath, 'r') as file:
                lines = file.readlines()
        except IOError as err:
            print("[ReadSyntheticData] read_file: Error reading file ", filepath, " Error: ", err)
            raise
        del lines[0]

        sequence = []
        prev_day = 0
        for line in lines:
            split_line = line.splitlines()[0].split(sep=',')
            if prev_day != int(split_line[0]):
                dummy = [None]*1440
                sequence.append([dummy])
                prev_day = int(split_line[0])
            stime = getMinuteFromTime(split_line[1])
            duration = int(int(split_line[2])/60)
            endtime = stime + duration+1
            sequence[-1][stime:endtime] = [split_line[-1]] * duration

        for r in range(len(sequence)):
            for c in range(len(sequence[r])):
                if sequence[r][c] is None:
                    print("None")
                    sequence[r][c] = sequence[r][c-1]
            print(sequence[r])

        return sequence

    def testSequenceMatching(self):
        filepath = "../data/synthetic_data/level1/parsed_data/synt_data_lvl1_days30_sd50_4.csv"
        sequence = self.getSyntheticSequence(filepath)

        avg = 0
        for s_row in range(len(sequence) - 1):
            # s = SequenceMatcher(lambda x: x == " ", sequence[s_row], sequence[s_row+1])
            s = SequenceMatcher(None, sequence[s_row], sequence[s_row + 1])
            avg += s.ratio()
            # print(s.quick_ratio())
            # print(s.ratio())
        print(avg/29)

    @staticmethod
    def getSequenceMatchingRatio(sequence):
        avg = 0
        quick_avg = 0
        avg_points = []
        q_avg_points = []
        for s_row in range(len(sequence) - 1):
            # s = SequenceMatcher(lambda x: x == " ", sequence[s_row], sequence[s_row+1])
            s = SequenceMatcher(None, sequence[s_row], sequence[s_row + 1])
            avg += s.ratio()
            avg_points.append(s.ratio())
            quick_avg += s.quick_ratio()
            q_avg_points.append(s.quick_ratio())
            # print(s.quick_ratio())
            # print(s.ratio())
        # print(avg/(len(sequence) - 1))
        return avg/(len(sequence) - 1), avg_points, quick_avg/(len(sequence) - 1), q_avg_points

    def plotGraphs3(self, x_axis, ratio, q_ratio, labels, x_label="Standard Deviation"):
        # cmap = plt.cm.get_cmap("hsv", len(prob_labels))
        cmap = plt.cm.jet(np.linspace(0, 1, len(labels)))
        fig, axs = plt.subplots(2, 1, sharex='none', sharey='none')
        for i in range(len(labels)):

            plt.rc('xtick', labelsize=15)
            plt.rc('ytick', labelsize=15)
            axs[0].plot(x_axis, ratio[i], c=cmap[i], label=str(labels[i]))
            axs[0].plot(x_axis, ratio[i], c=cmap[i], marker='o')
            axs[0].set_xlabel(x_label, fontsize=15)
            axs[0].set_ylabel("Similarity Ratio", fontsize=15)
            # axs[0].set_title("Short Axis Length for Synthetic Data")

            axs[1].plot(x_axis, q_ratio[i], c=cmap[i], label=str(labels[i]))
            axs[1].plot(x_axis, q_ratio[i], c=cmap[i], marker='o')
            axs[1].set_xlabel(x_label, fontsize=15)
            axs[1].set_ylabel("Similarity Q-Ratio", fontsize=15)
            # axs[1].set_title("Long Axis Length for Synthetic Data")

        axs[0].legend(prop={'size': 15}, loc="upper right")
        axs[1].legend(prop={'size': 15}, loc="upper right")
        if IMSAVE:
            # plt.savefig(self.results_path + "PoincareRes_Level"+str(level)+".png")
            pass
        if IMSHOW:
            plt.show()

    def plotGraphs(self, x_axis, ratio, ratio_points, quick_ratio, q_ratio_points, level):
        plt.rc('xtick', labelsize=15)
        plt.rc('ytick', labelsize=15)
        fig, axs = plt.subplots(2, 1, sharex='none', sharey='none')
        axs[0].plot(x_axis, ratio, 'k')
        axs[0].plot(x_axis, ratio, 'k', marker='o')
        for i in range(len(x_axis)):
            axs[0].scatter([x_axis[i]]*len(ratio_points[i]), ratio_points[i], color='black', s=5, marker='x')
        axs[0].set_xlabel("standard deviation", fontsize=15)
        axs[0].set_ylabel("Similarity Ratio", fontsize=15)
        # axs[0].set_title("Short Axis Length for Synthetic Data")
        axs[1].plot(x_axis, quick_ratio, 'k')
        axs[1].plot(x_axis, quick_ratio, 'k', marker='o')
        for i in range(len(x_axis)):
            axs[1].scatter([x_axis[i]]*len(q_ratio_points[i]), q_ratio_points[i], color='black', s=5, marker='x')
        axs[1].set_xlabel("standard deviation", fontsize=15)
        axs[1].set_ylabel("Similarity Q-Ratio", fontsize=15)

        if IMSAVE:
            # plt.savefig(self.results_path + "PoincareRes_Level"+str(level)+".png")
            pass
        if IMSHOW:
            plt.show()

    def synthetic(self):
        lvl = 3
        noise = [0]  # [5, 10, 15, 20]
        prob = [0.3, 0.5, 0.7, 0.9]
        sd = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        filenum = 5

        ratio_all = []
        ratio_points_all = []
        q_ratio_all = []
        q_ratio_points_all = []

        # for p in prob:
        # for n in noise:
        for p in prob:
            ratio_sd = []
            ratio_points_sd = []
            q_ratio_sd = []
            q_ratio_points_sd = []

            for d in sd:
                ratio = 0
                ratio_points = []
                q_ratio = 0
                q_ratio_points = []

                for f_num in range(1, filenum + 1):
                    if lvl == 1:
                        f_name = "synt_data_lvl" + str(lvl) + "_days30_sd" + str(d) + "_" + str(f_num)
                    elif lvl == 2:
                        f_name = "synt_data_lvl" + str(lvl) + "_days30_sd" + str(d) + "_noise" + str(n) + "_" + str(
                            f_num)
                    else:
                        f_name = "synt_data_lvl" + str(lvl) + "_days30_sd" + str(d) + "_prob" + str(p) + "_" + str(
                            f_num)

                    filepath = "../data/synthetic_data/level" + str(lvl) + "/parsed_data/" + f_name + ".csv"
                    print(filepath)

                    sequence = self.getSyntheticSequence(filepath)
                    r1, r1p, r2, r2p = self.getSequenceMatchingRatio(sequence)
                    ratio += r1
                    ratio_points = ratio_points + r1p
                    q_ratio += r2
                    q_ratio_points = q_ratio_points + r2p

                ratio = ratio/filenum
                q_ratio = q_ratio/filenum
                ratio_sd.append(ratio)
                ratio_points_sd.append(ratio_points)
                q_ratio_sd.append(q_ratio)
                q_ratio_points_sd.append(q_ratio_points)

            ratio_all.append(ratio_sd)
            ratio_points_all.append(ratio_points_sd)
            q_ratio_all.append(q_ratio_sd)
            q_ratio_points_all.append(q_ratio_points_sd)

        if lvl == 1:
            self.plotGraphs(sd, ratio_all[0], ratio_points_all[0], q_ratio_all[0], q_ratio_points_all[0], lvl)
        elif lvl == 2:
            self.plotGraphs3(sd, ratio_all, q_ratio_all, noise)
        else:
            self.plotGraphs3(sd, ratio_all, q_ratio_all, prob)

    def real(self):
        # subject_id = 1
        ratio_sub = []
        ratio_points_sub = []
        q_ratio_sub = []
        q_ratio_points_sub = []
        day_sub = []
        subject = [1, 2]
        for subject_id in subject:
            base_dir = "../data/real_data/Subject_"+str(subject_id)
            print("[ReadData] readFiles: Preparing data for subject ", subject_id, ".....")
            img = []
            print("[ReadData] readFiles: reading log files of data")

            log_files = glob.glob(base_dir + "/*.log")

            if len(log_files) == 0:
                print("[ReadData] readFiles: Error - No log files found in the folder ", base_dir)
                exit(-1)
            log_files.sort()

            num_days = len(log_files)

            try:
                for file in log_files[:num_days]:
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
                        # print(sample)
                        time_split = re.split(' |:', sample[1])
                        minute = int(time_split[1]) * 60 + int(time_split[2])
                        # print(minute)
                        if prev_min != minute or sample_num == len(data) - 1:
                            routine[prev_min] = max(container, key=container.get)
                            prev_min = minute
                            container = dict()

                        container[sample[-2]] = container.get(sample[-2], 0) + 1
                        # print(container)

                    # check for missing cells
                    for c in range(1440):
                        if routine[c] is None:
                            routine[c] = routine[c - 1]

                    # print(routine)
                    img.append(routine)

            except IOError as err:
                print("[ReadData] read_files: error reading log files. Error ", err)
                raise
            except Exception as err:
                print("[ReadData] read_files: Error ", err)
                raise

            filtered_data = img.copy()  # data.copy()
            # for day in data:
            #     filtered = median_filtering(day, window_size=6)
            #     # if scale_down != scale:
            #     #     filtered = scale_data(filtered, int(scale_down / scale))
            #     filtered_data.append(filtered)

            start_day = 0
            end_day = 14
            ratio = []
            ratio_points = []
            q_ratio = []
            q_ratio_points = []
            day = []
            while end_day <= len(filtered_data):
                routine14days = filtered_data[start_day:end_day]
                # print(routine14days)
                r1, r1p, r2, r2p = self.getSequenceMatchingRatio(routine14days)
                ratio.append(r1)
                ratio_points.append(r1p)
                q_ratio.append(r2)
                q_ratio_points.append(r2p)
                day.append(start_day)

                start_day += 7
                end_day += 7

            ratio_sub.append(ratio)
            ratio_points_sub.append(ratio_points)
            q_ratio_sub.append(q_ratio)
            q_ratio_points_sub.append(q_ratio_points)
            day_sub.append(day)

        self.plotGraphs3(day_sub, ratio_sub, q_ratio_sub, subject, x_label="Start Day")


if __name__ == '__main__':
    # ppobj = PoincarePlot()
    # ppobj.main()
    # ppobj.plotPairs()
    # ppobj.getActivityTableLevel3("../data/synthetic_data/level3/parsed_data/synt_data_lvl3_days30_sd5_prob0.7_4.csv")
    smobj = SequenceMatching()
    smobj.synthetic()
