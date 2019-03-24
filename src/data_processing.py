import numpy as np
import csv
import re

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
roomidx['N'] = 11
roomidx['R'] = 12
roomidx['S'] = 13
roomidx['T'] = 14
roomidx['Y'] = 15

# Class for processing data
class DataProcessing:
    
    def __init__ (self, csv_path, num_days, time_interval):
        self.datafilepath = csv_path
        self.unprocessed_data_dict, self.unprocessed_data_list = self.loadCSV()
        self.num_days = num_days
        self.time_interval = time_interval
        self.daywisedata = {}

    def loadCSV(self):
        with open(self.datafilepath) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            data_list = []
            data_dict = []

            flag = False
            for row in readCSV:
                if flag == False:
                    flag = True
                    continue
                dictionary = {}
                dictionary['space'] = row[1]
                dictionary['start time'] = row[2]
                dictionary['duration'] = row[3]
                dictionary['ms_avg'] = row[4]
                dictionary['ms_sd'] = row[5]

                data_list.append(row)
                data_dict.append(dictionary)
                # print (row[2])

        print(data_list[1])
        print(data_dict[1])
        return data_dict, data_list

    def dataByDays_UsingList(self):
        day = 0
        daynum = 0

        for record in data:
            # split time
            # print(record[2])
            starttime = re.split(' |-|:', record[2])
            # print(starttime)

            if day != (int)(starttime[2]):
                day = (int)(starttime[2])
                daynum += 1
                self.daywisedata[daynum] = []
            self.daywisedata[daynum].append(record)

    # feature vector structure: [curr_activity, time(Sec), duration(sec), prev_activities*15]
    # total dimentions: 1+1+1+15 = 18
    def buildFeatures(self, prev_act_feat_vec, curr_act):
        feat_vec = np.array([0]*18)
        feat_vec[0] = curr_act.space
        feat_vec[1] = curr_act.time(sec)
        feat_vec[2] = curr_act.duration(Sec)
        feat_vec[3:18] = prev_act_feat_vec[3:18]
        feat_vec[roomidx[prev_act_feat_vec[0]]+2] += 1
        return feat_vec

    # Y axis: number of days (num_days)
    # X axis: time: 5 sec interval (24*60*60/time_interval)
    def makeDataMatrix(self):
        ndays = self.num_days
        ntime = 24*60*60/self.time_interval


if __name__ == '__main__':
    dataobj = DataProcessing("../data/180724_180810_mod.csv")
    exit(1)