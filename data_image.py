import csv
import cv2
import re
import numpy as np

SAVEIMG = 1

# b: Bathroom1[H], Bathroom2[A]
# g: Bed (room3)[B]
# r: toilet2[C]
# c: room1[G], room2[M], room3[D] 
# m: sofa[F], Living room[L]
# y: Kitchen[K], refrigerator[R], Sink[S], Dining Table[T]
# k: Out[N]
# w: Balcony[Y]

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

roomclrcode = {}
roomclrcode['A'] = [255, 0, 0]  # blue
roomclrcode['B'] = [0, 255, 0]  # green
roomclrcode['C'] = [0, 0, 255]  # red
roomclrcode['D'] = [255, 255, 0]
roomclrcode['F'] = [0, 255, 255]
roomclrcode['G'] = [255, 0, 255]
roomclrcode['H'] = [100, 0, 0]
roomclrcode['K'] = [0, 100, 0]
roomclrcode['L'] = [0, 0, 100]
roomclrcode['M'] = [0, 100, 100]
roomclrcode['N'] = [100, 100, 0]
roomclrcode['R'] = [100, 0, 100]
roomclrcode['S'] = [75, 75, 75]
roomclrcode['T'] = [150, 150, 150]
roomclrcode['Y'] = [255, 255, 255]

roomnames = {}
roomnames['A'] = "Bathroom 2"
roomnames['B'] = "Bed (Room 3)"
roomnames['C'] = "Toilet 2"
roomnames['D'] = "Room 3"
roomnames['F'] = "Sofa (Living room)"
roomnames['G'] = "Room 1"
roomnames['H'] = "Bathroom 1"
roomnames['K'] = "Kitchen"
roomnames['L'] = "Living room"
roomnames['M'] = "Room 2"
roomnames['N'] = "OUT"
roomnames['R'] = "Refrigerator (Kitchen)"
roomnames['S'] = "Sink (Kitchen)"
roomnames['T'] = "Dining table (Kitchen)"
roomnames['Y'] = "Balcony"


def createImage(every_day_record):
    num_days = len(every_day_record)
    cols = (int)(24*60*60/10)
    rows = num_days * 50
    size = rows, cols, 3
    img = np.zeros(size, dtype=np.uint8)
    i = 0
    for idx in every_day_record:
        day = every_day_record[idx]
        i += 1
        for record in day:
            # print (record)
            starttime = re.split(' |-|:', record[2])
            startsec = (int)(starttime[3])*60*60 + (int)(starttime[4])*60 + (int)(starttime[5])
            startsec = (int)(startsec/10)
            # startmin = (int)(starttime[3])*60 + (int)(starttime[4])
            endsec = startsec + (int)(record[3]) / 10
            endsec = (int)(endsec)
            # endmin = startmin + (int)((int)(record[3])/60)
            # for c in range((i-1)*50, i*50+1):
            #     for r in range(startmin, endmin):
            #         img[c,r] = roomclrcode[record[1]]
            # img[(i-1)*50:, startmin:endmin] = roomclrcode[record[1]]
            img[(i-1)*50:i*50-2, startsec:endsec] = roomclrcode[record[1]]

    cv2.namedWindow("img",flags = cv2.WINDOW_NORMAL)
    cv2.imshow("img", img)
    cv2.waitKey(0)


def loadCSV(filepath):
    with open(filepath) as csvfile:
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
    return data_list


def divideDataInDays_UsingList(data):
    day = 0
    recordnum = 0
    every_day_record = {}
    for record in data:
        # print(record[2])
        starttime = re.split(' |-|:', record[2])
        # print(starttime)
        if day != (int)(starttime[2]):
            day = (int)(starttime[2])
            recordnum += 1
            every_day_record[recordnum] = []
        every_day_record[recordnum].append(record)
    return every_day_record


if __name__ == '__main__':
    print("Loading csv file")
    datalist = loadCSV("data/180724_180810(mod).csv")
    print("CSV loaded, dividing data into days")
    every_day_record = divideDataInDays_UsingList(datalist)
    print("number of days for which data is collected: ", len(every_day_record))
    print("sample day record for date 24th July:")
    createImage(every_day_record)
