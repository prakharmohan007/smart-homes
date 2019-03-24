import csv
import re
import matplotlib.pyplot as plt

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
roomclrcode['A'] = 'b'
roomclrcode['B'] = 'g'
roomclrcode['C'] = 'r'
roomclrcode['D'] = 'c'
roomclrcode['F'] = 'm'
roomclrcode['G'] = 'c--'
roomclrcode['H'] = 'b-'
roomclrcode['K'] = 'y'
roomclrcode['L'] = 'm--'
roomclrcode['M'] = 'c:'
roomclrcode['N'] = 'k'
roomclrcode['R'] = 'y--'
roomclrcode['S'] = 'y-,'
roomclrcode['T'] = 'y:'
roomclrcode['Y'] = 'k:'

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


def plotRoutine_TimeSpace(every_day_record):
    xaxis = list(range(24*60*60 + 1))
    for idx in every_day_record:
        day = every_day_record[idx]
        fig = plt.figure(figsize=(13.0, 8.0))
        labellist = []
        for record in day:
            # print (record)
            starttime = re.split(' |-|:', record[2])
            startsec = (int)(starttime[3])*60*60 + (int)(starttime[4])*60 + (int)(starttime[5])
            endsec = startsec+(int)(record[3])
            # print("startsec: ", startsec, ", endsec: ", endsec)
            # yaxis[startsec:endsec+1] = [roomidx[record[1]]]*(endsec+1-startsec)
            yaxis = [roomidx[record[1]]]*(endsec - startsec)
            # print(len(yaxis), len(xaxis[startsec:endsec]))
            plt.plot(xaxis[startsec:endsec], yaxis, roomclrcode[record[1]],
                    label = roomnames[record[1]] if roomclrcode[record[1]] not in labellist else '')
            if roomclrcode[record[1]] not in labellist:
                labellist.append(roomclrcode[record[1]])
    	
        plt.ylabel("Space")
        plt.xlabel("time in sec")
        title = "Routine: Room vs Duration for date: "+day[0][2].split()[0]
        plt.title(title, y = -0.1)
    	# plt.legend(bbox_to_anchor=(1.42, 1), loc='upper right', borderaxespad=0.)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                ncol=5, mode="expand", borderaxespad=0.)
        # plt.show()
        if SAVEIMG:
            imagename = "graphs/room-duration/"+day[0][2].split()[0]+".png"
            fig.savefig(imagename)

def plotRoutine_TimeSpace1(every_day_record):
    day = every_day_record[3]
    xaxis = list(range(24*60*60 + 1))
    # yaxis = [0]*(24*60*60)
    fig = plt.figure()
    labellist = []
    for record in day:
        # print (record)
        starttime = re.split(' |-|:', record[2])
        startsec = (int)(starttime[3])*60*60 + (int)(starttime[4])*60 + (int)(starttime[5])
        endsec = startsec+(int)(record[3])
        print("startsec: ", startsec, ", endsec: ", endsec)
        # yaxis[startsec:endsec+1] = [roomidx[record[1]]]*(endsec+1-startsec)
        yaxis = [roomidx[record[1]]]*(endsec - startsec)
        # print(len(yaxis), len(xaxis[startsec:endsec]))
        plt.plot(xaxis[startsec:endsec], yaxis, roomclrcode[record[1]],
                label = roomnames[record[1]] if roomclrcode[record[1]] not in labellist else '')
        if roomclrcode[record[1]] not in labellist:
            labellist.append(roomclrcode[record[1]])
    	
    plt.ylabel("Space")
    plt.xlabel("time in sec")
    title = "Routine: Room vs Duration for date: "+day[1][2].split()[0]
    plt.title(title, y = -0.1)
    # plt.legend(bbox_to_anchor=(1.42, 1), loc='upper right', borderaxespad=0.)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
            ncol=5, mode="expand", borderaxespad=0.)
    plt.show()
    if SAVEIMG:
        imagename = "graphs/room-duration/"+day[1][2].split()[0]+".png"
        fig.savefig(imagename)


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
    # for row in every_day_record[1]:
    print(every_day_record[1][1])
    plotRoutine_TimeSpace(every_day_record)
