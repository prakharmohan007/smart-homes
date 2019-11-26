import random
from pathlib import Path
import itertools
import operator
import csv


def str_time_to_sec(str_time):
    time_split = list(map(int, str_time.split(sep=':')))
    sec = time_split[0] * 60 * 60 + time_split[1] * 60 + time_split[2]
    return sec


def str_time_to_min(str_time):
    time_split = list(map(int, str_time.split(sep=':')))
    minute = time_split[0] * 60 + time_split[1]
    return minute


def minuteToString(minute):
    mm = int(minute % 60)
    hh = int(minute / 60)
    time_str = str(hh).zfill(2) + ":"  + str(mm).zfill(2)
    return time_str


def sec_to_time(sec):
    ss = int(sec % 60.0)
    sec = int(sec / 60)
    mm = int(sec % 60)
    hh = int(sec / 60)

    time_str = str(hh).zfill(2) + ":" + str(mm).zfill(2) + ":" + str(ss).zfill(2)
    time_split = (hh, mm, ss)
    return time_split, time_str


def writeFile(basepath, filename, file):

    try:
        path = Path(basepath)
        path.mkdir(parents=True, exist_ok=True)
    except FileExistsError as err:
        print("[SyntheticData] write_files: Path already exist. Warning: ", err)
    except FileNotFoundError as err:
        print("[SyntheticData] write_files: Parent folders already exist. Warning: ", err)
    except Exception as err:
        print("[SyntheticData] write_files: Error with folder ", basepath, ", Error: ", err)
        raise

    print("[writeFile]", basepath+"/"+filename)

    f = open(basepath+"/"+filename, 'w')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["Day", "Time", "Room-ID", "Room-Type", "Activity"])
    for sample in file:
        csv_writer.writerow(sample)
    f.close()


def newSyntheticRoutine():
    base_routine = [
        {
            "stime": "00:00:00",
            "etime": "07:00:00",
            "duration": 25201,
            "activity": "sleeping1",
            "loc": [[('R1', "room")], [('R2', "room")], [('R3', "room")]],
            "noise":3
        },
        {
            "stime": "07:00:01",
            "etime": "07:30:00",
            "duration": 1800,
            "activity": "hygiene1",
            "loc": [[('T1', "toilet")], [('T2', "toilet")]],
            "noise": 8
        },
        {
            "stime": "07:30:01",
            "etime": "09:00:00",
            "duration": 5400,
            "activity": "cookeat1",
            "loc": [[('K', "kitchen"), ('L', "living")]],
            "noise": 14
        },
        {
            "stime": "09:00:01",
            "etime": "12:00:00",
            "duration": 10800,
            "activity": "entertainment1",
            "loc": [[('L', "living")]],
            "noise": 14
        },
        {
            "stime": "12:00:01",
            "etime": "13:30:00",
            "duration": 5400,
            "activity": "cookeat2",
            "loc": [[('K', "kitchen"), ('L', "living")]],
            "noise": 14
        },
        {
            "stime": "13:30:01",
            "etime": "16:30:00",
            "duration": 10800,
            "activity": "sleeping2",
            "loc": [[('R1', "room")], [('R2', "room")], [('R3', "room")]],
            "noise": 3
        },
        {
            "stime": "16:30:01",
            "etime": "17:00:00",
            "duration": 1800,
            "activity": "hygiene2",
            "loc": [[('T1', "toilet")], [('T2', "toilet")]],
            "noise": 8
        },
        {
            "stime": "17:00:01",
            "etime": "19:00:00",
            "duration": 7200,
            "activity": "outside",
            "loc": [[('O', "out")]],
            "noise": 3
        },
        {
            "stime": "19:00:01",
            "etime": "20:30:00",
            "duration": 5400,
            "activity": "cookeat3",
            "loc": [[('K', "kitchen"), ('L', "living")]],
            "noise": 14
        },
        {
            "stime": "20:30:01",
            "etime": "23:00:00",
            "duration": 9000,
            "activity": "entertainment2",
            "loc": [[('L', "living")]],
            "noise": 14
        },
        {
            "stime": "23:00:01",
            "etime": "23:59:59",
            "duration": 3599,
            "activity": "sleeping3",
            "loc": [[('R1', "room")], [('R2', "room")], [('R3', "room")]],
            "noise": 3
        },
        {
            "stime": "__:__:__",
            "etime": "__:__:__",
            "duration": 480,
            "activity": "noise",
            "type": "noise",
            "loc": [('R1', "room"), ('R2', "room"), ('R3', "room"),
                    ('T1', "toilet"), ('T2', "toilet"), ('L', "living"),
                    ('K', "kitchen"), ('O', "out"), ('E', "entry")]
        }
    ]
    return base_routine


class SyntheticData:
    def __init__(self, scale=60, num_days=30, num_files=5, base_dir="../data/synthetic_data/"):

        # general required parameters, level1 parameters
        self.scale = scale
        self.num_days = num_days
        self.num_files = num_files
        self.base_dir = base_dir
        self.sd = [10, 20, 30]  # represent % of standard deviation
        self.base_routine = newSyntheticRoutine()

    def genLocSequence(self, duration, rooms):
        sequence = []
        dur = 0
        while dur < duration:
            d = abs(int(random.gauss(5, 2.5)))
            if d + dur > duration:
                d = duration - dur
            # print(d)
            room = random.choice(rooms)
            sequence = sequence + [room]*d
            dur = dur + d
        return sequence

    def genLevel1(self, day, controlled=False, sdp=10):
        synt_routine = []
        prev_end_time = -1
        is_full = False
        for sample in self.base_routine[:-2]:
            # STEP1: GET START TIME
            start_sec = prev_end_time + 1

            # STEP2: GET RANDOM DURATION for the ACTIVITY
            duration = int(sample["duration"]/60)
            if controlled:
                sd = int(duration * sdp / 100.0)
            else:
                # randomly select Standard Deviation
                sd_idx = random.randint(0, 2)  # 0->10%, 1->20%, 2->30%
                # calc standard deviation as a percent of duration
                sd = int(duration * self.sd[sd_idx] / 100.0)

            rand_duration = abs(int(random.gauss(duration, sd)))
            if rand_duration < 0:
                print(rand_duration)

            # check if activity crosses 24 hrs
            if start_sec + rand_duration > 24 * 60:
                rand_duration = 24 * 60 - start_sec
                is_full = True

            # STEP3: RANDOMLY SELECT LOCATION COMBINATION
            room_combination = random.choice(sample["loc"])

            # STEP4: GENERATE LOCATION SEQUENCE for the ACTIVITY
            sequence = self.genLocSequence(rand_duration, room_combination)

            d = 0
            while d < rand_duration:
                time_min = minuteToString(start_sec + d)
                synt_routine.append([str(day).zfill(2), time_min, sequence[d][0],
                                     sequence[d][1], sample["activity"]])
                d += 1

            prev_end_time = start_sec + rand_duration - 1
            if is_full:
                break

        # if day is still left, add the last routine activity
        if prev_end_time + 1 < 24 * 60:
            start_sec = prev_end_time + 1
            rand_duration = 24 * 60 - start_sec

            # STEP3: RANDOMLY SELECT LOCATION COMBINATION
            room_combination = random.choice(self.base_routine[-2]["loc"])

            # STEP4: GENERATE LOCATION SEQUENCE for the ACTIVITY
            sequence = self.genLocSequence(rand_duration, room_combination)

            d = 0
            while d < rand_duration:
                time_min = minuteToString(start_sec + d)
                synt_routine.append([str(day).zfill(2), time_min, sequence[d][0],
                                     sequence[d][1], self.base_routine[-2]["activity"]])
                d += 1

        return synt_routine

    def level_1(self, basepath, controlled=False, sdp=10):
        if controlled:
            print("[SyntheticData] Level_1: Generating level1 data for ", self.num_files, "files, ", self.num_days,
                  " days and controlled SD of " + str(sdp) + "% in each file.")
        else:
            print("[SyntheticData] Level_1: Generating level1 data for ", self.num_files, "files, ", self.num_days,
                  " days in each file.")

        basepath = basepath + "/level1"

        for f_num in range(1, self.num_files + 1):
            synt_routine = []
            for day in range(1, self.num_days + 1):
                one_day_routine = self.genLevel1(day, controlled, sdp)
                synt_routine = synt_routine + one_day_routine
            filename = "newsynt_level1_sd" + str(sdp) + "_" + str(f_num) + ".csv"
            writeFile(basepath, filename, synt_routine)
            # print(len(synt_routine))


def temporaryFunction():
    f = open("../../data/real_data/Subject_2/xandem_2018-11-28_v2_location.log", 'r')
    prev_minute = "11/28/2018 0:0"
    first_line = True
    dictionary = dict()
    for line in f.readlines():
        if first_line:
            first_line = False
            continue
        fields = line.splitlines()[0].split(sep=',')
        if fields[1] != prev_minute:
            print(prev_minute, max(dictionary.items(), key=operator.itemgetter(1))[0])
            dictionary = dict()
            prev_minute = fields[1]
        dictionary[fields[-1]] = dictionary.get(fields[-1], 0) + 1


if __name__ == "__main__":
    # obj = SyntheticData()
    # seq = obj.genLocSequence(100, [('K', "kitchen"), ('L', "living")])
    # obj.level_1("../../data/synthetic_data/new_synthetic_data", True, 20)
    temporaryFunction()