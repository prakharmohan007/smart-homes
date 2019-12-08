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
    time_str = str(hh).zfill(2) + ":" + str(mm).zfill(2)
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

    print("[writeFile]", basepath + "/" + filename)

    f = open(basepath + "/" + filename, 'w')
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
            "activity_type": "sleep",
            "loc": [[('R1', "room")], [('R2', "room")], [('R3', "room")]],
            "noise": (3, 8)
        },
        {
            "stime": "07:00:01",
            "etime": "07:30:00",
            "duration": 1800,
            "activity": "hygiene1",
            "activity_type": "hygiene",
            "loc": [[('T1', "toilet")], [('T2', "toilet")]],
            "noise": (6.5, 5)
        },
        {
            "stime": "07:30:01",
            "etime": "09:00:00",
            "duration": 5400,
            "activity": "cookeat1",
            "activity_type": "cookeat",
            "loc": [[('K', "kitchen"), ('L', "living")]],
            "noise": (15, 5)
        },
        {
            "stime": "09:00:01",
            "etime": "12:00:00",
            "duration": 10800,
            "activity": "entertainment1",
            "activity_type": "entertainment",
            "loc": [[('L', "living")]],
            "noise": (15, 8)
        },
        {
            "stime": "12:00:01",
            "etime": "13:30:00",
            "duration": 5400,
            "activity": "cookeat2",
            "activity_type": "cookeat",
            "loc": [[('K', "kitchen"), ('L', "living")]],
            "noise": (15, 8)
        },
        {
            "stime": "13:30:01",
            "etime": "16:30:00",
            "duration": 10800,
            "activity": "sleeping2",
            "activity_type": "sleep",
            "loc": [[('R1', "room")], [('R2', "room")], [('R3', "room")]],
            "noise": (3, 8)
        },
        {
            "stime": "16:30:01",
            "etime": "17:00:00",
            "duration": 1800,
            "activity": "hygiene2",
            "activity_type": "hygiene",
            "loc": [[('T1', "toilet")], [('T2', "toilet")]],
            "noise": (6.5, 5)
        },
        {
            "stime": "17:00:01",
            "etime": "19:00:00",
            "duration": 7200,
            "activity": "outside",
            "activity_type": "outside",
            "loc": [[('O', "out")]],
            "noise": (3, 8)
        },
        {
            "stime": "19:00:01",
            "etime": "20:30:00",
            "duration": 5400,
            "activity": "cookeat3",
            "activity_type": "cookeat",
            "loc": [[('K', "kitchen"), ('L', "living")]],
            "noise": (15, 5)
        },
        {
            "stime": "20:30:01",
            "etime": "23:00:00",
            "duration": 9000,
            "activity": "entertainment2",
            "activity_type": "entertainment",
            "loc": [[('L', "living")]],
            "noise": (15, 8)
        },
        {
            "stime": "23:00:01",
            "etime": "23:59:59",
            "duration": 3599,
            "activity": "sleeping3",
            "activity_type": "sleep",
            "loc": [[('R1', "room")], [('R2', "room")], [('R3', "room")]],
            "noise": (3, 8)
        },
        {
            "stime": "__:__:__",
            "etime": "__:__:__",
            "duration": 480,
            "activity": "noise",
            "type": "noise",
            "loc": {('R1', "room"), ('R2', "room"), ('R3', "room"),
                    ('T1', "toilet"), ('T2', "toilet"), ('L', "living"),
                    ('K', "kitchen"), ('O', "out"), ('E', "entry")}
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
        prev = 0
        while dur < duration:
            d = abs(int(random.gauss(5, 2)))
            if d + dur > duration:
                d = duration - dur
            # print(d)
            room = random.choice(rooms)
            # if len(rooms) > 1:
            #     while prev == room:
            #         room = random.choice(rooms)

            sequence = sequence + [room] * d
            dur = dur + d
            prev = room
        return sequence

    def genLevel1(self, day, controlled=False, sdp=10):
        synt_routine = []
        prev_end_time = -1
        is_full = False
        for sample in self.base_routine[:-2]:
            # STEP1: GET START TIME
            start_sec = prev_end_time + 1

            # STEP2: GET RANDOM DURATION for the ACTIVITY
            duration = int(sample["duration"] / 60)
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

    @staticmethod
    def genNoiseSequence(sequence, noise_inst, avg_noise_dur, noise_set):
        for n in range(noise_inst):
            # generate noise duration
            noise_dur = 0
            while noise_dur == 0:
                noise_dur = abs(int(random.gauss(avg_noise_dur, 2)))
            if noise_dur >= len(sequence) / 2:
                continue

            # generate noise start
            startcell = random.randint(a=0, b=len(sequence) - noise_dur)

            # gen loc instances
            num_loc = random.randint(a=1, b=noise_dur)
            loc_ins = [1] * num_loc
            while num_loc < noise_dur:
                num_loc += 1
                loc_ins[random.randint(a=0, b=len(loc_ins) - 1)] += 1

            # select locations
            for n in loc_ins:
                act = random.choice(list(noise_set))
                for _ in range(n):
                    sequence[startcell] = act
                    startcell += 1

        return sequence

    def genBasicSequence(self, level, day, controlled=False, sdp=10, noise=20):
        synt_routine = []
        prev_end_time = -1
        is_full = False
        for sample in self.base_routine[:-2]:
            # STEP1: GET START TIME
            start_sec = prev_end_time + 1

            # STEP2: GET RANDOM DURATION for the ACTIVITY
            duration = int(sample["duration"] / 60)
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

            # STEP4.1: ADD NOISE
            if level == 2:
                # noise_dur = abs(int(random.gauss(sample["noise"][1], 2)))
                noise_inst = int(sample["noise"][0] * noise / 100 + 1)
                noise_set = self.base_routine[-1]["loc"].copy()
                for r in room_combination:
                    noise_set.remove(r)
                sequence = self.genNoiseSequence(sequence, noise_inst, sample["noise"][1], noise_set)

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

    def genLevel3Sequence(self, day, controlled=False, sdp=10, prob=0.7):
        # get a set/dict of activities
        act_set = dict()
        for act in self.base_routine[:-1]:
            if act["activity_type"] not in act_set:
                act_set[act["activity_type"]] = []
            act_set[act["activity_type"]].append(act)

        prev_act = None
        is_full = False
        synt_routine = []
        prev_end_time = -1

        for sample in self.base_routine[:-2]:
            success = False
            while not success and not is_full:
                noise_set = act_set.copy()
                del noise_set[sample["activity_type"]]
                if prev_act is not None:
                    del noise_set[prev_act]

                # print(noise_set.keys())

                if random.random() <= prob:
                    act_type = sample["activity_type"]
                    duration = int(sample["duration"] / 60)
                    room_combination = random.choice(sample["loc"])
                    activity = sample["activity"]
                    success = True
                else:
                    # randomly choose an activity
                    act_type = random.sample(noise_set.keys(), 1)[0]
                    # print(act_type)
                    activity_inst = random.choice(act_set[act_type])
                    duration = int(activity_inst["duration"] / 60)
                    room_combination = random.choice(activity_inst["loc"])
                    activity = activity_inst["activity"]

                start_sec = prev_end_time + 1

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

                # STEP4: GENERATE LOCATION SEQUENCE for the ACTIVITY
                sequence = self.genLocSequence(rand_duration, room_combination)

                d = 0
                while d < rand_duration:
                    time_min = minuteToString(start_sec + d)
                    synt_routine.append([str(day).zfill(2), time_min, sequence[d][0],
                                         sequence[d][1], activity])
                    d += 1

                prev_end_time = start_sec + rand_duration - 1
                prev_act = act_type

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
                # one_day_routine = self.genLevel1(day, controlled, sdp)
                one_day_routine = self.genBasicSequence(level=1, day=day,
                                                        controlled=controlled,
                                                        sdp=sdp)
                synt_routine = synt_routine + one_day_routine
            filename = "newsynt_level1_sd" + str(sdp) + "_" + str(f_num) + ".csv"
            writeFile(basepath, filename, synt_routine)
            # print(len(synt_routine))

    def level_2(self, basepath, controlled=False, sdp=10, noise=20):
        if controlled:
            print("[SyntheticData] Level_2: Generating level2 data for ", self.num_files, "files, ", self.num_days,
                  " days and controlled SD of " + str(sdp) + "% in each file.")
        else:
            print("[SyntheticData] Level_2: Generating level2 data for ", self.num_files, "files, ", self.num_days,
                  " days in each file.")

        basepath = basepath + "/level2"

        print("sd:", sdp, ", noise:", noise)

        for f_num in range(1, self.num_files + 1):
            synt_routine = []
            for day in range(1, self.num_days + 1):
                # one_day_routine = self.genLevel1(day, controlled, sdp)
                one_day_routine = self.genBasicSequence(level=2, day=day,
                                                        controlled=controlled,
                                                        sdp=sdp, noise=noise)
                synt_routine = synt_routine + one_day_routine
            filename = "newsynt_level2_sd" + str(sdp) + "_noise" + str(noise) + "_" + str(f_num) + ".csv"
            writeFile(basepath, filename, synt_routine)
            # print(len(synt_routine))

    def level_3(self, basepath, controlled=False, sdp=10, prob=0.7):
        if controlled:
            print("[SyntheticData] Level_3: Generating level3 data for ", self.num_files, "files, ", self.num_days,
                  " days and controlled SD of " + str(sdp) + "% in each file.")
        else:
            print("[SyntheticData] Level_3: Generating level3 data for ", self.num_files, "files, ", self.num_days,
                  " days in each file.")

        basepath = basepath + "/level3"

        print("sd:", sdp, ", noise:", prob)

        for f_num in range(1, self.num_files + 1):
            synt_routine = []
            for day in range(1, self.num_days + 1):
                # one_day_routine = self.genLevel1(day, controlled, sdp)
                one_day_routine = self.genLevel3Sequence(day=day,
                                                         controlled=controlled,
                                                         sdp=sdp, prob=prob)
                synt_routine = synt_routine + one_day_routine
            filename = "newsynt_level3_sd" + str(sdp) + "_prob" + str(prob) + "_" + str(f_num) + ".csv"
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
    obj = SyntheticData(num_files=10)
    # seq = obj.genLocSequence(100, [('K', "kitchen"), ('L', "living")])
    # level1
    # for sd in [5, 10, 15, 20, 25, 30]:
    #     obj.level_1("../../data/synthetic_data/new_synthetic_data", True, sd)

    # LEVEL2
    # for sd in [5, 10, 15, 20, 25, 30]:
    #     for noise in [20, 30, 40]:
    #         obj.level_2("../../data/synthetic_data/new_synthetic_data", True, sd, noise)

    # LEVEL3
    for sd in [5, 10, 15, 20, 25, 30]:
        for prob in [0.3, 0.5, 0.7, 0.9]:
            obj.level_3("../../data/synthetic_data/new_synthetic_data", True, sd, prob)


    # temporaryFunction()
    # seq = obj.genLevel3Sequence(1, controlled=True, sdp=5, prob=0.9)
    # for s in seq:
    #     print(s)
