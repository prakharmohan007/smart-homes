# Author: Prakhar Mohan
# Title: Preparing Synthetic Routine of basic ADL
import csv
import random
from pathlib import Path
import itertools


class SyntheticData:
    def __init__(self, scale=30, num_days=30, num_files=5, max_noise_activity=5, base_dir="../data/synthetic_data/"):

        # general required parameters, level1 parameters
        self.scale = scale
        self.num_days = num_days
        self.num_files = num_files
        self.base_dir = base_dir
        self.sd = [10, 20, 30]  # represent % of standard deviation

        # level2/level3 required parameters, toilet activity noise
        self.max_noise_event = max_noise_activity

    @staticmethod
    def read_general_routine(file_name="../data/synthetic_data/synthetic_routine.csv"):
        try:
            with open(file_name, 'r') as csv_reader:
                data = csv_reader.readlines()
            del data[0]
        except IOError as err:
            print("Error reading the csv file. Error: ", err)
            raise
        except Exception as err:
            print("Error: ", err)
            raise

        # num_act = len(data) - 1  # last activity is toilet, to be introduced as noise in lvl2 and lvl3
        routine = []
        for sample in data:
            split_sample = sample.splitlines()[0].split(sep=', ')
            routine.append(split_sample)

        return routine

    @staticmethod
    def str_time_to_sec(str_time):
        time_split = list(map(int, str_time.split(sep=':')))
        sec = time_split[0] * 60 * 60 + time_split[1] * 60 + time_split[2]
        return sec

    @staticmethod
    def sec_to_time(sec):
        ss = int(sec % 60.0)
        sec = int(sec / 60)
        mm = int(sec % 60)
        hh = int(sec / 60)

        time_str = str(hh).zfill(2) + ":" + str(mm).zfill(2) + ":" + str(ss).zfill(2)
        time_split = (hh, mm, ss)
        return time_split, time_str

    @staticmethod
    def unpack_routine(routine, scale):
        unpacked_routine = []
        sec = 0
        for sample in routine:
            end_time = SyntheticData.str_time_to_sec(sample[1]) + int(sample[2])

            if sec >= 24 * 60 * 60:
                sec = 0

            while sec <= end_time:
                time_split, time_str = SyntheticData.sec_to_time(sec)
                unpacked_routine.append([sample[0], time_str, sample[3], sample[4]])
                sec = sec + scale
        return unpacked_routine

    @staticmethod
    def pack_routine(routine, scale):
        packed_routine = []
        count = 0
        prev_space = None
        prev_act = None
        stime = None

        for sample in range(len(routine)):
            if prev_space is None:
                stime = routine[sample][1]
                prev_space = routine[sample][-1]
                prev_act = routine[sample][-2]

            if routine[sample][-1] != prev_space or sample == len(routine) - 1:
                duration = count * scale
                packed_routine.append([routine[sample][0], stime, str(duration), prev_act, prev_space])
                prev_space = routine[sample][-1]
                prev_act = routine[sample][-2]
                stime = routine[sample][1]
                count = 0
            count = count + 1

        return packed_routine

    def write_files(self, routine, level=1, file_num=1, controlled=False, sdp=10, prob=None):
        dir_path = self.base_dir + "level" + str(level)
        parsed_file_path = dir_path + "/parsed_data"
        raw_file_path = dir_path + "/raw_data"

        # make raw file directory
        try:
            path = Path(raw_file_path)
            path.mkdir(parents=True, exist_ok=True)
        except FileExistsError as err:
            print("[SyntheticData] write_files: Path already exist. Warning: ", err)
        except FileNotFoundError as err:
            print("[SyntheticData] write_files: Parent folders already exist. Warning: ", err)
        except Exception as err:
            print("[SyntheticData] write_files: Error with folder ", raw_file_path, ", Error: ", err)
            raise

        # make parsed file directory
        try:
            path = Path(parsed_file_path)
            path.mkdir(parents=True, exist_ok=True)
        except FileExistsError as err:
            print("[SyntheticData] write_files: Path already exist. Warning: ", err)
        except FileNotFoundError as err:
            print("[SyntheticData] write_files: Parent folders already exist. Warning: ", err)
        except Exception as err:
            print("[SyntheticData] write_files: Error with folder ", parsed_file_path, ", Error: ", err)
            raise

        if prob is None:
            prob_str = ""
        else:
            prob_str = "_prob" + str(prob)

        if controlled:
            file_name = "synt_data_lvl" + str(level) + "_days" + str(self.num_days) + "_sd" + str(
                sdp) + prob_str + "_" + str(file_num) + ".csv"
        else:
            file_name = "synt_data_lvl" + str(level) + "_days" + str(self.num_days) + prob_str + "_" + str(
                file_num) + ".csv"

        f1 = open(raw_file_path + "/" + file_name, 'w')
        raw_writer = csv.writer(f1)
        raw_writer.writerow(["Day", "Time Stamp", "Activity", "Location"])

        f2 = open(parsed_file_path + "/" + file_name, 'w')
        parsed_writer = csv.writer(f2)
        parsed_writer.writerow(["Day", "Start Time", "Duration", "Activity", "Location"])

        sec = 0
        for sample in routine:
            parsed_writer.writerow(sample)
            end_time = self.str_time_to_sec(sample[1]) + int(sample[2])

            if sec > 24 * 60 * 60:
                sec = 0

            while sec <= end_time:
                time_split, time_str = self.sec_to_time(sec)
                raw_writer.writerow([sample[0], time_str, sample[3], sample[4]])
                sec = sec + self.scale

        f1.close()
        f2.close()

    def gen_level1(self, day, routine, controlled=False, sdp=10):
        synt_routine = []
        prev_end_time = 24 * 60 * 60
        is_full = False
        for sample in routine[:-2]:
            time_split = list(map(int, sample[0].split(sep=':')))
            start_sec = time_split[0] * 60 * 60 + time_split[1] * 60 + time_split[2]
            if start_sec != 0:
                start_sec = prev_end_time + 1

            duration = int(sample[2])
            if controlled:
                sd = int(duration * sdp / 100.0)
            else:
                # randomly select Standard Deviation
                sd_idx = random.randint(0, 2)  # 0->10%, 1->20%, 2->30%
                # calc standard deviation as a percent of duration
                sd = int(duration * self.sd[sd_idx] / 100.0)

            rand_duration = int(random.gauss(duration, sd))

            # check if activity crosses 24 hrs
            if start_sec + rand_duration > 24 * 60 * 60:
                rand_duration = 24 * 60 * 60 - start_sec
                is_full = True

            # convert sec to time string
            time_split, time_str = self.sec_to_time(start_sec)

            # add record in synt_routine
            synt_routine.append([str(day).zfill(2), time_str, str(rand_duration), sample[3], sample[4]])

            prev_end_time = start_sec + rand_duration
            if is_full:
                break

        # if day is still left, add the last routine activity
        if prev_end_time < 24 * 60 * 60:
            start_sec = prev_end_time + 1
            rand_duration = 24 * 60 * 60 - start_sec

            time_split, time_str = self.sec_to_time(start_sec)

            synt_routine.append([str(day).zfill(2), time_str, rand_duration, routine[-2][3], routine[-2][4]])

        return synt_routine

    def gen_level3(self, day, routine, prob=0.7, controlled=False, sdp=5, noise_num=0):

        # print(routine)
        loc_set = set()
        for act in routine[:-1]:
            loc_set.add(act[-1])
        # print(loc_set)

        prev_end_time = None
        # start_sec = 0
        is_full = False
        prev_loc = None
        synt_routine = []
        for sample in routine[:-2]:
            # time_split = list(map(int, sample[0].split(sep=':')))
            # start_sec = time_split[0] * 60 * 60 + time_split[1] * 60 + time_split[2]
            curr_act = sample[-1]

            dur_curr_act = int(sample[2])
            class2_set = loc_set.copy()
            class2_set.discard(curr_act)
            if prev_loc is not None:
                class2_set.discard(prev_loc)

            success = False
            while not success and not is_full:
                if random.random() <= prob:
                    # choose the actual activity
                    duration = dur_curr_act
                    loc = curr_act
                    activity = sample[3]
                    success = True
                else:
                    # choose random activity
                    duration = 2700
                    loc = random.sample(class2_set, 1)[0]
                    activity = "nonroutine" + str(noise_num)
                    noise_num += 1

                # ADD activity in day routine
                if prev_end_time is None:
                    start_sec = 0
                else:
                    start_sec = prev_end_time + 1

                if controlled:
                    sd = int(duration * sdp / 100.0)
                else:
                    # randomly select Standard Deviation
                    sd_idx = random.randint(0, 2)  # 0->10%, 1->20%, 2->30%
                    # calc standard deviation as a percent of duration
                    sd = int(duration * self.sd[sd_idx] / 100.0)

                rand_duration = int(random.gauss(duration, sd))

                # check if activity crosses 24 hrs
                if start_sec + rand_duration > 24 * 60 * 60:
                    rand_duration = 24 * 60 * 60 - start_sec
                    is_full = True

                # convert sec to time string
                time_split, time_str = self.sec_to_time(start_sec)
                # add record in synt_routine
                synt_routine.append([str(day).zfill(2), time_str, str(rand_duration), activity, loc])

                prev_end_time = start_sec + rand_duration
                prev_loc = loc

        # if day is still left, add the last routine activity
        if prev_end_time < 24 * 60 * 60:
            start_sec = prev_end_time + 1
            rand_duration = 24 * 60 * 60 - start_sec

            time_split, time_str = self.sec_to_time(start_sec)

            synt_routine.append([str(day).zfill(2), time_str, rand_duration, routine[-2][3], routine[-2][4]])

        return synt_routine, noise_num

    def level_1(self, routine, controlled=False, sdp=10):
        if controlled:
            print("[SyntheticData] Level_1: Generating level1 data for ", self.num_files, "files, ", self.num_days,
                  " days and controlled SD of " + str(sdp) + "% in each file.")
        else:
            print("[SyntheticData] Level_1: Generating level1 data for ", self.num_files, "files, ", self.num_days,
                  " days in each file.")

        for f_num in range(1, self.num_files + 1):
            synt_routine = []
            for day in range(1, self.num_days + 1):
                one_day_routine = self.gen_level1(day, routine, controlled, sdp)
                synt_routine = synt_routine + one_day_routine
            self.write_files(routine=synt_routine, level=1, file_num=f_num, controlled=controlled, sdp=sdp)

    # level2 = level1 with randomly distributed toilet activity
    def level_2(self, routine, controlled=False, sdp=10):
        if controlled:
            print("[SyntheticData] Level_2: Generating level2 data for ", self.num_files, "files, ", self.num_days,
                  " days and controlled SD of " + str(sdp) + "% in each file.")
        else:
            print("[SyntheticData] Level_2: Generating level2 data for ", self.num_files, "files, ", self.num_days,
                  " days in each file.")

        for f_num in range(1, self.num_files + 1):
            synt_routine = []
            noise_num = 0
            for day in range(1, self.num_days + 1):
                one_day_routine = self.gen_level1(day, routine, controlled, sdp)
                # print(one_day_routine)

                # introduce noise
                # ..get a random number of instances of noise
                num_act = random.randint(a=1, b=self.max_noise_event)
                act_dur = int(routine[-1][2])  # last record in routine is the noise
                act_sd = int(act_dur / 10)  # setting standard deviation to 10%
                num_cells = int(24 * 60 * 60 / self.scale)

                # unpack one-day-routine to easy introduce the noise
                unpacked = self.unpack_routine(one_day_routine, self.scale)
                # print(unpacked)

                # ..for each instance
                for i in range(num_act):
                    # ..get a random duration of noise
                    # rand_act_dur = int(random.gauss(act_dur, act_sd))
                    rand_act_dur = int(random.randint(1, 10) * act_dur / 10)

                    success = False
                    while not success:
                        # ..get a random time to introduce noise
                        act_cell = random.randint(a=1, b=num_cells)

                        # check if noise can be introduced
                        # ..activity should fit the time line
                        if act_cell * self.scale + rand_act_dur >= 24 * 60 * 60:
                            # ..regen random time
                            continue

                        end_cell = act_cell + int(rand_act_dur / self.scale)
                        # print(act_cell, end_cell, rand_act_dur)
                        # ..can not overlap with personal hygiene or already introduced toilet
                        overlap = list(itertools.chain.from_iterable(unpacked[act_cell:end_cell + 1]))
                        if 'T' in overlap:
                            # ..regen random time
                            continue
                        else:
                            # ..else introduce the noise
                            for row in unpacked[act_cell:end_cell + 1]:
                                row[2] = routine[-1][3] + str(noise_num)
                                row[3] = routine[-1][4]
                            noise_num = noise_num + 1
                            print(noise_num)
                            success = True

                # pack one-day-routine to actual format
                packed = self.pack_routine(unpacked, self.scale)
                synt_routine = synt_routine + packed
                # print(packed)

            self.write_files(routine=synt_routine, level=2, file_num=f_num, controlled=controlled, sdp=sdp)

    def level_3(self, routine, controlled=False, sdp=5, prob=0.7):
        if controlled:
            print("[SyntheticData] Level_3: Generating level3 data for ", self.num_files, "files, ", self.num_days,
                  " days and controlled SD of " + str(sdp) + "% in each file.")
        else:
            print("[SyntheticData] Level_3: Generating level3 data for ", self.num_files, "files, ", self.num_days,
                  " days in each file.")

        for f_num in range(1, self.num_files + 1):
            synt_routine = []
            noise_num = 0
            for day in range(1, self.num_days + 1):
                one_day_routine, noise_num = self.gen_level3(day=day, routine=routine,
                                                             controlled=controlled, sdp=sdp, prob=prob,
                                                             noise_num=noise_num)
                synt_routine = synt_routine + one_day_routine
            self.write_files(routine=synt_routine, level=3, file_num=f_num, controlled=controlled, sdp=sdp, prob=prob)

    # UCI DATA SET
    @staticmethod
    def read_uci_data(file_path):
        with open(file_path, 'r') as uci_csv:
            lines = uci_csv.readlines()

        uci_data = []
        for line in lines:
            record = line.splitlines()[0].split(sep=",")
            uci_data.append(record)
        del uci_data[0]
        return uci_data

    def write_uci_routine(self, file_path, routine):
        f = open(file_path, 'w')
        parsed_writer = csv.writer(f)
        parsed_writer.writerow(["Day", "Start Time", "Duration", "Activity", "Location"])

        for sample in routine:
            parsed_writer.writerow(sample)
        f.close()

    def gen_uci_routine(self, day, routine, controlled, sdp):
        synt_routine = []
        actual_prev_end_time = 0
        prev_end_time = self.str_time_to_sec("23:59:59")
        is_full = False
        for sample in routine[:-2]:
            start_time = list(map(int, sample[0].split(sep=':')))
            start_sec = start_time[0] * 60 * 60 + start_time[1] * 60 + start_time[2]
            if start_sec != 0:
                # start_sec - actual_prev_end_time = gap between two activities
                start_sec = prev_end_time + start_sec - actual_prev_end_time

            duration = int(sample[2])
            if controlled:
                sd = int(duration * sdp / 100.0)
            else:
                # randomly select Standard Deviation
                sd_idx = random.randint(0, 2)  # 0->10%, 1->20%, 2->30%
                # calc standard deviation as a percent of duration
                sd = int(duration * self.sd[sd_idx] / 100.0)

            rand_duration = int(random.gauss(duration, sd))

            # check if activity crosses 24 hrs
            if start_sec + rand_duration > self.str_time_to_sec("23:59:59"):
                rand_duration = 24 * 60 * 60 - start_sec
                is_full = True

            # convert sec to time string
            time_split, time_str = self.sec_to_time(start_sec)

            # add record in synt_routine
            synt_routine.append([str(day).zfill(2), time_str, str(rand_duration), sample[3], sample[4]])

            prev_end_time = start_sec + rand_duration
            actual_prev_end_time = self.str_time_to_sec(sample[1])
            if is_full:
                break

        # if day is still left, add the last routine activity
        if prev_end_time < self.str_time_to_sec("23:59:59"):
            start_sec = prev_end_time + self.str_time_to_sec(routine[-2][0]) - actual_prev_end_time
            rand_duration = self.str_time_to_sec("23:59:59") - start_sec

            time_split, time_str = self.sec_to_time(start_sec)

            synt_routine.append([str(day).zfill(2), time_str, rand_duration, routine[-2][3], routine[-2][4]])

        return synt_routine

    def gen_uci_synthetic_data(self, file_path, num_files=5, num_reps=2, controlled=True, sdp=10):
        uci_data = self.read_uci_data(file_path)
        for f_num in range(1, num_files+1):
            synthetic_data = []
            for reps in range(num_reps):
                record_num = 0
                for day in range(1, 17):
                    routine = list()
                    # routine.append(["Start Time", "End Time", "Duration", "Activity", "Activity"])
                    while record_num < len(uci_data) and int(uci_data[record_num][0]) == day:
                        start_time = uci_data[record_num][1]
                        duration = int(uci_data[record_num][2])
                        start_sec = self.str_time_to_sec(start_time)
                        end_sec = start_sec + duration
                        end_time = self.sec_to_time(end_sec)[1]
                        activity = uci_data[record_num][3]
                        routine.append([start_time, end_time, duration, activity, activity])
                        record_num += 1
                    routine.append(["_:_:_", "_:_:_", "___", "N.A.", "N.A."])
                    # print(routine)
                    one_day_synt_routine = self.gen_uci_routine(day+reps*16, routine, controlled, sdp)
                    # print(one_day_synt_routine)
                    synthetic_data += one_day_synt_routine
                # routine for 1 day made
            # routine for all days made
            target_file = "../data/synthetic_data/uci_adl/csv_files/" + "A_" + str(sdp) + "_" + str(f_num) + ".csv"
            self.write_uci_routine(target_file, synthetic_data)

    # level: What levels are required. 1: Lvl1, 2:lvl2, 3:lvl3, 12:lvl1&2, 13:lvl1&3, 23:lvl2&3, 123:lvl1,2&3
    # num_days: number of days in each data file
    # num_files: number of files to be generated
    def gen_synthetic_data(self, level=1, scale=30, num_days=30,
                           num_files=5, controlled=False, sdp=10, prob=0.7,
                           base_dir="../data/synthetic_data/",
                           file_name="synthetic_routine.csv"):
        self.scale = scale
        self.num_days = num_days
        self.num_files = num_files
        self.base_dir = base_dir

        routine = self.read_general_routine(self.base_dir + file_name)

        if level == 1:
            self.level_1(routine, controlled, sdp)
        elif level == 2:
            self.level_2(routine, controlled, sdp)
        elif level == 3:
            self.level_3(routine, controlled, sdp, prob)
        elif level == 12:
            self.level_1(routine, controlled, sdp)
            self.level_2(routine, controlled, sdp)


if __name__ == "__main__":
    obj = SyntheticData()
    # sdp = [5, 10, 15, 20, 25, 30]
    # prob = [0.3, 0.5, 0.7, 0.9]
    # for sd in sdp:
    #     for p in prob:
    #         obj.gen_synthetic_data(level=3, num_days=30, num_files=10, controlled=True, sdp=sd, prob=p)
    sd = [5,10,15,20,25,30]
    for sdp in sd:
        obj.gen_uci_synthetic_data(file_path="../data/synthetic_data/uci_adl/uci_adl_orig.csv",
                                   num_files=5,
                                   num_reps=2,
                                   controlled=True,
                                   sdp=sdp)