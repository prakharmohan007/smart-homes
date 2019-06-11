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

    def write_files(self, routine, level=1, file_num=1):
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

        file_name = "synt_data_lvl" + str(level) + "_days" + str(self.num_days) + "_" + str(file_num) + ".csv"

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

    def gen_level1(self, day, routine):
        synt_routine = []
        prev_end_time = 24 * 60 * 60
        is_full = False
        for sample in routine[:-2]:
            time_split = list(map(int, sample[0].split(sep=':')))
            start_sec = time_split[0] * 60 * 60 + time_split[1] * 60 + time_split[2]
            if start_sec != 0:
                start_sec = prev_end_time + 1

            # randomly select Standard Deviation
            sd_idx = random.randint(0, 2)  # 0->10%, 1->20%, 2->30%
            duration = int(sample[2])
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

    def level_1(self, routine):
        print("[SyntheticData] Level_1: Generating level1 data for ", self.num_files, "files, ", self.num_days,
              " days in each file.")

        for f_num in range(1, self.num_files + 1):
            synt_routine = []
            for day in range(1, self.num_days + 1):
                one_day_routine = self.gen_level1(day, routine)
                synt_routine = synt_routine + one_day_routine
            self.write_files(routine=synt_routine, level=1, file_num=f_num)

    # level2 = level1 with randomly distributed toilet activity
    def level_2(self, routine):
        print("[SyntheticData] Level_2: Generating level2 data for ", self.num_files, "files, ", self.num_days,
              " days in each file.")

        for f_num in range(1, self.num_files + 1):
            synt_routine = []
            for day in range(1, self.num_days + 1):
                one_day_routine = self.gen_level1(day, routine)
                # print(one_day_routine)

                # introduce noise
                # ..get a random number of instances of noise
                num_act = random.randint(a=1, b=self.max_noise_event)
                act_dur = int(routine[-1][2])  # last record in routine is the noise
                act_sd = int(act_dur / 10)  # setting standard deviation to 10%
                num_cells = int(24 * 60 * 60 / self.scale)

                # unpack one-day-routine to easy introduce the noise
                unpacked = self.unpack_routine(one_day_routine, self.scale)
                #print(unpacked)

                # ..for each instance
                for i in range(num_act):
                    # ..get a random duration of noise
                    rand_act_dur = int(random.gauss(act_dur, act_sd))

                    success = False
                    while not success:
                        # ..get a random time to introduce noise
                        act_cell = random.randint(a=1, b=num_cells)

                        # check if noise can be introduced
                        # ..activity should fit the time line
                        if act_cell * self.scale + rand_act_dur >= 24 * 60 * 60:
                            # ..regen random time
                            continue

                        end_cell = act_cell + int(rand_act_dur/self.scale)
                        # print(act_cell, end_cell, rand_act_dur)
                        # ..can not overlap with personal hygiene or already introduced toilet
                        overlap = list(itertools.chain.from_iterable(unpacked[act_cell:end_cell+1]))
                        if 'T' in overlap:
                            # ..regen random time
                            continue
                        else:
                            # ..else introduce the noise
                            for row in unpacked[act_cell:end_cell+1]:
                                row[2] = routine[-1][3]
                                row[3] = routine[-1][4]
                                success = True

                # pack one-day-routine to actual format
                packed = self.pack_routine(unpacked, self.scale)
                synt_routine = synt_routine + packed
                # print(packed)

            self.write_files(routine=synt_routine, level=2, file_num=f_num)

    # level: What levels are required. 1: Lvl1, 2:lvl2, 3:lvl3, 12:lvl1&2, 13:lvl1&3, 23:lvl2&3, 123:lvl1,2&3
    # num_days: number of days in each data file
    # num_files: number of files to be generated
    def gen_synthetic_data(self, level=1, scale=30, num_days=30, num_files=5, base_dir="../data/synthetic_data/",
                           file_name="synthetic_routine.csv"):
        self.scale = scale
        self.num_days = num_days
        self.num_files = num_files
        self.base_dir = base_dir

        routine = self.read_general_routine(self.base_dir + file_name)

        if level == 1:
            self.level_1(routine)
        elif level == 2:
            self.level_2(routine)
        elif level == 12:
            self.level_1(routine)
            self.level_2(routine)


if __name__ == "__main__":
    obj = SyntheticData()
    obj.gen_synthetic_data(level=12, num_days=30, num_files=5)
