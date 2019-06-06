class ReadSyntheticData:
    def __init__(self, file_path):
        self.data = self.read_file(file_path)

    # parse_time converts time in str(hh:mm:ss) to
    # int [hh, mm, ss] and total sec
    @staticmethod
    def parse_time(time_str):
        time_split = list(map(int, time_str.split(sep=':')))
        sec = time_split[0]*3600 + time_split[1]*60 + time_split[2]
        return time_split, sec

    # read_file read the parsed data as it is and returns the list
    # each row = [day(int), time(string), duration(int), activity(string), location(string / character)]
    @staticmethod
    def read_file(file_path):
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
        except IOError as err:
            print("[ReadSyntheticdata] read_file: Error reading file ", file_path, " Error: ", err)
            raise
        del lines[0]

        data = []
        for line in lines:
            split_line = line.splitlines()[0].split(sep=',')
            day = int(split_line[0])
            stime = split_line[1]
            dur = int(split_line[2])
            activity = split_line[3]
            loc = split_line[4]
            data.append([day, stime, dur, activity, loc])
        return data

    def get_data(self):
        return self.data


# feature convention: dictionary
# numpy arrays and integers and sets
class GenerateSyntheticCluster:
    def __init__(self, file_path):
        self.data = self.get_cluster_data(file_path)

    @staticmethod
    def get_cluster_data(file_path):
        data_obj = ReadSyntheticData(file_path)
        data = data_obj.get_data()
        return data

    def get_cluster_features(self):
        pass