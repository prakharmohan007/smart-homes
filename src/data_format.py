import glob
import csv


src_dir = "../data/experimental_data/Subject_2/processed_data"
target_dir = "../data/experimental_data/Subject_2/parsed_data"
print("Reading files")
log_files = glob.glob(src_dir + "/*.log")
if len(log_files) == 0:
    print("Error - No log files found in the folder ", dir)
    exit(-1)

for file in log_files:
    target_file = target_dir + "/" + file.split('/')[-1].split(sep='.')[0]+".csv"
    csv_writer = open(target_file, 'w')
    f_write = csv.writer(csv_writer, delimiter=',')
    f_write.writerow(["time", "duration", "is_motion_instances", "avg_motion_score", "room_id"])
    with open(file, 'r') as f_read:
        data = f_read.readlines()
    del data[0]

    prev_space = None
    stime = None
    duration = None
    count = 0
    mscore = 0
    mcount = 0.0

    for sample in range(len(data)):
        d = data[sample].splitlines()[0].split(sep=',')
        count = count + 1
        if prev_space is None:
            stime = d[0]
            prev_space = d[-1]

        if d[-1] != prev_space or sample == len(data)-1:
            duration = count * 5
            avgmscore = mscore / count
            num_motion = mcount / count
            f_write.writerow([stime, str(duration), str(num_motion), str(avgmscore), prev_space])
            prev_space = d[-1]
            stime = d[0]
            mscore = 0
            mcount = 0.0
            count = 1
        mscore = mscore + float(d[4])
        # print(d[3])
        if d[3] == "true":
            mcount = mcount + 1


    csv_writer.close()

