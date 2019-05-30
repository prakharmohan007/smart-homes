import glob
import csv
import matplotlib.pyplot as plt


# glob.glob("/data/experimental_data/Subject_2/parsed_data/")
with open("./data/experimental_data/Subject_2/parsed_data/xandem_2018-11-28.csv", 'r') as csv_reader:
    data = csv_reader.readlines()
del data[0]

duration = []
mscore = []
avg_mcount = []

for row in data:
    d = row.split(sep=',')
    if int(d[1]) < 50:
        duration.append(int(d[1]))
        mscore.append(float(d[3]))
        avg_mcount.append(float(d[2]))

fig1 = plt.figure()
plt.plot(duration, mscore, 'ro')
plt.xlabel("duration")
plt.ylabel("mscore")


fig2 = plt.figure()
plt.plot(mscore, duration, 'ro')
plt.ylabel("duration")
plt.xlabel("mscore")

fig3 = plt.figure()
plt.plot(duration, avg_mcount, 'ro')
plt.xlabel("duration")
plt.ylabel("motion_count")


fig4 = plt.figure()
plt.plot(mscore, avg_mcount, 'ro')
plt.ylabel("motion_count")
plt.xlabel("mscore")

plt.show()