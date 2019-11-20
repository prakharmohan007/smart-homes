import glob
import csv
import matplotlib.pyplot as plt


# glob.glob("/data/experimental_data/Subject_2/parsed_data/")
with open("./data/experimental_data/Subject_2/parsed_data/xandem_2018-11-28.csv", 'r') as csv_reader:
    data = csv_reader.readlines()
del data[0]

duration = []
mscore = []

for row in data:
    d = row.split(sep=',')
    if int(d[1]) < 100:
        duration.append(int(d[1]))
        mscore.append(float(d[2]))

fig = plt.figure()
plt.plot(duration, mscore, 'ro')
plt.xlabel("duration")
plt.ylabel("mscore")


fig = plt.figure()
plt.plot(mscore, duration, 'ro')
plt.ylabel("duration")
plt.xlabel("mscore")


plt.show()

