import cv2
from tkinter import *
import tkinter
import numpy as np


class DataVisualization:
    def __init__(self, image, labels, data=0):
        self.image = image
        self.labels = labels
        self.data = data

    # similarity measures
    @staticmethod
    def hist_inter_union(hist1, hist2):
        intersection = np.logical_and(hist1, hist2)
        logical_union = np.logical_or(hist1, hist2)

        inter = np.sum(intersection)
        union = np.sum(logical_union)

        if union > 0:
            return inter / union
        else:
            return 0

    @staticmethod
    def hist_inter_normalized(hist1, hist2):
        intersection = np.logical_and(hist1, hist2)
        inter = np.sum(intersection)

        mask = np.ones(shape=hist1.shape, dtype=int)
        len1 = np.sum(np.logical_and(hist1, mask))
        len2 = np.sum(np.logical_and(hist2, mask))

        factor = min(len1, len2)

        print(inter)

        if factor > 0:
            return inter / factor
        else:
            return 0

    def display_features(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            self.lc_id = self.labels[y, x]
            textl = "ClusterID: " + str(self.lc_id)
            self.tl.delete(0.0, tkinter.END)
            self.tl.insert('insert', textl + '\n')
            self.tl.update()

        if event == cv2.EVENT_RBUTTONDOWN:
            self.rc_id = self.labels[y, x]
            textr = "ClusterID: " + str(self.rc_id)
            self.tr.delete(0.0, tkinter.END)
            self.tr.insert('insert', textr + '\n')
            self.tr.update()

        if self.lc_id != -1 and self.rc_id != -1:
            cl = self.cf[self.lc_id]
            cr = self.cf[self.rc_id]

            time_hist_iu = self.hist_inter_union(cl["time_hist"], cr["time_hist"])
            dur_hist_iu = self.hist_inter_union(cl["dur_hist"], cr["dur_hist"])
            time_norm_hist_int = self.hist_inter_normalized(cl["time_hist"], cr["time_hist"])
            dur_norm_hist_int = self.hist_inter_normalized(cl["dur_hist"], cr["dur_hist"])
            avg_stime_diff = abs(cl["stime"] - cr["stime"]) / (24*60*60)

            textc = "Similarity between " + str(self.lc_id) + " and " + str(self.rc_id) + "\n" + \
                "Number of act: Left(" + str(cl["num_clusters"]) + ")\t Right(" + str(cr["num_clusters"]) + ")\n" + \
                "Space IDs: Left" + str(cl["loc"]) + "\tRight" + str(cr["loc"]) + "\n" + \
                "Histogram Intersection / Union (Time): " + str(time_hist_iu) + "\n" + \
                "Histogram Intersection / Union (Duration): " + str(dur_hist_iu) + "\n" + \
                "Normalized Histogram Intersection (Time): " + str(time_norm_hist_int) + "\n" + \
                "Normalized Histogram Intersection (Duration): " + str(dur_norm_hist_int) + "\n" + \
                "Average Start time difference: " + str(avg_stime_diff)
            self.ts.delete(0.0, tkinter.END)
            self.ts.insert('insert', textc + '\n')
            self.ts.update()

    def feature_comparison(self, cluster_feat={}):
        self.cf = cluster_feat

        # load the image, clone it, and setup the mouse callback function
        img = self.image.copy()
        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("image", self.display_features)
        self.lc_id = -1
        self.rc_id = -1

        self.master = Tk()
        self.master.title('Cluster Visualization')

        Label(self.master, text="Left Click Cluster Features", fg="green", font=("", 15, "bold")).grid(row=1, column=1)
        self.tl = Text(self.master, bd=0, width=50, height=5, font='Fixdsys -14')
        self.tl.grid(row=2, column=1)

        Label(self.master, text="Right Click Cluster Features", fg="green", font=("", 15, "bold")).grid(row=1, column=2)
        self.tr = Text(self.master, bd=0, width=50, height=5, font='Fixdsys -14')
        self.tr.grid(row=2, column=2)

        Label(self.master, text="Similarity / Distance", fg="green", font=("", 15, "bold")).grid(row=3, column=1,
                                                                                                 columnspan=2)
        self.ts = Text(self.master, bd=0, width=75, height=20, font='Fixdsys -14')
        self.ts.grid(row=4, column=1, columnspan=2, rowspan=2)

        while True:
            cv2.imshow("image", img)
            key = cv2.waitKey(1) & 0xFF

            # if the 'q' key is pressed, break from the loop
            if key == ord("q"):
                break

        self.master.destroy()