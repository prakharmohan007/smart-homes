import cv2
from tkinter import *
import tkinter


class DataVisualization:
    def __init__(self, image, labels, data=0):
        self.image = image
        self.labels = labels
        self.data = data

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

        textc = "Similarity between " + str(self.lc_id) + " and " + str(self.rc_id)
        self.ts.delete(0.0, tkinter.END)
        self.ts.insert('insert', textc + '\n')
        self.ts.update()

    def feature_comparison(self, cluster_feat={}):

        # load the image, clone it, and setup the mouse callback function
        img = self.image.copy()
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", self.display_features)
        self.lc_id = -1
        self.rc_id = -1

        self.master = Tk()
        self.master.title('Cluster Visualization')

        Label(self.master, text="Left Click Cluster Features", fg="green", font=("", 15, "bold")).grid(row=1, column=1)
        self.tl = Text(self.master, bd=0, width=50, height=50, font='Fixdsys -14')
        self.tl.grid(row=2, column=1)

        Label(self.master, text="Right Click Cluster Features", fg="green", font=("", 15, "bold")).grid(row=3, column=1)
        self.tr = Text(self.master, bd=0, width=50, height=50, font='Fixdsys -14')
        self.tr.grid(row=4, column=1)

        Label(self.master, text="Similarity / Distance", fg="green", font=("", 15, "bold")).grid(row=1, column=2)
        self.ts = Text(self.master, bd=0, width=50, height=100, font='Fixdsys -14')
        self.ts.grid(row=2, column=2, columnspan=1, rowspan=4)

        while True:
            cv2.imshow("image", img)
            key = cv2.waitKey(1) & 0xFF

            # if the 'q' key is pressed, break from the loop
            if key == ord("q"):
                break

        self.master.destroy()
