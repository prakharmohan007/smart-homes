import numpy as np
import sys
import random

desired_width = 2900
np.set_printoptions(threshold=sys.maxsize)
DEBUG = 0


class SEEDS:
    def __init__(self):
        self.seed_width = 0
        self.nr_levels = 0
        self.nr_superpixels = 0
        self.nr_bins = 0
        self.histogram_size = 0

        self.top_level = 0
        self.curr_level = 0

        self.width = 0  # array length or width

        self.labels = []  # [level][y * width + x] -> stores the label (cluster idx) at each level
        self.parents = []  # [level][y * width + x] -> parent label of current label
        self.nr_labels = []  # 1D, stores number of labels at each level
        self.histogram = []  # 3D, [level][label][bins]
        self.block_size = []  # [level][label] how many units in each label
        self.nr_partitions = []  # ?
        self.bin_index = []  # image_bins, bin index (histogram) of each image pixel [y*width + x]
        self.min_nr_sublabels = 0

    def initialize(self, width, scale, num_locs):
        if 24 * 60 * 60 / scale != width:
            print("[SEEDS] init: scale and width are inconsistent")
            raise

        if width == 2880:
            # 8 minute superpixel -> superpixel width = 16
            # 2, 4, 8, 16
            self.seed_width = 2
            self.nr_levels = 3
            self.nr_superpixels = 2880 / 16  # 180
        elif width == 1440:
            # 8 minute superpixel -> superpixel width = 8
            # 2, 4, 8
            self.seed_width = 2
            self.nr_levels = 4
            self.nr_superpixels = 1440 / 8  # 180
        elif width == 17280:
            # 8 minute superpixel -> superpixel width = 96
            # 6, 12, 24, 48, 96
            self.seed_width = 12
            self.nr_levels = 3
            self.nr_superpixels = 17280 / 48  # 180

        self.min_nr_sublabels = 1

        self.width = width
        self.nr_bins = num_locs
        self.histogram_size = num_locs

        self.top_level = self.nr_levels - 1
        self.labels = [None] * self.nr_levels
        self.parents = [None] * self.nr_levels
        self.nr_labels = [None] * self.nr_levels
        self.nr_partitions = [None] * self.nr_levels

        # initialize
        for level in range(0, self.nr_levels):
            num_blocks = int(width / (self.seed_width * 2 ** level))
            self.labels[level] = [None] * width
            self.parents[level] = [None] * num_blocks
            self.nr_labels[level] = num_blocks
            self.nr_partitions[level] = [None] * num_blocks

        # initialize histograms
        self.histogram = [None] * self.nr_levels
        self.block_size = [None] * self.nr_levels  # lock sizes are kept at each level and each label [level][label]
        for level in range(0, self.nr_levels):
            self.histogram[level] = [None] * self.nr_labels[level]
            self.block_size[level] = [0] * self.nr_labels[level]

            for label in range(0, self.nr_labels[level]):
                self.histogram[level][label] = np.zeros(self.histogram_size, dtype=int)

        if DEBUG:
            print("[SEEDS] initialize")
            print("\tnr_levels: ", self.nr_levels)
            print("\tseeds width: ", self.seed_width, ", levels: ", self.nr_levels)

            print("\tdimensions of self.labels: ", len(self.labels))
            for row in self.labels:
                print("\t\t", len(row), end=' ')

            print("\n\tdimensions of self.parent: ", len(self.parents))
            for row in self.parents:
                print("\t\t", len(row), end=' ')

            print("\n\tdimensions of self.nr_labels (1D array): ", len(self.nr_labels))
            for row in self.nr_labels:
                print("\t\t", row, end=' ')

            print("\n\tdimensions of self.histogram (3D array): ", len(self.histogram))
            for row in self.histogram:
                print("\t\t", len(row), end=' ')

            print("\n\tdimensions of self.nr_partitions (2D array): ", len(self.nr_partitions))
            for row in self.nr_partitions:
                print("\t\t", len(row), end=' ')

            print("\n\tdimensions of self.block_size (2D array): ", len(self.block_size))
            for row in self.block_size:
                print("\t\t", len(row), end=' ')

            print("\n[SEED] initialize: Initialization Done")

    def assign_labels(self):
        if DEBUG:
            print("[SEEDS] assign_labels: assigning labels and parents")

        for level in range(0, self.nr_levels):
            num_blocks = int(self.width / (self.seed_width * 2 ** level))  # nr_seeds
            step_w = int(self.seed_width * 2 ** level)

            for i in range(0, num_blocks):
                if level == 0:
                    self.nr_partitions[level][i] = 1
                else:
                    self.nr_partitions[level][i] = 0

            for w in range(0, self.width):
                self.labels[level][w] = int(w / step_w)
                if self.labels[level][w] > num_blocks:
                    self.labels[level][w] = num_blocks - 1

                # assign level as parents of level-1
                if level > 0:
                    self.parents[level - 1][self.labels[level - 1][w]] = self.labels[level][w]

        if DEBUG:
            print("[SEEDS] assign_labels: labels and parents assigned")

    # TODO
    # modify according to the actual features
    def calc_histogram_bin(self, routine):
        self.bin_index = routine

    def delete_element(self, level, label, w):
        self.histogram[level][label][self.bin_index[w]] -= 1
        self.block_size[level][label] -= 1

    def add_element(self, level, label, w):
        self.histogram[level][label][self.bin_index[w]] += 1
        self.block_size[level][label] += 1

    def add_block(self, level, parent_label, child_level, child_label):
        self.parents[child_level][child_label] = parent_label
        # child_level = level - 1
        self.histogram[level][parent_label] = self.histogram[level][parent_label] + self.histogram[child_level][
            child_label]
        self.block_size[level][parent_label] += self.block_size[child_level][child_label]
        self.nr_partitions[level][parent_label] += 1

    def compute_histograms(self, routine, until_level=-1):
        # self.assign_labels()

        # store bins of every element
        self.calc_histogram_bin(routine)

        if until_level == -1:
            until_level = self.nr_levels  # -1

        # clear all histograms
        if DEBUG:
            print("[SEEDS] compute_histograms: Clearing all histograms")
        for level in range(self.nr_levels):
            for label in range(self.nr_labels[level]):
                self.histogram[level][label] = np.zeros(self.histogram_size, dtype=int)
            self.block_size[level] = [0] * self.nr_labels[level]

        if DEBUG:
            print("[SEEDS] compute_histogram: building level0 histograms")
        # build histograms on the first level by adding the elements to the blocks at element level
        for w in range(0, self.width):
            self.add_element(0, self.labels[0][w], w)

        if DEBUG:
            print("[SEEDS] compute_histogram: building higher level histograms")
        for level in range(1, until_level):
            for label in range(self.nr_labels[level - 1]):
                self.add_block(level, self.parents[level - 1][label], level - 1, label)

        if DEBUG:
            print("[SEEDS] compute_histogram: histograms build")

    def delete_block(self, top_level, plabel, level, label):
        self.parents[level][label] = -1
        self.histogram[top_level][plabel] = self.histogram[top_level][plabel] - self.histogram[level][label]
        self.block_size[top_level][plabel] -= self.block_size[level][label]
        self.nr_partitions[top_level][plabel] -= 1
        # if DEBUG:
        #     print(self.block_size[top_level][plabel])

    def probability(self, histbin, label1, label2):
        p_l1 = self.histogram[self.top_level][label1][histbin] / self.block_size[self.top_level][label1]
        p_l2 = self.histogram[self.top_level][label2][histbin] / self.block_size[self.top_level][label2]
        return p_l2 > p_l1

    def intersection(self, level1, label1, level2, label2):
        sum1 = 0
        sum2 = 0

        hist1 = self.histogram[level1][label1]
        hist2 = self.histogram[level2][label2]

        count1 = self.block_size[level1][label1]
        count2 = self.block_size[level2][label2]

        if DEBUG:
            if count1 == 0 or count2 == 0:
                print("\t[SEEDS] intersection: level1: ", level1, ", label1: ", label1, ", count1: ", count1)
                print("\t[SEEDS] intersection: level2: ", level2, ", label2: ", label2, ", count2: ", count2, "\n")
                print("\t[SEEDS] intersection: level1: ", level1, ", label1: ", label1, ", hist1: ", hist1)
                print("\t[SEEDS] intersection: level2: ", level2, ", label2: ", label2, ", hist2: ", hist2)

        for b in range(self.nr_bins):
            if hist1[b] * count1 < hist2[b] * count2:
                sum1 += hist1[b]
            else:
                sum2 += hist2[b]

        return sum1 / count1 + sum2 / count2

    def update_labels(self, level):
        for w in range(self.width):
            self.labels[self.top_level][w] = self.parents[level][self.labels[level][w]]

    def update_blocks(self, level, req_confidence=0.0):

        for label in range(self.nr_labels[level] - 1):
            # parent of this label and next label
            sublabel = label
            label1 = self.parents[level][sublabel]
            label2 = self.parents[level][sublabel + 1]

            if label1 != label2:  # if the parents of two neighboring blocks are diff, they are on boundary
                done = False  # is forward update isn't happening, try backward

                # check if parent block can be broken
                if self.nr_partitions[self.top_level][label1] > self.min_nr_sublabels and \
                        self.block_size[self.top_level][label1] > self.block_size[level][sublabel]:

                    self.delete_block(self.top_level, label1, level, sublabel)
                    int1 = self.intersection(self.top_level, label1, level, sublabel)
                    int2 = self.intersection(self.top_level, label2, level, sublabel)
                    confidence = abs(int1 - int2)

                    # if intersection with neighbor is better, add the block to neighbor
                    # else add it back to orignal
                    if int2 > int1 and confidence > req_confidence:
                        self.add_block(self.top_level, label2, level, sublabel)
                        done = True

                    else:
                        self.add_block(self.top_level, label1, level, sublabel)

                sublabel = sublabel + 1
                # if forward update didn't work, try backward update
                if not done and self.nr_partitions[self.top_level][label2] > self.min_nr_sublabels and \
                        self.block_size[self.top_level][label2] > self.block_size[level][sublabel]:

                    self.delete_block(self.top_level, label2, level, sublabel)
                    int1 = self.intersection(self.top_level, label1, level, sublabel)
                    int2 = self.intersection(self.top_level, label2, level, sublabel)
                    confidence = abs(int1 - int2)

                    # if intersection with neighbor is better, add the block to neighbor
                    # else add it back to original
                    if int1 > int2 and confidence > req_confidence:
                        self.add_block(self.top_level, label1, level, sublabel)

                    else:
                        self.add_block(self.top_level, label2, level, sublabel)

        self.update_labels(level)

    def update_element(self, level, label_new, w):
        label_old = self.labels[level][w]
        self.delete_element(level, label_old, w)
        self.add_element(level, label_new, w)
        self.labels[level][w] = label_new

    def update_pixel(self):
        updated = [None] * self.width
        w = 0
        while w < self.width - 1:
            label1 = self.labels[self.top_level][w]
            label2 = self.labels[self.top_level][w + 1]

            if label1 != label2:
                if self.probability(self.bin_index[w], label1, label2) and updated[w] is None:
                    self.update_element(self.top_level, label2, w)
                    updated[w] = True
                    w -= 2
                elif self.probability(self.bin_index[w + 1], label2, label1) and updated[w + 1] is None:
                    self.update_element(self.top_level, label1, w + 1)
                    updated[w + 1] = True
            w += 1

    def go_down_one_level(self):
        old_level = self.curr_level
        new_level = self.curr_level - 1

        if new_level < 0:
            return -1

        # reset nr_partitions of top level
        for w in range(self.nr_labels[self.top_level]):
            self.nr_partitions[self.top_level][w] = 0

        # change the parents of new level to those of top level
        for w in range(self.nr_labels[new_level]):
            p = self.parents[old_level][self.parents[new_level][w]]
            self.parents[new_level][w] = p
            self.nr_partitions[self.top_level][p] += self.nr_partitions[new_level][w]

        return new_level

    def iterate(self):
        # start with one level lower than the top level
        # because top level is the final clusters

        if DEBUG:
            print("[SEEDS] iterate: starting block updates")

        self.curr_level = self.nr_levels - 2  # self.top_level - 1

        # block update
        while self.curr_level >= 0:
            if DEBUG:
                print("[SEEDS] iterate: updating level ", self.curr_level)
            self.update_blocks(self.curr_level)
            self.curr_level = self.go_down_one_level()

            if DEBUG:
                print("[SEEDS] iterate")
                print("\n\tBlock Sizes")
                for level in range(self.nr_levels):
                    print("\tlevel: ", level)
                    print("\t\t", self.block_size[level])

                print("\n\tnr_partitions")
                for level in range(self.nr_levels):
                    print("\tlevel: ", level)
                    print("\t\t", self.nr_partitions[level])

                print("\n\tLabels")
                for level in range(self.nr_levels):
                    print("\tlevel: ", level)
                    print("\t\t", self.labels[level])

                print("\n\tParents")
                for level in range(self.nr_levels):
                    print("\tlevel: ", level)
                    print("\t\t", self.parents[level])

        # update individual elements
        self.update_pixel()
        # self.update_pixel()
        # self.update_pixel()


if __name__ == '__main__':
    seedsobj = SEEDS()
    seedsobj.initialize(17280, 5, 7)
    seedsobj.assign_labels()

    # generate a routine thing
    # routine = np.zeros(2880, dtype=int)
    # routine[576:1152] = np.random.randint(2, size=576)
    # routine[1152:1728] = np.random.randint(3, size=576)
    # routine[1728:2304] = np.random.randint(4, size=576)
    # routine[2304:2880] = np.random.randint(5, size=576)
    routine = np.zeros(17280, dtype=int)
    routine[2160:4320] = np.random.randint(2, size=2160)
    routine[4320:6480] = np.random.randint(4, size=2160)
    routine[6480:8640] = np.random.randint(2, size=2160)
    routine[8640:10800] = np.random.randint(1, 4, size=2160)
    routine[10800:12960] = np.random.randint(4, 6, size=2160)
    routine[12960:15120] = np.array(random.choices([3, 5], k=2160))
    routine[15120:] = np.random.randint(4, 7, size=2160)

    # routine[576:1152] = np.ones(576)
    # routine[1152:1728] = np.array([2]*576)
    # routine[1728:2304] = np.array([3]*576)
    # routine[2304:2880] = np.array([4]*576)

    seedsobj.compute_histograms(routine)
    seedsobj.iterate()

    print(list(routine))
    print(seedsobj.labels[-1])

    label_routine = {}
    j = 0
    for i in range(len(seedsobj.labels[-1])):
        if i == len(seedsobj.labels[-1]) - 1 or seedsobj.labels[-1][i] != seedsobj.labels[-1][i + 1]:
            label_routine[seedsobj.labels[-1][i]] = routine[j:i + 1]
            j = i + 1

    for l in label_routine:
        print(l, label_routine[l])

    exit(1)
