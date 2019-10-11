from connected_components import UnionFind
import numpy as np
import math

DEBUG = 0


class RegionGrowth:
    def __init__(self):
        self.edge = {'a': 0, 'b': 0, 'w': 0}

    @staticmethod
    def entropy(hist):
        total = sum(hist)
        ent = 0  # -p*log(p) so ent = ent - p*log(p)
        for i in hist:
            if i != 0:
                ent = ent - math.log(i/total)*i/total

        return ent

    # entropy gain on merging the two histograms
    # gain = entropy before - entropy after
    def entropy_gain(self, hist1, hist2):
        ent_b = self.entropy(hist1)
        ent_b += self.entropy(hist2)

        ent_a = self.entropy(hist1 + hist2)
        return ent_b - ent_a

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

        if factor > 0:
            return inter / factor
        else:
            return 0

    @staticmethod
    def hist_euclidean_dist(hist1, hist2):
        abs_diff = np.abs(hist1 - hist2)
        sqrd_diff = abs_diff ** 2
        euc_dist = np.sqrt(sqrd_diff.sum())
        return euc_dist

    @staticmethod
    def hist_cosine_similarity(hist1, hist2):
        h1_mag = np.linalg.norm(hist1)
        if h1_mag == 0:
            h1_mag = 1
        h2_mag = np.linalg.norm(hist2)
        if h2_mag == 0:
            h2_mag = 1

        return np.dot(hist1, hist2) / (h1_mag * h2_mag)

    def hist_similarity_measures(self, c1, c2, measure):
        if measure == "hist_euclidean_dist":
            return self.hist_euclidean_dist(c1["prev_act_avg"], c2["prev_act_avg"])
        elif measure == "hist_inter_normalized":
            return self.hist_inter_normalized(c1["time_hist"], c2["time_hist"])
        elif measure == "hist_inter_union":
            return self.hist_inter_union(c1["time_hist"], c2["time_hist"])
        elif measure == "time_hist_cosine_sim":
            return self.hist_cosine_similarity(c1["time_hist"], c2["time_hist"])
        elif measure == "prevact_hist_cosine_sim":
            return self.hist_cosine_similarity(c1["prev_act_avg"], c2["prev_act_avg"])
        elif measure == "dur_hist_cosine_sim":
            return self.hist_cosine_similarity(c1["dur_hist"], c2["dur_hist"])
        elif measure == "timedur_hist_cosine_sim":
            thcs = self.hist_cosine_similarity(c1["time_hist"], c2["time_hist"])
            dhcs = self.hist_cosine_similarity(c1["dur_hist"], c2["dur_hist"])
            return (thcs + dhcs)/2
        elif measure == "durprevact_hist_cosine_sim":
            pahcs = self.hist_cosine_similarity(c1["prev_act_avg"], c2["prev_act_avg"])
            dhcs = self.hist_cosine_similarity(c1["dur_hist"], c2["dur_hist"])
            start_time_diff = abs(c1["stime"] - c2["stime"]) / (24*60*60)
            return (1-start_time_diff)*pahcs * dhcs

    def cluster_sameday_activity(self, cluster_feat):
        if DEBUG:
            print("[RegionGrowth] cluster_sameday_activity: ")

        cluster_old_new = dict()
        cluster_new_old = dict()
        cluster_new_hist = dict()

        cluster_new_old[0] = [0]
        cluster_old_new[0] = 0
        cluster_new_hist[0] = cluster_feat[0]["loc_array"]

        for i in range(1, len(cluster_feat)):
            # if entropy +ive, merge else don't merge
            if self.entropy_gain(cluster_feat[i]["loc_array"], cluster_new_hist[cluster_old_new[i-1]]) >= 0:
                cluster_old_new[i] = cluster_old_new[i-1]
                cluster_new_old[cluster_old_new[i]].append(i)
                cluster_new_hist[cluster_old_new[i]] = cluster_new_hist[cluster_old_new[i]] + cluster_feat[i]["loc_array"]
            else:
                cluster_old_new[i] = len(cluster_new_old)
                cluster_new_old[cluster_old_new[i]] = [i]
                cluster_new_hist[cluster_old_new[i]] = cluster_feat[i]["loc_array"]
        return cluster_new_old

    # synthetic data
    def synt_region_growth(self, cluster_pixels, cluster_coarse, cluster_feat, thresh=0.6, measure="hist_inter_normalized"):
        edges = []
        num_edges = 0
        num_clusters = len(cluster_feat)
        print("[RegionGrowth] number of clusters: ", num_clusters)

        # form edges
        print("[RegionGrowth] region_growth: Forming cluster edges")
        c_id = set(cluster_feat.keys())
        while len(c_id) > 1:
            c1 = c_id.pop()
            for c2 in c_id:
                temp_edge = self.edge.copy()
                temp_edge['a'] = c1
                temp_edge['b'] = c2

                if cluster_feat[c1]["loc_type"].issubset(cluster_feat[c2]["loc_type"]) or \
                        cluster_feat[c2]["loc_type"].issubset(cluster_feat[c1]["loc_type"]):
                    temp_edge['w'] = self.hist_similarity_measures(cluster_feat[c1], cluster_feat[c2], measure)
                    # temp_edge['w']=self.hist_inter_union(cluster_feat[c1]["time_hist"],cluster_feat[c2]["time_hist"])
                edges.append(temp_edge)

                num_edges = num_edges + 1
                # print(c1, c2)

        # Union
        success = False
        print("[RegionGrowth] region_growth: performing union-find")
        obj_uf = UnionFind(num_clusters)
        c_id = set()
        for i in range(num_edges):
            # print(edges[i]['a'], edges[i]['b'])
            a = obj_uf.find(edges[i]['a'] - 1)
            b = obj_uf.find(edges[i]['b'] - 1)
            if a not in c_id and b not in c_id and edges[i]['w'] >= thresh:
                obj_uf.union(a, b)
                # print (a, b, edges[i]['w'])
                c_id.add(a)
                c_id.add(b)
                success = True
            # if a != b and edges[i]['w'] >= thresh:
            #     obj_uf.union(a, b)
            #     success = True

        # collect information for clusters
        cluster_new_pixels = {}  # new cluster_id: pixels for image
        new_cluster_course = {}  # new cluster_id: original clusters
        cluster_new_old = {}  # new cluster_id: previous step cluster ids
        cluster_uf_new = {}  # cluster ID: union-find ids to actual new ids
        cluster_id = 1

        # c is the original cluster
        for c in cluster_pixels:
            c_id = obj_uf.find(c - 1)

            # if the cluster has not been allotted an id, alot it
            if c_id not in cluster_uf_new:
                cluster_uf_new[c_id] = cluster_id
                cluster_id = cluster_id + 1

            if cluster_uf_new[c_id] not in cluster_new_old:
                cluster_new_old[cluster_uf_new[c_id]] = []
                cluster_new_pixels[cluster_uf_new[c_id]] = []
                new_cluster_course[cluster_uf_new[c_id]] = []

            cluster_new_pixels[cluster_uf_new[c_id]] += cluster_pixels[c]
            new_cluster_course[cluster_uf_new[c_id]] += cluster_coarse[c]
            cluster_new_old[cluster_uf_new[c_id]].append(c)

        return cluster_new_old, cluster_new_pixels, new_cluster_course, success

    # real data
    def region_growth(self, cluster_pixels, cluster_coarse, cluster_feat, thresh=0.6,
                      measure="hist_inter_normalized"):
        edges = []
        num_edges = 0
        num_clusters = len(cluster_feat)
        print("[RegionGrowth] number of clusters: ", num_clusters)

        # form edges
        print("[RegionGrowth] region_growth: Forming cluster edges")
        c_id = set(cluster_feat.keys())
        while len(c_id) > 1:
            c1 = c_id.pop()
            for c2 in c_id:
                temp_edge = self.edge.copy()
                temp_edge['a'] = c1
                temp_edge['b'] = c2

                if self.hist_cosine_similarity(cluster_feat[c1]["loc_array"], cluster_feat[c2]["loc_array"]) > 0.8:
                    temp_edge['w'] = self.hist_similarity_measures(cluster_feat[c1], cluster_feat[c2], measure)
                    # temp_edge['w']=self.hist_inter_union(cluster_feat[c1]["time_hist"],cluster_feat[c2]["time_hist"])
                edges.append(temp_edge)

                num_edges = num_edges + 1
                # print(c1, c2)

        # Union
        success = False
        print("[RegionGrowth] region_growth: performing union-find")
        obj_uf = UnionFind(num_clusters)
        c_id = set()
        for i in range(num_edges):
            # print(edges[i]['a'], edges[i]['b'])
            a = obj_uf.find(edges[i]['a'] - 1)
            b = obj_uf.find(edges[i]['b'] - 1)
            if a not in c_id and b not in c_id and edges[i]['w'] >= thresh:
                obj_uf.union(a, b)
                # print (a, b, edges[i]['w'])
                c_id.add(a)
                c_id.add(b)
                success = True
            # if a != b and edges[i]['w'] >= thresh:
            #     obj_uf.union(a, b)
            #     success = True

        # collect information for clusters
        cluster_new_pixels = {}  # new cluster_id: pixels for image
        new_cluster_course = {}  # new cluster_id: original clusters
        cluster_new_old = {}  # new cluster_id: previous step cluster ids
        cluster_uf_new = {}  # cluster ID: union-find ids to actual new ids
        cluster_id = 1

        # c is the original cluster
        for c in cluster_pixels:
            c_id = obj_uf.find(c - 1)

            # if the cluster has not been allotted an id, alot it
            if c_id not in cluster_uf_new:
                cluster_uf_new[c_id] = cluster_id
                cluster_id = cluster_id + 1

            if cluster_uf_new[c_id] not in cluster_new_old:
                cluster_new_old[cluster_uf_new[c_id]] = []
                cluster_new_pixels[cluster_uf_new[c_id]] = []
                new_cluster_course[cluster_uf_new[c_id]] = []

            cluster_new_pixels[cluster_uf_new[c_id]] += cluster_pixels[c]
            new_cluster_course[cluster_uf_new[c_id]] += cluster_coarse[c]
            cluster_new_old[cluster_uf_new[c_id]].append(c)

        return cluster_new_old, cluster_new_pixels, new_cluster_course, success
