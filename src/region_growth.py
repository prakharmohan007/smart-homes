from connected_components import UnionFind
import numpy as np


class RegionGrowth:
    def __init__(self):
        self.edge = {'a': 0, 'b': 0, 'w': 0}

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

    def region_growth(self, cluster, cluster_feat, thresh=0.6):
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
                    temp_edge['w'] = self.hist_inter_union(cluster_feat[c1]["time_hist"], cluster_feat[c2]["time_hist"])
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
            # if a not in c_id and b not in c_id and edges[i]['w'] >= thresh:
            #     obj_uf.union(a, b)
            #     c_id.add(a)
            #     c_id.add(b)
            #     success = True
            if a != b and edges[i]['w'] >= thresh:
                obj_uf.union(a, b)
                success = True

        # collect information for clusters
        cluster_pixels = {}  # cluster_id: pixels for image
        cluster_elements = {}  # cluster_id: oriinal clluster ids
        old_new_cluster = {}  # cluster ID. cid from union-find are not true ids
        cluster_id = 1

        for c in cluster:
            c_id = obj_uf.find(c - 1)

            # if the cluster has not been allotted an id, alot it
            if c_id not in old_new_cluster:
                old_new_cluster[c_id] = cluster_id
                cluster_id = cluster_id + 1

            if old_new_cluster[c_id] not in cluster_elements:
                cluster_elements[old_new_cluster[c_id]] = []
                cluster_pixels[old_new_cluster[c_id]] = []

            cluster_pixels[old_new_cluster[c_id]] += cluster[c]
            cluster_elements[old_new_cluster[c_id]].append(c)

        return cluster_elements, cluster_pixels, success
