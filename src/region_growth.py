from connected_components import UnionFind
import numpy as np
import simplejson

DEBUG = 0


class RegionGrowth:
    def __init__(self):
        self.edge = {'a': 0, 'b': 0, 'w': 0}

    def first_pass_similarity(self, room_vec_a, room_vec_b):
        # vec_a = np.array(room_vec_a)
        # vec_b = np.array(room_vec_b)
        return np.dot(room_vec_a, room_vec_b)

    def similarity_room_type(self, a1, a2):
        print(a1["type"], a2["type"])
        if a1["type"] == a2["type"]:
            return 1
        else:
            return 0

    # distance measure (dist) dist:
    # 1 -> euclidean distance (normalized times)
    # 2 -> cosine similarity
    def similarity_time_duration(self, a1, a2, dist=1):
        distance = 0
        a1_stime = a1["stime"]
        a2_stime = a2["stime"]

        a1_dur = a1["duration"]
        a2_dur = a2["duration"]

        if dist == 1:
            secs = 24 * 60 * 60.0
            a1_stime = a1_stime / secs
            a2_stime = a2_stime / secs

            a1_dur = a1_dur / secs
            a2_dur = a2_dur / secs

            distance = ((a1_stime - a2_stime) ** 2 + (a1_dur - a2_dur) ** 2)
            distance = distance ** 1 / 2
        else:
            numerator = a1_stime * a2_stime + a1_dur * a2_dur
            denominator = (a1_stime ** 2 + a2_stime ** 2) ** 1 / 2 * (a1_dur ** 2 + a2_dur ** 2) ** 1 / 2
            distance = numerator / denominator
        print(distance)
        return distance

    def similarity_time(self, a1, a2, ):
        a1_stime = a1["stime"]
        a2_stime = a2["stime"]
        secs = 24 * 60 * 60.0
        distance = abs(a1_stime - a2_stime) / secs
        return distance

    # Cluster the activities within same day using the room id
    # return the dictionary of clusters where every key has a list
    # of pixel coordinates (row , column) in that cluster
    # TO-DO: write the similarity function
    def first_pass(self, data):
        rows = len(data)
        cols = len(data[0])
        # edges = [self.edge]*(rows*(cols-1))
        edges = []
        numedges = 0
        no_data_points = []

        # make edges with pixels for which data is available
        for y in range(rows):
            for x in range(cols - 1):
                if data[y][x] != 0 and data[y][x + 1] != 0:
                    temp_edge = self.edge.copy()
                    temp_edge['a'] = y * cols + x
                    temp_edge['b'] = y * cols + x + 1

                    # vec_a = data[y][x]["space"].copy()
                    vec_a = data[y][x].space.copy()
                    # vec_b = data[y][x + 1]["space"].copy()
                    vec_b = data[y][x + 1].space.copy()

                    temp_edge['w'] = self.first_pass_similarity(vec_a, vec_b)
                    edges.append(temp_edge)
                    numedges += 1

        # get points for which data is not available
        for y in range(rows):
            for x in range(cols):
                if data[y][x] == 0:
                    no_data_points.append(y * cols + x)

        if DEBUG:
            print("number of edges to check: ", numedges)
            print(edges)

        # make clusters
        # cluster all points (irrespective of location) for which data is not available as one

        # sorted_edges = sorted(edges, key=lambda i: i['x'], reverse=True)
        obj_uf = UnionFind(rows * cols)
        d = 0
        for i in range(numedges):
            # print(edges[i]['a'], edges[i]['b'])
            a = obj_uf.find(edges[i]['a'])
            b = obj_uf.find(edges[i]['b'])
            # print(i, a, b)
            if a != b and edges[i]['w'] == 1:
                # print(d)
                # d += 1
                obj_uf.union(a, b)

        # cluster all the no-data points into 1 cluster
        for i in range(len(no_data_points) - 1):
            a = obj_uf.find(no_data_points[i])
            b = obj_uf.find(no_data_points[i + 1])
            if a != b:
                obj_uf.union(a, b)

        # cluster label for points without data
        no_data_label = 0
        if len(no_data_points) != 0:
            no_data_label = obj_uf.find(no_data_points[0])

        # collect information for clusters
        cluster_elements = {}  # final dictionary of clusters []
        cluster_map = {}  # cluster ID. cid from union-find are not true ids
        cluster_id = 1

        # store the pixels for each cluster. [row, col]
        label_img = np.zeros(shape=(rows, cols), dtype=int)
        # label_img = label_img*(-1)
        for y in range(rows):
            for x in range(cols):
                cid = obj_uf.find(y * cols + x)
                if cid == no_data_label:
                    continue

                # if the cluster has not been allotted an id, allot it
                if cid not in cluster_map:
                    cluster_map[cid] = cluster_id
                    cluster_id = cluster_id + 1

                # If the cluster is not in the dictionary, initialize it as list
                if cluster_map[cid] not in cluster_elements:
                    cluster_elements[cluster_map[cid]] = []

                cluster_elements[cluster_map[cid]].append((y, x))
                label_img[y, x] = cluster_map[cid]

        return cluster_elements, label_img

    def merge_short_activities(self, clusters, cluster_feat, label_img, thresh=20):
        rows = len(label_img)
        cols = len(label_img[0])
        merge_ids = {}
        new_clusters = {}
        # new_label_img = np.ones((rows, cols))*(-1)
        new_label_img = label_img.copy()
        get_id = 0
        for r in range(rows):
            for c in range(cols-1):
                if label_img[r, c] != label_img[r, c+1]:  # check for potential merging cluster
                    c1 = label_img[r, c]
                    c2 = label_img[r, c+1]

                    # print(c1, cluster_feat[c1].duration, "\t", c2, cluster_feat[c2].duration)
                    # input()
                    if cluster_feat[c1].duration <= thresh:
                        # is both c1 and c2 are small or if c2 is large and c1 follows B-s-B
                        if cluster_feat[c2].duration <= thresh or c1 not in merge_ids:
                            # merge c1 and c2
                            if c1 in merge_ids:
                                merge_ids[c2] = merge_ids[c1]
                            else:
                                get_id = get_id + 1
                                merge_ids[c1] = get_id
                                merge_ids[c2] = get_id
                        else:  # cluster_feat[c2].duration > thresh and c1 is already added
                            get_id = get_id + 1
                            merge_ids[c2] = get_id
                    else:  # cluster_feat[c1].duration > thresh:
                        # add c1 as an individual cluster
                        if c1 not in merge_ids:
                            get_id = get_id+1
                            merge_ids[c1] = get_id

                elif c == cols-2:
                    # if the cluster is not added anywhere, add it
                    if label_img[r, c] not in merge_ids:
                        get_id = get_id + 1
                        merge_ids[label_img[r, c]] = get_id

        for key in merge_ids:
            new_label_img[new_label_img == key] = merge_ids[key]

        for y in range(rows):
            for x in range(cols):
                cid = new_label_img[y, x]
                if cid not in new_clusters:
                    new_clusters[cid] = []
                new_clusters[cid].append((y, x))

        return new_clusters, new_label_img

    def second_pass(self, clusters, cluster_feat):
        num_clusters = len(clusters)

        edges = []
        num_edges = 0
        # prepare edges within clusters
        for i in range(num_clusters - 1):
            for j in range(i + 1, num_clusters):
                temp_edge = self.edge.copy()
                temp_edge['a'] = i
                temp_edge['b'] = j
                # temp_edge['w'] = self.first_pass_similarity(vec_a, vec_b)
                # temp_edge['w'] *= self.similarity_time_duration(cluster_feat[i], cluster_feat[j], 1)

                if self.similarity_room_type(cluster_feat[i], cluster_feat[j]) == 1:
                    temp_edge['w'] = self.similarity_time(cluster_feat[i], cluster_feat[j])
                else:
                    temp_edge['w'] = 0
                edges.append(temp_edge)
                num_edges += 1

        if DEBUG:
            print("number of edges to check: ", num_edges)
            print(edges)

        sorted_edges = sorted(edges, key=lambda i: i['w'])
        obj_uf = UnionFind(num_clusters)
        for i in sorted_edges:
            # print(i['a'], i['b'])
            a = obj_uf.find(i['a'])
            b = obj_uf.find(i['b'])
            print(i['w'])
            # print(i, a, b)
            if a != b and i['w'] != 0 and i['w'] <= 0.1:
                # print(d)
                # d += 1
                obj_uf.union(a, b)

        # collect information for clusters
        cluster_elements = {}
        cluster_map = {}
        cluster_id = 0

        for i in range(num_clusters):
            cid = obj_uf.find(i)

            # if the cluster has not been allotted an id, allot it
            if cid not in cluster_map:
                cluster_map[cid] = cluster_id
                cluster_id += 1

            # If the cluster is not in the dictionary, initialize it as list
            if cluster_map[cid] not in cluster_elements:
                cluster_elements[cluster_map[cid]] = []

            cluster_elements[cluster_map[cid]] += clusters[i]

        return cluster_elements

    def hist_inter_union(self, hist1, hist2):
        inter = 0
        union = 0
        intersection = np.logical_and(hist1, hist2)
        logical_union = np.logical_or(hist1, hist2)

        # for i in range(len(hist1)):
        #     if hist1[i] > 0 and hist2[i] > 0:
        #         inter += 1
        #
        #     if hist1[i] > 0 or hist2[i] > 0:
        #         union += 1

        inter = np.sum(intersection)
        union = np.sum(logical_union)

        if union > 0:
            return inter / union
        else:
            return 0

    # def hist_init(self):
    #     secs = 24*60*60
    #     num_cells = int(secs / self.interval)
    #     histograms = {}
    #     histograms["space_id"] = np.array([0]*15)
    #     histograms["space_type"] = set()
    #     histograms["time_hist"] = np.array([0]*num_cells)
    #     histograms["space_hist"] = np.array([0]*num_cells)
    #     return histograms
    def merge_histograms(self, cluster1, cluster2):
        cluster3 = {}
        # ToDo: space is
        cluster3["space_id"] = np.array([0] * 15)
        cluster3["space_type"] = cluster1["space_type"].union(cluster2["space_type"])

        if len(cluster3["space_type"]) > 1:
            print("WARNING: [RegionGrowth] merge_histograms: More than 1 space type in a cluster")

        cluster3["space_hist"] = cluster1["space_hist"]+cluster2["space_hist"]
        cluster3["time_hist"] = cluster1["time_hist"] + cluster2["time_hist"]

        # ToDo: Previous and later activities
        return cluster3

    def cluster_by_time_hist(self, clusters, cluster_feat, th):
        num_clusters = len(clusters)

        edges = []
        num_edges = 0
        # prepare edges within clusters
        print("[RegionGrowth] cluster_by_time_hist: preparing edges....")
        for i in range(1, num_clusters):
            for j in range(i + 1, num_clusters+1):
                temp_edge = self.edge.copy()
                temp_edge['a'] = i
                temp_edge['b'] = j
                # temp_edge['w'] = self.first_pass_similarity(vec_a, vec_b)
                # temp_edge['w'] *= self.similarity_time_duration(cluster_feat[i], cluster_feat[j], 1)

                # if cluster_feat[i]["space_type"].issubset(cluster_feat[j]["space_type"]) or \
                #         cluster_feat[j]["space_type"].issubset(cluster_feat[i]["space_type"]):
                #     temp_edge['w'] = self.hist_inter_union(cluster_feat[i]["time_hist"], cluster_feat[j]["time_hist"])
                # else:
                #     temp_edge['w'] = 0
                if cluster_feat[i].space_type.issubset(cluster_feat[j].space_type) or \
                        cluster_feat[j].space_type.issubset(cluster_feat[i].space_type):
                    temp_edge['w'] = self.hist_inter_union(cluster_feat[i].time_hist, cluster_feat[j].time_hist)
                    edges.append(temp_edge)
                    num_edges += 1

        print("[RegionGrowth] cluster_by_time_hist: edges prepared")

        if DEBUG:
            print("number of edges to check: ", num_edges)
            print(edges)

        sorted_edges = sorted(edges, key=lambda k: k['w'], reverse=True)
        print("[RegionGrowth] cluster_by_time_hist: sorted edges by similarity")

        print("[RegionGrowth] cluster_by_time_hist: starting clustering....")
        obj_uf = UnionFind(num_clusters)
        for i in sorted_edges:
            # print(i['a'], i['b'])
            a = obj_uf.find(i['a'])
            b = obj_uf.find(i['b'])
            # print(i['w'])
            # print(i, a, b)
            if a != b and i['w'] >= th:
                # print(d)
                # d += 1
                obj_uf.union(a, b)
        print("[RegionGrowth] cluster_by_time_hist: clustering done. preparing clusters")

        # collect information for clusters
        cluster_elements = {}
        cluster_map = {}
        cluster_id = 0

        for i in range(num_clusters):
            cid = obj_uf.find(i)

            # if the cluster has not been allotted an id, allot it
            if cid not in cluster_map:
                cluster_map[cid] = cluster_id
                cluster_id += 1

            # If the cluster is not in the dictionary, initialize it as list
            if cluster_map[cid] not in cluster_elements:
                cluster_elements[cluster_map[cid]] = []

            cluster_elements[cluster_map[cid]] += clusters[i]
        print("[RegionGrowth] cluster_by_time_hist: clusters prepared")

        return cluster_elements

    # Update cluster features simultaneously
    # check distance with the updated features of the cluster
    # To do so:
    # - get pairs of clusters
    # - iterate through every pair
    # - - compare/calculate the similarity between the two bigger clusters (not individual cluster)
    # - - if similarity is less, merge the two clusters
    # - - update their features
    def cluster_by_time_hist2(self, clusters, cluster_feat):
        num_clusters = len(clusters)

        edges = []
        num_edges = 0
        # prepare edges within clusters
        for i in range(num_clusters - 1):
            for j in range(i + 1, num_clusters):
                temp_edge = self.edge.copy()
                temp_edge['a'] = i
                temp_edge['b'] = j

                if cluster_feat[i]["space_type"].issubset(cluster_feat[j]["space_type"]) or \
                        cluster_feat[j]["space_type"].issubset(cluster_feat[i]["space_type"]):
                    temp_edge['w'] = self.hist_inter_union(cluster_feat[i]["time_hist"],
                                                           cluster_feat[j]["time_hist"])
                else:
                    temp_edge['w'] = 0
                edges.append(temp_edge)
                num_edges += 1

        if DEBUG:
            print("number of edges to check: ", num_edges)
            print(edges)

        # sorted_edges = sorted(edges, key=lambda k: k['w'], reverse=True)
        obj_uf = UnionFind(num_clusters)
        for i in edges:
            # print(i['a'], i['b'])
            a = obj_uf.find(i['a'])
            b = obj_uf.find(i['b'])

            if a != b:
                # get cluster similarity
                if cluster_feat[a]["space_type"].issubset(cluster_feat[b]["space_type"]) or \
                        cluster_feat[b]["space_type"].issubset(cluster_feat[a]["space_type"]):
                    similarity_ab = self.hist_inter_union(cluster_feat[a]["time_hist"],
                                                          cluster_feat[b]["time_hist"])
                else:
                    similarity_ab = 0

                # is similar -> merge
                if similarity_ab >= 0.7:
                    par = obj_uf.union(a, b)
                    cluster_feat[par] = self.merge_histograms(cluster_feat[a], cluster_feat[b])
                    # update cluster features

        # collect information for clusters
        cluster_elements = {}
        cluster_map = {}
        cluster_id = 0

        for i in range(num_clusters):
            cid = obj_uf.find(i)

            # if the cluster has not been allotted an id, allot it
            if cid not in cluster_map:
                cluster_map[cid] = cluster_id
                cluster_id += 1

            # If the cluster is not in the dictionary, initialize it as list
            if cluster_map[cid] not in cluster_elements:
                cluster_elements[cluster_map[cid]] = []

            cluster_elements[cluster_map[cid]] += clusters[i]

        return cluster_elements
