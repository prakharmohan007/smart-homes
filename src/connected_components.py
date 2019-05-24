class UnionFind:
    def __init__(self, num_nodes):
        self.cluster = {"rank": 0, "size": 0, "parent": 0}
        self.elts = []
        for i in range(num_nodes):
            temp_cluster = self.cluster.copy()
            temp_cluster["rank"] = 0
            temp_cluster["size"] = 1
            temp_cluster["parent"] = i
            self.elts.append(temp_cluster)
        self.num_nodes = num_nodes

    def find(self, x):
        y = x
        while y != self.elts[y]["parent"]:
            y = self.elts[y]["parent"]
        self.elts[x]["parent"] = y
        return y

    # Unions the two components
    # returns (a, b) where a is the parent and b is the child. i.e. b is attached to a
    def union(self, x, y):
        if self.elts[x]["rank"] > self.elts[y]["rank"]:
            self.elts[y]["parent"] = x
            self.elts[x]["size"] += self.elts[y]["size"]
            parent = x
        else:
            self.elts[x]["parent"] = y
            self.elts[y]["size"] += self.elts[x]["size"]
            if self.elts[x]["size"] == self.elts[y]["size"]:
                self.elts[y]["Rank"] += 1
            parent = y
        self.num_nodes -= 1
        return parent
