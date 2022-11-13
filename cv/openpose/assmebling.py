"""
Bipartite Matching is used for doing the matching of keypoints

See algorithm for section 3.5

z[j1][j2] -> check if two detection candiates are valid

D_j = {
    [....] <- possible for partj (x,y)
    ...
    ...
    part j
}

Z = is the set of possible connection of parts in D_j

Basically - we want to maximize

E = 2d arr
for m in D:
    for n in D:
        E[m][n] = E_m_m * z[m][n] 
            <- E_m_m is the  weight confidence

https://en.wikipedia.org/wiki/Hungarian_algorithm is used for the matching
"""


"""
Based on readings of 
- https://yasenh.github.io/post/hungarian-algorithm-1/
- https://brilliant.org/wiki/hungarian-matching/
- http://www.columbia.edu/~cs2035/courses/ieor8100.F12/lec6.pdf
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

class Edge:
    def __init__(self, from_node, to_node, weight) -> None:
        self.from_node = from_node
        self.to_node = to_node
        self.weight = weight

class Graph:
    def __init__(self) -> None:
        self.nodes = {}
        self.from_edges = {}
        self.all_edges = {}
        self.edges = []
        self.nodes["Rob"] = len(self.nodes)
        self.nodes["Eva"] = len(self.nodes)
        self.nodes["Tom"] = len(self.nodes)

        self.nodes["Defense"] = len(self.nodes)
        self.nodes["Mid"] = len(self.nodes)
        self.nodes["Top"] = len(self.nodes)

        self.add_edge(
            Edge(self.nodes["Rob"], self.nodes["Top"], 4)
        )
        self.add_edge(
            Edge(self.nodes["Rob"], self.nodes["Defense"], 1)
        )
        self.add_edge(
            Edge(self.nodes["Eva"], self.nodes["Defense"], 6)
        )
        self.add_edge(
            Edge(self.nodes["Eva"], self.nodes["Mid"], 8)
        )
        self.add_edge(
            Edge(self.nodes["Tom"], self.nodes["Mid"], 6)
        )
        self.add_edge(
            Edge(self.nodes["Tom"], self.nodes["Top"], 1)
        )
        self.reverse_node = {
            value: key for key, value in self.nodes.items()
        }

    def node(self, i):
        return self.reverse_node[i]

    def add_edge(self, edge: Edge):
        self.edges.append(edge)
        from_node = edge.from_node
        to_node = edge.to_node

        if from_node not in self.from_edges:
            self.from_edges[from_node] = [edge]
        else:
            self.from_edges[from_node].append(edge)

        for node in [from_node, to_node]:
            if node not in self.all_edges:
                self.all_edges[node] = [edge]
            else:
                self.all_edges[node].append(edge)


class HungarianAlgorithm:
    def __init__(self) -> None:
        self.matched_graph = {}
        self.graph = Graph()
        # first iterate over all edges, and find the max from node
        self.visited = {}
        self.new_matched = {}

    def first_iteration(self):
        self.matched_graph = {}
        used_edges = {}
        for from_node in self.graph.from_edges:
            max_edge = None
            for i in self.graph.from_edges[from_node]:
                if max_edge is None:
                    max_edge = i
                elif max_edge.weight < i.weight:
                    max_edge = i
            if max_edge.to_node not in used_edges:
                used_edges[max_edge.to_node] = True
                self.matched_graph[from_node] = max_edge
            else:
                self.matched_graph[from_node] = None

    def loop_iteration(self):
        pass

    def augmentation_path(self, u):
        print(u)
        for edges in self.graph.all_edges[self.graph.nodes[u]]:
            #            print(edges.to_node)
            to_node = self.graph.node(edges.to_node)
            if self.visited.get(to_node, None) == None:
                self.visited[to_node] = True
                matched = self.new_matched.get(to_node, None)
                if matched is None or self.augmentation_path(matched):
                    self.new_matched[u] = to_node
                    self.new_matched[to_node] = u
                    return True
        return False
    
    def label_improving(self):
        """
        Updating the labels allows for new matching since the weights are changed.
        Labels = new weights.
        """
        weight = [
            i for i in self.graph.edges if i.from_node == self.graph.nodes["Tom"] and i.to_node == self.graph.nodes["Mid"]
        ][0].weight
        l_v = (self.matched_graph.get(self.graph.nodes['Eva'], Edge(None, None, 0)).weight)
        l_u = (self.matched_graph.get(self.graph.nodes['Tom'], None) or Edge(None, None, 0)).weight

        sigma = l_v + l_u - weight

        [
            i for i in self.graph.edges if i.from_node == self.graph.nodes["Tom"] and i.to_node == self.graph.nodes["Mid"]
        ][0].weight += sigma

        [
            i for i in self.graph.edges if i.from_node == self.graph.nodes["Eva"] and i.to_node == self.graph.nodes["Mid"]
        ][0].weight -= sigma

    def print_matched(self):
        # perfect match is denoted as every vertex having exactly one edge of matching
        for i in self.matched_graph:
            v = self.graph.node(
                self.matched_graph[i].to_node) if self.matched_graph[i] else "<none>"
            i = self.graph.node(i)
            print(f"{i} -> {v}")


if __name__ == "__main__":
    algorithm = HungarianAlgorithm()
    algorithm.first_iteration()
    algorithm.print_matched()

    algorithm.augmentation_path("Tom")
    algorithm.label_improving()

    algorithm.first_iteration()
    print(algorithm.new_matched)
    algorithm.print_matched()

    # Using scipy
    G = np.array([
                    [1, 0, 4],
                    [6, 8, 0],
                    [0, 6, 1],
                ])
    row_indices, col_indices = linear_sum_assignment(G, maximize=True)
    row_names = ['Rob', 'Eva', 'Tom']
    col_names = ['Defense', 'Mid', 'Top']
    edges = [((row_names[r], col_names[c]), G[r, c])
             for r, c in zip(row_indices, col_indices)]
    print(edges)
