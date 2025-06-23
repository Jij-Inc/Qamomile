import networkx as nx
from typing import List, Tuple


def unit_disk_graph(locs: List[Tuple[float, float]], unit: float) -> nx.Graph:
    """
    Create a unit disk graph given a list of 2D coordinates and a distance threshold.

    Nodes are indexed 0..len(locs)-1. An undirected edge is added between nodes i and j
    if the Euclidean distance between locs[i] and locs[j] is strictly less than unit.
    """
    g = nx.Graph()
    n = len(locs)
    g.add_nodes_from(range(n))

    unit_sq = unit * unit
    for i in range(n):
        xi, yi = locs[i]
        for j in range(i + 1, n):
            xj, yj = locs[j]
            dx = xi - xj
            dy = yi - yj
            if dx * dx + dy * dy < unit_sq:
                g.add_edge(i, j)

    return g
