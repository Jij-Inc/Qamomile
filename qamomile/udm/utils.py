import networkx as nx
import numpy as np
from typing import List, Tuple, Set, Dict, Any, Union, Optional

def simple_graph_from_edgelist(edgelist: List[Tuple[int, int]]) -> nx.Graph:
    """Create a simple graph from an edge list."""
    g = nx.Graph()
    for i, j in edgelist:
        g.add_edge(i, j)
    return g

# Geometric transformations
def rotate90(loc: Tuple[int, int]) -> Tuple[int, int]:
    """Rotate coordinates 90 degrees counterclockwise."""
    return (-loc[1], loc[0])

def reflectx(loc: Tuple[int, int]) -> Tuple[int, int]:
    """Reflect coordinates across x-axis."""
    return (loc[0], -loc[1])

def reflecty(loc: Tuple[int, int]) -> Tuple[int, int]:
    """Reflect coordinates across y-axis."""
    return (-loc[0], loc[1])

def reflectdiag(loc: Tuple[int, int]) -> Tuple[int, int]:
    """Reflect coordinates across main diagonal."""
    return (-loc[1], -loc[0])

def reflectoffdiag(loc: Tuple[int, int]) -> Tuple[int, int]:
    """Reflect coordinates across off-diagonal."""
    return (loc[1], loc[0])

# Apply transformations with a center
def apply_transform(loc: Tuple[int, int], center: Tuple[int, int], transform_func):
    """Apply a transformation function with respect to a center point."""
    dx, dy = transform_func((loc[0] - center[0], loc[1] - center[1]))
    return (center[0] + dx, center[1] + dy)

def rotate90_around(loc: Tuple[int, int], center: Tuple[int, int]) -> Tuple[int, int]:
    """Rotate coordinates 90 degrees counterclockwise around a center point."""
    return apply_transform(loc, center, rotate90)

def reflectx_around(loc: Tuple[int, int], center: Tuple[int, int]) -> Tuple[int, int]:
    """Reflect coordinates across x-axis passing through a center point."""
    return apply_transform(loc, center, reflectx)

def reflecty_around(loc: Tuple[int, int], center: Tuple[int, int]) -> Tuple[int, int]:
    """Reflect coordinates across y-axis passing through a center point."""
    return apply_transform(loc, center, reflecty)

def reflectdiag_around(loc: Tuple[int, int], center: Tuple[int, int]) -> Tuple[int, int]:
    """Reflect coordinates across main diagonal passing through a center point."""
    return apply_transform(loc, center, reflectdiag)

def reflectoffdiag_around(loc: Tuple[int, int], center: Tuple[int, int]) -> Tuple[int, int]:
    """Reflect coordinates across off-diagonal passing through a center point."""
    return apply_transform(loc, center, reflectoffdiag)

def unit_disk_graph(locs: List[Tuple[int, int]], unit: float) -> nx.Graph:
    """
    Create a unit disk graph with locations specified by locs and unit distance.
    
    A unit disk graph connects vertices if they are less than the unit distance apart.
    """
    n = len(locs)
    g = nx.Graph()
    
    # Add all nodes
    for i in range(n):
        g.add_node(i)
    
    # Connect nodes if their distance is less than unit
    for i in range(n):
        for j in range(i+1, n):
            if sum((a - b) ** 2 for a, b in zip(locs[i], locs[j])) < unit ** 2:
                g.add_edge(i, j)
    
    return g

def is_independent_set(g: nx.Graph, config: List[int]) -> bool:
    """
    Check if a configuration represents an independent set in the graph.
    
    An independent set has no adjacent vertices both in the set.
    """
    for u, v in g.edges():
        if config[u] == 1 and config[v] == 1:
            return False
    return True

def is_diff_by_const(arr1: np.ndarray, arr2: np.ndarray) -> Tuple[bool, float]:
    """
    Check if two arrays differ by a constant.
    
    Returns (True, constant_diff) if arrays differ by a constant,
    otherwise returns (False, 0).
    """
    diff = None
    
    for a, b in zip(arr1, arr2):
        # Handle infinities
        if np.isinf(a) and np.isinf(b):
            continue
        if np.isinf(a) or np.isinf(b):
            return False, 0
            
        # Check for constant difference
        if diff is None:
            diff = a - b
        elif diff != a - b:
            return False, 0
            
    return True, diff if diff is not None else 0

def is_unit_disk_graph(grid_graph) -> bool:
    """
    Check if a grid graph is a valid unit disk graph.
    
    A unit disk graph is a graph where:
    1. Nodes are placed in a Euclidean space
    2. Two nodes are connected by an edge if and only if their distance is at most the radius
    
    For our grid graphs, we check that nodes are connected if their Euclidean distance is less 
    than the grid graph's radius, and not connected otherwise.
    
    Args:
        grid_graph: A GridGraph object with nodes having location attributes
        
    Returns:
        True if the graph is a valid unit disk graph, False otherwise
    """
    # Get the unit distance from the grid_graph (called 'radius' in GridGraph)
    # Convert the grid graph to a NetworkX graph
    nx_graph = grid_graph.to_networkx()
    
    # Check all pairs of nodes
    for i, node_i in enumerate(grid_graph.nodes):
        loc_i = node_i.loc
        for j, node_j in enumerate(grid_graph.nodes[i+1:], i+1):
            loc_j = node_j.loc
            
            # Calculate Euclidean distance
            dist_squared = sum((a - b) ** 2 for a, b in zip(loc_i, loc_j))
            
            # Check if the nodes should be connected based on distance
            should_be_connected = dist_squared <= grid_graph.radius ** 2
            
            # Check if there is an edge between the nodes in the NetworkX graph
            is_connected = nx_graph.has_edge(i, j)
            
            # If there's a mismatch, this is not a valid unit disk graph
            if should_be_connected != is_connected:
                return False
                
    return True