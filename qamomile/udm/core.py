import networkx as nx
import numpy as np
from typing import List, Tuple, Dict, Any, Union, Optional
import math
from dataclasses import dataclass
from .utils import unit_disk_graph

# Constants
SHOW_WEIGHT = False


# UNWEIGHTED type - a singleton to represent the constant 1 for unweighted cases
class UNWEIGHTED:
    """A class representing the constant 1 for unweighted graphs."""

    def __str__(self):
        return "1"

    def __repr__(self):
        return "1"


# A singleton instance to use throughout the code
UNWEIGHTED_INSTANCE = UNWEIGHTED()


# Cell classes
class AbstractCell:
    """Abstract base class for all cell types."""

    def __init__(self, occupied: bool = True, weight: Any = UNWEIGHTED_INSTANCE):
        self.occupied = occupied
        self.weight = weight

    def __str__(self):
        return self.format_cell(show_weight=SHOW_WEIGHT)

    def __repr__(self):
        return self.__str__()

    def format_cell(self, show_weight: bool = False) -> str:
        """Format cell for display."""
        if self.occupied:
            return str(self.weight) if show_weight else "●"
        else:
            return "⋅"

    @property
    def is_empty(self) -> bool:
        """Check if cell is empty."""
        return not self.occupied

    def get_weight(self):
        """Get the weight of the cell."""
        return self.weight


class SimpleCell(AbstractCell):
    """A simple cell implementation."""

    @classmethod
    def create_empty(cls, weight_type=UNWEIGHTED):
        """Create an empty cell of a given weight type."""
        if weight_type == UNWEIGHTED:
            return cls(occupied=False, weight=UNWEIGHTED_INSTANCE)
        else:
            return cls(occupied=False, weight=1)

    def __add__(self, other):
        """Add two cells together."""
        if not isinstance(other, SimpleCell):
            return NotImplemented

        if not self.occupied:
            return other
        if not other.occupied:
            return self

        # Both occupied
        if isinstance(self.weight, UNWEIGHTED) and isinstance(other.weight, UNWEIGHTED):
            return SimpleCell(occupied=True, weight=UNWEIGHTED_INSTANCE)
        else:
            # Convert UNWEIGHTED to 1 if needed
            w1 = 1 if isinstance(self.weight, UNWEIGHTED) else self.weight
            w2 = 1 if isinstance(other.weight, UNWEIGHTED) else other.weight
            return SimpleCell(occupied=True, weight=w1 + w2)

    def __sub__(self, other):
        """Subtract a cell from another."""
        if not isinstance(other, SimpleCell):
            return NotImplemented

        if not self.occupied:
            return SimpleCell(occupied=other.occupied, weight=-other.weight)
        if not other.occupied:
            return self

        # Both occupied
        if isinstance(self.weight, UNWEIGHTED) and isinstance(other.weight, UNWEIGHTED):
            return SimpleCell(occupied=True, weight=0)
        else:
            # Convert UNWEIGHTED to 1 if needed
            w1 = 1 if isinstance(self.weight, UNWEIGHTED) else self.weight
            w2 = 1 if isinstance(other.weight, UNWEIGHTED) else other.weight
            return SimpleCell(occupied=True, weight=w1 - w2)

    def __neg__(self):
        """Negate a cell."""
        if not self.occupied:
            return self

        if isinstance(self.weight, UNWEIGHTED):
            return SimpleCell(occupied=True, weight=-1)
        else:
            return SimpleCell(occupied=True, weight=-self.weight)


# Type aliases for clarity
WeightedSimpleCell = SimpleCell  # SimpleCell with numeric weight
UnWeightedSimpleCell = SimpleCell  # SimpleCell with UNWEIGHTED weight


# Node class for graph representation
class Node:
    """A node in a grid graph with a location and weight."""

    def __init__(self, *args, **kwargs):
        # Handle different initialization patterns
        if len(args) == 1 and isinstance(args[0], tuple) and len(args[0]) == 2:
            # Node((x, y))
            self.loc = args[0]
            self.weight = kwargs.get("weight", UNWEIGHTED_INSTANCE)
        elif len(args) == 2 and all(isinstance(arg, (int, float)) for arg in args):
            # Node(x, y)
            self.loc = (int(args[0]), int(args[1]))
            self.weight = kwargs.get("weight", UNWEIGHTED_INSTANCE)
        elif len(args) == 3 and all(
            isinstance(args[i], (int, float)) for i in range(2)
        ):
            # Node(x, y, weight)
            self.loc = (int(args[0]), int(args[1]))
            self.weight = args[2]
        elif len(args) == 1 and isinstance(args[0], list) and len(args[0]) == 2:
            # Node([x, y])
            self.loc = (args[0][0], args[0][1])
            self.weight = kwargs.get("weight", UNWEIGHTED_INSTANCE)
        else:
            raise ValueError("Invalid arguments for Node initialization")

    def __getitem__(self, idx):
        """Get x or y coordinate by index."""
        return self.loc[idx]

    def __iter__(self):
        """Iterate over coordinates."""
        return iter(self.loc)

    def __len__(self):
        """Length is always 2 (x and y coordinates)."""
        return 2

    def __str__(self):
        """String representation."""
        if isinstance(self.weight, UNWEIGHTED):
            return f"Node({self.loc[0]}, {self.loc[1]})"
        else:
            return f"Node({self.loc[0]}, {self.loc[1]}, {self.weight})"

    def __repr__(self):
        return self.__str__()

    def get_xy(self):
        """Get the location as a tuple."""
        return self.loc

    def change_xy(self, new_loc):
        """Create a new node with the same weight but different location."""
        return Node(new_loc, weight=self.weight)

    def offset(self, xy_offset):
        """Create a new node offset by the given amount."""
        new_loc = (self.loc[0] + xy_offset[0], self.loc[1] + xy_offset[1])
        return self.change_xy(new_loc)


# Type aliases for Node
WeightedNode = Node  # Node with numeric weight
UnWeightedNode = Node  # Node with UNWEIGHTED weight


class GridGraph:
    """A graph embedded in a 2D grid with a unit disk connectivity radius."""

    def __init__(self, size, nodes, radius):
        """
        Initialize a GridGraph.

        Args:
            size: Tuple of (height, width) of the grid
            nodes: List of Node objects
            radius: The radius for determining connectivity in the unit disk graph
        """
        self.size = size
        self.nodes = nodes
        self.radius = float(radius)

    def __str__(self):
        """String representation of the grid graph."""
        result = [f"{self.__class__.__name__} (radius = {self.radius})"]
        grid = self.cell_matrix()

        for i in range(self.size[0]):
            row = []
            for j in range(self.size[1]):
                row.append(grid[i][j].format_cell(show_weight=SHOW_WEIGHT))
            result.append(" ".join(row))

        return "\n".join(result)

    def get_size(self):
        """Get the size of the grid."""
        return self.size

    def get_graph_and_weights(self):
        """Get the graph and weights as separate objects."""
        locs = [node.loc for node in self.nodes]
        weights = [node.weight for node in self.nodes]
        return unit_disk_graph(locs, self.radius), weights

    def to_networkx(self):
        """Convert to a NetworkX graph."""
        locs = [node.loc for node in self.nodes]
        return unit_disk_graph(locs, self.radius)

    def get_coordinates(self):
        """Get all node coordinates."""
        return [node.loc for node in self.nodes]

    def neighbors(self, i):
        """Get neighbors of node i based on the unit disk distance."""
        return [
            j
            for j in range(len(self.nodes))
            if i != j and self.distance(self.nodes[i], self.nodes[j]) <= self.radius
        ]

    @staticmethod
    def distance(n1, n2):
        """Calculate Euclidean distance between two nodes."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(n1.loc, n2.loc)))

    def num_vertices(self):
        """Get the number of vertices in the graph."""
        return len(self.nodes)

    def vertices(self):
        """Get all vertex indices."""
        return range(self.num_vertices())

    def cell_matrix(self):
        """Convert the grid graph to a matrix of cells."""
        # Create an empty matrix
        matrix = [
            [
                SimpleCell.create_empty(
                    UNWEIGHTED
                    if isinstance(self.nodes[0].weight, UNWEIGHTED)
                    else type(self.nodes[0].weight)
                )
                for _ in range(self.size[1])
            ]
            for _ in range(self.size[0])
        ]

        # Fill in the nodes
        for node in self.nodes:
            i, j = node.loc
            assert 0 <= i < self.size[0], f"i={i} is out of bounds [0, {self.size[0]})"
            assert 0 <= j < self.size[1], f"j={j} is out of bounds [0, {self.size[1]})"
            matrix[i][j] = SimpleCell(occupied=True, weight=node.weight)

        return matrix

    @classmethod
    def from_cell_matrix(cls, matrix, radius):
        """Create a GridGraph from a matrix of cells."""
        nodes = []
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                cell = matrix[i][j]
                if not cell.is_empty:
                    nodes.append(Node((i, j), weight=cell.weight))

        return cls((len(matrix), len(matrix[0])), nodes, radius)
