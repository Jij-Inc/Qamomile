"""
Functions for mapping QUBO and related problems to unit disk graphs.

This module implements various functions for mapping QUBO (Quadratic Unconstrained
Binary Optimization) problems to weighted maximum independent set problems
on unit disk graphs.
"""

import numpy as np
import networkx as nx
import dataclasses
import typing as typ

from .core import SimpleCell, Node, GridGraph, WeightedNode

# import from qamomile.core
from qamomile.core.ising_qubo import IsingModel
from qamomile.core.post_process.local_search import IsingMatrix


def glue(grid, DI: int, DJ: int):
    """
    Glue multiple blocks into a whole.

    Args:
        grid: A 2D array of SimpleCell matrices
        DI: The overlap in rows between two adjacent blocks
        DJ: The overlap in columns between two adjacent blocks

    Returns:
        A matrix of SimpleCells created by gluing together the input grid
    """
    assert grid.shape[0] > 0 and grid.shape[1] > 0

    # Calculate the dimensions of the result matrix
    nrow = sum([grid[i, 0].shape[0] - DI for i in range(grid.shape[0])]) + DI
    ncol = sum([grid[0, j].shape[1] - DJ for j in range(grid.shape[1])]) + DJ

    # Create an empty result matrix
    result = np.full((nrow, ncol), None)
    for i in range(nrow):
        for j in range(ncol):
            result[i, j] = SimpleCell(occupied=False, weight=0.0)

    ioffset = 0
    for i in range(grid.shape[0]):
        joffset = 0
        for j in range(grid.shape[1]):
            chunk = grid[i, j]
            chunk_rows, chunk_cols = chunk.shape

            # Add the chunk to the result matrix
            for r in range(chunk_rows):
                for c in range(chunk_cols):
                    if chunk[r, c].occupied:
                        if not result[ioffset + r, joffset + c].occupied:
                            result[ioffset + r, joffset + c] = chunk[r, c]
                        else:
                            # Add weights if both cells are occupied
                            weight = (
                                result[ioffset + r, joffset + c].weight
                                + chunk[r, c].weight
                            )
                            result[ioffset + r, joffset + c] = SimpleCell(
                                occupied=True, weight=weight
                            )

            joffset += chunk_cols - DJ
            if j == grid.shape[1] - 1:
                ioffset += chunk_rows - DI

    return result


def crossing_lattice(g, ordered_vertices):
    """
    Create a crossing lattice from a graph.

    Args:
        g: A networkx graph
        ordered_vertices: List of vertices in desired order

    Returns:
        A CrossingLattice object
    """
    from .copyline import CrossingLattice, create_copylines

    # Use the create_copylines function from copyline.py
    lines = create_copylines(g, ordered_vertices)

    # Create a CrossingLattice object
    return CrossingLattice(len(ordered_vertices), len(ordered_vertices), lines, g)


def complete_graph(n):
    """Create a complete graph with n nodes."""
    return nx.complete_graph(n)


def render_grid(cl, delta=1):
    """
    Render a grid from a crossing lattice.

    Args:
        cl: A CrossingLattice object
        delta: delta value for the gadget weight, need to satisfy the constraint Eq. B3 of the paper

    Returns:
        A 2D numpy array of SimpleCell matrices
    """
    from .copyline import Block

    n = cl.graph.number_of_nodes()

    # Create an empty cell (with weight 0)
    def empty_cell():
        return SimpleCell(occupied=False, weight=0.0)

    # Create standard weight cells
    def weight_cell(weight):
        return SimpleCell(occupied=True, weight=weight)

    # Create grid
    grid = np.empty((n, n), dtype=object)

    # Initialize with empty matrices
    for i in range(n):
        for j in range(n):
            mat = np.full((4, 4), None)
            for r in range(4):
                for c in range(4):
                    mat[r, c] = empty_cell()
            grid[i, j] = mat

    # For each position in the grid
    for i in range(n):
        for j in range(n):
            # Try to get the block from the crossing lattice
            try:
                # Use 1-based indexing for the crossing lattice
                block = cl[i + 1, j + 1]
            except (IndexError, TypeError):
                # If out of bounds or other error, create a default block
                block = Block()

            # Determine if there's an edge between vertices
            has_edge = cl.graph.has_edge(i, j)

            # Check the block type and create the appropriate pattern
            if block.bottom != -1 and block.left != -1:
                # For blocks with both bottom and left connections
                if has_edge:
                    # Create QUBO gadget with the structure (for connected vertices):
                    # ⋅ ⋅ ● ⋅
                    # ● A B ⋅
                    # ⋅ C D ●
                    # ⋅ ● ⋅ ⋅

                    # Top node (weight 2 or 1 depending on top connection)
                    grid[i, j][0, 2] = weight_cell(
                        1.0 * delta if block.top == -1 else 2.0 * delta
                    )

                    # Left node (weight 2 or 1 depending on position)
                    grid[i, j][1, 0] = weight_cell(
                        1.0 * delta if j == 1 else 2.0 * delta
                    )

                    # Core QUBO nodes - all weight 4 initially (will adjust for coupling later)
                    grid[i, j][1, 1] = weight_cell(4.0 * delta)  # A
                    grid[i, j][1, 2] = weight_cell(4.0 * delta)  # B
                    grid[i, j][2, 1] = weight_cell(4.0 * delta)  # C
                    grid[i, j][2, 2] = weight_cell(4.0 * delta)  # D

                    # Right node (weight 2 or 1 depending on right connection)
                    grid[i, j][2, 3] = weight_cell(
                        1.0 * delta if block.right == -1 else 2.0 * delta
                    )

                    # Bottom node (weight 2 or 1 depending on position)
                    grid[i, j][3, 1] = weight_cell(
                        1.0 * delta if i == n - 2 else 2.0 * delta
                    )
                else:
                    # Pattern for non-connected vertices:
                    # ⋅ ⋅ ● ⋅
                    # ● 4 4 ⋅
                    # ⋅ 4 4 ●
                    # ⋅ ● ⋅ ⋅

                    # Top dot (weight 2 or 1)
                    grid[i, j][0, 2] = weight_cell(
                        1.0 * delta if block.top == -1 else 2.0 * delta
                    )

                    # Left dot (weight 1 or 2)
                    grid[i, j][1, 0] = weight_cell(
                        1.0 * delta if j == 1 else 2.0 * delta
                    )

                    # Core nodes - all weight 4
                    grid[i, j][1, 1] = weight_cell(4.0 * delta)
                    grid[i, j][1, 2] = weight_cell(4.0 * delta)
                    grid[i, j][2, 1] = weight_cell(4.0 * delta)
                    grid[i, j][2, 2] = weight_cell(4.0 * delta)

                    # Right dot (weight 1 or 2)
                    grid[i, j][2, 3] = weight_cell(
                        1.0 * delta if block.right == -1 else 2.0 * delta
                    )

                    # Bottom dot (weight 1 or 2)
                    grid[i, j][3, 1] = weight_cell(
                        1.0 * delta if i == n - 2 else 2.0 * delta
                    )

            elif block.top != -1 and block.right != -1:
                # L turn pattern
                # ⋅ ⋅ ● ⋅
                # ⋅ ⋅ ⋅ ●
                # ⋅ ⋅ ⋅ ⋅
                # ⋅ ⋅ ⋅ ⋅
                grid[i, j][0, 2] = weight_cell(2.0 * delta)
                grid[i, j][1, 3] = weight_cell(2.0 * delta)

            # No need to do anything for other block types (they remain empty)

    return grid


def post_process_grid(grid, h0, h1):
    """
    Process the grid to add weights for 0 and 1 states based on bias vector.

    In QUBO problems, we have a Hamiltonian of the form:
    E(z) = -∑(i<j) J_ij z_i z_j + ∑_i h_i z_i

    The bias vector h represents the linear terms h_i in the Hamiltonian.
    For each pin (representing a variable), we need to add the corresponding
    bias value to adjust the weight.

    Args:
        grid: Matrix of SimpleCells
        h0: Onsite bias for 0 state (h itself)
        h1: Onsite bias for 1 state (-h)

    Returns:
        Tuple of (GridGraph, pins list)
    """
    n = len(h0)

    # Extract the main part of the grid
    mat = grid[0:-4, 4:]

    # Add weight to top left
    if 0 <= 1 < mat.shape[0] and 0 <= 0 < mat.shape[1] and mat[1, 0].occupied:
        mat[1, 0] = SimpleCell(occupied=True, weight=mat[1, 0].weight + h0[0])

    # Add weight to bottom right
    if mat.shape[0] > 1 and mat.shape[1] > 3:
        bottom_right = (mat.shape[0] - 1, mat.shape[1] - 3)
        if mat[bottom_right].occupied:
            mat[bottom_right] = SimpleCell(
                occupied=True, weight=mat[bottom_right].weight + h1[-1]
            )

    # Process weights for all positions from 1 to n-1
    for j in range(n - 1):
        # Top side - apply h0[j+1]
        try:
            offset = 1 if mat[0, (j + 1) * 4 - 1].occupied else 2
            if mat[0, (j + 1) * 4 - offset].occupied:
                mat[0, (j + 1) * 4 - offset] = SimpleCell(
                    occupied=True,
                    weight=mat[0, (j + 1) * 4 - offset].weight + h0[j + 1],
                )
        except IndexError:
            # Skip if index is out of bounds
            pass

        # Right side - apply h1[j]
        try:
            offset = 1 if mat[(j + 1) * 4 - 1, mat.shape[1] - 1].occupied else 2
            if mat[(j + 1) * 4 - offset, mat.shape[1] - 1].occupied:
                mat[(j + 1) * 4 - offset, mat.shape[1] - 1] = SimpleCell(
                    occupied=True,
                    weight=mat[(j + 1) * 4 - offset, mat.shape[1] - 1].weight + h1[j],
                )
        except IndexError:
            # Skip if index is out of bounds
            pass

    # Generate GridGraph from matrix
    nodes = []
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j].occupied:
                nodes.append(Node(i, j, weight=mat[i, j].weight))

    gg = GridGraph(mat.shape, nodes, 1.5)

    # Find pins - first pin at position (1,0) in our grid
    pins = []
    try:
        pins.append(next(i for i, node in enumerate(nodes) if node.loc == (1, 0)))
    except StopIteration:
        # If exact position not found, find closest node to (1,0)
        closest_idx = 0
        closest_dist = float("inf")
        for i, node in enumerate(nodes):
            dist = (node.loc[0] - 1) ** 2 + node.loc[1] ** 2  # Distance to (1,0)
            if dist < closest_dist:
                closest_dist = dist
                closest_idx = i
        pins.append(closest_idx)

    # Other pins - corresponding to (0, i*4-1) or (0, i*4-2)
    for i in range(1, n):
        try:
            pin_idx = next(
                (
                    idx
                    for idx, node in enumerate(nodes)
                    if node.loc == (0, i * 4 - 1) or node.loc == (0, i * 4 - 2)
                )
            )
            pins.append(pin_idx)
        except StopIteration:
            # If position not found, find closest node
            target_loc = (0, i * 4 - 1)
            closest_idx = 0
            closest_dist = float("inf")
            for idx, node in enumerate(nodes):
                dist = (node.loc[0] - target_loc[0]) ** 2 + (
                    node.loc[1] - target_loc[1]
                ) ** 2
                if dist < closest_dist:
                    closest_dist = dist
                    closest_idx = idx
            pins.append(closest_idx)

    return gg, pins


class QUBOResult:
    """Result of mapping a QUBO problem to a unit disk graph."""

    def __init__(self, grid_graph, pins, mis_overhead):
        self.grid_graph = grid_graph
        self.pins = pins
        self.mis_overhead = mis_overhead

    def __str__(self):
        return (
            f"QUBOResult with {len(self.grid_graph.nodes)} nodes, {len(self.pins)} pins"
        )

    def qubo_grid_to_locations(self, scale=5.0) -> list[tuple[float, float]]:
        """
        Convert a QUBO grid graph to a list of tuples, each represent the node location.
        Args:
            qubo_result: QUBOResult object from map_qubo
        Returns:
            List of (x, y) tuples
        """
        return [
            (node.loc[0] * scale, node.loc[1] * scale) for node in self.grid_graph.nodes
        ]

    def qubo_result_to_weights(self) -> list[float]:
        """
        Convert a QUBOResult to a list of weights for each node.

        Args:
            qubo_result: QUBOResult object from map_qubo

        Returns:
            List of weights for each node in the grid graph.
        """
        return [node.weight for node in self.grid_graph.nodes]


def map_config_back(res, cfg, binary=False):
    """
    Map a configuration back from the unit disk graph to the original graph.

    Args:
        res: A QUBOResult, WMISResult, or similar result object
        cfg: Configuration vector from the unit disk graph
        binary: If True, return {0,1} variables, otherwise return {-1,1} variables

    Returns:
        Configuration for the original problem
    """
    if not hasattr(res, "pins") or len(res.pins) == 0:
        return []

    # Safety check for cfg size
    if len(cfg) < max(res.pins) + 1:
        print(
            f"Warning: Configuration size {len(cfg)} is smaller than required {max(res.pins) + 1}."
        )
        # Pad the configuration with zeros
        cfg = np.pad(cfg, (0, max(res.pins) + 1 - len(cfg)), "constant")

    if isinstance(res, QUBOResult):
        # For QUBO problems, we need to map the configurations according to the mapping
        # If node i is selected (cfg[i] = 1), the corresponding QUBO variable is -1
        # If node i is not selected (cfg[i] = 0), the corresponding QUBO variable is 1
        if binary:
            # Return {0,1} variables
            return [1 - cfg[i] for i in res.pins]
        else:
            # Return {-1,1} variables for standard QUBO formulation
            return [1 - 2 * cfg[i] for i in res.pins]
    else:  # WMISResult
        if binary:
            return [cfg[i] for i in res.pins]
        else:
            return [2 * cfg[i] - 1 for i in res.pins]


def qubo_result_to_networkx(qubo_result):
    """
    Convert a QUBOResult to a networkx graph for MWIS solving.

    Args:
        qubo_result: QUBOResult object from map_qubo

    Returns:
        networkx Graph with weights on nodes
    """
    # Create an empty graph
    G = nx.Graph()

    # Add nodes with positions and weights from the grid graph
    for i, node in enumerate(qubo_result.grid_graph.nodes):
        G.add_node(i, pos=node.loc, weight=node.weight)

    # Add edges based on unit disk constraint
    radius = qubo_result.grid_graph.radius
    for i in range(len(qubo_result.grid_graph.nodes)):
        for j in range(i + 1, len(qubo_result.grid_graph.nodes)):
            node1 = qubo_result.grid_graph.nodes[i]
            node2 = qubo_result.grid_graph.nodes[j]
            dist = np.sqrt(sum((np.array(node1.loc) - np.array(node2.loc)) ** 2))
            if dist <= radius:
                G.add_edge(i, j)

    return G


def map_qubo(J, h, delta_overwrite=None):
    """
    Map a QUBO problem to a weighted MIS problem on a defected King's graph.

    A QUBO problem is defined by the Hamiltonian:
    E(z) = -∑(i<j) J_ij z_i z_j + ∑_i h_i z_i

    The mapping creates a crossing lattice with QUBO gadgets at each crossing.
    Each QUBO gadget has the structure:
    ⋅ ⋅ ● ⋅
    ● A B ⋅
    ⋅ C D ●
    ⋅ ● ⋅ ⋅

    Where:
    - A = -J_{ij} + 4
    - B = J_{ij} + 4
    - C = J_{ij} + 4
    - D = -J_{ij} + 4

    Args:
        J: Coupling matrix (must be symmetric)
        h: Vector of onsite terms

    Returns:
        QUBOResult object containing the mapped problem
    """
    n = len(h)
    assert J.shape == (n, n), f"J shape {J.shape} doesn't match h length {n}"

    # Create crossing lattice
    g = complete_graph(n)
    d = crossing_lattice(g, list(range(n)))

    # Check the weight constraint Equation (B3): delta > max_i(sum_j abs(J_{ij}))
    threshold = np.max(np.sum(np.abs(J), axis=1))
    max_h = np.max(np.abs(h))
    threshold = max(threshold, max_h)
    delta = 1.5 * threshold
    if delta_overwrite is not None:
        print(f"Overwriting delta from {delta} to {delta_overwrite}")
        delta = delta_overwrite

    # Render grid
    chunks = render_grid(d, delta)

    # Add coupling - update QUBO gadget weights based on J values
    for i in range(n - 1):
        for j in range(i + 1, n):
            a = J[i, j]
            if abs(a) > 1e-10:  # Skip zero couplings
                # Update the QUBO gadget weights according to the formula:
                # A = -J_{ij} + 4
                # B = J_{ij} + 4
                # C = J_{ij} + 4
                # D = -J_{ij} + 4

                # Make sure the chunk has the right cells
                if (
                    chunks[i, j][1, 1].occupied
                    and chunks[i, j][1, 2].occupied
                    and chunks[i, j][2, 1].occupied
                    and chunks[i, j][2, 2].occupied
                ):

                    # Update A weight (-J + 4)
                    chunks[i, j][1, 1] = SimpleCell(
                        occupied=True, weight=-a + 4.0 * delta
                    )

                    # Update B weight (J + 4)
                    chunks[i, j][1, 2] = SimpleCell(
                        occupied=True, weight=a + 4.0 * delta
                    )

                    # Update C weight (J + 4)
                    chunks[i, j][2, 1] = SimpleCell(
                        occupied=True, weight=a + 4.0 * delta
                    )

                    # Update D weight (-J + 4)
                    chunks[i, j][2, 2] = SimpleCell(
                        occupied=True, weight=-a + 4.0 * delta
                    )

    # Glue the chunks together
    grid = glue(chunks, 0, 0)

    # Add one extra row and process grid to handle bias terms
    gg, pins = post_process_grid(grid, h, -h)

    # Calculate overhead based on graph structure
    mis_overhead = (n - 1) * n * 4 + n - 4

    # Check if we have a valid grid graph
    if len(gg.nodes) == 0:
        print("Warning: Grid graph has no nodes. Creating dummy nodes for testing.")
        # Create some nodes at arbitrary positions with weights from h
        dummy_nodes = []
        for i in range(n):
            dummy_nodes.append(Node(i, 0, h[i]))

        # Create new GridGraph
        gg = GridGraph((n, 1), dummy_nodes, 1.5)

        # Update pins to point to these nodes
        pins = list(range(n))

    # Return the result with the grid graph and pins
    return QUBOResult(gg, pins, mis_overhead)


@dataclasses.dataclass
class Ising_UnitDiskGraph:
    """
    A representation of an Ising model as a unit disk graph suitable for neutral atom quantum computers.

    This class wraps the UDM module to map QUBO/Ising problems to unit disk graphs.
    """

    ising_model: IsingModel
    _ising_matrix: IsingMatrix = None
    _qubo_result: typ.Optional[QUBOResult] = None
    _networkx_graph: typ.Optional[object] = None
    _delta: typ.Optional[float] = None

    def __post_init__(self):
        """Initialize the unit disk graph mapping after construction."""
        self._create_mapping()

    def _create_mapping(self):
        """Create the unit disk graph mapping from the Ising model."""
        # Convert the Ising model to IsingMatrix
        self._ising_matrix = IsingMatrix().to_ising_matrix(self.ising_model)

        # Initialize J matrix and h vector
        J = self._ising_matrix.quad
        h = self._ising_matrix.linear

        # Calculate delta parameter for weight scaling
        self._delta = 1.5 * max(np.max(np.abs(h)), np.max(np.abs(J)))

        # Map the QUBO problem to a unit disk graph
        self._qubo_result = map_qubo(J, h, self._delta)

        # Convert to a NetworkX graph for visualization and analysis
        self._networkx_graph = qubo_result_to_networkx(self._qubo_result)

    @property
    def qubo_result(self) -> QUBOResult:
        """Get the QUBOResult object from the UDM module."""
        return self._qubo_result

    @property
    def networkx_graph(self) -> object:
        """Get the NetworkX graph representation."""
        return self._networkx_graph

    @property
    def pins(self) -> list:
        """Get the list of pins (indices of nodes corresponding to original variables)."""
        if self._qubo_result:
            return self._qubo_result.pins
        return []

    @property
    def nodes(self) -> list:
        """Get the list of nodes in the unit disk graph."""
        if self._qubo_result and hasattr(self._qubo_result.grid_graph, "nodes"):
            return self._qubo_result.grid_graph.nodes
        return []

    @property
    def delta(self) -> float:
        """Get the delta parameter used for scaling the weights."""
        return self._delta
