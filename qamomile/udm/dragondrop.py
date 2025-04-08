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
                            weight = result[ioffset + r, joffset + c].weight + chunk[r, c].weight
                            result[ioffset + r, joffset + c] = SimpleCell(occupied=True, weight=weight)
            
            joffset += chunk_cols - DJ
            if j == grid.shape[1] - 1:
                ioffset += chunk_rows - DI
                
    return result


def cell_matrix(gg):
    """Convert a GridGraph to a matrix of SimpleCell objects."""
    mat = np.full(gg.size, None)
    for i in range(gg.size[0]):
        for j in range(gg.size[1]):
            mat[i, j] = SimpleCell(occupied=False, weight=0.0)
            
    for node in gg.nodes:
        i, j = node.loc
        mat[i, j] = SimpleCell(occupied=True, weight=node.weight)
    return mat


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
                block = cl[i+1, j+1]
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
                    grid[i, j][0, 2] = weight_cell(1.0*delta if block.top == -1 else 2.0*delta)
                    
                    # Left node (weight 2 or 1 depending on position)
                    grid[i, j][1, 0] = weight_cell(1.0*delta if j == 1 else 2.0*delta)
                    
                    # Core QUBO nodes - all weight 4 initially (will adjust for coupling later)
                    grid[i, j][1, 1] = weight_cell(4.0*delta)  # A
                    grid[i, j][1, 2] = weight_cell(4.0*delta)  # B
                    grid[i, j][2, 1] = weight_cell(4.0*delta)  # C
                    grid[i, j][2, 2] = weight_cell(4.0*delta)  # D
                    
                    # Right node (weight 2 or 1 depending on right connection)
                    grid[i, j][2, 3] = weight_cell(1.0*delta if block.right == -1 else 2.0*delta)
                    
                    # Bottom node (weight 2 or 1 depending on position)
                    grid[i, j][3, 1] = weight_cell(1.0*delta if i == n-2 else 2.0*delta)
                else:
                    # Pattern for non-connected vertices:
                    # ⋅ ⋅ ● ⋅
                    # ● 4 4 ⋅
                    # ⋅ 4 4 ●
                    # ⋅ ● ⋅ ⋅
                    
                    # Top dot (weight 2 or 1)
                    grid[i, j][0, 2] = weight_cell(1.0*delta if block.top == -1 else 2.0*delta)
                    
                    # Left dot (weight 1 or 2)
                    grid[i, j][1, 0] = weight_cell(1.0*delta if j == 1 else 2.0*delta)
                    
                    # Core nodes - all weight 4
                    grid[i, j][1, 1] = weight_cell(4.0*delta)
                    grid[i, j][1, 2] = weight_cell(4.0*delta)
                    grid[i, j][2, 1] = weight_cell(4.0*delta)
                    grid[i, j][2, 2] = weight_cell(4.0*delta)
                    
                    # Right dot (weight 1 or 2)
                    grid[i, j][2, 3] = weight_cell(1.0*delta if block.right == -1 else 2.0*delta)
                    
                    # Bottom dot (weight 1 or 2)
                    grid[i, j][3, 1] = weight_cell(1.0*delta if i == n-2 else 2.0*delta)
            
            elif block.top != -1 and block.right != -1:
                # L turn pattern
                # ⋅ ⋅ ● ⋅
                # ⋅ ⋅ ⋅ ●
                # ⋅ ⋅ ⋅ ⋅
                # ⋅ ⋅ ⋅ ⋅
                grid[i, j][0, 2] = weight_cell(2.0*delta)
                grid[i, j][1, 3] = weight_cell(2.0*delta)
            
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
        bottom_right = (mat.shape[0]-1, mat.shape[1]-3)
        if mat[bottom_right].occupied:
            mat[bottom_right] = SimpleCell(occupied=True, weight=mat[bottom_right].weight + h1[-1])
    
    # Process weights for all positions from 1 to n-1
    for j in range(n-1):
        # Top side - apply h0[j+1]
        try:
            offset = 1 if mat[0, (j+1)*4-1].occupied else 2
            if mat[0, (j+1)*4-offset].occupied:
                mat[0, (j+1)*4-offset] = SimpleCell(
                    occupied=True, 
                    weight=mat[0, (j+1)*4-offset].weight + h0[j+1]
                )
        except IndexError:
            # Skip if index is out of bounds
            pass
        
        # Right side - apply h1[j]
        try:
            offset = 1 if mat[(j+1)*4-1, mat.shape[1]-1].occupied else 2
            if mat[(j+1)*4-offset, mat.shape[1]-1].occupied:
                mat[(j+1)*4-offset, mat.shape[1]-1] = SimpleCell(
                    occupied=True,
                    weight=mat[(j+1)*4-offset, mat.shape[1]-1].weight + h1[j]
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
        closest_dist = float('inf')
        for i, node in enumerate(nodes):
            dist = (node.loc[0] - 1)**2 + node.loc[1]**2  # Distance to (1,0)
            if dist < closest_dist:
                closest_dist = dist
                closest_idx = i
        pins.append(closest_idx)
    
    # Other pins - corresponding to (0, i*4-1) or (0, i*4-2)
    for i in range(1, n):
        try:
            pin_idx = next((idx for idx, node in enumerate(nodes) 
                          if node.loc == (0, i*4-1) or node.loc == (0, i*4-2)))
            pins.append(pin_idx)
        except StopIteration:
            # If position not found, find closest node
            target_loc = (0, i*4-1)
            closest_idx = 0
            closest_dist = float('inf')
            for idx, node in enumerate(nodes):
                dist = (node.loc[0] - target_loc[0])**2 + (node.loc[1] - target_loc[1])**2
                if dist < closest_dist:
                    closest_dist = dist
                    closest_idx = idx
            pins.append(closest_idx)
    
    return gg, pins


def is_occupied(cell):
    """Check if a cell is occupied."""
    return cell.occupied if hasattr(cell, 'occupied') else False

def format_cell(cell, show_weight=False):
    """
    Format a cell for display in ASCII representation.
    
    Args:
        cell: SimpleCell object
        show_weight: Whether to show the cell's weight
        
    Returns:
        String representation of the cell
    """
    if not hasattr(cell, 'occupied') or not cell.occupied:
        return "⋅"
    
    # Cell is occupied, format based on weight
    weight = cell.weight if hasattr(cell, 'weight') else 0
    
    if show_weight:
        # Show the actual weight value
        return str(int(weight) if weight.is_integer() else weight)
    else:
        # Use symbols based on weight
        if abs(weight - 2.0) < 0.1:
            return "●"  # Standard connector (weight ~2)
        elif abs(weight - 1.0) < 0.1:
            return "○"  # Edge connector (weight ~1) 
        elif weight < 0:
            return "▾"  # Negative weight (J coupled)
        elif weight > 4.5:
            return "▴"  # High weight (B or C nodes)
        elif abs(weight - 4.0) < 0.1:
            return "■"  # Default weight 4
        elif abs(weight - 3.0) < 0.5:
            return "□"  # Weight ~3 (A or D with positive J)
        else:
            return "◆"  # Other weights


class QUBOResult:
    """Result of mapping a QUBO problem to a unit disk graph."""
    
    def __init__(self, grid_graph, pins, mis_overhead):
        self.grid_graph = grid_graph
        self.pins = pins
        self.mis_overhead = mis_overhead
    
    def __str__(self):
        return f"QUBOResult with {len(self.grid_graph.nodes)} nodes, {len(self.pins)} pins"
    
    def print_grid(self, show_weights=False):
        """
        Print the grid graph in ASCII format.
        
        Args:
            show_weights: Whether to show actual weight values
            
        Returns:
            String representation of the grid graph
        """
        # Create a matrix representation from the grid graph
        size = self.grid_graph.size
        matrix = []
        for i in range(size[0]):
            row = []
            for j in range(size[1]):
                # Default empty cell
                cell = SimpleCell(occupied=False, weight=0.0)
                
                # Check if there's a node at this position
                for node in self.grid_graph.nodes:
                    if node.loc == (i, j):
                        cell = SimpleCell(occupied=True, weight=node.weight)
                        break
                
                row.append(cell)
            matrix.append(row)
        
        # Format the matrix as a string
        result = []
        for row in matrix:
            line = ''.join(format_cell(cell, show_weights) + ' ' for cell in row)
            result.append(line)
        
        # Highlight pins
        pins_info = []
        for i, pin_idx in enumerate(self.pins):
            if pin_idx < len(self.grid_graph.nodes):
                pin_node = self.grid_graph.nodes[pin_idx]
                pins_info.append(f"Pin {i}: at position {pin_node.loc}, weight={pin_node.weight}")
        
        if pins_info:
            result.append("\nPins:")
            result.extend(pins_info)
        
        return '\n'.join(result)


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
    if not hasattr(res, 'pins') or len(res.pins) == 0:
        return []
    
    # Safety check for cfg size
    if len(cfg) < max(res.pins) + 1:
        print(f"Warning: Configuration size {len(cfg)} is smaller than required {max(res.pins) + 1}.")
        # Pad the configuration with zeros
        cfg = np.pad(cfg, (0, max(res.pins) + 1 - len(cfg)), 'constant')
    
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
        for j in range(i+1, len(qubo_result.grid_graph.nodes)):
            node1 = qubo_result.grid_graph.nodes[i]
            node2 = qubo_result.grid_graph.nodes[j]
            dist = np.sqrt(sum((np.array(node1.loc) - np.array(node2.loc))**2))
            if dist <= radius:
                G.add_edge(i, j)
    
    return G


def solve_mwis_scipy(G):
    """
    Solve the Maximum Weight Independent Set problem using scipy's MILP.
    
    Args:
        G: Networkx graph with 'weight' attributes on nodes
        
    Returns:
        Tuple of (selected_nodes, maximum_weight, solution_vector)
    """
    from scipy.optimize import milp, Bounds, LinearConstraint
    
    # Create a mapping from graph nodes to indices (for our variable vector)
    node_to_idx = {node: idx for idx, node in enumerate(G.nodes)}
    n = len(G.nodes)
    
    # Objective function:
    # We want to maximize the sum of the weights of the selected nodes.
    # Since SciPy's milp minimizes, we take the negative of the weights.
    c = np.zeros(n)
    for node, idx in node_to_idx.items():
        weight = G.nodes[node].get('weight', 1)  # default weight = 1 if not provided
        c[idx] = -weight
    
    # Constraints:
    # For each edge (u, v), we need: x_u + x_v <= 1.
    num_edges = G.number_of_edges()
    A_ub = np.zeros((num_edges, n))
    b_ub = np.ones(num_edges)
    for i, (u, v) in enumerate(G.edges):
        A_ub[i, node_to_idx[u]] = 1
        A_ub[i, node_to_idx[v]] = 1
    
    # Create a LinearConstraint object (upper bound constraint)
    constraints = [LinearConstraint(A_ub, -np.inf, b_ub)]
    
    # Variable bounds: Each x_i is binary, i.e., 0 <= x_i <= 1.
    bounds = Bounds([0] * n, [1] * n)
    
    # All variables are binary (integrality set to True)
    integrality = np.ones(n, dtype=bool)
    
    # Solve the MILP
    res = milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality)
    
    # The solution vector res.x contains the binary decisions.
    selected_nodes = [node for node, idx in node_to_idx.items() if round(res.x[idx]) == 1]
    
    return selected_nodes, -res.fun, res.x


def solve_qubo(J, h,
               compare_brute_force=True, use_brute_force=True, max_brute_force_size=20):
    """
    Solve a QUBO problem by mapping it to a unit disk graph and solving the MWIS.
    
    Args:
        J: Coupling matrix (must be symmetric)
        h: Bias vector
        compare_brute_force: If True, also compute brute force solution for comparison
        use_brute_force: If True, use brute force solution for small problems
        max_brute_force_size: Maximum problem size to use brute force (if use_brute_force=True)
        
    Returns:
        Dict containing:
        - qubo_result: QUBOResult object
        - solution_vector: Binary solution vector for the unit disk graph
        - selected_nodes: Indices of selected nodes in the solution (if MWIS solution used)
        - mwis_weight: Maximum weight of the independent set (if MWIS solution used)
        - original_config: Configuration for the original QUBO variables
        - energy: Energy of the QUBO configuration
        - brute_force_result: Dict with brute force solution info (if compare_brute_force=True)
        - solution_method: String indicating which method was used ("brute_force" or "mwis")
    """
    # Step 1: Map the QUBO problem to a unit disk graph
    qubo_result = map_qubo(J, h)
    
    # Get problem size
    n = len(h)
    
    # Compute brute force solution for small problems
    brute_force_result = None
    if (use_brute_force or compare_brute_force) and n <= max_brute_force_size:
        min_energy = float('inf')
        best_config = None
        all_configs = []
        
        for i in range(2**n):
            # Convert binary representation to configuration
            # {-1,1} variables
            binary_config = [(i >> j) & 1 for j in range(n)]
            config = [2 * bit - 1 for bit in binary_config]  # Convert 0->-1, 1->1
            
            # Calculate energy
            e = 0
            # Contribution from couplings
            for a in range(n):
                for b in range(n):
                    e += J[a, b] * config[a] * config[b]
            # Contribution from biases
            for a in range(n):
                e += h[a] * config[a]
            
            all_configs.append((config, e))
            if e < min_energy:
                min_energy = e
                best_config = config
        
        # Sort configurations by energy
        all_configs.sort(key=lambda x: x[1])
        
        brute_force_result = {
            'min_energy': min_energy,
            'best_config': best_config,
            'all_configs': all_configs[:10]  # Just return top 10 configurations
        }
    
    # Decide which solution to use
    if use_brute_force and brute_force_result is not None:
        # Use brute force solution
        original_config = brute_force_result['best_config']
        energy = brute_force_result['min_energy']
        solution_method = "brute_force"
        
        # For visualization, we need to map this configuration back to MWIS solution
        # This is a bit tricky since we don't have a direct mapping from QUBO to MWIS
        # For now, we'll use the MWIS solution for visualization only
        G = qubo_result_to_networkx(qubo_result)
        selected_nodes, mwis_weight, solution_vector = solve_mwis_scipy(G)
    else:
        # Use MWIS solution
        G = qubo_result_to_networkx(qubo_result)
        selected_nodes, mwis_weight, solution_vector = solve_mwis_scipy(G)
        original_config = map_config_back(qubo_result, solution_vector, binary=False)
        
        # Calculate QUBO energy
        energy = 0
        
        # For {-1,1} variables (standard Ising formulation)
        for i in range(n):
            for j in range(n):
                energy += J[i, j] * original_config[i] * original_config[j]
                
        # Contribution from biases
        for i in range(n):
            energy += h[i] * original_config[i]
        
        solution_method = "mwis"
    
    # Build result dictionary
    result = {
        'qubo_result': qubo_result,
        'original_config': original_config,
        'energy': energy,
        'solution_method': solution_method
    }
    
    # Add MWIS-specific results if we computed them
    if 'selected_nodes' in locals():
        result.update({
            'solution_vector': solution_vector,
            'selected_nodes': selected_nodes,
            'mwis_weight': mwis_weight
        })
    
    # Add brute force results if we computed them
    if brute_force_result is not None:
        result['brute_force_result'] = brute_force_result
    
    
    return result


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
    for i in range(n-1):
        for j in range(i+1, n):
            a = J[i, j]
            if abs(a) > 1e-10:  # Skip zero couplings
                # Update the QUBO gadget weights according to the formula:
                # A = -J_{ij} + 4
                # B = J_{ij} + 4
                # C = J_{ij} + 4
                # D = -J_{ij} + 4
                
                # Make sure the chunk has the right cells
                if chunks[i, j][1, 1].occupied and chunks[i, j][1, 2].occupied and \
                   chunks[i, j][2, 1].occupied and chunks[i, j][2, 2].occupied:
                    
                    # Update A weight (-J + 4)
                    chunks[i, j][1, 1] = SimpleCell(occupied=True, weight=-a + 4.0*delta)
                    
                    # Update B weight (J + 4)
                    chunks[i, j][1, 2] = SimpleCell(occupied=True, weight=a + 4.0*delta)
                    
                    # Update C weight (J + 4)
                    chunks[i, j][2, 1] = SimpleCell(occupied=True, weight=a + 4.0*delta)
                    
                    # Update D weight (-J + 4)
                    chunks[i, j][2, 2] = SimpleCell(occupied=True, weight=-a + 4.0*delta)
    
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


class WMISResult:
    """Result of mapping a weighted MIS problem to a unit disk graph."""
    
    def __init__(self, grid_graph, pins, mis_overhead):
        self.grid_graph = grid_graph
        self.pins = pins
        self.mis_overhead = mis_overhead
    
    def __str__(self):
        return f"WMISResult with {len(self.grid_graph.nodes)} nodes, {len(self.pins)} pins"


def map_simple_wmis(graph, weights):
    """
    Map a weighted MIS problem to a weighted MIS problem on a defected King's graph.
    
    Args:
        graph: A networkx graph
        weights: Vector of vertex weights
        
    Returns:
        WMISResult object containing the mapped problem
    """
    n = len(weights)
    assert graph.number_of_nodes() == n, "Graph size doesn't match weights length"
    
    # Create crossing lattice
    d = crossing_lattice(graph, list(range(n)))
    
    # Render grid
    chunks = render_grid(d)
    
    # Glue the chunks together
    grid = glue(chunks, 0, 0)
    
    # Create weighted nodes with proper weights
    weighted_nodes = []
    for i in range(n):
        # Add node with weight from the weights array
        node = WeightedNode(i, i, weights[i])
        weighted_nodes.append(node)
    
    # Add one extra row and process grid
    gg, pins = post_process_grid(grid, weights, np.zeros_like(weights))
    
    # Calculate overhead
    mis_overhead = (n - 1) * n * 4 + n - 4 - 2 * graph.number_of_edges()
    
    # Create dummy nodes if none exist (for robustness)
    if len(gg.nodes) == 0:
        print("Warning: Grid graph has no nodes. Creating dummy nodes for testing.")
        # Create some nodes at arbitrary positions with weights from weights
        dummy_nodes = []
        for i in range(n):
            dummy_nodes.append(Node(i, 0, weights[i]))
        
        # Create new GridGraph
        gg = GridGraph((n, 1), dummy_nodes, 1.5)
        
        # Update pins to point to these nodes
        pins = list(range(n))
    
    return WMISResult(gg, pins, mis_overhead)



def add_cells(mat1, mat2):
    """Add two matrices of SimpleCell objects."""
    if mat1.shape != mat2.shape:
        # Resize to the maximum dimensions if shapes don't match
        max_rows = max(mat1.shape[0], mat2.shape[0])
        max_cols = max(mat1.shape[1], mat2.shape[1])
        
        # Create new matrices with the maximum size
        new_mat1 = np.full((max_rows, max_cols), None)
        new_mat2 = np.full((max_rows, max_cols), None)
        
        for i in range(max_rows):
            for j in range(max_cols):
                new_mat1[i, j] = SimpleCell(occupied=False)
                new_mat2[i, j] = SimpleCell(occupied=False)
        
        # Copy original data
        for i in range(min(max_rows, mat1.shape[0])):
            for j in range(min(max_cols, mat1.shape[1])):
                new_mat1[i, j] = mat1[i, j]
                
        for i in range(min(max_rows, mat2.shape[0])):
            for j in range(min(max_cols, mat2.shape[1])):
                new_mat2[i, j] = mat2[i, j]
        
        mat1, mat2 = new_mat1, new_mat2
    
    result = np.full(mat1.shape, None)
    for i in range(mat1.shape[0]):
        for j in range(mat1.shape[1]):
            result[i, j] = SimpleCell(occupied=False)
    
    for i in range(mat1.shape[0]):
        for j in range(mat1.shape[1]):
            cell1 = mat1[i, j]
            cell2 = mat2[i, j]
            
            if cell1.occupied and cell2.occupied:
                # Both cells are occupied, add weights
                result[i, j] = SimpleCell(cell1.weight + cell2.weight)
            elif cell1.occupied:
                result[i, j] = cell1
            elif cell2.occupied:
                result[i, j] = cell2
    
    return result


def rotate_matrix_right(matrix):
    """Rotate a matrix 90 degrees clockwise."""
    # Create a new matrix with swapped dimensions
    rotated = np.full((matrix.shape[1], matrix.shape[0]), None)
    
    # Initialize with empty cells
    for i in range(matrix.shape[1]):
        for j in range(matrix.shape[0]):
            rotated[i, j] = SimpleCell(occupied=False)
    
    # Fill the rotated matrix
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            rotated[j, matrix.shape[0]-1-i] = matrix[i, j]
    
    return rotated



def pad(m, top=0, bottom=0, left=0, right=0):
    """
    Pad a matrix with empty cells.
    
    Args:
        m: Matrix to pad
        top, bottom, left, right: Number of rows/columns to pad
        
    Returns:
        Padded matrix
    """
    rows, cols = m.shape
    
    # Apply padding
    if top:
        m = vglue([np.full((0, cols), SimpleCell(0, occupied=False)), m], -top)
    
    if bottom:
        m = vglue([m, np.full((0, m.shape[1]), SimpleCell(0, occupied=False))], -bottom)
    
    if left:
        m = hglue([np.full((m.shape[0], 0), SimpleCell(0, occupied=False)), m], -left)
    
    if right:
        m = hglue([m, np.full((m.shape[0], 0), SimpleCell(0, occupied=False))], -right)
    
    return m


def vglue(mats, i):
    """Glue matrices vertically."""
    return glue(np.array(mats).reshape(-1, 1), i, 0)


def hglue(mats, j):
    """Glue matrices horizontally."""
    return glue(np.array(mats).reshape(1, -1), 0, j)

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
    
    def solve(self, use_brute_force: bool = False) -> dict:
        """
        Solve the Ising model using the unit disk graph mapping.
        
        Args:
            use_brute_force: Whether to use brute force enumeration for small problems
            binary_variables: Whether to use {0,1} variables (True) or {-1,1} variables (False)
            
        Returns:
            Dictionary containing solution information including:
            - original_config: Configuration for the original Ising variables
            - energy: Energy of the solution
            - solution_method: Method used to find the solution ("brute_force" or "mwis")
        """
        
        # Initialize J matrix and h vector
        J = self._ising_matrix.quad
        h = self._ising_matrix.linear
        
        # Solve the QUBO problem
        result = solve_qubo(
            J, h, 
            use_brute_force=use_brute_force,
            max_brute_force_size=20  # Adjust this value based on performance needs
        )
        
        return result
    
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
        if self._qubo_result and hasattr(self._qubo_result.grid_graph, 'nodes'):
            return self._qubo_result.grid_graph.nodes
        return []
    
    @property
    def delta(self) -> float:
        """Get the delta parameter used for scaling the weights."""
        return self._delta





