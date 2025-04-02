"""
Standalone test for UDM module.
"""

import sys
import numpy as np
from qamomile.udm import map_qubo, solve_qubo, QUBOResult

def test_udm_standalone():
    """Test the UDM module by directly using its functions."""
    
    # Create a simple problem
    n = 3
    J = np.zeros((n, n))
    J[0, 1] = 1.0
    J[0, 2] = -0.5
    J[1, 2] = 0.8
    J = J + J.T - np.diag(np.diag(J))  # Make symmetric
    
    h = np.array([0.1, -0.2, 0.3])
    
    print("Created problem matrices")
    
    # Try mapping to a UDG
    qubo_result = map_qubo(J, h)
    print("Mapped problem to UDG successfully")
    
    # Check if the mapping worked
    print(f"Number of nodes: {len(qubo_result.grid_graph.nodes)}")
    print(f"Number of pins: {len(qubo_result.pins)}")
    
    # Try solving the problem
    result = solve_qubo(J, h, use_brute_force=True)
    print(f"Solution found: {result['original_config']}")
    print(f"Energy: {result['energy']}")
    print(f"Method: {result['solution_method']}")
    
    print("\nUDM module test successful!")
    assert "energy" in result
    assert isinstance(result["energy"], (int, float)) 
    
if __name__ == "__main__":
    success = test_udm_standalone()
    sys.exit(0 if success else 1)