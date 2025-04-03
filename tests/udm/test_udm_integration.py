"""
Simple test script to verify UDM integration.
"""

import sys
import numpy as np
from qamomile.core.ising_qubo import IsingModel, UnitDiskGraph

def test_udm_integration():
    """Test the UDM integration by creating and solving a simple Ising problem."""
    
    # Create a simple Ising model
    quad = {
        (0, 1): 1.0,
        (0, 2): -0.5,
        (1, 2): 0.8
    }
    
    linear = {
        0: 0.1,
        1: -0.2,
        2: 0.3
    }
    
    ising = IsingModel(quad=quad, linear=linear, constant=0.0)
    print("Created Ising model successfully")
    
    # Create UnitDiskGraph
    udg = ising.to_unit_disk_graph()
    print("Created UnitDiskGraph successfully")
    
    # Check if key structures are available
    print(f"Number of nodes: {len(udg.nodes)}")
    print(f"Number of pins: {len(udg.pins)}")
    
    # Try solving the problem
    result = udg.solve(use_brute_force=True)
    print(f"Solution found: {result['original_config']}")
    print(f"Energy: {result['energy']}")
    print(f"Method: {result['solution_method']}")
    
    print("\nUDM integration successful!")
    assert "energy" in result
    assert isinstance(result["energy"], (int, float))
    assert "original_config" in result
    assert isinstance(result["original_config"], list)
    assert "solution_method" in result
    assert isinstance(result["solution_method"], str)
    

if __name__ == "__main__":
    success = test_udm_integration()
    sys.exit(0 if success else 1)