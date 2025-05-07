"""
Simple test script to verify UDM integration.
"""

import sys
import numpy as np
from qamomile.core.ising_qubo import IsingModel
from qamomile.udm import Ising_UnitDiskGraph

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
    udg = Ising_UnitDiskGraph(ising)
    print("Created UnitDiskGraph successfully")
    
    # Check if key structures are available
    print(f"Number of nodes: {len(udg.nodes)}")
    print(f"Number of pins: {len(udg.pins)}")    

if __name__ == "__main__":
    success = test_udm_integration()
    sys.exit(0 if success else 1)