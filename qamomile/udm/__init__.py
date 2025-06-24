"""
Unit Disk Mapping (UDM) module for Qamomile.

This module implements algorithms for mapping various optimization problems (like QUBO and Ising models)
to unit disk grid graphs, which can be naturally encoded in neutral-atom quantum computers.
"""

from .mwis_solver import QUBOResult, Ising_UnitDiskGraph
from .transpiler import UDMTranspiler

__all__ = ["QUBOResult", "Ising_UnitDiskGraph", "UDMTranspiler"]
