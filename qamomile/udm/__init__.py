"""
Unit Disk Mapping (UDM) module for Qamomile.

This module implements algorithms for mapping various optimization problems (like QUBO and Ising models)
to unit disk grid graphs, which can be naturally encoded in neutral-atom quantum computers.
"""


from .dragondrop import (
    map_qubo,
    map_simple_wmis,
    solve_qubo,
    solve_mwis_scipy,
    qubo_result_to_networkx,
    QUBOResult,
    WMISResult,
    Ising_UnitDiskGraph
)

__all__ = [
    "Ising_UnitDiskGraph", "map_qubo", "map_simple_wmis", "solve_qubo",
    "solve_mwis_scipy", "qubo_result_to_networkx",
    "QUBOResult", "WMISResult",
]