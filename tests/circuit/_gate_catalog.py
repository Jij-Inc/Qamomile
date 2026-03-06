"""Shared gate catalogs for circuit tests.

This module centralizes frontend gate lists so tests avoid in-file duplication.
"""

import qamomile.circuit as qm
from qamomile.circuit.stdlib.qft import iqft, qft

SINGLE_QUBIT_GATES = [
    ("h", qm.h),
    ("x", qm.x),
    ("y", qm.y),
    ("z", qm.z),
    ("t", qm.t),
    ("s", qm.s),
    ("sdg", qm.sdg),
    ("tdg", qm.tdg),
]

ROTATION_GATES = [
    ("rx", qm.rx),
    ("ry", qm.ry),
    ("rz", qm.rz),
    ("p", qm.p),
]

TWO_QUBIT_GATES_NO_PARAM = [
    ("cx", qm.cx),
    ("cz", qm.cz),
    ("swap", qm.swap),
]

TWO_QUBIT_GATES_WITH_PARAM = [
    ("cp", qm.cp),
    ("rzz", qm.rzz),
]

THREE_QUBIT_GATES = [
    ("ccx", qm.ccx),
]

STDLIB_GATES = [
    ("qft", qft),
    ("iqft", iqft),
]

__all__ = [
    "SINGLE_QUBIT_GATES",
    "ROTATION_GATES",
    "TWO_QUBIT_GATES_NO_PARAM",
    "TWO_QUBIT_GATES_WITH_PARAM",
    "THREE_QUBIT_GATES",
    "STDLIB_GATES",
]
