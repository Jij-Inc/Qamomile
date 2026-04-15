"""Shared gate decomposition recipes for backend emitters.

This module defines the canonical decomposition of controlled gates (CH, CY,
CP, CRY, CRZ) into primitive operations (RY, RZ, CNOT, S, SDG).  Each recipe
is a frozen sequence of :class:`DecompStep` instances that encode:

* Which primitive gate to apply,
* Which qubit role receives the gate (``"control"`` or ``"target"``),
* An optional angle expression (e.g. ``"theta/2"``, ``"-pi/4"``).

The recipes are the **single source of truth** for how Qamomile decomposes
controlled gates when a backend cannot use a native controlled-U operation.

## Why data-only (no shared execution helper)

Each backend has its own emission dialect:

* **Qiskit** uses native ``circuit.ch()`` / ``circuit.cy()`` and never needs
  this decomposition.
* **QURI Parts** represents angles as parametric dicts (name -> coefficient);
  primitive ``emit_*`` methods take those dicts, not plain floats, so a
  generic "loop over the recipe and call ``emitter.emit_ry(angle)``" helper
  does not compose.
* **CUDA-Q** emits Python source as strings.  A generic helper that calls
  ``self.emit_ry`` interacts poorly with the tracing test emitter, which
  wraps every ``emit_*`` method and would double-record each call.

Because no single helper absorbs all three styles cleanly, backends inline
their decomposition using their own idiomatic emission, and reference the
recipe constants below from the ``emit_ch`` / ``emit_cy`` / ... docstrings to
document equivalence.  When changing a recipe, update the constant here and
ensure every backend's inline implementation matches.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class PrimitiveGate(Enum):
    """Gate primitives used in decomposition recipes."""

    RY = auto()
    RZ = auto()
    CNOT = auto()
    S = auto()
    SDG = auto()


@dataclass(frozen=True)
class DecompStep:
    """A single step in a decomposition recipe.

    Attributes:
        gate: The primitive gate to emit.
        target: ``"control"`` or ``"target"`` -- which qubit role receives
            the gate.  For ``CNOT`` this field indicates the target qubit;
            the control qubit of the CNOT is always the ``control`` role.
        angle: An optional angle expression such as ``"theta/2"`` or
            ``"-pi/4"``.  ``None`` for non-rotation gates.
    """

    gate: PrimitiveGate
    target: str  # "control" or "target"
    angle: str | None = None  # Expression like "theta/2", "-theta/2", "pi/4"


# --------------------------------------------------------------------------
# Decomposition recipes
# --------------------------------------------------------------------------

# CH: RY(pi/4, target) -> CNOT(ctrl, target) -> RY(-pi/4, target)
CH_DECOMPOSITION: list[DecompStep] = [
    DecompStep(PrimitiveGate.RY, "target", "pi/4"),
    DecompStep(PrimitiveGate.CNOT, "target"),
    DecompStep(PrimitiveGate.RY, "target", "-pi/4"),
]

# CY: SDG(target) -> CNOT(ctrl, target) -> S(target)
CY_DECOMPOSITION: list[DecompStep] = [
    DecompStep(PrimitiveGate.SDG, "target"),
    DecompStep(PrimitiveGate.CNOT, "target"),
    DecompStep(PrimitiveGate.S, "target"),
]

# CP(theta): RZ(theta/2, target) -> CNOT -> RZ(-theta/2, target) -> CNOT -> RZ(theta/2, control)
CP_DECOMPOSITION: list[DecompStep] = [
    DecompStep(PrimitiveGate.RZ, "target", "theta/2"),
    DecompStep(PrimitiveGate.CNOT, "target"),
    DecompStep(PrimitiveGate.RZ, "target", "-theta/2"),
    DecompStep(PrimitiveGate.CNOT, "target"),
    DecompStep(PrimitiveGate.RZ, "control", "theta/2"),
]

# CRY(theta): RY(theta/2, target) -> CNOT -> RY(-theta/2, target) -> CNOT
CRY_DECOMPOSITION: list[DecompStep] = [
    DecompStep(PrimitiveGate.RY, "target", "theta/2"),
    DecompStep(PrimitiveGate.CNOT, "target"),
    DecompStep(PrimitiveGate.RY, "target", "-theta/2"),
    DecompStep(PrimitiveGate.CNOT, "target"),
]

# CRZ(theta): RZ(theta/2, target) -> CNOT -> RZ(-theta/2, target) -> CNOT
CRZ_DECOMPOSITION: list[DecompStep] = [
    DecompStep(PrimitiveGate.RZ, "target", "theta/2"),
    DecompStep(PrimitiveGate.CNOT, "target"),
    DecompStep(PrimitiveGate.RZ, "target", "-theta/2"),
    DecompStep(PrimitiveGate.CNOT, "target"),
]
