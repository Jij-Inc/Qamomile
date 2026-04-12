"""Shared gate decomposition recipes for backend emitters.

This module defines the canonical decomposition of controlled gates (CH, CY, CP,
CRY, CRZ) into primitive operations (RY, RZ, CNOT, S, SDG).  Each recipe is a
sequence of ``DecompStep`` dataclass instances that encode:

* Which primitive gate to apply,
* Which qubit role receives the gate (``"control"`` or ``"target"``),
* An optional angle expression (e.g. ``"theta/2"``, ``"-pi/4"``).

Backend emitters that cannot use native controlled gates should implement
their decomposition methods so that the gate sequence follows the same
recipe defined here.  Backends with special angle representations
(CUDA-Q ``CudaqExpr`` strings, QURI Parts parametric dicts) may not be
able to call ``emit_decomposition`` directly but should reference the
recipe constants to ensure mathematical equivalence.

Typical usage (for backends with plain float angles)::

    emit_decomposition(emitter, circuit, CH_DECOMPOSITION, ctrl, tgt)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any


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


def evaluate_angle(expr: str, theta: float = 0.0) -> float:
    """Evaluate an angle expression.

    Supported tokens: ``theta`` (bound to *theta*), ``pi`` (``math.pi``),
    arithmetic operators ``+``, ``-``, ``*``, ``/``.

    Args:
        expr: Angle expression string, e.g. ``"theta/2"`` or ``"-pi/4"``.
        theta: Concrete value to substitute for the ``theta`` token.

    Returns:
        Evaluated float angle in radians.
    """
    return eval(expr, {"theta": theta, "pi": math.pi, "__builtins__": {}})  # noqa: S307


def emit_decomposition(
    emitter: Any,
    circuit: Any,
    recipe: list[DecompStep],
    control: int,
    target: int,
    theta: float = 0.0,
) -> None:
    """Execute a decomposition recipe using the emitter's primitive methods.

    This helper is suitable for backends whose ``emit_ry``, ``emit_rz``, etc.
    accept plain ``float`` angles.  Backends with symbolic or parametric angle
    representations (CUDA-Q ``CudaqExpr``, QURI Parts parametric dicts) should
    inline the recipe manually and reference the corresponding ``*_DECOMPOSITION``
    constant in a comment.

    The emitter must implement the following methods:
    ``emit_ry(circuit, qubit, angle)``,
    ``emit_rz(circuit, qubit, angle)``,
    ``emit_cx(circuit, control, target)``,
    ``emit_s(circuit, qubit)``,
    ``emit_sdg(circuit, qubit)``.

    Args:
        emitter: A gate emitter with primitive gate methods.
        circuit: The circuit object being built.
        recipe: Sequence of ``DecompStep`` defining the decomposition.
        control: Control qubit index.
        target: Target qubit index.
        theta: Angle parameter (used when recipe steps contain angle
            expressions referencing ``theta``).
    """
    for step in recipe:
        qubit = control if step.target == "control" else target
        angle = evaluate_angle(step.angle, theta) if step.angle else 0.0

        match step.gate:
            case PrimitiveGate.RY:
                emitter.emit_ry(circuit, qubit, angle)
            case PrimitiveGate.RZ:
                emitter.emit_rz(circuit, qubit, angle)
            case PrimitiveGate.CNOT:
                emitter.emit_cx(circuit, control, target)
            case PrimitiveGate.S:
                emitter.emit_s(circuit, qubit)
            case PrimitiveGate.SDG:
                emitter.emit_sdg(circuit, qubit)
