"""Single source of truth for gate classification and resource formulas.

Centralises gate sets, gate→GateCount classification,
metadata extraction, and QFT/IQFT formulas used by all estimators.
"""

from __future__ import annotations

import warnings

import sympy as sp

from qamomile.circuit.ir.operation.composite_gate import ResourceMetadata
from qamomile.circuit.ir.operation.gate import GateOperation

from ._gate_count import GateCount

# ------------------------------------------------------------------ #
#  Gate set constants                                                 #
# ------------------------------------------------------------------ #

CLIFFORD_GATES = {"h", "x", "y", "z", "s", "sdg", "cx", "cy", "cz", "swap"}
T_GATES = {"t", "tdg"}
SINGLE_QUBIT_GATES = {
    "h",
    "x",
    "y",
    "z",
    "s",
    "sdg",
    "t",
    "tdg",
    "rx",
    "ry",
    "rz",
    "p",
    "u",
    "u1",
    "u2",
    "u3",
}
TWO_QUBIT_GATES = {"cx", "cy", "cz", "swap", "cp", "crx", "cry", "crz", "rzz"}
ROTATION_GATES = {"rx", "ry", "rz", "p", "cp", "crx", "cry", "crz", "rzz"}
MULTI_QUBIT_GATES = {"toffoli", "ccx"}
_GATE_BASE_QUBITS: dict[str, int] = {"toffoli": 3, "ccx": 3}

# CX / CY / CZ with exactly 1 control are Clifford; CCX etc. are not.
_CONTROLLED_CLIFFORD_GATES = {"x", "y", "z"}

_ZERO = sp.Integer(0)
_ONE = sp.Integer(1)


# ------------------------------------------------------------------ #
#  Gate classification                                                #
# ------------------------------------------------------------------ #


def classify_gate(
    op: GateOperation,
    num_controls: int | sp.Expr = 0,
) -> GateCount:
    """Classify a single GateOperation into a GateCount.

    Handles both concrete and symbolic *num_controls*.  The result always
    has ``total=1``.
    """
    gate_name = op.gate_type.name.lower() if op.gate_type else "unknown"
    # Normalise alias
    if gate_name == "ccx":
        gate_name = "toffoli"

    has_controls = not (isinstance(num_controls, int) and num_controls == 0)

    if has_controls:
        return _classify_controlled(gate_name, num_controls)
    return _classify_uncontrolled(gate_name)


def _classify_uncontrolled(gate_name: str) -> GateCount:
    return GateCount(
        total=_ONE,
        single_qubit=_ONE if gate_name in SINGLE_QUBIT_GATES else _ZERO,
        two_qubit=_ONE if gate_name in TWO_QUBIT_GATES else _ZERO,
        multi_qubit=_ONE if gate_name in MULTI_QUBIT_GATES else _ZERO,
        t_gates=_ONE if gate_name in T_GATES else _ZERO,
        clifford_gates=_ONE if gate_name in CLIFFORD_GATES else _ZERO,
        rotation_gates=_ONE if gate_name in ROTATION_GATES else _ZERO,
    )


def _classify_controlled(
    gate_name: str,
    num_controls: int | sp.Expr,
) -> GateCount:
    from .gate_counter import GateCount

    is_single_base = gate_name in SINGLE_QUBIT_GATES
    is_two_base = gate_name in TWO_QUBIT_GATES

    if is_single_base:
        base_qubits = 1
    elif is_two_base:
        base_qubits = 2
    else:
        base_qubits = _GATE_BASE_QUBITS.get(gate_name, 1)

    total_qubits = num_controls + base_qubits

    if isinstance(num_controls, int):
        two = _ONE if total_qubits == 2 else _ZERO
        multi = _ONE if total_qubits > 2 else _ZERO
    else:
        is_classifiable = (
            is_single_base or is_two_base or gate_name in MULTI_QUBIT_GATES
        )
        if is_classifiable:
            two = sp.Piecewise((_ONE, sp.Eq(total_qubits, 2)), (_ZERO, True))
            multi = sp.Piecewise((_ONE, total_qubits > 2), (_ZERO, True))
        else:
            two, multi = _ZERO, _ZERO

    # Controlled T/Tdg → no longer T gates
    t_count = _ZERO
    # Only single-control CX/CY/CZ are Clifford
    if gate_name in _CONTROLLED_CLIFFORD_GATES:
        if isinstance(num_controls, int):
            clifford = _ONE if num_controls == 1 else _ZERO
        else:
            clifford = sp.Piecewise(
                (_ONE, sp.Eq(num_controls, 1)),
                (_ZERO, True),
            )
    else:
        clifford = _ZERO

    rotation = _ONE if gate_name in ROTATION_GATES else _ZERO

    return GateCount(
        total=_ONE,
        single_qubit=_ZERO,  # controlled gates are never single-qubit
        two_qubit=two,
        multi_qubit=multi,
        t_gates=t_count,
        clifford_gates=clifford,
        rotation_gates=rotation,
    )


# ------------------------------------------------------------------ #
#  ControlledU classification                                         #
# ------------------------------------------------------------------ #


def classify_controlled_u(
    nc: int | sp.Expr,
    num_targets: int,
) -> GateCount:
    """Classify a ControlledUOperation (opaque gate, total=1).

    ``power`` is NOT multiplied — each ControlledUOperation call counts
    as exactly one gate regardless of power.  Exponential oracle counts
    arise from loop structure (e.g. ``for _rep in range(2**k)``).
    """
    from .gate_counter import GateCount

    total_qubits = nc + num_targets

    if isinstance(nc, int):
        two = _ONE if total_qubits == 2 else _ZERO
        multi = _ONE if total_qubits > 2 else _ZERO
    else:
        two = sp.Piecewise((_ONE, sp.Eq(total_qubits, 2)), (_ZERO, True))
        multi = sp.Piecewise((_ONE, total_qubits > 2), (_ZERO, True))

    return GateCount(
        total=_ONE,
        single_qubit=_ZERO,
        two_qubit=two,
        multi_qubit=multi,
        t_gates=_ZERO,
        clifford_gates=_ZERO,
        rotation_gates=_ZERO,
    )


# ------------------------------------------------------------------ #
#  Metadata extraction                                                #
# ------------------------------------------------------------------ #


def extract_gate_count_from_metadata(meta: ResourceMetadata) -> GateCount:
    """Extract GateCount from ResourceMetadata.

    Emits ``UserWarning`` when *total_gates* is set but sub-categories
    have ``None`` gaps and the known sub-total is less than *total_gates*.
    This warning behaviour is required by ``test_metadata_warnings.py``.
    """
    from .gate_counter import GateCount

    single = meta.single_qubit_gates or 0
    two = meta.two_qubit_gates or 0
    multi = meta.multi_qubit_gates or 0
    t = meta.t_gates or 0
    clifford = meta.clifford_gates or 0
    rotation = meta.rotation_gates or 0

    if meta.total_gates:
        none_fields = [
            name
            for name, val in [
                ("single_qubit_gates", meta.single_qubit_gates),
                ("two_qubit_gates", meta.two_qubit_gates),
                ("multi_qubit_gates", meta.multi_qubit_gates),
            ]
            if val is None
        ]
        if none_fields:
            sub_total = single + two + multi
            if sub_total < meta.total_gates:
                warnings.warn(
                    f"ResourceMetadata has total_gates={meta.total_gates} "
                    f"but {', '.join(none_fields)} "
                    f"{'is' if len(none_fields) == 1 else 'are'} "
                    f"unspecified (None) and treated as 0. "
                    f"The known sub-total ({sub_total}) is less than "
                    f"total_gates ({meta.total_gates}). "
                    f"Set gate category fields explicitly for accurate "
                    f"sub-category counts.",
                    UserWarning,
                    stacklevel=2,
                )

    total = meta.total_gates if meta.total_gates else single + two + multi

    return GateCount(
        total=sp.Integer(total),
        single_qubit=sp.Integer(single),
        two_qubit=sp.Integer(two),
        multi_qubit=sp.Integer(multi),
        t_gates=sp.Integer(t),
        clifford_gates=sp.Integer(clifford),
        rotation_gates=sp.Integer(rotation),
    )


# ------------------------------------------------------------------ #
#  QFT / IQFT formulas                                                #
# ------------------------------------------------------------------ #


def qft_iqft_gate_count(n: sp.Expr) -> GateCount:
    """Gate count for QFT or IQFT on *n* qubits.

    QFT = n H + n(n-1)/2 CP + n//2 SWAP
    """
    from .gate_counter import GateCount

    h = n
    cp = n * (n - 1) / 2
    swap = n // 2
    return GateCount(
        total=h + cp + swap,
        single_qubit=h,
        two_qubit=cp + swap,
        multi_qubit=_ZERO,
        t_gates=_ZERO,
        clifford_gates=h + swap,  # H and SWAP are Clifford
        rotation_gates=cp,  # CP gates are rotation
    )
