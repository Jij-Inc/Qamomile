"""Single source of truth for gate classification and resource formulas.

Centralises gate sets, gate→GateCount classification,
metadata extraction, and QFT/IQFT formulas used by all estimators.
"""

from __future__ import annotations

import warnings
from typing import Any, cast

import sympy as sp

from qamomile.circuit.ir.operation.composite_gate import ResourceMetadata
from qamomile.circuit.ir.operation.gate import GateOperation

from ._gate_count import GateCount

# ------------------------------------------------------------------ #
#  Gate set constants                                                 #
# ------------------------------------------------------------------ #

CLIFFORD_GATES = {"h", "x", "y", "z", "s", "sdg", "cx", "cz", "swap"}
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
TWO_QUBIT_GATES = {"cx", "cz", "swap", "cp", "rzz"}
ROTATION_GATES = {"rx", "ry", "rz", "p", "cp", "rzz"}
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
    """Classify a single uncontrolled gate into a GateCount.

    Args:
        gate_name: Lowercase gate name (e.g. "h", "cx", "toffoli").

    Returns:
        GateCount with total=1 and appropriate category flags set.
    """
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
    """Classify a controlled gate into a GateCount.

    Handles both concrete and symbolic num_controls.
    A single-control X/Y/Z is Clifford; higher control counts are multi-qubit.

    Args:
        gate_name: Lowercase base gate name (e.g. "x", "rz").
        num_controls: Number of control qubits (concrete or symbolic).

    Returns:
        GateCount with total=1 and category flags based on control count.
    """
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
        two_qubit=cast(sp.Expr, two),
        multi_qubit=cast(sp.Expr, multi),
        t_gates=t_count,
        clifford_gates=cast(sp.Expr, clifford),
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

    Args:
        nc (int | sp.Expr): Number of control qubits (concrete or symbolic).
        num_targets (int): Number of target qubits.

    Returns:
        GateCount: Gate count with ``total=1`` and two_qubit / multi_qubit
            flags set based on the total qubit count (``nc + num_targets``).
    """
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
        two_qubit=cast(sp.Expr, two),
        multi_qubit=cast(sp.Expr, multi),
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

    Args:
        meta (ResourceMetadata): Metadata to extract gate counts from.
            Fields left as ``None`` are treated as 0.

    Returns:
        GateCount: Extracted gate counts as ``sp.Integer`` values.
    """
    single = meta.single_qubit_gates if meta.single_qubit_gates is not None else 0
    two = meta.two_qubit_gates if meta.two_qubit_gates is not None else 0
    multi = meta.multi_qubit_gates if meta.multi_qubit_gates is not None else 0
    t = meta.t_gates if meta.t_gates is not None else 0
    clifford = meta.clifford_gates if meta.clifford_gates is not None else 0
    rotation = meta.rotation_gates if meta.rotation_gates is not None else 0

    if meta.total_gates is not None:
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

    total = meta.total_gates if meta.total_gates is not None else single + two + multi

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

    Args:
        n (sp.Expr): Number of qubits (may be symbolic).

    Returns:
        GateCount: Symbolic gate counts for the standard QFT decomposition.
    """
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


# ------------------------------------------------------------------ #
#  PauliEvolve gate count                                            #
# ------------------------------------------------------------------ #


def classify_pauli_evolve(hamiltonian: Any) -> GateCount:
    """Gate count for PauliEvolve decomposition (Pauli gadget method).

    For each Hamiltonian term with *k* qubits (x_count X, y_count Y):
      - Basis change (pre):  x_count H + y_count (Sdg + H) = x_count + 2*y_count single-qubit
      - CNOT ladder:         2*(k-1) CX gates (k >= 2)
      - RZ gate:             1 rotation gate
      - Basis undo (post):   x_count H + y_count (H + S) = x_count + 2*y_count single-qubit
      Total single-qubit per term: 2*x_count + 4*y_count + 1 (RZ)
      Total two-qubit per term:    2*max(0, k-1) (CX ladder)

    Args:
        hamiltonian: A concrete Hamiltonian with known terms.

    Returns:
        GateCount: Total gate counts for the full Pauli evolution.
    """
    import qamomile.observable as qm_o

    total_single = _ZERO
    total_two = _ZERO
    total_clifford = _ZERO
    total_rotation = _ZERO
    total_total = _ZERO

    for operators, coeff in hamiltonian:
        if abs(complex(coeff)) < 1e-15:
            continue
        if not operators:
            continue

        k = len(operators)
        x_count = sum(1 for op in operators if op.pauli == qm_o.Pauli.X)
        y_count = sum(1 for op in operators if op.pauli == qm_o.Pauli.Y)

        # Basis change gates (Clifford): H for X, Sdg+H for Y (pre), H+S for Y (post)
        basis_change_single = sp.Integer(2 * x_count + 4 * y_count)
        # RZ rotation (1 per term)
        rz_count = _ONE
        # CNOT ladder (2*(k-1) for k >= 2, all Clifford)
        cx_count = sp.Integer(2 * max(0, k - 1))

        term_single = basis_change_single + rz_count
        term_two = cx_count
        term_clifford = basis_change_single + cx_count
        term_rotation = rz_count
        term_total = term_single + term_two

        total_single += term_single
        total_two += term_two
        total_clifford += term_clifford
        total_rotation += term_rotation
        total_total += term_total

    return GateCount(
        total=total_total,
        single_qubit=total_single,
        two_qubit=total_two,
        multi_qubit=_ZERO,
        t_gates=_ZERO,
        clifford_gates=total_clifford,
        rotation_gates=total_rotation,
    )
