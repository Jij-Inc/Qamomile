"""Classify logical gate names and arities."""

from __future__ import annotations

from collections.abc import Iterable

from qamomile.circuit.ir.operation.gate import (
    GateOperationType,
)

_IR_SINGLE_QUBIT_GATE_TYPES = frozenset(
    {
        GateOperationType.H,
        GateOperationType.X,
        GateOperationType.Y,
        GateOperationType.Z,
        GateOperationType.S,
        GateOperationType.SDG,
        GateOperationType.T,
        GateOperationType.TDG,
        GateOperationType.RX,
        GateOperationType.RY,
        GateOperationType.RZ,
        GateOperationType.P,
    }
)


_IR_TWO_QUBIT_GATE_TYPES = frozenset(
    {
        GateOperationType.CX,
        GateOperationType.CZ,
        GateOperationType.SWAP,
        GateOperationType.CP,
        GateOperationType.RZZ,
    }
)


_IR_MULTI_QUBIT_GATE_TYPES = frozenset({GateOperationType.TOFFOLI})


_CLEAN_ANCILLA_EXPLICIT_MULTI_TARGET_GATE_TYPES = frozenset(
    {
        GateOperationType.CX,
        GateOperationType.CZ,
        GateOperationType.SWAP,
        GateOperationType.CP,
        GateOperationType.RZZ,
        GateOperationType.TOFFOLI,
    }
)


def _validate_ir_gate_arity_profiles() -> None:
    """Require one and only one arity profile for every IR gate type.

    Raises:
        RuntimeError: If a ``GateOperationType`` is missing from the resource
            arity profile or appears in more than one arity class.
    """
    groups = (
        _IR_SINGLE_QUBIT_GATE_TYPES,
        _IR_TWO_QUBIT_GATE_TYPES,
        _IR_MULTI_QUBIT_GATE_TYPES,
    )
    covered = frozenset().union(*groups)
    duplicated = {
        gate_type
        for gate_type in covered
        if sum(gate_type in group for group in groups) != 1
    }
    missing = set(GateOperationType) - covered
    multi_target = _IR_TWO_QUBIT_GATE_TYPES | _IR_MULTI_QUBIT_GATE_TYPES
    missing_clean_ancilla = (
        multi_target - _CLEAN_ANCILLA_EXPLICIT_MULTI_TARGET_GATE_TYPES
    )
    extraneous_clean_ancilla = (
        _CLEAN_ANCILLA_EXPLICIT_MULTI_TARGET_GATE_TYPES - multi_target
    )
    if missing or duplicated or missing_clean_ancilla or extraneous_clean_ancilla:
        missing_names = sorted(gate_type.name for gate_type in missing)
        duplicated_names = sorted(gate_type.name for gate_type in duplicated)
        missing_clean_ancilla_names = sorted(
            gate_type.name for gate_type in missing_clean_ancilla
        )
        extraneous_clean_ancilla_names = sorted(
            gate_type.name for gate_type in extraneous_clean_ancilla
        )
        raise RuntimeError(
            "Resource estimation requires an exhaustive, disjoint IR gate "
            "arity profile; "
            f"missing={missing_names}, duplicated={duplicated_names}, "
            f"missing_clean_ancilla_multi_target={missing_clean_ancilla_names}, "
            f"extraneous_clean_ancilla_multi_target={extraneous_clean_ancilla_names}."
        )


_GATE_OPERATION_ARITY: dict[GateOperationType, int] = {
    **{gate_type: 1 for gate_type in _IR_SINGLE_QUBIT_GATE_TYPES},
    **{gate_type: 2 for gate_type in _IR_TWO_QUBIT_GATE_TYPES},
    **{gate_type: 3 for gate_type in _IR_MULTI_QUBIT_GATE_TYPES},
}


_CLIFFORD_GATE_TYPES = frozenset(
    {
        GateOperationType.H,
        GateOperationType.X,
        GateOperationType.Y,
        GateOperationType.Z,
        GateOperationType.S,
        GateOperationType.SDG,
        GateOperationType.CX,
        GateOperationType.CZ,
        GateOperationType.SWAP,
    }
)


_T_GATE_TYPES = frozenset({GateOperationType.T, GateOperationType.TDG})


_ROTATION_GATE_TYPES = frozenset(
    {
        GateOperationType.RX,
        GateOperationType.RY,
        GateOperationType.RZ,
        GateOperationType.P,
        GateOperationType.CP,
        GateOperationType.RZZ,
    }
)


_CONTROLLED_CLIFFORD_GATE_TYPES = frozenset(
    {
        GateOperationType.X,
        GateOperationType.Y,
        GateOperationType.Z,
    }
)


def _gate_type_names(gate_types: Iterable[GateOperationType]) -> set[str]:
    """Return canonical lowercase names for IR gate types.

    Args:
        gate_types (Iterable[GateOperationType]): IR gate types to name.

    Returns:
        set[str]: Canonical lowercase enum names.
    """
    return {gate_type.name.lower() for gate_type in gate_types}


_SYNTHETIC_SINGLE_QUBIT_GATE_NAMES = {"u", "u1", "u2", "u3"}


_SYNTHETIC_MULTI_QUBIT_GATE_NAMES = {"ccx"}


_CLIFFORD_GATES = _gate_type_names(_CLIFFORD_GATE_TYPES)


_T_GATES = _gate_type_names(_T_GATE_TYPES)


_SINGLE_QUBIT_GATES = (
    _gate_type_names(_IR_SINGLE_QUBIT_GATE_TYPES) | _SYNTHETIC_SINGLE_QUBIT_GATE_NAMES
)


_TWO_QUBIT_GATES = _gate_type_names(_IR_TWO_QUBIT_GATE_TYPES)


_ROTATION_GATES = _gate_type_names(_ROTATION_GATE_TYPES)


_MULTI_QUBIT_GATES = (
    _gate_type_names(_IR_MULTI_QUBIT_GATE_TYPES) | _SYNTHETIC_MULTI_QUBIT_GATE_NAMES
)


_GATE_BASE_QUBITS: dict[str, int] = {
    **{
        gate_type.name.lower(): arity
        for gate_type, arity in _GATE_OPERATION_ARITY.items()
    },
    "ccx": 3,
}


_CONTROLLED_CLIFFORD_GATES = _gate_type_names(_CONTROLLED_CLIFFORD_GATE_TYPES)
