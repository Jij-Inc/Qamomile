"""Verify Qiskit selects native semantic composites and portable fallbacks."""

from __future__ import annotations

from typing import Any

import qamomile.circuit as qmc
from qamomile.circuit.stdlib import (
    amplitude_encoding,
    mottonen_amplitude_encoding,
)
from qamomile.qiskit import QiskitTranspiler


@qmc.qkernel
def _state_preparation() -> qmc.Vector[qmc.Bit]:
    """Build a two-qubit state-preparation example.

    Returns:
        qmc.Vector[qmc.Bit]: State-preparation measurements.
    """
    qubits = qmc.qubit_array(2, "q")
    return qmc.measure(amplitude_encoding(qubits, [1.0, 2.0, 3.0, 4.0]))


@qmc.qkernel
def _mottonen_state_preparation() -> qmc.Vector[qmc.Bit]:
    """Build an explicitly Möttönen two-qubit state-preparation example.

    Returns:
        qmc.Vector[qmc.Bit]: State-preparation measurements.
    """
    qubits = qmc.qubit_array(2, "q")
    return qmc.measure(mottonen_amplitude_encoding(qubits, [1.0, 2.0, 3.0, 4.0]))


@qmc.qkernel
def _ripple_carry() -> tuple[
    qmc.Vector[qmc.Bit],
    qmc.Vector[qmc.Bit],
    qmc.Bit,
    qmc.Bit,
]:
    """Build a two-bit full-adder example.

    Returns:
        tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit], qmc.Bit, qmc.Bit]:
            Addend, accumulator, carry, and overflow measurements.
    """
    left = qmc.qubit_array(2, "left")
    right = qmc.qubit_array(2, "right")
    carry = qmc.qubit("carry")
    overflow = qmc.qubit("overflow")
    left, right, carry, overflow = qmc.ripple_carry_add(left, right, carry, overflow)
    return (
        qmc.measure(left),
        qmc.measure(right),
        qmc.measure(carry),
        qmc.measure(overflow),
    )


@qmc.qkernel
def _multi_controlled_x() -> tuple[qmc.Vector[qmc.Bit], qmc.Bit]:
    """Build a three-control X example.

    Returns:
        tuple[qmc.Vector[qmc.Bit], qmc.Bit]: Control and target measurements.
    """
    controls = qmc.qubit_array(3, "controls")
    target = qmc.qubit("target")
    controls, target = qmc.mcx(controls, target)
    return qmc.measure(controls), qmc.measure(target)


def _operations(kernel: qmc.QKernel, *, native: bool) -> list[Any]:
    """Return non-measurement operations from one transpiled Qiskit circuit.

    Args:
        kernel (qmc.QKernel): Kernel to transpile.
        native (bool): Whether to prefer native semantic composites.

    Returns:
        list[Any]: Qiskit operations in program order.
    """
    executable = QiskitTranspiler(use_native_composite=native).transpile(kernel)
    circuit = executable.compiled_quantum[0].circuit
    return [
        instruction.operation
        for instruction in circuit.data
        if instruction.operation.name != "measure"
    ]


def _nested_operations(operation: Any, *, depth: int = 0) -> list[Any]:
    """Return an operation and the operations nested in its definitions.

    Args:
        operation (Any): Qiskit operation to inspect.
        depth (int): Current recursion depth.

    Returns:
        list[Any]: The root operation and up to eight nested definition levels.
    """
    operations = [operation]
    definition = getattr(operation, "definition", None)
    if definition is None or depth == 8:
        return operations
    for instruction in definition.data:
        operations.extend(_nested_operations(instruction.operation, depth=depth + 1))
    return operations


def test_qiskit_uses_native_state_preparation() -> None:
    """Concrete amplitude encoding selects Qiskit's StatePreparation gate."""
    native = _operations(_state_preparation, native=True)
    fallback = _operations(_state_preparation, native=False)
    assert [(op.name, type(op).__name__) for op in native] == [
        ("state_preparation", "StatePreparation")
    ]
    assert [(op.name, type(op).__name__) for op in fallback] == [
        ("state_preparation", "Gate")
    ]


def test_qiskit_does_not_substitute_explicit_mottonen_encoding() -> None:
    """Explicit Möttönen encoding stays a named gate with its exact body."""
    for native in (True, False):
        operations = _operations(_mottonen_state_preparation, native=native)
        assert [(op.name, type(op).__name__) for op in operations] == [
            ("mottonen_amplitude_encoding", "Gate")
        ]

        nested = _nested_operations(operations[0])
        assert all(type(op).__name__ != "StatePreparation" for op in nested)
        nested_names = [op.name for op in nested]
        assert nested_names.count("ry") >= 3
        assert nested_names.count("cx") >= 2


def test_qiskit_uses_native_full_adder() -> None:
    """Ripple-carry addition selects Qiskit's abstract FullAdder gate."""
    assert [
        (op.name, type(op).__name__) for op in _operations(_ripple_carry, native=True)
    ] == [("FullAdder", "FullAdderGate")]
    assert [op.name for op in _operations(_ripple_carry, native=False)] == [
        "ripple_carry_add"
    ]


def test_qiskit_uses_native_multi_controlled_x() -> None:
    """Semantic MCX selects Qiskit's arbitrary-width MCX gate."""
    assert [
        (op.name, type(op).__name__)
        for op in _operations(_multi_controlled_x, native=True)
    ] == [("mcx", "MCXGate")]
    assert [op.name for op in _operations(_multi_controlled_x, native=False)] == [
        "multi_controlled_x"
    ]
