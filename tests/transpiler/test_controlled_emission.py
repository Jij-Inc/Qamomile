"""Tests for controlled-emission support helpers."""

from typing import Any

from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.composite_gate import (
    CompositeGateOperation,
    CompositeGateType,
)
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.types.primitives import QubitType
from qamomile.circuit.ir.value import Value
from qamomile.circuit.transpiler.passes.emit_support import (
    controlled_emission,
    inverse_emission,
)
from qamomile.circuit.transpiler.passes.emit_support.controlled_emission import (
    _gate_matches_qubit_count,
    emit_controlled_operations,
)


class _GateWithoutQubitCount:
    """Backend-gate stand-in with no qubit-count attribute."""


class _GateWithQubitCount:
    """Backend-gate stand-in with a qubit-count attribute."""

    def __init__(self, num_qubits: int | None) -> None:
        """Initialize the fake gate.

        Args:
            num_qubits (int | None): Fake backend gate width.
        """
        self.num_qubits = num_qubits


def test_gate_matches_qubit_count_rejects_unknown_width() -> None:
    """Unknown backend-gate width should force fallback emission."""
    assert not _gate_matches_qubit_count(_GateWithoutQubitCount(), 2)
    assert not _gate_matches_qubit_count(_GateWithQubitCount(None), 2)
    assert _gate_matches_qubit_count(_GateWithQubitCount(2), 2)
    assert not _gate_matches_qubit_count(_GateWithQubitCount(1), 2)


def test_controlled_dispatch_accepts_inverse_block(monkeypatch) -> None:
    """Controlled fallback dispatch routes inverse blocks explicitly."""
    q = Value(type=QubitType(), name="q")
    op = InverseBlockOperation(
        operands=[q],
        results=[q.next_version()],
        num_target_qubits=1,
        source_block=Block(),
        implementation_block=Block(),
    )
    calls: list[tuple[list[int], list[int]]] = []

    def fake_emit_inverse(
        emit_pass: Any,
        circuit: Any,
        op: InverseBlockOperation,
        control_indices: list[int],
        target_indices: list[int],
        bindings: dict[str, Any],
    ) -> None:
        """Record the resolved controlled inverse call."""
        del emit_pass, circuit, op, bindings
        calls.append((control_indices, target_indices))

    # The dispatch site imports lazily from ``inverse_emission`` at call
    # time, so the patch targets that module rather than
    # ``controlled_emission``.
    monkeypatch.setattr(
        inverse_emission,
        "emit_inverse_block_at_indices",
        fake_emit_inverse,
    )

    emit_controlled_operations(object(), object(), [op], 7, [3], {})

    assert calls == [([7], [3])]


def test_controlled_dispatch_accepts_composite_gate(monkeypatch) -> None:
    """Controlled fallback dispatch routes composite gates explicitly."""
    q = Value(type=QubitType(), name="q")
    op = CompositeGateOperation(
        operands=[q],
        results=[q.next_version()],
        gate_type=CompositeGateType.CUSTOM,
        num_target_qubits=1,
        implementation_block=Block(),
    )
    calls: list[tuple[list[int], list[int]]] = []

    def fake_emit_composite(
        emit_pass: Any,
        circuit: Any,
        op: CompositeGateOperation,
        control_indices: list[int],
        qubit_indices: list[int],
        bindings: dict[str, Any],
    ) -> None:
        """Record the resolved controlled composite call."""
        del emit_pass, circuit, op, bindings
        calls.append((control_indices, qubit_indices))

    monkeypatch.setattr(
        controlled_emission,
        "emit_controlled_composite_at_indices",
        fake_emit_composite,
    )

    emit_controlled_operations(object(), object(), [op], 7, [3], {})

    assert calls == [([7], [3])]
