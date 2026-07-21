"""Tests for lightweight trace-time structs."""

from __future__ import annotations

import dataclasses

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.transpiler.errors import QubitConsumedError


@qmc.struct
class _RegisterPair:
    """Group two qubit handles for test kernels.

    Args:
        control (qmc.Qubit): Control qubit handle.
        target (qmc.Qubit): Target qubit handle.
    """

    control: qmc.Qubit
    target: qmc.Qubit


def _prepare_bell_pair(registers: _RegisterPair) -> _RegisterPair:
    """Prepare a Bell pair and return the successor struct.

    Args:
        registers (_RegisterPair): Input trace-time register group.

    Returns:
        _RegisterPair: Successor group containing the returned handles.
    """
    control = qmc.h(registers.control)
    control, target = qmc.cx(
        control,
        registers.target,
    )
    return _RegisterPair(
        control=control,
        target=target,
    )


def test_struct_creates_an_immutable_record() -> None:
    """Structs generate constructors but reject in-place field updates."""
    control = object()
    target = object()
    registers = _RegisterPair(control, target)  # type: ignore[arg-type]

    assert dataclasses.is_dataclass(_RegisterPair)
    assert registers.control is control
    with pytest.raises(dataclasses.FrozenInstanceError):
        registers.control = target
    with pytest.raises(AttributeError):
        registers.temporary = control  # type: ignore[attr-defined]


def test_struct_groups_handles_without_changing_the_kernel_abi() -> None:
    """A helper can return grouped handles while the IR keeps flat qubits."""

    @qmc.qkernel
    def bell_pair() -> qmc.Vector[qmc.Bit]:
        """Prepare and measure a Bell pair.

        Returns:
            qmc.Vector[qmc.Bit]: Two correlated measurement bits.
        """
        registers = _RegisterPair(qmc.qubit("control"), qmc.qubit("target"))
        registers = _prepare_bell_pair(registers)
        output = qmc.bit_array(2)
        output[0] = qmc.measure(registers.control)
        output[1] = qmc.measure(registers.target)
        return output

    estimate = bell_pair.estimate_resources()

    assert bell_pair.input_types == {}
    assert bell_pair.output_types == [qmc.Vector[qmc.Bit]]
    assert estimate.qubits == 2
    assert estimate.gates.single_qubit == 1
    assert estimate.gates.two_qubit == 1


def test_stale_struct_reuse_obeys_contained_handle_affine_rule() -> None:
    """A stale record cannot hide reuse of one of its consumed qubits."""

    @qmc.qkernel
    def stale_workspace() -> qmc.Bit:
        """Reuse a consumed qubit through an alias of its old record.

        Returns:
            qmc.Bit: Unreachable measurement result.
        """
        registers = _RegisterPair(qmc.qubit("control"), qmc.qubit("target"))
        stale = registers
        registers = _prepare_bell_pair(registers)
        return qmc.measure(stale.control)

    with pytest.raises(QubitConsumedError):
        stale_workspace.estimate_resources()
