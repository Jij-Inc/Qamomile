"""Tests for lightweight trace-time structs."""

from __future__ import annotations

import dataclasses

import pytest

import qamomile.circuit as qmc


@qmc.struct
class _RegisterPair:
    """Group two qubit handles for test kernels.

    Args:
        control (qmc.Qubit): Control qubit handle.
        target (qmc.Qubit): Target qubit handle.
    """

    control: qmc.Qubit
    target: qmc.Qubit


def _prepare_bell_pair(registers: _RegisterPair) -> None:
    """Prepare a Bell pair by updating a struct in place.

    Args:
        registers (_RegisterPair): Mutable trace-time register group.
    """
    registers.control = qmc.h(registers.control)
    registers.control, registers.target = qmc.cx(
        registers.control,
        registers.target,
    )


def test_struct_creates_a_mutable_slotted_record() -> None:
    """Structs generate constructors while retaining in-place field updates."""
    control = object()
    target = object()
    registers = _RegisterPair(control, target)  # type: ignore[arg-type]

    assert dataclasses.is_dataclass(_RegisterPair)
    assert registers.control is control
    registers.control = target
    assert registers.control is target
    with pytest.raises(AttributeError):
        registers.temporary = control  # type: ignore[attr-defined]


def test_struct_groups_handles_without_changing_the_kernel_abi() -> None:
    """A helper can mutate grouped handles while the IR keeps flat qubits."""

    @qmc.qkernel
    def bell_pair() -> qmc.Vector[qmc.Bit]:
        """Prepare and measure a Bell pair.

        Returns:
            qmc.Vector[qmc.Bit]: Two correlated measurement bits.
        """
        registers = _RegisterPair(qmc.qubit("control"), qmc.qubit("target"))
        _prepare_bell_pair(registers)
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
