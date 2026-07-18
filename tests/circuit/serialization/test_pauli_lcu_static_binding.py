"""Tests for serialized Pauli LCU block-encoding binding slots."""

from __future__ import annotations

import base64
import copy
import dataclasses
import math
import subprocess
import sys
import uuid
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

import qamomile.circuit as qmc
from qamomile.circuit.frontend.operation.inverse import _BlockInverter
from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation import ExpvalOp
from qamomile.circuit.ir.operation.callable import InvokeOperation
from qamomile.circuit.ir.operation.control_flow import (
    ForOperation,
    HasNestedOps,
    IfOperation,
)
from qamomile.circuit.ir.operation.gate import ControlledUOperation
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.operation.return_operation import ReturnOperation
from qamomile.circuit.ir.operation.select import SelectOperation
from qamomile.circuit.ir.types.primitives import BitType, QubitType
from qamomile.circuit.ir.value import ArrayValue, Value
from qamomile.circuit.serialization import deserialize, serialize
from qamomile.circuit.serialization.encode import to_dict as kernel_to_dict
from qamomile.circuit.serialization.graph_protobuf import qkernel_from_graph_dict
from qamomile.circuit.serialization.proto import qamomile_ir_pb2 as pb
from qamomile.circuit.transpiler.compiler import QamomileCompiler
from qamomile.linalg import PauliLCU

I2 = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class _AlternativeLCUBlockEncoding(qmc.LCUBlockEncoding):
    """Represent a non-Pauli producer using the common LCU contract."""


@qmc.qkernel
def _identity_case(
    signal: qmc.Vector[qmc.Qubit],
    system: qmc.Vector[qmc.Qubit],
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
    """Return both block-encoding target registers unchanged.

    Args:
        signal (qmc.Vector[qmc.Qubit]): Signal register to preserve.
        system (qmc.Vector[qmc.Qubit]): System register to preserve.

    Returns:
        tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]: Unchanged
            registers in their original order.
    """
    return signal, system


@qmc.qkernel
def _apply_static_encoding(
    signal: qmc.Vector[qmc.Qubit],
    system: qmc.Vector[qmc.Qubit],
    encoding: qmc.LCUBlockEncoding,
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
    """Apply a descriptor received as this nested qkernel's static binding.

    Args:
        signal (qmc.Vector[qmc.Qubit]): Signal register.
        system (qmc.Vector[qmc.Qubit]): System register.
        encoding (qmc.LCUBlockEncoding): Compile-time descriptor to apply.

    Returns:
        tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]: Updated signal and
            system registers.
    """
    return encoding.unitary(signal, system)


@qmc.composite_gate(name="static_encoding_box")
def _apply_static_encoding_composite(
    signal: qmc.Vector[qmc.Qubit],
    system: qmc.Vector[qmc.Qubit],
    encoding: qmc.LCUBlockEncoding,
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
    """Apply a static descriptor through the named composite protocol.

    Args:
        signal (qmc.Vector[qmc.Qubit]): Signal register.
        system (qmc.Vector[qmc.Qubit]): System register.
        encoding (qmc.LCUBlockEncoding): Compile-time descriptor to apply.

    Returns:
        tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]: Updated signal and
            system registers.
    """
    return encoding.unitary(signal, system)


@qmc.qkernel
def _apply_static_normalization(
    qubit: qmc.Qubit,
    encoding: qmc.LCUBlockEncoding,
) -> qmc.Qubit:
    """Use a scalar field from a nested qkernel's static binding.

    Args:
        qubit (qmc.Qubit): Target qubit.
        encoding (qmc.LCUBlockEncoding): Descriptor whose normalization sets
            the rotation angle.

    Returns:
        qmc.Qubit: Rotated target qubit.
    """
    return qmc.rz(qubit, encoding.normalization)


@qmc.qkernel
def _apply_two_static_normalizations(
    qubit: qmc.Qubit,
    first: qmc.LCUBlockEncoding,
    second: qmc.LCUBlockEncoding,
) -> qmc.Qubit:
    """Use two descriptors supplied through independent static arguments.

    Args:
        qubit (qmc.Qubit): Target qubit.
        first (qmc.LCUBlockEncoding): Descriptor supplying the first angle.
        second (qmc.LCUBlockEncoding): Descriptor supplying the second angle.

    Returns:
        qmc.Qubit: Qubit after both descriptor-derived rotations.
    """
    qubit = qmc.rz(qubit, first.normalization)
    return qmc.rz(qubit, second.normalization)


@qmc.qkernel
def _direct_template(
    encoding: qmc.LCUBlockEncoding,
) -> tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]:
    """Apply a compile-time-bound encoding through its direct unitary."""
    signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
    system = qmc.qubit_array(encoding.num_system_qubits, "system")
    signal, system = encoding.unitary(signal, system)
    return qmc.measure(signal), qmc.measure(system)


@qmc.qkernel
def _pauli_specific_template(
    encoding: qmc.PauliLCUBlockEncoding,
) -> tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]:
    """Retain compatibility with the original Pauli-specific annotation."""
    signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
    system = qmc.qubit_array(encoding.num_system_qubits, "system")
    signal, system = encoding.unitary(signal, system)
    return qmc.measure(signal), qmc.measure(system)


@qmc.qkernel
def _ising_specific_template(
    encoding: qmc.IsingZBlockEncoding,
) -> tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]:
    """Retain a stable producer-specific Ising-Z annotation."""
    signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
    system = qmc.qubit_array(encoding.num_system_qubits, "system")
    signal, system = encoding.unitary(signal, system)
    return qmc.measure(signal), qmc.measure(system)


@qmc.qkernel
def _inverse_template(
    encoding: qmc.LCUBlockEncoding,
) -> tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]:
    """Apply the inverse of a compile-time-bound encoding unitary."""
    signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
    system = qmc.qubit_array(encoding.num_system_qubits, "system")
    signal, system = qmc.inverse(encoding.unitary)(signal, system)
    return qmc.measure(signal), qmc.measure(system)


@qmc.qkernel
def _inverse_static_argument_template(
    encoding: qmc.LCUBlockEncoding,
) -> tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]:
    """Invert a nested qkernel that receives the descriptor as an argument."""
    signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
    system = qmc.qubit_array(encoding.num_system_qubits, "system")
    signal, system = qmc.inverse(_apply_static_encoding)(
        signal,
        system,
        encoding,
    )
    return qmc.measure(signal), qmc.measure(system)


@qmc.qkernel
def _inverse_static_composite_template(
    encoding: qmc.LCUBlockEncoding,
) -> tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]:
    """Invert a named composite that receives a static descriptor argument."""
    signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
    system = qmc.qubit_array(encoding.num_system_qubits, "system")
    signal, system = qmc.inverse(_apply_static_encoding_composite)(
        signal,
        system,
        encoding,
    )
    return qmc.measure(signal), qmc.measure(system)


@qmc.qkernel
def _inverse_static_field_template(
    encoding: qmc.LCUBlockEncoding,
) -> qmc.Bit:
    """Invert a nested qkernel that reads a static descriptor field."""
    qubit = qmc.qubit("qubit")
    qubit = qmc.inverse(_apply_static_normalization)(qubit, encoding)
    return qmc.measure(qubit)


@qmc.qkernel
def _controlled_template(
    encoding: qmc.LCUBlockEncoding,
) -> tuple[qmc.Bit, qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]:
    """Apply a compile-time-bound encoding under an outer control."""
    outer = qmc.qubit("outer")
    signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
    system = qmc.qubit_array(encoding.num_system_qubits, "system")
    outer, signal, system = qmc.control(encoding.unitary)(outer, signal, system)
    return qmc.measure(outer), qmc.measure(signal), qmc.measure(system)


@qmc.qkernel
def _select_template(
    encoding: qmc.LCUBlockEncoding,
) -> tuple[
    qmc.Vector[qmc.Bit],
    qmc.Vector[qmc.Bit],
    qmc.Vector[qmc.Bit],
]:
    """Use a compile-time-bound encoding as an outer SELECT case."""
    index = qmc.qubit_array(1, "index")
    signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
    system = qmc.qubit_array(encoding.num_system_qubits, "system")
    index, signal, system = qmc.select(
        (_identity_case, encoding.unitary),
        num_index_qubits=1,
    )(index, signal, system)
    return qmc.measure(index), qmc.measure(signal), qmc.measure(system)


@qmc.qkernel
def _nested_scalar_capture_template(
    encoding: qmc.LCUBlockEncoding,
) -> qmc.Vector[qmc.Bit]:
    """Capture a static width field in a nested qkernel allocation."""

    @qmc.qkernel
    def allocate() -> qmc.Vector[qmc.Bit]:
        """Allocate and measure the captured static signal width."""
        signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
        return qmc.measure(signal)

    return allocate()


@qmc.qkernel
def _allocate_from_width(width: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    """Allocate and measure a compile-time width argument."""
    qubits = qmc.qubit_array(width, "qubits")
    return qmc.measure(qubits)


@qmc.qkernel
def _identity_width(width: qmc.UInt) -> qmc.UInt:
    """Return a compile-time width unchanged."""
    return width


@qmc.qkernel
def _nested_scalar_passthrough_template(
    encoding: qmc.LCUBlockEncoding,
) -> qmc.Vector[qmc.Bit]:
    """Use a static width returned by an ordinary nested qkernel."""
    width = _identity_width(encoding.num_signal_qubits)
    signal = qmc.qubit_array(width, "signal")
    return qmc.measure(signal)


@qmc.qkernel
def _nested_scalar_passthrough_chain_template(
    encoding: qmc.LCUBlockEncoding,
) -> qmc.Vector[qmc.Bit]:
    """Use a static width passed through two ordered helper calls."""
    first = _identity_width(encoding.num_signal_qubits)
    second = _identity_width(first)
    signal = qmc.qubit_array(second, "signal")
    return qmc.measure(signal)


@qmc.qkernel
def _nested_width_argument_template(
    encoding: qmc.LCUBlockEncoding,
) -> tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]:
    """Pass one static width through two ordinary nested qkernel calls."""
    first = _allocate_from_width(encoding.num_signal_qubits)
    second = _allocate_from_width(encoding.num_signal_qubits)
    return first, second


@qmc.qkernel
def _loop_field_alias_template(
    encoding: qmc.LCUBlockEncoding,
) -> qmc.Bit:
    """Provide a loop-local UInt producer beside an unused static slot."""
    qubit = qmc.qubit("qubit")
    for _ in qmc.range(2):
        qubit = qmc.x(qubit)
    return qmc.measure(qubit)


@qmc.qkernel
def _both_inverse_orders_template(
    encoding: qmc.LCUBlockEncoding,
) -> tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]:
    """Apply direct-inverse and inverse-direct pairs in one template."""
    signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
    system = qmc.qubit_array(encoding.num_system_qubits, "system")
    signal, system = encoding.unitary(signal, system)
    signal, system = qmc.inverse(encoding.unitary)(signal, system)
    signal, system = qmc.inverse(encoding.unitary)(signal, system)
    signal, system = encoding.unitary(signal, system)
    return qmc.measure(signal), qmc.measure(system)


@qmc.qkernel
def _double_inverse_template(
    encoding: qmc.LCUBlockEncoding,
) -> tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]:
    """Apply a double inverse of a deferred encoding member."""

    @qmc.qkernel
    def inverse_member(
        signal: qmc.Vector[qmc.Qubit],
        system: qmc.Vector[qmc.Qubit],
    ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
        """Expose the deferred inverse behind a composable qkernel wrapper."""
        return qmc.inverse(encoding.unitary)(signal, system)

    signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
    system = qmc.qubit_array(encoding.num_system_qubits, "system")
    signal, system = qmc.inverse(inverse_member)(signal, system)
    return qmc.measure(signal), qmc.measure(system)


@qmc.qkernel
def _controlled_inverse_template(
    encoding: qmc.LCUBlockEncoding,
) -> tuple[qmc.Bit, qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]:
    """Apply the inverse deferred member under an outer control."""

    @qmc.qkernel
    def inverse_member(
        signal: qmc.Vector[qmc.Qubit],
        system: qmc.Vector[qmc.Qubit],
    ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
        """Expose the deferred inverse behind a composable qkernel wrapper."""
        return qmc.inverse(encoding.unitary)(signal, system)

    outer = qmc.qubit("outer")
    signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
    system = qmc.qubit_array(encoding.num_system_qubits, "system")
    outer, signal, system = qmc.control(inverse_member)(
        outer,
        signal,
        system,
    )
    return qmc.measure(outer), qmc.measure(signal), qmc.measure(system)


@qmc.qkernel
def _inverse_select_template(
    encoding: qmc.LCUBlockEncoding,
) -> tuple[
    qmc.Vector[qmc.Bit],
    qmc.Vector[qmc.Bit],
    qmc.Vector[qmc.Bit],
]:
    """Use the inverse deferred member as an outer SELECT case."""

    @qmc.qkernel
    def inverse_member(
        signal: qmc.Vector[qmc.Qubit],
        system: qmc.Vector[qmc.Qubit],
    ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
        """Expose the deferred inverse behind a composable qkernel wrapper."""
        return qmc.inverse(encoding.unitary)(signal, system)

    index = qmc.qubit_array(1, "index")
    signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
    system = qmc.qubit_array(encoding.num_system_qubits, "system")
    index, signal, system = qmc.select(
        (_identity_case, inverse_member),
        num_index_qubits=1,
    )(index, signal, system)
    return qmc.measure(index), qmc.measure(signal), qmc.measure(system)


@qmc.qkernel
def _wrong_signal_direct(
    encoding: qmc.LCUBlockEncoding,
) -> qmc.Vector[qmc.Bit]:
    """Call a two-signal encoding with one explicit signal qubit."""
    signal = qmc.qubit_array(1, "signal")
    system = qmc.qubit_array(1, "system")
    signal, _ = encoding.unitary(signal, system)
    return qmc.measure(signal)


@qmc.qkernel
def _wrong_signal_inverse(
    encoding: qmc.LCUBlockEncoding,
) -> qmc.Vector[qmc.Bit]:
    """Inverse-call a two-signal encoding with one explicit signal qubit."""
    signal = qmc.qubit_array(1, "signal")
    system = qmc.qubit_array(1, "system")
    signal, _ = qmc.inverse(encoding.unitary)(signal, system)
    return qmc.measure(signal)


@qmc.qkernel
def _wrong_signal_control(
    encoding: qmc.LCUBlockEncoding,
) -> qmc.Vector[qmc.Bit]:
    """Control-call a two-signal encoding with one explicit signal qubit."""
    outer = qmc.qubit("outer")
    signal = qmc.qubit_array(1, "signal")
    system = qmc.qubit_array(1, "system")
    _, signal, _ = qmc.control(encoding.unitary)(outer, signal, system)
    return qmc.measure(signal)


@qmc.qkernel
def _wrong_signal_select(
    encoding: qmc.LCUBlockEncoding,
) -> qmc.Vector[qmc.Bit]:
    """SELECT-call a two-signal encoding with one explicit signal qubit."""
    index = qmc.qubit_array(1, "index")
    signal = qmc.qubit_array(1, "signal")
    system = qmc.qubit_array(1, "system")
    _, signal, _ = qmc.select(
        (_identity_case, encoding.unitary),
        num_index_qubits=1,
    )(index, signal, system)
    return qmc.measure(signal)


@qmc.qkernel
def _wrong_system_direct(
    encoding: qmc.LCUBlockEncoding,
) -> qmc.Vector[qmc.Bit]:
    """Call a one-system encoding with two explicit system qubits."""
    signal = qmc.qubit_array(2, "signal")
    system = qmc.qubit_array(2, "system")
    signal, _ = encoding.unitary(signal, system)
    return qmc.measure(signal)


@qmc.qkernel
def _phase_sample_template(encoding: qmc.LCUBlockEncoding) -> qmc.Bit:
    """Detect a bound encoding's phase with an outer Hadamard test."""
    outer = qmc.h(qmc.qubit("outer"))
    signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
    system = qmc.qubit_array(encoding.num_system_qubits, "system")
    outer, _, _ = qmc.control(encoding.unitary)(outer, signal, system)
    outer = qmc.h(outer)
    return qmc.measure(outer)


@qmc.qkernel
def _semantic_sample_template(encoding: qmc.LCUBlockEncoding) -> qmc.Bit:
    """Measure the system output of a statically bound encoding."""
    signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
    system = qmc.qubit_array(encoding.num_system_qubits, "system")
    _, system = encoding.unitary(signal, system)
    return qmc.measure(system[0])


@qmc.qkernel
def _semantic_expval_template(
    encoding: qmc.LCUBlockEncoding,
    observable: qmc.Observable,
) -> qmc.Float:
    """Evaluate an observable after a statically bound encoding."""
    signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
    system = qmc.qubit_array(encoding.num_system_qubits, "system")
    _, system = encoding.unitary(signal, system)
    return qmc.expval(system[0], observable)


@qmc.qkernel
def _normalization_runtime_template(
    encoding: qmc.LCUBlockEncoding,
    theta: qmc.Float,
) -> qmc.Bit:
    """Combine one static normalization field with a runtime parameter."""
    qubit = qmc.qubit("qubit")
    qubit = qmc.rx(qubit, encoding.normalization + theta)
    return qmc.measure(qubit)


@qmc.qkernel
def _complex_inverse_sample_template(
    encoding: qmc.LCUBlockEncoding,
) -> qmc.Bit:
    """Convert deferred-inverse phase kickback into a population."""

    @qmc.qkernel
    def inverse_member(
        signal: qmc.Vector[qmc.Qubit],
        system: qmc.Vector[qmc.Qubit],
    ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
        """Expose the deferred inverse through a controllable qkernel."""
        return qmc.inverse(encoding.unitary)(signal, system)

    outer = qmc.h(qmc.qubit("outer"))
    signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
    system = qmc.qubit_array(encoding.num_system_qubits, "system")
    outer, _, _ = qmc.control(inverse_member)(outer, signal, system)
    outer = qmc.sdg(outer)
    outer = qmc.h(outer)
    return qmc.measure(outer)


@qmc.qkernel
def _complex_inverse_expval_template(
    encoding: qmc.LCUBlockEncoding,
    observable: qmc.Observable,
) -> qmc.Float:
    """Estimate conjugated phase kickback from a deferred inverse."""

    @qmc.qkernel
    def inverse_member(
        signal: qmc.Vector[qmc.Qubit],
        system: qmc.Vector[qmc.Qubit],
    ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
        """Expose the deferred inverse through a controllable qkernel."""
        return qmc.inverse(encoding.unitary)(signal, system)

    outer = qmc.h(qmc.qubit("outer"))
    signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
    system = qmc.qubit_array(encoding.num_system_qubits, "system")
    outer, _, _ = qmc.control(inverse_member)(outer, signal, system)
    return qmc.expval(outer, observable)


@qmc.qkernel
def _complex_static_argument_inverse_sample_template(
    encoding: qmc.LCUBlockEncoding,
) -> qmc.Bit:
    """Sample phase from an inverted qkernel with a static descriptor input."""

    @qmc.qkernel
    def inverse_member(
        signal: qmc.Vector[qmc.Qubit],
        system: qmc.Vector[qmc.Qubit],
    ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
        """Expose the nested static-argument inverse under outer control."""
        return qmc.inverse(_apply_static_encoding)(signal, system, encoding)

    outer = qmc.h(qmc.qubit("outer"))
    signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
    system = qmc.qubit_array(encoding.num_system_qubits, "system")
    outer, _, _ = qmc.control(inverse_member)(outer, signal, system)
    outer = qmc.sdg(outer)
    outer = qmc.h(outer)
    return qmc.measure(outer)


@qmc.qkernel
def _complex_static_argument_inverse_expval_template(
    encoding: qmc.LCUBlockEncoding,
    observable: qmc.Observable,
) -> qmc.Float:
    """Estimate phase from an inverted static-descriptor qkernel."""

    @qmc.qkernel
    def inverse_member(
        signal: qmc.Vector[qmc.Qubit],
        system: qmc.Vector[qmc.Qubit],
    ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
        """Expose the nested static-argument inverse under outer control."""
        return qmc.inverse(_apply_static_encoding)(signal, system, encoding)

    outer = qmc.h(qmc.qubit("outer"))
    signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
    system = qmc.qubit_array(encoding.num_system_qubits, "system")
    outer, _, _ = qmc.control(inverse_member)(outer, signal, system)
    return qmc.expval(outer, observable)


@qmc.qkernel
def _inverse_tuple_expval_template(
    encoding: qmc.LCUBlockEncoding,
    observable: qmc.Observable,
) -> qmc.Float:
    """Estimate two qubits after applying a deferred inverse member."""
    signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
    system = qmc.qubit_array(encoding.num_system_qubits, "system")
    signal, system = qmc.inverse(encoding.unitary)(signal, system)
    return qmc.expval((signal[0], system[0]), observable)


@qmc.qkernel
def _complex_control_sample_template(
    encoding: qmc.LCUBlockEncoding,
) -> qmc.Bit:
    """Convert outer-control phase kickback into a sampled population."""
    outer = qmc.h(qmc.qubit("outer"))
    signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
    system = qmc.qubit_array(encoding.num_system_qubits, "system")
    outer, _, _ = qmc.control(encoding.unitary)(outer, signal, system)
    outer = qmc.sdg(outer)
    outer = qmc.h(outer)
    return qmc.measure(outer)


@qmc.qkernel
def _complex_control_expval_template(
    encoding: qmc.LCUBlockEncoding,
    observable: qmc.Observable,
) -> qmc.Float:
    """Estimate phase kickback from a deferred controlled member."""
    outer = qmc.h(qmc.qubit("outer"))
    signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
    system = qmc.qubit_array(encoding.num_system_qubits, "system")
    outer, _, _ = qmc.control(encoding.unitary)(outer, signal, system)
    return qmc.expval(outer, observable)


@qmc.qkernel
def _complex_select_sample_template(
    encoding: qmc.LCUBlockEncoding,
) -> qmc.Bit:
    """Convert nested-SELECT phase interference into a population."""
    index = qmc.qubit_array(1, "index")
    index[0] = qmc.h(index[0])
    signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
    system = qmc.qubit_array(encoding.num_system_qubits, "system")
    index, _, _ = qmc.select(
        (_identity_case, encoding.unitary),
        num_index_qubits=1,
    )(index, signal, system)
    index[0] = qmc.sdg(index[0])
    index[0] = qmc.h(index[0])
    return qmc.measure(index[0])


@qmc.qkernel
def _complex_select_expval_template(
    encoding: qmc.LCUBlockEncoding,
    observable: qmc.Observable,
) -> qmc.Float:
    """Estimate relative phase from a deferred nested-SELECT member."""
    index = qmc.qubit_array(1, "index")
    index[0] = qmc.h(index[0])
    signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
    system = qmc.qubit_array(encoding.num_system_qubits, "system")
    index, _, _ = qmc.select(
        (_identity_case, encoding.unitary),
        num_index_qubits=1,
    )(index, signal, system)
    return qmc.expval(index[0], observable)


def _encoding(matrix: np.ndarray) -> qmc.LCUBlockEncoding:
    """Build a Pauli LCU block encoding from a one-qubit matrix.

    Args:
        matrix (np.ndarray): One-qubit matrix to decompose.

    Returns:
        qmc.LCUBlockEncoding: Static descriptor for ``matrix``.
    """
    return qmc.pauli_lcu_block_encoding(PauliLCU.from_matrix(matrix, atol=1e-12))


def _recursive_ising_encoding() -> qmc.LCUBlockEncoding:
    r"""Build a two-level complex encoding without Pauli LCU conversion.

    Returns:
        qmc.LCUBlockEncoding: Encoding of ``0.5 Z + 0.75j I`` with
            normalization ``1.75``.
    """
    inner = qmc.ising_z_block_encoding({(): 1.0j, (0,): 0.5}, 1)
    return qmc.lcu_block_encoding(
        (
            qmc.LCUBlockEncodingTerm(1.0, inner),
            qmc.LCUBlockEncodingTerm(-0.25j, qmc.identity_block_encoding(1)),
        )
    )


def _alternative_encoding(matrix: np.ndarray) -> qmc.LCUBlockEncoding:
    """Wrap one unitary as a distinct producer-specific LCU descriptor.

    Args:
        matrix (np.ndarray): One-qubit matrix whose Pauli unitary supplies the
            test implementation.

    Returns:
        qmc.LCUBlockEncoding: Alternative nominal descriptor subtype with the
            same exact block semantics.
    """
    encoding = _encoding(matrix)
    return _AlternativeLCUBlockEncoding(
        unitary=encoding.unitary,
        normalization=encoding.normalization,
        num_signal_qubits=encoding.num_signal_qubits,
        num_system_qubits=encoding.num_system_qubits,
    )


def _generic_encoding(matrix: np.ndarray) -> qmc.LCUBlockEncoding:
    """Erase producer identity while preserving an exact LCU implementation.

    Args:
        matrix (np.ndarray): One-qubit matrix whose Pauli unitary supplies the
            test implementation.

    Returns:
        qmc.LCUBlockEncoding: Exact instance of the common descriptor class.
    """
    encoding = _encoding(matrix)
    return qmc.LCUBlockEncoding(
        unitary=encoding.unitary,
        normalization=encoding.normalization,
        num_signal_qubits=encoding.num_signal_qubits,
        num_system_qubits=encoding.num_system_qubits,
    )


def _reachable_blocks(root: Block) -> Iterator[Block]:
    """Yield each block reachable through owned operation bodies once.

    Args:
        root (Block): Root block whose owned graph should be traversed.

    Yields:
        Block: Reachable block, including ``root``.
    """
    pending = [root]
    seen: set[int] = set()
    while pending:
        block = pending.pop()
        if id(block) in seen:
            continue
        seen.add(id(block))
        yield block
        for operation in block.operations:
            if isinstance(operation, HasNestedOps):
                for nested_operations in operation.nested_op_lists():
                    pending.append(Block(operations=list(nested_operations)))
            if isinstance(operation, InvokeOperation):
                definition = operation.definition
                if definition is not None:
                    if definition.body is not None:
                        pending.append(definition.body)
                    for implementation in definition.implementations:
                        if implementation.body is not None:
                            pending.append(implementation.body)
            elif isinstance(operation, ControlledUOperation):
                if operation.block is not None:
                    pending.append(operation.block)
            elif isinstance(operation, InverseBlockOperation):
                if operation.source_block is not None:
                    pending.append(operation.source_block)
                if operation.implementation_block is not None:
                    pending.append(operation.implementation_block)
            elif isinstance(operation, SelectOperation):
                pending.extend(operation.case_blocks)


def _assert_static_binding_resolved(block: Block) -> None:
    """Assert that no static slot or deferred body reference remains.

    Args:
        block (Block): Specialized block to inspect recursively.

    Returns:
        None: Assertions fail if a static marker remains.
    """
    for reachable in _reachable_blocks(block):
        assert reachable.static_bindings == ()
        for operation in reachable.operations:
            if not isinstance(operation, InvokeOperation):
                continue
            body_ref = operation.body_ref
            assert body_ref is None or body_ref.kind != "static_binding"
            definition = operation.definition
            if definition is None:
                continue
            body_ref = definition.body_ref
            assert body_ref is None or body_ref.kind != "static_binding"
            for implementation in definition.implementations:
                body_ref = implementation.body_ref
                assert body_ref is None or body_ref.kind != "static_binding"


def _executor(case: Any) -> Any:
    """Return a local simulator executor for one SDK fixture case.

    Args:
        case (Any): Fixture carrying a backend name and transpiler.

    Returns:
        Any: Backend-specific local executor.
    """
    if case.backend_name == "qiskit":
        from qiskit.providers.basic_provider import BasicSimulator

        return case.transpiler.executor(backend=BasicSimulator())
    return case.transpiler.executor()


def _tuple_expval_carrier(
    kernel: Any,
) -> tuple[Block, ExpvalOp, ArrayValue]:
    """Locate the tuple-expval carrier in one serialized qkernel graph.

    Args:
        kernel (Any): QKernel-like object exposing a root ``block``.

    Returns:
        tuple[Block, ExpvalOp, ArrayValue]: Owning block, expval operation, and
            its synthetic tuple carrier.

    Raises:
        AssertionError: If no tuple-form expval operation exists.
    """
    for block in _reachable_blocks(kernel.block):
        for operation in block.operations:
            if isinstance(operation, ExpvalOp) and isinstance(
                operation.operands[0],
                ArrayValue,
            ):
                return block, operation, operation.operands[0]
    raise AssertionError("no tuple-form ExpvalOp")


def _protobuf_message(kernel: Any) -> pb.QKernel:
    """Return the canonical protobuf message for one qkernel.

    Args:
        kernel (Any): QKernel-like object to serialize.

    Returns:
        pb.QKernel: Parsed canonical protobuf message.
    """
    message = pb.QKernel()
    message.ParseFromString(serialize(kernel))
    return message


def _replace_payload_map_string(
    payload: pb.Payload,
    key: str,
    value: str,
) -> None:
    """Replace one string-valued entry in a protobuf payload map.

    Args:
        payload (pb.Payload): Map payload to mutate in place.
        key (str): Existing string key to locate.
        value (str): Replacement string value.

    Returns:
        None: ``payload`` is mutated in place.

    Raises:
        AssertionError: If ``key`` is absent or does not map to a string.
    """
    assert payload.HasField("map_value")
    for entry in payload.map_value.entries:
        if entry.key.string_value != key:
            continue
        assert entry.value.HasField("string_value")
        entry.value.string_value = value
        return
    raise AssertionError(f"payload map has no string entry {key!r}")


def _reorder_value_table(message: pb.QKernel, ordered_refs: list[str]) -> None:
    """Reorder a protobuf value table to a specified canonical visit order.

    Args:
        message (pb.QKernel): QKernel message to mutate in place.
        ordered_refs (list[str]): UUIDs in semantic encoder visit order.

    Returns:
        None: ``message.value_table`` is replaced in place.

    Raises:
        AssertionError: If ``ordered_refs`` is not a permutation of the table.
    """
    values: dict[str, pb.ValueNode] = {}
    for value in message.value_table:
        owned = pb.ValueNode()
        owned.CopyFrom(value)
        values[value.uuid] = owned
    assert len(values) == len(message.value_table)
    assert len(ordered_refs) == len(set(ordered_refs))
    assert set(ordered_refs) == set(values)
    del message.value_table[:]
    for ref in ordered_refs:
        message.value_table.add().CopyFrom(values[ref])


def test_unbound_template_uses_only_a_typed_static_manifest_slot() -> None:
    """The encoding parameter is absent from the ordinary value ABI."""
    block = _direct_template.block

    assert block.kind is BlockKind.HIERARCHICAL
    assert block.label_args == []
    assert block.input_values == []
    assert block.param_slots == ()
    assert len(block.static_bindings) == 1
    slot = block.static_bindings[0]
    assert slot.name == "encoding"
    assert slot.type_key == "qamomile.stdlib.lcu_block_encoding"
    assert [field.name for field in slot.fields] == [
        "normalization",
        "num_signal_qubits",
        "num_system_qubits",
    ]


def test_static_manifest_serialization_is_deterministic() -> None:
    """The unbound template round-trips without a concrete descriptor."""
    payload = serialize(_direct_template)
    restored = deserialize(payload)

    assert serialize(restored) == payload
    assert list(restored.signature.parameters) == ["encoding"]
    assert restored.input_types == {"encoding": qmc.LCUBlockEncoding}
    assert restored.block.label_args == []
    assert len(restored.block.static_bindings) == 1


def test_pauli_specific_static_manifest_remains_compatible() -> None:
    """The original Pauli annotation keeps its stable serialized adapter."""
    payload = serialize(_pauli_specific_template)
    restored = deserialize(payload)

    assert restored.input_types == {"encoding": qmc.PauliLCUBlockEncoding}
    assert restored.block.static_bindings[0].type_key == (
        "qamomile.stdlib.pauli_lcu_block_encoding"
    )
    specialized = restored.build(encoding=_encoding(I2))
    _assert_static_binding_resolved(specialized)


def test_ising_specific_static_manifest_has_a_stable_adapter() -> None:
    """The Ising-Z subtype annotation round-trips with its nominal type key."""
    payload = serialize(_ising_specific_template)
    restored = deserialize(payload)

    assert restored.input_types == {"encoding": qmc.IsingZBlockEncoding}
    assert restored.block.static_bindings[0].type_key == (
        "qamomile.stdlib.ising_z_block_encoding"
    )
    specialized = restored.build(
        encoding=qmc.ising_z_block_encoding({(): 1.0j, (0,): 0.5}, 1)
    )
    _assert_static_binding_resolved(specialized)


def test_graph_encoder_rejects_an_empty_static_binding_type() -> None:
    """An empty type key cannot satisfy the graph parameter type union."""
    envelope = kernel_to_dict(_direct_template)
    envelope["artifact"]["parameters"][0]["static_binding_type"] = ""

    with pytest.raises(ValueError, match="requires exactly one"):
        qkernel_from_graph_dict(envelope)


def test_protobuf_decoder_rejects_an_empty_static_binding_type() -> None:
    """A present-but-empty optional protobuf string is still malformed."""
    message = _protobuf_message(_direct_template)
    message.parameters[0].static_binding_type = ""
    assert message.parameters[0].HasField("static_binding_type")

    with pytest.raises(ValueError, match="requires exactly one"):
        deserialize(message.SerializeToString(deterministic=True))


def test_deserialize_rejects_consistently_forged_static_member() -> None:
    """A canonical marker cannot name an unregistered descriptor member."""
    message = _protobuf_message(_direct_template)
    marker_entry = next(
        entry
        for entry in message.callable_table
        if entry.definition.HasField("body_ref")
        and entry.definition.body_ref.kind == "static_binding"
    )
    marker_operation = next(
        operation
        for entry in message.callable_table
        if entry.definition.HasField("body")
        for operation in entry.definition.body.operations
        if operation.definition_ref == marker_entry.id
    )

    for attrs in (
        marker_operation.attrs,
        marker_entry.definition.attrs,
        marker_entry.definition.body_ref.attrs,
    ):
        _replace_payload_map_string(attrs, "member", "bogus")

    payload = message.SerializeToString(deterministic=True)
    with pytest.raises(ValueError, match="static binding.*member"):
        deserialize(payload)


@pytest.mark.parametrize("malformation", ["array", "parameterized"])
def test_deserialize_rejects_nonbare_static_scalar_field(
    malformation: str,
) -> None:
    """A static scalar projection must remain a bare scalar ``Value``."""
    message = _protobuf_message(_direct_template)
    normalization_ref = message.body.static_bindings[0].fields[0].value_ref
    normalization = next(
        value for value in message.value_table if value.uuid == normalization_ref
    )
    assert normalization.value_type.kind == pb.FLOAT_TYPE

    if malformation == "array":
        normalization.value_kind = pb.ARRAY_VALUE
    else:
        normalization.metadata.scalar.const_value.null_value.SetInParent()
        normalization.metadata.scalar.parameter_name = "forged"

    payload = message.SerializeToString(deterministic=True)
    with pytest.raises(ValueError, match="(?i)static binding.*field"):
        deserialize(payload)


def test_deserialize_rejects_static_field_aliasing_owned_input_shape() -> None:
    """A same-typed child dimension cannot replace a root static field."""
    message = _protobuf_message(_direct_template)
    root_slot = message.body.static_bindings[0]
    original_signal_width_ref = root_slot.fields[1].value_ref
    wrapper_entry = next(
        entry
        for entry in message.callable_table
        if entry.definition.HasField("body")
        and entry.definition.body.name == "encoding.unitary"
    )
    wrapper = wrapper_entry.definition.body
    values = {value.uuid: value for value in message.value_table}
    wrapper_signal_ref = wrapper.input_value_refs[0]
    wrapper_signal_width_ref = values[wrapper_signal_ref].shape_refs[0]
    assert values[wrapper_signal_width_ref].value_type.kind == pb.UINT_TYPE
    assert values[original_signal_width_ref].value_type.kind == pb.UINT_TYPE

    root_slot.fields[1].value_ref = wrapper_signal_width_ref
    marker_operation = wrapper.operations[0]
    root_operations = message.body.operations
    wrapper_system_ref = wrapper.input_value_refs[1]
    wrapper_system_width_ref = values[wrapper_system_ref].shape_refs[0]
    _reorder_value_table(
        message,
        [
            root_slot.fields[0].value_ref,
            wrapper_signal_width_ref,
            root_slot.fields[2].value_ref,
            root_operations[0].result_refs[0],
            original_signal_width_ref,
            root_operations[1].result_refs[0],
            *root_operations[2].result_refs,
            wrapper_signal_ref,
            wrapper_system_ref,
            wrapper_system_width_ref,
            *marker_operation.result_refs,
            root_operations[3].result_refs[0],
            root_operations[4].result_refs[0],
        ],
    )

    payload = message.SerializeToString(deterministic=True)
    with pytest.raises(ValueError, match="aliases root static binding fields"):
        deserialize(payload)


def test_deserialize_rejects_static_slot_on_owned_child_block() -> None:
    """Only the root qkernel may declare a static-binding manifest."""
    message = _protobuf_message(_direct_template)
    wrapper = next(
        entry.definition.body
        for entry in message.callable_table
        if entry.definition.HasField("body")
        and entry.definition.body.name == "encoding.unitary"
    )
    wrapper.static_bindings.add().CopyFrom(message.body.static_bindings[0])

    payload = message.SerializeToString(deterministic=True)
    with pytest.raises(ValueError, match="only the root qkernel owns binding slots"):
        deserialize(payload)


def test_serialize_rejects_static_field_logical_id_alias() -> None:
    """A static field cannot move to a new UUID with its old logical ID."""
    restored = deserialize(serialize(_direct_template))
    slot = restored.block.static_bindings[0]
    fields = list(slot.fields)
    original = fields[1]
    fields[1] = dataclasses.replace(
        original,
        value=dataclasses.replace(original.value, uuid=str(uuid.uuid4())),
    )
    restored.block.static_bindings = (dataclasses.replace(slot, fields=tuple(fields)),)

    with pytest.raises(ValueError, match="aliases static binding logical identities"):
        serialize(restored)


def test_serialize_rejects_static_field_aliasing_loop_local_producer() -> None:
    """A loop variable cannot also be a static-field producer."""
    restored = deserialize(serialize(_loop_field_alias_template))
    loop = next(
        operation
        for operation in restored.block.operations
        if isinstance(operation, ForOperation)
    )
    assert loop.loop_var_value is not None
    slot = restored.block.static_bindings[0]
    fields = list(slot.fields)
    fields[1] = dataclasses.replace(fields[1], value=loop.loop_var_value)
    restored.block.static_bindings = (dataclasses.replace(slot, fields=tuple(fields)),)

    with pytest.raises(ValueError, match="loop variable aliases a static binding"):
        serialize(restored)


def test_deserialized_template_rejects_missing_wrong_and_runtime_bindings() -> None:
    """Static descriptors are mandatory compile-time bindings of exact type."""
    restored = deserialize(serialize(_direct_template))

    with pytest.raises(ValueError, match="must be provided through bindings"):
        restored.build()
    with pytest.raises(TypeError, match="must be LCUBlockEncoding"):
        restored.build(encoding=object())
    with pytest.raises(TypeError, match="cannot be runtime parameters"):
        restored.build(parameters=["encoding"])


def test_deserialized_template_is_reusable_without_mutation() -> None:
    """One payload materializes zero, single, and complex multi-term LCUs."""
    payload = serialize(_direct_template)
    restored = deserialize(payload)
    encodings = (
        _encoding(np.zeros((2, 2), dtype=np.complex128)),
        _encoding(np.exp(0.37j) * I2),
        _encoding(1j * I2 + 0.5 * X + (0.2 - 0.3j) * Z),
        qmc.pauli_lcu_block_encoding(
            PauliLCU.from_matrix(
                np.exp(-0.23j) * np.eye(4, dtype=np.complex128),
                atol=1e-12,
            )
        ),
    )

    assert [encoding.num_signal_qubits for encoding in encodings] == [1, 1, 2, 1]
    assert [encoding.num_system_qubits for encoding in encodings] == [1, 1, 1, 2]
    for encoding in encodings:
        specialized = restored.build(encoding=encoding)
        assert specialized.kind is BlockKind.TRACED
        _assert_static_binding_resolved(specialized)
        assert serialize(restored) == payload


def test_generic_slot_accepts_an_alternative_lcu_descriptor_subclass() -> None:
    """One generic payload accepts common and producer-specific descriptors."""
    payload = serialize(_direct_template)
    restored = deserialize(payload)
    pauli = _encoding(1j * I2 + 0.5 * X)
    generic = _generic_encoding(1j * I2 + 0.5 * X)
    alternative = _alternative_encoding(1j * I2 + 0.5 * X)
    ising = qmc.ising_z_block_encoding({(): 1.0j, (0,): 0.5}, 1)
    recursive = _recursive_ising_encoding()

    for encoding in (generic, pauli, alternative, ising, recursive):
        specialized = restored.build(encoding=encoding)
        assert specialized.kind is BlockKind.TRACED
        _assert_static_binding_resolved(specialized)
        assert serialize(restored) == payload


def test_nested_qkernel_capture_of_static_width_is_materialized() -> None:
    """Lexically captured scalar fields are replaced in owned blocks."""
    restored = deserialize(serialize(_nested_scalar_capture_template))
    encoding = _encoding(1j * I2 + 0.5 * X + (0.2 - 0.3j) * Z)

    specialized = restored.build(encoding=encoding)

    _assert_static_binding_resolved(specialized)
    allocation_widths = [
        operation.results[0].shape[0].get_const()
        for block in _reachable_blocks(specialized)
        for operation in block.operations
        if isinstance(operation, QInitOperation)
        and operation.results
        and hasattr(operation.results[0], "shape")
    ]
    assert encoding.num_signal_qubits in allocation_widths
    QamomileCompiler().prepare(restored, bindings={"encoding": encoding})


def test_static_width_passed_to_repeated_nested_calls_is_materialized() -> None:
    """Array result shapes retain the bound static width without versioning."""
    restored = deserialize(serialize(_nested_width_argument_template))
    encoding = _encoding(1j * I2 + 0.5 * X + (0.2 - 0.3j) * Z)

    specialized = restored.build(encoding=encoding)

    _assert_static_binding_resolved(specialized)
    result_widths = [
        result.shape[0].get_const()
        for block in _reachable_blocks(specialized)
        for operation in block.operations
        for result in operation.results
        if hasattr(result, "shape") and result.shape
    ]
    assert result_widths.count(encoding.num_signal_qubits) >= 2


def test_static_width_returned_from_nested_qkernel_is_materialized() -> None:
    """A validated scalar pass-through receives the bound field value."""
    restored = deserialize(serialize(_nested_scalar_passthrough_template))
    encoding = qmc.pauli_lcu_block_encoding(PauliLCU.from_matrix(I2))

    built = restored.build(encoding=encoding)

    allocation_widths = [
        operation.results[0].shape[0].get_const()
        for block in _reachable_blocks(built)
        for operation in block.operations
        if isinstance(operation, QInitOperation)
        and operation.results
        and hasattr(operation.results[0], "shape")
    ]
    assert encoding.num_signal_qubits in allocation_widths
    QamomileCompiler().prepare(restored, bindings={"encoding": encoding})


def test_tuple_expval_carrier_round_trips_with_static_binding() -> None:
    """A tuple-expval carrier resolves its locally produced parent arrays."""
    payload = serialize(_inverse_tuple_expval_template)
    restored = deserialize(payload)
    encoding = _encoding(1j * I2 + 0.5 * X)

    specialized = restored.build(encoding=encoding)

    assert serialize(restored) == payload
    _assert_static_binding_resolved(specialized)


def test_serialize_rejects_unknown_tuple_expval_parent() -> None:
    """An inline expval carrier cannot name a foreign root array."""
    restored = copy.deepcopy(deserialize(serialize(_inverse_tuple_expval_template)))
    _, expval_operation, carrier = _tuple_expval_carrier(restored)
    runtime = carrier.metadata.array_runtime
    assert runtime is not None
    forged_runtime = dataclasses.replace(
        runtime,
        element_parent_uuids=("foreign-array", *runtime.element_parent_uuids[1:]),
    )
    expval_operation.operands[0] = dataclasses.replace(
        carrier,
        metadata=dataclasses.replace(
            carrier.metadata,
            array_runtime=forged_runtime,
        ),
    )

    with pytest.raises(ValueError, match="without a local producer"):
        serialize(restored)


def test_serialize_rejects_tuple_expval_carrier_used_by_return() -> None:
    """The inline-carrier exception applies only to one expval operand."""
    restored = copy.deepcopy(deserialize(serialize(_inverse_tuple_expval_template)))
    block, _, carrier = _tuple_expval_carrier(restored)
    block.operations.append(ReturnOperation(operands=[carrier]))

    with pytest.raises(ValueError, match="without a local producer"):
        serialize(restored)


def test_serialize_rejects_tuple_expval_carrier_reuse() -> None:
    """One synthetic tuple carrier cannot serve two expval operations."""
    restored = copy.deepcopy(deserialize(serialize(_inverse_tuple_expval_template)))
    block, expval_operation, _ = _tuple_expval_carrier(restored)
    duplicate = dataclasses.replace(
        expval_operation,
        operands=list(expval_operation.operands),
        results=[expval_operation.results[0].next_version()],
    )
    block.operations.insert(block.operations.index(expval_operation) + 1, duplicate)

    with pytest.raises(ValueError, match="without a local producer"):
        serialize(restored)


def test_serialize_rejects_tuple_carrier_observable_uuid_collision() -> None:
    """A tuple carrier cannot impersonate the expval observable operand."""
    restored = copy.deepcopy(deserialize(serialize(_inverse_tuple_expval_template)))
    _, expval_operation, carrier = _tuple_expval_carrier(restored)
    expval_operation.operands[0] = dataclasses.replace(
        carrier,
        uuid=expval_operation.operands[1].uuid,
    )

    with pytest.raises(ValueError, match="conflicting value identities"):
        serialize(restored)


def test_serialize_rejects_tuple_carrier_self_reference() -> None:
    """Tuple metadata cannot name its synthetic carrier as an element."""
    restored = copy.deepcopy(deserialize(serialize(_inverse_tuple_expval_template)))
    _, expval_operation, carrier = _tuple_expval_carrier(restored)
    runtime = carrier.metadata.array_runtime
    assert runtime is not None
    forged_runtime = dataclasses.replace(
        runtime,
        element_uuids=(carrier.uuid, *runtime.element_uuids[1:]),
        element_logical_ids=(
            carrier.logical_id,
            *runtime.element_logical_ids[1:],
        ),
    )
    expval_operation.operands[0] = dataclasses.replace(
        carrier,
        metadata=dataclasses.replace(
            carrier.metadata,
            array_runtime=forged_runtime,
        ),
    )

    with pytest.raises(ValueError, match="without a local producer"):
        serialize(restored)


def test_serialize_rejects_tuple_carrier_from_sibling_region() -> None:
    """A carrier produced in one branch cannot be consumed by its sibling."""
    restored = copy.deepcopy(deserialize(serialize(_inverse_tuple_expval_template)))
    block, expval_operation, carrier = _tuple_expval_carrier(restored)
    sibling_carrier = dataclasses.replace(
        carrier,
        uuid=str(uuid.uuid4()),
        logical_id=str(uuid.uuid4()),
    )
    sibling_expval = dataclasses.replace(
        expval_operation,
        operands=[sibling_carrier, expval_operation.operands[1]],
        results=[expval_operation.results[0].next_version()],
    )
    branch = IfOperation(
        operands=[Value(type=BitType(), name="condition").with_const(True)],
        true_operations=[QInitOperation(operands=[], results=[sibling_carrier])],
        false_operations=[sibling_expval],
    )
    block.operations.insert(block.operations.index(expval_operation), branch)

    with pytest.raises(ValueError, match="without a local producer"):
        serialize(restored)


def test_serialize_rejects_tuple_expval_parent_after_use() -> None:
    """A tuple carrier cannot refer to a producer after the expval use."""
    restored = copy.deepcopy(deserialize(serialize(_inverse_tuple_expval_template)))
    block, expval_operation, carrier = _tuple_expval_carrier(restored)
    runtime = carrier.metadata.array_runtime
    assert runtime is not None
    parent_uuid = runtime.element_parent_uuids[0]
    producer_index = next(
        index
        for index, operation in enumerate(block.operations)
        if any(result.uuid == parent_uuid for result in operation.results)
    )
    expval_index = block.operations.index(expval_operation)
    producer = block.operations.pop(producer_index)
    if producer_index < expval_index:
        expval_index -= 1
    block.operations.insert(expval_index + 1, producer)

    with pytest.raises(ValueError, match="without a local producer"):
        serialize(restored)


def test_serialize_rejects_conflicting_tuple_expval_element_uuid() -> None:
    """A parent-backed element UUID cannot redirect to another qubit."""
    restored = copy.deepcopy(deserialize(serialize(_inverse_tuple_expval_template)))
    block, expval_operation, carrier = _tuple_expval_carrier(restored)
    runtime = carrier.metadata.array_runtime
    assert runtime is not None
    foreign_qubit = Value(type=QubitType(), name="foreign")
    block.operations.insert(
        block.operations.index(expval_operation),
        QInitOperation(operands=[], results=[foreign_qubit]),
    )
    forged_runtime = dataclasses.replace(
        runtime,
        element_uuids=(foreign_qubit.uuid, *runtime.element_uuids[1:]),
        element_logical_ids=(
            foreign_qubit.logical_id,
            *runtime.element_logical_ids[1:],
        ),
    )
    expval_operation.operands[0] = dataclasses.replace(
        carrier,
        metadata=dataclasses.replace(
            carrier.metadata,
            array_runtime=forged_runtime,
        ),
    )

    with pytest.raises(ValueError, match="without a local producer"):
        serialize(restored)


def test_bound_static_width_rechecks_tuple_expval_parent_index() -> None:
    """A symbolic parent index is checked after the encoding width is bound."""
    restored = copy.deepcopy(deserialize(serialize(_inverse_tuple_expval_template)))
    _, expval_operation, carrier = _tuple_expval_carrier(restored)
    runtime = carrier.metadata.array_runtime
    assert runtime is not None
    forged_runtime = dataclasses.replace(
        runtime,
        element_parent_indices=(10**9, *runtime.element_parent_indices[1:]),
    )
    expval_operation.operands[0] = dataclasses.replace(
        carrier,
        metadata=dataclasses.replace(
            carrier.metadata,
            array_runtime=forged_runtime,
        ),
    )
    payload = serialize(restored)
    rebound = deserialize(payload)

    with pytest.raises(ValueError, match="outside array width"):
        rebound.build(encoding=_encoding(1j * I2 + 0.5 * X))


def test_deserialize_rejects_static_passthrough_use_before_producer() -> None:
    """Static alias chains are validated in lexical producer order."""
    restored = copy.deepcopy(
        deserialize(serialize(_nested_scalar_passthrough_chain_template))
    )
    invocations = [
        (index, operation)
        for index, operation in enumerate(restored.block.operations)
        if isinstance(operation, InvokeOperation)
    ]
    assert len(invocations) == 2
    (first_index, first), (second_index, second) = invocations
    assert second.operands[0].uuid == first.results[0].uuid
    restored.block.operations[first_index], restored.block.operations[second_index] = (
        second,
        first,
    )

    with pytest.raises(ValueError, match="aliases a static binding field"):
        serialize(restored)


def test_deserialize_rejects_static_logical_id_on_ordinary_input() -> None:
    """Even an unused ordinary formal cannot impersonate a static field."""
    restored = copy.deepcopy(deserialize(serialize(_normalization_runtime_template)))
    static_field = restored.block.static_bindings[0].fields[0].value
    ordinary_input = restored.block.input_values[0]
    restored.block.input_values[0] = dataclasses.replace(
        ordinary_input,
        logical_id=static_field.logical_id,
    )

    with pytest.raises(ValueError, match="input aliases a static binding field"):
        serialize(restored)


def test_deserialize_rejects_mistyped_static_passthrough_body() -> None:
    """A forged helper body cannot bless a static scalar result alias."""
    restored = copy.deepcopy(
        deserialize(serialize(_nested_scalar_passthrough_template))
    )
    invocation = next(
        operation
        for operation in restored.block.operations
        if isinstance(operation, InvokeOperation)
    )
    assert invocation.definition is not None
    body = invocation.definition.body
    assert body is not None
    forged = dataclasses.replace(body.input_values[0], type=QubitType())
    body.input_values[0] = forged
    body.output_values[0] = forged
    return_operation = next(
        operation
        for operation in body.operations
        if isinstance(operation, ReturnOperation)
    )
    return_operation.operands[0] = forged

    with pytest.raises(ValueError, match="aliases a static binding field"):
        serialize(restored)


def test_nested_qkernel_rejects_direct_static_field_return() -> None:
    """A captured static scalar cannot escape as a nested call result."""

    with pytest.raises(ValueError, match="cannot return a symbolic classical"):

        @qmc.qkernel
        def template(
            encoding: qmc.LCUBlockEncoding,
        ) -> qmc.Vector[qmc.Bit]:
            """Attempt to return a captured field through a helper."""

            @qmc.qkernel
            def signal_width() -> qmc.UInt:
                """Return the lexically captured signal width."""
                return encoding.num_signal_qubits

            signal = qmc.qubit_array(signal_width(), "signal")
            return qmc.measure(signal)

        _ = template.block


def test_prepared_module_does_not_retain_bound_descriptor() -> None:
    """Preparation consumes the descriptor without copying it downstream."""
    encoding = _encoding(1j * I2 + 0.5 * X)
    restored = deserialize(serialize(_direct_template))

    prepared = QamomileCompiler().prepare(
        restored,
        bindings={"encoding": encoding},
    )

    assert "encoding" not in prepared.bindings
    assert not any(
        isinstance(value, qmc.LCUBlockEncoding) for value in prepared.bindings.values()
    )
    _assert_static_binding_resolved(prepared.entrypoint)
    for definition in prepared.definitions.values():
        if definition.body is not None:
            _assert_static_binding_resolved(definition.body)


def test_build_rejects_mutated_replacement_member_abi() -> None:
    """Materialization revalidates a replaced member's mutable frontend ABI."""

    @qmc.qkernel
    def replacement(
        signal: qmc.Vector[qmc.Qubit],
        system: qmc.Vector[qmc.Qubit],
    ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
        """Return both registers through an initially valid replacement ABI."""
        return signal, system

    encoding = dataclasses.replace(
        _encoding(1j * I2 + 0.5 * X),
        unitary=replacement,
    )
    replacement.input_types = {
        "signal": qmc.Vector[qmc.Qubit],
        "system": qmc.Qubit,
    }
    restored = deserialize(serialize(_direct_template))

    with pytest.raises(TypeError, match="frontend ABI disagrees"):
        restored.build(encoding=encoding)


@pytest.mark.parametrize(
    "template",
    [
        _direct_template,
        _inverse_template,
        _inverse_static_argument_template,
        _inverse_static_composite_template,
        _inverse_static_field_template,
        _controlled_template,
        _select_template,
        _both_inverse_orders_template,
        _double_inverse_template,
        _controlled_inverse_template,
        _inverse_select_template,
    ],
    ids=[
        "direct",
        "inverse",
        "inverse_static_argument",
        "inverse_static_composite",
        "inverse_static_field",
        "control",
        "nested_select",
        "both_inverse_orders",
        "double_inverse",
        "controlled_inverse",
        "inverse_select",
    ],
)
def test_static_member_transforms_resolve_after_deserialization(template: Any) -> None:
    """Deferred transforms and inverse compositions resolve the same slot."""
    encoding = _encoding(1j * I2 + 0.5 * X)
    restored = deserialize(serialize(template))

    specialized = restored.build(encoding=encoding)

    _assert_static_binding_resolved(specialized)


def test_static_argument_inverse_materializes_scalar_target_width() -> None:
    """Binding replaces symbolic Vector arity with its scalar qubit width."""
    encoding = qmc.LCUBlockEncoding(_identity_case, 1.0, 2, 1)
    restored = deserialize(serialize(_inverse_static_argument_template))

    specialized = restored.build(encoding=encoding)
    inverse_widths = [
        operation.num_target_qubits
        for block in _reachable_blocks(specialized)
        for operation in block.operations
        if isinstance(operation, InverseBlockOperation)
    ]

    assert inverse_widths
    assert all(width == 3 for width in inverse_widths)


def test_inverse_qkernel_executes_with_a_concrete_static_binding_argument(
    sdk_transpiler: Any,
) -> None:
    """Every SDK executes a nested inverse specialized by a concrete object."""
    encoding = _encoding(X)

    @qmc.qkernel
    def template() -> tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]:
        """Invert a qkernel using a descriptor captured at trace time."""
        signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
        system = qmc.qubit_array(encoding.num_system_qubits, "system")
        signal, system = qmc.inverse(_apply_static_encoding)(
            signal,
            system,
            encoding,
        )
        return qmc.measure(signal), qmc.measure(system)

    assert template.block.static_bindings == ()
    _assert_static_binding_resolved(template.block)

    executable = sdk_transpiler.transpiler.transpile(template)
    result = executable.sample(
        _executor(sdk_transpiler),
        shots=128,
    ).result()
    assert result.results == [(((0,), (1,)), 128)]


def test_inverse_static_composite_executes_after_binding(
    sdk_transpiler: Any,
) -> None:
    """Every SDK executes a statically bound named-composite inverse."""
    restored = deserialize(serialize(_inverse_static_composite_template))
    executable = sdk_transpiler.transpiler.transpile(
        restored,
        bindings={"encoding": _encoding(X)},
    )

    result = executable.sample(
        _executor(sdk_transpiler),
        shots=128,
    ).result()
    assert result.results == [(((0,), (1,)), 128)]


def test_inverse_qkernel_mixes_symbolic_and_concrete_static_bindings() -> None:
    """One inverse call may combine a payload slot with a captured object."""
    second = _encoding(X)

    @qmc.qkernel
    def template(first: qmc.LCUBlockEncoding) -> qmc.Bit:
        """Invert a helper whose descriptors resolve at different times."""
        qubit = qmc.qubit("qubit")
        qubit = qmc.inverse(_apply_two_static_normalizations)(
            qubit,
            first,
            second,
        )
        return qmc.measure(qubit)

    restored = deserialize(serialize(template))
    specialized = restored.build(first=_encoding(I2))

    _assert_static_binding_resolved(specialized)


def test_inverted_block_preserves_static_bindings_through_serialization() -> None:
    """A standalone inverse retains the manifest needed for later binding."""
    restored = deserialize(serialize(_apply_static_normalization))
    inverted = _BlockInverter().invert_block(restored.block)

    assert inverted.static_bindings == restored.block.static_bindings

    inverse_kernel = dataclasses.replace(restored, _block=inverted)
    round_tripped = deserialize(serialize(inverse_kernel))
    assert round_tripped.block.static_bindings == restored.block.static_bindings

    specialized = round_tripped.build(encoding=_encoding(I2))
    _assert_static_binding_resolved(specialized)


def test_inverse_qkernel_rejects_a_renamed_symbolic_static_slot() -> None:
    """Deferred static arguments require explicit same-name slot identity."""

    @qmc.qkernel
    def template(descriptor: qmc.LCUBlockEncoding) -> qmc.Bit:
        """Pass a differently named root slot to the nested qkernel."""
        qubit = qmc.qubit("qubit")
        qubit = qmc.inverse(_apply_static_normalization)(qubit, descriptor)
        return qmc.measure(qubit)

    with pytest.raises(
        TypeError,
        match="must come from the same-named slot",
    ):
        _ = template.block


@pytest.mark.parametrize(
    "template",
    [
        _wrong_signal_direct,
        _wrong_signal_inverse,
        _wrong_signal_control,
        _wrong_signal_select,
    ],
    ids=["direct", "inverse", "control", "nested_select"],
)
def test_static_member_paths_share_signal_width_diagnostics(template: Any) -> None:
    """Every transform reports the bound descriptor's concrete signal width."""
    encoding = _encoding(1j * I2 + 0.5 * X + (0.2 - 0.3j) * Z)
    restored = deserialize(serialize(template))

    with pytest.raises(ValueError) as error:
        restored.build(encoding=encoding)

    assert str(error.value) == (
        "Pauli LCU block encoding requires 2 signal qubits, got 1."
    )


def test_static_member_reports_concrete_system_width_after_deserialization() -> None:
    """System-width validation also runs only after the descriptor is bound."""
    encoding = _encoding(1j * I2 + 0.5 * X + (0.2 - 0.3j) * Z)
    restored = deserialize(serialize(_wrong_system_direct))

    with pytest.raises(ValueError) as error:
        restored.build(encoding=encoding)

    assert str(error.value) == (
        "Pauli LCU block encoding requires 1 system qubit, got 2."
    )


def test_recursive_static_member_reports_concrete_signal_width() -> None:
    """A recursive descriptor retains its validator after deserialization."""
    restored = deserialize(serialize(_wrong_signal_direct))

    with pytest.raises(ValueError) as error:
        restored.build(encoding=_recursive_ising_encoding())

    assert str(error.value) == (
        "LCU block encoding requires 2 signal qubits, got 1."
    )


def test_serialized_static_binding_executes_phase_sample_on_every_sdk(
    sdk_transpiler: Any,
) -> None:
    """Every SDK detects a locally bound negative-identity global phase."""
    encoding = _encoding(-I2)
    sample_template = deserialize(serialize(_phase_sample_template))

    sample_executable = sdk_transpiler.transpiler.transpile(
        sample_template,
        bindings={"encoding": encoding},
    )
    sample_result = sample_executable.sample(
        _executor(sdk_transpiler),
        shots=128,
    ).result()
    assert sample_result.results == [(1, 128)]


def test_same_serialized_payload_rebinds_sampling_on_every_sdk(
    sdk_transpiler: Any,
) -> None:
    """One payload samples distinct I and X descriptors on every SDK."""
    sample_template = deserialize(serialize(_semantic_sample_template))
    encodings = (
        (_encoding(I2), 0),
        (_generic_encoding(X), 1),
        (_alternative_encoding(X), 1),
    )
    assert {
        (encoding.num_signal_qubits, encoding.num_system_qubits)
        for encoding, _ in encodings
    } == {(1, 1)}

    executor = _executor(sdk_transpiler)
    for encoding, expected_bit in encodings:
        sample_executable = sdk_transpiler.transpiler.transpile(
            sample_template,
            bindings={"encoding": encoding},
        )
        sample_result = sample_executable.sample(executor, shots=128).result()
        assert sample_result.results == [(expected_bit, 128)]


def test_same_serialized_payload_rebinds_expval_on_every_sdk(
    sdk_transpiler: Any,
) -> None:
    """One payload estimates distinct I and X descriptors on every SDK."""
    import qamomile.observable as qm_o

    expval_template = deserialize(serialize(_semantic_expval_template))
    encodings = (
        (_encoding(I2), 1.0),
        (_generic_encoding(X), -1.0),
        (_alternative_encoding(X), -1.0),
    )
    executor = _executor(sdk_transpiler)
    for encoding, expected_expval in encodings:
        expval_executable = sdk_transpiler.transpiler.transpile(
            expval_template,
            bindings={"encoding": encoding, "observable": qm_o.Z(0)},
        )
        expval = expval_executable.run(executor).result()
        tolerance = 1e-6 if sdk_transpiler.backend_name == "cudaq" else 1e-8
        assert float(expval) == pytest.approx(expected_expval, abs=tolerance)


def test_serialized_tuple_expval_executes_on_every_sdk(
    sdk_transpiler: Any,
) -> None:
    """Tuple carriers survive inverse materialization on every SDK."""
    import qamomile.observable as qm_o

    restored = deserialize(serialize(_inverse_tuple_expval_template))
    executable = sdk_transpiler.transpiler.transpile(
        restored,
        bindings={
            "encoding": _encoding(X),
            "observable": qm_o.Z(0) * qm_o.Z(1),
        },
    )

    observed = float(executable.run(_executor(sdk_transpiler)).result())
    tolerance = 1e-6 if sdk_transpiler.backend_name == "cudaq" else 1e-8
    assert observed == pytest.approx(-1.0, abs=tolerance)


def test_static_normalization_and_runtime_parameter_execute_on_every_sdk(
    sdk_transpiler: Any,
) -> None:
    """A static Float field and an ordinary runtime Float share one ABI."""
    encoding = _encoding(np.pi * I2)
    restored = deserialize(serialize(_normalization_runtime_template))

    specialized = restored.build(parameters=["theta"], encoding=encoding)
    assert list(specialized.parameters) == ["theta"]
    assert [slot.name for slot in specialized.param_slots] == ["theta"]
    _assert_static_binding_resolved(specialized)

    executable = sdk_transpiler.transpiler.transpile(
        restored,
        bindings={"encoding": encoding},
        parameters=["theta"],
    )
    result = executable.sample(
        _executor(sdk_transpiler),
        shots=128,
        bindings={"theta": 0.0},
    ).result()
    assert result.results == [(1, 128)]


@pytest.mark.parametrize(
    (
        "sample_template",
        "expval_template",
        "expected_zero_probability",
        "expected_expval",
    ),
    [
        (
            _complex_inverse_sample_template,
            _complex_inverse_expval_template,
            1.0 / 6.0,
            -2.0 / 3.0,
        ),
        (
            _complex_static_argument_inverse_sample_template,
            _complex_static_argument_inverse_expval_template,
            1.0 / 6.0,
            -2.0 / 3.0,
        ),
        (
            _complex_control_sample_template,
            _complex_control_expval_template,
            5.0 / 6.0,
            2.0 / 3.0,
        ),
        (
            _complex_select_sample_template,
            _complex_select_expval_template,
            5.0 / 6.0,
            2.0 / 3.0,
        ),
    ],
    ids=[
        "inverse",
        "inverse_static_argument",
        "outer_control",
        "nested_select",
    ],
)
def test_serialized_complex_transforms_execute_on_every_sdk(
    sdk_transpiler: Any,
    sample_template: Any,
    expval_template: Any,
    expected_zero_probability: float,
    expected_expval: float,
) -> None:
    """Complex inverse and interference paths match analytic block values.

    For ``A = iI + X/2`` and ``alpha = 3/2``, a controlled forward or SELECT
    branch in equal superposition has overlap ``2i/3``. Inverse conjugates that
    overlap. An S-dagger followed by H maps the phases to zero probabilities
    ``5/6`` and ``1/6``; direct Y expectations are ``2/3`` and ``-2/3``.
    """
    import qamomile.observable as qm_o

    encoding = _encoding(1j * I2 + 0.5 * X)
    restored_sample = deserialize(serialize(sample_template))
    restored_expval = deserialize(serialize(expval_template))
    shots = 4096
    executor = _executor(sdk_transpiler)

    sample_executable = sdk_transpiler.transpiler.transpile(
        restored_sample,
        bindings={"encoding": encoding},
    )
    sample_result = sample_executable.sample(executor, shots=shots).result()
    observed_zero_probability = (
        sum(count for outcome, count in sample_result.results if int(outcome) == 0)
        / shots
    )
    sampling_tolerance = (
        6.0
        * math.sqrt(
            expected_zero_probability * (1.0 - expected_zero_probability) / shots
        )
        + 0.02
    )
    assert observed_zero_probability == pytest.approx(
        expected_zero_probability,
        abs=sampling_tolerance,
    )

    expval_executable = sdk_transpiler.transpiler.transpile(
        restored_expval,
        bindings={"encoding": encoding, "observable": qm_o.Y(0)},
    )
    observed_expval = float(expval_executable.run(executor).result())
    expval_tolerance = 1e-6 if sdk_transpiler.backend_name == "cudaq" else 1e-8
    assert observed_expval == pytest.approx(expected_expval, abs=expval_tolerance)


@pytest.mark.parametrize(
    (
        "sample_template",
        "expval_template",
        "expected_zero_probability",
        "expected_expval",
    ),
    [
        (
            _complex_inverse_sample_template,
            _complex_inverse_expval_template,
            2.0 / 7.0,
            -3.0 / 7.0,
        ),
        (
            _complex_static_argument_inverse_sample_template,
            _complex_static_argument_inverse_expval_template,
            2.0 / 7.0,
            -3.0 / 7.0,
        ),
        (
            _complex_control_sample_template,
            _complex_control_expval_template,
            5.0 / 7.0,
            3.0 / 7.0,
        ),
        (
            _complex_select_sample_template,
            _complex_select_expval_template,
            5.0 / 7.0,
            3.0 / 7.0,
        ),
    ],
    ids=[
        "inverse",
        "inverse_static_argument",
        "outer_control",
        "nested_select",
    ],
)
def test_serialized_recursive_lcu_transforms_execute_on_every_sdk(
    sdk_transpiler: Any,
    sample_template: Any,
    expval_template: Any,
    expected_zero_probability: float,
    expected_expval: float,
) -> None:
    r"""Every SDK executes a recursively composed static descriptor.

    The Ising-Z child encodes ``iI + Z/2`` with normalization ``3/2``.
    Combining that child with ``-iI/4`` produces ``Z/2 + 3iI/4`` and parent
    normalization ``7/4``. Its all-zero overlap on ``|0>`` is therefore
    ``2/7 + 3i/7``. The expected interference values exercise child
    normalization, recursive SELECT, explicit coefficient phase, static
    binding, inverse, outer control, and outer SELECT together.
    """
    import qamomile.observable as qm_o

    encoding = _recursive_ising_encoding()
    restored_sample = deserialize(serialize(sample_template))
    restored_expval = deserialize(serialize(expval_template))
    shots = 2048
    executor = _executor(sdk_transpiler)

    sample_executable = sdk_transpiler.transpiler.transpile(
        restored_sample,
        bindings={"encoding": encoding},
    )
    sample_result = sample_executable.sample(executor, shots=shots).result()
    observed_zero_probability = (
        sum(count for outcome, count in sample_result.results if int(outcome) == 0)
        / shots
    )
    sampling_tolerance = (
        6.0
        * math.sqrt(
            expected_zero_probability * (1.0 - expected_zero_probability) / shots
        )
        + 0.02
    )
    assert observed_zero_probability == pytest.approx(
        expected_zero_probability,
        abs=sampling_tolerance,
    )

    expval_executable = sdk_transpiler.transpiler.transpile(
        restored_expval,
        bindings={"encoding": encoding, "observable": qm_o.Y(0)},
    )
    observed_expval = float(expval_executable.run(executor).result())
    expval_tolerance = 1e-6 if sdk_transpiler.backend_name == "cudaq" else 1e-8
    assert observed_expval == pytest.approx(expected_expval, abs=expval_tolerance)


def test_static_binding_deserializes_and_builds_in_fresh_interpreter() -> None:
    """A fresh process resolves the registered type without sender state."""
    payload = base64.b64encode(serialize(_direct_template)).decode("ascii")
    script = """
import base64
import sys

import numpy as np

import qamomile.circuit as qmc
from qamomile.circuit.serialization import deserialize
from qamomile.linalg import PauliLCU

restored = deserialize(base64.b64decode(sys.argv[1]))
matrix = np.array([[0, 1], [1, 0]], dtype=np.complex128)
encoding = qmc.pauli_lcu_block_encoding(PauliLCU.from_matrix(matrix, atol=1e-12))
block = restored.build(encoding=encoding)
assert block.kind.name == "TRACED"
assert block.static_bindings == ()
"""

    completed = subprocess.run(
        [sys.executable, "-c", script, payload],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr


def test_ising_static_binding_deserializes_in_fresh_interpreter() -> None:
    """A fresh process resolves the Ising-Z producer-specific type key."""
    payload = base64.b64encode(serialize(_ising_specific_template)).decode("ascii")
    script = """
import base64
import sys

import qamomile.circuit as qmc
from qamomile.circuit.serialization import deserialize

restored = deserialize(base64.b64decode(sys.argv[1]))
encoding = qmc.ising_z_block_encoding({(): 1.0j, (0,): 0.5}, 1)
block = restored.build(encoding=encoding)
assert block.kind.name == "TRACED"
assert block.static_bindings == ()
"""

    completed = subprocess.run(
        [sys.executable, "-c", script, payload],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
