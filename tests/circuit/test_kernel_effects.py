"""Tests for first-class qkernel effect aggregation and diagnostics."""

from __future__ import annotations

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.callable import InvokeOperation
from qamomile.circuit.ir.operation.select import SelectOperation
from qamomile.circuit.ir.types import QubitType
from qamomile.circuit.ir.value import Value
from qamomile.circuit.serialization import deserialize, serialize
from qamomile.circuit.stdlib.arithmetic import modmul_const
from qamomile.circuit.transpiler.errors import ValidationError


@qmc.qkernel
def _projected_layer(qubit: qmc.Qubit) -> qmc.Qubit:
    """Project a qubit while discarding the classical result."""
    qubit, _ = qmc.project_z(qubit)
    return qubit


@qmc.qkernel
def _reset_layer(qubit: qmc.Qubit) -> qmc.Qubit:
    """Reset and return one qubit."""
    return qmc.reset(qubit)


@qmc.qkernel
def _invoke_projected_layer(qubit: qmc.Qubit) -> qmc.Qubit:
    """Invoke the effectful projection layer."""
    return _projected_layer(qubit)


@qmc.qkernel
def _measure_bit_layer(qubit: qmc.Qubit) -> qmc.Bit:
    """Measure and return one classical bit."""
    return qmc.measure(qubit)


@qmc.qkernel
def _measurement_feed_forward(
    measured: qmc.Qubit,
    target: qmc.Qubit,
) -> qmc.Qubit:
    """Drive a quantum branch from an invoked measurement result."""
    bit = _measure_bit_layer(measured)
    if bit:
        target = qmc.x(target)
    return target


@qmc.qkernel
def _controlled_effectful_entrypoint() -> tuple[qmc.Bit, qmc.Bit]:
    """Attempt generic control of a measurement-effectful kernel."""
    control = qmc.qubit("control")
    target = qmc.qubit("target")
    control, target = qmc.control(_projected_layer)(control, target)
    return qmc.measure(control), qmc.measure(target)


@qmc.qkernel
def _inverse_effectful_entrypoint() -> qmc.Bit:
    """Attempt generic inversion of a reset-effectful kernel."""
    target = qmc.qubit("target")
    target = qmc.inverse(_reset_layer)(target)
    return qmc.measure(target)


@qmc.qkernel
def _effectful_select_entrypoint() -> tuple[qmc.Bit, qmc.Bit]:
    """Attempt SELECT construction with effectful case bodies."""
    index = qmc.qubit("index")
    target = qmc.qubit("target")
    index, target = qmc.select([_projected_layer, _projected_layer])(
        index,
        target,
    )
    return qmc.measure(index), qmc.measure(target)


@qmc.qkernel
def _mixed_measurement_expval(
    observable: qmc.Observable,
) -> tuple[qmc.Bit, qmc.Float]:
    """Mix sample-only measurement with expectation estimation."""
    measured = qmc.qubit("measured")
    estimated = qmc.qubit("estimated")
    return qmc.measure(measured), qmc.expval(estimated, observable)


@qmc.qkernel
def _explicit_controlled_modmul() -> tuple[qmc.Bit, qmc.Vector[qmc.Bit]]:
    """Use the stdlib kernel's explicit control argument."""
    control = qmc.qubit("control")
    register = qmc.qubit_array(2, name="register")
    control, register = modmul_const(
        register,
        multiplier=2,
        modulus=3,
        control=control,
    )
    return qmc.measure(control), qmc.measure(register)


def test_qkernel_block_and_invoke_expose_cached_effects() -> None:
    """QKernel, Block, and InvokeOperation expose one propagated effect set."""
    assert _projected_layer.effects == qmc.KernelEffect.MEASUREMENT
    assert _projected_layer.block.effects == qmc.KernelEffect.MEASUREMENT

    invocation = next(
        operation
        for operation in _invoke_projected_layer.block.operations
        if isinstance(operation, InvokeOperation)
    )
    assert invocation.effects == qmc.KernelEffect.MEASUREMENT
    assert _invoke_projected_layer.effects == qmc.KernelEffect.MEASUREMENT

    measured_invocation = next(
        operation
        for operation in _measurement_feed_forward.block.operations
        if isinstance(operation, InvokeOperation)
    )
    assert measured_invocation.measurement_result_indices == frozenset({0})


def test_reset_and_feed_forward_effects_are_distinct_and_composable() -> None:
    """Reset and measurement-backed control flow receive distinct flags."""
    assert _reset_layer.effects == qmc.KernelEffect.RESET
    assert _measurement_feed_forward.effects == (
        qmc.KernelEffect.MEASUREMENT | qmc.KernelEffect.FEED_FORWARD
    )


def test_serialized_qkernel_rebuilds_effect_metadata() -> None:
    """Deserialization restores direct effects and output provenance."""
    restored = deserialize(serialize(_measure_bit_layer))

    assert restored.effects == qmc.KernelEffect.MEASUREMENT
    assert restored.block.measurement_result_indices == frozenset({0})


def test_serialized_invocation_relinks_effect_metadata() -> None:
    """Deserialization reaches an effect fixed point across callable links."""
    restored = deserialize(serialize(_invoke_projected_layer))

    invocation = next(
        operation
        for operation in restored.block.operations
        if isinstance(operation, InvokeOperation)
    )
    assert invocation.effects == qmc.KernelEffect.MEASUREMENT
    assert restored.effects == qmc.KernelEffect.MEASUREMENT


def test_select_operation_propagates_case_effects() -> None:
    """A hand-built SELECT inherits cached effects from every case block."""
    index = Value(type=QubitType(), name="index")
    target = Value(type=QubitType(), name="target")
    select = SelectOperation(
        operands=[index, target],
        results=[index.next_version(), target.next_version()],
        num_index_qubits=1,
        num_index_args=1,
        case_blocks=[_projected_layer.block, _reset_layer.block],
    )
    block = Block(name="effectful_select", operations=[select])

    assert block.effects == (qmc.KernelEffect.MEASUREMENT | qmc.KernelEffect.RESET)


def test_select_frontend_rejects_effectful_cases_from_cached_metadata() -> None:
    """The SELECT frontend names the offending case effects immediately."""
    with pytest.raises(
        ValueError,
        match=r"case 0.*kernel effects \[MEASUREMENT\]",
    ):
        _ = _effectful_select_entrypoint.block


@pytest.mark.parametrize(
    ("kernel", "effect", "alternative"),
    [
        (_controlled_effectful_entrypoint, "MEASUREMENT", "explicit control"),
        (_inverse_effectful_entrypoint, "RESET", "explicit inverse"),
    ],
)
def test_generic_transforms_reject_effects_during_frontend_build(
    kernel: object,
    effect: str,
    alternative: str,
) -> None:
    """Generic control and inverse fail before analysis or backend emission."""
    with pytest.raises(
        ValueError,
        match=rf"non-unitary kernel effects \[{effect}\].*{alternative}",
    ):
        _ = kernel.block


def test_backend_transpilers_report_the_same_effect_diagnostic(
    sdk_transpiler: object,
) -> None:
    """Qiskit, QURI Parts, and CUDA-Q reject effects before backend emission."""
    with pytest.raises(
        ValueError,
        match=r"qmc\.control\(\).*_projected_layer.*MEASUREMENT",
    ):
        sdk_transpiler.transpiler.transpile(_controlled_effectful_entrypoint)


def test_measurement_plus_expval_is_rejected_as_sample_only(
    sdk_transpiler: object,
) -> None:
    """Every backend reports the shared early sample-only constraint."""
    import qamomile.observable as qmo

    with pytest.raises(
        ValidationError,
        match=r"MEASUREMENT.*sample-only.*qmc\.expval",
    ):
        sdk_transpiler.transpiler.transpile(
            _mixed_measurement_expval,
            bindings={"observable": qmo.Z(0)},
        )


def test_explicit_modmul_control_path_still_executes(sdk_transpiler: object) -> None:
    """Explicit stdlib control executes on reset-capable SDK backends."""
    _ = _explicit_controlled_modmul.block
    if sdk_transpiler.backend_name == "quri_parts":
        pytest.skip("QURI Parts cannot represent modmul's mid-circuit reset")

    transpiler = sdk_transpiler.transpiler
    result = (
        transpiler.transpile(_explicit_controlled_modmul)
        .sample(transpiler.executor(), shots=1)
        .result()
    )

    assert result.results == [((0, (0, 0)), 1)]
