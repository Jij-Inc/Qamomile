"""Tests for first-class qkernel effect aggregation and diagnostics."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.callable import (
    CallableDef,
    CallableImplementation,
    CallableRef,
    CallTransform,
    InvokeOperation,
)
from qamomile.circuit.ir.operation.gate import MeasureOperation
from qamomile.circuit.ir.operation.operation import CInitOperation, Operation
from qamomile.circuit.ir.operation.select import SelectOperation
from qamomile.circuit.ir.types import BitType, QubitType
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


def test_invoke_measurement_provenance_tracks_selected_implementation() -> None:
    """Exact provenance follows selection while the property remains a superset."""
    direct_target = Value(type=QubitType(), name="direct_target")
    direct_measured = Value(type=BitType(), name="direct_measured")
    direct_plain = Value(type=BitType(), name="direct_plain")
    direct_body = Block(
        input_values=[direct_target],
        output_values=[direct_measured, direct_plain],
        operations=[
            MeasureOperation(
                operands=[direct_target],
                results=[direct_measured],
            ),
            CInitOperation(results=[direct_plain]),
        ],
    )

    native_control = Value(type=QubitType(), name="native_control")
    native_target = Value(type=QubitType(), name="native_target")
    native_plain = Value(type=BitType(), name="native_plain")
    native_measured = Value(type=BitType(), name="native_measured")
    native_body = Block(
        input_values=[native_control, native_target],
        output_values=[
            native_control.next_version(),
            native_plain,
            native_measured,
        ],
        operations=[
            CInitOperation(results=[native_plain]),
            MeasureOperation(
                operands=[native_target],
                results=[native_measured],
            ),
        ],
    )

    ref = CallableRef(namespace="test", name="selected_measurement_provenance")
    definition = CallableDef(
        ref=ref,
        body=direct_body,
        implementations=[
            CallableImplementation(
                transform=CallTransform.CONTROLLED,
                backend="qiskit",
                strategy="native",
                body=native_body,
            )
        ],
    )
    control = Value(type=QubitType(), name="control")
    target = Value(type=QubitType(), name="target")
    operation = InvokeOperation(
        operands=[control, target],
        results=[
            control.next_version(),
            Value(type=BitType(), name="first"),
            Value(type=BitType(), name="second"),
        ],
        target=ref,
        transform=CallTransform.CONTROLLED,
        attrs={"num_control_qubits": 1, "num_target_qubits": 1},
        definition=definition,
    )

    exact_indices = {
        ("qiskit", "native"): frozenset({2}),
        ("quri_parts", "native"): frozenset({1}),
        ("qiskit", "portable"): frozenset({1}),
    }
    for (backend, strategy), expected in exact_indices.items():
        selected = operation.measurement_result_indices_for(
            backend=backend,
            strategy=strategy,
        )
        assert selected == expected
        assert selected <= operation.measurement_result_indices

    assert operation.measurement_result_indices == frozenset({1, 2})


@pytest.mark.parametrize(
    ("backend", "strategy"),
    [
        ("qiskit", None),
        (None, "native"),
    ],
)
def test_context_specific_implementation_keeps_fallback_effects(
    backend: str | None,
    strategy: str | None,
) -> None:
    """A specialized unitary body does not hide a measured fallback body."""
    direct_target = Value(type=QubitType(), name="direct_target")
    direct_result = Value(type=BitType(), name="direct_result")
    direct_body = Block(
        input_values=[direct_target],
        output_values=[direct_result],
        operations=[
            MeasureOperation(operands=[direct_target], results=[direct_result])
        ],
    )

    specialized_target = Value(type=QubitType(), name="specialized_target")
    specialized_result = Value(type=BitType(), name="specialized_result")
    specialized_body = Block(
        input_values=[specialized_target],
        output_values=[specialized_result],
        operations=[CInitOperation(results=[specialized_result])],
    )

    ref = CallableRef(namespace="test", name="context_specific_effect")
    definition = CallableDef(
        ref=ref,
        body=direct_body,
        implementations=[
            CallableImplementation(
                transform=CallTransform.DIRECT,
                backend=backend,
                strategy=strategy,
                body=specialized_body,
            )
        ],
    )
    result = Value(type=BitType(), name="result")
    operation = InvokeOperation(
        operands=[Value(type=QubitType(), name="target")],
        results=[result],
        target=ref,
        definition=definition,
    )

    assert operation.effects is qmc.KernelEffect.MEASUREMENT
    assert operation.measurement_result_indices == frozenset({0})


def test_fully_generic_implementation_shadows_fallback_effects() -> None:
    """A fully generic implementation makes the direct fallback unreachable."""
    direct_target = Value(type=QubitType(), name="direct_target")
    direct_result = Value(type=BitType(), name="direct_result")
    direct_body = Block(
        input_values=[direct_target],
        output_values=[direct_result],
        operations=[
            MeasureOperation(operands=[direct_target], results=[direct_result])
        ],
    )

    generic_target = Value(type=QubitType(), name="generic_target")
    generic_result = Value(type=BitType(), name="generic_result")
    generic_body = Block(
        input_values=[generic_target],
        output_values=[generic_result],
        operations=[CInitOperation(results=[generic_result])],
    )
    ref = CallableRef(namespace="test", name="generic_effect")
    operation = InvokeOperation(
        operands=[Value(type=QubitType(), name="target")],
        results=[Value(type=BitType(), name="result")],
        target=ref,
        definition=CallableDef(
            ref=ref,
            body=direct_body,
            implementations=[
                CallableImplementation(
                    transform=CallTransform.DIRECT,
                    body=generic_body,
                )
            ],
        ),
    )

    assert operation.effects is qmc.KernelEffect.NONE
    assert operation.measurement_result_indices == frozenset()


def test_block_effects_are_lazy_cached_and_replacement_invalidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Effect analysis runs on demand and fresh Blocks start invalidated."""
    import qamomile.circuit.ir.effect as effect_module

    original = effect_module.summarize_block_effects
    calls = 0

    def counting_summary(
        operations: Sequence[Operation],
        output_values: Sequence[object],
    ) -> tuple[qmc.KernelEffect, frozenset[int]]:
        """Count and delegate one block-effect analysis.

        Args:
            operations (Sequence[Operation]): Semantic operations to summarize.
            output_values (Sequence[object]): Public outputs to inspect.

        Returns:
            tuple[qmc.KernelEffect, frozenset[int]]: Delegated effect summary.
        """
        nonlocal calls
        calls += 1
        return original(operations, output_values)

    monkeypatch.setattr(effect_module, "summarize_block_effects", counting_summary)
    block = Block(name="lazy")
    assert calls == 0

    assert block.effects == qmc.KernelEffect.NONE
    assert block.measurement_result_indices == frozenset()
    assert calls == 1

    replacement = dataclasses.replace(block, name="replacement")
    assert calls == 1
    assert replacement.effects == qmc.KernelEffect.NONE
    assert calls == 2


@pytest.mark.parametrize("query_b_first", [False, True])
def test_mutually_recursive_effects_reach_an_order_independent_fixed_point(
    query_b_first: bool,
) -> None:
    """Mutual recursion propagates measurement effects in either query order."""
    block_a = Block(name="A")
    block_b = Block(name="B")
    ref_a = CallableRef(namespace="test", name="A")
    ref_b = CallableRef(namespace="test", name="B")
    definition_a = CallableDef(ref=ref_a, body=block_a)
    definition_b = CallableDef(ref=ref_b, body=block_b)

    output_a = Value(type=BitType(), name="output_a")
    output_b = Value(type=BitType(), name="output_b")
    nested_b = Value(type=BitType(), name="nested_b")
    block_a.output_values = [output_a]
    block_b.output_values = [output_b]
    block_a.operations.extend(
        [
            InvokeOperation(
                results=[nested_b],
                target=ref_b,
                definition=definition_b,
            ),
            MeasureOperation(
                operands=[Value(type=QubitType(), name="measured")],
                results=[output_a],
            ),
        ]
    )
    block_b.operations.append(
        InvokeOperation(
            results=[output_b],
            target=ref_a,
            definition=definition_a,
        )
    )

    first, second = (block_b, block_a) if query_b_first else (block_a, block_b)
    assert first.effects is qmc.KernelEffect.MEASUREMENT
    assert second.effects is qmc.KernelEffect.MEASUREMENT
    assert first.measurement_result_indices == frozenset({0})
    assert second.measurement_result_indices == frozenset({0})


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


def test_select_frontend_rejects_effectful_cases_structurally() -> None:
    """The SELECT frontend rejects non-unitary case operations directly."""
    with pytest.raises(
        ValueError,
        match=r"case 0.*non-unitary ProjectOperation",
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
