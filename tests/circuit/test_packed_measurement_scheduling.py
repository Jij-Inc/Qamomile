"""Keep independent packed-register measurements in one quantum execution."""

from __future__ import annotations

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.ir.operation import DecodeQFixedOperation, DecodeQIntOperation
from qamomile.circuit.serialization import deserialize, serialize
from qamomile.circuit.transpiler.errors import DependencyError
from qamomile.circuit.transpiler.passes.separate import SegmentationPass
from qamomile.circuit.transpiler.segments import (
    ClassicalStep,
    MultipleQuantumSegmentsError,
    QuantumStep,
)


@qmc.qkernel
def _two_qints() -> tuple[qmc.UInt, qmc.UInt]:
    """Measure disjoint registers with distinguishable little-endian values."""
    a = qmc.qubit_array(2, "a")
    b = qmc.qubit_array(3, "b")
    a[0] = qmc.x(a[0])
    b[2] = qmc.x(b[2])
    return qmc.measure(qmc.cast(a, qmc.QInt)), qmc.measure(qmc.cast(b, qmc.QInt))


@qmc.qkernel
def _qint_then_plain() -> tuple[qmc.UInt, qmc.Vector[qmc.Bit]]:
    """Measure a packed register before an independent plain vector."""
    a = qmc.qubit_array(2, "a")
    b = qmc.qubit_array(2, "b")
    a[0] = qmc.x(a[0])
    b[1] = qmc.x(b[1])
    return qmc.measure(qmc.cast(a, qmc.QInt)), qmc.measure(b)


@qmc.qkernel
def _plain_then_qint() -> tuple[qmc.Vector[qmc.Bit], qmc.UInt]:
    """Preserve the same values when the plain vector is measured first."""
    a = qmc.qubit_array(2, "a")
    b = qmc.qubit_array(2, "b")
    a[0] = qmc.x(a[0])
    b[1] = qmc.x(b[1])
    return qmc.measure(b), qmc.measure(qmc.cast(a, qmc.QInt))


@qmc.qkernel
def _qfixed_then_plain() -> tuple[qmc.Float, qmc.Vector[qmc.Bit]]:
    """Share the scheduling behavior with fractional packed registers."""
    a = qmc.qubit_array(2, "a")
    b = qmc.qubit_array(2, "b")
    a[0] = qmc.x(a[0])
    b[1] = qmc.x(b[1])
    return qmc.measure(qmc.cast(a, qmc.QFixed, int_bits=0)), qmc.measure(b)


@qmc.qkernel
def _plain_then_qfixed() -> tuple[qmc.Vector[qmc.Bit], qmc.Float]:
    """Preserve fixed-point values when the plain vector is measured first."""
    a = qmc.qubit_array(2, "a")
    b = qmc.qubit_array(2, "b")
    a[0] = qmc.x(a[0])
    b[1] = qmc.x(b[1])
    return qmc.measure(b), qmc.measure(qmc.cast(a, qmc.QFixed, int_bits=0))


@qmc.qkernel
def _qint_then_qfixed() -> tuple[qmc.UInt, qmc.Float]:
    """Run both packed-register decoders after their independent measurements."""
    a = qmc.qubit_array(2, "a")
    b = qmc.qubit_array(3, "b")
    a[0] = qmc.x(a[0])
    b[2] = qmc.x(b[2])
    return (
        qmc.measure(qmc.cast(a, qmc.QInt)),
        qmc.measure(qmc.cast(b, qmc.QFixed, int_bits=0)),
    )


@qmc.qkernel
def _zero_width_then_qint() -> tuple[qmc.UInt, qmc.UInt]:
    """Decode an empty register without changing later measurement results."""
    empty = qmc.qubit_array(0, "empty")
    b = qmc.qubit_array(2, "b")
    b[1] = qmc.x(b[1])
    return qmc.measure(qmc.cast(empty, qmc.QInt)), qmc.measure(qmc.cast(b, qmc.QInt))


@qmc.qkernel
def _interleaved_host_arithmetic() -> tuple[qmc.UInt, qmc.UInt]:
    """Evaluate arithmetic after its deferred decode and preserve return order."""
    a = qmc.qubit_array(2, "a")
    b = qmc.qubit_array(3, "b")
    a[0] = qmc.x(a[0])
    b[2] = qmc.x(b[2])
    first = qmc.measure(qmc.cast(a, qmc.QInt)) + 2
    second = qmc.measure(qmc.cast(b, qmc.QInt)) + first
    return second, first


@qmc.qkernel
def _qint_then_scalar() -> tuple[qmc.UInt, qmc.Bit]:
    """Keep independent scalar measurements after a packed decode in-circuit."""
    a = qmc.qubit_array(2, "a")
    b = qmc.qubit("b")
    a[0] = qmc.x(a[0])
    b = qmc.x(b)
    return qmc.measure(qmc.cast(a, qmc.QInt)), qmc.measure(b)


_INDEPENDENT_CASES = [
    pytest.param(_two_qints, (1, 4), 2, id="two-qints"),
    pytest.param(_qint_then_plain, (1, (0, 1)), 1, id="qint-plain"),
    pytest.param(_plain_then_qint, ((0, 1), 1), 1, id="plain-qint"),
    pytest.param(_qfixed_then_plain, (0.25, (0, 1)), 1, id="qfixed-plain"),
    pytest.param(_plain_then_qfixed, ((0, 1), 0.25), 1, id="plain-qfixed"),
    pytest.param(_qint_then_qfixed, (1, 0.5), 2, id="qint-qfixed"),
    pytest.param(_zero_width_then_qint, (0, 2), 2, id="zero-width"),
    pytest.param(_interleaved_host_arithmetic, (7, 3), 2, id="host-arithmetic"),
    pytest.param(_qint_then_scalar, (1, 1), 1, id="qint-scalar"),
]


@pytest.mark.parametrize("kernel, expected, decode_count", _INDEPENDENT_CASES)
@pytest.mark.parametrize("roundtrip", [False, True], ids=["direct", "serialized"])
def test_independent_packed_measurements_execute(
    sdk_transpiler, kernel, expected, decode_count, roundtrip
):
    """Each SDK executes one circuit and reconstructs all typed outputs."""
    if roundtrip:
        kernel = deserialize(serialize(kernel))
    transpiler = sdk_transpiler.transpiler
    executable = transpiler.transpile(kernel)
    plan = executable.plan
    assert plan is not None
    assert [type(step) for step in plan.steps] == [QuantumStep, ClassicalStep]
    quantum, post = plan.steps
    assert post.role == "post"
    decode_types = (DecodeQIntOperation, DecodeQFixedOperation)
    assert not any(isinstance(op, decode_types) for op in quantum.segment.operations)
    assert (
        sum(isinstance(op, decode_types) for op in post.segment.operations)
        == decode_count
    )
    assert all(boundary.source_segment_index == 0 for boundary in plan.boundaries)
    assert all(boundary.target_segment_index == 1 for boundary in plan.boundaries)
    for operation in post.segment.operations:
        if isinstance(operation, decode_types):
            bits_ref = operation.operands[0].uuid
            assert bits_ref in quantum.segment.output_refs
            assert bits_ref in post.segment.input_refs

    result = executable.sample(transpiler.executor(), shots=8).result()
    assert result.results == [(expected, 8)], sdk_transpiler.engine_name


@qmc.qkernel
def _decoded_qfixed_gate_parameter() -> qmc.Bit:
    """Require a host decode before a subsequent quantum gate."""
    a = qmc.qubit_array(2, "a")
    b = qmc.qubit("b")
    angle = qmc.measure(qmc.cast(a, qmc.QFixed, int_bits=0))
    b = qmc.rx(b, angle)
    return qmc.measure(b)


@qmc.qkernel
def _decoded_qint_indirect_gate_parameter() -> qmc.Bit:
    """Require a decoded integer through arithmetic before a quantum gate."""
    a = qmc.qubit_array(2, "a")
    b = qmc.qubit("b")
    angle = qmc.measure(qmc.cast(a, qmc.QInt)) * 0.5
    b = qmc.rx(b, angle)
    return qmc.measure(b)


@qmc.qkernel
def _decoded_qint_runtime_condition() -> qmc.Bit:
    """Require a decoded integer through a runtime quantum branch condition."""
    a = qmc.qubit_array(2, "a")
    b = qmc.qubit("b")
    measured = qmc.measure(qmc.cast(a, qmc.QInt))
    if measured == 1:
        b = qmc.x(b)
    return qmc.measure(b)


@pytest.mark.parametrize(
    "kernel, error, message",
    [
        (_decoded_qfixed_gate_parameter, DependencyError, "depends on measurement"),
        (
            _decoded_qint_indirect_gate_parameter,
            DependencyError,
            "depends on measurement",
        ),
        (
            _decoded_qint_runtime_condition,
            MultipleQuantumSegmentsError,
            "quantum segments",
        ),
    ],
    ids=["direct-gate-parameter", "indirect-gate-parameter", "runtime-condition"],
)
@pytest.mark.parametrize("roundtrip", [False, True], ids=["direct", "serialized"])
def test_dependent_packed_measurements_preserve_quantum_boundary(
    qiskit_transpiler, kernel, error, message, roundtrip
):
    """Decodes required by later quantum operations cannot move past them."""
    if roundtrip:
        kernel = deserialize(serialize(kernel))
    with pytest.raises(error, match=message):
        qiskit_transpiler.transpile(kernel)

    # Exercise scheduling directly too: gate parameters are normally rejected
    # earlier by AnalyzePass, before reaching the decode dependency guard.
    with pytest.raises(MultipleQuantumSegmentsError, match="quantum segments"):
        SegmentationPass().run(kernel.build())
