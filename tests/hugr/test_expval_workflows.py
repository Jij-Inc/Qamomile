"""Exercise structured HUGR expectation workflows across execution targets."""

from __future__ import annotations

import math

import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.ir.operation.callable import InvokeOperation
from qamomile.circuit.transpiler import (
    ExpvalJob,
    QamomileCompiler,
    RunJob,
    TargetCapabilityError,
)
from qamomile.hugr import HugrExecutor, HugrTranspiler, SeleneExecutionOptions
from qamomile.hugr._expval import ShotExpectationWorkflow, prepare_expval
from qamomile.hugr._nexus import NexusTransport
from tests.hugr.test_expval import _calling_expectation, _mock_helios_client


@qmc.qkernel
def _state(q: qmc.Qubit, angle: qmc.Float) -> qmc.Qubit:
    """Prepare one rotated qubit while retaining its callable boundary.

    Args:
        q (qmc.Qubit): Input state to rotate.
        angle (qmc.Float): Rotation angle in radians.

    Returns:
        qmc.Qubit: State after the Y rotation.
    """
    return qmc.ry(q, angle)


@qmc.qkernel
def _estimate(q: qmc.Qubit, observable: qmc.Observable) -> qmc.Float:
    """Consume a state in a helper-owned expectation.

    Args:
        q (qmc.Qubit): State to estimate.
        observable (qmc.Observable): One-qubit observable to measure.

    Returns:
        qmc.Float: Estimated observable expectation.
    """
    return qmc.expval(q, observable)


@qmc.qkernel
def _structured(
    theta: qmc.Float, scale: qmc.Float, observable: qmc.Observable
) -> tuple[qmc.Float, qmc.Float, qmc.Float, qmc.Float]:
    """Return estimates and classical expressions from independent qubits.

    Args:
        theta (qmc.Float): Twice the left qubit rotation angle in radians.
        scale (qmc.Float): Multiplier for the sum of the estimates.
        observable (qmc.Observable): One-qubit observable for both states.

    Returns:
        tuple: Four classical outputs:
            scaled_sum (qmc.Float): Scaled sum of both estimates.
            squared_first (qmc.Float): Square of the left state estimate.
            second (qmc.Float): Right state estimate.
            angle (qmc.Float): Left qubit rotation angle in radians.
    """
    angle = theta * 0.5
    left = qmc.qubit("left")
    right = qmc.qubit("right")
    left = _state(left, angle)
    right = qmc.x(right)
    first = _estimate(left, observable)
    second = _estimate(right, observable)
    return (first + second) * scale, first * first, second, theta * 0.5


@qmc.qkernel
def _array_postprocessing(
    values: qmc.Vector[qmc.Float], observable: qmc.Observable
) -> tuple[qmc.Float, qmc.Float, qmc.Float]:
    """Use runtime array slots before and after quantum execution.

    Args:
        values (qmc.Vector[qmc.Float]): Rotation angle and additive offset.
        observable (qmc.Observable): One-qubit observable to estimate.

    Returns:
        tuple: Three classical outputs:
            shifted (qmc.Float): Estimate plus the second array element.
            estimate (qmc.Float): Unmodified expectation.
            angle (qmc.Float): First array element, in radians.
    """
    q = qmc.qubit("q")
    q = qmc.ry(q, values[0])
    estimate = qmc.expval(q, observable)
    return estimate + values[1], estimate, values[0]


@qmc.qkernel
def _shifted(offset: qmc.Float, observable: qmc.Observable) -> qmc.Float:
    """Apply a compile-time shift to an otherwise identical measurement package.

    Args:
        offset (qmc.Float): Compile-time additive offset.
        observable (qmc.Observable): Observable to estimate on the zero state.

    Returns:
        qmc.Float: Estimated expectation plus the offset.
    """
    q = qmc.qubit("q")
    estimate = qmc.expval(q, observable)
    return estimate + offset


@qmc.qkernel
def _tuple_postprocessing(
    pair: qmc.Tuple[qmc.Float, qmc.Float], observable: qmc.Observable
) -> tuple[qmc.Float, qmc.Tuple[qmc.Float, qmc.Float]]:
    """Return a nested typed tuple alongside a processed expectation.

    Args:
        pair (qmc.Tuple[qmc.Float, qmc.Float]): Preserved classical values.
        observable (qmc.Observable): Observable to estimate on the zero state.

    Returns:
        tuple: Two structured outputs:
            shifted (qmc.Float): Estimate plus the second tuple element.
            pair (qmc.Tuple[qmc.Float, qmc.Float]): Original classical pair.
    """
    q = qmc.qubit("q")
    estimate = qmc.expval(q, observable)
    return estimate + pair[1], pair


@qmc.qkernel
def _loop_postprocessing(
    repeats: qmc.UInt, bias: qmc.Float, observable: qmc.Observable
) -> qmc.Float:
    """Run a classical loop after estimating the state.

    Args:
        repeats (qmc.UInt): Compile-time number of classical iterations.
        bias (qmc.Float): Offset added during each iteration.
        observable (qmc.Observable): Observable to estimate on the zero state.

    Returns:
        qmc.Float: Estimated expectation plus the accumulated bias.
    """
    q = qmc.qubit("q")
    estimate = qmc.expval(q, observable)
    for _ in qmc.range(repeats):
        estimate = estimate + bias
    return estimate


@qmc.qkernel
def _array_result(
    values: qmc.Vector[qmc.Float], observable: qmc.Observable
) -> tuple[qmc.Float, qmc.Vector[qmc.Float]]:
    """Store an estimated value in a classical array and return its contents.

    Args:
        values (qmc.Vector[qmc.Float]): Array with at least two elements.
        observable (qmc.Observable): Observable to estimate on the zero state.

    Returns:
        tuple: Two outputs:
            estimate (qmc.Float): Unmodified expectation.
            values (qmc.Vector[qmc.Float]): Array with twice the estimate
                stored at index one.
    """
    q = qmc.qubit("q")
    estimate = qmc.expval(q, observable)
    values[1] = estimate * 2.0
    return estimate, values


@qmc.qkernel
def _bound_branch(flag: qmc.UInt, observable: qmc.Observable) -> qmc.Float:
    """Select an expectation branch from a compile-time condition.

    Args:
        flag (qmc.UInt): Compile-time branch selector, compared with one.
        observable (qmc.Observable): Observable to estimate on the zero state.

    Returns:
        qmc.Float: Expectation from the selected branch.
    """
    q = qmc.qubit("q")
    if flag == 1:
        estimate = qmc.expval(q, observable)
    else:
        estimate = qmc.expval(q, observable)
    return estimate


@qmc.qkernel
def _resumed_quantum(observable: qmc.Observable) -> tuple[qmc.Float, qmc.Float]:
    """Attempt to resume quantum computation after obtaining an estimate.

    Args:
        observable (qmc.Observable): One-qubit observable for both states.

    Returns:
        tuple: Two estimates in an unsupported execution order:
            first (qmc.Float): Estimate preceding the second allocation.
            second (qmc.Float): Estimate after quantum work resumes.
    """
    left = qmc.qubit("left")
    first = qmc.expval(left, observable)
    right = qmc.qubit("right")
    second = qmc.expval(right, observable)
    return first, second


def _helios(monkeypatch, rows):
    """Use SDK-shaped mocked remote jobs without contacting hardware.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture replacing the Nexus transport.
        rows (list[list[tuple[int, ...]]]): Measurement rows grouped by term.

    Returns:
        tuple: The execution target and its mock SDK client:
            executor (HugrExecutor): Helios target using the mocked transport.
            client (SimpleNamespace): SDK-shaped client with inspectable calls.
    """
    client = _mock_helios_client(rows)
    monkeypatch.setattr(
        "qamomile.hugr.execution._NexusTransport",
        lambda options: NexusTransport(options, client=client),
    )
    return HugrExecutor("helios"), client


def test_expectation_helpers_preserve_unrelated_quantum_calls():
    """Host expectation legalization leaves the state-preparation helper intact."""
    source = QamomileCompiler().prepare(
        _structured, bindings={"observable": qm_o.Z(0)}, parameters=["theta", "scale"]
    )
    original_calls = [
        op for op in source.entrypoint.operations if isinstance(op, InvokeOperation)
    ]
    plan = prepare_expval(source)
    assert isinstance(plan, ShotExpectationWorkflow)
    assert len(original_calls) == 3
    for circuit in plan.circuits:
        calls = [
            op
            for op in circuit.entrypoint.operations
            if isinstance(op, InvokeOperation)
        ]
        assert [call.target for call in calls] == [original_calls[0].target]
    assert (
        len(
            [
                op
                for op in source.entrypoint.operations
                if isinstance(op, InvokeOperation)
            ]
        )
        == 3
    )
    assert plan.aggregate([[0], [1]], {"theta": 0.8, "scale": 3.0}) == pytest.approx(
        (0.0, 1.0, -1.0, 0.4), abs=1e-12
    )


def test_quantum_after_expectation_is_rejected_before_submission():
    """The workflow maintains the shared single quantum region execution contract."""
    with pytest.raises(TargetCapabilityError, match="resume quantum work"):
        HugrTranspiler().transpile(_resumed_quantum, bindings={"observable": qm_o.Z(0)})


@pytest.mark.hugr
def test_selene_structured_expectations_and_classical_work_match_qiskit(tmp_path):
    """Direct HUGR programs reproduce the common C-to-Q-to-expval-to-C workflow."""
    pytest.importorskip("selene_sim")
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitExecutor, QiskitTranspiler

    options = dict(bindings={"observable": qm_o.Z(0)}, parameters=["theta", "scale"])
    executable = HugrTranspiler().transpile(_structured, **options)
    reference = QiskitTranspiler().transpile(_structured, **options)
    executor = HugrExecutor(
        options=SeleneExecutionOptions(seed=12, n_qubits=2, build_dir=tmp_path)
    )
    for theta in (0.0, math.pi, 2 * math.pi):
        bindings = {"theta": theta, "scale": 2.0}
        job = executable.run(executor, bindings=bindings, shots=1024)
        assert isinstance(job, RunJob)
        actual = job.result()
        expected = reference.run(QiskitExecutor(), bindings=bindings).result()
        assert actual[0] == pytest.approx(expected[0], abs=0.12)
        assert actual[1:3] == pytest.approx(expected[1:3], abs=0.08)
        assert actual[3] == pytest.approx(expected[3], abs=1e-12)


@pytest.mark.hugr
def test_selene_pure_helper_estimate_preserves_expval_job(tmp_path):
    """Inlining an expectation helper retains scalar public API compatibility."""
    pytest.importorskip("selene_sim")
    executable = HugrTranspiler().transpile(
        _calling_expectation, bindings={"observable": qm_o.Z(0)}
    )
    executor = HugrExecutor(
        options=SeleneExecutionOptions(n_qubits=1, build_dir=tmp_path)
    )
    job = executable.run(executor, shots=8)
    assert isinstance(job, ExpvalJob)
    assert job.result() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.hugr
@pytest.mark.parametrize("indexed", [False, True])
def test_helios_array_postprocessing_restores_owned_runtime_values(
    monkeypatch, indexed
):
    """Indexed inputs, deferred host work, and restore share one binding snapshot."""
    pytest.importorskip("hugr")
    executor, client = _helios(monkeypatch, [[(0,), (0,), (0,), (1,)]])
    executable = HugrTranspiler().transpile(
        _array_postprocessing,
        bindings={"observable": qm_o.Z(0)},
        parameters=["values"],
        parameter_shapes={"values": (2,)},
    )
    bindings = (
        {"values[0]": 0.3, "values[1]": 2.0} if indexed else {"values": [0.3, 2.0]}
    )
    job = executable.run(executor, bindings=bindings, shots=4)
    snapshot = job.snapshot()
    if indexed:
        bindings["values[1]"] = 99.0
    else:
        bindings["values"][1] = 99.0
    assert job.result() == pytest.approx((2.5, 0.5, 0.3), abs=1e-12)
    restored = executable.restore(executor, snapshot, bindings={"values": [0.3, 2.0]})
    assert isinstance(restored, RunJob)
    assert restored.result() == pytest.approx(job.result(), abs=1e-12)
    assert client.start_execute_job.call_count == 1
    with pytest.raises(ValueError, match="Snapshot"):
        executable.restore(executor, snapshot, bindings={"values": [0.3, 3.0]})


@pytest.mark.hugr
def test_helios_multiple_estimates_restore_structured_output(monkeypatch):
    """Separate Pauli jobs reconstruct nested outputs before and after restore."""
    executor, client = _helios(monkeypatch, [[(0,), (0,), (0,), (1,)], [(1,)] * 4])
    options = dict(bindings={"observable": qm_o.Z(0)}, parameters=["theta", "scale"])
    executable = HugrTranspiler().transpile(_structured, **options)
    job = executable.run(executor, bindings={"theta": 0.7, "scale": 2.0}, shots=4)
    assert job.result() == pytest.approx((-1.0, 0.25, -1.0, 0.35), abs=1e-12)
    rebuilt = HugrTranspiler().transpile(_structured, **options)
    restored = rebuilt.restore(
        executor, job.snapshot(), bindings={"theta": 0.7, "scale": 2.0}
    )
    assert restored.result() == pytest.approx(job.result(), abs=1e-12)
    assert client.start_execute_job.call_count == 2


@pytest.mark.hugr
def test_restore_rejects_changed_classical_postprocessing(monkeypatch):
    """Equal measured graphs cannot authorize a different host-side result recipe."""
    executor, client = _helios(monkeypatch, [[(0,)]])
    transpiler = HugrTranspiler()
    first = transpiler.transpile(
        _shifted, bindings={"observable": qm_o.Z(0), "offset": 2.0}
    )
    second = transpiler.transpile(
        _shifted, bindings={"observable": qm_o.Z(0), "offset": 7.0}
    )
    snapshot = first.run(executor, shots=1).snapshot()
    with pytest.raises(ValueError, match="expectation recipe"):
        second.restore(executor, snapshot)
    client.jobs.get.assert_not_called()


@pytest.mark.hugr
def test_identity_only_workflow_evaluates_classical_result_without_provider(
    monkeypatch,
):
    """Identity estimates still resolve all runtime-dependent public outputs."""
    from unittest.mock import Mock

    observable = qm_o.Hamiltonian()
    observable.constant = 0.5
    executable = HugrTranspiler().transpile(
        _array_postprocessing,
        bindings={"observable": observable},
        parameters=["values"],
        parameter_shapes={"values": (2,)},
    )
    executor = Mock(spec=HugrExecutor)
    job = executable.run(executor, bindings={"values[0]": 0.3, "values[1]": 2.0})
    assert job.result() == pytest.approx((2.5, 0.5, 0.3), abs=1e-12)
    executor.submit.assert_not_called()


@pytest.mark.hugr
def test_helios_nested_tuple_and_classical_loop_outputs(monkeypatch):
    """Public tuple structure and host loop-carried arithmetic are preserved."""
    executor, _ = _helios(monkeypatch, [[(0,)], [(0,)]])
    nested = HugrTranspiler().transpile(
        _tuple_postprocessing,
        bindings={"observable": qm_o.Z(0), "pair": (0.5, 2.0)},
    )
    nested_result = nested.run(executor, shots=1).result()
    assert nested_result[0] == pytest.approx(3.0, abs=1e-12)
    assert nested_result[1] == pytest.approx((0.5, 2.0), abs=1e-12)
    loop = HugrTranspiler().transpile(
        _loop_postprocessing,
        bindings={"observable": qm_o.Z(0), "repeats": 3},
        parameters=["bias"],
    )
    assert loop.run(
        executor, bindings={"bias": 0.25}, shots=1
    ).result() == pytest.approx(1.75, abs=1e-12)


@pytest.mark.hugr
def test_helios_postprocessing_uses_canonical_float_inputs(monkeypatch):
    """Accepted Decimal inputs have the same Float semantics on device and host."""
    from decimal import Decimal

    executor, _ = _helios(monkeypatch, [[(0,)]])
    executable = HugrTranspiler().transpile(
        _array_postprocessing,
        bindings={"observable": qm_o.Z(0)},
        parameters=["values"],
        parameter_shapes={"values": (2,)},
    )
    job = executable.run(
        executor, bindings={"values": [Decimal("0.3"), Decimal("2.0")]}, shots=1
    )
    assert job.result() == pytest.approx((3.0, 1.0, 0.3), abs=1e-12)


@pytest.mark.hugr
def test_helios_array_output_retains_abi_container_and_input_ownership(monkeypatch):
    """Expectation-derived stores return the same tuple carrier as native arrays."""
    executor, _ = _helios(monkeypatch, [[(0,)]])
    executable = HugrTranspiler().transpile(
        _array_result,
        bindings={"observable": qm_o.Z(0)},
        parameters=["values"],
        parameter_shapes={"values": (3,)},
    )
    values = [0.2, 0.3, 0.4]
    result = executable.run(executor, bindings={"values": values}, shots=1).result()
    assert result[0] == pytest.approx(1.0, abs=1e-12)
    assert isinstance(result[1], tuple)
    assert result[1] == pytest.approx((0.2, 2.0, 0.4), abs=1e-12)
    assert values == pytest.approx((0.2, 0.3, 0.4), abs=1e-12)


@pytest.mark.hugr
def test_helios_dictionary_output_contains_estimate(monkeypatch):
    """A legal structured IR output resolves expectation-derived dictionary entries."""
    from dataclasses import replace
    from types import SimpleNamespace

    from qamomile.circuit.ir.operation.return_operation import ReturnOperation
    from qamomile.circuit.ir.types import UIntType
    from qamomile.circuit.ir.value import DictValue, Value

    bindings = {"observable": qm_o.Z(0), "offset": 2.0}
    prepared = QamomileCompiler().prepare(_shifted, bindings=bindings)
    key = Value(type=UIntType(), name="key").with_const(7)
    output = DictValue(
        name="estimates", entries=((key, prepared.abi.output_values[0]),)
    )
    block = replace(
        prepared.entrypoint,
        output_values=[output],
        output_names=[output.name],
        operations=[
            *(
                op
                for op in prepared.entrypoint.operations
                if not isinstance(op, ReturnOperation)
            ),
            ReturnOperation(operands=[output]),
        ],
    )
    kernel = SimpleNamespace(build=lambda **_: block)
    executable = HugrTranspiler().transpile(kernel, bindings=bindings)
    executor, _ = _helios(monkeypatch, [[(0,)]])
    job = executable.run(executor, shots=1)
    assert job.result() == pytest.approx({7: 3.0}, abs=1e-12)
    assert executable.restore(executor, job.snapshot()).result() == pytest.approx(
        {7: 3.0}, abs=1e-12
    )


@pytest.mark.hugr
@pytest.mark.parametrize("flag", [0, 1])
def test_bound_expectation_branches_are_specialized(monkeypatch, flag):
    """Compile-time branch specialization exposes a legal terminal expectation."""
    executor, _ = _helios(monkeypatch, [[(0,)]])
    executable = HugrTranspiler().transpile(
        _bound_branch,
        bindings={"observable": qm_o.Z(0), "flag": flag},
    )
    assert executable.run(executor, shots=1).result() == pytest.approx(1.0, abs=1e-12)
