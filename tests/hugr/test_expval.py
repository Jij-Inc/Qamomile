"""Verify hardware-compatible HUGR shot expectation preparation and estimation."""

from __future__ import annotations

import math
from typing import Any

import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.ir.operation.expval import ExpvalOp
from qamomile.circuit.ir.operation.gate import (
    GateOperation,
    GateOperationType,
    MeasureOperation,
)
from qamomile.circuit.transpiler import (
    CompiledProgram,
    PreparedModule,
    QamomileCompiler,
    QKernelLike,
)
from qamomile.circuit.transpiler.errors import TargetCapabilityError
from qamomile.hugr._expval import ShotExpectationPlan, prepare_expval
from qamomile.hugr.lowerer import HugrTarget


@qmc.qkernel
def _rotated(theta: qmc.Float, observable: qmc.Observable) -> qmc.Float:
    """Return one parameterized qubit's expectation.

    Args:
        theta (qmc.Float): Runtime rotation angle in radians.
        observable (qmc.Observable): Compile-time Hermitian Pauli observable.

    Returns:
        qmc.Float: Expectation of the rotated state.
    """
    q = qmc.qubit("q")
    q = qmc.ry(q, theta)
    return qmc.expval(q, observable)


@qmc.qkernel
def _entangled(theta: qmc.Float, observable: qmc.Observable) -> qmc.Float:
    """Prepare a phased Bell-like state with nontrivial Pauli correlations.

    Args:
        theta (qmc.Float): Runtime rotation angle in radians.
        observable (qmc.Observable): Compile-time Hermitian Pauli observable.

    Returns:
        qmc.Float: Expectation of the phased entangled state.
    """
    q = qmc.qubit_array(2, "q")
    q[0] = qmc.ry(q[0], theta)
    q[0], q[1] = qmc.cx(q[0], q[1])
    q[0] = qmc.s(q[0])
    return qmc.expval(q, observable)


@qmc.qkernel
def _tuple_order(observable: qmc.Observable) -> qmc.Float:
    """Measure reordered qubit operands with distinguishable eigenvalues.

    Args:
        observable (qmc.Observable): Compile-time Hermitian Pauli observable.

    Returns:
        qmc.Float: Expectation using the reordered tuple as its qubit basis.
    """
    q = qmc.qubit_array(3, "q")
    q[2] = qmc.x(q[2])
    return qmc.expval((q[2], q[0]), observable)


@qmc.qkernel
def _strided_order(observable: qmc.Observable) -> qmc.Float:
    """Measure a strided view in view-local observable order.

    Args:
        observable (qmc.Observable): Compile-time Hermitian Pauli observable.

    Returns:
        qmc.Float: Expectation using the strided view as its qubit basis.
    """
    q = qmc.qubit_array(4, "q")
    q[3] = qmc.x(q[3])
    return qmc.expval(q[1::2], observable)


@qmc.qkernel
def _measurement() -> qmc.Bit:
    """Return an ordinary measurement without an expectation.

    Returns:
        qmc.Bit: Measurement of the zero-state qubit.
    """
    q = qmc.qubit("q")
    return qmc.measure(q)


@qmc.qkernel
def _postprocessed(observable: qmc.Observable) -> qmc.Float:
    """Use an expectation in classical postprocessing.

    Args:
        observable (qmc.Observable): Compile-time Hermitian Pauli observable.

    Returns:
        qmc.Float: Classically postprocessed expectation.
    """
    q = qmc.qubit("q")
    result = qmc.expval(q, observable)
    return result + 1.0


def _prepare(
    kernel: QKernelLike,
    observable: qm_o.Hamiltonian,
    parameters: list[str] | None = None,
) -> PreparedModule:
    """Prepare an expectation program with a compile-time observable.

    Args:
        kernel (QKernelLike): Test kernel containing an expectation.
        observable (Hamiltonian): Compile-time Pauli observable.
        parameters (list[str] | None): Runtime parameter names.

    Returns:
        PreparedModule: Compiler-owned hierarchical semantics.
    """
    return QamomileCompiler().prepare(
        kernel, bindings={"observable": observable}, parameters=parameters
    )


def _compile_plan(plan: ShotExpectationPlan) -> list[CompiledProgram[Any]]:
    """Lower every measurement program without native validator side effects.

    Args:
        plan (ShotExpectationPlan): Prepared Pauli measurement programs.

    Returns:
        list[CompiledProgram[Any]]: Native packages in Pauli-term order.
    """
    pytest.importorskip("hugr")
    pytest.importorskip("tket_exts")
    target = HugrTarget()
    return [target.compile(circuit, target.plan(circuit)) for circuit in plan.circuits]


def test_expectation_keeps_runtime_parameters_and_source_ir():
    """Each Pauli circuit owns its semantics and retains the runtime ABI."""
    observable = 0.4 * qm_o.X(0) - 0.3 * qm_o.Y(0) + 0.7 * qm_o.Z(0) + 0.2
    prepared = _prepare(_rotated, observable, ["theta"])
    original_ops = tuple(prepared.entrypoint.operations)
    plan = prepare_expval(prepared)

    assert len(plan.circuits) == 3
    assert plan.constant == pytest.approx(0.2)
    assert tuple(prepared.entrypoint.operations) == original_ops
    assert any(isinstance(op, ExpvalOp) for op in prepared.entrypoint.operations)
    for circuit in plan.circuits:
        assert set(circuit.abi.public_inputs) == set(prepared.abi.public_inputs)
        assert set(circuit.bindings) == {"observable"}
        assert not any(isinstance(op, ExpvalOp) for op in circuit.entrypoint.operations)
        assert len(circuit.abi.output_values) == 1
    lowered = _compile_plan(plan)
    assert all(
        set(compiled.abi.public_inputs) == set(prepared.abi.public_inputs)
        for compiled in lowered
    )


def test_pauli_basis_changes_and_parity():
    """X uses H, Y uses S-adjoint then H, and products use joint parity."""
    observable = 0.5 * qm_o.X(0) * qm_o.Y(1) + 0.2 * qm_o.Z(0) + 0.7
    plan = prepare_expval(_prepare(_entangled, observable, ["theta"]))
    operations = plan.circuits[0].entrypoint.operations
    tail = operations[-6:]
    assert isinstance(tail[0], GateOperation)
    assert tail[0].gate_type is GateOperationType.H
    assert isinstance(tail[1], MeasureOperation)
    assert tail[2].gate_type is GateOperationType.SDG
    assert tail[3].gate_type is GateOperationType.H
    assert isinstance(tail[4], MeasureOperation)
    assert plan.parity_indices == ((0, 1), (0,))
    assert plan.aggregate(
        [[(0, 0), (1, 1), (1, 0), (0, 1)], [(1, 0)] * 4]
    ) == pytest.approx(0.5)
    _compile_plan(plan)


def test_expectation_identity_needs_no_quantum_jobs():
    """Identity-only observables return their constant without measurement jobs."""
    observable = qm_o.Hamiltonian()
    observable.constant = -2.5
    plan = prepare_expval(_prepare(_rotated, observable, ["theta"]))
    assert plan.circuits == ()
    assert plan.aggregate([]) == -2.5


def test_ordinary_program_has_no_expectation_plan():
    """Ordinary measurement compilation follows its existing path."""
    assert prepare_expval(QamomileCompiler().prepare(_measurement)) is None


@pytest.mark.parametrize("coefficient", [float("nan"), float("inf"), 1.0j])
@pytest.mark.parametrize("location", ["constant", "term"])
def test_invalid_observable_rejected_before_execution(coefficient, location):
    """Nonfinite or non-Hermitian observables fail before submitting jobs."""
    observable = qm_o.Z(0)
    if location == "constant":
        observable.constant = coefficient
    else:
        observable = qm_o.Hamiltonian()
        observable.add_term((qm_o.PauliOperator(qm_o.Pauli.Z, 0),), coefficient)
    with pytest.raises(TargetCapabilityError, match="finite real"):
        prepare_expval(_prepare(_rotated, observable, ["theta"]))


def test_observable_width_rejected():
    """Observable indices cannot exceed the supplied quantum state width."""
    with pytest.raises(TargetCapabilityError, match="more qubits"):
        prepare_expval(_prepare(_rotated, qm_o.Z(1), ["theta"]))


def test_expectation_postprocessing_uses_classical_interpreter():
    """Classical work executes once after reconstructing the estimate."""
    plan = prepare_expval(_prepare(_postprocessed, qm_o.Z(0)))
    assert plan.aggregate([[0, 0, 0, 1]]) == pytest.approx(1.5)
    _compile_plan(plan)


@pytest.mark.parametrize("rows", [[], [[]], [[(0, 1)]], [[2]], [[0.5]]])
def test_aggregate_rejects_malformed_service_results(rows):
    """Missing batches, empty shots, wrong shapes, and non-bits fail closed."""
    plan = prepare_expval(_prepare(_rotated, qm_o.Z(0), ["theta"]))
    with pytest.raises(ValueError):
        plan.aggregate(rows)


@pytest.mark.parametrize("kernel", [_tuple_order, _strided_order])
def test_nontrivial_state_carriers_lower_to_measurements(kernel):
    """Tuple and strided operands compile without losing observable-local order."""
    plan = prepare_expval(_prepare(kernel, qm_o.Z(0) - 0.5 * qm_o.Z(1)))
    assert plan.num_qubits == 2
    assert plan.parity_indices == ((0,), (1,))
    _compile_plan(plan)


@qmc.qkernel
def _multiple_expectations(observable: qmc.Observable) -> tuple[qmc.Float, qmc.Float]:
    """Return expectations for two separately allocated states.

    Args:
        observable (qmc.Observable): Compile-time Hermitian Pauli observable.

    Returns:
        tuple[qmc.Float, qmc.Float]: Independent expectations in public output order.
    """
    left = qmc.qubit("left")
    right = qmc.qubit("right")
    first = qmc.expval(left, observable)
    second = qmc.expval(right, observable)
    return first, second


@qmc.qkernel
def _nested_expectation(q: qmc.Qubit, observable: qmc.Observable) -> qmc.Float:
    """Return an expectation from a quantum helper.

    Args:
        q (qmc.Qubit): Consumed quantum state to measure.
        observable (qmc.Observable): Compile-time Hermitian Pauli observable.

    Returns:
        qmc.Float: Expectation produced inside the helper.
    """
    return qmc.expval(q, observable)


@qmc.qkernel
def _calling_expectation(observable: qmc.Observable) -> qmc.Float:
    """Invoke an expectation-valued helper from the entrypoint.

    Args:
        observable (qmc.Observable): Compile-time Hermitian Pauli observable.

    Returns:
        qmc.Float: Expectation produced by the helper.
    """
    q = qmc.qubit("q")
    return _nested_expectation(q, observable)


@pytest.mark.parametrize(
    ("kernel", "expected"),
    [(_multiple_expectations, (1.0, 1.0)), (_calling_expectation, 1.0)],
)
def test_multiple_and_helper_expectations_are_prepared(kernel, expected):
    """Independent estimates and inline-policy helpers use measured packages."""
    plan = prepare_expval(_prepare(kernel, qm_o.Z(0)))
    assert plan.aggregate([[0]] * len(plan.circuits)) == pytest.approx(
        expected, abs=1e-12
    )
    _compile_plan(plan)


@pytest.mark.hugr
def test_selene_expectation_repeated_runtime_values_match_qiskit(tmp_path):
    """One executable accepts different angles and estimates X/Y/Z correlations."""
    pytest.importorskip("selene_sim")
    pytest.importorskip("qiskit")
    from qamomile.circuit.transpiler.job import ExpvalJob
    from qamomile.hugr import HugrTranspiler
    from qamomile.hugr.execution import HugrExecutor, SeleneExecutionOptions
    from qamomile.qiskit import QiskitExecutor, QiskitTranspiler

    observable = (
        0.7 * qm_o.X(0) * qm_o.Y(1)
        - 0.3 * qm_o.Z(0)
        + 0.2 * qm_o.Z(0) * qm_o.Z(1)
        + 0.13
    )
    executable = HugrTranspiler().transpile(
        _entangled, bindings={"observable": observable}, parameters=["theta"]
    )
    reference = QiskitTranspiler().transpile(
        _entangled, bindings={"observable": observable}, parameters=["theta"]
    )
    executor = HugrExecutor(
        options=SeleneExecutionOptions(seed=42, n_qubits=2, build_dir=tmp_path)
    )
    for angle in (0.0, 0.61, math.pi / 2, math.pi):
        job = executable.run(executor, bindings={"theta": angle}, shots=2048)
        assert isinstance(job, ExpvalJob)
        actual = job.result()
        expected = reference.run(QiskitExecutor(), bindings={"theta": angle}).result()
        assert actual == pytest.approx(expected, abs=0.08)
    with pytest.raises(ValueError, match="Expectation"):
        executable.sample(executor, shots=8, bindings={"theta": 0.0})


@pytest.mark.hugr
@pytest.mark.parametrize(
    ("kernel", "expected"), [(_tuple_order, -1.5), (_strided_order, 1.5)]
)
def test_selene_expectation_respects_operand_order(tmp_path, kernel, expected):
    """Preserve indices: Z on |1,0> gives -1.5, and |0,1> gives 1.5."""
    pytest.importorskip("selene_sim")
    from qamomile.hugr import HugrTranspiler
    from qamomile.hugr.execution import HugrExecutor, SeleneExecutionOptions

    executable = HugrTranspiler().transpile(
        kernel, bindings={"observable": qm_o.Z(0) - 0.5 * qm_o.Z(1)}
    )
    executor = HugrExecutor(
        options=SeleneExecutionOptions(seed=8, n_qubits=4, build_dir=tmp_path)
    )
    assert executable.run(executor, shots=16).result() == expected


def _mock_helios_client(rows_by_term: list[list[tuple[int, ...]]]) -> Any:
    """Build an SDK-shaped Nexus client with separately retrievable term jobs.

    Args:
        rows_by_term (list[list[tuple[int, ...]]]): Public measurement rows
            grouped by Pauli term.

    Returns:
        Any: Mocked SDK client exposing upload, submit, status, and results.
    """
    from types import SimpleNamespace
    from unittest.mock import Mock

    from hugr.qsystem.result import QsysResult

    jobs = [
        SimpleNamespace(id=f"term-{index}", job_type="execute")
        for index in range(len(rows_by_term))
    ]
    downloaded = {
        job.id: QsysResult(
            [
                [
                    (f"qamomile.output.{index}.0", bool(value))
                    for index, value in enumerate(row)
                ]
                for row in rows
            ]
        )
        for job, rows in zip(jobs, rows_by_term, strict=True)
    }
    return SimpleNamespace(
        HeliosConfig=Mock(
            side_effect=lambda **kwargs: SimpleNamespace(type="HeliosConfig", **kwargs)
        ),
        hugr=SimpleNamespace(upload=Mock(return_value="uploaded-hugr")),
        start_execute_job=Mock(side_effect=jobs),
        jobs=SimpleNamespace(
            get=Mock(
                side_effect=lambda *, id: next(job for job in jobs if job.id == id)
            ),
            cancel=Mock(),
            status=Mock(return_value=SimpleNamespace(status="COMPLETED")),
            results=Mock(
                side_effect=lambda job, **_: [
                    SimpleNamespace(
                        download_result=Mock(return_value=downloaded[job.id])
                    )
                ]
            ),
        ),
    )


@pytest.mark.hugr
def test_helios_expectation_aggregates_and_restores_independent_term_jobs(monkeypatch):
    """Restore 0.7 + 0.5*(3-1)/4 + 0.2*(1-3)/4 = 0.85 from SDK shots."""
    pytest.importorskip("hugr")
    from qamomile.hugr import HugrExecutor, HugrTranspiler
    from qamomile.hugr._nexus import NexusTransport

    rows = [[(0, 0), (1, 1), (0, 0), (0, 1)], [(0, 1), (1, 0), (1, 0), (1, 0)]]
    client = _mock_helios_client(rows)
    monkeypatch.setattr(
        "qamomile.hugr.execution._NexusTransport",
        lambda options: NexusTransport(options, client=client),
    )
    executable = HugrTranspiler().transpile(
        _entangled,
        bindings={"observable": 0.5 * qm_o.X(0) * qm_o.Y(1) + 0.2 * qm_o.Z(0) + 0.7},
        parameters=["theta"],
    )
    executor = HugrExecutor("helios")
    job = executable.run(executor, bindings={"theta": 0.9}, shots=4)
    assert client.start_execute_job.call_count == 2
    assert all(
        call.kwargs["n_shots"] == [4]
        for call in client.start_execute_job.call_args_list
    )
    client.jobs.results.assert_not_called()
    assert job.result() == pytest.approx(0.85)
    snapshot = job.snapshot()
    assert len(snapshot.executions) == 2
    restored = executable.restore(executor, snapshot, bindings={"theta": 0.9})
    assert restored.result() == pytest.approx(0.85)
    assert client.start_execute_job.call_count == 2
    assert client.hugr.upload.call_count == 2


@pytest.mark.hugr
def test_helios_expectation_cancels_submitted_terms_when_later_submission_fails(
    monkeypatch,
):
    """A partial submission failure cleans up earlier remote Pauli jobs."""
    pytest.importorskip("hugr")
    from types import SimpleNamespace

    from qamomile.hugr import HugrExecutor, HugrTranspiler
    from qamomile.hugr._nexus import NexusTransport

    client = _mock_helios_client([[(0,)], [(1,)]])
    first = SimpleNamespace(id="term-0", job_type="execute")
    client.start_execute_job.side_effect = [first, RuntimeError("second term rejected")]
    monkeypatch.setattr(
        "qamomile.hugr.execution._NexusTransport",
        lambda options: NexusTransport(options, client=client),
    )
    executable = HugrTranspiler().transpile(
        _rotated, bindings={"observable": qm_o.X(0) + qm_o.Z(0)}, parameters=["theta"]
    )
    with pytest.raises(RuntimeError, match="second term rejected"):
        executable.run(HugrExecutor("helios"), bindings={"theta": 0.9}, shots=1)
    client.jobs.cancel.assert_called_once_with(first)


@pytest.mark.hugr
@pytest.mark.parametrize("failure_stage", ["status", "cancel", "both"])
@pytest.mark.parametrize("restored", [False, True])
def test_helios_expectation_cancel_attempts_all_terms_after_provider_errors(
    monkeypatch, failure_stage, restored
):
    """Public cancellation reaches all three remote terms before reporting errors."""
    pytest.importorskip("hugr")
    from types import SimpleNamespace

    from qamomile.hugr import HugrExecutor, HugrTranspiler
    from qamomile.hugr._nexus import NexusTransport

    client = _mock_helios_client([[(0,)], [(0,)], [(0,)]])
    monkeypatch.setattr(
        "qamomile.hugr.execution._NexusTransport",
        lambda options: NexusTransport(options, client=client),
    )
    executable = HugrTranspiler().transpile(
        _rotated,
        bindings={"observable": qm_o.X(0) + qm_o.Y(0) + qm_o.Z(0)},
        parameters=["theta"],
    )
    executor = HugrExecutor("helios")
    job = executable.run(executor, bindings={"theta": 0.9}, shots=1)
    if restored:
        job = executable.restore(executor, job.snapshot(), bindings={"theta": 0.9})
    status_error = RuntimeError("first term status unavailable")
    cancel_error = ValueError("second term cancellation rejected")
    events = []

    def status(native: Any) -> Any:
        """Record polling and optionally fail for the first remote term.

        Args:
            native (Any): Nexus execute-job reference.

        Returns:
            Any: Provider status record for an unfinished job.

        Raises:
            RuntimeError: If the first term's status lookup is configured to fail.
        """
        events.append(("status", native.id))
        if native.id == "term-0" and failure_stage in ("status", "both"):
            raise status_error
        return SimpleNamespace(status="RUNNING")

    def cancel(native: Any) -> None:
        """Record cancellation and optionally fail for the second remote term.

        Args:
            native (Any): Nexus execute-job reference.

        Raises:
            ValueError: If the second term's cancellation is configured to fail.
        """
        events.append(("cancel", native.id))
        if native.id == "term-1" and failure_stage in ("cancel", "both"):
            raise cancel_error

    client.jobs.status.side_effect = status
    client.jobs.cancel.side_effect = cancel

    with pytest.raises(ExceptionGroup) as caught:
        job.cancel()

    expected_errors = []
    if failure_stage in ("status", "both"):
        expected_errors.append(status_error)
    if failure_stage in ("cancel", "both"):
        expected_errors.append(cancel_error)
    assert caught.value.exceptions == tuple(expected_errors)
    assert events == [
        (operation, f"term-{index}")
        for index in range(3)
        for operation in ("status", "cancel")
    ]
    assert client.start_execute_job.call_count == 3
    assert client.hugr.upload.call_count == 3
    client.jobs.results.assert_not_called()


@pytest.mark.hugr
@pytest.mark.parametrize(
    "observable", [5.0 * qm_o.X(0) + qm_o.Z(0), qm_o.X(0) + qm_o.Z(0) + 9.0]
)
def test_expectation_restore_rejects_different_observable(monkeypatch, observable):
    """Identical measurement circuits cannot identify different Hamiltonians."""
    pytest.importorskip("hugr")
    from qamomile.hugr import HugrExecutor, HugrTranspiler
    from qamomile.hugr._nexus import NexusTransport

    client = _mock_helios_client([[(0,)], [(1,)]])
    monkeypatch.setattr(
        "qamomile.hugr.execution._NexusTransport",
        lambda options: NexusTransport(options, client=client),
    )
    original = HugrTranspiler().transpile(
        _rotated, bindings={"observable": qm_o.X(0) + qm_o.Z(0)}, parameters=["theta"]
    )
    executor = HugrExecutor("helios")
    snapshot = original.run(executor, bindings={"theta": 0.9}, shots=1).snapshot()
    altered = HugrTranspiler().transpile(
        _rotated, bindings={"observable": observable}, parameters=["theta"]
    )
    with pytest.raises(ValueError, match="Snapshot"):
        altered.restore(executor, snapshot, bindings={"theta": 0.9})
    client.jobs.get.assert_not_called()
