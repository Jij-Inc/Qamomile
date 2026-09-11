"""Restore typed jobs from raw local, grouped, native-batch, and legacy data."""

import json
import math
import sys
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

import qamomile.observable as qm_o
from qamomile.circuit.ir.types.primitives import FloatType
from qamomile.circuit.ir.value import Value
from qamomile.circuit.transpiler.compiled_segments import (
    CompiledExpvalSegment,
    CompiledQuantumSegment,
)
from qamomile.circuit.transpiler.errors import ExecutionError
from qamomile.circuit.transpiler.executable import ExecutableProgram
from qamomile.circuit.transpiler.execution_handle import (
    CompletedExecutionHandle,
    CompositeExecutionHandle,
    ExecutionHandle,
    ExecutionReference,
    JobStatus,
)
from qamomile.circuit.transpiler.execution_snapshot import ExecutionSnapshot
from qamomile.circuit.transpiler.job import JobKind, JobSnapshot
from qamomile.circuit.transpiler.parameter_binding import ParameterMetadata
from qamomile.circuit.transpiler.quantum_executor import QuantumExecutor
from qamomile.circuit.transpiler.segments import (
    ExpvalSegment,
    ExpvalStep,
    ProgramABI,
    ProgramPlan,
    QuantumSegment,
    QuantumStep,
)


def _program(arity: int) -> ExecutableProgram[str]:
    """Construct a minimal engine-independent plan with ordered float outputs.

    Args:
        arity (int): Number of scalar expectations in the test result ABI.

    Returns:
        ExecutableProgram[str]: Program using an opaque mock circuit artifact.
    """
    quantum = QuantumSegment()
    outputs = [Value(type=FloatType(), name=f"expectation_{i}") for i in range(arity)]
    segments = [ExpvalSegment(None, None, result_ref=value.uuid) for value in outputs]
    return ExecutableProgram(
        plan=ProgramPlan(
            steps=[
                QuantumStep(quantum),
                *(ExpvalStep(segment) for segment in segments),
            ],
            abi=ProgramABI(output_values=outputs),
        ),
        compiled_quantum=[
            CompiledQuantumSegment(
                quantum, "test-circuit", parameter_metadata=ParameterMetadata()
            )
        ],
        compiled_expval=[
            CompiledExpvalSegment(segment, qm_o.Z(0), result_ref=value.uuid)
            for segment, value in zip(segments, outputs, strict=True)
        ],
        output_values=outputs,
    )


class _Remote(ExecutionHandle):
    """Keep result retrieval observable, including provider-native batches."""

    def __init__(self, value: Any, reference: ExecutionReference) -> None:
        """Initialize an observable mock provider execution.

        Args:
            value (Any): Provider result configured by the test.
            reference (ExecutionReference): Identity used when reattaching.
        """
        self.value = value
        self.reference = reference
        self.result_calls = 0

    def result(self, timeout: float | None = None) -> Any:
        """Record retrieval of the provider value.

        Args:
            timeout (float | None): Ignored compatibility wait limit.

        Returns:
            Any: Configured scalar or native-batch result.
        """
        self.result_calls += 1
        return self.value

    def status(self) -> JobStatus:
        """Expose the controlled unfinished status.

        Returns:
            JobStatus: The queued state.
        """
        return JobStatus.QUEUED

    def references(self) -> tuple[ExecutionReference, ...]:
        """Expose the provider identity without retrieving a result.

        Returns:
            tuple[ExecutionReference, ...]: One logical provider reference.
        """
        return (self.reference,)


class _Executor(QuantumExecutor):
    """Distinguish original submissions, retrieval, and actual result reads."""

    def __init__(self, handle: ExecutionHandle[Any] | None) -> None:
        """Configure separately observable submit and restore methods.

        Args:
            handle (ExecutionHandle[Any] | None): Handle for original submission,
                or None for tests that must never submit.
        """
        self.handle = handle
        self.submit_estimates = Mock(return_value=handle)
        self.restore = Mock()

    def execute(self, circuit: Any, shots: int) -> dict[str, int]:
        """Reject any unexpected synchronous execution.

        Args:
            circuit (Any): Unused mock circuit.
            shots (int): Unused requested sample count.

        Returns:
            dict[str, int]: Never returned.

        Raises:
            AssertionError: Always, because restoration must not execute.
        """
        raise AssertionError("Restoration must not execute or resubmit")


def _assert_float_results(actual: Any, expected: tuple[float, ...]) -> None:
    """Check every saved mock value and the original float-tuple ABI.

    Mock values are copied without numerical computation, so zero tolerances
    assert exact persistence rather than a simulation accuracy claim.

    Args:
        actual (Any): Public restored result.
        expected (tuple[float, ...]): Original mock provider values.

    Raises:
        AssertionError: If result types, order, or values differ.
    """
    assert type(actual) is tuple
    assert all(type(value) is float for value in actual)
    np.testing.assert_allclose(actual, expected, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("native_batch", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
def test_remote_batch_and_legacy_snapshots_restore_without_resubmission(
    native_batch, legacy
):
    """One batch reference and several scalar references preserve the same ABI."""
    values = (-0.5, 0.25, 0.75)
    remote = (
        [_Remote(values, ExecutionReference("test", ("task-a", "task-b")))]
        if native_batch
        else [
            _Remote(value, ExecutionReference("test", (str(index),)))
            for index, value in enumerate(values)
        ]
    )
    executor = _Executor(
        remote[0] if native_batch else CompositeExecutionHandle(remote)
    )
    retrieved = {
        handle.reference.job_ids: _Remote(handle.value, handle.reference)
        for handle in remote
    }
    executor.restore.side_effect = lambda reference: retrieved[reference.job_ids]
    executable = _program(3)
    original = executable.run(executor)
    snapshot = original.snapshot()
    if legacy:
        snapshot = JobSnapshot(JobKind.RUN, snapshot.executions)
    data = json.loads(json.dumps(snapshot.to_dict()))
    restored = executable.restore(executor, JobSnapshot.from_dict(data))

    assert executor.submit_estimates.call_count == 1
    assert executor.restore.call_count == len(remote)
    assert all(handle.result_calls == 0 for handle in remote)
    assert all(handle.result_calls == 0 for handle in retrieved.values())
    assert type(restored) is type(original)
    _assert_float_results(restored.result(), values)
    _assert_float_results(original.result(), values)
    again = executable.restore(
        executor,
        JobSnapshot.from_dict(json.loads(json.dumps(restored.snapshot().to_dict()))),
    )
    _assert_float_results(again.result(), values)
    assert executor.submit_estimates.call_count == 1


@pytest.mark.parametrize("grouped", [False, True])
def test_local_batch_needs_no_provider_restoration_support(grouped):
    """Local scalar groups and completed native tuples both restore lazily."""
    values = (0.125, -2.0, 3.5)
    handle = (
        CompositeExecutionHandle([CompletedExecutionHandle(value) for value in values])
        if grouped
        else CompletedExecutionHandle(values)
    )
    executor = _Executor(handle)
    executable = _program(3)
    original = executable.run(executor)
    snapshot = JobSnapshot.from_dict(
        json.loads(json.dumps(original.snapshot().to_dict()))
    )
    assert snapshot.executions == ()
    restored = executable.restore(executor, snapshot)
    executor.restore.assert_not_called()
    assert executor.submit_estimates.call_count == 1
    assert type(restored) is type(original)
    _assert_float_results(restored.result(), values)
    _assert_float_results(original.result(), values)


@pytest.mark.parametrize(
    "execution",
    [
        ExecutionSnapshot("local", value=(0.25,)),
        ExecutionSnapshot("local", value=(0.25, "invalid")),
        ExecutionSnapshot("local", value=[0.25, 1.0]),
        ExecutionSnapshot("composite", children=()),
        ExecutionSnapshot(
            "composite",
            children=(
                ExecutionSnapshot(
                    "remote", reference=ExecutionReference("test", ("a",))
                ),
                ExecutionSnapshot("local", value=True),
            ),
        ),
        ExecutionSnapshot(
            "composite",
            children=(
                ExecutionSnapshot(
                    "remote", reference=ExecutionReference("test", ("a",))
                ),
                ExecutionSnapshot(
                    "composite", children=(ExecutionSnapshot("local", value=0.25),)
                ),
            ),
        ),
    ],
)
def test_invalid_estimate_shape_fails_before_provider_retrieval(execution):
    """Well-formed JSON must still match the executable's scalar expectation ABI."""
    executor = _Executor(None)
    snapshot = JobSnapshot(JobKind.RUN, execution.references(), execution=execution)
    with pytest.raises(ExecutionError, match="[Ee]xpectation|real scalar"):
        _program(2).restore(executor, snapshot)
    executor.restore.assert_not_called()
    executor.submit_estimates.assert_not_called()


@pytest.mark.parametrize("value", [(0.25,), (0.25, "wrong"), ((0.25,), 0.5)])
def test_invalid_remote_batch_fails_at_result_without_partial_output(value):
    """Unknown provider arity is validated lazily when the result arrives."""
    reference = ExecutionReference("test", ("batch",))
    handle = _Remote(value, reference)
    executor = _Executor(None)
    executor.restore.return_value = handle
    snapshot = JobSnapshot(JobKind.RUN, (reference,))
    restored = _program(2).restore(executor, snapshot)
    assert handle.result_calls == 0
    with pytest.raises(ExecutionError, match="count|real scalar"):
        restored.result()
    executor.submit_estimates.assert_not_called()


@pytest.mark.parametrize("value", [10**1000, -(10**1000)], ids=["positive", "negative"])
@pytest.mark.parametrize("shape", ["scalar", "native_batch", "mixed_group"])
def test_local_expectation_overflow_fails_before_provider_retrieval(value, shape):
    """Oversized saved integers fail with an execution diagnostic before retrieval."""
    if shape == "scalar":
        execution = ExecutionSnapshot("local", value=value)
        arity = 1
    elif shape == "native_batch":
        execution = ExecutionSnapshot("local", value=(0.25, value))
        arity = 2
    else:
        execution = ExecutionSnapshot(
            "composite",
            children=(
                ExecutionSnapshot(
                    "remote", reference=ExecutionReference("test", ("first",))
                ),
                ExecutionSnapshot("local", value=value),
            ),
        )
        arity = 2
    snapshot = JobSnapshot.from_dict(
        json.loads(
            json.dumps(
                JobSnapshot(
                    JobKind.RUN, execution.references(), execution=execution
                ).to_dict()
            )
        )
    )
    executor = _Executor(None)

    with pytest.raises(ExecutionError, match="finite float range") as error:
        _program(arity).restore(executor, snapshot)

    assert isinstance(error.value.__cause__, OverflowError)
    executor.restore.assert_not_called()
    executor.submit_estimates.assert_not_called()


@pytest.mark.parametrize("value", [10**1000, -(10**1000)], ids=["positive", "negative"])
@pytest.mark.parametrize("shape", ["scalar", "native_batch", "legacy_group"])
def test_provider_expectation_overflow_is_reported_when_result_is_requested(
    value, shape
):
    """Provider integers outside the float range produce a lazy execution error."""
    if shape == "scalar":
        values = [value]
        arity = 1
    elif shape == "native_batch":
        values = [(0.25, value)]
        arity = 2
    else:
        values = [0.25, value]
        arity = 2
    handles = [
        _Remote(item, ExecutionReference("test", (str(index),)))
        for index, item in enumerate(values)
    ]
    executor = _Executor(None)
    executor.restore.side_effect = handles
    snapshot = JobSnapshot(JobKind.RUN, tuple(handle.reference for handle in handles))
    restored = _program(arity).restore(executor, snapshot)
    assert all(handle.result_calls == 0 for handle in handles)

    with pytest.raises(ExecutionError, match="representable as float") as error:
        restored.result()

    assert isinstance(error.value.__cause__, OverflowError)
    assert all(handle.result_calls == 1 for handle in handles)
    executor.submit_estimates.assert_not_called()


@pytest.mark.parametrize("error_type", [TypeError, ValueError])
def test_provider_real_conversion_failure_is_reported_when_result_is_requested(
    error_type,
):
    """A real scalar may still reject float conversion after numeric validation."""

    class UnconvertibleReal(float):
        """Keep the Real type contract while failing the conversion protocol."""

        def __float__(self):
            """Raise a standard conversion error configured by the test.

            Returns:
                float: Never returned.

            Raises:
                TypeError: If the test configures an invalid conversion type.
                ValueError: If the test configures an invalid conversion value.
            """
            raise error_type("cannot convert this real scalar")

    reference = ExecutionReference("test", ("scalar",))
    handle = _Remote(UnconvertibleReal(0.25), reference)
    executor = _Executor(None)
    executor.restore.return_value = handle
    snapshot = JobSnapshot(JobKind.RUN, (reference,))
    restored = _program(1).restore(executor, snapshot)
    assert handle.result_calls == 0

    with pytest.raises(ExecutionError, match="representable as float") as error:
        restored.result()

    assert isinstance(error.value.__cause__, error_type)
    assert handle.result_calls == 1
    executor.submit_estimates.assert_not_called()


@pytest.mark.parametrize("source", ["local", "provider"])
@pytest.mark.parametrize(
    "value",
    [
        sys.float_info.max,
        -sys.float_info.max,
        int(sys.float_info.max),
        -int(sys.float_info.max),
    ],
    ids=["positive-float", "negative-float", "positive-int", "negative-int"],
)
def test_expectation_restore_retains_finite_float_boundaries(source, value):
    """Both restoration paths accept the finite float endpoints and their integers."""
    executor = _Executor(None)
    if source == "local":
        execution = ExecutionSnapshot("local", value=value)
        snapshot = JobSnapshot(JobKind.RUN, (), execution=execution)
    else:
        reference = ExecutionReference("test", ("scalar",))
        executor.restore.return_value = _Remote(value, reference)
        snapshot = JobSnapshot(JobKind.RUN, (reference,))

    result = _program(1).restore(executor, snapshot).result()

    assert type(result) is float
    assert math.isclose(result, float(value), rel_tol=0.0, abs_tol=0.0)
    executor.submit_estimates.assert_not_called()
    if source == "local":
        executor.restore.assert_not_called()


@pytest.mark.parametrize("kind,shots", [(JobKind.RUN, None), (JobKind.SAMPLE, 4)])
@pytest.mark.parametrize("counts", [0.25, {"2": 4}, {"0": True}, {"0": -1}])
def test_invalid_local_counts_are_diagnosed_without_submitting(kind, shots, counts):
    """A local value must satisfy the raw counts ABI before public conversion."""
    execution = ExecutionSnapshot("local", value=counts)
    snapshot = JobSnapshot(kind, (), shots=shots, execution=execution)
    executor = _Executor(None)
    with pytest.raises(ExecutionError, match="Local counts snapshot"):
        _program(0).restore(executor, snapshot)
    executor.restore.assert_not_called()
    executor.submit_estimates.assert_not_called()


@pytest.mark.parametrize(
    "counts",
    [{}, {"0": 0}, {"0": 2}, {"0": 1, "1": 1}, {"0": 1, "1": 0}],
    ids=["empty", "zero-count", "multiple-shots", "multiple-outcomes", "zero-outcome"],
)
def test_local_run_requires_exactly_one_outcome_before_provider_retrieval(counts):
    """Saved single-shot runs reject empty, extra, or unobserved outcomes."""
    execution = ExecutionSnapshot("local", value=counts)
    snapshot = JobSnapshot.from_dict(
        json.loads(
            json.dumps(JobSnapshot(JobKind.RUN, (), execution=execution).to_dict())
        )
    )
    executor = _Executor(None)
    executor.submit_sample = Mock()

    with pytest.raises(ExecutionError, match="exactly one bitstring with count 1"):
        _program(0).restore(executor, snapshot)

    executor.restore.assert_not_called()
    executor.submit_sample.assert_not_called()
    executor.submit_estimates.assert_not_called()


@pytest.mark.parametrize(
    "bitstring,expected",
    [("", ()), ("0", (0,)), ("1", (1,)), ("01", (1, 0))],
    ids=["empty-bitstring", "zero", "one", "multiple-bits"],
)
def test_local_run_round_trip_preserves_single_shot_result(bitstring, expected):
    """One saved outcome restores the public value, including zero measured bits."""
    executor = _Executor(None)
    executor.submit_sample = Mock(return_value=CompletedExecutionHandle({bitstring: 1}))
    executable = _program(0)
    original = executable.run(executor)
    snapshot = JobSnapshot.from_dict(
        json.loads(json.dumps(original.snapshot().to_dict()))
    )

    restored = executable.restore(executor, snapshot)

    assert type(restored) is type(original)
    assert restored.result() == original.result() == expected
    executor.restore.assert_not_called()
    executor.submit_sample.assert_called_once()
    assert executor.submit_sample.call_args.args[0].shots == 1
    executor.submit_estimates.assert_not_called()


def test_restore_revalidates_mutated_reference_inventory():
    """The public restore boundary rejects later mutations before retrieval."""
    reference = ExecutionReference("test", ("batch",), context={"kind": "estimate"})
    execution = ExecutionSnapshot("remote", reference=reference)
    snapshot = JobSnapshot(JobKind.RUN, execution.references(), execution=execution)
    snapshot.executions[0].context["kind"] = "sample"
    executor = _Executor(None)
    with pytest.raises(ValueError, match="disagree"):
        _program(2).restore(executor, snapshot)
    executor.restore.assert_not_called()


@pytest.mark.parametrize("counts", [{}, {"0": 2}, {"0": 0, "1": 3}])
def test_local_sample_preserves_returned_counts_and_requested_shots(counts):
    """Restoration does not invent counts when an adapter reports fewer samples."""
    executor = _Executor(None)
    executor.submit_sample = Mock(return_value=CompletedExecutionHandle(counts))
    executable = _program(0)
    original = executable.sample(executor, shots=4)
    snapshot = JobSnapshot.from_dict(
        json.loads(json.dumps(original.snapshot().to_dict()))
    )
    restored = executable.restore(executor, snapshot)
    assert restored.result() == original.result()
    assert restored.result().shots == 4
    executor.restore.assert_not_called()
    assert executor.submit_sample.call_count == 1
