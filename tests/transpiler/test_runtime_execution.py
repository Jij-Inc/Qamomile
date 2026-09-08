from __future__ import annotations

import numpy as np
import pytest

import qamomile.observable as qm_o
from qamomile.circuit.ir.operation.arithmetic_operations import (
    BinOp,
    BinOpKind,
    NotOp,
)
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    IfOperation,
    RegionArg,
    WhileOperation,
)
from qamomile.circuit.ir.types.primitives import BitType, FloatType, UIntType
from qamomile.circuit.ir.value import DictValue, Value
from qamomile.circuit.transpiler.classical_executor import ClassicalExecutor
from qamomile.circuit.transpiler.compiled_segments import (
    CompiledClassicalSegment,
    CompiledExpvalSegment,
    CompiledQuantumSegment,
)
from qamomile.circuit.transpiler.errors import ExecutionError
from qamomile.circuit.transpiler.executable import ExecutableProgram
from qamomile.circuit.transpiler.execution_context import ExecutionContext
from qamomile.circuit.transpiler.execution_handle import (
    ExecutionHandle,
    ExecutionReference,
    JobStatus,
)
from qamomile.circuit.transpiler.execution_request import (
    EstimateRequest,
    Exact,
    SampleRequest,
)
from qamomile.circuit.transpiler.job import JobKind, SampleJob
from qamomile.circuit.transpiler.parameter_binding import (
    ParameterInfo,
    ParameterMetadata,
)
from qamomile.circuit.transpiler.quantum_executor import QuantumExecutor
from qamomile.circuit.transpiler.segments import (
    ClassicalSegment,
    ClassicalStep,
    ExpvalSegment,
    ExpvalStep,
    ProgramABI,
    ProgramPlan,
    QuantumSegment,
    QuantumStep,
)


def _uint_const(value: int, name: str = "const") -> Value:
    return Value(type=UIntType(), name=name).with_const(value)


def _float_const(value: float, name: str = "const") -> Value:
    return Value(type=FloatType(), name=name).with_const(value)


class _FakeExecutor(QuantumExecutor[str]):
    def __init__(
        self,
        *,
        counts: dict[str, int] | None = None,
        expval: float = 0.0,
    ) -> None:
        self._counts = counts or {"": 1}
        self._expval = expval
        self.bound_bindings: dict[str, float] | None = None

    def execute(self, circuit: str, shots: int) -> dict[str, int]:
        return self._counts

    def bind_parameters(
        self,
        circuit: str,
        bindings: dict[str, float],
        parameter_metadata: ParameterMetadata,
    ) -> str:
        self.bound_bindings = bindings
        return circuit

    def estimate(
        self,
        circuit: str,
        hamiltonian: qm_o.Hamiltonian,
        params=None,
    ) -> float:
        return self._expval


class _DeferredFloatHandle(ExecutionHandle[float]):
    """Record lazy expectation result retrieval."""

    def __init__(
        self,
        value: float,
        reference: ExecutionReference | None = None,
    ) -> None:
        """Initialize a deferred value.

        Args:
            value (float): Result returned on retrieval.
            reference (ExecutionReference | None): Optional provider reference.
        """
        self.value = value
        self.reference = reference
        self.result_calls = 0

    def result(self, timeout: float | None = None) -> float:
        """Return and record the deferred value.

        Args:
            timeout (float | None): Ignored test timeout.

        Returns:
            float: Stored value.
        """
        self.result_calls += 1
        return self.value

    def status(self) -> JobStatus:
        """Return a pending or completed state.

        Returns:
            JobStatus: State derived from result retrieval.
        """
        return JobStatus.COMPLETED if self.result_calls else JobStatus.QUEUED

    def references(self) -> tuple[ExecutionReference, ...]:
        """Return the optional provider reference.

        Returns:
            tuple[ExecutionReference, ...]: Empty or one referenced execution.
        """
        return () if self.reference is None else (self.reference,)


class _DeferredEstimateExecutor(_FakeExecutor):
    """Return a provider-like handle from expectation submission."""

    def __init__(self, value: float) -> None:
        """Initialize a deferred executor.

        Args:
            value (float): Deferred expectation value.
        """
        super().__init__()
        self.handle = _DeferredFloatHandle(value)
        self.request: EstimateRequest[str] | None = None

    def submit_estimate(
        self,
        request: EstimateRequest[str],
    ) -> ExecutionHandle[float]:
        """Record a request and return without retrieving its result.

        Args:
            request (EstimateRequest[str]): Submitted expectation request.

        Returns:
            ExecutionHandle[float]: Deferred test handle.
        """
        self.request = request
        return self.handle


class _RestoringEstimateExecutor(_DeferredEstimateExecutor):
    """Restore one referenced expectation handle."""

    def __init__(self, value: float) -> None:
        """Initialize a restorable expectation executor.

        Args:
            value (float): Deferred expectation value.
        """
        super().__init__(value)
        self.reference = ExecutionReference("test", ("estimate-1",))
        self.handle = _DeferredFloatHandle(value, self.reference)

    def restore(
        self,
        reference: ExecutionReference,
    ) -> ExecutionHandle[float]:
        """Restore the referenced expectation handle.

        Args:
            reference (ExecutionReference): Reference to validate.

        Returns:
            ExecutionHandle[float]: Restored deferred expectation handle.
        """
        assert reference == self.reference
        return _DeferredFloatHandle(self.handle.value, reference)


class _RestorableCountsHandle(ExecutionHandle[dict[str, int]]):
    """Expose restorable raw counts for typed restoration tests."""

    def __init__(
        self,
        counts: dict[str, int],
        reference: ExecutionReference,
    ) -> None:
        """Initialize a referenced counts handle.

        Args:
            counts (dict[str, int]): Counts returned on retrieval.
            reference (ExecutionReference): Stable provider reference.
        """
        self._counts = counts
        self._reference = reference

    def result(self, timeout: float | None = None) -> dict[str, int]:
        """Return stored counts.

        Args:
            timeout (float | None): Ignored test timeout.

        Returns:
            dict[str, int]: Stored raw counts.
        """
        return dict(self._counts)

    def status(self) -> JobStatus:
        """Return a completed test status.

        Returns:
            JobStatus: Always completed.
        """
        return JobStatus.COMPLETED

    def references(self) -> tuple[ExecutionReference, ...]:
        """Return the stable provider reference.

        Returns:
            tuple[ExecutionReference, ...]: One test reference.
        """
        return (self._reference,)


class _RestoringExecutor(_FakeExecutor):
    """Submit and restore referenced counts handles."""

    def __init__(self, counts: dict[str, int]) -> None:
        """Initialize original and restored test handles.

        Args:
            counts (dict[str, int]): Counts returned by both handles.
        """
        super().__init__(counts=counts)
        self.reference = ExecutionReference("test", ("job-1",))

    def submit_sample(
        self,
        request: SampleRequest[str],
    ) -> ExecutionHandle[dict[str, int]]:
        """Return a referenced sampling handle.

        Args:
            request (SampleRequest[str]): Sampling request ignored by the test
                executor.

        Returns:
            ExecutionHandle[dict[str, int]]: Original referenced handle.
        """
        return _RestorableCountsHandle(self._counts, self.reference)

    def restore(
        self,
        reference: ExecutionReference,
    ) -> ExecutionHandle[dict[str, int]]:
        """Recreate the counts handle from its reference.

        Args:
            reference (ExecutionReference): Reference to validate.

        Returns:
            ExecutionHandle[dict[str, int]]: Restored counts handle.
        """
        assert reference == self.reference
        return _RestorableCountsHandle(self._counts, reference)


class TestClassicalExecutorControlFlow:
    def test_executes_if_with_merge(self) -> None:
        cond = Value(type=BitType(), name="cond")
        true_result = Value(type=UIntType(), name="true_result")
        false_result = Value(type=UIntType(), name="false_result")
        merged = Value(type=UIntType(), name="merged")

        if_op = IfOperation(
            operands=[cond],
            true_operations=[
                BinOp(
                    kind=BinOpKind.ADD,
                    operands=[_uint_const(1, "one"), _uint_const(1, "one")],
                    results=[true_result],
                )
            ],
            false_operations=[
                BinOp(
                    kind=BinOpKind.ADD,
                    operands=[_uint_const(1, "one"), _uint_const(2, "two")],
                    results=[false_result],
                )
            ],
        )
        if_op.add_merge(true_result, false_result, merged)

        context = ExecutionContext({cond.uuid: True})
        results = ClassicalExecutor().execute(
            ClassicalSegment(operations=[if_op]),
            context,
        )

        assert results[merged.uuid] == 2

    def test_executes_for_loop(self) -> None:
        loop_var = Value(type=UIntType(), name="i")
        loop_out = Value(type=UIntType(), name="loop_out")
        for_op = ForOperation(
            loop_var="i",
            loop_var_value=loop_var,
            operands=[_uint_const(0), _uint_const(3), _uint_const(1)],
            operations=[
                BinOp(
                    kind=BinOpKind.ADD,
                    operands=[loop_var, _uint_const(1)],
                    results=[loop_out],
                )
            ],
        )

        results = ClassicalExecutor().execute(
            ClassicalSegment(operations=[for_op]),
            ExecutionContext(),
        )

        assert results[loop_out.uuid] == 3

    def test_executes_for_items_loop(self) -> None:
        coeff = Value(type=FloatType(), name="coeff")
        out = Value(type=FloatType(), name="out")
        iterable = DictValue(
            name="weights",
        ).with_dict_runtime_metadata({0: 1.5, 2: 2.5})
        for_items = ForItemsOperation(
            key_vars=["i"],
            key_var_values=(Value(type=UIntType(), name="i"),),
            value_var="coeff",
            value_var_value=coeff,
            operands=[iterable],
            operations=[
                BinOp(
                    kind=BinOpKind.ADD,
                    operands=[coeff, _float_const(1.0)],
                    results=[out],
                )
            ],
        )

        results = ClassicalExecutor().execute(
            ClassicalSegment(operations=[for_items]),
            ExecutionContext(),
        )

        assert results[out.uuid] == pytest.approx(3.5)

    def test_empty_for_items_publishes_region_arg_initializer(self) -> None:
        """An explicitly bound empty dict takes the zero-trip carry path."""
        key = Value(type=UIntType(), name="key")
        value = Value(type=FloatType(), name="value")
        init = _uint_const(7, "init")
        block_arg = Value(type=UIntType(), name="carry")
        result = Value(type=UIntType(), name="carry_result")
        operation = ForItemsOperation(
            key_vars=["key"],
            key_var_values=(key,),
            value_var="value",
            value_var_value=value,
            operands=[DictValue(name="empty").with_dict_runtime_metadata({})],
            operations=[],
            region_args=(
                RegionArg(
                    var_name="carry",
                    init=init,
                    block_arg=block_arg,
                    yielded=block_arg,
                    result=result,
                ),
            ),
            results=[result],
        )

        results = ClassicalExecutor().execute(
            ClassicalSegment(operations=[operation]),
            ExecutionContext(),
        )

        assert results[result.uuid] == 7

    def test_executes_while_loop(self) -> None:
        cond_in = Value(type=BitType(), name="cond")
        cond_out = Value(type=BitType(), name="cond_next")
        while_op = WhileOperation(
            operands=[cond_in, cond_out],
            operations=[NotOp(operands=[cond_in], results=[cond_out])],
        )

        results = ClassicalExecutor().execute(
            ClassicalSegment(operations=[while_op]),
            ExecutionContext({cond_in.uuid: True}),
        )

        assert results[cond_out.uuid] is False

    def test_executes_nested_control_flow(self) -> None:
        cond = Value(type=BitType(), name="cond")
        loop_var = Value(type=UIntType(), name="i")
        out = Value(type=UIntType(), name="out")
        nested = IfOperation(
            operands=[cond],
            true_operations=[
                ForOperation(
                    loop_var="i",
                    loop_var_value=loop_var,
                    operands=[_uint_const(0), _uint_const(2), _uint_const(1)],
                    operations=[
                        BinOp(
                            kind=BinOpKind.ADD,
                            operands=[loop_var, _uint_const(10)],
                            results=[out],
                        )
                    ],
                )
            ],
            false_operations=[],
        )

        results = ClassicalExecutor().execute(
            ClassicalSegment(operations=[nested]),
            ExecutionContext({cond.uuid: True}),
        )

        assert results[out.uuid] == 11


class TestExecutableProgramRuntime:
    def test_sample_snapshot_restores_typed_public_result(self) -> None:
        """Restoration reapplies the executable program's result conversion."""
        quantum_segment = QuantumSegment()
        executable = ExecutableProgram[str](
            compiled_quantum=[
                CompiledQuantumSegment(
                    segment=quantum_segment,
                    circuit="quantum",
                    implicit_output_qubit_indices=(0,),
                )
            ]
        )
        executor = _RestoringExecutor({"0": 2, "1": 3})

        snapshot = executable.sample(executor, shots=5).snapshot()
        restored = executable.restore(executor, snapshot)

        assert snapshot.kind is JobKind.SAMPLE
        assert snapshot.shots == 5
        assert restored.result().results == [((0,), 2), ((1,), 3)]

    def test_run_snapshot_restores_typed_public_result(self) -> None:
        """A restored one-shot run returns the kernel-level output type."""
        quantum_segment = QuantumSegment()
        executable = ExecutableProgram[str](
            compiled_quantum=[
                CompiledQuantumSegment(
                    segment=quantum_segment,
                    circuit="quantum",
                    implicit_output_qubit_indices=(0,),
                )
            ]
        )
        executor = _RestoringExecutor({"1": 1})

        snapshot = executable.run(executor).snapshot()
        restored = executable.restore(executor, snapshot)

        assert snapshot.kind is JobKind.RUN
        assert snapshot.shots is None
        assert restored.result() == (1,)

    def test_sample_projects_implicit_outputs_and_aggregates_hidden_bits(self) -> None:
        """Internal ancilla states do not leak into implicit sample outputs."""
        quantum_segment = QuantumSegment()
        executable = ExecutableProgram[str](
            compiled_quantum=[
                CompiledQuantumSegment(
                    segment=quantum_segment,
                    circuit="quantum",
                    implicit_output_qubit_indices=(0, 1),
                )
            ]
        )

        result = executable.sample(
            _FakeExecutor(counts={"000": 2, "100": 3}),
            shots=5,
        ).result()

        assert result.results == [((0, 0), 5)]

    def test_run_projects_implicit_outputs_in_declared_order(self) -> None:
        """Implicit run outputs follow declared physical-index order."""
        quantum_segment = QuantumSegment()
        executable = ExecutableProgram[str](
            compiled_quantum=[
                CompiledQuantumSegment(
                    segment=quantum_segment,
                    circuit="quantum",
                    implicit_output_qubit_indices=(2, 0),
                )
            ]
        )

        result = executable.run(_FakeExecutor(counts={"101": 1})).result()

        assert result == (1, 1)

    def test_empty_implicit_output_mapping_hides_every_physical_qubit(self) -> None:
        """An explicit empty map distinguishes zero logical qubits from unknown."""
        quantum_segment = QuantumSegment()
        executable = ExecutableProgram[str](
            compiled_quantum=[
                CompiledQuantumSegment(
                    segment=quantum_segment,
                    circuit="quantum",
                    implicit_output_qubit_indices=(),
                )
            ]
        )

        result = executable.run(_FakeExecutor(counts={"0": 1})).result()

        assert result == ()

    @pytest.mark.parametrize("invalid_index", [-1, 2])
    def test_invalid_implicit_output_index_is_rejected(
        self,
        invalid_index: int,
    ) -> None:
        """Malformed implicit-output metadata fails instead of exposing a bit."""
        quantum_segment = QuantumSegment()
        executable = ExecutableProgram[str](
            compiled_quantum=[
                CompiledQuantumSegment(
                    segment=quantum_segment,
                    circuit="quantum",
                    implicit_output_qubit_indices=(invalid_index,),
                )
            ]
        )

        with pytest.raises(ExecutionError, match="outside the backend bitstring"):
            executable.run(_FakeExecutor(counts={"0": 1})).result()

    def test_sample_rejects_unresolved_typed_output(self) -> None:
        """Typed output provenance gaps fail instead of sampling ``None``."""
        output = Value(type=UIntType(), name="missing_output")
        quantum_segment = QuantumSegment()
        executable = ExecutableProgram[str](
            plan=ProgramPlan(
                steps=[QuantumStep(segment=quantum_segment)],
                abi=ProgramABI(output_values=[output]),
            ),
            compiled_quantum=[
                CompiledQuantumSegment(
                    segment=quantum_segment,
                    circuit="quantum",
                    parameter_metadata=ParameterMetadata(),
                )
            ],
            output_values=[output],
        )

        with pytest.raises(ExecutionError, match="Typed output 'missing_output'"):
            executable.sample(_FakeExecutor(counts={"": 2}), shots=2).result()

    def test_sample_job_aggregates_duplicate_projected_outputs(self) -> None:
        """Raw states collapsing to one public value have their counts summed."""
        job = SampleJob(
            {"000": 2, "100": 3},
            lambda counts: [((0,), count) for count in counts.values()],
            shots=5,
        )

        assert job.result().results == [((0,), 5)]

    def test_sample_job_keeps_bool_and_int_outputs_distinct(self) -> None:
        """Aggregation preserves scalar type instead of using Python equality."""
        job = SampleJob(
            {"0": 2, "1": 3},
            lambda counts: [(True, counts["0"]), (1, counts["1"])],
            shots=5,
        )

        assert job.result().results == [(True, 2), (1, 3)]

    def test_sample_job_aggregates_equal_numpy_outputs(self) -> None:
        """Independent arrays with equal dtype, shape, and values aggregate."""
        first = np.array([1, 2], dtype=np.int32)
        second = np.array([1, 2], dtype=np.int32)
        job = SampleJob(
            {"0": 2, "1": 3},
            lambda counts: [(first, counts["0"]), (second, counts["1"])],
            shots=5,
        )

        results = job.result().results
        assert len(results) == 1
        assert results[0][0] is first
        assert results[0][1] == 5

    def test_sample_job_preserves_numpy_dtype_and_shape(self) -> None:
        """Aggregation does not collapse arrays with different public types."""
        values = (
            np.array([1, 2], dtype=np.int32),
            np.array([1, 2], dtype=np.int64),
            np.array([[1, 2]], dtype=np.int32),
        )
        job = SampleJob(
            {"00": 1, "01": 1, "10": 1},
            lambda counts: [
                (value, count)
                for value, count in zip(values, counts.values(), strict=True)
            ],
            shots=3,
        )

        results = job.result().results
        assert [count for _, count in results] == [1, 1, 1]
        assert [value.dtype for value, _ in results] == [
            np.dtype(np.int32),
            np.dtype(np.int64),
            np.dtype(np.int32),
        ]
        assert [value.shape for value, _ in results] == [(2,), (2,), (1, 2)]

    def test_run_executes_expval_before_classical_post(self) -> None:
        exp_result = Value(type=FloatType(), name="exp_result")
        output = Value(type=FloatType(), name="output")
        quantum_segment = QuantumSegment()
        classical_segment = ClassicalSegment(
            operations=[
                BinOp(
                    kind=BinOpKind.ADD,
                    operands=[exp_result, _float_const(1.0)],
                    results=[output],
                )
            ]
        )
        expval_segment = qm_o.Hamiltonian()
        executable = ExecutableProgram[str](
            plan=ProgramPlan(
                steps=[
                    QuantumStep(segment=quantum_segment),
                    ExpvalStep(
                        segment=(
                            exp_segment := ExpvalSegment(
                                hamiltonian_value=None,
                                qubits_value=None,
                                result_ref=exp_result.uuid,
                            )
                        )
                    ),
                    ClassicalStep(segment=classical_segment, role="post"),
                ],
                abi=ProgramABI(output_values=[output]),
            ),
            compiled_quantum=[
                CompiledQuantumSegment(
                    segment=quantum_segment,
                    circuit="quantum",
                    parameter_metadata=ParameterMetadata(),
                )
            ],
            compiled_classical=[CompiledClassicalSegment(segment=classical_segment)],
            compiled_expval=[
                CompiledExpvalSegment(
                    segment=exp_segment,
                    hamiltonian=expval_segment,
                    result_ref=exp_result.uuid,
                )
            ],
            output_values=[output],
        )

        job = executable.run(_FakeExecutor(expval=0.25))
        assert job.result() == pytest.approx(1.25)

    def test_run_defers_expval_result_and_propagates_accuracy(self) -> None:
        """Orchestration submits first and performs host work on retrieval."""
        exp_result = Value(type=FloatType(), name="exp_result")
        output = Value(type=FloatType(), name="output")
        quantum_segment = QuantumSegment()
        classical_segment = ClassicalSegment(
            operations=[
                BinOp(
                    kind=BinOpKind.ADD,
                    operands=[exp_result, _float_const(1.0)],
                    results=[output],
                )
            ]
        )
        exp_segment = ExpvalSegment(
            hamiltonian_value=None,
            qubits_value=None,
            result_ref=exp_result.uuid,
        )
        executable = ExecutableProgram[str](
            plan=ProgramPlan(
                steps=[
                    QuantumStep(segment=quantum_segment),
                    ExpvalStep(segment=exp_segment),
                    ClassicalStep(segment=classical_segment, role="post"),
                ],
                abi=ProgramABI(output_values=[output]),
            ),
            compiled_quantum=[
                CompiledQuantumSegment(
                    segment=quantum_segment,
                    circuit="quantum",
                    parameter_metadata=ParameterMetadata(),
                )
            ],
            compiled_classical=[CompiledClassicalSegment(segment=classical_segment)],
            compiled_expval=[
                CompiledExpvalSegment(
                    segment=exp_segment,
                    hamiltonian=qm_o.Hamiltonian(),
                    result_ref=exp_result.uuid,
                )
            ],
            output_values=[output],
        )
        executor = _DeferredEstimateExecutor(0.25)

        job = executable.run(executor, estimation=Exact())

        assert executor.handle.result_calls == 0
        assert job.status() is JobStatus.QUEUED
        assert executor.request is not None
        assert isinstance(executor.request.accuracy, Exact)
        assert job.result() == pytest.approx(1.25)
        assert executor.handle.result_calls == 1

    def test_expval_snapshot_restores_float_job(self) -> None:
        """A restored pure expectation execution remains an ExpvalJob."""
        quantum_segment = QuantumSegment()
        exp_segment = ExpvalSegment(
            hamiltonian_value=None,
            qubits_value=None,
            result_ref="expval",
        )
        executable = ExecutableProgram[str](
            plan=ProgramPlan(
                steps=[
                    QuantumStep(segment=quantum_segment),
                    ExpvalStep(segment=exp_segment),
                ],
            ),
            compiled_quantum=[
                CompiledQuantumSegment(
                    segment=quantum_segment,
                    circuit="quantum",
                    parameter_metadata=ParameterMetadata(),
                )
            ],
            compiled_expval=[
                CompiledExpvalSegment(
                    segment=exp_segment,
                    hamiltonian=qm_o.Hamiltonian(),
                    result_ref="expval",
                )
            ],
        )
        executor = _RestoringEstimateExecutor(0.375)

        snapshot = executable.run(executor).snapshot()
        restored = executable.restore(executor, snapshot)

        assert restored.result() == pytest.approx(0.375)

    def test_sample_rejects_expval_programs(self) -> None:
        quantum_segment = QuantumSegment()
        exp_segment = ExpvalSegment(
            hamiltonian_value=None,
            qubits_value=None,
            result_ref="expval",
        )
        executable = ExecutableProgram[str](
            plan=ProgramPlan(
                steps=[
                    QuantumStep(segment=quantum_segment),
                    ExpvalStep(segment=exp_segment),
                ],
            ),
            compiled_quantum=[
                CompiledQuantumSegment(
                    segment=quantum_segment,
                    circuit="quantum",
                    parameter_metadata=ParameterMetadata(),
                )
            ],
            compiled_expval=[
                CompiledExpvalSegment(
                    segment=exp_segment,
                    hamiltonian=qm_o.Hamiltonian(),
                    result_ref="expval",
                )
            ],
        )

        with pytest.raises(ExecutionError, match="sample\\(\\) does not support"):
            executable.sample(_FakeExecutor())

    def test_sample_executes_classical_prep_with_runtime_bindings(self) -> None:
        theta = Value(type=FloatType(), name="theta").with_parameter("theta")
        output = Value(type=FloatType(), name="output")
        prep_segment = ClassicalSegment(
            operations=[
                BinOp(
                    kind=BinOpKind.ADD,
                    operands=[theta, _float_const(1.0)],
                    results=[output],
                )
            ]
        )
        quantum_segment = QuantumSegment()
        executable = ExecutableProgram[str](
            plan=ProgramPlan(
                steps=[
                    ClassicalStep(segment=prep_segment, role="prep"),
                    QuantumStep(segment=quantum_segment),
                ],
                abi=ProgramABI(
                    public_inputs={"theta": theta},
                    output_values=[output],
                ),
            ),
            compiled_quantum=[
                CompiledQuantumSegment(
                    segment=quantum_segment,
                    circuit="quantum",
                    parameter_metadata=ParameterMetadata(),
                )
            ],
            compiled_classical=[CompiledClassicalSegment(segment=prep_segment)],
            output_values=[output],
        )

        result = executable.sample(
            _FakeExecutor(counts={"": 2}),
            shots=2,
            bindings={"theta": 2.0},
        ).result()

        assert result.results == [(3.0, 2)]

    def test_sample_binds_quantum_parameters_from_classical_prep(self) -> None:
        theta = Value(type=FloatType(), name="theta").with_parameter("theta")
        theta2 = Value(type=FloatType(), name="theta2")
        prep_segment = ClassicalSegment(
            operations=[
                BinOp(
                    kind=BinOpKind.ADD,
                    operands=[theta, _float_const(1.0)],
                    results=[theta2],
                )
            ]
        )
        quantum_segment = QuantumSegment()
        executable = ExecutableProgram[str](
            plan=ProgramPlan(
                steps=[
                    ClassicalStep(segment=prep_segment, role="prep"),
                    QuantumStep(segment=quantum_segment),
                ],
                abi=ProgramABI(public_inputs={"theta": theta}),
            ),
            compiled_quantum=[
                CompiledQuantumSegment(
                    segment=quantum_segment,
                    circuit="quantum",
                    parameter_metadata=ParameterMetadata(
                        parameters=[
                            ParameterInfo(
                                name="theta2",
                                array_name="theta2",
                                index=None,
                                backend_param="theta2_backend",
                                source_ref=theta2.uuid,
                            )
                        ]
                    ),
                )
            ],
            compiled_classical=[CompiledClassicalSegment(segment=prep_segment)],
        )

        executor = _FakeExecutor(counts={"": 1})
        executable.sample(executor, shots=1, bindings={"theta": 2.0}).result()

        assert executor.bound_bindings == {"theta2": pytest.approx(3.0)}
