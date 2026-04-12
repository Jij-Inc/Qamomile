from __future__ import annotations

import pytest

import qamomile.observable as qm_o
from qamomile.circuit.ir.operation.arithmetic_operations import (
    BinOp,
    BinOpKind,
    NotOp,
    PhiOp,
)
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    IfOperation,
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


class TestClassicalExecutorControlFlow:
    def test_executes_if_with_phi_merge(self) -> None:
        cond = Value(type=BitType(), name="cond")
        true_result = Value(type=UIntType(), name="true_result")
        false_result = Value(type=UIntType(), name="false_result")
        merged = Value(type=UIntType(), name="merged")

        if_op = IfOperation(
            operands=[cond],
            results=[merged],
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
            phi_ops=[
                PhiOp(
                    operands=[cond, true_result, false_result],
                    results=[merged],
                )
            ],
        )

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
            value_var="coeff",
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
            phi_ops=[],
        )

        results = ClassicalExecutor().execute(
            ClassicalSegment(operations=[nested]),
            ExecutionContext({cond.uuid: True}),
        )

        assert results[out.uuid] == 11


class TestExecutableProgramRuntime:
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
                abi=ProgramABI(output_refs=[output.uuid]),
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
            output_refs=[output.uuid],
        )

        job = executable.run(_FakeExecutor(expval=0.25))
        assert job.result() == pytest.approx(1.25)

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
                    output_refs=[output.uuid],
                ),
            ),
            compiled_quantum=[
                CompiledQuantumSegment(
                    segment=quantum_segment,
                    circuit="quantum",
                    parameter_metadata=ParameterMetadata(),
                )
            ],
            compiled_classical=[
                CompiledClassicalSegment(segment=prep_segment)
            ],
            output_refs=[output.uuid],
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
