"""Regression tests for equality comparisons on Bit handles.

Bit values represent classical predicates in qkernel code. Equality between
two Bit handles must therefore emit a CompOp instead of using dataclass field
comparison and collapsing to a Python Boolean during tracing. These tests
cover IR emission, compile-time folding, and measurement-derived runtime
conditions.
"""

from __future__ import annotations

import operator
from collections.abc import Callable

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.frontend.tracer import Tracer, trace
from qamomile.circuit.ir.operation.arithmetic_operations import (
    CompOp,
    CompOpKind,
)
from qamomile.circuit.ir.types.primitives import BitType
from qamomile.circuit.ir.value import Value

Comparison = Callable[[object, object], object]

_COMPARISON_CASES = [
    pytest.param(operator.eq, CompOpKind.EQ, id="eq"),
    pytest.param(operator.ne, CompOpKind.NEQ, id="ne"),
]


def _make_bit_handle(name: str) -> qmc.Bit:
    """Build a non-constant Bit handle for tracing.

    Args:
        name (str): Name assigned to the wrapped IR value.

    Returns:
        qmc.Bit: A non-constant bit handle.
    """
    return qmc.Bit(value=Value(type=BitType(), name=name))


@qmc.qkernel
def _bound_bit_comparison(
    a: qmc.Bit,
    b: qmc.Bit,
) -> qmc.Vector[qmc.Bit]:
    """Expose equality and inequality as compile-time branch conditions.

    Args:
        a (qmc.Bit): Left comparison operand.
        b (qmc.Bit): Right comparison operand.

    Returns:
        qmc.Vector[qmc.Bit]: Measurements whose qubits identify the selected
            equality and inequality branches.
    """
    q = qmc.qubit_array(2, name="q")
    if a == b:
        q[0] = qmc.x(q[0])
    if a != b:
        q[1] = qmc.x(q[1])
    return qmc.measure(q)


@qmc.qkernel
def _runtime_bit_eq(second_one: qmc.Bit) -> qmc.Bit:
    """Compare two measurement-derived bits for equality.

    Args:
        second_one (qmc.Bit): Compile-time flag selecting the second measured
            bit's basis state.

    Returns:
        qmc.Bit: One when the two measurement results are equal.
    """
    q0 = qmc.qubit("q0")
    q1 = qmc.qubit("q1")
    output = qmc.qubit("output")
    q0 = qmc.x(q0)
    if second_one:
        q1 = qmc.x(q1)
    a = qmc.measure(q0)
    b = qmc.measure(q1)
    if a == b:
        output = qmc.x(output)
    return qmc.measure(output)


@qmc.qkernel
def _runtime_bit_ne(second_one: qmc.Bit) -> qmc.Bit:
    """Compare two measurement-derived bits for inequality.

    Args:
        second_one (qmc.Bit): Compile-time flag selecting the second measured
            bit's basis state.

    Returns:
        qmc.Bit: One when the two measurement results are different.
    """
    q0 = qmc.qubit("q0")
    q1 = qmc.qubit("q1")
    output = qmc.qubit("output")
    q0 = qmc.x(q0)
    if second_one:
        q1 = qmc.x(q1)
    a = qmc.measure(q0)
    b = qmc.measure(q1)
    if a != b:
        output = qmc.x(output)
    return qmc.measure(output)


class TestBitComparisonEmitsIR:
    """Verify Bit equality and inequality emit comparison IR."""

    @pytest.mark.parametrize("comparison,kind", _COMPARISON_CASES)
    def test_bit_pair_emits_compop(
        self,
        comparison: Comparison,
        kind: CompOpKind,
    ) -> None:
        """Verify a pair of Bit handles emits the requested comparison kind."""
        tracer = Tracer()
        left = _make_bit_handle("left")
        right = _make_bit_handle("right")

        with trace(tracer):
            result = comparison(left, right)

        assert len(tracer.operations) == 1
        emitted = tracer.operations[0]
        assert isinstance(emitted, CompOp)
        assert emitted.kind is kind
        assert emitted.lhs is left.value
        assert emitted.rhs is right.value
        assert isinstance(emitted.output.type, BitType)
        assert isinstance(result, qmc.Bit)
        assert not isinstance(result, bool)

    @pytest.mark.parametrize("comparison,kind", _COMPARISON_CASES)
    @pytest.mark.parametrize(
        "constant,expected",
        [
            pytest.param(False, False, id="false"),
            pytest.param(True, True, id="true"),
            pytest.param(0, False, id="zero"),
            pytest.param(1, True, id="one"),
        ],
    )
    def test_python_constant_is_normalized(
        self,
        comparison: Comparison,
        kind: CompOpKind,
        constant: bool | int,
        expected: bool,
    ) -> None:
        """Verify Boolean and zero-or-one operands become constant Bit values."""
        tracer = Tracer()
        left = _make_bit_handle("left")

        with trace(tracer):
            result = comparison(left, constant)

        assert len(tracer.operations) == 1
        emitted = tracer.operations[0]
        assert isinstance(emitted, CompOp)
        assert emitted.kind is kind
        assert emitted.lhs is left.value
        assert isinstance(emitted.rhs.type, BitType)
        assert emitted.rhs.get_const() is expected
        assert isinstance(result, qmc.Bit)

    @pytest.mark.parametrize("comparison,kind", _COMPARISON_CASES)
    def test_reflected_boolean_is_normalized(
        self,
        comparison: Comparison,
        kind: CompOpKind,
    ) -> None:
        """Verify a Boolean on the left delegates to the Bit comparison."""
        tracer = Tracer()
        right = _make_bit_handle("right")

        with trace(tracer):
            result = comparison(True, right)

        assert len(tracer.operations) == 1
        emitted = tracer.operations[0]
        assert isinstance(emitted, CompOp)
        assert emitted.kind is kind
        assert emitted.lhs is right.value
        assert emitted.rhs.get_const() is True
        assert isinstance(result, qmc.Bit)

    def test_unsupported_operands_follow_python_protocol(self) -> None:
        """Verify unsupported equality operands use Python's fallback result."""
        bit = _make_bit_handle("bit")

        assert bit.__eq__("not a bit") is NotImplemented
        assert bit.__ne__(2) is NotImplemented
        assert (bit == "not a bit") is False
        assert (bit != 2) is True

    def test_bit_remains_unhashable(self) -> None:
        """Verify adding DSL equality does not expand Bit's container API."""
        bit = _make_bit_handle("bit")

        with pytest.raises(TypeError, match="unhashable"):
            hash(bit)


class TestBitComparisonCompileTimeFold:
    """Verify bound Bit comparisons select the correct branch."""

    @pytest.mark.parametrize(
        "a,b,expected_qubit",
        [
            pytest.param(True, True, 0, id="true-equals-true"),
            pytest.param(True, False, 1, id="true-differs-false"),
            pytest.param(False, True, 1, id="false-differs-true"),
            pytest.param(False, False, 0, id="false-equals-false"),
        ],
    )
    def test_eq_ne_fold(
        self,
        qiskit_transpiler,
        a: bool,
        b: bool,
        expected_qubit: int,
    ) -> None:
        """Verify exactly the equality or inequality branch survives."""
        executable = qiskit_transpiler.transpile(
            _bound_bit_comparison,
            bindings={"a": a, "b": b},
        )
        circuit = executable.compiled_quantum[0].circuit
        x_qubits = [
            circuit.find_bit(instruction.qubits[0]).index
            for instruction in circuit.data
            if instruction.operation.name == "x"
        ]

        assert x_qubits == [expected_qubit]
        assert all(
            instruction.operation.name != "if_else" for instruction in circuit.data
        )


class TestBitComparisonRuntime:
    """Verify measurement-derived comparisons execute as runtime predicates."""

    @pytest.mark.parametrize(
        "second_one,expected",
        [
            pytest.param(False, 0, id="different"),
            pytest.param(True, 1, id="equal"),
        ],
    )
    def test_runtime_eq_executes(
        self,
        sdk_transpiler,
        second_one: bool,
        expected: int,
    ) -> None:
        """Verify supported dynamic backends execute runtime equality."""
        if sdk_transpiler.backend_name == "quri_parts":
            pytest.skip("QuriParts does not support measurement-dependent control flow")
        transpiler = sdk_transpiler.transpiler
        executable = transpiler.transpile(
            _runtime_bit_eq,
            bindings={"second_one": second_one},
        )
        result = executable.sample(transpiler.executor(), shots=100).result()

        assert result.results == [(expected, 100)]

    @pytest.mark.parametrize(
        "second_one,expected",
        [
            pytest.param(False, 1, id="different"),
            pytest.param(True, 0, id="equal"),
        ],
    )
    def test_runtime_ne_executes(
        self,
        sdk_transpiler,
        second_one: bool,
        expected: int,
    ) -> None:
        """Verify supported dynamic backends execute runtime inequality."""
        if sdk_transpiler.backend_name == "quri_parts":
            pytest.skip("QuriParts does not support measurement-dependent control flow")
        transpiler = sdk_transpiler.transpiler
        executable = transpiler.transpile(
            _runtime_bit_ne,
            bindings={"second_one": second_one},
        )
        result = executable.sample(transpiler.executor(), shots=100).result()

        assert result.results == [(expected, 100)]

    def test_runtime_eq_emits_classical_expr(self, qiskit_transpiler) -> None:
        """Verify Qiskit receives an expression rather than a folded Boolean."""
        from qiskit.circuit.classical import expr
        from qiskit.circuit.controlflow import IfElseOp

        executable = qiskit_transpiler.transpile(
            _runtime_bit_eq,
            bindings={"second_one": False},
        )
        circuit = executable.compiled_quantum[0].circuit
        if_operations = [
            instruction.operation
            for instruction in circuit.data
            if isinstance(instruction.operation, IfElseOp)
        ]

        assert len(if_operations) == 1
        assert isinstance(if_operations[0].condition, expr.Expr)
