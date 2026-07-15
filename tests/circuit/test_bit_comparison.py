"""Regression tests for equality comparisons on Bit handles.

Bit values represent classical predicates in qkernel code. Equality between
two Bit handles must therefore emit a CompOp instead of using dataclass field
comparison and collapsing to a Python Boolean during tracing. These tests
also cover mixed Bit and UInt equality, including the abstract circuit IR,
compile-time folding, and measurement-derived runtime conditions.
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
from qamomile.circuit.ir.types.primitives import BitType, UIntType
from qamomile.circuit.ir.value import Value
from qamomile.circuit.transpiler.circuit_ir import (
    BinaryExpr,
    BinaryOperator,
    ClassicalBitExpr,
    IfInstruction,
    LiteralExpr,
    lower_circuit_plan,
)

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


def _make_uint_handle(name: str) -> qmc.UInt:
    """Build a non-constant UInt handle for tracing.

    Args:
        name (str): Name assigned to the wrapped IR value.

    Returns:
        qmc.UInt: A non-constant unsigned-integer handle.
    """
    return qmc.UInt(value=Value(type=UIntType(), name=name))


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
def _bound_bit_uint_comparison(
    bit: qmc.Bit,
    integer: qmc.UInt,
) -> qmc.Vector[qmc.Bit]:
    """Expose mixed Bit and UInt equality as compile-time conditions.

    Args:
        bit (qmc.Bit): Boolean comparison operand.
        integer (qmc.UInt): Unsigned-integer comparison operand.

    Returns:
        qmc.Vector[qmc.Bit]: Measurements identifying the selected equality
            and inequality branches.
    """
    q = qmc.qubit_array(2, name="q")
    if bit == integer:
        q[0] = qmc.x(q[0])
    if integer != bit:
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


@qmc.qkernel
def _runtime_bit_uint_eq(source_one: qmc.Bit, value: qmc.UInt) -> qmc.Bit:
    """Compare a measured Bit with a bound UInt for equality.

    Args:
        source_one (qmc.Bit): Compile-time flag selecting the measured value.
        value (qmc.UInt): Compile-time unsigned-integer comparison operand.

    Returns:
        qmc.Bit: One when the measured value equals ``value``.
    """
    source = qmc.qubit("source")
    if source_one:
        source = qmc.x(source)
    measured = qmc.measure(source)
    output = qmc.qubit("output")
    if measured == value:
        output = qmc.x(output)
    return qmc.measure(output)


@qmc.qkernel
def _runtime_uint_bit_ne(source_one: qmc.Bit, value: qmc.UInt) -> qmc.Bit:
    """Compare a bound UInt with a measured Bit for inequality.

    Args:
        source_one (qmc.Bit): Compile-time flag selecting the measured value.
        value (qmc.UInt): Compile-time unsigned-integer comparison operand.

    Returns:
        qmc.Bit: One when ``value`` differs from the measured value.
    """
    source = qmc.qubit("source")
    if source_one:
        source = qmc.x(source)
    measured = qmc.measure(source)
    output = qmc.qubit("output")
    if value != measured:
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

    @pytest.mark.parametrize("comparison,kind", _COMPARISON_CASES)
    @pytest.mark.parametrize("bit_first", [True, False], ids=["bit-uint", "uint-bit"])
    def test_bit_uint_pair_emits_compop(
        self,
        comparison: Comparison,
        kind: CompOpKind,
        bit_first: bool,
    ) -> None:
        """Verify mixed Bit and UInt handles emit comparison IR both ways."""
        tracer = Tracer()
        bit = _make_bit_handle("bit")
        integer = _make_uint_handle("integer")
        left, right = (bit, integer) if bit_first else (integer, bit)

        with trace(tracer):
            result = comparison(left, right)

        assert len(tracer.operations) == 1
        emitted = tracer.operations[0]
        assert isinstance(emitted, CompOp)
        assert emitted.kind is kind
        assert emitted.lhs is left.value
        assert emitted.rhs is right.value
        assert isinstance(result, qmc.Bit)
        assert not isinstance(result, bool)

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

    @pytest.mark.parametrize(
        "bit,integer,expected_qubit",
        [
            pytest.param(False, 0, 0, id="false-equals-zero"),
            pytest.param(False, 1, 1, id="false-differs-one"),
            pytest.param(True, 0, 1, id="true-differs-zero"),
            pytest.param(True, 1, 0, id="true-equals-one"),
            pytest.param(True, 2, 1, id="true-differs-two"),
        ],
    )
    def test_bit_uint_eq_ne_fold(
        self,
        qiskit_transpiler,
        bit: bool,
        integer: int,
        expected_qubit: int,
    ) -> None:
        """Verify bound Bit and UInt comparisons select one branch."""
        executable = qiskit_transpiler.transpile(
            _bound_bit_uint_comparison,
            bindings={"bit": bit, "integer": integer},
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

    @pytest.mark.parametrize(
        "bit,integer,expected",
        [
            pytest.param(False, 0, (1, 0), id="false-equals-zero"),
            pytest.param(True, 1, (1, 0), id="true-equals-one"),
            pytest.param(False, 2, (0, 1), id="false-differs-two"),
        ],
    )
    def test_bit_uint_fold_executes_cross_backend(
        self,
        sdk_transpiler,
        bit: bool,
        integer: int,
        expected: tuple[int, int],
    ) -> None:
        """Verify every backend executes the folded mixed comparison."""
        transpiler = sdk_transpiler.transpiler
        executable = transpiler.transpile(
            _bound_bit_uint_comparison,
            bindings={"bit": bit, "integer": integer},
        )
        result = executable.sample(transpiler.executor(), shots=32).result()

        assert result.results == [(expected, 32)]


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

    @pytest.mark.parametrize(
        "kernel,value",
        [
            pytest.param(_runtime_bit_uint_eq, 0, id="bit-eq-zero"),
            pytest.param(_runtime_bit_uint_eq, 1, id="bit-eq-one"),
            pytest.param(_runtime_bit_uint_eq, 2, id="bit-eq-two"),
            pytest.param(_runtime_uint_bit_ne, 0, id="zero-ne-bit"),
            pytest.param(_runtime_uint_bit_ne, 1, id="one-ne-bit"),
            pytest.param(_runtime_uint_bit_ne, 2, id="two-ne-bit"),
        ],
    )
    @pytest.mark.parametrize(
        "source_one", [False, True], ids=["source-zero", "source-one"]
    )
    def test_runtime_bit_uint_executes(
        self,
        sdk_transpiler,
        kernel,
        value: int,
        source_one: bool,
    ) -> None:
        """Verify mixed Bit and UInt predicates execute on dynamic backends."""
        if sdk_transpiler.backend_name == "quri_parts":
            pytest.skip("QuriParts does not support measurement-dependent control flow")
        transpiler = sdk_transpiler.transpiler
        executable = transpiler.transpile(
            kernel,
            bindings={"source_one": source_one, "value": value},
        )
        result = executable.sample(transpiler.executor(), shots=100).result()
        equal = int(source_one) == value
        expected = int(equal if kernel is _runtime_bit_uint_eq else not equal)

        assert result.results == [(expected, 100)]

    def test_circuit_ir_preserves_mixed_comparison(self, qiskit_transpiler) -> None:
        """Verify shared circuit IR retains one abstract mixed equality."""
        bindings = {"source_one": False, "value": 2}
        prepared = qiskit_transpiler.prepare(_runtime_bit_uint_eq, bindings=bindings)
        plan = qiskit_transpiler.plan_circuit(prepared, bindings=bindings)
        program = lower_circuit_plan(plan, bindings=bindings).quantum_circuit
        [branch] = [
            operation
            for operation in program.operations
            if isinstance(operation, IfInstruction)
        ]

        condition = branch.condition
        assert isinstance(condition, BinaryExpr)
        assert condition.operator is BinaryOperator.EQ
        assert isinstance(condition.left, ClassicalBitExpr)
        assert isinstance(condition.right, LiteralExpr)
        assert condition.right.value == 2

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
