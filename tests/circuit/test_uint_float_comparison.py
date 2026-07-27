"""Regression tests for mixed UInt and Float comparisons.

Classical handle comparisons must emit CompOp IR nodes during tracing.
Previously, equality between UInt and Float handles returned NotImplemented
from both sides, allowing Python to collapse the result to a Boolean. Ordering
comparisons with a Python float could instead access a missing value attribute.
The comparison methods now normalize supported operands while retaining mixed
handle types in the abstract IR.

The tests cover direct IR emission and compile-time folding through the
qkernel decorator.
"""

from __future__ import annotations

import operator
from collections.abc import Callable

import numpy as np
import pytest

import qamomile.circuit as qmc
from qamomile.circuit.frontend.tracer import Tracer, trace
from qamomile.circuit.ir.operation.arithmetic_operations import (
    CompOp,
    CompOpKind,
)
from qamomile.circuit.ir.types.primitives import FloatType, UIntType
from qamomile.circuit.ir.value import Value

Comparison = Callable[[object, object], object]

_COMPARISON_CASES = [
    pytest.param(operator.eq, CompOpKind.EQ, id="eq"),
    pytest.param(operator.ne, CompOpKind.NEQ, id="ne"),
    pytest.param(operator.lt, CompOpKind.LT, id="lt"),
    pytest.param(operator.gt, CompOpKind.GT, id="gt"),
    pytest.param(operator.le, CompOpKind.LE, id="le"),
    pytest.param(operator.ge, CompOpKind.GE, id="ge"),
]


def _make_uint_handle(name: str) -> qmc.UInt:
    """Build a non-constant UInt handle for tracing.

    Args:
        name (str): Name assigned to the wrapped IR value.

    Returns:
        qmc.UInt: A non-constant unsigned-integer handle.
    """
    return qmc.UInt(value=Value(type=UIntType(), name=name))


def _make_float_handle(name: str) -> qmc.Float:
    """Build a non-constant Float handle for tracing.

    Args:
        name (str): Name assigned to the wrapped IR value.

    Returns:
        qmc.Float: A non-constant floating-point handle.
    """
    return qmc.Float(value=Value(type=FloatType(), name=name))


@qmc.qkernel
def _mixed_less_than_sample(threshold: qmc.Float) -> qmc.Bit:
    """Expose a bound mixed comparison through deterministic sampling.

    Args:
        threshold (qmc.Float): Compile-time floating-point comparison value.

    Returns:
        qmc.Bit: One when the constant UInt value is below the threshold.
    """
    q = qmc.qubit("q")
    if qmc.uint(1) < threshold:
        q = qmc.x(q)
    return qmc.measure(q)


class TestUIntFloatComparisonEmitsIR:
    """Verify that mixed numeric comparisons remain in the IR."""

    @pytest.mark.parametrize("comparison,kind", _COMPARISON_CASES)
    def test_uint_op_float(
        self,
        comparison: Comparison,
        kind: CompOpKind,
    ) -> None:
        """Verify every UInt-to-Float comparison emits one CompOp."""
        tracer = Tracer()
        uint_value = _make_uint_handle("uint_value")
        float_value = _make_float_handle("float_value")

        with trace(tracer):
            result = comparison(uint_value, float_value)

        assert len(tracer.operations) == 1
        emitted = tracer.operations[0]
        assert isinstance(emitted, CompOp)
        assert emitted.kind is kind
        assert emitted.lhs is uint_value.value
        assert emitted.rhs is float_value.value
        assert isinstance(emitted.lhs.type, UIntType)
        assert isinstance(emitted.rhs.type, FloatType)
        assert isinstance(result, qmc.Bit)
        assert not isinstance(result, bool)

    @pytest.mark.parametrize("comparison,kind", _COMPARISON_CASES)
    def test_float_op_uint(
        self,
        comparison: Comparison,
        kind: CompOpKind,
    ) -> None:
        """Verify every Float-to-UInt comparison emits one CompOp."""
        tracer = Tracer()
        float_value = _make_float_handle("float_value")
        uint_value = _make_uint_handle("uint_value")

        with trace(tracer):
            result = comparison(float_value, uint_value)

        assert len(tracer.operations) == 1
        emitted = tracer.operations[0]
        assert isinstance(emitted, CompOp)
        assert emitted.kind is kind
        assert emitted.lhs is float_value.value
        assert emitted.rhs is uint_value.value
        assert isinstance(emitted.lhs.type, FloatType)
        assert isinstance(emitted.rhs.type, UIntType)
        assert isinstance(result, qmc.Bit)
        assert not isinstance(result, bool)

    def test_python_float_is_normalized(self) -> None:
        """Verify a Python float becomes a constant Float operand."""
        tracer = Tracer()
        uint_value = _make_uint_handle("uint_value")

        with trace(tracer):
            result = uint_value < 1.5

        assert len(tracer.operations) == 1
        emitted = tracer.operations[0]
        assert isinstance(emitted, CompOp)
        assert emitted.kind is CompOpKind.LT
        assert isinstance(emitted.rhs.type, FloatType)
        np.testing.assert_allclose(
            emitted.rhs.get_const(),
            1.5,
            rtol=1e-12,
            atol=1e-12,
        )
        assert isinstance(result, qmc.Bit)

    def test_reflected_python_float_is_normalized(self) -> None:
        """Verify reflected ordering delegates to the matching UInt method."""
        tracer = Tracer()
        uint_value = _make_uint_handle("uint_value")

        with trace(tracer):
            result = 1.5 < uint_value

        assert len(tracer.operations) == 1
        emitted = tracer.operations[0]
        assert isinstance(emitted, CompOp)
        assert emitted.kind is CompOpKind.GT
        assert isinstance(emitted.lhs.type, UIntType)
        assert isinstance(emitted.rhs.type, FloatType)
        np.testing.assert_allclose(
            emitted.rhs.get_const(),
            1.5,
            rtol=1e-12,
            atol=1e-12,
        )
        assert isinstance(result, qmc.Bit)

    def test_same_type_comparisons_still_emit_compop(self) -> None:
        """Verify operand normalization preserves homogeneous comparisons."""
        tracer = Tracer()

        with trace(tracer):
            uint_result = _make_uint_handle("u1") == _make_uint_handle("u2")
            float_result = _make_float_handle("f1") < _make_float_handle("f2")

        assert isinstance(uint_result, qmc.Bit)
        assert isinstance(float_result, qmc.Bit)
        assert len(tracer.operations) == 2
        assert all(isinstance(op, CompOp) for op in tracer.operations)

    def test_unsupported_operands_follow_python_protocol(self) -> None:
        """Verify unsupported comparisons return NotImplemented or fall back."""
        uint_value = _make_uint_handle("uint_value")
        float_value = _make_float_handle("float_value")

        assert uint_value.__eq__("not a number") is NotImplemented
        assert uint_value.__lt__(object()) is NotImplemented
        assert float_value.__ne__("not a number") is NotImplemented
        assert float_value.__ge__(object()) is NotImplemented
        assert (uint_value == "not a number") is False
        assert (float_value != "not a number") is True
        with pytest.raises(TypeError):
            _ = uint_value < object()


class TestUIntFloatCompileTimeFold:
    """Verify mixed comparisons fold after compile-time binding."""

    @pytest.mark.parametrize(
        "threshold,expected_indices",
        [
            pytest.param(0.0, [0], id="lower-bound"),
            pytest.param(1.0, [1], id="integer-middle"),
            pytest.param(2.0, [2], id="upper-bound"),
            pytest.param(1.5, [], id="fractional"),
        ],
    )
    def test_uint_eq_float_folds(
        self,
        qiskit_transpiler,
        threshold: float,
        expected_indices: list[int],
    ) -> None:
        """Verify equality retains only the matching loop iteration."""

        @qmc.qkernel
        def kernel(threshold: qmc.Float) -> qmc.Vector[qmc.Bit]:
            """Mark the loop index equal to a bound floating-point threshold.

            Args:
                threshold (qmc.Float): Compile-time comparison threshold.

            Returns:
                qmc.Vector[qmc.Bit]: Measurements of the three marked qubits.
            """
            q = qmc.qubit_array(3, name="q")
            for i in qmc.range(3):
                if i == threshold:
                    q[i] = qmc.x(q[i])
            return qmc.measure(q)

        executable = qiskit_transpiler.transpile(
            kernel,
            bindings={"threshold": threshold},
        )
        circuit = executable.compiled_quantum[0].circuit
        x_indices = [
            circuit.find_bit(instruction.qubits[0]).index
            for instruction in circuit.data
            if instruction.operation.name == "x"
        ]
        assert x_indices == expected_indices
        assert all(
            instruction.operation.name != "if_else" for instruction in circuit.data
        )

    @pytest.mark.parametrize(
        "threshold,expected_indices",
        [
            pytest.param(-0.5, [], id="below-range"),
            pytest.param(0.0, [], id="lower-bound"),
            pytest.param(0.5, [0], id="fractional-low"),
            pytest.param(1.5, [0, 1], id="fractional-high"),
            pytest.param(3.0, [0, 1, 2], id="above-range"),
        ],
    )
    def test_uint_lt_float_folds(
        self,
        qiskit_transpiler,
        threshold: float,
        expected_indices: list[int],
    ) -> None:
        """Verify ordering retains exactly the loop indices below the threshold."""

        @qmc.qkernel
        def kernel(threshold: qmc.Float) -> qmc.Vector[qmc.Bit]:
            """Mark every loop index below a bound floating-point threshold.

            Args:
                threshold (qmc.Float): Compile-time comparison threshold.

            Returns:
                qmc.Vector[qmc.Bit]: Measurements of the three marked qubits.
            """
            q = qmc.qubit_array(3, name="q")
            for i in qmc.range(3):
                if i < threshold:
                    q[i] = qmc.x(q[i])
            return qmc.measure(q)

        executable = qiskit_transpiler.transpile(
            kernel,
            bindings={"threshold": threshold},
        )
        circuit = executable.compiled_quantum[0].circuit
        x_indices = [
            circuit.find_bit(instruction.qubits[0]).index
            for instruction in circuit.data
            if instruction.operation.name == "x"
        ]
        assert x_indices == expected_indices
        assert all(
            instruction.operation.name != "if_else" for instruction in circuit.data
        )


@pytest.mark.parametrize(
    "threshold,expected",
    [
        pytest.param(0.5, 0, id="false"),
        pytest.param(1.0, 0, id="boundary"),
        pytest.param(1.5, 1, id="true"),
    ],
)
def test_mixed_comparison_executes_cross_backend(
    sdk_transpiler,
    threshold: float,
    expected: int,
) -> None:
    """Verify every supported SDK backend executes the folded comparison."""
    transpiler = sdk_transpiler.transpiler
    executable = transpiler.transpile(
        _mixed_less_than_sample,
        bindings={"threshold": threshold},
    )
    result = executable.sample(transpiler.executor(), shots=32).result()

    assert result.results == [(expected, 32)]
