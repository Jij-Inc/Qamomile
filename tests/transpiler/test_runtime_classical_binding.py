"""Tests for runtime classical binding contract.

Verifies that ClassicalExecutor._get_value() resolves user bindings
by parameter_name, name, and array-element fallback, and that
ExecutableProgram entry points (run, sample, _run_expval) properly
populate the runtime classical execution context.
"""

import pytest

import qamomile.observable as qm_o
from qamomile.circuit.ir.operation.arithmetic_operations import BinOp, BinOpKind
from qamomile.circuit.ir.types.primitives import FloatType, UIntType
from qamomile.circuit.ir.value import ArrayValue, Value
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
    ExpvalSegment,
    QuantumSegment,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class DummyExecutor(QuantumExecutor[object]):
    def bind_parameters(self, circuit, bindings, metadata):
        return circuit

    def execute(self, circuit, shots):
        return {"0": shots}

    def estimate(self, circuit, hamiltonian, params=None):
        return 1.23


def _make_add_program(*, with_expval: bool = False) -> ExecutableProgram[object]:
    """Build a minimal program with classical_prep that computes a + b."""
    aval = Value(type=FloatType(), name="a", params={"parameter": "a"})
    bval = Value(type=FloatType(), name="b", params={"parameter": "b"})
    out = Value(type=FloatType(), name="sum")
    binop = BinOp(operands=[aval, bval], results=[out], kind=BinOpKind.ADD)
    classical = ClassicalSegment(operations=[binop])
    quantum = QuantumSegment(operations=[])
    meta = ParameterMetadata(
        parameters=[
            ParameterInfo(name="a", array_name="a", index=None, backend_param="a"),
            ParameterInfo(name="b", array_name="b", index=None, backend_param="b"),
        ]
    )

    kwargs: dict = dict(
        compiled_quantum=[
            CompiledQuantumSegment(
                segment=quantum, circuit=object(), parameter_metadata=meta
            )
        ],
        compiled_classical=[CompiledClassicalSegment(segment=classical)],
        execution_order=[("classical", 0), ("quantum", 0)],
    )

    if with_expval:
        kwargs["compiled_expval"] = [
            CompiledExpvalSegment(
                segment=ExpvalSegment(result_ref="ev"),
                hamiltonian=qm_o.Z(0),
                result_ref="ev",
            )
        ]
        kwargs["execution_order"].append(("expval", 0))

    return ExecutableProgram(**kwargs)


def _make_vector_lookup_program(
    *,
    with_expval: bool = False,
) -> ExecutableProgram[object]:
    """Build a program whose classical_prep resolves params[i] at runtime."""

    params = ArrayValue(type=FloatType(), name="params", params={"parameter": "params"})
    idx = Value(type=UIntType(), name="i", params={"parameter": "i"})
    element = Value(
        type=FloatType(),
        name="params[i]",
        parent_array=params,
        element_indices=(idx,),
        params={"parameter": "params[i]"},
    )
    bias = Value(type=FloatType(), name="bias", params={"const": 0.5})
    out = Value(type=FloatType(), name="shifted")
    binop = BinOp(operands=[element, bias], results=[out], kind=BinOpKind.ADD)
    classical = ClassicalSegment(operations=[binop])
    quantum = QuantumSegment(operations=[])
    meta = ParameterMetadata(
        parameters=[
            ParameterInfo(
                name="params[0]",
                array_name="params",
                index=0,
                backend_param="params[0]",
            ),
            ParameterInfo(
                name="params[1]",
                array_name="params",
                index=1,
                backend_param="params[1]",
            ),
        ]
    )

    kwargs: dict = dict(
        compiled_quantum=[
            CompiledQuantumSegment(
                segment=quantum, circuit=object(), parameter_metadata=meta
            )
        ],
        compiled_classical=[CompiledClassicalSegment(segment=classical)],
        execution_order=[("classical", 0), ("quantum", 0)],
        output_refs=[out.uuid],
    )

    if with_expval:
        kwargs["compiled_expval"] = [
            CompiledExpvalSegment(
                segment=ExpvalSegment(result_ref="ev"),
                hamiltonian=qm_o.Z(0),
                result_ref="ev",
            )
        ]
        kwargs["execution_order"].append(("expval", 0))

    return ExecutableProgram(**kwargs)


# ---------------------------------------------------------------------------
# ClassicalExecutor._get_value unit tests
# ---------------------------------------------------------------------------


class TestGetValueFallback:
    """ClassicalExecutor._get_value resolves runtime-bound classical values."""

    def test_resolve_by_uuid(self):
        val = Value(type=FloatType(), name="x", params={"parameter": "x"})
        ctx = ExecutionContext()
        results = {val.uuid: 42.0}
        assert ClassicalExecutor()._get_value(val, ctx, results) == 42.0

    def test_resolve_by_context_uuid(self):
        val = Value(type=FloatType(), name="x", params={"parameter": "x"})
        ctx = ExecutionContext()
        ctx.set(val.uuid, 99.0)
        assert ClassicalExecutor()._get_value(val, ctx, {}) == 99.0

    def test_resolve_by_parameter_name(self):
        val = Value(type=FloatType(), name="x", params={"parameter": "x"})
        ctx = ExecutionContext(initial_bindings={"x": 3.14})
        assert ClassicalExecutor()._get_value(val, ctx, {}) == 3.14

    def test_resolve_by_value_name(self):
        val = Value(type=FloatType(), name="myvar")
        ctx = ExecutionContext(initial_bindings={"myvar": 2.71})
        assert ClassicalExecutor()._get_value(val, ctx, {}) == 2.71

    def test_resolve_constant(self):
        val = Value(type=FloatType(), name="c", params={"const": 7.0})
        ctx = ExecutionContext()
        assert ClassicalExecutor()._get_value(val, ctx, {}) == 7.0

    def test_resolve_symbolic_array_element_from_context(self):
        params = ArrayValue(
            type=FloatType(), name="params", params={"parameter": "params"}
        )
        idx = Value(type=UIntType(), name="i", params={"parameter": "i"})
        val = Value(
            type=FloatType(),
            name="params[i]",
            parent_array=params,
            element_indices=(idx,),
            params={"parameter": "params[i]"},
        )
        ctx = ExecutionContext(initial_bindings={"params": [1.0, 2.0], "i": 1})
        assert ClassicalExecutor()._get_value(val, ctx, {}) == 2.0

    def test_resolve_symbolic_array_element_index_from_results(self):
        params = ArrayValue(
            type=FloatType(), name="params", params={"parameter": "params"}
        )
        idx = Value(type=UIntType(), name="i")
        val = Value(
            type=FloatType(),
            name="params[i]",
            parent_array=params,
            element_indices=(idx,),
            params={"parameter": "params[i]"},
        )
        ctx = ExecutionContext(initial_bindings={"params": [1.0, 2.0]})
        results = {idx.uuid: 1}
        assert ClassicalExecutor()._get_value(val, ctx, results) == 2.0

    def test_unresolved_symbolic_array_index_raises(self):
        params = ArrayValue(
            type=FloatType(), name="params", params={"parameter": "params"}
        )
        idx = Value(type=UIntType(), name="i", params={"parameter": "i"})
        val = Value(
            type=FloatType(),
            name="params[i]",
            parent_array=params,
            element_indices=(idx,),
            params={"parameter": "params[i]"},
        )
        ctx = ExecutionContext(initial_bindings={"params": [1.0, 2.0]})
        with pytest.raises(ExecutionError, match="params\\[i\\]"):
            ClassicalExecutor()._get_value(val, ctx, {})

    def test_missing_array_binding_raises(self):
        params = ArrayValue(
            type=FloatType(), name="params", params={"parameter": "params"}
        )
        idx = Value(type=UIntType(), name="i", params={"parameter": "i"})
        val = Value(
            type=FloatType(),
            name="params[i]",
            parent_array=params,
            element_indices=(idx,),
            params={"parameter": "params[i]"},
        )
        ctx = ExecutionContext(initial_bindings={"i": 0})
        with pytest.raises(ExecutionError, match="params\\[i\\]"):
            ClassicalExecutor()._get_value(val, ctx, {})

    def test_near_zero_divisor_not_rejected(self):
        """Exact non-zero divisors (even very small) must not be rejected."""
        lhs = Value(type=FloatType(), name="lhs", params={"const": 1.0})
        rhs = Value(type=FloatType(), name="rhs", params={"const": 1e-320})
        out = Value(type=FloatType(), name="out")
        op = BinOp(operands=[lhs, rhs], results=[out], kind=BinOpKind.DIV)
        seg = ClassicalSegment(operations=[op])
        executor = ClassicalExecutor()
        results = executor.execute(seg, ExecutionContext())
        assert out.uuid in results


# ---------------------------------------------------------------------------
# ExecutableProgram entry point tests
# ---------------------------------------------------------------------------


class TestRuntimeClassicalPrep:
    """ExecutableProgram entry points populate context for runtime classical_prep."""

    def test_run_with_classical_prep(self):
        exe = _make_add_program()
        job = exe.run(DummyExecutor(), bindings={"a": 1.0, "b": 2.0})
        result = job.result()
        # Should succeed without ExecutionError
        assert result is not None

    def test_sample_with_classical_prep(self):
        exe = _make_add_program()
        job = exe.sample(DummyExecutor(), shots=2, bindings={"a": 1.0, "b": 2.0})
        result = job.result()
        assert result is not None

    def test_expval_with_classical_prep(self):
        exe = _make_add_program(with_expval=True)
        job = exe.run(DummyExecutor(), bindings={"a": 1.0, "b": 2.0})
        result = job.result()
        assert result == 1.23

    def test_classical_prep_assert_present(self):
        """Verify execution_order contains classical segment (not folded away)."""
        exe = _make_add_program()
        assert ("classical", 0) in exe.execution_order

    def test_run_with_vector_runtime_binding(self):
        exe = _make_vector_lookup_program()
        result = exe.run(
            DummyExecutor(), bindings={"params": [1.0, 2.0], "i": 1}
        ).result()
        assert result == 2.5

    def test_sample_with_vector_runtime_binding(self):
        exe = _make_vector_lookup_program()
        result = exe.sample(
            DummyExecutor(),
            shots=2,
            bindings={"params": [1.0, 2.0], "i": 1},
        ).result()
        assert result.results == [(2.5, 2)]

    def test_expval_with_vector_runtime_binding(self):
        exe = _make_vector_lookup_program(with_expval=True)
        result = exe.run(
            DummyExecutor(), bindings={"params": [1.0, 2.0], "i": 1}
        ).result()
        assert result == 1.23
