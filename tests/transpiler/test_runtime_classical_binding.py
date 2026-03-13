"""Tests for runtime classical binding contract.

Verifies that ClassicalExecutor._get_value() resolves user bindings
by parameter_name and name fallback, and that ExecutableProgram entry
points (run, sample, _run_expval) properly populate the runtime
classical execution context.
"""

from qamomile.circuit.ir.types.primitives import FloatType
from qamomile.circuit.ir.value import Value
from qamomile.circuit.ir.operation.arithmetic_operations import BinOp, BinOpKind
from qamomile.circuit.transpiler.classical_executor import ClassicalExecutor
from qamomile.circuit.transpiler.compiled_segments import (
    CompiledClassicalSegment,
    CompiledExpvalSegment,
    CompiledQuantumSegment,
)
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
import qamomile.observable as qm_o


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


# ---------------------------------------------------------------------------
# ClassicalExecutor._get_value unit tests
# ---------------------------------------------------------------------------


class TestGetValueFallback:
    """ClassicalExecutor._get_value resolves values via parameter_name and name."""

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
