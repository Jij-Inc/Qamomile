"""Tests for classical_prep parametric BinOp emit contract.

Verifies that StandardEmitPass._pre_evaluate_classical() uses the full
symbolic resolver (via _evaluate_binop) so that unbound parameters in
classical_prep produce backend parameter expressions instead of being
silently dropped (which would cause _resolve_angle to return 0.0).

Close conditions:
  C1: scalar parametric classical-prep (theta = a + b) emits parametric expr
  C2: const-index vector param (theta = params[1] + 0.5) emits parametric expr
  C3: unresolved symbolic index is fail-closed (EmitError), not silent 0.0
  C4: existing runtime classical / zero-division tests unaffected
"""

from dataclasses import dataclass
from typing import Any

import pytest

from qamomile.circuit.ir.types.primitives import FloatType, QubitType, UIntType
from qamomile.circuit.ir.value import ArrayValue, Value
from qamomile.circuit.ir.operation.arithmetic_operations import BinOp, BinOpKind
from qamomile.circuit.ir.operation.gate import GateOperation, GateOperationType
from qamomile.circuit.transpiler.errors import EmitError
from qamomile.circuit.transpiler.passes.standard_emit import StandardEmitPass
from qamomile.circuit.transpiler.segments import ClassicalSegment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class ParamExpr:
    """Minimal parametric expression that records symbolic arithmetic."""

    expr: str

    def __add__(self, other: Any) -> "ParamExpr":
        return ParamExpr(f"({self.expr}+{other})")

    def __radd__(self, other: Any) -> "ParamExpr":
        return ParamExpr(f"({other}+{self.expr})")

    def __sub__(self, other: Any) -> "ParamExpr":
        return ParamExpr(f"({self.expr}-{other})")

    def __rsub__(self, other: Any) -> "ParamExpr":
        return ParamExpr(f"({other}-{self.expr})")

    def __mul__(self, other: Any) -> "ParamExpr":
        return ParamExpr(f"({self.expr}*{other})")

    def __rmul__(self, other: Any) -> "ParamExpr":
        return ParamExpr(f"({other}*{self.expr})")

    def __truediv__(self, other: Any) -> "ParamExpr":
        return ParamExpr(f"({self.expr}/{other})")

    def __rtruediv__(self, other: Any) -> "ParamExpr":
        return ParamExpr(f"({other}/{self.expr})")

    def __floordiv__(self, other: Any) -> "ParamExpr":
        return ParamExpr(f"({self.expr}//{other})")

    def __pow__(self, other: Any) -> "ParamExpr":
        return ParamExpr(f"({self.expr}**{other})")

    def __repr__(self) -> str:
        return self.expr


class DummyEmitter:
    def create_parameter(self, name: str) -> ParamExpr:
        return ParamExpr(name)


def _make_gate_with_theta(theta_value: Value) -> GateOperation:
    """Create a minimal RY gate referencing *theta_value*."""
    q = Value(type=QubitType(), name="q")
    return GateOperation(
        operands=[q],
        results=[q.next_version()],
        gate_type=GateOperationType.RY,
        theta=theta_value,
    )


# ---------------------------------------------------------------------------
# C1: scalar parametric classical-prep
# ---------------------------------------------------------------------------


class TestScalarParametricClassicalPrep:
    """theta = a + b must be preserved as a parametric expression."""

    def test_scalar_add_produces_parametric_expression(self):
        a = Value(type=FloatType(), name="a", params={"parameter": "a"})
        b = Value(type=FloatType(), name="b", params={"parameter": "b"})
        out = Value(type=FloatType(), name="theta")
        op = BinOp(operands=[a, b], results=[out], kind=BinOpKind.ADD)
        seg = ClassicalSegment(operations=[op])

        std = StandardEmitPass(DummyEmitter(), bindings={}, parameters=["a", "b"])
        std._pre_evaluate_classical(seg)

        # The output must be stored in bindings
        assert out.uuid in std.bindings or out.name in std.bindings
        stored = std.bindings.get(out.uuid) or std.bindings.get(out.name)
        assert isinstance(stored, ParamExpr), (
            f"Expected ParamExpr, got {type(stored)}: {stored}"
        )

    def test_scalar_add_resolve_angle_returns_expression(self):
        a = Value(type=FloatType(), name="a", params={"parameter": "a"})
        b = Value(type=FloatType(), name="b", params={"parameter": "b"})
        out = Value(type=FloatType(), name="theta")
        op = BinOp(operands=[a, b], results=[out], kind=BinOpKind.ADD)
        seg = ClassicalSegment(operations=[op])

        std = StandardEmitPass(DummyEmitter(), bindings={}, parameters=["a", "b"])
        std._pre_evaluate_classical(seg)

        gate = _make_gate_with_theta(out)
        angle = std._resolve_angle(gate, std.bindings)
        # Must NOT be 0.0
        assert not isinstance(angle, (int, float)) or angle != 0.0
        assert isinstance(angle, ParamExpr)

    def test_scalar_sub_produces_parametric_expression(self):
        a = Value(type=FloatType(), name="a", params={"parameter": "a"})
        b = Value(type=FloatType(), name="b", params={"parameter": "b"})
        out = Value(type=FloatType(), name="diff")
        op = BinOp(operands=[a, b], results=[out], kind=BinOpKind.SUB)
        seg = ClassicalSegment(operations=[op])

        std = StandardEmitPass(DummyEmitter(), bindings={}, parameters=["a", "b"])
        std._pre_evaluate_classical(seg)

        stored = std.bindings.get(out.uuid) or std.bindings.get(out.name)
        assert isinstance(stored, ParamExpr)

    def test_scalar_mul_produces_parametric_expression(self):
        a = Value(type=FloatType(), name="a", params={"parameter": "a"})
        b = Value(type=FloatType(), name="b", params={"parameter": "b"})
        out = Value(type=FloatType(), name="prod")
        op = BinOp(operands=[a, b], results=[out], kind=BinOpKind.MUL)
        seg = ClassicalSegment(operations=[op])

        std = StandardEmitPass(DummyEmitter(), bindings={}, parameters=["a", "b"])
        std._pre_evaluate_classical(seg)

        stored = std.bindings.get(out.uuid) or std.bindings.get(out.name)
        assert isinstance(stored, ParamExpr)


# ---------------------------------------------------------------------------
# C2: const-index vector parameter
# ---------------------------------------------------------------------------


class TestVectorConstIndexParametricClassicalPrep:
    """theta = params[1] + 0.5 must be preserved as parametric expression."""

    def test_const_index_vector_add_produces_expression(self):
        params = ArrayValue(
            type=FloatType(), name="params", params={"parameter": "params"}
        )
        idx = Value(type=UIntType(), name="idx1", params={"const": 1})
        element = Value(
            type=FloatType(),
            name="params[1]",
            parent_array=params,
            element_indices=(idx,),
            params={"parameter": "params[1]"},
        )
        bias = Value(type=FloatType(), name="bias", params={"const": 0.5})
        out = Value(type=FloatType(), name="shifted")
        op = BinOp(operands=[element, bias], results=[out], kind=BinOpKind.ADD)
        seg = ClassicalSegment(operations=[op])

        std = StandardEmitPass(DummyEmitter(), bindings={}, parameters=["params"])
        std._pre_evaluate_classical(seg)

        stored = std.bindings.get(out.uuid) or std.bindings.get(out.name)
        assert isinstance(stored, ParamExpr), (
            f"Expected ParamExpr, got {type(stored)}: {stored}"
        )

    def test_concrete_index_at_emit_produces_expression(self):
        """params[i] + 0.5 with bindings={'i': 1} is resolvable."""
        params = ArrayValue(
            type=FloatType(), name="params", params={"parameter": "params"}
        )
        i = Value(type=UIntType(), name="i", params={"parameter": "i"})
        element = Value(
            type=FloatType(),
            name="params[i]",
            parent_array=params,
            element_indices=(i,),
            params={"parameter": "params[i]"},
        )
        bias = Value(type=FloatType(), name="bias", params={"const": 0.5})
        out = Value(type=FloatType(), name="theta")
        op = BinOp(operands=[element, bias], results=[out], kind=BinOpKind.ADD)
        seg = ClassicalSegment(operations=[op])

        std = StandardEmitPass(DummyEmitter(), bindings={"i": 1}, parameters=["params"])
        std._pre_evaluate_classical(seg)

        stored = std.bindings.get(out.uuid) or std.bindings.get(out.name)
        assert isinstance(stored, ParamExpr), (
            f"Expected ParamExpr, got {type(stored)}: {stored}"
        )


# ---------------------------------------------------------------------------
# C3: fail-closed for unresolved theta
# ---------------------------------------------------------------------------


class TestFailClosedUnresolvedTheta:
    """Classical-prep outputs that could not be resolved must raise EmitError."""

    def test_unresolved_prep_output_raises_emit_error(self):
        """A classical-prep BinOp output that _evaluate_binop could not
        resolve must cause _resolve_angle to raise EmitError."""
        # Simulate an unresolvable BinOp in classical_prep
        a = Value(type=FloatType(), name="a", params={"parameter": "a"})
        # b has no const and no param — _evaluate_binop can't resolve it
        b = Value(type=FloatType(), name="unknown")
        out = Value(type=FloatType(), name="theta")
        op = BinOp(operands=[a, b], results=[out], kind=BinOpKind.ADD)
        seg = ClassicalSegment(operations=[op])

        std = StandardEmitPass(DummyEmitter(), bindings={}, parameters=["a"])
        std._pre_evaluate_classical(seg)

        # out should be tracked as unresolved prep output
        gate = _make_gate_with_theta(out)
        with pytest.raises(EmitError, match="Cannot resolve gate angle"):
            std._resolve_angle(gate, std.bindings)

    def test_unresolved_prep_output_does_not_reuse_same_name_temporary(self):
        """Fail closed even when an earlier temporary reused the same name.

        Synthetic temporaries (float_tmp, uint_tmp, bit_tmp) are stored by
        UUID only — never by name — to avoid same-name collision.  The
        resolved earlier temp must be reachable via UUID, and the later
        unresolved temp must still trigger EmitError.
        """
        a = Value(type=FloatType(), name="a", params={"parameter": "a"})
        one = Value(type=FloatType(), name="one", params={"const": 1.0})
        resolved_out = Value(type=FloatType(), name="float_tmp")
        resolved_op = BinOp(
            operands=[a, one],
            results=[resolved_out],
            kind=BinOpKind.ADD,
        )

        unknown = Value(type=FloatType(), name="unknown")
        bias = Value(type=FloatType(), name="bias", params={"const": 0.5})
        unresolved_out = Value(type=FloatType(), name="float_tmp")
        unresolved_op = BinOp(
            operands=[unknown, bias],
            results=[unresolved_out],
            kind=BinOpKind.ADD,
        )
        seg = ClassicalSegment(operations=[resolved_op, unresolved_op])

        std = StandardEmitPass(DummyEmitter(), bindings={}, parameters=["a"])
        std._pre_evaluate_classical(seg)

        # UUID-only contract: resolved temp is stored by UUID, not by name
        assert resolved_out.uuid in std.bindings
        assert resolved_out.name not in std.bindings
        gate = _make_gate_with_theta(unresolved_out)
        with pytest.raises(EmitError, match="Cannot resolve gate angle"):
            std._resolve_angle(gate, std.bindings)

    def test_unresolved_symbolic_index_does_not_produce_zero(self):
        """params[i] + 0.5 with unresolved i must not silently produce 0.0.

        _evaluate_binop cannot resolve either operand so bindings stays empty.
        When the gate theta then tries to resolve, it should fail-closed.
        """
        params = ArrayValue(
            type=FloatType(), name="params", params={"parameter": "params"}
        )
        i = Value(type=UIntType(), name="i", params={"parameter": "i"})
        element = Value(
            type=FloatType(),
            name="params[i]",
            parent_array=params,
            element_indices=(i,),
            params={"parameter": "params[i]"},
        )
        bias = Value(type=FloatType(), name="bias", params={"const": 0.5})
        out = Value(type=FloatType(), name="theta")
        op = BinOp(operands=[element, bias], results=[out], kind=BinOpKind.ADD)
        seg = ClassicalSegment(operations=[op])

        std = StandardEmitPass(DummyEmitter(), bindings={}, parameters=["params", "i"])
        std._pre_evaluate_classical(seg)

        # out should NOT be in bindings because _evaluate_binop can't
        # resolve the element with unresolved index
        gate = _make_gate_with_theta(out)
        with pytest.raises(EmitError, match="Cannot resolve gate angle"):
            std._resolve_angle(gate, std.bindings)
