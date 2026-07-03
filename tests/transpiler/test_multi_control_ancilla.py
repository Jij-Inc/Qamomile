"""Tests for clean-ancilla planning of the multi-controlled decomposition."""

import numpy as np
import pytest

from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.control_flow import ForOperation
from qamomile.circuit.ir.operation.gate import (
    ConcreteControlledU,
    GateOperation,
    GateOperationType,
    SymbolicControlledU,
)
from qamomile.circuit.ir.types.primitives import (
    FloatType,
    QubitType,
    UIntType,
)
from qamomile.circuit.ir.value import ArrayValue, Value
from qamomile.circuit.transpiler.errors import EmitError
from qamomile.circuit.transpiler.passes.emit_support.multi_control_ancilla import (
    _MAX_PRECISE_FOR_LOOP_ITERATIONS,
    MultiControlAncillaPool,
    estimate_multi_control_ancilla_demand,
)
from qamomile.circuit.transpiler.passes.emit_support.value_resolver import (
    ValueResolver,
)


def _qubit(name: str) -> Value:
    """Build a fresh scalar qubit Value."""
    return Value(type=QubitType(), name=name)


def _fixed_gate(gate_type: GateOperationType, num_qubits: int) -> GateOperation:
    """Build a fixed GateOperation over fresh qubit values."""
    qubits = [_qubit(f"q{i}") for i in range(num_qubits)]
    return GateOperation.fixed(gate_type, qubits, [q.next_version() for q in qubits])


def _rotation_gate(gate_type: GateOperationType) -> GateOperation:
    """Build a single-qubit rotation GateOperation with a constant angle."""
    qubit = _qubit("q0")
    theta = Value(type=FloatType(), name="theta").with_const(0.5)
    return GateOperation.rotation(gate_type, [qubit], theta, [qubit.next_version()])


def _controlled(num_controls: int, inner: GateOperation) -> ConcreteControlledU:
    """Wrap one inner gate in a ConcreteControlledU with fresh controls."""
    controls = [_qubit(f"c{i}") for i in range(num_controls)]
    target = _qubit("t")
    operands = [*controls, target]
    return ConcreteControlledU(
        operands=operands,
        results=[v.next_version() for v in operands],
        num_controls=num_controls,
        block=Block(operations=[inner]),
    )


def _estimate(operations: list) -> int:
    """Run the demand estimate with a fresh resolver and empty bindings."""
    return estimate_multi_control_ancilla_demand(operations, ValueResolver(), {})


def test_pool_take_returns_leading_indices() -> None:
    """take() hands out the first k reserved indices."""
    pool = MultiControlAncillaPool(first_index=7, count=3)
    assert pool.take(2) == [7, 8]
    assert pool.take(3) == [7, 8, 9]


def test_pool_take_shortfall_returns_none() -> None:
    """take() signals a shortfall with None instead of raising."""
    pool = MultiControlAncillaPool(first_index=7, count=1)
    assert pool.take(2) is None


def test_estimate_no_controlled_operations_is_zero() -> None:
    """A segment without controlled operations needs no ancillas."""
    assert _estimate([_fixed_gate(GateOperationType.X, 1)]) == 0
    assert _estimate([_fixed_gate(GateOperationType.TOFFOLI, 3)]) == 0


def test_estimate_single_control_gates_are_native() -> None:
    """One-control X / RY lower natively (CX / CRY) — no ancillas."""
    assert _estimate([_controlled(1, _fixed_gate(GateOperationType.X, 1))]) == 0
    assert _estimate([_controlled(1, _rotation_gate(GateOperationType.RY))]) == 0


def test_estimate_two_control_x_uses_toffoli() -> None:
    """Two-control X lowers to a native Toffoli — no ancillas."""
    assert _estimate([_controlled(2, _fixed_gate(GateOperationType.X, 1))]) == 0


def test_estimate_three_control_x_needs_two_ancillas() -> None:
    """An n-control X (n >= 3) cascades on n - 1 ancillas."""
    assert _estimate([_controlled(3, _fixed_gate(GateOperationType.X, 1))]) == 2


def test_estimate_two_control_rotation_needs_one_ancilla() -> None:
    """An n-control rotation (n >= 2) cascades on n - 1 ancillas."""
    assert _estimate([_controlled(2, _rotation_gate(GateOperationType.RY))]) == 1


def test_estimate_absorbs_gate_own_controls() -> None:
    """CX / TOFFOLI absorb their own controls before the cascade check."""
    # 1 outer control + CX -> 2-control X -> Toffoli, no ancillas.
    assert _estimate([_controlled(1, _fixed_gate(GateOperationType.CX, 2))]) == 0
    # 2 outer controls + CX -> 3-control X -> 2 ancillas.
    assert _estimate([_controlled(2, _fixed_gate(GateOperationType.CX, 2))]) == 2
    # 3 outer controls + TOFFOLI -> 5-control X -> 4 ancillas.
    assert _estimate([_controlled(3, _fixed_gate(GateOperationType.TOFFOLI, 3))]) == 4


def test_estimate_composes_nested_controls() -> None:
    """Nested controlled-U control counts compose additively."""
    inner = _controlled(2, _fixed_gate(GateOperationType.X, 1))
    outer = _controlled(1, inner)  # type: ignore[arg-type]
    # Composed 3-control X -> 2 ancillas.
    assert _estimate([outer]) == 2


def test_estimate_takes_max_over_segment() -> None:
    """The segment demand is the max, not the sum, of per-gate demands."""
    ops = [
        _controlled(4, _fixed_gate(GateOperationType.X, 1)),  # 3 ancillas
        _controlled(2, _rotation_gate(GateOperationType.RZ)),  # 1 ancilla
    ]
    assert _estimate(ops) == 3


def test_estimate_recurses_into_for_bodies() -> None:
    """Controlled gates inside a ForOperation body are counted."""
    start = Value(type=UIntType(), name="start").with_const(0)
    stop = Value(type=UIntType(), name="stop").with_const(4)
    step = Value(type=UIntType(), name="step").with_const(1)
    loop = ForOperation(
        operands=[start, stop, step],
        results=[],
        loop_var="k",
        loop_var_value=Value(type=UIntType(), name="k"),
        operations=[_controlled(3, _fixed_gate(GateOperationType.X, 1))],
    )
    assert _estimate([loop]) == 2


def test_estimate_symbolic_num_controls_resolves_constant() -> None:
    """A SymbolicControlledU with a constant num_controls resolves exactly."""
    controls = [_qubit(f"c{i}") for i in range(3)]
    target = _qubit("t")
    operands = [*controls, target]
    op = SymbolicControlledU(
        operands=operands,
        results=[v.next_version() for v in operands],
        num_controls=Value(type=UIntType(), name="nc").with_const(3),
        num_control_args=3,
        block=Block(operations=[_fixed_gate(GateOperationType.X, 1)]),
    )
    assert _estimate([op]) == 2


def test_estimate_loop_variable_dependent_num_controls() -> None:
    """A num_controls tied to the loop variable resolves per iteration.

    Mirrors the modular increment shape (``qmc.control(qmc.x,
    num_controls=k)`` inside ``qmc.range(...)``): the demand is the
    maximum over the unrolled iterations, reached at ``k = 4`` here.
    """
    loop_var = Value(type=UIntType(), name="k")
    controls = [_qubit(f"c{i}") for i in range(4)]
    target = _qubit("t")
    operands = [*controls, target]
    mcx = SymbolicControlledU(
        operands=operands,
        results=[v.next_version() for v in operands],
        num_controls=loop_var,
        num_control_args=4,
        block=Block(operations=[_fixed_gate(GateOperationType.X, 1)]),
    )
    start = Value(type=UIntType(), name="start").with_const(1)
    stop = Value(type=UIntType(), name="stop").with_const(5)
    step = Value(type=UIntType(), name="step").with_const(1)
    loop = ForOperation(
        operands=[start, stop, step],
        results=[],
        loop_var="k",
        loop_var_value=loop_var,
        operations=[mcx],
    )
    # k runs over 1..4; the widest iteration is a 4-control X.
    assert _estimate([loop]) == 3


def test_estimate_large_loop_falls_back_to_symbolic_body_walk() -> None:
    """Large loops avoid per-iteration estimation and use operand width."""
    loop_var = Value(type=UIntType(), name="k")
    width = _MAX_PRECISE_FOR_LOOP_ITERATIONS + 1
    controls = ArrayValue(
        type=QubitType(),
        name="controls",
        shape=(Value(type=UIntType(), name="n").with_const(width),),
    )
    target = _qubit("t")
    operands = [controls, target]
    mcx = SymbolicControlledU(
        operands=operands,
        results=[v.next_version() for v in operands],
        num_controls=loop_var,
        num_control_args=1,
        block=Block(operations=[_fixed_gate(GateOperationType.X, 1)]),
    )
    loop = ForOperation(
        operands=[
            Value(type=UIntType(), name="start").with_const(0),
            Value(type=UIntType(), name="stop").with_const(width),
            Value(type=UIntType(), name="step").with_const(1),
        ],
        results=[],
        loop_var="k",
        loop_var_value=loop_var,
        operations=[mcx],
    )
    # The symbolic walk falls back to the vector control width.
    assert _estimate([loop]) == width - 1


def test_estimate_symbolic_num_controls_falls_back_to_operand_width() -> None:
    """An unresolvable num_controls falls back to the control-prefix width."""
    controls = [_qubit(f"c{i}") for i in range(4)]
    target = _qubit("t")
    operands = [*controls, target]
    op = SymbolicControlledU(
        operands=operands,
        results=[v.next_version() for v in operands],
        num_controls=Value(type=UIntType(), name="nc"),
        num_control_args=4,
        block=Block(operations=[_fixed_gate(GateOperationType.X, 1)]),
    )
    # Four scalar control operands -> upper bound of a 4-control X.
    assert _estimate([op]) == 3


def test_estimate_symbolic_num_controls_accepts_numpy_integral_shape() -> None:
    """Operand-width fallback accepts NumPy integral shape constants."""
    num_controls = Value(type=UIntType(), name="nc")
    controls = ArrayValue(
        type=QubitType(),
        name="controls",
        shape=(Value(type=UIntType(), name="n").with_const(np.int64(4)),),
    )
    target = _qubit("t")
    operands = [controls, target]
    op = SymbolicControlledU(
        operands=operands,
        results=[v.next_version() for v in operands],
        num_controls=num_controls,
        num_control_args=1,
        block=Block(operations=[_fixed_gate(GateOperationType.X, 1)]),
    )
    # Four vector controls -> upper bound of a 4-control X.
    assert _estimate([op]) == 3


def test_estimate_symbolic_num_controls_rejects_unresolved_vector_width() -> None:
    """Operand-width fallback fails fast when vector width is unresolved."""
    num_controls = Value(type=UIntType(), name="nc")
    controls = ArrayValue(
        type=QubitType(),
        name="controls",
        shape=(Value(type=UIntType(), name="n"),),
    )
    target = _qubit("t")
    operands = [controls, target]
    op = SymbolicControlledU(
        operands=operands,
        results=[v.next_version() for v in operands],
        num_controls=num_controls,
        num_control_args=1,
        block=Block(operations=[_fixed_gate(GateOperationType.X, 1)]),
    )
    with pytest.raises(EmitError, match=r"Cannot resolve Vector\[Qubit\]"):
        _estimate([op])
