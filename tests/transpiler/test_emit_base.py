"""Tests for emit_base pass — LoopAnalyzer BinOp dependency detection.

These tests verify that ``LoopAnalyzer.should_unroll`` correctly identifies
``ForOperation`` loops containing BinOps that depend on the loop variable
(directly or inside nested control-flow), and that theta array-element
access referencing the loop variable triggers unrolling.

The second section contains integration tests that verify UInt BinOp
operations (``//``, ``**``) are correctly folded by the constant folding
pass so that the resulting constant can be used as a ``qmc.range`` argument
for loop unrolling in the emit pass.
"""

from typing import Any, TYPE_CHECKING

import numpy as np
import pytest

import qamomile.circuit as qmc
from qamomile.circuit.ir.operation.arithmetic_operations import BinOp, BinOpKind
from qamomile.circuit.ir.operation.control_flow import (
    ForOperation,
    IfOperation,
    WhileOperation,
)
from qamomile.circuit.ir.operation.gate import (
    GateOperation,
    GateOperationType,
)
from qamomile.circuit.ir.types.primitives import FloatType, QubitType, UIntType
from qamomile.circuit.ir.value import ArrayValue, Value
from qamomile.circuit.transpiler.passes.emit_base import LoopAnalyzer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _qubit(name: str = "q") -> Value:
    """Create a qubit Value."""
    return Value(type=QubitType(), name=name)


def _float_val(name: str = "theta", *, const: float | None = None) -> Value:
    """Create a float Value, optionally constant."""
    params: dict[str, object] = {}
    if const is not None:
        params["const"] = const
    return Value(type=FloatType(), name=name, params=params)


def _uint_val(name: str, *, const: int | None = None) -> Value:
    """Create a UInt Value, optionally constant."""
    params: dict[str, object] = {}
    if const is not None:
        params["const"] = const
    return Value(type=UIntType(), name=name, params=params)


def _make_binop(
    lhs: Value,
    rhs: Value,
    kind: BinOpKind,
    result_name: str = "binop_result",
) -> tuple[BinOp, Value]:
    """Create a BinOp and its result Value.

    The result type is inferred from the left-hand operand.
    """
    result = Value(type=lhs.type, name=result_name)
    op = BinOp(
        operands=[lhs, rhs],
        results=[result],
        kind=kind,
    )
    return op, result


def _make_gate(
    gate_type: GateOperationType,
    qubits: list[Value],
    theta: float | Value | None = None,
) -> GateOperation:
    """Create a GateOperation with given qubits and theta."""
    results = [q.next_version() for q in qubits]
    return GateOperation(
        operands=qubits,
        results=results,
        gate_type=gate_type,
        theta=theta,
    )


# ===========================================================================
# LoopAnalyzer._has_loop_var_binop
# ===========================================================================


class TestLoopAnalyzerBinOp:
    """Tests for LoopAnalyzer detecting BinOps dependent on loop variables."""

    def setup_method(self) -> None:
        self.analyzer = LoopAnalyzer()

    def test_direct_binop_dependency_triggers_unroll(self) -> None:
        """A BinOp using the loop variable directly should trigger unrolling."""
        loop_var_val = _uint_val("i")
        const_val = _float_val("c", const=0.5)
        binop, _ = _make_binop(loop_var_val, const_val, BinOpKind.MUL)

        q = _qubit()
        gate = _make_gate(GateOperationType.RZ, [q], theta=0.1)

        start = _uint_val("start", const=0)
        stop = _uint_val("stop", const=5)
        step = _uint_val("step", const=1)

        for_op = ForOperation(
            operands=[start, stop, step],
            results=[],
            loop_var="i",
            operations=[binop, gate],
        )

        assert self.analyzer.should_unroll(for_op, {}) is True

    def test_no_binop_no_unroll(self) -> None:
        """A loop with no BinOps and no array access should not unroll."""
        q = _qubit()
        gate = _make_gate(GateOperationType.H, [q])

        start = _uint_val("start", const=0)
        stop = _uint_val("stop", const=3)
        step = _uint_val("step", const=1)

        for_op = ForOperation(
            operands=[start, stop, step],
            results=[],
            loop_var="i",
            operations=[gate],
        )

        assert self.analyzer.should_unroll(for_op, {}) is False

    def test_binop_not_using_loop_var_no_unroll(self) -> None:
        """A BinOp not referencing the loop variable should not trigger unrolling."""
        a = _float_val("a", const=1.0)
        b = _float_val("b", const=2.0)
        binop, _ = _make_binop(a, b, BinOpKind.ADD)

        q = _qubit()
        gate = _make_gate(GateOperationType.H, [q])

        start = _uint_val("start", const=0)
        stop = _uint_val("stop", const=3)
        step = _uint_val("step", const=1)

        for_op = ForOperation(
            operands=[start, stop, step],
            results=[],
            loop_var="i",
            operations=[binop, gate],
        )

        assert self.analyzer.should_unroll(for_op, {}) is False

    def test_binop_in_nested_for_triggers_unroll(self) -> None:
        """A BinOp inside a nested ForOperation referencing the outer loop var."""
        loop_var_val = _uint_val("i")
        const_val = _float_val("c", const=0.1)
        binop, _ = _make_binop(loop_var_val, const_val, BinOpKind.MUL)

        inner_start = _uint_val("is", const=0)
        inner_stop = _uint_val("ie", const=2)
        inner_step = _uint_val("ist", const=1)

        inner_for = ForOperation(
            operands=[inner_start, inner_stop, inner_step],
            results=[],
            loop_var="j",
            operations=[binop],
        )

        start = _uint_val("start", const=0)
        stop = _uint_val("stop", const=3)
        step = _uint_val("step", const=1)

        for_op = ForOperation(
            operands=[start, stop, step],
            results=[],
            loop_var="i",
            operations=[inner_for],
        )

        assert self.analyzer.should_unroll(for_op, {}) is True

    def test_binop_in_if_true_branch_triggers_unroll(self) -> None:
        """A BinOp in the true branch of an IfOperation triggers unrolling."""
        loop_var_val = _uint_val("i")
        const_val = _float_val("c", const=2.0)
        binop, _ = _make_binop(loop_var_val, const_val, BinOpKind.ADD)

        cond = Value(type=FloatType(), name="cond")
        if_op = IfOperation(
            operands=[cond],
            results=[],
            true_operations=[binop],
            false_operations=[],
        )

        start = _uint_val("start", const=0)
        stop = _uint_val("stop", const=3)
        step = _uint_val("step", const=1)

        for_op = ForOperation(
            operands=[start, stop, step],
            results=[],
            loop_var="i",
            operations=[if_op],
        )

        assert self.analyzer.should_unroll(for_op, {}) is True

    def test_binop_in_if_false_branch_triggers_unroll(self) -> None:
        """A BinOp in the false branch of an IfOperation triggers unrolling."""
        loop_var_val = _uint_val("i")
        const_val = _float_val("c", const=3.0)
        binop, _ = _make_binop(loop_var_val, const_val, BinOpKind.SUB)

        cond = Value(type=FloatType(), name="cond")
        if_op = IfOperation(
            operands=[cond],
            results=[],
            true_operations=[],
            false_operations=[binop],
        )

        start = _uint_val("start", const=0)
        stop = _uint_val("stop", const=3)
        step = _uint_val("step", const=1)

        for_op = ForOperation(
            operands=[start, stop, step],
            results=[],
            loop_var="i",
            operations=[if_op],
        )

        assert self.analyzer.should_unroll(for_op, {}) is True

    def test_binop_in_while_triggers_unroll(self) -> None:
        """A BinOp inside a WhileOperation referencing loop var triggers unrolling."""
        loop_var_val = _uint_val("i")
        const_val = _float_val("c", const=1.0)
        binop, _ = _make_binop(loop_var_val, const_val, BinOpKind.MUL)

        cond = Value(type=FloatType(), name="cond")
        while_op = WhileOperation(
            operands=[cond],
            results=[],
            operations=[binop],
        )

        start = _uint_val("start", const=0)
        stop = _uint_val("stop", const=3)
        step = _uint_val("step", const=1)

        for_op = ForOperation(
            operands=[start, stop, step],
            results=[],
            loop_var="i",
            operations=[while_op],
        )

        assert self.analyzer.should_unroll(for_op, {}) is True


# ===========================================================================
# LoopAnalyzer — theta array element access
# ===========================================================================


class TestLoopAnalyzerThetaArrayAccess:
    """Tests that theta array element access with loop var triggers unrolling."""

    def setup_method(self) -> None:
        self.analyzer = LoopAnalyzer()

    def test_theta_array_element_with_loop_var_triggers_unroll(self) -> None:
        """Gate with theta = gammas[i] where i is the loop var should unroll."""
        gammas_array = ArrayValue(type=FloatType(), name="gammas")
        loop_idx = Value(type=UIntType(), name="i")
        theta_elem = Value(
            type=FloatType(),
            name="gammas_i",
            parent_array=gammas_array,
            element_indices=(loop_idx,),
        )

        q = _qubit()
        gate = _make_gate(GateOperationType.RZ, [q], theta=theta_elem)

        start = _uint_val("start", const=0)
        stop = _uint_val("stop", const=3)
        step = _uint_val("step", const=1)

        for_op = ForOperation(
            operands=[start, stop, step],
            results=[],
            loop_var="i",
            operations=[gate],
        )

        assert self.analyzer.should_unroll(for_op, {}) is True

    def test_theta_array_element_without_loop_var_no_unroll(self) -> None:
        """Gate with theta = gammas[0] (constant index) should not unroll."""
        gammas_array = ArrayValue(type=FloatType(), name="gammas")
        const_idx = _uint_val("idx", const=0)
        theta_elem = Value(
            type=FloatType(),
            name="gammas_0",
            parent_array=gammas_array,
            element_indices=(const_idx,),
        )

        q = _qubit()
        gate = _make_gate(GateOperationType.RZ, [q], theta=theta_elem)

        start = _uint_val("start", const=0)
        stop = _uint_val("stop", const=3)
        step = _uint_val("step", const=1)

        for_op = ForOperation(
            operands=[start, stop, step],
            results=[],
            loop_var="i",
            operations=[gate],
        )

        assert self.analyzer.should_unroll(for_op, {}) is False


# ===========================================================================
# Integration tests — UInt BinOp (floordiv, pow) folding into loop bounds
# ===========================================================================

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit

from qamomile.qiskit.transpiler import QiskitTranspiler  # noqa: E402


# ---------------------------------------------------------------------------
# Module-level @qkernel definitions (required for inspect.getsource)
# ---------------------------------------------------------------------------


@qmc.qkernel
def binop_floordiv_circuit(
    n: qmc.UInt, theta: qmc.Float
) -> qmc.Vector[qmc.Bit]:
    """Apply RX(theta) to first n // 2 qubits of a 4-qubit register."""
    q = qmc.qubit_array(4, "q")
    count = n // 2
    for i in qmc.range(count):
        q[i] = qmc.rx(q[i], angle=theta)
    return qmc.measure(q)


@qmc.qkernel
def binop_pow_circuit(
    n: qmc.UInt, theta: qmc.Float
) -> qmc.Vector[qmc.Bit]:
    """Apply RX(theta) to first n ** 2 qubits of a 4-qubit register."""
    q = qmc.qubit_array(4, "q")
    count = n ** 2
    for i in qmc.range(count):
        q[i] = qmc.rx(q[i], angle=theta)
    return qmc.measure(q)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _transpile_and_get_circuit(
    kernel: Any,
    bindings: dict[str, Any] | None = None,
) -> tuple[Any, "QuantumCircuit"]:
    """Transpile a kernel and return ``(executable, circuit)``."""
    transpiler = QiskitTranspiler()
    exe = transpiler.transpile(kernel, bindings=bindings)
    qc = exe.compiled_quantum[0].circuit
    return exe, qc


def _extract_rx_angles(qc: "QuantumCircuit") -> list[float]:
    """Extract all RX gate angles from a Qiskit circuit."""
    return [
        float(inst.operation.params[0])
        for inst in qc.data
        if inst.operation.name == "rx"
    ]


# ---------------------------------------------------------------------------
# Tests — UInt BinOp (floordiv, pow) folding into loop bounds
# ---------------------------------------------------------------------------


class TestUIntBinOpFolding:
    """Tests for UInt BinOp kinds (``//``, ``**``) that affect loop bounds.

    At tracing time ``n`` is a symbolic ``UInt`` handle, so ``n // 2``
    emits a ``BinOp(FLOORDIV)`` into the IR.  The constant folding pass
    resolves the parameter binding and evaluates the BinOp, producing
    a concrete loop bound that the emit pass can unroll.
    """

    @pytest.mark.parametrize(
        "n, theta, expected_rx_count",
        [
            (4, 0.5, 2),   # 4 // 2 = 2
            (6, 0.3, 3),   # 6 // 2 = 3
            (2, 1.0, 1),   # 2 // 2 = 1
        ],
        ids=["4//2=2", "6//2=3", "2//2=1"],
    )
    def test_floordiv_loop_bound(
        self, n: int, theta: float, expected_rx_count: int
    ) -> None:
        """``n // 2`` correctly folded as loop bound; angles verified."""
        _, qc = _transpile_and_get_circuit(
            binop_floordiv_circuit, bindings={"n": n, "theta": theta}
        )
        rx_angles = _extract_rx_angles(qc)

        assert len(rx_angles) == expected_rx_count, (
            f"Expected {expected_rx_count} RX gates (n={n}, n//2={n // 2}), "
            f"got {len(rx_angles)}"
        )
        for i, angle in enumerate(rx_angles):
            assert np.isclose(angle, theta), (
                f"RX[{i}] angle {angle} != expected {theta}"
            )

    @pytest.mark.parametrize(
        "n, theta, expected_rx_count",
        [
            (2, 0.3, 4),   # 2 ** 2 = 4
            (1, 0.5, 1),   # 1 ** 2 = 1
        ],
        ids=["2**2=4", "1**2=1"],
    )
    def test_pow_loop_bound(
        self, n: int, theta: float, expected_rx_count: int
    ) -> None:
        """``n ** 2`` correctly folded as loop bound; angles verified."""
        _, qc = _transpile_and_get_circuit(
            binop_pow_circuit, bindings={"n": n, "theta": theta}
        )
        rx_angles = _extract_rx_angles(qc)

        assert len(rx_angles) == expected_rx_count, (
            f"Expected {expected_rx_count} RX gates (n={n}, n**2={n ** 2}), "
            f"got {len(rx_angles)}"
        )
        for i, angle in enumerate(rx_angles):
            assert np.isclose(angle, theta), (
                f"RX[{i}] angle {angle} != expected {theta}"
            )
