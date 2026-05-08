"""Tests for self-recursive ``@qkernel`` support.

The recursion is resolved by the transpiler's inline ↔ partial_eval
fixed-point loop: each iteration unrolls one layer of self-referential
``CallBlockOperation`` and then folds the base-case ``if`` under the
provided bindings.  When the recursion driver is not concretized by the
bindings the self-call is left in the IR; when it is concrete but the
recursion does not terminate the loop raises ``FrontendTransformError``.
"""

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.ir.operation.call_block_ops import CallBlockOperation
from qamomile.circuit.transpiler.errors import FrontendTransformError
from qamomile.qiskit import QiskitTranspiler


@qmc.qkernel
def _leaf(q: qmc.Qubit) -> qmc.Qubit:
    return qmc.h(q)


@qmc.qkernel
def _rec(k: qmc.UInt, q: qmc.Qubit) -> qmc.Qubit:
    if k == 0:
        q = _leaf(q)
    else:
        q = _rec(k - 1, q)
    return q


@qmc.qkernel
def _outer_of_rec(k: qmc.UInt) -> qmc.Bit:
    q = qmc.qubit(name="q")
    q = _rec(k, q)
    return qmc.measure(q)


@qmc.qkernel
def _non_terminating(k: qmc.UInt, q: qmc.Qubit) -> qmc.Qubit:
    if k == 0:
        q = _leaf(q)
    else:
        q = _non_terminating(k + 1, q)
    return q


@qmc.qkernel
def _outer_of_non_terminating(k: qmc.UInt) -> qmc.Bit:
    q = qmc.qubit(name="q")
    q = _non_terminating(k, q)
    return qmc.measure(q)


def test_build_of_self_recursive_kernel_succeeds():
    """A self-recursive kernel builds into a hierarchical block with a
    self-referential CallBlockOperation inside its body."""
    block = _rec.block
    self_refs = 0
    pending = [block.operations]
    while pending:
        ops = pending.pop()
        for op in ops:
            if isinstance(op, CallBlockOperation) and op.block is block:
                self_refs += 1
            if hasattr(op, "nested_op_lists"):
                for body in op.nested_op_lists():
                    pending.append(body)
    assert self_refs >= 1
    assert not _rec._pending_self_calls


def test_transpile_with_concrete_driver_succeeds():
    """When bindings make the recursion driver concrete, the transpiler
    unrolls and produces a flat circuit."""
    tr = QiskitTranspiler()
    for k in (0, 1, 3):
        exe = tr.transpile(_outer_of_rec, bindings={"k": k})
        circuit = exe.compiled_quantum[0].circuit
        h_count = sum(1 for instr in circuit.data if instr.operation.name == "h")
        assert h_count == 1, f"k={k}: expected 1 H gate, got {h_count}"


def test_non_terminating_recursion_raises():
    """A recursion whose driver never reaches the base case raises
    ``FrontendTransformError`` after MAX_UNROLL_DEPTH iterations."""
    tr = QiskitTranspiler()
    with pytest.raises(FrontendTransformError, match="did not terminate"):
        tr.transpile(_outer_of_non_terminating, bindings={"k": 3})


def test_non_recursive_kernel_still_works():
    """Regression: forward-ref machinery must not affect non-recursive
    kernels."""
    tr = QiskitTranspiler()

    @qmc.qkernel
    def _simple(q: qmc.Qubit) -> qmc.Qubit:
        return qmc.h(q)

    @qmc.qkernel
    def _outer_simple() -> qmc.Bit:
        q = qmc.qubit(name="q")
        q = _simple(q)
        return qmc.measure(q)

    exe = tr.transpile(_outer_simple)
    circuit = exe.compiled_quantum[0].circuit
    assert any(instr.operation.name == "h" for instr in circuit.data)
    assert any(instr.operation.name == "measure" for instr in circuit.data)
