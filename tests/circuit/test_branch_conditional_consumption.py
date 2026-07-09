"""Regression tests for conditional-consumption affine soundness (S1).

Affine typing forbids using a quantum value more than once. A qubit
consumed (measured/projected) inside a runtime ``if`` / ``else`` branch and
then reused *after* the branch used to compile silently: the phi merge
carried the consumed value forward as a fresh, live handle, so a later
``q = qmc.h(q)`` produced a circuit that applied a gate to an
already-measured wire on the branch that consumed it. Written without the
enclosing ``if`` the identical program raises ``QubitConsumedError`` at
trace time — the ``if`` was the only thing hiding the violation. This is the
dual of the branch-*discard* problem covered in
``test_branch_quantum_discard.py``: discard = consumed-by-zero-paths,
conditional consumption = consumed-on-one-path-and-reused.

The frontend now applies the conditional-move rule (as in Rust's borrow
checker): a quantum value consumed on *any* branch of an if/else is treated
as consumed after the merge, so reusing it raises ``QubitConsumedError``
with a message that names the consuming operation and branch. Dropping the
value (never using it after the if) stays legal — affinity permits drop.

Note: Do NOT use ``from __future__ import annotations`` here — the @qkernel
AST transformer relies on resolved type annotations.
"""

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.transpiler.errors import QubitConsumedError

pytest.importorskip("qiskit")

from qamomile.qiskit import QiskitTranspiler


def _transpile(kernel, bindings=None):
    """Transpile a kernel on the Qiskit backend."""
    return QiskitTranspiler().transpile(kernel, bindings=bindings or {})


# ---------------------------------------------------------------------------
# Rejected: consume on a branch, reuse after the if
# ---------------------------------------------------------------------------


def test_consume_true_branch_then_reuse_raises():
    """Measuring q in the true branch, then reusing q after, is rejected."""

    @qmc.qkernel
    def circuit() -> qmc.Bit:
        q = qmc.qubit("q")
        sel = qmc.measure(qmc.h(qmc.qubit("r")))
        if sel:
            _ = qmc.measure(q)
        q = qmc.h(q)
        return qmc.measure(q)

    with pytest.raises(QubitConsumedError, match="true branch"):
        _transpile(circuit)


def test_consume_false_branch_then_reuse_raises():
    """Consuming q only in the else branch, then reusing q, is rejected."""

    @qmc.qkernel
    def circuit() -> qmc.Bit:
        q = qmc.qubit("q")
        sel = qmc.measure(qmc.h(qmc.qubit("r")))
        if sel:
            q = qmc.x(q)
        else:
            _ = qmc.measure(q)
        q = qmc.h(q)
        return qmc.measure(q)

    with pytest.raises(QubitConsumedError, match="false branch"):
        _transpile(circuit)


def test_consume_both_branches_then_reuse_raises():
    """Consuming q on both branches, then reusing q, is rejected."""

    @qmc.qkernel
    def circuit() -> qmc.Bit:
        q = qmc.qubit("q")
        sel = qmc.measure(qmc.h(qmc.qubit("r")))
        if sel:
            _ = qmc.measure(q)
        else:
            _ = qmc.project_z(q)
        q = qmc.h(q)
        return qmc.measure(q)

    with pytest.raises(QubitConsumedError, match="both branches"):
        _transpile(circuit)


def test_consume_in_nested_if_then_reuse_raises():
    """Consuming q in a nested inner branch, then reusing q, is rejected.

    The merged handle's consumed state must propagate out through both the
    inner and outer merge, and the error message must not duplicate the
    conditional-consumption suffix.
    """

    @qmc.qkernel
    def circuit() -> qmc.Bit:
        q = qmc.qubit("q")
        s1 = qmc.measure(qmc.h(qmc.qubit("r1")))
        s2 = qmc.measure(qmc.h(qmc.qubit("r2")))
        if s1:
            if s2:
                _ = qmc.measure(q)
        q = qmc.h(q)
        return qmc.measure(q)

    with pytest.raises(QubitConsumedError) as exc_info:
        _transpile(circuit)
    message = str(exc_info.value)
    # The conditional-consumption suffix appears exactly once (no doubling).
    assert message.count("of the preceding if/else") == 1


def test_error_message_names_consuming_operation():
    """The affine error identifies the consuming op and points at reuse."""

    @qmc.qkernel
    def circuit() -> qmc.Bit:
        q = qmc.qubit("q")
        sel = qmc.measure(qmc.h(qmc.qubit("r")))
        if sel:
            _ = qmc.measure(q)
        q = qmc.h(q)
        return qmc.measure(q)

    with pytest.raises(QubitConsumedError) as exc_info:
        _transpile(circuit)
    message = str(exc_info.value)
    assert "measure" in message
    assert "if/else" in message
    assert "in 'H'" in message  # the reuse site


# ---------------------------------------------------------------------------
# Allowed: drop after conditional consumption, or rebinding both branches
# ---------------------------------------------------------------------------


def test_consume_in_branch_and_drop_is_allowed():
    """Consuming q in a branch and never reusing it afterwards is legal."""

    @qmc.qkernel
    def circuit() -> qmc.Bit:
        q = qmc.qubit("q")
        sel = qmc.measure(qmc.h(qmc.qubit("r")))
        out = qmc.bit(False)
        if sel:
            out = qmc.measure(q)
        else:
            q = qmc.h(q)
            out = qmc.measure(q)
        return out

    # Both branches consume q; q is never reused after the if.
    executable = _transpile(circuit)
    assert executable is not None


def test_both_branch_rebind_then_reuse_is_allowed():
    """Rebinding q with a gate in both branches, then reusing q, is legal."""

    @qmc.qkernel
    def circuit() -> qmc.Bit:
        q = qmc.qubit("q")
        sel = qmc.measure(qmc.h(qmc.qubit("r")))
        if sel:
            q = qmc.h(q)
        else:
            q = qmc.x(q)
        q = qmc.h(q)
        return qmc.measure(q)

    executable = _transpile(circuit)
    assert executable is not None


def test_no_branch_consumption_then_reuse_is_allowed():
    """A branch that only applies a gate (no consumption) leaves q reusable."""

    @qmc.qkernel
    def circuit() -> qmc.Bit:
        q = qmc.qubit("q")
        sel = qmc.measure(qmc.h(qmc.qubit("r")))
        if sel:
            q = qmc.x(q)
        q = qmc.h(q)
        return qmc.measure(q)

    executable = _transpile(circuit)
    assert executable is not None


# ---------------------------------------------------------------------------
# Vector element granularity: conditional consumption of qs[i]
# ---------------------------------------------------------------------------


def test_element_consume_in_branch_then_reuse_raises():
    """Measuring qs[0] in a branch, then reusing qs[0] after, is rejected.

    The trace-handle conditional-move rule covered scalar variables but each
    branch traces against a fresh array copy whose element-borrow table
    starts empty, so the branch's destructive element consume used to be
    dropped at the phi merge — ``if sel: _ = measure(qs[0])`` followed by
    ``qs[0] = h(qs[0])`` compiled silently. The merge now carries unreturned
    branch element borrows onto the merged handle, routing the post-if
    element access into the existing borrow enforcement.
    """
    from qamomile.circuit.transpiler.errors import QubitBorrowConflictError

    @qmc.qkernel
    def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(n, "qs")
        s = qmc.qubit("s")
        s = qmc.h(s)
        sel = qmc.measure(s)
        if sel:
            _ = qmc.measure(qs[0])
        qs[0] = qmc.h(qs[0])
        return qmc.measure(qs)

    with pytest.raises((QubitConsumedError, QubitBorrowConflictError)):
        _transpile(circuit, bindings={"n": 2})


def test_element_consume_in_else_branch_then_reuse_raises():
    """The else-branch variant of the element conditional consume is rejected."""
    from qamomile.circuit.transpiler.errors import QubitBorrowConflictError

    @qmc.qkernel
    def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(n, "qs")
        s = qmc.qubit("s")
        s = qmc.h(s)
        sel = qmc.measure(s)
        if sel:
            qs[1] = qmc.x(qs[1])
        else:
            _ = qmc.measure(qs[0])
        qs[0] = qmc.h(qs[0])
        return qmc.measure(qs)

    with pytest.raises((QubitConsumedError, QubitBorrowConflictError)):
        _transpile(circuit, bindings={"n": 2})


def test_element_gate_returned_in_branch_stays_legal():
    """An element borrowed AND returned inside the branch stays reusable.

    ``if sel: qs[0] = x(qs[0])`` releases its borrow before the branch ends,
    so the post-if ``qs[0] = h(qs[0])`` must keep compiling.
    """

    @qmc.qkernel
    def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(n, "qs")
        s = qmc.qubit("s")
        s = qmc.h(s)
        sel = qmc.measure(s)
        if sel:
            qs[0] = qmc.x(qs[0])
        qs[0] = qmc.h(qs[0])
        return qmc.measure(qs)

    executable = _transpile(circuit, bindings={"n": 2})
    assert executable is not None


def test_element_consume_in_branch_other_element_reuse_stays_legal():
    """Conditionally consuming qs[0] leaves the untouched qs[1] reusable."""

    @qmc.qkernel
    def circuit(n: qmc.UInt) -> qmc.Bit:
        qs = qmc.qubit_array(n, "qs")
        s = qmc.qubit("s")
        s = qmc.h(s)
        sel = qmc.measure(s)
        if sel:
            _ = qmc.measure(qs[0])
        qs[1] = qmc.h(qs[1])
        return qmc.measure(qs[1])

    executable = _transpile(circuit, bindings={"n": 2})
    assert executable is not None
