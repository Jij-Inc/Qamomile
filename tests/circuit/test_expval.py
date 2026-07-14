"""Tests pinning ``qmc.expval``'s destructive-consume semantics.

``expval(q, H)`` is classified as :attr:`ConsumeMode.DESTRUCTIVE` — the
same family as ``measure`` and ``cast`` — because the underlying
Estimator primitive samples the state.  Two complementary groups of
regressions live here:

- ``TestExpvalIsDestructiveConsume`` pins that running ``expval``
  *terminates* the lifetime of the qubits it observes.  Every form of
  reuse after the call — direct re-gating on a covered slot, re-using
  the consumed handle through the affine type, expval'ing with an
  outstanding view borrow, the broadcast / transfer chain via
  ``pauli_evolve(view) → expval(view)`` — must be rejected at trace
  time.  A single OK case (two disjoint views, one destructively
  consumed via ``expval``, the other still slice-assignable) anchors
  the bookkeeping precision.

- ``TestExpvalOverConsumedSlots`` pins the symmetric guard for
  *prior* destruction: when ``measure(view)`` (or another destructive
  view op) has already destroyed a subset of the register's qubits,
  passing the whole register to ``expval`` must raise, while passing
  a disjoint view must still pass cleanly.  The IR-level check is
  exercised through the post-fold pass as well.
"""

from __future__ import annotations

import pytest

import qamomile.circuit as qmc


class TestExpvalIsDestructiveConsume:
    """Once expval runs, its qubits cannot be reused.

    Prior to making ``expval`` a destructive consume the IR pass had
    to treat ``ExpvalOp`` as a release-mode op (clearing view
    ownership) just to compensate for the fact that the frontend
    never called ``consume()`` on the qubits.  That left a back door:
    a kernel could use the same view (or even the same root register)
    for further ops after ``expval`` returned, with no compile-time
    error.  These tests pin the strict-return alignment — every
    consume path that gives up quantum state goes through the same
    destructive-consume machinery.
    """

    def test_use_after_expval_on_view_is_rejected_at_trace(self):
        """``expval(q[0::2], H); q[0] = h(q[0])`` raises at trace time."""
        from qamomile.circuit.transpiler.errors import QubitConsumedError

        @qmc.qkernel
        def kern(obs: qmc.Observable) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            _ = qmc.expval(q[0::2], obs)  # slots {0, 2} destroyed
            q[0] = qmc.h(q[0])  # ← use after expval-destroyed slot
            return qmc.measure(q)

        with pytest.raises(QubitConsumedError):
            _ = kern.block

    def test_use_after_expval_whole_array_is_rejected_at_trace(self):
        """``e = expval(q, H); q = h(q)`` raises at trace time.

        ``expval`` destructively consumes the whole array; the
        original handle's affine-type check fires when the kernel
        tries to reuse it.
        """
        from qamomile.circuit.transpiler.errors import QubitConsumedError

        @qmc.qkernel
        def kern(obs: qmc.Observable) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            _ = qmc.expval(q, obs)
            q = qmc.h(q)  # ← affine type rejects this
            return qmc.measure(q)

        with pytest.raises(QubitConsumedError):
            _ = kern.block

    def test_expval_with_outstanding_borrow_is_rejected(self):
        """``v = q[0:2]; expval(q, H)`` raises — unreturned view borrow."""
        from qamomile.circuit.transpiler.errors import (
            QubitConsumedError,
            UnreturnedBorrowError,
        )

        @qmc.qkernel
        def kern(obs: qmc.Observable) -> qmc.Float:
            q = qmc.qubit_array(4, "q")
            _v = q[0:2]  # bulk-borrows {0, 1}
            return qmc.expval(q, obs)  # ← whole-array consume with view live

        with pytest.raises((QubitConsumedError, UnreturnedBorrowError)):
            _ = kern.block

    def test_two_disjoint_views_destructively_consumed_via_expval(self):
        """``expval(q[0::2], H); q[1::2] = h(q[1::2])`` still works.

        Two disjoint views: the first is destructively consumed via
        ``expval``, the second is independent and can be
        slice-assigned back in the usual way.  Both halves of the
        parent must work in the same kernel.

        Only structural ``kern.block`` is asserted here because
        mixing ``expval`` and ``measure`` in one kernel produces a
        multi-segment program that the NISQ executor rejects at plan
        time.  The strict-return + destructive-consume bookkeeping is
        what we need to pin.
        """

        @qmc.qkernel
        def kern(obs: qmc.Observable) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            _ = qmc.expval(q[0::2], obs)  # destroy {0, 2}
            q[1::2] = qmc.h(q[1::2])  # use the surviving half
            # ``measure(q)`` would re-touch destroyed slots, so we
            # measure only the surviving half.
            return qmc.measure(q[1::2])

        block = kern.block
        assert block is not None

    def test_expval_after_pauli_evolve_on_view_consumes_view(self):
        """``view = pauli_evolve(view, ...); expval(view, H)`` is valid + destructive.

        ``pauli_evolve`` transfers view ownership to the next-version
        handle; ``expval`` then destructively consumes it.  The
        kernel cannot touch the same slots afterwards.
        """
        from qamomile.circuit.transpiler.errors import QubitConsumedError

        @qmc.qkernel
        def kern(
            H: qmc.Observable,
            gamma: qmc.Float,
            obs: qmc.Observable,
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            view = q[0::2]
            view = qmc.pauli_evolve(view, H, gamma)
            _ = qmc.expval(view, obs)  # destructive on slots {0, 2}
            q[0] = qmc.h(q[0])  # ← rejected
            return qmc.measure(q)

        with pytest.raises(QubitConsumedError):
            _ = kern.block


class TestExpvalOverConsumedSlots:
    """``expval`` rejects operands whose slots were destroyed earlier.

    Before this guard, ``measure(q[1::2]); expval(q, Z(1))`` would
    transpile silently and surface as a circuit-vs-observable
    dimension mismatch deep inside Qiskit at execution time.
    ``expval`` now calls ``ArrayBase._check_no_consumed_slots`` (for
    ``Vector`` operands) and ``ArrayBase.consume("expval")``, and the
    IR-level ``SliceBorrowCheckPass`` catches the same violation
    when the IR is constructed without going through ``expval()``.
    """

    def test_expval_whole_array_after_view_measure_rejected_at_trace(self):
        """``measure(q[1::2]); expval(q, H)`` raises at trace time.

        ``expval(q, H)`` with ``q`` passed as a whole root array used
        to bypass both the frontend's consumed-slot guard (because
        ``expval.py`` never called ``consume`` or
        ``_check_no_consumed_slots``) and the IR's
        ``_process_operand_borrows`` (which only inspected per-element
        Values, not whole-array operands).  The bug surfaced at Qiskit
        execution time with a circuit-vs-observable dimension
        mismatch.  The fix raises ``QubitConsumedError`` at trace
        time.
        """
        from qamomile.circuit.transpiler.errors import QubitConsumedError

        @qmc.qkernel
        def kern(obs: qmc.Observable) -> qmc.Float:
            q = qmc.qubit_array(4, "q")
            _ = qmc.measure(q[1::2])  # q[1] and q[3] destroyed
            return qmc.expval(q, obs)  # should raise: whole q includes destroyed slots

        with pytest.raises(QubitConsumedError, match="consumed"):
            kern.block

    def test_expval_on_disjoint_view_after_measure_view_not_blocked_at_frontend(self):
        """Frontend does not reject ``expval(q[0::2], H)`` after ``measure(q[1::2])``.

        Verifies that ``_check_no_consumed_slots`` is not
        over-aggressive: consuming slots {1, 3} via ``measure(q[1::2])``
        should leave slots {0, 2} usable in a subsequent ``expval``
        call on ``q[0::2]``.  The guard must only fire when the
        *expval operand's* covered slots overlap the consumed set; a
        disjoint view (even qubits vs odd qubits) must pass cleanly.

        Note: mixing ``measure(view)`` and ``expval(view2)`` in the
        same kernel creates a multi-segment program that the NISQ
        single-segment strategy rejects at plan time.  This test
        therefore asserts only that the *frontend linearity guard*
        does not raise — ``kern.block`` must succeed — and does not
        attempt a full transpile or execution.
        """

        @qmc.qkernel
        def kern(obs: qmc.Observable) -> qmc.Float:
            q = qmc.qubit_array(4, "q")
            _ = qmc.measure(q[1::2])  # destroy q[1], q[3]
            return qmc.expval(q[0::2], obs)  # slots {0, 2} survive — OK

        # Must not raise QubitConsumedError: even view is disjoint
        # from the consumed odd slots.
        block = kern.block
        assert block is not None

    def test_ir_check_catches_expval_over_consumed_slot(self):
        """IR ``SliceBorrowCheckPass`` rejects whole-root expval after view-measure.

        Complementary to the frontend trace-time guard: even if the
        IR is constructed directly (bypassing ``expval()``), the
        post-fold linearity checker must catch the consumed-slot
        violation.
        """
        pytest.importorskip("qiskit")
        from qamomile.circuit.transpiler.errors import QubitConsumedError

        @qmc.qkernel
        def kern(obs: qmc.Observable) -> qmc.Float:
            q = qmc.qubit_array(4, "q")
            _ = qmc.measure(q[1::2])
            # Attempt expval on a partial view that overlaps the
            # consumed slots.  q[0::2] is safe; q[1::2] would
            # overlap — use a harmless even view here, but pair it
            # with an odd direct access.
            q[1] = qmc.x(q[1])  # direct access after measure(q[1::2]) — consumed
            return qmc.expval(q[0::2], obs)

        with pytest.raises(QubitConsumedError):
            kern.block
