"""Regression tests for call-boundary borrow aliasing (CB-1 / CB-2).

Two shapes used to hand user code two "independent" quantum handles that
were physically one wire, with no diagnostic — measuring both gave only
(0,0)/(1,1), perfect shot-by-shot correlation:

- CB-1: a borrowed element and its parent register passed as two
  arguments of ONE sub-kernel call. Now rejected at the call boundary by
  the footprint-aware ``reject_aliased_quantum_args`` ("overlapping
  physical region" ``QubitConsumedError``) — pinned here on the exact
  borrowed-element-plus-parent shape.
- CB-2: a callee borrowing an element and returning it ALONGSIDE its
  parent (``e = arr[0]; e = h(e); return e, arr``). The trace-end borrow
  validator (``_validate_returned_arrays``) existed only on the
  ``func_to_block`` path; the ``create_traced_block`` path (``.build()``
  and call-time specialization) never ran it, so the alias escaped and
  the caller double-measured the slot. This suite's fix wires the
  validator into ``create_traced_block``: co-returning the borrowed
  element does not exempt the parent, and the documented
  ``return arr``-only shape now raises on ``.build()`` too.

Legal spellings (write back, disjoint slots) are pinned by execution,
not just transpile success.
"""

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.transpiler.errors import (
    QubitBorrowConflictError,
    QubitConsumedError,
    UnreturnedBorrowError,
)
from qamomile.qiskit import QiskitTranspiler


def _sample_single(kernel, bindings, shots=100):
    """Transpile, execute on Qiskit, and return the single deterministic outcome.

    Args:
        kernel (QKernel): The qkernel to compile and run.
        bindings (dict[str, int]): Compile-time bindings.
        shots (int): Number of shots. Defaults to 100.

    Returns:
        int | tuple[int, ...]: The unique measured value across all shots.
    """
    transpiler = QiskitTranspiler()
    executable = transpiler.transpile(kernel, bindings=bindings)
    result = executable.sample(transpiler.executor(), shots=shots).result()
    assert len(result.results) == 1, f"expected deterministic outcome: {result}"
    return result.results[0][0]


@qmc.qkernel
def _two_param_sub(e: qmc.Qubit, arr: qmc.Vector[qmc.Qubit]) -> tuple[qmc.Bit, qmc.Bit]:
    e = qmc.h(e)
    b1 = qmc.measure(e)
    a0 = arr[0]
    b2 = qmc.measure(a0)
    return b1, b2


@qmc.qkernel
def _co_return_sub(
    arr: qmc.Vector[qmc.Qubit],
) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit]]:
    e = arr[0]
    e = qmc.h(e)
    return e, arr


class TestCoArgumentAliasingRejected:
    """CB-1: element + parent co-arguments raise at the call boundary."""

    def test_element_and_parent_same_call_rejected(self):
        """The CB-1 shape is rejected instead of compiling with both callee
        params on one wire. The call-boundary footprint check
        (``reject_aliased_quantum_args``) catches the element↔parent
        overlap before argument consumption, with a dedicated message."""

        @qmc.qkernel
        def caller(dummy: qmc.UInt) -> tuple[qmc.Bit, qmc.Bit]:
            qs = qmc.qubit_array(2, "qs")
            e = qs[0]
            b1, b2 = _two_param_sub(e, qs)
            return b1, b2

        with pytest.raises(QubitConsumedError, match="overlapping physical region"):
            QiskitTranspiler().transpile(caller, bindings={"dummy": 0})

    def test_parent_with_outstanding_borrow_rejected(self):
        """Deferred CB-1 pin: passing the parent while an un-passed element
        borrow is outstanding keeps raising (pre-existing behavior the
        co-argument case now matches)."""

        @qmc.qkernel
        def whole_sub(arr: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
            a1 = arr[1]
            a1 = qmc.x(a1)
            arr[1] = a1
            return arr

        @qmc.qkernel
        def caller(dummy: qmc.UInt) -> tuple[qmc.Bit, qmc.Bit]:
            qs = qmc.qubit_array(2, "qs")
            e = qs[0]
            qs2 = whole_sub(qs)
            b1 = qmc.measure(e)
            e2 = qs2[0]
            b2 = qmc.measure(e2)
            return b1, b2

        with pytest.raises(UnreturnedBorrowError, match="unreturned borrowed"):
            QiskitTranspiler().transpile(caller, bindings={"dummy": 0})

    def test_overlapping_views_rejected_at_creation(self):
        """Pin: overlapping sibling views (``qs[0:2]`` and ``qs[1:3]``) are
        already rejected when the second view is created — no dedicated
        call-boundary footprint check is needed for them."""

        @qmc.qkernel
        def two_views(
            a: qmc.Vector[qmc.Qubit], b: qmc.Vector[qmc.Qubit]
        ) -> tuple[qmc.Bit, qmc.Bit]:
            x = a[0]
            b1 = qmc.measure(x)
            y = b[0]
            b2 = qmc.measure(y)
            return b1, b2

        @qmc.qkernel
        def caller(dummy: qmc.UInt) -> tuple[qmc.Bit, qmc.Bit]:
            qs = qmc.qubit_array(3, "qs")
            b1, b2 = two_views(qs[0:2], qs[1:3])
            return b1, b2

        with pytest.raises(QubitBorrowConflictError):
            QiskitTranspiler().transpile(caller, bindings={"dummy": 0})

    def test_identical_duplicate_arguments_rejected(self):
        """Pin: the existing uuid check keeps rejecting ``sub(q, q)``."""

        @qmc.qkernel
        def pair(a: qmc.Qubit, b: qmc.Qubit) -> tuple[qmc.Bit, qmc.Bit]:
            a, b = qmc.cx(a, b)
            return qmc.measure(a), qmc.measure(b)

        @qmc.qkernel
        def caller(dummy: qmc.UInt) -> tuple[qmc.Bit, qmc.Bit]:
            q = qmc.qubit("q")
            b1, b2 = pair(q, q)
            return b1, b2

        with pytest.raises(QubitConsumedError):
            QiskitTranspiler().transpile(caller, bindings={"dummy": 0})


class TestCoReturnedBorrowRejected:
    """CB-2: a still-borrowed element returned alongside its parent raises."""

    def test_co_returned_borrow_rejected_at_standalone_build(self):
        """The callee alone fails its build: co-returning the borrowed
        element does not exempt the parent's outstanding borrow."""

        with pytest.raises(UnreturnedBorrowError, match="unreturned borrowed"):
            _co_return_sub.build()

    def test_co_returned_borrow_rejected_through_caller(self):
        """The call-time specialization path validates too — the CB-2
        caller can no longer double-measure the slot."""

        @qmc.qkernel
        def caller(dummy: qmc.UInt) -> tuple[qmc.Bit, qmc.Bit]:
            qs = qmc.qubit_array(1, "qs")
            e, qs2 = _co_return_sub(qs)
            b1 = qmc.measure(e)
            e2 = qs2[0]
            b2 = qmc.measure(e2)
            return b1, b2

        with pytest.raises(UnreturnedBorrowError, match="unreturned borrowed"):
            QiskitTranspiler().transpile(caller, bindings={"dummy": 0})

    def test_parent_only_return_with_outstanding_borrow_rejected(self):
        """The documented validator target (``e = arr[0]; return arr``
        without returning e) raises on the ``.build()`` path as well —
        previously only the ``.block`` path checked it."""

        @qmc.qkernel
        def leak(arr: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
            e = arr[0]
            e = qmc.h(e)
            return arr

        with pytest.raises(UnreturnedBorrowError, match="unreturned borrowed"):
            leak.build()


class TestLegalSpellingsStillExecute:
    """The sanctioned flows keep compiling AND executing correctly."""

    def test_writeback_then_pass_whole_array_executes(self):
        """Write the element back, then pass the whole register: slot 0
        gets X twice (identity), slot 1 gets X once — measures (0, 1)."""

        @qmc.qkernel
        def flip_all(arr: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
            for i in qmc.range(2):
                arr[i] = qmc.x(arr[i])
            return arr

        @qmc.qkernel
        def caller(dummy: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            qs = qmc.qubit_array(2, "qs")
            e = qs[0]
            e = qmc.x(e)
            qs[0] = e
            qs = flip_all(qs)
            return qmc.measure(qs)

        assert _sample_single(caller, {"dummy": 0}) == (0, 1)

    def test_disjoint_elements_as_co_arguments_execute(self):
        """Two DIFFERENT slots of one register as co-arguments stay legal:
        X on the first then CX gives deterministic (1, 1)."""

        @qmc.qkernel
        def cx_pair(a: qmc.Qubit, b: qmc.Qubit) -> tuple[qmc.Bit, qmc.Bit]:
            a = qmc.x(a)
            a, b = qmc.cx(a, b)
            return qmc.measure(a), qmc.measure(b)

        @qmc.qkernel
        def caller(dummy: qmc.UInt) -> tuple[qmc.Bit, qmc.Bit]:
            qs = qmc.qubit_array(2, "qs")
            b1, b2 = cx_pair(qs[0], qs[1])
            return b1, b2

        assert _sample_single(caller, {"dummy": 0}) == (1, 1)

    def test_callee_internal_writeback_return_parent_executes(self):
        """A callee that borrows, writes back, and returns the parent has a
        clean borrow table — the new trace-end validation must not
        false-positive. X on slot 0 measures (1,)."""

        @qmc.qkernel
        def flip0(arr: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
            e = arr[0]
            e = qmc.x(e)
            arr[0] = e
            return arr

        @qmc.qkernel
        def caller(dummy: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            qs = qmc.qubit_array(1, "qs")
            qs = flip0(qs)
            return qmc.measure(qs)

        assert _sample_single(caller, {"dummy": 0}) == (1,)
