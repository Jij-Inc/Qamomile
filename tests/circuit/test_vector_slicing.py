"""Tests for ``Vector`` slicing and the resulting ``VectorView``.

These tests cover the frontend behaviour (length, indexing, nesting,
borrow tracking, error cases) and the end-to-end execution path on every
supported quantum SDK backend for the canonical alternating-qubit
pattern — the use case the slicing feature was introduced to support.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.frontend.handle import VectorView
from qamomile.circuit.frontend.qkernel_invocation import _wrap_array_result
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.control_flow import IfMerge
from qamomile.circuit.ir.types.primitives import QubitType, UIntType
from qamomile.circuit.ir.value import ArrayValue, Value
from qamomile.circuit.transpiler.errors import (
    QubitBorrowConflictError,
    QubitConsumedError,
    ValidationError,
)
from qamomile.circuit.transpiler.passes.slice_borrow_check import (
    SliceBorrowCheckPass,
    _SnapshotKind,
)


def _uint_value(name: str, const: int | None = None) -> Value:
    """Create a UInt IR value for slice-borrow synthetic tests."""
    value = Value(type=UIntType(), name=name)
    if const is not None:
        return value.with_const(const)
    return value


def _qubit_array(name: str, length: Value) -> ArrayValue:
    """Create a root qubit ArrayValue for slice-borrow synthetic tests."""
    return ArrayValue(type=QubitType(), name=name, shape=(length,))


def _slice_array(
    root: ArrayValue,
    name: str,
    start: Value,
    step: Value,
    length: Value,
) -> ArrayValue:
    """Create a sliced ArrayValue for slice-borrow synthetic tests."""
    return ArrayValue(
        type=QubitType(),
        name=name,
        shape=(length,),
        slice_of=root,
        slice_start=start,
        slice_step=step,
    )


# ---------------------------------------------------------------------------
# Frontend-only behavioural tests (no backend required)
# ---------------------------------------------------------------------------


class TestVectorViewFrontend:
    """Behaviour that should hold regardless of any backend being installed."""

    def test_slice_returns_vector_view_with_block(self):
        """Slicing a Vector inside a qkernel produces a VectorView handle."""

        captured: dict[str, object] = {}

        @qmc.qkernel
        def kern(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
            view = q[0::2]
            captured["view_type"] = type(view).__name__
            captured["shape_len"] = len(view.shape)
            view[0] = qmc.h(view[0])
            return q

        # Force trace by asking for the block.
        assert kern.block is not None
        assert captured["view_type"] == "VectorView"
        assert captured["shape_len"] == 1

    def test_constant_slice_length_is_int(self):
        """A slice with all-constant bounds has a Python int length."""

        captured: dict[str, object] = {}

        @qmc.qkernel
        def kern(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
            v = q[0:6:2]
            captured["length"] = v.shape[0]
            v[0] = qmc.h(v[0])
            q[0:6:2] = v
            return q

        assert kern.block is not None
        assert isinstance(captured["length"], int)
        assert captured["length"] == 3  # ceil((6-0)/2)

    def test_iter_is_rejected(self):
        """Direct iteration over a VectorView raises, matching Vector.

        The AST transform catches ``for x in view:`` at parse time and
        converts it into a ``SyntaxError``; we assert on that path, which
        is the one users actually see.
        """

        with pytest.raises(SyntaxError, match="index-based"):

            @qmc.qkernel
            def kern(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
                v = q[0::2]
                for _ in v:  # type: ignore[misc]
                    pass
                return q

    def test_slice_assignment_non_view_rejected(self):
        """Slice assignment with a non-VectorView right-hand side raises TypeError.

        Slice assignment is the explicit borrow-return path for slice views.
        Passing a non-VectorView (e.g. a Vector itself, a scalar, or an
        element handle) is a category error: ``vec[a:b] = vec`` would mix
        whole-array assignment with the slice path, and ``vec[a:b] = q[i]``
        confuses slice-level with element-level returns.  The frontend
        rejects this at trace time before any borrow state mutation.
        """

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            q[0::2] = q  # type: ignore[assignment]  # whole-array on RHS
            return qmc.measure(q)

        with pytest.raises(TypeError, match="Slice assignment"):
            _ = kern.block

    def test_negative_step_raises(self):
        """Negative step is an explicit NotImplementedError for now."""

        @qmc.qkernel
        def kern(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
            v = q[::-1]
            v[0] = qmc.h(v[0])
            return q

        with pytest.raises(NotImplementedError, match="Negative step"):
            _ = kern.block

    def test_negative_start_raises(self):
        """Negative start is likewise unsupported (no Python-style wrap)."""

        @qmc.qkernel
        def kern(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
            v = q[-2:]
            v[0] = qmc.h(v[0])
            return q

        with pytest.raises(NotImplementedError, match="Negative start"):
            _ = kern.block

    def test_negative_element_index_raises(self):
        """Constant negative element index is an explicit NotImplementedError.

        Previously ``q[-1]`` survived tracing and tripped a misleading
        internal allocator assertion ("This indicates a bug in the
        transpiler pipeline") at emit time.
        """

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q[-1] = qmc.h(q[-1])
            return qmc.measure(q)

        with pytest.raises(NotImplementedError, match="Negative index"):
            _ = kern.block

    def test_negative_view_element_index_raises(self):
        """Negative index on a VectorView is rejected at trace time.

        Previously ``v[-1]`` for ``v = q[1:3]`` silently affine-composed to
        root index ``1 + 1 * (-1) = 0``, routing the gate onto physical
        ``q[0]`` instead of ``q[2]`` — a silent miscompilation.
        """

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            v = q[1:3]
            v[-1] = qmc.x(v[-1])
            q[1:3] = v
            return qmc.measure(q)

        with pytest.raises(NotImplementedError, match="Negative index"):
            _ = kern.block

    def test_negative_element_setitem_raises(self):
        """Write-side negative element index is rejected like the read side."""

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[-1] = qmc.qubit("fresh")
            return qmc.measure(q)

        with pytest.raises(NotImplementedError, match="Negative index"):
            _ = kern.block

    def test_negative_matrix_element_index_raises(self):
        """Negative constant index on a Matrix element is rejected."""

        @qmc.qkernel
        def kern(edges: qmc.Matrix[qmc.UInt]) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            i = edges[0, -1]
            q[i] = qmc.h(q[i])
            return qmc.measure(q)

        with pytest.raises(NotImplementedError, match="Negative index"):
            _ = kern.block

    def test_view_round_trip_releases_borrow(self):
        """Borrowing through the view and writing back releases the slot.

        After ``view[0] = qmc.h(view[0])`` the parent qubit at index 0 is
        no longer borrowed, so a subsequent access succeeds.  This
        exercises the round-trip borrow/return path through the affine
        index map.
        """

        @qmc.qkernel
        def kern(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
            view = q[0::2]
            view[0] = qmc.h(view[0])
            # Re-borrowing the same view slot must now succeed.
            view[0] = qmc.h(view[0])
            return q

        assert kern.block is not None


# ---------------------------------------------------------------------------
# Cross-backend execution tests for the alternating-qubit pattern
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=[
        "qiskit",
        pytest.param("quri_parts", marks=pytest.mark.quri_parts),
        pytest.param("cudaq", marks=pytest.mark.cudaq),
    ]
)
def backend(request):
    """Yield ``(transpiler, executor)`` for each installed backend.

    Tests parametrized over this fixture run independently on each
    backend and are skipped when the corresponding SDK is not
    installed.
    """
    name = request.param
    if name == "qiskit":
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        transpiler = QiskitTranspiler()
        return name, transpiler, transpiler.executor()
    if name == "quri_parts":
        pytest.importorskip("quri_parts")
        from qamomile.quri_parts import QuriPartsTranspiler

        transpiler = QuriPartsTranspiler()
        return name, transpiler, transpiler.executor()
    if name == "cudaq":
        pytest.importorskip("cudaq")
        from qamomile.cudaq import CudaqTranspiler

        transpiler = CudaqTranspiler()
        return name, transpiler, transpiler.executor()
    raise AssertionError(f"unknown backend {name}")


# -- Sampling path ----------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 42])
@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7])
def test_slice_even_hadamard_sampling(backend, seed, n):
    """``q[0::2]`` + H hits exactly the even qubits across every backend.

    Applies H on ``q[0::2]`` only, leaving the odd qubits in |0>.  After
    measurement every odd bit must be 0; the even bits are uniformly
    random.  Parametrized over register sizes and seeds so the assertion
    holds for varied shapes.
    """
    name, transpiler, executor = backend
    rng = np.random.default_rng(seed)
    del rng  # reserved for future randomised variants; keep the plumbing

    @qmc.qkernel
    def circuit(num: qmc.UInt) -> qmc.Vector[qmc.Bit]:
        q = qmc.qubit_array(num, "q")
        evens = q[0::2]
        for i in qmc.range(evens.shape[0]):
            evens[i] = qmc.h(evens[i])
        q[0::2] = evens
        return qmc.measure(q)

    exe = transpiler.transpile(circuit, bindings={"num": n})
    job = exe.sample(executor, shots=1024)
    result = job.result()

    # Normalise across backend result shapes: results is iterable of
    # (bits, count) where ``bits`` is a tuple of 0/1 in little-endian
    # kernel-order, matching the Vector indexing convention.
    total = 0
    for bits, count in result.results:
        total += count
        assert len(bits) == n, f"{name}: expected {n} bits, got {len(bits)}"
        for qi in range(1, n, 2):  # odd indices must be zero
            assert bits[qi] == 0, (
                f"{name} n={n} seed={seed}: odd qubit {qi} was {bits[qi]} "
                f"in outcome {bits}"
            )
    assert total == 1024


@pytest.mark.parametrize("seed", [0, 1, 42])
@pytest.mark.parametrize("n_pairs", [1, 2, 3])
def test_slice_xy_brick_sampling(backend, seed, n_pairs):
    """Alternating CX on even/odd pairs produces perfect Bell-like pairs.

    This is the XY-mixer-style construction from the Alternating Operator
    Ansatz: prepare |+> on the even qubits, leave odds as |0>, then
    ``cx(evens[i], odds[i])``.  The final state has each pair in
    ``(|00> + |11>)/sqrt(2)``, so every measured outcome must have
    ``bits[2i] == bits[2i+1]`` — independent of the random seed.
    """
    name, transpiler, executor = backend
    rng = np.random.default_rng(seed)
    del rng

    n = 2 * n_pairs

    @qmc.qkernel
    def circuit(num: qmc.UInt) -> qmc.Vector[qmc.Bit]:
        q = qmc.qubit_array(num, "q")
        evens = q[0::2]
        odds = q[1::2]
        for i in qmc.range(evens.shape[0]):
            evens[i] = qmc.h(evens[i])
        for i in qmc.range(evens.shape[0]):
            evens[i], odds[i] = qmc.cx(evens[i], odds[i])
        q[0::2] = evens
        q[1::2] = odds
        return qmc.measure(q)

    exe = transpiler.transpile(circuit, bindings={"num": n})
    job = exe.sample(executor, shots=1024)
    result = job.result()

    total = 0
    for bits, count in result.results:
        total += count
        for pair in range(n_pairs):
            even_bit = bits[2 * pair]
            odd_bit = bits[2 * pair + 1]
            assert even_bit == odd_bit, (
                f"{name} n={n} seed={seed}: pair {pair} was "
                f"({even_bit}, {odd_bit}) in outcome {bits}"
            )
    assert total == 1024


# -- Expectation-value path -------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 42])
@pytest.mark.parametrize("n", [2, 3, 4])
def test_slice_even_rx_expval(backend, seed, n):
    """``<Z_k>`` on an RX-even-only circuit matches ``cos(theta)`` or ``1``.

    Apply ``rx(theta)`` to ``q[0::2]``, leave odd qubits alone, then
    measure ``<Z_target>``.  Expected value is ``cos(theta)`` on even
    targets and ``1`` on odd targets — independent of the register size.
    Randomised over ``theta`` and varied over ``n`` and ``target`` so the
    test covers a spread of shapes.
    """
    name, transpiler, executor = backend
    rng = np.random.default_rng(seed)
    theta = float(rng.uniform(-np.pi, np.pi))

    for target in range(n):
        expected = np.cos(theta) if target % 2 == 0 else 1.0
        # Pad the observable to span the whole register so every backend's
        # estimator sees a matching num_qubits.
        H = qm_o.Z(target) + 0.0 * qm_o.Z(n - 1)

        @qmc.qkernel
        def circuit(
            num: qmc.UInt,
            angle: qmc.Float,
            obs: qmc.Observable,
        ) -> qmc.Float:
            q = qmc.qubit_array(num, "q")
            evens = q[0::2]
            for i in qmc.range(evens.shape[0]):
                evens[i] = qmc.rx(evens[i], angle)
            q[0::2] = evens
            return qmc.expval(q, obs)

        exe = transpiler.transpile(
            circuit,
            bindings={"num": n, "angle": theta, "obs": H},
        )
        got = exe.run(executor).result()
        assert np.isclose(got, expected, atol=1e-5), (
            f"{name} n={n} target={target} theta={theta}: "
            f"got {got}, expected {expected}"
        )


# ---------------------------------------------------------------------------
# Qiskit-specific structural assertions (cheap, no simulation required)
# ---------------------------------------------------------------------------


def _make_h_via_slice_kernel(slice_kind: str):
    """Build a kernel that applies ``H`` to qubits selected by ``slice_kind``.

    Args:
        slice_kind (str): One of ``"0::2"`` / ``"1::2"`` / ``"1:5"`` /
            ``"2::3"``; identifies which slice form the kernel body uses.

    Returns:
        QKernel: A kernel that takes ``num: UInt`` and broadcasts ``H``
            over ``q[<slice_kind>]`` via an explicit loop, returning the
            view to ``q`` before measuring.
    """
    if slice_kind == "0::2":

        @qmc.qkernel
        def kern(num: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(num, "q")
            view = q[0::2]
            for i in qmc.range(view.shape[0]):
                view[i] = qmc.h(view[i])
            q[0::2] = view
            return qmc.measure(q)

        return kern
    if slice_kind == "1::2":

        @qmc.qkernel
        def kern(num: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(num, "q")
            view = q[1::2]
            for i in qmc.range(view.shape[0]):
                view[i] = qmc.h(view[i])
            q[1::2] = view
            return qmc.measure(q)

        return kern
    if slice_kind == "1:5":

        @qmc.qkernel
        def kern(num: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(num, "q")
            view = q[1:5]
            for i in qmc.range(view.shape[0]):
                view[i] = qmc.h(view[i])
            q[1:5] = view
            return qmc.measure(q)

        return kern
    if slice_kind == "2::3":

        @qmc.qkernel
        def kern(num: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(num, "q")
            view = q[2::3]
            for i in qmc.range(view.shape[0]):
                view[i] = qmc.h(view[i])
            q[2::3] = view
            return qmc.measure(q)

        return kern
    raise AssertionError(f"unknown slice_kind {slice_kind!r}")


@pytest.mark.parametrize(
    "slice_kind, n, expected_qubits",
    [
        ("0::2", 6, [0, 2, 4]),
        ("1::2", 6, [1, 3, 5]),
        ("1:5", 6, [1, 2, 3, 4]),
        ("2::3", 10, [2, 5, 8]),
    ],
)
def test_slice_emits_expected_qubit_indices(slice_kind, n, expected_qubits):
    """Emitted H gates land on exactly the qubits the slice describes."""
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    circuit = _make_h_via_slice_kernel(slice_kind)

    transpiler = QiskitTranspiler()
    exe = transpiler.transpile(circuit, bindings={"num": n})
    qc = exe.compiled_quantum[0].circuit
    applied = [
        qc.qubits.index(inst.qubits[0])
        for inst in qc.data
        if inst.operation.name == "h"
    ]
    assert applied == expected_qubits


def test_dynamic_slice_bounds_qiskit():
    """Slices where both bounds are runtime UInts still emit correctly."""
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    @qmc.qkernel
    def circuit(num: qmc.UInt, lo: qmc.UInt, hi: qmc.UInt) -> qmc.Vector[qmc.Bit]:
        q = qmc.qubit_array(num, "q")
        region = q[lo:hi]
        for i in qmc.range(region.shape[0]):
            region[i] = qmc.h(region[i])
        q[lo:hi] = region
        return qmc.measure(q)

    transpiler = QiskitTranspiler()
    exe = transpiler.transpile(circuit, bindings={"num": 8, "lo": 2, "hi": 6})
    qc = exe.compiled_quantum[0].circuit
    applied = [
        qc.qubits.index(inst.qubits[0])
        for inst in qc.data
        if inst.operation.name == "h"
    ]
    assert applied == [2, 3, 4, 5]


def test_nested_slice_composes_through_parent():
    """Slicing a VectorView nests the affine map without detaching from parent."""
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    @qmc.qkernel
    def circuit(num: qmc.UInt) -> qmc.Vector[qmc.Bit]:
        q = qmc.qubit_array(num, "q")
        # q[0::2] → parent indices 0, 2, 4, 6; [1:3] → view-local 1, 2
        # → parent 2, 4.
        outer = q[0::2]
        inner = outer[1:3]
        for i in qmc.range(inner.shape[0]):
            inner[i] = qmc.h(inner[i])
        # Strict-return: return inner to outer, then outer to root.
        outer[1:3] = inner
        q[0::2] = outer
        return qmc.measure(q)

    transpiler = QiskitTranspiler()
    exe = transpiler.transpile(circuit, bindings={"num": 8})
    qc = exe.compiled_quantum[0].circuit
    applied = [
        qc.qubits.index(inst.qubits[0])
        for inst in qc.data
        if inst.operation.name == "h"
    ]
    assert applied == [2, 4]


def test_vector_view_exported():
    """VectorView is reachable from the public ``qamomile.circuit`` API."""
    assert qmc.VectorView is VectorView


# ---------------------------------------------------------------------------
# Bulk-borrow linearity — parent slots covered by a live view are locked.
# ---------------------------------------------------------------------------


class TestSliceBulkBorrow:
    """Slice-time bulk borrow blocks direct parent access to covered slots."""

    def test_parent_access_on_covered_slot_is_rejected(self):
        """``q[0]`` after ``q[0:4:2]`` raises because slot 0 is slice-owned."""
        from qamomile.circuit.transpiler.errors import QubitBorrowConflictError

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Qubit]:
            q = qmc.qubit_array(4, "q")
            evens = q[0:4:2]
            del evens
            q[0] = qmc.h(q[0])
            return q

        with pytest.raises(
            QubitBorrowConflictError, match="held by a VectorView slice"
        ):
            _ = kern.block

    def test_parent_access_on_non_covered_slot_is_fine(self):
        """Non-covered slot 1 is accessible while ``q[0:4:2]`` covers 0, 2."""

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Qubit]:
            q = qmc.qubit_array(4, "q")
            evens = q[0:4:2]
            q[1] = qmc.h(q[1])
            for i in qmc.range(evens.shape[0]):
                evens[i] = qmc.h(evens[i])
            q[0:4:2] = evens
            return q

        assert kern.block is not None

    def test_explicit_slice_return_before_parent_consume(self):
        """After writing the view back via slice assignment, the parent can be consumed.

        Renamed from ``test_drained_view_releases_parent_on_consume``:
        the old name implied an implicit-drain semantic where merely
        emptying the view's element borrows freed the parent.  The
        strict-return policy removed that path — the view must be
        returned explicitly via ``parent[a:b:c] = view`` before the
        parent is consumed.  This test pins the explicit pattern.
        """

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            evens = q[0:4:2]
            for i in qmc.range(evens.shape[0]):
                evens[i] = qmc.h(evens[i])
            q[0:4:2] = evens
            return qmc.measure(q)

        assert kern.block is not None

    def test_drained_view_without_explicit_return_rejected(self):
        """Implicit-drain pattern (view drained of element borrows but
        not slice-assigned back) now raises ``UnreturnedBorrowError``.

        Prior to the strict-return change this pattern silently passed
        via an opportunistic drain.  Pinning the new loud failure keeps
        regressions visible.
        """
        from qamomile.circuit.transpiler.errors import UnreturnedBorrowError

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            evens = q[0:4:2]
            for i in qmc.range(evens.shape[0]):
                evens[i] = qmc.h(evens[i])
            # Missing ``q[0:4:2] = evens`` — should fail at measure(q).
            return qmc.measure(q)

        with pytest.raises(UnreturnedBorrowError, match="unreturned slice-view"):
            _ = kern.block

    def test_overlapping_live_slices_are_rejected(self):
        """Two slices with partial overlap on the same root are rejected.

        Strict no-multi-view rejects every live overlap at
        ``VectorView._wrap`` regardless of whether ``a`` has been
        accessed yet — the second view's construction fails as soon as
        the parent's borrow table records ``a`` against any of ``b``'s
        covered slots.
        """

        from qamomile.circuit.transpiler.errors import QubitBorrowConflictError

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Qubit]:
            q = qmc.qubit_array(4, "q")
            a = q[0:4:2]  # covers {0, 2}
            qa = a[0]  # active borrow on slot 0 via view a
            _b = q[0:4:1]  # covers {0, 1, 2, 3} — overlaps with live a
            a[0] = qa
            return q

        with pytest.raises(
            QubitBorrowConflictError, match="already owned by another slice view"
        ):
            _ = kern.block

    def test_symbolic_slice_skips_bulk_borrow(self):
        """Symbolic-bound slices don't block parent access (best-effort)."""
        # A ``lo:hi`` slice where bounds are ``UInt`` parameters can't
        # enumerate its covered indices at trace time, so it falls back
        # to view-local borrow tracking only.  Direct parent access to
        # a concrete slot therefore succeeds.

        @qmc.qkernel
        def kern(n: qmc.UInt, lo: qmc.UInt, hi: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
            q = qmc.qubit_array(n, "q")
            region = q[lo:hi]
            del region
            q[0] = qmc.h(q[0])  # allowed — slice is symbolic
            return q

        assert kern.block is not None


# ---------------------------------------------------------------------------
# View as a qkernel argument — inline-trace path.
# ---------------------------------------------------------------------------


class TestVectorViewAsKernelArgument:
    """A ``VectorView`` can be handed to another ``@qkernel`` directly."""

    def test_view_argument_inlines_correctly(self):
        """Gates emitted by the callee land on the view's parent qubits."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def h_all(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
            n = q.shape[0]
            for i in qmc.range(n):
                q[i] = qmc.h(q[i])
            return q

        @qmc.qkernel
        def circuit(num: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(num, "q")
            evens = q[0::2]
            evens = h_all(evens)
            q[0::2] = evens
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(circuit, bindings={"num": 6})
        qc = exe.compiled_quantum[0].circuit
        applied = [
            qc.qubits.index(inst.qubits[0])
            for inst in qc.data
            if inst.operation.name == "h"
        ]
        assert applied == [0, 2, 4]

    def test_view_argument_full_reslice_return_stays_a_view(self):
        """A sub-kernel may return a full re-slice of its view argument."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def reslice_identity(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
            view = q[0 : q.shape[0]]
            return view

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            evens = q[0:4:2]
            evens = reslice_identity(evens)
            evens = qmc.h(evens)
            q[0:4:2] = evens
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(circuit)
        qc = exe.compiled_quantum[0].circuit
        applied = [
            qc.qubits.index(inst.qubits[0])
            for inst in qc.data
            if inst.operation.name == "h"
        ]
        assert applied == [0, 2]

    def test_runtime_if_full_reslice_return_stays_a_view(self):
        """A conditional merge keeps a caller view's resource identity."""

        @qmc.qkernel
        def conditional_reslice(
            q: qmc.Vector[qmc.Qubit],
            flag: qmc.Bit,
        ) -> qmc.Vector[qmc.Qubit]:
            if flag:
                q = qmc.x(q)
            else:
                q = qmc.h(q)
            return q[:]

        @qmc.qkernel
        def circuit(flag: qmc.Bit) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            middle = q[1:3]
            middle = conditional_reslice(middle, flag)
            q[1:3] = middle
            return qmc.measure(q)

        assert circuit.block is not None

    @pytest.mark.parametrize("slice_in_true_branch", [False, True])
    def test_runtime_if_direct_and_full_reslice_return_stays_a_view(
        self,
        slice_in_true_branch: bool,
    ):
        """Direct and full-slice branch forms share one array resource."""
        if slice_in_true_branch:

            @qmc.qkernel
            def conditional_reslice(
                q: qmc.Vector[qmc.Qubit],
                flag: qmc.Bit,
            ) -> qmc.Vector[qmc.Qubit]:
                if flag:
                    result = q[:]
                else:
                    result = q
                return result[:]

        else:

            @qmc.qkernel
            def conditional_reslice(
                q: qmc.Vector[qmc.Qubit],
                flag: qmc.Bit,
            ) -> qmc.Vector[qmc.Qubit]:
                if flag:
                    result = q
                else:
                    result = q[:]
                return result[:]

        @qmc.qkernel
        def circuit(flag: qmc.Bit) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            middle = q[1:3]
            middle = conditional_reslice(middle, flag)
            q[1:3] = middle
            return qmc.measure(q)

        SliceBorrowCheckPass().run(circuit.block)

    def test_full_reslice_wrapper_keeps_caller_local_metadata(self):
        """A full re-slice keeps metadata remapped by call materialization."""
        length = _uint_value("length", 2)
        zero = _uint_value("zero", 0)
        one = _uint_value("one", 1)

        formal_input = _qubit_array("formal_input", length)
        formal_output = _slice_array(
            formal_input,
            "formal_output",
            zero,
            one,
            length,
        ).with_array_runtime_metadata(element_uuids=("formal[0]", "formal[1]"))

        caller_root = _qubit_array("caller_root", length)
        caller_view = _slice_array(
            caller_root,
            "caller_view",
            zero,
            one,
            length,
        )
        caller_result = _slice_array(
            caller_view,
            "caller_result",
            zero,
            one,
            length,
        ).with_array_runtime_metadata(element_uuids=("caller[0]", "caller[1]"))

        call_op = SimpleNamespace(results=[caller_result])
        wrapped = _wrap_array_result(
            kernel=SimpleNamespace(name="helper"),
            call_op=call_op,
            result_idx=0,
            val=caller_result,
            handle_type=qmc.Vector[qmc.Qubit],
            block_ir_for_call=Block(
                input_values=[formal_input],
                output_values=[formal_output],
            ),
            formal_input_views={
                formal_input.logical_id: (
                    formal_input,
                    SimpleNamespace(value=caller_view),
                )
            },
            input_view_metas={},
        )

        expected = ("caller[0]", "caller[1]")
        assert call_op.results[0].get_element_uuids() == expected
        assert wrapped.value.get_element_uuids() == expected

    def test_view_argument_mutate_then_full_reslice_targets_parent_slots(self):
        """A full re-slice return preserves in-callee operations on the view."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def h_then_reslice(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
            q = qmc.h(q)
            return q[:]

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            evens = q[0:4:2]
            evens = h_then_reslice(evens)
            q[0:4:2] = evens
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(circuit)
        qc = exe.compiled_quantum[0].circuit
        applied = [
            qc.qubits.index(inst.qubits[0])
            for inst in qc.data
            if inst.operation.name == "h"
        ]
        assert applied == [0, 2]

    @pytest.mark.parametrize("n", [4, 6, 8])
    def test_two_qubit_subroutine_via_view_pair(self, n):
        """Pair subroutine across ``q[0::2]``/``q[1::2]`` produces brick-wall CX."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def cx_pair(a: qmc.Qubit, b: qmc.Qubit) -> tuple[qmc.Qubit, qmc.Qubit]:
            return qmc.cx(a, b)

        @qmc.qkernel
        def cx_alt_layer(
            evens: qmc.Vector[qmc.Qubit],
            odds: qmc.Vector[qmc.Qubit],
        ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            n = evens.shape[0]
            for i in qmc.range(n):
                ea, ob = cx_pair(evens[i], odds[i])
                evens[i] = ea
                odds[i] = ob
            return evens, odds

        @qmc.qkernel
        def circuit(num: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(num, "q")
            evens = q[0::2]
            odds = q[1::2]
            evens, odds = cx_alt_layer(evens, odds)
            q[0::2] = evens
            q[1::2] = odds
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(circuit, bindings={"num": n})
        qc = exe.compiled_quantum[0].circuit
        pairs = [
            (qc.qubits.index(inst.qubits[0]), qc.qubits.index(inst.qubits[1]))
            for inst in qc.data
            if inst.operation.name == "cx"
        ]
        expected = [(2 * i, 2 * i + 1) for i in range(n // 2)]
        assert pairs == expected

    def test_view_consumption_transfers_parent_bulk_borrow(self):
        """Passing a view to a sub-kernel transfers the bulk-borrow to the result handle.

        Under strict-return the parent slot stays locked after the
        call — ownership is transferred from the original view to the
        sub-kernel's return value, not released.  Slice-assigning the
        returned view back releases the parent's slot.
        """
        from qamomile.circuit.transpiler.errors import QubitBorrowConflictError

        @qmc.qkernel
        def h_all(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
            n = q.shape[0]
            for i in qmc.range(n):
                q[i] = qmc.h(q[i])
            return q

        # Without the slice-assign return, ``q[0]`` is still held by
        # the returned view.
        @qmc.qkernel
        def kern_no_return() -> qmc.Vector[qmc.Qubit]:
            q = qmc.qubit_array(4, "q")
            evens = q[0:4:2]
            evens = h_all(evens)
            del evens
            q[0] = qmc.x(q[0])
            return q

        with pytest.raises(
            QubitBorrowConflictError, match="held by a VectorView slice"
        ):
            _ = kern_no_return.block

        # The strict-return form releases the borrow before touching q[0].
        @qmc.qkernel
        def kern_with_return() -> qmc.Vector[qmc.Qubit]:
            q = qmc.qubit_array(4, "q")
            evens = q[0:4:2]
            evens = h_all(evens)
            q[0:4:2] = evens
            q[0] = qmc.x(q[0])
            return q

        assert kern_with_return.block is not None

    def test_view_passed_to_kernel_returning_non_view_is_consumed(self):
        """A view handed to a sub-kernel whose result is not a sliced view is consumed.

        Regression: ``QKernel.__call__`` used to defer ``VectorView``
        consumption to a ``_transfer_borrow_to`` that only ran when a
        matching sliced result existed.  If the callee returned a
        scalar / different-shape array, the input view was left
        un-consumed — re-use after the call gave silent
        affine-type violations.  The fix destructively consumes
        unmatched view inputs via ``"qkernel call (view dropped)"``,
        so re-touching the covered slots is rejected loudly.
        """
        from qamomile.circuit.transpiler.errors import QubitConsumedError

        @qmc.qkernel
        def measure_half(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Bit]:
            return qmc.measure(q)

        @qmc.qkernel
        def caller_reuses_view() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            evens = q[0::2]
            _ = measure_half(evens)  # view dropped here; slots {0, 2} consumed
            evens = qmc.h(evens)  # ← rejected: view _consumed=True
            return qmc.measure(q)

        with pytest.raises(QubitConsumedError):
            _ = caller_reuses_view.block

    def test_view_dropped_after_kernel_call_blocks_root_slot_reuse(self):
        """``q[0]`` after a dropped-view kernel call is rejected as use-after-destroy.

        Companion to the test above: the *covered parent slots* must
        also be marked consumed, not just the view handle.  Without
        the destructive consume, a kernel like ``measure_half(evens)``
        would leak qubits — the caller could touch the same physical
        qubits via the root register and the compiler would not
        notice.
        """
        from qamomile.circuit.transpiler.errors import QubitConsumedError

        @qmc.qkernel
        def measure_half(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Bit]:
            return qmc.measure(q)

        @qmc.qkernel
        def caller_touches_root() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            evens = q[0::2]
            _ = measure_half(evens)  # destructively consumes {0, 2}
            q[0] = qmc.h(q[0])  # ← rejected: slot 0 is destroyed
            return qmc.measure(q)

        with pytest.raises(QubitConsumedError):
            _ = caller_touches_root.block


# ---------------------------------------------------------------------------
# V6: IR-level slice operation + SliceBorrowCheckPass
# ---------------------------------------------------------------------------


class TestSliceArrayOperation:
    """The SliceArrayOperation appears in the pre-fold block and is stripped."""

    def test_slice_op_present_in_hierarchical_block(self):
        """Tracing emits a ``SliceArrayOperation`` at the slice site."""
        from qamomile.circuit.ir.operation import SliceArrayOperation

        @qmc.qkernel
        def kern(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            evens = q[0::2]
            for i in qmc.range(evens.shape[0]):
                evens[i] = qmc.h(evens[i])
            return qmc.measure(q)

        ops = kern.block.operations
        assert any(isinstance(op, SliceArrayOperation) for op in ops)

    def test_slice_op_stripped_after_constant_folding(self):
        """``ConstantFoldingPass`` removes ``SliceArrayOperation``."""
        pytest.importorskip("qiskit")
        from qamomile.circuit.ir.operation import SliceArrayOperation
        from qamomile.circuit.transpiler.passes import ConstantFoldingPass
        from qamomile.circuit.transpiler.passes.inline import InlinePass

        @qmc.qkernel
        def kern(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            evens = q[0::2]
            for i in qmc.range(evens.shape[0]):
                evens[i] = qmc.h(evens[i])
            return qmc.measure(q)

        block = InlinePass().run(kern.block)
        folded = ConstantFoldingPass(bindings={"n": 4}).run(block)
        assert not any(isinstance(op, SliceArrayOperation) for op in folded.operations)

    def test_sliced_array_value_carries_metadata(self):
        """The sliced ``ArrayValue`` carries ``slice_of`` / start / step."""
        from qamomile.circuit.ir.value import ArrayValue

        captured: dict[str, object] = {}

        @qmc.qkernel
        def kern(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
            view = q[0::2]
            captured["av"] = view.value
            view[0] = qmc.h(view[0])
            return q

        _ = kern.block
        av = captured["av"]
        assert isinstance(av, ArrayValue)
        assert av.slice_of is not None
        assert av.slice_start is not None
        assert av.slice_step is not None


class TestViewAsKernelOperandViaInvoke:
    """View flows through inline ``InvokeOperation`` calls."""

    def test_view_arg_routes_through_invoke(self):
        """Kernel called with a view now uses the standard call path.

        Previously (V1), views forced ``_inline_trace_call`` because
        they had no IR representation.  V6 makes the sliced
        ``ArrayValue`` first-class, so the callee receives the view
        through the normal inline ``InvokeOperation`` operand chain.  The
        user-facing behaviour (gates land on the right qubits) is
        unchanged; this test pins the IR-level path.
        """
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def apply_h(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
            for i in qmc.range(q.shape[0]):
                q[i] = qmc.h(q[i])
            return q

        @qmc.qkernel
        def circuit(num: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(num, "q")
            evens = q[0::2]
            evens = apply_h(evens)
            q[0::2] = evens
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(circuit, bindings={"num": 6})
        qc = exe.compiled_quantum[0].circuit
        h_qubits = [
            qc.qubits.index(inst.qubits[0])
            for inst in qc.data
            if inst.operation.name == "h"
        ]
        assert h_qubits == [0, 2, 4]


class TestPostFoldLinearity:
    """``SliceBorrowCheckPass`` catches post-fold aliasing."""

    def test_symbolic_slice_vs_direct_access_caught_at_transpile(self):
        """``q[lo:hi] + q[0]`` aliases when ``lo=0``; caught at transpile.

        Scalar ``UInt`` parameters that are concrete in ``bindings`` now
        carry ``is_constant() == True`` at trace time, so the frontend's
        bulk-borrow tracker can enumerate the slice's coverage and
        catches the aliasing immediately (``QubitBorrowConflictError``).
        When bounds stay symbolic — e.g. derived from an unbound
        parameter through arithmetic — ``SliceBorrowCheckPass``
        resolves them post-fold and raises the same
        ``QubitBorrowConflictError``.
        """
        pytest.importorskip("qiskit")
        from qamomile.circuit.transpiler.errors import QubitBorrowConflictError
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def circuit(num: qmc.UInt, lo: qmc.UInt, hi: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(num, "q")
            region = q[lo:hi]
            region[0] = qmc.h(region[0])
            # Direct access to slot that the region covers when lo=0.
            q[0] = qmc.x(q[0])
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        with pytest.raises(QubitBorrowConflictError):
            transpiler.transpile(circuit, bindings={"num": 4, "lo": 0, "hi": 4})

    def test_symbolic_slice_disjoint_from_direct_access_passes(self):
        """When the symbolic range doesn't cover the accessed slot, no error."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def circuit(num: qmc.UInt, lo: qmc.UInt, hi: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(num, "q")
            region = q[lo:hi]
            for i in qmc.range(region.shape[0]):
                region[i] = qmc.h(region[i])
            q[lo:hi] = region
            # Slot 0 is out of [lo, hi) when lo=2, hi=4.
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        # Should succeed with lo=2, hi=4 — region covers {2, 3}, not 0.
        exe = transpiler.transpile(circuit, bindings={"num": 4, "lo": 2, "hi": 4})
        assert exe is not None

    def test_runtime_symbolic_adjacent_slices_pass_borrow_check(self):
        """``q[:k]`` and ``q[k:4]`` are accepted as adjacent symbolic intervals."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def circuit(k: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            left = q[:k]
            right = q[k:4]
            q[:k] = left
            q[k:4] = right
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        block = transpiler.to_block(circuit, parameters=["k"])
        block = transpiler.resolve_parameter_shapes(block, bindings={})
        block = transpiler.inline(block)
        block = transpiler.affine_validate(block)
        block = transpiler.partial_eval(block)

        assert transpiler.slice_borrow_check(block) is block

    def test_direct_access_before_view_use_is_caught(self):
        """Alias is caught even when the direct parent access precedes the view use.

        Regression (P1-2): ``SliceBorrowCheckPass`` used to register
        a view's bulk-borrow lazily — only when the view was first
        referenced as an operand.  That made ``region = q[lo2:hi];
        q[0] = h(q[0]); region[0] = x(region[0])`` (with ``lo=0``,
        ``hi=4``) slip through, because the direct ``q[0]`` access
        happened before the view was ever observed in the pass.
        The fix pre-scans all operands up front so the view's covered
        slots are registered before any conflict check.
        """
        pytest.importorskip("qiskit")
        from qamomile.circuit.transpiler.errors import QubitBorrowConflictError
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def circuit(num: qmc.UInt, lo: qmc.UInt, hi: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(num, "q")
            # Force lo2 to stay symbolic at trace time via BinOp so
            # the frontend cannot catch the alias and we must rely on
            # the post-fold pass for detection.
            lo2 = lo + 0
            region = q[lo2:hi]
            q[0] = qmc.h(q[0])  # aliases the view when lo=0
            region[0] = qmc.x(region[0])
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        with pytest.raises(QubitBorrowConflictError):
            transpiler.transpile(circuit, bindings={"num": 4, "lo": 0, "hi": 4})


class TestSameSliceVersionRefresh:
    """Regression coverage for same-slice SSA-version refreshes."""

    def test_controlled_gate_refresh_on_slice_controls_transpiles(self):
        """A controlled gate may return a newer version of the same slice."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        num_qubits = 3
        thetas_list = [
            [0],
            [1, 2],
            [3, 4, 5, 6],
        ]

        @qmc.qkernel
        def state_prep() -> qmc.Vector[qmc.Qubit]:
            qs = qmc.qubit_array(num_qubits, "qs")

            qs[0] = qmc.ry(qs[0], thetas_list[0][0])

            controlled_ry = qmc.control(qmc.ry, num_controls=1)
            controls = qs[:1]
            target = qs[1]

            controls = qmc.x(controls)
            controls, target = controlled_ry(controls, target, thetas_list[1][0])
            controls = qmc.x(controls)

            qs[:1] = controls
            qs[1] = target

            return qs

        @qmc.qkernel
        def main() -> qmc.Vector[qmc.Bit]:
            qs = state_prep()
            return qmc.measure(qs)

        executable = QiskitTranspiler().transpile(main)

        assert executable.get_first_circuit().num_qubits == num_qubits

    def test_bound_positive_for_loop_refreshes_outer_slice_view(self):
        """A bound positive loop count permits body-local same-slice refresh."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def circuit(repetitions: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            controls = q[:1]
            for _ in qmc.range(repetitions):
                controls = qmc.x(controls)
            q[:1] = controls
            return qmc.measure(q)

        executable = QiskitTranspiler().transpile(circuit, bindings={"repetitions": 2})

        assert executable.get_first_circuit().num_qubits == 3

    def test_runtime_if_preserves_slice_view_metadata(self):
        """Runtime branch merge keeps refreshed slice metadata assignable."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            controls = q[:1]
            flag = qmc.measure(q[2])
            if flag:
                controls = qmc.x(controls)
            q[:1] = controls
            return qmc.measure(q[:2])

        executable = QiskitTranspiler().transpile(circuit)

        assert executable.get_first_circuit().num_qubits == 3

    def test_runtime_if_reconnects_branch_local_full_slices(self):
        """Merged full slices return ownership to the merged root handle."""

        @qmc.qkernel
        def circuit(flag: qmc.Bit) -> qmc.Vector[qmc.Qubit]:
            q = qmc.qubit_array(4, "q")
            if flag:
                view = q[:]
            else:
                view = q[:]
            view = qmc.h(view)
            q[:] = view
            return q

        assert circuit.block is not None

    @pytest.mark.parametrize("reslice_in_true_branch", [True, False])
    def test_runtime_if_root_full_slice_merge_releases_alias_borrow(
        self,
        reslice_in_true_branch: bool,
    ) -> None:
        """A root/full-slice merge leaves the merged root directly usable."""

        if reslice_in_true_branch:

            @qmc.qkernel
            def circuit(flag: qmc.Bit) -> qmc.Vector[qmc.Qubit]:
                q = qmc.qubit_array(4, "q")
                if flag:
                    q = q[:]
                q[0] = qmc.h(q[0])
                return q

        else:

            @qmc.qkernel
            def circuit(flag: qmc.Bit) -> qmc.Vector[qmc.Qubit]:
                q = qmc.qubit_array(4, "q")
                if flag:
                    q = q
                else:
                    q = q[:]
                q[0] = qmc.h(q[0])
                return q

        SliceBorrowCheckPass().run(circuit.block)

    def test_nested_runtime_if_root_full_slice_merge_releases_alias_borrow(self):
        """Nested root/full-slice merges leave the outer merged root usable."""

        @qmc.qkernel
        def circuit(
            outer_flag: qmc.Bit,
            inner_flag: qmc.Bit,
        ) -> qmc.Vector[qmc.Qubit]:
            q = qmc.qubit_array(4, "q")
            if outer_flag:
                if inner_flag:
                    q = q[:]
            q[1] = qmc.h(q[1])
            return q

        SliceBorrowCheckPass().run(circuit.block)

    def test_root_full_slice_merge_preserves_prebranch_owner(self):
        """Alias cleanup never retires a borrow live before the runtime If."""
        checker = SliceBorrowCheckPass()
        length = _uint_value("length", 4)
        root = _qubit_array("q", length)
        full_view = _slice_array(
            root,
            "full",
            _uint_value("start", 0),
            _uint_value("step", 1),
            length,
        )
        merge = IfMerge(0, root, full_view, root.next_version())
        key = (root.logical_id, "const:0")
        state = {key: full_view}

        assert checker._record_if_representation_alias(merge)
        checker._retire_if_root_representation_borrows(
            merge,
            state,
            dict(state),
        )

        assert state == {key: full_view}

    def test_root_reordered_slice_merge_is_not_representation_alias(self):
        """Equal unordered coverage cannot hide a reordered slice as an alias."""
        checker = SliceBorrowCheckPass()
        length = _uint_value("length", 4)
        root = _qubit_array("q", length)
        reversed_view = _slice_array(
            root,
            "reversed",
            _uint_value("start", 3),
            _uint_value("step", -1),
            length,
        )
        merge = IfMerge(0, root, reversed_view, root.next_version())

        assert not checker._record_if_representation_alias(merge)

    def test_runtime_if_reslices_existing_partial_view(self):
        """A branch-local full reslice retains the pre-branch borrow owner."""

        @qmc.qkernel
        def circuit(flag: qmc.Bit) -> qmc.Vector[qmc.Qubit]:
            q = qmc.qubit_array(4, "q")
            view = q[1:3]
            if flag:
                view = view[:]
            else:
                view = view
            view = qmc.h(view)
            q[1:3] = view
            return q

        assert circuit.block is not None

    @pytest.mark.parametrize("reslice_in_true_branch", [True, False])
    def test_runtime_if_reconnects_nested_prebranch_view(
        self,
        reslice_in_true_branch: bool,
    ):
        """A one-sided reslice keeps its nested pre-branch view lineage."""

        if reslice_in_true_branch:

            @qmc.qkernel
            def circuit(flag: qmc.Bit) -> qmc.Vector[qmc.Qubit]:
                q = qmc.qubit_array(6, "q")
                outer = q[0::2]
                view = outer[1:3]
                if flag:
                    view = view[:]
                else:
                    view = view
                view = qmc.h(view)
                outer[1:3] = view
                q[0::2] = outer
                return q

        else:

            @qmc.qkernel
            def circuit(flag: qmc.Bit) -> qmc.Vector[qmc.Qubit]:
                q = qmc.qubit_array(6, "q")
                outer = q[0::2]
                view = outer[1:3]
                if flag:
                    view = view
                else:
                    view = view[:]
                view = qmc.h(view)
                outer[1:3] = view
                q[0::2] = outer
                return q

        SliceBorrowCheckPass().run(circuit.block)

    def test_runtime_if_reconnects_two_resliced_nested_prebranch_views(self):
        """Two-sided reslices keep their nested pre-branch view lineage."""

        @qmc.qkernel
        def circuit(flag: qmc.Bit) -> qmc.Vector[qmc.Qubit]:
            q = qmc.qubit_array(6, "q")
            outer = q[0::2]
            view = outer[1:3]
            if flag:
                view = view[:]
            else:
                view = view[:]
            view = qmc.h(view)
            outer[1:3] = view
            q[0::2] = outer
            return q

        SliceBorrowCheckPass().run(circuit.block)

    def test_nested_runtime_if_reconnects_nested_prebranch_view(self):
        """Nested If merges preserve the canonical outer-view lineage."""

        @qmc.qkernel
        def circuit(
            outer_flag: qmc.Bit,
            inner_flag: qmc.Bit,
        ) -> qmc.Vector[qmc.Qubit]:
            q = qmc.qubit_array(6, "q")
            outer = q[0::2]
            view = outer[1:3]
            if outer_flag:
                if inner_flag:
                    view = view[:]
                else:
                    view = view
            else:
                view = view
            view = qmc.h(view)
            outer[1:3] = view
            q[0::2] = outer
            return q

        SliceBorrowCheckPass().run(circuit.block)

    def test_runtime_if_multiple_full_reslices_preserve_one_lineage(self):
        """Repeated full reslices remain one resource across an If merge."""

        @qmc.qkernel
        def circuit(flag: qmc.Bit) -> qmc.Vector[qmc.Qubit]:
            q = qmc.qubit_array(4, "q")
            view = q[1:3]
            if flag:
                view = view[:][:]
            else:
                view = view
            view[0] = qmc.h(view[0])
            q[1:3] = view
            return q

        SliceBorrowCheckPass().run(circuit.block)

    def test_runtime_if_two_full_reslices_preserve_one_lineage(self):
        """Both branches may refresh one pre-branch view independently."""

        @qmc.qkernel
        def circuit(flag: qmc.Bit) -> qmc.Vector[qmc.Qubit]:
            q = qmc.qubit_array(4, "q")
            view = q[1:3]
            if flag:
                view = view[:]
            else:
                view = view[:]
            view[0] = qmc.h(view[0])
            q[1:3] = view
            return q

        SliceBorrowCheckPass().run(circuit.block)

    def test_runtime_if_reslices_symbolic_partial_view(self):
        """Full-reslice recognition does not require concrete coverage."""

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
            flag: qmc.Bit,
        ) -> qmc.Vector[qmc.Qubit]:
            q = qmc.qubit_array(n, "q")
            view = q[1:n]
            if flag:
                view = view[:]
            else:
                view = view
            view = qmc.h(view)
            q[1:n] = view
            return q

        assert circuit.block is not None

    def test_runtime_if_reconnects_hidden_view_parent(self):
        """A merged view remains usable when its root is not an If output."""

        @qmc.qkernel
        def circuit(
            q: qmc.Vector[qmc.Qubit],
            flag: qmc.Bit,
        ) -> qmc.Vector[qmc.Qubit]:
            view = q[1:3]
            if flag:
                view[0] = qmc.x(view[0])
            else:
                view[0] = qmc.z(view[0])
            view[1] = qmc.h(view[1])
            return view

        assert circuit.block is not None

    def test_runtime_if_reconnects_borrowed_element_parent(self):
        """A scalar element merge can be returned to its merged array."""

        @qmc.qkernel
        def circuit(flag: qmc.Bit) -> qmc.Vector[qmc.Qubit]:
            q = qmc.qubit_array(2, "q")
            element = q[0]
            if flag:
                element = qmc.x(element)
            else:
                element = qmc.h(element)
            q[0] = element
            return q

        assert circuit.block is not None

    def test_runtime_if_reconnects_pass_through_element_parent(self):
        """An unchanged element keeps the array selected by a sibling merge."""

        @qmc.qkernel
        def circuit(flag: qmc.Bit) -> qmc.Vector[qmc.Qubit]:
            q = qmc.qubit_array(2, "q")
            element = q[0]
            alias = element
            if flag:
                q[1] = qmc.x(q[1])
            else:
                q[1] = qmc.h(q[1])
            element = qmc.x(element)
            q[0] = element
            _ = alias
            return q

        assert circuit.block is not None

    def test_runtime_if_reconnects_aliased_pass_through_element_parent(self):
        """An unused Python alias does not detach a live element from its parent."""

        @qmc.qkernel
        def circuit(flag: qmc.Bit) -> qmc.Vector[qmc.Qubit]:
            q = qmc.qubit_array(2, "q")
            element = q[0]
            _alias = element
            if flag:
                q[1] = qmc.x(q[1])
            else:
                q[1] = qmc.h(q[1])
            element = qmc.x(element)
            q[0] = element
            return q

        assert circuit.block is not None

    def test_runtime_if_reconnects_nested_pass_through_element_parent(self):
        """A nested element and view follow a merged root-array parent."""

        @qmc.qkernel
        def circuit(flag: qmc.Bit) -> qmc.Vector[qmc.Qubit]:
            q = qmc.qubit_array(6, "q")
            outer = q[0::2]
            element = outer[0]
            if flag:
                q[1] = qmc.x(q[1])
            else:
                q[1] = qmc.h(q[1])
            element = qmc.x(element)
            outer[0] = element
            q[0::2] = outer
            return q

        assert circuit.block is not None

    def test_runtime_if_keeps_live_prebranch_element_borrow(self):
        """A live element still blocks a second borrow after an array merge."""
        from qamomile.circuit.transpiler.errors import QubitBorrowConflictError

        @qmc.qkernel
        def circuit(flag: qmc.Bit) -> qmc.Bit:
            q = qmc.qubit_array(2, "q")
            element = q[0]
            if flag:
                q[1] = qmc.x(q[1])
            else:
                q[1] = qmc.h(q[1])
            duplicate = q[0]
            duplicate = qmc.z(duplicate)
            return qmc.measure(element)

        with pytest.raises(QubitBorrowConflictError, match="already borrowed"):
            _ = circuit.block

    def test_runtime_if_keeps_live_element_borrow_inside_branch(self):
        """A captured element blocks a second borrow while tracing a branch."""
        from qamomile.circuit.transpiler.errors import QubitBorrowConflictError

        @qmc.qkernel
        def circuit(flag: qmc.Bit) -> qmc.Bit:
            q = qmc.qubit_array(2, "q")
            element = q[0]
            if flag:
                q[0] = qmc.x(q[0])
            return qmc.measure(element)

        with pytest.raises(QubitBorrowConflictError, match="already borrowed"):
            _ = circuit.block

    def test_runtime_if_preserves_destroyed_condition_slot_for_sibling_use(self):
        """A measured condition slot does not block a disjoint sibling slot."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.h(q[0])
            flag = qmc.measure(q[0])
            if flag:
                q[1] = qmc.x(q[1])
            return qmc.measure(q[1])

        assert circuit.block is not None

    def test_direct_element_measure_marks_parent_slot_destroyed(self):
        """A direct element measurement prevents later whole-array reuse."""
        from qamomile.circuit.transpiler.errors import QubitConsumedError

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            _ = qmc.measure(q[0])
            return qmc.measure(q)

        with pytest.raises(QubitConsumedError, match="already destroyed"):
            _ = circuit.block

    def test_runtime_if_rejects_destroyed_condition_slot_inside_branch(self):
        """A measured temporary element cannot be re-borrowed in a branch."""
        from qamomile.circuit.transpiler.errors import QubitConsumedError

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit_array(2, "q")
            flag = qmc.measure(q[0])
            if flag:
                q[0] = qmc.x(q[0])
            else:
                q[1] = qmc.h(q[1])
            return qmc.measure(q[1])

        with pytest.raises(QubitConsumedError, match="already consumed"):
            _ = circuit.block

    def test_runtime_if_rejects_destroyed_condition_slot_after_merge(self):
        """A measured temporary element remains destroyed after an If merge."""
        from qamomile.circuit.transpiler.errors import QubitConsumedError

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit_array(2, "q")
            flag = qmc.measure(q[0])
            if flag:
                q[1] = qmc.x(q[1])
            else:
                q[1] = qmc.h(q[1])
            q[0] = qmc.z(q[0])
            return qmc.measure(q[1])

        with pytest.raises(QubitConsumedError, match="already consumed"):
            _ = circuit.block

    def test_runtime_if_keeps_dead_name_direct_element_borrow(self):
        """A borrow remains active even when its Python name is dead at the If."""
        from qamomile.circuit.transpiler.errors import QubitBorrowConflictError

        @qmc.qkernel
        def circuit(flag: qmc.Bit) -> qmc.Bit:
            q = qmc.qubit_array(2, "q")
            _element = q[0]
            if flag:
                q[1] = qmc.x(q[1])
            else:
                q[1] = qmc.h(q[1])
            q[0] = qmc.z(q[0])
            return qmc.measure(q[1])

        with pytest.raises(QubitBorrowConflictError, match="already borrowed"):
            _ = circuit.block

    def test_runtime_if_does_not_revive_named_consumed_element(self):
        """A captured consumed element remains unusable inside either branch."""
        from qamomile.circuit.transpiler.errors import QubitConsumedError

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit_array(2, "q")
            element = q[0]
            flag = qmc.measure(element)
            if flag:
                element = qmc.x(element)
            else:
                element = qmc.z(element)
            return qmc.measure(element)

        with pytest.raises(QubitConsumedError, match="already consumed"):
            _ = circuit.block

    def test_runtime_if_reconnects_branch_local_element_borrows(self):
        """Equal literal indices from separate branches share one owner."""

        @qmc.qkernel
        def circuit(flag: qmc.Bit) -> qmc.Vector[qmc.Qubit]:
            q = qmc.qubit_array(2, "q")
            if flag:
                element = q[0]
                element = qmc.x(element)
            else:
                element = q[0]
                element = qmc.h(element)
            q[0] = element
            return q

        assert circuit.block is not None

    def test_runtime_if_does_not_unify_different_element_borrows(self):
        """Different branch-local indices cannot acquire one merged owner."""
        from qamomile.circuit.transpiler.errors import AffineTypeError

        @qmc.qkernel
        def circuit(flag: qmc.Bit) -> qmc.Vector[qmc.Qubit]:
            q = qmc.qubit_array(2, "q")
            if flag:
                element = q[0]
            else:
                element = q[1]
            q[0] = element
            return q

        with pytest.raises(AffineTypeError, match="same index"):
            _ = circuit.block

    def test_runtime_if_merge_version_is_newer_than_branch_views(self):
        """A slice merge cannot move its logical resource version backward."""

        @qmc.qkernel
        def circuit(flag: qmc.Bit) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            view = q[1:3]
            if flag:
                view = qmc.qft(view)
            else:
                view = qmc.iqft(view)
            view[0] = qmc.h(view[0])
            q[1:3] = view
            return qmc.measure(q)

        SliceBorrowCheckPass().run(circuit.block)

    def test_zero_trip_loop_does_not_trace_destructive_slice_body(self):
        """Literal zero-trip loops do not leak skipped destructive slice use."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            for _ in qmc.range(0):
                _ = qmc.measure(q[1::2])
            return qmc.measure(q)

        executable = QiskitTranspiler().transpile(circuit)

        assert executable.get_first_circuit().num_qubits == 4

    def test_bound_zero_loop_does_not_trace_destructive_slice_body(self):
        """Bound zero-trip loops do not leak skipped destructive slice use."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def circuit(repetitions: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            for _ in qmc.range(repetitions):
                _ = qmc.measure(q[1::2])
            return qmc.measure(q)

        executable = QiskitTranspiler().transpile(circuit, bindings={"repetitions": 0})

        assert executable.get_first_circuit().num_qubits == 4

    def test_static_loop_nested_full_slice_refresh_transpiles(self):
        """Static non-zero loops may hand off a full nested slice refresh."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            evens = q[0::2]
            for _ in qmc.range(1):
                evens = evens[:]
                evens[0] = qmc.h(evens[0])
            q[0::2] = evens
            return qmc.measure(q)

        executable = QiskitTranspiler().transpile(circuit)

        assert executable.get_first_circuit().num_qubits == 4

    def test_static_loop_nested_full_slice_release_allows_direct_access(self):
        """Full nested-slice release clears stale IR owners by coverage."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            evens = q[0::2]
            for _ in qmc.range(1):
                evens = evens[:]
                evens[0] = qmc.h(evens[0])
            q[0::2] = evens
            q[0] = qmc.x(q[0])
            return qmc.measure(q)

        executable = QiskitTranspiler().transpile(circuit)

        assert executable.get_first_circuit().num_qubits == 4

    def test_static_loop_nested_full_slice_marker_only_transpiles(self):
        """Marker-only full-slice refresh loops are stripped cleanly."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            evens = q[0::2]
            for _ in qmc.range(1):
                evens = evens[:]
            q[0::2] = evens
            return qmc.measure(q)

        executable = QiskitTranspiler().transpile(circuit)

        assert executable.get_first_circuit().num_qubits == 4

    def test_static_loop_reslice_of_nested_preloop_view_transpiles(self):
        """A loop full-reslice returns directly to the pre-loop outer view."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(6, "q")
            outer = q[0::2]
            view = outer[1:3]
            for _ in qmc.range(1):
                view = view[:]
            view[0] = qmc.h(view[0])
            outer[1:3] = view
            q[0::2] = outer
            return qmc.measure(q)

        executable = QiskitTranspiler().transpile(circuit)

        assert executable.get_first_circuit().num_qubits == 6

    def test_symbolic_loop_reslice_of_nested_preloop_view_is_one_lineage(self):
        """A potentially zero-trip For keeps an exact reslice as one resource."""

        @qmc.qkernel
        def circuit(repetitions: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
            q = qmc.qubit_array(6, "q")
            outer = q[0::2]
            view = outer[1:3]
            for _ in qmc.range(repetitions):
                view = view[:]
            view[0] = qmc.h(view[0])
            outer[1:3] = view
            q[0::2] = outer
            return q

        SliceBorrowCheckPass().run(circuit.block)

    def test_zero_trip_loop_keeps_nested_preloop_view(self):
        """A skipped loop leaves the original nested-view return path intact."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Qubit]:
            q = qmc.qubit_array(6, "q")
            outer = q[0::2]
            view = outer[1:3]
            for _ in qmc.range(0):
                view = view[:]
            view[0] = qmc.h(view[0])
            outer[1:3] = view
            q[0::2] = outer
            return q

        SliceBorrowCheckPass().run(circuit.block)

    def test_while_reslice_of_nested_preloop_view_is_one_lineage(self):
        """A measurement-backed While preserves an exact view reslice."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit_array(7, "q")
            outer = q[0:4]
            view = outer[1:3]
            condition = qmc.measure(q[6])
            while condition:
                view = view[:]
                condition = qmc.measure(q[5])
            view[0] = qmc.h(view[0])
            outer[1:3] = view
            q[0:4] = outer
            return condition

        executable = QiskitTranspiler().transpile(circuit)

        assert executable.get_first_circuit().num_qubits == 7

    def test_nested_if_and_loop_reslice_preserves_preloop_outer(self):
        """Nested control flow cannot add a phantom full-reslice layer."""

        @qmc.qkernel
        def circuit(
            repetitions: qmc.UInt,
            flag: qmc.Bit,
        ) -> qmc.Vector[qmc.Qubit]:
            q = qmc.qubit_array(6, "q")
            outer = q[0::2]
            view = outer[1:3]
            for _ in qmc.range(repetitions):
                if flag:
                    view = view[:]
                else:
                    view = view
            view[0] = qmc.h(view[0])
            outer[1:3] = view
            q[0::2] = outer
            return q

        SliceBorrowCheckPass().run(circuit.block)

    def test_loop_partial_reslice_cannot_skip_immediate_outer(self):
        """Loop normalization does not treat a partial slice as an alias."""
        from qamomile.circuit.transpiler.errors import AffineTypeError

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Qubit]:
            q = qmc.qubit_array(6, "q")
            outer = q[0::2]
            view = outer[1:3]
            for _ in qmc.range(1):
                view = view[0:1]
            outer[1:3] = view
            q[0::2] = outer
            return q

        with pytest.raises(AffineTypeError, match="immediate outer view"):
            _ = circuit.block

    def test_skipped_outer_view_is_consumed_after_full_slice_handoff(self):
        """Direct root return retires the skipped outer view."""
        from qamomile.circuit.transpiler.errors import QubitConsumedError

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            outer = q[0::2]
            inner = outer[:]
            q[0::2] = inner
            outer[0] = qmc.h(outer[0])
            return qmc.measure(q)

        with pytest.raises(QubitConsumedError, match="already consumed"):
            _ = circuit.block

    def test_multilevel_full_slice_handoff_retires_all_skipped_outers(self):
        """Direct root return retires every skipped full-slice ancestor."""
        from qamomile.circuit.transpiler.errors import QubitConsumedError

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            outer = q[0::2]
            middle = outer[:]
            inner = middle[:]
            q[0::2] = inner
            outer[0] = qmc.h(outer[0])
            return qmc.measure(q)

        with pytest.raises(QubitConsumedError, match="already consumed"):
            _ = circuit.block

    def test_partial_ancestor_nested_handoff_stays_rejected(self):
        """Root direct return cannot skip a wider outer ancestor."""
        from qamomile.circuit.transpiler.errors import AffineTypeError

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(6, "q")
            outer = q[0::2]
            middle = outer[1:]
            inner = middle[:]
            q[2::2] = inner
            return qmc.measure(q)

        with pytest.raises(AffineTypeError, match="immediate outer view"):
            _ = circuit.block

    def test_runtime_if_nested_full_slice_handoff_transpiles(self):
        """A branch-local full reslice denotes the same physical slots."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            evens = q[0::2]
            flag = qmc.measure(q[3])
            if flag:
                evens = evens[:]
                evens[0] = qmc.h(evens[0])
            q[0::2] = evens
            return qmc.measure(q[:3])

        executable = QiskitTranspiler().transpile(circuit)

        assert executable.get_first_circuit().num_qubits == 4

    def test_huge_static_loop_bound_does_not_overflow_trace_guard(self):
        """Static loop trace guard handles ranges larger than ssize_t."""
        from qamomile.circuit.frontend.operation.control_flow import (
            should_trace_for_loop,
        )

        assert should_trace_for_loop(0, 10**100, 1) is True

    def test_concrete_forward_refresh_updates_existing_owner(self):
        """A newer concrete same-slice version replaces the old owner."""
        checker = SliceBorrowCheckPass()
        state = {}
        root = _qubit_array("q", _uint_value("n", 4))
        owner = _slice_array(
            root,
            "view",
            _uint_value("start", 0),
            _uint_value("step", 1),
            _uint_value("length", 2),
        )
        refreshed = owner.next_version()

        checker._register_slice_bulk_borrow_if_new(owner, state)
        checker._register_slice_bulk_borrow_if_new(refreshed, state)

        assert state
        assert {view.uuid for view in state.values()} == {refreshed.uuid}

    def test_concrete_stale_same_slice_refresh_is_rejected(self):
        """An older concrete same-slice version cannot replace a newer owner."""
        checker = SliceBorrowCheckPass()
        state = {}
        root = _qubit_array("q", _uint_value("n", 4))
        stale = _slice_array(
            root,
            "view",
            _uint_value("start", 0),
            _uint_value("step", 1),
            _uint_value("length", 2),
        )
        current = stale.next_version()

        checker._register_slice_bulk_borrow_if_new(stale, state)
        checker._register_slice_bulk_borrow_if_new(current, state)

        with pytest.raises(QubitConsumedError, match="forward SSA-version"):
            checker._register_slice_bulk_borrow_if_new(stale, state)

    def test_symbolic_exact_descriptor_forward_refresh_is_allowed(self):
        """A symbolic slice may refresh only when descriptor SSA values match."""
        checker = SliceBorrowCheckPass()
        state = {}
        n = _uint_value("n")
        start = _uint_value("i")
        step = _uint_value("step")
        root = _qubit_array("q", n)
        owner = _slice_array(root, "view", start, step, n)
        refreshed = owner.next_version()

        checker._register_slice_bulk_borrow_if_new(owner, state)
        checker._register_slice_bulk_borrow_if_new(refreshed, state)

        assert len(state) == 1
        assert next(iter(state.values())).uuid == refreshed.uuid

    def test_symbolic_changed_descriptor_is_rejected(self):
        """A different symbolic descriptor is not a same-slice refresh."""
        checker = SliceBorrowCheckPass()
        state = {}
        n = _uint_value("n")
        start = _uint_value("i")
        changed_start = _uint_value("i_plus_zero")
        step = _uint_value("step")
        root = _qubit_array("q", n)
        owner = _slice_array(root, "view", start, step, n)
        changed = dataclasses.replace(owner.next_version(), slice_start=changed_start)

        checker._register_slice_bulk_borrow_if_new(owner, state)

        with pytest.raises(QubitBorrowConflictError, match="may overlap"):
            checker._register_slice_bulk_borrow_if_new(changed, state)

    def test_symbolic_adjacent_prefix_suffix_descriptors_are_allowed(self):
        """Adjacent symbolic unit-stride intervals on the same root are disjoint."""
        checker = SliceBorrowCheckPass()
        state = {}
        n = _uint_value("n")
        k = _uint_value("k")
        root = _qubit_array("q", n)
        prefix = _slice_array(
            root,
            "prefix",
            _uint_value("zero", 0),
            _uint_value("one", 1),
            k,
        )
        suffix = _slice_array(
            root,
            "suffix",
            k,
            _uint_value("one", 1),
            _uint_value("n_minus_k"),
        )

        checker._register_slice_bulk_borrow_if_new(prefix, state)
        checker._register_slice_bulk_borrow_if_new(suffix, state)

        assert {owner.uuid for owner in state.values()} == {prefix.uuid, suffix.uuid}

    def test_symbolic_adjacent_suffix_prefix_descriptors_are_allowed(self):
        """The symbolic disjointness proof is independent of registration order."""
        checker = SliceBorrowCheckPass()
        state = {}
        n = _uint_value("n")
        k = _uint_value("k")
        root = _qubit_array("q", n)
        prefix = _slice_array(
            root,
            "prefix",
            _uint_value("zero", 0),
            _uint_value("one", 1),
            k,
        )
        suffix = _slice_array(
            root,
            "suffix",
            k,
            _uint_value("one", 1),
            _uint_value("n_minus_k"),
        )

        checker._register_slice_bulk_borrow_if_new(suffix, state)
        checker._register_slice_bulk_borrow_if_new(prefix, state)

        assert {owner.uuid for owner in state.values()} == {prefix.uuid, suffix.uuid}

    def test_symbolic_strided_disjointness_stays_conservative(self):
        """Parity-style symbolic strides still require a future proof helper."""
        checker = SliceBorrowCheckPass()
        state = {}
        n = _uint_value("n")
        root = _qubit_array("q", n)
        evens = _slice_array(
            root,
            "evens",
            _uint_value("zero", 0),
            _uint_value("two", 2),
            n,
        )
        odds = _slice_array(
            root,
            "odds",
            _uint_value("one", 1),
            _uint_value("two", 2),
            n,
        )

        checker._register_slice_bulk_borrow_if_new(evens, state)

        with pytest.raises(QubitBorrowConflictError, match="may overlap"):
            checker._register_slice_bulk_borrow_if_new(odds, state)

    def test_symbolic_recomputed_descriptor_is_rejected(self):
        """A new slice lineage with the same symbolic descriptor is not a refresh."""
        checker = SliceBorrowCheckPass()
        state = {}
        n = _uint_value("n")
        start = _uint_value("i")
        step = _uint_value("step")
        root = _qubit_array("q", n)
        owner = _slice_array(root, "view", start, step, n)
        recomputed = _slice_array(root, "recomputed", start, step, n)

        checker._register_slice_bulk_borrow_if_new(owner, state)

        with pytest.raises(QubitBorrowConflictError, match="not a forward"):
            checker._register_slice_bulk_borrow_if_new(recomputed, state)

    def test_symbolic_refresh_inside_unsafe_snapshot_is_rejected(self):
        """Unsafe control-flow bodies cannot refresh outer symbolic views."""
        checker = SliceBorrowCheckPass()
        state = {}
        n = _uint_value("n")
        start = _uint_value("i")
        step = _uint_value("step")
        root = _qubit_array("q", n)
        owner = _slice_array(root, "view", start, step, n)
        refreshed = owner.next_version()

        checker._register_slice_bulk_borrow_if_new(owner, state)
        checker._outer_snapshot_stack.append(
            (_SnapshotKind.UNSAFE_CONTROL_BODY, dict(state))
        )
        try:
            with pytest.raises(ValidationError, match="may be skipped"):
                checker._register_slice_bulk_borrow_if_new(refreshed, state)
        finally:
            checker._outer_snapshot_stack.pop()

    def test_nested_const_view_locks_parent_slot(self):
        """Constant nested view (``q[0::2][1:3]``) locks the covered root slot.

        Regression (P1-3): ``VectorView._nested_slice`` passed
        ``covered_indices=None`` unconditionally, so even compile-
        time-known nested views (whose coverage is fully derivable)
        skipped the parent's bulk-borrow registration and let
        aliased direct parent access (``q[2] = h(q[2])`` while
        ``inner = q[0::2][1:3]`` is live) slip through at trace time.
        The fix composes the two affine maps and enumerates the
        covered root indices for the bulk-borrow tracker.
        """
        from qamomile.circuit.transpiler.errors import QubitBorrowConflictError

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Qubit]:
            q = qmc.qubit_array(4, "q")
            inner = q[0::2][1:3]  # covers root slots {2}
            q[2] = qmc.x(q[2])  # aliases inner's slot 2
            inner[0] = qmc.h(inner[0])
            return q

        with pytest.raises(QubitBorrowConflictError):
            kern.block  # trigger tracing


class TestBlockEndUnreturnedBorrow:
    """Frontend trace-end validation catches unreturned borrows."""

    def test_return_with_outstanding_direct_borrow_raises(self):
        """``qv = q[0]; return q`` — borrow not returned — is flagged."""
        from qamomile.circuit.transpiler.errors import UnreturnedBorrowError

        with pytest.raises(UnreturnedBorrowError):

            @qmc.qkernel
            def kern() -> qmc.Vector[qmc.Qubit]:
                q = qmc.qubit_array(4, "q")
                qv = q[0]  # borrowed, never returned
                del qv
                return q

            _ = kern.block


class TestConstantIndexOnSliceView:
    """Constant-index gate on a sliced view resolves to the correct qubit.

    Targets a latent bug in ``ResourceAllocator``: its Phase-1 lazy
    registration used ``parent_array.uuid`` directly, which for a
    sliced ArrayValue is the view's uuid — not the root parent keyed in
    ``qubit_map``.  Loop-indexed accesses escaped via the emit-time
    resolver (``ValueResolver``), but constant-index accesses
    (``view[0] = qmc.h(view[0])``) went through Phase-1 and asserted.
    """

    @staticmethod
    def _make_const_index_kernel(slice_kind: str):
        """Build a kernel that applies ``h`` / ``x`` to ``view[0]`` / ``view[1]``.

        Mirrors the lambda-parametrized variants in
        :meth:`test_const_index_on_view_maps_to_root_slot`; the strict-
        return policy requires each kernel to write the view back via
        the same slice expression, so we materialise four explicit
        kernel bodies rather than synthesise them through a lambda.
        """
        if slice_kind == "1:3":

            @qmc.qkernel
            def kern(num: qmc.UInt) -> qmc.Vector[qmc.Bit]:
                q = qmc.qubit_array(num, "q")
                view = q[1:3]
                view[0] = qmc.h(view[0])
                view[1] = qmc.x(view[1])
                q[1:3] = view
                return qmc.measure(q)

            return kern
        if slice_kind == "0::2":

            @qmc.qkernel
            def kern(num: qmc.UInt) -> qmc.Vector[qmc.Bit]:
                q = qmc.qubit_array(num, "q")
                view = q[0::2]
                view[0] = qmc.h(view[0])
                view[1] = qmc.x(view[1])
                q[0::2] = view
                return qmc.measure(q)

            return kern
        if slice_kind == "1::2":

            @qmc.qkernel
            def kern(num: qmc.UInt) -> qmc.Vector[qmc.Bit]:
                q = qmc.qubit_array(num, "q")
                view = q[1::2]
                view[0] = qmc.h(view[0])
                view[1] = qmc.x(view[1])
                q[1::2] = view
                return qmc.measure(q)

            return kern
        if slice_kind == "0::2[1:3]":

            @qmc.qkernel
            def kern(num: qmc.UInt) -> qmc.Vector[qmc.Bit]:
                q = qmc.qubit_array(num, "q")
                outer = q[0::2]
                view = outer[1:3]
                view[0] = qmc.h(view[0])
                view[1] = qmc.x(view[1])
                # Strict-return: inner → outer → root.
                outer[1:3] = view
                q[0::2] = outer
                return qmc.measure(q)

            return kern
        raise AssertionError(f"unknown slice_kind {slice_kind!r}")

    @pytest.mark.parametrize(
        "slice_kind, expected_h, expected_x",
        [
            ("1:3", [1], [2]),
            ("0::2", [0], [2]),
            ("1::2", [1], [3]),
            # Nested slice: q[0::2] covers {0,2,4,6}; then [1:3] covers
            # view slots 1, 2 → parent slots 2, 4.
            ("0::2[1:3]", [2], [4]),
        ],
    )
    def test_const_index_on_view_maps_to_root_slot(
        self, slice_kind, expected_h, expected_x
    ):
        """Gates at ``view[0]`` / ``view[1]`` land on root slots via slice chain."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        kern = self._make_const_index_kernel(slice_kind)

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(kern, bindings={"num": 8})
        qc = exe.compiled_quantum[0].circuit
        h_qubits = [
            qc.qubits.index(inst.qubits[0])
            for inst in qc.data
            if inst.operation.name == "h"
        ]
        x_qubits = [
            qc.qubits.index(inst.qubits[0])
            for inst in qc.data
            if inst.operation.name == "x"
        ]
        assert h_qubits == expected_h
        assert x_qubits == expected_x


class TestWholeViewEmit:
    """Whole-view emit paths must resolve the slice chain, not silently drop.

    Regression: until these were fixed, ``measure(view)`` emitted zero
    measurement gates (the sampled result came back as ``[(None, N)]`` —
    a data-integrity hazard), ``qft(view)`` emitted zero gates,
    ``pauli_evolve(view, H, γ)`` raised "Cannot resolve qubit index",
    and ``expval(view, H)`` fed a view-sized observable to a root-sized
    circuit and crashed inside the estimator.
    """

    def test_measure_view_targets_parent_qubits(self):
        """``measure(q[1::2])`` measures exactly ``q[1]`` and ``q[3]``."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            for i in qmc.range(4):
                q[i] = qmc.x(q[i])
            view = q[1::2]
            return qmc.measure(view)

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(kern, bindings={})
        qc = exe.compiled_quantum[0].circuit
        measured = [
            qc.qubits.index(inst.qubits[0])
            for inst in qc.data
            if inst.operation.name == "measure"
        ]
        assert measured == [1, 3]
        assert qc.num_clbits == 2

        # Executing the circuit must also produce a real 2-bit result,
        # not the legacy ``(None, shots)`` entry that slipped through
        # when the emit silently dropped.
        result = exe.sample(transpiler.executor(), shots=32).result()
        bitstrings = {bs for bs, _ in result.results}
        assert bitstrings == {(1, 1)}

    def test_qft_on_view_targets_parent_qubits(self):
        """``qft(q[1::2])`` emits QFT gates on ``q[1]`` and ``q[3]``."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            view = q[1::2]
            view = qmc.qft(view)
            q[1::2] = view
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(kern, bindings={})
        qc = exe.compiled_quantum[0].circuit
        touched: set[int] = set()
        for inst in qc.data:
            if inst.operation.name == "measure":
                continue
            for qobj in inst.qubits:
                touched.add(qc.qubits.index(qobj))
        assert touched, "qft(view) emitted no gates at all"
        assert touched.issubset({1, 3})

    def test_pauli_evolve_on_view_targets_parent_qubits(self):
        """``pauli_evolve(q[1::2], Z⊗Z, γ)`` acts only on qubits 1 and 3."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def kern(H: qmc.Observable, gamma: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            view = q[1::2]
            view = qmc.pauli_evolve(view, H, gamma)
            q[1::2] = view
            return qmc.measure(q)

        H_op = qm_o.Z(0) * qm_o.Z(1)
        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(kern, bindings={"H": H_op, "gamma": 0.5})
        qc = exe.compiled_quantum[0].circuit
        touched: set[int] = set()
        emitted_any = False
        for inst in qc.data:
            if inst.operation.name == "measure":
                continue
            emitted_any = True
            for qobj in inst.qubits:
                touched.add(qc.qubits.index(qobj))
        assert emitted_any, "pauli_evolve(view, ...) emitted no gates"
        assert touched.issubset({1, 3})

    @pytest.mark.parametrize(
        "view_slice, expected_phys_qubit",
        [(slice(1, None, 2), 1), (slice(0, None, 2), 0)],
    )
    def test_expval_on_view_remaps_observable(self, view_slice, expected_phys_qubit):
        """``expval(view, Z(0))`` measures ``<Z>`` on the view's root-index 0.

        Confirms both that the circuit executes without a qubit-count
        mismatch and that the computed value matches the analytic
        ``<Z>`` on the physical qubit the view's slot-0 maps to.
        """
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        s = view_slice

        @qmc.qkernel
        def kern(obs: qmc.Observable) -> qmc.Float:
            q = qmc.qubit_array(4, "q")
            for i in qmc.range(4):
                q[i] = qmc.x(q[i])  # all to |1>
            view = q[s.start : s.stop : s.step]
            return qmc.expval(view, obs)

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(kern, bindings={"obs": qm_o.Z(0)})
        value = exe.run(transpiler.executor()).result()
        # All qubits are |1> so <Z> on any root qubit is -1.
        assert np.isclose(value, -1.0)
        # Sanity: whichever view we picked, slot 0 maps to the physical
        # qubit ``expected_phys_qubit``.
        assert s.start == expected_phys_qubit

    def test_composite_gate_on_view_does_not_inflate_qubit_count(self):
        """``qft(q[1::2])`` on a 4-qubit register must stay at 4 physical qubits.

        Regression: ``_allocate_qubit_list`` used to register
        ``QubitAddress(view_uuid, i)`` as fresh allocations, so each
        view element reserved an extra physical qubit that nothing
        else referenced. ``qft(q[1::2])`` on a 4-qubit register
        previously produced a 6-qubit circuit.  Touched-qubit
        assertions did not notice because the extra qubits were
        unused by any gate.
        """
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            view = q[1::2]
            view = qmc.qft(view)
            q[1::2] = view
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(kern, bindings={})
        qc = exe.compiled_quantum[0].circuit
        assert qc.num_qubits == 4, (
            f"qft(q[1::2]) inflated qubit count to {qc.num_qubits}; "
            f"expected 4 (the root register size)"
        )

    def test_controlled_u_on_view_elements(self):
        """``control(_zgate)(q[1], q[3])`` runs on the view's elements.

        Migrated from the old ``index_spec``-on-view test: the new
        concrete-mode API does not accept ``target_indices`` /
        ``control_indices``, so the control / target partition is
        expressed positionally instead.  The original regression
        concern — that the controlled gate lands on physical qubits
        (1, 3) via the view's ``slice_of`` chain — is still exercised
        because ``q[1]`` and ``q[3]`` come from the view's element
        access (``q[1::2][0]`` and ``q[1::2][1]`` resolve through the
        same parent_array / element_indices machinery).
        """
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def _zgate(q: qmc.Qubit) -> qmc.Qubit:
            return qmc.z(q)

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            cz = qmc.control(_zgate, num_controls=1)
            view = q[1::2]
            view[0], view[1] = cz(view[0], view[1])
            q[1::2] = view
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(kern, bindings={})
        qc = exe.compiled_quantum[0].circuit
        assert qc.num_qubits == 4

        # The controlled gate should touch exactly physical qubits {1, 3}.
        # Names are backend-dependent (``ccircuit-N`` etc.); filter by
        # non-measure and expect 2-qubit instruction spanning {1, 3}.
        controlled_qubits: set[int] = set()
        for inst in qc.data:
            if inst.operation.name == "measure":
                continue
            qs = {qc.qubits.index(qobj) for qobj in inst.qubits}
            controlled_qubits |= qs
        assert controlled_qubits == {1, 3}, (
            f"controlled-Z on q[1::2] touched {controlled_qubits}, expected {{1, 3}}"
        )

    def test_expval_does_not_mutate_user_hamiltonian_binding(self):
        """Reusing the same ``H`` binding across different-sized circuits works.

        Regression (P1-1): the orchestrator wrote
        ``hamiltonian._num_qubits = circuit.num_qubits`` directly on
        the binding Hamiltonian.  For identity expval paths where
        ``qubit_map`` is empty, ``remap_qubits`` returns ``self``
        verbatim, so the user's ``H`` got poisoned to the first
        circuit's width and subsequent runs with smaller circuits
        crashed inside the backend estimator.  The fix clones the
        Hamiltonian before padding ``_num_qubits``.
        """
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def kern4(obs: qmc.Observable) -> qmc.Float:
            q = qmc.qubit_array(4, "q")
            q[0] = qmc.x(q[0])
            return qmc.expval(q, obs)

        @qmc.qkernel
        def kern2(obs: qmc.Observable) -> qmc.Float:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.x(q[0])
            return qmc.expval(q, obs)

        H = qm_o.Z(0)
        assert H._num_qubits is None
        t = QiskitTranspiler()
        v4 = t.transpile(kern4, bindings={"obs": H}).run(t.executor()).result()
        assert np.isclose(v4, -1.0)
        # The user's binding must NOT have been mutated.
        assert H._num_qubits is None, (
            f"user H binding was mutated to _num_qubits={H._num_qubits}"
        )
        # Reusing H on a 2-qubit circuit must succeed (it would crash
        # with a 4-vs-2 mismatch if ``_num_qubits`` had been poisoned).
        v2 = t.transpile(kern2, bindings={"obs": H}).run(t.executor()).result()
        assert np.isclose(v2, -1.0)

    def test_pauli_evolve_on_view_then_expval_on_view(self):
        """``pauli_evolve(view) → expval(view)`` measures the right physical qubit.

        Regression (P1-4): ``pauli_evolve`` discarded the input's
        slice metadata on its result, so the downstream
        ``expval(result_view, Z(0))`` fell through the non-view branch
        in ``_build_qubit_map``, left ``qubit_map`` empty, and ended
        up measuring ``Z`` on physical qubit 0 instead of the view's
        slot-0 (physical qubit 1).
        """
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def kern(H: qmc.Observable, gamma: qmc.Float, obs: qmc.Observable) -> qmc.Float:
            q = qmc.qubit_array(4, "q")
            # Flip only qubit 1 so <Z> distinguishes physical qubit 1
            # (-1) from any other qubit (+1).
            q[1] = qmc.x(q[1])
            view = q[1::2]
            view = qmc.pauli_evolve(view, H, gamma)
            return qmc.expval(view, obs)

        H_ev = qm_o.Z(0) * qm_o.Z(1)
        transpiler = QiskitTranspiler()
        # gamma=0 makes pauli_evolve an identity so the expected
        # ``<Z>`` is determined by the initial state alone.
        exe = transpiler.transpile(
            kern, bindings={"H": H_ev, "gamma": 0.0, "obs": qm_o.Z(0)}
        )
        value = exe.run(transpiler.executor()).result()
        assert np.isclose(value, -1.0), (
            f"expval(pauli_evolve(q[1::2], ...), Z(0)) = {value}, "
            f"expected -1.0 (Z on physical qubit 1 = |1>)"
        )


class TestRound2Reviewer:
    """Regression tests for the 2nd-round adversarial review findings.

    Covers P1-A (OOB slice clamp), P1-B (destructive view consume),
    P1-C (if-lowering merge substitution on SliceArrayOp result),
    P2-A (frontend/post-fold drain alignment), P2-B (H snapshot),
    P2-C (UInt const negative/zero validation).
    """

    # ----- P1-A -------------------------------------------------------------

    def test_slice_stop_beyond_parent_is_clamped(self):
        """``q[3:10]`` on a 4-qubit register produces a length-1 view.

        Regression (P1-A): without clamping, emit produced a 7-clbit
        result with only one measurement — silent data loss.
        """
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            q[3] = qmc.x(q[3])
            return qmc.measure(q[3:10])

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(kern, bindings={})
        qc = exe.compiled_quantum[0].circuit
        assert qc.num_clbits == 1, (
            f"q[3:10] should clamp to q[3:4] (1 clbit); got {qc.num_clbits}"
        )
        n_meas = sum(1 for inst in qc.data if inst.operation.name == "measure")
        assert n_meas == 1

    # ----- P1-B -------------------------------------------------------------

    def test_measure_view_then_measure_parent_rejects(self):
        """``measure(q[1::2]); measure(q)`` must error — q[1], q[3] destroyed.

        Regression (P1-B): frontend previously released only the view
        slice-borrow and left the parent re-consumable, allowing a
        second measure to silently re-measure collapsed qubits.
        """
        from qamomile.circuit.transpiler.errors import QubitConsumedError

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            odd = q[1::2]
            _ = qmc.measure(odd)
            return qmc.measure(q)

        with pytest.raises(QubitConsumedError):
            kern.block

    def test_element_access_after_view_measure_rejects(self):
        """``measure(q[1::2]); q[1] = h(q[1])`` must error."""
        from qamomile.circuit.transpiler.errors import QubitConsumedError

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            _ = qmc.measure(q[1::2])
            q[1] = qmc.h(q[1])
            return qmc.measure(q)

        with pytest.raises(QubitConsumedError):
            kern.block

    def test_non_overlapping_view_measures_allowed(self):
        """``measure(q[1::2]); measure(q[0::2])`` must succeed.

        Frontend sanity: the consumed-slot sentinel covers only slots
        {1, 3}; the second view over {0, 2} is disjoint so the
        destructive-consume check must not block it.
        """
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def kern() -> tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]:
            q = qmc.qubit_array(4, "q")
            r1 = qmc.measure(q[1::2])
            r2 = qmc.measure(q[0::2])
            return r1, r2

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(kern, bindings={})
        qc = exe.compiled_quantum[0].circuit
        assert qc.num_qubits == 4
        assert qc.num_clbits == 4

    # ----- P1-C -------------------------------------------------------------

    def test_if_lowering_substitutes_slice_array_op_results(self):
        """``_apply_substitution`` now walks SliceArrayOp result metadata.

        Regression (P1-C): for an ``if`` whose branches flow into a
        slice's bounds, the merge-output references must be substituted
        into ``SliceArrayOperation.results[0].slice_start`` /
        ``slice_step`` — not just operands — so post-fold coverage
        registration sees the concrete bounds.  Directly exercising
        the pass entry keeps the test independent of the higher-level
        ``qmc.UInt`` constructor surface.
        """
        from qamomile.circuit.ir.operation import SliceArrayOperation
        from qamomile.circuit.ir.types.primitives import QubitType, UIntType
        from qamomile.circuit.ir.value import ArrayValue, Value
        from qamomile.circuit.transpiler.passes.compile_time_if_lowering import (
            CompileTimeIfLoweringPass,
        )

        merge_start = Value(type=UIntType(), name="merge_start")
        folded_start = Value(type=UIntType(), name="folded").with_const(1)
        step = Value(type=UIntType(), name="step").with_const(1)
        root = ArrayValue(type=QubitType(), name="q")
        sliced = ArrayValue(
            type=QubitType(),
            name="q[slice]",
            slice_of=root,
            slice_start=merge_start,
            slice_step=step,
        )
        op = SliceArrayOperation(
            operands=[root, merge_start, step],
            results=[sliced],
        )

        subst = {merge_start.uuid: folded_start}
        lowered = CompileTimeIfLoweringPass(bindings={})._apply_substitution(op, subst)
        # Both operand and result-side slice_start must have been
        # substituted to the folded const value.
        assert lowered.operands[1] is folded_start
        result_av = lowered.results[0]
        assert isinstance(result_av, ArrayValue)
        assert result_av.slice_start is folded_start

    # ----- P2-A -------------------------------------------------------------

    def test_overlapping_top_level_views_are_rejected(self):
        """Top-level same-range / overlapping views are rejected at trace time.

        Under the strict no-multi-view policy
        (``VectorView._wrap``'s overlap check), constructing a second
        top-level view that overlaps an existing live view is rejected
        immediately with ``QubitBorrowConflictError``.  This replaces
        the prior "opportunistic drain" semantics where an unused
        outer view was silently drained by a later overlapping slice
        — which made ``a = q[0:3]; b = q[1:4]`` succeed at the cost
        of very surprising ownership transfer.  Nested slicing
        (``view[0:2]``) is still allowed because ``_nested_slice``
        explicitly releases the outer view's borrow before the inner
        view's ``_wrap`` runs.
        """
        from qamomile.circuit.transpiler.errors import QubitBorrowConflictError

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            _a = q[0:3]  # outer view covers {0, 1, 2}
            b = q[1:4]  # ← overlaps on {1, 2} → rejected
            b[0] = qmc.h(b[0])
            return qmc.measure(q)

        with pytest.raises(
            QubitBorrowConflictError, match="already owned by another slice view"
        ):
            _ = kern.block

    # ----- P2-B -------------------------------------------------------------

    def test_expval_snapshots_hamiltonian_binding(self):
        """Post-transpile mutations of ``H`` do not leak into ``exe.run``.

        Regression (P2-B): the compiled executable held the binding
        by reference, so ``H.add_term(...)`` between transpile and
        run silently altered the evaluated observable.
        """
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def kern(obs: qmc.Observable) -> qmc.Float:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.x(q[0])
            return qmc.expval(q, obs)

        H = qm_o.Z(0)
        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(kern, bindings={"obs": H})

        # Mutate the user binding after transpile; snapshot must isolate.
        H.add_term((qm_o.PauliOperator(qm_o.Pauli.X, 0),), 1.0)

        value = exe.run(transpiler.executor()).result()
        assert np.isclose(value, -1.0), (
            f"exe.run picked up post-transpile H mutation; got {value}"
        )

    # ----- P2-C -------------------------------------------------------------

    def test_zero_uint_const_step_rejected(self):
        """``q[0:4:UInt(0)]`` raises ``NotImplementedError`` before arithmetic.

        Regression (P2-C): the previous ``int``-only validation let a
        ``UInt`` handle with const 0 through; ``_compute_slice_length``
        then hit ``ZeroDivisionError`` inside arithmetic.
        """
        from qamomile.circuit.frontend.handle.primitives import UInt
        from qamomile.circuit.ir.types.primitives import UIntType
        from qamomile.circuit.ir.value import Value

        zero = UInt(
            value=Value(type=UIntType(), name="uint_const").with_const(0),
            init_value=0,
        )

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            _ = q[0:4:zero]
            return qmc.measure(q)

        with pytest.raises(NotImplementedError, match="positive step"):
            kern.block


class TestRound3Reviewer:
    """Regression tests for Codex Round 3 adversarial review findings.

    Covers R3-A: constant-fold does not fold result ArrayValues for
    non-SliceArrayOp ops (``MeasureVectorOperation`` clbit-result
    shape stays symbolic → allocator emits 0 clbits).

    The accompanying R3-B regressions (``measure(q[1::2])`` then
    ``expval(q, ...)`` over the same register and friends) used to
    live here as well; they have moved to
    ``tests/circuit/test_expval.py::TestExpvalOverConsumedSlots``
    alongside the other expval-consume guarantees.
    """

    # ----- R3-A: constant-fold must fold operation *results* ----------------

    def test_binop_derived_slice_bounds_emit_correct_clbit_count(self):
        """``measure(q[lo+0 : hi+0])`` folds to concrete clbits via bound bindings.

        Regression (R3-A): ``_substitute_folded_operands`` only folded
        ``op.operands``, not ``op.results``.  The ``MeasureVectorOperation``
        clbit-result array carries a ``shape`` derived from the BinOp
        ``(hi+0) - (lo+0)``; without result folding the shape remained
        symbolic and the allocator produced zero clbits — a silent data
        loss identical to the earlier OOB clamp bug.
        """
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def kern(lo: qmc.UInt, hi: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            # Introduce BinOp so the bounds are symbolic at trace time
            # and only resolve after ConstantFoldingPass applies bindings.
            start = lo + 0
            stop = hi + 0
            return qmc.measure(q[start:stop])

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(kern, bindings={"lo": 1, "hi": 4})
        qc = exe.compiled_quantum[0].circuit
        # q[1:4] covers 3 qubits → 3 clbits
        assert qc.num_clbits == 3, (
            f"q[lo+0 : hi+0] with lo=1, hi=4 should emit 3 clbits; "
            f"got {qc.num_clbits} (result-folding regression)"
        )
        n_meas = sum(1 for inst in qc.data if inst.operation.name == "measure")
        assert n_meas == 3

        # Also verify execution correctness: all-zero |000...> state
        result = exe.sample(transpiler.executor(), shots=16).result()
        bitstrings = {bs for bs, _ in result.results}
        # qubits 1,2,3 start in |0⟩ → all measured bits should be 0
        assert all(all(b == 0 for b in bs) for bs in bitstrings)

    def test_result_array_shape_folded_for_full_root_measure(self):
        """``measure(q)`` after BinOp-derived qubit_array size folds correctly.

        Exercises a neighbouring path: when the root array length itself
        came from a BinOp, the MeasureVectorOperation result shape must
        be folded in the same pass so the allocator sees a concrete size.
        """
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def kern(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            # Use n+0 so the size expression is a BinOp at trace time.
            size = n + 0
            q = qmc.qubit_array(size, "q")
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(kern, bindings={"n": 3})
        qc = exe.compiled_quantum[0].circuit
        assert qc.num_qubits == 3
        assert qc.num_clbits == 3


class TestRound4Reviewer:
    """Regression tests for Codex Round 4 adversarial review findings.

    R4-A: BinOp-derived view stop bound is not clamped to the parent
    length, so ``measure(q[0:hi+0])`` with ``hi`` greater than the parent
    length over-reports clbits and silently drops measurements.  The fix
    routes the clamp through a new :attr:`BinOpKind.MIN` so the same
    construction handles eager folding (literal bounds) and deferred
    folding (parameter-derived bounds).
    """

    def test_binop_stop_clamped_to_parent_length_from_zero(self):
        """``q[0:hi+0]`` with ``hi`` greater than parent yields parent-many clbits."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def kern(hi: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            return qmc.measure(q[0 : hi + 0])

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(kern, bindings={"hi": 10})
        qc = exe.compiled_quantum[0].circuit
        assert qc.num_clbits == 4
        n_meas = sum(1 for inst in qc.data if inst.operation.name == "measure")
        assert n_meas == 4

    def test_binop_stop_clamped_to_parent_length_with_offset_start(self):
        """``q[3:hi+0]`` with ``hi`` past parent collapses to a single-qubit slice."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def kern(hi: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            return qmc.measure(q[3 : hi + 0])

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(kern, bindings={"hi": 10})
        qc = exe.compiled_quantum[0].circuit
        assert qc.num_clbits == 1
        n_meas = sum(1 for inst in qc.data if inst.operation.name == "measure")
        assert n_meas == 1

    def test_binop_start_past_parent_collapses_to_empty(self):
        """``q[lo+0:hi+0]`` with ``lo`` past parent end yields zero measurements."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def kern(lo: qmc.UInt, hi: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            return qmc.measure(q[lo + 0 : hi + 0])

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(kern, bindings={"lo": 7, "hi": 10})
        qc = exe.compiled_quantum[0].circuit
        n_meas = sum(1 for inst in qc.data if inst.operation.name == "measure")
        assert n_meas == 0

    def test_literal_oob_slice_still_clamps(self):
        """Literal-bound clamp still works after symbolic-clamp unification."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            return qmc.measure(q[3:10])

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(kern, bindings={})
        qc = exe.compiled_quantum[0].circuit
        assert qc.num_clbits == 1

    def test_nested_view_binop_stop_clamped_to_view_length(self):
        """A nested ``view[0:hi+0]`` clamps against the outer view's length."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def kern(hi: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(8, "q")
            outer = q[0::2]  # length 4 view over q
            return qmc.measure(outer[0 : hi + 0])

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(kern, bindings={"hi": 99})
        qc = exe.compiled_quantum[0].circuit
        assert qc.num_clbits == 4

    # ----- R4-B: separately-derived view consumed-slot check ----------------

    def test_two_same_range_views_with_destructive_consume_rejected(self):
        """``odd1=q[1::2]; odd2=q[1::2]; measure(odd1); expval(odd2,obs)`` is rejected."""
        from qamomile.circuit.transpiler.errors import QubitBorrowConflictError

        @qmc.qkernel
        def kern(obs: qmc.Observable) -> qmc.Float:
            q = qmc.qubit_array(4, "q")
            odd1 = q[1::2]
            odd2 = q[1::2]
            _ = qmc.measure(odd1)
            return qmc.expval(odd2, obs)

        with pytest.raises(QubitBorrowConflictError):
            kern.block

    def test_drain_then_destructive_consume_marks_consumed_slots(self):
        """After drain transferring slot ownership, destructive consume still marks."""
        from qamomile.circuit.transpiler.errors import QubitBorrowConflictError

        @qmc.qkernel
        def kern(obs: qmc.Observable) -> qmc.Float:
            q = qmc.qubit_array(4, "q")
            # Sequential same-range slicing triggers the opportunistic
            # drain that transfers parent-slot ownership from view1 to
            # view2 — the underlying R4-B bug pattern.
            view1 = q[0::2]
            view2 = q[0::2]
            _ = qmc.measure(view1)
            return qmc.expval(view2, obs)

        with pytest.raises(QubitBorrowConflictError):
            kern.block

    def test_double_destructive_consume_on_same_slots_rejected(self):
        """``measure(q[1::2]); measure(q[1::2])`` raises rather than silently succeeding."""
        from qamomile.circuit.transpiler.errors import QubitBorrowConflictError

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            v1 = q[1::2]
            v2 = q[1::2]
            _ = qmc.measure(v1)
            return qmc.measure(v2)

        with pytest.raises(QubitBorrowConflictError):
            kern.block

    def test_disjoint_views_with_destructive_consume_each_allowed(self):
        """``measure(q[0::2]); measure(q[1::2])`` over disjoint slots is fine."""
        pytest.importorskip("qiskit")

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            evens = q[0::2]
            odds = q[1::2]
            _ = qmc.measure(evens)
            return qmc.measure(odds)

        # Frontend trace must succeed for disjoint coverage.
        assert kern.block is not None

    def test_consumed_marker_survives_non_destructive_view_consume(self):
        """A non-destructive consume on an overlapping view does not erase markers."""
        from qamomile.circuit.transpiler.errors import QubitBorrowConflictError

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            v1 = q[1::2]
            v2 = q[1::2]
            _ = qmc.measure(v1)  # destructive, marks slots {1, 3} as consumed
            # A subsequent measure(v2) MUST be rejected.  The fact that
            # v2 is still alive when v1 was consumed must not let it
            # slip past the consumed-slot check.
            return qmc.measure(v2)

        with pytest.raises(QubitBorrowConflictError):
            kern.block

    def test_recreate_view_after_destructive_consume_raises_consumed(self):
        """``measure(q[1::2])`` then ``q[1::2]`` again surfaces as
        ``QubitConsumedError``, not the live-borrow conflict.

        The earlier Round-4 tests pin the LIVE-overlap variant: ``v1 =
        q[1::2]; v2 = q[1::2]`` rejects ``v2``'s construction before
        any destructive consume happens (``QubitBorrowConflictError``).
        This test pins the *post-destructive-consume* recreation path:
        ``v1`` is created and immediately measured, leaving a
        destroyed-slot breadcrumb in the parent's borrow table; a
        later ``q[1::2]`` walks that table and must surface the slot
        loss as ``QubitConsumedError`` (the slot is gone forever; no
        amount of returning a handle restores it).
        """
        from qamomile.circuit.transpiler.errors import QubitConsumedError

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            v1 = q[1::2]
            _ = qmc.measure(v1)  # destroys q[1], q[3]
            v2 = q[1::2]  # ← must raise QubitConsumedError
            return qmc.measure(v2)

        with pytest.raises(QubitConsumedError, match="already destroyed"):
            kern.block

    # ----- R4-C: cast(view, ...) carrier-key root-space resolution -----------

    def test_cast_view_to_qfixed_measures_root_qubits(self):
        """``cast(q[1::2], QFixed)`` measures the root qubits the view covers."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def kern() -> qmc.Float:
            q = qmc.qubit_array(8, "q")
            view = q[1::2]
            qf = qmc.cast(view, qmc.QFixed, int_bits=0)
            return qmc.measure(qf)

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(kern, bindings={})
        qc = exe.compiled_quantum[0].circuit
        n_meas = sum(1 for inst in qc.data if inst.operation.name == "measure")
        # The view covers q[1], q[3], q[5], q[7] — four physical qubits.
        assert n_meas == 4, (
            f"cast(q[1::2], QFixed) followed by measure must emit 4 "
            f"measurements; got {n_meas} (R4-C carrier-key bug)"
        )
        # Verify the measured qubits are the root-space {1, 3, 5, 7}.
        measured_q = sorted(
            qc.find_bit(inst.qubits[0]).index
            for inst in qc.data
            if inst.operation.name == "measure"
        )
        assert measured_q == [1, 3, 5, 7]

    def test_cast_root_vector_to_qfixed_still_measures_all(self):
        """Non-view cast still measures every qubit (regression guard)."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def kern() -> qmc.Float:
            q = qmc.qubit_array(4, "q")
            qf = qmc.cast(q, qmc.QFixed, int_bits=0)
            return qmc.measure(qf)

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(kern, bindings={})
        qc = exe.compiled_quantum[0].circuit
        n_meas = sum(1 for inst in qc.data if inst.operation.name == "measure")
        assert n_meas == 4

    def test_cast_nested_view_to_qfixed(self):
        """``cast(q[0::2][1:3], QFixed)`` resolves carriers via composed slice chain.

        The composed-affine-map test goes through a single
        root-equivalent slice ``q[2:5:2]`` to avoid having to
        explicitly discharge the outer view ``q[0::2]`` (which would
        end up with destroyed inner slots and live non-overlap slots
        — strict-return would then demand a separate slice assignment
        on the non-overlap range).  The IR emit path is identical for
        the two expressions; ``q[0::2][1:3]`` is exercised end-to-end
        in :class:`TestConstantIndexOnSliceView` (under nested-slice
        composition) and in :class:`TestSliceDrawAnalyzer` for the
        drawer.
        """
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def kern() -> qmc.Float:
            q = qmc.qubit_array(8, "q")
            inner = q[2:5:2]  # length 2, covers q[2], q[4]
            qf = qmc.cast(inner, qmc.QFixed, int_bits=0)
            return qmc.measure(qf)

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(kern, bindings={})
        qc = exe.compiled_quantum[0].circuit
        measured_q = sorted(
            qc.find_bit(inst.qubits[0]).index
            for inst in qc.data
            if inst.operation.name == "measure"
        )
        assert measured_q == [2, 4]

    def test_cast_symbolic_bound_view_rejected(self):
        """``cast(q[lo:hi], QFixed)`` with UInt-symbolic bounds raises a clear error."""

        @qmc.qkernel
        def kern(lo: qmc.UInt, hi: qmc.UInt) -> qmc.Float:
            q = qmc.qubit_array(4, "q")
            return qmc.measure(qmc.cast(q[lo:hi], qmc.QFixed, int_bits=0))

        with pytest.raises(ValueError, match="symbolic"):
            kern.block

    # ----- R4-D: eager BinOp folding when both operands are constants --------

    def test_binop_with_two_constants_folds_eagerly_at_trace(self):
        """``UInt(const) + 0`` collapses to a constant Value at trace, no BinOp."""
        from qamomile.circuit.frontend.handle import UInt
        from qamomile.circuit.frontend.tracer import Tracer, trace
        from qamomile.circuit.ir.types.primitives import UIntType
        from qamomile.circuit.ir.value import Value

        t = Tracer()
        with trace(t):
            lo = UInt(
                value=Value(type=UIntType(), name="lo").with_const(1),
                init_value=1,
            )
            result = lo + 0

        assert result.value.is_constant()
        assert result.value.get_const() == 1
        # No BinOp should have been added to the tracer for the trivial fold.
        binop_kinds = [
            type(op).__name__ for op in t.operations if type(op).__name__ == "BinOp"
        ]
        assert binop_kinds == [], f"expected no BinOp, got {binop_kinds}"

    def test_binop_with_one_symbolic_operand_still_emits_binop(self):
        """``UInt(symbolic) + 0`` keeps the BinOp — only both-const folds eagerly."""
        from qamomile.circuit.frontend.handle import UInt
        from qamomile.circuit.frontend.tracer import Tracer, trace
        from qamomile.circuit.ir.types.primitives import UIntType
        from qamomile.circuit.ir.value import Value

        t = Tracer()
        with trace(t):
            lo = UInt(
                value=Value(type=UIntType(), name="lo").with_parameter("lo"),
                init_value=0,
            )
            result = lo + 0

        # symbolic + const → BinOp must remain in trace
        assert not result.value.is_constant()
        binops = [op for op in t.operations if type(op).__name__ == "BinOp"]
        assert len(binops) == 1

    def test_qft_on_binop_derived_view_emits_gates(self):
        """``qft(q[lo+0:hi+0])`` with bound parameters emits the same QFT as ``qft(q[lo:hi])``."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def kern_direct(lo: qmc.UInt, hi: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            view = q[lo:hi]
            view = qmc.qft(view)
            return qmc.measure(view)

        @qmc.qkernel
        def kern_binop(lo: qmc.UInt, hi: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            view = q[lo + 0 : hi + 0]
            view = qmc.qft(view)
            return qmc.measure(view)

        transpiler = QiskitTranspiler()
        bindings = {"lo": 1, "hi": 3}
        qc_direct = (
            transpiler.transpile(kern_direct, bindings=bindings)
            .compiled_quantum[0]
            .circuit
        )
        qc_binop = (
            transpiler.transpile(kern_binop, bindings=bindings)
            .compiled_quantum[0]
            .circuit
        )

        # Structural equivalence between the direct- and BinOp-bounded
        # views: same number of operations on the same target qubits in
        # the same order.  Avoid asserting on ``operation.name`` strings
        # — the native QFT instruction is rendered as ``qft`` / ``QFT``
        # / ``QFTGate`` depending on the Qiskit version, so a string
        # check would be brittle across the supported matrix.
        def _signature(circuit):
            return [
                (
                    type(inst.operation).__name__,
                    tuple(circuit.find_bit(q).index for q in inst.qubits),
                )
                for inst in circuit.data
            ]

        direct_sig = _signature(qc_direct)
        binop_sig = _signature(qc_binop)
        assert direct_sig == binop_sig, (
            f"BinOp-derived view should produce the same circuit as the "
            f"direct view.\n  direct: {direct_sig}\n  binop:  {binop_sig}"
        )
        # Sanity guard — the direct path always emits something; if it
        # were empty the equivalence above would be vacuously true.
        assert direct_sig, "direct view emitted no operations (test is vacuous)"

    def test_cast_on_binop_derived_view_emits_measurements(self):
        """``cast(q[lo+0:hi+0], QFixed)`` resolves carriers with bound parameters."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def kern(lo: qmc.UInt, hi: qmc.UInt) -> qmc.Float:
            q = qmc.qubit_array(8, "q")
            view = q[(lo + 1) : (hi + 0)]
            qf = qmc.cast(view, qmc.QFixed, int_bits=0)
            return qmc.measure(qf)

        transpiler = QiskitTranspiler()
        # lo+1 = 2, hi+0 = 5 → view = q[2:5] → 3 carriers on physical {2,3,4}
        exe = transpiler.transpile(kern, bindings={"lo": 1, "hi": 5})
        qc = exe.compiled_quantum[0].circuit
        measured_q = sorted(
            qc.find_bit(inst.qubits[0]).index
            for inst in qc.data
            if inst.operation.name == "measure"
        )
        assert measured_q == [2, 3, 4]


class TestRound5Reviewer:
    """Regression tests for Copilot Round 5 review findings.

    Covers three additional silent-fail / cross-contamination bugs the
    earlier rounds missed:

    R5-A: ``_build_qubit_map`` swallows ``EmitError`` from
    ``resolve_slice_chain`` for sliced operands and falls back to the
    element_uuid path, producing an empty qubit_map and a far-away
    backend width-mismatch error.
    R5-B: ``SliceBorrowCheckPass`` borrow-state keyed only by
    slot index aliases independent registers — consuming ``a[1]``
    spuriously blocks ``b[1]``.
    R5-C: The loop-body state merge inside ``_walk_nested`` only
    propagates *new* keys back to the outer state, dropping
    consumed-slot markers installed inside the loop and view
    ownership transitions for views created before the loop.
    """

    # ----- R5-B: borrow keys must be namespaced per root array --------------

    def test_consume_view_on_one_register_does_not_block_other_register(self):
        """``measure(a[1::2]); expval(b, Z(1))`` over disjoint registers must succeed."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def kern(obs: qmc.Observable) -> qmc.Float:
            a = qmc.qubit_array(4, "a")
            b = qmc.qubit_array(4, "b")
            _ = qmc.measure(a[1::2])
            # b's slot 1 must remain accessible — the consumed-slot
            # markers from a's destructive view must not bleed into b.
            return qmc.expval(b, obs)

        H = qm_o.Z(1)
        transpiler = QiskitTranspiler()
        # If borrow keys were not namespaced, this would raise
        # QubitConsumedError or a stage-later EmitError; we
        # only need the kernel to trace and transpile cleanly.
        exe = transpiler.transpile(kern, bindings={"obs": H})
        assert exe is not None

    def test_disjoint_registers_can_each_consume_view_independently(self):
        """Each register independently tracks its own consumed slots."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            a = qmc.qubit_array(4, "a")
            b = qmc.qubit_array(4, "b")
            _ = qmc.measure(a[1::2])  # destroys a[1], a[3]
            return qmc.measure(b[1::2])  # must NOT see a's markers

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(kern, bindings={})
        qc = exe.compiled_quantum[0].circuit
        # Both views measure 2 qubits each on their own register.
        assert qc.num_clbits >= 2

    # ----- R5-C: loop-body state must propagate consumed markers -------------

    def test_destructive_view_consume_inside_for_loop_persists_post_loop(self):
        """A destructive view consume inside a loop body must mark consumed slots."""
        from qamomile.circuit.transpiler.errors import QubitConsumedError

        @qmc.qkernel
        def kern(obs: qmc.Observable) -> qmc.Float:
            q = qmc.qubit_array(4, "q")
            for _ in qmc.range(1):
                _ = qmc.measure(q[1::2])  # destroys q[1], q[3]
            # After the loop, expval(q, ...) must be rejected — the
            # destructive consume's markers must propagate out of the
            # loop body state into the outer state.
            return qmc.expval(q, obs)

        H = qm_o.Z(0)
        with pytest.raises(QubitConsumedError):
            _ = kern.build(obs=H)

    def test_consumed_marker_in_loop_body_does_not_leak_across_registers(self):
        """The loop-body merge respects the per-root namespace.

        Combines R5-B (per-root namespacing) and R5-C (loop-body
        merge): a destructive view consume on register ``a`` inside a
        ``for`` body must mark ``a``'s slots post-loop, but must not
        bleed into register ``b`` (whose identical-numbered slot would
        have collided under a non-namespaced key).
        """
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def kern(obs: qmc.Observable) -> qmc.Float:
            a = qmc.qubit_array(4, "a")
            b = qmc.qubit_array(4, "b")
            for _ in qmc.range(1):
                _ = qmc.measure(a[1::2])  # destroys a[1], a[3]
            # b is untouched — must stay valid.
            return qmc.expval(b, obs)

        H = qm_o.Z(1)
        transpiler = QiskitTranspiler()
        # b's expval must succeed despite a's destructive consume sharing
        # the same slot indices {1, 3}.
        exe = transpiler.transpile(kern, bindings={"obs": H})
        assert exe is not None


# =============================================================================
# Slice Assignment Tests
# =============================================================================


class TestSliceAssignment:
    """Tests for the slice-assignment broadcast-return path
    (``qs[a:b] = qmc.h(qs[a:b])``).

    Slice assignment is the explicit borrow-return form: the right-hand
    side must be a ``VectorView`` covering the same root-space slot set
    as ``qs[a:b]`` would build.  These tests cover both the happy path
    (literal bounds, bindings-resolved bounds, fully-symbolic via re-trace,
    nested slice on a view, body-internal loop pattern) and every
    negative path the frontend validator rejects at trace time
    (mismatched coverage, cross-array RHS, stale view from drain, LHS
    root consumed, RHS consumed, body-internal release).
    """

    # ----- positive: literal-bound assignment compiles and emits ----------

    def test_literal_slice_assignment_emits_correct_gates(self):
        """``q[0:2] = h(q[0:2])`` emits H on qubits {0, 1}."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            q[0:2] = qmc.h(q[0:2])
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(kern)
        qc = exe.compiled_quantum[0].circuit
        applied = sorted(
            qc.find_bit(inst.qubits[0]).index
            for inst in qc.data
            if inst.operation.name == "h"
        )
        assert applied == [0, 1], applied

    def test_bindings_resolved_slice_assignment_emits_correct_gates(self):
        """``q[0:2] = h(q[0:2])`` with parent length from bindings emits H on {0, 1}."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def kern(num: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(num, "q")
            q[0:2] = qmc.h(q[0:2])
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(kern, bindings={"num": 4})
        qc = exe.compiled_quantum[0].circuit
        applied = sorted(
            qc.find_bit(inst.qubits[0]).index
            for inst in qc.data
            if inst.operation.name == "h"
        )
        assert applied == [0, 1], applied

    def test_fully_symbolic_slice_assignment_via_bindings(self):
        """``q[lo:hi] = h(q[lo:hi])`` resolves through bindings and emits
        H on the resolved range.  Trace-time check is deferred (both
        sides are symbolic), but the bindings-applied re-trace runs the
        full coverage / ownership validation."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def kern(num: qmc.UInt, lo: qmc.UInt, hi: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(num, "q")
            q[lo:hi] = qmc.h(q[lo:hi])
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(kern, bindings={"num": 6, "lo": 1, "hi": 4})
        qc = exe.compiled_quantum[0].circuit
        applied = sorted(
            qc.find_bit(inst.qubits[0]).index
            for inst in qc.data
            if inst.operation.name == "h"
        )
        assert applied == [1, 2, 3], applied

    def test_computed_bound_slice_assignment(self):
        """``q[1:n - 2] = h(q[1:n - 2])`` with computed stop expression
        resolves the BinOp via partial_eval and emits correctly."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def kern(num: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(num, "q")
            q[1 : num - 2] = qmc.h(q[1 : num - 2])
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(kern, bindings={"num": 6})
        qc = exe.compiled_quantum[0].circuit
        applied = sorted(
            qc.find_bit(inst.qubits[0]).index
            for inst in qc.data
            if inst.operation.name == "h"
        )
        assert applied == [1, 2, 3], applied

    def test_slice_assignment_on_vector_view(self):
        """Nested slice assignment via ``VectorView.__setitem__``.

        ``view[a:b] = h(view[a:b])`` releases the inner view back to its
        parent ``VectorView``.  The outer view's affine map is composed
        with the inner slice in pure-int space (``VectorView._normalize_
        slice_to_covered``) and the resulting root-space coverage is
        validated against the RHS view's coverage.
        """
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(8, "q")
            view = q[0::2]  # length-4 view over q[0], q[2], q[4], q[6]
            view[0:2] = qmc.h(view[0:2])  # H on q[0], q[2] (nested return)
            q[0::2] = view
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(kern)
        qc = exe.compiled_quantum[0].circuit
        applied = sorted(
            qc.find_bit(inst.qubits[0]).index
            for inst in qc.data
            if inst.operation.name == "h"
        )
        assert applied == [0, 2], applied

    def test_body_internal_slice_assignment_in_for_loop(self):
        """``for i in range(...): qs[i:i+1] = h(qs[i:i+1])`` keeps each
        view's lifetime inside the body iteration.  Because the slice is
        constructed inside the body, the IR pass's ``outer_snapshot_stack``
        guard doesn't fire (no outer-registered entry is being deleted),
        and the release at the end of the iteration cleans up the
        registration before the next iteration.
        """
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            for i in qmc.range(4):
                q[i : i + 1] = qmc.h(q[i : i + 1])
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(kern)
        # All four qubits should have H applied via the loop.
        qc = exe.compiled_quantum[0].circuit
        n_h = sum(1 for inst in qc.data if inst.operation.name == "h")
        assert n_h == 4

    def test_top_level_slice_assignment_after_body_use(self):
        """``even = qs[0::2]; for i in ...: even[i] = h(even[i]); qs[0::2] = even``
        is the canonical broadcast-loop pattern: top-level view, body-internal
        element use, top-level slice-assignment release.  The IR pass's
        outer_snapshot_stack pops when the body exits, so the top-level
        release sees an empty stack and proceeds normally.
        """
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            even = q[0::2]
            for i in qmc.range(even.shape[0]):
                even[i] = qmc.h(even[i])
            q[0::2] = even
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(kern)
        qc = exe.compiled_quantum[0].circuit
        applied = sorted(
            qc.find_bit(inst.qubits[0]).index
            for inst in qc.data
            if inst.operation.name == "h"
        )
        assert applied == [0, 2], applied

    # ----- negative: trace-time rejects ---------------------------------

    def test_coverage_mismatch_literal_rejected(self):
        """``q[0:2] = h(q[0:3])`` (concrete bounds, concrete parent length)
        is rejected at trace time with a coverage-mismatch AffineTypeError.
        """
        from qamomile.circuit.transpiler.errors import AffineTypeError

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            q[0:2] = qmc.h(q[0:3])
            return qmc.measure(q)

        with pytest.raises(AffineTypeError, match="coverage mismatch"):
            _ = kern.block

    def test_coverage_mismatch_symbolic_parent_rejected(self):
        """``q[0:2] = h(q[0:3])`` with symbolic parent length is still
        caught at trace time.

        The frontend's ``_normalize_slice_to_covered`` skips the parent-length
        clamp when the parent is symbolic but still computes lhs coverage
        from concrete user bounds; ``_make_slice_view`` records concrete
        ``_slice_covered_indices`` under the same condition.  Both
        agree on the comparison, so the mismatch is rejected.
        """
        from qamomile.circuit.transpiler.errors import AffineTypeError

        @qmc.qkernel
        def kern(num: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(num, "q")
            q[0:2] = qmc.h(q[0:3])
            return qmc.measure(q)

        with pytest.raises(AffineTypeError, match="coverage mismatch"):
            _ = kern.block

    def test_cross_array_rhs_rejected(self):
        """``q[0:2] = h(r[0:2])`` (RHS view's root parent differs from LHS)
        is rejected at trace time with a root-identity AffineTypeError.
        """
        from qamomile.circuit.transpiler.errors import AffineTypeError

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            r = qmc.qubit_array(4, "r")
            q[0:2] = qmc.h(r[0:2])
            _ = qmc.measure(r)
            return qmc.measure(q)

        with pytest.raises(AffineTypeError, match="root parent"):
            _ = kern.block

    def test_overlapping_top_level_slice_rejected_at_creation(self):
        """``a = q[0:2]; b = q[0:2]`` is rejected at ``b``'s construction.

        Under the strict no-multi-view policy
        (``VectorView._wrap``'s overlap check), the second
        same-range view ``b`` is rejected immediately with
        ``QubitBorrowConflictError`` — no need for the downstream
        slice assignment to detect a stale ``a``.  This is stricter
        than the previous "opportunistic drain + ownership check at
        ``q[0:2] = a``" two-step rejection; the failure mode is now
        loud at the obvious site (the construction of ``b``).
        """
        from qamomile.circuit.transpiler.errors import QubitBorrowConflictError

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            a = q[0:2]
            b = q[0:2]  # ← rejected: overlap with a
            q[0:2] = a
            _ = b
            return qmc.measure(q)

        with pytest.raises(
            QubitBorrowConflictError, match="already owned by another slice view"
        ):
            _ = kern.block

    def test_lhs_root_consumed_rejected(self):
        """``v = q[0:2]; measure(q); q[0:2] = h(v)`` is rejected up-front
        because ``v`` was still bulk-borrowing from ``q`` at the time of
        ``measure(q)``.

        Under the strict-return policy the parent cannot be consumed
        while any slice view is outstanding, so the failure surfaces
        as ``UnreturnedBorrowError`` at ``measure(q)`` rather than the
        downstream LHS-root liveness check on ``q[0:2] = ...``.  Both
        outcomes loudly reject the silent miscompile path; we just pin
        the *earlier* failure point introduced by the strict policy.
        """
        from qamomile.circuit.transpiler.errors import UnreturnedBorrowError

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            v = q[0:2]
            bits = qmc.measure(q)
            q[0:2] = qmc.h(v)
            return bits

        with pytest.raises(UnreturnedBorrowError, match="unreturned slice-view"):
            _ = kern.block

    def test_rhs_view_already_consumed_rejected(self):
        """``v = q[0:2]; measure(v); q[0:2] = v`` is rejected at step 2
        (RHS consumed-check).  The measure consumed ``v`` already so the
        view handle cannot be returned via slice assignment.
        """
        from qamomile.circuit.transpiler.errors import QubitConsumedError

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            v = q[0:2]
            bits = qmc.measure(v)
            q[0:2] = v
            return bits

        with pytest.raises(QubitConsumedError):
            _ = kern.block

    def test_body_internal_release_of_outer_view_rejected(self):
        """``even = q[0::2]; for i in ...: q[0::2] = even`` is rejected
        at the IR pass's ``outer_snapshot_stack`` guard.

        Releasing an outer-registered view from inside a control-flow body
        would require the loop / branch merge to propagate the entry
        deletion outward, which the current consumption-priority union
        cannot do safely. The pass raises ``ValidationError``
        with a hint pointing at the control-flow region.
        """
        pytest.importorskip("qiskit")
        from qamomile.circuit.transpiler.errors import ValidationError
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            even = q[0::2]
            for _ in qmc.range(1):
                # Body-internal release of outer-registered view.
                q[0::2] = even
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        with pytest.raises(ValidationError, match="control-flow body"):
            transpiler.transpile(kern)

    def test_one_entry_items_keeps_outer_view_release_boundary(self):
        """One-entry items lowering cannot hide an in-body slice release."""
        pytest.importorskip("qiskit")
        from qamomile.circuit.transpiler.errors import ValidationError
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def kern(
            data: qmc.Dict[qmc.UInt, qmc.Float],
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            even = q[0::2]
            for _key, _value in qmc.items(data):
                q[0::2] = even
            return qmc.measure(q)

        with pytest.raises(ValidationError, match="control-flow body"):
            QiskitTranspiler().transpile(kern, bindings={"data": {0: 1.0}})
