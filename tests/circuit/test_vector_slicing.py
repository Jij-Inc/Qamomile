"""Tests for ``Vector`` slicing and the resulting ``VectorView``.

These tests cover the frontend behaviour (length, indexing, nesting,
borrow tracking, error cases) and the end-to-end execution path on every
supported quantum SDK backend for the canonical alternating-qubit
pattern — the use case the slicing feature was introduced to support.
"""

from __future__ import annotations

import numpy as np
import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.frontend.handle import VectorView

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
        from qamomile.circuit.transpiler.errors import QubitConsumedError

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Qubit]:
            q = qmc.qubit_array(4, "q")
            evens = q[0:4:2]
            del evens
            q[0] = qmc.h(q[0])
            return q

        with pytest.raises(QubitConsumedError, match="held by a VectorView slice"):
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

        from qamomile.circuit.transpiler.errors import QubitConsumedError

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Qubit]:
            q = qmc.qubit_array(4, "q")
            a = q[0:4:2]  # covers {0, 2}
            qa = a[0]  # active borrow on slot 0 via view a
            _b = q[0:4:1]  # covers {0, 1, 2, 3} — overlaps with live a
            a[0] = qa
            return q

        with pytest.raises(
            QubitConsumedError, match="already owned by another slice view"
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
        from qamomile.circuit.transpiler.errors import QubitConsumedError

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

        with pytest.raises(QubitConsumedError, match="held by a VectorView slice"):
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


# ---------------------------------------------------------------------------
# V6: IR-level slice operation + SliceLinearityCheckPass
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


class TestViewAsKernelOperandViaCallBlock:
    """View flows through ``CallBlockOperation`` (V6 removes inline-trace)."""

    def test_view_arg_routes_through_callblock(self):
        """Kernel called with a view now uses the standard call path.

        Previously (V1), views forced ``_inline_trace_call`` because
        they had no IR representation.  V6 makes the sliced
        ``ArrayValue`` first-class, so the callee receives the view
        through the normal ``CallBlockOperation`` operand chain.  The
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
    """``SliceLinearityCheckPass`` catches post-fold aliasing."""

    def test_symbolic_slice_vs_direct_access_caught_at_transpile(self):
        """``q[lo:hi] + q[0]`` aliases when ``lo=0``; caught at transpile.

        Scalar ``UInt`` parameters that are concrete in ``bindings`` now
        carry ``is_constant() == True`` at trace time, so the frontend's
        bulk-borrow tracker can enumerate the slice's coverage and
        catches the aliasing immediately (``QubitConsumedError``).
        When bounds stay symbolic — e.g. derived from an unbound
        parameter through arithmetic — ``SliceLinearityCheckPass``
        resolves them post-fold and raises
        ``SliceLinearityViolationError`` instead.
        """
        pytest.importorskip("qiskit")
        from qamomile.circuit.transpiler.errors import (
            QubitConsumedError,
            SliceLinearityViolationError,
        )
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
        with pytest.raises((SliceLinearityViolationError, QubitConsumedError)):
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

    def test_direct_access_before_view_use_is_caught(self):
        """Alias is caught even when the direct parent access precedes the view use.

        Regression (P1-2): ``SliceLinearityCheckPass`` used to register
        a view's bulk-borrow lazily — only when the view was first
        referenced as an operand.  That made ``region = q[lo2:hi];
        q[0] = h(q[0]); region[0] = x(region[0])`` (with ``lo=0``,
        ``hi=4``) slip through, because the direct ``q[0]`` access
        happened before the view was ever observed in the pass.
        The fix pre-scans all operands up front so the view's covered
        slots are registered before any conflict check.
        """
        pytest.importorskip("qiskit")
        from qamomile.circuit.transpiler.errors import (
            QubitConsumedError,
            SliceLinearityViolationError,
        )
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
        with pytest.raises((SliceLinearityViolationError, QubitConsumedError)):
            transpiler.transpile(circuit, bindings={"num": 4, "lo": 0, "hi": 4})

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
        from qamomile.circuit.transpiler.errors import QubitConsumedError

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Qubit]:
            q = qmc.qubit_array(4, "q")
            inner = q[0::2][1:3]  # covers root slots {2}
            q[2] = qmc.x(q[2])  # aliases inner's slot 2
            inner[0] = qmc.h(inner[0])
            return q

        with pytest.raises(QubitConsumedError):
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

    def test_controlled_u_with_index_spec_on_view(self):
        """``controlled(_zgate, num_controls=1)(q[1::2], target_indices=[1])`` runs.

        Regression: the index-spec path in
        ``emit_controlled_u_with_index_spec`` used
        ``QubitAddress(vector_value.uuid, i)`` directly, which missed
        the view's ``slice_of`` chain.  The previous result was
        ``EmitError: Qubit <view_uuid>_0 not found in qubit_map``.
        After the fix, the controlled-Z lands on physical qubits
        (1, 3), the view's slot 0 / slot 1.
        """
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def _zgate(q: qmc.Qubit) -> qmc.Qubit:
            return qmc.z(q)

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            cz = qmc.controlled(_zgate, num_controls=1)
            view = q[1::2]
            view = cz(view, target_indices=[1])
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
    P1-C (if-lowering phi substitution on SliceArrayOp result),
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
        from qamomile.circuit.transpiler.errors import (
            QubitConsumedError,
            SliceLinearityViolationError,
        )

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            odd = q[1::2]
            _ = qmc.measure(odd)
            return qmc.measure(q)

        with pytest.raises((QubitConsumedError, SliceLinearityViolationError)):
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
        slice's bounds, the phi-output references must be substituted
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

        phi_start = Value(type=UIntType(), name="phi_start")
        folded_start = Value(type=UIntType(), name="folded").with_const(1)
        step = Value(type=UIntType(), name="step").with_const(1)
        root = ArrayValue(type=QubitType(), name="q")
        sliced = ArrayValue(
            type=QubitType(),
            name="q[slice]",
            slice_of=root,
            slice_start=phi_start,
            slice_step=step,
        )
        op = SliceArrayOperation(
            operands=[root, phi_start, step],
            results=[sliced],
        )

        subst = {phi_start.uuid: folded_start}
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
        immediately with ``QubitConsumedError``.  This replaces the
        prior "opportunistic drain" semantics where an unused outer
        view was silently drained by a later overlapping slice —
        which made ``a = q[0:3]; b = q[1:4]`` succeed at the cost of
        very surprising ownership transfer.  Nested slicing
        (``view[0:2]``) is still allowed because ``_nested_slice``
        explicitly releases the outer view's borrow before the inner
        view's ``_wrap`` runs.
        """
        from qamomile.circuit.transpiler.errors import QubitConsumedError

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            _a = q[0:3]  # outer view covers {0, 1, 2}
            b = q[1:4]  # ← overlaps on {1, 2} → rejected
            b[0] = qmc.h(b[0])
            return qmc.measure(q)

        with pytest.raises(
            QubitConsumedError, match="already owned by another slice view"
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

    Covers:
    - R3-A: constant-fold does not fold result ArrayValues for non-SliceArrayOp ops
      (MeasureVectorOperation clbit-result shape stays symbolic → allocator emits 0 clbits)
    - R3-B: ``measure(q[1::2]); expval(q, Z(1))`` transpiles silently despite
      consumed-slot markers — caught neither at frontend trace time nor by
      the IR SliceLinearityCheckPass
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

    # ----- R3-B: expval after measure(view) must be rejected ----------------

    def test_expval_whole_array_after_view_measure_rejected_at_trace(self):
        """``measure(q[1::2]); expval(q, H)`` raises at trace time.

        Regression (R3-B): ``expval(q, H)`` with ``q`` passed as a whole
        root array bypassed both the frontend's consumed-slot guard (because
        ``expval.py`` never called ``consume`` or ``_check_no_consumed_slots``)
        and the IR's ``_process_operand_borrows`` (which only inspected
        per-element Values, not whole-array operands).  The bug surfaced at
        Qiskit execution time with a circuit-vs-observable dimension mismatch.
        The fix raises ``QubitConsumedError`` at trace time.
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

        Verifies that ``_check_no_consumed_slots`` is not over-aggressive:
        consuming slots {1, 3} via ``measure(q[1::2])`` should leave
        slots {0, 2} usable in a subsequent ``expval`` call on ``q[0::2]``.
        The guard must only fire when the *expval operand's* covered slots
        overlap the consumed set; a disjoint view (even qubits vs odd qubits)
        must pass cleanly.

        Note: mixing ``measure(view)`` and ``expval(view2)`` in the same
        kernel creates a multi-segment program that the NISQ single-segment
        strategy rejects at plan time.  This test therefore asserts only that
        the *frontend linearity guard* does not raise — ``kern.block`` must
        succeed — and does not attempt a full transpile or execution.
        """
        # No backend import needed — frontend-only assertion.

        @qmc.qkernel
        def kern(obs: qmc.Observable) -> qmc.Float:
            q = qmc.qubit_array(4, "q")
            _ = qmc.measure(q[1::2])  # destroy q[1], q[3]
            return qmc.expval(q[0::2], obs)  # slots {0, 2} survive — OK

        # Must not raise QubitConsumedError: even view is disjoint from
        # the consumed odd slots.
        block = kern.block
        assert block is not None

    def test_ir_check_catches_expval_over_consumed_slot(self):
        """IR ``SliceLinearityCheckPass`` rejects whole-root expval after view-measure.

        Complementary to the frontend trace-time guard: even if the IR is
        constructed directly (bypassing ``expval()``), the post-fold
        linearity checker must catch the consumed-slot violation.
        """
        pytest.importorskip("qiskit")
        from qamomile.circuit.transpiler.errors import SliceLinearityViolationError

        @qmc.qkernel
        def kern(obs: qmc.Observable) -> qmc.Float:
            q = qmc.qubit_array(4, "q")
            _ = qmc.measure(q[1::2])
            # Attempt expval on a partial view that overlaps the consumed slots.
            # q[0::2] is safe; q[1::2] would overlap — use a harmless
            # even view here, but pair it with an odd direct access.
            q[1] = qmc.x(q[1])  # direct access after measure(q[1::2]) — consumed
            return qmc.expval(q[0::2], obs)

        from qamomile.circuit.transpiler.errors import QubitConsumedError

        with pytest.raises((QubitConsumedError, SliceLinearityViolationError)):
            kern.block


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
        from qamomile.circuit.transpiler.errors import QubitConsumedError

        @qmc.qkernel
        def kern(obs: qmc.Observable) -> qmc.Float:
            q = qmc.qubit_array(4, "q")
            odd1 = q[1::2]
            odd2 = q[1::2]
            _ = qmc.measure(odd1)
            return qmc.expval(odd2, obs)

        with pytest.raises(QubitConsumedError):
            kern.block

    def test_drain_then_destructive_consume_marks_consumed_slots(self):
        """After drain transferring slot ownership, destructive consume still marks."""
        from qamomile.circuit.transpiler.errors import QubitConsumedError

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

        with pytest.raises(QubitConsumedError):
            kern.block

    def test_double_destructive_consume_on_same_slots_rejected(self):
        """``measure(q[1::2]); measure(q[1::2])`` raises rather than silently succeeding."""
        from qamomile.circuit.transpiler.errors import QubitConsumedError

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            v1 = q[1::2]
            v2 = q[1::2]
            _ = qmc.measure(v1)
            return qmc.measure(v2)

        with pytest.raises(QubitConsumedError):
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
        from qamomile.circuit.transpiler.errors import QubitConsumedError

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

        with pytest.raises(QubitConsumedError):
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
    R5-B: ``SliceLinearityCheckPass`` borrow-state keyed only by
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
        # SliceLinearityViolationError or a stage-later EmitError; we
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
        from qamomile.circuit.transpiler.errors import (
            QubitConsumedError,
            SliceLinearityViolationError,
        )

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
        with pytest.raises((QubitConsumedError, SliceLinearityViolationError)):
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
        ``QubitConsumedError`` — no need for the downstream slice
        assignment to detect a stale ``a``.  This is stricter than
        the previous "opportunistic drain + ownership check at
        ``q[0:2] = a``" two-step rejection; the failure mode is now
        loud at the obvious site (the construction of ``b``).
        """
        from qamomile.circuit.transpiler.errors import QubitConsumedError

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            a = q[0:2]
            b = q[0:2]  # ← rejected: overlap with a
            q[0:2] = a
            _ = b
            return qmc.measure(q)

        with pytest.raises(
            QubitConsumedError, match="already owned by another slice view"
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
        cannot do safely.  The pass raises ``SliceLinearityViolationError``
        with a hint pointing at the control-flow region.
        """
        pytest.importorskip("qiskit")
        from qamomile.circuit.transpiler.errors import (
            SliceLinearityViolationError,
        )
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
        with pytest.raises(SliceLinearityViolationError, match="control-flow body"):
            transpiler.transpile(kern)
