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

    def test_slice_assignment_rejected(self):
        """Slice assignment on a Vector is rejected with a clear message."""

        @qmc.qkernel
        def kern(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
            q[0::2] = q[0::2]  # type: ignore[assignment]
            return q

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


@pytest.mark.parametrize(
    "slicer, n, expected_qubits",
    [
        (lambda q: q[0::2], 6, [0, 2, 4]),
        (lambda q: q[1::2], 6, [1, 3, 5]),
        (lambda q: q[1:5], 6, [1, 2, 3, 4]),
        (lambda q: q[2::3], 10, [2, 5, 8]),
    ],
)
def test_slice_emits_expected_qubit_indices(slicer, n, expected_qubits):
    """Emitted H gates land on exactly the qubits the slice describes."""
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    @qmc.qkernel
    def circuit(num: qmc.UInt) -> qmc.Vector[qmc.Bit]:
        q = qmc.qubit_array(num, "q")
        view = slicer(q)
        for i in qmc.range(view.shape[0]):
            view[i] = qmc.h(view[i])
        return qmc.measure(q)

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
        inner = q[0::2][1:3]
        for i in qmc.range(inner.shape[0]):
            inner[i] = qmc.h(inner[i])
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

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Qubit]:
            q = qmc.qubit_array(4, "q")
            evens = q[0:4:2]
            del evens
            q[0] = qmc.h(q[0])
            return q

        with pytest.raises(Exception, match="held by a VectorView slice"):
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
            return q

        assert kern.block is not None

    def test_drained_view_releases_parent_on_consume(self):
        """After writing all view slots back, the parent can be consumed."""

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            evens = q[0:4:2]
            for i in qmc.range(evens.shape[0]):
                evens[i] = qmc.h(evens[i])
            # ``measure`` consumes ``q``; its ``validate_all_returned``
            # must observe that the view is drained and release the
            # slice-borrows on slots 0 and 2.
            return qmc.measure(q)

        assert kern.block is not None

    def test_overlapping_slices_are_rejected(self):
        """Two constant slices covering the same slot can't both be live."""

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Qubit]:
            q = qmc.qubit_array(4, "q")
            _a = q[0:4:2]  # covers {0, 2}
            _b = q[0:4:1]  # also covers {0}
            return q

        with pytest.raises(Exception, match="already owned by another slice view"):
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

    def test_view_consumption_releases_parent_bulk_borrow(self):
        """Passing a view to a kernel releases the parent slice-borrow.

        After the call, the parent's covered slots can be accessed
        directly again — otherwise a subsequent ``q[0]`` would be
        rejected as "held by a slice view".
        """

        @qmc.qkernel
        def h_all(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
            n = q.shape[0]
            for i in qmc.range(n):
                q[i] = qmc.h(q[i])
            return q

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Qubit]:
            q = qmc.qubit_array(4, "q")
            evens = q[0:4:2]
            evens = h_all(evens)
            del evens
            q[0] = qmc.x(q[0])  # must be accessible again
            return q

        assert circuit.block is not None
