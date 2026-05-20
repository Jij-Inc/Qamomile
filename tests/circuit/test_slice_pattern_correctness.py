"""Cross-backend correctness for representative Vector slicing patterns.

This file groups slice tests into two complementary halves so any
regression — silent miscompilation on the legal side, swallowed error
on the illegal side — surfaces loudly under one ``pytest`` invocation.

**Legal patterns** (transpile + execute + resource estimate).  Each
kernel is a small, self-contained example of a way users are expected
to write slicing code — evens / odds Bell-pair loop, ``qft`` on a
slice, ``cast`` to ``QFixed``, in-line ``q[a:b] = h(q[a:b])`` broadcast,
top-level slice with a body loop, Python-style auto-truncation, and
passing a view as a sub-kernel argument.  Each kernel is transpiled
on every supported SDK (Qiskit / QuriParts / CUDA-Q via
``importorskip``), the result statevector compared to an analytical
Qiskit reference (or ``run()`` for the QFixed Float-return kernel),
and ``estimate_resources`` pinned to the expected qubit / gate counts.
A backend-agnostic IR-level resource estimator is intentionally
re-asserted here so the per-pattern claim stays auditable.

**Illegal patterns** (error-raising), grouped by concept:

- ``TestStrictReturnViolations`` — every transfer-style view consume
  (element loop, broadcast, sub-kernel call, ``pauli_evolve``) must
  be followed by a slice-assign return; otherwise ``measure(q)``
  raises ``UnreturnedBorrowError``.
- ``TestMultiViewOverlap`` — two live views that share parent slots
  are rejected at trace time by ``VectorView._wrap``.
- ``TestSliceAssignmentConsistency`` — left- and right-hand sides of
  ``q[a:b] = ...`` must cover the same slot set.
- ``TestControlFlowBodyViolations`` — releasing an outer-registered
  view from inside a loop / branch body is post-fold-rejected by
  ``SliceBorrowCheckPass``'s outer-snapshot guard.
- ``TestNestedSliceReturnOrder`` — nested slices must come back
  inner→outer→root; inner-direct-to-root, outer-while-inner-live,
  and outer-element-at-inner-slot all raise.

Each failure case asserts both the error type and the detection point
(frontend trace-time vs. post-fold IR pass) so the regression
surfaces are explicit.

Helper convention used throughout the file: kernels appear next to
the test that exercises them.  When a kernel is shared across the
``test_*`` methods of a single class, it lives inside that class as
a class-level ``@qmc.qkernel`` attribute (Python looks ``self._kern``
up via the class's ``__dict__``; ``QKernel`` is not a descriptor so
no spurious ``self``-binding happens).  When a kernel is used by
exactly one test method it is defined inline at the top of that
method.  Only the cross-backend statevector helpers and the
analytical reference builder are module-level — they are genuinely
shared by every legal-pattern test class.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import qamomile.circuit as qmc

qiskit = pytest.importorskip("qiskit")
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.quantum_info import Statevector  # noqa: E402

from qamomile.circuit.estimator import estimate_resources  # noqa: E402
from qamomile.qiskit import QiskitTranspiler  # noqa: E402
from tests.transpiler.gate_test_specs import statevectors_equal  # noqa: E402

# ---------------------------------------------------------------------------
# Cross-backend helpers (used by every legal-pattern test class below).
# ---------------------------------------------------------------------------


def _qiskit_statevector(kern, bindings: dict) -> np.ndarray:
    """Transpile with Qiskit and return the unitary statevector.

    Args:
        kern: Compiled qkernel.
        bindings (dict): Compile-time bindings forwarded to
            ``transpile``.

    Returns:
        np.ndarray: Complex amplitudes after stripping the final
            measurement instructions from the compiled circuit.
    """
    transpiler = QiskitTranspiler()
    exe = transpiler.transpile(kern, bindings=bindings)
    qc = exe.compiled_quantum[0].circuit
    return np.array(Statevector(qc.remove_final_measurements(inplace=False)).data)


def _quri_parts_statevector(kern, bindings: dict) -> np.ndarray:
    """Transpile with QuriParts and read the unitary statevector via Qulacs.

    Skips if ``quri_parts`` / ``quri_parts.qulacs`` is unavailable.

    Args:
        kern: Compiled qkernel.
        bindings (dict): Compile-time bindings.

    Returns:
        np.ndarray: Complex amplitudes from the Qulacs simulator.
    """
    pytest.importorskip("quri_parts")
    pytest.importorskip("quri_parts.qulacs")
    from quri_parts.core.state import GeneralCircuitQuantumState
    from quri_parts.qulacs.simulator import evaluate_state_to_vector

    from qamomile.quri_parts import QuriPartsTranspiler

    transpiler = QuriPartsTranspiler()
    exe = transpiler.transpile(kern, bindings=bindings)
    qp_circuit = exe.compiled_quantum[0].circuit
    if hasattr(qp_circuit, "parameter_count") and qp_circuit.parameter_count > 0:
        bound = qp_circuit.bind_parameters([0.0] * qp_circuit.parameter_count)
    elif hasattr(qp_circuit, "bind_parameters"):
        bound = qp_circuit.bind_parameters([])
    else:
        bound = qp_circuit
    state = GeneralCircuitQuantumState(bound.qubit_count, bound)
    return np.array(evaluate_state_to_vector(state).vector)


def _cudaq_statevector(kern, bindings: dict) -> np.ndarray:
    """Transpile with CUDA-Q and read the unitary statevector.

    Skips if ``cudaq`` is unavailable.

    Args:
        kern: Compiled qkernel.
        bindings (dict): Compile-time bindings.

    Returns:
        np.ndarray: Complex amplitudes from the CUDA-Q state simulator.
    """
    pytest.importorskip("cudaq")
    import cudaq

    from qamomile.cudaq import CudaqTranspiler

    transpiler = CudaqTranspiler()
    exe = transpiler.transpile(kern, bindings=bindings)
    cudaq_circuit = exe.compiled_quantum[0].circuit
    return np.array(cudaq.get_state(cudaq_circuit.kernel_func))


def _assert_cross_backend_matches(kern, bindings: dict, expected_sv: np.ndarray):
    """Assert that every available backend matches the reference statevector.

    Backends that are not installed are silently skipped via
    ``pytest.importorskip`` inside the per-backend helpers; the Qiskit
    branch is always exercised.

    Args:
        kern: Compiled qkernel.
        bindings (dict): Compile-time bindings.
        expected_sv (np.ndarray): Reference unitary statevector.
    """
    actual_qiskit = _qiskit_statevector(kern, bindings)
    assert statevectors_equal(actual_qiskit, expected_sv), (
        f"[qiskit] statevector mismatch.\n"
        f"  actual:   {actual_qiskit}\n"
        f"  expected: {expected_sv}"
    )

    actual_quri = _quri_parts_statevector(kern, bindings)
    assert statevectors_equal(actual_quri, expected_sv), (
        f"[quri_parts] statevector mismatch.\n"
        f"  actual:   {actual_quri}\n"
        f"  expected: {expected_sv}"
    )

    actual_cudaq = _cudaq_statevector(kern, bindings)
    assert statevectors_equal(actual_cudaq, expected_sv), (
        f"[cudaq] statevector mismatch.\n"
        f"  actual:   {actual_cudaq}\n"
        f"  expected: {expected_sv}"
    )


def _reference_statevector(n_qubits: int, h_qubits: list[int]) -> np.ndarray:
    """Build a reference statevector: apply H to each of ``h_qubits`` on |0>.

    Args:
        n_qubits (int): Total qubits in the reference circuit.
        h_qubits (list[int]): Qubit indices the kernel should apply
            ``H`` to.

    Returns:
        np.ndarray: Complex amplitudes after applying H to those qubits.
    """
    qc = QuantumCircuit(n_qubits)
    for q in h_qubits:
        qc.h(q)
    return np.array(Statevector(qc).data)


# ---------------------------------------------------------------------------
# Legal slicing patterns — each class defines its own kernel inline.
# ---------------------------------------------------------------------------


class TestEvensOddsCxPattern:
    """H on evens then CX(evens[i], odds[i]) produces Bell pairs."""

    @qmc.qkernel
    def _kern(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
        """Evens / odds + H + CX, explicit slice-assign return."""
        q = qmc.qubit_array(n, "q")
        evens = q[0::2]
        odds = q[1::2]
        for i in qmc.range(odds.shape[0]):
            evens[i] = qmc.h(evens[i])
            evens[i], odds[i] = qmc.cx(evens[i], odds[i])
        q[0::2] = evens
        q[1::2] = odds
        return qmc.measure(q)

    @pytest.mark.parametrize("n", [4, 6])
    def test_transpile_and_statevector_match(self, n: int):
        """Cross-backend statevector matches ``H_evens then CX(evens, odds)``."""
        bindings = {"n": n}
        ref_qc = QuantumCircuit(n)
        n_pairs = n // 2
        for i in range(n_pairs):
            ref_qc.h(2 * i)
        for i in range(n_pairs):
            ref_qc.cx(2 * i, 2 * i + 1)
        expected_sv = np.array(Statevector(ref_qc).data)
        _assert_cross_backend_matches(self._kern, bindings, expected_sv)

    @pytest.mark.parametrize(
        "n, exp_qubits, exp_single, exp_two",
        [(4, 4, 2, 2), (6, 6, 3, 3)],
    )
    def test_resource_estimate(
        self, n: int, exp_qubits: int, exp_single: int, exp_two: int
    ):
        """Resource estimate matches H + CX counts inside the loop."""
        est = estimate_resources(self._kern.block, bindings={"n": n})
        assert int(est.qubits) == exp_qubits
        assert int(est.gates.single_qubit) == exp_single
        assert int(est.gates.two_qubit) == exp_two
        assert int(est.gates.total) == exp_single + exp_two


class TestQftOnView:
    """``qft(q[lo:hi])`` applies QFT to the parent's slice."""

    @qmc.qkernel
    def _kern(lo: qmc.UInt, hi: qmc.UInt) -> qmc.Vector[qmc.Bit]:
        """``qft`` on a slice view, explicit slice-assign return."""
        q = qmc.qubit_array(8, "q")
        view = q[lo:hi]
        view = qmc.qft(view)
        q[lo:hi] = view
        return qmc.measure(q)

    def test_transpile_and_statevector_match(self):
        """``qft(|000>) = |+++>``; the rest of the register stays |0>."""
        bindings = {"lo": 1, "hi": 4}
        # ``qft(|000>) = (1/sqrt(8)) * sum_k |k>`` for a 3-qubit register,
        # so on the full 8-qubit |00000000> input the slice ``q[1:4]``
        # becomes a uniform superposition while q[0] / q[4..7] stay |0>.
        # The expected statevector has amplitude 1/sqrt(8) at every
        # basis state with q[0]=q[4]=q[5]=q[6]=q[7]=0 (any q[1..3]).
        n_full = 8
        expected_sv = np.zeros(1 << n_full, dtype=complex)
        amp = 1.0 / math.sqrt(8)
        for k in range(8):
            # Qiskit indexes qubit 0 as the least-significant bit.
            idx = (k & 0b1) << 1 | ((k >> 1) & 0b1) << 2 | ((k >> 2) & 0b1) << 3
            expected_sv[idx] = amp
        _assert_cross_backend_matches(self._kern, bindings, expected_sv)

    def test_resource_estimate(self):
        """``qft`` is wrapped as a composite gate; the top-level block reports 0 leaf gates.

        ``estimate_resources`` walks the IR ``Block`` at the level the
        kernel produced it.  ``CompositeGateOperation`` is one logical
        op, not a tree of H / CP, so leaf gate counters return zero.
        The qubit count (8 — the full ``qubit_array`` size) is still
        reported correctly.  This is intentional behaviour, not a slice
        regression; we pin it so any future estimator change that
        starts unpacking composite gates is noticed.
        """
        est = estimate_resources(self._kern.block, bindings={"lo": 1, "hi": 4})
        assert int(est.qubits) == 8
        assert int(est.gates.total) == 0


class TestCastSliceToQFixed:
    """``cast(q[1::2], QFixed)`` then ``measure(qf)``.

    The kernel applies no gates, so every qubit stays |0>.  The QFixed
    decode of two |0> qubits with ``int_bits=0`` is the value 0.0.
    """

    @qmc.qkernel
    def _kern() -> qmc.Float:
        """Cast a slice view to ``QFixed`` and measure."""
        q = qmc.qubit_array(4, "q")
        qf = qmc.cast(q[1::2], qmc.QFixed, int_bits=0)
        return qmc.measure(qf)

    def test_transpile_and_run_returns_zero(self):
        """The Float result of measuring |0>^2 as QFixed is 0.0."""
        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(self._kern)
        got = exe.run(transpiler.executor()).result()
        assert float(got) == pytest.approx(0.0, abs=1e-9)

    def test_quri_parts_run_returns_zero(self):
        """Cross-backend: QuriParts ``run`` also returns 0.0."""
        pytest.importorskip("quri_parts")
        pytest.importorskip("quri_parts.qulacs")
        from qamomile.quri_parts import QuriPartsTranspiler

        transpiler = QuriPartsTranspiler()
        exe = transpiler.transpile(self._kern)
        got = exe.run(transpiler.executor()).result()
        assert float(got) == pytest.approx(0.0, abs=1e-9)

    def test_cudaq_run_returns_zero(self):
        """Cross-backend: CUDA-Q ``run`` also returns 0.0."""
        pytest.importorskip("cudaq")
        from qamomile.cudaq import CudaqTranspiler

        transpiler = CudaqTranspiler()
        exe = transpiler.transpile(self._kern)
        got = exe.run(transpiler.executor()).result()
        assert float(got) == pytest.approx(0.0, abs=1e-9)

    def test_resource_estimate(self):
        """4 qubits allocated, no gates."""
        est = estimate_resources(self._kern.block)
        assert int(est.qubits) == 4
        assert int(est.gates.total) == 0


class TestInlineSliceAssignBroadcast:
    """In-place ``q[1:n-2] = qmc.h(q[1:n-2])``.

    With ``n=6`` the slice covers ``q[1:4] = {1, 2, 3}``, so the
    expected unitary is H on those three qubits.
    """

    @qmc.qkernel
    def _kern(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
        """In-place ``q[1:n-2] = qmc.h(q[1:n-2])`` slice assignment."""
        q = qmc.qubit_array(n, "q")
        q[1 : n - 2] = qmc.h(q[1 : n - 2])
        return qmc.measure(q)

    def test_transpile_and_statevector_match(self):
        bindings = {"n": 6}
        expected_sv = _reference_statevector(6, [1, 2, 3])
        _assert_cross_backend_matches(self._kern, bindings, expected_sv)

    def test_resource_estimate(self):
        est = estimate_resources(self._kern.block, bindings={"n": 6})
        assert int(est.qubits) == 6
        assert int(est.gates.single_qubit) == 3
        assert int(est.gates.two_qubit) == 0
        assert int(est.gates.total) == 3


class TestTopLevelSliceWithBodyLoop:
    """``evens = q[0::2]; for i ...; q[0::2] = evens``."""

    @qmc.qkernel
    def _kern(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
        """Top-level slice + per-element loop + top-level release."""
        q = qmc.qubit_array(n, "q")
        evens = q[0::2]
        for i in qmc.range(evens.shape[0]):
            evens[i] = qmc.h(evens[i])
        q[0::2] = evens
        return qmc.measure(q)

    def test_transpile_and_statevector_match(self):
        bindings = {"n": 4}
        expected_sv = _reference_statevector(4, [0, 2])
        _assert_cross_backend_matches(self._kern, bindings, expected_sv)

    def test_resource_estimate(self):
        est = estimate_resources(self._kern.block, bindings={"n": 4})
        assert int(est.qubits) == 4
        assert int(est.gates.single_qubit) == 2
        assert int(est.gates.two_qubit) == 0
        assert int(est.gates.total) == 2


class TestAutoTruncatedSlice:
    """Out-of-range stop is truncated to the parent length."""

    @qmc.qkernel
    def _kern() -> qmc.Vector[qmc.Bit]:
        """Python-style auto-truncation (``q[3:10]`` → ``q[3:4]``)."""
        q = qmc.qubit_array(4, "q")
        q[3:10] = qmc.h(q[3:10])
        return qmc.measure(q)

    def test_transpile_and_statevector_match(self):
        # q[3:10] on a length-4 array is truncated to q[3:4]; only q[3]
        # gets H.
        expected_sv = _reference_statevector(4, [3])
        _assert_cross_backend_matches(self._kern, {}, expected_sv)

    def test_resource_estimate(self):
        est = estimate_resources(self._kern.block)
        assert int(est.qubits) == 4
        assert int(est.gates.single_qubit) == 1
        assert int(est.gates.two_qubit) == 0
        assert int(est.gates.total) == 1


# ``_h_all`` is the sub-kernel that ``TestViewPassedToSubKernel._kern``
# passes its slice view to.  It is *only* used by that test class, but
# it must live at module scope because the body of ``_kern`` references
# it as a bare name and Python's name resolution looks free names up
# via the function's ``__globals__`` (the enclosing module's globals),
# not via the surrounding class body.  Keeping ``_h_all`` immediately
# above the class preserves visual proximity.
@qmc.qkernel
def _h_all(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Broadcast H over every element of a ``Vector[Qubit]`` via a per-element loop."""
    n = q.shape[0]
    for i in qmc.range(n):
        q[i] = qmc.h(q[i])
    return q


class TestViewPassedToSubKernel:
    """Passing a slice view as a ``Vector[Qubit]`` argument to another kernel."""

    @qmc.qkernel
    def _kern(num: qmc.UInt) -> qmc.Vector[qmc.Bit]:
        """Pass a slice view as a ``Vector[Qubit]`` argument."""
        q = qmc.qubit_array(num, "q")
        evens = q[0::2]
        evens = _h_all(evens)
        q[0::2] = evens
        return qmc.measure(q)

    def test_transpile_and_statevector_match(self):
        bindings = {"num": 6}
        expected_sv = _reference_statevector(6, [0, 2, 4])
        _assert_cross_backend_matches(self._kern, bindings, expected_sv)

    def test_resource_estimate(self):
        est = estimate_resources(self._kern.block, bindings={"num": 6})
        assert int(est.qubits) == 6
        assert int(est.gates.single_qubit) == 3
        assert int(est.gates.two_qubit) == 0
        assert int(est.gates.total) == 3


# ---------------------------------------------------------------------------
# Illegal slicing patterns — each test method inlines its own kernel
# ---------------------------------------------------------------------------


class TestStrictReturnViolations:
    """Every view-consuming op that's not destructive demands a slice-assign return.

    Skipping that return leaves the parent's bulk-borrow live and the
    next parent consume (``measure(q)`` etc.) raises
    ``UnreturnedBorrowError``.  Four entry points exercise the four
    transfer-style ops the policy calls out: element loop, broadcast
    gate, sub-kernel call, and ``pauli_evolve``.
    """

    def test_loop_pattern_without_slice_assign_raises(self):
        """Element loop + no slice-assign on the loop's view."""
        from qamomile.circuit.transpiler.errors import UnreturnedBorrowError

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            evens = q[0:4:2]
            for i in qmc.range(evens.shape[0]):
                evens[i] = qmc.h(evens[i])
            return qmc.measure(q)

        with pytest.raises(UnreturnedBorrowError, match="unreturned slice-view"):
            _ = kern.block

    def test_broadcast_without_slice_assign_raises(self):
        """``view = qmc.h(view)`` + no slice-assign on the returned view."""
        from qamomile.circuit.transpiler.errors import UnreturnedBorrowError

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            view = q[0::2]
            view = qmc.h(view)
            return qmc.measure(q)

        with pytest.raises(UnreturnedBorrowError, match="unreturned slice-view"):
            _ = kern.block

    def test_qkernel_call_without_slice_assign_raises(self):
        """``view = sub_kernel(view)`` + no slice-assign on the returned view."""
        from qamomile.circuit.transpiler.errors import UnreturnedBorrowError

        @qmc.qkernel
        def callee(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
            for i in qmc.range(2):
                q[i] = qmc.h(q[i])
            return q

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            view = q[0:2]
            view = callee(view)
            return qmc.measure(q)

        with pytest.raises(UnreturnedBorrowError, match="unreturned slice-view"):
            _ = kern.block

    def test_pauli_evolve_without_slice_assign_raises(self):
        """``view = qmc.pauli_evolve(view, ...)`` + no slice-assign on the returned view."""
        from qamomile.circuit.transpiler.errors import UnreturnedBorrowError

        @qmc.qkernel
        def kern(H: qmc.Observable, gamma: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            view = q[0::2]
            view = qmc.pauli_evolve(view, H, gamma)
            return qmc.measure(q)

        with pytest.raises(UnreturnedBorrowError, match="unreturned slice-view"):
            _ = kern.block


class TestMultiViewOverlap:
    """Two live views on the same root with overlapping coverage are rejected.

    Strict no-multi-view: at most one live ``VectorView`` may own any
    given parent slot.  Concrete-bounds overlap is caught at trace
    time by ``VectorView._wrap``; the same pattern inside a
    control-flow body is rejected for the same reason (the body's view
    construction sees the outer view's claim on the parent slot).
    """

    def test_top_level_overlap_raises_at_trace(self):
        """``a = q[0:3]; b = q[0:2]`` — two live views with overlapping coverage."""
        from qamomile.circuit.transpiler.errors import QubitConsumedError

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            a = q[0:3]
            b = q[0:2]
            _ = a
            _ = b
            return qmc.measure(q)

        with pytest.raises(
            QubitConsumedError, match="already owned by another slice view"
        ):
            _ = kern.block

    def test_body_internal_overlap_raises_at_trace(self):
        """Outer view live across the body, body creates an overlapping inner view."""
        from qamomile.circuit.transpiler.errors import QubitConsumedError

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            even = q[0::2]  # noqa: F841 — kept live to force the overlap conflict
            for _ in qmc.range(1):
                v = q[0:3]
                v = qmc.h(v)
                q[0:3] = v
            return qmc.measure(q)

        with pytest.raises(
            QubitConsumedError, match="already owned by another slice view"
        ):
            _ = kern.block


class TestSliceAssignmentConsistency:
    """Slice assignment requires LHS and RHS to describe the same slot set.

    Coverage mismatch is the canonical case: the right-hand side
    covers a different number of parent slots than the left-hand
    slice.  Caught at trace time inside ``_return_slice_view`` even
    when the right-hand side has been broadcast-consumed (the helper
    reconstructs coverage from the IR ``ArrayValue``).
    """

    def test_coverage_mismatch_raises_at_trace(self):
        """``q[0:2] = qmc.h(q[0:3])`` — LHS and RHS coverages differ."""
        from qamomile.circuit.transpiler.errors import AffineTypeError

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            q[0:2] = qmc.h(q[0:3])
            return qmc.measure(q)

        with pytest.raises(AffineTypeError, match="coverage mismatch"):
            _ = kern.block


class TestControlFlowBodyViolations:
    """Releasing an outer-registered view from inside a control-flow body is rejected.

    The IR pass's ``outer_snapshot_stack`` guard catches this: the
    body cannot delete a borrow-table entry that the enclosing block
    is still relying on, because the loop / branch merge cannot
    propagate the deletion outward.  Caught post-fold.
    """

    def test_release_of_outer_view_in_body_raises(self):
        """Slice-assign that releases an outer view from inside a for body."""
        from qamomile.circuit.transpiler.errors import SliceBorrowViolationError

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            even = q[0::2]
            for _ in qmc.range(1):
                q[0::2] = even
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        with pytest.raises(SliceBorrowViolationError):
            transpiler.transpile(kern)


class TestNestedSliceReturnOrder:
    """Nested slices must be returned inner→outer→root.

    The slice assignment in :meth:`VectorView._return_slice_view`
    refuses to release an inner view directly to the root parent,
    refuses to release the outer view while the inner still owns
    parent slots, and the element accessor refuses to touch an
    outer-view slot that is currently owned by a nested inner.
    """

    def test_inner_returned_to_root_directly_raises(self):
        """``a = q[1:9]; b = a[1:5]; q[2:6] = b`` skips the outer-view return."""
        from qamomile.circuit.transpiler.errors import AffineTypeError

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(10, "q")
            a = q[1:9]  # noqa: F841
            b = a[1:5]
            q[2:6] = b
            return qmc.measure(q)

        with pytest.raises(
            AffineTypeError, match="returned through its immediate outer view"
        ):
            _ = kern.block

    def test_outer_returned_while_inner_live_raises(self):
        """``q[1:9] = a`` with ``b`` still owning a's overlap slots."""
        from qamomile.circuit.transpiler.errors import AffineTypeError

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(10, "q")
            a = q[1:9]
            b = a[1:5]  # noqa: F841
            q[1:9] = a
            return qmc.measure(q)

        with pytest.raises(AffineTypeError, match="no longer owns"):
            _ = kern.block

    def test_outer_access_at_inner_slot_raises(self):
        """Touching ``a[overlap_idx]`` while inner ``b`` owns that slot."""
        from qamomile.circuit.transpiler.errors import QubitConsumedError

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(10, "q")
            a = q[1:9]
            b = a[1:5]  # noqa: F841
            a[1] = qmc.h(a[1])  # a[1] is root q[2], owned by b
            return qmc.measure(q)

        with pytest.raises(
            QubitConsumedError, match="currently held by another slice view"
        ):
            _ = kern.block
