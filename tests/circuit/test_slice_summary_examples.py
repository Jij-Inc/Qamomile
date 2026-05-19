"""Literal-as-written verification of every example in the slicing summary.

The slicing summary (`.claude/my_claude/pr_357_slicing_summary.md`) shows
worked-out code samples in Section 1 (legal patterns) and Section 2
(illegal / error patterns).  This file pins every one of those samples
under exactly the form the summary presents them in — same kernel body,
same bindings — and asserts:

1. **Transpile** succeeds on every supported SDK
   (Qiskit / QuriParts / CUDA-Q; the latter two via ``importorskip``).
2. **Execute** produces the expected outcome — full-amplitude
   statevector check against a Qiskit reference circuit for the
   measurement kernels, ``run()`` for the QFixed-Float-return kernel.
3. **Resource estimation** (``estimate_resources``) returns the same
   qubit / gate counts on every run — even though the resource
   estimator looks at the IR and is backend-agnostic, repeating it here
   keeps the summary's claims auditable.

For Section 2 (negative patterns), the file pins the error type and the
detection point (frontend trace vs. post-fold IR pass) so that any
future regression that re-introduces silent miscompilation surfaces
loudly here.

The file is deliberately a one-to-one mirror of the summary's prose —
if a sample is added / changed in the summary, the corresponding test
here should be added / updated together with it.
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
# Section 1 OK kernels — defined at module level so ``@qmc.qkernel`` can
# read their source for AST tracing.
# ---------------------------------------------------------------------------


@qmc.qkernel
def _section_1_1(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    """Section 1-1: evens / odds + H + CX, explicit slice-assign return."""
    q = qmc.qubit_array(n, "q")
    evens = q[0::2]
    odds = q[1::2]
    for i in qmc.range(odds.shape[0]):
        evens[i] = qmc.h(evens[i])
        evens[i], odds[i] = qmc.cx(evens[i], odds[i])
    q[0::2] = evens
    q[1::2] = odds
    return qmc.measure(q)


@qmc.qkernel
def _section_1_2(lo: qmc.UInt, hi: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    """Section 1-2: ``qft`` on a slice view, explicit slice-assign return."""
    q = qmc.qubit_array(8, "q")
    view = q[lo:hi]
    view = qmc.qft(view)
    q[lo:hi] = view
    return qmc.measure(q)


@qmc.qkernel
def _section_1_3() -> qmc.Float:
    """Section 1-3: cast a slice view to ``QFixed`` and measure."""
    q = qmc.qubit_array(4, "q")
    qf = qmc.cast(q[1::2], qmc.QFixed, int_bits=0)
    return qmc.measure(qf)


@qmc.qkernel
def _section_1_4(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    """Section 1-4: in-place ``q[1:n-2] = qmc.h(q[1:n-2])`` slice assignment."""
    q = qmc.qubit_array(n, "q")
    q[1 : n - 2] = qmc.h(q[1 : n - 2])
    return qmc.measure(q)


@qmc.qkernel
def _section_1_5(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    """Section 1-5: top-level slice + per-element loop + top-level release."""
    q = qmc.qubit_array(n, "q")
    evens = q[0::2]
    for i in qmc.range(evens.shape[0]):
        evens[i] = qmc.h(evens[i])
    q[0::2] = evens
    return qmc.measure(q)


@qmc.qkernel
def _section_1_6() -> qmc.Vector[qmc.Bit]:
    """Section 1-6: Python-style auto-truncation (``q[3:10]`` → ``q[3:4]``)."""
    q = qmc.qubit_array(4, "q")
    q[3:10] = qmc.h(q[3:10])
    return qmc.measure(q)


@qmc.qkernel
def _h_all(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Helper for Section 1-7: H broadcast via per-element loop."""
    n = q.shape[0]
    for i in qmc.range(n):
        q[i] = qmc.h(q[i])
    return q


@qmc.qkernel
def _section_1_7(num: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    """Section 1-7: pass a slice view as a ``Vector[Qubit]`` argument."""
    q = qmc.qubit_array(num, "q")
    evens = q[0::2]
    evens = _h_all(evens)
    return qmc.measure(q)


# ---------------------------------------------------------------------------
# Cross-backend statevector / sampling helpers
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
# Section 1-1: evens / odds + H + CX, explicit return
# ---------------------------------------------------------------------------


class TestSection_1_1_EvensOddsCx:
    """Section 1-1: H on evens then CX(evens[i], odds[i]) produces Bell pairs."""

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
        _assert_cross_backend_matches(_section_1_1, bindings, expected_sv)

    @pytest.mark.parametrize(
        "n, exp_qubits, exp_single, exp_two",
        [(4, 4, 2, 2), (6, 6, 3, 3)],
    )
    def test_resource_estimate(
        self, n: int, exp_qubits: int, exp_single: int, exp_two: int
    ):
        """Resource estimate matches H + CX counts inside the loop."""
        est = estimate_resources(_section_1_1.block, bindings={"n": n})
        assert int(est.qubits) == exp_qubits
        assert int(est.gates.single_qubit) == exp_single
        assert int(est.gates.two_qubit) == exp_two
        assert int(est.gates.total) == exp_single + exp_two


# ---------------------------------------------------------------------------
# Section 1-2: qft on a slice view, explicit slice-assign return
# ---------------------------------------------------------------------------


class TestSection_1_2_QftOnView:
    """Section 1-2: ``qft(q[lo:hi])`` applies QFT to the parent's slice."""

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
        _assert_cross_backend_matches(_section_1_2, bindings, expected_sv)

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
        est = estimate_resources(_section_1_2.block, bindings={"lo": 1, "hi": 4})
        assert int(est.qubits) == 8
        assert int(est.gates.total) == 0


# ---------------------------------------------------------------------------
# Section 1-3: cast a slice view to QFixed and measure
# ---------------------------------------------------------------------------


class TestSection_1_3_CastSliceToQFixed:
    """Section 1-3: ``cast(q[1::2], QFixed)`` then ``measure(qf)``.

    The kernel applies no gates, so every qubit stays |0>.  The QFixed
    decode of two |0> qubits with ``int_bits=0`` is the value 0.0.
    """

    def test_transpile_and_run_returns_zero(self):
        """The Float result of measuring |0>^2 as QFixed is 0.0."""
        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(_section_1_3)
        got = exe.run(transpiler.executor()).result()
        assert float(got) == pytest.approx(0.0, abs=1e-9)

    def test_quri_parts_run_returns_zero(self):
        """Cross-backend: QuriParts ``run`` also returns 0.0."""
        pytest.importorskip("quri_parts")
        pytest.importorskip("quri_parts.qulacs")
        from qamomile.quri_parts import QuriPartsTranspiler

        transpiler = QuriPartsTranspiler()
        exe = transpiler.transpile(_section_1_3)
        got = exe.run(transpiler.executor()).result()
        assert float(got) == pytest.approx(0.0, abs=1e-9)

    def test_cudaq_run_returns_zero(self):
        """Cross-backend: CUDA-Q ``run`` also returns 0.0."""
        pytest.importorskip("cudaq")
        from qamomile.cudaq import CudaqTranspiler

        transpiler = CudaqTranspiler()
        exe = transpiler.transpile(_section_1_3)
        got = exe.run(transpiler.executor()).result()
        assert float(got) == pytest.approx(0.0, abs=1e-9)

    def test_resource_estimate(self):
        """4 qubits allocated, no gates."""
        est = estimate_resources(_section_1_3.block)
        assert int(est.qubits) == 4
        assert int(est.gates.total) == 0


# ---------------------------------------------------------------------------
# Section 1-4: q[1:n-2] = qmc.h(q[1:n-2])
# ---------------------------------------------------------------------------


class TestSection_1_4_SymbolicSliceAssign:
    """Section 1-4: in-place ``q[1:n-2] = qmc.h(q[1:n-2])``.

    With ``n=6`` the slice covers ``q[1:4] = {1, 2, 3}``, so the
    expected unitary is H on those three qubits.
    """

    def test_transpile_and_statevector_match(self):
        bindings = {"n": 6}
        expected_sv = _reference_statevector(6, [1, 2, 3])
        _assert_cross_backend_matches(_section_1_4, bindings, expected_sv)

    def test_resource_estimate(self):
        est = estimate_resources(_section_1_4.block, bindings={"n": 6})
        assert int(est.qubits) == 6
        assert int(est.gates.single_qubit) == 3
        assert int(est.gates.two_qubit) == 0
        assert int(est.gates.total) == 3


# ---------------------------------------------------------------------------
# Section 1-5: top-level slice + body loop + top-level release
# ---------------------------------------------------------------------------


class TestSection_1_5_TopLevelSliceWithBodyLoop:
    """Section 1-5: ``evens = q[0::2]; for i ...; q[0::2] = evens``."""

    def test_transpile_and_statevector_match(self):
        bindings = {"n": 4}
        expected_sv = _reference_statevector(4, [0, 2])
        _assert_cross_backend_matches(_section_1_5, bindings, expected_sv)

    def test_resource_estimate(self):
        est = estimate_resources(_section_1_5.block, bindings={"n": 4})
        assert int(est.qubits) == 4
        assert int(est.gates.single_qubit) == 2
        assert int(est.gates.two_qubit) == 0
        assert int(est.gates.total) == 2


# ---------------------------------------------------------------------------
# Section 1-6: auto-truncated slice (q[3:10] → q[3:4])
# ---------------------------------------------------------------------------


class TestSection_1_6_AutoTruncatedSlice:
    """Section 1-6: out-of-range stop is truncated to the parent length."""

    def test_transpile_and_statevector_match(self):
        # q[3:10] on a length-4 array is truncated to q[3:4]; only q[3]
        # gets H.
        expected_sv = _reference_statevector(4, [3])
        _assert_cross_backend_matches(_section_1_6, {}, expected_sv)

    def test_resource_estimate(self):
        est = estimate_resources(_section_1_6.block)
        assert int(est.qubits) == 4
        assert int(est.gates.single_qubit) == 1
        assert int(est.gates.two_qubit) == 0
        assert int(est.gates.total) == 1


# ---------------------------------------------------------------------------
# Section 1-7: view as qkernel argument
# ---------------------------------------------------------------------------


class TestSection_1_7_ViewAsKernelArgument:
    """Section 1-7: ``evens = h_all(evens)`` passes a slice view to a sub-kernel."""

    def test_transpile_and_statevector_match(self):
        bindings = {"num": 6}
        expected_sv = _reference_statevector(6, [0, 2, 4])
        _assert_cross_backend_matches(_section_1_7, bindings, expected_sv)

    def test_resource_estimate(self):
        est = estimate_resources(_section_1_7.block, bindings={"num": 6})
        assert int(est.qubits) == 6
        assert int(est.gates.single_qubit) == 3
        assert int(est.gates.two_qubit) == 0
        assert int(est.gates.total) == 3


# ---------------------------------------------------------------------------
# Section 2: error patterns — module-level kernels for source-readable tracing
# ---------------------------------------------------------------------------


@qmc.qkernel
def _ng_a_body_release() -> qmc.Vector[qmc.Bit]:
    """NG A: body-internal release of an outer-registered view."""
    q = qmc.qubit_array(4, "q")
    even = q[0::2]
    for _ in qmc.range(1):
        q[0::2] = even  # ← rejected post-fold by ``outer_snapshot_stack``
    return qmc.measure(q)


@qmc.qkernel
def _ng_b_body_overlap() -> qmc.Vector[qmc.Bit]:
    """NG B: body-internal slice that overlaps an outer view (concrete bounds)."""
    q = qmc.qubit_array(4, "q")
    even = q[0::2]  # noqa: F841 — kept live to force the overlap conflict
    for _ in qmc.range(1):
        v = q[0:3]  # ← rejected at trace time by ``VectorView._wrap``
        v = qmc.h(v)
        q[0:3] = v
    return qmc.measure(q)


@qmc.qkernel
def _ng_c_coverage_mismatch() -> qmc.Vector[qmc.Bit]:
    """NG C: slice-assignment LHS / RHS coverage mismatch."""
    q = qmc.qubit_array(4, "q")
    q[0:2] = qmc.h(q[0:3])  # ← AffineTypeError at trace time
    return qmc.measure(q)


@qmc.qkernel
def _ng_d_top_level_overlap() -> qmc.Vector[qmc.Bit]:
    """NG D: two overlapping live views on the same root."""
    q = qmc.qubit_array(4, "q")
    a = q[0:3]
    b = q[0:2]  # ← QubitConsumedError at trace time
    _ = a
    _ = b
    return qmc.measure(q)


@qmc.qkernel
def _ng_e_strict_return_violation() -> qmc.Vector[qmc.Bit]:
    """NG E: drained view not returned via slice assignment before parent consume."""
    q = qmc.qubit_array(4, "q")
    evens = q[0:4:2]
    for i in qmc.range(evens.shape[0]):
        evens[i] = qmc.h(evens[i])
    # Missing ``q[0:4:2] = evens`` — strict-return policy rejects.
    return qmc.measure(q)


class TestSection_2_NegativePatterns:
    """Section 2 NG patterns must raise the documented error type."""

    def test_ng_a_body_release_raises(self):
        """``q[0::2] = even`` inside a for body trips the post-fold guard."""
        from qamomile.circuit.transpiler.errors import SliceLinearityViolationError

        transpiler = QiskitTranspiler()
        with pytest.raises(SliceLinearityViolationError):
            transpiler.transpile(_ng_a_body_release)

    def test_ng_b_body_overlap_raises_at_trace(self):
        """Concrete-bounds overlap is rejected during ``VectorView._wrap``."""
        from qamomile.circuit.transpiler.errors import QubitConsumedError

        with pytest.raises(
            QubitConsumedError, match="already owned by another slice view"
        ):
            _ = _ng_b_body_overlap.block

    def test_ng_c_coverage_mismatch_raises(self):
        """LHS / RHS slice coverage mismatch is rejected at trace time."""
        from qamomile.circuit.transpiler.errors import AffineTypeError

        with pytest.raises(AffineTypeError, match="coverage mismatch"):
            _ = _ng_c_coverage_mismatch.block

    def test_ng_d_top_level_overlap_raises_at_trace(self):
        """``a = q[0:3]; b = q[0:2]`` overlap is rejected at ``b``'s ``_wrap``."""
        from qamomile.circuit.transpiler.errors import QubitConsumedError

        with pytest.raises(
            QubitConsumedError, match="already owned by another slice view"
        ):
            _ = _ng_d_top_level_overlap.block

    def test_ng_e_strict_return_violation_raises(self):
        """Drained view + ``measure(q)`` without slice-assign-back is rejected."""
        from qamomile.circuit.transpiler.errors import UnreturnedBorrowError

        with pytest.raises(UnreturnedBorrowError, match="unreturned slice-view"):
            _ = _ng_e_strict_return_violation.block
