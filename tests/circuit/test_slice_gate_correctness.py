"""Per-gate correctness tests for slice-broadcast and slice-assignment patterns.

This file is the slicing-specific complement to the existing
single-gate / cross-backend tests in ``tests/circuit/test_*.py``: for
every supported single-qubit gate (and a representative two-qubit
loop pattern), it pins down that **applying the gate through a
slice produces the same physical result as applying it qubit-by-qubit
to the equivalent set of root qubits**.

Coverage matrix:

* All single-qubit fixed gates (``h`` / ``x`` / ``y`` / ``z`` / ``s`` /
  ``t`` / ``sdg`` / ``tdg``) via slice broadcast and via slice-assignment.
* All rotation / phase single-qubit gates (``rx`` / ``ry`` / ``rz`` /
  ``p``) with parametrized ``theta``.
* Two-qubit pairing pattern: alternating ``CX`` over ``q[0::2]`` /
  ``q[1::2]`` views.

For each kernel the test asserts **both**:

1. **Statevector equality** against a reference circuit built directly
   with Qiskit (``qc.h(qi)`` etc., applied per qubit index).  This
   gives a deterministic, full-amplitude check that the slice-broadcast
   path emits the same unitary as the equivalent root-qubit gate
   sequence.
2. **Measurement distribution** matches the analytic probability mass
   derived from the same statevector.  The sample-side assertion uses
   a high shot count and a generous tolerance (chi-square style) so
   that any silent miscompile that changes the gate set or the
   targeted qubits surfaces as a distribution mismatch rather than a
   slightly-off count.

Qiskit is the reference simulator; ``QuriParts`` / ``CUDA-Q`` cross-
backend slicing is already covered by the existing alternating-
hadamard suite.  We keep the reference deterministic here so that a
single failing assertion points clearly at the slicing path rather
than at backend-specific sampling noise.
"""

from __future__ import annotations

import hashlib
import linecache
import math

import numpy as np
import pytest

import qamomile.circuit as qmc

qiskit = pytest.importorskip("qiskit")
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.quantum_info import Statevector  # noqa: E402

from qamomile.qiskit import QiskitTranspiler  # noqa: E402

# Import the same global-phase-aware statevector comparator the
# rest of the suite uses.
from tests.transpiler.gate_test_specs import statevectors_equal  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_final_measurements(qc: QuantumCircuit) -> QuantumCircuit:
    """Return a copy of ``qc`` with all trailing measurements removed.

    ``Statevector`` cannot be computed from a circuit that still
    contains measurement instructions; ``remove_final_measurements``
    drops them in-place on a copy.

    Args:
        qc: The compiled Qiskit circuit (typically obtained via
            ``exe.compiled_quantum[0].circuit``).

    Returns:
        A new ``QuantumCircuit`` instance without any final
        ``Measure`` instructions.  Other gate ops are preserved.
    """
    return qc.remove_final_measurements(inplace=False)


def _compiled_circuit(transpiler: QiskitTranspiler, kern, bindings: dict | None = None):
    """Transpile a kernel and return the underlying Qiskit circuit.

    Args:
        transpiler: The Qiskit transpiler instance to use.
        kern: The ``@qmc.qkernel`` to compile.
        bindings: Optional bindings to forward to ``transpile``.

    Returns:
        The single Qiskit ``QuantumCircuit`` the transpiler produced
        for the quantum segment of this kernel.  All four tests in
        this file have a single quantum segment by construction, so
        ``compiled_quantum[0].circuit`` is safe.
    """
    exe = transpiler.transpile(kern, bindings=bindings or {})
    return exe, exe.compiled_quantum[0].circuit


def _statevector_from_kernel(
    transpiler: QiskitTranspiler, kern, bindings=None
) -> Statevector:
    """Compile ``kern`` and return the statevector of its unitary core.

    The kernel's final measurement is stripped before statevector
    extraction; ``Statevector(qc)`` then evolves the all-zeros input
    through the compiled gate sequence.

    Args:
        transpiler: Qiskit transpiler instance.
        kern: Kernel to compile.
        bindings: Optional kernel bindings.

    Returns:
        The output ``Statevector`` after evolving ``|0...0>``
        through the kernel's gate sequence.
    """
    _exe, qc = _compiled_circuit(transpiler, kern, bindings)
    qc_no_meas = _strip_final_measurements(qc)
    return Statevector(qc_no_meas)


def _reference_statevector(
    n_qubits: int,
    target_qubits: list[int],
    qiskit_gate_apply,
) -> Statevector:
    """Build the expected statevector by applying ``qiskit_gate_apply`` per index.

    Args:
        n_qubits: Number of qubits in the reference circuit; must
            match the qubit count of the kernel under test.
        target_qubits: Qubit indices the gate should land on.  The
            same set the slice on the kernel side covers.
        qiskit_gate_apply: ``Callable[[QuantumCircuit, int], None]``
            that applies one gate to one qubit (e.g.
            ``lambda qc, qi: qc.h(qi)``).

    Returns:
        The reference ``Statevector`` after evolving the all-zeros
        input through the equivalent per-qubit gate sequence.
    """
    qc = QuantumCircuit(n_qubits)
    for qi in target_qubits:
        qiskit_gate_apply(qc, qi)
    return Statevector(qc)


def _assert_sample_matches_statevector(
    transpiler: QiskitTranspiler,
    kern,
    expected_sv: Statevector,
    bindings=None,
    shots: int = 4096,
    rtol: float = 0.06,
):
    """Sample the kernel and assert the bitstring distribution matches expected.

    For each measured bitstring the empirical frequency
    ``count / shots`` must be within ``rtol`` of the analytic
    probability ``|<bitstring|expected_sv>|^2``.  Bitstrings the
    expected statevector assigns negligible amplitude to (below
    ``1e-9``) must also appear with negligible frequency
    (``count / shots < rtol``).

    Args:
        transpiler: Qiskit transpiler instance.
        kern: Kernel to sample.
        expected_sv: Reference statevector to compare against.
        bindings: Optional kernel bindings.
        shots: Number of shots to sample.  4096 gives ~1% statistical
            noise on a 50/50 split which is well within the
            6% ``rtol`` band.
        rtol: Tolerance on per-bitstring frequency vs probability.
    """
    exe, qc = _compiled_circuit(transpiler, kern, bindings)
    n_qubits = qc.num_qubits
    job = exe.sample(transpiler.executor(), shots=shots)
    result = job.result()

    # Build the expected probability for each bitstring from the
    # statevector.  ``Statevector`` indexes amplitudes in Qiskit's
    # little-endian convention; we materialize the full probability
    # vector once.
    probs = expected_sv.probabilities()

    # Count empirical occurrences.  Qamomile result bits come as a
    # tuple of 0/1 in kernel-order (little-endian to match the
    # Vector indexing convention).
    empirical: dict[int, int] = {}
    for bits, count in result.results:
        # Compose the bitstring into a Qiskit-indexed integer.
        # Qiskit indexes qubit 0 as the least-significant bit.
        idx = 0
        for qi, b in enumerate(bits):
            if b:
                idx |= 1 << qi
        empirical[idx] = empirical.get(idx, 0) + count

    total = sum(empirical.values())
    assert total > 0, "sampler returned zero shots"

    # Every bitstring with non-negligible expected probability must
    # have an empirical frequency within ``rtol`` of the analytic
    # value.  Every bitstring with negligible expected probability
    # must have a low empirical frequency too — otherwise the
    # circuit is targeting the wrong qubits and producing leakage
    # outside the expected support.
    n_states = 2**n_qubits
    for state_idx in range(n_states):
        empirical_freq = empirical.get(state_idx, 0) / total
        expected_p = probs[state_idx]
        if expected_p > 1e-9:
            assert abs(empirical_freq - expected_p) < rtol, (
                f"bitstring {state_idx:0{n_qubits}b}: "
                f"empirical {empirical_freq:.4f} vs expected {expected_p:.4f}"
            )
        else:
            assert empirical_freq < rtol, (
                f"unexpected leakage at bitstring {state_idx:0{n_qubits}b}: "
                f"empirical {empirical_freq:.4f} (expected 0)"
            )


def _assert_slice_gate(
    n_qubits: int,
    slice_qubits: list[int],
    kern,
    qiskit_gate_apply,
    bindings=None,
):
    """End-to-end check: statevector + sample for a slice-broadcast gate.

    Args:
        n_qubits: Total qubits in the kernel.
        slice_qubits: Indices the slice should cover (in root order).
        kern: Slice-using qkernel under test.
        qiskit_gate_apply: Per-qubit reference gate applicator
            (``lambda qc, qi: qc.h(qi)`` style).
        bindings: Optional kernel bindings.
    """
    transpiler = QiskitTranspiler()
    actual_sv = _statevector_from_kernel(transpiler, kern, bindings)
    expected_sv = _reference_statevector(n_qubits, slice_qubits, qiskit_gate_apply)
    assert statevectors_equal(actual_sv.data, expected_sv.data), (
        f"statevector mismatch for slice gate over qubits {slice_qubits}.\n"
        f"  actual:   {actual_sv.data}\n"
        f"  expected: {expected_sv.data}"
    )
    _assert_sample_matches_statevector(transpiler, kern, expected_sv, bindings)


# ---------------------------------------------------------------------------
# Fixed single-qubit gates: H / X / Y / Z / S / T / Sdg / Tdg via slice
# ---------------------------------------------------------------------------


class TestSliceBroadcastFixedSingleQubit:
    """Each fixed single-qubit gate broadcast over a slice produces the
    same unitary as applying it per-qubit to the equivalent root indices.

    The slice pattern ``q[1::2]`` over 4 qubits hits ``{1, 3}``; the
    expected outcome therefore is "non-trivial action on q[1] and q[3],
    identity on q[0] and q[2]".  Both the statevector and the sampled
    distribution are pinned.
    """

    def test_h_via_slice(self):
        """``q[1::2] = h(q[1::2])`` on |0000> produces a uniform mixture over {1, 3}."""

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            q[1::2] = qmc.h(q[1::2])
            return qmc.measure(q)

        _assert_slice_gate(4, [1, 3], kern, lambda qc, qi: qc.h(qi))

    def test_x_via_slice(self):
        """``q[1::2] = x(q[1::2])`` on |0000> deterministically yields the bit pattern 1010 (qubit 1, 3 set)."""

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            q[1::2] = qmc.x(q[1::2])
            return qmc.measure(q)

        _assert_slice_gate(4, [1, 3], kern, lambda qc, qi: qc.x(qi))

    def test_y_via_slice(self):
        """``q[1::2] = y(q[1::2])`` produces the same probability distribution as X (up to global phase)."""

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            q[1::2] = qmc.y(q[1::2])
            return qmc.measure(q)

        _assert_slice_gate(4, [1, 3], kern, lambda qc, qi: qc.y(qi))

    def test_z_via_slice(self):
        """``q[1::2] = z(q[1::2])`` on |0000> leaves the state unchanged (Z|0> = |0>)."""

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            q[1::2] = qmc.z(q[1::2])
            return qmc.measure(q)

        _assert_slice_gate(4, [1, 3], kern, lambda qc, qi: qc.z(qi))

    def test_s_via_slice(self):
        """``q[1::2] = s(q[1::2])`` on |0000> leaves the state unchanged (S|0> = |0>)."""

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            q[1::2] = qmc.s(q[1::2])
            return qmc.measure(q)

        _assert_slice_gate(4, [1, 3], kern, lambda qc, qi: qc.s(qi))

    def test_t_via_slice(self):
        """``q[1::2] = t(q[1::2])`` on |0000> leaves the state unchanged (T|0> = |0>)."""

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            q[1::2] = qmc.t(q[1::2])
            return qmc.measure(q)

        _assert_slice_gate(4, [1, 3], kern, lambda qc, qi: qc.t(qi))

    def test_sdg_via_slice(self):
        """``q[1::2] = sdg(q[1::2])`` on |0000> leaves the state unchanged."""

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            q[1::2] = qmc.sdg(q[1::2])
            return qmc.measure(q)

        _assert_slice_gate(4, [1, 3], kern, lambda qc, qi: qc.sdg(qi))

    def test_tdg_via_slice(self):
        """``q[1::2] = tdg(q[1::2])`` on |0000> leaves the state unchanged."""

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            q[1::2] = qmc.tdg(q[1::2])
            return qmc.measure(q)

        _assert_slice_gate(4, [1, 3], kern, lambda qc, qi: qc.tdg(qi))


# ---------------------------------------------------------------------------
# Rotation / phase single-qubit gates over slice: RX / RY / RZ / P
# ---------------------------------------------------------------------------


class TestSliceBroadcastRotationGates:
    """Rotation / phase gates broadcast over a slice with varied ``theta``.

    ``theta`` is varied across a small set covering 0, π/4, π/2, π,
    3π/4 to exercise the analytic probability bands and the
    statevector elementwise.  Each parametrized case picks one
    angle; the slice covers ``{1, 3}`` of a 4-qubit register so
    even/odd qubit behaviour is also exercised in passing.

    ``theta`` is forwarded through kernel bindings so the slice
    coverage is decided by the literal ``[1::2]`` (no symbolic-bound
    interaction with the slice path).
    """

    THETAS = [0.0, math.pi / 4, math.pi / 2, math.pi, 3 * math.pi / 4]

    @pytest.mark.parametrize("theta", THETAS)
    def test_rx_via_slice(self, theta):
        """``q[1::2] = rx(q[1::2], theta)``: P(1) per touched qubit is sin²(θ/2)."""

        @qmc.qkernel
        def kern(angle: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            q[1::2] = qmc.rx(q[1::2], angle)
            return qmc.measure(q)

        _assert_slice_gate(
            4,
            [1, 3],
            kern,
            lambda qc, qi: qc.rx(theta, qi),
            bindings={"angle": theta},
        )

    @pytest.mark.parametrize("theta", THETAS)
    def test_ry_via_slice(self, theta):
        """``q[1::2] = ry(q[1::2], theta)``: P(1) per touched qubit is sin²(θ/2)."""

        @qmc.qkernel
        def kern(angle: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            q[1::2] = qmc.ry(q[1::2], angle)
            return qmc.measure(q)

        _assert_slice_gate(
            4,
            [1, 3],
            kern,
            lambda qc, qi: qc.ry(theta, qi),
            bindings={"angle": theta},
        )

    @pytest.mark.parametrize("theta", THETAS)
    def test_rz_via_slice(self, theta):
        """``q[1::2] = rz(q[1::2], theta)`` on |0000> leaves measurement probs unchanged (RZ adds a phase only)."""

        @qmc.qkernel
        def kern(angle: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            q[1::2] = qmc.rz(q[1::2], angle)
            return qmc.measure(q)

        _assert_slice_gate(
            4,
            [1, 3],
            kern,
            lambda qc, qi: qc.rz(theta, qi),
            bindings={"angle": theta},
        )

    @pytest.mark.parametrize("theta", THETAS)
    def test_p_via_slice(self, theta):
        """``q[1::2] = p(q[1::2], theta)`` on |0000> leaves measurement probs unchanged (P adds a phase only)."""

        @qmc.qkernel
        def kern(angle: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            q[1::2] = qmc.p(q[1::2], angle)
            return qmc.measure(q)

        _assert_slice_gate(
            4,
            [1, 3],
            kern,
            lambda qc, qi: qc.p(theta, qi),
            bindings={"angle": theta},
        )


# ---------------------------------------------------------------------------
# Multiple slice patterns under one gate
# ---------------------------------------------------------------------------


class TestSlicePatternsUnderH:
    """The same gate (``h``) applied through different slice patterns.

    Confirms that the slice path correctly carries different
    affine maps (start / step / stop combinations) to the right
    root qubits, not just for the ``q[1::2]`` pattern the gate
    sweep above uses.
    """

    def test_h_on_even_slice(self):
        """``q[0::2]`` on 6 qubits covers {0, 2, 4}."""

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(6, "q")
            q[0::2] = qmc.h(q[0::2])
            return qmc.measure(q)

        _assert_slice_gate(6, [0, 2, 4], kern, lambda qc, qi: qc.h(qi))

    def test_h_on_contiguous_slice(self):
        """``q[1:4]`` on 6 qubits covers {1, 2, 3}."""

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(6, "q")
            q[1:4] = qmc.h(q[1:4])
            return qmc.measure(q)

        _assert_slice_gate(6, [1, 2, 3], kern, lambda qc, qi: qc.h(qi))

    def test_h_on_step3_slice(self):
        """``q[0:6:3]`` on 6 qubits covers {0, 3}."""

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(6, "q")
            q[0:6:3] = qmc.h(q[0:6:3])
            return qmc.measure(q)

        _assert_slice_gate(6, [0, 3], kern, lambda qc, qi: qc.h(qi))

    def test_h_on_dynamic_slice(self):
        """``q[lo:hi]`` with ``bindings={lo:1, hi:5}`` covers {1, 2, 3, 4}."""

        @qmc.qkernel
        def kern(lo: qmc.UInt, hi: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(6, "q")
            view = q[lo:hi]
            for i in qmc.range(view.shape[0]):
                view[i] = qmc.h(view[i])
            q[lo:hi] = view
            return qmc.measure(q)

        _assert_slice_gate(
            6, [1, 2, 3, 4], kern, lambda qc, qi: qc.h(qi), bindings={"lo": 1, "hi": 5}
        )


# ---------------------------------------------------------------------------
# Two-qubit pairing pattern: CX between even / odd views
# ---------------------------------------------------------------------------


class TestSlicePairCX:
    """Two views paired via index loop and ``cx`` per iteration.

    The pattern ``for i in range(...): evens[i], odds[i] = cx(evens[i], odds[i])``
    is the canonical use of slices for entangling.  Verifying it
    against a Qiskit reference checks both that the per-element
    accesses through the views resolve to the right physical qubits
    and that the borrow/return cycle does not perturb the gate sequence.
    """

    def test_h_then_cx_pairing(self):
        """``q[0::2]`` Hadamards + per-element CX paired with ``q[1::2]``
        produces Bell pairs across each (even, odd) qubit pair.

        For 4 qubits this yields the state ``|Φ+>_{0,1} ⊗ |Φ+>_{2,3}``
        — measurement outcomes are concentrated on the bitstrings
        ``0000``, ``0011``, ``1100``, ``1111`` each with probability
        0.25, and 0 everywhere else.
        """

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            evens = q[0::2]
            odds = q[1::2]
            for i in qmc.range(evens.shape[0]):
                evens[i] = qmc.h(evens[i])
                evens[i], odds[i] = qmc.cx(evens[i], odds[i])
            q[0::2] = evens
            q[1::2] = odds
            return qmc.measure(q)

        def _reference(qc):
            # Apply H to evens then CX(even -> odd) for each pair.
            for ei in (0, 2):
                qc.h(ei)
            for ei, oi in ((0, 1), (2, 3)):
                qc.cx(ei, oi)

        transpiler = QiskitTranspiler()
        actual_sv = _statevector_from_kernel(transpiler, kern)

        ref_qc = QuantumCircuit(4)
        _reference(ref_qc)
        expected_sv = Statevector(ref_qc)

        assert statevectors_equal(actual_sv.data, expected_sv.data), (
            f"statevector mismatch for paired CX over views.\n"
            f"  actual:   {actual_sv.data}\n"
            f"  expected: {expected_sv.data}"
        )
        _assert_sample_matches_statevector(transpiler, kern, expected_sv)


# ---------------------------------------------------------------------------
# Sanity: random gate sequence over slice matches per-qubit reference
# ---------------------------------------------------------------------------


class TestSliceRandomMix:
    """A pseudo-random gate sequence applied through slices must match
    the equivalent per-qubit reference for any random seed.

    The kernel applies an alternating sequence of Hadamards (slice
    broadcast), per-element rotations (loop), and pair-wise CX (loop)
    across the even and odd views.  The reference rebuilds the same
    sequence with Qiskit's per-qubit API.  Parametrizing over seeds
    catches anomalies that only manifest at specific angle / qubit
    combinations.
    """

    @pytest.mark.parametrize("seed", [0, 1, 7, 42])
    def test_mixed_sequence_matches_reference(self, seed):
        """Cross-check a hand-crafted slice sequence against Qiskit reference.

        The sequence is deterministic in ``seed`` so this test is
        reproducible.  Both ``rx`` angles and ``ry`` angles are drawn
        from a uniform distribution over ``[0, 2π)``; the slice
        pattern is fixed (``q[0::2]`` / ``q[1::2]`` on n=4).
        """
        rng = np.random.default_rng(seed)
        rx_angles = [float(rng.uniform(0.0, 2 * math.pi)) for _ in range(2)]
        ry_angles = [float(rng.uniform(0.0, 2 * math.pi)) for _ in range(2)]

        @qmc.qkernel
        def kern(
            rx0: qmc.Float,
            rx1: qmc.Float,
            ry0: qmc.Float,
            ry1: qmc.Float,
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            # 1. Hadamard on the evens via slice-assign broadcast
            q[0::2] = qmc.h(q[0::2])
            # 2. RX on the odds via per-element loop, each its own angle
            odds = q[1::2]
            odds[0] = qmc.rx(odds[0], rx0)
            odds[1] = qmc.rx(odds[1], rx1)
            # 3. CX pairing evens -> odds (evens and odds cover disjoint slots
            #    so they may coexist as live views).
            evens = q[0::2]
            evens[0], odds[0] = qmc.cx(evens[0], odds[0])
            evens[1], odds[1] = qmc.cx(evens[1], odds[1])
            # 4. RY on the evens via slice-assign broadcast (one angle reused)
            evens[0] = qmc.ry(evens[0], ry0)
            evens[1] = qmc.ry(evens[1], ry1)
            # Strict-return: explicitly return both views to ``q`` before
            # ``measure(q)`` consumes the parent.
            q[0::2] = evens
            q[1::2] = odds
            return qmc.measure(q)

        ref_qc = QuantumCircuit(4)
        # Steps 1-4 in qubit-index space
        ref_qc.h(0)
        ref_qc.h(2)
        ref_qc.rx(rx_angles[0], 1)
        ref_qc.rx(rx_angles[1], 3)
        ref_qc.cx(0, 1)
        ref_qc.cx(2, 3)
        ref_qc.ry(ry_angles[0], 0)
        ref_qc.ry(ry_angles[1], 2)
        expected_sv = Statevector(ref_qc)

        transpiler = QiskitTranspiler()
        bindings = {
            "rx0": rx_angles[0],
            "rx1": rx_angles[1],
            "ry0": ry_angles[0],
            "ry1": ry_angles[1],
        }
        actual_sv = _statevector_from_kernel(transpiler, kern, bindings)

        assert statevectors_equal(actual_sv.data, expected_sv.data), (
            f"statevector mismatch for mixed sequence (seed={seed}).\n"
            f"  rx angles: {rx_angles}\n"
            f"  ry angles: {ry_angles}\n"
            f"  actual:    {actual_sv.data}\n"
            f"  expected:  {expected_sv.data}"
        )
        _assert_sample_matches_statevector(transpiler, kern, expected_sv, bindings)


# ---------------------------------------------------------------------------
# Property-based randomized slice tests across Qiskit / QuriParts / CUDA-Q
# ---------------------------------------------------------------------------
#
# These tests exercise three different slice patterns
# (``q[lo:hi:step]`` / ``q[lo:hi]`` / ``q[lo::step]``) over a randomly
# chosen qubit register size and a randomly chosen sequence of
# single-qubit gates applied inside ``for i in qmc.range(view.shape[0])``.
# For each random instance the same kernel is transpiled with all three
# supported SDK backends and the resulting statevector is compared against
# the theoretical statevector built directly with Qiskit.  The slice
# bounds are passed through ``bindings`` so every random case also
# exercises the symbolic-bounds → bindings-resolved path.


# Fixed single-qubit gates (no angle).  Mapped to the Qiskit attribute
# name in ``_qiskit_apply``.
_FIXED_GATES = ("h", "x", "y", "z", "s", "t", "sdg", "tdg")

# Single-qubit gates that take a real angle.
_ROTATION_GATES = ("rx", "ry", "rz", "p")


def _make_kern_from_source(src: str):
    """Compile, register in linecache, exec a kernel source string.

    ``@qmc.qkernel`` calls ``inspect.getsource`` on the decorated
    function, which in turn reads from ``linecache``.  Registering
    the source under a unique stable filename makes a dynamically
    constructed kernel introspectable just like a normal source-file
    function — which is required for the trace pass to recover the
    return statement and parameter annotations.

    Args:
        src (str): Full ``@qmc.qkernel`` function source, including the
            decorator.  The function must be named ``_kern``.

    Returns:
        QKernel: The compiled kernel object ready to be passed to a
            transpiler.
    """
    digest = hashlib.md5(src.encode()).hexdigest()
    fname = f"<dynamic_kern_{digest}>"
    linecache.cache[fname] = (
        len(src),
        None,
        [line + "\n" for line in src.splitlines()],
        fname,
    )
    code = compile(src, fname, "exec")
    namespace: dict = {"qmc": qmc}
    exec(code, namespace)
    return namespace["_kern"]


def _build_gate_sequence(rng: np.random.Generator, n_gates: int):
    """Pick ``n_gates`` random single-qubit gates with random angles.

    Args:
        rng (np.random.Generator): Seeded RNG for reproducibility.
        n_gates (int): Number of gates in the inner sequence
            (1 ≤ ``n_gates`` ≤ 4 in practice).

    Returns:
        list[tuple[str, float | None]]: Sequence of ``(gate_name,
            angle)`` pairs.  ``angle`` is ``None`` for fixed gates and a
            float in ``[0, 2π)`` for rotation/phase gates.
    """
    all_gates = _FIXED_GATES + _ROTATION_GATES
    seq: list[tuple[str, float | None]] = []
    for _ in range(n_gates):
        name = str(rng.choice(all_gates))
        if name in _ROTATION_GATES:
            seq.append((name, float(rng.uniform(0.0, 2 * math.pi))))
        else:
            seq.append((name, None))
    return seq


def _qiskit_apply(qc: QuantumCircuit, name: str, angle: float | None, qi: int) -> None:
    """Apply a single gate to qubit ``qi`` on ``qc`` using its Qiskit API.

    Args:
        qc (QuantumCircuit): Target circuit.
        name (str): Gate name (``h`` / ``x`` / ... / ``rx`` / ``ry`` /
            ``rz`` / ``p``).
        angle (float | None): Angle for rotation/phase gates, ``None``
            for fixed gates.
        qi (int): Qubit index.
    """
    if angle is None:
        getattr(qc, name)(qi)
    else:
        getattr(qc, name)(angle, qi)


def _build_kernel_source(
    n_qubits: int,
    slice_expr: str,
    gate_seq: list[tuple[str, float | None]],
    bind_params: list[str],
) -> str:
    """Construct a qkernel source for the property test.

    The kernel takes the slice bounds (``lo`` / ``hi`` / ``step`` —
    whichever subset the variant uses) and one ``Float`` parameter per
    rotation gate as ``bindings``-resolved arguments.  Inside, it takes
    ``view = q[<slice_expr>]`` and runs a single ``for i in
    qmc.range(view.shape[0])`` loop body that applies the chosen gates
    to ``view[i]`` in sequence.

    Args:
        n_qubits (int): Concrete qubit register size; baked as a
            literal in the kernel source.
        slice_expr (str): The slice notation used inside the kernel,
            e.g. ``"lo:hi:step"`` / ``"lo:hi"`` / ``"lo::step"``.
        gate_seq (list[tuple[str, float | None]]): Gate sequence to
            apply per loop iteration.  Each angle is passed via a
            separate ``theta{idx}`` binding.
        bind_params (list[str]): Names of the integer-typed slice-
            bound parameters (``lo`` / ``hi`` / ``step``) that should
            appear in the kernel signature.  The angle parameters are
            appended automatically.

    Returns:
        str: Full kernel source ready for ``_make_kern_from_source``.
    """
    angle_params: list[str] = []
    body_lines: list[str] = []
    angle_idx = 0
    for name, angle in gate_seq:
        if angle is None:
            body_lines.append(f"        view[i] = qmc.{name}(view[i])")
        else:
            param = f"theta{angle_idx}"
            angle_params.append(param)
            body_lines.append(f"        view[i] = qmc.{name}(view[i], {param})")
            angle_idx += 1

    sig_parts = [f"{p}: qmc.UInt" for p in bind_params]
    sig_parts += [f"{p}: qmc.Float" for p in angle_params]
    sig = ", ".join(sig_parts)

    body = "\n".join(body_lines) if body_lines else "        pass"

    return (
        f"@qmc.qkernel\n"
        f"def _kern({sig}) -> qmc.Vector[qmc.Bit]:\n"
        f'    q = qmc.qubit_array({n_qubits}, "q")\n'
        f"    view = q[{slice_expr}]\n"
        f"    for i in qmc.range(view.shape[0]):\n"
        f"{body}\n"
        f"    q[{slice_expr}] = view\n"
        f"    return qmc.measure(q)\n"
    )


def _theoretical_statevector(
    n_qubits: int,
    slice_qubits: list[int],
    gate_seq: list[tuple[str, float | None]],
) -> Statevector:
    """Build the reference statevector via Qiskit per-qubit gate application.

    For each qubit in ``slice_qubits`` (in the order the slice walks
    them), apply the gate sequence to that qubit on a fresh Qiskit
    circuit.  Since all gates in the sequence are single-qubit and the
    qubits are distinct, the order of the outer loop does not affect
    the resulting statevector — but we follow the kernel's loop order
    to keep the reference structurally analogous.

    Args:
        n_qubits (int): Total qubits in the register.
        slice_qubits (list[int]): Root-space indices the slice covers.
        gate_seq (list[tuple[str, float | None]]): Gate sequence to
            apply per qubit.

    Returns:
        Statevector: Reference statevector after running the gate
            sequence over the slice qubits.
    """
    qc = QuantumCircuit(n_qubits)
    for qi in slice_qubits:
        for name, angle in gate_seq:
            _qiskit_apply(qc, name, angle, qi)
    return Statevector(qc)


def _qiskit_statevector(kern, bindings: dict) -> np.ndarray:
    """Transpile and simulate on Qiskit, returning the unitary statevector.

    Args:
        kern: Compiled qkernel from ``_make_kern_from_source``.
        bindings (dict): Bindings forwarded to ``transpile``.

    Returns:
        np.ndarray: Complex amplitudes of the unitary core (measurement
            instructions are stripped before extraction).
    """
    transpiler = QiskitTranspiler()
    exe = transpiler.transpile(kern, bindings=bindings)
    qc = exe.compiled_quantum[0].circuit
    qc_no_meas = qc.remove_final_measurements(inplace=False)
    return np.array(Statevector(qc_no_meas).data)


def _quri_parts_statevector(kern, bindings: dict) -> np.ndarray:
    """Transpile with QuriParts and simulate on Qulacs.

    QuriParts ``measurement_mode`` is ``STATIC`` (see
    ``qamomile.quri_parts.emitter``), so the compiled circuit contains
    only unitary gates — no extra stripping is required.

    Args:
        kern: Compiled qkernel.
        bindings (dict): Bindings forwarded to ``transpile``.

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
    """Transpile with CUDA-Q and read the simulator statevector.

    CUDA-Q ``measurement_mode`` defaults to ``STATIC`` (see
    ``qamomile.cudaq.emitter``), so ``cudaq.get_state`` returns the
    pre-measurement amplitudes directly.

    Args:
        kern: Compiled qkernel.
        bindings (dict): Bindings forwarded to ``transpile``.

    Returns:
        np.ndarray: Complex amplitudes from the CUDA-Q simulator.
    """
    pytest.importorskip("cudaq")
    import cudaq

    from qamomile.cudaq import CudaqTranspiler

    transpiler = CudaqTranspiler()
    exe = transpiler.transpile(kern, bindings=bindings)
    cudaq_circuit = exe.compiled_quantum[0].circuit
    return np.array(cudaq.get_state(cudaq_circuit.kernel_func))


def _random_slice_lo_hi_step(rng: np.random.Generator):
    """Sample ``(n_qubits, lo, hi, step)`` with at least 1 covered qubit.

    Args:
        rng (np.random.Generator): RNG for reproducibility.

    Returns:
        tuple[int, int, int, int]: ``(n_qubits, lo, hi, step)`` such
            that ``list(range(lo, hi, step))`` is non-empty.
    """
    n_qubits = int(rng.integers(4, 11))  # [4, 10]
    step = int(rng.integers(1, 4))  # [1, 3]
    lo = int(rng.integers(0, n_qubits))
    # ``hi`` must be at least ``lo + step`` so that the first iteration
    # of ``range(lo, hi, step)`` produces a valid index.
    hi_low = min(lo + step, n_qubits)
    hi = int(rng.integers(hi_low, n_qubits + 1))
    return n_qubits, lo, hi, step


def _random_slice_lo_hi(rng: np.random.Generator):
    """Sample ``(n_qubits, lo, hi)`` with step=1, at least 1 element.

    Args:
        rng (np.random.Generator): RNG for reproducibility.

    Returns:
        tuple[int, int, int]: ``(n_qubits, lo, hi)`` such that
            ``hi > lo``.
    """
    n_qubits = int(rng.integers(4, 11))
    lo = int(rng.integers(0, n_qubits))
    hi = int(rng.integers(lo + 1, n_qubits + 1))
    return n_qubits, lo, hi


def _random_slice_lo_step(rng: np.random.Generator):
    """Sample ``(n_qubits, lo, step)`` for ``q[lo::step]``, ≥ 1 element.

    Args:
        rng (np.random.Generator): RNG for reproducibility.

    Returns:
        tuple[int, int, int]: ``(n_qubits, lo, step)`` such that
            ``lo < n_qubits`` (so ``range(lo, n_qubits, step)`` is
            non-empty).
    """
    n_qubits = int(rng.integers(4, 11))
    step = int(rng.integers(1, 4))
    lo = int(rng.integers(0, n_qubits))  # 0 ≤ lo < n_qubits
    return n_qubits, lo, step


_PROPERTY_SEEDS = [0, 1, 7, 11, 42, 123, 999, 2024]


def _run_property_case(
    kern,
    bindings: dict,
    expected_sv: Statevector,
    case_label: str,
):
    """Run the kernel on every available backend and assert SV match.

    The Qiskit backend is mandatory; QuriParts and CUDA-Q are guarded
    by ``pytest.importorskip`` inside their helpers and skip cleanly
    when the corresponding SDK is not installed.

    Args:
        kern: Compiled qkernel.
        bindings (dict): Bindings to pass to every backend's
            ``transpile``.
        expected_sv (Statevector): Reference statevector.
        case_label (str): Human-readable label describing the random
            case (e.g. the slice expression and gate sequence) used in
            assertion failure messages.
    """
    expected_data = np.array(expected_sv.data)

    actual_qiskit = _qiskit_statevector(kern, bindings)
    assert statevectors_equal(actual_qiskit, expected_data), (
        f"[qiskit] statevector mismatch for {case_label}.\n"
        f"  actual:   {actual_qiskit}\n"
        f"  expected: {expected_data}"
    )

    actual_quri_parts = _quri_parts_statevector(kern, bindings)
    assert statevectors_equal(actual_quri_parts, expected_data), (
        f"[quri_parts] statevector mismatch for {case_label}.\n"
        f"  actual:   {actual_quri_parts}\n"
        f"  expected: {expected_data}"
    )

    actual_cudaq = _cudaq_statevector(kern, bindings)
    assert statevectors_equal(actual_cudaq, expected_data), (
        f"[cudaq] statevector mismatch for {case_label}.\n"
        f"  actual:   {actual_cudaq}\n"
        f"  expected: {expected_data}"
    )


class TestPropertyBasedSliceCrossBackend:
    """Property-style: random register size, random slice, random gates
    must produce the same statevector on Qiskit, QuriParts, and CUDA-Q
    as the per-qubit Qiskit reference.

    The seed drives:
      1. ``n_qubits`` in ``[4, 10]``.
      2. Slice bounds (variant-dependent; see each test).
      3. A 1–4 element sequence of single-qubit gates with random
         angles for rotation/phase gates.

    The slice bounds are passed via ``bindings`` so every case also
    exercises the symbolic-bounds → ``ConstantFoldingPass`` →
    ``SliceBorrowCheckPass`` pipeline rather than fully-concrete
    literal slicing.
    """

    @pytest.mark.parametrize("seed", _PROPERTY_SEEDS)
    def test_random_slice_lo_hi_step(self, seed: int):
        """``q[lo:hi:step]`` with random ``(n_qubits, lo, hi, step)``.

        Exercises the most general slice form: all three of ``start``,
        ``stop``, and ``step`` are non-trivial and bindings-resolved.
        """
        rng = np.random.default_rng(seed)
        n_qubits, lo, hi, step = _random_slice_lo_hi_step(rng)
        slice_qubits = list(range(lo, hi, step))
        assert slice_qubits, "sampler must yield a non-empty slice"

        n_gates = int(rng.integers(1, 5))  # 1..4 gates
        gate_seq = _build_gate_sequence(rng, n_gates)

        src = _build_kernel_source(
            n_qubits=n_qubits,
            slice_expr="lo:hi:step",
            gate_seq=gate_seq,
            bind_params=["lo", "hi", "step"],
        )
        kern = _make_kern_from_source(src)

        bindings: dict = {"lo": lo, "hi": hi, "step": step}
        for idx, (_, angle) in enumerate((n, a) for n, a in gate_seq if a is not None):
            bindings[f"theta{idx}"] = angle

        expected_sv = _theoretical_statevector(n_qubits, slice_qubits, gate_seq)

        _run_property_case(
            kern,
            bindings,
            expected_sv,
            case_label=(
                f"n={n_qubits}, q[{lo}:{hi}:{step}] (covers {slice_qubits}), "
                f"gates={gate_seq}, seed={seed}"
            ),
        )

    @pytest.mark.parametrize("seed", _PROPERTY_SEEDS)
    def test_random_slice_lo_hi(self, seed: int):
        """``q[lo:hi]`` (step omitted → 1) with random ``(n_qubits, lo, hi)``.

        Exercises the no-step form to ensure the default-step branch
        of the slice normaliser matches the Qiskit reference.
        """
        rng = np.random.default_rng(seed)
        n_qubits, lo, hi = _random_slice_lo_hi(rng)
        slice_qubits = list(range(lo, hi))
        assert slice_qubits, "sampler must yield a non-empty slice"

        n_gates = int(rng.integers(1, 5))
        gate_seq = _build_gate_sequence(rng, n_gates)

        src = _build_kernel_source(
            n_qubits=n_qubits,
            slice_expr="lo:hi",
            gate_seq=gate_seq,
            bind_params=["lo", "hi"],
        )
        kern = _make_kern_from_source(src)

        bindings: dict = {"lo": lo, "hi": hi}
        for idx, (_, angle) in enumerate((n, a) for n, a in gate_seq if a is not None):
            bindings[f"theta{idx}"] = angle

        expected_sv = _theoretical_statevector(n_qubits, slice_qubits, gate_seq)

        _run_property_case(
            kern,
            bindings,
            expected_sv,
            case_label=(
                f"n={n_qubits}, q[{lo}:{hi}] (covers {slice_qubits}), "
                f"gates={gate_seq}, seed={seed}"
            ),
        )

    @pytest.mark.parametrize("seed", _PROPERTY_SEEDS)
    def test_random_slice_lo_step(self, seed: int):
        """``q[lo::step]`` (no ``hi`` → end of parent) with random
        ``(n_qubits, lo, step)``.

        Exercises the open-ended-stop branch: the normaliser fills the
        missing ``stop`` from the parent length, and the test confirms
        the resulting coverage matches ``range(lo, n_qubits, step)``.
        """
        rng = np.random.default_rng(seed)
        n_qubits, lo, step = _random_slice_lo_step(rng)
        slice_qubits = list(range(lo, n_qubits, step))
        assert slice_qubits, "sampler must yield a non-empty slice"

        n_gates = int(rng.integers(1, 5))
        gate_seq = _build_gate_sequence(rng, n_gates)

        src = _build_kernel_source(
            n_qubits=n_qubits,
            slice_expr="lo::step",
            gate_seq=gate_seq,
            bind_params=["lo", "step"],
        )
        kern = _make_kern_from_source(src)

        bindings: dict = {"lo": lo, "step": step}
        for idx, (_, angle) in enumerate((n, a) for n, a in gate_seq if a is not None):
            bindings[f"theta{idx}"] = angle

        expected_sv = _theoretical_statevector(n_qubits, slice_qubits, gate_seq)

        _run_property_case(
            kern,
            bindings,
            expected_sv,
            case_label=(
                f"n={n_qubits}, q[{lo}::{step}] (covers {slice_qubits}), "
                f"gates={gate_seq}, seed={seed}"
            ),
        )
