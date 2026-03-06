"""Tests for QPE: naive implementation vs built-in qmc.qpe()."""

import math
import random

import pytest

import qamomile.circuit as qmc


def _decode_phase(bits: list) -> float:
    """Decode measurement bits into a phase estimate."""
    return sum(bit * (1 / (2 ** (i + 1))) for i, bit in enumerate(reversed(bits)))


@pytest.fixture
def qiskit_transpiler():
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    return QiskitTranspiler()


# -- Helper: naive QPE -------------------------------------------------------


@qmc.qkernel
def _iqft(qubits: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    n = qubits.shape[0]
    for j in qmc.range(n // 2):
        qubits[j], qubits[n - j - 1] = qmc.swap(qubits[j], qubits[n - j - 1])
    for j in qmc.range(n):
        for k in qmc.range(j):
            angle = -math.pi / (2 ** (j - k))
            qubits[j], qubits[k] = qmc.cp(qubits[j], qubits[k], angle)
        qubits[j] = qmc.h(qubits[j])
    return qubits


@qmc.qkernel
def _phase_gate(q: qmc.Qubit, theta: float, iter: int) -> qmc.Qubit:
    for _ in qmc.range(iter):
        q = qmc.p(q, theta)
    return q


@qmc.qkernel
def naive_qpe(n: int, phase: float) -> qmc.Vector[qmc.Bit]:
    phase_register = qmc.qubit_array(n, name="phase_reg")
    target = qmc.qubit(name="target")
    target = qmc.x(target)
    controlled_phase_gate = qmc.controlled(_phase_gate)
    num = phase_register.shape[0]
    for i in qmc.range(num):
        phase_register[i] = qmc.h(phase_register[i])
    for i in qmc.range(num):
        phase_register[i], target = controlled_phase_gate(
            phase_register[i], target, theta=phase, iter=2**i
        )
    phase_register = _iqft(phase_register)
    bits = qmc.measure(phase_register)
    return bits


# -- Helper: built-in QPE ----------------------------------------------------


@qmc.qkernel
def _p_gate(q: qmc.Qubit, theta: float) -> qmc.Qubit:
    return qmc.p(q, theta)


@qmc.qkernel
def builtin_qpe(n: int, phase: float) -> qmc.Float:
    q_phase = qmc.qubit_array(n, name="phase_reg")
    target = qmc.qubit(name="target")
    target = qmc.x(target)
    phase_q: qmc.QFixed = qmc.qpe(target, q_phase, _p_gate, theta=phase)
    return qmc.measure(phase_q)


# -- Tests --------------------------------------------------------------------


class TestQPENaive:
    """Naive QPE (manual IQFT + iter parameter) tests."""

    def test_naive_qpe_pi_over_2(self, qiskit_transpiler):
        """theta=pi/2 -> phase=0.25."""
        executable = qiskit_transpiler.transpile(
            naive_qpe, bindings={"n": 3, "phase": math.pi / 2}
        )
        job = executable.sample(qiskit_transpiler.executor(), shots=1024)
        result = job.result()

        for bits, count in result.results:
            phase_estimate = _decode_phase(bits)
            assert phase_estimate == pytest.approx(0.25), (
                f"Naive QPE: expected 0.25, got {phase_estimate} "
                f"(bits={bits}, count={count})"
            )

    def test_naive_qpe_pi_over_4(self, qiskit_transpiler):
        """theta=pi/4 -> phase=0.125."""
        executable = qiskit_transpiler.transpile(
            naive_qpe, bindings={"n": 3, "phase": math.pi / 4}
        )
        job = executable.sample(qiskit_transpiler.executor(), shots=1024)
        result = job.result()

        for bits, count in result.results:
            phase_estimate = _decode_phase(bits)
            assert phase_estimate == pytest.approx(0.125), (
                f"Naive QPE: expected 0.125, got {phase_estimate}"
            )


class TestQPEBuiltin:
    """Built-in QPE (qmc.qpe()) tests."""

    def test_builtin_qpe_pi_over_2(self, qiskit_transpiler):
        """theta=pi/2 -> phase=0.25."""
        executable = qiskit_transpiler.transpile(
            builtin_qpe, bindings={"n": 3, "phase": math.pi / 2}
        )
        job = executable.sample(qiskit_transpiler.executor(), shots=1024)
        result = job.result()

        for value, count in result.results:
            assert value == pytest.approx(0.25), (
                f"Built-in QPE: expected 0.25, got {value} (count={count})"
            )

    def test_builtin_qpe_pi_over_4(self, qiskit_transpiler):
        """theta=pi/4 -> phase=0.125."""
        executable = qiskit_transpiler.transpile(
            builtin_qpe, bindings={"n": 3, "phase": math.pi / 4}
        )
        job = executable.sample(qiskit_transpiler.executor(), shots=1024)
        result = job.result()

        for value, count in result.results:
            assert value == pytest.approx(0.125), (
                f"Built-in QPE: expected 0.125, got {value} (count={count})"
            )


class TestQPEConsistency:
    """Verify naive QPE and built-in QPE return the same phase estimates."""

    @pytest.mark.parametrize(
        "theta,expected_phase",
        [
            (math.pi / 2, 0.25),
            (math.pi / 4, 0.125),
            (math.pi, 0.5),
        ],
    )
    def test_naive_and_builtin_agree(self, qiskit_transpiler, theta, expected_phase):
        """Both implementations estimate the same phase."""
        # Naive QPE
        naive_exec = qiskit_transpiler.transpile(
            naive_qpe, bindings={"n": 3, "phase": theta}
        )
        naive_job = naive_exec.sample(qiskit_transpiler.executor(), shots=1024)
        naive_result = naive_job.result()
        naive_phases = set()
        for bits, count in naive_result.results:
            phase_est = _decode_phase(bits)
            naive_phases.add(round(phase_est, 6))

        # Built-in QPE
        builtin_exec = qiskit_transpiler.transpile(
            builtin_qpe, bindings={"n": 3, "phase": theta}
        )
        builtin_job = builtin_exec.sample(qiskit_transpiler.executor(), shots=1024)
        builtin_result = builtin_job.result()
        builtin_phases = set()
        for value, count in builtin_result.results:
            builtin_phases.add(round(value, 6))

        # Both return the same phase set
        assert naive_phases == builtin_phases, (
            f"theta={theta}: naive got {naive_phases}, built-in got {builtin_phases}"
        )

        # Expected phase is in the results
        assert expected_phase in naive_phases, (
            f"Expected phase {expected_phase} not in naive results"
        )

    @pytest.mark.parametrize("n_qubits", [3, 5, 7])
    @pytest.mark.parametrize("seed", [901 + i for i in range(100)])
    def test_random_angle_consistency(self, qiskit_transpiler, seed, n_qubits):
        r"""Both QPE implementations return a theoretically valid phase for random angles.

        Mathematical background
        -----------------------
        For a unitary U with eigenvalue e^{2\pi i \phi}, an n-qubit QPE measures
        an integer m \in \{0, ..., 2^n - 1\} with probability:

            P(m) = \frac{\sin^2(2^n \pi (\phi - m/2^n))}
                        {2^n \sin^2(\pi (\phi - m/2^n))}

        The two highest-probability outcomes are the two integers nearest to
        \phi \cdot 2^n, i.e., m_0 = floor(\phi \cdot 2^n) and
        m_1 = ceil(\phi \cdot 2^n). Their combined probability satisfies:

            P(m_0) + P(m_1) >= 8 / \pi^2 \approx 0.81

        even in the worst case (equidistant: \phi \cdot 2^n is a half-integer).
        For non-equidistant phases the dominant outcome is even more concentrated.

        Two-tier verification
        ---------------------
        We define EQUIDISTANT_THRESHOLD = 0.1. Let frac = (\phi \cdot 2^n) mod 1.

        - Non-equidistant (|frac - 0.5| >= 0.1):
          The dominant phase m_0/2^n has probability significantly higher than
          the runner-up m_1/2^n. Specifically, when |frac - 0.5| >= 0.1 the
          gap P(m_0) - P(m_1) is large enough that with 4096 shots the top
          result is deterministic. We assert:
            (a) naive_top == builtin_top  (exact match)
            (b) naive_top in valid_phases (accuracy)

        - Equidistant (|frac - 0.5| < 0.1):
          P(m_0) and P(m_1) are nearly equal, so sampling noise can flip the
          top result between the two implementations. We only assert:
            (a) naive_top in valid_phases
            (b) builtin_top in valid_phases
          Both m_0/2^n and m_1/2^n are correct QPE outputs.

        Bug detection
        -------------
        When the bug is present (power not resolved), the built-in QPE applies
        U^1 on every counting qubit instead of U^{2^k}, producing a phase
        unrelated to the correct \phi. This fails both tier checks for the
        vast majority of random angles.
        """
        EQUIDISTANT_THRESHOLD = 0.1

        rng = random.Random(seed)
        theta = rng.uniform(0, 2 * math.pi)
        shots = 4096

        # True phase and the two nearest representable phases
        phi = (theta / (2 * math.pi)) % 1.0
        phi_scaled = phi * (2**n_qubits)
        lower = (math.floor(phi_scaled) % (2**n_qubits)) / (2**n_qubits)
        upper = (math.ceil(phi_scaled) % (2**n_qubits)) / (2**n_qubits)
        valid_phases = {round(lower, 6), round(upper, 6)}

        # Equidistant detection
        frac = phi_scaled % 1.0
        is_equidistant = abs(frac - 0.5) < EQUIDISTANT_THRESHOLD

        # Naive QPE
        naive_exec = qiskit_transpiler.transpile(
            naive_qpe, bindings={"n": n_qubits, "phase": theta}
        )
        naive_job = naive_exec.sample(qiskit_transpiler.executor(), shots=shots)
        naive_result = naive_job.result()
        naive_phase_counts = {}
        for bits, count in naive_result.results:
            phase_est = _decode_phase(bits)
            naive_phase_counts[round(phase_est, 6)] = count
        naive_top = max(naive_phase_counts, key=naive_phase_counts.get)

        # Built-in QPE
        builtin_exec = qiskit_transpiler.transpile(
            builtin_qpe, bindings={"n": n_qubits, "phase": theta}
        )
        builtin_job = builtin_exec.sample(qiskit_transpiler.executor(), shots=shots)
        builtin_result = builtin_job.result()
        builtin_phase_counts = {}
        for value, count in builtin_result.results:
            builtin_phase_counts[round(value, 6)] = count
        builtin_top = max(builtin_phase_counts, key=builtin_phase_counts.get)

        if is_equidistant:
            # Equidistant: both must be valid QPE results, may differ
            assert round(naive_top, 6) in valid_phases, (
                f"[equidistant] seed={seed}, n={n_qubits}, "
                f"theta={theta:.6f}, phi={phi:.6f}: "
                f"naive top={naive_top} not in valid {valid_phases}"
            )
            assert round(builtin_top, 6) in valid_phases, (
                f"[equidistant] seed={seed}, n={n_qubits}, "
                f"theta={theta:.6f}, phi={phi:.6f}: "
                f"builtin top={builtin_top} not in valid {valid_phases}"
            )
        else:
            # Non-equidistant: exact match required
            assert naive_top == pytest.approx(builtin_top, abs=1e-6), (
                f"[non-equidistant] seed={seed}, n={n_qubits}, "
                f"theta={theta:.6f}, phi={phi:.6f}: "
                f"naive top={naive_top} != builtin top={builtin_top}"
            )
            assert round(naive_top, 6) in valid_phases, (
                f"[non-equidistant] seed={seed}, n={n_qubits}, "
                f"theta={theta:.6f}, phi={phi:.6f}: "
                f"top={naive_top} not in valid {valid_phases}"
            )
