"""Tests for the FinITE converter."""

from __future__ import annotations

import math

import numpy as np
import ommx.v1
import pytest

import qamomile.observable as qm_o
from qamomile.circuit.transpiler.job import SampleResult
from qamomile.optimization.binary_model import BinaryModel, VarType
from qamomile.optimization.finite_ite import (
    FinITEConverter,
    finite_ite_beta_threshold,
)

# The 3-qubit instance from the FinITE prototype notebook: W = 6, E0 = -4
# (non-degenerate, bitstring 101), gamma0 = 1/8, gap = 2.
NOTEBOOK_TERMS = {
    (0,): 1.0,
    (1,): -1.0,
    (2,): 1.0,
    (0, 1): -1.0,
    (1, 2): 1.0,
    (0, 2): -1.0,
}
NOTEBOOK_BETA = 1.22


def _notebook_converter() -> FinITEConverter:
    """Build the converter for the prototype notebook's Ising instance."""
    return FinITEConverter(BinaryModel.from_higher_ising(dict(NOTEBOOK_TERMS)))


# ---------------------------------------------------------------------------
# Classical references. These are exponential in the qubit count and exist
# only to check the converter, which is why they live here and not in the
# library.
# ---------------------------------------------------------------------------


def _energies(terms: dict[tuple[int, ...], float], num_qubits: int) -> np.ndarray:
    """Return the diagonal energies, qubit 0 being the least significant bit."""
    return np.array(
        [
            sum(
                coefficient * (-1.0 if sum((basis >> q) & 1 for q in word) & 1 else 1.0)
                for word, coefficient in terms.items()
            )
            for basis in range(1 << num_qubits)
        ]
    )


def _spectrum(
    terms: dict[tuple[int, ...], float],
    num_qubits: int,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Return ``(energies, ground_mask, E0, gamma0, gap)`` for the uniform state."""
    energies = _energies(terms, num_qubits)
    ground_energy = float(energies.min())
    ground_mask = np.isclose(energies, ground_energy)
    excited = energies[~ground_mask]
    gap = float(excited.min() - ground_energy) if excited.size else math.inf
    overlap = float(ground_mask.sum()) / 2**num_qubits
    return energies, ground_mask, ground_energy, overlap, gap


def _predicted_metrics(
    terms: dict[tuple[int, ...], float],
    beta: float,
    num_qubits: int,
) -> tuple[float, float]:
    """Return Proposition 1's closed-form ``(P_LCU, F_g)`` for ``|+>^n``.

    Eqs. 14-15 of arXiv:2604.27482. Energies are shifted by E0 before
    exponentiating so large beta cannot overflow; the common factor cancels
    out of the fidelity ratio.
    """
    energies, ground_mask, ground_energy, _, _ = _spectrum(terms, num_qubits)
    weight_norm = sum(abs(value) for value in terms.values())
    shifted = np.exp(-2.0 * beta * (energies - ground_energy))
    weight_sum = float(shifted.sum()) / 2**num_qubits
    ground_weight = float(shifted[ground_mask].sum()) / 2**num_qubits
    success = math.exp(-2.0 * beta * (weight_norm + ground_energy)) * weight_sum
    return success, ground_weight / weight_sum


def _postselected_amplitudes(
    converter: FinITEConverter,
    beta: float,
) -> np.ndarray:
    """Return the all-zero-ancilla system amplitudes by exact simulation."""
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Statevector

    from qamomile.qiskit import QiskitTranspiler

    circuit = converter.transpile(
        QiskitTranspiler(), beta=beta
    ).quantum_circuit.remove_final_measurements(inplace=False)
    statevector = np.asarray(Statevector.from_instruction(circuit).data)
    # `signal` is allocated first, so it occupies the low wires.
    return statevector[
        [
            basis << converter.num_ancilla_bits
            for basis in range(1 << converter.spin_model.num_bits)
        ]
    ]


def _simulated_metrics(
    converter: FinITEConverter,
    terms: dict[tuple[int, ...], float],
    beta: float,
) -> tuple[float, float]:
    """Return ``(P_LCU, F_g)`` read off an exact statevector simulation."""
    block = _postselected_amplitudes(converter, beta)
    success = float(np.vdot(block, block).real)
    normalized = block / math.sqrt(success)
    _, ground_mask, _, _, _ = _spectrum(terms, converter.spin_model.num_bits)
    return success, float((np.abs(normalized[ground_mask]) ** 2).sum())


# ---------------------------------------------------------------------------
# Construction and problem intake
# ---------------------------------------------------------------------------


def test_converter_derives_weight_norm_and_ancilla_count() -> None:
    """W is the one-norm of the identity-free terms, one ancilla each."""
    converter = _notebook_converter()

    assert converter.weight_norm == pytest.approx(6.0)
    assert converter.num_ancilla_bits == 6


def test_constant_term_is_left_out_of_the_block_encoding() -> None:
    """The model constant does not inflate W or add an ancilla."""
    converter = FinITEConverter(
        BinaryModel.from_higher_ising({(0,): 1.0, (0, 1): -1.0}, constant=3.5)
    )

    assert converter.spin_model.constant == pytest.approx(3.5)
    assert converter.weight_norm == pytest.approx(2.0)
    assert converter.num_ancilla_bits == 2


def test_converter_rejects_a_constant_only_model() -> None:
    """A model with no Pauli term has nothing to block-encode."""
    with pytest.raises(ValueError, match="at least one non-identity Pauli term"):
        FinITEConverter(BinaryModel.from_higher_ising({}, constant=1.0))


def test_converter_accepts_higher_order_terms_without_quadratization() -> None:
    """Cubic HUBO terms get their own ancilla, with no auxiliary variables."""
    converter = FinITEConverter(
        BinaryModel.from_higher_ising({(0, 1, 2): 1.0, (0,): -2.0})
    )

    assert converter.spin_model.num_bits == 3
    assert converter.num_ancilla_bits == 2
    assert converter.weight_norm == pytest.approx(3.0)


def test_converter_accepts_an_ommx_instance_without_mutating_it() -> None:
    """OMMX intake deep-copies, as the shared normalize_problem_input promises."""
    x0 = ommx.v1.DecisionVariable.binary(0, name="x0")
    x1 = ommx.v1.DecisionVariable.binary(1, name="x1")
    instance = ommx.v1.Instance.from_components(
        decision_variables=[x0, x1],
        objective=x0 * x1 - x0,
        constraints=[],
        sense=ommx.v1.Instance.MINIMIZE,
    )
    snapshot = instance.to_bytes()

    converter = FinITEConverter(instance)

    assert instance.to_bytes() == snapshot
    assert converter.original_vartype == VarType.BINARY
    assert converter.num_ancilla_bits >= 1


def test_cost_hamiltonian_matches_the_encoded_terms() -> None:
    """The exposed Hamiltonian carries exactly the block-encoded Pauli terms."""
    converter = _notebook_converter()
    hamiltonian = converter.get_cost_hamiltonian()

    assert hamiltonian.constant == pytest.approx(0.0)
    rebuilt = {
        tuple(sorted(operator.index for operator in operators)): coefficient
        for operators, coefficient in hamiltonian.terms.items()
    }
    assert rebuilt == pytest.approx(NOTEBOOK_TERMS)
    assert all(
        operator.pauli == qm_o.Pauli.Z
        for operators in hamiltonian.terms
        for operator in operators
    )


def test_cost_hamiltonian_carries_the_model_constant() -> None:
    """The held-out constant surfaces on the Hamiltonian, not as a Pauli term."""
    converter = FinITEConverter(
        BinaryModel.from_higher_ising({(0,): 1.0}, constant=-2.5)
    )
    hamiltonian = converter.get_cost_hamiltonian()

    assert hamiltonian.constant == pytest.approx(-2.5)
    assert len(hamiltonian.terms) == 1


# ---------------------------------------------------------------------------
# Block encoding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("beta", [0.0, 0.3, NOTEBOOK_BETA, 2.5])
def test_block_encoding_normalization_is_exp_beta_weight_norm(beta: float) -> None:
    """The chained normalization equals ``exp(beta * W)``."""
    converter = _notebook_converter()
    encoding = converter.block_encoding(beta)

    assert encoding.normalization == pytest.approx(
        math.exp(beta * converter.weight_norm)
    )
    assert encoding.num_signal_qubits == converter.num_ancilla_bits
    assert encoding.num_system_qubits == converter.spin_model.num_bits


@pytest.mark.parametrize("seed", [0, 1, 2, 42])
@pytest.mark.parametrize("beta", [0.0, 0.45, 1.22])
def test_block_encoding_projects_to_the_scaled_imaginary_time_state(
    seed: int,
    beta: float,
) -> None:
    """Projecting the ancillas onto zero yields ``m(beta) exp(-beta H)|+>^n``."""
    rng = np.random.default_rng(seed)
    num_bits = 3
    terms = {
        (0,): float(rng.uniform(-1.0, 1.0)),
        (1,): float(rng.uniform(-1.0, 1.0)),
        (0, 2): float(rng.uniform(-1.0, 1.0)),
    }
    converter = FinITEConverter(BinaryModel.from_higher_ising(dict(terms)))

    psi0 = np.ones(1 << num_bits) / math.sqrt(1 << num_bits)
    expected = (
        math.exp(-beta * converter.weight_norm)
        * np.exp(-beta * _energies(terms, num_bits))
        * psi0
    )

    np.testing.assert_allclose(
        _postselected_amplitudes(converter, beta), expected, atol=1e-12, rtol=0.0
    )


@pytest.mark.parametrize("beta", [-1.0, math.inf, math.nan])
def test_invalid_beta_is_rejected(beta: float) -> None:
    """Negative and non-finite imaginary times are rejected before any work."""
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    converter = _notebook_converter()
    with pytest.raises(ValueError, match="beta must be"):
        converter.block_encoding(beta)
    with pytest.raises(ValueError, match="beta must be"):
        converter.transpile(QiskitTranspiler(), beta=beta)


# ---------------------------------------------------------------------------
# Post-selection
# ---------------------------------------------------------------------------


def test_postselection_keeps_only_all_zero_ancilla_shots() -> None:
    """Shots with any ancilla bit set are discarded, the rest are decoded."""
    converter = _notebook_converter()
    raw = SampleResult(
        results=[
            (([0, 0, 0, 0, 0, 0], [1, 0, 1]), 30),
            (([0, 0, 1, 0, 0, 0], [1, 0, 1]), 55),
            (([0, 0, 0, 0, 0, 0], [1, 1, 1]), 10),
            (([1, 1, 1, 1, 1, 1], [0, 0, 0]), 5),
        ],
        shots=100,
    )

    assert converter.success_probability(raw) == pytest.approx(0.40)

    decoded = converter.decode_to_binary_sampleset(raw)
    assert sum(decoded.num_occurrences) == 40
    assert decoded.vartype == VarType.SPIN
    # Measurement 1 -> spin -1, so the kept 101 shot is the E0 = -4 ground state.
    best_sample, best_energy, best_occurrences = decoded.lowest()
    assert best_energy == pytest.approx(-4.0)
    assert best_sample == {0: -1, 1: 1, 2: -1}
    assert best_occurrences == 30


def test_postselection_of_an_all_failed_run_yields_an_empty_sampleset() -> None:
    """Losing every shot is a legitimate low-yield outcome, not an error."""
    converter = _notebook_converter()
    raw = SampleResult(results=[(([1, 0, 0, 0, 0, 0], [0, 0, 0]), 12)], shots=12)

    assert converter.success_probability(raw) == pytest.approx(0.0)
    decoded = converter.decode_to_binary_sampleset(raw)
    assert decoded.samples == []
    assert decoded.num_occurrences == []


def test_success_probability_of_zero_shots_is_zero() -> None:
    """An empty run reports no acceptance rather than dividing by zero."""
    converter = _notebook_converter()
    assert converter.success_probability(SampleResult(results=[], shots=0)) == 0.0


def test_postselection_rejects_a_single_register_payload() -> None:
    """Results that are not (signal, system) pairs are a programming error."""
    converter = _notebook_converter()
    raw = SampleResult(results=[([0, 0, 0], 4)], shots=4)

    with pytest.raises(ValueError, match=r"\(signal, system\) tuples"):
        converter.success_probability(raw)


def test_decode_from_an_ommx_instance_returns_an_ommx_sampleset() -> None:
    """OMMX-backed converters decode post-selected shots into an OMMX SampleSet."""
    x0 = ommx.v1.DecisionVariable.binary(0, name="x0")
    x1 = ommx.v1.DecisionVariable.binary(1, name="x1")
    instance = ommx.v1.Instance.from_components(
        decision_variables=[x0, x1],
        objective=-2.0 * x0 * x1,
        constraints=[],
        sense=ommx.v1.Instance.MINIMIZE,
    )
    converter = FinITEConverter(instance)
    ancillas = [0] * converter.num_ancilla_bits
    raw = SampleResult(
        results=[
            ((ancillas, [1, 1]), 7),
            (([1] + ancillas[1:], [0, 0]), 3),
        ],
        shots=10,
    )

    decoded = converter.decode(raw)

    assert isinstance(decoded, ommx.v1.SampleSet)
    assert converter.success_probability(raw) == pytest.approx(0.7)
    assert decoded.best_feasible.objective == pytest.approx(-2.0)


# ---------------------------------------------------------------------------
# Paper identities, checked against simulation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("beta", [0.0, 0.25, 0.75, NOTEBOOK_BETA, 2.0])
def test_simulated_metrics_match_proposition_1(beta: float) -> None:
    """Simulation reproduces the closed forms of Proposition 1 (Eqs. 14-15)."""
    converter = _notebook_converter()
    simulated = _simulated_metrics(converter, NOTEBOOK_TERMS, beta)
    predicted = _predicted_metrics(NOTEBOOK_TERMS, beta, 3)

    assert simulated[0] == pytest.approx(predicted[0], rel=1e-9)
    assert simulated[1] == pytest.approx(predicted[1], rel=1e-9)


def test_simulation_reproduces_the_notebook_numbers() -> None:
    """The reference instance reproduces the prototype's published values."""
    converter = _notebook_converter()
    success, fidelity = _simulated_metrics(converter, NOTEBOOK_TERMS, NOTEBOOK_BETA)

    assert success == pytest.approx(0.0009641659, rel=1e-6)
    assert fidelity == pytest.approx(0.9849205342, rel=1e-6)


@pytest.mark.parametrize("beta", [0.0, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0])
def test_success_probability_times_fidelity_is_pinned_to_the_envelope(
    beta: float,
) -> None:
    """Proposition 1: P_LCU * F_g = gamma0 * exp(-2 beta (W + E0)) for every beta."""
    _, _, ground_energy, overlap, _ = _spectrum(NOTEBOOK_TERMS, 3)
    weight_norm = sum(abs(value) for value in NOTEBOOK_TERMS.values())
    success, fidelity = _predicted_metrics(NOTEBOOK_TERMS, beta, 3)
    envelope = overlap * math.exp(-2.0 * beta * (weight_norm + ground_energy))

    assert success * fidelity == pytest.approx(envelope, rel=1e-12)


def test_fidelity_increases_monotonically_with_beta() -> None:
    """Longer imaginary time concentrates more weight on the ground subspace."""
    _, _, _, overlap, _ = _spectrum(NOTEBOOK_TERMS, 3)
    fidelities = [
        _predicted_metrics(NOTEBOOK_TERMS, beta, 3)[1]
        for beta in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
    ]

    assert fidelities == sorted(fidelities)
    assert fidelities[0] == pytest.approx(overlap)
    assert fidelities[-1] > 0.999


# ---------------------------------------------------------------------------
# Corollary 2 threshold
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", [0.2, 0.5, 0.9, 0.99, 0.999])
def test_beta_threshold_actually_reaches_the_target_fidelity(target: float) -> None:
    """Corollary 2's threshold is sufficient: F_g(beta*) >= target."""
    _, _, _, overlap, gap = _spectrum(NOTEBOOK_TERMS, 3)
    threshold = finite_ite_beta_threshold(
        target, spectral_gap=gap, ground_overlap=overlap
    )

    assert math.isfinite(threshold)
    assert _predicted_metrics(NOTEBOOK_TERMS, threshold, 3)[1] >= target


def test_beta_threshold_is_zero_when_the_initial_state_already_qualifies() -> None:
    """Targets at or below gamma0 are met by the uniform state at beta = 0."""
    assert (
        finite_ite_beta_threshold(0.125, spectral_gap=2.0, ground_overlap=0.125) == 0.0
    )
    assert (
        finite_ite_beta_threshold(0.05, spectral_gap=2.0, ground_overlap=0.125) == 0.0
    )


def test_beta_threshold_inverts_the_gap_bound_exactly() -> None:
    """The threshold saturates Corollary 1's bound at the target fidelity."""
    target, gap, overlap = 0.95, 1.5, 0.05
    threshold = finite_ite_beta_threshold(
        target, spectral_gap=gap, ground_overlap=overlap
    )
    bound = overlap / (overlap + (1.0 - overlap) * math.exp(-2.0 * threshold * gap))

    assert bound == pytest.approx(target)


def test_beta_threshold_is_infinite_without_any_initial_overlap() -> None:
    """No finite imaginary time can populate a subspace with zero overlap."""
    assert (
        finite_ite_beta_threshold(0.5, spectral_gap=1.0, ground_overlap=0.0) == math.inf
    )


@pytest.mark.parametrize(
    ("target", "gap", "overlap"),
    [
        (1.0, 1.0, 0.5),
        (-0.1, 1.0, 0.5),
        (0.5, 0.0, 0.5),
        (0.5, math.inf, 0.5),
        (0.5, 1.0, 1.5),
    ],
)
def test_beta_threshold_rejects_out_of_range_arguments(
    target: float,
    gap: float,
    overlap: float,
) -> None:
    """Fidelities, gaps and overlaps outside their domains are rejected."""
    with pytest.raises(ValueError):
        finite_ite_beta_threshold(target, spectral_gap=gap, ground_overlap=overlap)


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


def test_end_to_end_sampling_finds_the_ground_state() -> None:
    """A shot-based run post-selects down to the notebook's optimum, 101."""
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    converter = _notebook_converter()
    transpiler = QiskitTranspiler()
    program = converter.transpile(transpiler, beta=NOTEBOOK_BETA)
    result = program.sample(transpiler.executor(), shots=200_000).result()

    expected_success, _ = _predicted_metrics(NOTEBOOK_TERMS, NOTEBOOK_BETA, 3)
    tolerance = 6.0 * math.sqrt(expected_success * (1.0 - expected_success) / 200_000)
    assert converter.success_probability(result) == pytest.approx(
        expected_success, abs=tolerance
    )

    decoded = converter.decode(result)
    best_sample, best_energy, _ = decoded.lowest()
    assert best_energy == pytest.approx(-4.0)
    assert best_sample == {0: -1, 1: 1, 2: -1}


def test_transpiled_program_has_no_runtime_parameters() -> None:
    """beta is structural, so the emitted circuit exposes nothing to bind."""
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    converter = _notebook_converter()
    program = converter.transpile(QiskitTranspiler(), beta=NOTEBOOK_BETA)

    assert program.quantum_circuit.num_qubits == (
        converter.spin_model.num_bits + converter.num_ancilla_bits
    )
    assert list(program.quantum_circuit.parameters) == []
