"""Tests for the QSVT eigenstate-filtering converter."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

import qamomile.observable as qm_o
from qamomile.circuit.transpiler.job import SampleResult
from qamomile.optimization.binary_model import BinaryModel
from qamomile.optimization.qsvt_filter import QSVTFilterConverter

qiskit = pytest.importorskip("qiskit")


@pytest.fixture
def transpiler() -> Any:
    """Return a Qiskit transpiler for converter compilation.

    Returns:
        Any: A ``QiskitTranspiler`` instance.
    """
    from qamomile.qiskit import QiskitTranspiler

    return QiskitTranspiler()


def _ising_diagonal(
    coefficients: dict[tuple[int, ...], float], num_system_qubits: int
) -> np.ndarray:
    """Return the eigenvalues of a diagonal Ising-Z Hamiltonian.

    Args:
        coefficients (dict[tuple[int, ...], float]): Ising coefficients keyed by
            products of spin indices.
        num_system_qubits (int): Number of spins.

    Returns:
        np.ndarray: Energies indexed by little-endian computational basis state.
    """
    diagonal = np.zeros(1 << num_system_qubits)
    for basis in range(1 << num_system_qubits):
        for word, coefficient in coefficients.items():
            parity = sum((basis >> index) & 1 for index in word) & 1
            diagonal[basis] += coefficient * (-1.0 if parity else 1.0)
    return diagonal


def _exact_success_probability(
    converter: QSVTFilterConverter,
    transpiler: Any,
    mu: float,
    phases: list[float],
    num_system_qubits: int,
) -> float:
    """Return the noiseless all-zero ancilla probability of the probe circuit.

    Args:
        converter (QSVTFilterConverter): Converter under test.
        transpiler (Any): Qiskit transpiler used for compilation.
        mu (float): Energy threshold.
        phases (list[float]): Reflection-convention phases.
        num_system_qubits (int): Width of the system register.

    Returns:
        float: Exact statevector probability of measuring every ancilla zero.
    """
    from qiskit.quantum_info import Statevector

    executable = converter.transpile(transpiler, mu=mu, phi=phases)
    circuit = executable.quantum_circuit.remove_final_measurements(inplace=False)
    state = Statevector.from_instruction(circuit).data
    num_ancilla = converter.num_ancilla_bits
    amplitudes = np.array(
        [state[basis << num_ancilla] for basis in range(1 << num_system_qubits)]
    )
    return float(np.sum(np.abs(amplitudes) ** 2))


def test_encoding_captures_the_full_cost_hamiltonian() -> None:
    """The block encoding covers every term, constant included."""
    model = BinaryModel.from_higher_ising(
        {(0,): 1.0, (1,): -1.0, (0, 1): -1.0}, constant=0.5
    )
    converter = QSVTFilterConverter(model)

    assert converter.encoding.num_system_qubits == 2
    # 1-norm of all coefficients, including the constant term.
    assert converter.normalization == pytest.approx(3.5)


def test_cost_hamiltonian_matches_the_spin_model() -> None:
    """The reported Hamiltonian is the Ising operator being block encoded."""
    coefficients = {(0,): 1.0, (1,): -1.0, (0, 1): -1.0, (0, 1, 2): 0.25}
    model = BinaryModel.from_higher_ising(coefficients, constant=0.5)
    converter = QSVTFilterConverter(model)

    hamiltonian = converter.get_cost_hamiltonian()
    expected = qm_o.Hamiltonian()
    for indices, coefficient in coefficients.items():
        expected.add_term(
            tuple(qm_o.PauliOperator(qm_o.Pauli.Z, index) for index in indices),
            coefficient,
        )
    expected.constant = 0.5

    assert hamiltonian.terms == expected.terms
    assert hamiltonian.constant == pytest.approx(expected.constant)


@pytest.mark.parametrize("mu", [-2.0, 0.0, 1.5])
def test_shifted_encoding_normalization_grows_with_the_shift(mu: float) -> None:
    """Composing with the identity term adds |mu| to the subnormalization."""
    model = BinaryModel.from_higher_ising({(0,): 1.0, (0, 1): -1.0})
    converter = QSVTFilterConverter(model)

    shifted = converter.shifted_encoding(mu)

    assert shifted.num_system_qubits == converter.encoding.num_system_qubits
    assert shifted.normalization == pytest.approx(converter.normalization + abs(mu))
    assert shifted.num_signal_qubits >= converter.encoding.num_signal_qubits


def test_qsp_phases_are_odd_length_and_cached() -> None:
    """Phase generation returns degree+1 phases and reuses its cache."""
    pytest.importorskip("pyqsp")
    model = BinaryModel.from_higher_ising({(0,): 1.0})
    converter = QSVTFilterConverter(model)

    phases = converter.qsp_phases(degree=11, delta=5)
    assert len(phases) == 12

    cached = converter.qsp_phases(degree=11, delta=5)
    assert cached == phases
    # The cache hands out copies, so callers cannot corrupt it.
    cached[0] = 0.0
    assert converter.qsp_phases(degree=11, delta=5) == phases

    for degree in (0, -1, 10):
        with pytest.raises(ValueError, match="degree"):
            converter.qsp_phases(degree=degree)


def test_transpile_rejects_odd_length_phase_sequences(transpiler: Any) -> None:
    """The alternation needs an even phase count (odd polynomial degree)."""
    model = BinaryModel.from_higher_ising({(0,): 1.0})
    converter = QSVTFilterConverter(model)

    for phases in ([0.1], [0.1, 0.2, 0.3]):
        with pytest.raises(ValueError, match="even number"):
            converter.transpile(transpiler, mu=0.0, phi=phases)


def test_transpile_sizes_the_circuit_from_the_shifted_encoding(
    transpiler: Any,
) -> None:
    """The probe allocates one projector qubit plus signal and system."""
    model = BinaryModel.from_higher_ising({(0,): 1.0, (1,): -1.0, (0, 1): -1.0})
    converter = QSVTFilterConverter(model)
    shifted = converter.shifted_encoding(0.5)

    executable = converter.transpile(transpiler, mu=0.5, phi=[0.1, 0.2])

    expected = 1 + shifted.num_signal_qubits + shifted.num_system_qubits
    assert executable.quantum_circuit.num_qubits == expected
    assert converter.num_ancilla_bits == 1 + shifted.num_signal_qubits


@pytest.mark.parametrize(
    ("mu", "expected_fraction"),
    [(-3.5, 0.0), (-0.5, 0.75), (3.5, 1.0)],
)
def test_success_probability_counts_the_states_below_the_threshold(
    transpiler: Any, mu: float, expected_fraction: float
) -> None:
    """The filtered weight of the uniform superposition tracks the spectrum.

    The probe starts from the uniform superposition, so the exact all-zero
    ancilla probability is the fraction of eigenstates with energy below
    ``mu`` — the predicate the Lin & Tong binary search thresholds.
    """
    pytest.importorskip("pyqsp")
    coefficients = {(0,): 1.0, (1,): -1.0, (0, 1): -1.0}
    model = BinaryModel.from_higher_ising(coefficients)
    converter = QSVTFilterConverter(model)
    phases = converter.qsp_phases()

    probability = _exact_success_probability(converter, transpiler, mu, phases, 2)

    energies = _ising_diagonal(coefficients, 2)
    assert float((energies < mu).mean()) == pytest.approx(expected_fraction)
    assert probability == pytest.approx(expected_fraction, abs=0.05)


def test_sampled_filter_recovers_the_ground_states(transpiler: Any) -> None:
    """Post-selected samples concentrate on the states the filter keeps."""
    pytest.importorskip("pyqsp")
    coefficients = {(0,): 1.0, (1,): -1.0, (0, 1): -1.0}
    model = BinaryModel.from_higher_ising(coefficients)
    converter = QSVTFilterConverter(model)
    phases = converter.qsp_phases()

    executable = converter.transpile(transpiler, mu=-0.5, phi=phases)
    result = executable.sample(transpiler.executor(), shots=2000).result()

    assert converter.success_probability(result) == pytest.approx(0.75, abs=0.05)

    sampleset = converter.decode_to_binary_sampleset(result)
    ground_energy = float(_ising_diagonal(coefficients, 2).min())
    kept = sum(sampleset.num_occurrences)
    ground_shots = sum(
        occurrences
        for energy, occurrences in zip(sampleset.energy, sampleset.num_occurrences)
        if math.isclose(energy, ground_energy, abs_tol=1e-9)
    )
    assert kept > 0
    assert ground_shots / kept > 0.95


def test_decode_rejects_results_that_are_not_probe_measurements() -> None:
    """Decoding fails loudly when handed results from another circuit."""
    model = BinaryModel.from_higher_ising({(0,): 1.0, (0, 1): -1.0})
    converter = QSVTFilterConverter(model)
    samples: SampleResult[Any] = SampleResult(results=[([0, 1], 10)], shots=10)

    with pytest.raises(ValueError, match="projector, signal, system"):
        converter.decode_to_binary_sampleset(samples)
    with pytest.raises(ValueError, match="projector, signal, system"):
        converter.success_probability(samples)


def test_success_probability_of_an_empty_result_is_zero() -> None:
    """A zero-shot result reports no filtered weight instead of dividing by 0."""
    model = BinaryModel.from_higher_ising({(0,): 1.0})
    converter = QSVTFilterConverter(model)
    empty: SampleResult[Any] = SampleResult(results=[], shots=0)

    assert converter.success_probability(empty) == 0.0
