"""Tests for the FinITE block encoding and sampling kernel."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.algorithm import finite_ite_block_encoding, finite_ite_state
from qamomile.circuit.algorithm.finite_ite import _term_block_encoding

# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------


def _executor(case: Any) -> Any:
    """Return a local simulator executor for one SDK fixture case."""
    if case.backend_name == "qiskit":
        from qiskit.providers.basic_provider import BasicSimulator

        return case.transpiler.executor(backend=BasicSimulator())
    return case.transpiler.executor()


def _signal_zero_probability(results: list[tuple[Any, int]]) -> float:
    """Return the observed fraction of shots whose signal bits are all zero."""
    total = sum(count for _, count in results)
    accepted = sum(
        count for (signal, _system), count in results if not any(int(b) for b in signal)
    )
    return accepted / total


def _zero_projector(num_qubits: int) -> qm_o.Hamiltonian:
    """Return the Pauli expansion of the all-zero projector on a register."""
    coefficient = 1.0 / (1 << num_qubits)
    projector = qm_o.Hamiltonian.identity(coefficient, num_qubits=num_qubits)
    for mask in range(1, 1 << num_qubits):
        operators = tuple(
            qm_o.PauliOperator(qm_o.Pauli.Z, index)
            for index in range(num_qubits)
            if mask & (1 << index)
        )
        projector.add_term(operators, coefficient)
    return projector


# ---------------------------------------------------------------------------
# Classical references
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


def _reference_block(
    terms: dict[tuple[int, ...], float],
    beta: float,
    num_qubits: int,
) -> np.ndarray:
    """Return ``m(beta) exp(-beta H)|+>^n`` computed classically."""
    weight_norm = sum(abs(value) for value in terms.values())
    psi0 = np.ones(1 << num_qubits) / math.sqrt(1 << num_qubits)
    return (
        math.exp(-beta * weight_norm)
        * np.exp(-beta * _energies(terms, num_qubits))
        * psi0
    )


def _reference_success(
    terms: dict[tuple[int, ...], float],
    beta: float,
    num_qubits: int,
) -> float:
    """Return the exact LCU success probability ``P_LCU(beta)``."""
    block = _reference_block(terms, beta, num_qubits)
    return float(np.vdot(block, block).real)


def _random_terms(
    rng: np.random.Generator,
    num_qubits: int,
    max_terms: int = 4,
) -> dict[tuple[int, ...], float]:
    """Draw a small random identity-free Ising-Z coefficient mapping."""
    words = [
        tuple(i for i in range(num_qubits) if mask & (1 << i))
        for mask in range(1, 1 << num_qubits)
    ]
    chosen = rng.permutation(len(words))[: min(max_terms, len(words))]
    # Keep magnitudes away from zero so no term silently degenerates.
    return {
        words[int(index)]: float(rng.choice([-1.0, 1.0]) * rng.uniform(0.3, 1.0))
        for index in chosen
    }


def _uniform_state_block(
    encoding: qmc.LCUBlockEncoding,
    num_qubits: int,
) -> np.ndarray:
    """Return the all-zero-ancilla amplitudes after applying to |+>^n."""
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Statevector

    from qamomile.qiskit import QiskitTranspiler

    circuit = (
        QiskitTranspiler()
        .transpile(finite_ite_state(encoding))
        .quantum_circuit.remove_final_measurements(inplace=False)
    )
    statevector = np.asarray(Statevector.from_instruction(circuit).data)
    # `signal` is allocated first, so it occupies the low wires.
    return statevector[
        [basis << encoding.num_signal_qubits for basis in range(1 << num_qubits)]
    ]


# ---------------------------------------------------------------------------
# Per-term factor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("coefficient", [-1.5, -0.4, 0.4, 1.5])
@pytest.mark.parametrize("beta", [0.0, 0.3, 1.22])
def test_term_normalization_is_the_exponential_of_the_weight(
    coefficient: float,
    beta: float,
) -> None:
    """One FinITE factor has coefficient one-norm ``exp(beta * |x|)``."""
    encoding = _term_block_encoding((0,), coefficient, beta, 2)

    assert encoding.normalization == pytest.approx(math.exp(beta * abs(coefficient)))
    assert encoding.num_signal_qubits == 1
    assert encoding.num_system_qubits == 2


def test_term_at_zero_beta_is_the_identity_block() -> None:
    """At ``beta = 0`` the sinh term drops out, leaving an identity block."""
    encoding = _term_block_encoding((0, 1), -1.0, 0.0, 2)

    assert encoding.normalization == pytest.approx(1.0)
    assert encoding.num_signal_qubits == 1


@pytest.mark.parametrize("coefficient", [-1.0, 1.0])
def test_term_block_equals_the_scaled_exponential(coefficient: float) -> None:
    """The projected block is ``exp(-beta x Z) / exp(beta |x|)`` on the diagonal."""
    beta = 0.8
    encoding = _term_block_encoding((0,), coefficient, beta, 1)

    psi0 = np.ones(2) / math.sqrt(2)
    expected = np.exp(-beta * coefficient * np.array([1.0, -1.0])) * psi0
    expected /= math.exp(beta * abs(coefficient))
    np.testing.assert_allclose(
        _uniform_state_block(encoding, 1), expected, atol=1e-12, rtol=0.0
    )


@pytest.mark.parametrize("coefficient", [math.inf, math.nan])
def test_term_rejects_non_finite_coefficients(coefficient: float) -> None:
    """A non-finite Pauli coefficient has no finite hyperbolic expansion."""
    with pytest.raises(ValueError, match="coefficient must be finite"):
        _term_block_encoding((0,), coefficient, 0.5, 1)


# ---------------------------------------------------------------------------
# Composed encoding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 42])
@pytest.mark.parametrize("num_qubits", [1, 2, 3])
def test_normalization_is_exp_beta_weight_norm(seed: int, num_qubits: int) -> None:
    """The chained normalization equals ``exp(beta * W)``, one ancilla per term."""
    rng = np.random.default_rng(seed)
    terms = _random_terms(rng, num_qubits)
    beta = float(rng.uniform(0.0, 1.5))
    weight_norm = sum(abs(value) for value in terms.values())

    encoding = finite_ite_block_encoding(terms, beta, num_qubits)

    assert encoding.normalization == pytest.approx(math.exp(beta * weight_norm))
    assert encoding.num_signal_qubits == len(terms)
    assert encoding.num_system_qubits == num_qubits


@pytest.mark.parametrize("seed", [0, 1, 2, 42])
@pytest.mark.parametrize("num_qubits", [1, 2, 3])
@pytest.mark.parametrize("beta", [0.0, 0.45, 1.22])
def test_projected_block_equals_scaled_imaginary_time_state(
    seed: int,
    num_qubits: int,
    beta: float,
) -> None:
    """Projecting the ancillas onto zero yields ``m(beta) exp(-beta H)|+>^n``."""
    rng = np.random.default_rng(seed)
    terms = _random_terms(rng, num_qubits)
    encoding = finite_ite_block_encoding(terms, beta, num_qubits)

    np.testing.assert_allclose(
        _uniform_state_block(encoding, num_qubits),
        _reference_block(terms, beta, num_qubits),
        atol=1e-12,
        rtol=0.0,
    )


def test_zero_beta_leaves_the_uniform_superposition_untouched() -> None:
    """At ``beta = 0`` the encoding is the identity, so post-selection is certain."""
    encoding = finite_ite_block_encoding({(0,): 1.0, (0, 1): -0.5}, 0.0, 2)
    block = _uniform_state_block(encoding, 2)

    assert encoding.normalization == pytest.approx(1.0)
    assert float(np.vdot(block, block).real) == pytest.approx(1.0)


def test_term_order_does_not_change_the_encoded_operator() -> None:
    """Terms are sorted internally, so mapping order cannot alter the block."""
    forward = {(0,): 1.0, (1,): -0.5, (0, 1): 0.25}
    shuffled = {(0, 1): 0.25, (0,): 1.0, (1,): -0.5}

    np.testing.assert_allclose(
        _uniform_state_block(finite_ite_block_encoding(forward, 0.6, 2), 2),
        _uniform_state_block(finite_ite_block_encoding(shuffled, 0.6, 2), 2),
        atol=1e-12,
        rtol=0.0,
    )


def test_identity_and_empty_coefficients_are_rejected() -> None:
    """Identity words must be applied classically, and there must be a term."""
    with pytest.raises(ValueError, match="identity-free"):
        finite_ite_block_encoding({(): 1.0, (0,): 1.0}, 0.5, 1)

    with pytest.raises(ValueError, match="at least one Pauli term"):
        finite_ite_block_encoding({}, 0.5, 1)


def test_non_mapping_coefficients_are_rejected() -> None:
    """The coefficient container must be a mapping, not a sequence of pairs."""
    with pytest.raises(TypeError, match="must be a mapping"):
        finite_ite_block_encoding([((0,), 1.0)], 0.5, 1)


@pytest.mark.parametrize("beta", [-1e-9, -1.0, math.inf, math.nan])
def test_invalid_beta_is_rejected(beta: float) -> None:
    """Negative and non-finite imaginary times are rejected."""
    with pytest.raises(ValueError, match="beta must be"):
        finite_ite_block_encoding({(0,): 1.0}, beta, 1)


# ---------------------------------------------------------------------------
# Sampling kernel
# ---------------------------------------------------------------------------


def test_kernel_reads_its_register_widths_from_the_encoding() -> None:
    """The probe allocates exactly the descriptor's signal and system widths."""
    encoding = finite_ite_block_encoding({(0,): 1.0, (0, 1): -1.0}, 0.5, 2)

    assert finite_ite_state(encoding).output_types == [
        qmc.Vector[qmc.Bit],
        qmc.Vector[qmc.Bit],
    ]


def test_kernel_rejects_a_non_descriptor_argument() -> None:
    """The probe consumes a block-encoding descriptor, not raw coefficients."""
    with pytest.raises(TypeError, match="LCUBlockEncoding"):
        finite_ite_state({(0,): 1.0})


def test_kernel_accepts_any_block_encoding_descriptor() -> None:
    """The probe is descriptor-generic, not tied to the FinITE producer."""
    kernel = finite_ite_state(qmc.identity_block_encoding(2))

    assert kernel.output_types == [qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]


# ---------------------------------------------------------------------------
# Cross-backend execution: sampling and expectation value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 42])
@pytest.mark.parametrize("num_qubits", [1, 2, 3])
@pytest.mark.parametrize("beta", [0.0, 0.25])
def test_finite_ite_samples_and_estimates_on_every_sdk(
    sdk_transpiler: Any,
    seed: int,
    num_qubits: int,
    beta: float,
) -> None:
    """Sampler and estimator paths both reproduce the exact success probability."""
    rng = np.random.default_rng(seed)
    terms = _random_terms(rng, num_qubits)
    success = _reference_success(terms, beta, num_qubits)
    encoding = finite_ite_block_encoding(terms, beta, num_qubits)

    @qmc.qkernel
    def expval_kernel(observable: qmc.Observable) -> qmc.Float:
        """Estimate the all-zero ancilla projector after the FinITE encoding."""
        signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
        system = qmc.qubit_array(encoding.num_system_qubits, "system")
        system = qmc.h(system)
        signal, system = encoding.unitary(signal, system)
        return qmc.expval(signal, observable)

    shots = 4096
    sample_program = sdk_transpiler.transpiler.transpile(finite_ite_state(encoding))
    sample_result = sample_program.sample(
        _executor(sdk_transpiler), shots=shots
    ).result()
    tolerance = 6.0 * math.sqrt(success * (1.0 - success) / shots) + 0.02
    assert _signal_zero_probability(sample_result.results) == pytest.approx(
        success, abs=tolerance
    )

    expval_program = sdk_transpiler.transpiler.transpile(
        expval_kernel,
        bindings={"observable": _zero_projector(encoding.num_signal_qubits)},
    )
    observed = float(expval_program.run(_executor(sdk_transpiler)).result())
    atol = 1e-6 if sdk_transpiler.backend_name == "cudaq" else 1e-8
    assert observed == pytest.approx(success, abs=atol)


@pytest.mark.parametrize("num_qubits", [1, 2, 3])
def test_finite_ite_state_measures_both_registers_on_every_sdk(
    sdk_transpiler: Any,
    num_qubits: int,
) -> None:
    """Every shot carries the ancilla flag alongside the system bitstring."""
    terms = {(index,): 1.0 for index in range(num_qubits)}
    encoding = finite_ite_block_encoding(terms, 0.3, num_qubits)
    program = sdk_transpiler.transpiler.transpile(finite_ite_state(encoding))
    result = program.sample(_executor(sdk_transpiler), shots=256).result()

    for (signal, system), _count in result.results:
        assert len(signal) == num_qubits
        assert len(system) == num_qubits
        assert all(int(bit) in (0, 1) for bit in signal)
        assert all(int(bit) in (0, 1) for bit in system)
