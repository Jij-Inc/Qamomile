"""Tests for ordered products of exact block encodings."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o


def _executor(case: Any) -> Any:
    """Return a local simulator executor for one SDK fixture case."""
    if case.backend_name == "qiskit":
        from qiskit.providers.basic_provider import BasicSimulator

        return case.transpiler.executor(backend=BasicSimulator())
    return case.transpiler.executor()


def _zero_probability(results: list[tuple[Any, int]]) -> float:
    """Return the observed probability of an all-zero signal outcome."""
    total = sum(count for _, count in results)
    zero = sum(
        count for outcome, count in results if not any(int(bit) for bit in outcome)
    )
    return zero / total


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


def _ising_matrix(
    coefficients: dict[tuple[int, ...], complex],
    num_qubits: int,
) -> np.ndarray:
    """Return the dense diagonal matrix represented by Ising-Z words."""
    diagonal = np.zeros(1 << num_qubits, dtype=np.complex128)
    for basis in range(1 << num_qubits):
        for word, coefficient in coefficients.items():
            parity = sum((basis >> index) & 1 for index in word) & 1
            diagonal[basis] += coefficient * (-1.0 if parity else 1.0)
    return np.diag(diagonal)


def _random_terms(
    rng: np.random.Generator,
    num_qubits: int,
    max_terms: int = 2,
) -> dict[tuple[int, ...], float]:
    """Draw a small random identity-free Ising-Z coefficient mapping."""
    words = [
        tuple(i for i in range(num_qubits) if mask & (1 << i))
        for mask in range(1, 1 << num_qubits)
    ]
    chosen = rng.permutation(len(words))[: min(max_terms, len(words))]
    return {
        words[int(index)]: float(rng.choice([-1.0, 1.0]) * rng.uniform(0.3, 1.0))
        for index in chosen
    }


def _projected_block(encoding: qmc.LCUBlockEncoding) -> np.ndarray:
    """Return the all-zero-signal block of an encoding via Qiskit."""
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Operator

    from qamomile.qiskit import QiskitTranspiler

    @qmc.qkernel
    def kernel() -> qmc.Bit:
        """Allocate the registers and apply the encoding once."""
        signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
        system = qmc.qubit_array(encoding.num_system_qubits, "system")
        signal, _ = encoding.unitary(signal, system)
        return qmc.measure(signal[0])

    circuit = (
        QiskitTranspiler()
        .transpile(kernel)
        .quantum_circuit.remove_final_measurements(inplace=False)
    )
    unitary = np.asarray(Operator(circuit).data)
    indices = [
        basis << encoding.num_signal_qubits
        for basis in range(1 << encoding.num_system_qubits)
    ]
    return unitary[np.ix_(indices, indices)]


def test_product_reports_summed_widths_and_multiplied_norms() -> None:
    """The product descriptor concatenates signals and multiplies normalizations."""
    first = qmc.ising_z_block_encoding({(): 0.25, (0,): -1.0}, 2)
    second = qmc.ising_z_block_encoding({(0, 1): 0.5, (1,): 0.5}, 2)
    product = qmc.product_block_encoding([first, second])

    assert product.num_system_qubits == 2
    assert product.num_signal_qubits == (
        first.num_signal_qubits + second.num_signal_qubits
    )
    assert product.normalization == pytest.approx(
        first.normalization * second.normalization
    )


@pytest.mark.parametrize("seed", [0, 1, 2, 42])
@pytest.mark.parametrize("num_qubits", [1, 2, 3])
def test_product_matches_the_ordered_dense_product(seed: int, num_qubits: int) -> None:
    """The projected block equals the ordered matrix product of the children."""
    rng = np.random.default_rng(seed)
    first_terms = _random_terms(rng, num_qubits)
    second_terms = _random_terms(rng, num_qubits)
    first = qmc.ising_z_block_encoding(first_terms, num_qubits)
    second = qmc.ising_z_block_encoding(second_terms, num_qubits)
    product = qmc.product_block_encoding([first, second])

    # encodings[0] acts first, so it is the rightmost matrix factor.
    expected = (
        _ising_matrix(second_terms, num_qubits) @ _ising_matrix(first_terms, num_qubits)
    ) / product.normalization
    np.testing.assert_allclose(
        _projected_block(product), expected, atol=1e-10, rtol=0.0
    )


def test_product_respects_order_for_noncommuting_children() -> None:
    """With a non-diagonal child the composition order is observable.

    Ising-Z children are all diagonal and therefore commute, which would hide
    a reversed chain. Pairing one with an ``X``-bearing Pauli encoding pins
    the ordering down.
    """
    pytest.importorskip("qiskit")
    from qamomile.linalg import PauliLCU

    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    pauli_z = np.array([[1.0, 0.0], [0.0, -1.0]])

    first = qmc.ising_z_block_encoding({(0,): 1.0}, 1)
    second = qmc.pauli_lcu_block_encoding(
        PauliLCU.from_matrix(0.5 * np.eye(2) + 0.7 * pauli_x)
    )
    product = qmc.product_block_encoding([first, second])

    left = 0.5 * np.eye(2) + 0.7 * pauli_x
    forward = (left @ pauli_z) / product.normalization
    reversed_order = (pauli_z @ left) / product.normalization

    np.testing.assert_allclose(_projected_block(product), forward, atol=1e-10, rtol=0.0)
    # Guard the guard: the two orders must actually differ, or this proves nothing.
    assert not np.allclose(forward, reversed_order, atol=1e-10)


def test_product_of_a_single_child_reproduces_that_child() -> None:
    """A one-element product is the child itself, widths and block included."""
    child = qmc.ising_z_block_encoding({(0,): 0.75, (): 0.25}, 1)
    product = qmc.product_block_encoding([child])

    assert product.num_signal_qubits == child.num_signal_qubits
    assert product.normalization == pytest.approx(child.normalization)
    np.testing.assert_allclose(
        _projected_block(product), _projected_block(child), atol=1e-12, rtol=0.0
    )


def test_product_composes_recursively_with_itself() -> None:
    """A product descriptor is a valid child of another product."""
    identity = qmc.identity_block_encoding(1)
    inner = qmc.product_block_encoding([identity, identity])
    outer = qmc.product_block_encoding([inner, identity])

    assert outer.num_signal_qubits == 3
    assert outer.normalization == pytest.approx(1.0)


def test_product_rejects_empty_mismatched_and_non_descriptor_inputs() -> None:
    """Empty sequences, disagreeing widths and non-descriptors are rejected."""
    with pytest.raises(ValueError, match="nonempty"):
        qmc.product_block_encoding([])

    with pytest.raises(ValueError, match="one system width"):
        qmc.product_block_encoding(
            [
                qmc.ising_z_block_encoding({(0,): 1.0}, 1),
                qmc.ising_z_block_encoding({(0,): 1.0}, 2),
            ]
        )

    with pytest.raises(TypeError, match="LCUBlockEncoding"):
        qmc.product_block_encoding([qmc.ising_z_block_encoding({(0,): 1.0}, 1), "nope"])

    with pytest.raises(TypeError, match="ordered sequence"):
        qmc.product_block_encoding(42)


# ---------------------------------------------------------------------------
# Cross-backend execution: sampling and expectation value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 42])
@pytest.mark.parametrize("num_qubits", [1, 2, 3])
@pytest.mark.parametrize("num_children", [1, 2, 3])
def test_product_samples_and_estimates_on_every_sdk(
    sdk_transpiler: Any,
    seed: int,
    num_qubits: int,
    num_children: int,
) -> None:
    """Sampler and estimator paths agree with the analytic success probability.

    Applying the product to the all-zero basis state leaves the ancillas in
    the all-zero state with probability ``|prod_j A_j[0, 0] / alpha|**2``,
    since every child is diagonal.
    """
    rng = np.random.default_rng(seed)
    children_terms = [_random_terms(rng, num_qubits) for _ in range(num_children)]
    children = [
        qmc.ising_z_block_encoding(terms, num_qubits) for terms in children_terms
    ]
    product = qmc.product_block_encoding(children)

    # Basis state |0...0>: every Z word evaluates to +1, so each child's
    # diagonal entry is just the sum of its coefficients.
    amplitude = 1.0
    for terms in children_terms:
        amplitude *= sum(terms.values())
    success = (amplitude / product.normalization) ** 2

    @qmc.qkernel
    def sample_kernel() -> qmc.Vector[qmc.Bit]:
        """Apply the product to |0...0> and measure the signal register."""
        signal = qmc.qubit_array(product.num_signal_qubits, "signal")
        system = qmc.qubit_array(product.num_system_qubits, "system")
        signal, _ = product.unitary(signal, system)
        return qmc.measure(signal)

    @qmc.qkernel
    def expval_kernel(observable: qmc.Observable) -> qmc.Float:
        """Estimate the all-zero signal projector after the product."""
        signal = qmc.qubit_array(product.num_signal_qubits, "signal")
        system = qmc.qubit_array(product.num_system_qubits, "system")
        signal, _ = product.unitary(signal, system)
        return qmc.expval(signal, observable)

    shots = 4096
    sample_program = sdk_transpiler.transpiler.transpile(sample_kernel)
    sample_result = sample_program.sample(
        _executor(sdk_transpiler), shots=shots
    ).result()
    tolerance = 6.0 * math.sqrt(success * (1.0 - success) / shots) + 0.02
    assert _zero_probability(sample_result.results) == pytest.approx(
        success, abs=tolerance
    )

    expval_program = sdk_transpiler.transpiler.transpile(
        expval_kernel,
        bindings={"observable": _zero_projector(product.num_signal_qubits)},
    )
    observed = float(expval_program.run(_executor(sdk_transpiler)).result())
    atol = 1e-6 if sdk_transpiler.backend_name == "cudaq" else 1e-8
    assert observed == pytest.approx(success, abs=atol)


def test_product_of_identities_always_succeeds_on_every_sdk(
    sdk_transpiler: Any,
) -> None:
    """A product of identity encodings is a boundary case with certain success."""
    product = qmc.product_block_encoding([qmc.identity_block_encoding(2)] * 3)

    @qmc.qkernel
    def sample_kernel() -> qmc.Vector[qmc.Bit]:
        """Apply the identity product and measure the signal register."""
        signal = qmc.qubit_array(product.num_signal_qubits, "signal")
        system = qmc.qubit_array(product.num_system_qubits, "system")
        signal, _ = product.unitary(signal, system)
        return qmc.measure(signal)

    program = sdk_transpiler.transpiler.transpile(sample_kernel)
    result = program.sample(_executor(sdk_transpiler), shots=512).result()

    assert product.normalization == pytest.approx(1.0)
    assert _zero_probability(result.results) == pytest.approx(1.0)
