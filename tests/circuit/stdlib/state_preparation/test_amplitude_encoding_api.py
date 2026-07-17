"""Cross-backend tests for the generic and explicit amplitude-encoding APIs."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o

EncodingApi = Literal["generic", "mottonen"]

_ENCODING_APIS: list[pytest.ParameterSet] = [
    pytest.param("generic", id="generic"),
    pytest.param("mottonen", id="mottonen"),
]
_RANDOM_CASES: list[pytest.ParameterSet] = [
    pytest.param(1, 0, id="1q-seed0"),
    pytest.param(2, 1, id="2q-seed1"),
    pytest.param(3, 2, id="3q-seed2"),
    pytest.param(2, 42, id="2q-seed42"),
]
_SHOTS = 2048
_STD_TOLERANCE = 6.0


def _random_complex_amplitudes(n_qubits: int, seed: int) -> list[complex]:
    """Return deterministic random complex amplitudes.

    Args:
        n_qubits (int): Number of qubits represented by the amplitudes.
        seed (int): Seed for ``numpy.random.default_rng``.

    Returns:
        list[complex]: Complex vector of length ``2**n_qubits``.
    """
    rng = np.random.default_rng(seed)
    real = rng.standard_normal(2**n_qubits)
    imaginary = rng.standard_normal(2**n_qubits)
    return (real + 1j * imaginary).tolist()


def _build_sample_kernel(
    api: EncodingApi,
    amplitudes: list[complex],
) -> qmc.QKernel:
    """Build a sampling kernel for one amplitude-encoding API.

    Args:
        api (EncodingApi): Public API variant to exercise.
        amplitudes (list[complex]): Amplitudes to prepare.

    Returns:
        qmc.QKernel: Kernel measuring the prepared register.
    """
    n_qubits = int(round(np.log2(len(amplitudes))))

    if api == "generic":

        @qmc.qkernel
        def generic_kernel() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n_qubits, "q")
            q = qmc.amplitude_encoding(q, amplitudes)
            return qmc.measure(q)

        return generic_kernel

    @qmc.qkernel
    def mottonen_kernel() -> qmc.Vector[qmc.Bit]:
        q = qmc.qubit_array(n_qubits, "q")
        q = qmc.mottonen_amplitude_encoding(q, amplitudes)
        return qmc.measure(q)

    return mottonen_kernel


def _build_expval_kernel(
    api: EncodingApi,
    amplitudes: list[complex],
) -> qmc.QKernel:
    """Build a phase-sensitive expectation-value kernel.

    Args:
        api (EncodingApi): Public API variant to exercise.
        amplitudes (list[complex]): Amplitudes to prepare.

    Returns:
        qmc.QKernel: Kernel evaluating a runtime-bound observable.
    """
    n_qubits = int(round(np.log2(len(amplitudes))))

    if api == "generic":

        @qmc.qkernel
        def generic_kernel(H: qmc.Observable) -> qmc.Float:
            q = qmc.qubit_array(n_qubits, "q")
            q = qmc.amplitude_encoding(q, amplitudes)
            return qmc.expval(q, H)

        return generic_kernel

    @qmc.qkernel
    def mottonen_kernel(H: qmc.Observable) -> qmc.Float:
        q = qmc.qubit_array(n_qubits, "q")
        q = qmc.mottonen_amplitude_encoding(q, amplitudes)
        return qmc.expval(q, H)

    return mottonen_kernel


@qmc.qkernel
def _generic_bound_kernel(
    amplitudes: qmc.Vector[qmc.Float],
) -> qmc.Vector[qmc.Bit]:
    """Build a generic amplitude-encoding kernel with bound input data.

    Args:
        amplitudes (qmc.Vector[qmc.Float]): Real amplitudes supplied through
            transpile-time bindings.

    Returns:
        qmc.Vector[qmc.Bit]: Measurements of the prepared two-qubit register.
    """
    qubits = qmc.qubit_array(2, "q")
    qubits = qmc.amplitude_encoding(qubits, amplitudes)
    return qmc.measure(qubits)


def _bits_to_index(bits: tuple[int, ...] | int) -> int:
    """Convert a little-endian sample value to a basis-state index.

    Args:
        bits (tuple[int, ...] | int): Sampled scalar or register value.

    Returns:
        int: Corresponding computational-basis index.
    """
    if isinstance(bits, int):
        return bits
    return sum(int(bit) << index for index, bit in enumerate(bits))


def _expected_y0(amplitudes: list[complex]) -> float:
    """Return the analytic expectation value of Pauli Y on qubit zero.

    Args:
        amplitudes (list[complex]): Unnormalized state amplitudes.

    Returns:
        float: Little-endian ``<Y_0>`` expectation value.
    """
    state = np.asarray(amplitudes, dtype=complex)
    state /= np.linalg.norm(state)
    return float(
        sum(
            2.0 * np.imag(np.conjugate(state[index]) * state[index + 1])
            for index in range(0, len(state), 2)
        )
    )


def _assert_sample_and_expval(
    sdk_transpiler,
    api: EncodingApi,
    amplitudes: list[complex],
) -> None:
    """Transpile and execute sampling and expval paths for one backend.

    Args:
        sdk_transpiler: Supported-backend fixture case.
        api (EncodingApi): Public API variant to exercise.
        amplitudes (list[complex]): Unnormalized state amplitudes.
    """
    transpiler = sdk_transpiler.transpiler
    executor = transpiler.executor()
    expected_probabilities = (
        np.abs(np.asarray(amplitudes, dtype=complex) / np.linalg.norm(amplitudes)) ** 2
    )

    sample_executable = transpiler.transpile(_build_sample_kernel(api, amplitudes))
    samples = sample_executable.sample(executor, shots=_SHOTS).result().results
    observed_probabilities = np.zeros_like(expected_probabilities)
    total = 0
    for bits, count in samples:
        observed_probabilities[_bits_to_index(bits)] = count / _SHOTS
        total += count
    assert total == _SHOTS

    for index, expected_probability in enumerate(expected_probabilities):
        variance = max(
            expected_probability * (1.0 - expected_probability), 1.0 / _SHOTS
        )
        tolerance = _STD_TOLERANCE * np.sqrt(variance / _SHOTS)
        assert observed_probabilities[index] == pytest.approx(
            expected_probability, abs=tolerance
        )

    n_qubits = int(round(np.log2(len(amplitudes))))
    observable = qm_o.Y(0) + qm_o.Hamiltonian.zero(num_qubits=n_qubits)
    expval_executable = transpiler.transpile(
        _build_expval_kernel(api, amplitudes), bindings={"H": observable}
    )
    result = expval_executable.run(executor).result()
    assert float(result) == pytest.approx(_expected_y0(amplitudes), abs=1e-8)


@pytest.mark.parametrize("api", _ENCODING_APIS)
@pytest.mark.parametrize(("n_qubits", "seed"), _RANDOM_CASES)
def test_randomized_sampling_and_expval_on_every_backend(
    sdk_transpiler,
    api: EncodingApi,
    n_qubits: int,
    seed: int,
) -> None:
    """Both APIs execute randomized sampling and expval on every backend."""
    amplitudes = _random_complex_amplitudes(n_qubits, seed)
    _assert_sample_and_expval(sdk_transpiler, api, amplitudes)


@pytest.mark.parametrize("api", _ENCODING_APIS)
def test_basis_state_boundary_on_every_backend(
    sdk_transpiler, api: EncodingApi
) -> None:
    """Both APIs execute a sparse computational-basis boundary case."""
    _assert_sample_and_expval(
        sdk_transpiler,
        api,
        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
    )


def test_generic_api_accepts_transpile_time_amplitude_bindings(
    qiskit_transpiler,
) -> None:
    """The existing generic API keeps its bound ``Vector[Float]`` behavior."""
    executable = qiskit_transpiler.transpile(
        _generic_bound_kernel,
        bindings={"amplitudes": [0.0, 0.0, 0.0, 1.0]},
    )
    samples = executable.sample(qiskit_transpiler.executor(), shots=128).result()
    assert (
        sum(count for bits, count in samples.results if _bits_to_index(bits) == 3)
        == 128
    )


def test_generic_api_rejects_runtime_amplitude_parameters(
    qiskit_transpiler,
) -> None:
    """The generic API still directs symbolic amplitudes to the angle API."""
    with pytest.raises(
        ValueError,
        match="mottonen_amplitude_encoding_from_angles",
    ):
        qiskit_transpiler.transpile(
            _generic_bound_kernel,
            parameters=["amplitudes"],
        )
