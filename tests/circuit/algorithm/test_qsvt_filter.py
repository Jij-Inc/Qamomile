"""Tests for the QSVT eigenstate-filtering kernels.

The quantum singular value transformation itself is covered by
``tests/circuit/stdlib/test_qsvt.py``; these tests exercise only the Lin & Tong
wrapper this module adds on top of :func:`qmc.qsvt`: the Hadamard-test projector
and the sampling probe.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.algorithm import (
    eigenstate_filter_probe,
    eigenstate_filter_projector,
)


def _executor(case: Any) -> Any:
    """Return a local simulator executor for one SDK fixture case.

    Args:
        case (Any): SDK fixture bundling a backend name and transpiler.

    Returns:
        Any: Executor bound to a local statevector simulator.
    """
    if case.backend_name == "qiskit":
        from qiskit.providers.basic_provider import BasicSimulator

        return case.transpiler.executor(backend=BasicSimulator())
    return case.transpiler.executor()


def _flatten(value: Any) -> tuple[int, ...]:
    """Flatten a nested measurement value into integer bits.

    Args:
        value (Any): Measurement result, possibly a nested tuple of registers.

    Returns:
        tuple[int, ...]: Flat tuple of measured bits.
    """
    if isinstance(value, tuple):
        return tuple(bit for item in value for bit in _flatten(item))
    return (int(value),)


def _zero_probability(results: list[tuple[Any, int]], num_bits: int) -> float:
    """Return the observed probability that the leading bits are all zero.

    Args:
        results (list[tuple[Any, int]]): Sampled ``(value, count)`` pairs.
        num_bits (int): Number of leading measured bits to require zero.

    Returns:
        float: Fraction of shots whose first ``num_bits`` bits are zero.
    """
    total = sum(count for _, count in results)
    zero = sum(
        count
        for outcome, count in results
        if all(bit == 0 for bit in _flatten(outcome)[:num_bits])
    )
    return zero / total


def _zero_projector(num_qubits: int) -> qm_o.Hamiltonian:
    """Return the Pauli expansion of the all-zero projector.

    Args:
        num_qubits (int): Positive projector-register width.

    Returns:
        qm_o.Hamiltonian: ``product((I + Z_i) / 2)`` on the register.
    """
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


def _qiskit_unitary(kernel: qmc.QKernel, **bindings: Any) -> np.ndarray:
    """Transpile a kernel with Qiskit and return its dense unitary.

    Args:
        kernel (qmc.QKernel): Kernel to compile.
        **bindings (Any): Compile-time bindings forwarded to the transpiler.

    Returns:
        np.ndarray: Unitary of the circuit with measurements removed.
    """
    from qiskit.quantum_info import Operator

    from qamomile.qiskit import QiskitTranspiler

    executable = QiskitTranspiler().transpile(kernel, bindings=bindings or None)
    circuit = executable.quantum_circuit.remove_final_measurements(inplace=False)
    return np.asarray(Operator(circuit).data)


def _top_left_block(
    unitary: np.ndarray, num_ancilla: int, num_system: int
) -> np.ndarray:
    """Extract the block selected by all-zero ancillas.

    Args:
        unitary (np.ndarray): Dense unitary of the full circuit.
        num_ancilla (int): Number of ancilla qubits, which occupy the low bits.
        num_system (int): Number of system qubits.

    Returns:
        np.ndarray: The ``2**num_system`` square good block.
    """
    indices = [state << num_ancilla for state in range(1 << num_system)]
    return unitary[np.ix_(indices, indices)]


def _ising_diagonal(
    coefficients: dict[tuple[int, ...], float], num_system_qubits: int
) -> np.ndarray:
    """Return the eigenvalues of a diagonal Ising-Z operator.

    Args:
        coefficients (dict[tuple[int, ...], float]): Ising-Z coefficients keyed
            by products of system-qubit indices.
        num_system_qubits (int): Width of the system register.

    Returns:
        np.ndarray: Diagonal entries in little-endian basis order.
    """
    diagonal = np.zeros(1 << num_system_qubits)
    for basis in range(1 << num_system_qubits):
        for word, coefficient in coefficients.items():
            parity = sum((basis >> index) & 1 for index in word) & 1
            diagonal[basis] += coefficient * (-1.0 if parity else 1.0)
    return diagonal


def _wx_qsp_polynomial(wx_phases: np.ndarray, x: float) -> complex:
    """Evaluate the Wx-convention QSP product at one signal value.

    This is the textbook reference the QSVT sequence reproduces on a
    single-signal-qubit block encoding: the 2x2 product of signal rotations
    ``W(x)`` and phase rotations, read at its top-left entry.

    Args:
        wx_phases (np.ndarray): Phases in the Wx (signal-rotation) convention.
        x (float): Signal value in ``[-1, 1]``.

    Returns:
        complex: ``<0|U(x)|0>`` for the QSP product.
    """
    sine = math.sqrt(max(0.0, 1.0 - x**2))
    signal = np.array([[x, 1j * sine], [1j * sine, x]])

    def phase(angle: float) -> np.ndarray:
        return np.array([[np.exp(1j * angle), 0.0], [0.0, np.exp(-1j * angle)]])

    product = phase(wx_phases[0])
    for angle in wx_phases[1:]:
        product = product @ signal @ phase(angle)
    return complex(product[0, 0])


def _to_reflection_phases(wx_phases: np.ndarray) -> list[float]:
    """Convert Wx-convention phases to the projector-rotation convention.

    On a single-signal-qubit block encoding, feeding these phases to
    :func:`qmc.qsvt` makes its good block equal ``-i`` times
    :func:`_wx_qsp_polynomial`. This is the test-side inverse of the convention
    ``qmc.qsvt`` documents and is independent of the converter's own phase
    synthesis.

    Args:
        wx_phases (np.ndarray): Phases as produced by ``pyqsp``.

    Returns:
        list[float]: Phases the QSVT primitive consumes.
    """
    phases = np.asarray(wx_phases, dtype=float).copy()
    phases[0] += math.pi / 4
    phases[-1] += math.pi / 4
    phases[1:-1] += math.pi / 2
    return [float(phase) for phase in phases]


def _reference_qsvt(
    encoding_unitary: np.ndarray,
    phases: list[float],
    signal_dimension: int,
) -> np.ndarray:
    """Compose the documented QSVT sequence as a dense unitary.

    Args:
        encoding_unitary (np.ndarray): Dense block-encoding unitary.
        phases (list[float]): Projector phases in sequence order.
        signal_dimension (int): Dimension of the signal register.

    Returns:
        np.ndarray: Dense QSVT unitary in Qiskit qubit order.
    """
    dimension = encoding_unitary.shape[0]
    system_dimension = dimension // signal_dimension

    def projector_rotation(phase: float) -> np.ndarray:
        signal = np.eye(signal_dimension, dtype=np.complex128) * np.exp(-1j * phase)
        signal[0, 0] = np.exp(1j * phase)
        return np.kron(np.eye(system_dimension), signal)

    transformed = projector_rotation(phases[0])
    for index, phase in enumerate(phases[1:]):
        step = encoding_unitary if index % 2 == 0 else encoding_unitary.conj().T
        transformed = projector_rotation(phase) @ step @ transformed
    return transformed


def _encoding_unitary(encoding: qmc.LCUBlockEncoding) -> np.ndarray:
    """Return the dense unitary of a block encoding in Qiskit qubit order.

    Args:
        encoding (qmc.LCUBlockEncoding): Descriptor whose ``unitary`` kernel is
            compiled.

    Returns:
        np.ndarray: Dense block-encoding unitary, signal on the low bits.
    """
    num_signal_qubits = encoding.num_signal_qubits
    num_system_qubits = encoding.num_system_qubits

    @qmc.qkernel
    def encoding_top() -> tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]:
        """Apply the encoding unitary to freshly allocated registers."""
        signal = qmc.qubit_array(num_signal_qubits, "signal")
        system = qmc.qubit_array(num_system_qubits, "system")
        signal, system = encoding.unitary(signal, system)
        return qmc.measure(signal), qmc.measure(system)

    return _qiskit_unitary(encoding_top)


def test_filter_builders_reject_non_descriptors() -> None:
    """Only block-encoding descriptors can drive the filter builders."""
    with pytest.raises(TypeError, match="LCUBlockEncoding"):
        eigenstate_filter_projector(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="LCUBlockEncoding"):
        eigenstate_filter_probe(object())  # type: ignore[arg-type]


@pytest.mark.parametrize("seed", [0, 1, 2, 42])
def test_projector_block_is_the_filtered_half_sum(seed: int) -> None:
    """The Hadamard test realizes (I - R)/2 for the QSVT reflection block R."""
    pytest.importorskip("qiskit")
    rng = np.random.default_rng(seed)
    coefficients = {
        (0,): float(rng.uniform(-1.0, 1.0)),
        (1,): float(rng.uniform(-1.0, 1.0)),
        (0, 1): float(rng.uniform(-1.0, 1.0)),
    }
    phases = rng.uniform(-math.pi, math.pi, size=4).tolist()
    encoding = qmc.ising_z_block_encoding(coefficients, 2)
    projector = eigenstate_filter_projector(encoding)

    @qmc.qkernel
    def top(phi: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
        """Apply the projector to freshly allocated registers."""
        proj = qmc.qubit_array(1, "proj")
        signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
        system = qmc.qubit_array(encoding.num_system_qubits, "system")
        proj, signal, system = projector(proj, signal, system, phi)
        return qmc.measure(proj)

    num_ancilla = 1 + encoding.num_signal_qubits
    block = _top_left_block(_qiskit_unitary(top, phi=phases), num_ancilla, 2)

    reflection = _top_left_block(
        _reference_qsvt(
            _encoding_unitary(encoding),
            phases,
            1 << encoding.num_signal_qubits,
        ),
        encoding.num_signal_qubits,
        2,
    )
    expected = (np.eye(1 << 2) - reflection) / 2.0

    np.testing.assert_allclose(block, expected, atol=1e-8, rtol=0.0)


@pytest.mark.parametrize("num_system_qubits", [1, 2])
@pytest.mark.parametrize("seed", [0, 1, 2, 42])
def test_probe_samples_and_estimates_on_every_sdk(
    sdk_transpiler: Any,
    num_system_qubits: int,
    seed: int,
) -> None:
    """The probe's all-zero ancilla rate matches its exact success probability.

    The encodings here have a single signal qubit, so the QSVT good block is
    ``-i`` times the Wx-convention QSP polynomial and the success probability
    has a closed form. Random phases keep the circuit short (degree 3) while
    exercising the full stack — encoding, inverse encoding, the controlled QSVT
    reflection — on every backend, through both the sampler and the estimator.
    """
    rng = np.random.default_rng(seed)
    # At most two Ising terms keep the encoding to one signal qubit.
    words = ((0,),) if num_system_qubits == 1 else ((0,), (1,))
    coefficients = {word: float(rng.uniform(-1.0, 1.0)) for word in words}
    wx_phases = rng.uniform(-math.pi, math.pi, size=4)
    phases = _to_reflection_phases(wx_phases)
    encoding = qmc.ising_z_block_encoding(coefficients, num_system_qubits)
    assert encoding.num_signal_qubits == 1
    num_ancilla = 1 + encoding.num_signal_qubits
    probe = eigenstate_filter_probe(encoding)

    signal_values = _ising_diagonal(coefficients, num_system_qubits)
    signal_values = signal_values / encoding.normalization
    amplitudes = np.array(
        [(1.0 - (-1j * _wx_qsp_polynomial(wx_phases, x))) / 2.0 for x in signal_values]
    )
    # The probe starts from the uniform superposition, so every eigenstate
    # contributes with weight 1 / 2**n.
    success = float(np.sum(np.abs(amplitudes) ** 2) / (1 << num_system_qubits))

    @qmc.qkernel
    def expval_kernel(
        phi: qmc.Vector[qmc.Float], observable: qmc.Observable
    ) -> qmc.Float:
        """Estimate the all-zero ancilla projector after the filter."""
        ancilla = qmc.qubit_array(num_ancilla, "ancilla")
        system = qmc.qubit_array(encoding.num_system_qubits, "system")
        for index in range(encoding.num_system_qubits):
            system[index] = qmc.h(system[index])
        proj = ancilla[0:1]
        signal = ancilla[1:num_ancilla]
        proj, signal, system = eigenstate_filter_projector(encoding)(
            proj, signal, system, phi
        )
        ancilla[0:1] = proj
        ancilla[1:num_ancilla] = signal
        return qmc.expval(ancilla, observable)

    shots = 4096
    sample = sdk_transpiler.transpiler.transpile(probe, bindings={"phi": phases})
    result = sample.sample(_executor(sdk_transpiler), shots=shots).result()
    tolerance = 6.0 * math.sqrt(success * (1.0 - success) / shots) + 0.02
    assert _zero_probability(result.results, num_ancilla) == pytest.approx(
        success, abs=tolerance
    )

    expval = sdk_transpiler.transpiler.transpile(
        expval_kernel,
        bindings={"phi": phases, "observable": _zero_projector(num_ancilla)},
    )
    observed = float(expval.run(_executor(sdk_transpiler)).result())
    atol = 1e-6 if sdk_transpiler.backend_name == "cudaq" else 1e-8
    assert observed == pytest.approx(success, abs=atol)
