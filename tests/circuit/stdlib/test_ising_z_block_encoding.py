"""Tests for dedicated Ising-Z block encodings."""

from __future__ import annotations

import dataclasses
import importlib
import math
from typing import Any

import numpy as np
import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o


@qmc.qkernel
def _identity_case(
    signal: qmc.Vector[qmc.Qubit],
    system: qmc.Vector[qmc.Qubit],
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
    """Return complete signal and system registers unchanged."""
    return signal, system


def _executor(case: Any) -> Any:
    """Return a local simulator executor for one SDK fixture case."""
    if case.backend_name == "qiskit":
        from qiskit.providers.basic_provider import BasicSimulator

        return case.transpiler.executor(backend=BasicSimulator())
    return case.transpiler.executor()


def _flatten(value: Any) -> tuple[int, ...]:
    """Flatten a nested measurement value into integer bits."""
    if isinstance(value, tuple):
        return tuple(bit for item in value for bit in _flatten(item))
    return (int(value),)


def _zero_probability(results: list[tuple[Any, int]]) -> float:
    """Return the observed probability of an all-zero result."""
    total = sum(count for _, count in results)
    zero = sum(
        count
        for outcome, count in results
        if all(bit == 0 for bit in _flatten(outcome))
    )
    return zero / total


def _ising_matrix(
    coefficients: dict[tuple[int, ...], complex],
    num_system_qubits: int,
) -> np.ndarray:
    """Return the dense diagonal matrix represented by Ising-Z words."""
    diagonal = np.zeros(1 << num_system_qubits, dtype=np.complex128)
    for basis in range(1 << num_system_qubits):
        for word, coefficient in coefficients.items():
            parity = sum((basis >> index) & 1 for index in word) & 1
            diagonal[basis] += coefficient * (-1.0 if parity else 1.0)
    return np.diag(diagonal)


def _build_unitary_kernel(
    encoding: qmc.IsingZBlockEncoding,
    *,
    invert: bool = False,
) -> qmc.QKernel:
    """Build an allocation-only kernel exposing an Ising-Z unitary."""
    applied = qmc.inverse(encoding.kernel) if invert else encoding.kernel

    @qmc.qkernel
    def kernel() -> qmc.Bit:
        """Allocate the registers and apply the selected direction."""
        signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
        system = qmc.qubit_array(encoding.num_system_qubits, "system")
        signal, _ = applied(signal, system)
        return qmc.measure(signal[0])

    return kernel


def _qiskit_unitary(
    encoding: qmc.IsingZBlockEncoding,
    *,
    invert: bool = False,
) -> np.ndarray:
    """Return the exact dense unitary emitted through Qiskit."""
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Operator

    from qamomile.qiskit import QiskitTranspiler

    executable = QiskitTranspiler().transpile(
        _build_unitary_kernel(encoding, invert=invert)
    )
    circuit = executable.quantum_circuit.remove_final_measurements(inplace=False)
    return np.asarray(Operator(circuit).data)


def _top_left_block(
    unitary: np.ndarray,
    encoding: qmc.IsingZBlockEncoding,
) -> np.ndarray:
    """Extract the all-zero-signal block in Qamomile LSB order."""
    indices = [
        basis << encoding.num_signal_qubits
        for basis in range(1 << encoding.num_system_qubits)
    ]
    return unitary[np.ix_(indices, indices)]


def test_descriptor_is_frozen_noncallable_and_identity_based() -> None:
    """The Ising-Z descriptor follows the static four-field convention."""
    encoding = qmc.ising_z_block_encoding({(): 1.0}, 1)
    repeated = qmc.ising_z_block_encoding({(): 1.0}, 1)

    assert tuple(field.name for field in dataclasses.fields(encoding)) == (
        "kernel",
        "normalization",
        "num_signal_qubits",
        "num_system_qubits",
    )
    assert tuple(encoding.kernel.signature.parameters) == ("signal", "system")
    assert encoding.normalization == 1.0
    assert encoding.num_signal_qubits == 1
    assert encoding.num_system_qubits == 1
    assert not callable(encoding)
    assert not hasattr(encoding, "__dict__")
    assert encoding is not repeated
    assert encoding != repeated

    with pytest.raises(dataclasses.FrozenInstanceError):
        encoding.normalization = 2.0  # type: ignore[misc]


def test_factory_validates_mapping_coefficients_words_and_width() -> None:
    """The dedicated factory rejects values outside its finite Ising domain."""
    with pytest.raises(TypeError, match="mapping"):
        qmc.ising_z_block_encoding([], 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="tuple"):
        qmc.ising_z_block_encoding({"0": 1.0}, 1)  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="integers"):
        qmc.ising_z_block_encoding({(True,): 1.0}, 1)
    with pytest.raises(TypeError, match="coefficient"):
        qmc.ising_z_block_encoding({(): True}, 1)
    with pytest.raises(ValueError, match="outside"):
        qmc.ising_z_block_encoding({(1,): 1.0}, 1)
    with pytest.raises(TypeError, match="coefficient"):
        qmc.ising_z_block_encoding({(): object()}, 1)  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="finite"):
        qmc.ising_z_block_encoding({(): complex(math.inf, 0.0)}, 1)
    with pytest.raises(TypeError, match="num_system_qubits"):
        qmc.ising_z_block_encoding({}, True)
    with pytest.raises(ValueError, match="num_system_qubits"):
        qmc.ising_z_block_encoding({}, 0)


@pytest.mark.parametrize(
    ("coefficients", "num_system_qubits"),
    [
        ({}, 1),
        ({(): -2.0}, 1),
        ({(0,): -2.0}, 1),
        ({(0,): np.exp(1j * math.pi / 3.0)}, 1),
        ({(0, 1, 0): 1.0 + 2.0j, (1,): 2.0 - 2.0j}, 2),
        ({(): 0.5j, (0,): -0.25, (0, 1): 0.75 - 0.1j}, 2),
    ],
    ids=[
        "zero",
        "negative_identity",
        "negative_z",
        "complex_z",
        "repeated_index_aggregation",
        "multi_term",
    ],
)
def test_zero_single_complex_repeated_and_multi_terms_have_exact_blocks(
    coefficients: dict[tuple[int, ...], complex],
    num_system_qubits: int,
) -> None:
    """All term-count paths encode the requested diagonal operator exactly."""
    encoding = qmc.ising_z_block_encoding(coefficients, num_system_qubits)
    matrix = _ising_matrix(coefficients, num_system_qubits)

    np.testing.assert_allclose(
        _top_left_block(_qiskit_unitary(encoding), encoding),
        matrix / encoding.normalization,
        atol=1e-10,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        _top_left_block(_qiskit_unitary(encoding, invert=True), encoding),
        matrix.conj().T / encoding.normalization,
        atol=1e-10,
        rtol=0.0,
    )
    if not coefficients:
        assert encoding.normalization == 1.0
        assert encoding.num_signal_qubits == 1


def test_canonical_aggregation_is_mapping_order_invariant() -> None:
    """Equivalent words aggregate independently of mapping insertion order."""
    entries = [
        ((1,), 2.0 - 2.0j),
        ((0, 1, 0), 1.0 + 2.0j),
        ((0, 0), -0.0 + 0.0j),
    ]
    forward = qmc.ising_z_block_encoding(dict(entries), 2)
    reverse = qmc.ising_z_block_encoding(dict(reversed(entries)), 2)

    assert forward.normalization == reverse.normalization == 3.0
    assert forward.num_signal_qubits == reverse.num_signal_qubits == 1
    np.testing.assert_allclose(
        _top_left_block(_qiskit_unitary(forward), forward),
        _top_left_block(_qiskit_unitary(reverse), reverse),
        atol=1e-10,
        rtol=0.0,
    )


def test_single_term_preserves_every_signal_state_in_the_full_unitary() -> None:
    """The one-term optimization leaves its positive-width signal untouched."""
    phase = math.pi / 3.0
    encoding = qmc.ising_z_block_encoding({(0,): np.exp(1j * phase)}, 1)
    z_matrix = np.diag([1.0, -1.0]).astype(np.complex128)
    expected = np.exp(1j * phase) * np.kron(z_matrix, np.eye(2))

    np.testing.assert_allclose(
        _qiskit_unitary(encoding),
        expected,
        atol=1e-10,
        rtol=0.0,
    )


def test_equivalent_words_can_cancel_to_the_exact_zero_path() -> None:
    """Canonical-word cancellation selects the explicit zero encoding."""
    encoding = qmc.ising_z_block_encoding({(0,): 1.0, (0, 0, 0): -1.0}, 1)

    assert encoding.normalization == 1.0
    assert encoding.num_signal_qubits == 1
    np.testing.assert_allclose(
        _top_left_block(_qiskit_unitary(encoding), encoding),
        np.zeros((2, 2)),
        atol=1e-10,
    )


def test_factory_does_not_delegate_to_pauli_lcu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The specialized Ising-Z path is independent of PauliLCUEncoding."""
    module = importlib.import_module("qamomile.circuit.stdlib.ising_z_block_encoding")

    assert "PauliLCU" not in module.__dict__
    assert "pauli_lcu_block_encoding" not in module.__dict__

    def fail_if_called(*args: object, **kwargs: object) -> None:
        """Fail if the generic Pauli LCU factory is unexpectedly invoked."""
        del args, kwargs
        raise AssertionError("Ising-Z construction delegated to PauliLCUEncoding")

    monkeypatch.setattr(qmc, "pauli_lcu_block_encoding", fail_if_called)
    encoding = qmc.ising_z_block_encoding({(): 0.25, (0,): -1.0}, 1)
    assert encoding.normalization == 1.25


def test_register_width_errors_survive_inverse_control_and_outer_select() -> None:
    """Every composition path retains the Ising-Z signal-width diagnostic."""
    encoding = qmc.ising_z_block_encoding({(): 1.0, (0,): 0.5}, 1)
    inverse = qmc.inverse(encoding.kernel)
    controlled = qmc.control(encoding.kernel)
    selected = qmc.select((_identity_case, encoding.kernel), num_index_qubits=1)

    @qmc.qkernel
    def plain() -> qmc.Bit:
        """Call the encoding with a wide signal register."""
        signal = qmc.qubit_array(2, "signal")
        system = qmc.qubit_array(1, "system")
        signal, _ = encoding.kernel(signal, system)
        return qmc.measure(signal[0])

    @qmc.qkernel
    def inverted() -> qmc.Bit:
        """Call the inverse with a wide signal register."""
        signal = qmc.qubit_array(2, "signal")
        system = qmc.qubit_array(1, "system")
        signal, _ = inverse(signal, system)
        return qmc.measure(signal[0])

    @qmc.qkernel
    def controlled_call() -> qmc.Bit:
        """Call the controlled encoding with a wide signal register."""
        control = qmc.qubit("control")
        signal = qmc.qubit_array(2, "signal")
        system = qmc.qubit_array(1, "system")
        control, signal, _ = controlled(control, signal, system)
        return qmc.measure(control)

    @qmc.qkernel
    def selected_call() -> qmc.Bit:
        """Call an outer SELECT with a wide signal register."""
        outer = qmc.qubit_array(1, "outer")
        signal = qmc.qubit_array(2, "signal")
        system = qmc.qubit_array(1, "system")
        outer, signal, _ = selected(outer, signal, system)
        return qmc.measure(outer[0])

    for kernel in (plain, inverted, controlled_call, selected_call):
        with pytest.raises(ValueError, match=r"requires 1 signal qubit, got 2"):
            kernel.build()


@pytest.mark.parametrize("seed", [0, 1, 2, 42])
def test_random_complex_ising_lcu_samples_and_estimates_on_every_sdk(
    sdk_transpiler: Any,
    seed: int,
) -> None:
    """Random Ising coefficients execute through sampler and estimator paths."""
    rng = np.random.default_rng(seed)
    identity_weight = complex(*rng.uniform(-1.0, 1.0, size=2))
    z_weight = complex(*rng.uniform(-1.0, 1.0, size=2))
    basis = seed & 1
    encoding = qmc.ising_z_block_encoding(
        {(): identity_weight, (0,): z_weight},
        1,
    )
    signed_z = 1.0 if basis == 0 else -1.0
    success = abs(identity_weight + signed_z * z_weight) ** 2
    success /= encoding.normalization**2

    @qmc.qkernel
    def sample_kernel() -> qmc.Bit:
        """Apply the Ising-Z encoding and measure its success signal."""
        signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
        system = qmc.qubit_array(1, "system")
        if basis:
            system[0] = qmc.x(system[0])
        signal, _ = encoding.kernel(signal, system)
        return qmc.measure(signal[0])

    @qmc.qkernel
    def expval_kernel(observable: qmc.Observable) -> qmc.Float:
        """Estimate signal Z after the Ising-Z encoding."""
        signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
        system = qmc.qubit_array(1, "system")
        if basis:
            system[0] = qmc.x(system[0])
        signal, _ = encoding.kernel(signal, system)
        return qmc.expval(signal[0], observable)

    shots = 2048
    sample = sdk_transpiler.transpiler.transpile(sample_kernel)
    sample_result = sample.sample(_executor(sdk_transpiler), shots=shots).result()
    tolerance = 6.0 * math.sqrt(success * (1.0 - success) / shots) + 0.02
    assert _zero_probability(sample_result.results) == pytest.approx(
        success,
        abs=tolerance,
    )

    expval = sdk_transpiler.transpiler.transpile(
        expval_kernel,
        bindings={"observable": qm_o.Z(0)},
    )
    observed = float(expval.run(_executor(sdk_transpiler)).result())
    atol = 1e-6 if sdk_transpiler.backend_name == "cudaq" else 1e-8
    assert observed == pytest.approx(2.0 * success - 1.0, abs=atol)


def test_zero_encoding_samples_and_estimates_on_every_sdk(
    sdk_transpiler: Any,
) -> None:
    """The explicit zero path flips signal and executes on every backend."""
    encoding = qmc.ising_z_block_encoding({}, 1)

    @qmc.qkernel
    def sample_kernel() -> qmc.Bit:
        """Apply the zero encoding and measure its signal."""
        signal = qmc.qubit_array(1, "signal")
        system = qmc.qubit_array(1, "system")
        signal, _ = encoding.kernel(signal, system)
        return qmc.measure(signal[0])

    @qmc.qkernel
    def expval_kernel(observable: qmc.Observable) -> qmc.Float:
        """Apply the zero encoding and estimate signal Z."""
        signal = qmc.qubit_array(1, "signal")
        system = qmc.qubit_array(1, "system")
        signal, _ = encoding.kernel(signal, system)
        return qmc.expval(signal[0], observable)

    sample = sdk_transpiler.transpiler.transpile(sample_kernel)
    result = sample.sample(_executor(sdk_transpiler), shots=64).result()
    assert result.results == [(1, 64)]

    expval = sdk_transpiler.transpiler.transpile(
        expval_kernel,
        bindings={"observable": qm_o.Z(0)},
    )
    observed = float(expval.run(_executor(sdk_transpiler)).result())
    assert observed == pytest.approx(-1.0, abs=1e-8)
