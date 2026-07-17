"""Tests for recursively composed LCU block encodings."""

from __future__ import annotations

import dataclasses
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
    """Return both block-encoding registers unchanged."""
    return signal, system


@qmc.qkernel
def _two_signal_z_case(
    signal: qmc.Vector[qmc.Qubit],
    system: qmc.Vector[qmc.Qubit],
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
    """Apply Z while preserving a two-qubit signal ABI."""
    system[0] = qmc.z(system[0])
    return signal, system


@qmc.qkernel
def _flip_child_signal_case(
    signal: qmc.Vector[qmc.Qubit],
    system: qmc.Vector[qmc.Qubit],
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
    """Flip one child signal while preserving its system register."""
    signal[0] = qmc.x(signal[0])
    return signal, system


@qmc.qkernel
def _invalid_scalar_kernel(
    signal: qmc.Qubit,
    system: qmc.Qubit,
) -> tuple[qmc.Qubit, qmc.Qubit]:
    """Expose an intentionally invalid scalar block-encoding ABI."""
    return signal, system


def _executor(case: Any) -> Any:
    """Return a local executor for one cross-backend fixture case."""
    if case.backend_name == "qiskit":
        from qiskit.providers.basic_provider import BasicSimulator

        return case.transpiler.executor(backend=BasicSimulator())
    return case.transpiler.executor()


def _flatten(value: Any) -> tuple[int, ...]:
    """Flatten a nested measurement result into integer bits."""
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


def _z_encoding(*, num_signal_qubits: int = 2) -> qmc.LCUBlockEncoding:
    """Return a hand-written Z leaf with a configurable physical signal width."""
    return qmc.LCUBlockEncoding(
        _two_signal_z_case,
        1.0,
        num_signal_qubits,
        1,
    )


def _build_unitary_kernel(
    encoding: qmc.LCUBlockEncoding,
    *,
    invert: bool = False,
) -> qmc.QKernel:
    """Build an allocation-only kernel exposing an encoding unitary."""
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
    encoding: qmc.LCUBlockEncoding,
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
    encoding: qmc.LCUBlockEncoding,
) -> np.ndarray:
    """Extract the all-zero-signal block in Qamomile LSB order."""
    indices = [
        basis << encoding.num_signal_qubits
        for basis in range(1 << encoding.num_system_qubits)
    ]
    return unitary[np.ix_(indices, indices)]


def _recursive_encoding() -> tuple[qmc.LCUBlockEncoding, np.ndarray]:
    """Build a two-level heterogeneous LCU and its represented matrix."""
    identity = qmc.identity_block_encoding(1)
    inner_i = 0.6 + 0.2j
    inner_z = -0.3 + 0.4j
    inner = qmc.lcu_block_encoding(
        (
            qmc.LCUBlockEncodingTerm(inner_i, identity),
            qmc.LCUBlockEncodingTerm(inner_z, _z_encoding()),
        )
    )
    outer_inner = -0.7 + 0.1j
    outer_i = 0.25j
    outer = qmc.lcu_block_encoding(
        (
            qmc.LCUBlockEncodingTerm(outer_inner, inner),
            qmc.LCUBlockEncodingTerm(outer_i, identity),
        )
    )
    represented = outer_inner * np.diag(
        [inner_i + inner_z, inner_i - inner_z]
    ) + outer_i * np.eye(2, dtype=np.complex128)
    return outer, represented


def test_descriptor_and_term_are_frozen_noncallable_identity_objects() -> None:
    """Descriptors expose the static four-field ABI without value equality."""
    encoding = qmc.identity_block_encoding(1)
    repeated = qmc.identity_block_encoding(1)
    term = qmc.LCUBlockEncodingTerm(1.0 + 2.0j, encoding)

    assert tuple(field.name for field in dataclasses.fields(encoding)) == (
        "kernel",
        "normalization",
        "num_signal_qubits",
        "num_system_qubits",
    )
    assert tuple(field.name for field in dataclasses.fields(term)) == (
        "coefficient",
        "encoding",
    )
    assert tuple(encoding.kernel.signature.parameters) == ("signal", "system")
    assert encoding.normalization == 1.0
    assert encoding.num_signal_qubits == 1
    assert encoding.num_system_qubits == 1
    assert term.coefficient == 1.0 + 2.0j
    assert term.encoding is encoding
    assert not callable(encoding)
    assert not hasattr(encoding, "__dict__")
    assert encoding is not repeated
    assert encoding != repeated

    with pytest.raises(dataclasses.FrozenInstanceError):
        encoding.normalization = 2.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        term.coefficient = 3.0  # type: ignore[misc]


@pytest.mark.parametrize("normalization", [0.0, -1.0, math.inf, math.nan])
def test_descriptor_rejects_invalid_normalization(normalization: float) -> None:
    """The descriptor requires a finite positive normalization."""
    with pytest.raises(ValueError, match="normalization"):
        qmc.LCUBlockEncoding(_identity_case, normalization, 1, 1)


def test_descriptor_and_term_reject_invalid_values() -> None:
    """Static metadata, kernel ABI, coefficients, and children are validated."""
    identity = qmc.identity_block_encoding(1)

    with pytest.raises(TypeError, match="kernel"):
        qmc.LCUBlockEncoding(object(), 1.0, 1, 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="signature"):
        qmc.LCUBlockEncoding(_invalid_scalar_kernel, 1.0, 1, 1)
    with pytest.raises(TypeError, match="num_signal_qubits"):
        qmc.LCUBlockEncoding(_identity_case, 1.0, True, 1)
    with pytest.raises(ValueError, match="num_signal_qubits"):
        qmc.LCUBlockEncoding(_identity_case, 1.0, 0, 1)
    with pytest.raises(ValueError, match="num_system_qubits"):
        qmc.LCUBlockEncoding(_identity_case, 1.0, 1, 0)
    with pytest.raises(ValueError, match="coefficient"):
        qmc.LCUBlockEncodingTerm(math.inf, identity)
    with pytest.raises(TypeError, match="coefficient"):
        qmc.LCUBlockEncodingTerm(True, identity)
    with pytest.raises(TypeError, match="encoding"):
        qmc.LCUBlockEncodingTerm(1.0, object())


def test_factory_rejects_invalid_terms_and_mixed_system_widths() -> None:
    """An ordered composition requires terms over one common system register."""
    one_qubit = qmc.identity_block_encoding(1)
    two_qubit = qmc.identity_block_encoding(2)

    with pytest.raises(ValueError, match="at least one term"):
        qmc.lcu_block_encoding(())
    with pytest.raises(TypeError, match="LCUBlockEncodingTerm"):
        qmc.lcu_block_encoding((object(),))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="system"):
        qmc.lcu_block_encoding(
            (
                qmc.LCUBlockEncodingTerm(1.0, one_qubit),
                qmc.LCUBlockEncodingTerm(1.0, two_qubit),
            )
        )


def test_identity_zero_and_single_term_paths_have_exact_blocks() -> None:
    """Degenerate paths keep a positive signal ABI and preserve term phase."""
    identity = qmc.identity_block_encoding(1)
    zero = qmc.lcu_block_encoding((qmc.LCUBlockEncodingTerm(0.0, identity),))
    single = qmc.lcu_block_encoding((qmc.LCUBlockEncodingTerm(-2.0j, identity),))

    assert identity.normalization == 1.0
    assert zero.normalization == 1.0
    assert single.normalization == 2.0
    assert identity.num_signal_qubits == zero.num_signal_qubits == 1
    assert single.num_signal_qubits == identity.num_signal_qubits
    identity_unitary = _qiskit_unitary(identity)
    single_unitary = _qiskit_unitary(single)
    np.testing.assert_allclose(identity_unitary, np.eye(4), atol=1e-10)
    np.testing.assert_allclose(single_unitary, -1j * np.eye(4), atol=1e-10)
    np.testing.assert_allclose(_top_left_block(identity_unitary, identity), np.eye(2))
    np.testing.assert_allclose(
        _top_left_block(_qiskit_unitary(zero), zero),
        np.zeros((2, 2)),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        _top_left_block(single_unitary, single),
        -1j * np.eye(2),
        atol=1e-10,
    )


def test_uniform_case_preserves_unused_child_pool_padding_exactly() -> None:
    """A heterogeneous SELECT wrapper acts as identity on every unused wire."""
    from qamomile.circuit.stdlib.lcu_block_encoding import (
        _build_uniform_case,
        _ValidatedTerm,
    )

    child = qmc.LCUBlockEncoding(_flip_child_signal_case, 1.0, 1, 1)
    term = _ValidatedTerm(1.0 + 0.0j, child, 1.0, 1, 1)
    case = _build_uniform_case(term, child_pool_width=2)
    assert tuple(case.signature.parameters) == ("signal", "system")

    @qmc.qkernel
    def kernel() -> qmc.Bit:
        """Expose the uniform wrapper as a three-qubit unitary."""
        signal = qmc.qubit_array(2, "signal")
        system = qmc.qubit_array(1, "system")
        signal, _ = case(signal, system)
        return qmc.measure(signal[0])

    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Operator

    from qamomile.qiskit import QiskitTranspiler

    executable = QiskitTranspiler().transpile(kernel)
    circuit = executable.quantum_circuit.remove_final_measurements(inplace=False)
    observed = np.asarray(Operator(circuit).data)
    expected = np.kron(np.eye(4, dtype=np.complex128), np.array([[0, 1], [1, 0]]))
    np.testing.assert_allclose(observed, expected, atol=1e-10, rtol=0.0)


@pytest.mark.parametrize("seed", [0, 1, 2, 42])
def test_multi_term_lcu_has_lambda_weights_and_exact_complex_block(seed: int) -> None:
    """Random complex weights obey Lambda and preserve relative coefficient phase."""
    rng = np.random.default_rng(seed)
    coefficient_i = complex(*rng.uniform(-1.0, 1.0, size=2))
    coefficient_z = complex(*rng.uniform(-1.0, 1.0, size=2))
    identity = qmc.identity_block_encoding(1)
    z_encoding = _z_encoding()
    encoding = qmc.lcu_block_encoding(
        (
            qmc.LCUBlockEncodingTerm(coefficient_i, identity),
            qmc.LCUBlockEncodingTerm(coefficient_z, z_encoding),
        )
    )

    expected_lambda = abs(coefficient_i) + abs(coefficient_z)
    expected = np.diag([coefficient_i + coefficient_z, coefficient_i - coefficient_z])
    assert encoding.normalization == pytest.approx(expected_lambda)
    assert encoding.num_signal_qubits == 3
    np.testing.assert_allclose(
        _top_left_block(_qiskit_unitary(encoding), encoding),
        expected / expected_lambda,
        atol=1e-10,
        rtol=0.0,
    )


def test_two_level_recursion_and_inverse_have_exact_projected_blocks() -> None:
    """An LCU descriptor remains a valid child and its inverse exposes A dagger."""
    encoding, matrix = _recursive_encoding()

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


def test_shifted_ising_composes_recursively_with_child_normalizations() -> None:
    """The intended Ising shift workflow uses each child's non-unit alpha."""
    h_coefficients = {
        (): 0.7,
        (0,): -1.2,
        (0, 1): 0.4j,
    }
    h_encoding = qmc.ising_z_block_encoding(h_coefficients, 2)
    identity = qmc.identity_block_encoding(2)
    mu = 0.35
    shifted = qmc.lcu_block_encoding(
        (
            qmc.LCUBlockEncodingTerm(1.0, h_encoding),
            qmc.LCUBlockEncodingTerm(-mu, identity),
        )
    )
    nested_h_weight = -0.25 + 0.1j
    nested_shift_weight = 0.5j
    nested = qmc.lcu_block_encoding(
        (
            qmc.LCUBlockEncodingTerm(nested_shift_weight, shifted),
            qmc.LCUBlockEncodingTerm(nested_h_weight, h_encoding),
        )
    )

    h_matrix = np.zeros((4, 4), dtype=np.complex128)
    for basis in range(4):
        z0 = 1.0 if not basis & 1 else -1.0
        z1 = 1.0 if not basis & 2 else -1.0
        h_matrix[basis, basis] = 0.7 - 1.2 * z0 + 0.4j * z0 * z1
    shifted_matrix = h_matrix - mu * np.eye(4, dtype=np.complex128)
    nested_matrix = nested_shift_weight * shifted_matrix + nested_h_weight * h_matrix

    assert h_encoding.normalization == pytest.approx(2.3)
    assert shifted.normalization == pytest.approx(h_encoding.normalization + mu)
    assert nested.normalization == pytest.approx(
        abs(nested_shift_weight) * shifted.normalization
        + abs(nested_h_weight) * h_encoding.normalization
    )
    np.testing.assert_allclose(
        _top_left_block(_qiskit_unitary(nested), nested),
        nested_matrix / nested.normalization,
        atol=1e-10,
        rtol=0.0,
    )


def test_composite_rejects_wrong_widths_through_all_composition_paths() -> None:
    """Plain, inverse, control, and outer SELECT share width diagnostics."""
    encoding = qmc.lcu_block_encoding(
        (
            qmc.LCUBlockEncodingTerm(1.0, qmc.identity_block_encoding(1)),
            qmc.LCUBlockEncodingTerm(1.0j, _z_encoding()),
        )
    )
    inverse = qmc.inverse(encoding.kernel)
    controlled = qmc.control(encoding.kernel)
    selected = qmc.select((_identity_case, encoding.kernel), num_index_qubits=1)

    @qmc.qkernel
    def plain() -> qmc.Bit:
        """Call the encoding with a short signal."""
        signal = qmc.qubit_array(2, "signal")
        system = qmc.qubit_array(1, "system")
        signal, _ = encoding.kernel(signal, system)
        return qmc.measure(signal[0])

    @qmc.qkernel
    def inverted() -> qmc.Bit:
        """Call the inverse with a short signal."""
        signal = qmc.qubit_array(2, "signal")
        system = qmc.qubit_array(1, "system")
        signal, _ = inverse(signal, system)
        return qmc.measure(signal[0])

    @qmc.qkernel
    def controlled_call() -> qmc.Bit:
        """Call the controlled encoding with a short signal."""
        control = qmc.qubit("control")
        signal = qmc.qubit_array(2, "signal")
        system = qmc.qubit_array(1, "system")
        control, signal, _ = controlled(control, signal, system)
        return qmc.measure(control)

    @qmc.qkernel
    def selected_call() -> qmc.Bit:
        """Call an outer SELECT with a short signal."""
        outer = qmc.qubit_array(1, "outer")
        signal = qmc.qubit_array(2, "signal")
        system = qmc.qubit_array(1, "system")
        outer, signal, _ = selected(outer, signal, system)
        return qmc.measure(outer[0])

    for kernel in (plain, inverted, controlled_call, selected_call):
        with pytest.raises(ValueError, match=r"requires 3 signal qubits, got 2"):
            kernel.build()


def test_recursive_inverse_round_trip_samples_and_estimates_on_every_sdk(
    sdk_transpiler: Any,
) -> None:
    """Recursive U followed by U dagger restores arbitrary signal and system bits."""
    encoding, _ = _recursive_encoding()
    inverse = qmc.inverse(encoding.kernel)
    marked_signal = encoding.num_signal_qubits - 1

    @qmc.qkernel
    def sample_kernel() -> tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]:
        """Round-trip a nonzero state and measure every logical wire."""
        signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
        system = qmc.qubit_array(1, "system")
        signal[marked_signal] = qmc.x(signal[marked_signal])
        system[0] = qmc.x(system[0])
        signal, system = encoding.kernel(signal, system)
        signal, system = inverse(signal, system)
        return qmc.measure(signal), qmc.measure(system)

    @qmc.qkernel
    def expval_kernel(observable: qmc.Observable) -> qmc.Float:
        """Estimate a restored nonzero signal bit after the round trip."""
        signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
        system = qmc.qubit_array(1, "system")
        signal[marked_signal] = qmc.x(signal[marked_signal])
        system[0] = qmc.x(system[0])
        signal, system = encoding.kernel(signal, system)
        signal, _ = inverse(signal, system)
        return qmc.expval(signal[marked_signal], observable)

    sample = sdk_transpiler.transpiler.transpile(sample_kernel)
    result = sample.sample(_executor(sdk_transpiler), shots=64).result()
    expected_bits = (0,) * marked_signal + (1, 1)
    assert all(_flatten(outcome) == expected_bits for outcome, _ in result.results)
    assert sum(count for _, count in result.results) == 64

    expval = sdk_transpiler.transpiler.transpile(
        expval_kernel,
        bindings={"observable": qm_o.Z(0)},
    )
    observed = float(expval.run(_executor(sdk_transpiler)).result())
    atol = 1e-6 if sdk_transpiler.backend_name == "cudaq" else 1e-8
    assert observed == pytest.approx(-1.0, abs=atol)


def test_recursive_lcu_control_samples_and_estimates_on_every_sdk(
    sdk_transpiler: Any,
) -> None:
    """Two-level LCU recursion executes through control, sample, and estimator."""
    encoding, matrix = _recursive_encoding()
    overlap = matrix[0, 0] / encoding.normalization
    controlled = qmc.control(encoding.kernel)

    @qmc.qkernel
    def sample_kernel() -> qmc.Bit:
        """Run a Hadamard test for the recursive encoding's real overlap."""
        outer = qmc.h(qmc.qubit("outer"))
        signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
        system = qmc.qubit_array(1, "system")
        outer, signal, system = controlled(outer, signal, system)
        outer = qmc.h(outer)
        return qmc.measure(outer)

    @qmc.qkernel
    def expval_kernel(observable: qmc.Observable) -> qmc.Float:
        """Estimate the recursive encoding's imaginary overlap."""
        outer = qmc.h(qmc.qubit("outer"))
        signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
        system = qmc.qubit_array(1, "system")
        outer, signal, system = controlled(outer, signal, system)
        return qmc.expval(outer, observable)

    shots = 2048
    sample = sdk_transpiler.transpiler.transpile(sample_kernel)
    sample_result = sample.sample(_executor(sdk_transpiler), shots=shots).result()
    expected_zero = (1.0 + float(np.real(overlap))) / 2.0
    tolerance = 6.0 * math.sqrt(expected_zero * (1.0 - expected_zero) / shots) + 0.02
    assert _zero_probability(sample_result.results) == pytest.approx(
        expected_zero,
        abs=tolerance,
    )

    expval = sdk_transpiler.transpiler.transpile(
        expval_kernel,
        bindings={"observable": qm_o.Y(0)},
    )
    observed = float(expval.run(_executor(sdk_transpiler)).result())
    atol = 1e-6 if sdk_transpiler.backend_name == "cudaq" else 1e-8
    assert observed == pytest.approx(float(np.imag(overlap)), abs=atol)


def test_outer_select_of_lcu_samples_and_estimates_phase_on_every_sdk(
    sdk_transpiler: Any,
) -> None:
    """A generic encoding remains a phase-bearing case of another SELECT."""
    phase = math.pi / 3.0
    encoding = qmc.lcu_block_encoding(
        (
            qmc.LCUBlockEncodingTerm(
                np.exp(1j * phase),
                qmc.identity_block_encoding(1),
            ),
        )
    )
    selected = qmc.select((_identity_case, encoding.kernel), num_index_qubits=1)

    @qmc.qkernel
    def sample_kernel() -> qmc.Bit:
        """Interfere the outer identity and phased encoding cases."""
        outer = qmc.qubit_array(1, "outer")
        outer[0] = qmc.h(outer[0])
        signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
        system = qmc.qubit_array(1, "system")
        outer, signal, system = selected(outer, signal, system)
        outer[0] = qmc.h(outer[0])
        return qmc.measure(outer[0])

    @qmc.qkernel
    def expval_kernel(observable: qmc.Observable) -> qmc.Float:
        """Estimate the phase kickback on the outer selector."""
        outer = qmc.qubit_array(1, "outer")
        outer[0] = qmc.h(outer[0])
        signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
        system = qmc.qubit_array(1, "system")
        outer, signal, system = selected(outer, signal, system)
        return qmc.expval(outer[0], observable)

    shots = 2048
    sample = sdk_transpiler.transpiler.transpile(sample_kernel)
    sample_result = sample.sample(_executor(sdk_transpiler), shots=shots).result()
    expected_zero = (1.0 + math.cos(phase)) / 2.0
    tolerance = 6.0 * math.sqrt(expected_zero * (1.0 - expected_zero) / shots) + 0.02
    assert _zero_probability(sample_result.results) == pytest.approx(
        expected_zero,
        abs=tolerance,
    )

    expval = sdk_transpiler.transpiler.transpile(
        expval_kernel,
        bindings={"observable": qm_o.Y(0)},
    )
    observed = float(expval.run(_executor(sdk_transpiler)).result())
    atol = 1e-6 if sdk_transpiler.backend_name == "cudaq" else 1e-8
    assert observed == pytest.approx(math.sin(phase), abs=atol)
