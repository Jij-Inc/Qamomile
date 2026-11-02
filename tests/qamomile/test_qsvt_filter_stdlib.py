"""Real-part extraction, phase certification, and shared filter execution."""

from __future__ import annotations

import math
import sys
from types import ModuleType

import numpy as np
import pytest

import qamomile as qm
from qamomile._native import compiler as native
from qamomile.stdlib import (
    BlockEncoding,
    qsvt_filter_probe,
    qsvt_filter_projector,
    synthesize_qsvt_polynomial,
    synthesize_sign_filter,
)


@qm.qkernel
def z_encoding(state: qm.Q[qm.Bits[2]]) -> qm.Q[qm.Bits[2]]:
    """Encode system Z with an untouched zero signal qubit."""
    return qm.z(state, target=1)


_encoding = BlockEncoding(z_encoding, 1.0, 1, 1)
_projector = qsvt_filter_projector(_encoding, phases=(0.0, 0.0))
_projector_step = _projector.unitary
_probe = qsvt_filter_probe(_encoding, phases=(0.0, 0.0))


@qm.qkernel
def lower_energy_state() -> qm.Q[qm.Bits[4]]:
    """Keep the negative-Z eigenstate with every success flag zero."""
    return _projector_step(qm.prepare((0, 0, 0, 1)))


def exercise_qsvt_filter_backend(*, endpoint: object) -> None:
    """Execute shared sampling and expectation witnesses on a local target."""
    artifact = qm.compile(_probe, to=endpoint.target)
    assert isinstance(artifact, qm.CompiledProgram)
    result = qm.execute(artifact, on=endpoint, shots=64, seed=0)
    assert set(result.values) == {(True, (1,)), (False, (0,))}
    assert result.values[(True, (1,))] / result.shots == pytest.approx(0.5, abs=0.2)
    assert (
        qm.expectation(lower_energy_state, qm.Z(3), on=endpoint, shots=4, seed=0).value
        == -1.0
    )
    assert (
        qm.expectation(
            lower_energy_state,
            qm.Z(0) * qm.Z(1) * qm.Z(2),
            on=endpoint,
            shots=4,
            seed=0,
        ).value
        == 1.0
    )


def test_qsvt_filter_reference_execution() -> None:
    """The filter returns typed success and retains only the lower eigenstate."""
    exercise_qsvt_filter_backend(endpoint=qm.Reference())


def test_projector_extracts_real_polynomial_before_hadamard_test() -> None:
    """Complementary imaginary QSP amplitude never counts as filter success."""
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Operator

    from qamomile.qiskit import Qiskit, to_circuit

    certificate = synthesize_qsvt_polynomial((0.0, 0.2))
    filtered = qsvt_filter_projector(_encoding, polynomial=certificate)
    assert filtered.normalization == 1.0
    assert filtered.signal_width == 3
    assert filtered.system_width == 1
    assert filtered.error_bound == certificate.uniform_error_bound / 2
    transform = filtered.unitary

    @qm.qkernel
    def full() -> qm.Q[qm.Bits[4]]:
        return transform(qm.qubits(4))

    matrix = Operator(to_circuit(full, target=Qiskit())).data
    block = matrix[np.ix_((0, 8), (0, 8))]
    np.testing.assert_allclose(block, np.diag((0.4, 0.6)), atol=2e-13)
    # Omitting real extraction would add 0.24 to each success probability.
    np.testing.assert_allclose(np.abs(np.diag(block)) ** 2, (0.16, 0.36), atol=2e-13)


def test_sign_candidate_is_replayed_by_rust_with_positive_orientation(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """External imaginary-component phases become a checked positive real target."""
    polynomial_module = ModuleType("pyqsp.poly")
    angle_module = ModuleType("pyqsp.angle_sequence")

    class Producer:
        def generate(self, **kwargs):
            print("external polynomial progress")
            return np.array((0.0, 0.2)), 1.0

    def phases(*args, **kwargs):
        print("external phase progress")
        angle = math.asin(0.2) / 2
        return (angle, angle), (), 1

    polynomial_module.PolySign = Producer
    angle_module.QuantumSignalProcessingPhases = phases
    monkeypatch.setitem(sys.modules, "pyqsp", ModuleType("pyqsp"))
    monkeypatch.setitem(sys.modules, "pyqsp.poly", polynomial_module)
    monkeypatch.setitem(sys.modules, "pyqsp.angle_sequence", angle_module)
    certificate = synthesize_sign_filter(degree=1, delta=1.0, scale=1.0)
    assert certificate.chebyshev_coefficients == (0.0, 0.2)
    assert certificate.odd
    assert certificate.uniform_error_bound <= 1e-6
    assert capsys.readouterr().out == ""
    angle_module.QuantumSignalProcessingPhases = lambda *args, **kwargs: (
        (0.0, 0.0),
        (),
        1,
    )
    with pytest.raises(ValueError, match="Q7709"):
        synthesize_sign_filter(degree=1, delta=1.0, scale=1.0)
    monkeypatch.setattr(native, "verify_qsvt_polynomial", None)
    with pytest.raises(TypeError, match="native QSVT polynomial verification"):
        synthesize_sign_filter(degree=1)


@pytest.mark.parametrize(
    "arguments, error",
    [
        ({"degree": True}, TypeError),
        ({"degree": 0}, ValueError),
        ({"degree": 2}, ValueError),
        ({"degree": 257}, ValueError),
        ({"delta": False}, TypeError),
        ({"delta": 0.0}, ValueError),
        ({"delta": math.inf}, ValueError),
        ({"scale": 0.0}, ValueError),
        ({"scale": math.nan}, ValueError),
        ({"max_error": 0.0}, ValueError),
    ],
)
def test_sign_candidate_rejects_invalid_inputs_before_external_work(arguments, error):
    """Malformed and unbounded requests fail before the external producer runs."""
    with pytest.raises(error):
        synthesize_sign_filter(**arguments)


def test_projector_rejects_uncertified_source_error_and_ambiguous_phases() -> None:
    """An input perturbation needs a separate bound, and phase sources are exclusive."""
    inaccurate = BlockEncoding(z_encoding, 1.0, 1, 1, error_bound=0.01)
    with pytest.raises(ValueError, match="exact block encoding"):
        qsvt_filter_projector(inaccurate, phases=(0.0, 0.0))
    with pytest.raises(TypeError, match="exactly one"):
        qsvt_filter_projector(_encoding)
    with pytest.raises(TypeError, match="exactly one"):
        qsvt_filter_projector(
            _encoding,
            phases=(0.0, 0.0),
            polynomial=synthesize_qsvt_polynomial((0.0, 1.0)),
        )


def test_raw_phase_filter_skips_phase_synthesis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Supplied phases build the filter without invoking phase synthesis."""
    monkeypatch.setitem(sys.modules, "pyqsp", None)
    monkeypatch.setitem(sys.modules, "pyqsp.poly", None)
    monkeypatch.setitem(sys.modules, "pyqsp.angle_sequence", None)
    filtered = qsvt_filter_projector(_encoding, phases=(0.0, 0.0))
    assert filtered.signal_width == 3


@pytest.mark.parametrize("scale", (1.0, -1.0))
def test_sign_producer_matches_its_real_target(scale: float) -> None:
    """Real pyqsp candidates retain both orientations across the native boundary."""
    certificate = synthesize_sign_filter(degree=11, delta=3.0, scale=scale)
    assert (
        scale * np.polynomial.chebyshev.chebval(0.8, certificate.chebyshev_coefficients)
        > 0.5
    )
    assert certificate.uniform_error_bound <= 1e-6
