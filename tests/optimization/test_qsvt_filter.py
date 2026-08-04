"""Tests for the QSVT eigenstate-filtering converter."""

from __future__ import annotations

import logging
import math
import re
from typing import Any

import numpy as np
import ommx.v1
import pytest

from qamomile.circuit.transpiler.job import SampleResult
from qamomile.optimization.binary_model import BinaryModel
from qamomile.optimization.qsvt_filter import QSVTFilterConverter


@pytest.fixture
def transpiler() -> Any:
    """Return a Qiskit transpiler for converter compilation.

    Returns:
        Any: A ``QiskitTranspiler`` instance.
    """
    pytest.importorskip("qiskit")
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
    num_ancilla = converter.num_ancilla_bits(mu)
    amplitudes = np.array(
        [state[basis << num_ancilla] for basis in range(1 << num_system_qubits)]
    )
    return float(np.sum(np.abs(amplitudes) ** 2))


def test_encoding_holds_out_the_constant_term() -> None:
    """The constant is tracked as an offset instead of costing subnormalization.

    An identity term shifts every eigenvalue equally, so it cannot change which
    state is best, but it would consume the subnormalization budget that sets
    the filter's resolution. It is folded into the threshold instead.
    """
    model = BinaryModel.from_higher_ising(
        {(0,): 1.0, (1,): -1.0, (0, 1): -1.0}, constant=0.5
    )
    converter = QSVTFilterConverter(model)

    assert converter.encoding.num_system_qubits == 2
    # 1-norm of the non-constant coefficients only.
    assert converter.normalization == pytest.approx(3.0)
    assert converter.energy_offset == pytest.approx(0.5)
    # The pair still bounds the spectrum, which is what picks a threshold.
    energies = _ising_diagonal({(0,): 1.0, (1,): -1.0, (0, 1): -1.0}, 2) + 0.5
    low = converter.energy_offset - converter.normalization
    high = converter.energy_offset + converter.normalization
    assert low <= energies.min() and energies.max() <= high


def test_cost_hamiltonian_is_not_exposed() -> None:
    """The converter block encodes the cost operator instead of exposing it."""
    model = BinaryModel.from_higher_ising({(0,): 1.0, (0, 1): -1.0})
    converter = QSVTFilterConverter(model)

    with pytest.raises(NotImplementedError, match="cost Hamiltonian"):
        converter.get_cost_hamiltonian()


@pytest.mark.parametrize(
    ("mu", "expected_signal_qubits"),
    [
        # A residual shift keeps both LCU terms, so the composition adds one
        # selector qubit on top of the child encoding's single signal qubit.
        (-2.0, 2),
        (1.5, 2),
        # lcu_block_encoding drops zero-coefficient terms, so a threshold
        # sitting on the offset collapses back to the unshifted width.
        (0.75, 1),
    ],
)
def test_shifted_encoding_normalization_grows_with_the_residual_shift(
    mu: float, expected_signal_qubits: int
) -> None:
    """Composing with the identity adds |mu - offset| to the subnormalization.

    The constant is already held out of the encoding, so only the part of the
    threshold that is not the offset has to be paid for.
    """
    model = BinaryModel.from_higher_ising({(0,): 1.0, (0, 1): -1.0}, constant=0.75)
    converter = QSVTFilterConverter(model)

    shifted = converter._shifted_encoding(mu)

    assert converter.energy_offset == pytest.approx(0.75)
    assert converter.encoding.num_signal_qubits == 1
    assert shifted.num_system_qubits == converter.encoding.num_system_qubits
    assert shifted.normalization == pytest.approx(
        converter.normalization + abs(mu - converter.energy_offset)
    )
    assert shifted.num_signal_qubits == expected_signal_qubits
    assert converter.num_ancilla_bits(mu) == 1 + expected_signal_qubits


def test_holding_out_the_constant_shrinks_the_subnormalization() -> None:
    """The offset is pure savings: it never widens the shifted normalization.

    Encoding the constant would cost ``|c| + |mu|``; folding it into the
    threshold costs ``|mu - c|``, which the triangle inequality caps at the
    same value and which is far smaller when ``mu`` sits near ``c`` — the usual
    case, since a spectrum is centred on its own constant.
    """
    coefficients = {(0,): 1.0, (1,): -1.0, (0, 1): -1.0}
    constant = 5.0
    converter = QSVTFilterConverter(
        BinaryModel.from_higher_ising(coefficients, constant=constant)
    )
    bare = QSVTFilterConverter(BinaryModel.from_higher_ising(coefficients))

    for mu in (constant, constant - 1.0, constant + 2.0, 0.0):
        held_out = converter._shifted_encoding(mu).normalization
        encoded = bare.normalization + abs(constant) + abs(mu)
        assert held_out <= encoded + 1e-12
    # At mu == c the residual shift vanishes entirely.
    assert converter._shifted_encoding(constant).normalization == pytest.approx(
        converter.normalization
    )


def test_qsp_phases_are_odd_length_and_cached() -> None:
    """Phase generation returns degree+1 phases and reuses its cache."""
    pytest.importorskip("pyqsp")
    model = BinaryModel.from_higher_ising({(0,): 1.0})
    converter = QSVTFilterConverter(model)

    phases = converter._qsp_phases(degree=11, delta=5)
    assert len(phases) == 12

    cached = converter._qsp_phases(degree=11, delta=5)
    assert cached == phases
    # The cache hands out copies, so callers cannot corrupt it.
    cached[0] = 0.0
    assert converter._qsp_phases(degree=11, delta=5) == phases


def test_qsp_phases_log_the_summary_and_demote_pyqsp_output(
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Synthesis writes nothing to stdout; both summary and trace are logged.

    ``pyqsp`` prints unconditionally, so its output is captured and demoted to
    ``DEBUG``. The converter's own one-line summary goes to ``INFO``, leaving a
    downstream application's console untouched.
    """
    pytest.importorskip("pyqsp")
    model = BinaryModel.from_higher_ising({(0,): 1.0})
    converter = QSVTFilterConverter(model)

    with caplog.at_level(logging.DEBUG, logger="qamomile.optimization.qsvt_filter"):
        converter._qsp_phases(degree=11, delta=5, scale=-1.1)

    assert capsys.readouterr().out == ""

    summaries = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert summaries == [
        "synthesizing sign filter: degree=11, delta=5, scale=-1.1 -> 12 phases"
    ]
    # pyqsp's own chatter survives, one level down, rather than being dropped.
    debug = "\n".join(
        r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG
    )
    assert "[pyqsp.poly.PolySign] degree=11, delta=5" in debug


def test_qsp_phases_reject_a_polynomial_outside_the_qsp_bound() -> None:
    """A joint (degree, delta, scale) violation fails loudly, not silently.

    ``ensure_bounded`` leaves headroom that depends on degree and delta, so
    ``scale`` alone cannot decide whether |p| <= 1 holds. At (21, 20) the
    default magnitude overshoots; the phases extracted from such a polynomial
    approximate nothing, which surfaces as a filter that keeps ~80% of the
    spectrum instead of 3%.
    """
    pytest.importorskip("pyqsp")
    model = BinaryModel.from_higher_ising({(0,): 1.0})
    converter = QSVTFilterConverter(model)

    with pytest.raises(ValueError, match=r"QSP bound") as excinfo:
        converter._qsp_phases(degree=21, delta=20, scale=1.10)

    # The message quotes the measured peak so the overshoot is actionable.
    assert re.search(r"\|p\|=1\.\d+", str(excinfo.value))
    # Nothing broken is cached for a later call to hand back.
    assert (21, 20.0, 1.10) not in converter._phase_cache


def test_qsp_phases_accept_a_bounded_polynomial() -> None:
    """The default combination clears the bound and still synthesizes."""
    pytest.importorskip("pyqsp")
    model = BinaryModel.from_higher_ising({(0,): 1.0})
    converter = QSVTFilterConverter(model)

    phases = converter._qsp_phases(degree=61, delta=20, scale=1.10)

    assert len(phases) == 62


@pytest.mark.parametrize("degree", [0, -1, 10])
def test_qsp_phases_reject_non_odd_positive_degrees(degree: int) -> None:
    """The sign approximation is an odd polynomial, so degree must be odd."""
    model = BinaryModel.from_higher_ising({(0,): 1.0})
    converter = QSVTFilterConverter(model)

    with pytest.raises(ValueError, match="degree"):
        converter._qsp_phases(degree=degree)


@pytest.mark.parametrize("delta", [0.0, -1.0])
def test_qsp_phases_reject_non_positive_transition_widths(delta: float) -> None:
    """A non-positive transition width has no sign-approximation meaning."""
    model = BinaryModel.from_higher_ising({(0,): 1.0})
    converter = QSVTFilterConverter(model)

    with pytest.raises(ValueError, match="delta"):
        converter._qsp_phases(delta=delta)


@pytest.mark.parametrize("scale", [0.0, 1.3, -5.0])
def test_qsp_phases_reject_scales_outside_the_qsp_bound(scale: float) -> None:
    """Rescaling past the QSP bound would silently yield meaningless phases."""
    model = BinaryModel.from_higher_ising({(0,): 1.0})
    converter = QSVTFilterConverter(model)

    with pytest.raises(ValueError, match="scale"):
        converter._qsp_phases(scale=scale)


@pytest.mark.parametrize("phases", [[0.1], [0.1, 0.2, 0.3]])
def test_transpile_rejects_odd_length_phase_sequences(
    transpiler: Any, phases: list[float]
) -> None:
    """The alternation needs an even phase count (odd polynomial degree)."""
    model = BinaryModel.from_higher_ising({(0,): 1.0})
    converter = QSVTFilterConverter(model)

    with pytest.raises(ValueError, match="even number"):
        converter.transpile(transpiler, mu=0.0, phi=phases)


def test_transpile_sizes_the_circuit_from_the_shifted_encoding(
    transpiler: Any,
) -> None:
    """The probe allocates one projector qubit plus signal and system."""
    model = BinaryModel.from_higher_ising({(0,): 1.0, (1,): -1.0, (0, 1): -1.0})
    converter = QSVTFilterConverter(model)
    shifted = converter._shifted_encoding(0.5)

    executable = converter.transpile(transpiler, mu=0.5, phi=[0.1, 0.2])

    # One projector qubit, the signal and system registers, plus the single
    # clean auxiliary qmc.qsvt allocates for its projector rotations. That
    # auxiliary is restored to zero and never measured, so it is not part of
    # the post-selected ancilla block.
    expected = 2 + shifted.num_signal_qubits + shifted.num_system_qubits
    assert executable.quantum_circuit.num_qubits == expected
    assert converter.num_ancilla_bits(0.5) == 1 + shifted.num_signal_qubits


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
    phases = converter._qsp_phases()

    probability = _exact_success_probability(converter, transpiler, mu, phases, 2)

    energies = _ising_diagonal(coefficients, 2)
    assert float((energies < mu).mean()) == pytest.approx(expected_fraction)
    assert probability == pytest.approx(expected_fraction, abs=0.05)


def test_the_predicate_is_unchanged_by_holding_out_the_constant(
    transpiler: Any,
) -> None:
    """Holding the constant out sharpens the filter without moving the answer.

    ``mu`` stays in the problem's own energy units, so the same threshold must
    select the same fraction of the spectrum whether or not the constant sits
    inside the encoding. Only the subnormalization -- and hence the resolution
    -- improves.
    """
    pytest.importorskip("pyqsp")
    coefficients = {(0,): 1.0, (1,): -1.0, (0, 1): -1.0}
    constant = 5.0
    converter = QSVTFilterConverter(
        BinaryModel.from_higher_ising(coefficients, constant=constant)
    )
    phases = converter._qsp_phases()

    # Energies are the bare spectrum plus the constant; mu is quoted in those
    # same units, so the shifted threshold lands between the two lowest levels.
    energies = _ising_diagonal(coefficients, 2) + constant
    mu = constant - 0.5
    probability = _exact_success_probability(converter, transpiler, mu, phases, 2)

    assert float((energies < mu).mean()) == pytest.approx(0.75)
    assert probability == pytest.approx(0.75, abs=0.05)


def test_a_positive_scale_inverts_the_filter(transpiler: Any) -> None:
    """The sign of ``scale`` is the only thing selecting which side is kept.

    Same circuit, same threshold: the default keeps the three quarters of the
    spectrum below ``mu``, and negating the polynomial keeps the complementary
    quarter above it.
    """
    pytest.importorskip("pyqsp")
    coefficients = {(0,): 1.0, (1,): -1.0, (0, 1): -1.0}
    model = BinaryModel.from_higher_ising(coefficients)
    converter = QSVTFilterConverter(model)

    below = _exact_success_probability(
        converter, transpiler, -0.5, converter._qsp_phases(), 2
    )
    above = _exact_success_probability(
        converter, transpiler, -0.5, converter._qsp_phases(scale=1.10), 2
    )

    assert below == pytest.approx(0.75, abs=0.05)
    assert above == pytest.approx(0.25, abs=0.05)
    assert below + above == pytest.approx(1.0, abs=0.05)


def test_sampled_filter_recovers_the_ground_states(transpiler: Any) -> None:
    """Post-selected samples concentrate on the states the filter keeps."""
    pytest.importorskip("pyqsp")
    coefficients = {(0,): 1.0, (1,): -1.0, (0, 1): -1.0}
    model = BinaryModel.from_higher_ising(coefficients)
    converter = QSVTFilterConverter(model)
    phases = converter._qsp_phases()

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


def test_ommx_decode_post_selects_before_evaluating_the_original_instance() -> None:
    """The OMMX output path sees only the post-selected shots.

    ``decode`` is inherited from the base converter and routes through this
    class's ``decode_to_binary_sampleset`` override, so the projector/signal
    post-selection must survive all the way into the ``ommx.v1.SampleSet``.

    The QUBO energy is derived by hand: with ``objective = -10 * x0`` and the
    equality ``x0 + x1 == 1`` absorbed at penalty weight 2, the kept shot
    ``x0 = x1 = 1`` scores ``-10 + 2 * (1 + 1 - 1)**2 = -8``, while OMMX
    reports the un-penalized original objective ``-10`` and marks it
    infeasible.
    """
    x0 = ommx.v1.DecisionVariable.binary(0, name="x0")
    x1 = ommx.v1.DecisionVariable.binary(1, name="x1")
    instance = ommx.v1.Instance.from_components(
        decision_variables=[x0, x1],
        objective=-10.0 * x0,
        constraints=[(x0 + x1 == 1).set_id(0)],
        sense=ommx.v1.Instance.MINIMIZE,
    )
    converter = QSVTFilterConverter(instance, uniform_penalty_weight=2.0)

    # Measured bit 1 decodes to spin -1, i.e. binary 1.
    raw: SampleResult[Any] = SampleResult(
        results=[
            (([0], [0], [1, 1]), 3),  # kept: every ancilla zero
            (([1], [0], [0, 1]), 1),  # dropped: projector fired
            (([0], [1], [1, 0]), 1),  # dropped: signal register non-zero
        ],
        shots=5,
    )

    assert converter.success_probability(raw) == pytest.approx(0.6)

    binary = converter.decode_to_binary_sampleset(raw)
    assert binary.samples == [{0: 1, 1: 1}]
    assert binary.num_occurrences == [3]
    assert binary.energy == pytest.approx([-8.0])

    decoded = converter.decode(raw)
    assert isinstance(decoded, ommx.v1.SampleSet)
    assert decoded.get(0).objective == pytest.approx(-10.0)
    assert not decoded.get(0).feasible


def test_success_probability_of_an_empty_result_is_zero() -> None:
    """A zero-shot result reports no filtered weight instead of dividing by 0."""
    model = BinaryModel.from_higher_ising({(0,): 1.0})
    converter = QSVTFilterConverter(model)
    empty: SampleResult[Any] = SampleResult(results=[], shots=0)

    assert converter.success_probability(empty) == 0.0
