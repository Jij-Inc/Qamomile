"""Typed frequency statistics and explicit measurement/readout correspondence."""

import math
from dataclasses import FrozenInstanceError
from fractions import Fraction

import pytest

import qamomile as qm
from qamomile import statistics as stats
from qamomile._native import compiler as native
from qamomile.compilation import ResultABI
from qamomile.types import RecordType, as_type


def samples(values, result_type=qm.Shot[qm.Real], width=64):
    """Encode a small typed histogram through the ordinary native ABI."""
    abi = ResultABI(result_type, tuple(range(width)), width)
    result = qm.SampleResult.from_counts(
        backend="statistics-test",
        counts={abi.encode_value(value): count for value, count in values},
        expected_width=width,
    )
    from dataclasses import replace

    return replace(result, result_abi=abi)


def test_mean_uses_frequency_counts_and_unbiased_sample_variance():
    result = stats.mean(samples([(1.0, 2), (4.0, 1)]))
    assert result.value == 2.0
    assert result.sample_variance == 3.0
    assert result.standard_error == 1.0
    assert result.shots == 3
    with pytest.raises(FrozenInstanceError):
        result._native = None
    with pytest.raises(TypeError):
        stats.MeanEstimate()


def test_single_observation_has_unknown_uncertainty():
    estimate = stats.mean(samples([(7.0, 1)]))
    assert estimate.value == 7.0
    assert estimate.sample_variance is None
    assert estimate.standard_error is None
    assert stats.combine_independent([estimate], offset=2.0).standard_error is None
    zero = stats.combine_independent([estimate], coefficients=[0.0], offset=2.0)
    assert zero.value == 2.0
    assert zero.standard_error == 0.0


def test_large_counts_do_not_expand_shots():
    count = 10**100
    estimate = stats.mean(samples([(-1.0, count), (1.0, count)]))
    assert estimate.shots == 2 * count
    assert estimate.value == 0.0
    assert estimate.sample_variance == pytest.approx(1.0)
    assert estimate.standard_error == pytest.approx(
        1 / math.sqrt(2 * count), rel=1e-14, abs=0
    )


def test_moments_do_not_lose_variance_to_cancellation():
    estimate = stats.mean(samples([(1e12, 1), (1e12 + 1, 1), (1e12 + 2, 1)]))
    assert estimate.value == 1e12 + 1
    assert estimate.sample_variance == 1.0
    assert estimate.standard_error == pytest.approx(math.sqrt(1 / 3))


def test_weighted_moments_agree_with_fraction_oracle():
    observations = [(0.1, 3), (-7.25, 5), (2.0, 8)]
    count = sum(n for _, n in observations)
    mean = sum(Fraction(x) * n for x, n in observations) / count
    variance = sum(n * (Fraction(x) - mean) ** 2 for x, n in observations) / (count - 1)
    estimate = stats.mean(samples(observations))
    assert estimate.value == float(mean)
    assert estimate.sample_variance == float(variance)
    assert estimate.standard_error == math.sqrt(float(variance / count))


def test_paths_select_nested_records_without_reading_real_transport_bits():
    histogram = samples(
        [
            ({"scores": (2.0, 4.0), "measured": (1, 0)}, 3),
            ({"scores": (4.0, 16.0), "measured": (0, 1)}, 1),
        ],
        RecordType(
            "Score",
            (
                ("scores", as_type(tuple[qm.Shot[qm.Real], qm.Shot[qm.Real]])),
                ("measured", qm.Shot[qm.Bits[2]]),
            ),
        ),
        130,
    )
    assert stats.mean(histogram, path=("scores", 0)).value == 2.5
    assert stats.mean(histogram, path=("scores", 1)).value == 7.0
    assert stats.parity_mean(histogram, path=("measured",), indices=(0,)).value == -0.5
    with pytest.raises(ValueError, match="Q7400"):
        stats.parity_mean(histogram, path=("scores",), indices=(0,))
    with pytest.raises(ValueError, match="Q7400"):
        stats.mean(histogram)
    with pytest.raises(ValueError, match="Q7400"):
        stats.mean(histogram, path=("missing",))


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_joint_linear_combination_preserves_covariance(sign):
    histogram = qm.SampleResult.from_counts(
        backend="joint", counts={(0, 0): 2, (1, 1): 2}
    )
    estimate = stats.linear_combination(
        histogram, [(1.0, stats.Parity((0,))), (sign, stats.Parity((1,)))], offset=3.0
    )
    assert estimate.value == 3.0
    assert estimate.sample_variance == pytest.approx(16 / 3 if sign == 1 else 0.0)
    assert estimate.standard_error == pytest.approx(
        math.sqrt(4 / 3) if sign == 1 else 0.0
    )


def test_independent_groups_add_mean_variances_not_shot_variances():
    first = stats.mean(samples([(0.0, 2), (2.0, 2)]))
    second = stats.mean(samples([(0.0, 4), (2.0, 4)]))
    combined = stats.combine_independent(
        [first, second], coefficients=[2.0, -3.0], offset=0.5
    )
    assert combined.value == -0.5
    assert combined.standard_error == pytest.approx(math.sqrt(4 / 3 + 9 / 7))
    assert combined.shots == 12
    assert not hasattr(combined, "sample_variance")
    with pytest.raises(ValueError, match="independent twice"):
        stats.combine_independent([first, first])


def test_identity_offset_needs_no_measurements():
    result = stats.combine_independent([], offset=1.25)
    assert (result.value, result.standard_error, result.shots) == (1.25, 0.0, 0)


@pytest.mark.parametrize("indices", [(0, 0), (-1,), (True,)])
def test_invalid_parity_coordinates_fail(indices):
    with pytest.raises((ValueError, TypeError)):
        stats.Parity(indices)


def test_parity_checks_logical_width_and_explicit_numeric_interpretation():
    histogram = qm.SampleResult.from_counts(
        backend="bits", counts={(1, 0): 3, (0, 0): 1}
    )
    assert stats.parity_mean(histogram, indices=(0,)).value == -0.5
    assert stats.mean(histogram, path=(0,)).value == 0.75
    with pytest.raises(ValueError, match="Q7400"):
        stats.parity_mean(histogram, indices=(2,))
    with pytest.raises(ValueError, match="Q7400"):
        stats.parity_mean(samples([(0.0, 2)]), indices=(0,))
    assert stats.parity_mean(histogram, indices=()).value == 1.0


@pytest.mark.parametrize("coefficient", [math.nan, math.inf, -math.inf])
def test_nonfinite_coefficients_fail_closed(coefficient):
    with pytest.raises(ValueError, match="Q7401"):
        stats.linear_combination(samples([(1.0, 2)]), [(coefficient, stats.Value())])


def test_nonrepresentable_variance_is_not_reported_as_zero():
    for magnitude in (1e308, 1e-200):
        with pytest.raises(ValueError, match="Q7401"):
            stats.mean(samples([(-magnitude, 1), (magnitude, 1)]))


def test_numeric_and_structural_budgets_are_checked():
    with pytest.raises(ValueError, match="Q7402"):
        stats.Value((0,) * (native.MAX_STATISTICS_PATH + 1))
    with pytest.raises(ValueError, match="Q7402"):
        stats.mean(samples([(1.0, 1 << native.MAX_STATISTICS_COUNT_BITS)]))
    with pytest.raises(ValueError, match="Q7402"):
        stats.linear_combination(
            samples([(1.0, 2)]),
            [(1.0, stats.Value())] * (native.MAX_STATISTICS_TERMS + 1),
        )


def test_missing_native_statistics_has_no_python_fallback(monkeypatch):
    histogram = samples([(1.0, 2)])
    monkeypatch.setattr(native, "statistics_mean", None)
    with pytest.raises(qm.CapabilityError, match="Q7404"):
        stats.mean(histogram)


@pytest.mark.parametrize("aer", [False, True])
def test_measurement_basis_and_readout_are_explicit(aer):
    """The same parity estimates different original-state observables by basis."""
    source = """
import qamomile as qm
@qm.qkernel
def measure(x_basis: qm.Const[qm.Bool]) -> qm.Shot[qm.Bits[1]]:
    q = qm.h(qm.qubits(1))
    if x_basis:
        q = qm.h(q)
    return qm.measure(q)
"""
    if aer:
        pytest.importorskip("qiskit_aer")
        from qamomile.qiskit import Aer

        endpoint = Aer()
    else:
        endpoint = qm.Reference()
    module = qm.build_source(source)
    x_samples = qm.run(
        module.specialize("measure", x_basis=True), on=endpoint, shots=128, seed=1
    ).result
    z_samples = qm.run(
        module.specialize("measure", x_basis=False), on=endpoint, shots=128, seed=2
    ).result
    x = stats.parity_mean(x_samples, indices=(0,))
    z = stats.parity_mean(z_samples, indices=(0,))
    assert x.value == 1.0
    assert abs(z.value) < 0.3
    assert x.standard_error == 0.0
    assert z.standard_error > 0


@pytest.mark.parametrize("aer", [False, True])
def test_nested_basis_selection_preserves_captures_and_y_sign(aer):
    source = """
import qamomile as qm
@qm.qkernel
def measure(basis: qm.Const[qm.Vector[qm.Int, 2]]) -> qm.Shot[qm.Bits[2]]:
    q = qm.qubits(2)
    q = qm.h(q, target=0)
    q = qm.s(q, target=0)
    q = qm.x(q, target=1)
    for target in range(2):
        if basis[target] == 1:
            q = qm.h(q, target=target)
        elif basis[target] == 2:
            q = qm.sdg(q, target=target)
            q = qm.h(q, target=target)
    return qm.measure(q)
"""
    if aer:
        pytest.importorskip("qiskit_aer")
        from qamomile.qiskit import Aer

        endpoint = Aer()
    else:
        endpoint = qm.Reference()
    program = qm.build_source(source).specialize("measure", basis=(2, 0))
    result = qm.run(program, on=endpoint, shots=16, seed=9).result
    assert result.values == {(0, 1): 16}
    assert stats.parity_mean(result, indices=(0,)).value == 1.0
    assert stats.parity_mean(result, indices=(0, 1)).value == -1.0


def test_native_statistics_preflights_container_sizes():
    with pytest.raises(ValueError, match="Q7402"):
        native.statistics_readout([None] * (native.MAX_STATISTICS_PATH + 1))
    with pytest.raises(ValueError, match="Q7402"):
        native.statistics_combine_independent(
            [None] * (native.MAX_STATISTICS_TERMS + 1), 0.0
        )
