"""Grover Adaptive Search composition and cross-backend witnesses."""

from __future__ import annotations

import math
import pickle
from fractions import Fraction
from itertools import product

import pytest

import qamomile as qm
from qamomile.compilation import ExecutionEndpoint
from qamomile.optimization import (
    BinaryModel,
    CompiledGASExperiment,
    GASExperiment,
    VarType,
)
from qamomile.persistence import deserialize, serialize


def _search_model() -> BinaryModel[VarType]:
    """Return a two-variable objective with one marked assignment.

    Returns:
        BinaryModel[VarType]: The objective ``x4 + x9``.
    """

    return BinaryModel.from_hubo({(4,): 1.0, (9,): 1.0}, variables=(4, 9))


def exercise_gas_backend(*, endpoint: ExecutionEndpoint, seed: int = 0) -> None:
    """Compile and execute the same closed GAS source on one endpoint.

    GAS exposes sampling rather than expectation-value semantics. With a
    threshold of one half, exactly ``00`` is marked; one Grover iteration over
    four assignments therefore returns that assignment deterministically.

    Args:
        endpoint (ExecutionEndpoint): Backend endpoint under test.
        seed (int): Optional deterministic simulator seed.

    Raises:
        AssertionError: If compilation, execution, or source-order decoding
            differs across targets.
    """

    experiment = GASExperiment(_search_model(), threshold=0.5, iterations=1)
    run = experiment.run(on=endpoint, shots=8, seed=seed)

    assert run.samples.vartype is VarType.BINARY
    assert run.samples.total_occurrences == 8
    assert len(run.samples.samples) == 1
    assert run.samples.lowest.assignment == ((4, 0), (9, 0))
    assert run.samples.lowest.energy == 0.0
    assert run.samples.lowest.occurrences == 8

    exact = GASExperiment(
        BinaryModel.from_hubo({(4,): 0.25, (9,): 0.5}, variables=(4, 9)),
        threshold=0.2,
        approximate_real_coefficients=False,
    )
    compiled = exact.compile(to=endpoint.target)
    exact_run = compiled.run(on=endpoint, shots=8, seed=seed)
    assert exact.quantization_scale == 4.0
    assert exact_run.samples.lowest.assignment == ((4, 0), (9, 0))
    assert exact_run.samples.lowest.occurrences == 8
    assert exact_run.samples.lowest.energy == 0.0


def test_gas_amplifies_the_only_assignment_below_threshold() -> None:
    """The reference semantics implement the fixed-threshold Grover step."""

    exercise_gas_backend(endpoint=qm.Reference(), seed=4)


def test_gas_experiment_retains_identity_and_compiled_artifact() -> None:
    """The high-level lifecycle remains an ordinary closed program."""

    experiment = GASExperiment(_search_model(), threshold=0.5, iterations=1)
    endpoint = qm.Reference()
    compiled = experiment.compile(to=endpoint.target)
    run = compiled.run(on=endpoint, shots=4, seed=2)

    assert "quantum.gas" not in experiment.program.inspect()
    assert run.execution.artifact is compiled.artifact
    assert run.identity == compiled.identity == experiment.identity
    assert run.identity.problem_digest == experiment.model.semantic_digest
    assert run.invocation.values() == {}
    with pytest.raises(TypeError, match="process-local"):
        pickle.dumps(compiled)
    restored = deserialize(serialize(compiled.artifact))
    assert serialize(restored) == serialize(compiled.artifact)


def test_compiled_gas_rejects_an_artifact_for_another_experiment() -> None:
    """A target artifact cannot be attached to a different GAS program."""

    expected = GASExperiment(_search_model(), threshold=0.5)
    foreign = GASExperiment(_search_model(), threshold=1.5)
    artifact = qm.compile(foreign.program, to=qm.Reference().target)

    with pytest.raises(ValueError, match="artifact does not match"):
        CompiledGASExperiment(expected, artifact)


def test_gas_preserves_arbitrary_variable_ids_and_higher_order_terms() -> None:
    """Dense qubit positions decode back to the original HUBO variable IDs."""

    model = BinaryModel.from_hubo(
        {(3,): 1.0, (8,): 1.0, (11,): 1.0, (3, 8, 11): -1.0},
        variables=(3, 8, 11),
    )
    experiment = GASExperiment(model, threshold=0.5, iterations=0)
    run = experiment.run(on=qm.Reference(), shots=16, seed=3)

    assert run.samples.total_occurrences == 16
    assert {
        variable for sample in run.samples.samples for variable, _ in sample.assignment
    } == {3, 8, 11}


def test_gas_quantizes_real_coefficients_and_threshold_together() -> None:
    """Real objectives expose the exact integer circuit approximation."""

    with pytest.warns(UserWarning, match="quantized"):
        experiment = GASExperiment(
            BinaryModel.from_hubo({(0,): 0.25, (1,): 0.5}),
            threshold=0.3,
            quantization_parameter=4,
        )

    assert experiment.quantization_scale == 16.0
    assert tuple(experiment.effective_model.coefficients.values()) == (4.0, 8.0)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    (
        ({"iterations": -1}, ValueError, "nonnegative"),
        ({"iterations": True}, TypeError, "integer"),
        ({"output_bits": 1}, ValueError, "too small"),
        ({"threshold": float("inf")}, ValueError, "finite"),
        ({"approximate_real_coefficients": 1}, TypeError, "Boolean"),
    ),
)
def test_gas_rejects_invalid_static_configuration(
    kwargs: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    """Malformed host configuration fails before target compilation.

    Args:
        kwargs (dict[str, object]): Invalid constructor arguments.
        error (type[Exception]): Expected diagnostic class.
        message (str): Expected diagnostic fragment.
    """

    options: dict[str, object] = {"threshold": 0.5}
    options.update(kwargs)
    with pytest.raises(error, match=message):
        GASExperiment(_search_model(), **options)  # type: ignore[arg-type]


@pytest.mark.parametrize("threshold", [0.0, 0.2, 0.5, -0.2])
def test_gas_exact_fractional_encoding_preserves_strict_threshold(threshold):
    """Every small assignment has the same exact strict comparison after scaling."""
    model = BinaryModel.from_hubo(
        {(0,): 0.25, (1,): -0.5, (0, 1): 0.125}, constant=-0.125
    )
    experiment = GASExperiment(
        model, threshold=threshold, iterations=0, approximate_real_coefficients=False
    )
    assert experiment.quantization_scale == 8.0
    scale = Fraction(experiment.quantization_scale)
    for bits in product((0, 1), repeat=2):
        value = Fraction(model.constant) + sum(
            Fraction(coefficient) * product_value(variables, bits)
            for variables, coefficient in model.coefficients.items()
        )
        encoded = Fraction(experiment.effective_model.constant) + sum(
            Fraction(coefficient) * product_value(variables, bits)
            for variables, coefficient in experiment.effective_model.coefficients.items()
        )
        assert encoded == value * scale
        assert (value < Fraction(threshold)) == (
            encoded < -(-Fraction(threshold) * scale // 1)
        )


def product_value(variables, bits):
    """Evaluate one tiny binary monomial for the differential test oracle."""
    result = 1
    for variable in variables:
        result *= bits[variable]
    return result


def test_gas_exact_mode_does_not_snap_small_coefficients():
    """The no-approximation path preserves scientific coefficients near zero."""
    experiment = GASExperiment(
        BinaryModel.from_hubo({(0,): 2.0**-45}),
        threshold=0.0,
        iterations=0,
        approximate_real_coefficients=False,
    )
    assert experiment.quantization_scale == 2.0**45
    assert experiment.effective_model.coefficients[(0,)] == 1.0


def test_gas_exact_mode_rejects_unsafe_width_before_building(monkeypatch):
    """A huge exact scale difference fails before constructing a qkernel graph."""

    def fail_build(**kwargs):
        raise AssertionError("over-budget encoding reached circuit construction")

    import importlib

    module = importlib.import_module("qamomile.optimization.gas")
    monkeypatch.setattr(module, "_gas_sample_program", fail_build)
    for coefficients in ({(0,): 2.0**-60, (1,): 1.0}, {(0,): 5e-324}):
        with pytest.raises(ValueError, match="Q7570"):
            GASExperiment(
                BinaryModel.from_hubo(coefficients),
                threshold=0.0,
                approximate_real_coefficients=False,
            )


def test_gas_exact_mode_fails_closed_without_native_support():
    """An unavailable native encoding never selects the approximation path."""
    model = BinaryModel.from_hubo({(0,): 0.25})
    object.__setattr__(model, "_native_model", object())
    with pytest.raises(TypeError, match="Q7560.*unavailable"):
        GASExperiment(model, threshold=0.0, approximate_real_coefficients=False)


@pytest.mark.parametrize(
    "threshold",
    [
        0.2,
        -0.2,
        0.125,
        math.nextafter(0.125, -math.inf),
        math.nextafter(0.125, math.inf),
    ],
)
def test_exact_fractional_quantum_dictionary_encodes_each_assignment(threshold):
    """The emitted arithmetic sign matches the original exact objective."""
    from qamomile.optimization.gas import _gas_preparation

    model = BinaryModel.from_hubo({(0,): 0.25, (1,): -0.5}, constant=0.125)
    experiment = GASExperiment(
        model, threshold=threshold, iterations=0, approximate_real_coefficients=False
    )
    projection = model._native_model.exact_integer_encoding(threshold)
    integer_bound = int(projection[2])
    width = experiment.resolved_output_bits
    prepare = _gas_preparation(
        model=experiment.effective_model, threshold=integer_bound, output_width=width
    )
    for bits in product((0, 1), repeat=2):
        initial = (*bits, *(0 for _ in range(width)))

        @qm.qkernel
        def sample() -> qm.Shot[qm.Bits[2 + width]]:
            state = qm.prepare(initial)
            state = qm.h(state, target=0)
            state = qm.h(state, target=1)
            return qm.measure(prepare(state))

        counts = qm.run(sample, on=qm.Reference(), shots=4, seed=7).result.counts
        assert len(counts) == 1
        observed = next(iter(counts))
        exact = Fraction(model.constant) + sum(
            Fraction(coefficient) * product_value(variables, bits)
            for variables, coefficient in model.coefficients.items()
        )
        assert observed[:2] == bits
        assert bool(observed[-1]) is (exact < Fraction(threshold))
