"""Native-model QSVT filtering preserves energy units and typed samples."""

from __future__ import annotations

import itertools
import math
from dataclasses import FrozenInstanceError, replace
from importlib import import_module

import ommx.v1
import pytest

import qamomile as qm
from qamomile.optimization import (
    BinaryModel,
    CompiledQSVTEigenstateFilterExperiment,
    OMMXProblem,
    QSVTEigenstateFilterExperiment,
    QSVTFilterRun,
    VarType,
)
from qamomile.persistence import deserialize, serialize
from qamomile.stdlib import QSVTPhases, signals, synthesize_qsvt_polynomial

from .test_qsvt_filter_stdlib import exercise_qsvt_filter_backend


def _sparse_model():
    """Retain an unused variable and a nonzero BINARY-to-SPIN offset."""

    return BinaryModel.from_hubo({(7,): 2.0}, constant=-0.5, variables=(101, 7))


def _linear_phases() -> QSVTPhases:
    """Encode the independently known Wx polynomial p(x)=x."""

    return QSVTPhases.from_qsp_wx((0.0, 0.0))


def _expected_weights(experiment):
    """Compute success weights directly from original problem energies."""

    scale = experiment.shifted_encoding.normalization
    return {
        bits: ((1.0 - (experiment.model.energy(bits) - experiment.mu) / scale) / 2.0)
        ** 2
        / (1 << experiment.model.num_variables)
        for bits in itertools.product((0, 1), repeat=experiment.model.num_variables)
    }


def _uniform_filter_state(experiment):
    """Expose the same coherent projector for the expectation execution path."""

    transform = experiment.filter_encoding.unitary
    width = experiment.filter_encoding.total_width
    signal_width = experiment.filter_encoding.signal_width
    system_width = experiment.filter_encoding.system_width

    @qm.qkernel
    def state() -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
        """Prepare a uniform problem input and keep every coherent flag."""

        flags, system = qm.split_at(qm.qubits(width), signal_width)
        for index in range(system_width):
            system = qm.h(system, target=index)
        return transform(qm.concat(flags, system))

    return state


def _zero_signal_projector(experiment) -> qm.Hamiltonian:
    """Expand the complete all-zero flag projector in commuting Z words."""

    observable = qm.Hamiltonian.identity(
        1.0, num_qubits=experiment.filter_encoding.total_width
    )
    for index in range(experiment.num_ancilla_bits):
        observable = observable * (qm.Hamiltonian.identity(0.5) + 0.5 * qm.Z(index))
    return observable


@pytest.mark.parametrize("mu", (-0.25, 0.75))
def test_filter_threshold_uses_original_energy_units_and_immutable_model(mu):
    """Offsets, either shift sign, sparse IDs, and domain survive composition."""

    model = _sparse_model()
    original = model.to_bytes()
    experiment = QSVTEigenstateFilterExperiment(model, mu, phases=_linear_phases())

    assert experiment.model is model
    assert model.to_bytes() == original
    assert model.variables == (7, 101)
    assert model.vartype is VarType.BINARY
    assert experiment.energy_offset == 0.5
    assert experiment.normalization == experiment.encoding.normalization == 1.0
    assert experiment.shifted_encoding.normalization == 1.0 + abs(mu - 0.5)
    assert experiment.encoding.system_width == 2
    assert experiment.filter_encoding.system_width == 2
    assert experiment.num_ancilla_bits == experiment.shifted_encoding.signal_width + 2
    assert "quantum.qsvt" not in experiment.program.inspect()
    assert experiment.identity.problem_digest == model.semantic_digest
    with pytest.raises(FrozenInstanceError):
        experiment.mu = 0.0

    endpoint = qm.Reference()
    compiled = experiment.compile(to=endpoint.target)
    run = compiled.run(on=endpoint, shots=128, seed=31)
    assert run.identity == compiled.identity == experiment.identity
    assert run.execution.artifact is compiled.artifact
    assert run.success.shots == run.execution.result.shots == 128
    assert run.success_probability == experiment.success_probability(
        run.execution.result
    )
    assert run.success_probability == run.success.successes / 128
    assert run.samples is not None
    assert run.samples.vartype is VarType.BINARY
    assert run.samples.total_occurrences == run.success.successes
    weights = _expected_weights(experiment)
    for sample in run.samples.samples:
        assert tuple(variable for variable, _ in sample.assignment) == (7, 101)
        assert sample.energy == model.energy(sample.values)
        bits = tuple(sample.values[variable] for variable in model.variables)
        expected = weights[bits]
        assert (
            abs(sample.occurrences / 128 - expected)
            <= 6 * math.sqrt(expected * (1 - expected) / 128) + 1 / 128
        )
    restored = deserialize(serialize(compiled.artifact))
    assert qm.execute(restored, on=endpoint, shots=4, seed=9).width == 3


def test_filter_accepts_a_native_certified_polynomial():
    """A bounded nontrivial polynomial reaches the ordinary compiler path."""

    polynomial = synthesize_qsvt_polynomial((0.0, 0.2))
    experiment = QSVTEigenstateFilterExperiment(
        _sparse_model(), 0.5, polynomial=polynomial
    )
    observed = qm.expectation(
        _uniform_filter_state(experiment),
        _zero_signal_projector(experiment),
        on=qm.Reference(),
        shots=128,
        seed=12,
        grouping="qwc",
    ).value

    assert experiment.polynomial is polynomial
    # At mu=offset the normalized eigenvalues are +/-1, so p(x)=0.2*x.
    assert observed == pytest.approx(((0.4**2) + (0.6**2)) / 2, abs=0.10)


def test_filter_successes_compose_with_original_ommx_evaluation():
    """Filtering and original constrained-problem evaluation remain separate."""

    left = ommx.v1.DecisionVariable.binary(7)
    right = ommx.v1.DecisionVariable.binary(101)
    instance = ommx.v1.Instance.from_components(
        decision_variables=[left, right],
        objective=-2.0 * left + 0.75 * right + 3.0,
        constraints=[(left + right == 1).set_id(9)],
        sense=ommx.v1.Instance.MINIMIZE,
    )
    original = instance.to_bytes()
    problem = OMMXProblem.from_instance(instance, uniform_penalty_weight=2.0)
    experiment = QSVTEigenstateFilterExperiment(
        problem.model, 2.0, phases=_linear_phases()
    )
    run = experiment.run(on=qm.Reference(), shots=128, seed=23)

    assert instance.to_bytes() == original
    assert run.samples is not None
    evaluated = problem.evaluate(run.samples)
    assert evaluated.total_occurrences == run.success.successes
    for sample_id, sample in enumerate(run.samples.samples):
        assert sample.energy == problem.model.energy(sample.values)
        assert evaluated.get(sample_id).objective == pytest.approx(
            -2.0 * sample.values[7] + 0.75 * sample.values[101] + 3.0
        )
        assert evaluated.occurrences[sample_id] == sample.occurrences


def test_filter_no_success_is_explicit_and_rejects_untyped_or_wrong_width_results():
    """Failure shots stay in the denominator without inventing decoded rows."""

    experiment = QSVTEigenstateFilterExperiment(
        _sparse_model(), 0.75, phases=_linear_phases()
    )
    result = experiment.run(on=qm.Reference(), shots=4, seed=19).execution.result
    failures = replace(result, _counts=(((0, 1, 0), 12),))

    assert experiment.success_probability(failures) == 0.0
    assert signals.estimate_success(failures).successes == 0
    assert experiment.decode(failures) is None
    untyped = qm.SampleResult.from_counts(backend="test", counts={(1, 0, 1): 2})
    with pytest.raises(TypeError):
        experiment.decode(untyped)
    wrong_width = replace(result, _counts=(((1, 0), 2),))
    with pytest.raises((TypeError, ValueError)):
        experiment.decode(wrong_width)


def test_filter_decodes_spin_values_and_validates_run_success_accounting():
    """SPIN payloads use original IDs and corrupt conditional counts fail."""

    model = BinaryModel.from_ising({(23,): 1.0}, variables=(23, 81))
    experiment = QSVTEigenstateFilterExperiment(model, 0.0, phases=_linear_phases())
    run = experiment.run(on=qm.Reference(), shots=64, seed=13)

    assert run.samples is not None
    assert run.samples.vartype is VarType.SPIN
    assert all(sample.values[23] == -1 for sample in run.samples.samples)
    assert all(set(sample.values) == {23, 81} for sample in run.samples.samples)
    assert all(sample.energy == -1.0 for sample in run.samples.samples)
    with pytest.raises(ValueError, match="success"):
        replace(run, success=replace(run.success, successes=run.success.successes + 1))
    with pytest.raises(ValueError, match="successful shots"):
        replace(run, samples=None)


@pytest.mark.parametrize(("mu", "probability"), ((1.0, 0.0), (2.0, 0.25), (3.0, 1.0)))
def test_filter_constant_objective_handles_zero_and_identity_blocks(mu, probability):
    """Constant models retain disconnected variables and can accept no shots."""

    model = BinaryModel.from_ising({}, constant=2.0, variables=(71,))
    experiment = QSVTEigenstateFilterExperiment(model, mu, phases=_linear_phases())
    run = experiment.run(on=qm.Reference(), shots=64, seed=21)

    assert run.success_probability == pytest.approx(probability, abs=0.15)
    if probability == 0.0:
        assert run.success.successes == 0
        assert run.samples is None
    else:
        assert run.samples is not None
        assert run.samples.total_occurrences == run.success.successes
        assert all(tuple(sample.values) == (71,) for sample in run.samples.samples)
        assert all(sample.energy == 2.0 for sample in run.samples.samples)


@pytest.mark.parametrize(
    ("mu", "error"),
    (
        (True, TypeError),
        ("0.5", TypeError),
        (math.inf, ValueError),
        (math.nan, ValueError),
    ),
)
def test_filter_rejects_invalid_threshold_before_building_a_program(mu, error):
    """Threshold type and finiteness are checked in original model units."""

    with pytest.raises(error, match="mu"):
        QSVTEigenstateFilterExperiment(_sparse_model(), mu, phases=_linear_phases())


def test_filter_rejects_an_artifact_or_execution_for_another_threshold():
    """A compiled filter and its run retain the exact threshold program."""

    first = QSVTEigenstateFilterExperiment(
        _sparse_model(), -0.25, phases=_linear_phases()
    )
    second = QSVTEigenstateFilterExperiment(first.model, 0.75, phases=_linear_phases())
    endpoint = qm.Reference()
    foreign = second.compile(to=endpoint.target)
    with pytest.raises(ValueError, match="artifact|experiment|identity"):
        CompiledQSVTEigenstateFilterExperiment(first, foreign.artifact)
    run = foreign.run(on=endpoint, shots=4, seed=9)
    with pytest.raises(ValueError, match="artifact|experiment|identity"):
        QSVTFilterRun(first.identity, run.execution, run.success, run.samples)


@pytest.fixture(scope="module")
def filter_witness():
    """Close one source program once for every local backend integration."""

    experiment = QSVTEigenstateFilterExperiment(
        _sparse_model(), 0.75, phases=_linear_phases()
    )
    return experiment, _uniform_filter_state(experiment).specialize()


_BACKENDS = [
    ("reference", "qamomile", "Reference", None),
    ("qiskit", "qamomile.qiskit", "Aer", "qiskit_aer"),
    pytest.param(
        "cirq", "qamomile.cirq", "CirqSimulator", "cirq", marks=pytest.mark.cirq
    ),
    pytest.param(
        "quri_parts",
        "qamomile.quri_parts",
        "Qulacs",
        "qulacs",
        marks=pytest.mark.quri_parts,
    ),
    pytest.param(
        "cudaq", "qamomile.cudaq", "CudaqLocal", "cudaq", marks=pytest.mark.cudaq
    ),
    pytest.param(
        "braket",
        "qamomile.braket",
        "BraketLocal",
        "braket.circuits",
        marks=pytest.mark.braket,
    ),
    pytest.param(
        "hugr", "qamomile.hugr", "Selene", "selene_sim", marks=pytest.mark.hugr
    ),
    pytest.param(
        "quration",
        "qamomile.quration",
        "QurationLocal",
        "pyqret",
        marks=pytest.mark.quration,
    ),
]


@pytest.mark.backend_slow
@pytest.mark.parametrize(
    ("backend", "module", "endpoint_name", "dependency"), _BACKENDS[:6]
)
def test_filter_same_source_executes_sampling_and_expectation_across_backends(
    filter_witness, backend, module, endpoint_name, dependency
):
    """Each local integration executes both paths against analytic p(x)=x.

    The shared witness uses five qubits, degree one, and at most 512 shots.
    Commuting grouping measures the complete signal projector in one basis.
    """

    if dependency:
        pytest.importorskip(dependency)
    endpoint = getattr(import_module(module), endpoint_name)()
    experiment, state_program = filter_witness
    expected = sum(_expected_weights(experiment).values())
    tolerance = 6.0 * math.sqrt(expected * (1.0 - expected) / 512)
    compiled = experiment.compile(to=endpoint.target)
    assert compiled.artifact.source_digest.hex() == experiment.program.semantic_digest
    run = compiled.run(on=endpoint, shots=512, seed=0)

    assert run.execution.artifact is compiled.artifact
    assert run.success.shots == 512
    assert abs(run.success_probability - expected) <= tolerance
    assert run.samples is not None
    assert run.samples.total_occurrences == run.success.successes
    assert all(
        sample.energy == experiment.model.energy(sample.values)
        for sample in run.samples.samples
    )
    observed = qm.expectation(
        state_program,
        _zero_signal_projector(experiment),
        on=endpoint,
        shots=512,
        seed=0,
        grouping="qwc",
    )
    assert abs(observed.value - expected) <= tolerance


@pytest.mark.backend_slow
@pytest.mark.parametrize(
    ("backend", "module", "endpoint_name", "dependency"), _BACKENDS[:-1]
)
def test_filter_pauli_source_executes_across_capable_local_backends(
    backend, module, endpoint_name, dependency
):
    """The same Pauli filter executes typed samples and exact expectations."""

    if dependency:
        pytest.importorskip(dependency)
    endpoint = getattr(import_module(module), endpoint_name)()
    exercise_qsvt_filter_backend(endpoint=endpoint)


@pytest.mark.parametrize(
    ("backend", "module", "dependency", "code"),
    (
        pytest.param("hugr", "qamomile.hugr", "hugr", "Q8660", marks=pytest.mark.hugr),
        pytest.param(
            "quration",
            "qamomile.quration",
            "pyqret",
            "Q8904",
            marks=pytest.mark.quration,
        ),
    ),
)
def test_filter_shifted_lcu_rejects_unsupported_nested_controls(
    filter_witness, backend, module, dependency, code
):
    """LCU preparation under nested controls fails closed on restricted targets.

    Direct HUGR rejects multiply controlled RY. Quration rejects three controls
    on phase gates, including those in the simpler Pauli witness.
    """

    pytest.importorskip(dependency)
    integration = import_module(module)
    target = integration.Hugr() if backend == "hugr" else integration.Quration()
    experiment, _ = filter_witness
    with pytest.raises(qm.CapabilityError, match=code):
        experiment.compile(to=target)


@pytest.mark.quration
def test_filter_pauli_source_rejects_unsupported_quration_phase_controls():
    """Even zero-angle nested phases exceed Quration's declared control profile."""

    pytest.importorskip("pyqret")
    from qamomile.quration import QurationLocal

    with pytest.raises(qm.CapabilityError, match="Q8904.*at most two controls"):
        exercise_qsvt_filter_backend(endpoint=QurationLocal())
