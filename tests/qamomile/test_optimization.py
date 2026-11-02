"""Optimization-model replacement tests over Qamomile execution values."""

from __future__ import annotations

import itertools
import math
import pickle
import random
from types import SimpleNamespace

import ommx.v1
import pytest

import qamomile as qm
import qamomile.optimization.ommx as ommx_module
from qamomile.optimization import (
    BinaryModel,
    CompiledQAOAExperiment,
    DecodedSample,
    DecodedSamples,
    FQAOAExperiment,
    LocalSearch,
    LocalSearchMethod,
    OMMXProblem,
    PauliCorrelationEncoding,
    QAOAExperiment,
    QRACEncoding,
    QRACVariant,
    VarType,
)
from qamomile.persistence import deserialize, serialize
from qamomile.stdlib import qaoa_state


@qm.qkernel
def optimization_qaoa_sample(
    hamiltonian: qm.Const[qm.Observable],
    gammas: qm.Param[qm.Vector[qm.Real, 1]],
    betas: qm.Param[qm.Vector[qm.Real, 1]],
) -> qm.Shot[qm.Bits[1]]:
    """Sample one typed QAOA state generated from an optimization model."""

    return qm.measure(qaoa_state(1, 1, hamiltonian, gammas, betas))


def test_model_canonicalizes_domain_identities_and_preserves_original_ids() -> None:
    """BINARY and SPIN terms use distinct exact product identities."""

    binary = BinaryModel.from_hubo(
        {
            (9, 9, 2): 3.0,
            (2, 9): -1.0,
            (): 2.0,
            (7,): 0.0,
        },
        constant=0.5,
    )
    spin = BinaryModel.from_ising({(9, 9, 2): 3.0, (2,): -1.0})

    assert binary.vartype is VarType.BINARY
    assert binary.variables == (2, 9)
    assert binary.coefficients == {(2, 9): 2.0}
    assert binary.constant == 2.5
    assert binary.order == 2
    assert spin.coefficients == {(2,): 2.0}
    assert not hasattr(binary, "linear")
    assert not hasattr(binary, "quad")
    assert not hasattr(binary, "higher")
    assert not hasattr(binary, "as_vartype")
    with pytest.raises(TypeError):
        binary.coefficients[(2,)] = 1.0  # type: ignore[index]


def test_model_native_normal_form_retains_exact_sums_and_unbounded_ids() -> None:
    """Canonical aliases and energies use Rust without narrowing Python IDs."""

    huge = (1 << 20_000) + 3
    residue = float.fromhex("0x0.0000000000001p-1022")
    model = BinaryModel.from_hubo(
        {
            (0,): float.fromhex("0x1.fffffffffffffp+1023"),
            (0, 0): -float.fromhex("0x1.fffffffffffffp+1023"),
            (0, 0, 0): residue,
            (huge,): 0.25,
        },
        variables=(huge, 0),
    )

    assert model.variables == (0, huge)
    assert model.coefficients == {(0,): residue, (huge,): 0.25}
    assert model.energy((1, 0)) == residue

    restored = pickle.loads(pickle.dumps(model))
    assert restored == model
    assert restored.energy((1, 0)) == residue


def test_model_coefficient_normalizations_are_native_immutable_and_persistent() -> None:
    """Absolute-maximum and degree-wise RMS scaling retain model identity data."""

    model = BinaryModel.from_hubo(
        {(1,): 1.0, (3,): 2.0, (1, 3): 3.0},
        constant=5.0,
        variables=(8, 3, 1),
    )
    abs_max = model.normalize_by_abs_max()
    rms = model.normalize_by_rms()
    rms_factor = math.sqrt(11.5)

    assert model.coefficients == {(1,): 1.0, (1, 3): 3.0, (3,): 2.0}
    assert model.constant == 5.0
    assert abs_max.vartype is model.vartype
    assert abs_max.variables == rms.variables == model.variables
    assert abs_max.coefficients == {(1,): 1.0 / 3.0, (1, 3): 1.0, (3,): 2.0 / 3.0}
    assert abs_max.constant == 5.0 / 3.0
    assert rms.coefficients == {
        (1,): 1.0 / rms_factor,
        (1, 3): 3.0 / rms_factor,
        (3,): 2.0 / rms_factor,
    }
    assert rms.constant == 5.0 / rms_factor
    assert abs_max.energy({1: 1, 3: 0, 8: 0}) == model.energy({1: 1, 3: 0, 8: 0}) / 3.0
    assert BinaryModel.from_bytes(rms.to_bytes()) == rms


def test_model_coefficient_normalization_handles_constants_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Constant objectives are stable and no Python scaling fallback exists."""

    constant = BinaryModel.from_ising({}, constant=2.0, variables=(7,))
    assert constant.normalize_by_abs_max() == constant
    assert constant.normalize_by_rms() == constant

    extreme = BinaryModel.from_hubo(
        {(0,): float.fromhex("0x0.0000000000001p-1022")},
        constant=float.fromhex("0x1.fffffffffffffp+1023"),
    )
    with pytest.raises(ValueError, match="Q7565"):
        extreme.normalize_by_abs_max()

    from qamomile.optimization import model as model_module

    source = BinaryModel.from_hubo({(0,): 1.0})
    object.__setattr__(source, "_native_model", SimpleNamespace())
    with pytest.raises(TypeError, match="Q7560"):
        source.normalize_by_abs_max()
    with pytest.raises(TypeError, match="Q7560"):
        source.normalize_by_rms()

    class MalformedNormalization:
        """Return an invalid projection from both native normalization methods."""

        def normalized_by_abs_max(self) -> object:
            """Return one malformed absolute-maximum normalization."""

            return object()

        def normalized_by_rms(self) -> object:
            """Return one malformed RMS normalization."""

            return object()

    object.__setattr__(source, "_native_model", MalformedNormalization())
    monkeypatch.setattr(model_module.native, "SparsePolynomial", object)
    with pytest.raises(TypeError, match="Q7568"):
        source.normalize_by_abs_max()
    with pytest.raises(TypeError, match="Q7568"):
        source.normalize_by_rms()


def test_model_problem_artifact_is_canonical_portable_and_integrity_checked() -> None:
    """Equivalent host inputs share one Rust-owned problem identity."""

    first = BinaryModel.from_hubo(
        {(9, 2): 1.5, (2,): -0.25},
        constant=0.5,
        variables=(12, 9, 2),
    )
    second = BinaryModel.from_hubo(
        {(2,): -0.25, (2, 9): 1.5},
        constant=0.5,
        variables=(2, 12, 9),
    )

    payload = first.to_bytes()
    restored = BinaryModel.from_bytes(payload)

    assert payload == second.to_bytes()
    assert first.semantic_digest == second.semantic_digest
    assert restored == first
    assert restored.to_bytes() == payload
    assert pickle.loads(pickle.dumps(first)).to_bytes() == payload

    corrupted = bytearray(payload)
    corrupted[-1] ^= 1
    with pytest.raises(ValueError, match="Q7574"):
        BinaryModel.from_bytes(corrupted)


def test_model_problem_artifact_fails_closed_without_native_persistence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persistence never falls back to Python coefficient reconstruction."""

    from qamomile.optimization import model as model_module

    model = BinaryModel.from_hubo({(0,): 1.0})
    object.__setattr__(model, "_native_model", SimpleNamespace(semantic_digest="bad"))

    with pytest.raises(TypeError, match="Q7560"):
        model.to_bytes()
    with pytest.raises(TypeError, match="Q7576"):
        _ = model.semantic_digest
    monkeypatch.setattr(model_module.native, "SparsePolynomial", SimpleNamespace())
    with pytest.raises(TypeError, match="Q7560"):
        BinaryModel.from_bytes(b"not-a-problem")


def test_model_requires_a_well_formed_native_normal_form(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Polynomial semantics cannot fall back to Python normalization or evaluation."""

    from qamomile.optimization import model as model_module

    monkeypatch.setattr(model_module.native, "SparsePolynomial", None)
    with pytest.raises(TypeError, match="Q7560"):
        BinaryModel.from_hubo({(0,): 1.0})

    class MalformedPolynomial:
        """Return an invalid noncanonical native projection."""

        def __init__(self, *_args: object) -> None:
            self.domain = "BINARY"
            self.terms = [(["1", "0"], 1.0)]
            self.constant = 0.0
            self.variables = ["0", "1"]

    monkeypatch.setattr(model_module.native, "SparsePolynomial", MalformedPolynomial)
    with pytest.raises(TypeError, match="Q7568"):
        BinaryModel.from_hubo({(0,): 1.0})


def test_model_assignment_semantics_cross_the_native_boundary_once() -> None:
    """Rust alone owns original-ID layout, value-domain checks, and energy."""

    model = BinaryModel.from_hubo({(3,): 1.0, (8,): -2.0}, variables=(13, 8, 3))
    native_model = model._native_model
    calls = 0

    class ObservedPolynomial:
        """Observe the single checked assignment-and-energy entrypoint."""

        def energy_assignment(
            self,
            values: list[str],
            variables: list[str] | None,
        ) -> float:
            nonlocal calls
            calls += 1
            return native_model.energy_assignment(values, variables)

    object.__setattr__(model, "_native_model", ObservedPolynomial())

    assert model.energy({13: 0, 8: 1, 3: 1}) == -1.0
    assert calls == 1


def test_model_assignment_validation_fails_closed_in_native_code() -> None:
    """No Python fallback accepts a wrong layout, value, or native result."""

    model = BinaryModel.from_hubo({(3,): 1.0}, variables=(3, 8))
    with pytest.raises(ValueError, match="Q7567"):
        model.energy({3: 1})
    with pytest.raises(ValueError, match="Q7567"):
        model.energy({3: 1, 9: 0})
    with pytest.raises(ValueError, match="Q7567"):
        model.energy((1, 2))
    with pytest.raises(TypeError, match="integers"):
        model.energy((True, 0))

    class MissingAssignment:
        """Represent a native model predating checked assignments."""

        energy_assignment = None

    object.__setattr__(model, "_native_model", MissingAssignment())
    with pytest.raises(TypeError, match="Q7560"):
        model.energy((1, 0))

    class MalformedAssignment:
        """Return a malformed native energy projection."""

        def energy_assignment(self, *_args: object) -> object:
            return "1.0"

    object.__setattr__(model, "_native_model", MalformedAssignment())
    with pytest.raises(TypeError, match="Q7568"):
        model.energy((1, 0))


def test_model_retains_an_explicit_zero_cost_variable_universe() -> None:
    """Declared disconnected variables survive conversion and sample decoding."""

    binary = BinaryModel.from_hubo(
        {(9,): 2.0},
        variables=(12, 2, 9),
    )
    spin = binary.to_ising()
    decoded = spin.decode(
        qm.SampleResult.from_counts(
            backend="test",
            counts={(0, 1, 0): 2},
        )
    )

    assert binary.variables == (2, 9, 12)
    assert spin.variables == binary.variables
    assert spin.hamiltonian.num_qubits == 3
    assert decoded.samples == (DecodedSample(((2, 1), (9, -1), (12, 1)), 2.0, 2),)


@pytest.mark.parametrize(
    ("variables", "error", "message"),
    (
        ((0, 0), ValueError, "duplicate"),
        ((0, -1), ValueError, "non-negative"),
        ((0, 1.0), TypeError, "integers"),
        ((0,), ValueError, "absent"),
    ),
)
def test_model_rejects_an_invalid_explicit_variable_universe(
    variables: object,
    error: type[Exception],
    message: str,
) -> None:
    """The declared model layout must be a closed set of unique IDs."""

    with pytest.raises(error, match=message):
        BinaryModel.from_hubo({(1,): 1.0}, variables=variables)  # type: ignore[arg-type]


def test_arbitrary_order_domain_roundtrip_preserves_every_energy() -> None:
    """Exact substitution preserves a sparse-ID fifth-order objective."""

    binary = BinaryModel.from_hubo(
        {
            (1,): -0.25,
            (1, 4): 1.5,
            (1, 4, 7): -2.0,
            (1, 4, 7, 8, 12): 0.75,
        },
        constant=0.125,
    )
    spin = binary.to_ising()
    restored = spin.to_binary()

    assert spin.vartype is VarType.SPIN
    assert restored.variables == binary.variables
    for values in itertools.product((0, 1), repeat=binary.num_variables):
        assert restored.energy(values) == pytest.approx(binary.energy(values))
        spin_values = tuple(1 - 2 * value for value in values)
        assert spin.energy(spin_values) == pytest.approx(binary.energy(values))


def test_hamiltonian_is_the_dense_higher_order_ising_objective() -> None:
    """All polynomial degrees map to ordinary commuting Pauli-Z words."""

    model = BinaryModel.from_hubo(
        {(10,): 1.0, (10, 20): -2.0, (10, 20, 30): 0.5},
        constant=0.25,
    )
    hamiltonian = model.hamiltonian

    assert hamiltonian.num_qubits == 3
    assert hamiltonian.is_hermitian
    assert hamiltonian.is_pairwise_commuting
    assert max(len(word.factors) for word, _ in hamiltonian.terms) == 3
    for bits in itertools.product((0, 1), repeat=3):
        diagonal = hamiltonian.constant.real + sum(
            coefficient.real
            * math.prod(1 - 2 * bits[factor.index] for factor in word.factors)
            for word, coefficient in hamiltonian.terms
        )
        assert diagonal == pytest.approx(model.energy(bits))


def test_model_hamiltonian_is_one_fail_closed_native_projection() -> None:
    """Python does not reproduce polynomial-to-Pauli conversion semantics."""

    model = BinaryModel.from_hubo({(3,): 1.0, (3, 8): -2.0}, variables=(13, 8, 3))
    native_model = model._native_model
    calls = 0

    class ObservedPolynomial:
        """Observe only the canonical Hamiltonian projection."""

        def hamiltonian(self) -> object:
            nonlocal calls
            calls += 1
            return native_model.hamiltonian()

    object.__setattr__(model, "_native_model", ObservedPolynomial())
    hamiltonian = model.hamiltonian

    assert calls == 1
    assert hamiltonian.num_qubits == 3
    assert hamiltonian.is_hermitian

    class MissingHamiltonian:
        """Represent a native model predating Hamiltonian projection."""

        hamiltonian = None

    object.__setattr__(model, "_native_model", MissingHamiltonian())
    with pytest.raises(TypeError, match="Q7560"):
        _ = model.hamiltonian

    class MalformedHamiltonian:
        """Return a malformed native observable header."""

        def hamiltonian(self) -> object:
            return object()

    object.__setattr__(model, "_native_model", MalformedHamiltonian())
    with pytest.raises(TypeError, match="observable"):
        _ = model.hamiltonian


def test_decode_returns_typed_immutable_distribution_in_original_id_order() -> None:
    """Logical source-order bits decode without backend-specific conventions."""

    model = BinaryModel.from_ising({(3,): 1.0, (3, 8): -2.0}, constant=0.5)
    result = qm.SampleResult.from_counts(
        backend="test",
        counts={(0, 0): 3, (1, 0): 1},
    )

    decoded = model.decode(result)

    assert decoded.vartype is VarType.SPIN
    assert decoded.total_occurrences == 4
    assert decoded.samples[0].assignment == ((3, 1), (8, 1))
    assert decoded.samples[1].assignment == ((3, -1), (8, 1))
    assert decoded.mean_energy == pytest.approx(0.0)
    assert decoded.lowest.assignment == ((3, 1), (8, 1))
    assert decoded.lowest.energy == pytest.approx(-0.5)


def test_model_decode_crosses_the_native_boundary_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All logical rows decode in one call without row-wise energy crossings."""

    model = BinaryModel.from_hubo({(3,): 1.0, (8,): -2.0})
    native_model = model._native_model
    calls = 0

    class ObservedPolynomial:
        """Observe only the bulk decode entrypoint of one native model."""

        def decode_samples(self, summary: object) -> object:
            nonlocal calls
            calls += 1
            return native_model.decode_samples(summary)

        def energy(self, _values: object) -> float:
            raise AssertionError("sample decoding re-entered per-row energy")

    object.__setattr__(model, "_native_model", ObservedPolynomial())
    result = qm.SampleResult.from_counts(
        backend="test",
        counts={(0, 0): 2, (0, 1): 3, (1, 0): 1},
    )

    decoded = model.decode(result)

    assert calls == 1
    assert decoded.total_occurrences == 6
    assert len(decoded.samples) == 3


def test_decoded_sample_construction_uses_the_native_row_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Standalone rows share Rust layout, numeric, and occurrence semantics."""

    sample = DecodedSample(((3, -7), (8, 2)), 1, 1 << 200)
    assert sample == DecodedSample(((3, -7), (8, 2)), 1.0, 1 << 200)
    with pytest.raises(ValueError, match="Q7569.*ordered"):
        DecodedSample(((8, 2), (3, -7)), 1.0, 1)
    with pytest.raises(ValueError, match="Q7569.*ordered"):
        DecodedSample(((3, 2), (3, -7)), 1.0, 1)
    with pytest.raises(ValueError, match="Q7563"):
        DecodedSample(((-1, 0),), 1.0, 1)
    with pytest.raises(ValueError, match="Q7569.*finite"):
        DecodedSample(((3, 0),), math.nan, 1)
    with pytest.raises(ValueError, match="Q7569.*positive"):
        DecodedSample(((3, 0),), 1.0, 0)

    from qamomile.optimization import model as model_module

    monkeypatch.setattr(model_module.native, "validate_decoded_samples", None)
    with pytest.raises(TypeError, match="Q7560"):
        DecodedSample(((3, 0),), 1.0, 1)
    monkeypatch.setattr(
        model_module.native,
        "validate_decoded_samples",
        lambda _domain, _rows: object(),
    )
    with pytest.raises(TypeError, match="Q7568"):
        DecodedSample(((3, 0),), 1.0, 1)


def test_decoded_distribution_fails_closed_without_native_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Decoded distributions cannot restore Python validation or statistics."""

    from qamomile.optimization import model as model_module

    sample = DecodedSample(((0, 0),), 0.0, 1)
    validator = model_module.native.validate_decoded_samples
    monkeypatch.setattr(model_module.native, "validate_decoded_samples", None)
    with pytest.raises(TypeError, match="Q7560"):
        DecodedSamples(VarType.BINARY, (sample,))
    monkeypatch.setattr(
        model_module.native,
        "validate_decoded_samples",
        lambda _domain, _rows: object(),
    )
    with pytest.raises(TypeError, match="Q7568"):
        DecodedSamples(VarType.BINARY, (sample,))
    monkeypatch.setattr(model_module.native, "validate_decoded_samples", validator)
    with pytest.raises(ValueError, match="Q7569.*empty"):
        DecodedSamples(VarType.BINARY, ())
    with pytest.raises(ValueError, match="Q7569.*domain"):
        DecodedSamples(
            VarType.BINARY,
            (DecodedSample(((0, -1),), 0.0, 1),),
        )


def test_model_hamiltonian_drives_plain_qaoa_and_decodes_the_result() -> None:
    """Problem, standard library, execution, and decode share one public path."""

    model = BinaryModel.from_hubo({(42,): 1.0})
    record = qm.run(
        optimization_qaoa_sample,
        on=qm.Reference(),
        **{"hamiltonian": model.hamiltonian},
        parameters={"gammas": (0.0,), "betas": (0.0,)},
        shots=8,
        seed=4,
    )

    decoded = model.decode(record.result)

    assert decoded.vartype is VarType.BINARY
    assert decoded.total_occurrences == 8
    assert {index for sample in decoded.samples for index, _ in sample.assignment} == {
        42
    }


def test_qaoa_experiment_closes_topology_and_returns_decoded_execution() -> None:
    """The high-level lifecycle remains ordinary body-derived composition."""

    model = BinaryModel.from_hubo({(4,): -1.0, (4, 9): 0.5})
    experiment = QAOAExperiment(model, layers=2)
    endpoint = qm.Reference()
    compiled = experiment.compile(to=endpoint.target)

    study = qm.resources.logical(experiment.program)
    run = compiled.run(
        on=endpoint,
        gammas=(0.0, 0.0),
        betas=(0.0, 0.0),
        shots=6,
        seed=3,
    )

    assert "quantum.qaoa" not in experiment.program.inspect()
    assert study.metric(
        qm.resources.logical_operation_count("quantum.pauli_evolution")
    ).numeric_enclosure() == (2, 2)
    assert (
        run.execution.artifact.source_digest.hex() == experiment.program.semantic_digest
    )
    assert run.execution.artifact is compiled.artifact
    assert run.identity == compiled.identity == experiment.identity
    assert run.identity.problem_digest == model.semantic_digest
    assert run.invocation.values() == {
        "gammas": (0.0, 0.0),
        "betas": (0.0, 0.0),
    }
    assert run.samples.vartype is VarType.BINARY
    assert run.samples.total_occurrences == 6
    with pytest.raises(TypeError, match="process-local"):
        pickle.dumps(compiled)
    restored = deserialize(serialize(compiled.artifact))
    assert serialize(restored) == serialize(compiled.artifact)


def test_compiled_qaoa_rejects_an_artifact_for_another_experiment() -> None:
    """A target artifact cannot be attached to a different model identity."""

    expected = QAOAExperiment(BinaryModel.from_hubo({(4,): -1.0}), layers=1)
    foreign = QAOAExperiment(BinaryModel.from_hubo({(9,): 1.0}), layers=1)
    artifact = qm.compile(foreign.program, to=qm.Reference())

    with pytest.raises(ValueError, match="Q7670"):
        CompiledQAOAExperiment(expected, artifact)


@pytest.mark.parametrize(
    ("model", "layers", "error", "message"),
    (
        (BinaryModel.from_hubo({}), 1, ValueError, "at least one"),
        (BinaryModel.from_hubo({(0,): 1.0}), -1, ValueError, "nonnegative"),
        (BinaryModel.from_hubo({(0,): 1.0}), True, TypeError, "integer"),
    ),
)
def test_qaoa_experiment_rejects_invalid_static_topology(
    model: BinaryModel[VarType],
    layers: int,
    error: type[Exception],
    message: str,
) -> None:
    """Malformed compile-time topology fails before program construction."""

    with pytest.raises(error, match=message):
        QAOAExperiment(model, layers=layers)


@pytest.mark.parametrize(
    ("gammas", "betas", "error", "message"),
    (
        ((0.0,), (0.0, 0.0), ValueError, "gammas.*exactly 2"),
        ((0.0, 0.0), (0.0,), ValueError, "betas.*exactly 2"),
        ((0.0, math.inf), (0.0, 0.0), ValueError, "finite"),
        ((0.0, True), (0.0, 0.0), TypeError, "real numbers"),
    ),
)
def test_qaoa_experiment_validates_variational_coordinates(
    gammas: tuple[object, ...],
    betas: tuple[object, ...],
    error: type[Exception],
    message: str,
) -> None:
    """Parameter shape and finiteness fail at the host experiment boundary."""

    experiment = QAOAExperiment(BinaryModel.from_hubo({(0,): 1.0}), layers=2)
    with pytest.raises(error, match=message):
        experiment.run(
            on=qm.Reference(),
            gammas=gammas,  # type: ignore[arg-type]
            betas=betas,  # type: ignore[arg-type]
            shots=1,
        )


@pytest.mark.parametrize(
    ("factory", "coefficients", "error", "message"),
    (
        (BinaryModel.from_hubo, {(True,): 1.0}, TypeError, "integers"),
        (BinaryModel.from_hubo, {(-1,): 1.0}, ValueError, "non-negative"),
        (BinaryModel.from_hubo, {(0,): True}, TypeError, "real number"),
        (BinaryModel.from_ising, {(0,): math.inf}, ValueError, "finite"),
    ),
)
def test_model_rejects_ambiguous_or_nonfinite_input(
    factory: object,
    coefficients: dict[tuple[int, ...], float],
    error: type[Exception],
    message: str,
) -> None:
    """Host normalization rejects invalid scientific data eagerly."""

    with pytest.raises(error, match=message):
        factory(coefficients)  # type: ignore[operator]


def test_decode_rejects_a_result_for_another_problem_width() -> None:
    """A sample distribution cannot be silently truncated or padded."""

    model = BinaryModel.from_hubo({(0,): 1.0, (1,): 1.0})
    result = qm.SampleResult.from_counts(backend="test", counts={(0,): 1})

    with pytest.raises(ValueError, match="sample width 1.*model width 2"):
        model.decode(result)


@pytest.mark.parametrize("vartype", (VarType.BINARY, VarType.SPIN))
def test_local_search_reaches_an_arbitrary_order_one_flip_minimum(
    vartype: VarType,
) -> None:
    """Sparse original IDs and cubic terms use the model's native algebra."""

    binary = BinaryModel.from_hubo(
        {
            (2,): 0.5,
            (7,): -1.0,
            (11,): 0.25,
            (2, 7): -1.5,
            (2, 7, 11): -3.0,
        }
    )
    model = binary if vartype is VarType.BINARY else binary.to_ising()
    initial = (0, 0, 0) if vartype is VarType.BINARY else (1, 1, 1)

    result = LocalSearch(model).run(initial)

    assert result.energy <= model.energy(initial)
    values = dict(result.assignment)
    for variable in model.variables:
        neighbor = values.copy()
        neighbor[variable] = (
            1 - neighbor[variable] if vartype is VarType.BINARY else -neighbor[variable]
        )
        assert model.energy(neighbor) >= result.energy


def test_local_search_selection_and_step_bound_are_deterministic() -> None:
    """First and best improvement make their documented one-step choices."""

    model = BinaryModel.from_hubo({(4,): -1.0, (9,): -3.0})

    first = LocalSearch(model, method="first").run((0, 0), max_steps=1)
    best = LocalSearch(model, method=LocalSearchMethod.BEST).run(
        (0, 0),
        max_steps=1,
    )

    assert first.assignment == ((4, 1), (9, 0))
    assert best.assignment == ((4, 0), (9, 1))


def test_local_search_uses_exact_native_delta_accumulation() -> None:
    """A subnormal residue between cancelling extremes still selects a flip."""

    residue = float.fromhex("0x0.0000000000001p-1022")
    model = BinaryModel.from_hubo(
        {
            (0,): float.fromhex("0x1.fffffffffffffp+1023"),
            (0, 1): residue,
            (0, 2): -float.fromhex("0x1.fffffffffffffp+1023"),
        }
    )

    result = LocalSearch(model, method="first").run((1, 1, 1), max_steps=1)

    assert result.assignment == ((0, 0), (1, 1), (2, 1))
    assert result.energy == 0.0


@pytest.mark.parametrize("vartype", (VarType.BINARY, VarType.SPIN))
@pytest.mark.parametrize("method", (LocalSearchMethod.FIRST, LocalSearchMethod.BEST))
def test_local_search_matches_exhaustive_python_reference(
    vartype: VarType,
    method: LocalSearchMethod,
) -> None:
    """Random exact dyadic models retain the prior deterministic search path."""

    generator = random.Random(f"755-{vartype.value}-{method.value}")
    variables = (4, 9, 15)
    products = tuple(
        tuple(variables[index] for index in range(3) if mask & (1 << index))
        for mask in range(1, 8)
    )
    allowed = (0, 1) if vartype is VarType.BINARY else (-1, 1)
    for _ in range(30):
        coefficients = {
            product: coefficient / 4.0
            for product in products
            if (coefficient := generator.randint(-5, 5)) != 0
        }
        model = BinaryModel(
            vartype,
            coefficients,
            generator.randint(-3, 3) / 4.0,
            variables=variables,
        )
        initial = tuple(generator.choice(allowed) for _ in variables)
        max_steps = generator.choice((0, 1, 2, 4, None))
        expected = list(initial)
        steps = 0
        while max_steps is None or steps < max_steps:
            current_energy = model.energy(expected)
            candidates: list[tuple[float, int]] = []
            for position in range(len(expected)):
                neighbor = expected.copy()
                neighbor[position] = (
                    1 - neighbor[position]
                    if vartype is VarType.BINARY
                    else -neighbor[position]
                )
                delta = model.energy(neighbor) - current_energy
                if delta < 0.0:
                    candidates.append((delta, position))
                    if method is LocalSearchMethod.FIRST:
                        break
            if not candidates:
                break
            position = min(candidates)[1]
            expected[position] = (
                1 - expected[position]
                if vartype is VarType.BINARY
                else -expected[position]
            )
            steps += 1

        result = LocalSearch(model, method=method).run(initial, max_steps=max_steps)

        assert result.assignment == tuple(zip(variables, expected, strict=True))
        assert result.energy == model.energy(expected)


def test_local_search_requires_well_formed_native_result() -> None:
    """Search semantics cannot fall back to the former Python implementation."""

    model = BinaryModel.from_hubo({(0,): -1.0})

    class MissingSearch:
        """Represent an outdated native polynomial handle."""

        minimize = None

    object.__setattr__(model, "_native_model", MissingSearch())
    with pytest.raises(TypeError, match="Q7550"):
        LocalSearch(model).run((0,))

    class MalformedSearch:
        """Return an invalid native search projection."""

        def minimize_assignment(self, *_args: object) -> object:
            return ([2], -1.0, 1)

    object.__setattr__(model, "_native_model", MalformedSearch())
    with pytest.raises(TypeError, match="Q7558"):
        LocalSearch(model).run((0,))


def test_local_search_refines_and_merges_a_decoded_distribution() -> None:
    """All decoded rows retain multiplicity while converging to one minimum."""

    model = BinaryModel.from_hubo({(4,): -1.0, (9,): -2.0})
    decoded = model.decode(
        qm.SampleResult.from_counts(
            backend="test",
            counts={(0, 0): 2, (1, 0): 3, (0, 1): 5},
        )
    )

    refined = LocalSearch(model).refine(decoded)

    assert refined.samples == (DecodedSample(((4, 1), (9, 1)), -3.0, 10),)
    assert refined.total_occurrences == decoded.total_occurrences


def test_local_search_composes_with_original_ommx_evaluation() -> None:
    """Refinement reuses the existing immutable OMMX evaluation boundary."""

    variable = ommx.v1.DecisionVariable.binary(3)
    instance = ommx.v1.Instance.from_components(
        decision_variables=[variable],
        objective=-variable,
        constraints=[],
        sense=ommx.v1.Instance.MINIMIZE,
    )
    problem = OMMXProblem.from_instance(instance)
    decoded = problem.model.decode(
        qm.SampleResult.from_counts(backend="test", counts={(0,): 4})
    )

    refined = LocalSearch(problem.model).refine(decoded)
    evaluated = problem.evaluate(refined)

    assert refined.samples == (DecodedSample(((3, 1),), -1.0, 4),)
    assert evaluated.occurrences == {0: 4}
    assert evaluated.total_occurrences == 4
    assert all(
        evaluated.get(sample_id).objective == -1.0 for sample_id in evaluated.sample_ids
    )


@pytest.mark.parametrize(
    ("action", "error", "message"),
    (
        (lambda model: LocalSearch(model, method="unknown"), ValueError, "method"),
        (
            lambda model: LocalSearch(model).run((0,), max_steps=-1),
            ValueError,
            "max_steps",
        ),
        (
            lambda model: LocalSearch(model).run((0,), max_steps=True),
            TypeError,
            "max_steps",
        ),
        (
            lambda model: LocalSearch(model).run((0,), max_steps=1_000_001),
            ValueError,
            "Q7556",
        ),
        (lambda model: LocalSearch(model).run((2,)), ValueError, "Q7567"),
        (
            lambda model: LocalSearch(model).run(DecodedSample(((8, 0),), 0.0, 1)),
            ValueError,
            "Q7567",
        ),
    ),
)
def test_local_search_rejects_ambiguous_inputs(
    action: object,
    error: type[Exception],
    message: str,
) -> None:
    """Invalid policies, bounds, layouts, and domain values fail eagerly."""

    model = BinaryModel.from_hubo({(3,): -1.0})
    with pytest.raises(error, match=message):
        action(model)  # type: ignore[operator]


def test_ommx_problem_preserves_caller_and_separates_penalty_energy() -> None:
    """OMMX normalization mutates only its copy and original evaluation is clean."""

    left = ommx.v1.DecisionVariable.binary(0, name="left")
    right = ommx.v1.DecisionVariable.binary(1, name="right")
    instance = ommx.v1.Instance.from_components(
        decision_variables=[left, right],
        objective=-10.0 * left,
        constraints=[(left + right == 1).set_id(0)],
        sense=ommx.v1.Instance.MINIMIZE,
    )
    original_bytes = instance.to_bytes()

    problem = OMMXProblem.from_instance(instance, uniform_penalty_weight=2.0)
    decoded = problem.model.decode(
        qm.SampleResult.from_counts(backend="test", counts={(1, 1): 3})
    )
    evaluated = problem.evaluate(decoded)

    assert instance.to_bytes() == original_bytes
    assert decoded.samples[0].energy == pytest.approx(-8.0)
    assert evaluated.sample_ids == (0,)
    assert evaluated.occurrences == {0: 3}
    assert evaluated.total_occurrences == 3
    assert all(
        evaluated.get(sample_id).objective == -10.0
        for sample_id in evaluated.sample_ids
    )
    assert not any(evaluated.feasible.values())


def test_ommx_evaluation_never_expands_arbitrary_precision_occurrences() -> None:
    """One unique solution stays one OMMX row regardless of shot multiplicity."""

    variable = ommx.v1.DecisionVariable.binary(0)
    instance = ommx.v1.Instance.from_components(
        decision_variables=[variable],
        objective=variable,
        constraints=[],
        sense=ommx.v1.Instance.MINIMIZE,
    )
    problem = OMMXProblem.from_instance(instance)
    occurrences = 10**30
    decoded = DecodedSamples(
        VarType.BINARY,
        (DecodedSample(((0, 1),), 1.0, occurrences),),
    )

    evaluated = problem.evaluate(decoded)
    restored = pickle.loads(pickle.dumps(evaluated))

    assert evaluated.sample_ids == (0,)
    assert evaluated.occurrences == {0: occurrences}
    assert evaluated.total_occurrences == occurrences
    assert evaluated.sample_set.sample_ids == [0]
    assert restored == evaluated


def test_ommx_evaluation_rejects_corruption_and_budget_overflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persistence fails before parsing malformed or oversized external data."""

    with pytest.raises(ValueError, match="Q7672"):
        ommx_module._restore_ommx_evaluation(b"not an OMMX artifact", ((0, 1),))
    with pytest.raises(ValueError, match="Q7672.*canonical"):
        ommx_module._restore_ommx_evaluation(b"", ((1, 1),))

    monkeypatch.setattr(ommx_module, "_MAX_OMMX_EVALUATION_BYTES", 1)
    with pytest.raises(ValueError, match="Q7673.*64 MiB"):
        ommx_module._restore_ommx_evaluation(b"xx", ())


def test_ommx_problem_retains_unused_binary_variables_only() -> None:
    """The sampling ABI includes disconnected binaries but not substituted integers."""

    active = ommx.v1.DecisionVariable.binary(0, name="active")
    integer = ommx.v1.DecisionVariable.integer(
        1,
        lower=0,
        upper=3,
        name="integer",
    )
    unused = ommx.v1.DecisionVariable.binary(9, name="unused")
    instance = ommx.v1.Instance.from_components(
        decision_variables=[active, integer, unused],
        objective=active + 2 * integer,
        constraints=[],
        sense=ommx.v1.Instance.MINIMIZE,
    )

    problem = OMMXProblem.from_instance(instance)

    assert 0 in problem.model.variables
    assert 9 in problem.model.variables
    assert 1 not in problem.model.variables
    assert problem.model.num_variables == 4


def test_ommx_problem_reconstructs_original_integer_variables() -> None:
    """The retained transformed instance inverts its binary substitutions."""

    binary = ommx.v1.DecisionVariable.binary(0, name="binary")
    integer = ommx.v1.DecisionVariable.integer(
        1,
        lower=0,
        upper=3,
        name="integer",
    )
    instance = ommx.v1.Instance.from_components(
        decision_variables=[binary, integer],
        objective=binary + 2 * integer,
        constraints=[],
        sense=ommx.v1.Instance.MINIMIZE,
    )
    problem = OMMXProblem.from_instance(instance)
    zero_bits = (0,) * problem.model.num_variables
    decoded = problem.model.decode(
        qm.SampleResult.from_counts(backend="test", counts={zero_bits: 1})
    )

    solution = problem.evaluate(decoded).get(0)
    values = solution.decision_variables_df["value"]

    assert 0 in values.index
    assert 1 in values.index
    assert float(values.loc[0]) in {0.0, 1.0}
    assert 0.0 <= float(values.loc[1]) <= 3.0
    assert float(values.loc[1]).is_integer()


def test_ommx_problem_rejects_samples_from_another_model() -> None:
    """Original-problem evaluation cannot accept a coincidental bit width."""

    variable = ommx.v1.DecisionVariable.binary(7)
    instance = ommx.v1.Instance.from_components(
        decision_variables=[variable],
        objective=variable,
        constraints=[],
        sense=ommx.v1.Instance.MINIMIZE,
    )
    problem = OMMXProblem.from_instance(instance)
    other = BinaryModel.from_hubo({(8,): 1.0}).decode(
        qm.SampleResult.from_counts(backend="test", counts={(0,): 1})
    )

    with pytest.raises(ValueError, match="variables do not match"):
        problem.evaluate(other)


@pytest.mark.parametrize(
    ("num_variables", "order", "expected"),
    (
        (0, 3, 3),
        (9, 2, 2),
        (10, 2, 3),
        (27, 3, 3),
        (28, 3, 4),
    ),
)
def test_pce_capacity_is_the_exact_combinatorial_minimum(
    num_variables: int,
    order: int,
    expected: int,
) -> None:
    """PCE uses the smallest register containing enough distinct words."""

    assert PauliCorrelationEncoding.min_num_qubits(num_variables, order) == expected


def test_pce_enumerates_distinct_observables_in_original_variable_order() -> None:
    """Sparse variable IDs map to deterministic lexicographic Pauli words."""

    model = BinaryModel.from_ising({(10,): 1.0, (20,): 1.0, (30,): 1.0})
    encoding = PauliCorrelationEncoding(model, correlator_order=1)

    assert encoding.num_qubits == 1
    assert tuple(encoding.correlations) == (10, 20, 30)
    factors = tuple(
        observable.terms[0][0].factors for observable in encoding.observables
    )
    assert factors == (
        (qm.PauliFactor(0, qm.Pauli.X),),
        (qm.PauliFactor(0, qm.Pauli.Y),),
        (qm.PauliFactor(0, qm.Pauli.Z),),
    )
    assert all(observable.num_qubits == 1 for observable in encoding.observables)
    assert len(set(encoding.observables)) == 3


def test_pce_relaxed_cost_and_decode_share_the_spin_polynomial() -> None:
    """Continuous cost and sign rounding use one canonical variable layout."""

    model = BinaryModel.from_ising(
        {(10,): 1.0, (20,): -2.0, (10, 20): 0.5},
        constant=0.25,
    )
    encoding = PauliCorrelationEncoding(model, correlator_order=2)

    assert encoding.relaxed_cost((0.4, -0.5)) == pytest.approx(1.55)
    decoded = encoding.decode((0.4, -0.5))
    assert decoded.vartype is VarType.SPIN
    assert decoded.samples[0].assignment == ((10, 1), (20, -1))
    assert decoded.samples[0].energy == pytest.approx(2.75)
    assert encoding.relaxed_cost((1.0, -1.0)) == pytest.approx(
        decoded.samples[0].energy
    )


def test_pce_decodes_back_to_the_original_binary_domain() -> None:
    """A BINARY problem is not exposed as an internal SPIN assignment."""

    model = BinaryModel.from_hubo({(4,): 1.0, (4, 9): -1.0})
    decoded = PauliCorrelationEncoding(model, correlator_order=2).decode((0.2, -0.7))

    assert decoded.vartype is VarType.BINARY
    assert decoded.samples[0].assignment == ((4, 0), (9, 1))
    assert decoded.samples[0].energy == 0.0


def test_pce_pickle_reconstructs_its_native_plan() -> None:
    """Serialization persists stable inputs rather than a process-local handle."""

    encoding = PauliCorrelationEncoding(
        BinaryModel.from_ising({(4,): 1.0, (9,): -2.0}),
        correlator_order=2,
    )

    restored = pickle.loads(pickle.dumps(encoding))

    assert restored == encoding
    assert restored.relaxed_cost((0.25, -0.5)) == pytest.approx(1.25)


def test_pce_output_composes_with_ommx_original_evaluation() -> None:
    """PCE and OMMX remain independent adapters over one decoded type."""

    left = ommx.v1.DecisionVariable.binary(0)
    right = ommx.v1.DecisionVariable.binary(1)
    instance = ommx.v1.Instance.from_components(
        decision_variables=[left, right],
        objective=left - right,
        constraints=[(left + right == 1).set_id(0)],
        sense=ommx.v1.Instance.MINIMIZE,
    )
    problem = OMMXProblem.from_instance(instance)
    encoding = PauliCorrelationEncoding(problem.model, correlator_order=2)

    evaluated = problem.evaluate(encoding.decode((-0.8, 0.6)))
    solution = evaluated.get(0)

    assert solution.objective == pytest.approx(1.0)
    assert solution.feasible


@pytest.mark.parametrize(
    ("operation", "error", "message"),
    (
        (
            lambda: PauliCorrelationEncoding(
                BinaryModel.from_hubo({(0, 1, 2): 1.0}),
                correlator_order=2,
            ),
            ValueError,
            "quadratic",
        ),
        (
            lambda: PauliCorrelationEncoding(
                BinaryModel.from_ising({(0,): 1.0, (1,): 1.0}),
                correlator_order=2,
                num_qubits=1,
            ),
            ValueError,
            "too small",
        ),
        (
            lambda: PauliCorrelationEncoding.min_num_qubits(1, True),
            TypeError,
            "integer",
        ),
        (
            lambda: PauliCorrelationEncoding.min_num_qubits(-1, 2),
            ValueError,
            "non-negative",
        ),
    ),
)
def test_pce_rejects_invalid_topology(
    operation: object,
    error: type[Exception],
    message: str,
) -> None:
    """Unsupported PCE scientific domains fail at construction."""

    with pytest.raises(error, match=message):
        operation()  # type: ignore[operator]


@pytest.mark.parametrize(
    ("values", "error", "message"),
    (
        ((0.2,), ValueError, "expected 2"),
        ((0.2, math.nan), ValueError, "finite"),
        ((0.2, 1.01), ValueError, r"\[-1, 1\]"),
        ((0.2, True), TypeError, "real numbers"),
    ),
)
def test_pce_rejects_invalid_expectation_coordinates(
    values: tuple[object, ...],
    error: type[Exception],
    message: str,
) -> None:
    """Expectation coordinates are complete, finite, and physically bounded."""

    encoding = PauliCorrelationEncoding(
        BinaryModel.from_ising({(0,): 1.0, (1,): 1.0}),
        correlator_order=2,
    )
    with pytest.raises(error, match=message):
        encoding.decode(values)  # type: ignore[arg-type]


def test_pce_fails_closed_for_missing_or_malformed_native_authority() -> None:
    """PCE never falls back to Python planning, evaluation, or decoding."""

    missing = BinaryModel.from_ising({(0,): 1.0})
    object.__setattr__(missing, "_native_model", SimpleNamespace())
    with pytest.raises(TypeError, match="Q7570"):
        PauliCorrelationEncoding(missing, correlator_order=1)

    malformed = BinaryModel.from_ising({(0,): 1.0, (1,): 1.0})
    object.__setattr__(
        malformed,
        "_native_model",
        SimpleNamespace(
            pauli_correlation_plan=lambda _order, _width: SimpleNamespace(
                width=1,
                words=[[(0, "X")], [(0, "X")]],
            )
        ),
    )
    with pytest.raises(TypeError, match="Q7575"):
        PauliCorrelationEncoding(malformed, correlator_order=1)

    encoding = PauliCorrelationEncoding(
        BinaryModel.from_ising({(0,): 1.0}),
        correlator_order=1,
    )
    object.__setattr__(
        encoding,
        "_native_plan",
        SimpleNamespace(
            relaxed_cost=lambda _values: "not-a-cost",
            decode=lambda _values: ([0], 1.0),
        ),
    )
    with pytest.raises(TypeError, match="Q7575"):
        encoding.relaxed_cost((0.5,))
    with pytest.raises(TypeError, match="Q7575"):
        encoding.decode((0.5,))


@pytest.mark.parametrize(
    ("variant", "capacity"),
    (
        (QRACVariant.QRAC21, 2),
        (QRACVariant.QRAC31, 3),
        (QRACVariant.QRAC32, 3),
    ),
)
def test_qrac_coloring_is_deterministic_independent_and_capacity_bounded(
    variant: QRACVariant,
    capacity: int,
) -> None:
    """Interacting variables never share one random-access code block."""

    model = BinaryModel.from_ising(
        {
            (0,): 1.0,
            (1,): 1.0,
            (2,): 1.0,
            (3,): 1.0,
            (0, 1): -1.0,
            (1, 2): -1.0,
            (2, 3): -1.0,
        }
    )
    encoding = QRACEncoding(model, variant)
    group_of = {
        variable: group_index
        for group_index, group in enumerate(encoding.groups)
        for variable in group
    }

    assert all(len(group) <= capacity for group in encoding.groups)
    assert group_of[0] != group_of[1]
    assert group_of[1] != group_of[2]
    assert group_of[2] != group_of[3]
    assert tuple(encoding.correlations) == model.variables
    assert all(
        observable.num_qubits == encoding.num_qubits
        for observable in encoding.observables
    )


def test_qrac21_linear_relaxation_uses_occupancy_scaling() -> None:
    """Two noninteracting spins share Z/X coordinates with sqrt(2) scale."""

    encoding = QRACEncoding(
        BinaryModel.from_ising({(4,): 2.0, (9,): -3.0}, constant=0.5),
        QRACVariant.QRAC21,
    )

    assert encoding.groups == ((4, 9),)
    assert encoding.num_qubits == 1
    assert encoding.observables == (qm.Z(0), qm.X(0))
    assert encoding.hamiltonian.constant.real == 0.5
    coefficient_by_pauli = {
        word.factors[0].pauli: coefficient.real
        for word, coefficient in encoding.hamiltonian.terms
    }
    assert coefficient_by_pauli[qm.Pauli.Z] == pytest.approx(2.0 * math.sqrt(2))
    assert coefficient_by_pauli[qm.Pauli.X] == pytest.approx(-3.0 * math.sqrt(2))


def test_qrac32_uses_two_qubit_prime_observables_for_shared_groups() -> None:
    """Three variables share normalized prime coordinates on two qubits."""

    encoding = QRACEncoding(
        BinaryModel.from_ising({(0,): 1.0, (1,): 1.0, (2,): 1.0}),
        QRACVariant.QRAC32,
    )

    assert encoding.groups == ((0, 1, 2),)
    assert encoding.num_qubits == 2
    assert all(observable.num_qubits == 2 for observable in encoding.observables)
    assert all(len(observable.terms) == 3 for observable in encoding.observables)
    assert encoding.hamiltonian.is_hermitian


def test_space_efficient_qrac_uses_z_for_same_qubit_interaction() -> None:
    """The sequential 2:1 encoding retains its special local product rule."""

    encoding = QRACEncoding(
        BinaryModel.from_ising({(0, 1): 2.0}),
        QRACVariant.SPACE_EFFICIENT,
    )

    assert encoding.groups == ((0, 1),)
    assert encoding.num_qubits == 1
    assert encoding.observables == (qm.X(0), qm.Y(0))
    assert encoding.hamiltonian.terms == (
        (
            qm.PauliWord((qm.PauliFactor(0, qm.Pauli.Z),)),
            complex(2.0 * math.sqrt(3.0)),
        ),
    )


def test_qrac_decode_aggregates_rounding_rows_in_the_original_domain() -> None:
    """Repeated rounded rows become occurrences instead of duplicate samples."""

    model = BinaryModel.from_hubo({(10,): 1.0, (10, 20): -2.0})
    encoding = QRACEncoding(model, QRACVariant.QRAC31)
    decoded = encoding.decode_many(((1, -1), (1, -1), (-1, 1)))

    assert decoded.vartype is VarType.BINARY
    assert decoded.total_occurrences == 3
    assert decoded.samples == (
        DecodedSample(((10, 0), (20, 1)), 0.0, 2),
        DecodedSample(((10, 1), (20, 0)), 1.0, 1),
    )
    sign_sample = encoding.sign_decode((0.2, -0.3)).samples[0]
    assert sign_sample.assignment == decoded.samples[0].assignment
    assert sign_sample.energy == decoded.samples[0].energy


def test_qrac_decode_composes_with_ommx_evaluation() -> None:
    """Rounded QRAC rows reuse the same original-problem evaluation boundary."""

    left = ommx.v1.DecisionVariable.binary(0)
    right = ommx.v1.DecisionVariable.binary(1)
    instance = ommx.v1.Instance.from_components(
        decision_variables=[left, right],
        objective=left + 2 * right,
        constraints=[],
        sense=ommx.v1.Instance.MINIMIZE,
    )
    problem = OMMXProblem.from_instance(instance)
    encoding = QRACEncoding(problem.model, QRACVariant.QRAC21)

    evaluated = problem.evaluate(encoding.decode((-1, 1)))

    assert evaluated.get(0).objective == pytest.approx(1.0)


def test_qrac_pickle_reconstructs_its_native_plan() -> None:
    """Serialization persists the model and variant rather than a native handle."""

    encoding = QRACEncoding(
        BinaryModel.from_ising({(4,): 1.0, (9,): -2.0}),
        QRACVariant.QRAC21,
    )

    restored = pickle.loads(pickle.dumps(encoding))

    assert restored == encoding
    assert restored.sign_decode((0.25, -0.5)) == encoding.sign_decode((0.25, -0.5))


def test_qrac_fails_closed_for_missing_or_malformed_native_authority() -> None:
    """QRAC never selects Python grouping, observable, or decode semantics."""

    missing = BinaryModel.from_ising({(0,): 1.0})
    object.__setattr__(missing, "_native_model", SimpleNamespace())
    with pytest.raises(TypeError, match="Q7580"):
        QRACEncoding(missing)

    malformed = BinaryModel.from_ising({(0,): 1.0})
    object.__setattr__(
        malformed,
        "_native_model",
        SimpleNamespace(
            qrac_plan=lambda _variant: SimpleNamespace(
                width=1,
                max_decode_rows=1_000_000,
                max_decode_cells=10_000_000,
                groups=[[0], [0]],
                observables=[None],
                hamiltonian=None,
            )
        ),
    )
    with pytest.raises(TypeError, match="Q7587"):
        QRACEncoding(malformed)

    encoding = QRACEncoding(BinaryModel.from_ising({(0,): 1.0}))
    object.__setattr__(
        encoding,
        "_native_plan",
        SimpleNamespace(sign_decode=lambda _values: [([0], 0.0, 1)]),
    )
    with pytest.raises(TypeError, match="Q7587"):
        encoding.sign_decode((0.5,))
    bounded = QRACEncoding(BinaryModel.from_ising({(0,): 1.0}))
    with pytest.raises(ValueError, match="Q7585"):
        bounded.decode_many(range(1_000_001))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("operation", "error", "message"),
    (
        (
            lambda: QRACEncoding(
                BinaryModel.from_hubo({(0, 1, 2): 1.0}),
                QRACVariant.QRAC31,
            ),
            ValueError,
            "quadratic",
        ),
        (
            lambda: QRACEncoding(
                BinaryModel.from_ising({(0,): 1.0}),
                "3,1",  # type: ignore[arg-type]
            ),
            TypeError,
            "QRACVariant",
        ),
        (
            lambda: QRACEncoding(
                BinaryModel.from_ising({(0,): 1.0, (1,): 1.0}),
            ).decode((1, 0)),
            ValueError,
            r"-1 or \+1",
        ),
        (
            lambda: QRACEncoding(
                BinaryModel.from_ising({(0,): 1.0, (1,): 1.0}),
            ).sign_decode((0.0, math.inf)),
            ValueError,
            "finite",
        ),
    ),
)
def test_qrac_rejects_invalid_scientific_coordinates(
    operation: object,
    error: type[Exception],
    message: str,
) -> None:
    """QRAC topology and rounded coordinates fail closed."""

    with pytest.raises(error, match=message):
        operation()  # type: ignore[operator]


def _fqaoa_grid_instance(
    *,
    ids: tuple[int, int, int, int] = (0, 1, 2, 3),
    particles: int = 2,
) -> ommx.v1.Instance:
    """Build one two-site, two-orbital binary OMMX instance."""

    variables = [
        ommx.v1.DecisionVariable.binary(
            variable_id,
            name="x",
            subscripts=[site, orbital],
        )
        for variable_id, (site, orbital) in zip(
            ids,
            ((0, 0), (0, 1), (1, 0), (1, 1)),
            strict=True,
        )
    ]
    return ommx.v1.Instance.from_components(
        decision_variables=variables,
        objective=(
            variables[0]
            + 2 * variables[1]
            - variables[2]
            + 0.5 * variables[0] * variables[3]
        ),
        constraints=[(sum(variables) == particles).set_id(17)],
        sense=ommx.v1.Instance.MINIMIZE,
    )


def test_fqaoa_experiment_closes_geometry_and_retains_runtime_coordinates() -> None:
    """Only variational coordinates survive the closed FQAOA program ABI."""

    experiment = FQAOAExperiment.from_instance(
        _fqaoa_grid_instance(),
        particles=2,
        layers=2,
        uniform_penalty_weight=4.0,
    )
    runtime_parameters = tuple(
        parameter.name
        for parameter in experiment.program.entry.parameters
        if str(parameter.value.type).startswith("Param[")
    )
    study = qm.resources.logical(experiment.program)

    assert experiment.num_sites == 2
    assert experiment.num_orbitals == 2
    assert experiment.num_qubits == 4
    assert experiment.variable_order == (0, 2, 1, 3)
    assert len(experiment.givens_pairs) == len(experiment.givens_angles)
    assert runtime_parameters == ("hopping", "gammas", "betas")
    assert "quantum.fqaoa" not in experiment.program.inspect()
    for bits in itertools.product((0, 1), repeat=experiment.num_qubits):
        diagonal = experiment.hamiltonian.constant.real + sum(
            coefficient.real
            * math.prod(1 - 2 * bits[factor.index] for factor in word.factors)
            for word, coefficient in experiment.hamiltonian.terms
        )
        assignment = dict(zip(experiment.variable_order, bits, strict=True))
        assert diagonal == pytest.approx(experiment.problem.model.energy(assignment))
    assert study.metric(
        qm.resources.logical_operation_count("quantum.pauli_evolution")
    ).numeric_enclosure() == (2, 2)


def test_fqaoa_run_preserves_particles_and_decodes_original_sparse_ids() -> None:
    """Physical cyclic mode order is inverted before typed model decoding."""

    experiment = FQAOAExperiment.from_instance(
        _fqaoa_grid_instance(ids=(40, 7, 20, 3)),
        particles=2,
        layers=1,
        uniform_penalty_weight=3.0,
    )
    endpoint = qm.Reference()
    compiled = experiment.compile(to=endpoint.target)
    run = compiled.run(
        on=endpoint,
        gammas=(0.0,),
        betas=(0.0,),
        hopping=0.0,
        shots=12,
        seed=9,
    )

    assert experiment.variable_order == (40, 20, 7, 3)
    assert run.execution.artifact is compiled.artifact
    assert run.identity == compiled.identity == experiment.identity
    assert run.identity.problem_digest == experiment.problem.model.semantic_digest
    assert run.invocation.values() == {
        "hopping": 0.0,
        "gammas": (0.0,),
        "betas": (0.0,),
    }
    assert run.samples.total_occurrences == 12
    assert all(
        sum(value for _, value in sample.assignment) == 2
        for sample in run.samples.samples
    )
    assert all(
        tuple(index for index, _ in sample.assignment) == (3, 7, 20, 40)
        for sample in run.samples.samples
    )
    assert all(
        sample.energy == pytest.approx(experiment.problem.model.energy(sample.values))
        for sample in run.samples.samples
    )
    evaluated = experiment.problem.evaluate(run.samples)
    assert len(evaluated.sample_ids) == len(run.samples.samples)
    assert evaluated.total_occurrences == 12
    assert evaluated.occurrences == {
        sample_id: sample.occurrences
        for sample_id, sample in enumerate(run.samples.samples)
    }
    assert all(evaluated.feasible.values())


def test_fqaoa_one_particle_givens_preparation_is_uniform_on_the_ring() -> None:
    """The constant occupied orbital gives equal single-mode marginals."""

    experiment = FQAOAExperiment.from_instance(
        _fqaoa_grid_instance(particles=1),
        particles=1,
    )
    run = experiment.run(
        on=qm.Reference(),
        gammas=(0.0,),
        betas=(0.0,),
        hopping=0.0,
        shots=128,
        seed=23,
    )
    marginals = [0] * experiment.num_qubits
    for sample in run.samples.samples:
        physical = {
            variable: index for index, variable in enumerate(experiment.variable_order)
        }
        for variable, value in sample.assignment:
            marginals[physical[variable]] += value * sample.occurrences

    expected = 0.25
    tolerance = 6.0 * math.sqrt(
        expected * (1.0 - expected) / run.samples.total_occurrences
    )
    assert all(
        abs(marginal / run.samples.total_occurrences - expected) <= tolerance
        for marginal in marginals
    )


def test_fqaoa_pickle_reconstructs_its_native_plan() -> None:
    """Serialization persists public experiment inputs rather than a native handle."""

    experiment = FQAOAExperiment.from_instance(
        _fqaoa_grid_instance(ids=(40, 7, 20, 3)),
        particles=2,
    )

    restored = pickle.loads(pickle.dumps(experiment))

    assert restored == experiment
    assert restored.variable_order == experiment.variable_order
    assert restored.hamiltonian == experiment.hamiltonian


def test_fqaoa_fails_closed_for_missing_or_malformed_native_authority() -> None:
    """FQAOA never selects Python geometry, preparation, cost, or decode semantics."""

    missing = OMMXProblem.from_instance(_fqaoa_grid_instance())
    object.__setattr__(missing.model, "_native_model", SimpleNamespace())
    with pytest.raises(TypeError, match="Q7590"):
        FQAOAExperiment(missing, particles=2)

    malformed = OMMXProblem.from_instance(_fqaoa_grid_instance())
    object.__setattr__(
        malformed.model,
        "_native_model",
        SimpleNamespace(
            fqaoa_plan=lambda _ids, _positions, _particles: SimpleNamespace(
                num_sites=2,
                num_orbitals=2,
                variable_order=[0, 0, 1, 2],
                givens_pairs=[],
                givens_angles=[],
                hamiltonian=None,
                max_decode_rows=1_000_000,
                max_decode_cells=10_000_000,
            )
        ),
    )
    with pytest.raises(TypeError, match="Q7590"):
        FQAOAExperiment(malformed, particles=2)

    experiment = FQAOAExperiment.from_instance(
        _fqaoa_grid_instance(),
        particles=2,
    )
    result = qm.SampleResult.from_counts(
        backend="test",
        counts={(1, 1, 0, 0): 1},
        expected_width=4,
    )
    object.__setattr__(
        experiment,
        "_native_plan",
        SimpleNamespace(decode=lambda _rows: [([0], 0.0, 1)]),
    )
    with pytest.raises(TypeError, match="Q7590"):
        experiment._decode(result)


@pytest.mark.parametrize("particles", (0, 4))
def test_fqaoa_extreme_particle_sectors_need_no_givens_rotations(
    particles: int,
) -> None:
    """Empty and full sectors compile with a statically empty preparation."""

    experiment = FQAOAExperiment.from_instance(
        _fqaoa_grid_instance(particles=particles),
        particles=particles,
    )

    assert experiment.givens_pairs == ()
    assert experiment.givens_angles == ()
    assert experiment.program.entry.contract.kind.value == "instrument"


def test_fqaoa_rejects_a_mismatched_cardinality_sector() -> None:
    """An explicit cardinality constraint cannot disagree with preparation."""

    with pytest.raises(ValueError, match="cardinality constraint 17"):
        FQAOAExperiment.from_instance(
            _fqaoa_grid_instance(particles=2),
            particles=1,
        )


def test_fqaoa_rejects_nonrectangular_and_auxiliary_mode_layouts() -> None:
    """Every sampled cost variable must own exactly one fermionic coordinate."""

    malformed = ommx.v1.DecisionVariable.binary(0, subscripts=[0])
    malformed_instance = ommx.v1.Instance.from_components(
        decision_variables=[malformed],
        objective=malformed,
        constraints=[],
        sense=ommx.v1.Instance.MINIMIZE,
    )
    grid = _fqaoa_grid_instance()
    constrained = ommx.v1.Instance.from_components(
        decision_variables=list(grid.decision_variables),
        objective=grid.objective,
        constraints=[
            (grid.decision_variables[0] + grid.decision_variables[1] <= 1).set_id(9)
        ],
        sense=ommx.v1.Instance.MINIMIZE,
    )

    with pytest.raises(ValueError, match="exactly two"):
        FQAOAExperiment.from_instance(malformed_instance, particles=1)
    with pytest.raises(ValueError, match="auxiliary variables"):
        FQAOAExperiment.from_instance(constrained, particles=1)


@pytest.mark.parametrize(
    ("particles", "layers", "error", "message"),
    (
        (True, 1, TypeError, "particles.*integer"),
        (-1, 1, ValueError, "between zero"),
        (5, 1, ValueError, "between zero"),
        (2, 0, ValueError, "layers.*positive"),
        (2, True, TypeError, "layers.*integer"),
    ),
)
def test_fqaoa_rejects_invalid_static_coordinates(
    particles: object,
    layers: object,
    error: type[Exception],
    message: str,
) -> None:
    """Static FQAOA topology fails before compiler construction."""

    with pytest.raises(error, match=message):
        FQAOAExperiment.from_instance(
            _fqaoa_grid_instance(),
            particles=particles,  # type: ignore[arg-type]
            layers=layers,  # type: ignore[arg-type]
        )
