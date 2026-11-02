"""Observable algebra and abstract Pauli-evolution tests."""

from __future__ import annotations

import importlib
import math
from typing import Literal

import pytest

import qamomile as qm
from qamomile.ir import Program

from .conftest import PAULI_EVOLUTION_SOURCE, STATE_PREPARATION_SOURCE

SYMBOLIC_OBSERVABLE_SOURCE = """
import qamomile as qm

@qm.qkernel
def transverse_field(
    n: qm.Const[qm.Int],
    coefficients: qm.Const[qm.Vector[qm.Real, n]],
) -> qm.Observable:
    hamiltonian = qm.zero_observable(n)
    for index in range(n):
        hamiltonian = hamiltonian + coefficients[index] * qm.X(index)
    return hamiltonian

@qm.qkernel
def coupled_model(
    coupling: qm.Const[qm.Real],
    field: qm.Const[qm.Real],
) -> qm.Observable:
    return coupling * qm.X(0) * qm.Y(1) + field * qm.Z(0) + 0.25

@qm.qkernel
def evolve(
    n: qm.Const[qm.Int],
    coefficients: qm.Const[qm.Vector[qm.Real, n]],
    time: qm.Param[qm.Real],
) -> qm.Shot[qm.Bits[n]]:
    hamiltonian = transverse_field(n, coefficients)
    state = qm.pauli_evolve(qm.qubits(n), hamiltonian, time)
    return qm.measure(state)
"""


def _pauli_program() -> Program:
    """Build a two-qubit exact Pauli evolution program."""

    hamiltonian = 0.5 * qm.Z(0) * qm.Z(1)
    return qm.build_source(PAULI_EVOLUTION_SOURCE).specialize(
        "pauli_evolution",
        **{"hamiltonian": hamiltonian},
    )


def test_symbolic_observable_recipe_stays_structural_until_materialization() -> None:
    """A portable builder retains its loop and realizes one canonical value."""

    module = qm.build_source(
        SYMBOLIC_OBSERVABLE_SOURCE,
        module_name="symbolic_observable",
    )
    inspection = module.function("transverse_field").inspect()
    program = module.specialize(
        "transverse_field",
        **{"n": 2, "coefficients": (0.5, -1.25)},
    )
    restored = qm.Program.from_bytes(program.to_bytes())
    hamiltonian = qm.evaluate(restored)
    coupled = qm.evaluate(
        module.specialize(
            "coupled_model",
            **{"coupling": 0.5, "field": 2.0},
        )
    )

    assert inspection.count("qcore.for") == 1
    assert inspection.count("classical.observable_zero") == 1
    assert inspection.count("classical.observable_pauli") == 1
    assert hamiltonian == 0.5 * qm.X(0) - 1.25 * qm.X(1)
    assert hamiltonian.num_qubits == 2
    assert coupled == 0.5 * qm.X(0) * qm.Y(1) + 2.0 * qm.Z(0) + 0.25
    assert restored.to_bytes() == program.to_bytes()


def test_symbolic_observable_drives_reference_pauli_evolution() -> None:
    """Const coefficient data materializes before abstract Pauli evolution."""

    program = qm.build_source(
        SYMBOLIC_OBSERVABLE_SOURCE,
        module_name="symbolic_observable_evolution",
    ).specialize(
        "evolve",
        **{"n": 2, "coefficients": (1.0, 0.0)},
    )

    assert qm.evaluate(program, time=math.pi / 2) == (1, 0)
    assert "quantum.pauli_evolution" in program.inspect()


def test_symbolic_observable_rejects_runtime_coefficients() -> None:
    """Observable construction remains in the Const stage in the first version."""

    source = """
import qamomile as qm

@qm.qkernel
def invalid(coefficient: qm.Param[qm.Real]) -> qm.Observable:
    return coefficient * qm.X(0)
"""

    with pytest.raises(qm.TypeCheckError, match="Q4504:.*Const Observable"):
        qm.build_source(source, module_name="runtime_observable")


def test_zero_observable_preserves_empty_width_and_rejects_negative_width() -> None:
    """The additive identity retains width zero and fails on invalid dimensions."""

    source = """
import qamomile as qm

@qm.qkernel
def empty() -> qm.Observable:
    return qm.zero_observable(0)

@qm.qkernel
def invalid() -> qm.Observable:
    return qm.zero_observable(-1)
"""
    module = qm.build_source(source, module_name="zero_observable")

    assert qm.evaluate(module.specialize("empty")) == qm.Hamiltonian.zero()
    with pytest.raises(qm.SpecializationError, match="Q7404"):
        module.specialize("invalid")


def test_observable_algebra_is_canonical_and_portable() -> None:
    """Pauli products simplify once and round-trip as closed literal data."""

    hamiltonian = 0.5 * qm.X(0) * qm.X(1) + 2.0 * qm.Z(0) + 0.25
    restored = qm.Hamiltonian.from_portable(hamiltonian.to_portable())

    assert restored == hamiltonian
    assert qm.X(0) * qm.X(0) == qm.Hamiltonian.identity(1, num_qubits=1)
    assert qm.commutator(qm.X(0), qm.Y(0)) == 2j * qm.Z(0)


def test_observable_multiplication_crosses_the_native_algebra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pauli phases and canonical products come from the required Rust boundary."""

    observable_module = importlib.import_module("qamomile.observable")
    native_multiply = observable_module.native.multiply_observables
    calls = 0
    payloads: tuple[bytes, bytes] | None = None

    def record_native_multiplication(left: bytes, right: bytes) -> object:
        """Record one compiler-owned observable product."""

        nonlocal calls, payloads
        calls += 1
        payloads = (left, right)
        return native_multiply(left, right)

    left = qm.X(0)
    right = qm.Y(0)
    monkeypatch.setattr(
        observable_module.native,
        "multiply_observables",
        record_native_multiplication,
    )

    assert left * right == 1j * qm.Z(0)
    assert calls == 1
    assert payloads is not None
    assert payloads[0] is left._native_literal
    assert payloads[1] is right._native_literal


def test_observable_affine_algebra_crosses_the_native_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Addition, subtraction, and scalar action use one Rust coefficient law."""

    observable_module = importlib.import_module("qamomile.observable")
    native_affine = observable_module.native.affine_observable
    native_combine = observable_module.native.add_scaled_observable
    affine_calls: list[tuple[bytes, tuple[float, float], tuple[float, float]]] = []
    combine_calls: list[tuple[bytes, bytes, tuple[float, float]]] = []

    def record_native_affine(
        payload: bytes,
        scale: tuple[float, float],
        shift: tuple[float, float],
    ) -> object:
        """Record and delegate one compiler-owned affine operation."""

        affine_calls.append((payload, scale, shift))
        return native_affine(payload, scale, shift)

    def record_native_combination(
        left: bytes,
        right: bytes,
        scale: tuple[float, float],
    ) -> object:
        """Record and delegate one compiler-owned binary combination."""

        combine_calls.append((left, right, scale))
        return native_combine(left, right, scale)

    left = qm.X(0) + 2
    right = qm.Y(0)
    monkeypatch.setattr(
        observable_module.native, "affine_observable", record_native_affine
    )
    monkeypatch.setattr(
        observable_module.native,
        "add_scaled_observable",
        record_native_combination,
    )

    assert 3 - 2j * left == 3 - 4j - 2j * qm.X(0)
    assert left - right == 2 + qm.X(0) - qm.Y(0)
    assert affine_calls[0][0] is left._native_literal
    assert combine_calls[0][0] is left._native_literal
    assert combine_calls[0][1] is right._native_literal


def test_observable_affine_algebra_fails_closed_and_rejects_overflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No Python fallback or nonfinite compiler-owned result can escape."""

    observable = qm.X(0)
    observable_module = importlib.import_module("qamomile.observable")
    maximum = float.fromhex("0x1.fffffffffffffp+1023")

    with pytest.raises(ValueError, match="Q7411"):
        _ = (maximum * observable) * 2

    monkeypatch.setattr(observable_module.native, "affine_observable", None)
    with pytest.raises(TypeError, match="Q7400"):
        _ = -observable


def test_observable_construction_fails_closed_without_native_algebra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hamiltonians cannot select a shadow Python canonicalizer."""

    observable_module = importlib.import_module("qamomile.observable")
    monkeypatch.setattr(observable_module.native, "normalize_observable", None)

    with pytest.raises(TypeError, match="Q7400"):
        qm.Hamiltonian.zero()


def test_observable_construction_rejects_nonfinite_coefficients() -> None:
    """Canonical observable values contain only finite binary64 coordinates."""

    with pytest.raises(ValueError, match="finite"):
        qm.Hamiltonian.identity(float("inf"))


def test_pairwise_commutation_is_native_lazy_and_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Quadratic commutation analysis runs only when requested and only once."""

    observable = qm.X(0) + qm.Z(0)
    observable_module = importlib.import_module("qamomile.observable")
    native_inspect = observable_module.native.observable_is_pairwise_commuting
    calls = 0
    payload: bytes | None = None

    def record_native_inspection(value: bytes) -> bool:
        """Record one lazy compiler-owned commutation query."""

        nonlocal calls, payload
        calls += 1
        payload = value
        return bool(native_inspect(value))

    monkeypatch.setattr(
        observable_module.native,
        "observable_is_pairwise_commuting",
        record_native_inspection,
    )

    assert observable.is_pairwise_commuting is False
    assert observable.is_pairwise_commuting is False
    assert calls == 1
    assert payload is observable._native_literal


def test_pauli_word_commutation_crosses_the_native_algebra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Individual word commutation uses the shared Rust Pauli semantics."""

    observable_module = importlib.import_module("qamomile.observable")
    native_commute = observable_module.native.pauli_words_commute
    calls = 0

    def record_native_commutation(
        left: tuple[tuple[int, str], ...],
        right: tuple[tuple[int, str], ...],
    ) -> bool:
        """Record one compiler-owned Pauli-word commutation query."""

        nonlocal calls
        calls += 1
        return bool(native_commute(left, right))

    monkeypatch.setattr(
        observable_module.native,
        "pauli_words_commute",
        record_native_commutation,
    )

    assert qm.X(0).terms[0][0].commutes_with(qm.Z(0).terms[0][0]) is False
    assert calls == 1


def test_observable_scaling_normalizes_signed_zero_before_digesting() -> None:
    """Equivalent negative Pauli data has one Python/Rust package digest."""

    hamiltonian = -0.45 * qm.Z(0) * qm.Z(1)
    portable = hamiltonian.to_portable()
    raw_terms = portable["terms"]
    assert isinstance(raw_terms, tuple)
    constant = complex(portable["constant"])
    coefficient = complex(raw_terms[0][0])
    program = qm.build_source(PAULI_EVOLUTION_SOURCE).specialize(
        "pauli_evolution",
        **{"hamiltonian": hamiltonian},
    )

    assert math.copysign(1.0, constant.real) == 1.0
    assert math.copysign(1.0, constant.imag) == 1.0
    assert math.copysign(1.0, coefficient.imag) == 1.0
    assert "quantum.pauli_evolution" in program.inspect()


def test_observable_remap_is_native_and_preserves_explicit_width_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rust relabeling cannot silently shrink an explicitly sized observable."""

    observable = qm.Hamiltonian.identity(num_qubits=4) + qm.Z(3)
    observable_module = importlib.import_module("qamomile.observable")
    native_remap = observable_module.native.remap_observable
    calls: list[tuple[bytes, tuple[tuple[int, int], ...]]] = []

    def record_native_remap(
        payload: bytes,
        indices: tuple[tuple[int, int], ...],
    ) -> object:
        """Record and delegate one compiler-owned index substitution."""

        calls.append((payload, indices))
        return native_remap(payload, indices)

    monkeypatch.setattr(
        observable_module.native,
        "remap_observable",
        record_native_remap,
    )

    assert observable.remap({3: 0}).num_qubits == 4
    assert observable.remap({3: 6}).num_qubits == 7
    assert calls[0][0] is observable._native_literal
    assert calls[0][1] == ((3, 0),)

    monkeypatch.setattr(observable_module.native, "remap_observable", None)
    with pytest.raises(TypeError, match="Q7400"):
        observable.remap({3: 0})


@pytest.mark.parametrize(
    ("indices", "error", "message"),
    (
        ({True: 0}, TypeError, "integers"),
        ({0: False}, TypeError, "integers"),
        ({-1: 0}, ValueError, "non-negative"),
        ({4: 0}, ValueError, "Q7415.*outside"),
        ({1: 0}, ValueError, "Q7415.*same index"),
    ),
)
def test_observable_remap_rejects_invalid_index_substitutions(
    indices: object,
    error: type[Exception],
    message: str,
) -> None:
    """Malformed and axis-collapsing remaps fail before a result escapes."""

    observable = qm.X(0) * qm.Z(1)

    with pytest.raises(error, match=message):
        observable.remap(indices)  # type: ignore[arg-type]


def test_reference_executes_abstract_pauli_evolution() -> None:
    """The native Reference runtime evaluates the semantic operation directly."""

    from qamomile._native import compiler as native
    from qamomile.circuit import lower_to_circuit
    from qamomile.compilation import RealizationCapability

    program = _pauli_program()
    result = qm.run(program, on=qm.Reference(), shots=4, seed=2).result
    fallback = lower_to_circuit(program)
    optimized = lower_to_circuit(
        program,
        realization_capabilities=frozenset({RealizationCapability.RZZ}),
    )
    fallback_opcodes = tuple(
        instruction.opcode for instruction in fallback.body.instructions
    )
    optimized_opcodes = tuple(
        instruction.opcode for instruction in optimized.body.instructions
    )

    assert "quantum.pauli_evolution" in program.inspect()
    assert fallback_opcodes.count("cx") == 2
    assert fallback_opcodes.count("rz") == 1
    assert optimized_opcodes.count("rzz") == 1
    assert "rz" not in optimized_opcodes
    assert "cx" not in optimized_opcodes
    assert fallback.operation_count == optimized.operation_count + 2
    with pytest.raises(ValueError, match="Q8041"):
        native.inspect_static_circuit(program.to_bytes(), ["future"])
    with pytest.raises(ValueError, match="Q8042"):
        native.inspect_static_circuit(program.to_bytes(), ["rzz", "rzz"])
    assert result.counts == {(1, 0): 4}
    assert (
        qm.Program.from_bytes(program.to_bytes()).semantic_digest
        == program.semantic_digest
    )


def test_expectation_accepts_a_functional_state_program(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one-use helper delegates to the reusable estimator and total budget."""
    module = importlib.import_module("qamomile.measurement")
    native_prepare = module.native.prepare_pauli_measurement
    calls = []

    def prepare(payload, factors):
        calls.append(factors)
        return native_prepare(payload, factors)

    monkeypatch.setattr(module.native, "prepare_pauli_measurement", prepare)
    state = qm.build_source("""
import qamomile as qm
@qm.qkernel
def zero_state() -> qm.Q[qm.Bits[2]]:
    return qm.qubits(2)
""")["zero_state"]
    result = qm.expectation(
        state, qm.Z(0) + 2.0 * qm.Z(1) + 0.25, on=qm.Reference(), shots=4, seed=2
    )
    assert result.value == 3.25
    assert result.standard_error == 0.0
    assert result.shots == 4
    assert calls == [((0, "Z"),), ((1, "Z"),)]
    assert len({group.seed for group in result.groups}) == 2


def test_qwc_expectation_shares_one_total_shot_budget_and_exposes_groups() -> None:
    """One inspected QWC group replaces three independent backend launches."""

    state = qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def zero_state() -> qm.Q[qm.Bits[2]]:
    return qm.qubits(2)
""",
        module_name="qwc_expectation",
    )["zero_state"]
    observable = qm.Z(0) + 2.0 * qm.Z(1) + 3.0 * qm.Z(0) * qm.Z(1)
    result = qm.expectation(
        state,
        observable,
        on=qm.Reference(),
        shots=4,
        seed=2,
        grouping="qwc",
    )
    independent = qm.expectation(
        state,
        observable,
        on=qm.Reference(),
        shots=4,
        seed=2,
    )

    assert result.value == 6.0
    assert independent.value == result.value
    assert tuple(term.mean for term in independent.terms) == tuple(
        term.mean for term in result.terms
    )
    assert independent.shots == 4
    assert result.standard_error == 0.0
    assert result.shots == 4
    assert len(result.terms) == 3
    assert len(result.groups) == 1
    assert result.groups[0].basis == qm.PauliWord(
        (qm.PauliFactor(0, qm.Pauli.Z), qm.PauliFactor(1, qm.Pauli.Z))
    )
    assert result.groups[0].term_indices == (0, 1, 2)
    assert result.groups[0].shots == 4
    assert result.groups[0].variance == 0.0
    assert result.groups[0].standard_error == 0.0


def test_commuting_expectation_diagonalizes_non_qwc_terms_once() -> None:
    """One Clifford measurement jointly estimates Bell XX and ZZ stabilizers."""

    state = qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def bell_state() -> qm.Q[qm.Bits[2]]:
    state = qm.h(qm.qubits(2), target=0)
    return qm.cx(state, control=0, target=1)
""",
        module_name="commuting_expectation",
    )["bell_state"]
    observable = qm.X(0) * qm.X(1) - qm.Y(0) * qm.Y(1) + qm.Z(0) * qm.Z(1)
    result = qm.expectation(
        state,
        observable,
        on=qm.Reference(),
        shots=4,
        seed=2,
        grouping="commuting",
    )
    qwc = qm.expectation(
        state,
        observable,
        on=qm.Reference(),
        shots=4,
        seed=2,
        grouping="qwc",
    )

    assert result.value == qwc.value == 3.0
    assert result.standard_error == 0.0
    assert qwc.standard_error is None
    assert result.shots == qwc.shots == 4
    assert len(result.groups) == 1
    assert len(qwc.groups) == 3
    group = result.groups[0]
    assert group.basis is None
    assert group.term_indices == (0, 1, 2)
    assert tuple(readout.term_index for readout in group.readouts) == (0, 1, 2)
    assert all(
        all(factor.pauli is qm.Pauli.Z for factor in readout.word.factors)
        for readout in group.readouts
    )
    assert group.diagonalizing_gates > 0
    assert group.strategy == "commuting"
    assert any(readout.negative for readout in group.readouts)


def test_commuting_expectation_fails_closed_without_native_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """General grouping rejects missing and malformed Rust plan evidence."""

    expectation_module = importlib.import_module("qamomile._measurement_plan")
    state = qm.build_source(STATE_PREPARATION_SOURCE)["plus_state"]

    monkeypatch.setattr(
        expectation_module.native,
        "plan_commuting_pauli_expectation",
        None,
    )
    with pytest.raises(qm.CapabilityError, match="Q7300"):
        qm.expectation(
            state,
            qm.X(0),
            on=qm.Reference(),
            shots=1,
            grouping="commuting",
        )

    monkeypatch.setattr(
        expectation_module.native,
        "plan_commuting_pauli_expectation",
        lambda _observable, _shots, _seed: [([0], [(0, [], False)], 0, "1", 0)],
    )
    with pytest.raises(qm.ExecutionError, match="Q7310"):
        qm.expectation(
            state,
            qm.X(0),
            on=qm.Reference(),
            shots=1,
            grouping="commuting",
        )


@pytest.mark.parametrize("grouping", ["qwc", "commuting"])
def test_grouped_identity_expectation_uses_no_device_shots(
    monkeypatch: pytest.MonkeyPatch,
    grouping: Literal["qwc", "commuting"],
) -> None:
    """A constant observable is exact and does not schedule a measurement."""

    state = qm.build_source(STATE_PREPARATION_SOURCE)["plus_state"]

    def reject_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("identity expectation must not launch an endpoint")

    monkeypatch.setattr(
        importlib.import_module("qamomile.measurement"), "run", reject_run
    )
    result = qm.expectation(
        state,
        qm.Hamiltonian.identity(0.25, num_qubits=1),
        on=qm.Reference(),
        shots=4,
        grouping=grouping,
    )

    assert result.value == 0.25
    assert result.standard_error == 0.0
    assert result.shots == 0
    assert result.terms == ()
    assert result.groups == ()


def test_qwc_expectation_fails_closed_on_strategy_plan_and_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown strategies, missing native plans, and undersized budgets fail."""

    expectation_module = importlib.import_module("qamomile._measurement_plan")
    native_plan = expectation_module.native.plan_qwc_pauli_expectation
    native_estimate = expectation_module.native.statistics_mean
    native_prepare = expectation_module.native.prepare_qwc_pauli_measurement
    state = qm.build_source(STATE_PREPARATION_SOURCE)["plus_state"]
    with pytest.raises(TypeError, match="grouping"):
        qm.expectation(
            state,
            qm.X(0),
            on=qm.Reference(),
            shots=1,
            grouping="unknown",  # type: ignore[arg-type]
        )
    with pytest.raises(qm.ExecutionError, match="Q7318"):
        qm.expectation(
            state,
            qm.X(0) + qm.Z(0),
            on=qm.Reference(),
            shots=1,
            grouping="qwc",
        )

    monkeypatch.setattr(
        expectation_module.native,
        "plan_qwc_pauli_expectation",
        lambda _observable, _shots, _seed: [([], [0], "1", 0)],
    )
    with pytest.raises(qm.ExecutionError, match="Q7310"):
        qm.expectation(
            state,
            qm.X(0),
            on=qm.Reference(),
            shots=1,
            grouping="qwc",
        )

    monkeypatch.setattr(
        expectation_module.native, "plan_qwc_pauli_expectation", native_plan
    )
    monkeypatch.setattr(expectation_module.native, "statistics_mean", None)
    with pytest.raises(qm.CapabilityError, match="Q7300|Q7404"):
        qm.expectation(
            state,
            qm.X(0),
            on=qm.Reference(),
            shots=1,
            grouping="qwc",
        )
    monkeypatch.setattr(
        expectation_module.native,
        "statistics_mean",
        native_estimate,
    )

    monkeypatch.setattr(
        expectation_module.native, "prepare_qwc_pauli_measurement", None
    )
    with pytest.raises(qm.CapabilityError, match="Q7300|Q7404"):
        qm.expectation(
            state,
            qm.X(0),
            on=qm.Reference(),
            shots=1,
            grouping="qwc",
        )
    monkeypatch.setattr(
        expectation_module.native, "prepare_qwc_pauli_measurement", native_prepare
    )

    monkeypatch.setattr(expectation_module.native, "plan_qwc_pauli_expectation", None)
    with pytest.raises(qm.CapabilityError, match="Q7300|Q7404"):
        qm.expectation(
            state,
            qm.X(0),
            on=qm.Reference(),
            shots=1,
            grouping="qwc",
        )


def test_expectation_seed_planning_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expectation never reuses a Python seed fallback or malformed native row."""

    expectation_module = importlib.import_module("qamomile._measurement_plan")
    state = qm.build_source(STATE_PREPARATION_SOURCE)["plus_state"]

    with pytest.raises(qm.ExecutionError, match="Q8138"):
        qm.expectation(state, qm.X(0), on=qm.Reference(), shots=1, seed=-1)

    monkeypatch.setattr(
        expectation_module.native,
        "plan_independent_pauli_expectation",
        None,
    )
    with pytest.raises(qm.CapabilityError, match="Q7300"):
        qm.expectation(state, qm.X(0), on=qm.Reference(), shots=1)

    monkeypatch.setattr(
        expectation_module.native,
        "plan_independent_pauli_expectation",
        lambda _observable, _shots, _seed: [],
    )
    with pytest.raises(qm.ExecutionError, match="Q7310"):
        qm.expectation(state, qm.X(0), on=qm.Reference(), shots=1)


def test_expectation_seed_plan_is_portable_and_preserves_unseeded_execution() -> None:
    """Rust derives distinct 32-bit streams or the endpoint-default sentinel."""

    expectation_module = importlib.import_module("qamomile._measurement_plan")

    observable = sum((qm.Z(i) for i in range(4)), qm.zero_observable(4))
    seeded = expectation_module._native_qwc_expectation_plan(
        observable, shots=4, seed=2, independent=True
    )
    seeds = tuple(group.seed for group in seeded)
    assert len(set(seeds)) == 4
    assert all(0 <= seed <= (1 << 32) - 1 for seed in seeds)
    unseeded = expectation_module._native_qwc_expectation_plan(
        observable, shots=4, seed=0, independent=True
    )
    assert tuple(group.seed for group in unseeded) == (0, 0, 0, 0)


def test_expectation_fails_closed_without_native_measurement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public API cannot fall back to a shadow Python IR transform."""

    expectation_module = importlib.import_module("qamomile._measurement_plan")
    monkeypatch.setattr(
        expectation_module.native,
        "prepare_pauli_measurement",
        None,
    )
    state = qm.build_source(STATE_PREPARATION_SOURCE)["plus_state"]

    with pytest.raises(qm.CapabilityError, match="Q7300"):
        qm.expectation(state, qm.X(0), on=qm.Reference(), shots=1)


def test_expectation_fails_closed_without_native_aggregation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even identity-only estimates require Rust-owned statistics."""

    expectation_module = importlib.import_module("qamomile._measurement_plan")
    monkeypatch.setattr(
        expectation_module.native,
        "statistics_combine_independent",
        None,
    )
    state = qm.build_source(STATE_PREPARATION_SOURCE)["plus_state"]

    with pytest.raises(qm.CapabilityError, match="Q7404"):
        qm.expectation(
            state,
            qm.Hamiltonian.identity(0.25, num_qubits=1),
            on=qm.Reference(),
            shots=1,
        )

    monkeypatch.setattr(
        expectation_module.native,
        "statistics_combine_independent",
        lambda _terms, _constant: (0.25, 0.0, [(1.0, 0.0)]),
    )
    with pytest.raises(qm.CapabilityError, match="Q7404"):
        qm.expectation(
            state,
            qm.Hamiltonian.identity(0.25, num_qubits=1),
            on=qm.Reference(),
            shots=1,
        )


def test_noncommuting_sum_requires_an_explicit_approximation_algorithm() -> None:
    """The exact core never silently inserts a first-order product formula."""

    hamiltonian = qm.X(0) + qm.Z(0)
    program = qm.build_source(PAULI_EVOLUTION_SOURCE).specialize(
        "pauli_evolution",
        **{"hamiltonian": hamiltonian},
    )

    with pytest.raises(qm.ExecutionError, match="pairwise-commuting"):
        qm.run(program, on=qm.Reference(), shots=1)


def test_pauli_evolution_participates_in_automatic_uncomputation() -> None:
    """A dead evolution uses the same modular adjoint cleanup proof."""

    source = """
import qamomile as qm

@qm.qkernel
def cleanup(hamiltonian: qm.Const[qm.Observable]) -> qm.Unit:
    state = qm.qubits(1)
    state = qm.pauli_evolve(state, hamiltonian, 0.125)
    return None
"""
    program = qm.build_source(source).specialize(
        "cleanup",
        **{"hamiltonian": qm.Z(0)},
    )

    assert "original='quantum.pauli_evolution'" in program.inspect()
    assert qm.evaluate(program) is None
