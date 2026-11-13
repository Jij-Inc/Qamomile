"""Semantic persistence, exact alternatives, and late target selection."""

from __future__ import annotations

import hashlib
import importlib

import numpy as np
import pytest

import qamomile as qm
from qamomile._native import compiler as native
from qamomile.circuit import (
    lower_to_circuit,
    lower_to_circuit_at,
    lower_to_circuit_family,
)
from qamomile.compilation import CapabilitySnapshot, RealizationCapability
from qamomile.lowering import RealizationOptions, list_realizations
from qamomile.persistence import deserialize, serialize

from ._programs import (
    first_class_mod_pow_sample,
    mod_pow_identity_sample,
    operator_arithmetic_sample,
    realization_add_coherent_sample,
    realization_add_coherent_state,
    realization_add_sample,
    realization_add_state,
    realization_subtract_sample,
    stdlib_arithmetic_ux_sample,
)


def test_discovery_matches_validation_and_is_immutable() -> None:
    """Each definition validates under exactly its advertised requirements."""
    from dataclasses import FrozenInstanceError

    choices = list_realizations()
    assert len(choices) == 16
    assert len({choice.name for choice in choices}) == len(choices)
    for choice in choices:
        assert not choice.name.endswith(".v1")
        implementation = (
            qm.realize(choice.name, padding_bits=8)
            if choice.name == "mod_pow.coset_windowed"
            else choice.name
        )
        options = RealizationOptions({choice.family: implementation})
        options.require_capabilities(choice.required_capabilities)
        assert choice.family in {
            "quantum.add_into",
            "quantum.mod_mul",
            "quantum.mod_mul_add",
            "quantum.mod_multi_pow",
            "quantum.mod_pow",
            "quantum.pauli_evolution",
            "quantum.arithmetic",
        }
        assert choice.description
        with pytest.raises(FrozenInstanceError):
            choice.name = "changed"
        for missing in choice.required_capabilities:
            with pytest.raises(qm.CapabilityError, match="Q8045"):
                options.require_capabilities(
                    tuple(cap for cap in choice.required_capabilities if cap != missing)
                )
    with pytest.raises(ValueError, match="Q8043"):
        RealizationOptions({qm.add_into: "add.qft.v1"})


def test_parameterized_realization_is_native_validated_and_canonical() -> None:
    """Keep parameters out of names while retaining one canonical fingerprint."""

    configured = qm.realize("mod_pow.coset_windowed", padding_bits=40)
    options = RealizationOptions({qm.mod_pow: configured})
    assert options.overrides == ("mod_pow.coset_windowed[padding_bits=40]",)
    with pytest.raises(ValueError, match="Q8052"):
        qm.realize("mod_pow.coset_windowed", padding_bits=0)
    with pytest.raises(ValueError, match="Q8052"):
        qm.realize("mod_pow.coset_windowed", padding_bits=-1)
    with pytest.raises(ValueError, match="Q8052"):
        qm.realize("mod_pow.coset_windowed", padding_bits=65_536)
    with pytest.raises(ValueError, match="Q8052"):
        qm.realize("mod_pow.coset_windowed", precision=40)
    with pytest.raises(ValueError, match="Q8052"):
        qm.realize("add.qft", padding_bits=8)
    with pytest.raises(TypeError, match="integer"):
        qm.realize("mod_pow.coset_windowed", padding_bits=True)


@pytest.mark.parametrize(
    "rows",
    [
        None,
        [("x", "family", "summary", ())],
        [("x", "family", "summary", [1])],
        [("x", "family", "summary", [])] * 2,
    ],
)
def test_discovery_rejects_malformed_projection(monkeypatch, rows) -> None:
    """A stale native ABI never becomes an alternate Python catalog."""
    monkeypatch.setattr(native, "list_realizations", lambda: rows)
    with pytest.raises(TypeError, match="ABI"):
        list_realizations()


def test_discovery_requires_native_owner(monkeypatch) -> None:
    """Discovery fails closed without its compiler definition."""
    monkeypatch.setattr(native, "list_realizations", None)
    with pytest.raises(qm.CapabilityError, match="Q8049"):
        list_realizations()


def exercise_realization_backend(
    endpoint: object, *, supports_mod_pow: bool = True
) -> None:
    """Execute sampling, inverse, interference and expectation on one target.

    Args:
        endpoint: Backend endpoint under test.
        supports_mod_pow: Whether its target admits exact modular
            exponentiation realization.
    """
    program = realization_add_sample.specialize()
    artifact = qm.compile(program, to=endpoint.target)
    assert isinstance(artifact, qm.CompiledProgram)
    assert any(stage.startswith("realization:add.") for stage in artifact.report.stages)
    restored = deserialize(serialize(artifact))
    assert qm.execute(restored, on=endpoint, shots=4, seed=0).counts == {
        (1, 0, 0, 0): 4
    }
    assert qm.run(
        realization_subtract_sample, on=endpoint, shots=4, seed=0
    ).result.counts == {(1, 0, 1, 1): 4}
    assert qm.run(
        operator_arithmetic_sample, on=endpoint, shots=4, seed=0
    ).result.counts == {(1, 1, 0, 1, 1, 1, 0, 1): 4}
    assert qm.run(
        stdlib_arithmetic_ux_sample, on=endpoint, shots=4, seed=0
    ).result.counts == {(1, 1, 1, 0, 1, 1, 0, 0, 1, 1): 4}
    if supports_mod_pow:
        assert qm.run(
            mod_pow_identity_sample, on=endpoint, shots=4, seed=0
        ).result.counts == {(1, 1): 4}
    assert qm.run(
        realization_add_coherent_sample, on=endpoint, shots=4, seed=0
    ).result.counts == {(0, 0, 0, 0, 0): 4}
    assert (
        qm.expectation(
            realization_add_state, qm.Z(0), on=endpoint, shots=4, seed=0
        ).value
        == -1.0
    )
    assert (
        qm.expectation(
            realization_add_coherent_state, qm.Z(0), on=endpoint, shots=4, seed=0
        ).value
        == 1.0
    )


@pytest.mark.parametrize(
    "choice", ["add.increments", "add.qft", "add.ripple", "add.native"]
)
@pytest.mark.parametrize("width", [1, 2, 3])
def test_addition_matrix_matches_full_complex_unitary(choice: str, width: int) -> None:
    """All basis amplitudes and phases agree with an independent permutation."""
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Operator

    from qamomile.qiskit import Qiskit, to_circuit

    program = qm.build_source(f"""
import qamomile as qm
@qm.qkernel
def example() -> qm.Q[qm.Bits[{2 * width}]]:
    return qm.add_into(qm.qubits({2 * width}))
""").specialize("example")
    target = Qiskit(realizations=RealizationOptions({qm.add_into: choice}))
    circuit = to_circuit(program, target=target)
    actual = Operator(circuit).data
    expected = np.zeros_like(actual)
    mask = (1 << width) - 1
    for index in range(1 << (2 * width)):
        a, b = index & mask, index >> width
        expected[a | (((a + b) & mask) << width), index] = 1
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=0)


def test_realization_does_not_change_semantic_identity_or_logical_count() -> None:
    """Choice is late and transient while QCore retains one semantic addition."""
    program = realization_add_sample.specialize()
    payload = program.to_bytes()
    restored = qm.Program.from_bytes(payload)
    assert restored.to_bytes() == payload
    assert restored.inspect().count("quantum.add_into") == 1
    counts = []
    for choice in ("add.increments", "add.qft"):
        circuit = lower_to_circuit(
            restored, realizations=RealizationOptions({qm.add_into: choice})
        )
        assert circuit.realizations == (choice,)
        assert circuit.semantic_id == program.semantic_digest
        counts.append(circuit.operation_count)
    assert counts[0] != counts[1]
    assert restored.to_bytes() == payload
    assert qm.resources.logical(restored).metric(
        qm.resources.logical_operation_count("quantum.add_into")
    ).numeric_enclosure() == (1, 1)


def test_invalid_realization_options_are_rejected() -> None:
    """Reject unknown implementations and mismatched semantic families."""

    with pytest.raises(ValueError, match="Q8043"):
        RealizationOptions({qm.add_into: "add.unknown.v1"})
    with pytest.raises(ValueError, match="implements quantum.add_into"):
        RealizationOptions({qm.mod_mul: "add.qft"})


def test_explicit_unavailable_choice_never_silently_falls_back() -> None:
    program = realization_add_sample.specialize()
    with pytest.raises(qm.CapabilityError, match="Q8045"):
        lower_to_circuit(
            program, realizations=RealizationOptions({qm.add_into: "add.native"})
        )
    with pytest.raises(ValueError, match="Q8041"):
        native.inspect_static_circuit(program.to_bytes(), ["unknown"])
    with pytest.raises(ValueError, match="Q8042"):
        native.inspect_static_circuit(program.to_bytes(), ["rzz", "rzz"])


def test_realization_options_require_native_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("qamomile.lowering")
    monkeypatch.setattr(module.native, "validate_lowering_options", None)
    with pytest.raises(qm.CapabilityError, match="Q8049"):
        RealizationOptions({qm.add_into: "add.qft"})


def test_arithmetic_preflights_width_work_and_native_bypass() -> None:
    for width in (0, 1, 3):
        with pytest.raises(ValueError, match="Q8047"):
            native.plan_addition(width)
    for choice in ("add.increments", "add.qft"):
        with pytest.raises(ValueError, match="Q8048"):
            native.plan_addition(1 << 40, [], [choice])
    identity, steps = native.plan_addition(1000, ["add_into"])
    assert identity == "add.native" and len(steps) == 1


def test_mod_pow_prefers_native_capability_unless_portable_is_selected() -> None:
    """Target support preserves semantics while an explicit recipe takes precedence."""

    native_plan = native.plan_mod_pow(3, 2, 2, 5, False, 0, ["mod_pow"])
    assert native_plan[:3] == ("native", "mod_pow.native", 0)
    assert native_plan[3] == []

    portable = native.plan_mod_pow(
        3,
        2,
        2,
        5,
        False,
        0,
        ["mod_pow"],
        ["mod_pow.square_and_multiply"],
    )
    assert portable[0] == "gates"
    assert portable[1] == "mod_pow.square_and_multiply"
    assert portable[2] == 1  # One clean wire decomposes three-control X.
    assert portable[3]
    assert portable[4] == ["arithmetic.portable"]


def test_static_circuit_preserves_native_mod_pow_and_honors_portable_override() -> None:
    """Circuit projection retains a supported composite without defeating overrides."""

    program = first_class_mod_pow_sample.specialize()
    native_circuit = lower_to_circuit(
        program,
        realization_capabilities=frozenset({RealizationCapability.MOD_POW}),
    )
    assert native_circuit.realizations == ("mod_pow.native",)
    [native_instruction] = [
        instruction
        for instruction in native_circuit.body.instructions
        if instruction.opcode == "mod_pow"
    ]
    assert native_instruction.parameters == (3, 2, 2, 5)

    portable_circuit = lower_to_circuit(
        program,
        realization_capabilities=frozenset({RealizationCapability.MOD_POW}),
        realizations=RealizationOptions({qm.mod_pow: "mod_pow.square_and_multiply"}),
    )
    assert portable_circuit.realizations == (
        "mod_pow.square_and_multiply",
        "arithmetic.portable",
    )
    assert all(
        instruction.opcode != "mod_pow"
        for instruction in portable_circuit.body.instructions
    )


def test_malformed_addition_shape_fails_before_execution() -> None:
    with pytest.raises(qm.QamomileError, match="Q1035"):
        qm.build_source("""
import qamomile as qm
@qm.qkernel
def malformed() -> qm.Bits[3]:
    return qm.measure(qm.add_into(qm.qubits(3)))
""")


def test_unsupported_unused_choice_fails_at_target_negotiation() -> None:
    """A direct-HUGR route cannot silently ignore an unavailable option."""
    with pytest.raises(qm.CapabilityError, match="Q8045"):
        CapabilitySnapshot(
            "test",
            frozenset(),
            realizations=RealizationOptions({qm.add_into: "add.native"}),
        )


def test_unrealized_gate_costs_are_named_unknowns_not_free_work() -> None:
    """Unselected implementations cannot justify exact physical gate costs."""
    study = qm.resources.logical(realization_add_sample)
    for metric in (
        qm.resources.LOGICAL_T_GATES,
        qm.resources.LOGICAL_NON_CLIFFORD_GATES,
        qm.resources.LOGICAL_GATE_DEPTH,
    ):
        value = study.metric(metric)
        assert value.numeric_enclosure()[1] is None
        assert any(
            "quantum.add_into" in reason for reason in value.unknown_contributors
        )
    physical = qm.resources.surface_code(study)
    assert (
        physical.metric(qm.resources.PHYSICAL_DURATION).numeric_enclosure()[1] is None
    )


def test_dependent_shape_is_rechecked_after_specialization() -> None:
    """Abstract addition cannot hide an invalid specialized wire count."""
    module = qm.build_source("""
import qamomile as qm
@qm.qkernel
def add(width: qm.Const[qm.Int]) -> qm.Q[qm.Bits[width]]:
    return qm.add_into(qm.qubits(width))
""")
    assert "quantum.add_into" in module.specialize("add", width=4).inspect()
    with pytest.raises(qm.QamomileError, match="Q1035"):
        module.specialize("add", width=3)


def test_persistent_verification_rejects_forged_addition_width() -> None:
    """A valid seal cannot bypass Rust's semantic addition equation."""
    from qamomile.proto import qcore_pb2 as wire

    program = qm.build_source("""
import qamomile as qm
@qm.qkernel
def sample() -> qm.Q[qm.Bits[3]]:
    return qm.h(qm.qubits(3))
""").specialize("sample")
    package = wire.QCorePackageProto.FromString(program.to_bytes())
    gate = next(
        op for op in package.functions[0].body.operations if op.opcode == "quantum.h"
    )
    gate.opcode = "quantum.add_into"
    del gate.attributes[:]
    del gate.operands[1:]
    package.package_digest = b""
    package.package_digest = hashlib.sha256(
        package.SerializeToString(deterministic=True)
    ).digest()
    with pytest.raises(ValueError, match="Q1035"):
        native.inspect_program(package.SerializeToString(deterministic=True))


def test_family_and_launch_lowering_preserve_the_same_override() -> None:
    """Dispatch/materialization do not revert to a different recipe."""
    program = qm.build_source("""
import qamomile as qm
@qm.qkernel
def choose(enabled: qm.Param[qm.Bool]) -> qm.Bits[4]:
    state = qm.qubits(4)
    if enabled:
        state = qm.add_into(state)
    return qm.measure(state)
""").specialize("choose")
    options = RealizationOptions({qm.add_into: "add.qft"})
    family = lower_to_circuit_family(program, realizations=options)
    assert any(
        variant.circuit.realizations == options.overrides for variant in family.variants
    )
    circuit = lower_to_circuit_at(program, {"enabled": True}, realizations=options)
    assert circuit.realizations == options.overrides


@pytest.mark.parametrize("choice", ["pauli.native", "pauli.rzz", "pauli.parity"])
@pytest.mark.parametrize("inverse", [False, True])
def test_pauli_choices_preserve_controlled_global_phase(
    choice: str, inverse: bool
) -> None:
    """Compare controlled forward/inverse matrices without ignoring phase."""
    from qiskit.quantum_info import Operator

    from qamomile.qiskit import Qiskit, to_circuit

    call = (
        "qm.adjoint(evolution, state, hamiltonian)"
        if inverse
        else "evolution(state, hamiltonian)"
    )
    program = qm.build_source(f"""
import qamomile as qm
@qm.qkernel
def evolution(state: qm.Q[qm.Bits[2]], hamiltonian: qm.Const[qm.Observable]) -> qm.Q[qm.Bits[2]]:
    return qm.pauli_evolve(state, hamiltonian, 0.4)
@qm.qkernel
def transformed(state: qm.Q[qm.Bits[2]], hamiltonian: qm.Const[qm.Observable]) -> qm.Q[qm.Bits[2]]:
    return {call}
@qm.qkernel
def example(hamiltonian: qm.Const[qm.Observable]) -> qm.Q[qm.Bits[3]]:
    control, state = qm.split_at(qm.qubits(3), 1)
    control, state = qm.control(transformed, control, state, hamiltonian)
    return qm.concat(control, state)
""").specialize("example", hamiltonian=0.3 + 0.7 * qm.Z(0) * qm.Z(1))
    actual = Operator(
        to_circuit(
            program,
            target=Qiskit(realizations=RealizationOptions({qm.pauli_evolve: choice})),
        )
    ).data
    expected = np.eye(8, dtype=complex)
    sign = 1 if inverse else -1
    for index in range(1, 8, 2):
        zz = 1 if ((index >> 1) & 1) == ((index >> 2) & 1) else -1
        expected[index, index] = np.exp(sign * 1j * 0.4 * (0.3 + 0.7 * zz))
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=0)
