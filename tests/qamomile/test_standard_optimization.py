"""Exact standard QCore optimization and cross-backend execution witnesses."""

from __future__ import annotations

import pytest

import qamomile as qm
from qamomile._native import compiler as native
from qamomile.circuit import lower_to_circuit
from qamomile.persistence import export_payload

from ._programs import standard_optimization_sample, standard_optimization_state


@qm.qkernel
def _unitary(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    """Return one reversible body used by the call-cancellation witness."""

    return qm.x(state)


@qm.qkernel
def _call_adjoint_sample() -> qm.Bits[1]:
    """Apply one callable and its exact adjoint."""

    state = _unitary(qm.qubits(1))
    state = qm.adjoint(_unitary, state)
    return qm.measure(state)


@qm.qkernel
def _overlapping_sample() -> qm.Bits[2]:
    """Keep inverse gates separated by an overlapping transformation."""

    state = qm.x(qm.qubits(2), target=0)
    state = qm.h(state, target=0)
    state = qm.x(state, target=0)
    return qm.measure(state)


@qm.qkernel
def _vacuous_owners() -> None:
    """Create immediately released allocation and preparation roots."""

    _unused_allocation = qm.qubits(1)
    _unused_preparation = qm.prepare((1, 0))


@qm.qkernel
def _structural_roundtrip_sample() -> qm.Bits[2]:
    """Split and immediately reconstruct one owned register."""

    left, right = qm.split_at(qm.qubits(2), 1)
    return qm.measure(qm.concat(left, right))


def exercise_standard_optimization_backend(*, endpoint: object) -> None:
    """Compile and execute the shared exact-optimization witness.

    Args:
        endpoint: Backend endpoint exposing its matching artifact target.

    Raises:
        AssertionError: If the target omits the optimizer stage or changes the
            optimized program's result.
    """

    artifact = qm.compile(standard_optimization_sample, to=endpoint.target)
    assert artifact.report.stages.count("rust-qcore-standard-optimization") == 1
    result = qm.execute(artifact, on=endpoint, shots=4, seed=0)
    assert result.counts == {(0, 0): 4}
    assert (
        qm.expectation(
            standard_optimization_state,
            qm.Z(0),
            on=endpoint,
            shots=4,
            seed=0,
        ).value
        == 1.0
    )


def test_standard_optimizer_removes_exact_identities_and_is_deterministic() -> None:
    """The Rust pass reseals a stable smaller image and preserves Program input."""

    program = standard_optimization_sample.specialize()
    payload = program.to_bytes()
    first = native.optimize_standard_program(payload)
    second = native.optimize_standard_program(payload)
    optimized = qm.Program.from_bytes(first)

    assert first == second
    assert first != payload
    assert optimized.to_bytes() == first
    assert program.to_bytes() == payload
    assert [
        instruction.opcode
        for instruction in lower_to_circuit(program).body.instructions
    ] == ["measure", "measure"]


def test_standard_optimizer_cancels_call_with_adjoint() -> None:
    """An adjacent call and exact adjoint disappear before circuit lowering."""

    circuit = lower_to_circuit(_call_adjoint_sample.specialize())
    assert [instruction.opcode for instruction in circuit.body.instructions] == [
        "measure"
    ]


def test_standard_optimizer_cancels_beyond_the_old_lookback_distance() -> None:
    """A verified frontier cancels across more than 256 independent gates."""

    separation = 300
    middle = "".join(
        f"    state = qm.h(state, target={target})\n"
        for target in range(1, separation + 1)
    )
    source = (
        "import qamomile as qm\n\n"
        "@qm.qkernel\n"
        f"def sample() -> qm.Bits[{separation + 1}]:\n"
        f"    state = qm.qubits({separation + 1})\n"
        "    state = qm.x(state, target=0)\n"
        f"{middle}"
        "    state = qm.x(state, target=0)\n"
        "    return qm.measure(state)\n"
    )
    program = qm.build_source(
        source,
        module_name="long_frontier",
        display_name="long_frontier.py",
    ).specialize("sample")
    opcodes = [
        instruction.opcode
        for instruction in lower_to_circuit(program).body.instructions
    ]

    assert opcodes.count("x") == 0
    assert opcodes.count("h") == separation


def test_standard_optimizer_keeps_overlapping_dependencies() -> None:
    """Operations touching the same logical qubit remain ordered and visible."""

    circuit = lower_to_circuit(_overlapping_sample.specialize())
    assert [instruction.opcode for instruction in circuit.body.instructions] == [
        "x",
        "h",
        "x",
        "measure",
        "measure",
    ]


def test_standard_optimizer_removes_vacuous_owned_roots() -> None:
    """Immediate alloc/release and prepare/unprepare pairs leave no circuit."""

    circuit = lower_to_circuit(_vacuous_owners.specialize())
    assert circuit.operation_count == 0
    assert circuit.body.instructions == ()


def test_standard_optimizer_removes_structural_roundtrip() -> None:
    """Tuple, unpack, and reshape roundtrips collapse to the original owner."""

    program = _structural_roundtrip_sample.specialize()
    optimized = qm.Program.from_bytes(
        native.optimize_standard_program(program.to_bytes())
    )

    assert "qcore.reshape" in program.inspect()
    assert "qcore.unpack" in program.inspect()
    assert "qcore.tuple" in program.inspect()
    assert "qcore.reshape" not in optimized.inspect()
    assert "qcore.unpack" not in optimized.inspect()
    assert "qcore.tuple" not in optimized.inspect()


def test_reference_compilation_retains_the_prepared_qcore_once() -> None:
    """Reference stores the verified optimized image used by native execution."""

    program = standard_optimization_sample.specialize()
    artifact = qm.compile(program, to=qm.Reference())
    optimized = native.optimize_standard_program(program.to_bytes())

    assert artifact.report.stages.count("rust-qcore-standard-optimization") == 1
    assert export_payload(artifact) == optimized
    assert export_payload(artifact) != program.to_bytes()
    assert native.optimize_standard_program(export_payload(artifact)) == export_payload(
        artifact
    )
    assert qm.execute(artifact, on=qm.Reference(), shots=4, seed=0).counts == {
        (0, 0): 4
    }


def test_reference_compilation_requires_the_native_optimizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reference fails closed when the Rust optimization boundary is absent."""

    monkeypatch.setattr(native, "optimize_standard_program", None)
    with pytest.raises(qm.ArtifactError, match="Q7049"):
        qm.compile(standard_optimization_sample, to=qm.Reference())


def test_standard_optimizer_rejects_non_qcore_bytes() -> None:
    """The optimizer never treats malformed bytes as a verified program."""

    with pytest.raises(ValueError, match="Q4008"):
        native.optimize_standard_program(b"not-qcore")


@pytest.mark.parametrize(
    ("factor", "modulus"),
    [(1, 5), (-4, 5), (0, 1), (-(2**63), 1)],
)
def test_modular_identity_optimization_preserves_verified_program(
    factor: int, modulus: int
) -> None:
    """Identity removal reseals deterministically without mutating source or ABI."""

    source = f"""import qamomile as qm

@qm.qkernel
def sample() -> qm.Bits[5]:
    target = qm.as_uint(qm.prepare((1, 0, 1)))
    exponent = qm.as_uint(qm.prepare((0, 1)))
    target = qm.mod_mul(target, factor={factor}, modulus={modulus})
    target, exponent = qm.mod_pow(target, exponent, base={factor}, modulus={modulus})
    return qm.measure(qm.concat(qm.as_bits(target), qm.as_bits(exponent)))
"""
    program = qm.build_source(
        source,
        module_name="modular_identity",
        display_name="modular_identity.py",
    ).specialize("sample")
    original = program.to_bytes()
    optimized = native.optimize_standard_program(original)
    verified = qm.Program.from_bytes(optimized)

    assert optimized == native.optimize_standard_program(original)
    assert verified.to_bytes() == optimized
    assert program.to_bytes() == original
    assert "quantum.mod_mul" in program.inspect()
    assert "quantum.mod_pow" in program.inspect()
    assert "quantum.mod_mul" not in verified.inspect()
    assert "quantum.mod_pow" not in verified.inspect()
    before = lower_to_circuit(program)
    after = lower_to_circuit(verified)
    assert before.result_abi == after.result_abi
    assert before.width == after.width == 5
    assert before.body == after.body
