"""Classical execution entries and quantum component inspection remain distinct."""

import hashlib
from dataclasses import replace

import pytest

import qamomile as qm
from qamomile.compilation import ParameterABI, ParameterSlot, ResultABI
from qamomile.persistence import deserialize, serialize
from qamomile.proto import qcore_pb2 as wire
from qamomile.serialization import encode_type

SOURCE = """
import qamomile as qm
@qm.qkernel
def state(theta: qm.Param[qm.Real]) -> qm.Q[qm.Bits[1]]:
    return qm.ry(qm.qubits(1), theta)
@qm.qkernel
def nested() -> tuple[qm.Int, tuple[qm.Q[qm.Bits[1]], qm.Int]]:
    return 1, (qm.qubits(1), 2)
@qm.qkernel
def consume(q: qm.Q[qm.Bits[1]]) -> qm.Shot[qm.Bits[1]]:
    return qm.measure(q)
@qm.qkernel
def nested_input(pair: tuple[qm.Q[qm.Bits[1]], qm.Int]) -> qm.Shot[qm.Bits[1]]:
    q, n = pair
    return qm.measure(q)
@qm.qkernel
def sample(theta: qm.Param[qm.Real]) -> qm.Shot[qm.Bits[1]]:
    return consume(state(theta))
"""


@pytest.mark.parametrize("entry", ["state", "nested", "consume", "nested_input"])
def test_quantum_entry_rejected_before_target_lowering(entry, monkeypatch):
    """Recursive source types reject execution before SDK emission is reached."""
    from qamomile.qiskit import Qiskit

    def forbidden(*args, **kwargs):
        raise AssertionError("must reject before SDK lowering")

    monkeypatch.setattr(Qiskit, "lower_circuit", forbidden)
    program = qm.build_source(SOURCE).specialize(entry)
    assert qm.Program.from_bytes(program.to_bytes()) == program
    with pytest.raises(qm.QamomileError, match="Q6304"):
        qm.compile(program, to=Qiskit())


def test_quantum_helpers_execute_inside_classical_entry():
    """Quantum boundaries are legal within a measured classical execution."""
    artifact = qm.compile(qm.build_source(SOURCE)["sample"], to=qm.Reference())
    result = qm.execute(
        deserialize(serialize(artifact)),
        on=qm.Reference(),
        parameters={"theta": 0.0},
        shots=3,
    )
    assert result.counts == {(0,): 3}


def test_inspection_preserves_symbolic_and_concrete_parameters():
    """Circuit inspection adds no measurements or executable quantum ABI."""
    from qiskit.quantum_info import Statevector

    from qamomile.qiskit import to_circuit

    source = qm.build_source(SOURCE)["state"]
    symbolic = to_circuit(source)
    assert len(symbolic.parameters) == 1
    concrete = to_circuit(source, parameters={"theta": 0.0})
    assert not concrete.parameters and not concrete.num_clbits
    assert Statevector.from_instruction(concrete).probabilities_dict() == {"0": 1.0}
    concrete.x(0)
    assert len(to_circuit(source, parameters={"theta": 0.0})) + 1 == len(concrete)


@pytest.mark.parametrize("field", ["parameters", "result", "source"])
def test_live_artifacts_reject_quantum_contracts(field):
    """Constructing or replacing an artifact cannot bypass native type checks."""
    module = qm.build_source(SOURCE)
    artifact = qm.compile(module["sample"], to=qm.Reference())
    if field == "parameters":
        updates = {
            "parameter_abi": ParameterABI((ParameterSlot("q", qm.Q[qm.Bits[1]]),))
        }
    elif field == "result":
        updates = {"result_abi": ResultABI(qm.Q[qm.Bits[1]], (), 0)}
    else:
        updates = {
            "report": replace(
                artifact.report,
                resources=qm.resources.logical(module.specialize("state")),
            )
        }
    with pytest.raises(qm.ArtifactError):
        replace(artifact, **updates)


@pytest.mark.parametrize("field", ["parameters", "result"])
def test_forged_quantum_abi_rejected_before_payload_load(field, monkeypatch):
    """A valid checksum does not allow quantum values through an old archive."""
    import qamomile.persistence as persistence

    artifact = qm.compile(qm.build_source(SOURCE)["sample"], to=qm.Reference())
    encoded = wire.ArtifactProto.FromString(serialize(artifact))
    target = (
        encoded.parameter_abi.slots[0].type
        if field == "parameters"
        else encoded.result_abi.type
    )
    target.Clear()
    encode_type(qm.Q[qm.Bits[1]], target)
    encoded.artifact_digest = b""
    encoded.artifact_digest = hashlib.sha256(
        encoded.SerializeToString(deterministic=True)
    ).digest()

    def forbidden(*args):
        raise AssertionError("must reject before restoring SDK payload")

    monkeypatch.setattr(persistence, "_restore_payload", forbidden)
    with pytest.raises(qm.ArtifactError):
        deserialize(encoded.SerializeToString(deterministic=True))


def test_missing_native_boundary_validator_fails_closed(monkeypatch):
    import qamomile.compilation as compilation

    artifact = qm.compile(qm.build_source(SOURCE)["sample"], to=qm.Reference())
    monkeypatch.setattr(compilation.native, "validate_execution_interface", None)
    with pytest.raises(qm.ArtifactError, match="Q7016"):
        replace(artifact)
