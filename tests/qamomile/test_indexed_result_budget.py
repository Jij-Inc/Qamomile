"""Verify indexed result work is bounded before provider submission."""

import pytest

import qamomile as qm


def test_indexed_result_budget_precedes_backend_submission(monkeypatch) -> None:
    """Large result plans fail before a measured SDK batch can be submitted."""
    pytest.importorskip("qiskit")
    from qiskit.providers.basic_provider import BasicSimulator

    from qamomile._native import compiler as native
    from qamomile.proto import result_postprocessing_pb2 as pb2
    from qamomile.qiskit import QiskitBackend
    from qamomile.qiskit._payload import QiskitExecutable

    backend = BasicSimulator()
    endpoint = QiskitBackend(backend)
    artifact = qm.compile(
        qm.build_source("""
import qamomile as qm

@qm.qkernel
def sample() -> qm.Shot[qm.Bool]:
    measured = qm.measure(qm.prepare(False))
    return qm.get_current_shot() == 0
""").specialize("sample"),
        to=endpoint.target,
    )
    payload = artifact.execution_payload(QiskitExecutable)
    assert payload.shot_template is not None
    assert payload.circuit().num_qubits == 1
    template = pb2.ShotCircuitTemplateProto.FromString(
        dict(artifact.metadata.entries)["qamomile.native_shot_template.v1"]
    )
    result_plan = pb2.CircuitResultPostprocessingProto.FromString(template.result_plan)
    result_plan.scalars.extend(
        [pb2.CircuitResultScalarProto(bool_value=False)] * 300_000
    )
    template.result_plan = result_plan.SerializeToString(deterministic=True)
    payload.shot_template = native.NativeShotCircuitTemplate.from_bytes(
        template.SerializeToString(deterministic=True),
        artifact.parameter_abi._to_bytes(),
        artifact.result_abi._to_bytes(),
        artifact.target_fingerprint,
        artifact.source_digest.hex(),
    )

    def forbidden(*args, **kwargs):
        """Keep the oversized invocation entirely before the SDK boundary."""
        raise AssertionError("result work must be checked before SDK submission")

    monkeypatch.setattr(backend, "run", forbidden)
    with pytest.raises(qm.ExecutionError, match="Q4481.*work budget"):
        qm.execute(artifact, on=endpoint, shots=256)
