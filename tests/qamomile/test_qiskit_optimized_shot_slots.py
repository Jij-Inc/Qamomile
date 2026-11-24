"""Compiler slot identities survive SDK removal of parameterized gates."""

from types import SimpleNamespace

import pytest

import qamomile as qm
from qamomile.persistence import deserialize, serialize


def _source(gates):
    """Build independent rotations whose native slot positions stay observable."""
    return f"""
import qamomile as qm

@qm.qkernel
def sample() -> tuple[qm.Shot[qm.Bit], qm.Shot[qm.Bit], qm.Shot[qm.Bit], qm.Shot[qm.Bool]]:
    index = qm.get_current_shot()
    a = float(index % 2) * 3.141592653589793
    b = float(index % 3) * 3.141592653589793
    c = float((index + 1) % 2) * 3.141592653589793
    q0 = qm.{gates[0]}(qm.prepare(False), a)
    q1 = qm.{gates[1]}(qm.prepare(False), b)
    q2 = qm.{gates[2]}(qm.prepare(False), c)
    return qm.measure(q0), qm.measure(q1), qm.measure(q2), index == 0
"""


_REMOVALS = [
    pytest.param(("rz", "ry", "ry"), (1, 2), id="first"),
    pytest.param(("ry", "rz", "ry"), (0, 2), id="middle"),
    pytest.param(("rz", "rz", "rz"), (), id="all"),
]


def _expected(gates):
    """Return exact results of zero or pi rotations at four logical indices."""
    counts = {}
    for index in range(4):
        row = (
            0 if gates[0] == "rz" else index % 2,
            0 if gates[1] == "rz" else (index % 3) % 2,
            0 if gates[2] == "rz" else (index + 1) % 2,
            int(index == 0),
        )
        counts[row] = counts.get(row, 0) + 1
    return counts


def _check_slots(artifact, retained):
    """Check all native columns while only retained SDK symbols remain free."""
    from qamomile.qiskit._payload import QiskitExecutable
    from qamomile.qiskit._shot_template import (
        aer_parameter_bindings,
        template_inputs,
    )

    payload = artifact.execution_payload(QiskitExecutable)
    circuit, hidden, rows = template_inputs(artifact, qm.Invocation.create(shots=4))
    assert payload.shot_template.slot_count == rows.slot_count == len(hidden) == 3
    assert set(circuit.parameters) == {hidden[index] for index in retained}
    assert aer_parameter_bindings(circuit, hidden, rows) == {
        hidden[index]: [row[index] for row in rows] for index in retained
    }


@pytest.mark.parametrize("gates,retained", _REMOVALS)
@pytest.mark.parametrize("restore", [False, True])
@pytest.mark.parametrize("target_kind", ["basis", "backend"])
def test_compile_time_slot_removal_preserves_native_columns(
    gates, retained, restore, target_kind
):
    """Real SDK compilation may remove a prefix, middle, or every numeric slot."""
    from qiskit.providers.basic_provider import BasicSimulator

    from qamomile.qiskit import Qiskit, QiskitBackend

    endpoint = QiskitBackend(BasicSimulator(), optimization_level=3)
    target = (
        Qiskit(basis_gates=("rz", "sx", "x", "cx"), optimization_level=3)
        if target_kind == "basis"
        else endpoint.target
    )
    artifact = qm.compile(
        qm.build_source(_source(gates)).specialize("sample"), to=target
    )
    if restore:
        artifact = deserialize(serialize(artifact))
    _check_slots(artifact, retained)
    if target_kind == "backend":
        assert qm.execute(artifact, on=endpoint, shots=4).counts == _expected(gates)


@pytest.mark.parametrize("gates,retained", _REMOVALS)
@pytest.mark.parametrize("restore", [False, True])
def test_runtime_filters_removed_columns_after_compile(
    gates, retained, restore, monkeypatch
):
    """Runtime binds surviving SDK symbols from their original native columns."""
    from qiskit.primitives import StatevectorSampler
    from qiskit.providers.basic_provider import BasicSimulator

    import qamomile.qiskit.runtime as runtime
    from qamomile.qiskit import IBMRuntime

    submissions = []

    class Sampler:
        """Exercise real shaped SDK parameter binding without provider access."""

        def __init__(self, *, mode, options):
            self.mode = mode

        def backend(self):
            return self.mode

        def run(self, pubs, *, shots):
            submissions.append((pubs, shots))
            result = StatevectorSampler(seed=0).run(pubs, shots=shots).result()
            return SimpleNamespace(
                job_id=lambda: "optimized-slots",
                result=lambda: result,
                status=lambda: "DONE",
            )

    monkeypatch.setattr(
        runtime, "_runtime_module", lambda: SimpleNamespace(SamplerV2=Sampler)
    )
    endpoint = IBMRuntime(BasicSimulator(), optimization_level=3)
    artifact = qm.compile(
        qm.build_source(_source(gates)).specialize("sample"), to=endpoint.target
    )
    if restore:
        artifact = deserialize(serialize(artifact))
    _check_slots(artifact, retained)
    assert qm.execute(artifact, on=endpoint, shots=4).counts == _expected(gates)
    assert len(submissions) == 1
    pubs, shots = submissions[0]
    assert len(pubs) == 1 and shots == 1
    assert pubs[0][1].shape == (4,)


@pytest.mark.parametrize("restore", [False, True])
def test_removed_numeric_slot_keeps_later_source_errors(restore):
    """Erasing an irrelevant gate cannot suppress numeric checks at later shots."""
    from qiskit.providers.basic_provider import BasicSimulator

    from qamomile.qiskit import QiskitBackend

    endpoint = QiskitBackend(BasicSimulator(), optimization_level=3)
    artifact = qm.compile(
        qm.build_source("""import qamomile as qm
@qm.qkernel
def sample() -> qm.Shot[qm.Bit]:
    index = qm.get_current_shot()
    angle = 1.0 / float(1 - index)
    return qm.measure(qm.rz(qm.prepare(False), angle))
""").specialize("sample"),
        to=endpoint.target,
    )
    if restore:
        artifact = deserialize(serialize(artifact))
    assert qm.execute(artifact, on=endpoint, shots=1).counts == {(0,): 1}
    with pytest.raises(qm.ExecutionError, match="Q4481.*zero"):
        qm.execute(artifact, on=endpoint, shots=2)
