"""Fixed quantum loops preserve indexed launch and result evaluation."""

import pytest

import qamomile as qm
from qamomile.persistence import deserialize, serialize

_SOURCES = [
    """
@qm.qkernel
def sample() -> qm.Shot[qm.Bool]:
    state = qm.prepare(False)
    for i in range(3):
        state = qm.x(state)
    return qm.measure(state) ^ (qm.get_current_shot() % 2 == 1)
""",
    """
@qm.qkernel
def rotate(state: qm.Q[qm.Bool]) -> qm.Q[qm.Bool]:
    return qm.ry(state, float(qm.get_current_shot() % 2) * 3.141592653589793)

@qm.qkernel
def sample() -> qm.Shot[qm.Bit]:
    state = qm.prepare(True)
    for i in range(3):
        state = rotate(state)
    return qm.measure(state)
""",
    """
@qm.qkernel
def rotate(state: qm.Q[qm.Bool]) -> qm.Q[qm.Bool]:
    for i in range(3):
        state = qm.x(state)
    return qm.ry(state, float(qm.get_current_shot() % 2) * 3.141592653589793)

@qm.qkernel
def sample() -> qm.Shot[qm.Bit]:
    return qm.measure(rotate(qm.prepare(False)))
""",
]


@pytest.mark.parametrize("source", _SOURCES, ids=["result", "helper", "helper-loop"])
def test_fixed_loops_reuse_persisted_indexed_template(source, monkeypatch) -> None:
    """Loop-local nodes cannot escape into independent numeric launch plans."""
    pytest.importorskip("qiskit_aer")
    import qamomile.compilation as compilation
    import qamomile.qiskit._integration as integration
    from qamomile.qiskit import Aer

    program = qm.build_source("import qamomile as qm\n" + source).specialize("sample")
    reference = qm.Reference()
    endpoint = Aer()
    reference_artifact = qm.compile(program, to=reference.target)
    artifact = deserialize(serialize(qm.compile(program, to=endpoint.target)))

    def forbidden(*args, **kwargs):
        raise AssertionError("fixed loops must reuse their compiled template")

    monkeypatch.setattr(compilation, "compile", forbidden)
    monkeypatch.setattr(integration, "_emit_circuit", forbidden)
    monkeypatch.setattr(integration, "_emit_for_target", forbidden)
    for shots in (5, 8, 5):
        expected = {True: (shots + 1) // 2, False: shots // 2}
        assert (
            qm.execute(reference_artifact, on=reference, shots=shots).values == expected
        )
        assert qm.execute(artifact, on=endpoint, shots=shots).values == expected
