"""Keep logical-shot materialization separate from ordinary invocation batches."""

from types import SimpleNamespace

import pytest

import qamomile as qm
import qamomile.compilation as compilation


def _materialization_witness(monkeypatch, *, fail_preflight=False):
    """Keep source ordering real while replacing repeated child compilation."""
    pytest.importorskip("qiskit")
    from qiskit.providers.basic_provider import BasicSimulator

    from qamomile.qiskit import QiskitBackend
    from qamomile.qiskit._integration import _CAPABILITIES

    backend = BasicSimulator()
    endpoint = QiskitBackend(backend, requirements=_CAPABILITIES)
    artifact = qm.compile(
        qm.build_source("""
import qamomile as qm
@qm.qkernel
def sample() -> qm.Shot[qm.Bit]:
    state = qm.prepare(False)
    if qm.get_current_shot() % 2 == 0:
        state = qm.x(state)
    return qm.measure(state)
""").specialize("sample"),
        to=endpoint.target,
    )
    recipe = artifact.execution_payload(compilation._LaunchMaterializerPayload)
    child = qm.compile(
        qm.build_source("""
import qamomile as qm
@qm.qkernel
def sample() -> qm.Shot[qm.Bit]:
    return qm.measure(qm.prepare(False))
""").specialize("sample"),
        to=endpoint.target,
    )
    checked = []
    indices = []

    def preflight(shots):
        """Record that native source checks precede every materialized index."""
        checked.append(shots)
        if fail_preflight:
            raise ValueError("native indexed source work budget")

    def bind(index, shots):
        """Record the unchanged invocation-wide logical domain."""
        assert checked == [shots]
        indices.append((index, shots))
        return SimpleNamespace(to_bytes=lambda: b"bound-source")

    owner = SimpleNamespace(check_indexed_invocation=preflight, bind_current_shot=bind)
    monkeypatch.setattr(compilation.Program, "_native_program", lambda self: owner)
    monkeypatch.setattr(
        compilation.Program, "from_bytes", staticmethod(lambda payload: recipe.program)
    )
    monkeypatch.setattr(compilation, "compile", lambda *args, **kwargs: child)
    return artifact, endpoint, backend, child, checked, indices


def test_logical_recipe_crosses_old_invocation_batch_count_without_submission(
    monkeypatch,
) -> None:
    """4097 logical indices use one native domain, without executing SDK shots."""
    artifact, endpoint, _, child, checked, indices = _materialization_witness(
        monkeypatch
    )
    invocation = qm.Invocation.create(shots=4097)
    children = compilation._current_shot_children(artifact, endpoint, invocation)
    assert checked == [4097]
    assert len(children) == 4097
    assert indices == [(index, 4097) for index in range(4097)]
    assert all(item is child and request.shots == 1 for item, request in children)


def test_native_preflight_failure_precedes_logical_source_expansion(
    monkeypatch,
) -> None:
    """Resource rejection cannot leave partially generated child invocations."""
    artifact, endpoint, _, _, checked, indices = _materialization_witness(
        monkeypatch, fail_preflight=True
    )
    with pytest.raises(
        qm.ExecutionError, match="Q4480.*native indexed source work budget"
    ):
        compilation._current_shot_children(
            artifact, endpoint, qm.Invocation.create(shots=10_000_000)
        )
    assert checked == [10_000_000]
    assert indices == []


def test_recipe_retention_checks_backend_capacity_before_next_child(
    monkeypatch,
) -> None:
    """Declared capacity stops generation before an oversized SDK batch exists."""
    artifact, endpoint, backend, _, _, indices = _materialization_witness(monkeypatch)
    monkeypatch.setattr(type(backend), "max_circuits", property(lambda self: 1))
    with pytest.raises(qm.ExecutionError, match="Q8234.*backend limit is 1"):
        compilation._current_shot_children(
            artifact, endpoint, qm.Invocation.create(shots=4097)
        )
    assert indices == [(0, 4097), (1, 4097)]
