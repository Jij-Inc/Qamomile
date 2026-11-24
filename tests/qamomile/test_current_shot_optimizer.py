"""Exact quantum cancellation preserves invocation execution cardinality."""

import pytest

import qamomile as qm
from qamomile.compilation import OptimizationPipeline
from qamomile.effects import Effect, Requirement
from qamomile.persistence import deserialize, serialize
from qamomile.qiskit import Aer


@pytest.mark.parametrize("endpoint_type", [qm.Reference, Aer], ids=["reference", "aer"])
@pytest.mark.parametrize("optimizer_name", ["default", "standard", "multilevel"])
@pytest.mark.parametrize("result_type", ["Bool", "Unit"])
@pytest.mark.parametrize("context_use", ["unused", "angle", "transitive"])
def test_inverse_cancellation_preserves_contextual_completion(
    endpoint_type, optimizer_name, result_type, context_use
):
    """Helper/inverse pairs retain every completion through optimization and reuse."""
    if endpoint_type is Aer:
        pytest.importorskip("qiskit")
        pytest.importorskip("qiskit_aer")
    helper = (
        "    ignored = qm.get_current_shot()\n    return qm.x(state)\n"
        if context_use != "angle"
        else "    return qm.ry(state, float(qm.get_current_shot()) * 0.1)\n"
    )
    wrapper = (
        "@qm.qkernel\n"
        "def wrapped(state: qm.Q[qm.Bool]) -> qm.Q[qm.Bool]:\n"
        "    return action(state)\n"
        if context_use == "transitive"
        else ""
    )
    callee = "wrapped" if context_use == "transitive" else "action"
    source = (
        "import qamomile as qm\n"
        "@qm.qkernel\n"
        "def action(state: qm.Q[qm.Bool]) -> qm.Q[qm.Bool]:\n"
        + helper
        + wrapper
        + "@qm.qkernel\n"
        + f"def sample() -> qm.{result_type}:\n"
        + f"    state = {callee}(qm.prepare(False))\n"
        + f"    state = qm.adjoint({callee}, state)\n"
        + "    qm.discard(state)\n"
        + ("    return True\n" if result_type == "Bool" else "    return\n")
    )
    program = qm.build_source(source).specialize("sample")
    program = qm.Program.from_bytes(program.to_bytes())
    assert Requirement.LOGICAL_SHOT_CONTEXT in program.entry.requirements
    assert Effect.MEASUREMENT not in program.entry.effects
    assert Effect.PROBABILISTIC not in program.entry.effects
    optimizer = {
        "default": None,
        "standard": OptimizationPipeline.standard(),
        "multilevel": OptimizationPipeline.multilevel(family=False),
    }[optimizer_name]
    endpoint = endpoint_type()
    artifact = deserialize(
        serialize(qm.compile(program, to=endpoint.target, optimizer=optimizer))
    )
    assert artifact.result_abi.per_shot
    assert Requirement.LOGICAL_SHOT_CONTEXT in artifact.execution_contract.requirements
    expected = True if result_type == "Bool" else None
    for shots in (1, 5, 3, 5):
        result = qm.execute(artifact, on=endpoint, shots=shots)
        assert isinstance(result, qm.SampleResult)
        assert result.shots == shots
        assert result.values == {expected: shots}
        restored = qm.ExecutionSnapshot.from_bytes(
            qm.ExecutionSnapshot.completed(result).to_bytes()
        )
        assert restored.result.values == {expected: shots}
