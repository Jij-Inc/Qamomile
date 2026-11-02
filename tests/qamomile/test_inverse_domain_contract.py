"""Inverse cancellation uses successful-execution semantics and static diagnostics."""

import pytest

import qamomile as qm
from qamomile._native import compiler as native
from qamomile.compilation import OptimizationPipeline
from qamomile.persistence import deserialize, serialize

PIPELINES = [
    None,
    OptimizationPipeline.standard(),
    OptimizationPipeline.multilevel(family=False),
]

SOURCE = """import qamomile as qm
@qm.qkernel
def action(state: qm.Q[qm.Bits[n]], i: qm.Param[qm.Int]) -> qm.Q[qm.Bits[n]]:
    return qm.x(state, target=i)
@qm.qkernel
def sample(i: qm.Param[qm.Int]) -> qm.Shot[qm.Bits[1]]:
    state = action(qm.qubits(1), i)
    return qm.measure(qm.adjoint(action, state, i))
"""


@pytest.mark.parametrize("optimizer", PIPELINES)
def test_unproved_call_pair_cancels_before_runtime_binding(optimizer):
    """An unknown selector does not require a proof of the callable's domain."""
    program = qm.build_source(SOURCE).specialize("sample")
    prepared = qm.Program.from_bytes(
        native.optimize_standard_program(program.to_bytes())
    )
    assert "qcore.call" not in prepared.entry.inspect()
    assert "qcore.adjoint_call" not in prepared.entry.inspect()
    artifact = deserialize(
        serialize(qm.compile(program, to=qm.Reference(), optimizer=optimizer))
    )
    for i in (0, 2):
        assert qm.execute(
            artifact, on=qm.Reference(), parameters={"i": i}, shots=2
        ).counts == {(0,): 2}


@pytest.mark.parametrize("optimizer", [*PIPELINES, OptimizationPipeline()])
@pytest.mark.parametrize("i", [-1, 1, 3])
def test_known_invalid_callee_selector_is_a_compile_error(optimizer, i):
    """Actual argument values and dependent owner widths reach callee diagnostics."""
    source = SOURCE.replace("def sample(i: qm.Param[qm.Int])", "def sample()")
    source = source.replace("action(qm.qubits(1), i)", f"action(qm.qubits(1), {i})")
    source = source.replace(
        "qm.adjoint(action, state, i)", f"qm.adjoint(action, state, {i})"
    )
    program = qm.build_source(source).specialize("sample")
    with pytest.raises(qm.QamomileError, match="Q7804.*module.action.*quantum.x"):
        qm.compile(program, to=qm.Reference(), optimizer=optimizer)


@pytest.mark.parametrize("enabled", [False, True])
def test_only_the_statically_selected_branch_is_diagnosed(enabled):
    """An invalid operation on an unselected path cannot reject a valid call."""
    source = f"""import qamomile as qm
@qm.qkernel
def action(state: qm.Q[qm.Bits[1]], enabled: qm.Param[qm.Bool], i: qm.Param[qm.Int]) -> qm.Q[qm.Bits[1]]:
    if enabled:
        state = qm.x(state, target=i)
    return state
@qm.qkernel
def sample() -> qm.Shot[qm.Bits[1]]:
    state = action(qm.qubits(1), {enabled}, 3)
    return qm.measure(qm.adjoint(action, state, {enabled}, 3))
"""
    program = qm.build_source(source).specialize("sample")
    if enabled:
        with pytest.raises(qm.QamomileError, match="Q7804.*outside width"):
            qm.compile(program, to=qm.Reference())
    else:
        assert qm.run(program, on=qm.Reference(), shots=2).result.counts == {(0,): 2}


@pytest.mark.parametrize("rounds", [0, 1])
def test_finite_loop_diagnostics_preserve_empty_ranges(rounds):
    """Checking a known loop follows its actual iterations without emitting gates."""
    source = f"""import qamomile as qm
@qm.qkernel
def action(state: qm.Q[qm.Bits[1]], rounds: qm.Const[qm.Int]) -> qm.Q[qm.Bits[1]]:
    for index in range(rounds):
        state = qm.x(state, target=index + 1)
    return state
@qm.qkernel
def sample() -> qm.Shot[qm.Bits[1]]:
    state = action(qm.qubits(1), {rounds})
    return qm.measure(qm.adjoint(action, state, {rounds}))
"""
    program = qm.build_source(source).specialize("sample")
    if rounds:
        with pytest.raises(qm.QamomileError, match="Q7804.*outside width"):
            qm.compile(program, to=qm.Reference())
    else:
        assert qm.run(program, on=qm.Reference(), shots=2).result.counts == {(0,): 2}


@pytest.mark.parametrize(
    "expression,argument,reason",
    [
        ("1.0 / value", "0.0", "division by zero"),
        ("qm.sqrt(value)", "-1.0", "negative|nonnegative"),
    ],
)
def test_known_numeric_failure_survives_nested_call_cancellation(
    expression, argument, reason
):
    """The shared native numeric algebra diagnoses errors across call boundaries."""
    source = f"""import qamomile as qm
@qm.qkernel
def angle(value: qm.Param[qm.Real]) -> qm.Real:
    return {expression}
@qm.qkernel
def action(state: qm.Q[qm.Bits[1]], value: qm.Param[qm.Real]) -> qm.Q[qm.Bits[1]]:
    return qm.rz(state, angle(value))
@qm.qkernel
def sample() -> qm.Shot[qm.Bits[1]]:
    state = action(qm.qubits(1), {argument})
    return qm.measure(qm.adjoint(action, state, {argument}))
"""
    program = qm.build_source(source).specialize("sample")
    with pytest.raises(qm.QamomileError, match=f"Q7804.*({reason})"):
        qm.compile(program, to=qm.Reference())


def test_unknown_control_flow_does_not_require_a_success_proof():
    """An unknown branch may contain a domain error; the inverse law still applies."""
    source = """import qamomile as qm
@qm.qkernel
def action(state: qm.Q[qm.Bits[1]], enabled: qm.Param[qm.Bool], i: qm.Param[qm.Int]) -> qm.Q[qm.Bits[1]]:
    if enabled:
        state = qm.x(state, target=i)
    return state
@qm.qkernel
def sample(enabled: qm.Param[qm.Bool]) -> qm.Shot[qm.Bits[1]]:
    state = action(qm.qubits(1), enabled, 5)
    return qm.measure(qm.adjoint(action, state, enabled, 5))
"""
    artifact = qm.compile(
        qm.build_source(source).specialize("sample"), to=qm.Reference()
    )
    assert qm.execute(
        artifact, on=qm.Reference(), parameters={"enabled": True}, shots=2
    ).counts == {(0,): 2}


def test_unknown_numerator_with_known_zero_denominator_is_diagnosed():
    """A denominator proves failure without knowing every operand."""
    source = """import qamomile as qm
@qm.qkernel
def action(state: qm.Q[qm.Bits[1]], theta: qm.Param[qm.Real], denominator: qm.Param[qm.Real]) -> qm.Q[qm.Bits[1]]:
    return qm.rz(state, theta / denominator)
@qm.qkernel
def sample(theta: qm.Param[qm.Real]) -> qm.Shot[qm.Bits[1]]:
    state = action(qm.qubits(1), theta, 0.0)
    return qm.measure(qm.adjoint(action, state, theta, 0.0))
"""
    program = qm.build_source(source).specialize("sample")
    with pytest.raises(qm.QamomileError, match="Q7804.*division by zero"):
        qm.compile(program, to=qm.Reference())


def test_same_unknown_selector_proves_aliasing_before_cancellation():
    """Repeated SSA selectors prove a domain error even without numeric values."""
    source = SOURCE.replace(
        "return qm.x(state, target=i)", "return qm.cx(state, control=i, target=i)"
    )
    program = qm.build_source(source).specialize("sample")
    with pytest.raises(qm.QamomileError, match="Q7804.*distinct"):
        qm.compile(program, to=qm.Reference())


def test_diagnostic_work_exhaustion_does_not_prevent_cancellation():
    """A large finite loop stays compact and requires no termination proof."""
    source = """import qamomile as qm
@qm.qkernel
def action(state: qm.Q[qm.Bits[1]], rounds: qm.Const[qm.Int]) -> qm.Q[qm.Bits[1]]:
    for index in range(rounds):
        state = qm.x(state)
    return state
@qm.qkernel
def sample() -> qm.Shot[qm.Bits[1]]:
    state = action(qm.qubits(1), 1000000000)
    return qm.measure(qm.adjoint(action, state, 1000000000))
"""
    program = qm.build_source(source).specialize("sample")
    payload = program.to_bytes()
    prepared = qm.Program.from_bytes(native.optimize_standard_program(payload))
    assert "qcore.call" not in prepared.entry.inspect()
    assert len(prepared.to_bytes()) <= len(payload)
    assert program.to_bytes() == payload
