"""Checked integer preparation respects measured exits from fixed loops."""

import pytest

import qamomile as qm
from qamomile._native import compiler as native


@pytest.mark.parametrize("exit_kind", ["break", "continue"])
@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("value", [0, 1, 2])
@pytest.mark.parametrize("optimized", [False, True])
def test_preparation_checks_follow_measured_loop_exits(
    exit_kind, initial, value, optimized
):
    """An exited suffix stays lazy while an executed preparation is checked."""
    source = f"""
import qamomile as qm

@qm.qkernel
def sample(value: qm.Param[qm.Int]) -> qm.Shot[qm.Bit]:
    flag = qm.measure(qm.prepare({initial}))
    result = flag
    for _ in range(3):
        if flag:
            {exit_kind}
        result = qm.measure(qm.prepare(value))
    return result
"""
    program = qm.build_source(source).specialize("sample")
    if optimized:
        program = qm.Program.from_bytes(
            native.optimize_standard_program(program.to_bytes())
        )
    program = qm.Program.from_bytes(program.to_bytes())
    endpoint = qm.Reference()
    artifact = qm.compile(program, to=endpoint.target)
    if not initial and value == 2:
        with pytest.raises(qm.QamomileError, match="0 or 1"):
            qm.evaluate(program, value=value)
        with pytest.raises(qm.QamomileError, match="0 or 1"):
            qm.execute(artifact, on=endpoint, parameters={"value": value}, shots=4)
    else:
        expected = 1 if initial else value
        result = qm.evaluate(program, value=value)
        assert type(result) is int and result == expected
        samples = qm.execute(
            artifact, on=endpoint, parameters={"value": value}, shots=4, seed=3
        )
        assert samples.values == {expected: 4}


@pytest.mark.parametrize("exit_kind", ["break", "continue"])
@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("optimized", [False, True])
def test_measured_loop_exits_preserve_dependent_nested_bounds(
    exit_kind, initial, optimized
):
    """Safe constant bounds still fold after a measured loop exit."""
    source = f"""
import qamomile as qm

@qm.qkernel
def sample() -> qm.Shot[qm.Bit]:
    flag = qm.measure(qm.prepare({initial}))
    result = flag
    for i in range(3):
        if flag:
            {exit_kind}
        for j in range(i + 1):
            result = qm.measure(qm.prepare(0))
    return result
"""
    program = qm.build_source(source).specialize("sample")
    if optimized:
        program = qm.Program.from_bytes(
            native.optimize_standard_program(program.to_bytes())
        )
    program = qm.Program.from_bytes(program.to_bytes())
    expected = int(initial)
    assert qm.evaluate(program) == expected
    endpoint = qm.Reference()
    artifact = qm.compile(program, to=endpoint.target)
    samples = qm.execute(artifact, on=endpoint, shots=4, seed=3)
    assert samples.values == {expected: 4}


@pytest.mark.parametrize("exit_kind", ["break", "continue"])
@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("value", [0, 1, 2])
@pytest.mark.parametrize("optimized", [False, True])
def test_preparation_checks_follow_measured_loop_else(
    exit_kind, initial, value, optimized
):
    """Only a measured break skips checked preparation in a loop else."""
    source = f"""
import qamomile as qm

@qm.qkernel
def sample(value: qm.Param[qm.Int]) -> qm.Shot[qm.Bit]:
    flag = qm.measure(qm.prepare({initial}))
    result = flag
    for _ in range(3):
        if flag:
            {exit_kind}
    else:
        result = qm.measure(qm.prepare(value))
    return result
"""
    program = qm.build_source(source).specialize("sample")
    if optimized:
        program = qm.Program.from_bytes(
            native.optimize_standard_program(program.to_bytes())
        )
    program = qm.Program.from_bytes(program.to_bytes())
    endpoint = qm.Reference()
    artifact = qm.compile(program, to=endpoint.target)
    skipped = initial and exit_kind == "break"
    if not skipped and value == 2:
        with pytest.raises(qm.QamomileError, match="0 or 1"):
            qm.evaluate(program, value=value)
        with pytest.raises(qm.QamomileError, match="0 or 1"):
            qm.execute(artifact, on=endpoint, parameters={"value": value}, shots=4)
    else:
        expected = 1 if skipped else value
        assert qm.evaluate(program, value=value) == expected
        samples = qm.execute(
            artifact, on=endpoint, parameters={"value": value}, shots=4, seed=3
        )
        assert samples.values == {expected: 4}
