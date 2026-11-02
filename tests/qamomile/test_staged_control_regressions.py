"""Regression witnesses for measured loop exits and sealed control arguments."""

from __future__ import annotations

import cmath

import pytest

import qamomile as qm
from qamomile._native import compiler as native
from qamomile.circuit import launch_control_bindings
from qamomile.compilation import ExecutionEndpoint

MEASURED_BREAK_SOURCE = """
import qamomile as qm

@qm.qkernel
def sample(flag: qm.Const[qm.Bool]) -> qm.Shot[qm.Bits[1]]:
    bit = qm.measure(qm.prepare(flag))
    marker = qm.qubits(1)
    for _ in range(3):
        if not bit:
            break
        marker = qm.x(marker)
        bit = qm.measure(qm.prepare(False))
    return qm.measure(marker)
"""

CONTROLLED_CONST_SOURCE = """
import qamomile as qm

@qm.qkernel
def leaf(q: qm.Q[qm.Bits[1]], theta: qm.Real, mode: qm.Const[qm.Bool]) -> qm.Q[qm.Bits[1]]:
    if mode:
        q = qm.x(q)
    return qm.p(q, theta)

@qm.qkernel
def component(theta: qm.Param[qm.Real], mode: qm.Const[qm.Bool]) -> qm.Q[qm.Bits[2]]:
    c = qm.prepare((1,))
    q = qm.qubits(1)
    c, q = qm.control(leaf, c, q, theta, mode)
    return qm.concat(c, q)
"""


@pytest.mark.parametrize("flag", [False, True])
@pytest.mark.parametrize("endpoint", [qm.Reference(), qm.qiskit.Aer()])
def test_measured_break_runs_after_program_roundtrip(
    flag: bool, endpoint: ExecutionEndpoint
) -> None:
    """A measured break reaches the same deterministic result on both endpoints."""
    program = qm.build_source(MEASURED_BREAK_SOURCE).specialize("sample", flag=flag)
    program = qm.Program.from_bytes(program.to_bytes())
    result = qm.run(program, on=endpoint, shots=8, seed=7).result
    assert isinstance(result, qm.SampleResult)
    assert result.counts == {(int(flag),): 8}


@pytest.mark.parametrize("limit", [3, 5000])
@pytest.mark.parametrize("exit_kind", ["break", "continue"])
def test_measured_loop_exit_resources_remain_enclosures(
    limit: int, exit_kind: str
) -> None:
    """Both bounded interpretation and symbolic repetition retain possible work."""
    source = f"""
import qamomile as qm

@qm.qkernel
def sample(flag: qm.Const[qm.Bool]) -> qm.Shot[qm.Bits[1]]:
    bit = qm.measure(qm.prepare(flag))
    marker = qm.qubits(1)
    for _ in range({limit}):
        if bit:
            {exit_kind}
        marker = qm.x(marker)
    else:
        marker = qm.h(marker)
    return qm.measure(marker)
"""
    program = qm.build_source(source).specialize("sample", flag=True)
    study = qm.resources.logical(program)
    metric = study.metric(qm.resources.logical_operation_count("quantum.x"))
    assert metric.relation is qm.resources.Relation.ENCLOSURE
    assert metric.numeric_enclosure() == (0, limit)
    otherwise = study.metric(qm.resources.logical_operation_count("quantum.h"))
    assert otherwise.numeric_enclosure() == ((0, 1) if exit_kind == "break" else (1, 1))


@pytest.mark.parametrize("flag", [False, True])
@pytest.mark.parametrize("endpoint", [qm.Reference(), qm.qiskit.Aer()])
def test_measured_nested_break_consumes_only_inner_exit(
    flag: bool, endpoint: ExecutionEndpoint
) -> None:
    """An inner measured break skips its else but preserves the outer iterations."""
    source = """
import qamomile as qm

@qm.qkernel
def sample(flag: qm.Const[qm.Bool]) -> qm.Shot[qm.Bits[1]]:
    bit = qm.measure(qm.prepare(flag))
    marker = qm.qubits(1)
    for _ in range(3):
        for _ in range(2):
            if bit:
                if bit:
                    break
            marker = qm.x(marker)
        else:
            marker = qm.x(marker)
    return qm.measure(marker)
"""
    program = qm.build_source(source).specialize("sample", flag=flag)
    result = qm.run(program, on=endpoint, shots=8, seed=7).result
    assert isinstance(result, qm.SampleResult)
    assert result.counts == {(int(not flag),): 8}


@pytest.mark.parametrize("mode", [False, True])
def test_numeric_controlled_circuit_omits_sealed_const(mode: bool) -> None:
    """Numeric extraction preserves the controlled phase and both Const branches."""
    from qiskit.quantum_info import Statevector

    program = qm.build_source(CONTROLLED_CONST_SOURCE).specialize(
        "component", mode=mode
    )
    program = qm.Program.from_bytes(program.to_bytes())
    assert native.inspect_program(program.to_bytes()).entry_control_parameters == []
    assert launch_control_bindings(program, {"theta": 0.63}) == {}
    symbolic = qm.qiskit.to_circuit(program)
    parameter = next(iter(symbolic.parameters))
    expected = symbolic.assign_parameters({parameter: 0.63})
    concrete = qm.qiskit.to_circuit(program, parameters={"theta": 0.63})
    assert not concrete.parameters
    actual_state = Statevector.from_instruction(concrete).data
    assert actual_state == pytest.approx(Statevector.from_instruction(expected).data)
    amplitude_index = 3 if mode else 1
    assert actual_state[amplitude_index] == pytest.approx(
        cmath.exp(0.63j) if mode else 1
    )


@pytest.mark.parametrize("flag", [False, True])
@pytest.mark.parametrize("exit_kind", ["break", "continue"])
@pytest.mark.parametrize("endpoint", [qm.Reference(), qm.qiskit.Aer()])
def test_measured_exits_preserve_updated_carries(
    flag: bool, exit_kind: str, endpoint: ExecutionEndpoint
) -> None:
    """Exit snapshots preserve updates before an exit and suppress its suffix."""
    source = f"""
import qamomile as qm

@qm.qkernel
def sample(flag: qm.Const[qm.Bool], expected: qm.Const[qm.Int]) -> tuple[qm.Shot[qm.Bits[1]], qm.Shot[qm.Bool]]:
    bit = qm.measure(qm.prepare(flag))
    marker = qm.qubits(1)
    count = 0
    for _ in range(4):
        count += 1
        if bit:
            {exit_kind}
        marker = qm.x(marker)
        count += 100
    else:
        marker = qm.x(marker)
        count += 10
    return qm.measure(marker), count == expected
"""
    count = 1 if flag and exit_kind == "break" else (14 if flag else 414)
    program = qm.build_source(source).specialize("sample", flag=flag, expected=count)
    result = qm.run(program, on=endpoint, shots=8, seed=7).result
    assert isinstance(result, qm.SampleResult)
    assert result.values == {((0 if flag and exit_kind == "break" else 1,), True): 8}


@pytest.mark.parametrize("flag", [False, True])
@pytest.mark.parametrize("endpoint", [qm.Reference(), qm.qiskit.Aer()])
def test_measured_branch_breaks_or_continues_without_fallthrough(
    flag: bool, endpoint: ExecutionEndpoint
) -> None:
    """Both exiting branches carry values even when no branch yields normally."""
    source = """
import qamomile as qm

@qm.qkernel
def sample(flag: qm.Const[qm.Bool], expected: qm.Const[qm.Int]) -> qm.Shot[qm.Bool]:
    bit = qm.measure(qm.prepare(flag))
    count = 0
    for _ in range(3):
        count += 1
        if bit:
            break
        else:
            continue
    return count == expected
"""
    program = qm.build_source(source).specialize(
        "sample", flag=flag, expected=1 if flag else 3
    )
    result = qm.run(program, on=endpoint, shots=8, seed=7).result
    assert isinstance(result, qm.SampleResult)
    assert result.values == {True: 8}


def test_numeric_controlled_circuit_still_requires_runtime_dispatch() -> None:
    """Filtering a sealed Const retains live launch branches and their validation."""
    source = (
        CONTROLLED_CONST_SOURCE.replace(
            "mode: qm.Const[qm.Bool])",
            "mode: qm.Const[qm.Bool], select: qm.Param[qm.Bool])",
        )
        .replace(
            "return qm.p(q, theta)",
            "if select:\n        q = qm.z(q)\n    return qm.p(q, theta)",
        )
        .replace("c, q, theta, mode)", "c, q, theta, mode, select)")
    )
    program = qm.build_source(source).specialize("component", mode=True)
    assert native.inspect_program(program.to_bytes()).entry_control_parameters == [2]
    assert launch_control_bindings(program, {"theta": 0.63, "select": True}) == {
        "select": True
    }
    with pytest.raises(qm.ExecutionError, match="Q8102.*select"):
        qm.qiskit.to_circuit(program, parameters={"theta": 0.63})
    with pytest.raises(qm.ExecutionError, match="select.*Bool"):
        qm.qiskit.to_circuit(program, parameters={"theta": 0.63, "select": 1})
    circuit = qm.qiskit.to_circuit(program, parameters={"theta": 0.63, "select": True})
    from qiskit.quantum_info import Statevector

    assert Statevector.from_instruction(circuit).data[3] == pytest.approx(
        -cmath.exp(0.63j)
    )


@pytest.mark.parametrize("flag", [False, True])
@pytest.mark.parametrize("endpoint", [qm.Reference(), qm.qiskit.Aer()])
def test_measured_break_rejects_path_dependent_return_in_loop_else(
    flag: bool, endpoint: ExecutionEndpoint
) -> None:
    """Mixed return and carry ABIs fail before any endpoint can return a wrong value."""
    source = """
import qamomile as qm

@qm.qkernel
def sample(flag: qm.Const[qm.Bool]) -> qm.Shot[qm.Bits[1]]:
    bit = qm.measure(qm.prepare(flag))
    marker = qm.qubits(1)
    for _ in range(3):
        if bit:
            break
    else:
        return qm.measure(marker)
    marker = qm.x(marker)
    return qm.measure(marker)
"""
    program = qm.build_source(source).specialize("sample", flag=flag)
    with pytest.raises(qm.CapabilityError, match="Q4211.*path-dependent.*return"):
        qm.run(program, on=endpoint, shots=8, seed=7)
