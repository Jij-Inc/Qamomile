"""Quantum ownership is preserved across supported assignment spellings."""

import pytest

import qamomile as qm


@pytest.mark.parametrize(
    "assignment",
    [
        "first = qm.x(second)",
        "first, = (qm.x(second),)",
        "[first] = (qm.x(second),)",
        "first = qm.qubits(1)",
        "[first] = (qm.qubits(1),)",
    ],
)
def test_rebinding_cannot_abandon_an_input_owner(assignment):
    """Renaming another owner or allocating a replacement cannot lose an input."""
    source = f"""
import qamomile as qm

@qm.qkernel
def replace_input(first: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    second = qm.qubits(1)
    {assignment}
    return first
"""
    with pytest.raises(qm.VerificationError, match="Q2001.*observed 0 uses"):
        qm.build_source(source).specialize("replace_input")


@pytest.mark.parametrize(
    "assignment", ["first = qm.x(first)", "[first] = (qm.x(first),)"]
)
def test_self_update_preserves_an_input_owner(assignment):
    """An unused local register does not make input self-updates invalid."""
    source = f"""
import qamomile as qm

@qm.qkernel
def update_input(first: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    second = qm.qubits(1)
    {assignment}
    return first
"""
    program = qm.build_source(source).specialize("update_input")
    assert qm.Program.from_bytes(program.to_bytes()) == program


@pytest.mark.parametrize(
    "assignment",
    [
        "first, *rest = (qm.x(second), third)",
        "*rest, first = (third, qm.x(second))",
        "first, (second, third) = (qm.x(second), (first, third))",
        "[first, [second, third]] = (qm.x(second), (first, third))",
    ],
)
def test_unsupported_assignment_shapes_fail_before_execution(assignment):
    """Nested and starred targets cannot bypass source validation."""
    source = f"""
import qamomile as qm

@qm.qkernel
def unsupported() -> qm.Shot[qm.Bits[1]]:
    first = qm.qubits(1)
    second = qm.qubits(1)
    third = qm.qubits(1)
    {assignment}
    return qm.measure(first)
"""
    with pytest.raises(qm.FrontendError, match="Q5056"):
        qm.build_source(source)


@pytest.fixture(params=["reference", "qiskit"])
def endpoint(request):
    """Run the same owned-register source on local execution endpoints."""
    if request.param == "reference":
        return qm.Reference()
    pytest.importorskip("qiskit_aer")
    from qamomile.qiskit import Aer

    return Aer()


@pytest.mark.parametrize(
    ("assignment", "expected"),
    [
        ("first, second = second, first", (0, 1, 1)),
        ("[first, second] = (second, first)", (0, 1, 1)),
        ("first = qm.x(first)", (0, 0, 1)),
        ("[first] = (qm.x(first),)", (0, 0, 1)),
    ],
)
def test_supported_swaps_and_self_updates_preserve_owned_values(
    assignment, expected, endpoint
):
    """Different-width owners retain their bit order after legal rebinding."""
    source = f"""
import qamomile as qm

@qm.qkernel
def rebind() -> qm.Shot[qm.Bits[3]]:
    first = qm.prepare((1,))
    second = qm.prepare((0, 1))
    {assignment}
    return qm.measure(qm.concat(first, second))
"""
    program = qm.build_source(source).specialize("rebind")
    restored = qm.Program.from_bytes(program.to_bytes())
    record = qm.run(restored, on=endpoint, shots=8, seed=17)

    assert record.result.counts == {expected: 8}


@pytest.mark.parametrize(
    "assignment",
    [
        "first = qm.x(second)",
        "first, = (qm.x(second),)",
        "[first] = (qm.x(second),)",
    ],
)
def test_rebinding_a_local_owner_preserves_verified_cleanup(assignment, endpoint):
    """An overwritten local preparation is reversed and released safely."""
    source = f"""
import qamomile as qm

@qm.qkernel
def replace_local() -> qm.Shot[qm.Bits[1]]:
    first = qm.qubits(1)
    first = qm.h(first)
    first = qm.t(first)
    second = qm.qubits(1)
    {assignment}
    return qm.measure(first)
"""
    program = qm.build_source(source).specialize("replace_local")
    study = qm.resources.logical(program)
    released = study.metric(qm.resources.logical_operation_count("quantum.release"))

    assert released.numeric_enclosure() == (1, 1)
    assert qm.run(program, on=endpoint, shots=8, seed=17).result.counts == {(1,): 8}
