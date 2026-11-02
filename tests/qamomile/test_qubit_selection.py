"""Owned noncontiguous selection preserves coordinates without routing gates."""

from __future__ import annotations

import pytest

import qamomile as qm
from qamomile.circuit import lower_to_circuit
from qamomile.stdlib import on_qubits


@qm.qkernel
def flip_selected(state: qm.Q[qm.Bits[n]]) -> qm.Q[qm.Bits[n]]:  # noqa: F821
    """Flip every qubit passed by the owned selection."""
    return qm.x(state)


@qm.qkernel
def flip_first(state: qm.Q[qm.Bits[n]]) -> qm.Q[qm.Bits[n]]:  # noqa: F821
    """Expose the ordering of the selected helper coordinates."""
    return qm.x(state, target=0)


@qm.qkernel
def bell_selected(state: qm.Q[qm.Bits[2]]) -> qm.Q[qm.Bits[2]]:
    """Entangle only the selected coordinates."""
    return qm.cx(qm.h(state, target=0), control=0, target=1)


@qm.qkernel
def pair_flip(state: qm.Q[qm.Bits[2]]) -> qm.Q[qm.Bits[2]]:
    """Provide the same fixed-width helper ABI for every SDK."""
    return qm.x(state)


@qm.qkernel
def pair_first(state: qm.Q[qm.Bits[2]]) -> qm.Q[qm.Bits[2]]:
    """Expose the first coordinate through a fixed-width helper ABI."""
    return qm.x(state, target=0)


def selection_sample(process, *, width, indices):
    """Close an owned selection in a measured sample kernel."""
    selected = on_qubits(process, width=width, indices=indices)

    @qm.qkernel
    def sample() -> qm.Shot[qm.Bits[width]]:
        return qm.measure(selected(qm.qubits(width)))

    return sample


@pytest.mark.parametrize("indices", [(0, 2), (3, 1), (3, 2, 1, 0), (2,), (0, 1, 2, 3)])
def test_selection_preserves_order_without_extra_quantum_gates(indices):
    """Static regrouping emits exactly the helper's single X gate."""
    program = qm.specialize(selection_sample(flip_first, width=4, indices=indices))
    restored = qm.Program.from_bytes(program.to_bytes())
    result = qm.run(restored, on=qm.Reference(), shots=8, seed=7).result
    expected = tuple(int(index == indices[0]) for index in range(4))
    assert result.counts == {expected: 8}
    gates = [
        i
        for i in lower_to_circuit(restored).body.instructions
        if i.opcode in {"x", "swap", "cx"}
    ]
    assert [(i.opcode, i.qubits) for i in gates] == [("x", (indices[0],))]
    assert qm.resources.logical(restored).metric(
        qm.resources.LOGICAL_GATES
    ).numeric_enclosure() == (1, 1)


def exercise_qubit_selection_backend(endpoint, *, seed=7):
    """Execute a strided selection and an asymmetric ordered selection."""
    sample = selection_sample(pair_flip, width=4, indices=(0, 2))
    program = qm.Program.from_bytes(qm.specialize(sample).to_bytes())
    artifact = qm.compile(program, to=endpoint.target)
    result = qm.execute(artifact, on=endpoint, shots=8, seed=seed)
    assert result.counts == {(1, 0, 1, 0): 8}
    ordered = selection_sample(pair_first, width=4, indices=(3, 1))
    assert qm.run(ordered, on=endpoint, shots=8, seed=seed).result.counts == {
        (0, 0, 0, 1): 8
    }
    selected = on_qubits(pair_first, width=4, indices=(3, 1))

    @qm.qkernel
    def state() -> qm.Q[qm.Bits[4]]:
        return selected(qm.qubits(4))

    assert qm.expectation(state, qm.Z(3), on=endpoint, shots=8, seed=seed).value == -1.0

    @qm.qkernel
    def controlled() -> qm.Shot[qm.Bits[5]]:
        control = qm.prepare((1,))
        control, data = qm.control(selected, control, qm.qubits(4))
        return qm.measure(qm.concat(control, data))

    assert qm.run(controlled, on=endpoint, shots=8, seed=seed).result.counts == {
        (1, 0, 0, 0, 1): 8
    }

    @qm.qkernel
    def inverse() -> qm.Shot[qm.Bits[4]]:
        return qm.measure(qm.adjoint(selected, selected(qm.qubits(4))))

    assert qm.run(inverse, on=endpoint, shots=8, seed=seed).result.counts == {
        (0, 0, 0, 0): 8
    }


def test_reference_selection_entanglement_and_inverse():
    """Untouched coordinates survive entanglement and inverse application."""
    selected = on_qubits(bell_selected, width=4, indices=(3, 1))

    @qm.qkernel
    def sample() -> qm.Shot[qm.Bits[4]]:
        state = qm.x(qm.qubits(4), target=0)
        state = selected(state)
        return qm.measure(state)

    assert set(qm.run(sample, on=qm.Reference(), shots=64, seed=7).result.counts) == {
        (1, 0, 0, 0),
        (1, 1, 0, 1),
    }

    @qm.qkernel
    def inverse() -> qm.Shot[qm.Bits[4]]:
        state = selected(qm.qubits(4))
        return qm.measure(qm.adjoint(selected, state))

    assert qm.run(inverse, on=qm.Reference(), shots=8, seed=7).result.counts == {
        (0, 0, 0, 0): 8
    }


@pytest.mark.parametrize(
    "width,indices,error",
    [
        (0, (0,), ValueError),
        (129, (0,), ValueError),
        (True, (0,), TypeError),
        (4, (), ValueError),
        (4, (0, 0), ValueError),
        (4, (-1,), ValueError),
        (4, (4,), ValueError),
        (4, (True,), TypeError),
        (4, [0, 2], TypeError),
    ],
)
def test_selection_rejects_invalid_configuration(width, indices, error):
    """Host topology is bounded before native source construction."""
    with pytest.raises(error):
        on_qubits(flip_selected, width=width, indices=indices)


def test_selection_checks_helper_abi():
    """A selected helper must preserve the selected coherent Bits shape."""
    with pytest.raises(TypeError, match="selected Bits"):
        on_qubits(bell_selected, width=4, indices=(0,))


def test_reference_selection_shared_regressions():
    """Reference executes the same shared selection sources as each SDK."""
    exercise_qubit_selection_backend(qm.Reference())
