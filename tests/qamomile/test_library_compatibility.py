"""Shared execution witnesses for zero-layer QAOA and approximate Fourier transforms."""

from __future__ import annotations

import pytest

import qamomile as qm
from qamomile.compilation import ExecutionEndpoint
from qamomile.optimization import BinaryModel, QAOAExperiment
from qamomile.stdlib import approximate_qft, qft


@qm.qkernel
def approximate_component(
    distance: qm.Const[qm.Int],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[5]]:  # type: ignore[valid-type]
    """Expose a bounded operator for checking cutoff semantics independently."""
    return approximate_qft(qm.qubits(5), 5, distance)


@qm.qkernel
def exact_component() -> qm.Q[qm.Bits[5]]:  # type: ignore[valid-type]
    """Expose the existing exact transform for resource and operator comparison."""
    return qft(qm.qubits(5), 5)


@qm.qkernel
def approximate_prefix() -> qm.Q[qm.Bits[5]]:  # type: ignore[valid-type]
    """Transform a three-wire prefix without touching the trailing components."""
    return approximate_qft(qm.qubits(5), 3, 1)


@qm.qkernel
def approximate_roundtrip_state() -> qm.Q[qm.Bits[5]]:  # type: ignore[valid-type]
    """Restore an asymmetric register through an approximate prefix and its inverse."""
    state = qm.prepare((1, 0, 1, 1, 0))
    state = approximate_qft(state, 3, 1)
    state = qm.adjoint(approximate_qft, state, 3, 1)
    return state


@qm.qkernel
def approximate_roundtrip() -> qm.Shot[qm.Bits[5]]:  # type: ignore[valid-type]
    """Sample the restored register in canonical source order."""
    return qm.measure(approximate_roundtrip_state())


@qm.qkernel
def approximate_value() -> qm.Shot[qm.Bool]:  # type: ignore[valid-type]
    """Expose the same sampled state through a scalar result projection."""
    bits = approximate_roundtrip()
    return bool(bits[3])


def exercise_approximate_qft_backend(
    endpoint: ExecutionEndpoint, *, seed: int = 7
) -> None:
    """Compile and execute the shared Fourier witness on a local endpoint."""
    program = qm.Program.from_bytes(approximate_roundtrip.specialize().to_bytes())
    artifact = qm.compile(program, to=endpoint.target)
    result = qm.execute(artifact, on=endpoint, shots=8, seed=seed)
    assert isinstance(result, qm.SampleResult)
    assert result.counts == {(1, 0, 1, 1, 0): 8}
    value = qm.run(approximate_value, on=endpoint, shots=8, seed=seed).result
    assert isinstance(value, qm.SampleResult)
    assert value.values == {True: 8}
    estimate = qm.expectation(
        approximate_roundtrip_state, qm.Z(3), on=endpoint, shots=8, seed=seed
    )
    assert estimate.value == -1.0


def exercise_zero_layer_qaoa_backend(
    endpoint: ExecutionEndpoint, *, seed: int = 7
) -> None:
    """Execute the same initial-state program and retain model and invocation identity."""
    model = BinaryModel.from_hubo({(4,): -1.0, (4, 9): 0.5})
    experiment = QAOAExperiment(model, layers=0)
    compiled = experiment.compile(to=endpoint.target)
    run = compiled.run(on=endpoint, gammas=(), betas=(), shots=128, seed=seed)
    result = run.execution.result
    assert isinstance(result, qm.SampleResult)
    assert set(result.counts) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    for count in result.counts.values():
        assert abs(count / 128 - 0.25) < 0.15
    assert run.execution.artifact is compiled.artifact
    assert run.invocation.values() == {}
    assert not compiled.artifact.parameter_abi.slots
    assert run.identity == experiment.identity
    assert run.samples.total_occurrences == 128
    assert {
        index for sample in run.samples.samples for index, _ in sample.assignment
    } == {4, 9}


def test_reference_library_compatibility() -> None:
    """Both library paths retain their source-derived semantics on Reference."""
    exercise_zero_layer_qaoa_backend(qm.Reference())
    exercise_approximate_qft_backend(qm.Reference())


@pytest.mark.parametrize("distance", [-1, 0, 1, 2, 4, 6])
def test_approximate_qft_matches_independent_sdk_operator(distance: int) -> None:
    """A trusted SDK transform independently validates every retained phase and swap."""
    from qiskit.quantum_info import Operator
    from qiskit.synthesis import synth_qft_full

    program = approximate_component.specialize(distance=distance)
    circuit = qm.qiskit.to_circuit(program)
    degree = max(0, 4 - max(0, distance))
    expected = Operator(synth_qft_full(5, approximation_degree=degree)).data
    assert Operator(circuit).data == pytest.approx(expected, abs=1e-12)


def test_approximate_qft_preserves_prefix_and_exact_limit() -> None:
    """The cutoff reduces work, while a full cutoff recovers the existing exact QFT."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from qiskit.synthesis import synth_qft_full

    prefix = QuantumCircuit(5)
    prefix.append(synth_qft_full(3, approximation_degree=1), [0, 1, 2])
    assert Operator(qm.qiskit.to_circuit(approximate_prefix)).data == pytest.approx(
        Operator(prefix).data, abs=1e-12
    )
    exact = exact_component.specialize()
    full = approximate_component.specialize(distance=4)
    assert Operator(qm.qiskit.to_circuit(full)).data == pytest.approx(
        Operator(qm.qiskit.to_circuit(exact)).data, abs=1e-12
    )
    metric = qm.resources.logical_operation_count("quantum.cp")
    assert qm.resources.logical(exact).metric(metric).numeric_enclosure() == (10, 10)
    assert qm.resources.logical(approximate_component.specialize(distance=2)).metric(
        metric
    ).numeric_enclosure() == (7, 7)


@pytest.mark.parametrize("distance", [True, 0.5])
def test_approximate_qft_rejects_noninteger_cutoffs(distance: object) -> None:
    """The cutoff crosses the existing checked Const integer boundary."""
    with pytest.raises(qm.SpecializationError, match="Q3006.*distance.*Int"):
        approximate_component.specialize(distance=distance)


@pytest.mark.parametrize("name", ["gammas", "betas"])
def test_zero_layer_qaoa_rejects_nonempty_coordinates(name: str) -> None:
    """Zero-layer coordinates must still satisfy the experiment's fixed topology."""
    experiment = QAOAExperiment(BinaryModel.from_hubo({(0,): -1.0}), layers=0)
    values: dict[str, tuple[float, ...]] = {"gammas": (), "betas": ()}
    values[name] = (0.2,)
    with pytest.raises(ValueError, match=f"{name}.*exactly 0"):
        experiment.run(
            on=qm.Reference(), shots=1, gammas=values["gammas"], betas=values["betas"]
        )


def test_zero_layer_qaoa_resources_and_direct_launch() -> None:
    """The initial state has no angle ABI, cost evolution, or mixer rotations."""
    experiment = QAOAExperiment(BinaryModel.from_hubo({(0,): -1.0}), layers=0)
    study = qm.resources.logical(experiment.program)
    for opcode in ["quantum.rx", "quantum.pauli_evolution"]:
        assert (
            qm.resources.logical_operation_count(opcode).identifier not in study.values
        )
    assert study.metric(
        qm.resources.logical_operation_count("quantum.h")
    ).numeric_enclosure() == (1, 1)
    run = experiment.run(on=qm.Reference(), gammas=[], betas=[], shots=8, seed=7)
    assert run.samples.total_occurrences == 8
