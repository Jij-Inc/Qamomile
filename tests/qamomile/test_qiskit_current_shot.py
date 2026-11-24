"""Verify ordered Qiskit transport for logical invocation indices."""

from types import SimpleNamespace

import pytest

import qamomile as qm
from qamomile.qiskit.runtime import _runtime_indexed_counts


def test_indexed_batch_checks_declared_backend_limit() -> None:
    """Unsupported batch cardinality fails before provider submission."""
    from qamomile.qiskit._indexed import require_batch_capacity

    with pytest.raises(qm.ExecutionError, match="backend limit is 2"):
        require_batch_capacity(SimpleNamespace(max_circuits=2), 3)


def _unmeasured_backend_result(*, success=True, experiment_success=True, shots=1):
    """Build a real SDK result without classical measurement data."""
    from qiskit.result import Result

    return Result.from_dict(
        {
            "backend_name": "transport-witness",
            "backend_version": "1",
            "job_id": "local-witness",
            "success": success,
            "results": [{"success": experiment_success, "shots": shots, "data": {}}],
        }
    )


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"success": False}, "execution did not succeed"),
        ({"success": None}, "execution did not succeed"),
        ({"experiment_success": False}, "experiment did not succeed"),
        ({"experiment_success": None}, "experiment did not succeed"),
        ({"shots": 0}, "one shot"),
        ({"shots": 2}, "one shot"),
        ({"shots": True}, "one shot"),
    ],
)
def test_indexed_backend_rejects_invalid_unmeasured_experiment(
    fields: dict, message: str
) -> None:
    """No-classical-bit synthesis requires a successful one-shot SDK row."""
    from qiskit import QuantumCircuit

    from qamomile.qiskit._indexed import backend_indexed_counts

    with pytest.raises(qm.ExecutionError, match=message):
        backend_indexed_counts(
            _unmeasured_backend_result(**fields), [QuantumCircuit(1)]
        )


@pytest.mark.parametrize("experiment_count", [0, 2])
def test_indexed_backend_rejects_unmeasured_result_cardinality(
    experiment_count: int,
) -> None:
    """A successful SDK result must still contain each submitted circuit once."""
    from qiskit import QuantumCircuit

    from qamomile.qiskit._indexed import backend_indexed_counts

    result = _unmeasured_backend_result()
    result.results *= experiment_count
    with pytest.raises(qm.ExecutionError, match="result count"):
        backend_indexed_counts(result, [QuantumCircuit(1)])


def test_indexed_backend_accepts_successful_unmeasured_experiment() -> None:
    """Successful one-shot execution permits empty data for an unmeasured row."""
    from qiskit import QuantumCircuit

    from qamomile.qiskit._indexed import backend_indexed_counts

    assert backend_indexed_counts(
        _unmeasured_backend_result(),
        [QuantumCircuit(0), QuantumCircuit(1), QuantumCircuit(0)],
    ) == ({"": 1}, {"": 1}, {"": 1})


@pytest.mark.parametrize("fixed_template", [False, True])
def test_current_shot_backend_normalizes_missing_measured_counts(
    fixed_template: bool, monkeypatch
) -> None:
    """Missing measured SDK counts raise the public execution error on both routes."""
    from qiskit.exceptions import QiskitError
    from qiskit.providers.basic_provider import BasicSimulator

    from qamomile.qiskit import QiskitBackend
    from qamomile.qiskit._integration import _CAPABILITIES

    backend = BasicSimulator()
    endpoint = QiskitBackend(backend, requirements=_CAPABILITIES)
    conditional = (
        ""
        if fixed_template
        else "    if qm.get_current_shot() % 2 == 0:\n        state = qm.x(state)\n"
    )
    source = (
        "import qamomile as qm\n@qm.qkernel\n"
        "def sample() -> qm.Shot[qm.Bool]:\n"
        "    state = qm.prepare(False)\n"
        + conditional
        + "    measured = qm.measure(state)\n"
        "    return measured ^ (qm.get_current_shot() == 0)\n"
    )
    artifact = qm.compile(
        qm.build_source(source).specialize("sample"), to=endpoint.target
    )
    submissions = []

    def run(circuits, *, shots):
        """Return a successful real SDK result with missing measurement data."""
        submissions.append((circuits, shots))
        result = _unmeasured_backend_result()
        result.results *= len(circuits)
        return SimpleNamespace(
            job_id=lambda: "missing-counts", result=lambda timeout=None: result
        )

    monkeypatch.setattr(backend, "run", run)
    with pytest.raises(qm.ExecutionError, match="Q8233.*no sampling counts") as raised:
        qm.execute(artifact, on=endpoint, shots=2)
    assert isinstance(raised.value.__cause__, QiskitError)
    assert len(submissions) == 1
    assert len(submissions[0][0]) == 2
    assert all(circuit.clbits for circuit in submissions[0][0])
    assert submissions[0][1] == 1


_CHILD_SOURCE = """
import qamomile as qm

@qm.qkernel
def sample(flip: qm.Const[qm.Bool]) -> qm.Shot[qm.Bool]:
    bits = qm.measure(qm.qubits(1))
    return bits[0] ^ flip
"""


def _children(endpoint):
    """Compile two different native result evaluators for transport witnesses."""
    module = qm.build_source(_CHILD_SOURCE)
    return tuple(
        (
            qm.compile(module.specialize("sample", flip=flip), to=endpoint.target),
            qm.Invocation.create(shots=1, seed=None),
        )
        for flip in (False, True)
    )


@pytest.mark.parametrize(
    "raw_counts",
    [
        {"00": 2, "10": -1},
        {"00": 1.5, "10": -0.5},
        {"00": 0.5, "10": 0.5},
        {"00": True},
        {"00": "1"},
        {"00": 1 + 0j},
        [("00", 1)],
        None,
    ],
    ids=[
        "negative",
        "fractional-cancellation",
        "fractional",
        "boolean",
        "string",
        "complex",
        "sequence",
        "missing",
    ],
)
def test_indexed_result_rejects_malformed_provider_counts(raw_counts) -> None:
    """Invalid SDK counts cannot collapse into a valid one-shot logical result."""
    from qiskit import QuantumCircuit

    from qamomile.qiskit import Qiskit
    from qamomile.qiskit._indexed import indexed_result

    source = """import qamomile as qm
@qm.qkernel
def sample() -> qm.Shot[qm.Bool]:
    bits = qm.measure(qm.qubits(2))
    return bits[0] ^ False
"""
    artifact = qm.compile(qm.build_source(source).specialize("sample"), to=Qiskit())
    children = ((artifact, qm.Invocation.create(shots=1, seed=None)),)
    with pytest.raises(qm.ExecutionError, match="Q8233"):
        indexed_result(
            artifact,
            children,
            [QuantumCircuit(2, 2)],
            [raw_counts],
            backend="malformed-provider-witness",
        )


def test_indexed_result_accepts_zero_count_provider_entries() -> None:
    """Zero counts are valid nonnegative integers alongside one actual sample."""
    from qiskit import QuantumCircuit

    from qamomile.qiskit import Qiskit
    from qamomile.qiskit._indexed import indexed_result

    child, invocation = _children(SimpleNamespace(target=Qiskit()))[0]
    result = indexed_result(
        child,
        ((child, invocation),),
        [QuantumCircuit(1, 1)],
        [{"0": 1, "1": 0}],
        backend="valid-provider-witness",
    )
    assert result.values[False] == 1
    assert sum(result.values.values()) == 1


def _classified(rows: list[str], *, parameter_rows: bool = False):
    """Build classified register data without a provider submission."""
    from qiskit.primitives.containers import BitArray

    data = BitArray.from_samples(rows, num_bits=2)
    if parameter_rows:
        data = data.reshape(len(rows), 1)
    return SimpleNamespace(data=SimpleNamespace(meas=data))


@pytest.mark.parametrize("parameter_rows", [False, True])
def test_runtime_keeps_indexed_rows_separate(parameter_rows: bool) -> None:
    """Asymmetric rows remain in logical order until result evaluation."""
    rows = ["10", "01", "11"]
    results = (
        [_classified(rows, parameter_rows=True)]
        if parameter_rows
        else [_classified([row]) for row in rows]
    )
    decoded = _runtime_indexed_counts(
        results,
        bit_maps=[(("meas", 0), ("meas", 1))] * len(rows),
        parameter_rows=parameter_rows,
    )
    assert decoded == ({"10": 1}, {"01": 1}, {"11": 1})


def test_runtime_rejects_missing_indexed_result() -> None:
    """Missing logical results cannot silently shorten an invocation."""
    with pytest.raises(ValueError, match="result count"):
        _runtime_indexed_counts(
            [_classified(["00"])],
            bit_maps=[(("meas", 0),)] * 2,
        )


def test_runtime_rejects_many_shots_for_one_logical_index() -> None:
    """One-shot logical coordinates cannot contain an unlabeled population."""
    with pytest.raises(ValueError, match="one shot"):
        _runtime_indexed_counts(
            [_classified(["00", "01"])],
            bit_maps=[(("meas", 0),)],
        )


def test_runtime_rejects_flattened_parameter_rows() -> None:
    """Flattened counts do not prove parameter-row correspondence."""
    with pytest.raises(ValueError, match="shape"):
        _runtime_indexed_counts(
            [_classified(["00", "01"])],
            bit_maps=[(("meas", 0),)] * 2,
            parameter_rows=True,
        )


@pytest.mark.parametrize("parameter_rows", [False, True])
@pytest.mark.parametrize("completed_shots", [0, 1, 2, None, True, 1.0])
def test_runtime_unmeasured_execution_requires_completed_one_shot(
    parameter_rows: bool, completed_shots: object, monkeypatch
) -> None:
    """Both compiled routes reject missing completion evidence for quantum work."""
    from qiskit.primitives.containers import DataBin
    from qiskit.providers.basic_provider import BasicSimulator

    import qamomile.qiskit.runtime as runtime
    from qamomile.qiskit import IBMRuntime
    from qamomile.qiskit._integration import _CAPABILITIES

    submissions = []

    class Sampler:
        """Return real empty SDK data containers with controlled completion counts."""

        def __init__(self, *, mode, options):
            self.mode = mode

        def backend(self):
            return self.mode

        def run(self, pubs, *, shots):
            submissions.append((pubs, shots))
            shape = (2,) if parameter_rows else ()
            result = [
                SimpleNamespace(
                    data=DataBin(shape=shape), metadata={"shots": completed_shots}
                )
                for _ in pubs
            ]
            return SimpleNamespace(
                job_id=lambda: "unmeasured-completion",
                result=lambda: result,
                status=lambda: "DONE",
            )

    monkeypatch.setattr(
        runtime, "_runtime_module", lambda: SimpleNamespace(SamplerV2=Sampler)
    )
    endpoint = IBMRuntime(BasicSimulator(), requirements=_CAPABILITIES)
    conditional = (
        "    if qm.get_current_shot() % 2 == 0:\n        state = qm.x(state)\n"
        if not parameter_rows
        else ""
    )
    source = (
        "import qamomile as qm\n@qm.qkernel\n"
        "def sample() -> qm.Shot[qm.Bool]:\n"
        "    state = qm.prepare(False)\n" + conditional + "    qm.discard(state)\n"
        "    return qm.get_current_shot() == 0\n"
    )
    artifact = qm.compile(
        qm.build_source(source).specialize("sample"), to=endpoint.target
    )
    if type(completed_shots) is int and completed_shots == 1:
        assert qm.execute(artifact, on=endpoint, shots=2).values == {True: 1, False: 1}
    else:
        with pytest.raises(qm.ExecutionError, match="Q8233.*one shot"):
            qm.execute(artifact, on=endpoint, shots=2)
    assert len(submissions) == 1
    assert len(submissions[0][0]) == (1 if parameter_rows else 2)
    assert submissions[0][1] == 1


@pytest.mark.parametrize("parameter_rows", [False, True])
def test_runtime_rejects_unmeasured_pub_with_wrong_coordinate_shape(
    parameter_rows: bool,
) -> None:
    """One-shot metadata cannot certify missing or additional parameter positions."""
    from qiskit.primitives.containers import DataBin

    pub = SimpleNamespace(
        data=DataBin(shape=() if parameter_rows else (2,)), metadata={"shots": 1}
    )
    with pytest.raises(qm.ExecutionError, match="Q8233.*one shot"):
        _runtime_indexed_counts(
            [pub] if parameter_rows else [pub, pub],
            bit_maps=[(), ()],
            parameter_rows=parameter_rows,
        )


@pytest.mark.parametrize("endpoint_name", ["aer", "backend"])
def test_indexed_backend_batch_evaluates_children_before_merging(
    endpoint_name: str,
) -> None:
    """Identical measured rows still reach their distinct native result plans."""
    from qiskit.providers.basic_provider import BasicSimulator

    from qamomile.qiskit import Aer, QiskitBackend

    endpoint = Aer() if endpoint_name == "aer" else QiskitBackend(BasicSimulator())
    children = _children(endpoint)
    job = endpoint._launch_indexed(
        children[0][0], qm.Invocation.create(shots=2, seed=None), children
    )
    assert job.result().counts == {(0,): 1, (1,): 1}


def test_indexed_runtime_uses_one_submission_and_decodes_each_pub(monkeypatch) -> None:
    """One Runtime job preserves each child's result context."""
    from qiskit.providers.basic_provider import BasicSimulator

    import qamomile.qiskit.runtime as runtime
    from qamomile.qiskit import IBMRuntime

    submissions = []
    native_jobs = {}

    class Sampler:
        """Return measurement rows whose meaning depends on their PUB identity."""

        def __init__(self, *, mode, options):
            self.mode = mode

        def backend(self):
            return self.mode

        def run(self, pubs, *, shots):
            from qiskit.primitives.containers import BitArray

            submissions.append((pubs, shots))
            result = [
                SimpleNamespace(
                    data=SimpleNamespace(
                        **{
                            register.name: BitArray.from_samples(
                                [str(1 - index)], num_bits=len(register)
                            )
                            for register in circuit.cregs
                        }
                    )
                )
                for index, circuit in enumerate(pubs)
            ]
            job = SimpleNamespace(job_id=lambda: "indexed-job", result=lambda: result)
            native_jobs["indexed-job"] = job
            return job

    monkeypatch.setattr(
        runtime, "_runtime_module", lambda: SimpleNamespace(SamplerV2=Sampler)
    )
    endpoint = IBMRuntime(
        BasicSimulator(),
        service=SimpleNamespace(job=lambda identity: native_jobs[identity]),
    )
    children = _children(endpoint)
    job = endpoint._launch_indexed(
        children[0][0], qm.Invocation.create(shots=2, seed=None), children
    )
    assert len(submissions) == 1
    assert len(submissions[0][0]) == 2
    assert submissions[0][1] == 1
    snapshot = qm.ExecutionSnapshot.from_bytes(job.snapshot().to_bytes())
    context = snapshot.references()[0].context_values
    assert context["kind"] == "indexed_sample"
    assert len(context["bits_digest"]) == 64
    assert "bits" not in context
    import qamomile.compilation as compilation

    monkeypatch.setattr(
        compilation,
        "_current_shot_children",
        lambda artifact, endpoint, invocation: children,
        raising=False,
    )
    restored = endpoint.restore(snapshot, artifact=children[0][0])
    assert restored.snapshot().to_bytes() == snapshot.to_bytes()
    assert restored.result().counts == {(1,): 2}
    assert len(submissions) == 1
    assert job.result().counts == {(1,): 2}


@pytest.mark.parametrize("endpoint_name", ["aer", "backend"])
def test_current_shot_public_qiskit_lifecycle(endpoint_name: str) -> None:
    """One compiled artifact restarts logical numbering for each invocation."""
    from qiskit.providers.basic_provider import BasicSimulator

    from qamomile.qiskit import Aer, QiskitBackend

    endpoint = Aer() if endpoint_name == "aer" else QiskitBackend(BasicSimulator())
    program = qm.build_source("""
import qamomile as qm

@qm.qkernel
def sample() -> qm.Shot[qm.Bool]:
    index = qm.get_current_shot()
    bits = qm.measure(qm.qubits(1))
    return bits[0] ^ (index % 2 == 1)
""").specialize("sample")
    artifact = qm.compile(program, to=endpoint.target)
    assert not artifact.parameter_abi.slots
    assert qm.execute(artifact, on=endpoint, shots=3).counts == {(0,): 2, (1,): 1}
    assert qm.execute(artifact, on=endpoint, shots=2).counts == {(0,): 1, (1,): 1}


def test_current_shot_aer_without_measurements() -> None:
    """Pure contextual results preserve invocation shot cardinality."""
    from qamomile.qiskit import Aer

    endpoint = Aer()
    program = qm.build_source("""
import qamomile as qm

@qm.qkernel
def sample() -> qm.Shot[qm.Bool]:
    return qm.get_current_shot() == 0
""").specialize("sample")
    artifact = qm.compile(program, to=endpoint.target)
    assert qm.execute(artifact, on=endpoint, shots=3).counts == {(0,): 2, (1,): 1}
