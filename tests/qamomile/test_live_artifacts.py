"""Compilation retains executables; persistence is an explicit separate action."""

from __future__ import annotations

import importlib
import math

import pytest

import qamomile as qm
from qamomile._execution_payload import ExecutablePayload
from qamomile.persistence import deserialize, serialize

_SOURCE = """
import qamomile as qm
@qm.qkernel
def sample(theta: qm.Param[qm.Real]) -> qm.Bits[1]:
    state = qm.qubits(1)
    state = qm.rx(state, theta)
    return qm.measure(state)
"""

_BACKENDS = [
    ("qamomile", "Reference", None),
    ("qamomile.qiskit", "Aer", "qiskit_aer"),
    pytest.param("qamomile.cirq", "CirqSimulator", "cirq", marks=pytest.mark.cirq),
    pytest.param(
        "qamomile.quri_parts", "Qulacs", "qulacs", marks=pytest.mark.quri_parts
    ),
    pytest.param("qamomile.cudaq", "CudaqLocal", "cudaq", marks=pytest.mark.cudaq),
    pytest.param("qamomile.braket", "BraketLocal", "braket", marks=pytest.mark.braket),
    pytest.param("qamomile.hugr", "Selene", "selene_sim", marks=pytest.mark.hugr),
    pytest.param(
        "qamomile.quration", "QurationLocal", "pyqret", marks=pytest.mark.quration
    ),
]


@pytest.mark.parametrize(("module", "endpoint_name", "dependency"), _BACKENDS)
def test_repeated_execution_needs_no_artifact_serialization(
    monkeypatch, module, endpoint_name, dependency
):
    """Compile and rebind without export, restore, or shared quantum state."""
    if dependency:
        pytest.importorskip(dependency)
    endpoint = getattr(importlib.import_module(module), endpoint_name)()
    static = endpoint_name == "QurationLocal"
    source = (
        _SOURCE.replace("qm.Param[qm.Real]", "qm.Const[qm.Real]") if static else _SOURCE
    )
    program = qm.build_source(source).specialize(
        "sample", **({"theta": math.pi} if static else {})
    )

    def forbid(*args, **kwargs):
        raise AssertionError(
            "Ordinary compilation/execution must not persist artifacts"
        )

    with monkeypatch.context() as patch:
        patch.setattr("qamomile.persistence.serialize", forbid)
        patch.setattr("qamomile.persistence.deserialize", forbid)
        if endpoint_name == "Aer":
            from qiskit import qpy

            patch.setattr(qpy, "dump", forbid)
            patch.setattr(qpy, "load", forbid)
        elif endpoint_name == "Qulacs":
            from qamomile.circuit import CircuitProgram

            patch.setattr(CircuitProgram, "to_bytes", forbid)
        elif endpoint_name == "CirqSimulator":
            import cirq

            patch.setattr(cirq, "to_json", forbid)
            patch.setattr(cirq, "read_json", forbid)
        elif endpoint_name == "QurationLocal":
            from pyqret.frontend import Context, Module

            patch.setattr(Module, "dumps", forbid)
            patch.setattr(Context, "loads", forbid)
        artifact = qm.compile(program, to=endpoint.target)
        assert isinstance(artifact.payload, ExecutablePayload)
        patch.setattr(type(artifact.payload), "to_bytes", forbid, raising=False)
        for theta in (0.0, math.pi, 0.0):
            result = qm.execute(
                artifact,
                on=endpoint,
                parameters={} if static else {"theta": theta},
                shots=4,
                seed=0,
            )
            assert result.counts == {(int(static or theta != 0.0),): 4}
    # Persistence remains available, but is never invoked by the above launches.
    restored = deserialize(serialize(artifact))
    assert qm.execute(
        restored,
        on=endpoint,
        parameters={} if static else {"theta": math.pi},
        shots=4,
        seed=0,
    ).counts == {(1,): 4}


def test_host_family_keeps_child_objects(monkeypatch):
    """Repeated host dispatch selects retained children without byte envelopes."""
    pytest.importorskip("qiskit_aer")
    from qamomile.compilation import _ArtifactFamilyPayload
    from qamomile.qiskit import Aer

    source = """
import qamomile as qm
@qm.qkernel
def sample(enabled: qm.Param[qm.Bool]) -> qm.Bits[1]:
    state = qm.qubits(1)
    if enabled:
        state = qm.x(state)
    return qm.measure(state)
"""
    endpoint = Aer()
    program = qm.build_source(source).specialize("sample")

    def forbid(*args, **kwargs):
        raise AssertionError("Host selection must not serialize child artifacts")

    monkeypatch.setattr("qamomile.persistence.serialize", forbid)
    monkeypatch.setattr("qamomile.persistence.deserialize", forbid)
    artifact = qm.compile(program, to=endpoint.target)
    assert isinstance(artifact.payload, _ArtifactFamilyPayload)
    children = tuple(variant.artifact for variant in artifact.payload.variants)
    for enabled in (True, False, True):
        assert qm.execute(
            artifact, on=endpoint, parameters={"enabled": enabled}, shots=4
        ).counts == {(int(enabled),): 4}
    assert tuple(variant.artifact for variant in artifact.payload.variants) == children


@pytest.mark.cudaq
def test_cudaq_defines_one_kernel_for_repeated_launches(monkeypatch):
    """The generated module and helper owner graph are not recreated per shot run."""
    pytest.importorskip("cudaq")
    from unittest.mock import Mock

    from qamomile.cudaq import CudaqLocal, _integration

    load = Mock(wraps=_integration._load_source)
    monkeypatch.setattr(_integration, "_load_source", load)
    endpoint = CudaqLocal()
    artifact = qm.compile(
        qm.build_source(_SOURCE).specialize("sample"), to=endpoint.target
    )
    for theta in (math.pi, 0.0, math.pi):
        assert qm.execute(
            artifact, on=endpoint, parameters={"theta": theta}, shots=4
        ).counts == {(int(theta != 0),): 4}
    assert load.call_count == 1


@pytest.mark.hugr
def test_selene_builds_once_and_retains_build_directory(monkeypatch):
    """Repeated shots reuse one executable while the owning artifact is alive."""
    selene = pytest.importorskip("selene_sim")
    from unittest.mock import Mock

    from qamomile.hugr import Selene

    build = Mock(wraps=selene.build)
    monkeypatch.setattr(selene, "build", build)
    source = _SOURCE.replace("qm.Param[qm.Real]", "qm.Const[qm.Real]")
    endpoint = Selene(timeout=30)
    artifact = qm.compile(
        qm.build_source(source).specialize("sample", theta=math.pi), to=endpoint.target
    )
    for _ in range(2):
        assert qm.execute(artifact, on=endpoint, shots=4).counts == {(1,): 4}
    assert build.call_count == 1
    directory = build.call_args.kwargs["build_dir"]
    assert directory.is_dir()
    import gc
    import weakref

    owner = weakref.ref(artifact.payload)
    del artifact
    gc.collect()
    assert owner() is None
    assert not directory.exists()


@pytest.mark.quri_parts
def test_quri_parts_binds_nonlinear_controlled_angles_without_rebuilding_constants():
    """Typed deferred expressions retain controlled phase across changed values."""
    pytest.importorskip("qulacs")
    from qamomile.quri_parts import Qulacs

    source = """
import qamomile as qm
@qm.qkernel
def phase(state: qm.Q[qm.Bits[1]], theta: qm.Param[qm.Real]) -> qm.Q[qm.Bits[1]]:
    return qm.global_phase(state, theta * theta)
@qm.qkernel
def sample(theta: qm.Param[qm.Real]) -> qm.Bits[2]:
    control, target = qm.split_at(qm.qubits(2), 1)
    control = qm.h(control)
    control, target = qm.control(phase, control, target, theta)
    control = qm.h(control)
    return qm.measure(qm.concat(control, target))
"""
    endpoint = Qulacs()
    artifact = qm.compile(
        qm.build_source(source).specialize("sample"), to=endpoint.target
    )
    for candidate in (artifact, deserialize(serialize(artifact))):
        for theta in (0.0, math.sqrt(math.pi), 0.0):
            assert qm.execute(
                candidate, on=endpoint, parameters={"theta": theta}, shots=4
            ).counts == {(int(theta != 0), 0): 4}


def test_qiskit_retains_native_addition_and_isolates_public_circuit_mutation():
    """Storage never dictates arithmetic lowering or lets binding mutate templates."""
    pytest.importorskip("qiskit_aer")
    from qiskit.circuit.library import ModularAdderGate

    from qamomile.qiskit import Aer, load_circuit

    source = """
import qamomile as qm
@qm.qkernel
def sample() -> qm.Bits[4]:
    state = qm.prepare((1, 0, 1, 0))
    state = qm.add_into(state)
    return qm.measure(state)
"""
    endpoint = Aer(realizations=qm.RealizationOptions({qm.add_into: "add.native"}))
    artifact = qm.compile(
        qm.build_source(source).specialize("sample"), to=endpoint.target
    )
    circuit = load_circuit(artifact)
    assert any(isinstance(row.operation, ModularAdderGate) for row in circuit.data)
    circuit.x(0)
    for candidate in (artifact, deserialize(serialize(artifact))):
        assert qm.execute(candidate, on=endpoint, shots=4).counts == {(1, 0, 0, 1): 4}


def test_reference_restores_a_native_handle_only_once(monkeypatch):
    """Live and imported artifacts each own one verified native program handle."""
    from unittest.mock import Mock

    from qamomile._native import compiler as native

    constructor = Mock(wraps=native.ReferenceProgramHandle)
    monkeypatch.setattr(native, "ReferenceProgramHandle", constructor)
    endpoint = qm.Reference()
    artifact = qm.compile(qm.build_source(_SOURCE).specialize("sample"), to=endpoint)
    assert constructor.call_count == 1
    for _ in range(2):
        assert qm.execute(
            artifact, on=endpoint, parameters={"theta": math.pi}, shots=4
        ).counts == {(1,): 4}
    assert constructor.call_count == 1
    restored = deserialize(serialize(artifact))
    assert constructor.call_count == 2
    for _ in range(2):
        assert qm.execute(
            restored, on=endpoint, parameters={"theta": 0.0}, shots=4
        ).counts == {(0,): 4}
    assert constructor.call_count == 2


def test_launch_recipe_reuses_only_its_most_recent_materialization(monkeypatch):
    """An unbounded parameter domain cannot grow an unbounded executable cache."""
    pytest.importorskip("qiskit_aer")
    from qamomile.compilation import _LaunchMaterializerPayload
    from qamomile.qiskit import Aer, Qiskit

    source = """
import qamomile as qm
@qm.qkernel
def sample(rounds: qm.Param[qm.Int]) -> qm.Bits[1]:
    state = qm.qubits(1)
    for i in range(rounds):
        state = qm.x(state)
    return qm.measure(state)
"""
    endpoint = Aer()
    artifact = qm.compile(
        qm.build_source(source).specialize("sample"), to=endpoint.target
    )
    assert isinstance(artifact.payload, _LaunchMaterializerPayload)
    original = Qiskit.lower_circuit
    calls = []

    def lower(target, logical):
        calls.append(logical)
        return original(target, logical)

    monkeypatch.setattr(Qiskit, "lower_circuit", lower)
    for rounds in (1, 1, 2, 2, 1):
        assert qm.execute(
            artifact, on=endpoint, parameters={"rounds": rounds}, shots=4
        ).counts == {(rounds % 2,): 4}
    assert len(calls) == 3


@pytest.mark.hugr
@pytest.mark.parametrize(
    ("operation", "target"),
    [
        ("state = qm.x(state, target=target)", -1),
        ("state = qm.x(state, target=target)", 2),
        ("state = qm.cx(state, control=0, target=target)", 0),
        ("bit, state = qm.measure_z(state, target=target)", -1),
        ("bit, state = qm.measure_z(state, target=target)", 2),
    ],
)
def test_hugr_checks_launch_selectors_without_projecting_a_throwaway_circuit(
    monkeypatch, operation, target
):
    """Direct SDK emission retains Rust's selector bounds and alias diagnostics."""
    pytest.importorskip("selene_sim")
    import qamomile.circuit
    from qamomile.hugr import Selene

    source = f"""
import qamomile as qm
@qm.qkernel
def sample(target: qm.Param[qm.Int]) -> qm.Bits[2]:
    state = qm.qubits(2)
    {operation}
    return qm.measure(state)
"""

    def forbid(*args, **kwargs):
        raise AssertionError("HUGR must not project a validation-only circuit")

    monkeypatch.setattr(qamomile.circuit, "lower_to_circuit_at", forbid)
    endpoint = Selene()
    artifact = qm.compile(
        qm.build_source(source).specialize("sample"), to=endpoint.target
    )
    with pytest.raises(qm.ExecutionError, match="Q4303"):
        qm.execute(artifact, on=endpoint, parameters={"target": target})


@pytest.mark.quri_parts
def test_quri_parts_deferred_expression_failure_has_stable_diagnostic():
    """An invalid bound angle fails before entering the simulator."""
    pytest.importorskip("qulacs")
    from qamomile.quri_parts import Qulacs

    source = _SOURCE.replace("qm.rx(state, theta)", "qm.rx(state, 1.0 / theta)")
    endpoint = Qulacs()
    artifact = qm.compile(
        qm.build_source(source).specialize("sample"), to=endpoint.target
    )
    with pytest.raises(qm.ExecutionError, match="Q8711"):
        qm.execute(artifact, on=endpoint, parameters={"theta": 0.0})


@pytest.mark.hugr
@pytest.mark.parametrize("index", [-1, 2])
def test_hugr_checks_launch_control_component(index):
    """A dynamic control component uses the same native selector diagnostics."""
    pytest.importorskip("selene_sim")
    from qamomile.hugr import Selene

    source = """
import qamomile as qm
@qm.qkernel
def flip(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(state)
@qm.qkernel
def sample(index: qm.Param[qm.Int]) -> qm.Bits[3]:
    controls, target = qm.split_at(qm.qubits(3), 2)
    controls, target = qm.control(flip, controls, target, at=index)
    return qm.measure(qm.concat(controls, target))
"""
    endpoint = Selene()
    artifact = qm.compile(
        qm.build_source(source).specialize("sample"), to=endpoint.target
    )
    with pytest.raises(qm.ExecutionError, match="Q4303"):
        qm.execute(artifact, on=endpoint, parameters={"index": index})


def test_execution_contract_requires_no_persistence(monkeypatch):
    """A backend without codecs can compile and run with no saved-data path."""
    import pickle

    from qamomile._native import compiler as native
    from qamomile.compilation import _ReferenceExecutable

    monkeypatch.delattr(_ReferenceExecutable, "to_bytes")
    monkeypatch.delattr(_ReferenceExecutable, "from_bytes")

    def forbidden(*args, **kwargs):
        raise AssertionError("Execution attempted artifact persistence")

    monkeypatch.setattr(native, "seal_artifact", forbidden)
    monkeypatch.setattr(native, "inspect_artifact", forbidden)
    compiled = qm.compile(
        qm.build_source(_SOURCE).specialize("sample"), to=qm.Reference()
    )
    assert isinstance(compiled, qm.CompiledProgram)
    assert not hasattr(qm, "Artifact")
    assert not hasattr(compiled, "to_bytes")
    assert not hasattr(compiled, "payload_bytes")
    assert len(compiled.execution_identity) == 64
    for theta in (0.0, math.pi, 0.0):
        assert qm.execute(
            compiled, on=qm.Reference(), parameters={"theta": theta}, shots=4
        ).counts == {(int(theta != 0),): 4}
    with pytest.raises(TypeError, match="process-local"):
        pickle.dumps(compiled)
    with pytest.raises(TypeError, match="does not support persistence"):
        serialize(compiled)


def test_execution_identity_excludes_storage_and_checks_contract_changes():
    """Equivalent source/profile retains identity across compilation and saving."""
    from dataclasses import replace

    from qamomile.ir import FrozenMapping

    program = qm.build_source(_SOURCE).specialize("sample")
    left = qm.compile(program, to=qm.Reference())
    right = qm.compile(program, to=qm.Reference())
    assert left.execution_identity == right.execution_identity
    assert deserialize(serialize(left)).execution_identity == left.execution_identity
    assert (
        replace(left, target_fingerprint="different").execution_identity
        != left.execution_identity
    )
    assert (
        replace(
            left, metadata=FrozenMapping.from_mapping({"changed": True})
        ).execution_identity
        != left.execution_identity
    )
