"""Tests for the artifact-target and execution-endpoint split."""

from __future__ import annotations

import copy
import importlib
import json
import math
import pickle
from dataclasses import dataclass, replace
from types import MappingProxyType, SimpleNamespace
from unittest.mock import patch

import pytest

import qamomile as qm
import qamomile.compilation as compilation_module
from qamomile._execution_payload import ExecutablePayload
from qamomile._native import compiler as native
from qamomile.compilation import (
    CapabilitySnapshot,
    CompletedJob,
    Invocation,
    LoweredProgram,
    ParameterABI,
    RealizationCapability,
    ResultABI,
)
from qamomile.effects import Requirement
from qamomile.ir import Program
from qamomile.persistence import deserialize, export_payload, serialize

from . import conftest as qamomile_test_config
from .conftest import (
    BASIS_SOURCE,
    BELL_SOURCE,
    exercise_broadcast_gate_backend,
    exercise_uint_measurement_backend,
)


class _ProbeExecutable(ExecutablePayload):
    """Tiny optional codec used only by the target-protocol witnesses."""

    def __init__(self, payload):
        self.payload = payload

    def to_bytes(self):
        return self.payload


def _load_probe(format, payload):
    """Restore an explicitly supplied test backend codec."""
    assert format.startswith("probe.")
    return _ProbeExecutable(payload)


_HOST_DISPATCH_SOURCE = """
import qamomile as qm
from qamomile._execution_payload import ExecutablePayload

@qm.qkernel

def sample(
    first: qm.Param[qm.Bool],
    theta: qm.Param[qm.Real],
    limit: qm.Param[qm.Int],
) -> qm.Bits[1]:
    state = qm.qubits(1)
    if first:
        state = qm.x(state)
    if limit > 0:
        state = qm.x(state)
    state = qm.rz(state, theta)
    return qm.measure(state)
"""

_DERIVED_RESULT_DISPATCH_SOURCE = """
import qamomile as qm
from qamomile._execution_payload import ExecutablePayload

@qm.qkernel
def sample(enabled: qm.Param[qm.Bool]) -> qm.Shot[qm.Bool]:
    state = qm.qubits(1)
    if enabled:
        state = qm.x(state)
    bit, state = qm.measure_z(state, 0)
    qm.discard(state)
    return not bit
"""


def test_run_exposes_both_artifact_and_result() -> None:
    """Convenience execution must not hide compilation provenance."""

    program = qm.build_source(BELL_SOURCE).specialize("bell")
    record = qm.run(program, on=qm.Reference(), shots=32, seed=1)

    assert record.artifact.format == "qamomile.qcore.program"
    assert export_payload(record.artifact) == native.optimize_standard_program(
        program.to_bytes()
    )
    assert record.artifact.report.stages[0] == "native-lowering-planning"
    assert set(record.result.counts) == {(0, 0), (1, 1)}
    observed = record.resources.metric(qm.resources.OBSERVED_SHOTS)
    assert observed.numeric_enclosure() == (32, 32)
    assert observed.metric.coordinate is qm.resources.Coordinate.OBSERVED


def test_reference_measures_unsigned_integer_registers() -> None:
    """Reference execution exposes UInt inputs and sampled values as integers."""

    exercise_uint_measurement_backend(endpoint=qm.Reference(), seed=2)


def test_reference_broadcasts_unselected_unary_gates() -> None:
    """Reference applies an omitted unary selector to every logical lane."""

    exercise_broadcast_gate_backend(endpoint=qm.Reference(), seed=2)


def test_compile_is_the_only_public_target_compilation_entrypoint() -> None:
    """Intermediate planning and emission objects are not public lifecycle APIs."""

    assert callable(compilation_module.compile)
    assert not hasattr(compilation_module, "plan")
    assert not hasattr(compilation_module, "emit")
    assert not hasattr(compilation_module, "CompilationPlan")


def test_compile_reports_only_the_direct_targets_actual_stages() -> None:
    """A target retaining source QCore must not claim standard optimization."""

    class PassthroughTarget:
        """Retain a verified source without running an optimizer."""

        name = "probe.passthrough.v1"
        fingerprint = "probe-passthrough-v1"
        capabilities = CapabilitySnapshot(name, frozenset(Requirement))

        def lower(self, program: Program) -> LoweredProgram:
            """Retain the original bytes and record only their transport."""

            return LoweredProgram(
                payload=_ProbeExecutable(program.to_bytes()),
                parameter_abi=ParameterABI.from_program(program),
                result_abi=ResultABI.from_program(program),
                stages=("passthrough",),
            )

    program = qm.build_source(BELL_SOURCE).specialize("bell")
    artifact = qm.compile(program, to=PassthroughTarget())

    assert export_payload(artifact) == program.to_bytes()
    assert artifact.report.stages == ("native-lowering-planning", "passthrough")
    restored = deserialize(serialize(artifact), payload_loader=_load_probe)
    assert restored.report.stages == artifact.report.stages


@pytest.mark.parametrize(
    ("limit_name", "limit", "shots", "message"),
    [
        ("_MAX_BACKEND_TEST_QUBITS", 1, 1, "limited to 1 qubits; got 2"),
        ("_MAX_FAST_BACKEND_TEST_GATE_DEPTH", 0, 1, "limited to gate depth 0"),
        (
            "_MAX_FAST_BACKEND_TEST_EMITTED_INSTRUCTIONS",
            0,
            1,
            "bounded shallow example",
        ),
        ("_MAX_FAST_BACKEND_TEST_SHOTS", 1, 2, "limited to 1 shots; got 2"),
    ],
)
def test_reference_execution_obeys_shared_workload_bounds(
    monkeypatch: pytest.MonkeyPatch,
    limit_name: str,
    limit: int,
    shots: int,
    message: str,
) -> None:
    """Every simulator launch fails before work exceeds the test budget."""

    monkeypatch.setattr(qamomile_test_config, limit_name, limit)
    program = qm.build_source(
        BELL_SOURCE,
        module_name=f"bounded_reference_{limit_name.lower()}",
    ).specialize("bell")

    with pytest.raises(AssertionError, match=message):
        qm.run(program, on=qm.Reference(), shots=shots, seed=0)


def test_execute_never_calls_a_compiler() -> None:
    """An endpoint receives only an existing artifact and invocation."""

    program = qm.build_source(BELL_SOURCE).specialize("bell")
    artifact = qm.compile(program, to=qm.Reference())

    @dataclass
    class ForbiddenTarget:
        name: str = "forbidden"

        @property
        def fingerprint(self) -> str:
            return qm.Reference().fingerprint

        def lower(self, _program: Program) -> qm.CompiledProgram:
            raise AssertionError("execution attempted to lower")

    class Endpoint:
        name = "spy"
        target = ForbiddenTarget()
        capabilities = CapabilitySnapshot(
            "qamomile.qcore.program",
            frozenset(Requirement),
        )

        def launch(
            self,
            received: qm.CompiledProgram,
            invocation: Invocation,
        ) -> qm.Job:
            assert received is artifact
            return CompletedJob(
                "spy-job",
                qm.SampleResult.from_counts(
                    backend=self.name,
                    counts={(0, 0): invocation.shots},
                ),
            )

    result = qm.execute(artifact, on=Endpoint(), shots=3)
    assert result.counts == {(0, 0): 3}


def test_run_reuses_an_artifact_and_records_the_native_bound_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compile-once execution keeps canonical values and skips lowering."""

    source = """
import qamomile as qm
from qamomile._execution_payload import ExecutablePayload

@qm.qkernel
def choose(
    values: qm.Param[qm.Vector[qm.Int, 2]],
    enabled: qm.Param[qm.Bool] = True,
) -> qm.Int:
    return values[0] if enabled else values[1]
"""
    endpoint = qm.Reference()
    program = qm.build_source(source, module_name="recorded_invocation").specialize(
        "choose"
    )
    artifact = qm.compile(program, to=endpoint.target)

    def forbidden_compile(*_args: object, **_kwargs: object) -> object:
        """Fail if an existing artifact re-enters compilation."""

        raise AssertionError("artifact execution attempted to compile")

    monkeypatch.setattr(compilation_module, "compile", forbidden_compile)
    record = qm.run(
        artifact,
        on=endpoint,
        parameters={"values": [3, 5]},
    )

    assert record.artifact is artifact
    assert record.invocation.values() == {"values": (3, 5), "enabled": True}
    assert record.result.value == 3
    with pytest.raises(TypeError, match="process-local"):
        pickle.dumps(record)
    restored = deserialize(serialize(record.artifact))
    assert serialize(restored) == serialize(record.artifact)


def test_compile_analyzes_logical_resources_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Target emission must not repeat or own semantic resource analysis."""

    program = qm.build_source(BELL_SOURCE).specialize("bell")
    original = qm.resources.logical
    calls = 0

    def counted(source: object, *args: object, **kwargs: object) -> object:
        """Count calls while delegating to the ordinary logical analysis."""

        nonlocal calls
        calls += 1
        return original(source, *args, **kwargs)

    monkeypatch.setattr(qm.resources, "logical", counted)

    artifact = qm.compile(program, to=qm.Reference())

    assert calls == 1
    assert artifact.report.resources.semantic_digest == program.semantic_digest


def test_compile_and_submit_dispatch_a_portable_static_artifact_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The core selects sealed backend children without recompiling at launch."""

    class StaticProbeTarget:
        """Emit inspectable circuit JSON while rejecting direct QCore lowering."""

        name = "probe.static-circuit.v1"
        fingerprint = "probe-static-circuit-v1"
        capabilities = CapabilitySnapshot(
            name,
            frozenset(Requirement) - {Requirement.HOST_DISPATCH},
        )

        def __init__(self) -> None:
            """Initialize emitted payload evidence."""

            self.payloads: list[bytes] = []

        def lower(self, _program: Program) -> LoweredProgram:
            """Reject whole-program lowering for this static-only target."""

            raise AssertionError("artifact family called direct target lowering")

        def lower_circuit(self, circuit: object) -> LoweredProgram:
            """Emit one static circuit through the target-author protocol."""

            from qamomile.circuit import CircuitProgram

            assert isinstance(circuit, CircuitProgram)
            payload = circuit.to_bytes()
            self.payloads.append(payload)
            return LoweredProgram(
                payload=_ProbeExecutable(payload),
                parameter_abi=circuit.parameter_abi,
                result_abi=circuit.result_abi,
                stages=("probe-static-emission",),
                circuit_qubits=circuit.width,
                circuit_gates=circuit.operation_count,
                emitted_instructions=circuit.operation_count,
            )

    target = StaticProbeTarget()
    program = qm.build_source(
        _HOST_DISPATCH_SOURCE,
        module_name="portable_static_artifact_family",
    ).specialize("sample")
    artifact = qm.compile(program, to=target)
    restored = deserialize(serialize(artifact), payload_loader=_load_probe)

    assert artifact.format == "qamomile.artifact-family.v3"
    assert [slot.name for slot in artifact.parameter_abi.slots] == [
        "first",
        "theta",
        "limit",
    ]
    assert len(target.payloads) == 4
    assert "host-dispatch-artifact-family" in artifact.report.stages
    assert artifact.report.stages.count("rust-qcore-standard-optimization") == 1

    class Endpoint:
        """Inspect the selected child before returning one deterministic shot."""

        name = "probe"

        def __init__(self, target: StaticProbeTarget) -> None:
            """Retain the exact static target used during compilation."""

            self._target = target

        @property
        def capabilities(self) -> CapabilitySnapshot:
            """Return the selected target's execution capabilities."""

            return self._target.capabilities

        @property
        def target(self) -> StaticProbeTarget:
            """Return the exact target used for family compilation."""

            return self._target

        def launch(
            self,
            child: qm.CompiledProgram,
            invocation: Invocation,
        ) -> qm.Job:
            """Return whether the selected static child contains an X gate."""

            assert child.report.stages.count("rust-qcore-standard-optimization") == 1
            assert child.format == self._target.name
            assert [slot.name for slot in child.parameter_abi.slots] == ["theta"]
            assert [name for name, _ in invocation.parameters.entries] == ["theta"]
            opcodes = [
                instruction["op"]
                for instruction in json.loads(export_payload(child))["body"]
            ]
            bit = opcodes.count("x") % 2
            return CompletedJob(
                f"probe-{bit}",
                qm.SampleResult.from_counts(
                    backend=self.name,
                    counts={(bit,): invocation.shots},
                ),
            )

    endpoint = Endpoint(target)
    for first, limit, expected in (
        (False, -3, 0),
        (False, 2, 1),
        (True, -3, 1),
        (True, 2, 0),
    ):
        result = qm.execute(
            restored,
            on=endpoint,
            parameters={"first": first, "theta": 0.25, "limit": limit},
            shots=3,
        )
        assert result.counts == {(expected,): 3}
    with pytest.raises(qm.ArtifactError, match="Q7001"):
        qm.execute(
            replace(restored, payload=b"not-a-family"),
            on=endpoint,
            parameters={"first": False, "theta": 0.25, "limit": 0},
        )
    import qamomile.compilation as compilation_module
    from qamomile.persistence import (
        _decode_artifact_family_payload,
        _encode_artifact_family_payload,
    )
    from qamomile.proto import qcore_pb2 as qcore_wire
    from qamomile.serialization import encode_literal

    decoded = _decode_artifact_family_payload(
        export_payload(restored), payload_loader=_load_probe
    )
    legacy = qcore_wire.LiteralProto()
    encode_literal(
        (
            "qamomile.artifact-family",
            3,
            decoded.target_format,
            decoded.dispatch_parameters,
            decoded.selection_plan,
            tuple(
                (variant.decisions, serialize(variant.artifact))
                for variant in decoded.variants
            ),
        ),
        legacy,
    )
    assert export_payload(restored) == legacy.SerializeToString(deterministic=True)
    with pytest.raises(qm.ArtifactError, match="Q7045"):
        _encode_artifact_family_payload(
            replace(decoded, selection_plan=b"not-a-selector")
        )
    reordered = replace(
        decoded,
        variants=(decoded.variants[1], decoded.variants[0], *decoded.variants[2:]),
    )
    with pytest.raises(qm.ArtifactError, match="Q7045"):
        _encode_artifact_family_payload(reordered)
    with monkeypatch.context() as patch:
        patch.delattr(compilation_module.native, "seal_artifact_family_payload")
        with pytest.raises(qm.ArtifactError, match="Q7045"):
            _encode_artifact_family_payload(decoded)
    with monkeypatch.context() as patch:
        patch.delattr(compilation_module.native, "inspect_artifact_family_payload")
        with pytest.raises(qm.ExecutionError, match="Q8130"):
            _decode_artifact_family_payload(
                export_payload(restored), payload_loader=_load_probe
            )


def test_static_target_must_opt_in_to_result_postprocessing() -> None:
    """A circuit target cannot silently discard measurement-derived results."""

    class UnsafeStaticTarget:
        """Expose the static-circuit protocol without result processing."""

        name = "probe.no-result-processing.v1"
        fingerprint = "probe-no-result-processing-v1"
        capabilities = CapabilitySnapshot(
            name,
            frozenset(Requirement) - {Requirement.HOST_DISPATCH},
        )

        def lower(self, _program: Program) -> LoweredProgram:
            """Reject direct QCore lowering."""

            raise AssertionError("derived result used direct target lowering")

        def lower_circuit(self, _circuit: object) -> LoweredProgram:
            """Fail if an unsafe circuit is delivered to the emitter."""

            raise AssertionError("derived result reached an unsafe emitter")

    program = qm.build_source(
        _DERIVED_RESULT_DISPATCH_SOURCE,
        module_name="derived_result_dispatch",
    ).specialize("sample")

    with pytest.raises(qm.ArtifactError, match="Q7048"):
        qm.compile(program, to=UnsafeStaticTarget())

    class SafeStaticTarget(UnsafeStaticTarget):
        """A target that explicitly preserves result expressions."""

        supports_circuit_result_postprocessing = True

        def lower_circuit(self, circuit: object) -> LoweredProgram:
            """Emit the complete canonical circuit projection."""

            from qamomile.circuit import CircuitProgram

            assert isinstance(circuit, CircuitProgram)
            assert circuit.requires_result_postprocessing
            return LoweredProgram(
                payload=_ProbeExecutable(circuit.to_bytes()),
                parameter_abi=circuit.parameter_abi,
                result_abi=circuit.result_abi,
                stages=("result-aware-static-emission",),
            )

    artifact = qm.compile(program, to=SafeStaticTarget())
    assert artifact.format == "qamomile.artifact-family.v3"


def test_sdk_targets_advertise_result_postprocessing() -> None:
    """Every static SDK target opts in only after preserving result trees."""

    from qamomile.braket import Braket
    from qamomile.cudaq import Cudaq
    from qamomile.qiskit import Qiskit
    from qamomile.quri_parts import QuriParts

    for target in (Qiskit(), Braket(), Cudaq(), QuriParts()):
        assert target.supports_circuit_result_postprocessing is True


def test_direct_circuit_target_cannot_change_native_result_abi() -> None:
    """Direct circuit emission rejects a changed source-bit order."""

    from qamomile.circuit import CircuitProgram, lower_circuit_target

    class BadEmitter:
        """Return a valid payload with an incompatible result ABI."""

        capabilities = CapabilitySnapshot("probe.circuit.v1", frozenset(Requirement))

        def lower_circuit(self, circuit: CircuitProgram) -> LoweredProgram:
            """Change only the native-derived result ordering."""

            return LoweredProgram(
                payload=_ProbeExecutable(b"probe"),
                parameter_abi=circuit.parameter_abi,
                result_abi=replace(
                    circuit.result_abi,
                    source_order=tuple(reversed(circuit.result_abi.source_order)),
                ),
                stages=("probe-emission",),
            )

    program = qm.build_source(BASIS_SOURCE).specialize("basis")
    with pytest.raises(qm.ArtifactError, match="Q7044"):
        lower_circuit_target(program, BadEmitter())  # type: ignore[arg-type]


def test_compile_and_submit_materializes_launch_dependent_loops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sealed program becomes one backend circuit after launch values exist."""

    source = """
import qamomile as qm
from qamomile._execution_payload import ExecutablePayload

@qm.qkernel
def repeat(rounds: qm.Param[qm.Int], theta: qm.Param[qm.Real]) -> qm.Bits[1]:
    state = qm.qubits(1)
    for _ in range(rounds):
        state = qm.x(state)
    state = qm.rz(state, theta)
    return qm.measure(state)
"""

    class StaticProbeTarget:
        """Record circuits emitted only after exact loop bounds are known."""

        name = "probe.launch-circuit.v1"
        fingerprint = "probe-launch-circuit-v1"
        capabilities = CapabilitySnapshot(
            name,
            frozenset(Requirement) - {Requirement.HOST_DISPATCH},
        )

        def __init__(self) -> None:
            """Initialize launch materialization evidence."""

            self.payloads: list[bytes] = []

        def lower(self, _program: Program) -> LoweredProgram:
            """Reject direct whole-program lowering."""

            raise AssertionError("launch materializer called direct target lowering")

        def lower_circuit(self, circuit: object) -> LoweredProgram:
            """Emit one invocation-specific static circuit."""

            from qamomile.circuit import CircuitProgram

            assert isinstance(circuit, CircuitProgram)
            payload = circuit.to_bytes()
            self.payloads.append(payload)
            return LoweredProgram(
                payload=_ProbeExecutable(payload),
                parameter_abi=circuit.parameter_abi,
                result_abi=circuit.result_abi,
                stages=("probe-launch-emission",),
                circuit_qubits=circuit.width,
                circuit_gates=circuit.operation_count,
                emitted_instructions=circuit.operation_count,
            )

    class Endpoint:
        """Inspect each invocation-specific static child."""

        name = "probe-launch"

        def __init__(self, target: StaticProbeTarget) -> None:
            """Retain the target bound into the outer artifact."""

            self._target = target

        @property
        def capabilities(self) -> CapabilitySnapshot:
            """Return static child execution capabilities."""

            return self._target.capabilities

        @property
        def target(self) -> StaticProbeTarget:
            """Return the exact materialization target."""

            return self._target

        def launch(
            self,
            child: qm.CompiledProgram,
            invocation: Invocation,
        ) -> qm.Job:
            """Return the parity encoded by the emitted X sequence."""

            assert child.report.stages.count("rust-qcore-standard-optimization") == 1
            assert child.format == self._target.name
            assert [slot.name for slot in child.parameter_abi.slots] == ["theta"]
            assert [name for name, _ in invocation.parameters.entries] == ["theta"]
            opcodes = [
                instruction["op"]
                for instruction in json.loads(export_payload(child))["body"]
            ]
            bit = opcodes.count("x") % 2
            return CompletedJob(
                f"launch-{bit}",
                qm.SampleResult.from_counts(
                    backend=self.name,
                    counts={(bit,): invocation.shots},
                ),
            )

    target = StaticProbeTarget()
    program = qm.build_source(
        source,
        module_name="portable_launch_loop_materializer",
    ).specialize("repeat")
    artifact = qm.compile(program, to=target)
    restored = deserialize(serialize(artifact), payload_loader=_load_probe)

    assert artifact.format == "qamomile.launch-materializer.v1"
    assert target.payloads == []
    assert "retained-launch-materializer" in artifact.report.stages

    import qamomile.compilation as compilation_module
    from qamomile.compilation import _decode_launch_materializer_program
    from qamomile.proto import qcore_pb2 as qcore_wire
    from qamomile.serialization import encode_literal

    legacy = qcore_wire.LiteralProto()
    encode_literal(
        ("qamomile.launch-materializer", 1, target.name, program.to_bytes()),
        legacy,
    )
    assert export_payload(artifact) == legacy.SerializeToString(deterministic=True)

    endpoint = Endpoint(target)
    even = qm.execute(
        restored,
        on=endpoint,
        parameters={"rounds": 2, "theta": 0.25},
        shots=3,
    )
    odd = qm.execute(
        restored,
        on=endpoint,
        parameters={"rounds": 3, "theta": 0.25},
        shots=2,
    )

    assert even.counts == {(0,): 3}
    assert odd.counts == {(1,): 2}
    assert len(target.payloads) == 2
    with pytest.raises(qm.ExecutionError, match="Q8133"):
        _decode_launch_materializer_program(
            restored,
            target_format="probe.other.v1",
            materialized_requirements=frozenset({Requirement.HOST_DISPATCH}),
        )
    with pytest.raises(qm.ExecutionError, match="Q8133"):
        _decode_launch_materializer_program(
            replace(restored, parameter_abi=ParameterABI()),
            target_format=target.name,
            materialized_requirements=frozenset({Requirement.HOST_DISPATCH}),
        )
    with pytest.raises(qm.ArtifactError, match="Q7001"):
        qm.execute(
            replace(restored, payload=b"not-a-materializer"),
            on=endpoint,
            parameters={"rounds": 1, "theta": 0.25},
        )
    with pytest.raises(qm.ExecutionError, match="Q8134.*1000001 iterations"):
        qm.execute(
            restored,
            on=endpoint,
            parameters={"rounds": 1_000_001, "theta": 0.25},
        )
    with monkeypatch.context() as patch:
        patch.delattr(compilation_module.native, "seal_launch_materializer_payload")
        with pytest.raises(qm.ArtifactError, match="Q7047"):
            serialize(qm.compile(program, to=target))
    with monkeypatch.context() as patch:
        patch.delattr(compilation_module.native, "inspect_launch_materializer_artifact")
        assert (
            _decode_launch_materializer_program(
                restored,
                target_format=target.name,
                materialized_requirements=frozenset({Requirement.HOST_DISPATCH}),
            ).semantic_digest
            == program.semantic_digest
        )


def test_structured_target_materializes_exact_launch_abi_without_a_circuit() -> None:
    """A QCore target can close Param and host control in one explicit step."""

    class StructuredTarget:
        """Emit an invocation-bound payload directly from structured QCore."""

        name = "probe.structured-launch.v1"
        fingerprint = "probe-structured-launch-v1"
        capabilities = CapabilitySnapshot(
            name,
            frozenset(Requirement)
            - {
                Requirement.HOST_DISPATCH,
                Requirement.PARAMETERIZED_EXECUTION,
            },
        )
        launch_materialized_requirements = frozenset(
            {
                Requirement.HOST_DISPATCH,
                Requirement.PARAMETERIZED_EXECUTION,
            }
        )

        def __init__(self) -> None:
            """Initialize exact lowering evidence."""

            self.lowered: list[dict[str, object]] = []

        def lower(self, _program: Program) -> LoweredProgram:
            """Reject unresolved direct lowering."""

            raise AssertionError("structured target bypassed launch materialization")

        def lower_at(
            self,
            program: Program,
            parameters: dict[str, object],
        ) -> LoweredProgram:
            """Emit one argument-free structured child artifact."""

            values = dict(parameters)
            self.lowered.append(values)
            return LoweredProgram(
                payload=_ProbeExecutable(json.dumps(values, sort_keys=True).encode()),
                parameter_abi=ParameterABI(),
                result_abi=ResultABI.from_program(program),
                stages=("probe-structured-launch-emission",),
            )

    class Endpoint:
        """Execute the sealed structured child without launch parameters."""

        name = "probe-structured-launch"

        def __init__(self, target: StructuredTarget) -> None:
            """Retain the target identity used during compilation."""

            self._target = target

        @property
        def target(self) -> StructuredTarget:
            """Return the exact structured target."""

            return self._target

        @property
        def capabilities(self) -> CapabilitySnapshot:
            """Return capabilities required by the closed child only."""

            return self._target.capabilities

        def launch(
            self,
            child: qm.CompiledProgram,
            invocation: Invocation,
        ) -> qm.Job:
            """Return the bound branch value encoded by the child."""

            assert "rust-qcore-standard-optimization" not in child.report.stages
            assert child.parameter_abi == ParameterABI()
            assert invocation.parameters.entries == ()
            values = json.loads(export_payload(child))
            bit = int(values["first"]) ^ int(values["limit"] > 0)
            return CompletedJob(
                "structured-launch",
                qm.SampleResult.from_counts(
                    backend=self.name,
                    counts={(bit,): invocation.shots},
                ),
            )

    target = StructuredTarget()
    program = qm.build_source(
        _HOST_DISPATCH_SOURCE,
        module_name="structured_launch_materializer",
    ).specialize("sample")
    artifact = qm.compile(program, to=target)

    assert artifact.format == "qamomile.launch-materializer.v1"
    assert target.lowered == []
    result = qm.execute(
        artifact,
        on=Endpoint(target),
        parameters={"first": True, "theta": 0.25, "limit": 0},
        shots=3,
    )

    assert result.counts == {(1,): 3}
    assert target.lowered == [{"first": True, "theta": 0.25, "limit": 0}]


def test_launch_target_cannot_claim_to_discharge_quantum_requirements() -> None:
    """Exact host values close staging requirements, not device semantics."""

    class InvalidTarget:
        """Advertise an invalid attempt to materialize quantum execution."""

        name = "probe.invalid-launch.v1"
        fingerprint = "probe-invalid-launch-v1"
        capabilities = CapabilitySnapshot(
            name,
            frozenset(Requirement) - {Requirement.QUANTUM},
        )
        launch_materialized_requirements = frozenset({Requirement.QUANTUM})

        def lower(self, _program: Program) -> LoweredProgram:
            """Reject emission because planning must fail first."""

            raise AssertionError("invalid launch target reached emission")

        def lower_at(
            self,
            _program: Program,
            _parameters: dict[str, object],
        ) -> LoweredProgram:
            """Reject materialization because planning must fail first."""

            raise AssertionError("invalid launch target reached materialization")

    program = qm.build_source(BELL_SOURCE).specialize("bell")

    with pytest.raises(qm.ArtifactError, match="Q7046"):
        qm.compile(program, to=InvalidTarget())


def test_parameter_abi_rejects_unknown_launch_values() -> None:
    """Launch values must match the names sealed into the artifact ABI."""

    program = qm.build_source(BELL_SOURCE).specialize("bell")
    artifact = qm.compile(program, to=qm.Reference())
    with pytest.raises(qm.ExecutionError, match="unknown artifact parameters"):
        qm.execute(artifact, on=qm.Reference(), parameters={"ghost": 1})


def test_parameter_abi_binding_is_native_canonical_and_structural() -> None:
    """Rust must own defaults, scalar kinds, and fixed aggregate shapes."""

    source = """
import qamomile as qm
from qamomile._execution_payload import ExecutablePayload

@qm.qkernel
def choose(
    values: qm.Param[qm.Vector[qm.Int, 2]],
    enabled: qm.Param[qm.Bool] = True,
) -> qm.Int:
    return values[0] if enabled else values[1]
"""
    program = qm.build_source(source, module_name="native_invocation_abi").specialize(
        "choose"
    )
    abi = ParameterABI.from_program(program)

    canonical = {"values": (3, 5)}
    assert abi.bind(canonical) == (
        ("values", (3, 5)),
        ("enabled", True),
    )
    assert abi.bind({"values": [3, 5]}) == abi.bind(canonical)
    with pytest.raises(qm.ExecutionError, match="Q8109.*values") as mismatch:
        abi.bind({"values": [3]})
    assert mismatch.value.diagnostic.code == "Q8109"
    with pytest.raises(qm.ExecutionError, match="Q8109.*enabled"):
        abi.bind({"values": [3, 5], "enabled": 1})


def test_parameter_abi_rejects_nonportable_values_before_native_binding() -> None:
    """Invocation values must enter Rust through QCore's portable domain."""

    program = qm.build_source(BELL_SOURCE).specialize("bell")
    abi = ParameterABI.from_program(program)

    with pytest.raises(qm.ExecutionError, match="Q8135.*not portable"):
        abi.bind({"ghost": object()})


def test_reference_executes_values_already_validated_by_the_artifact_abi() -> None:
    """Reference execution must not repeat the Python runtime type contract."""

    source = """
import qamomile as qm
from qamomile._execution_payload import ExecutablePayload

@qm.qkernel
def choose(
    values: qm.Param[qm.Vector[qm.Int, 2]],
    enabled: qm.Param[qm.Bool] = True,
) -> qm.Int:
    return values[0] if enabled else values[1]
"""
    program = qm.build_source(source, module_name="bound_reference").specialize(
        "choose"
    )
    artifact = qm.compile(program, to=qm.Reference())

    result = qm.execute(
        artifact,
        on=qm.Reference(),
        parameters={"values": [3, 5]},
    )

    assert result.value == 3


def test_reference_native_value_restores_public_record_results() -> None:
    """Native portable mappings retain the source record field contract."""

    source = """
from typing import NamedTuple
import qamomile as qm
from qamomile._execution_payload import ExecutablePayload

class Summary(NamedTuple):
    total: qm.Int
    outcome: tuple[qm.Bool, qm.Int]

@qm.qkernel
def summarize(total: qm.Param[qm.Int]) -> Summary:
    return Summary(outcome=(True, 1), total=total)
"""
    program = qm.build_source(source, module_name="native_reference_record").specialize(
        "summarize"
    )

    result = qm.run(
        program,
        on=qm.Reference(),
        parameters={"total": 7},
    ).result

    assert isinstance(result.value, MappingProxyType)
    assert list(result.value.items()) == [("total", 7), ("outcome", (True, 1))]


def test_reference_prefers_native_sampling_for_supported_static_circuits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Supported sampled artifacts enter the native runtime directly."""

    from qamomile import _qamomile_native as native

    program = qm.build_source(
        """
import qamomile as qm
from qamomile._execution_payload import ExecutablePayload

@qm.qkernel
def sample(theta: qm.Param[qm.Real]) -> qm.Shot[qm.Bits[1]]:
    return qm.measure(qm.rx(qm.qubits(1), theta))
""",
        module_name="native_reference_sampling",
    ).specialize("sample")
    executor = native.execute_reference_at
    calls: list[tuple[dict[str, object], int, int]] = []

    def recording_executor(
        payload: bytes,
        parameters: dict[str, object],
        shots: int,
        seed: int,
    ) -> object:
        """Record and delegate one native Reference invocation."""

        calls.append((parameters, shots, seed))
        return executor(payload, parameters, shots, seed)

    monkeypatch.setattr(native, "execute_reference_at", recording_executor)

    result = qm.run(
        program,
        on=qm.Reference(),
        parameters={"theta": 3.141592653589793},
        shots=4,
        seed=17,
    ).result

    assert result.counts == {(1,): 4}
    assert calls == [({"theta": 3.141592653589793}, 4, 17)]


def test_execute_batch_reuses_one_restored_reference_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reference batches preserve row order and cross the native bridge once."""

    from qamomile import _qamomile_native as native

    program = qm.build_source(
        """
import qamomile as qm
from qamomile._execution_payload import ExecutablePayload

@qm.qkernel
def sample(enabled: qm.Param[qm.Bool]) -> qm.Shot[qm.Bits[1]]:
    state = qm.qubits(1)
    if enabled:
        state = qm.x(state)
    return qm.measure(state)
""",
        module_name="native_reference_batch",
    ).specialize("sample")
    artifact = deserialize(serialize(qm.compile(program, to=qm.Reference())))
    executor = native.execute_reference_batch_at
    calls: list[tuple[list[dict[str, object]], int, int]] = []

    def recording_executor(
        payload: bytes,
        parameters: list[dict[str, object]],
        shots: int,
        seed: int,
    ) -> object:
        """Record and delegate one native Reference batch."""

        calls.append((parameters, shots, seed))
        return executor(payload, parameters, shots, seed)

    monkeypatch.setattr(native, "execute_reference_batch_at", recording_executor)
    results = qm.execute_batch(
        artifact,
        on=qm.Reference(),
        parameters=[{"enabled": False}, {"enabled": True}, {"enabled": False}],
        shots=4,
        seed=11,
    )

    assert tuple(result.counts for result in results) == (
        {(0,): 4},
        {(1,): 4},
        {(0,): 4},
    )
    assert calls == [
        (
            [{"enabled": False}, {"enabled": True}, {"enabled": False}],
            4,
            11,
        )
    ]


@pytest.mark.parametrize(
    ("sdk", "module_name", "endpoint_name", "rows"),
    [
        ("qiskit_aer", "qamomile.qiskit", "Aer", 2),
        ("cirq", "qamomile.cirq", "CirqSimulator", 2),
        ("qulacs", "qamomile.quri_parts", "Qulacs", 2),
        ("braket", "qamomile.braket", "BraketLocal", 1),
    ],
)
def test_local_sdk_batches_execute_without_workers(
    sdk: str,
    module_name: str,
    endpoint_name: str,
    rows: int,
) -> None:
    """Run local SDK artifacts without workers under each seed contract."""

    pytest.importorskip(sdk)
    endpoint_type = getattr(importlib.import_module(module_name), endpoint_name)
    endpoint = endpoint_type()
    program = qm.build_source(BASIS_SOURCE).specialize("basis")
    artifact = deserialize(serialize(qm.compile(program, to=endpoint.target)))
    with patch(
        "subprocess.run",
        side_effect=AssertionError("worker used"),
    ):
        results = qm.execute_batch(
            artifact,
            on=endpoint,
            parameters=[{} for _ in range(rows)],
            shots=4,
            seed=0,
        )

    assert tuple(result.counts for result in results) == ({(1, 0, 0): 4},) * rows
    assert all(result.worker_pid is None for result in results)


def test_execute_batch_uses_endpoint_batch_capability() -> None:
    """Any structurally batch-capable endpoint can launch the batch once."""

    from qamomile.compilation import BatchExecutionEndpoint, Job

    reference = qm.Reference()

    class RecordingBatchEndpoint:
        """Delegate Reference execution while recording batch dispatch."""

        def __init__(self) -> None:
            """Initialize an empty dispatch log."""

            self.batches: list[tuple[qm.Invocation, ...]] = []

        @property
        def name(self) -> str:
            """Return the delegated endpoint name."""

            return reference.name

        @property
        def target(self) -> qm.Reference:
            """Return the delegated compilation target."""

            return reference

        @property
        def capabilities(self):
            """Return the delegated runtime capabilities."""

            return reference.capabilities

        def launch(
            self,
            artifact: qm.CompiledProgram,
            invocation: qm.Invocation,
        ) -> Job:
            """Fail if the batch path falls back to scalar launch."""

            del artifact, invocation
            raise AssertionError("scalar launch must not be used")

        def launch_batch(
            self,
            artifact: qm.CompiledProgram,
            invocations: tuple[qm.Invocation, ...],
        ) -> tuple[Job, ...]:
            """Record and delegate one normalized invocation batch."""

            self.batches.append(invocations)
            return reference.launch_batch(artifact, invocations)

    program = qm.build_source(
        """
import qamomile as qm
from qamomile._execution_payload import ExecutablePayload

@qm.qkernel
def choose(value: qm.Param[qm.Int]) -> qm.Int:
    return value
""",
        module_name="protocol_batch_endpoint",
    ).specialize("choose")
    endpoint = RecordingBatchEndpoint()
    assert isinstance(endpoint, BatchExecutionEndpoint)
    artifact = qm.compile(program, to=endpoint.target)

    results = qm.execute_batch(
        artifact,
        on=endpoint,
        parameters=[{"value": 7}, {"value": -3}],
        seed=5,
    )

    assert tuple(result.value for result in results) == (7, -3)
    assert len(endpoint.batches) == 1
    assert tuple(invocation.seed for invocation in endpoint.batches[0]) == (5, 6)


def test_execute_batch_supports_empty_and_value_batches() -> None:
    """Reference batch execution handles empty input and typed values."""

    program = qm.build_source(
        """
import qamomile as qm
from qamomile._execution_payload import ExecutablePayload

@qm.qkernel
def choose(value: qm.Param[qm.Int]) -> qm.Int:
    return value
""",
        module_name="native_reference_value_batch",
    ).specialize("choose")
    endpoint = qm.Reference()
    artifact = qm.compile(program, to=endpoint)

    assert qm.execute_batch(artifact, on=endpoint, parameters=[]) == ()
    results = qm.execute_batch(
        artifact,
        on=endpoint,
        parameters=[{"value": 7}, {"value": -3}],
        seed=5,
    )
    assert tuple(result.value for result in results) == (7, -3)


def test_execute_batch_fails_closed_on_limits_and_native_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Oversized or malformed Reference batches fail before unchecked use."""

    from qamomile import _qamomile_native as native

    program = qm.build_source(
        """
import qamomile as qm
from qamomile._execution_payload import ExecutablePayload

@qm.qkernel
def choose(value: qm.Param[qm.Int]) -> qm.Int:
    return value
""",
        module_name="invalid_native_reference_batch",
    ).specialize("choose")
    artifact = qm.compile(program, to=qm.Reference())

    with pytest.raises(ValueError, match="at most 4096"):
        qm.execute_batch(
            artifact,
            on=qm.Reference(),
            parameters=[{"value": 0}] * 4097,
        )

    monkeypatch.delattr(native, "execute_reference_batch_at")
    with pytest.raises(qm.ExecutionError, match="Q8145.*unavailable"):
        qm.execute_batch(
            artifact,
            on=qm.Reference(),
            parameters=[{"value": 1}],
        )


def test_evaluate_uses_the_native_reference_runtime_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one-shot convenience API shares native binding and execution."""

    from qamomile import _qamomile_native as native

    program = qm.build_source(
        """
import qamomile as qm
from qamomile._execution_payload import ExecutablePayload

@qm.qkernel
def sample(enabled: qm.Param[qm.Bool]) -> qm.Shot[qm.Bits[1]]:
    state = qm.qubits(1)
    if enabled:
        state = qm.x(state)
    return qm.measure(state)
""",
        module_name="native_evaluate_lifecycle",
    ).specialize("sample")
    executor = native.execute_reference_at
    calls: list[tuple[dict[str, object], int, int]] = []

    def recording_executor(
        payload: bytes,
        parameters: dict[str, object],
        shots: int,
        seed: int,
    ) -> object:
        """Record and delegate the native one-shot invocation.

        Args:
            payload (bytes): Canonical verified program payload.
            parameters (dict[str, object]): ABI-bound launch parameters.
            shots (int): Requested observation count.
            seed (int): Deterministic simulation seed.

        Returns:
            object: Native sampler result forwarded unchanged.
        """

        calls.append((parameters, shots, seed))
        return executor(payload, parameters, shots, seed)

    monkeypatch.setattr(native, "execute_reference_at", recording_executor)

    assert qm.evaluate(program, enabled=True, seed=23) == (1,)
    assert calls == [({"enabled": True}, 1, 23)]


def test_evaluate_samples_measurement_derived_booleans_natively() -> None:
    """Shot-time Boolean expressions remain in the native scalar DAG."""

    program = qm.build_source(
        """
import qamomile as qm
from qamomile._execution_payload import ExecutablePayload

@qm.qkernel
def sample() -> tuple[qm.Shot[qm.Bool], qm.Shot[qm.Bool]]:
    bit = qm.measure(qm.h(qm.prepare(True)))
    return not bit, bit
""",
        module_name="native_measurement_result_expression",
    ).specialize("sample")

    negated, measured = qm.evaluate(program, seed=3)

    assert (negated, measured) in {(False, True), (True, False)}


def test_evaluate_selects_branch_local_measurements_natively() -> None:
    """A measurement ``if`` lazily evaluates only its selected scalar result."""

    program = qm.build_source(
        """
import qamomile as qm
from qamomile._execution_payload import ExecutablePayload

@qm.qkernel
def choose() -> tuple[qm.Shot[qm.Bool], qm.Shot[qm.Bool]]:
    selector = qm.measure(qm.h(qm.prepare(False)))
    if selector:
        selected = qm.measure(qm.prepare(True))
    else:
        selected = qm.measure(qm.prepare(False))
    return selector, selected
""",
        module_name="native_measurement_if_phi",
    ).specialize("choose")

    selector, selected = qm.evaluate(program, seed=11)

    assert selected is selector


def test_evaluate_composes_measurement_tail_returns_through_calls_natively() -> None:
    """Two measurement-selected returns merge before caller continuation."""

    module = qm.build_source(
        """
import qamomile as qm
from qamomile._execution_payload import ExecutablePayload

@qm.qkernel
def choose() -> qm.Shot[qm.Bool]:
    selector = qm.measure(qm.h(qm.prepare(False)))
    if selector:
        return qm.measure(qm.prepare(True))
    else:
        return qm.measure(qm.prepare(False))

@qm.qkernel
def caller() -> tuple[qm.Shot[qm.Bool], qm.Shot[qm.Bool]]:
    selected = choose()
    return selected, not selected
""",
        module_name="native_measurement_tail_return",
    )

    selected, negated = qm.evaluate(module.specialize("caller"), seed=11)

    assert bool(selected) is not bool(negated)


def test_evaluate_normalizes_measurement_early_return_continuation_natively() -> None:
    """A one-arm return moves its continuation into the fallthrough arm."""

    import qamomile._compiler.lowering as lowering

    assert not hasattr(lowering, "_FunctionLowerer")

    module = qm.build_source(
        """
import qamomile as qm
from qamomile._execution_payload import ExecutablePayload

@qm.qkernel
def choose() -> qm.Shot[qm.Bool]:
    selector = qm.measure(qm.h(qm.prepare(False)))
    if selector:
        return qm.measure(qm.prepare(True))
    return qm.measure(qm.prepare(False))

@qm.qkernel
def caller() -> tuple[qm.Shot[qm.Bool], qm.Shot[qm.Bool]]:
    selected = choose()
    return selected, not selected
""",
        module_name="native_measurement_early_return",
    )

    program = module.specialize("caller")
    selected, negated = qm.evaluate(program, seed=17)

    assert bool(selected) is not bool(negated)
    choose = next(
        function
        for function in program.functions
        if function.symbol.endswith(".choose")
    )
    branch = next(
        operation
        for operation in choose.body.operations
        if operation.opcode == "qcore.if"
    )
    assert choose.body.terminator.kind == "unreachable"
    assert [region.entry.terminator.kind for region in branch.regions] == [
        "return",
        "return",
    ]


@pytest.mark.parametrize("value", [1 << 4096, -(1 << 4096)])
def test_evaluate_preserves_unbounded_python_integers_natively(
    value: int,
) -> None:
    """The native value bridge does not narrow Python integers to ``i64``."""

    program = qm.build_source(
        """
import qamomile as qm
from qamomile._execution_payload import ExecutablePayload

@qm.qkernel
def successor(value: qm.Param[qm.Int]) -> qm.Int:
    state = qm.qubits(1)
    if value > 0:
        state = qm.x(state)
    limit = value + 1
    if value < limit:
        state = qm.x(state)
        value += 1
    qm.discard(state)
    return value
""",
        module_name="native_evaluate_bigint",
    ).specialize("successor")

    assert qm.evaluate(program, value=value) == value + 1


def test_reference_preserves_nonfinite_binary64_real_values() -> None:
    """QCore Real retains NaN and infinity until a restricted consumer."""

    source = """
import qamomile as qm
from qamomile._execution_payload import ExecutablePayload

@qm.qkernel
def identity(value: qm.Param[qm.Real]) -> qm.Real:
    return value
"""
    program = qm.build_source(source, module_name="reference_binary64_real").specialize(
        "identity"
    )

    assert math.isinf(qm.evaluate(program, value=math.inf))
    assert math.isnan(qm.evaluate(program, value=math.nan))


@pytest.mark.parametrize("value", [math.inf, math.nan])
def test_reference_rejects_nonfinite_real_at_angle_consumption(value: float) -> None:
    """A nonfinite QCore Real fails only when consumed as a quantum angle."""

    source = """
import qamomile as qm
from qamomile._execution_payload import ExecutablePayload

@qm.qkernel
def rotate(theta: qm.Param[qm.Real]) -> qm.Unit:
    state = qm.qubits(1)
    state = qm.ry(state, theta, target=0)
    qm.discard(state)
"""
    program = qm.build_source(
        source, module_name="reference_nonfinite_angle"
    ).specialize("rotate")

    with pytest.raises(qm.ExecutionError, match="Q4402"):
        qm.evaluate(program, theta=value)


def test_evaluate_returns_exact_portable_values_after_quantum_effects() -> None:
    """Hybrid execution retains exact BigInt and Complex result leaves in Rust."""

    large_integer = 1 << 4096
    program = qm.build_source(
        f"""
import qamomile as qm
from qamomile._execution_payload import ExecutablePayload

@qm.qkernel
def exact_values() -> tuple[qm.Int, qm.Complex]:
    qm.discard(qm.x(qm.qubits(1)))
    return ({large_integer}, (1 + 2j) * (3 - 4j))
""",
        module_name="native_evaluate_portable_values",
    ).specialize("exact_values")

    assert qm.evaluate(program) == (large_integer, 11 + 2j)


def test_evaluate_accepts_complex_launch_values_natively() -> None:
    """Complex parameters stay native in pure and hybrid value execution."""

    module = qm.build_source(
        """
import qamomile as qm
from qamomile._execution_payload import ExecutablePayload

@qm.qkernel
def transform(value: qm.Param[qm.Complex]) -> qm.Complex:
    return value * (1 + 2j)

@qm.qkernel
def transform_after_effect(value: qm.Param[qm.Complex]) -> qm.Complex:
    qm.discard(qm.x(qm.qubits(1)))
    return value * (1 + 2j)
""",
        module_name="native_evaluate_complex_parameter",
    )

    for entrypoint in ("transform", "transform_after_effect"):
        program = module.specialize(entrypoint)
        assert qm.evaluate(program, value=1.5 - 2.25j) == 6 + 0.75j


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1 << 4096, (1,)), (-(1 << 4096), (0,))],
)
def test_reference_materializes_bigint_quantum_control_natively(
    value: int,
    expected: tuple[int, ...],
) -> None:
    """Arbitrary-precision host control selects native quantum topology."""

    program = qm.build_source(
        """
import qamomile as qm
from qamomile._execution_payload import ExecutablePayload

@qm.qkernel
def controlled(value: qm.Param[qm.Int]) -> qm.Shot[qm.Bits[1]]:
    state = qm.qubits(1)
    if value > 0:
        state = qm.x(state)
    return qm.measure(state)
""",
        module_name="native_bigint_quantum_control",
    ).specialize("controlled")

    assert qm.evaluate(program, value=value) == expected


def test_reference_prepares_and_natively_samples_abstract_adjoint_calls() -> None:
    """Reference stores the optimized image and executes its equivalent result."""

    source = """
import qamomile as qm
from qamomile._execution_payload import ExecutablePayload

@qm.qkernel
def prepare_one(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    state = qm.h(state)
    return qm.t(state)

@qm.qkernel
def undo() -> qm.Shot[qm.Bits[1]]:
    state = qm.qubits(1)
    state = prepare_one(state)
    state = qm.adjoint(prepare_one, state)
    return qm.measure(state)
"""
    program = qm.build_source(
        source, module_name="reference_abstract_adjoint"
    ).specialize("undo")
    artifact = qm.compile(program, to=qm.Reference())
    inspection = Program.from_bytes(export_payload(artifact)).inspect()

    assert "qcore.adjoint_call" in program.inspect()
    assert "qcore.adjoint_call" not in inspection
    assert "qcore.adjoint(" not in inspection
    assert qm.execute(artifact, on=qm.Reference(), shots=8, seed=5).counts == {(0,): 8}


def test_reference_quantum_unit_and_native_failures_are_disjoint() -> None:
    """Quantum effects compose with typed values while failures stay closed."""

    source = """
from typing import NamedTuple

import qamomile as qm
from qamomile._execution_payload import ExecutablePayload

class Summary(NamedTuple):
    total: qm.Int
    accepted: qm.Bool

@qm.qkernel
def quantum_value() -> qm.Unit:
    qm.discard(qm.x(qm.qubits(1)))
    return None

@qm.qkernel
def quantum_summary(
    value: qm.Param[qm.Int], accepted: qm.Param[qm.Bool]
) -> Summary:
    qm.discard(qm.h(qm.qubits(1)))
    return Summary(total=value + 2, accepted=not accepted)

@qm.qkernel
def invalid(value: qm.Param[qm.Int]) -> qm.Real:
    return value / 0

@qm.qkernel
def invalid_after_effect(value: qm.Param[qm.Int]) -> qm.Real:
    qm.discard(qm.qubits(1))
    return value / 0

@qm.qkernel
def runtime_invalid(value: qm.Param[qm.Int], divisor: qm.Param[qm.Int]) -> qm.Real:
    return value / divisor
"""
    module = qm.build_source(source, module_name="reference_value_edges")

    result = qm.run(
        module.specialize("quantum_value"),
        on=qm.Reference(),
    ).result

    assert result.value is None
    summary = qm.run(
        module.specialize("quantum_summary"),
        on=qm.Reference(),
        parameters={"value": 5, "accepted": False},
        seed=19,
    ).result.value
    assert dict(summary) == {"total": 7, "accepted": True}
    with pytest.raises(qm.QamomileError, match="Q7804.*division by zero"):
        qm.run(
            module.specialize("invalid"),
            on=qm.Reference(),
            parameters={"value": 1},
        )
    with pytest.raises(qm.QamomileError, match="Q7804.*division by zero"):
        qm.run(
            module.specialize("invalid_after_effect"),
            on=qm.Reference(),
            parameters={"value": 1},
        )

    with pytest.raises(qm.ExecutionError, match="Q8145.*Q4502.*division by zero"):
        qm.run(
            module.specialize("runtime_invalid"),
            on=qm.Reference(),
            parameters={"value": 1, "divisor": 0},
        )


def test_reference_rejects_an_explicit_native_sampling_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An installed native sampling rejection cannot re-enter Python."""

    from qamomile import _qamomile_native as native

    program = qm.build_source(BELL_SOURCE).specialize("bell")

    def unsupported(*_args: object, **_kwargs: object) -> object:
        """Report one explicit native simulator coverage gap."""

        raise NotImplementedError("Q4402: simulated coverage gap")

    monkeypatch.setattr(native, "execute_reference_at", unsupported)

    with pytest.raises(qm.ExecutionError, match="Q8145.*Q4402"):
        qm.run(program, on=qm.Reference(), shots=8, seed=3)
    with pytest.raises(qm.ExecutionError, match="Q8145.*Q4402"):
        qm.evaluate(program, seed=3)


def test_reference_rejects_an_explicit_native_value_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An installed native value rejection cannot re-enter Python."""

    from qamomile import _qamomile_native as native

    program = qm.build_source(
        """
import qamomile as qm
from qamomile._execution_payload import ExecutablePayload

@qm.qkernel
def identity(value: qm.Param[qm.Int]) -> qm.Int:
    return value
""",
        module_name="native_reference_value_authority",
    ).specialize("identity")

    def unsupported(*_args: object, **_kwargs: object) -> object:
        """Report one explicit native value evaluator coverage gap."""

        raise NotImplementedError("Q4501: simulated value coverage gap")

    monkeypatch.setattr(native, "execute_reference_at", unsupported)

    with pytest.raises(qm.ExecutionError, match="Q8145.*Q4501"):
        qm.run(
            program,
            on=qm.Reference(),
            parameters={"value": 1},
        )
    with pytest.raises(qm.ExecutionError, match="Q8145.*Q4501"):
        qm.evaluate(program, value=1)


def test_reference_rejects_native_sampling_failures_and_nonportable_seeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native integrity failures and seeds outside the shared ABI fail closed."""

    from qamomile import _qamomile_native as native

    program = qm.build_source(BELL_SOURCE).specialize("bell")
    artifact = qm.compile(program, to=qm.Reference())

    def malformed(*_args: object, **_kwargs: object) -> object:
        """Simulate a native statevector integrity failure."""

        raise ValueError("Q4404: simulated malformed state")

    monkeypatch.setattr(native, "execute_reference_at", malformed)
    with pytest.raises(qm.ExecutionError, match="Q8145.*Q4404"):
        qm.execute(artifact, on=qm.Reference(), shots=1)

    monkeypatch.setattr(
        native,
        "execute_reference_at",
        lambda *_args: SimpleNamespace(kind="value", counts=[], value=b""),
    )
    with pytest.raises(qm.ExecutionError, match="Q8145.*sampled ABI"):
        qm.execute(artifact, on=qm.Reference(), shots=1)

    for seed in (True, -1, 1 << 64):
        with pytest.raises(qm.ExecutionError, match="Q8138"):
            qm.execute(artifact, on=qm.Reference(), seed=seed)  # type: ignore[arg-type]


def test_reference_rejects_a_native_value_outside_its_result_abi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A canonical but wrongly typed native value fails at the Rust ABI gate."""

    from qamomile import _qamomile_native as native
    from qamomile.proto import qcore_pb2 as wire

    program = qm.build_source(
        """
import qamomile as qm
from qamomile._execution_payload import ExecutablePayload

@qm.qkernel
def value() -> qm.Int:
    return 1
""",
        module_name="invalid_native_result_value",
    ).specialize("value")
    artifact = qm.compile(program, to=qm.Reference())
    wrong_value = wire.LiteralProto(bool_value=True).SerializeToString(
        deterministic=True
    )
    monkeypatch.setattr(
        native,
        "execute_reference_at",
        lambda *_args: SimpleNamespace(kind="value", counts=[], value=wrong_value),
    )

    with pytest.raises(qm.ExecutionError, match="Q8145.*Q8143"):
        qm.execute(artifact, on=qm.Reference())


def test_reference_rejects_payload_and_envelope_abi_disagreement() -> None:
    """Reference launch must derive its parameter and result ABIs from QCore."""

    target = qm.Reference()
    parameterized = qm.compile(
        qm.build_source(
            _HOST_DISPATCH_SOURCE,
            module_name="reference_parameter_contract",
        ).specialize("sample"),
        to=target,
    )
    with pytest.raises(qm.ExecutionError, match="Q8136"):
        qm.execute(
            replace(parameterized, parameter_abi=ParameterABI()),
            on=target,
        )

    bell = qm.compile(
        qm.build_source(BELL_SOURCE).specialize("bell"),
        to=target,
    )
    reversed_result = ResultABI(
        bell.result_abi.type,
        tuple(reversed(bell.result_abi.source_order)),
        bell.result_abi.native_width,
    )
    with pytest.raises(qm.ExecutionError, match="Q8136"):
        qm.execute(replace(bell, result_abi=reversed_result), on=target)


def test_reference_rejects_an_invocation_that_bypasses_abi_normalization() -> None:
    """Direct endpoint use must fail closed when launch values are incomplete."""

    target = qm.Reference()
    artifact = qm.compile(
        qm.build_source(
            _HOST_DISPATCH_SOURCE,
            module_name="reference_invocation_contract",
        ).specialize("sample"),
        to=target,
    )

    with pytest.raises(qm.ExecutionError, match="Q8137"):
        target.launch(artifact, Invocation.create())


def test_execute_rejects_incompatible_capability_snapshot_before_backend() -> None:
    """A live endpoint must satisfy the contract sealed into the artifact."""

    artifact = qm.compile(
        qm.build_source(BELL_SOURCE).specialize("bell"),
        to=qm.Reference(),
    )

    class IncompatibleEndpoint:
        name = "incompatible"
        target = qm.Reference()
        capabilities = CapabilitySnapshot(
            "qamomile.qcore.program",
            frozenset({Requirement.QUANTUM}),
        )

        def launch(
            self,
            _artifact: qm.CompiledProgram,
            _invocation: Invocation,
        ) -> qm.Job:
            raise AssertionError("incompatible endpoint was invoked")

    with pytest.raises(qm.ExecutionError, match="lacks required capabilities"):
        qm.execute(artifact, on=IncompatibleEndpoint(), shots=1)


def test_compile_rejects_target_name_and_capability_format_disagreement() -> None:
    """A target cannot hide a format mismatch behind an empty missing set."""

    class MismatchedTarget:
        """Advertise a format different from the payload identity."""

        name = "qamomile.qcore.program"
        fingerprint = "mismatched.v1"
        capabilities = CapabilitySnapshot("different.format", frozenset(Requirement))

        def lower(self, _program: Program) -> LoweredProgram:
            """Fail if compilation reaches lowering after negotiation."""

            raise AssertionError("mismatched target was lowered")

    with pytest.raises(qm.ArtifactError, match="Q7014"):
        qm.compile(
            qm.build_source(BELL_SOURCE).specialize("bell"),
            to=MismatchedTarget(),
        )


def test_compile_uses_native_entry_requirements_before_target_lowering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rust rejects a target before resource analysis or target lowering."""

    def forbidden_analysis(_program: object) -> object:
        """Fail if an incompatible target triggers semantic resource analysis."""

        raise AssertionError("incompatible target was analyzed")

    monkeypatch.setattr(qm.resources, "logical", forbidden_analysis)

    class IncapableTarget:
        """Expose an empty profile and fail if lowering is reached."""

        name = "qamomile.qcore.program"
        fingerprint = "incapable.v1"
        capabilities = CapabilitySnapshot(name, frozenset())

        def lower(self, _program: Program) -> LoweredProgram:
            """Fail if native capability planning does not stop compilation."""

            raise AssertionError("incompatible target was lowered")

    with pytest.raises(qm.ArtifactError, match="Q7015.*measurement.*quantum"):
        qm.compile(
            qm.build_source(BELL_SOURCE).specialize("bell"),
            to=IncapableTarget(),
        )


def test_installed_native_lowering_planner_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale or inconsistent native route cannot reactivate Python planning."""

    from qamomile import _qamomile_native as native

    program = qm.build_source(BELL_SOURCE).specialize("bell")
    planner = native.plan_target_lowering
    monkeypatch.setattr(native, "plan_target_lowering", None)
    with pytest.raises(qm.ArtifactError, match="Q7016.*no lowering planner"):
        qm.compile(program, to=qm.Reference())

    def invalid_route(*args: object) -> object:
        """Return valid native facts with a contradictory route tag."""

        plan = planner(*args)
        return SimpleNamespace(
            required=plan.required,
            supported=plan.supported,
            missing=plan.missing,
            materialized=plan.materialized,
            unsupported=plan.unsupported,
            route="unsupported",
        )

    monkeypatch.setattr(native, "plan_target_lowering", invalid_route)
    with pytest.raises(qm.ArtifactError, match="Q7016.*returned an invalid route"):
        qm.compile(program, to=qm.Reference())

    def wrong_protocol(*args: object) -> object:
        """Return valid native facts with a route for another target protocol."""

        plan = planner(*args)
        return SimpleNamespace(
            required=plan.required,
            supported=plan.supported,
            missing=plan.missing,
            materialized=plan.materialized,
            unsupported=plan.unsupported,
            route="circuit_family",
        )

    monkeypatch.setattr(native, "plan_target_lowering", wrong_protocol)
    with pytest.raises(qm.ArtifactError, match="Q7016.*invalid circuit-family route"):
        qm.compile(program, to=qm.Reference())


def test_execute_rejects_artifact_for_another_target_before_backend() -> None:
    """Submission must not send a target-bound artifact to another target."""

    artifact = qm.compile(
        qm.build_source(BELL_SOURCE).specialize("bell"),
        to=qm.Reference(),
    )

    @dataclass
    class OtherTarget:
        name: str = "other"

        @property
        def fingerprint(self) -> str:
            return "other.v1"

        def lower(self, _program: Program) -> qm.CompiledProgram:
            raise AssertionError("execution attempted to lower")

    class OtherEndpoint:
        name = "other"
        target = OtherTarget()
        capabilities = CapabilitySnapshot(
            "qamomile.qcore.program",
            frozenset(Requirement),
        )

        def launch(
            self,
            _artifact: qm.CompiledProgram,
            _invocation: Invocation,
        ) -> qm.Job:
            raise AssertionError("mismatched endpoint was invoked")

    with pytest.raises(qm.ExecutionError, match="artifact target does not match"):
        qm.execute(artifact, on=OtherEndpoint(), shots=1)


def test_capability_snapshot_owns_immutable_requirements() -> None:
    """Caller-owned mutable requirements cannot alter a capability snapshot."""

    requirements = {Requirement.QUANTUM}
    realizations = {RealizationCapability.RZZ}
    snapshot = CapabilitySnapshot(
        "qamomile.qcore.program",
        requirements,  # type: ignore[arg-type]
        realizations,  # type: ignore[arg-type]
    )
    requirements.add(Requirement.DISCARD)
    realizations.clear()

    assert snapshot.format == "qamomile.qcore.program"
    assert snapshot.requirements == frozenset({Requirement.QUANTUM})
    assert snapshot.realization_capabilities == frozenset({RealizationCapability.RZZ})
    with pytest.raises(TypeError, match="RealizationCapability"):
        CapabilitySnapshot(
            "qamomile.qcore.program",
            frozenset(),
            frozenset({"rzz"}),  # type: ignore[arg-type]
        )


def test_result_abi_allows_repeated_logical_measurement_sources() -> None:
    """Products may expose the same observed bit in more than one field."""

    abi = ResultABI(tuple[qm.Bits[1], qm.Bits[1]], (0, 0), 1)

    assert abi.decode_big_endian("1") == (1, 1)
    assert abi.decode_value((1, 1)) == ((1,), (1,))
    assert abi.encode_value(((1,), (1,))) == (1, 1)


def test_result_abi_normalizes_native_order_before_fixed_point_decode() -> None:
    """SDK register order is removed before semantic numeric interpretation."""

    abi = ResultABI(qm.Shot[qm.UFixed[0, 3]], (0, 1, 2), 4)
    qiskit_bits = abi.decode_big_endian("1100")
    braket_bits = abi.decode_q0_first("0011")

    assert qiskit_bits == braket_bits == (0, 0, 1)
    assert abi.decode_value(qiskit_bits) == 0.125
    assert abi.encode_value(0.125) == qiskit_bits
    result = qm.SampleResult(
        "probe",
        (((0, 0, 1), 2),),
        result_abi=abi,
    )
    assert result.values == {0.125: 2}
    assert result.most_likely_value == 0.125


def test_result_abi_decodes_uint_as_an_integer_without_changing_bits() -> None:
    """UInt keeps source bit order while exposing its numeric interpretation."""

    abi = ResultABI(qm.Shot[qm.UInt[3]], (2, 0, 1), 3)
    logical_bits = abi.decode_big_endian("110")

    assert logical_bits == (1, 0, 1)
    assert abi.decode_value(logical_bits) == 5
    assert abi.encode_value(5) == logical_bits
    result = qm.SampleResult("probe", (((1, 0, 1), 2),), result_abi=abi)
    assert result.values == {5: 2}
    assert result.most_likely_value == 5


def test_result_abi_delegates_sdk_text_projection_to_rust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public SDK-order helpers fail closed around the native result codec."""

    import qamomile.compilation as compilation

    abi = ResultABI(qm.Bits[2], (2, 0), 3)
    original = compilation.native.decode_native_result_bitstring
    calls: list[tuple[str, bool]] = []

    def tracked(payload: bytes, bitstring: str, q0_first: bool) -> list[int]:
        """Record one native boundary call before forwarding it."""

        calls.append((bitstring, q0_first))
        return original(payload, bitstring, q0_first)

    monkeypatch.setattr(compilation.native, "decode_native_result_bitstring", tracked)
    assert abi.decode_big_endian("110") == (1, 0)
    assert abi.decode_q0_first("110") == (0, 1)
    assert calls == [("110", False), ("110", True)]

    monkeypatch.setattr(compilation.native, "decode_native_result_bitstring", None)
    with pytest.raises(qm.ExecutionError, match="Q8144"):
        abi.decode_big_endian("110")

    monkeypatch.setattr(
        compilation.native,
        "decode_native_result_bitstring",
        lambda *_args: (1, 0),
    )
    with pytest.raises(qm.ExecutionError, match="Q8144"):
        abi.decode_q0_first("110")


def test_result_abi_delegates_value_validation_to_rust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public typed decoding requires the complete native value contract."""

    import qamomile.compilation as compilation

    abi = ResultABI(qm.Bits[2], (0, 1), 2)
    original = compilation.native.validate_result_value
    calls: list[tuple[bytes, bytes]] = []

    def tracked(result_abi: bytes, value: bytes) -> None:
        """Record and forward one native result validation."""

        calls.append((result_abi, value))
        original(result_abi, value)

    monkeypatch.setattr(compilation.native, "validate_result_value", tracked)
    assert abi.decode_value((1, 0)) == (1, 0)
    assert len(calls) == 1

    monkeypatch.setattr(compilation.native, "validate_result_value", None)
    with pytest.raises(qm.ExecutionError, match="Q8144.*validation"):
        abi.decode_value((1, 0))


@pytest.mark.parametrize(
    "counts",
    [
        {(2, 0): 1},
        {(0,): 1, (1, 0): 1},
        {(0, 1): 0.5},
    ],
)
def test_sample_result_rejects_malformed_logical_counts(
    counts: dict[tuple[int, ...], object],
) -> None:
    """Backend count data is checked before it enters the public result API."""

    with pytest.raises(qm.ExecutionError):
        qm.SampleResult.from_counts(backend="invalid", counts=counts)  # type: ignore[arg-type]


def test_sample_result_uses_one_native_histogram_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Construction caches Rust-derived ordering, shots, and deterministic mode."""

    from qamomile import compilation

    original = compilation.native.normalize_sample_counts
    calls = 0

    def tracked(rows: object, expected_width: object) -> object:
        """Record and forward one native histogram request."""

        nonlocal calls
        calls += 1
        return original(rows, expected_width)

    monkeypatch.setattr(compilation.native, "normalize_sample_counts", tracked)
    large = 1 << 100
    result = qm.SampleResult.from_counts(
        backend="reference",
        counts={(1, 0): large, (0, 1): large, (0, 0): 0},
        expected_width=2,
    )

    assert result.counts == {(0, 0): 0, (0, 1): large, (1, 0): large}
    assert result.shots == 1 << 101
    assert result.most_likely == (1, 0)
    assert calls == 1

    monkeypatch.setattr(compilation.native, "normalize_sample_counts", None)
    with pytest.raises(qm.ExecutionError, match="Q8128"):
        qm.SampleResult.from_counts(backend="reference", counts={(0,): 1})

    monkeypatch.setattr(
        compilation.native,
        "normalize_sample_counts",
        lambda *_args: SimpleNamespace(entries=[], shots="1", most_likely=[0]),
    )
    with pytest.raises(qm.ExecutionError, match="Q8128"):
        qm.SampleResult.from_counts(backend="reference", counts={(0,): 1})


def test_sample_result_uses_native_probability_and_bulk_typed_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Derived result views reuse the native summary without Python reductions."""

    huge = 1 << 2048
    abi = ResultABI(qm.Bits[1], (0,), 1)
    result = qm.SampleResult(
        "reference",
        (((0,), huge), ((1,), 1)),
        result_abi=abi,
    )
    monkeypatch.setattr(
        ResultABI,
        "decode_value",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("per-row Python result decoding was used")
        ),
    )

    assert result.probability((0,)) == 1.0
    assert result.probability((1,)) == 0.0
    assert result.expectation_z(0) == 1.0
    assert result.values == {(0,): huge, (1,): 1}


def test_sample_result_copy_and_pickle_rebuild_native_summary() -> None:
    """Python value operations must not expose the retained native cache."""

    abi = ResultABI(qm.Bits[1], (0,), 1)
    result = qm.SampleResult(
        "reference",
        (((0,), 2), ((1,), 1)),
        worker_pid=42,
        result_abi=abi,
    )

    assert copy.copy(result) is result
    assert copy.deepcopy(result) is result
    restored = pickle.loads(pickle.dumps(result))
    assert restored == result
    assert restored.counts == result.counts
    assert restored.probability((0,)) == pytest.approx(2 / 3)
    assert restored.values == {(0,): 2, (1,): 1}


def test_sample_result_validates_observable_indices() -> None:
    """Observable helpers reject negative and out-of-range logical indices."""

    result = qm.SampleResult.from_counts(
        backend="reference",
        counts={(0, 1): 2},
        expected_width=2,
    )

    with pytest.raises(ValueError, match="indices"):
        result.expectation_z(-1)
    with pytest.raises(ValueError, match="indices"):
        result.expectation_z(2)


def test_sample_result_expectation_uses_required_native_statistics() -> None:
    """The convenience API shares the same Rust parity implementation."""

    result = qm.SampleResult.from_counts(
        backend="reference",
        counts={(0, 1): 1, (1, 1): 3},
        expected_width=2,
    )
    native_summary = result._native_summary
    calls = 0

    def record_native_estimate(indices: list[int]) -> object:
        """Record and delegate one retained-summary aggregation."""

        nonlocal calls
        calls += 1
        return native_summary.expectation_z(indices)

    object.__setattr__(
        result,
        "_native_summary",
        SimpleNamespace(expectation_z=record_native_estimate),
    )

    assert result.expectation_z(0) == -0.5
    assert calls == 1

    object.__setattr__(result, "_native_summary", SimpleNamespace())
    with pytest.raises(qm.CapabilityError, match="Q7300"):
        result.expectation_z(0)

    object.__setattr__(
        result,
        "_native_summary",
        SimpleNamespace(expectation_z=lambda _indices: float("nan")),
    )
    with pytest.raises(qm.ExecutionError, match="Q7310"):
        result.expectation_z(0)


def test_structured_sample_result_uses_typed_layout_abi() -> None:
    """Sampled products preserve flat transport and typed public values."""

    source = """
import qamomile as qm
from qamomile._execution_payload import ExecutablePayload

@qm.qkernel
def structured() -> tuple[qm.Shot[qm.Bits[1]], qm.Shot[qm.Bits[1]]]:
    left = qm.qubits(1)
    left_bits = qm.measure(left)
    right = qm.qubits(1)
    right_bits = qm.measure(right)
    return (left_bits, right_bits)
"""
    program = qm.build_source(source).specialize("structured")

    artifact = qm.compile(program, to=qm.Reference())
    result = qm.execute(artifact, on=qm.Reference(), shots=4, seed=2)

    assert artifact.result_abi.width == 2
    assert result.counts == {(0, 0): 4}
    assert result.values == {((0,), (0,)): 4}


def _line_profile(**overrides):
    """Build a small three-qubit profile with one symmetric edge."""
    from qamomile.compilation import (
        CouplingEdge,
        EdgeProperty,
        QubitProperty,
        TargetCalibration,
        TargetProfile,
        TargetTopology,
    )

    facts = {
        "format": "qiskit.qpy.v1",
        "topology": TargetTopology(
            3,
            (CouplingEdge(1, 2, True), CouplingEdge(0, 1)),
            ("x", "rz"),
            ("cx",),
        ),
        "calibration": TargetCalibration(
            (QubitProperty(1, "t1", 1e-4, "s"), QubitProperty(0, "t1", 2e-4, "s")),
            (EdgeProperty(2, 1, "cx.error", 0.01, "probability"),),
            "test",
        ),
        "backend_name": "fake_line",
    }
    facts.update(overrides)
    return TargetProfile(**facts)


def test_target_profile_is_canonical_native_validated_identity() -> None:
    """Equal facts in any order share bytes and digest; Rust rejects bad facts."""
    from qamomile.compilation import (
        CouplingEdge,
        EdgeProperty,
        LogicalProfile,
        QubitProperty,
        TargetCalibration,
        TargetProfile,
        TargetTopology,
    )

    profile = _line_profile()
    reordered = _line_profile(
        topology=TargetTopology(
            3, (CouplingEdge(0, 1), CouplingEdge(1, 2, True)), ("rz", "x"), ("cx",)
        ),
        calibration=TargetCalibration(
            (QubitProperty(0, "t1", 2e-4, "s"), QubitProperty(1, "t1", 1e-4, "s")),
            (EdgeProperty(2, 1, "cx.error", 0.01, "probability"),),
            "test",
        ),
    )
    assert profile == reordered and hash(profile) == hash(reordered)
    assert profile.to_bytes() == reordered.to_bytes()
    assert profile.digest == reordered.digest and len(profile.digest) == 64
    assert profile.topology.edges == (CouplingEdge(0, 1), CouplingEdge(1, 2, True))
    assert profile.topology.one_qubit_operations == ("rz", "x")
    assert profile.calibration.qubit_properties[0].qubit == 0
    assert TargetProfile.from_bytes(profile.to_bytes()) == profile
    assert pickle.loads(pickle.dumps(profile)) == profile
    logical = _line_profile(
        logical=LogicalProfile(
            "surface", 11, 1e-6, "s", 100.0, "cycles", "rs", (("b", 1.0), ("a", 2.0))
        )
    )
    assert logical.logical.rotation_synthesis_constants == (("a", 2.0), ("b", 1.0))
    assert logical.digest != profile.digest

    with pytest.raises(qm.ArtifactError, match="Q7032.*outside"):
        _line_profile(topology=TargetTopology(3, (CouplingEdge(0, 3),)))
    with pytest.raises(qm.ArtifactError, match="Q7032.*repeated"):
        _line_profile(
            topology=TargetTopology(3, (CouplingEdge(0, 1), CouplingEdge(0, 1)))
        )
    with pytest.raises(qm.ArtifactError, match="Q7033"):
        _line_profile(
            calibration=TargetCalibration((QubitProperty(0, "t1", math.nan),))
        )
    with pytest.raises(qm.ArtifactError, match="Q7032.*undeclared"):
        _line_profile(
            calibration=TargetCalibration((), (EdgeProperty(2, 0, "e", 0.1),))
        )
    with pytest.raises(qm.ArtifactError, match="Q7031"):
        _line_profile(topology=TargetTopology(70_000))
    with pytest.raises(qm.ArtifactError, match="Q7030"):
        TargetProfile.from_bytes(b"\xff\xff")
    with pytest.raises(qm.ArtifactError, match="Q7030.*canonical"):
        TargetProfile.from_bytes(profile.to_bytes() + b"\x42\x00")
    with pytest.raises(TypeError, match="declared types"):
        _line_profile(topology=TargetTopology("3"))
    with pytest.raises(TypeError, match="TargetTopology"):
        TargetProfile("f", "not a topology")


@pytest.fixture
def oversized_target_profile_payload() -> bytes:
    """Append a flat unknown field beyond the native 16 MiB wire limit."""
    from qamomile.compilation import TargetProfile, TargetTopology

    canonical = TargetProfile("test", TargetTopology(1)).to_bytes()
    return canonical + b"\x4a\x80\x80\x80\x08" + b"x" * (16 * 1024 * 1024)


def test_target_profile_from_bytes_rejects_oversized_original_payload(
    oversized_target_profile_payload: bytes,
) -> None:
    """Unknown fields still count toward the profile wire-size bound."""
    from qamomile.compilation import TargetProfile

    with pytest.raises(qm.ArtifactError, match="Q7031"):
        TargetProfile.from_bytes(oversized_target_profile_payload)


def test_target_profile_from_bytes_rejects_oversized_payload_before_python_decode(
    monkeypatch: pytest.MonkeyPatch,
    oversized_target_profile_payload: bytes,
) -> None:
    """Native bounds reject oversized bytes before Python expands protobufs."""
    from qamomile.compilation import TargetProfile
    from qamomile.proto import target_profile_pb2

    def reject_python_decode(_payload: bytes) -> None:
        raise AssertionError("Python decoded target profile before native preflight")

    monkeypatch.setattr(
        target_profile_pb2.TargetProfileProto, "FromString", reject_python_decode
    )

    with pytest.raises(qm.ArtifactError, match="Q7031"):
        TargetProfile.from_bytes(oversized_target_profile_payload)


def test_target_profile_from_bytes_rejects_unavailable_native_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restoring valid bytes requires the native validation capability."""
    from qamomile.compilation import TargetProfile

    canonical = _line_profile().to_bytes()
    monkeypatch.setattr(compilation_module.native, "validate_target_profile", None)

    with pytest.raises(qm.ArtifactError, match="Q7030.*unavailable"):
        TargetProfile.from_bytes(canonical)


def test_target_profile_from_bytes_rejects_changed_subclass_fields() -> None:
    """Subclass construction cannot change the facts encoded in restored bytes."""
    from qamomile.compilation import TargetProfile

    class ChangedProfile(TargetProfile):
        def __post_init__(self) -> None:
            object.__setattr__(self, "backend_name", "changed")
            super().__post_init__()

    with pytest.raises(qm.ArtifactError, match="Q7030.*canonical"):
        ChangedProfile.from_bytes(_line_profile().to_bytes())


@pytest.mark.parametrize(
    "result",
    [
        None,
        (),
        (b"",),
        [b"", b"x" * 32],
        ("bytes", b"x" * 32),
        (b"", "digest"),
        (b"", b"x"),
    ],
)
def test_target_profile_from_bytes_rejects_malformed_native_result(
    monkeypatch: pytest.MonkeyPatch, result: object
) -> None:
    """Malformed bridge results cannot restore a usable profile."""
    from qamomile.compilation import TargetProfile

    canonical = _line_profile().to_bytes()
    monkeypatch.setattr(
        compilation_module.native, "validate_target_profile", lambda _payload: result
    )

    with pytest.raises(qm.ArtifactError, match="Q7030.*malformed"):
        TargetProfile.from_bytes(canonical)


def test_target_profile_is_snapshot_identity_not_capability(monkeypatch) -> None:
    """Snapshots carry the profile for context; ``require`` never consults it."""
    profile = _line_profile()
    snapshot = CapabilitySnapshot(
        "qiskit.qpy.v1", frozenset(Requirement), profile=profile
    )
    bare = CapabilitySnapshot("qiskit.qpy.v1", frozenset(Requirement))
    assert snapshot.profile is profile and bare.profile is None
    with pytest.raises(TypeError, match="TargetProfile"):
        CapabilitySnapshot("qiskit.qpy.v1", frozenset(), profile="profile")

    program = qm.build_source(BELL_SOURCE).specialize("bell")
    artifact = qm.compile(program, to=qm.Reference())
    contract = replace(artifact.execution_contract, format="qiskit.qpy.v1")
    snapshot.require(contract)
    bare.require(contract)

    monkeypatch.setattr(compilation_module.native, "validate_target_profile", None)
    with pytest.raises(qm.ArtifactError, match="Q7030.*unavailable"):
        _line_profile()
