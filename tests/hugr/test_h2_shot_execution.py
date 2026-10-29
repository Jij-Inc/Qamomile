"""Execute validated H2 QIR from real shot-indexed qkernels with local callbacks."""

from __future__ import annotations

import ctypes
import json
import math
import re
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.transpiler import ExecutionError, JobSnapshot
from qamomile.hugr import (
    HugrExecutor,
    HugrTranspiler,
    NexusExecutionOptions,
    SeleneExecutionOptions,
)
from qamomile.hugr._qir import to_h2_qir
from qamomile.hugr._runtime import prepare_execution
from qamomile.hugr._shot_range import (
    SHOT_SUCCESS_TAG,
    normalize_h2_shots,
    prepare_h2_shot_range,
)

pytest.importorskip("hugr")
pytest.importorskip("tket_exts")
pytest.importorskip("hugr_qir.hugr_to_qir")
pyqir = pytest.importorskip("pyqir")
llvm = pytest.importorskip("llvmlite.binding")

pytestmark = pytest.mark.hugr

_MAX_UINT64 = (1 << 64) - 1
_SHOTS = 4
_DIRECT = [False, True, True, False]
_AFFINE = [False, False, False, True, False, True, False, False]


@qmc.qkernel
def _direct_uint(values: qmc.Vector[qmc.UInt]) -> tuple[qmc.Bit, qmc.Bit]:
    selected = values[qmc.get_current_shot()] == 1
    q = qmc.qubit("q")
    if selected:
        q = qmc.x(q)
    return selected, qmc.measure(q)


@qmc.qkernel
def _direct_bit(values: qmc.Vector[qmc.Bit]) -> tuple[qmc.Bit, qmc.Bit]:
    selected = values[qmc.get_current_shot()]
    q = qmc.qubit("q")
    if selected:
        q = qmc.x(q)
    return selected, qmc.measure(q)


@qmc.qkernel
def _affine_uint(values: qmc.Vector[qmc.UInt]) -> tuple[qmc.Bit, qmc.Bit]:
    selected = values[2 * qmc.get_current_shot() + 1] == 1
    q = qmc.qubit("q")
    if selected:
        q = qmc.x(q)
    return selected, qmc.measure(q)


@qmc.qkernel
def _affine_bit(values: qmc.Vector[qmc.Bit]) -> tuple[qmc.Bit, qmc.Bit]:
    selected = values[2 * qmc.get_current_shot() + 1]
    q = qmc.qubit("q")
    if selected:
        q = qmc.x(q)
    return selected, qmc.measure(q)


@qmc.qkernel
def _repeated_shot(values: qmc.Vector[qmc.Bit]) -> tuple[qmc.Bit, qmc.Bit, qmc.Bit]:
    shot = qmc.get_current_shot()
    selected = values[shot]
    return selected, shot == 0, qmc.get_current_shot() == 1


@qmc.qkernel
def _nested_lookup(values: qmc.Vector[qmc.UInt]) -> qmc.Bit:
    return values[qmc.get_current_shot()] == 1


@qmc.qkernel
def _nested_entry(values: qmc.Vector[qmc.UInt]) -> tuple[qmc.Bit, qmc.Bit]:
    selected = _nested_lookup(values)
    q = qmc.qubit("q")
    if selected:
        q = qmc.x(q)
    return selected, qmc.measure(q)


@qmc.qkernel
def _lookup_after_quantum_operations(
    values: qmc.Vector[qmc.UInt],
) -> tuple[qmc.Bit, qmc.Bit]:
    q = qmc.qubit("q")
    q = qmc.x(q)
    selected = values[qmc.get_current_shot()] == 1
    if selected:
        q = qmc.x(q)
    return selected, qmc.measure(q)


@qmc.qkernel
def _shot_without_array() -> qmc.Bit:
    q = qmc.qubit("q")
    if qmc.get_current_shot() == 0:
        q = qmc.x(q)
    return qmc.measure(q)


@qmc.qkernel
def _multi_elif_shot(values: qmc.Vector[qmc.UInt]) -> qmc.Bit:
    q = qmc.qubit("q")
    shot = qmc.get_current_shot()
    if shot == 0:
        q = qmc.x(q)
    elif shot == 1:
        pass
    elif shot == 2:
        q = qmc.x(q)
    elif shot == 3:
        pass
    else:
        q = qmc.y(q)
    return qmc.measure(q)


@qmc.qkernel
def _table_and_condition(values: qmc.Vector[qmc.Bit]) -> tuple[qmc.Bit, qmc.Bit]:
    shot = qmc.get_current_shot()
    selected = values[shot]
    q = qmc.qubit("q")
    if shot == 0:
        q = qmc.x(q)
    return selected, qmc.measure(q)


@qmc.qkernel
def _manually_guarded(values: qmc.Vector[qmc.UInt]) -> qmc.Bit:
    shot = qmc.get_current_shot()
    selected = qmc.uint(0)
    index = shot * shot
    if index < values.shape[0]:
        selected = values[index]
    return selected == 1


def _compile_and_prepare(kernel, values):
    if isinstance(values[0], bool):
        compiled = HugrTranspiler().compile(kernel, bindings={"values": values})
        return compiled, prepare_execution(compiled)
    program = HugrTranspiler().transpile(
        kernel, parameters=["values"], parameter_shapes={"values": (len(values),)}
    )
    compiled = program._compiled
    return compiled, prepare_execution(compiled, {"values": values})


@dataclass
class _Trace:
    raw_shot: int = 0
    quantum_events: list[str] = field(default_factory=list)
    outputs: list[tuple[str, bool]] = field(default_factory=list)
    qubits: dict[int | None, bool] = field(default_factory=dict)
    results: dict[int | None, bool] = field(default_factory=dict)
    unsupported: list[str] = field(default_factory=list)


class _QirExecution:
    """Keep native callbacks and the executable module alive for each assertion."""

    def __init__(self, bitcode: bytes, *, deterministic_x: bool = False):
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()
        self.trace = _Trace()
        self._deterministic_x = deterministic_x
        self._callbacks = []
        pointer = ctypes.c_void_p
        self._register("__quantum__rt__initialize", None, [pointer], lambda _: None)
        self._register(
            "___get_current_shot", ctypes.c_uint64, [], lambda: self.trace.raw_shot
        )
        self._register(
            "__quantum__qis__phasedx__body",
            None,
            [ctypes.c_double, ctypes.c_double, pointer],
            self._phased_x,
        )
        self._register(
            "__quantum__qis__mz__body",
            None,
            [pointer, pointer],
            self._measure,
        )
        self._register(
            "__quantum__qis__reset__body",
            None,
            [pointer],
            self._reset,
        )
        self._register(
            "__quantum__rt__read_result", ctypes.c_bool, [pointer], self._read_result
        )
        self._register(
            "__quantum__rt__bool_record_output",
            None,
            [ctypes.c_bool, pointer],
            lambda value, label: self.trace.outputs.append(
                (ctypes.string_at(label).decode(), bool(value))
            ),
        )
        ir = str(pyqir.Module.from_bitcode(pyqir.Context(), bitcode))
        assert "@abort" not in ir
        if deterministic_x:
            assert set(re.findall(r"@(__quantum__qis__\w+)", ir)) <= {
                "__quantum__qis__phasedx__body",
                "__quantum__qis__mz__body",
                "__quantum__qis__reset__body",
            }, ir
        entry = re.search(r"define i64 @([^ (]+)\(", ir)
        assert entry is not None, ir
        machine = llvm.Target.from_default_triple().create_target_machine()
        native = re.sub(
            r'target triple = ".*"',
            f'target triple = "{llvm.get_default_triple()}"',
            ir,
        )
        native = re.sub(
            r'target datalayout = ".*"',
            f'target datalayout = "{machine.target_data}"',
            native,
        )
        module = llvm.parse_assembly(native)
        module.verify()
        self._engine = llvm.create_mcjit_compiler(module, machine)
        self._engine.finalize_object()
        self._entry = ctypes.CFUNCTYPE(ctypes.c_uint64)(
            self._engine.get_function_address(entry.group(1))
        )

    def _phased_x(self, theta, phase, qubit):
        self.trace.quantum_events.append("x")
        if self._deterministic_x:
            if not (math.isclose(theta, math.pi) and math.isclose(phase, 0.0)):
                self.trace.unsupported.append(f"phasedx({theta}, {phase})")
            self.trace.qubits[qubit] = not self.trace.qubits.get(qubit, False)

    def _measure(self, qubit, result):
        self.trace.quantum_events.append("measure")
        if self._deterministic_x:
            self.trace.results[result] = self.trace.qubits.get(qubit, False)

    def _reset(self, qubit):
        self.trace.quantum_events.append("reset")
        self.trace.qubits[qubit] = False

    def _read_result(self, result):
        if self._deterministic_x:
            if result not in self.trace.results:
                self.trace.unsupported.append("read_result before measurement")
            return self.trace.results.get(result, False)
        return False

    def _register(self, name, result, arguments, callback):
        native = ctypes.CFUNCTYPE(result, *arguments)(callback)
        self._callbacks.append(native)
        llvm.add_symbol(name, ctypes.cast(native, ctypes.c_void_p).value)

    def run(self, raw_shot: int) -> tuple[int, _Trace]:
        self.trace = _Trace(raw_shot=raw_shot)
        code = self._entry()
        assert self.trace.unsupported == [], self.trace
        return code, self.trace


@pytest.mark.parametrize("origin", [0, 1])
@pytest.mark.parametrize(
    ("kernel", "values", "scale", "offset", "initial_gates"),
    [
        (_direct_uint, [int(v) for v in _DIRECT], 1, 0, 0),
        (_direct_bit, _DIRECT, 1, 0, 0),
        (_affine_uint, [int(v) for v in _AFFINE], 2, 1, 0),
        (_affine_bit, _AFFINE, 2, 1, 0),
        (_nested_entry, [int(v) for v in _DIRECT], 1, 0, 0),
        (_lookup_after_quantum_operations, [int(v) for v in _DIRECT], 1, 0, 1),
    ],
)
def test_real_qkernel_qir_selects_normalized_shots_and_guards_all_user_operations(
    origin, kernel, values, scale, offset, initial_gates
):
    compiled, prepared = _compile_and_prepare(kernel, values)
    before = prepared.package.to_bytes()
    specialized, guarded = prepare_h2_shot_range(prepared.package, _SHOTS, origin)
    assert guarded
    bitcode, manifest = to_h2_qir(specialized)
    assert manifest[SHOT_SUCCESS_TAG] == ("bool", 1)
    assert prepared.package.to_bytes() == before
    execution = _QirExecution(bitcode)
    raw_shots = [*range(origin, origin + _SHOTS), origin + _SHOTS, _MAX_UINT64]
    if origin:
        raw_shots.append(origin - 1)
    for raw in raw_shots:
        code, trace = execution.run(raw)
        valid = origin <= raw < origin + _SHOTS
        assert code == (0 if valid else 1), (origin, raw, trace)
        assert len(trace.outputs) == len(dict(trace.outputs)), trace
        outputs = dict(trace.outputs)
        assert outputs[SHOT_SUCCESS_TAG] is valid, trace
        if valid:
            selected = bool(values[scale * (raw - origin) + offset])
            assert outputs == {
                SHOT_SUCCESS_TAG: True,
                prepared.output_tags[0]: selected,
                prepared.output_tags[1]: False,
            }, trace
            assert trace.outputs[-1] == (SHOT_SUCCESS_TAG, True), trace
            assert trace.quantum_events.count("x") == initial_gates + int(selected), (
                trace
            )
            assert trace.quantum_events.count("measure") == 1, trace
        else:
            assert outputs == {
                **{tag: False for tag in prepared.output_tags},
                SHOT_SUCCESS_TAG: False,
            }, trace
            assert trace.outputs[-1] == (SHOT_SUCCESS_TAG, False), trace
            assert trace.quantum_events == [], trace


@pytest.mark.parametrize("origin", [0, 1])
def test_normalization_updates_all_consumers_and_repeated_shot_reads(origin):
    compiled, prepared = _compile_and_prepare(_repeated_shot, _DIRECT)
    specialized, guarded = prepare_h2_shot_range(prepared.package, _SHOTS, origin)
    assert guarded
    execution = _QirExecution(to_h2_qir(specialized)[0])
    for shot in range(_SHOTS):
        code, trace = execution.run(origin + shot)
        assert code == 0
        assert dict(trace.outputs) == {
            prepared.output_tags[0]: _DIRECT[shot],
            prepared.output_tags[1]: shot == 0,
            prepared.output_tags[2]: shot == 1,
            SHOT_SUCCESS_TAG: True,
        }


def test_specializing_for_h2_preserves_the_original_selene_executable():
    pytest.importorskip("selene_sim")
    compiled, prepared = _compile_and_prepare(_direct_bit, _DIRECT)
    original = compiled.artifact.to_bytes()
    prepare_h2_shot_range(prepared.package, _SHOTS, 1)
    executor = HugrExecutor("selene", options=SeleneExecutionOptions(seed=3))
    records = executor.submit(prepared.package, _SHOTS).result()
    assert records == [
        {prepared.output_tags[0]: bit, prepared.output_tags[1]: bit} for bit in _DIRECT
    ]
    assert compiled.artifact.to_bytes() == original


def test_shot_without_array_is_normalized_and_guarded():
    compiled = HugrTranspiler().compile(_shot_without_array)
    prepared = prepare_execution(compiled)
    before = prepared.package.to_bytes()
    package, guarded = normalize_h2_shots(prepared.package, _SHOTS, 1)
    assert guarded
    assert package is not prepared.package
    assert prepared.package.to_bytes() == before
    assert SHOT_SUCCESS_TAG in to_h2_qir(package)[1]


@pytest.mark.parametrize("marker", [True, False, None])
def test_public_sample_and_restore_use_real_guarded_qir(monkeypatch, marker):
    pytest.importorskip("pytket")
    from tests.hugr.test_h2_qir import _install_h2_client

    tags = [SHOT_SUCCESS_TAG, "qamomile.output.1.0", "qamomile.output.0.0"]
    readouts = [[1, 0, int(value)] for value in _DIRECT]
    readouts[-1][0] = int(bool(marker))
    if marker is None:
        tags = tags[1:]
        readouts = [row[1:] for row in readouts]
    client = _install_h2_client(monkeypatch, tags, readouts, layout="interleaved")
    program = HugrTranspiler().transpile(
        _direct_uint, parameters=["values"], parameter_shapes={"values": (_SHOTS,)}
    )
    bindings = {"values": [int(value) for value in _DIRECT]}
    executor = HugrExecutor("nexus", options=NexusExecutionOptions(system_name="H2-1E"))
    job = program.sample(executor, shots=_SHOTS, bindings=bindings)
    saved = JobSnapshot.from_dict(json.loads(json.dumps(job.snapshot().to_dict())))
    monkeypatch.setattr(
        "qamomile.hugr._nexus.to_h2_qir",
        Mock(side_effect=AssertionError("Restoration must not reconvert or resubmit")),
    )
    restored = program.restore(executor, saved, bindings)
    assert saved.executions[0].context["shot_success_tag"] == SHOT_SUCCESS_TAG
    assert saved.executions[0].context["shot_origin"] == "1"
    assert "h2_shot_guard" not in saved.executions[0].context
    assert "h2_shot_origin" not in saved.executions[0].context
    bitcode = client.qir.upload.call_args.kwargs["qir"]
    ir = str(pyqir.Module.from_bitcode(pyqir.Context(), bitcode))
    assert "@abort" not in ir
    assert SHOT_SUCCESS_TAG in ir
    for handle in (job, restored):
        if marker:
            assert handle.result().results == [((False, False), 2), ((True, False), 2)]
        else:
            message = (
                "Missing QIR result bits"
                if marker is None
                else "shot range check failed"
            )
            with pytest.raises(ExecutionError, match=message):
                handle.result()
    client.qir.upload.assert_called_once()
    client.start_execute_job.assert_called_once()


def test_public_sample_rejects_a_short_table_before_provider_calls(monkeypatch):
    pytest.importorskip("pytket")
    from tests.hugr.test_h2_qir import _install_h2_client

    client = _install_h2_client(monkeypatch, ["unused"], [[1]])
    program = HugrTranspiler().transpile(
        _direct_uint, bindings={"values": [0, 1, 0, 1]}
    )
    executor = HugrExecutor("nexus", options=NexusExecutionOptions(system_name="H2-1E"))
    with pytest.raises(ExecutionError, match="exceeds table length"):
        program.sample(executor, shots=_SHOTS + 1)
    client.login.assert_not_called()
    client.qir.upload.assert_not_called()
    client.start_execute_job.assert_not_called()


def _uploaded_execution(monkeypatch, program, *, bindings=None):
    from qamomile.hugr._nexus import NexusTransport
    from tests.hugr.test_h2_nexus import _client

    client = _client()
    monkeypatch.setattr(
        "qamomile.hugr.execution._NexusTransport",
        lambda options: NexusTransport(options, client=client),
    )
    executor = HugrExecutor("nexus", options=NexusExecutionOptions(system_name="H2-1E"))
    job = program.sample(executor, shots=_SHOTS, bindings=bindings)
    bitcode = client.qir.upload.call_args.kwargs["qir"]
    execution = _QirExecution(bitcode, deterministic_x=True)
    traces = []
    for raw in range(1, _SHOTS + 1):
        code, trace = execution.run(raw)
        assert code == 0
        traces.append(trace)
    labels = list(reversed(dict(traces[0].outputs)))
    raw = SimpleNamespace(
        get_bitlist=Mock(
            return_value=[SimpleNamespace(reg_name=tag, index=[0]) for tag in labels]
        ),
        get_shots=Mock(
            return_value=[
                [int(dict(trace.outputs)[tag]) for tag in labels] for trace in traces
            ]
        ),
    )
    client.jobs.results.return_value[0].download_result.return_value = raw
    return executor, job, client, execution, traces


def test_uploaded_no_array_qir_and_selene_flip_only_logical_shot_zero(monkeypatch):
    pytest.importorskip("selene_sim")
    program = HugrTranspiler().transpile(_shot_without_array)
    original = program._compiled.artifact.to_bytes()
    executor, job, client, execution, traces = _uploaded_execution(monkeypatch, program)
    assert [dict(trace.outputs)["qamomile.output.0.0"] for trace in traces] == [
        True,
        False,
        False,
        False,
    ]
    assert dict(job.result().results) == {False: 3, True: 1}
    for raw in (0, 5, _MAX_UINT64):
        code, trace = execution.run(raw)
        assert code == 1
        assert trace.outputs == [
            ("qamomile.output.0.0", False),
            (SHOT_SUCCESS_TAG, False),
        ]
        assert trace.quantum_events == []
    assert program._compiled.artifact.to_bytes() == original
    selene = HugrExecutor("selene", options=SeleneExecutionOptions(seed=7))
    prepared = prepare_execution(program._compiled)
    package_bytes = prepared.package.to_bytes()
    records = selene.submit(prepared.package, _SHOTS).result()
    assert [record[prepared.output_tags[0]] for record in records] == [
        True,
        False,
        False,
        False,
    ]
    assert prepared.package.to_bytes() == package_bytes
    assert program._compiled.artifact.to_bytes() == original
    saved = JobSnapshot.from_dict(json.loads(json.dumps(job.snapshot().to_dict())))
    assert dict(program.restore(executor, saved).result().results) == {
        False: 3,
        True: 1,
    }
    client.qir.upload.assert_called_once()
    client.start_execute_job.assert_called_once()


def test_uploaded_multi_elif_qir_and_selene_preserve_shot_branches(monkeypatch):
    pytest.importorskip("selene_sim")
    program = HugrTranspiler().transpile(
        _multi_elif_shot, bindings={"values": [0, 1, 0, 1]}
    )
    original = program._compiled.artifact.to_bytes()
    _, job, client, execution, traces = _uploaded_execution(monkeypatch, program)
    expected = [True, False, True, False]
    assert [dict(trace.outputs)["qamomile.output.0.0"] for trace in traces] == expected
    assert all(dict(trace.outputs)[SHOT_SUCCESS_TAG] is True for trace in traces)
    assert dict(job.result().results) == {False: 2, True: 2}
    for raw in (0, _SHOTS + 1):
        code, trace = execution.run(raw)
        assert code == 1
        assert trace.outputs == [
            ("qamomile.output.0.0", False),
            (SHOT_SUCCESS_TAG, False),
        ]
        assert trace.quantum_events == []
    assert program._compiled.artifact.to_bytes() == original
    selene = HugrExecutor("selene", options=SeleneExecutionOptions(seed=7))
    prepared = prepare_execution(program._compiled)
    package_bytes = prepared.package.to_bytes()
    records = selene.submit(prepared.package, _SHOTS).result()
    assert [record[prepared.output_tags[0]] for record in records] == expected
    assert prepared.package.to_bytes() == package_bytes
    assert program._compiled.artifact.to_bytes() == original
    client.qir.upload.assert_called_once()
    client.start_execute_job.assert_called_once()


@pytest.mark.parametrize(
    "kernel,expected",
    [
        (_direct_bit, [(False, False), (True, True), (True, True), (False, False)]),
        (
            _table_and_condition,
            [(False, True), (True, False), (True, False), (False, False)],
        ),
        (
            _repeated_shot,
            [
                (False, True, False),
                (True, False, True),
                (True, False, False),
                (False, False, False),
            ],
        ),
    ],
)
def test_uploaded_qir_normalizes_table_conditions_and_repeated_reads_once(
    monkeypatch, kernel, expected
):
    program = HugrTranspiler().transpile(kernel, bindings={"values": _DIRECT})
    before = program._compiled.artifact.to_bytes()
    _, job, client, _, traces = _uploaded_execution(monkeypatch, program)
    for trace, row in zip(traces, expected, strict=True):
        outputs = dict(trace.outputs)
        assert outputs.pop(SHOT_SUCCESS_TAG) is True
        assert (
            tuple(outputs[f"qamomile.output.{index}.0"] for index in range(len(row)))
            == row
        )
    counts = {}
    for row in expected:
        counts[row] = counts.get(row, 0) + 1
    assert dict(job.result().results) == counts
    assert program._compiled.artifact.to_bytes() == before
    client.qir.upload.assert_called_once()
    client.start_execute_job.assert_called_once()


def test_convertible_manual_guard_needs_no_limited_table_proof(monkeypatch):
    monkeypatch.setattr(
        "qamomile.hugr._nexus.prepare_h2_shot_range",
        Mock(side_effect=AssertionError("Already convertible tables need no proof")),
    )
    values = [0] * 16
    values[0] = values[9] = 1
    program = HugrTranspiler().transpile(_manually_guarded, bindings={"values": values})
    _, job, _, _, traces = _uploaded_execution(monkeypatch, program)
    assert [dict(trace.outputs)["qamomile.output.0.0"] for trace in traces] == [
        True,
        False,
        False,
        True,
    ]
    assert dict(job.result().results) == {False: 2, True: 2}


def test_helios_uploads_copied_shot_correction_without_h2_normalization(monkeypatch):
    from qamomile.hugr._nexus import NexusTransport
    from tests.hugr.test_h2_nexus import _client

    client = _client()
    normalize = Mock(side_effect=AssertionError("Helios does not use H2 shot guards"))
    monkeypatch.setattr("qamomile.hugr._nexus.normalize_h2_shots", normalize)
    monkeypatch.setattr(
        "qamomile.hugr.execution._NexusTransport",
        lambda options: NexusTransport(options, client=client),
    )
    prepared = prepare_execution(HugrTranspiler().compile(_shot_without_array))
    before = prepared.package.to_bytes()
    executor = HugrExecutor(
        "nexus", options=NexusExecutionOptions(system_name="Helios-1E", max_cost=1.0)
    )
    executor.submit(prepared.package, _SHOTS)
    uploaded = client.hugr.upload.call_args.kwargs["hugr_package"]
    assert uploaded is not prepared.package
    assert uploaded.to_bytes() != before
    assert SHOT_SUCCESS_TAG.encode() not in uploaded.to_bytes()
    assert prepared.package.to_bytes() == before
    normalize.assert_not_called()
    client.qir.upload.assert_not_called()


def test_h2_syntax_checker_rejects_normalized_no_array_program_before_auth(monkeypatch):
    from qamomile.hugr._nexus import NexusTransport
    from tests.hugr.test_h2_nexus import _client

    client = _client()
    prepared = prepare_execution(HugrTranspiler().compile(_shot_without_array))
    transport = NexusTransport(
        NexusExecutionOptions(system_name="H2-1SC"), client=client
    )
    with pytest.raises(ExecutionError, match="synthetic zero results"):
        transport.submit(prepared.package, _SHOTS)
    client.login.assert_not_called()
    client.qir.upload.assert_not_called()
    client.start_execute_job.assert_not_called()
