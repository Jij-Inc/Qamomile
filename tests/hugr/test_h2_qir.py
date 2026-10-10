"""Exercise real offline H2 conversion and the public typed execution path."""

from __future__ import annotations

import hashlib
import math
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.transpiler import JobSnapshot
from qamomile.circuit.transpiler.errors import ExecutionError
from qamomile.hugr import HugrExecutor, HugrTranspiler, NexusExecutionOptions
from qamomile.hugr._nexus import NexusTransport
from qamomile.hugr._qir import to_h2_qir
from qamomile.hugr._runtime import prepare_execution
from tests.hugr.test_execution import _complex, _integer, _rotation
from tests.hugr.test_expval import _rotated
from tests.hugr.test_nexus import _client
from tests.hugr.test_runtime import _runtime_bit_order

pytestmark = pytest.mark.hugr


@qmc.qkernel
def _bell() -> tuple[qmc.Bit, qmc.Bit]:
    """Return two correlated measurements from a Bell state.

    Returns:
        tuple[qmc.Bit, qmc.Bit]: Measurements in public qubit order.
    """
    left = qmc.qubit("left")
    right = qmc.qubit("right")
    left = qmc.h(left)
    left, right = qmc.cx(left, right)
    return qmc.measure(left), qmc.measure(right)


def _require_converter():
    """Skip real conversion checks when the optional converter is absent.

    Returns:
        ModuleType: Installed PyQIR module for inspecting bitcode.
    """
    pytest.importorskip("hugr_qir.hugr_to_qir")
    return pytest.importorskip("pyqir")


@pytest.mark.parametrize("kernel", [_bell, _runtime_bit_order])
def test_real_qir_conversion_preserves_all_public_tags_and_source(kernel):
    """Real QIR validation preserves tuple and array output labels without mutation."""
    pyqir = _require_converter()
    compiled = HugrTranspiler().compile(kernel)
    original = compiled.artifact.to_bytes()
    prepared = prepare_execution(compiled)
    before = prepared.package.to_bytes()
    bitcode, outputs = to_h2_qir(prepared.package)
    assert bitcode.startswith(b"BC\xc0\xde")
    llvm = str(pyqir.Module.from_bitcode(pyqir.Context(), bitcode))
    assert set(outputs) == set(prepared.output_tags)
    assert set(outputs.values()) == {("bool", 1)}
    for tag in prepared.output_tags:
        assert tag in llvm
    assert "__quantum__rt__bool_record_output" in llvm
    assert '"entry_point"' in llvm
    assert prepared.package.to_bytes() == before
    assert compiled.artifact.to_bytes() == original


def test_real_conversion_resolves_runtime_bindings_without_changing_body():
    """Distinct runtime angles produce distinct valid QIR from one compiled body."""
    _require_converter()
    compiled = HugrTranspiler().compile(_rotation, parameters=["theta"])
    original = compiled.artifact.to_bytes()
    bitcodes = [
        to_h2_qir(prepare_execution(compiled, {"theta": angle}).package)[0]
        for angle in (0.0, math.pi)
    ]
    assert bitcodes[0] != bitcodes[1]
    assert compiled.artifact.to_bytes() == original


@pytest.mark.parametrize(
    "kernel,bindings,kind",
    [(_complex, {"theta": 0.0, "label": 0}, "Float"), (_integer, {}, "UInt")],
)
def test_unsupported_h2_outputs_fail_before_loading_converter(
    monkeypatch, kernel, bindings, kind
):
    """Unsupported output types fail locally even without converter installation."""
    prepared = prepare_execution(HugrTranspiler().compile(kernel, bindings=bindings))
    load = Mock(side_effect=AssertionError("Converter must not be loaded"))
    monkeypatch.setattr("qamomile.hugr._qir.importlib.import_module", load)
    with pytest.raises(ExecutionError, match=kind):
        to_h2_qir(prepared.package)
    load.assert_not_called()


def test_missing_converter_has_actionable_installation_error(monkeypatch):
    """H2 explains its optional dependency without changing Helios installation."""
    prepared = prepare_execution(HugrTranspiler().compile(_bell))
    monkeypatch.setattr(
        "qamomile.hugr._qir.importlib.import_module",
        Mock(side_effect=ModuleNotFoundError("hugr_qir")),
    )
    with pytest.raises(ImportError, match=r"qamomile\[hugr,hugr-qir\]"):
        to_h2_qir(prepared.package)


def test_unsupported_python_version_has_actionable_error(monkeypatch):
    """Unsupported Python versions fail locally before importing the validator."""
    prepared = prepare_execution(HugrTranspiler().compile(_bell))
    monkeypatch.setattr("qamomile.hugr._qir.sys.version_info", (3, 13, 0))
    with pytest.raises(ImportError, match="Python 3.11 or 3.12"):
        to_h2_qir(prepared.package)


@pytest.mark.parametrize("value", [b"not-bitcode", "text-instead-of-bytes", None])
def test_invalid_converter_output_is_rejected(monkeypatch, value):
    """A malformed conversion result cannot be uploaded as executable bitcode."""
    prepared = prepare_execution(HugrTranspiler().compile(_bell))
    converter = SimpleNamespace(hugr_to_qir=Mock(return_value=value))
    output = SimpleNamespace(OutputFormat=SimpleNamespace(BITCODE="bitcode"))
    monkeypatch.setattr(
        "qamomile.hugr._qir.importlib.import_module",
        Mock(side_effect=[converter, output]),
    )
    with pytest.raises(ExecutionError, match="LLVM bitcode"):
        to_h2_qir(prepared.package)
    assert converter.hugr_to_qir.call_args.kwargs["validate_qir"] is True
    assert converter.hugr_to_qir.call_args.kwargs["validate_hugr"] is True


def _install_h2_client(monkeypatch, tags, readouts):
    """Use real SDK result storage with an entirely local submission client.

    Args:
        monkeypatch (pytest.MonkeyPatch): Scoped transport replacement.
        tags (list[str]): Register labels in provider storage order.
        readouts (list[list[int]]): Measured bits in that storage order.

    Returns:
        SimpleNamespace: Nexus-shaped client recording uploads and submissions.
    """
    from pytket.backends.backendresult import BackendResult
    from pytket.circuit import Bit
    from pytket.utils.outcomearray import OutcomeArray

    raw = BackendResult(
        c_bits=[Bit(tag, 0) for tag in tags],
        shots=OutcomeArray.from_readouts(np.asarray(readouts, dtype=np.uint8)),
    )
    client = _client()
    client.QuantinuumConfig = Mock(
        side_effect=lambda **kwargs: SimpleNamespace(type="QuantinuumConfig", **kwargs)
    )
    client.qir = SimpleNamespace(upload=Mock(return_value="uploaded-qir"))
    client.jobs.results.return_value[0].download_result.return_value = raw
    monkeypatch.setattr(
        "qamomile.hugr.execution._NexusTransport",
        lambda options: NexusTransport(options, client=client),
    )
    return client


def test_real_converter_typed_array_sample_and_restore(monkeypatch):
    """Real conversion plus shuffled provider columns preserves typed sample order."""
    _require_converter()
    pytest.importorskip("pytket")
    tags = ["qamomile.output.0.2", "qamomile.output.0.0", "qamomile.output.0.1"]
    client = _install_h2_client(monkeypatch, tags, [[0, 1, 0], [1, 0, 0]])
    program = HugrTranspiler().transpile(_runtime_bit_order)
    executor = HugrExecutor("nexus", options=NexusExecutionOptions(system_name="H2-1E"))
    job = program.sample(executor, shots=2)
    result = job.result()
    assert result.results == [((True, False, False), 1), ((False, False, True), 1)]
    saved = JobSnapshot.from_dict(job.snapshot().to_dict())
    assert (
        program.restore(HugrExecutor("nexus"), saved).result().results == result.results
    )
    client.qir.upload.assert_called_once()
    client.hugr.upload.assert_not_called()
    client.start_execute_job.assert_called_once()
    assert client.qir.upload.call_args.kwargs["qir"].startswith(b"BC\xc0\xde")
    reference = saved.executions[0]
    expected = prepare_execution(program._compiled).package.to_bytes()
    assert reference.context["package_sha256"] == hashlib.sha256(expected).hexdigest()


def test_real_converter_expectation_uses_bit_outputs_and_restores(monkeypatch):
    """Expectation Float results are computed from H2 bits on the host."""
    _require_converter()
    pytest.importorskip("pytket")
    client = _install_h2_client(monkeypatch, ["qamomile.output.0.0"], [[0], [1], [0]])
    program = HugrTranspiler().transpile(
        _rotated, bindings={"observable": qm_o.Z(0)}, parameters=["theta"]
    )
    executor = HugrExecutor("nexus", options=NexusExecutionOptions(system_name="H2-1E"))
    job = program.run(executor, bindings={"theta": 0.25}, shots=3)
    assert job.result() == pytest.approx(1 / 3)
    saved = JobSnapshot.from_dict(job.snapshot().to_dict())
    assert program.restore(executor, saved, {"theta": 0.25}).result() == pytest.approx(
        1 / 3
    )
    client.qir.upload.assert_called_once()
    client.start_execute_job.assert_called_once()
