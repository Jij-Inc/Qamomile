"""Check abstract shot context construction and unsupported circuit targets."""

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.frontend.composite_gate import configure_composite
from qamomile.circuit.frontend.tracer import trace
from qamomile.circuit.ir.operation import GetCurrentShotOperation
from qamomile.circuit.ir.operation.callable import CallableImplementation
from qamomile.circuit.ir.types.primitives import UIntType
from qamomile.circuit.transpiler import TargetCapabilityError


def test_context_read_is_abstract_and_requires_tracing():
    """Each call emits one nonconstant UInt result without operands."""
    with pytest.raises(RuntimeError, match="No active tracer"):
        qmc.get_current_shot()
    with trace() as tracer:
        result = qmc.get_current_shot()
    assert isinstance(result, qmc.UInt)
    assert not result.value.is_constant()
    [operation] = tracer.operations
    assert isinstance(operation, GetCurrentShotOperation)
    assert operation.operands == []
    assert operation.signature.results[0].type == UIntType()


@qmc.qkernel
def _uses_context() -> qmc.UInt:
    """Read the current repetition without requiring any quantum allocation."""
    return qmc.get_current_shot()


@qmc.qkernel
def _calls_context() -> qmc.UInt:
    """Preserve the read behind an ordinary callable boundary."""
    return _uses_context()


@pytest.mark.parametrize("kernel", [_uses_context, _calls_context])
@pytest.mark.parametrize(
    "engine",
    [
        "qiskit",
        pytest.param("quri_parts", marks=pytest.mark.quri_parts),
        pytest.param("cudaq", marks=pytest.mark.cudaq),
        pytest.param("braket", marks=pytest.mark.braket),
    ],
)
def test_circuit_targets_reject_shot_context_before_emission(kernel, engine):
    """Circuit targets diagnose the missing device capability, including callees."""
    import importlib

    modules = {
        "qiskit": "qiskit",
        "quri_parts": "quri_parts",
        "cudaq": "cudaq",
        "braket": "braket",
    }
    pytest.importorskip(modules[engine])
    names = {
        "qiskit": "QiskitTranspiler",
        "quri_parts": "QuriPartsTranspiler",
        "cudaq": "CudaqTranspiler",
        "braket": "BraketTranspiler",
    }
    module = importlib.import_module(f"qamomile.{engine}")
    transpiler = getattr(module, names[engine])()
    with pytest.raises(
        TargetCapabilityError, match=r"does not support get_current_shot.*HUGR"
    ):
        transpiler.transpile(kernel)


@qmc.qkernel
def _optional_context(q: qmc.Qubit, enabled: qmc.UInt) -> qmc.Qubit:
    """Read unsupported context only in an enabled compile-time branch."""
    if enabled == 1:
        if qmc.get_current_shot() == 1:
            q = qmc.x(q)
    return q


@qmc.qkernel
def _conditional_context(enabled: qmc.UInt) -> qmc.Bit:
    """Prune a disabled context read in the entrypoint."""
    q = qmc.qubit("q")
    if enabled == 1:
        if qmc.get_current_shot() == 1:
            q = qmc.x(q)
    return qmc.measure(q)


@qmc.qkernel
def _nested_conditional_context(enabled: qmc.UInt) -> qmc.Bit:
    """Prune a disabled context read in a called kernel."""
    q = qmc.qubit("q")
    q = _optional_context(q, enabled)
    return qmc.measure(q)


@qmc.qkernel
def _boxed_optional_context(q: qmc.Qubit, enabled: qmc.UInt) -> qmc.Qubit:
    """Keep compile-time selection behind a preserved callable boundary."""
    return _optional_context(q, enabled + 0)


configure_composite(_boxed_optional_context)


@qmc.qkernel
def _boxed_conditional_context(enabled: qmc.UInt) -> qmc.Bit:
    """Bind a preserved callable's structural input from the entrypoint."""
    q = qmc.qubit("q")
    q = _boxed_optional_context(q, enabled)
    return qmc.measure(q)


@qmc.qkernel
def _zero_trip_context(enabled: qmc.UInt) -> qmc.Bit:
    """Skip context reads in a statically empty loop."""
    q = qmc.qubit("q")
    for i in qmc.range(enabled):
        if qmc.get_current_shot() == 1:
            q = qmc.x(q)
    return qmc.measure(q)


@qmc.qkernel
def _portable_default(q: qmc.Qubit) -> qmc.Qubit:
    """Provide a circuit-compatible default implementation."""
    return qmc.x(q)


@qmc.qkernel
def _hugr_context_implementation(q: qmc.Qubit) -> qmc.Qubit:
    """Use shot context only in a HUGR implementation."""
    if qmc.get_current_shot() == 1:
        q = qmc.x(q)
    return q


configure_composite(
    _portable_default,
    implementations=[
        CallableImplementation(engine="hugr", body=_hugr_context_implementation.block)
    ],
)
configure_composite(_hugr_context_implementation)


@qmc.qkernel
def _calls_portable_default() -> qmc.Bit:
    """Select the default body rather than an unrelated backend implementation."""
    q = qmc.qubit("q")
    q = _portable_default(q)
    return qmc.measure(q)


@qmc.qkernel
def _calls_boxed_context() -> qmc.Bit:
    """Retain an unsupported read behind a preserved composite boundary."""
    q = qmc.qubit("q")
    q = _hugr_context_implementation(q)
    return qmc.measure(q)


@pytest.mark.parametrize(
    "kernel",
    [
        _conditional_context,
        _nested_conditional_context,
        _boxed_conditional_context,
        _zero_trip_context,
    ],
)
def test_disabled_context_branch_is_pruned_before_capability_check(kernel):
    """Compile-time-disabled reads do not block circuit execution."""
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    transpiler = QiskitTranspiler()
    p = transpiler.transpile(kernel, bindings={"enabled": 0})
    assert p.sample(transpiler.executor(), shots=2).result().results == [(False, 2)]
    with pytest.raises(
        TargetCapabilityError, match="does not support get_current_shot"
    ):
        transpiler.transpile(kernel, bindings={"enabled": 1})


def test_unselected_backend_implementation_does_not_require_its_capabilities():
    """A HUGR-only shot read does not prevent the portable default from running."""
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    transpiler = QiskitTranspiler()
    p = transpiler.transpile(_calls_portable_default)
    assert p.sample(transpiler.executor(), shots=2).result().results == [(True, 2)]


def test_reachable_boxed_context_is_still_rejected():
    """Selected callable bodies retain the actionable unsupported-target error."""
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    with pytest.raises(
        TargetCapabilityError, match="does not support get_current_shot"
    ):
        QiskitTranspiler().transpile(_calls_boxed_context)
