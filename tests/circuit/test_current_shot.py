"""Check abstract shot context construction and unsupported circuit targets."""

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.frontend.tracer import trace
from qamomile.circuit.ir.operation import GetCurrentShotOperation
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
