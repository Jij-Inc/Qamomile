"""Build abstract device shot-context reads during qkernel tracing."""

from qamomile.circuit.frontend.handle.primitives import UInt
from qamomile.circuit.frontend.tracer import get_current_tracer
from qamomile.circuit.ir.operation.shot import GetCurrentShotOperation
from qamomile.circuit.ir.types.primitives import UIntType
from qamomile.circuit.ir.value import Value


def get_current_shot() -> UInt:
    """Read the zero-based shot index within the current program execution.

    Repeated reads in one shot return the same value. HUGR targets implement
    this device context; circuit SDK targets reject it during compilation.
    Local Selene starts at zero for each sample submission. Nexus forwards
    native indices; provider-internal splitting and retries are not assigned
    Qamomile offsets. Restoring a job retrieves existing results without
    another submission.

    Returns:
        UInt: Device-supplied index, suitable for runtime classical operations
            and fixed-size classical parameter-table selection.

    Raises:
        RuntimeError: If called outside an active qkernel trace.
    """
    tracer = get_current_tracer()
    result = UInt(value=Value(type=UIntType(), name="current_shot"))
    tracer.add_operation(GetCurrentShotOperation(results=[result.value]))
    return result
