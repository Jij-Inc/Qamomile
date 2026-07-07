import contextvars
import dataclasses
import typing
from contextlib import contextmanager

from qamomile.circuit.ir.operation.control_flow import LoopCarriedRebind
from qamomile.circuit.ir.operation.operation import Operation


@dataclasses.dataclass
class Tracer:
    """Collects operations (and loop-rebind records) during tracing.

    Attributes:
        _operations (list[Operation]): Operations captured in trace order.
        loop_carried_rebinds (tuple[LoopCarriedRebind, ...]): Classical
            scalar rebinds recorded by ``record_loop_rebinds`` while this
            tracer captured a loop body. The loop builder copies them onto
            the loop operation after the body trace completes.
        promoted_carry_inits (dict[str, typing.Any]): Initial Python
            values of the plain-number loop-carry candidates promoted by
            ``promote_loop_carry`` while this tracer captured a loop
            body, keyed by the promoted symbolic ``Value``'s UUID. The
            loop builder reads these to synthesize the carry's
            ``iter_arg`` constants.
    """

    _operations: list[Operation] = dataclasses.field(default_factory=list)
    loop_carried_rebinds: tuple[LoopCarriedRebind, ...] = ()
    promoted_carry_inits: dict[str, typing.Any] = dataclasses.field(
        default_factory=dict
    )

    @property
    def operations(self) -> list[Operation]:
        return self._operations

    def add_operation(self, op) -> None:
        self._operations.append(op)


_current_tracer: contextvars.ContextVar["Tracer | None"] = contextvars.ContextVar(
    "qamomile_tracer", default=None
)


def get_current_tracer() -> "Tracer":
    tracer = _current_tracer.get()
    if tracer is None:
        raise RuntimeError("No active tracer found in context")
    return tracer


@contextmanager
def trace(tracer: Tracer | None = None) -> typing.Generator[Tracer, None, None]:
    """Context manager to set the current tracer."""
    if tracer is None:
        tracer = Tracer()
    token = _current_tracer.set(tracer)
    try:
        yield tracer
    finally:
        _current_tracer.reset(token)
