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
        region_entries (dict[str, typing.Any]): Pending loop region
            arguments created by ``loop_region_enter`` while this tracer
            captured a loop body, keyed by variable name (values are
            ``_PendingRegionArg`` instances from the frontend
            control-flow module). The loop builder converts them into
            ``RegionArg`` records on the loop operation.
        loop_region_results (dict[str, typing.Any]): Post-loop result
            handles published by the most recently closed loop traced
            under this tracer, keyed by variable name. Consumed (popped)
            by the AST-injected ``loop_region_result`` assignments that
            run immediately after the loop's ``with`` block exits.
    """

    _operations: list[Operation] = dataclasses.field(default_factory=list)
    loop_carried_rebinds: tuple[LoopCarriedRebind, ...] = ()
    region_entries: dict[str, typing.Any] = dataclasses.field(default_factory=dict)
    loop_region_results: dict[str, typing.Any] = dataclasses.field(default_factory=dict)

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
