import typing
import dataclasses
import contextvars
from contextlib import contextmanager

from qamomile.circuit.ir.operation.operation import Operation


@dataclasses.dataclass
class Tracer:
    _operations: list[Operation] = dataclasses.field(default_factory=list)

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
