"""Calls that a Qamomile QKernel must reject during static analysis."""

import qamomile as qm


@qm.qkernel
def add(left: int, right: int) -> int:
    """Return a scalar sum inside Qamomile source."""

    return left + right


@qm.qkernel(kind="classical")
def choose(value: float, *, enabled: bool = True) -> float:
    """Return a value selected by a keyword-only source parameter."""

    if enabled:
        return value
    return 0.0


add("one", 2)
add(1)
choose(1.0, enabled="yes")
text: str = add(1, 2)
