"""Small source-backed callables shared by functional library factories."""

from __future__ import annotations

import qamomile as qm


@qm.qkernel
def flip(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:  # type: ignore[valid-type]
    """Apply a bit flip as a first-class unary callable."""

    return qm.x(state)


__all__ = ["flip"]
