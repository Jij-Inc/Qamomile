"""Separate-file Qamomile function used by source-linking tests."""

import qamomile as qm


@qm.qkernel
def imported_flip(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    """Flip one state from a separately imported Python module."""

    return qm.x(state)
