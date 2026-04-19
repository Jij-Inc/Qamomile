"""Tests for the self-recursive @qkernel diagnostic.

A @qkernel whose body calls itself by name used to run into Python's
recursion limit because the AST-transformed DSL's exec namespace
shadowed the QKernel with the raw DSL function, causing self-calls to
bypass ``__call__`` and retrace the body in-place forever.

The two fixes together:

1. ``QKernel.__init__`` rebinds the DSL's ``__globals__`` entry for the
   kernel's own name to ``self``, so self-calls go through ``__call__``.
2. ``QKernel.block`` uses a ``_block_building`` flag to detect re-entry
   during construction and raise a clear ``FrontendTransformError``
   instead of a cryptic ``RecursionError``.
"""

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.transpiler.errors import FrontendTransformError


@qmc.qkernel
def _leaf(q: qmc.Qubit) -> qmc.Qubit:
    return qmc.h(q)


@qmc.qkernel
def _rec(k: qmc.UInt, q: qmc.Qubit) -> qmc.Qubit:
    if k == 0:
        q = _leaf(q)
    else:
        q = _rec(k - 1, q)
    return q


@qmc.qkernel
def _outer_of_rec(k: qmc.UInt) -> qmc.Bit:
    q = qmc.qubit(name="q")
    q = _rec(k, q)
    return qmc.measure(q)


def test_self_recursive_kernel_raises_frontend_transform_error():
    with pytest.raises(FrontendTransformError, match="Self-recursive @qkernel"):
        _outer_of_rec.build(k=1)


def test_self_recursive_kernel_error_is_actionable():
    with pytest.raises(FrontendTransformError) as exc_info:
        _outer_of_rec.build(k=1)

    msg = str(exc_info.value)
    # Name of the offending kernel must appear.
    assert "_rec" in msg
    # Actionable guidance pointing at the Python-level recursion pattern.
    assert "Python level" in msg
