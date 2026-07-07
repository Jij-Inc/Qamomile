"""Cross-backend execution tests for loop-carried classical values.

Loop-carry slots are resolved by two different executors depending on
where the loop lands: the classical segment interpreter (block-output
reductions) and emit-time unrolling (gate parameters). Both paths must
produce Python semantics on every supported SDK backend, so each case
transpiles AND executes, verifying the classical outcome value.

The quri_parts / cudaq variants are marker-gated (deselected by the
default ``uv run pytest`` run) and skip when the SDK is missing.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

import qamomile.circuit as qmc

BACKENDS = [
    pytest.param("qiskit", id="qiskit"),
    pytest.param("quri_parts", marks=pytest.mark.quri_parts, id="quri_parts"),
    pytest.param("cudaq", marks=pytest.mark.cudaq, id="cudaq"),
]


def _make_transpiler(backend: str) -> Any:
    """Build the requested backend transpiler, skipping when unavailable.

    Args:
        backend (str): One of ``"qiskit"`` / ``"quri_parts"`` / ``"cudaq"``.

    Returns:
        Any: The backend transpiler instance.
    """
    if backend == "qiskit":
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        return QiskitTranspiler()
    if backend == "quri_parts":
        pytest.importorskip("quri_parts")
        pytest.importorskip("quri_parts.qulacs")
        from qamomile.quri_parts import QuriPartsTranspiler

        return QuriPartsTranspiler()
    pytest.importorskip("cudaq")
    from qamomile.cudaq import CudaqTranspiler

    return CudaqTranspiler()


def _sample_single(backend: str, kernel: Any, bindings: dict[str, Any]) -> Any:
    """Transpile and sample a kernel, returning the deterministic outcome.

    Args:
        backend (str): Backend name accepted by :func:`_make_transpiler`.
        kernel (Any): The qkernel to run.
        bindings (dict[str, Any]): Compile-time bindings.

    Returns:
        Any: The single deterministic sampled outcome.
    """
    transpiler = _make_transpiler(backend)
    executable = transpiler.transpile(kernel, bindings=bindings)
    result = executable.sample(transpiler.executor(), shots=100).result()
    assert len(result.results) == 1, f"expected deterministic result: {result}"
    return result.results[0][0]


@qmc.qkernel
def sum_kernel(n: qmc.UInt) -> qmc.UInt:
    q = qmc.qubit("q")
    qmc.measure(q)
    total = qmc.uint(0)
    for i in qmc.range(n):
        total = total + i
    return total


@qmc.qkernel
def swap_kernel(n: qmc.UInt) -> tuple[qmc.UInt, qmc.UInt]:
    q = qmc.qubit("q")
    qmc.measure(q)
    a = qmc.uint(1)
    b = qmc.uint(2)
    for _i in qmc.range(n):
        a, b = b, a
    return a, b


@qmc.qkernel
def angle_kernel(n: qmc.UInt) -> qmc.Bit:
    total = 0.0
    for _i in qmc.range(n):
        total = total + math.pi
    q = qmc.qubit("q")
    q = qmc.rx(q, total)
    return qmc.measure(q)


@pytest.mark.parametrize("backend", BACKENDS)
class TestLoopCarriedValuesAcrossBackends:
    """Carried loops execute with Python semantics on every backend."""

    def test_classical_segment_accumulation(self, backend):
        """sum(range(4)) == 6 through the classical segment interpreter."""
        assert _sample_single(backend, sum_kernel, {"n": 4}) == 6

    def test_classical_segment_swap(self, backend):
        """Three swaps of (1, 2) land on (2, 1)."""
        assert _sample_single(backend, swap_kernel, {"n": 3}) == (2, 1)

    def test_emit_absorbed_angle_accumulation(self, backend):
        """An accumulated rx angle of 3*pi flips |0> deterministically.

        The carried loop feeds a gate parameter, so it is absorbed into
        the quantum segment and evaluated by emit-time unrolling.
        """
        assert _sample_single(backend, angle_kernel, {"n": 3}) == 1
