"""Cross-backend regressions for RegionArgs inside controlled blocks."""

from __future__ import annotations

from typing import Any

import pytest

import qamomile.circuit as qmc

BACKENDS = [
    pytest.param("qiskit", id="qiskit"),
    pytest.param("quri_parts", marks=pytest.mark.quri_parts, id="quri_parts"),
    pytest.param("cudaq", marks=pytest.mark.cudaq, id="cudaq"),
]


def _make_transpiler(backend: str) -> Any:
    """Build one installed backend transpiler or skip its test."""
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


@qmc.qkernel
def _carried_index_body(
    targets: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Flip successive targets and then consume the final RegionArg result."""
    index = qmc.uint(0)
    for _iteration in qmc.range(2):
        targets[index] = qmc.x(targets[index])
        index = index + 1
    targets[index - 1] = qmc.x(targets[index - 1])
    return targets


@qmc.qkernel
def _nested_carried_index_body(
    targets: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Carry an index through nested statically replayed loops."""
    index = qmc.uint(0)
    for _outer in qmc.range(2):
        for _inner in qmc.range(1):
            targets[index] = qmc.x(targets[index])
            index = index + 1
    return targets


def _controlled_sample_kernel(body: Any) -> Any:
    """Build a sample kernel applying ``body`` under an enabled control."""
    controlled = qmc.control(body)

    @qmc.qkernel
    def kernel() -> qmc.Vector[qmc.Bit]:
        """Enable the control, apply the body, and measure its targets."""
        control = qmc.x(qmc.qubit("control"))
        targets = qmc.qubit_array(2, "targets")
        control, targets = controlled(control, targets)
        return qmc.measure(targets)

    return kernel


def _sample(backend: str, body: Any) -> tuple[int, ...]:
    """Execute one deterministic controlled body on ``backend``."""
    transpiler = _make_transpiler(backend)
    executable = transpiler.transpile(_controlled_sample_kernel(body))
    sampled = executable.sample(transpiler.executor(), shots=16).result()
    assert len(sampled.results) == 1
    value, count = sampled.results[0]
    assert count == 16
    return value


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (_carried_index_body, (1, 0)),
        (_nested_carried_index_body, (1, 1)),
    ],
)
def test_controlled_region_args_execute_across_backends(
    backend: str,
    body: Any,
    expected: tuple[int, ...],
) -> None:
    """Controlled loop carries advance on every nested static iteration."""
    assert _sample(backend, body) == expected


def test_qiskit_controlled_fallback_threads_region_args(monkeypatch) -> None:
    """The generic mapped walker threads carries when native boxing is off."""
    pytest.importorskip("qiskit")
    from qamomile.qiskit.transpiler import QiskitEmitPass

    monkeypatch.setattr(
        QiskitEmitPass,
        "_blockvalue_to_gate",
        lambda self, *args, **kwargs: None,
    )

    assert _sample("qiskit", _nested_carried_index_body) == (1, 1)
