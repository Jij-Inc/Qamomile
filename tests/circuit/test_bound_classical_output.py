"""Regression tests for compile-time-constant block outputs.

A block output whose value is fixed at compile time — a ``bindings``-bound
classical argument returned directly, or a classical expression fully folded
by ``partial_eval`` such as ``return x * 2.0`` with ``x`` bound — has no
producing operation left in the IR after constant folding. It therefore never
enters the runtime execution context, and previously ``sample`` / ``run``
returned ``None`` for that output instead of the constant.

These tests pin the fix across every supported local SDK backend (Qiskit,
QURI Parts, CUDA-Q): the segmentation pass records such outputs in
``ProgramABI.output_constants`` and ``ProgramOrchestrator._resolve_outputs``
falls back to that map. Both the computed and pass-through forms are covered,
together with the ``parameters=[...]`` (runtime) variant that must keep
resolving from the execution context.
"""

from __future__ import annotations

from typing import Any

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.transpiler.errors import ExecutionError


@pytest.fixture(
    params=[
        "qiskit",
        pytest.param("quri_parts", marks=pytest.mark.quri_parts),
        pytest.param("cudaq", marks=pytest.mark.cudaq),
    ]
)
def backend(request) -> tuple[str, Any, Any]:
    """Yield ``(name, transpiler, executor)`` for each installed SDK backend."""
    name = request.param
    if name == "qiskit":
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        transpiler = QiskitTranspiler()
        return name, transpiler, transpiler.executor()
    if name == "quri_parts":
        pytest.importorskip("quri_parts")
        pytest.importorskip("quri_parts.qulacs")
        from qamomile.quri_parts import QuriPartsTranspiler

        transpiler = QuriPartsTranspiler()
        return name, transpiler, transpiler.executor()
    if name == "cudaq":
        pytest.importorskip("cudaq")
        from qamomile.cudaq import CudaqTranspiler

        transpiler = CudaqTranspiler()
        return name, transpiler, transpiler.executor()
    raise AssertionError(f"unknown backend {name}")


@qmc.qkernel
def _computed_output(
    x: qmc.Float, n: qmc.UInt
) -> tuple[qmc.Float, qmc.Vector[qmc.Bit]]:
    """Return a folded classical expression alongside a measured register."""
    y = x * 2.0
    qs = qmc.qubit_array(n, name="qs")
    qs = qmc.h(qs)
    bits = qmc.measure(qs)
    return y, bits


@qmc.qkernel
def _passthrough_output(
    x: qmc.Float, n: qmc.UInt
) -> tuple[qmc.Float, qmc.Vector[qmc.Bit]]:
    """Return a bound classical argument directly alongside a measured register."""
    qs = qmc.qubit_array(n, name="qs")
    qs = qmc.h(qs)
    bits = qmc.measure(qs)
    return x, bits


@pytest.mark.parametrize("x", [0.0, 1.5, 3.0, -0.8])
@pytest.mark.parametrize("n", [1, 2, 3])
def test_computed_bound_output_survives_fold(
    backend: tuple[str, Any, Any], x: float, n: int
) -> None:
    """A folded ``x * 2.0`` output resolves to the constant via sample and run."""
    name, transpiler, executor = backend
    executable = transpiler.transpile(_computed_output, bindings={"x": x, "n": n})

    run_out = executable.run(executor).result()
    assert run_out[0] == pytest.approx(x * 2.0), name
    assert len(run_out[1]) == n, name

    sample = executable.sample(executor, shots=32).result()
    assert sample.results, f"{name}: sampler returned no counts"
    for output, _count in sample.results:
        assert output[0] == pytest.approx(x * 2.0), name
        assert len(output[1]) == n, name


@pytest.mark.parametrize("x", [0.0, 2.0, 7.0, -1.25])
@pytest.mark.parametrize("n", [1, 2, 3])
def test_passthrough_bound_output_survives_fold(
    backend: tuple[str, Any, Any], x: float, n: int
) -> None:
    """A directly returned bound argument resolves to its value, not ``None``."""
    name, transpiler, executor = backend
    executable = transpiler.transpile(_passthrough_output, bindings={"x": x, "n": n})

    run_out = executable.run(executor).result()
    assert run_out[0] == pytest.approx(x), name
    assert len(run_out[1]) == n, name

    sample = executable.sample(executor, shots=32).result()
    assert sample.results, f"{name}: sampler returned no counts"
    for output, _count in sample.results:
        assert output[0] == pytest.approx(x), name
        assert len(output[1]) == n, name


@pytest.mark.parametrize("x", [0.0, 3.0, -2.5])
def test_runtime_parameter_output_still_resolves(
    backend: tuple[str, Any, Any], x: float
) -> None:
    """A runtime-parameter output keeps resolving from the execution context.

    When ``x`` is a runtime parameter the classical expression is not folded;
    it materializes through a classical segment. This must remain correct so
    the compile-time-constant fallback does not shadow the context path.
    """
    name, transpiler, executor = backend
    computed = transpiler.transpile(
        _computed_output, bindings={"n": 2}, parameters=["x"]
    )
    run_out = computed.run(executor, bindings={"x": x}).result()
    assert run_out[0] == pytest.approx(x * 2.0), name

    passthrough = transpiler.transpile(
        _passthrough_output, bindings={"n": 2}, parameters=["x"]
    )
    run_out = passthrough.run(executor, bindings={"x": x}).result()
    assert run_out[0] == pytest.approx(x), name


def test_output_constants_recorded_in_abi() -> None:
    """The segmentation ABI records folded scalar outputs as constants."""
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    transpiler = QiskitTranspiler()
    executable = transpiler.transpile(_computed_output, bindings={"x": 3.0, "n": 2})

    abi = executable.plan.abi
    # The first output ref (the folded Float) is a compile-time constant; the
    # second (the measured Vector) is not and resolves from the context.
    float_ref = abi.output_refs[0]
    assert abi.output_constants[float_ref] == pytest.approx(6.0)
    assert abi.output_refs[1] not in abi.output_constants


def test_unresolvable_output_raises_execution_error() -> None:
    """An output ref with no context value or constant raises ``ExecutionError``.

    Guards the diagnostic path: rather than silently returning ``None`` for a
    block output the pipeline failed to materialize, ``_resolve_outputs`` now
    raises so the miscompilation surfaces loudly.
    """
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    transpiler = QiskitTranspiler()
    executable = transpiler.transpile(_passthrough_output, bindings={"x": 3.0, "n": 2})

    # Corrupt the ABI so the pass-through output can be resolved neither from
    # the context nor from the recorded constants, mimicking a compiler bug.
    executable.plan.abi.output_constants = {}

    with pytest.raises(ExecutionError, match="could not be resolved"):
        executable.run(transpiler.executor()).result()
