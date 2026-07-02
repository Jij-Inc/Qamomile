"""Cross-backend execution tests for ``Transpiler.transpile_block``.

``transpile_block`` is the entry point for IR that arrives as data —
a ``Block`` reconstructed by ``load_json`` / ``load_msgpack`` in a
different process from the one that traced the kernel (e.g. a workflow
runner executing the quantum node of a larger hybrid program). These
tests round-trip representative kernels through the wire format and
then execute the *deserialized* block on every supported SDK backend,
covering both the sampling and the expectation-value paths.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.serialize import (
    dump_json,
    dump_msgpack,
    load_json,
    load_msgpack,
)
from qamomile.circuit.transpiler.passes.analyze import AnalyzePass
from qamomile.circuit.transpiler.passes.inline import InlinePass

Backend = tuple[str, Any, Any]


@pytest.fixture(
    params=[
        "qiskit",
        pytest.param("quri_parts", marks=pytest.mark.quri_parts),
        pytest.param("cudaq", marks=pytest.mark.cudaq),
    ]
)
def backend(request) -> Backend:
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
def _rx_entangle(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
    """Two-qubit Rx + CX kernel with one runtime rotation parameter."""
    q = qmc.qubit_array(2, "q")
    q[0] = qmc.rx(q[0], theta)
    q[0], q[1] = qmc.cx(q[0], q[1])
    return qmc.measure(q)


@qmc.qkernel
def _rx_layer(thetas: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
    """Rx layer whose register size follows the bound ``thetas`` length."""
    n = thetas.shape[0]
    q = qmc.qubit_array(n, "q")
    for i in qmc.range(n):
        q[i] = qmc.rx(q[i], thetas[i])
    return qmc.measure(q)


@qmc.qkernel
def _rx_expval(theta: qmc.Float, H: qmc.Observable) -> qmc.Float:
    """Single-qubit Rx state prepared for an expectation-value readout."""
    q = qmc.qubit(name="q")
    q = qmc.rx(q, theta)
    return qmc.expval(q, H)  # type: ignore[arg-type]


def _wire_round_trip(kernel: qmc.QKernel, fmt: str) -> Block:
    """Trace ``kernel`` to AFFINE and round-trip it through a wire format.

    Args:
        kernel (qmc.QKernel): The kernel to trace.
        fmt (str): ``"json"`` or ``"msgpack"``.

    Returns:
        Block: The block reconstructed from the wire payload.
    """
    block = InlinePass().run(kernel.block)
    if fmt == "json":
        return load_json(dump_json(block))
    return load_msgpack(dump_msgpack(block))


def _z0_probability(counts: dict[tuple[int, ...], int], theta: float) -> None:
    """Assert the first bit's distribution matches Rx(theta) analytics.

    Args:
        counts (dict[tuple[int, ...], int]): Sample counts keyed by bit
            tuples.
        theta (float): The Rx rotation angle applied to qubit 0.
    """
    shots = sum(counts.values())
    p1 = sum(c for bits, c in counts.items() if bits[0] == 1) / shots
    expected = math.sin(theta / 2.0) ** 2
    assert abs(p1 - expected) < 0.1, f"P(1)={p1:.3f}, expected {expected:.3f}"


def _counts(result: Any) -> dict[tuple[int, ...], int]:
    """Convert backend sample results to a bit-tuple count map.

    Args:
        result (Any): Qamomile ``SampleResult``-like object whose
            ``results`` iterable yields ``(bits, count)`` pairs.

    Returns:
        dict[tuple[int, ...], int]: Counts keyed by kernel-order bit
        tuples.
    """
    counts: dict[tuple[int, ...], int] = {}
    for bits, count in result.results:
        bit_tuple = tuple(int(bit) for bit in bits)
        counts[bit_tuple] = counts.get(bit_tuple, 0) + int(count)
    return counts


@pytest.mark.parametrize("fmt", ["json", "msgpack"])
@pytest.mark.parametrize("seed", [0, 1, 42])
def test_transpile_block_sample_runtime_parameter(
    backend: Backend, fmt: str, seed: int
) -> None:
    """A deserialized block samples correctly with a runtime parameter.

    The entangling kernel makes both bits equal, so the marginal of the
    first bit pins the Rx angle actually used by the backend.
    """
    name, transpiler, executor = backend
    rng = np.random.default_rng(seed)
    theta = float(rng.uniform(0.2, math.pi - 0.2))

    block = _wire_round_trip(_rx_entangle, fmt)
    executable = transpiler.transpile_block(block, parameters=["theta"])
    counts = _counts(
        executable.sample(executor, shots=2048, bindings={"theta": theta}).result()
    )

    assert set(counts).issubset({(0, 0), (1, 1)}), f"{name}: got {counts}"
    _z0_probability(counts, theta)


@pytest.mark.parametrize("fmt", ["json", "msgpack"])
@pytest.mark.parametrize("theta", [0.0, math.pi / 3, math.pi, 2.0 * math.pi])
def test_transpile_block_sample_compile_time_binding(
    backend: Backend, fmt: str, theta: float
) -> None:
    """A deserialized block samples correctly with the angle baked in."""
    name, transpiler, executor = backend

    block = _wire_round_trip(_rx_entangle, fmt)
    executable = transpiler.transpile_block(block, bindings={"theta": theta})
    counts = _counts(executable.sample(executor, shots=2048).result())

    assert set(counts).issubset({(0, 0), (1, 1)}), f"{name}: got {counts}"
    _z0_probability(counts, theta)


@pytest.mark.parametrize("fmt", ["json", "msgpack"])
@pytest.mark.parametrize("n", [1, 2, 3, 5])
@pytest.mark.parametrize("seed", [0, 7])
def test_transpile_block_receiver_side_array_binding(
    backend: Backend, fmt: str, n: int, seed: int
) -> None:
    """The receiving process can bind array parameters after loading.

    The client serializes an *unbound* kernel whose register size is
    symbolic (``thetas.shape[0]``); the receiver supplies the concrete
    array, which must resolve the loop bound and qubit count.
    """
    name, transpiler, executor = backend
    rng = np.random.default_rng(seed)
    thetas = rng.uniform(0.0, 2.0 * math.pi, size=n)

    block = _wire_round_trip(_rx_layer, fmt)
    executable = transpiler.transpile_block(block, bindings={"thetas": thetas})
    counts = _counts(executable.sample(executor, shots=2048).result())

    shots = sum(counts.values())
    for i in range(n):
        p1 = sum(c for bits, c in counts.items() if bits[i] == 1) / shots
        expected = math.sin(thetas[i] / 2.0) ** 2
        assert abs(p1 - expected) < 0.1, (
            f"{name}: qubit {i} P(1)={p1:.3f}, expected {expected:.3f}"
        )


@pytest.mark.parametrize("fmt", ["json", "msgpack"])
@pytest.mark.parametrize("seed", [0, 1, 42])
def test_transpile_block_expval(backend: Backend, fmt: str, seed: int) -> None:
    """A deserialized block reproduces the analytic Rx expectation value.

    ``<Z>`` after ``Rx(theta)`` is ``cos(theta)``. A scalar
    ``Observable`` argument is always a runtime parameter, so the
    receiving side supplies the Hamiltonian via ``bindings`` at
    ``transpile_block`` time — the same contract ``transpile`` uses.
    """
    name, transpiler, executor = backend
    rng = np.random.default_rng(seed)
    theta = float(rng.uniform(0.0, 2.0 * math.pi))

    restored = _wire_round_trip(_rx_expval, fmt)

    executable = transpiler.transpile_block(
        restored, bindings={"H": qm_o.Z(0)}, parameters=["theta"]
    )
    got = executable.run(executor, bindings={"theta": theta}).result()
    assert np.isclose(got, math.cos(theta), atol=1e-5), (
        f"{name}: got {got}, expected {math.cos(theta)}"
    )


@qmc.qkernel
def _trotter_bound(Hs: qmc.Vector[qmc.Observable], t: qmc.Float) -> qmc.Vector[qmc.Bit]:
    """Trotter-style kernel whose observables are bound at build time."""
    q = qmc.qubit_array(1, "q")
    for i in qmc.range(Hs.shape[0]):
        q = qmc.pauli_evolve(q, Hs[i], t)
    return qmc.measure(q)


@pytest.mark.parametrize("fmt", ["json", "msgpack"])
def test_transpile_block_restores_baked_observable_bindings(
    backend: Backend, fmt: str
) -> None:
    """Hamiltonians baked at build time survive serialization to emit.

    A bound ``Vector[Observable]`` is recorded as a
    ``COMPILE_TIME_BOUND`` slot carrying the ``$hamiltonian`` payloads;
    ``transpile_block`` must rebuild the emit-time bindings from that
    manifest without the caller re-supplying them. ``exp(-i * (pi/2) *
    X)`` maps |0> to (-i)|1>, so the single qubit measures 1
    deterministically.
    """
    name, transpiler, executor = backend
    traced = _trotter_bound.build(Hs=[1.0 * qm_o.X(0)])
    block = InlinePass().run(traced)
    restored = (
        load_json(dump_json(block))
        if fmt == "json"
        else load_msgpack(dump_msgpack(block))
    )

    executable = transpiler.transpile_block(restored, bindings={"t": math.pi / 2.0})
    counts = _counts(executable.sample(executor, shots=256).result())
    assert set(counts) == {(1,)}, f"{name}: got {counts}"


def test_transpile_block_rejects_rebinding_baked_slot() -> None:
    """Re-binding a compile-time-bound slot after loading raises."""
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    traced = _rx_layer.build(thetas=np.array([0.1, 0.2]))
    block = InlinePass().run(traced)
    restored = load_json(dump_json(block))

    with pytest.raises(ValueError, match="already compile-time bound"):
        QiskitTranspiler().transpile_block(
            restored, bindings={"thetas": np.array([0.3, 0.4])}
        )
    with pytest.raises(ValueError, match="cannot be turned into a runtime"):
        QiskitTranspiler().transpile_block(restored, parameters=["thetas"])


@pytest.mark.parametrize("fmt", ["json", "msgpack"])
def test_transpile_block_accepts_analyzed_block(backend: Backend, fmt: str) -> None:
    """An ANALYZED block round-trips and compiles like an AFFINE one."""
    name, transpiler, executor = backend
    analyzed = AnalyzePass().run(InlinePass().run(_rx_entangle.block))
    assert analyzed.kind is BlockKind.ANALYZED
    restored = (
        load_json(dump_json(analyzed))
        if fmt == "json"
        else load_msgpack(dump_msgpack(analyzed))
    )

    executable = transpiler.transpile_block(restored, bindings={"theta": math.pi})
    counts = _counts(executable.sample(executor, shots=256).result())
    assert set(counts) == {(1, 1)}, f"{name}: got {counts}"


def test_transpile_block_rejects_binding_parameter_overlap() -> None:
    """A name in both ``bindings`` and ``parameters`` raises ValueError."""
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    block = InlinePass().run(_rx_entangle.block)
    with pytest.raises(ValueError, match="appear in both"):
        QiskitTranspiler().transpile_block(
            block, bindings={"theta": 1.0}, parameters=["theta"]
        )


def test_transpile_block_rejects_hierarchical_block() -> None:
    """A HIERARCHICAL block (not yet inlined) is rejected with guidance."""
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    with pytest.raises(ValueError, match="AFFINE or"):
        QiskitTranspiler().transpile_block(_rx_entangle.block)


def test_transpile_block_rejects_quantum_io_block() -> None:
    """A quantum-I/O block is rejected by entrypoint validation."""
    pytest.importorskip("qiskit")
    from qamomile.circuit.transpiler.errors import EntrypointValidationError
    from qamomile.qiskit import QiskitTranspiler

    @qmc.qkernel
    def quantum_io(q: qmc.Qubit) -> qmc.Qubit:
        return qmc.h(q)

    block = InlinePass().run(quantum_io.block)
    with pytest.raises(EntrypointValidationError):
        QiskitTranspiler().transpile_block(block)
