"""Regression tests: ``expval`` binds the observable to the right physical qubit
for Vector-element qubits.

Before the fix, ``expval((vector[i],), obs)`` resolved the observable's qubit by a
flat ``QubitAddress(uuid)`` lookup with no fallback. An ungated Vector element (or
one re-accessed after a gate/composite) has a UUID that was never registered in the
quantum segment's qubit_map, so the lookup missed, the observable's qubit_map came
back empty, ``remap_qubits`` was skipped, and the Pauli term silently fell back to
physical qubit 0. These tests pin the corrected behaviour and the invariants that
must not regress (whole-Vector, sliced-view, and standalone-qubit forms).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.ir.operation.expval import ExpvalOp
from qamomile.circuit.ir.serialize import dump_json, load_json
from qamomile.circuit.transpiler.passes.inline import InlinePass

_EXPVAL_ATOL = 1e-6


@pytest.fixture(
    params=[
        "qiskit",
        pytest.param("quri_parts", marks=pytest.mark.quri_parts),
        pytest.param("cudaq", marks=pytest.mark.cudaq),
    ]
)
def backend(request):
    """Yield ``(transpiler, executor)`` for each installed quantum SDK backend.

    Args:
        request (pytest.FixtureRequest): Parametrization carrier selecting the
            backend name.

    Returns:
        tuple: ``(transpiler, executor)`` for the selected backend.
    """
    name = request.param
    if name == "qiskit":
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        t = QiskitTranspiler()
        return t, t.executor()
    if name == "quri_parts":
        pytest.importorskip("quri_parts")
        pytest.importorskip("quri_parts.qulacs")
        from qamomile.quri_parts import QuriPartsTranspiler

        t = QuriPartsTranspiler()
        return t, t.executor()
    if name == "cudaq":
        pytest.importorskip("cudaq")
        from qamomile.cudaq import CudaqTranspiler

        t = CudaqTranspiler()
        return t, t.executor()
    raise AssertionError(f"unknown backend {name}")


def _expval(backend, kernel, bindings) -> float:
    """Transpile, run, and return the scalar expectation value.

    Args:
        backend (tuple): ``(transpiler, executor)`` pair from the fixture.
        kernel (qmc.QKernel): The expval kernel to evaluate.
        bindings (dict): Compile-time bindings (observable, bits, angles).

    Returns:
        float: The estimated expectation value.
    """
    transpiler, executor = backend
    exe = transpiler.transpile(kernel, bindings=bindings)
    return float(exe.run(executor).result())


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@qmc.qkernel
def _ungated_vec_ancilla(obs: qmc.Observable) -> qmc.Float:
    """Both clock qubits are ``|1>``; an ungated Vector-element ancilla is observed.

    The clock occupies physical qubits 0..1, so the pre-fix empty-map binding put
    ``Z(0)`` on a ``|1>`` clock qubit (yielding ``-1``); the correct binding reaches
    the ``|0>`` ancilla and yields ``+1``.
    """
    clock = qmc.qubit_array(2, name="clock")
    clock[0] = qmc.x(clock[0])
    clock[1] = qmc.x(clock[1])
    anc_vec = qmc.qubit_array(1, name="anc")
    return qmc.expval((anc_vec[0],), obs)


@qmc.qkernel
def _gated_vec_ancilla(obs: qmc.Observable) -> qmc.Float:
    """``reg[0]`` stays ``|0>``; the Vector-element ancilla ``reg[1]`` is flipped to ``|1>``.

    ``reg[0]`` occupies physical qubit 0, so the pre-fix empty-map binding put
    ``Z(0)`` on the ``|0>`` ``reg[0]`` (yielding ``+1``); the correct binding reaches
    the ``|1>`` ancilla ``reg[1]`` and yields ``-1``. ``reg[1]`` is re-accessed after
    the gate (a fresh element UUID), exercising the root-address fallback.
    """
    reg = qmc.qubit_array(2, name="reg")
    reg[1] = qmc.x(reg[1])
    return qmc.expval((reg[1],), obs)


@qmc.qkernel
def _qft_result_element(obs: qmc.Observable) -> qmc.Float:
    """Observe one element of the Vector returned by a ``qft`` CompositeGate."""
    q = qmc.qubit_array(3, name="q")
    q = qmc.qft(q)
    return qmc.expval((q[0],), obs)


@qmc.qkernel
def _cry_result_ancilla(theta: qmc.Float, obs: qmc.Observable) -> qmc.Float:
    """Observe a Vector-element ancilla produced as a controlled-U result."""
    clock = qmc.qubit_array(1, name="clock")
    clock[0] = qmc.x(clock[0])  # control = |1> so the controlled-Ry always fires
    anc = qmc.qubit_array(1, name="anc")
    cry = qmc.control(qmc.ry, num_controls=1)
    res = cry(clock[0], anc[0], angle=theta)
    clock[0] = res[0]
    anc[0] = res[1]
    return qmc.expval((anc[0],), obs)


@qmc.qkernel
def _whole_vector(obs: qmc.Observable) -> qmc.Float:
    """Whole-Vector expval after gating element 0 to ``|1>``."""
    q = qmc.qubit_array(2, name="q")
    q[0] = qmc.x(q[0])
    return qmc.expval(q, obs)


@qmc.qkernel
def _sliced_view(obs: qmc.Observable) -> qmc.Float:
    """Sliced-view expval: ``q[1::2]`` with ``q[1]`` gated to ``|1>``."""
    q = qmc.qubit_array(4, name="q")
    q[1] = qmc.x(q[1])
    return qmc.expval(q[1::2], obs)


@qmc.qkernel
def _standalone_tuple(obs: qmc.Observable) -> qmc.Float:
    """Standalone-qubit tuple expval: ``q1`` gated to ``|1>``, ``q0`` stays ``|0>``."""
    q0 = qmc.qubit("q0")
    q1 = qmc.qubit("q1")
    q1 = qmc.x(q1)
    return qmc.expval((q0, q1), obs)


@qmc.qkernel
def _expval_two_vec_elems(obs: qmc.Observable) -> qmc.Float:
    """Tuple of two ungated Vector elements (used for the serialization test)."""
    q = qmc.qubit_array(2, name="q")
    return qmc.expval((q[0], q[1]), obs)


# ---------------------------------------------------------------------------
# Case (a): ungated Vector-element ancilla — the core regression
# ---------------------------------------------------------------------------


def test_ungated_vector_element_ancilla_is_plus_one(backend):
    """``<Z>`` of an ungated Vector-element ancilla is ``+1`` (clock qubits are |1>).

    Before the fix the empty qubit_map left ``Z(0)`` on physical qubit 0 (a ``|1>``
    clock qubit), yielding ``-1``.
    """
    got = _expval(backend, _ungated_vec_ancilla, {"obs": qm_o.Z(0)})
    assert math.isclose(got, 1.0, abs_tol=_EXPVAL_ATOL)


def test_gated_vector_element_ancilla_is_minus_one(backend):
    """A Vector-element ancilla flipped to ``|1>`` gives ``<Z> = -1`` (clock stays |0>).

    Together with the ungated case this pins ``Z(0)`` to the ancilla in both
    directions, not to ``clock[0]``.
    """
    got = _expval(backend, _gated_vec_ancilla, {"obs": qm_o.Z(0)})
    assert math.isclose(got, -1.0, abs_tol=_EXPVAL_ATOL)


# ---------------------------------------------------------------------------
# Case (b): Vector element produced by a composite / controlled-U, then expval
# ---------------------------------------------------------------------------


def test_qft_result_vector_element(backend):
    """``<Z>`` on a qubit of ``qft(|000>)`` is ``0`` (uniform superposition)."""
    got = _expval(backend, _qft_result_element, {"obs": qm_o.Z(0)})
    assert math.isclose(got, 0.0, abs_tol=_EXPVAL_ATOL)


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_controlled_u_result_vector_element(backend, seed):
    """``<Z>`` of a controlled-Ry result ancilla matches ``cos(theta)``.

    The control is held in ``|1>`` so the rotation always fires; the ancilla is
    re-accessed after the gate (a fresh element UUID), exercising the root-address
    fallback.
    """
    rng = np.random.default_rng(seed)
    for theta in [0.0, math.pi, 2.0 * math.pi, float(rng.uniform(0, 2 * math.pi))]:
        got = _expval(backend, _cry_result_ancilla, {"theta": theta, "obs": qm_o.Z(0)})
        assert math.isclose(got, math.cos(theta), abs_tol=_EXPVAL_ATOL), (
            f"theta={theta}"
        )


# ---------------------------------------------------------------------------
# Invariants that must not regress
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("qubit_idx, expected", [(0, -1.0), (1, 1.0)])
def test_whole_vector_invariant(backend, qubit_idx, expected):
    """Whole-Vector expval binds ``Z(k)`` to ``q[k]`` (``q[0]=|1>``, ``q[1]=|0>``)."""
    got = _expval(backend, _whole_vector, {"obs": qm_o.Z(qubit_idx)})
    assert math.isclose(got, expected, abs_tol=_EXPVAL_ATOL)


def test_sliced_view_invariant(backend):
    """``expval(q[1::2], Z(0))`` observes ``q[1]`` (``=|1>``) -> ``-1``."""
    got = _expval(backend, _sliced_view, {"obs": qm_o.Z(0)})
    assert math.isclose(got, -1.0, abs_tol=_EXPVAL_ATOL)


@pytest.mark.parametrize("qubit_idx, expected", [(0, 1.0), (1, -1.0)])
def test_standalone_qubit_tuple_invariant(backend, qubit_idx, expected):
    """Standalone-qubit tuple expval: ``q0=|0>`` -> ``+1``, ``q1=|1>`` -> ``-1``."""
    got = _expval(backend, _standalone_tuple, {"obs": qm_o.Z(qubit_idx)})
    assert math.isclose(got, expected, abs_tol=_EXPVAL_ATOL)


# ---------------------------------------------------------------------------
# Serialization: the per-element root metadata round-trips
# ---------------------------------------------------------------------------


def _find_expval(block) -> ExpvalOp:
    """Return the single ``ExpvalOp`` in ``block``.

    Args:
        block (Block): The IR block to search.

    Returns:
        ExpvalOp: The expectation-value operation.

    Raises:
        AssertionError: If no ``ExpvalOp`` is present.
    """
    for op in block.operations:
        if isinstance(op, ExpvalOp):
            return op
    raise AssertionError("no ExpvalOp in block")


def test_expval_vector_element_root_metadata_round_trips():
    """Tuple-form expval root ``(uuid, index)`` metadata survives JSON round-trip."""
    block = InlinePass().run(_expval_two_vec_elems.block)
    rt = _find_expval(block).qubits.metadata.array_runtime
    assert rt is not None
    assert len(rt.element_parent_uuids) == 2
    assert all(u for u in rt.element_parent_uuids)  # no standalone sentinel
    assert tuple(rt.element_parent_indices) == (0, 1)

    restored = load_json(dump_json(block))
    r_rt = _find_expval(restored).qubits.metadata.array_runtime
    assert r_rt is not None
    assert r_rt.element_parent_uuids == rt.element_parent_uuids
    assert r_rt.element_parent_indices == rt.element_parent_indices
