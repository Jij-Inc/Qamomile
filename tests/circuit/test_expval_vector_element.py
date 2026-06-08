"""Regression tests: ``expval`` binds observables to the right physical qubits
for Vector-element and whole-Vector operands.

Before the fix, ``expval((vector[i],), obs)`` resolved the observable's qubit by a
flat ``QubitAddress(uuid)`` lookup with no fallback. An ungated Vector element (or
one re-accessed after a gate/composite) has a UUID that was never registered in the
quantum segment's qubit_map, so the lookup missed, the observable's qubit_map came
back empty, ``remap_qubits`` was skipped, and the Pauli term silently fell back to
physical qubit 0. These tests pin the corrected behaviour and the invariants that
must not regress (whole-Vector, offset whole-Vector, sliced-view, and
standalone-qubit forms).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.ir.operation.expval import ExpvalOp
from qamomile.circuit.ir.serialize import dump_json, load_json
from qamomile.circuit.ir.types.primitives import QubitType
from qamomile.circuit.ir.value import ArrayValue
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
def _bare_ungated_vec_ancilla(obs: qmc.Observable) -> qmc.Float:
    """Ungated Vector-element ancilla passed as a bare ``Qubit`` (no tuple).

    Same as the tuple case but ``expval(anc[0], obs)`` is called without an
    enclosing tuple, exercising the single-Value branch of ``_build_qubit_map``.
    """
    clock = qmc.qubit_array(2, name="clock")
    clock[0] = qmc.x(clock[0])
    clock[1] = qmc.x(clock[1])
    anc_vec = qmc.qubit_array(1, name="anc")
    return qmc.expval(anc_vec[0], obs)  # bare Qubit, no tuple


@qmc.qkernel
def _bare_gated_vec_ancilla(obs: qmc.Observable) -> qmc.Float:
    """Vector-element ancilla ``reg[1]`` gated to ``|1>``, passed as a bare ``Qubit``."""
    reg = qmc.qubit_array(2, name="reg")
    reg[1] = qmc.x(reg[1])
    return qmc.expval(reg[1], obs)  # bare Qubit, no tuple


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


def test_bare_ungated_vector_element_ancilla_is_plus_one(backend):
    """``expval(anc[0], Z(0))`` (bare Qubit, no tuple) binds to the ancilla -> +1.

    The bare single-Qubit form goes through the single-Value branch of
    ``_build_qubit_map``; before the fix it missed the root-address fallback and
    returned ``-1`` (bound to a ``|1>`` clock qubit).
    """
    got = _expval(backend, _bare_ungated_vec_ancilla, {"obs": qm_o.Z(0)})
    assert math.isclose(got, 1.0, abs_tol=_EXPVAL_ATOL)


def test_bare_gated_vector_element_ancilla_is_minus_one(backend):
    """``expval(reg[1], Z(0))`` (bare Qubit) where ``reg[1] = |1>`` -> -1."""
    got = _expval(backend, _bare_gated_vec_ancilla, {"obs": qm_o.Z(0)})
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
    obs = qm_o.Z(0)
    angles = [0.0, math.pi, 2.0 * math.pi, float(rng.uniform(0, 2 * math.pi))]
    for theta in angles:
        got = _expval(backend, _cry_result_ancilla, {"theta": theta, "obs": obs})
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


def test_get_element_parent_addresses_aligns_with_element_uuids():
    """The accessor returns one entry per element even with no parent fields set.

    Guards the contract that ``get_element_parent_addresses()`` is aligned with
    ``get_element_uuids()`` (so callers can index by element position). Metadata
    that only sets ``element_uuids`` must yield ``None`` per element, not a
    truncated/empty tuple.
    """
    av = ArrayValue(type=QubitType(), name="a", shape=tuple())
    only_uuids = av.with_array_runtime_metadata(element_uuids=("u0", "u1", "u2"))
    addrs = only_uuids.get_element_parent_addresses()
    assert addrs == (None, None, None)
    assert len(addrs) == len(only_uuids.get_element_uuids())

    # A mix of resolved root, standalone sentinel, and resolved root.
    mixed = av.with_array_runtime_metadata(
        element_uuids=("u0", "u1", "u2"),
        element_parent_uuids=("root", "", "root"),
        element_parent_indices=(0, -1, 2),
    )
    assert mixed.get_element_parent_addresses() == (("root", 0), None, ("root", 2))


# ---------------------------------------------------------------------------
# Whole-Vector operands allocated after unrelated qubits
# ---------------------------------------------------------------------------


@qmc.qkernel
def _flip_second_in_qkernel(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Flip logical qubit 1 inside a sub-kernel.

    Args:
        q (qmc.Vector[qmc.Qubit]): Qubit register to mutate.

    Returns:
        qmc.Vector[qmc.Qubit]: Register with logical qubit 1 flipped.
    """
    q[1] = qmc.x(q[1])
    return q


@qmc.qkernel
def _random_ry_layer(
    q: qmc.Vector[qmc.Qubit],
    angles: qmc.Vector[qmc.Float],
    n: qmc.UInt,
) -> qmc.Vector[qmc.Qubit]:
    """Apply per-qubit RY angles inside a sub-kernel.

    Args:
        q (qmc.Vector[qmc.Qubit]): Qubit register to rotate.
        angles (qmc.Vector[qmc.Float]): Per-qubit rotation angles.
        n (qmc.UInt): Number of logical qubits to rotate.

    Returns:
        qmc.Vector[qmc.Qubit]: Rotated qubit register.
    """
    for i in qmc.range(n):
        q[i] = qmc.ry(q[i], angles[i])
    return q


@qmc.qkernel
def _offset_native_gate(obs: qmc.Observable) -> qmc.Float:
    """Prepare an offset register with a native element gate.

    Args:
        obs (qmc.Observable): Observable evaluated over the whole register.

    Returns:
        qmc.Float: Expectation value of the observable.
    """
    anc = qmc.qubit("anc")
    anc = qmc.x(anc)
    q = qmc.qubit_array(2, "q")
    q[1] = qmc.x(q[1])
    return qmc.expval(q, obs)


@qmc.qkernel
def _offset_qkernel_call(obs: qmc.Observable) -> qmc.Float:
    """Prepare an offset register through a sub-kernel call.

    Args:
        obs (qmc.Observable): Observable evaluated over the whole register.

    Returns:
        qmc.Float: Expectation value of the observable.
    """
    anc = qmc.qubit("anc")
    anc = qmc.x(anc)
    q = qmc.qubit_array(2, "q")
    q = _flip_second_in_qkernel(q)
    return qmc.expval(q, obs)


@qmc.qkernel
def _offset_composite_gate(obs: qmc.Observable) -> qmc.Float:
    """Prepare an offset register through CompositeGate operations.

    Args:
        obs (qmc.Observable): Observable evaluated over the whole register.

    Returns:
        qmc.Float: Expectation value of the observable.
    """
    anc = qmc.qubit("anc")
    anc = qmc.x(anc)
    q = qmc.qubit_array(2, "q")
    q[1] = qmc.x(q[1])
    q = qmc.qft(q)
    q = qmc.iqft(q)
    return qmc.expval(q, obs)


@qmc.qkernel
def _offset_broadcast(obs: qmc.Observable) -> qmc.Float:
    """Prepare an offset register with whole-Vector gate broadcast.

    Args:
        obs (qmc.Observable): Observable evaluated over the whole register.

    Returns:
        qmc.Float: Expectation value of the observable.
    """
    _ = qmc.qubit("anc")
    q = qmc.qubit_array(2, "q")
    q = qmc.x(q)
    return qmc.expval(q, obs)


@qmc.qkernel
def _offset_view_broadcast(obs: qmc.Observable) -> qmc.Float:
    """Prepare an offset register with view broadcast then whole expval.

    Args:
        obs (qmc.Observable): Observable evaluated over the whole register.

    Returns:
        qmc.Float: Expectation value of the observable.
    """
    anc = qmc.qubit("anc")
    anc = qmc.x(anc)
    q = qmc.qubit_array(3, "q")
    view = q[1:3]
    view = qmc.x(view)
    q[1:3] = view
    return qmc.expval(q, obs)


@qmc.qkernel
def _offset_random_native(
    n: qmc.UInt,
    angles: qmc.Vector[qmc.Float],
    obs: qmc.Observable,
) -> qmc.Float:
    """Prepare random per-qubit RY states with native element gates.

    Args:
        n (qmc.UInt): Number of logical qubits to allocate.
        angles (qmc.Vector[qmc.Float]): Per-qubit rotation angles.
        obs (qmc.Observable): Observable evaluated over the whole register.

    Returns:
        qmc.Float: Expectation value of the observable.
    """
    anc = qmc.qubit("anc")
    anc = qmc.x(anc)
    q = qmc.qubit_array(n, "q")
    for i in qmc.range(n):
        q[i] = qmc.ry(q[i], angles[i])
    return qmc.expval(q, obs)


@qmc.qkernel
def _offset_random_qkernel(
    n: qmc.UInt,
    angles: qmc.Vector[qmc.Float],
    obs: qmc.Observable,
) -> qmc.Float:
    """Prepare random per-qubit RY states through a sub-kernel.

    Args:
        n (qmc.UInt): Number of logical qubits to allocate.
        angles (qmc.Vector[qmc.Float]): Per-qubit rotation angles.
        obs (qmc.Observable): Observable evaluated over the whole register.

    Returns:
        qmc.Float: Expectation value of the observable.
    """
    anc = qmc.qubit("anc")
    anc = qmc.x(anc)
    q = qmc.qubit_array(n, "q")
    q = _random_ry_layer(q, angles, n)
    return qmc.expval(q, obs)


@qmc.qkernel
def _offset_random_composite(
    n: qmc.UInt,
    angles: qmc.Vector[qmc.Float],
    obs: qmc.Observable,
) -> qmc.Float:
    """Sandwich random RY states between QFT and IQFT composite gates.

    Args:
        n (qmc.UInt): Number of logical qubits to allocate.
        angles (qmc.Vector[qmc.Float]): Per-qubit rotation angles.
        obs (qmc.Observable): Observable evaluated over the whole register.

    Returns:
        qmc.Float: Expectation value of the observable.
    """
    anc = qmc.qubit("anc")
    anc = qmc.x(anc)
    q = qmc.qubit_array(n, "q")
    for i in qmc.range(n):
        q[i] = qmc.ry(q[i], angles[i])
    q = qmc.qft(q)
    q = qmc.iqft(q)
    return qmc.expval(q, obs)


@qmc.qkernel
def _offset_random_broadcast(
    theta: qmc.Float,
    obs: qmc.Observable,
) -> qmc.Float:
    """Prepare random RY states with whole-Vector broadcast.

    Args:
        theta (qmc.Float): Broadcast rotation angle.
        obs (qmc.Observable): Observable evaluated over the whole register.

    Returns:
        qmc.Float: Expectation value of the observable.
    """
    anc = qmc.qubit("anc")
    anc = qmc.x(anc)
    q = qmc.qubit_array(2, "q")
    q = qmc.ry(q, theta)
    return qmc.expval(q, obs)


@qmc.qkernel
def _offset_random_view_broadcast(
    theta0: qmc.Float,
    theta_view: qmc.Float,
    obs: qmc.Observable,
) -> qmc.Float:
    """Prepare random RY states with a view broadcast.

    Args:
        theta0 (qmc.Float): Rotation angle for logical qubit 0.
        theta_view (qmc.Float): Broadcast rotation angle for the slice view.
        obs (qmc.Observable): Observable evaluated over the whole register.

    Returns:
        qmc.Float: Expectation value of the observable.
    """
    anc = qmc.qubit("anc")
    anc = qmc.x(anc)
    q = qmc.qubit_array(3, "q")
    q[0] = qmc.ry(q[0], theta0)
    view = q[1:3]
    view = qmc.ry(view, theta_view)
    q[1:3] = view
    return qmc.expval(q, obs)


DETERMINISTIC_CASES = [
    pytest.param(
        "native",
        _offset_native_gate,
        2,
        {0: 1, 1: 2},
        [1.0, -1.0],
        id="native-gate",
    ),
    pytest.param(
        "qkernel",
        _offset_qkernel_call,
        2,
        {0: 1, 1: 2},
        [1.0, -1.0],
        id="qkernel-call",
    ),
    pytest.param(
        "composite",
        _offset_composite_gate,
        2,
        {0: 1, 1: 2},
        [1.0, -1.0],
        id="composite-gate",
    ),
    pytest.param(
        "broadcast",
        _offset_broadcast,
        2,
        {0: 1, 1: 2},
        [-1.0, -1.0],
        id="broadcast",
    ),
    pytest.param(
        "view_broadcast",
        _offset_view_broadcast,
        3,
        {0: 1, 1: 2, 2: 3},
        [1.0, -1.0, -1.0],
        id="view-broadcast",
    ),
]

RANDOM_CASES = [
    pytest.param("native", _offset_random_native, id="native-gate"),
    pytest.param("qkernel", _offset_random_qkernel, id="qkernel-call"),
    pytest.param("composite", _offset_random_composite, id="composite-gate"),
]


def _run_whole_vector_expval(backend, kernel, bindings, target, expected_map):
    """Run a whole-Vector expval and assert the compiled remap.

    Args:
        backend (tuple): ``(transpiler, executor)`` pair from the fixture.
        kernel (object): Qkernel that accepts ``obs`` through bindings.
        bindings (dict[str, object]): Additional compile-time bindings.
        target (int): Logical observable target index.
        expected_map (dict[int, int]): Expected Pauli-index remap.

    Returns:
        float: Executed expectation value.
    """
    transpiler, executor = backend
    exe = transpiler.transpile(
        kernel,
        bindings={**bindings, "obs": qm_o.Z(target)},
    )
    assert exe.compiled_expval[0].qubit_map == expected_map
    return float(exe.run(executor).result())


@pytest.mark.parametrize(
    "case_name,kernel,n,expected_map,expected_values", DETERMINISTIC_CASES
)
def test_offset_whole_vector_expval_deterministic_forms(
    backend,
    case_name,
    kernel,
    n,
    expected_map,
    expected_values,
):
    """Deterministic frontend forms remap ``Z(k)`` onto logical ``q[k]``."""
    del case_name
    for target in range(n):
        got = _run_whole_vector_expval(backend, kernel, {}, target, expected_map)
        assert np.isclose(got, expected_values[target], atol=1e-6), (
            f"target={target}: got {got}, expected {expected_values[target]}"
        )


@pytest.mark.parametrize("case_name,kernel", RANDOM_CASES)
@pytest.mark.parametrize("seed", [0, 1, 2, 42])
@pytest.mark.parametrize("n", [2, 4])
def test_offset_whole_vector_expval_random_ry_forms(
    backend,
    case_name,
    kernel,
    seed,
    n,
):
    """Random per-qubit RY forms compare each logical ``Z(k)`` analytically."""
    rng = np.random.default_rng(seed)
    angles = rng.uniform(-math.pi, math.pi, size=n)
    expected_map = {i: i + 1 for i in range(n)}
    bindings = {"n": n, "angles": angles}

    for target in range(n):
        got = _run_whole_vector_expval(backend, kernel, bindings, target, expected_map)
        expected = math.cos(float(angles[target]))
        assert np.isclose(got, expected, atol=1e-6), (
            f"{case_name} seed={seed} n={n} target={target}: "
            f"got {got}, expected {expected}"
        )


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_offset_whole_vector_expval_random_broadcast(backend, seed):
    """Random whole-Vector broadcast keeps expval on the offset register."""
    rng = np.random.default_rng(seed)
    theta = float(rng.uniform(-math.pi, math.pi))
    expected_map = {0: 1, 1: 2}

    for target in range(2):
        got = _run_whole_vector_expval(
            backend,
            _offset_random_broadcast,
            {"theta": theta},
            target,
            expected_map,
        )
        expected = math.cos(theta)
        assert np.isclose(got, expected, atol=1e-6), (
            f"broadcast seed={seed} target={target}: got {got}, expected {expected}"
        )


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_offset_whole_vector_expval_random_view_broadcast(backend, seed):
    """Random view broadcast keeps whole expval mapped after slice return."""
    rng = np.random.default_rng(seed)
    theta0 = float(rng.uniform(-math.pi, math.pi))
    theta_view = float(rng.uniform(-math.pi, math.pi))
    expected_map = {0: 1, 1: 2, 2: 3}
    expected_values = [math.cos(theta0), math.cos(theta_view), math.cos(theta_view)]

    for target in range(3):
        got = _run_whole_vector_expval(
            backend,
            _offset_random_view_broadcast,
            {"theta0": theta0, "theta_view": theta_view},
            target,
            expected_map,
        )
        assert np.isclose(got, expected_values[target], atol=1e-6), (
            f"view+broadcast seed={seed} target={target}: "
            f"got {got}, expected {expected_values[target]}"
        )


# Tuple-form frontend patterns through InlinePass and register updates
# ---------------------------------------------------------------------------


@qmc.qkernel
def _helper_first(q: qmc.Vector[qmc.Qubit], obs: qmc.Observable) -> qmc.Float:
    """Evaluate the first element of an inlined vector argument.

    Args:
        q (qmc.Vector[qmc.Qubit]): Vector argument supplied by the caller.
        obs (qmc.Observable): Observable to evaluate against the selected
            tuple element.

    Returns:
        qmc.Float: Expectation value on the first vector element.
    """
    return qmc.expval((q[0],), obs)


@qmc.qkernel
def _helper_second(q: qmc.Vector[qmc.Qubit], obs: qmc.Observable) -> qmc.Float:
    """Evaluate the second element of an inlined vector argument.

    Args:
        q (qmc.Vector[qmc.Qubit]): Vector argument supplied by the caller.
        obs (qmc.Observable): Observable to evaluate against the selected
            tuple element.

    Returns:
        qmc.Float: Expectation value on the second vector element.
    """
    return qmc.expval((q[1],), obs)


@qmc.qkernel
def _scalar_x_helper(q: qmc.Qubit, obs: qmc.Observable) -> qmc.Float:
    """Evaluate a scalar helper result after a native gate.

    Args:
        q (qmc.Qubit): Scalar qubit argument supplied by the caller.
        obs (qmc.Observable): Observable to evaluate against the updated
            scalar qubit.

    Returns:
        qmc.Float: Expectation value after applying an X gate.
    """
    q = qmc.x(q)
    return qmc.expval((q,), obs)


@qmc.qkernel
def _scalar_identity_helper(q: qmc.Qubit, obs: qmc.Observable) -> qmc.Float:
    """Evaluate an unchanged scalar helper argument.

    Args:
        q (qmc.Qubit): Scalar qubit argument supplied by the caller.
        obs (qmc.Observable): Observable to evaluate against the scalar qubit.

    Returns:
        qmc.Float: Expectation value for the unchanged scalar qubit.
    """
    return qmc.expval((q,), obs)


@qmc.qkernel
def _scalar_pair_identity_helper(
    q0: qmc.Qubit,
    q1: qmc.Qubit,
    obs: qmc.Observable,
) -> qmc.Float:
    """Evaluate unchanged scalar helper arguments as a tuple.

    Args:
        q0 (qmc.Qubit): First scalar qubit argument supplied by the caller.
        q1 (qmc.Qubit): Second scalar qubit argument supplied by the caller.
        obs (qmc.Observable): Observable to evaluate against the tuple.

    Returns:
        qmc.Float: Expectation value for the unchanged scalar tuple.
    """
    return qmc.expval((q0, q1), obs)


@qmc.qkernel
def _scalar_ry_helper(
    q: qmc.Qubit,
    theta: qmc.Float,
    obs: qmc.Observable,
) -> qmc.Float:
    """Evaluate a scalar helper result after a random native gate.

    Args:
        q (qmc.Qubit): Scalar qubit argument supplied by the caller.
        theta (qmc.Float): Rotation angle for the RY gate.
        obs (qmc.Observable): Observable to evaluate against the updated
            scalar qubit.

    Returns:
        qmc.Float: Expectation value after applying an RY gate.
    """
    q = qmc.ry(q, theta)
    return qmc.expval((q,), obs)


@qmc.qkernel
def _deterministic_qkernel(obs: qmc.Observable) -> qmc.Float:
    """Route tuple expval through an inlined qkernel argument.

    Args:
        obs (qmc.Observable): Observable to evaluate on the selected qubit.

    Returns:
        qmc.Float: Expectation value for the inlined vector element.
    """
    q = qmc.qubit_array(3, "q")
    q[1] = qmc.x(q[1])
    return _helper_second(q, obs)


@qmc.qkernel
def _deterministic_native_gate(obs: qmc.Observable) -> qmc.Float:
    """Evaluate a tuple element after a native scalar gate.

    Args:
        obs (qmc.Observable): Observable to evaluate on the selected qubit.

    Returns:
        qmc.Float: Expectation value for the directly updated element.
    """
    q = qmc.qubit_array(3, "q")
    q[2] = qmc.x(q[2])
    return qmc.expval((q[2],), obs)


@qmc.qkernel
def _deterministic_scalar_qkernel(obs: qmc.Observable) -> qmc.Float:
    """Evaluate tuple expval after an inlined scalar-qkernel gate.

    Args:
        obs (qmc.Observable): Observable to evaluate on the selected qubit.

    Returns:
        qmc.Float: Expectation value for the inlined scalar argument.
    """
    q = qmc.qubit_array(3, "q")
    return _scalar_x_helper(q[1], obs)


@qmc.qkernel
def _deterministic_scalar_identity_qkernel(obs: qmc.Observable) -> qmc.Float:
    """Evaluate an unchanged inlined scalar argument.

    Args:
        obs (qmc.Observable): Observable to evaluate on the selected qubit.

    Returns:
        qmc.Float: Expectation value for the inlined scalar argument.
    """
    q = qmc.qubit_array(2, "q")
    q[0] = qmc.x(q[0])
    return _scalar_identity_helper(q[1], obs)


@qmc.qkernel
def _deterministic_scalar_pair_identity_qkernel(
    obs: qmc.Observable,
) -> qmc.Float:
    """Evaluate unchanged inlined scalar arguments in tuple form.

    Args:
        obs (qmc.Observable): Observable to evaluate on the selected tuple slot.

    Returns:
        qmc.Float: Expectation value for the first inlined scalar argument.
    """
    q = qmc.qubit_array(2, "q")
    q[0] = qmc.x(q[0])
    return _scalar_pair_identity_helper(q[1], q[0], obs)


@qmc.qkernel
def _deterministic_broadcast(obs: qmc.Observable) -> qmc.Float:
    """Evaluate a tuple element after a full-register broadcast gate.

    Args:
        obs (qmc.Observable): Observable to evaluate on the selected qubit.

    Returns:
        qmc.Float: Expectation value for the broadcast-updated element.
    """
    q = qmc.qubit_array(3, "q")
    q = qmc.x(q)
    return qmc.expval((q[2],), obs)


@qmc.qkernel
def _deterministic_view_broadcast(obs: qmc.Observable) -> qmc.Float:
    """Evaluate a tuple element from a broadcast-updated view argument.

    Args:
        obs (qmc.Observable): Observable to evaluate on the selected qubit.

    Returns:
        qmc.Float: Expectation value for the inlined view element.
    """
    q = qmc.qubit_array(4, "q")
    view = q[1::2]
    view = qmc.x(view)
    q[1::2] = view
    return _helper_first(q[1::2], obs)


@qmc.qkernel
def _deterministic_composite_gate(obs: qmc.Observable) -> qmc.Float:
    """Evaluate a tuple element after composite gates that round-trip state.

    Args:
        obs (qmc.Observable): Observable to evaluate on the selected qubit.

    Returns:
        qmc.Float: Expectation value after the QFT and inverse-QFT round trip.
    """
    q = qmc.qubit_array(3, "q")
    q[2] = qmc.x(q[2])
    q = qmc.qft(q)
    q = qmc.iqft(q)
    return qmc.expval((q[2],), obs)


@qmc.qkernel
def _random_qkernel(theta: qmc.Float, obs: qmc.Observable) -> qmc.Float:
    """Route a random-angle tuple expval through an inlined qkernel.

    Args:
        theta (qmc.Float): Rotation angle for the RY gate.
        obs (qmc.Observable): Observable to evaluate on the selected qubit.

    Returns:
        qmc.Float: Expectation value for the inlined vector element.
    """
    q = qmc.qubit_array(3, "q")
    q[1] = qmc.ry(q[1], theta)
    return _helper_second(q, obs)


@qmc.qkernel
def _random_native_gate(theta: qmc.Float, obs: qmc.Observable) -> qmc.Float:
    """Evaluate a tuple element after a random native scalar gate.

    Args:
        theta (qmc.Float): Rotation angle for the RY gate.
        obs (qmc.Observable): Observable to evaluate on the selected qubit.

    Returns:
        qmc.Float: Expectation value for the directly updated element.
    """
    q = qmc.qubit_array(3, "q")
    q[2] = qmc.ry(q[2], theta)
    return qmc.expval((q[2],), obs)


@qmc.qkernel
def _random_scalar_qkernel(theta: qmc.Float, obs: qmc.Observable) -> qmc.Float:
    """Evaluate tuple expval after a random inlined scalar-qkernel gate.

    Args:
        theta (qmc.Float): Rotation angle for the RY gate.
        obs (qmc.Observable): Observable to evaluate on the selected qubit.

    Returns:
        qmc.Float: Expectation value for the inlined scalar argument.
    """
    q = qmc.qubit_array(3, "q")
    return _scalar_ry_helper(q[1], theta, obs)


@qmc.qkernel
def _random_broadcast(theta: qmc.Float, obs: qmc.Observable) -> qmc.Float:
    """Evaluate a tuple element after a random full-register broadcast gate.

    Args:
        theta (qmc.Float): Rotation angle broadcast across the vector.
        obs (qmc.Observable): Observable to evaluate on the selected qubit.

    Returns:
        qmc.Float: Expectation value for the broadcast-updated element.
    """
    q = qmc.qubit_array(3, "q")
    q = qmc.ry(q, theta)
    return qmc.expval((q[0],), obs)


@qmc.qkernel
def _random_view_broadcast(theta: qmc.Float, obs: qmc.Observable) -> qmc.Float:
    """Evaluate a tuple element from a random broadcast-updated view.

    Args:
        theta (qmc.Float): Rotation angle broadcast across the sliced view.
        obs (qmc.Observable): Observable to evaluate on the selected qubit.

    Returns:
        qmc.Float: Expectation value for the inlined view element.
    """
    q = qmc.qubit_array(4, "q")
    view = q[1::2]
    view = qmc.ry(view, theta)
    q[1::2] = view
    return _helper_first(q[1::2], obs)


@qmc.qkernel
def _random_composite_gate(theta: qmc.Float, obs: qmc.Observable) -> qmc.Float:
    """Evaluate a random tuple element after composite gate round-trip.

    Args:
        theta (qmc.Float): Rotation angle for the RY gate.
        obs (qmc.Observable): Observable to evaluate on the selected qubit.

    Returns:
        qmc.Float: Expectation value after the QFT and inverse-QFT round trip.
    """
    q = qmc.qubit_array(3, "q")
    q[2] = qmc.ry(q[2], theta)
    q = qmc.qft(q)
    q = qmc.iqft(q)
    return qmc.expval((q[2],), obs)


def _expval_with_qubit_map(
    backend: tuple[object, object],
    kernel: qmc.QKernel,
    bindings: dict[str, object],
) -> tuple[float, dict[int, int]]:
    """Transpile and run one expval kernel, returning its qubit map.

    Args:
        backend (tuple[object, object]): Transpiler and executor from the
            ``backend`` fixture.
        kernel (qmc.QKernel): Kernel to transpile and execute.
        bindings (dict[str, object]): Compile-time bindings.

    Returns:
        tuple[float, dict[int, int]]: Observed expectation value and compiled
            expval Pauli-index to physical-qubit map.
    """
    transpiler, executor = backend
    exe = transpiler.transpile(kernel, bindings=bindings)
    value = exe.run(executor).result()
    assert exe.compiled_expval
    return float(value), exe.compiled_expval[0].qubit_map


DETERMINISTIC_INLINE_CASES: tuple[tuple[str, qmc.QKernel, int, float], ...] = (
    ("qkernel", _deterministic_qkernel, 1, -1.0),
    ("native_gate", _deterministic_native_gate, 2, -1.0),
    ("scalar_qkernel", _deterministic_scalar_qkernel, 1, -1.0),
    ("scalar_identity_qkernel", _deterministic_scalar_identity_qkernel, 1, 1.0),
    ("broadcast", _deterministic_broadcast, 2, -1.0),
    ("view_broadcast", _deterministic_view_broadcast, 1, -1.0),
    ("composite_gate", _deterministic_composite_gate, 2, -1.0),
)


@pytest.mark.parametrize(
    ("case_name", "kernel", "expected_physical", "expected"),
    DETERMINISTIC_INLINE_CASES,
    ids=[case[0] for case in DETERMINISTIC_INLINE_CASES],
)
def test_tuple_expval_inline_frontend_patterns_deterministic(
    backend,
    case_name: str,
    kernel: qmc.QKernel,
    expected_physical: int,
    expected: float,
):
    """Tuple-form expval targets the intended physical qubit deterministically."""
    value, qubit_map = _expval_with_qubit_map(backend, kernel, {"obs": qm_o.Z(0)})
    assert qubit_map == {0: expected_physical}, case_name
    assert math.isclose(value, expected, abs_tol=_EXPVAL_ATOL), case_name


def test_tuple_expval_inline_scalar_pair_identity(backend):
    """Tuple metadata roots are rewritten for unchanged scalar arguments."""
    value, qubit_map = _expval_with_qubit_map(
        backend,
        _deterministic_scalar_pair_identity_qkernel,
        {"obs": qm_o.Z(0)},
    )
    assert qubit_map == {0: 1, 1: 0}
    assert math.isclose(value, 1.0, abs_tol=_EXPVAL_ATOL)


RANDOM_INLINE_CASES: tuple[
    tuple[str, qmc.QKernel, int, Callable[[float], float]], ...
] = (
    ("qkernel", _random_qkernel, 1, math.cos),
    ("native_gate", _random_native_gate, 2, math.cos),
    ("scalar_qkernel", _random_scalar_qkernel, 1, math.cos),
    ("broadcast", _random_broadcast, 0, math.cos),
    ("view_broadcast", _random_view_broadcast, 1, math.cos),
    ("composite_gate", _random_composite_gate, 2, math.cos),
)


@pytest.mark.parametrize("seed", [0, 1, 2, 42])
@pytest.mark.parametrize(
    ("case_name", "kernel", "expected_physical", "expected_fn"),
    RANDOM_INLINE_CASES,
    ids=[case[0] for case in RANDOM_INLINE_CASES],
)
def test_tuple_expval_inline_frontend_patterns_random_angles(
    backend,
    seed: int,
    case_name: str,
    kernel: qmc.QKernel,
    expected_physical: int,
    expected_fn: Callable[[float], float],
):
    """Tuple-form expval matches seeded random analytic expectations."""
    rng = np.random.default_rng(seed)
    theta = float(rng.uniform(-2.0 * math.pi, 2.0 * math.pi))
    value, qubit_map = _expval_with_qubit_map(
        backend,
        kernel,
        {"theta": theta, "obs": qm_o.Z(0)},
    )
    assert qubit_map == {0: expected_physical}, case_name
    assert math.isclose(value, expected_fn(theta), abs_tol=_EXPVAL_ATOL), case_name
