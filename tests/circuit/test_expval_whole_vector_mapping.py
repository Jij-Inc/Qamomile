"""Regression tests for whole-Vector expval observable remapping.

The register under test is deliberately allocated after an ancilla so
its logical index 0 does not coincide with physical qubit 0.  Each test
checks that ``expval(q, Z(k))`` evaluates logical ``q[k]`` rather than
falling through to identity physical-qubit binding.
"""

import math

import numpy as np
import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o


@pytest.fixture(
    params=[
        "qiskit",
        pytest.param("quri_parts", marks=pytest.mark.quri_parts),
        pytest.param("cudaq", marks=pytest.mark.cudaq),
    ]
)
def backend(request):
    """Yield an installed transpiler for each supported SDK backend.

    Args:
        request (pytest.FixtureRequest): Parametrized pytest fixture request.

    Returns:
        tuple[str, object]: Backend name and transpiler instance.
    """
    name = request.param
    if name == "qiskit":
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        return name, QiskitTranspiler()
    if name == "quri_parts":
        pytest.importorskip("quri_parts")
        from qamomile.quri_parts import QuriPartsTranspiler

        return name, QuriPartsTranspiler()
    if name == "cudaq":
        pytest.importorskip("cudaq")
        from qamomile.cudaq import CudaqTranspiler

        return name, CudaqTranspiler()
    raise AssertionError(f"unknown backend {name}")


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


def _run_expval(transpiler, kernel, bindings, target, expected_map):
    """Run a whole-Vector expval and assert the compiled remap.

    Args:
        transpiler (object): Backend transpiler under test.
        kernel (object): Qkernel that accepts ``obs`` through bindings.
        bindings (dict[str, object]): Additional compile-time bindings.
        target (int): Logical observable target index.
        expected_map (dict[int, int]): Expected Pauli-index remap.

    Returns:
        float: Executed expectation value.
    """
    exe = transpiler.transpile(
        kernel,
        bindings={**bindings, "obs": qm_o.Z(target)},
    )
    assert exe.compiled_expval[0].qubit_map == expected_map
    return exe.run(transpiler.executor()).result()


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
    backend_name, transpiler = backend
    del case_name
    for target in range(n):
        got = _run_expval(transpiler, kernel, {}, target, expected_map)
        assert np.isclose(got, expected_values[target], atol=1e-6), (
            f"{backend_name} target={target}: got {got}, "
            f"expected {expected_values[target]}"
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
    backend_name, transpiler = backend
    rng = np.random.default_rng(seed)
    angles = rng.uniform(-math.pi, math.pi, size=n)
    expected_map = {i: i + 1 for i in range(n)}
    bindings = {"n": n, "angles": angles}

    for target in range(n):
        got = _run_expval(transpiler, kernel, bindings, target, expected_map)
        expected = math.cos(float(angles[target]))
        assert np.isclose(got, expected, atol=1e-6), (
            f"{backend_name} {case_name} seed={seed} n={n} target={target}: "
            f"got {got}, expected {expected}"
        )


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_offset_whole_vector_expval_random_broadcast(backend, seed):
    """Random whole-Vector broadcast keeps expval on the offset register."""
    backend_name, transpiler = backend
    rng = np.random.default_rng(seed)
    theta = float(rng.uniform(-math.pi, math.pi))
    expected_map = {0: 1, 1: 2}

    for target in range(2):
        got = _run_expval(
            transpiler,
            _offset_random_broadcast,
            {"theta": theta},
            target,
            expected_map,
        )
        expected = math.cos(theta)
        assert np.isclose(got, expected, atol=1e-6), (
            f"{backend_name} broadcast seed={seed} target={target}: "
            f"got {got}, expected {expected}"
        )


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_offset_whole_vector_expval_random_view_broadcast(backend, seed):
    """Random view broadcast keeps whole expval mapped after slice return."""
    backend_name, transpiler = backend
    rng = np.random.default_rng(seed)
    theta0 = float(rng.uniform(-math.pi, math.pi))
    theta_view = float(rng.uniform(-math.pi, math.pi))
    expected_map = {0: 1, 1: 2, 2: 3}
    expected_values = [math.cos(theta0), math.cos(theta_view), math.cos(theta_view)]

    for target in range(3):
        got = _run_expval(
            transpiler,
            _offset_random_view_broadcast,
            {"theta0": theta0, "theta_view": theta_view},
            target,
            expected_map,
        )
        assert np.isclose(got, expected_values[target], atol=1e-6), (
            f"{backend_name} view+broadcast seed={seed} target={target}: "
            f"got {got}, expected {expected_values[target]}"
        )
