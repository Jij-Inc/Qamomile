"""Cross-backend tuple-form expval tests for frontend-allowed qubit flows."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o


@pytest.fixture(
    params=[
        "qiskit",
        pytest.param("quri_parts", marks=pytest.mark.quri_parts),
        pytest.param("cudaq", marks=pytest.mark.cudaq),
    ],
)
def backend(request):
    """Yield an installed backend's transpiler and executor.

    Args:
        request (pytest.FixtureRequest): Parametrized pytest fixture
            request carrying the backend name.

    Returns:
        tuple[str, object, object]: Backend name, transpiler, and executor.

    Raises:
        AssertionError: If the fixture parameter names an unknown backend.
    """
    name = request.param
    if name == "qiskit":
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        transpiler = QiskitTranspiler()
        return name, transpiler, transpiler.executor()
    if name == "quri_parts":
        pytest.importorskip("quri_parts")
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


def _run_expval(
    backend: tuple[object, object, object],
    kernel: qmc.QKernel,
    bindings: dict[str, object],
) -> tuple[float, dict[int, int]]:
    """Transpile and run one expval kernel.

    Args:
        backend (tuple[object, object, object]): Backend name,
            transpiler, and executor from the ``backend`` fixture.
        kernel (qmc.QKernel): Kernel to transpile and execute.
        bindings (dict[str, object]): Compile-time bindings.

    Returns:
        tuple[float, dict[int, int]]: Observed expectation value and
            compiled expval Pauli-index to physical-qubit map.
    """
    _, transpiler, executor = backend
    exe = transpiler.transpile(kernel, bindings=bindings)
    value = exe.run(executor).result()
    assert exe.compiled_expval
    return float(value), exe.compiled_expval[0].qubit_map


DETERMINISTIC_CASES: tuple[tuple[str, qmc.QKernel, int, float], ...] = (
    ("qkernel", _deterministic_qkernel, 1, -1.0),
    ("native_gate", _deterministic_native_gate, 2, -1.0),
    ("scalar_qkernel", _deterministic_scalar_qkernel, 1, -1.0),
    ("broadcast", _deterministic_broadcast, 2, -1.0),
    ("view_broadcast", _deterministic_view_broadcast, 1, -1.0),
    ("composite_gate", _deterministic_composite_gate, 2, -1.0),
)


@pytest.mark.parametrize(
    ("case_name", "kernel", "expected_physical", "expected"),
    DETERMINISTIC_CASES,
    ids=[case[0] for case in DETERMINISTIC_CASES],
)
def test_tuple_expval_frontend_patterns_deterministic(
    backend,
    case_name: str,
    kernel: qmc.QKernel,
    expected_physical: int,
    expected: float,
):
    """Tuple-form expval targets the intended physical qubit deterministically."""
    value, qubit_map = _run_expval(backend, kernel, {"obs": qm_o.Z(0)})
    assert qubit_map == {0: expected_physical}, case_name
    assert np.isclose(value, expected, atol=1e-8), case_name


RANDOM_CASES: tuple[tuple[str, qmc.QKernel, int, Callable[[float], float]], ...] = (
    ("qkernel", _random_qkernel, 1, np.cos),
    ("native_gate", _random_native_gate, 2, np.cos),
    ("scalar_qkernel", _random_scalar_qkernel, 1, np.cos),
    ("broadcast", _random_broadcast, 0, np.cos),
    ("view_broadcast", _random_view_broadcast, 1, np.cos),
    ("composite_gate", _random_composite_gate, 2, np.cos),
)


@pytest.mark.parametrize("seed", [0, 1, 2, 42])
@pytest.mark.parametrize(
    ("case_name", "kernel", "expected_physical", "expected_fn"),
    RANDOM_CASES,
    ids=[case[0] for case in RANDOM_CASES],
)
def test_tuple_expval_frontend_patterns_random_angles(
    backend,
    seed: int,
    case_name: str,
    kernel: qmc.QKernel,
    expected_physical: int,
    expected_fn: Callable[[float], float],
):
    """Tuple-form expval matches seeded random analytic expectations."""
    rng = np.random.default_rng(seed)
    theta = float(rng.uniform(-2.0 * np.pi, 2.0 * np.pi))
    value, qubit_map = _run_expval(
        backend,
        kernel,
        {"theta": theta, "obs": qm_o.Z(0)},
    )
    assert qubit_map == {0: expected_physical}, case_name
    assert np.isclose(value, expected_fn(theta), atol=1e-8), case_name
