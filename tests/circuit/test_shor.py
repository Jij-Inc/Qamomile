"""Tests for executable, body-derived Shor order finding."""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

import qamomile.circuit as qmc
from qamomile.circuit.frontend.qkernel import QKernel


def _basis_value(bits: tuple[int, ...]) -> int:
    """Decode a little-endian measured bit tuple.

    Args:
        bits (tuple[int, ...]): Measured bits in register order.

    Returns:
        int: Unsigned basis-state value.
    """
    return sum(bit << index for index, bit in enumerate(bits))


def test_shor_factory_returns_one_executable_qkernel() -> None:
    """The same returned QKernel supports estimation and transpilation."""
    kernel = qmc.shor_order_finding(base=2, modulus=15)

    assert isinstance(kernel, QKernel)
    estimate = kernel.estimate_resources()
    n = estimate.parameters["n"]
    positive_n = sp.Symbol("positive_n", integer=True, positive=True)
    assert estimate.qubits == 6 * n + 5
    assert estimate.width.dirty_ancilla_qubits == 0
    assert sp.Poly(estimate.gates.total.subs(n, positive_n), positive_n).degree() == 3
    assert sp.Poly(estimate.gates.toffoli.subs(n, positive_n), positive_n).degree() == 3
    assert "modmul_const" not in estimate.calls.calls_by_name
    assert estimate.trace is None

    concrete = kernel.estimate_resources(inputs={"n": 2048})
    assert concrete.gates.total == estimate.gates.total.subs(n, 2048)
    assert concrete.gates.toffoli == estimate.gates.toffoli.subs(n, 2048)


def test_shor_default_width_builds_and_too_narrow_width_fails() -> None:
    """The execution default fits the modulus and explicit narrow widths fail."""
    kernel = qmc.shor_order_finding(base=2, modulus=3)

    assert kernel.build().operations
    with pytest.raises(ValueError, match="too small.*require n >= 2"):
        kernel.build(n=1)


@pytest.mark.parametrize(
    ("base", "modulus", "message"),
    [
        (2, 2, "greater than two"),
        (1, 15, "1 < base"),
        (3, 15, "coprime"),
    ],
)
def test_shor_rejects_invalid_problem_instances(
    base: int,
    modulus: int,
    message: str,
) -> None:
    """Invalid order-finding instances fail before tracing.

    Args:
        base (int): Candidate base.
        modulus (int): Candidate modulus.
        message (str): Expected diagnostic fragment.
    """
    with pytest.raises(ValueError, match=message):
        qmc.shor_order_finding(base=base, modulus=modulus)


def test_small_shor_order_finding_recovers_period_two(sdk_transpiler) -> None:
    """The simulatable two-bit instance recovers the period-two peaks."""
    kernel = qmc.shor_order_finding(base=2, modulus=3)
    transpiler = sdk_transpiler.transpiler
    executable = transpiler.transpile(kernel, bindings={"n": 2})
    result = executable.sample(transpiler.executor(), shots=128).result()

    counts = {_basis_value(bits): count for bits, count in result.results}
    targets = {0, 8}
    on_peak = sum(counts.get(value, 0) for value in targets)

    assert on_peak / result.shots > 0.9
    assert all(counts.get(value, 0) / result.shots > 0.3 for value in targets)


def test_four_bit_shor_transpiles_without_statevector_execution(
    sdk_transpiler,
) -> None:
    """Transpile the 29-qubit benchmark without allocating its statevector.

    A dense 29-qubit statevector needs at least 8 GiB before simulator
    workspaces, so local sampling is intentionally outside this test's scope.

    Args:
        sdk_transpiler: Parametrized supported backend fixture.
    """
    kernel = qmc.shor_order_finding(base=2, modulus=15)
    executable = sdk_transpiler.transpiler.transpile(kernel, bindings={"n": 4})

    assert kernel.estimate_resources(inputs={"n": 4}).qubits == 29
    assert executable.compiled_quantum
    assert executable.plan.steps


@qmc.qkernel
def _shor_state_expval(observable: qmc.Observable) -> qmc.Float:
    """Evaluate an observable for the simulatable 2 mod 3 schedule."""
    counting = qmc.qubit_array(4, "counting")
    work = qmc.qubit_array(2, "work")
    work[0] = qmc.x(work[0])
    counting = qmc.h(counting)
    counting[0], work = qmc.modmul_const(
        work,
        multiplier=2,
        modulus=3,
        control=counting[0],
    )
    counting = qmc.iqft(counting)
    return qmc.expval(counting, observable)


def test_shor_cross_backend_expval(sdk_transpiler) -> None:
    """The modular-exponentiation state has its analytic high-bit expval.

    For the two-bit ``2 mod 3`` instance, the four-qubit counting register has
    ``<Z_3> = 0`` after IQFT. One observable is sufficient to exercise each
    backend's expval path; the sampling test validates the complete output
    distribution independently.

    Args:
        sdk_transpiler: Parametrized supported backend fixture.
    """
    import qamomile.observable as qm_o

    observable = qm_o.Z(3)

    transpiler = sdk_transpiler.transpiler
    executable = transpiler.transpile(
        _shor_state_expval,
        bindings={"observable": observable},
    )
    actual = executable.run(transpiler.executor()).result()

    tolerance = 1e-6 if sdk_transpiler.backend_name == "cudaq" else 1e-8
    assert np.isclose(actual, 0.0, atol=tolerance)
