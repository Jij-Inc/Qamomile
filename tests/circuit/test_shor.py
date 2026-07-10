"""Tests for executable, resource-modeled Shor order finding."""

from __future__ import annotations

import numpy as np
import pytest

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
    assert estimate.qubits == 12
    assert estimate.width.dirty_ancilla_qubits == 4
    # Eight primitive modmul models compose to the reference 0.3*n^3 term.
    # IQFT rotations are reported separately and also contribute to the broader
    # non-Clifford total.
    assert estimate.gates.toffoli == pytest.approx(19.2)
    assert estimate.gates.rotation == 28
    assert estimate.gates.non_clifford == pytest.approx(47.2)
    assert estimate.calls.calls_by_name["modmul_const"] == 8
    assert estimate.trace is None


def test_shor_exact_body_uses_the_executable_implementation() -> None:
    """EXACT_BODY traverses real gates instead of an estimation-only stub."""
    kernel = qmc.shor_order_finding(base=2, modulus=15)

    exact = kernel.estimate_resources(policy=qmc.ResourcePolicy.EXACT_BODY)

    assert exact.qubits == 12
    assert exact.gates.total > 0
    assert "modmul_const" not in exact.calls.calls_by_name


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


def test_shor_order_finding_recovers_period_four(sdk_transpiler) -> None:
    """The public kernel recovers the four phase peaks for 2 mod 15."""
    kernel = qmc.shor_order_finding(base=2, modulus=15)
    transpiler = sdk_transpiler.transpiler
    executable = transpiler.transpile(kernel)
    result = executable.sample(transpiler.executor(), shots=2048).result()

    counts = {_basis_value(bits): count for bits, count in result.results}
    targets = {0, 64, 128, 192}
    on_peak = sum(counts.get(value, 0) for value in targets)

    assert on_peak / result.shots > 0.7
    assert all(counts.get(value, 0) / result.shots > 0.1 for value in targets)


@qmc.qkernel
def _shor_state_expval(observable: qmc.Observable) -> qmc.Float:
    """Evaluate an observable before measurement for the 2 mod 15 schedule."""
    counting = qmc.qubit_array(8, "counting")
    work = qmc.qubit_array(4, "work")
    work[0] = qmc.x(work[0])
    counting = qmc.h(counting)
    counting[0], work = qmc.modmul_const(
        work,
        multiplier=2,
        modulus=15,
        control=counting[0],
    )
    counting[1], work = qmc.modmul_const(
        work,
        multiplier=4,
        modulus=15,
        control=counting[1],
    )
    counting = qmc.iqft(counting)
    return qmc.expval(counting, observable)


@pytest.mark.parametrize("qubit", [0, 1, 3])
def test_shor_cross_backend_expval(sdk_transpiler, qubit: int) -> None:
    """The modular-exponentiation state agrees on the expval path.

    Args:
        sdk_transpiler: Parametrized supported backend fixture.
        qubit (int): Counting-register Z observable index.
    """
    pytest.importorskip("qiskit")
    import qamomile.observable as qm_o
    from qamomile.qiskit import QiskitTranspiler

    observable = qm_o.Z(qubit)
    reference_transpiler = QiskitTranspiler()
    reference_executable = reference_transpiler.transpile(
        _shor_state_expval,
        bindings={"observable": observable},
    )
    reference = reference_executable.run(reference_transpiler.executor()).result()

    transpiler = sdk_transpiler.transpiler
    executable = transpiler.transpile(
        _shor_state_expval,
        bindings={"observable": observable},
    )
    actual = executable.run(transpiler.executor()).result()

    tolerance = 1e-6 if sdk_transpiler.backend_name == "cudaq" else 1e-8
    assert np.isclose(actual, reference, atol=tolerance)
