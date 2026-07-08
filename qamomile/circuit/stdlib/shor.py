"""Shor order-finding resource-estimation kernel.

:func:`shor_order_finding` is a ready-to-estimate qkernel that expresses the
*resource structure* of Shor's order-finding / RSA-factoring algorithm: a
``2n``-qubit phase (counting) register, an ``n``-qubit work register, ``2n``
controlled constant modular multiplications (one per exponent bit), and an
inverse QFT. Each step is a
:func:`~qamomile.circuit.stdlib.arithmetic.modmul_const` box whose resource
model is independent of *which* constant is multiplied, so estimating the
kernel reproduces the book's RSA-2048 figures directly. Pass the bit-width via
``substitutions`` to get a concrete figure without building a large circuit:

    >>> est = shor_order_finding.estimate_resources(substitutions={"n": 2048})
    >>> int(est.qubits)                        # ~3n algorithmic logical qubits
    6144
    >>> round(float(est.gates.non_clifford) / 1e9, 3)   # ~0.3 n^3 non-Clifford
    2.585

**Scope — this is a resource-estimation kernel, not an executable order
finder.** Each controlled step is an *abstract* constant modular multiplication
(:func:`~qamomile.circuit.stdlib.arithmetic.modmul_const` with no specific
``base``/``modulus``): the resource estimate is genuinely schedule-independent
(a constant modular multiplication costs the same for every constant), so the
kernel commits to no fake schedule and no fake modulus. Consequently it has no
executable body and raises at transpile time — it exists to be *estimated*, not
run. For a genuinely period-correct executable order finder, unroll the true
``base**(2**k) mod N`` schedule at a concrete width using ``modmul_const`` with
concrete cyclic-shift constants (see
``tests/circuit/test_shor.py::test_shor_true_schedule_finds_period``).
"""

from __future__ import annotations

import qamomile.circuit as qmc
from qamomile.circuit.stdlib.arithmetic import modmul_const


@qmc.qkernel
def shor_order_finding(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    """Build the Shor order-finding resource structure on an ``n``-bit modulus.

    Lays out the canonical order-finding structure: ``2n`` counting qubits in
    uniform superposition, an ``n``-qubit work register, ``2n`` controlled
    abstract constant modular multiplications (one per exponent bit), and an
    inverse QFT on the counting register. At a symbolic ``n`` the attached
    modular multiplication resource models make ``estimate_resources`` reproduce
    the literature RSA figures (``~3n`` logical qubits, ``~0.3 n**3``
    non-Clifford gates). Because the modular multiplications are abstract
    (estimation-only), the kernel has no executable body and raises at transpile
    time — see the module docstring.

    Args:
        n (qmc.UInt): Modulus bit-width. Left symbolic for resource estimation
            (``estimate_resources(substitutions={"n": 2048})`` for a concrete
            figure).

    Returns:
        qmc.Vector[qmc.Bit]: Measured counting register.
    """
    counting = qmc.qubit_array(2 * n, name="counting")
    work = qmc.qubit_array(n, name="work")
    work[0] = qmc.x(work[0])
    for k in qmc.range(2 * n):
        counting[k] = qmc.h(counting[k])
    for k in qmc.range(2 * n):
        counting[k], work = modmul_const(work, control=counting[k])
    counting = qmc.iqft(counting)
    return qmc.measure(counting)


__all__ = ["shor_order_finding"]
