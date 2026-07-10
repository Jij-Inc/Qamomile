"""Build executable Shor order-finding kernels with attached cost models."""

from __future__ import annotations

import math
from collections.abc import Sequence

import qamomile.circuit as qmc
from qamomile.circuit.frontend.qkernel import QKernel
from qamomile.circuit.stdlib.arithmetic import modmul_const


def _emit_modular_power_schedule(
    counting: qmc.Vector[qmc.Qubit],
    work: qmc.Vector[qmc.Qubit],
    schedule: Sequence[tuple[int, int]],
    modulus: int,
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
    """Emit classically precomputed controlled modular multiplications.

    Args:
        counting (qmc.Vector[qmc.Qubit]): Phase-estimation control register.
        work (qmc.Vector[qmc.Qubit]): Modular arithmetic register.
        schedule (Sequence[tuple[int, int]]): ``(control_index, multiplier)``
            entries for all exponent bits.
        modulus (int): Arithmetic modulus.

    Returns:
        tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]: Updated counting
        and work registers.
    """
    for control_index, multiplier in schedule:
        result = modmul_const(
            work,
            multiplier=multiplier,
            modulus=modulus,
            control=counting[control_index],
        )
        assert isinstance(result, tuple)
        control, work = result
        counting[control_index] = control
    return counting, work


def shor_order_finding(
    base: int,
    modulus: int,
) -> QKernel[..., qmc.Vector[qmc.Bit]]:
    """Create an executable order-finding qkernel for ``base mod modulus``.

    The returned object is an ordinary ``QKernel``. Users transpile and run it
    directly, or pass the same object to ``estimate_resources``. The default
    estimate walks this body and composes the attached primitive models;
    ``EXACT_BODY`` instead traverses the concrete reversible gates used for
    execution.

    Args:
        base (int): Integer whose multiplicative order should be found.
        modulus (int): Composite modulus greater than two.

    Returns:
        QKernel[..., qmc.Vector[qmc.Bit]]: Zero-argument executable kernel whose
        output is the measured phase-estimation register.

    Raises:
        ValueError: If the inputs do not define a reversible modular map.

    Example:
        >>> order_finding = shor_order_finding(base=2, modulus=15)
        >>> estimate = order_finding.estimate_resources()
        >>> int(estimate.qubits)
        12
    """
    if modulus <= 2:
        raise ValueError(f"modulus must be greater than two, got {modulus}.")
    if base <= 1 or base >= modulus:
        raise ValueError(
            f"base must satisfy 1 < base < modulus; got {base} and {modulus}."
        )
    if math.gcd(base, modulus) != 1:
        raise ValueError(
            f"base and modulus must be coprime; got gcd({base}, {modulus}) != 1."
        )

    num_bits = modulus.bit_length()
    counting_bits = 2 * num_bits
    schedule = tuple(
        (index, pow(base, 1 << index, modulus)) for index in range(counting_bits)
    )

    @qmc.composite_gate(name="shor_order_finding")
    def implementation(
        counting: qmc.Vector[qmc.Qubit],
        work: qmc.Vector[qmc.Qubit],
    ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
        """Execute phase estimation over modular exponentiation.

        Args:
            counting (qmc.Vector[qmc.Qubit]): Phase-estimation register.
            work (qmc.Vector[qmc.Qubit]): Modular arithmetic register.

        Returns:
            tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]: Updated
            counting and work registers.
        """
        work[0] = qmc.x(work[0])
        counting = qmc.h(counting)
        counting, work = _emit_modular_power_schedule(
            counting,
            work,
            schedule,
            modulus,
        )
        counting = qmc.iqft(counting)
        return counting, work

    @qmc.qkernel
    def entrypoint() -> qmc.Vector[qmc.Bit]:
        """Invoke the executable, modeled order-finding composite.

        Returns:
            qmc.Vector[qmc.Bit]: Measured phase-estimation register.
        """
        counting = qmc.qubit_array(counting_bits, name="counting")
        work = qmc.qubit_array(num_bits, name="work")
        counting, _work = implementation(counting, work)
        return qmc.measure(counting)

    return entrypoint


__all__ = ["shor_order_finding"]
