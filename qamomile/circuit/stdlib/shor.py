"""Build executable Shor order-finding kernels with body-derived resources."""

from __future__ import annotations

import math
import typing

import qamomile.circuit as qmc
from qamomile.circuit.frontend.qkernel import QKernel
from qamomile.circuit.stdlib.arithmetic import _modmul_const_body


def _validate_work_register_width(n: qmc.UInt, modulus: int) -> qmc.UInt:
    """Validate a concrete Shor work-register width during tracing.

    Args:
        n (qmc.UInt): Concrete or symbolic work-register width.
        modulus (int): Problem modulus whose bit length must fit.

    Returns:
        qmc.UInt: The original width handle.

    Raises:
        ValueError: If a concrete width is smaller than
            ``modulus.bit_length()``.
    """
    if n.value.is_constant():
        width = int(n.value.get_const())
        required = modulus.bit_length()
        if width < required:
            raise ValueError(
                f"Shor work-register width n={width} is too small for "
                f"modulus={modulus}; require n >= {required}."
            )
    return n


def shor_order_finding(
    base: int,
    modulus: int,
) -> QKernel[..., qmc.Vector[qmc.Bit]]:
    """Create an executable order-finding qkernel for ``base mod modulus``.

    The returned object is an ordinary ``QKernel``. Users transpile and run it
    directly, or estimate the same body symbolically. The kernel's ``n`` input
    defaults to ``modulus.bit_length()`` for execution, while resource
    estimation keeps it symbolic until ``inputs={"n": ...}`` specializes the
    resulting expressions. Execution binds ``n=modulus.bit_length()``.

    Args:
        base (int): Integer whose multiplicative order should be found.
        modulus (int): Composite modulus greater than two.

    Returns:
        QKernel[..., qmc.Vector[qmc.Bit]]: Executable kernel whose ``n`` input
            controls the register width and whose output is the measured
            phase-estimation register.

    Raises:
        ValueError: If the inputs do not define a reversible modular map.

    Example:
        >>> order_finding = shor_order_finding(base=2, modulus=15)
        >>> estimate = order_finding.estimate_resources()
        >>> "n" in estimate.parameters
        True
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

    inverse_base = pow(base, -1, modulus)

    @qmc.qkernel
    def modular_power_schedule(
        counting: qmc.Vector[qmc.Qubit],
        work: qmc.Vector[qmc.Qubit],
        accumulator: qmc.Vector[qmc.Qubit],
        addend: qmc.Vector[qmc.Qubit],
        modulus_register: qmc.Vector[qmc.Qubit],
        carry: qmc.Qubit,
        overflow: qmc.Qubit,
        flag: qmc.Qubit,
        domain: qmc.Qubit,
        enable: qmc.Qubit,
    ) -> tuple[
        qmc.Vector[qmc.Qubit],
        qmc.Vector[qmc.Qubit],
        qmc.Vector[qmc.Qubit],
        qmc.Vector[qmc.Qubit],
        qmc.Vector[qmc.Qubit],
        qmc.Qubit,
        qmc.Qubit,
        qmc.Qubit,
        qmc.Qubit,
        qmc.Qubit,
    ]:
        """Apply all controlled modular powers in one symbolic loop.

        Args:
            counting (qmc.Vector[qmc.Qubit]): Phase-estimation controls.
            work (qmc.Vector[qmc.Qubit]): Modular arithmetic register.
            accumulator (qmc.Vector[qmc.Qubit]): Clean multiplication
                accumulator.
            addend (qmc.Vector[qmc.Qubit]): Clean constant-addend workspace.
            modulus_register (qmc.Vector[qmc.Qubit]): Clean modulus workspace.
            carry (qmc.Qubit): Clean ripple-carry workspace.
            overflow (qmc.Qubit): Clean high-bit workspace.
            flag (qmc.Qubit): Clean modular-reduction flag.
            domain (qmc.Qubit): Clean input-domain flag.
            enable (qmc.Qubit): Clean conjunction workspace.

        Returns:
            tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit],
            qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit],
            qmc.Vector[qmc.Qubit], qmc.Qubit, qmc.Qubit, qmc.Qubit,
            qmc.Qubit, qmc.Qubit]: Updated counting/work handles and restored
                workspace.
        """
        multiplier = qmc.uint(base % modulus)
        inverse_multiplier = qmc.uint(inverse_base % modulus)
        for control_index in qmc.range(counting.shape[0]):
            (
                control,
                work,
                accumulator,
                addend,
                modulus_register,
                carry,
                overflow,
                flag,
                domain,
                enable,
            ) = _modmul_const_body(
                counting[control_index],
                work,
                accumulator,
                addend,
                modulus_register,
                carry,
                overflow,
                flag,
                domain,
                enable,
                multiplier,
                inverse_multiplier,
                modulus,
            )
            counting[control_index] = control
            multiplier = (multiplier * multiplier) % modulus
            inverse_multiplier = (inverse_multiplier * inverse_multiplier) % modulus
        return (
            counting,
            work,
            accumulator,
            addend,
            modulus_register,
            carry,
            overflow,
            flag,
            domain,
            enable,
        )

    @qmc.composite_gate(name="shor_order_finding")
    def implementation(
        counting: qmc.Vector[qmc.Qubit],
        work: qmc.Vector[qmc.Qubit],
        accumulator: qmc.Vector[qmc.Qubit],
        addend: qmc.Vector[qmc.Qubit],
        modulus_register: qmc.Vector[qmc.Qubit],
        carry: qmc.Qubit,
        overflow: qmc.Qubit,
        flag: qmc.Qubit,
        domain: qmc.Qubit,
        enable: qmc.Qubit,
    ) -> tuple[
        qmc.Vector[qmc.Qubit],
        qmc.Vector[qmc.Qubit],
        qmc.Vector[qmc.Qubit],
        qmc.Vector[qmc.Qubit],
        qmc.Vector[qmc.Qubit],
        qmc.Qubit,
        qmc.Qubit,
        qmc.Qubit,
        qmc.Qubit,
        qmc.Qubit,
    ]:
        """Execute phase estimation over modular exponentiation.

        Args:
            counting (qmc.Vector[qmc.Qubit]): Phase-estimation register.
            work (qmc.Vector[qmc.Qubit]): Modular arithmetic register.
            accumulator (qmc.Vector[qmc.Qubit]): Multiplication accumulator.
            addend (qmc.Vector[qmc.Qubit]): Constant-addend workspace.
            modulus_register (qmc.Vector[qmc.Qubit]): Modulus workspace.
            carry (qmc.Qubit): Ripple-carry workspace.
            overflow (qmc.Qubit): High-bit workspace.
            flag (qmc.Qubit): Modular-reduction flag.
            domain (qmc.Qubit): Input-domain flag.
            enable (qmc.Qubit): Conjunction workspace.

        Returns:
            tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit],
            qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit],
            qmc.Vector[qmc.Qubit], qmc.Qubit, qmc.Qubit, qmc.Qubit,
            qmc.Qubit, qmc.Qubit]: Updated algorithm state and restored
                workspace.
        """
        work[0] = qmc.x(work[0])
        counting = qmc.h(counting)
        (
            counting,
            work,
            accumulator,
            addend,
            modulus_register,
            carry,
            overflow,
            flag,
            domain,
            enable,
        ) = modular_power_schedule(
            counting,
            work,
            accumulator,
            addend,
            modulus_register,
            carry,
            overflow,
            flag,
            domain,
            enable,
        )
        counting = qmc.iqft(counting)
        return (
            counting,
            work,
            accumulator,
            addend,
            modulus_register,
            carry,
            overflow,
            flag,
            domain,
            enable,
        )

    @qmc.qkernel
    def entrypoint(
        n: qmc.UInt = typing.cast(qmc.UInt, modulus.bit_length()),
    ) -> qmc.Vector[qmc.Bit]:
        """Invoke the executable body-backed order-finding composite.

        Args:
            n (qmc.UInt): Work-register width. Defaults to
                ``modulus.bit_length()`` and must not be smaller.

        Returns:
            qmc.Vector[qmc.Bit]: Measured phase-estimation register.
        """
        n = _validate_work_register_width(n, modulus)
        counting = qmc.qubit_array(2 * n, name="counting")
        work = qmc.qubit_array(n, name="work")
        accumulator = qmc.qubit_array(n, name="accumulator")
        addend = qmc.qubit_array(n, name="addend")
        modulus_register = qmc.qubit_array(n, name="modulus")
        carry = qmc.qubit("carry")
        overflow = qmc.qubit("overflow")
        flag = qmc.qubit("flag")
        domain = qmc.qubit("domain")
        enable = qmc.qubit("enable")
        (
            counting,
            _work,
            _accumulator,
            _addend,
            _modulus_register,
            _carry,
            _overflow,
            _flag,
            _domain,
            _enable,
        ) = implementation(
            counting,
            work,
            accumulator,
            addend,
            modulus_register,
            carry,
            overflow,
            flag,
            domain,
            enable,
        )
        return qmc.measure(counting)

    return entrypoint


__all__ = ["shor_order_finding"]
