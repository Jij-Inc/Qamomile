"""Build executable Shor order-finding kernels with body-derived resources."""

from __future__ import annotations

import dataclasses
import math

import qamomile.circuit as qmc
from qamomile.circuit.frontend.qkernel import QKernel
from qamomile.circuit.stdlib.arithmetic import _modmul_const_body, _phase_shift_if


@qmc.struct
class _ShorWorkspace:
    """Group the reusable quantum state for iterative phase estimation.

    Args:
        phase (qmc.Qubit): Recycled phase-estimation qubit.
        work (qmc.Vector[qmc.Qubit]): Modular value register.
        accumulator (qmc.Vector[qmc.Qubit]): Multiplication accumulator.
        lookup (qmc.Vector[qmc.Qubit]): Lookup output register.
        address (qmc.Vector[qmc.Qubit]): Fixed-window lookup address.
        carry (qmc.Qubit): First carry-venting workspace qubit.
        vent (qmc.Qubit): Second carry-venting workspace qubit.
        overflow (qmc.Qubit): Extended-adder high bit.
        flag (qmc.Qubit): Modular-reduction flag.
        domain (qmc.Qubit): Valid-domain flag.
        enable (qmc.Qubit): Controlled-window conjunction workspace.
    """

    phase: qmc.Qubit
    work: qmc.Vector[qmc.Qubit]
    accumulator: qmc.Vector[qmc.Qubit]
    lookup: qmc.Vector[qmc.Qubit]
    address: qmc.Vector[qmc.Qubit]
    carry: qmc.Qubit
    vent: qmc.Qubit
    overflow: qmc.Qubit
    flag: qmc.Qubit
    domain: qmc.Qubit
    enable: qmc.Qubit


def _allocate_shor_workspace(
    register_size: int,
    window_size: int,
) -> _ShorWorkspace:
    """Allocate a clean workspace for one factoring kernel.

    Args:
        register_size (int): Concrete modular-register width.
        window_size (int): Lookup address width.

    Returns:
        _ShorWorkspace: Newly allocated frontend handles.
    """
    return _ShorWorkspace(
        phase=qmc.qubit("phase"),
        work=qmc.qubit_array(register_size, name="work"),
        accumulator=qmc.qubit_array(register_size, name="accumulator"),
        lookup=qmc.qubit_array(register_size, name="lookup"),
        address=qmc.qubit_array(window_size, name="address"),
        carry=qmc.qubit("carry"),
        vent=qmc.qubit("vent"),
        overflow=qmc.qubit("overflow"),
        flag=qmc.qubit("flag"),
        domain=qmc.qubit("domain"),
        enable=qmc.qubit("enable"),
    )


def _apply_modular_multiplication(
    workspace: _ShorWorkspace,
    multiplier: int,
    inverse_multiplier: int,
    modulus: int,
    register_size: int,
    window_size: int,
) -> _ShorWorkspace:
    """Apply one controlled modular multiplication to a workspace.

    Args:
        workspace (_ShorWorkspace): Input phase and arithmetic state.
        multiplier (int): Compile-time modular multiplier.
        inverse_multiplier (int): Multiplicative inverse modulo ``modulus``.
        modulus (int): Compile-time modular-arithmetic modulus.
        register_size (int): Concrete modular-register width.
        window_size (int): Lookup address width.

    Returns:
        _ShorWorkspace: Successor workspace containing the returned handles.
    """
    (
        phase,
        work,
        accumulator,
        lookup,
        address,
        carry,
        vent,
        overflow,
        flag,
        domain,
        enable,
    ) = _modmul_const_body(
        workspace.phase,
        workspace.work,
        workspace.accumulator,
        workspace.lookup,
        workspace.address,
        workspace.carry,
        workspace.vent,
        workspace.overflow,
        workspace.flag,
        workspace.domain,
        workspace.enable,
        qmc.uint(multiplier),
        qmc.uint(inverse_multiplier),
        qmc.uint(modulus),
        register_size,
        window_size,
    )
    return _ShorWorkspace(
        phase=phase,
        work=work,
        accumulator=accumulator,
        lookup=lookup,
        address=address,
        carry=carry,
        vent=vent,
        overflow=overflow,
        flag=flag,
        domain=domain,
        enable=enable,
    )


def _apply_semiclassical_iqft_feedback(
    phase: qmc.Qubit,
    previous_bits: list[qmc.Bit],
    round_index: int,
) -> qmc.Qubit:
    """Apply inverse-QFT feedback from already measured low-order bits.

    Args:
        phase (qmc.Qubit): Reused phase-estimation qubit.
        previous_bits (list[qmc.Bit]): Previously measured bits, least
            significant first.
        round_index (int): Zero-based iterative-QPE round.

    Returns:
        qmc.Qubit: Phase qubit after all classically controlled corrections.
    """
    for previous_index, bit in enumerate(previous_bits):
        distance = round_index - previous_index + 1
        phase = _phase_shift_if(bit, phase, -2 * math.pi / (2**distance))
    return phase


def _iterative_modular_phase_bits(
    workspace: _ShorWorkspace,
    *,
    multipliers: tuple[int, ...],
    modulus: int,
    register_size: int,
    window_size: int,
) -> tuple[_ShorWorkspace, tuple[qmc.Bit, ...]]:
    """Measure one iterative modular-eigenphase schedule.

    Args:
        workspace (_ShorWorkspace): Reusable phase and arithmetic state.
        multipliers (tuple[int, ...]): Controlled modular multipliers ordered
            from the largest binary power to the smallest.
        modulus (int): Modular-arithmetic modulus.
        register_size (int): Concrete modular-register width.
        window_size (int): Lookup address width.

    Returns:
        tuple[_ShorWorkspace, tuple[qmc.Bit, ...]]: Successor workspace and
            little-endian phase bits.
    """
    measured_bits: list[qmc.Bit] = []
    for round_index, multiplier in enumerate(multipliers):
        phase = qmc.h(workspace.phase)
        if multiplier != 1:
            inverse_multiplier = pow(multiplier, -1, modulus)
            workspace = dataclasses.replace(workspace, phase=phase)
            workspace = _apply_modular_multiplication(
                workspace,
                multiplier,
                inverse_multiplier,
                modulus,
                register_size,
                window_size,
            )
            phase = workspace.phase
        phase = _apply_semiclassical_iqft_feedback(
            phase,
            measured_bits,
            round_index,
        )
        phase = qmc.h(phase)
        phase, measured = qmc.measure_reset(phase)
        workspace = dataclasses.replace(workspace, phase=phase)
        measured_bits.append(measured)
    return workspace, tuple(measured_bits)


def _iterative_order_finding_body(
    workspace: _ShorWorkspace,
    *,
    base: int,
    modulus: int,
    precision: int,
    register_size: int,
    window_size: int,
) -> qmc.Vector[qmc.Bit]:
    """Execute iterative phase estimation for modular multiplication.

    Controlled powers are visited from largest to smallest. Each round first
    yields the next least-significant phase bit, so the returned vector keeps
    Qamomile's usual little-endian measurement order.

    Args:
        workspace (_ShorWorkspace): Reusable phase and arithmetic state.
        base (int): Modular multiplication base.
        modulus (int): Composite modulus.
        precision (int): Number of phase bits to measure.
        register_size (int): Concrete modular-register width.
        window_size (int): Lookup address width.

    Returns:
        qmc.Vector[qmc.Bit]: Measured phase bits, least significant first.
    """
    workspace.work[0] = qmc.x(workspace.work[0])
    multipliers = tuple(
        pow(base, 2**exponent, modulus) for exponent in range(precision - 1, -1, -1)
    )
    _, measured_bits = _iterative_modular_phase_bits(
        workspace,
        multipliers=multipliers,
        modulus=modulus,
        register_size=register_size,
        window_size=window_size,
    )
    output = qmc.bit_array(precision, name="phase_bits")
    for index, measured in enumerate(measured_bits):
        output[index] = measured
    return output


def _store_measured_bits(
    output: qmc.Vector[qmc.Bit],
    measured_bits: tuple[qmc.Bit, ...],
    *,
    offset: int,
) -> qmc.Vector[qmc.Bit]:
    """Store a trace-time tuple of measurements into a Bit vector.

    This helper deliberately performs Python iteration outside the decorated
    qkernel AST. The tuple length and offset are factory-time constants, so
    the frontend emits one explicit classical store per destination slot.

    Args:
        output (qmc.Vector[qmc.Bit]): Destination Bit vector.
        measured_bits (tuple[qmc.Bit, ...]): Measurement handles in output
            order.
        offset (int): First destination index.

    Returns:
        qmc.Vector[qmc.Bit]: The destination after all stores.
    """
    for index, measured in enumerate(measured_bits):
        output[offset + index] = measured
    return output


def shor_order_finding(
    base: int,
    modulus: int,
    *,
    window_size: int = 2,
    precision: int | None = None,
) -> QKernel[..., qmc.Vector[qmc.Bit]]:
    """Create an executable order-finding qkernel for ``base mod modulus``.

    The modulus fixes the work-register width, so the returned kernel has no
    artificial ``n`` runtime argument. It uses one recycled phase qubit and
    measurement feed-forward instead of a coherent ``2n`` counting register.
    With fixed ``window_size``, the body therefore has ``3n + O(1)`` peak
    width and ``O(n**3)`` gates at the default ``2n`` phase precision.

    Args:
        base (int): Integer whose multiplicative order should be found.
        modulus (int): Composite modulus greater than two.
        window_size (int): Lookup width for modular multiplication. Defaults
            to 2.
        precision (int | None): Number of measured phase bits. Defaults to
            twice ``modulus.bit_length()``.

    Returns:
        QKernel[..., qmc.Vector[qmc.Bit]]: Argument-free executable kernel
            returning little-endian phase bits.

    Raises:
        ValueError: If the inputs do not define a reversible modular map.

    Example:
        >>> order_finding = shor_order_finding(base=2, modulus=15)
        >>> estimate = order_finding.estimate_resources()
        >>> estimate.qubits
        21
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
    if window_size < 1:
        raise ValueError(f"window_size must be positive, got {window_size}.")
    register_size = modulus.bit_length()
    if precision is None:
        precision = 2 * register_size
    if precision < 1:
        raise ValueError(f"precision must be positive, got {precision}.")

    def entrypoint() -> qmc.Vector[qmc.Bit]:
        """Execute low-width iterative order finding.

        Returns:
            qmc.Vector[qmc.Bit]: Little-endian phase-estimation bits.
        """
        return _iterative_order_finding_body(
            _allocate_shor_workspace(register_size, window_size),
            base=base,
            modulus=modulus,
            precision=precision,
            register_size=register_size,
            window_size=window_size,
        )

    return qmc.qkernel(entrypoint)


def ekera_hastad_factoring(
    generator: int,
    modulus: int,
    *,
    window_size: int = 2,
) -> QKernel[..., qmc.Vector[qmc.Bit]]:
    """Create the quantum short-DLP stage for Ekerå–Håstad factoring.

    For a balanced semiprime ``N = p*q``, this measures the two modular phase
    schedules associated with ``g`` and ``y**-1``, where
    ``y = g**(N + 1) mod N``. Their precisions are ``2m`` and ``m`` for
    ``m = ceil(n / 2) + 1``. Both schedules recycle the same phase qubit and
    arithmetic workspace; classical lattice post-processing remains outside
    this qkernel.

    Args:
        generator (int): Group element ``g`` coprime to ``modulus``.
        modulus (int): Balanced semiprime to factor.
        window_size (int): Lookup width for modular multiplication. Defaults
            to 2.

    Returns:
        QKernel[..., qmc.Vector[qmc.Bit]]: Argument-free kernel returning the
            ``2m`` long-schedule bits followed by the ``m`` short-schedule
            bits. Each group is little-endian.

    Raises:
        ValueError: If the modular map or lookup width is invalid.
    """
    if modulus <= 2:
        raise ValueError(f"modulus must be greater than two, got {modulus}.")
    if generator <= 1 or generator >= modulus:
        raise ValueError(
            "generator must satisfy 1 < generator < modulus; "
            f"got {generator} and {modulus}."
        )
    if math.gcd(generator, modulus) != 1:
        raise ValueError("generator and modulus must be coprime.")
    if window_size < 1:
        raise ValueError(f"window_size must be positive, got {window_size}.")

    register_size = modulus.bit_length()
    phase_parameter_size = (register_size + 1) // 2 + 1
    long_precision = 2 * phase_parameter_size
    short_precision = phase_parameter_size
    y = pow(generator, modulus + 1, modulus)
    inverse_y = pow(y, -1, modulus)

    def entrypoint() -> qmc.Vector[qmc.Bit]:
        """Execute the low-width short-DLP quantum stage.

        Returns:
            qmc.Vector[qmc.Bit]: Concatenated long and short phase samples.
        """
        workspace = _allocate_shor_workspace(register_size, window_size)
        workspace.work[0] = qmc.x(workspace.work[0])
        long_multipliers = tuple(
            pow(generator, 2**exponent, modulus)
            for exponent in range(long_precision - 1, -1, -1)
        )
        workspace, long_bits = _iterative_modular_phase_bits(
            workspace,
            multipliers=long_multipliers,
            modulus=modulus,
            register_size=register_size,
            window_size=window_size,
        )
        short_multipliers = tuple(
            pow(inverse_y, 2**exponent, modulus)
            for exponent in range(short_precision - 1, -1, -1)
        )
        _, short_bits = _iterative_modular_phase_bits(
            workspace,
            multipliers=short_multipliers,
            modulus=modulus,
            register_size=register_size,
            window_size=window_size,
        )
        output = qmc.bit_array(
            long_precision + short_precision,
            name="phase_bits",
        )
        output = _store_measured_bits(output, long_bits, offset=0)
        output = _store_measured_bits(
            output,
            short_bits,
            offset=long_precision,
        )
        return output

    return qmc.qkernel(entrypoint)


__all__ = ["ekera_hastad_factoring", "shor_order_finding"]
