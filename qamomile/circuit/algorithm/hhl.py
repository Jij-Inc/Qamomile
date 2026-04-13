"""Register-sized HHL circuit skeleton.

The Harrow-Hassidim-Lloyd (HHL) algorithm prepares a quantum state
|x> proportional to A^{-1}|b> for a Hermitian matrix A. The algorithm uses
quantum phase estimation (QPE) to extract eigenvalues, a reciprocal rotation
to embed 1/lambda into ancilla amplitudes, and inverse QPE to uncompute
the clock register.

The solution state is obtained by post-selecting on the ancilla qubit
measuring |1>.

Important limitations / conventions of this implementation:

* ``system`` is an n-qubit system register representing a ``2**n``-dimensional
  state |b>.  This generalizes the previous single-qubit-only API.
* The QPE clock register is interpreted through an explicit eigenvalue-decoding
  rule controlled by ``phase_scale``.  The caller must set ``phase_scale``
  consistently with the unitary used in QPE.  For the standard convention
  ``U = exp(i A t)``, use ``phase_scale = 2*pi / t``.
* If ``signed=True``, the clock register is interpreted in two's-complement
  form, representing a spectrum in ``[-phase_scale/2, phase_scale/2)``.
* If ``signed=False`` (default), the clock register is interpreted as an
  unsigned phase estimate in ``[0, phase_scale)``.
* The zero-eigenvalue bin cannot be inverted.  This implementation leaves
  that branch untouched, so the caller must ensure that its support is
  negligible or intentionally excluded.
* Standard QPE convention assumes that the IQFT used inside phase estimation
  does **not** include final bit-reversal swaps.  If your underlying
  ``iqft()`` does include those swaps, set ``iqft_includes_swaps=True`` so
  that the reciprocal-rotation control patterns use the correct physical
  qubit order.
* ``strict=True`` is only operationally meaningful when
  ``supported_raw_bins`` is provided.  In that case, invalid reciprocal
  rotations are rejected only on the declared populated bins.
* This is still a circuit skeleton, not a full sparse-oracle / Hamiltonian-
  simulation framework.  The caller must prepare |b> on the full system
  register and supply kernels implementing ``U = exp(i A t)`` and
  ``U^dagger`` on that register.
* As in standard HHL, the output is a quantum state proportional to
  ``A^{-1}|b>`` on the system register, not a classical list of all solution
  components.

Example::

    import math
    import qamomile.circuit as qmc
    from qamomile.circuit.algorithm.hhl import hhl

    @qmc.qkernel
    def u_gate(
        q0: qmc.Qubit,
        q1: qmc.Qubit,
        theta: qmc.Float,
    ) -> tuple[qmc.Qubit, qmc.Qubit]:
        # Example full-register unitary: acts diagonally on the first system
        # qubit and trivially on the second.
        q0 = qmc.p(q0, theta)
        return q0, q1

    @qmc.qkernel
    def u_inv_gate(
        q0: qmc.Qubit,
        q1: qmc.Qubit,
        theta: qmc.Float,
    ) -> tuple[qmc.Qubit, qmc.Qubit]:
        q0 = qmc.p(q0, -1.0 * theta)
        return q0, q1

    @qmc.qkernel
    def solve() -> qmc.Bit:
        sys = qmc.qubit_array(2, name="sys")
        # Prepare a basis state supported entirely on the eigenphase-1/2
        # subspace of the example unitary.
        sys[0] = qmc.x(sys[0])
        clock = qmc.qubit_array(2, name="clock")
        anc = qmc.qubit("anc")

        # On states with sys[0] = |1>, U contributes eigenphase 1/2.
        # With 2 clock qubits and phase_scale = 2*pi, that maps exactly
        # to raw bin 2 and eigenvalue lambda = pi.
        sys, clock, anc = hhl(
            sys, clock, anc,
            unitary=u_gate,
            inv_unitary=u_inv_gate,
            scaling=0.25,
            phase_scale=2.0 * math.pi,
            supported_raw_bins=(2,),
            strict=True,
            iqft_includes_swaps=False,
            theta=math.pi,
        )
        return qmc.measure(anc)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Collection

import qamomile.circuit as qmc
from qamomile.circuit.frontend.handle import Qubit, Vector
from qamomile.circuit.frontend.handle.utils import get_size
from qamomile.circuit.stdlib.qft import iqft, qft

if TYPE_CHECKING:
    from qamomile.circuit.frontend.qkernel import QKernel


@qmc.qkernel
def _ry_gate(q: qmc.Qubit, angle: qmc.Float) -> qmc.Qubit:
    """Single RY gate wrapper for creating multi-controlled versions."""
    return qmc.ry(q, angle)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _basis_bit(
    raw: int,
    qubit_index: int,
    n_clock: int,
    little_endian_clock: bool,
) -> int:
    """Return the logical bit seen by ``clock[qubit_index]`` for basis state *raw*.

    If *little_endian_clock* is True, ``clock[i]`` is treated as the i-th
    least-significant bit.  Otherwise ``clock[0]`` is the most-significant bit.
    """
    if little_endian_clock:
        shift = qubit_index
    else:
        shift = n_clock - 1 - qubit_index
    return (raw >> shift) & 1


def _power_exponent_for_qubit(
    qubit_index: int,
    n_clock: int,
    little_endian_clock: bool,
) -> int:
    """Map a physical clock-qubit index to the logical QPE exponent."""
    if little_endian_clock:
        return qubit_index
    return n_clock - 1 - qubit_index


def _apply_controlled_register_kernel(
    controlled_kernel: "QKernel",
    control: Qubit,
    target: Vector[Qubit],
    **params: Any,
) -> tuple[Qubit, Vector[Qubit]]:
    """Apply a controlled kernel to an entire target register.

    Assumes the controlled kernel follows the calling convention

        controlled_kernel(control, *target, **params)

    and returns a tuple shaped like

        (updated_control, *updated_target_qubits).

    If qamomile uses a different controlled-kernel convention, adapt this
    helper only; the rest of the HHL circuit can stay unchanged.
    """
    n_target = get_size(target)
    target_qubits = [target[i] for i in range(n_target)]

    result = controlled_kernel(control, *target_qubits, **params)

    try:
        result_len = len(result)
    except TypeError as exc:
        raise TypeError(
            "Controlled register kernel must return a tuple-like object "
            f"containing (control, *target_qubits), got {type(result).__name__}."
        ) from exc

    expected_len = 1 + n_target
    if result_len != expected_len:
        raise ValueError(
            "Controlled register kernel returned an unexpected number of "
            f"outputs: got {result_len}, expected {expected_len} "
            "(control + full target register)."
        )

    control = result[0]
    for i in range(n_target):
        target[i] = result[1 + i]

    return control, target


def _normalize_supported_raw_bins(
    supported_raw_bins: Collection[int] | None,
    n_clock: int,
) -> frozenset[int] | None:
    """Validate and normalise *supported_raw_bins* into a frozenset.

    Returns ``None`` when the caller passed ``None`` (meaning "all bins are
    potentially populated").  Otherwise every element must satisfy
    ``0 <= raw < 2**n_clock``.

    The zero bin is rejected because the corresponding eigenvalue estimate is
    zero and therefore not invertible in HHL.
    """
    if supported_raw_bins is None:
        return None
    N = 2**n_clock
    if any((not isinstance(r, int)) or isinstance(r, bool) for r in supported_raw_bins):
        raise TypeError("supported_raw_bins must contain integers only.")
    bins = frozenset(supported_raw_bins)
    if not bins:
        raise ValueError(
            "supported_raw_bins must be non-empty when provided."
        )
    bad = [r for r in bins if r < 0 or r >= N]
    if bad:
        raise ValueError(
            f"supported_raw_bins contains out-of-range values {bad} "
            f"for n_clock={n_clock} (valid range 0..{N - 1})."
        )
    if 0 in bins:
        raise ValueError(
            "supported_raw_bins must not contain 0 because the "
            "zero-eigenvalue branch is not invertible."
        )
    return bins


def _decode_eigenvalue_from_clock(
    raw: int,
    n_clock: int,
    phase_scale: float,
    signed: bool,
) -> float:
    """Decode a computational-basis clock value into an eigenvalue estimate.

    Unsigned mode (``signed=False``)::

        raw in {0, ..., 2^m - 1}
        lambda_hat = phase_scale * raw / 2^m

    Signed / two's-complement mode (``signed=True``)::

        signed_raw = raw           if raw <  2^(m-1)
                   = raw - 2^m     if raw >= 2^(m-1)
        lambda_hat = phase_scale * signed_raw / 2^m

    so the represented interval is ``[-phase_scale/2, phase_scale/2)``.

    Args:
        raw: Integer value of the clock register basis state.
        n_clock: Number of clock qubits.
        phase_scale: Physical scale mapping phase estimates back to
            eigenvalues.  For ``U = exp(i A t)``, use ``2*pi/t``.
        signed: Whether to interpret *raw* as two's-complement.

    Returns:
        Decoded eigenvalue estimate.

    Raises:
        ValueError: If *n_clock* < 1 or *phase_scale* <= 0.
    """
    if n_clock < 1:
        raise ValueError("n_clock must be >= 1.")
    if phase_scale <= 0.0:
        raise ValueError("phase_scale must be positive.")

    N = 2**n_clock

    if signed:
        signed_raw = raw if raw < (N // 2) else raw - N
        return phase_scale * signed_raw / float(N)

    return phase_scale * raw / float(N)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def reciprocal_rotation(
    clock: Vector[Qubit],
    ancilla: Qubit,
    scaling: float,
    phase_scale: float,
    signed: bool = False,
    little_endian_clock: bool = True,
    strict: bool = True,
    supported_raw_bins: Collection[int] | None = None,
    iqft_includes_swaps: bool = False,
) -> tuple[Vector[Qubit], Qubit]:
    """Apply eigenvalue-conditioned reciprocal rotation on the ancilla.

    For each computational basis state |raw> of the clock register, decode
    the corresponding eigenvalue estimate

        lambda_hat = phase_scale * raw / 2^m     (unsigned)

    or its two's-complement variant (signed), then apply

        theta_raw = 2 * arcsin(C / lambda_hat)

    to the ancilla whenever that angle is well-defined.  After this rotation,
    the ancilla's |1> amplitude is proportional to ``C / lambda_hat`` on that
    branch.

    In signed mode, negative eigenvalues produce negative rotation angles,
    which preserves the sign of the reciprocal.

    Args:
        clock: Clock register containing eigenvalue encoding from QPE.
            Must have a fixed (concrete) size.
        ancilla: Ancilla qubit (initialized to |0>) to receive the
            1/lambda amplitude encoding.
        scaling: Normalization constant C used in the reciprocal map.
            Must satisfy ``|C / lambda_hat| <= 1`` for every populated
            eigenvalue bin.
        phase_scale: Positive factor converting a QPE clock value into a
            physical eigenvalue estimate.  For ``U = exp(i A t)``, use
            ``phase_scale = 2*pi / t``.
        signed: If True, interpret the QPE clock value using two's
            complement, representing a signed spectrum in
            ``[-phase_scale/2, phase_scale/2)``.
        little_endian_clock: Whether ``clock[0]`` is the least-significant
            clock bit.  This affects the bit-flip control patterns.
        strict: If True **and** *supported_raw_bins* is provided, raise
            ``ValueError`` when a bin in *supported_raw_bins* would
            require ``|C / lambda_hat| > 1``.  When *supported_raw_bins*
            is ``None``, *strict* has no effect because the set of
            populated bins is unknown.
        supported_raw_bins: Optional collection of clock basis-state
            integers that the QPE is expected to populate.  When given,
            bins outside this set are skipped entirely (no gate emitted),
            and the *strict* check is applied only to these bins.
            Pass ``None`` (default) to emit rotations for all
            representable nonzero bins.
        iqft_includes_swaps: If True, the IQFT used in the preceding QPE
            step includes bit-reversal swaps, so the physical qubit order
            after IQFT is reversed relative to the logical order.  This
            flag flips the endianness used for control-bit patterns.
            Standard QPE convention is ``False``.

    Returns:
        Updated (clock, ancilla) registers.
    """
    n_clock = get_size(clock)
    if scaling <= 0.0:
        raise ValueError("scaling must be positive.")

    populated_bins = _normalize_supported_raw_bins(supported_raw_bins, n_clock)

    # If iqft_includes_swaps, the physical qubit order after IQFT is
    # bit-reversed relative to the logical order, so we flip endianness
    # for the control-bit pattern.
    rotation_little_endian = little_endian_clock ^ iqft_includes_swaps

    N = 2**n_clock

    mc_ry = qmc.controlled(_ry_gate, num_controls=n_clock)

    for raw in range(N):
        lambda_hat = _decode_eigenvalue_from_clock(
            raw=raw,
            n_clock=n_clock,
            phase_scale=phase_scale,
            signed=signed,
        )

        # Zero-eigenvalue branch is not invertible.
        if math.isclose(lambda_hat, 0.0, abs_tol=1e-15):
            continue

        # Skip bins the caller declared as unpopulated.
        if populated_bins is not None and raw not in populated_bins:
            continue

        ratio = scaling / lambda_hat
        if abs(ratio) > 1.0:
            if strict and populated_bins is not None:
                raise ValueError(
                    "Invalid reciprocal rotation: |scaling / lambda_hat| > 1 "
                    f"for raw={raw} (lambda_hat={lambda_hat:.6g}). "
                    "Choose a smaller scaling, a larger phase_scale, or "
                    "remove this bin from supported_raw_bins."
                )
            continue

        theta = 2.0 * math.asin(ratio)

        # Flip 0-bits to create control-on-|raw> pattern
        for bit_idx in range(n_clock):
            if not _basis_bit(raw, bit_idx, n_clock, rotation_little_endian):
                clock[bit_idx] = qmc.x(clock[bit_idx])

        # Apply multi-controlled RY (fires when all controls are |1>)
        qubit_list = [clock[i] for i in range(n_clock)]
        result = mc_ry(*qubit_list, ancilla, angle=theta)

        # Write results back
        for i in range(n_clock):
            clock[i] = result[i]
        ancilla = result[n_clock]

        # Undo bit flips
        for bit_idx in range(n_clock):
            if not _basis_bit(raw, bit_idx, n_clock, rotation_little_endian):
                clock[bit_idx] = qmc.x(clock[bit_idx])

    return clock, ancilla


def hhl(
    system: Vector[Qubit],
    clock: Vector[Qubit],
    ancilla: Qubit,
    unitary: "QKernel",
    inv_unitary: "QKernel",
    scaling: float,
    phase_scale: float,
    signed: bool = False,
    little_endian_clock: bool = True,
    strict: bool = True,
    supported_raw_bins: Collection[int] | None = None,
    iqft_includes_swaps: bool = False,
    **params: Any,
) -> tuple[Vector[Qubit], Vector[Qubit], Qubit]:
    """Build an HHL circuit skeleton over an n-qubit system register.

    Constructs an HHL-style circuit that prepares a state proportional
    to the inverse action A^{-1}|b> on the supported spectral subspace,
    conditioned on measuring |1> in the ancilla qubit.

    The system register must already be prepared in state |b> before
    calling this function.  The unitary should implement U = e^{iAt}
    (or any unitary whose eigenvalues encode A's spectrum via QPE),
    and inv_unitary should implement its adjoint U-dagger.

    This function assumes:

    * ``system`` is an n-qubit register (Hilbert-space dimension ``2**n``);
    * the QPE clock register stores eigenvalue bins decoded via
      ``phase_scale``;
    * the zero-eigenvalue bin is absent from the supported subspace;
    * negative eigenvalues require ``signed=True``.
    * If ``strict=True`` is intended to validate only physically populated
      branches, then ``supported_raw_bins`` should be provided.
    * ``unitary`` and ``inv_unitary`` act on the **full** system register.

    Circuit structure::

        clock:  --[H^m]--[CU^(2^k)]--[IQFT]---------[QFT]--[CU'^(2^k)]--[H^m]--
        system: -------------------***********        ***********-----------------
        ancilla:---------------------------[RY(theta)]----------------------------

    Steps:
        1. Forward QPE: H on clock, CU^(2^k), then IQFT
        2. Reciprocal rotation: controlled RY on ancilla
        3. Inverse QPE: QFT, CU-dagger^(2^k), then H on clock

    Args:
        system: System register, already prepared in |b>.
        clock: Clock register (m qubits), initialized to |0>^m.
        ancilla: Ancilla qubit, initialized to |0>.
        unitary: QKernel implementing U = e^{iAt} on the full system
            register.  Its controlled form is assumed to accept
            ``(control: Qubit, *system: Qubit, **params)`` and return
            ``(control, *system)``.
        inv_unitary: QKernel implementing U-dagger = e^{-iAt} on the full
            system register, with the same convention as ``unitary``.
        scaling: Normalization constant for the reciprocal rotation.
            Controls the tradeoff between success probability and accuracy.
        phase_scale: Positive factor converting clock phases into
            eigenvalue estimates.  For the convention ``U = exp(i A t)``,
            use ``phase_scale = 2*pi / t``.
        signed: If True, interpret the clock register using two's
            complement, representing a signed spectrum in
            ``[-phase_scale/2, phase_scale/2)``.
        little_endian_clock: Whether ``clock[0]`` is the least-significant
            clock bit.  This affects both the controlled-U power schedule
            and the reciprocal-rotation control patterns.
        strict: Passed through to ``reciprocal_rotation()``.
        supported_raw_bins: Optional raw QPE bins that may actually be
            populated for the chosen toy instance.  Use this together with
            ``strict=True`` to avoid over-constraining unpopulated bins.
        iqft_includes_swaps: Whether the underlying ``iqft()`` includes the
            final bit-reversal swaps.  Standard QPE convention is ``False``.
        **params: Parameters passed to unitary and inv_unitary kernels
            (e.g., evolution time, rotation angle).

    Returns:
        Tuple ``(system, clock, ancilla)``.  Measure the ancilla; if the
        result is |1>, the system register holds an HHL solution state
        proportional to A^{-1}|b> within the assumptions above.

    Example::

        @qmc.qkernel
        def u(
            q0: qmc.Qubit,
            q1: qmc.Qubit,
            theta: qmc.Float,
        ) -> tuple[qmc.Qubit, qmc.Qubit]:
            q0 = qmc.p(q0, theta)
            return q0, q1

        @qmc.qkernel
        def u_inv(
            q0: qmc.Qubit,
            q1: qmc.Qubit,
            theta: qmc.Float,
        ) -> tuple[qmc.Qubit, qmc.Qubit]:
            q0 = qmc.p(q0, -1.0 * theta)
            return q0, q1

        @qmc.qkernel
        def solve() -> qmc.Bit:
            sys = qmc.qubit_array(2, name="sys")
            sys[0] = qmc.x(sys[0])
            clock = qmc.qubit_array(2, name="clock")
            anc = qmc.qubit("anc")
            sys, clock, anc = hhl(
                sys,
                clock,
                anc,
                u,
                u_inv,
                scaling=0.25,
                phase_scale=2.0 * math.pi,
                supported_raw_bins=(2,),
                strict=True,
                iqft_includes_swaps=False,
                theta=math.pi,
            )
            return qmc.measure(anc)
    """
    n_system = get_size(system)
    if n_system < 1:
        raise ValueError("system register must contain at least one qubit.")
    n_clock = get_size(clock)
    if n_clock < 1:
        raise ValueError("clock register must contain at least one qubit.")

    controlled_u = qmc.controlled(unitary)
    controlled_u_inv = qmc.controlled(inv_unitary)

    # === Step 1: Forward QPE ===
    # 1a. Hadamard on all clock qubits
    for i in range(n_clock):
        clock[i] = qmc.h(clock[i])

    # 1b. Controlled-U^(2^k) operations
    for i in range(n_clock):
        exponent = _power_exponent_for_qubit(i, n_clock, little_endian_clock)
        clock[i], system = _apply_controlled_register_kernel(
            controlled_u,
            clock[i],
            system,
            power=2**exponent,
            **params,
        )

    # 1c. Inverse QFT on clock register
    clock = iqft(clock)

    # === Step 2: Reciprocal rotation ===
    clock, ancilla = reciprocal_rotation(
        clock=clock,
        ancilla=ancilla,
        scaling=scaling,
        phase_scale=phase_scale,
        signed=signed,
        little_endian_clock=little_endian_clock,
        strict=strict,
        supported_raw_bins=supported_raw_bins,
        iqft_includes_swaps=iqft_includes_swaps,
    )

    # === Step 3: Inverse QPE (uncompute eigenvalue register) ===
    # 3a. QFT on clock (inverse of IQFT)
    clock = qft(clock)

    # 3b. Controlled-U-dagger^(2^k) in reverse order
    for i in range(n_clock - 1, -1, -1):
        exponent = _power_exponent_for_qubit(i, n_clock, little_endian_clock)
        clock[i], system = _apply_controlled_register_kernel(
            controlled_u_inv,
            clock[i],
            system,
            power=2**exponent,
            **params,
        )

    # 3c. Hadamard on all clock qubits
    for i in range(n_clock):
        clock[i] = qmc.h(clock[i])

    return system, clock, ancilla
