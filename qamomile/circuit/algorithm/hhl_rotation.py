"""HHL eigenvalue-inversion rotation block for a 2-qubit clock register.

The Harrow-Hassidim-Lloyd (HHL) algorithm inverts a Hermitian matrix by
extracting its eigenvalues onto a *clock* register via quantum phase
estimation (QPE), rotating an ancilla qubit by an angle proportional to
``1 / lambda``, and uncomputing the clock with inverse QPE.  This module
provides the two building blocks for the eigenvalue-inversion step,
specialised to a **2-qubit clock register**:

* :func:`reciprocal_rotation_2clock_le` -- the bare reciprocal rotation:
  a doubly-controlled ``Ry`` on the ancilla for each non-zero clock
  basis state.
* :func:`hhl_middle_block_2clock_le` -- the full middle block
  ``IQFT -> reciprocal rotation -> QFT`` that consumes a phase-encoded
  clock register (a QPE output) and re-encodes it for the inverse-QPE
  uncompute.

The ``_le`` suffix denotes the **little-endian** clock convention: ``c0``
is the least-significant bit and ``c1`` the most-significant bit, so the
clock basis state ``|raw>`` has ``raw = c0 + 2 * c1``.

Two implementation choices keep these blocks portable across backends:

* The doubly-controlled ``Ry`` is decomposed into single-controlled
  ``Ry`` and ``CX`` gates (see :func:`_ccry`) rather than emitted as a
  ``controlled(..., num_controls=2)`` operation, which not every backend
  emit pass can lower.
* The QFT/IQFT steps use the :class:`~qamomile.circuit.stdlib.qft.QFT` /
  :class:`~qamomile.circuit.stdlib.qft.IQFT` composite-gate classes
  constructed with an explicit qubit count, rather than the
  :func:`~qamomile.circuit.stdlib.qft.qft` /
  :func:`~qamomile.circuit.stdlib.qft.iqft` vector factories.  The
  factories infer the qubit count from the vector's shape, which is
  symbolic for a qkernel parameter; the explicit-count class form keeps
  these blocks usable as sub-kernels.
"""

from __future__ import annotations

import qamomile.circuit as qmc
from qamomile.circuit.stdlib.qft import IQFT, QFT


@qmc.qkernel
def _ry_gate(q: qmc.Qubit, angle: qmc.Float) -> qmc.Qubit:
    """Apply a single ``Ry`` rotation to one qubit.

    Used as the base kernel for the single-controlled ``Ry`` built by
    :data:`_c_ry`.

    Args:
        q (qmc.Qubit): Target qubit.
        angle (qmc.Float): Rotation angle in radians.

    Returns:
        qmc.Qubit: The rotated qubit.
    """
    return qmc.ry(q, angle)


# Single-controlled Ry.  A doubly-controlled Ry is built from this in
# `_ccry` via the standard CCRy = CRy(t/2)*CX*CRy(-t/2)*CX*CRy(t/2)
# identity, rather than `controlled(_ry_gate, num_controls=2)`: the
# multi-control form is not lowerable by every backend emit pass, while
# single-controlled gates and CX are.
_c_ry = qmc.controlled(_ry_gate, num_controls=1)


@qmc.qkernel
def _ccry(
    control_a: qmc.Qubit,
    control_b: qmc.Qubit,
    target: qmc.Qubit,
    theta: qmc.Float,
) -> tuple[qmc.Qubit, qmc.Qubit, qmc.Qubit]:
    """Apply a doubly-controlled ``Ry`` rotation.

    Rotates ``target`` by ``Ry(theta)`` only when both ``control_a`` and
    ``control_b`` are ``|1>``.  Implemented with the standard
    Sleator-Weinfurter decomposition ``CCRy(theta) =
    CRy(theta/2) . CX . CRy(-theta/2) . CX . CRy(theta/2)``, which is
    exact for every ``theta`` because ``Ry`` rotations form a
    one-parameter group (``Ry(theta/2)`` squares to ``Ry(theta)`` and
    ``Ry(-theta/2)`` is its inverse).  This uses only single-controlled
    ``Ry`` and ``CX`` gates so it lowers on every backend.

    Args:
        control_a (qmc.Qubit): First control qubit.
        control_b (qmc.Qubit): Second control qubit.
        target (qmc.Qubit): Target qubit to rotate.
        theta (qmc.Float): Rotation angle in radians applied when both
            controls are ``|1>``.

    Returns:
        tuple[qmc.Qubit, qmc.Qubit, qmc.Qubit]: The two control qubits
        (unchanged) and the conditionally-rotated ``target``.
    """
    control_b, target = _c_ry(control_b, target, angle=0.5 * theta)
    control_a, control_b = qmc.cx(control_a, control_b)
    control_b, target = _c_ry(control_b, target, angle=-0.5 * theta)
    control_a, control_b = qmc.cx(control_a, control_b)
    control_a, target = _c_ry(control_a, target, angle=0.5 * theta)
    return control_a, control_b, target


@qmc.qkernel
def reciprocal_rotation_2clock_le(
    c0: qmc.Qubit,
    c1: qmc.Qubit,
    ancilla: qmc.Qubit,
    theta_raw1: qmc.Float,
    theta_raw2: qmc.Float,
    theta_raw3: qmc.Float,
) -> tuple[qmc.Qubit, qmc.Qubit, qmc.Qubit]:
    """Apply the HHL reciprocal rotation for a 2-qubit little-endian clock.

    For each non-zero clock basis state ``|raw>`` (``raw`` in
    ``{1, 2, 3}``) this applies ``Ry(theta_raw)`` to ``ancilla``,
    doubly-controlled on the clock register.  The controls are flipped
    with ``X`` gates so that the doubly-controlled ``Ry`` fires exactly
    on the intended ``raw``; the flips are undone afterwards so the
    clock register is left unchanged.

    The clock is little-endian -- ``c0`` is the least-significant bit
    and ``c1`` the most-significant bit -- so the targeted basis states
    are ``raw = 1`` (``c0=1, c1=0``), ``raw = 2`` (``c0=0, c1=1``) and
    ``raw = 3`` (``c0=1, c1=1``).  The ``raw = 0`` bin corresponds to a
    zero eigenvalue, which cannot be inverted, so it is intentionally
    left untouched.

    Args:
        c0 (qmc.Qubit): Clock qubit holding the least-significant bit.
        c1 (qmc.Qubit): Clock qubit holding the most-significant bit.
        ancilla (qmc.Qubit): Ancilla qubit rotated by the reciprocal
            angle; HHL post-selects on this qubit measuring ``|1>``.
        theta_raw1 (qmc.Float): ``Ry`` angle, in radians, applied to the
            ancilla when the clock is in ``|raw=1>``.
        theta_raw2 (qmc.Float): ``Ry`` angle, in radians, for ``|raw=2>``.
        theta_raw3 (qmc.Float): ``Ry`` angle, in radians, for ``|raw=3>``.

    Returns:
        tuple[qmc.Qubit, qmc.Qubit, qmc.Qubit]: The clock qubits ``c0``
        and ``c1`` (unchanged) and the rotated ``ancilla``.

    Example:
        >>> import math
        >>> import qamomile.circuit as qmc
        >>> from qamomile.circuit.algorithm import reciprocal_rotation_2clock_le
        >>>
        >>> @qmc.qkernel
        ... def rotate_for_bin1() -> qmc.Bit:
        ...     c0 = qmc.x(qmc.qubit("c0"))  # clock = |raw=1>
        ...     c1 = qmc.qubit("c1")
        ...     anc = qmc.qubit("anc")
        ...     c0, c1, anc = reciprocal_rotation_2clock_le(
        ...         c0, c1, anc, math.pi, 0.0, 0.0,
        ...     )
        ...     return qmc.measure(anc)  # always |1>: Ry(pi) on |raw=1>
    """
    # raw = 1  ->  c0 = 1, c1 = 0
    c1 = qmc.x(c1)
    c0, c1, ancilla = _ccry(c0, c1, ancilla, theta_raw1)
    c1 = qmc.x(c1)

    # raw = 2  ->  c0 = 0, c1 = 1
    c0 = qmc.x(c0)
    c0, c1, ancilla = _ccry(c0, c1, ancilla, theta_raw2)
    c0 = qmc.x(c0)

    # raw = 3  ->  c0 = 1, c1 = 1
    c0, c1, ancilla = _ccry(c0, c1, ancilla, theta_raw3)

    return c0, c1, ancilla


@qmc.qkernel
def hhl_middle_block_2clock_le(
    c0: qmc.Qubit,
    c1: qmc.Qubit,
    ancilla: qmc.Qubit,
    theta_raw1: qmc.Float,
    theta_raw2: qmc.Float,
    theta_raw3: qmc.Float,
) -> tuple[qmc.Qubit, qmc.Qubit, qmc.Qubit]:
    """Apply the HHL middle block (IQFT -> reciprocal rotation -> QFT).

    This is the eigenvalue-inversion stage of HHL for a 2-qubit clock
    register.  The clock register is expected to hold a *phase-encoded*
    eigenvalue, i.e. the output of quantum phase estimation.  The block:

    1. applies the inverse QFT, mapping the phase encoding to the
       computational-basis eigenvalue bin ``|raw>``;
    2. applies :func:`reciprocal_rotation_2clock_le`, embedding the
       reciprocal ``1 / lambda`` into the ancilla amplitude;
    3. applies the QFT, re-encoding the clock so the caller can run the
       inverse-QPE uncompute.

    The clock is little-endian (``c0`` is the least-significant bit).
    The IQFT/QFT use the :class:`~qamomile.circuit.stdlib.qft.IQFT` /
    :class:`~qamomile.circuit.stdlib.qft.QFT` composite-gate classes with
    an explicit two-qubit count so the block works correctly when called
    as a sub-kernel.

    Args:
        c0 (qmc.Qubit): Clock qubit holding the least-significant bit.
        c1 (qmc.Qubit): Clock qubit holding the most-significant bit.
        ancilla (qmc.Qubit): Ancilla qubit rotated by the reciprocal
            angle; HHL post-selects on this qubit measuring ``|1>``.
        theta_raw1 (qmc.Float): ``Ry`` angle, in radians, applied to the
            ancilla for clock eigenvalue bin ``raw = 1``.
        theta_raw2 (qmc.Float): ``Ry`` angle, in radians, for bin
            ``raw = 2``.
        theta_raw3 (qmc.Float): ``Ry`` angle, in radians, for bin
            ``raw = 3``.

    Returns:
        tuple[qmc.Qubit, qmc.Qubit, qmc.Qubit]: The phase-re-encoded
        clock qubits ``c0`` and ``c1`` and the rotated ``ancilla``.

    Example:
        >>> import math
        >>> import qamomile.circuit as qmc
        >>> from qamomile.circuit.algorithm import hhl_middle_block_2clock_le
        >>> from qamomile.circuit.stdlib.qft import qft
        >>>
        >>> @qmc.qkernel
        ... def invert_bin1() -> qmc.Bit:
        ...     clock = qmc.qubit_array(2, name="clock")
        ...     clock[0] = qmc.x(clock[0])       # computational |raw=1>
        ...     clock = qft(clock)               # phase-encode it
        ...     anc = qmc.qubit("anc")
        ...     c0, c1, anc = hhl_middle_block_2clock_le(
        ...         clock[0], clock[1], anc, math.pi, 0.0, 0.0,
        ...     )
        ...     return qmc.measure(anc)
    """
    # IQFT(2) / QFT(2): explicit-count composite-gate classes applied to
    # loose qubits.  The `iqft`/`qft` vector factories infer the count
    # from a Vector shape, which is symbolic for a qkernel parameter.
    c0, c1 = IQFT(2)(c0, c1)

    c0, c1, ancilla = reciprocal_rotation_2clock_le(
        c0, c1, ancilla, theta_raw1, theta_raw2, theta_raw3
    )

    c0, c1 = QFT(2)(c0, c1)

    return c0, c1, ancilla
