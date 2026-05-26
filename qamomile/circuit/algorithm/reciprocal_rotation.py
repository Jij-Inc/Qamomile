"""HHL eigenvalue-inversion building blocks for an n-qubit clock register.

The Harrow-Hassidim-Lloyd (HHL) algorithm inverts a Hermitian matrix by
extracting its eigenvalues onto a *clock* register via quantum phase
estimation (QPE), rotating an ancilla qubit by an angle proportional to
``1 / lambda``, and uncomputing the clock with inverse QPE.  This module
provides the building blocks for the eigenvalue-inversion step, for an
arbitrary-size clock:

* :func:`reciprocal_rotation` -- the bare reciprocal rotation.  For each
  non-zero clock basis state ``|raw>`` it applies ``Ry(2 * arcsin(c /
  raw))`` to the ancilla, controlled on the clock being in ``|raw>``.
  This implements the canonical HHL transform

      |raw>_C |0>_S
          -> |raw>_C (sqrt(1 - (c/raw)^2) |0>_S + (c/raw) |1>_S).

* :func:`hhl_middle_block` -- the full middle stage of HHL,
  ``IQFT -> reciprocal_rotation -> QFT``, consuming a phase-encoded
  clock register (a QPE output) and re-encoding it for the inverse-QPE
  uncompute.

Both are **plain Python functions** that compose Qamomile operations
when called from inside a ``@qmc.qkernel`` body.  They are *not*
``@qmc.qkernel``-decorated themselves: the per-``raw`` enumeration with
classically-precomputed ``2 * arcsin(c / raw)`` angles relies on
trace-time Python iteration that a ``@qmc.qkernel`` body cannot
currently express (Qamomile's symbolic ``UInt`` lacks the bitwise
operators and ``arcsin`` math that the alternative qkernel form would
require).  The same plain-function pattern is used by
``qamomile.circuit.algorithm.hhl.reciprocal_rotation``.

The clock is **little-endian**: ``qubits[0]`` is the least-significant
bit, so the clock basis state ``|raw>`` has
``raw = sum(qubits[i] * 2**i for i in range(n))``.
"""

from __future__ import annotations

import math

import qamomile.circuit as qmc
from qamomile.circuit.frontend.handle.utils import get_size
from qamomile.circuit.stdlib.qft import iqft, qft


def reciprocal_rotation(
    qubits: qmc.Vector[qmc.Qubit],
    ancilla: qmc.Qubit,
    c: float,
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Qubit]:
    """Apply the HHL reciprocal rotation on an n-qubit clock register.

    For each non-zero clock basis state ``|raw>``
    (``raw in {1, ..., 2**n - 1}``), this applies
    ``Ry(2 * arcsin(c / raw))`` to ``ancilla``, controlled on the clock
    being in ``|raw>``.  The controls are flipped with ``X`` gates so
    that the n-controlled ``Ry`` fires exactly on the intended ``raw``;
    the flips are undone afterwards so the clock register is left
    unchanged.

    The ``raw = 0`` bin corresponds to a zero eigenvalue, which cannot
    be inverted, so it is intentionally left untouched.

    Eigenvalue convention: ``raw`` is the *integer clock value*, not
    the physical eigenvalue ``lambda``.  In a full HHL pipeline
    ``lambda`` is typically proportional to ``raw`` -- e.g. with QPE
    over ``U = exp(i * A * t)``,
    ``lambda = (2 * pi / t) * raw / 2**n``.  This function uses
    ``raw`` as-is and does *not* perform that mapping; the caller is
    responsible for absorbing the proportionality constant into ``c``
    so that the ancilla ``|1>`` amplitude is the physically intended
    ``c_phys / lambda`` rather than ``c / raw``.

    Implementation note: the n-controlled ``Ry`` is emitted with
    ``qmc.controlled(qmc.ry, num_controls=n)`` and applied directly,
    without any manual Sleator-Weinfurter-style decomposition into
    single-controlled rotations.  Backends whose emit pass cannot
    lower a multi-controlled custom-block gate (e.g. QURI Parts) will
    raise ``EmitError`` at transpile time; backends with native
    multi-control support (Qiskit, CUDA-Q) emit it directly.

    Scaling: enumerating the ``2**n - 1`` non-zero clock basis states
    makes the gate count grow as ``O(n * 2**n)``.  This is appropriate
    for small clock registers used in demonstration or small-system
    HHL; truly large ``n`` calls for a different implementation
    strategy (quantum arithmetic to compute ``c / raw`` into an
    arithmetic register, or block-encoding of the reciprocal map).

    Args:
        qubits (qmc.Vector[qmc.Qubit]): Clock register of size n.  The
            size must be concrete (non-symbolic) at trace time, so the
            caller's qkernel must allocate ``qubits`` with a fixed
            shape (e.g. ``qmc.qubit_array(n, name=...)``).
        ancilla (qmc.Qubit): Ancilla qubit, expected to start in
            ``|0>``, that receives the reciprocal rotation.  HHL
            post-selects on this qubit measuring ``|1>``.
        c (float): Scaling constant.  Must be positive and at most
            ``1`` so that ``|c / raw| <= 1`` for every non-zero
            ``raw``; otherwise the rotation angle is not real-valued.

    Returns:
        tuple[qmc.Vector[qmc.Qubit], qmc.Qubit]: The clock register
            (unchanged) and the rotated ancilla.

    Raises:
        ValueError: If the clock register has size zero, or if ``c`` is
            non-finite, non-positive, or greater than ``1`` (which
            would make ``|c / raw| > 1`` at ``raw = 1``).

    Example:
        >>> import qamomile.circuit as qmc
        >>> from qamomile.circuit.algorithm import reciprocal_rotation
        >>>
        >>> @qmc.qkernel
        ... def invert_basis_1() -> qmc.Bit:
        ...     clock = qmc.qubit_array(2, name="clock")
        ...     clock[0] = qmc.x(clock[0])    # clock = |raw=1>
        ...     anc = qmc.qubit("anc")
        ...     clock, anc = reciprocal_rotation(clock, anc, c=0.5)
        ...     return qmc.measure(anc)   # P(anc=1) = (0.5/1)^2 = 0.25
    """
    n = get_size(qubits)

    if n <= 0:
        raise ValueError(f"Clock register must contain at least one qubit, got n={n}.")
    if not math.isfinite(c):
        raise ValueError(f"scaling c must be finite, got c={c}")
    if c <= 0.0:
        raise ValueError(f"scaling c must be positive, got c={c}")
    if c > 1.0:
        raise ValueError(
            f"Reciprocal rotation undefined: c > 1 (got c={c}) makes "
            "|c/raw| > 1 at raw=1.  Choose c <= 1."
        )

    mc_ry = qmc.controlled(qmc.ry, num_controls=n)

    for raw in range(1, 2**n):
        theta = 2.0 * math.asin(c / raw)

        # X-flip the 0-bits of `raw` so the multi-controlled Ry fires
        # exactly on the |raw> basis state.  Little-endian: bit i of
        # `raw` corresponds to qubits[i].
        for i in range(n):
            if not ((raw >> i) & 1):
                qubits[i] = qmc.x(qubits[i])

        # Apply the n-controlled Ry.  With concrete int `num_controls`,
        # ControlledGate uses the positional-args calling convention:
        # the first n args are controls, the remaining args are targets.
        qubit_list = [qubits[i] for i in range(n)]
        result = mc_ry(*qubit_list, ancilla, angle=theta)
        for i in range(n):
            qubits[i] = result[i]
        ancilla = result[n]

        # Undo the X-flips so the clock register returns to |raw>.
        for i in range(n):
            if not ((raw >> i) & 1):
                qubits[i] = qmc.x(qubits[i])

    return qubits, ancilla


def hhl_middle_block(
    qubits: qmc.Vector[qmc.Qubit],
    ancilla: qmc.Qubit,
    c: float,
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Qubit]:
    """Apply the HHL middle block (IQFT -> reciprocal rotation -> QFT).

    Composes the eigenvalue-inversion stage of HHL on an n-qubit clock
    register.  The block:

    1. applies the inverse QFT, mapping a phase-encoded clock state to
       the computational-basis eigenvalue bin ``|raw>``;
    2. applies :func:`reciprocal_rotation`, embedding the reciprocal
       ``1 / raw`` (scaled by ``c``) into the ancilla amplitude;
    3. applies the QFT, re-encoding the clock so the caller can run
       inverse-QPE-style uncompute.

    Input state contract: the clock is expected to be in a
    *phase-encoded* state -- equivalently, ``qft(|raw>)`` for some
    integer ``raw``, or the QPE intermediate before its final inverse-QFT.
    **Do not** call this on a clock that is already in the computational
    basis ``|raw>`` (e.g. the output of a standard QPE that ends with
    its own IQFT): ``hhl_middle_block`` would then add an unintended
    IQFT (its step 1) that no later step cancels.  Use the table below.

    +----------------------------------------+--------------------------+
    | Clock state at the call site           | Function to call         |
    +========================================+==========================+
    | Computational ``|raw>``                | ``reciprocal_rotation``  |
    | (standard QPE, IQFT already applied)   |                          |
    +----------------------------------------+--------------------------+
    | Phase-encoded ``qft(|raw>)``           | ``hhl_middle_block``     |
    | (QPE intermediate, before final IQFT)  |                          |
    +----------------------------------------+--------------------------+

    The clock is little-endian (``qubits[0]`` is the least-significant
    bit).  Like :func:`reciprocal_rotation`, this is a plain Python
    function; call it from inside a ``@qmc.qkernel`` body with a
    concrete-shape ``qubits`` register.  The eigenvalue convention
    (``raw`` vs. physical ``lambda``) is identical to
    :func:`reciprocal_rotation` -- see its docstring.

    Args:
        qubits (qmc.Vector[qmc.Qubit]): Clock register of size n,
            expected to hold a phase-encoded eigenvalue.  The size must
            be concrete (non-symbolic) at trace time.
        ancilla (qmc.Qubit): Ancilla qubit, expected to start in
            ``|0>``.
        c (float): Scaling constant; same constraint as
            :func:`reciprocal_rotation`.

    Returns:
        tuple[qmc.Vector[qmc.Qubit], qmc.Qubit]: The phase-re-encoded
            clock register and the rotated ancilla.

    Raises:
        ValueError: Propagated from :func:`reciprocal_rotation` when
            ``c`` is non-finite, non-positive, or ``c > 1``, or when
            the clock register has size zero.

    Note:
        The ``iqft`` / ``qft`` round trip uses
        :mod:`qamomile.circuit.stdlib.qft`, whose stdlib implementation
        preserves bit order (its internal swaps undo QFT's inherent
        reversal).  This matches the little-endian convention assumed
        throughout this module and is verified by the round-trip tests
        in ``tests/circuit/algorithm/test_reciprocal_rotation.py``.

    Example:
        >>> import qamomile.circuit as qmc
        >>> from qamomile.circuit.algorithm import hhl_middle_block
        >>> from qamomile.circuit.stdlib.qft import qft
        >>>
        >>> @qmc.qkernel
        ... def invert_phase_basis_1() -> qmc.Bit:
        ...     clock = qmc.qubit_array(2, name="clock")
        ...     clock[0] = qmc.x(clock[0])     # computational |raw=1>
        ...     clock = qft(clock)             # phase-encode it
        ...     anc = qmc.qubit("anc")
        ...     clock, anc = hhl_middle_block(clock, anc, c=0.5)
        ...     return qmc.measure(anc)        # P(anc=1) = 0.25
    """
    qubits = iqft(qubits)
    qubits, ancilla = reciprocal_rotation(qubits, ancilla, c)
    qubits = qft(qubits)
    return qubits, ancilla
