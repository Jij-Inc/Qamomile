"""State preparation via Möttönen's uniformly-controlled-rotation construction.

Prepares the n-qubit state

.. math::

    |\\psi\\rangle = \\sum_{i=0}^{2^n - 1} a_i \\,|i\\rangle

from :math:`|0\\rangle^{\\otimes n}` for a normalised amplitude vector
``a``.  The construction follows Möttönen et al., "Transformation of
quantum states using uniformly controlled rotations"
(arXiv:quant-ph/0407010).  Both the RY-only "amplitude distribution"
stage and the RZ "phase restoration" stage are supported, so general
complex amplitudes work end-to-end.

.. important::

    **Pre-condition: the input register must be in the all-zero state**
    :math:`|0\\rangle^{\\otimes n}`.  The Möttönen construction
    decomposes the unitary that takes :math:`|0\\rangle^{\\otimes n}`
    to the target :math:`|\\psi\\rangle`; applying it to any other
    initial state yields a *different* output (the same unitary
    applied to a different input), not the target amplitude vector.
    There is no runtime guard for this — Qamomile does not track
    qubit states — so it is the caller's responsibility to ensure the
    register has not yet been mutated when ``amplitude_encoding`` /
    ``amplitude_encoding_from_angles`` is invoked.  In practice, call
    these helpers immediately after ``qmc.qubit_array(n, ...)``
    inside a kernel.

This module hosts only the **gate-emission side** of the algorithm:
the ``MottonenAmplitudeEncoding`` ``CompositeGate``, the function
wrappers ``amplitude_encoding`` / ``amplitude_encoding_from_angles``,
and the Gray-walk emitter ``_emit_mottonen_gates``.  The classical
angle precomputation lives in :mod:`qamomile.linalg.mottonen` so that
hybrid loops can call ``compute_mottonen_amplitude_encoding_*_angles``
outside any kernel and feed the result back through
``amplitude_encoding_from_angles`` with ``parameters=[...]``.

Pipeline
--------

1. Validate and normalise the input — see
   :func:`qamomile.linalg.mottonen.validate_and_normalize_amplitudes`
   (length must be a power of two, all-zero rejected).
2. Determine whether the input is genuinely complex (has a non-zero
   imaginary part).  Real inputs (including complex with zero imag)
   keep the original signed-RY path — negative real amplitudes flow
   through the sign of ``arctan2(a_1, a_0)`` naturally, with no RZ
   overhead.  Complex inputs use the iterative disentangling
   construction.
3. For real inputs, compute the per-level RY rotation angles by
   splitting each chunk into upper / lower halves and using
   ``arctan2`` of the two sub-block norms (or signed ``arctan2`` at
   the leaf).  See
   :func:`qamomile.linalg.mottonen.compute_all_ry_angles_per_level`.
4. For complex inputs, iteratively disentangle the target amplitude
   vector qubit-by-qubit from LSB to MSB.  See
   :func:`qamomile.linalg.mottonen.compute_disentangling_angles_per_level`.
5. At each level ``k >= 1`` apply a uniformly controlled rotation over
   the previously prepared ``k`` qubits.  We use the standard Gray-code
   RY / CNOT decomposition for the magnitude stage and the same
   structure with RZ for the phase stage.  The emitted gate order is
   "all RY layers, then all RZ layers".  Pairwise
   ``[U_y^(k), U_z^(k')]`` does NOT commute in general (including
   ``k != k'`` cases, because earlier RY targets can be controls of
   later RZ multiplexers), but the FULL sweep product equals the
   per-level interleaved product
   ``(U_z^(0) U_y^(0)) ... (U_z^(n-1) U_y^(n-1))`` as unitaries — this
   is a structural identity verified by
   :class:`tests.circuit.algorithm.state_preparation.test_mottonen_amplitude_encoding.TestRyRzOrdering`
   with arbitrary (non-disentangling) per-level angles.  Within each
   level the order RY-before-RZ is preserved in both schemes.

Lazy angle precomputation
-------------------------

``MottonenAmplitudeEncoding.__init__`` runs the cheap normalisation
pass eagerly (``O(2^n)``) so input errors — wrong shape, length not a
power of two, all-zero amplitudes — surface at construction time.  The
expensive angle precomputation (``O(n * 2^n)``) is deferred: it runs
lazily on the first ``_decompose()`` call and is cached afterwards.

In practice the surrounding ``CompositeGate.__call__`` framework
eagerly invokes ``_decompose()`` when the gate is invoked inside a
``@qkernel`` (to populate the invocation's ``CallableDef.body``), so
kernel-side ``estimate_resources()`` does still pay the angle cost
today. The lazy aspect is the right shape for a future framework
refactor that defers decomposition-body construction until emit time,
and is verified standalone by
``tests.circuit.algorithm.state_preparation.test_mottonen_amplitude_encoding.TestLazyConstruction``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from qamomile.circuit.frontend.composite_gate import CompositeGate
from qamomile.circuit.frontend.handle import Float, Qubit, Vector
from qamomile.circuit.frontend.handle.utils import get_size
from qamomile.circuit.frontend.operation.qubit_gates import cx, ry, rz
from qamomile.circuit.ir.operation.callable import CompositeGateType
from qamomile.linalg.mottonen import (
    compute_all_ry_angles_per_level,
    compute_disentangling_angles_per_level,
    validate_and_normalize_amplitudes,
)

__all__ = [
    "MottonenAmplitudeEncoding",
    "amplitude_encoding",
    "amplitude_encoding_from_angles",
]


# ---------------------------------------------------------------------------
# Gate-emission helpers
# ---------------------------------------------------------------------------


def _get_cnot_controls(k: int) -> list[int]:
    """Compute the CNOT control positions of the Gray-code Ry walk.

    The Gray-code decomposition of a ``k``-control uniformly controlled
    :math:`R_y` interleaves ``2^k`` rotations with ``2^k`` CNOTs.  The
    ``i``-th CNOT toggles the control bit that differs between
    ``Gray(i-1)`` and ``Gray(i)``, which is the trailing-zero position of
    ``i`` for ``i < 2^k`` and ``k-1`` for the closing CNOT that returns
    to the starting Gray word.

    Args:
        k (int): Number of control qubits driving the uniformly
            controlled gate.

    Returns:
        list[int]: A list of length ``2**k`` whose ``i``-th entry is the
            index of the control qubit (in ``0..k-1``) that the ``i``-th
            CNOT acts upon.
    """
    controls = []
    for i in range(1, 2**k + 1):
        if i == 2**k:
            controls.append(k - 1)
        else:
            lowest_set_bit = i & -i
            controls.append(lowest_set_bit.bit_length() - 1)
    return controls


def _emit_mottonen_gates(
    qubits: list[Qubit],
    num_qubits: int,
    angles: Sequence[float] | np.ndarray | Vector[Float],
    gate: str = "ry",
) -> None:
    """Emit a Gray-walk Ry / CNOT or Rz / CNOT sequence in place on *qubits*.

    Level ``k`` targets qubit ``num_qubits - 1 - k`` so that the first
    rotation lands on the most-significant qubit (consistent with the
    chunk-splitting convention used by the
    :mod:`qamomile.linalg.mottonen` per-level builders).

    *angles* may be either a concrete numerical sequence (Python list,
    NumPy array of dtype float) or a ``Vector[Float]`` handle — the
    latter is what makes the runtime-parametric path
    (:func:`amplitude_encoding_from_angles`) work: indexing a
    ``Vector[Float]`` with a Python ``int`` produces a ``Float`` handle
    that ``qmc.ry`` / ``qmc.rz`` accept directly.

    For the concrete-sequence and ``np.ndarray`` cases, each element is
    coerced to a builtin ``float`` before being handed to ``rotation``
    so that the angle stored in IR metadata stays a plain Python
    ``float`` — not a NumPy scalar (``np.float64``), which would slip
    past ``Value.get_const()``-based ``isinstance(c, float)`` checks
    downstream.  ``Vector[Float]`` indexing returns a ``Float`` handle
    and is forwarded as-is.

    Args:
        qubits (list[Qubit]): Mutable list of length ``num_qubits`` of
            qubit handles.  Entries are overwritten as gates are applied.
        num_qubits (int): Number of qubits in the register.
        angles (Sequence[float] | np.ndarray | Vector[Float]): Flat
            sequence of rotation angles of length ``2**num_qubits - 1``,
            laid out level by level.
        gate (str): ``"ry"`` for the magnitude stage or ``"rz"`` for the
            phase stage.  Defaults to ``"ry"``.

    Returns:
        None: The *qubits* list is mutated in place.

    Raises:
        ValueError: If *gate* is not ``"ry"`` or ``"rz"``.
    """
    match gate:
        case "ry":
            rotation = ry
        case "rz":
            rotation = rz
        case _:
            raise ValueError(f"gate must be 'ry' or 'rz', got {gate!r}")

    def _angle(at: int) -> Float | float:
        """Return ``angles[at]`` as the right type for ``rotation``."""
        if isinstance(angles, Vector):
            return angles[at]
        return float(angles[at])

    idx = 0
    for k in range(num_qubits):
        tgt = num_qubits - 1 - k
        if k == 0:
            qubits[tgt] = rotation(qubits[tgt], _angle(idx))
            idx += 1
            continue

        cnot_seq = _get_cnot_controls(k)
        for step in range(2**k):
            qubits[tgt] = rotation(qubits[tgt], _angle(idx))
            idx += 1
            ctrl = num_qubits - 1 - cnot_seq[step]
            qubits[ctrl], qubits[tgt] = cx(qubits[ctrl], qubits[tgt])


# ---------------------------------------------------------------------------
# Composite gate
# ---------------------------------------------------------------------------


class MottonenAmplitudeEncoding(CompositeGate):
    """Möttönen amplitude encoding for normalised real or complex vectors.

    Prepares the state :math:`\\sum_i a_i |i\\rangle` from
    :math:`|0\\rangle^{\\otimes n}` using uniformly controlled Y (and,
    for genuinely complex inputs, Z) rotations decomposed into ``RY`` /
    ``RZ`` and ``CNOT`` gates with Gray-code ordering.

    .. important::

        **Pre-condition: the input qubits must be in the all-zero state**
        :math:`|0\\rangle^{\\otimes n}`.  The gate emits the unitary
        that maps :math:`|0\\rangle^{\\otimes n}` to the target
        :math:`|\\psi\\rangle`.  Applied to any other input it
        produces ``U |\\phi\\rangle`` for that ``|\\phi\\rangle`` —
        which is *not* the target amplitude vector.  Qamomile does
        not track qubit states at runtime, so this is the caller's
        responsibility.

    Notes:
        * Input amplitudes are normalised automatically.
        * Real inputs (negative entries allowed) use a single RY stage:
          the sign of ``arctan2`` at the leaf preserves negative signs
          natively, so no RZ overhead is incurred.
        * Complex inputs with non-zero imaginary part use the full
          two-stage construction (RY for magnitudes, RZ for phases).
        * Complex inputs whose imaginary part is identically zero (e.g.
          ``[1+0j, -1+0j]``) are silently coerced to real and follow the
          single-stage path.
        * ``__init__`` runs the cheap pass eagerly (``O(2^n)``):
          shape / length validation, normalisation, and complex
          detection, so input errors (wrong shape, length not a
          power of two, all-zero amplitudes) surface at construction
          time.  The expensive pass — full per-level angle
          precomputation, ``O(n * 2^n)`` — is deferred until the
          first :meth:`_decompose` call and cached afterwards.
          :meth:`_resources` reads only the cached complex flag and
          never triggers the angle pass.

    Example::

        # Real amplitudes (signed allowed)
        gate = MottonenAmplitudeEncoding([1.0, 0.0, 0.0, 1.0])
        q0, q1 = gate(q0, q1)

        # Complex amplitudes
        gate = MottonenAmplitudeEncoding([1+0j, 1j, -1+0j, -1j])
        q0, q1 = gate(q0, q1)
    """

    gate_type = CompositeGateType.CUSTOM
    custom_name = "mottonen_amplitude_encoding"

    def __init__(self, amplitudes: Sequence[float] | Sequence[complex] | np.ndarray):
        """Initialise the gate with a concrete amplitude vector.

        Runs the full normalisation + complex-detection pass eagerly
        (``O(2^n)``) so that input errors — wrong shape, length not a
        power of two, all-zero amplitudes — surface at construction
        time rather than at the first ``_resources()`` /
        ``_decompose()`` call.  Angle precomputation
        (``O(n * 2^n)``) stays deferred: it runs lazily on the first
        ``_decompose()`` call and is cached afterwards.

        This keeps the dominant cost of Möttönen out of
        kernel-build time when the gate is later only used for
        resource estimation.  In practice the
        ``CompositeGate.__call__`` framework eagerly invokes
        ``_decompose()`` to build the implementation block when the
        gate is invoked inside a ``@qkernel``, so the lazy aspect
        currently bites only for code that constructs
        ``MottonenAmplitudeEncoding`` standalone — but the laziness
        still matters there, and is the right shape for a future
        framework refactor that defers ``_decompose()`` until emit
        time.

        Args:
            amplitudes (Sequence[float] | Sequence[complex] | np.ndarray):
                Amplitude vector of length ``2**n``.  Real or complex;
                it is automatically normalised.  Complex inputs with
                zero imaginary part are coerced to real (single-stage
                RY path).

        Raises:
            ValueError: If the input is not a 1-D vector, the length
                is not a power of two (or is less than 2, i.e., would
                map to a zero-qubit register), or all amplitudes are
                zero.
        """
        normalized, num_qubits, is_complex = validate_and_normalize_amplitudes(
            amplitudes
        )
        self._normalized: np.ndarray = normalized
        self._num_qubits: int = num_qubits
        self._is_complex: bool = is_complex

        # Angle precomputation is deferred — populated lazily on the
        # first call to :meth:`_ensure_angles` (driven from
        # :meth:`_decompose`).  Resource estimation uses ``self._is_complex``
        # alone, so it never triggers the angle pass.
        self._ry_angles_per_level_cache: list[np.ndarray] | None = None
        self._rz_angles_per_level_cache: list[np.ndarray] | None = None

    def _ensure_angles(self) -> None:
        """Run the angle-precomputation pass if not yet cached.

        ``O(n * 2^n)`` — populates the per-level Ry (and, for complex
        inputs, Rz) caches.  Idempotent.
        """
        if self._ry_angles_per_level_cache is not None:
            return
        if self._is_complex:
            ry, rz_levels = compute_disentangling_angles_per_level(
                self._normalized, self._num_qubits
            )
            self._ry_angles_per_level_cache = ry
            self._rz_angles_per_level_cache = rz_levels
        else:
            self._ry_angles_per_level_cache = compute_all_ry_angles_per_level(
                self._normalized, self._num_qubits
            )
            self._rz_angles_per_level_cache = None

    @property
    def num_target_qubits(self) -> int:
        """Number of qubits the gate acts on.

        Returns:
            int: The number of qubits (``log2`` of the amplitude vector
                length).  Cheap — known from ``__init__``.
        """
        return self._num_qubits

    def _decompose(
        self,
        qubits: Vector[Qubit] | tuple[Qubit, ...],
    ) -> tuple[Qubit, ...]:
        """Decompose into RY/CNOT (and RZ/CNOT for complex inputs) gates.

        Triggers full angle precomputation on first call.  Emits the
        magnitude stage (RY layers, Gray-walk order) and then, if the
        input is genuinely complex, the phase stage (RZ layers, same
        structure).

        Args:
            qubits (Vector[Qubit] | tuple[Qubit, ...]):
                ``num_target_qubits`` input qubits as a tuple or
                ``Vector`` handle, expected to start in
                :math:`|0\\rangle^{\\otimes n}`.

        Returns:
            tuple[Qubit, ...]: Output qubits in the encoded state.
        """
        self._ensure_angles()
        assert self._ry_angles_per_level_cache is not None
        qubit_list = [qubits[i] for i in range(self._num_qubits)]
        ry_angles = [float(a) for a in np.concatenate(self._ry_angles_per_level_cache)]
        _emit_mottonen_gates(qubit_list, self._num_qubits, ry_angles, gate="ry")
        if self._rz_angles_per_level_cache is not None:
            rz_angles = [
                float(a) for a in np.concatenate(self._rz_angles_per_level_cache)
            ]
            _emit_mottonen_gates(qubit_list, self._num_qubits, rz_angles, gate="rz")
        return tuple(qubit_list)

# ---------------------------------------------------------------------------
# Convenience function wrappers
# ---------------------------------------------------------------------------


def amplitude_encoding(
    qubits: Vector[Qubit],
    amplitudes: Sequence[float] | Sequence[complex] | np.ndarray | Vector[Float],
) -> Vector[Qubit]:
    """Apply Möttönen amplitude encoding to *qubits* in place.

    Convenience wrapper around :class:`MottonenAmplitudeEncoding` that
    accepts a ``Vector`` handle and writes the gated qubits back into
    the same vector.  Real and complex amplitudes are both supported;
    see the class docstring for the gate-count tradeoff between the two
    paths.

    .. important::

        **Pre-condition: ``qubits`` must currently be in the all-zero
        state** :math:`|0\\rangle^{\\otimes n}`.  Möttönen encodes the
        unitary that takes :math:`|0\\rangle^{\\otimes n}` to the
        normalised target; applied to any other state it produces a
        different (in general meaningless) output.  In practice call
        this immediately after ``qmc.qubit_array(n, ...)`` inside a
        kernel.

    *amplitudes* may be supplied as one of three forms:

    * A concrete Python ``Sequence[float]`` / ``Sequence[complex]`` /
      ``np.ndarray``.  Use this when the amplitudes are known where you
      build the kernel (closed over from the surrounding Python scope).
      This is the only form that supports **complex** amplitude vectors.
    * A ``Vector[Float]`` handle obtained from a kernel parameter that
      is **bound at compile time** via
      ``transpiler.transpile(kernel, bindings={"amps": [...]})``.  The
      handle's bound concrete values are extracted at trace time and
      flow through the same angle-computation path.  This makes
      ``bindings={"amps": ...}`` ergonomic without forcing the user to
      pre-compute Möttönen angles.  ``Vector[Float]`` carries real
      numbers only; for complex amplitudes via a kernel parameter,
      either pass a concrete ``np.ndarray`` directly (closure form), or
      use :func:`amplitude_encoding_from_angles` with separate
      ``ry_angles`` and ``rz_angles`` parameters.
    * **Not** a ``Vector[Float]`` left symbolic by
      ``parameters=["amps"]`` — the angle computation requires concrete
      values at trace time.  Use :func:`amplitude_encoding_from_angles`
      with ``parameters=["ry_angles", "rz_angles"]`` for the
      runtime-parametric case.

    Args:
        qubits (Vector[Qubit]): Vector of ``n`` qubit handles, expected
            to start in :math:`|0\\rangle^{\\otimes n}`.
        amplitudes (Sequence[float] | Sequence[complex] | np.ndarray | Vector[Float]):
            Amplitude vector of length ``2**n``.  Concrete sequences and
            ``np.ndarray`` accept both real and complex inputs and are
            normalised automatically.  ``Vector[Float]`` is accepted
            only when (a) its concrete values are available at trace
            time (i.e., it came from a ``bindings={...}`` entry, not
            from ``parameters=[...]``) and (b) the values are real;
            complex amplitudes via a kernel parameter must instead go
            through :func:`amplitude_encoding_from_angles`.

    Returns:
        Vector[Qubit]: The same *qubits* vector, with each element
            updated to the post-encoding qubit handle.

    Raises:
        ValueError: If *qubits* has a symbolic shape (no compile-time
            known size — e.g., it is a sub-kernel parameter traced
            standalone before its shape is resolved), the amplitude
            length is not a power of two (or is less than 2, i.e.,
            would map to a zero-qubit register), all amplitudes are
            zero, the qubit count does not match
            ``log2(len(amplitudes))``, or *amplitudes* is a
            ``Vector[Float]`` handle whose concrete values are not
            available at trace time (use
            :func:`amplitude_encoding_from_angles` with
            ``parameters=[...]`` for runtime-parametric angles).

    Example::

        # Concrete Python amplitudes
        @qmc.qkernel
        def prepare() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, name="q")
            q = amplitude_encoding(q, [1.0, 0.0, 0.0, 1.0])
            return qmc.measure(q)

        # Bound Vector[Float] kernel parameter
        @qmc.qkernel
        def prepare(amps: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, name="q")
            q = amplitude_encoding(q, amps)
            return qmc.measure(q)

        transpiler.transpile(prepare, bindings={"amps": [1.0, 0.0, 0.0, 1.0]})
    """
    try:
        n = get_size(qubits)
    except ValueError as e:
        raise ValueError(
            "amplitude_encoding requires a Vector[Qubit] with a compile-time "
            "known size; received a symbolic-shape vector. Bind the qubit "
            "count via transpiler.transpile(kernel, bindings={...}) so the "
            "Möttönen angle pre-computation can run at trace time."
        ) from e

    if isinstance(amplitudes, Vector):
        const_array = amplitudes.value.get_const_array()
        if const_array is None:
            raise ValueError(
                "amplitude_encoding received a Vector[Float] handle without "
                "concrete values at trace time. Bind it via "
                "transpiler.transpile(kernel, bindings={...}) for compile-time "
                "amplitudes, or use amplitude_encoding_from_angles with "
                "parameters=[...] for runtime-parametric angles."
            )
        concrete_amplitudes: Sequence[float] | Sequence[complex] | np.ndarray = (
            np.asarray(const_array)
        )
    else:
        concrete_amplitudes = amplitudes

    gate = MottonenAmplitudeEncoding(concrete_amplitudes)
    if gate.num_target_qubits != n:
        raise ValueError(
            f"amplitude_encoding requires {gate.num_target_qubits} qubits "
            f"for an amplitude vector of length {len(concrete_amplitudes)}, "
            f"got {n}"
        )

    qubit_list: list[Qubit] = [qubits[i] for i in range(n)]
    result = gate(*qubit_list)
    for i in range(n):
        qubits[i] = result[i]
    return qubits


def amplitude_encoding_from_angles(
    qubits: Vector[Qubit],
    ry_angles: Sequence[float] | np.ndarray | Vector[Float],
    rz_angles: Sequence[float] | np.ndarray | Vector[Float] | None = None,
) -> Vector[Qubit]:
    """Apply Möttönen amplitude encoding from pre-computed Ry / Rz angles.

    .. important::

        **Pre-condition: ``qubits`` must currently be in the all-zero
        state** :math:`|0\\rangle^{\\otimes n}`.  The Möttönen Gray-walk
        emission produced by these angle vectors only encodes the
        intended state when starting from :math:`|0\\rangle^{\\otimes n}`;
        applied to any other input it produces ``U |\\phi\\rangle`` for
        the same ``U`` and a different ``|\\phi\\rangle``, which is in
        general not the target amplitude vector.

    Companion to :func:`amplitude_encoding` for the **parametric** use
    case: the user pre-computes the Gray-walk Ry (and optionally Rz)
    angles classically with
    :func:`qamomile.linalg.compute_mottonen_amplitude_encoding_ry_angles`
    and
    :func:`qamomile.linalg.compute_mottonen_amplitude_encoding_rz_angles`,
    then passes them in as either concrete sequences or as
    ``Vector[Float]`` handles obtained from kernel parameters.  In the
    latter case the angles can be left as runtime parameters
    (``transpiler.transpile(kernel, parameters=["ry_angles", ...])``)
    so the same compiled circuit can be re-bound to different
    amplitude vectors without recompilation — useful inside hybrid
    optimisation loops.

    Unlike :func:`amplitude_encoding`, this function does NOT wrap the
    emission in a :class:`MottonenAmplitudeEncoding` ``CompositeGate``
    box on the IR side; the Ry / Rz / CNOT Gray-walk gates are emitted
    directly into the surrounding kernel.  Resource estimation /
    visualization will therefore see the elementary gates rather than
    a single high-level op.

    Args:
        qubits (Vector[Qubit]): Vector of ``n`` qubit handles, expected
            to start in :math:`|0\\rangle^{\\otimes n}`.
        ry_angles (Sequence[float] | np.ndarray | Vector[Float]):
            Gray-walk Ry angles for the magnitude stage.  Must have
            length ``2**n - 1``.
        rz_angles (Sequence[float] | np.ndarray | Vector[Float] | None):
            Gray-walk Rz angles for the phase stage.  Pass ``None``
            (default) to skip the Rz stage entirely (real-amplitude
            path); otherwise must have length ``2**n - 1`` as well.

    Returns:
        Vector[Qubit]: The same *qubits* vector, with each element
            updated to the post-encoding qubit handle.

    Raises:
        ValueError: If ``ry_angles`` / ``rz_angles`` is a concrete
            sequence whose length does not match ``2**n - 1``, or if the
            ``qubits`` vector has an unresolved symbolic shape that
            ``get_size`` cannot reduce to a concrete integer.  When the
            angle argument is a ``Vector[Float]`` handle the length check
            is skipped (the shape may be symbolic at trace time); a
            runtime mismatch then surfaces as a backend bind-time error
            instead.

    Example::

        from qamomile.linalg import (
            compute_mottonen_amplitude_encoding_ry_angles,
            compute_mottonen_amplitude_encoding_rz_angles,
        )

        # Pre-compute classically (outside the kernel)
        ry = compute_mottonen_amplitude_encoding_ry_angles(amps)
        rz = compute_mottonen_amplitude_encoding_rz_angles(amps)

        @qmc.qkernel
        def prepare(
            ry_a: qmc.Vector[qmc.Float],
            rz_a: qmc.Vector[qmc.Float],
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, name="q")
            q = amplitude_encoding_from_angles(q, ry_a, rz_a)
            return qmc.measure(q)

        exe = transpiler.transpile(prepare, parameters=["ry_a", "rz_a"])
        # Same compiled circuit, re-bound at "runtime":
        exe.run(transpiler.executor(),
                bindings={"ry_a": ry.tolist(), "rz_a": rz.tolist()})
    """
    n = get_size(qubits)
    expected_len = 2**n - 1

    # Length validation only applies to concrete sequences.  ``Vector[Float]``
    # handles obtained from kernel parameters carry a symbolic shape at
    # trace time, so a static check would either spuriously fail (symbolic
    # shape resolves to 0) or require resolving bindings inside this helper.
    # In the parametric case the user contracts to bind exactly ``2**n - 1``
    # values; a mismatch surfaces at backend bind time.
    if not isinstance(ry_angles, Vector):
        ry_len = len(ry_angles)
        if ry_len != expected_len:
            raise ValueError(
                f"ry_angles must have length 2**n - 1 = {expected_len} for "
                f"n={n} qubits, got {ry_len}"
            )
    if rz_angles is not None and not isinstance(rz_angles, Vector):
        rz_len = len(rz_angles)
        if rz_len != expected_len:
            raise ValueError(
                f"rz_angles must have length 2**n - 1 = {expected_len} for "
                f"n={n} qubits, got {rz_len}"
            )

    qubit_list: list[Qubit] = [qubits[i] for i in range(n)]
    _emit_mottonen_gates(qubit_list, n, ry_angles, gate="ry")
    if rz_angles is not None:
        _emit_mottonen_gates(qubit_list, n, rz_angles, gate="rz")
    for i in range(n):
        qubits[i] = qubit_list[i]
    return qubits
