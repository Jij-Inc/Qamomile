"""Apply quantum singular value transformation to an LCU block encoding.

The public :func:`qsvt` helper composes a static
:class:`~qamomile.circuit.stdlib.LCUBlockEncoding` with a QSVT phase sequence.
Both inputs may remain unresolved while an enclosing qkernel is traced and
serialized. The block encoding is supplied through Qamomile's static-binding
contract at transpile time, while phase values may be compile-time bindings or
backend runtime parameters when an explicit compile-time ``phase_count`` is
provided.
"""

from __future__ import annotations

import qamomile.circuit as qmc
from qamomile.circuit.frontend.operation.control_flow import for_loop

from .block_encoding import LCUBlockEncoding


def _flip_projector_auxiliary(
    signal: qmc.Vector[qmc.Qubit],
    auxiliary: qmc.Qubit,
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Qubit]:
    """Flip an auxiliary qubit exactly on the all-zero signal subspace.

    The signal register is inverted around a multi-controlled X whose target is
    ``auxiliary``. The control width is the complete signal width and is always
    positive under the :class:`~qamomile.circuit.stdlib.LCUBlockEncoding`
    contract, including the one-signal-qubit case.

    Args:
        signal (qmc.Vector[qmc.Qubit]): Non-empty block-encoding signal
            register whose all-zero state defines the encoded block.
        auxiliary (qmc.Qubit): Clean reusable auxiliary qubit.

    Returns:
        tuple[qmc.Vector[qmc.Qubit], qmc.Qubit]: Preserved signal register and
            auxiliary flipped exactly on its all-zero subspace.
    """
    num_signal = signal.shape[0]

    with for_loop(0, num_signal, var_name="qsvt_signal_index") as index:
        signal[index] = qmc.x(signal[index])

    controlled_x = qmc.control(qmc.x, num_controls=num_signal)
    signal, auxiliary = controlled_x(signal, auxiliary)

    with for_loop(0, num_signal, var_name="qsvt_signal_index") as index:
        signal[index] = qmc.x(signal[index])

    return signal, auxiliary


def _projector_phase_rotation(
    signal: qmc.Vector[qmc.Qubit],
    auxiliary: qmc.Qubit,
    phase: qmc.Float,
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Qubit]:
    r"""Apply ``exp(i * phase * (2 |0><0| - I))`` to the signal register.

    A clean auxiliary is flipped on the all-zero signal subspace, rotated by
    ``RZ(2 * phase)``, and uncomputed. The auxiliary therefore returns to zero
    and can be reused for every phase in one QSVT sequence. This branch-free
    construction also supports a one-qubit signal register.

    Args:
        signal (qmc.Vector[qmc.Qubit]): Non-empty block-encoding signal
            register whose all-zero state defines the encoded block.
        auxiliary (qmc.Qubit): Clean reusable auxiliary qubit.
        phase (qmc.Float): Projector phase angle in radians.

    Returns:
        tuple[qmc.Vector[qmc.Qubit], qmc.Qubit]: Rotated signal register and
            restored clean auxiliary.
    """
    signal, auxiliary = _flip_projector_auxiliary(signal, auxiliary)
    auxiliary = qmc.rz(auxiliary, 2.0 * phase)
    signal, auxiliary = _flip_projector_auxiliary(signal, auxiliary)
    return signal, auxiliary


def _validate_phase_count(value: int | qmc.UInt) -> None:
    """Validate a concrete phase count when available.

    Args:
        value (int | qmc.UInt): Concrete integer or UInt handle to inspect.

    Raises:
        TypeError: If ``value`` is a boolean or is neither an integer nor a
            UInt handle.
        ValueError: If a concrete integer is not positive.
    """
    if isinstance(value, bool):
        raise TypeError("phase_count must be an integer or qmc.UInt, not bool.")
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("phase_count must be positive.")
        return
    if not isinstance(value, qmc.UInt):
        raise TypeError("phase_count must be an integer or qmc.UInt.")
    if value.value.is_constant():
        resolved = int(value.value.get_const())
        if resolved <= 0:
            raise ValueError("phase_count must be positive.")


def qsvt(
    signal: qmc.Vector[qmc.Qubit],
    system: qmc.Vector[qmc.Qubit],
    phases: qmc.Vector[qmc.Float],
    encoding: LCUBlockEncoding,
    phase_count: int | qmc.UInt | None = None,
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
    r"""Apply quantum singular value transformation to a block encoding.

    For ``phases = (phi_0, ..., phi_d)``, Qamomile applies
    ``R(phi_0), U, R(phi_1), U dagger, ...``, where
    ``R(phi) = exp(i * phi * (2 Pi - I))`` and ``Pi`` projects the complete
    signal register onto all zero. One clean internal auxiliary qubit realizes
    these projector rotations and is restored after every phase. At least one
    phase is required.

    If the leading block of ``encoding.unitary`` is ``A / alpha``, with
    ``alpha = encoding.normalization``, the supplied phase sequence transforms
    singular values of ``A / alpha``. Phase synthesis, polynomial parity, and
    approximation-domain checks are problem dependent and remain the caller's
    responsibility; this helper implements the quantum sequence itself. The
    input values must already use the projector-rotation convention above.
    Raw phases produced for a different QSP signal operator or rotation
    convention, including pyqsp's ``Wx`` convention, must be converted first.

    By default, ``phase_count`` comes from ``phases.shape[0]``. This is the
    shortest API when phases are supplied through ``bindings`` at transpile
    time. To compile once and retain ``phases`` as a backend runtime parameter
    array, pass a separate ``phase_count`` argument and bind that count at
    transpile time; the phase values can then be supplied when the executable
    runs. For a phase vector resolved through transpile-time ``bindings``, an
    explicit count selects a prefix; the vector must contain at least that many
    values, and any remaining values are ignored. For a phase vector preserved
    as a runtime parameter, the executable ABI requires exactly ``phase_count``
    values; shorter and longer runtime vectors are rejected. Omitting the count
    applies the complete compile-time vector.

    The helper emits serializable IR directly into the enclosing qkernel. That
    qkernel may therefore be traced and serialized before either ``encoding``
    or the phase values are known. Do not capture a concrete descriptor in a
    module global; declare an ``LCUBlockEncoding`` argument on the enclosing
    qkernel and bind it after deserialization. A source-backed call rejects a
    non-positive concrete count immediately; the equivalent invalid binding on
    a deserialized template is rejected as a compilation error when the empty
    selected phase prefix is resolved.

    Args:
        signal (qmc.Vector[qmc.Qubit]): Signal register with exactly
            ``encoding.num_signal_qubits`` qubits.
        system (qmc.Vector[qmc.Qubit]): System register with exactly
            ``encoding.num_system_qubits`` qubits.
        phases (qmc.Vector[qmc.Float]): QSVT projector phases in radians, in
            sequence order.
        encoding (LCUBlockEncoding): Static exact LCU block encoding whose
            normalized leading block is transformed.
        phase_count (int | qmc.UInt | None): Number of phase elements to use.
            Defaults to the phase vector length. For compile-time phase
            bindings, an explicit count selects a prefix and the vector must
            contain at least that many values. When ``phases`` is preserved as
            a runtime parameter array, the count must be supplied at compile
            time and the runtime vector must have exactly that length.

    Returns:
        tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]: Transformed signal
            and system registers.

    Raises:
        TypeError: If an explicit ``phase_count`` is a boolean or is neither an
            integer nor a UInt handle.
        ValueError: If a concrete count is not positive.

    Example:
        >>> import qamomile.circuit as qmc
        >>> from qamomile.circuit.serialization import deserialize, serialize
        >>> from qamomile.linalg import PeriodicShiftLCU
        >>> from qamomile.qiskit import QiskitTranspiler
        >>> @qmc.qkernel
        ... def transform(
        ...     encoding: qmc.LCUBlockEncoding,
        ...     phases: qmc.Vector[qmc.Float],
        ... ) -> tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]:
        ...     signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
        ...     system = qmc.qubit_array(encoding.num_system_qubits, "system")
        ...     signal, system = qmc.qsvt(signal, system, phases, encoding)
        ...     return qmc.measure(signal), qmc.measure(system)
        >>> payload = serialize(transform)
        >>> received = deserialize(payload)
        >>> lcu = PeriodicShiftLCU.from_coefficients(
        ...     {-1: 1.0, 0: -2.0, 1: 1.0},
        ...     register_sizes=(2,),
        ... )
        >>> block_encoding = qmc.periodic_shift_lcu_block_encoding(lcu)
        >>> executable = QiskitTranspiler().transpile(
        ...     received,
        ...     bindings={
        ...         "encoding": block_encoding,
        ...         "phases": [0.2, -0.4, 0.7],
        ...     },
        ... )
    """
    phase_length = phases.shape[0]
    resolved_count: int | qmc.UInt = (
        phase_length if phase_count is None else phase_count
    )

    _validate_phase_count(resolved_count)

    selected_phases = phases[0:resolved_count]
    selected_count = resolved_count
    inverse_unitary = qmc.inverse(encoding.unitary)

    projector_auxiliary = qmc.qubit("qsvt_projector_auxiliary")
    signal, projector_auxiliary = _projector_phase_rotation(
        signal,
        projector_auxiliary,
        selected_phases[0],
    )

    remaining_count = selected_count - 1
    pair_count = remaining_count // 2
    with for_loop(0, pair_count, var_name="qsvt_pair") as pair:
        odd_index = pair + pair + 1
        even_index = odd_index + 1

        signal, system = encoding.unitary(signal, system)
        signal, projector_auxiliary = _projector_phase_rotation(
            signal,
            projector_auxiliary,
            selected_phases[odd_index],
        )
        signal, system = inverse_unitary(signal, system)
        signal, projector_auxiliary = _projector_phase_rotation(
            signal,
            projector_auxiliary,
            selected_phases[even_index],
        )

    tail_count = remaining_count % 2
    with for_loop(0, tail_count, var_name="qsvt_tail"):
        final_index = selected_count - 1
        signal, system = encoding.unitary(signal, system)
        signal, projector_auxiliary = _projector_phase_rotation(
            signal,
            projector_auxiliary,
            selected_phases[final_index],
        )

    return signal, system


__all__ = ["qsvt"]
