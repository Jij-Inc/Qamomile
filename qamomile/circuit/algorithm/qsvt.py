r"""Build QSVT and eigenstate-filtering kernels on top of a block encoding.

Quantum singular value transformation (QSVT) applies a polynomial to the
operator hidden inside a block encoding. Given a descriptor ``U`` whose good
block is :math:`A/\alpha` and an odd-degree phase sequence :math:`\Phi`, the
alternation

.. math::

    U_{\Phi} = e^{i\phi_0 (2\Pi - I)}
    \prod_{k} \left[ U\, e^{i\phi_{2k+1}(2\Pi - I)}\, U^{\dagger}\,
    e^{i\phi_{2k+2}(2\Pi - I)} \right] U\, e^{i\phi_{\ell-1}(2\Pi - I)}

encodes :math:`P(A/\alpha)` in the same block, with :math:`\Pi` the projector
onto the all-zero signal register.

Taking :math:`P \approx \mathrm{sign}` turns the alternation into a reflection
about the eigenspace below the shift, and one Hadamard-test qubit turns that
reflection into the projector :math:`P_{<\mu} = (I - R_{<\mu})/2` of Lin & Tong
(2020, arXiv:2002.12508). Sampling the probe kernel below and post-selecting on
all-zero ancillas estimates :math:`\lVert P_{<\mu}\lvert\varphi_0\rangle\rVert^2`,
the single scalar a classical binary search over :math:`\mu` consumes to locate
the ground energy.

The builders take an :class:`~qamomile.circuit.LCUBlockEncoding` and close over
it, because a controlled call (``qmc.control``) cannot forward a descriptor
argument to the wrapped kernel. Any producer satisfying the descriptor contract
— Ising-Z, general Pauli, or a nested LCU composition — can be passed in.
"""

from __future__ import annotations

import math

import qamomile.circuit as qmc
from qamomile.circuit.frontend.qkernel import QKernel
from qamomile.circuit.stdlib.lcu_block_encoding import LCUBlockEncoding

_HALF_PI = math.pi / 2
"""Quarter-turn phase used to cancel the alternation convention factor."""


def qsvt_projector_phase(num_signal_qubits: int) -> QKernel:
    r"""Build the QSVT projector-controlled phase for a signal-register width.

    The returned kernel implements :math:`e^{i\phi(2\Pi - I)}` exactly, where
    :math:`\Pi` projects the whole signal register onto :math:`\lvert 0
    \rangle`: one multi-controlled phase of angle :math:`2\phi` conjugated by
    ``X`` on every signal qubit, plus a compensating :math:`e^{-i\phi}` global
    phase. That correction is not cosmetic — the alternation is applied under a
    control in :func:`eigenstate_filter_projector`, where an uncompensated
    phase would become an observable relative phase on the control qubit.

    The width is a builder argument rather than an in-kernel ``shape[0]``
    because a single-qubit signal register degenerates to a plain phase gate,
    and a qkernel body cannot branch on a register width at build time.

    Args:
        num_signal_qubits (int): Positive width of the signal register the
            kernel acts on.

    Returns:
        QKernel: A qkernel with signature ``(signal: Vector[Qubit], phase:
            Float) -> Vector[Qubit]``.

    Raises:
        TypeError: If ``num_signal_qubits`` is not a Python ``int``.
        ValueError: If ``num_signal_qubits`` is not positive.
    """
    if isinstance(num_signal_qubits, bool) or not isinstance(num_signal_qubits, int):
        raise TypeError(
            f"num_signal_qubits must be an int; got {type(num_signal_qubits).__name__}."
        )
    if num_signal_qubits < 1:
        raise ValueError(
            f"num_signal_qubits must be positive; got {num_signal_qubits}."
        )
    width = num_signal_qubits

    # A qkernel body cannot branch on the width: `if` statements are traced
    # into control-flow regions, so both arms would be built and the
    # single-qubit arm would ask for a zero-control gate. Pick the body here,
    # in plain Python, instead.
    if width == 1:

        @qmc.qkernel
        def uncorrected(
            signal: qmc.Vector[qmc.Qubit], phase: qmc.Float
        ) -> qmc.Vector[qmc.Qubit]:
            """Apply the uncorrected phase on a one-qubit signal register.

            Args:
                signal (qmc.Vector[qmc.Qubit]): One-qubit signal register.
                phase (qmc.Float): Rotation angle :math:`\\phi` in radians.

            Returns:
                qmc.Vector[qmc.Qubit]: The signal register after the phase.
            """
            signal[0] = qmc.x(signal[0])
            signal[0] = qmc.p(signal[0], 2.0 * phase)
            signal[0] = qmc.x(signal[0])
            return signal

    else:

        @qmc.qkernel
        def uncorrected(
            signal: qmc.Vector[qmc.Qubit], phase: qmc.Float
        ) -> qmc.Vector[qmc.Qubit]:
            """Apply the uncorrected phase via a multi-controlled phase gate.

            Args:
                signal (qmc.Vector[qmc.Qubit]): Signal register of the encoding.
                phase (qmc.Float): Rotation angle :math:`\\phi` in radians.

            Returns:
                qmc.Vector[qmc.Qubit]: The signal register after the phase.
            """
            multi_controlled_phase = qmc.control(qmc.p, num_controls=width - 1)
            for i in range(width):
                signal[i] = qmc.x(signal[i])
            controls = signal[1:]
            target = signal[0]
            controls, target = multi_controlled_phase(controls, target, 2.0 * phase)
            signal[1:] = controls
            signal[0] = target
            for i in range(width):
                signal[i] = qmc.x(signal[i])
            return signal

    @qmc.qkernel
    def projector_phase(
        signal: qmc.Vector[qmc.Qubit], phase: qmc.Float
    ) -> qmc.Vector[qmc.Qubit]:
        """Apply the projector-controlled phase to the signal register.

        Args:
            signal (qmc.Vector[qmc.Qubit]): Signal register of the encoding.
            phase (qmc.Float): Rotation angle :math:`\\phi` in radians.

        Returns:
            qmc.Vector[qmc.Qubit]: The signal register after the phase.
        """
        # The conjugated phase gate realizes diag(exp(2i*phase), 1, ..., 1),
        # which is exp(i*phase) * exp(i*phase*(2*Pi - I)). Cancel that factor so
        # the operator stays exact under an outer control.
        corrected = qmc.global_phase(uncorrected, (-1.0) * phase)
        return corrected(signal, phase)

    return projector_phase


def qsvt_alternation(encoding: LCUBlockEncoding) -> QKernel:
    r"""Build the odd-parity QSVT alternation for one block encoding.

    The returned kernel interleaves ``encoding.unitary``, its inverse, and
    :func:`qsvt_projector_phase`, so that the real part of its good block is
    :math:`P(A/\alpha)` for the degree ``len(phi) - 1`` polynomial defined by
    the phases — the imaginary part carries the complementary QSP component and
    vanishes wherever :math:`\lvert P \rvert \to 1`. The phase vector must have
    even length: the alternation applies the encoding an odd number of times,
    which is the parity a sign/step approximation needs.

    Phases follow the reflection (QSVT) convention. Sequences produced in the
    Wx (signal-rotation) convention — ``pyqsp`` returns those — are converted by
    adding :math:`\pi/4` to the two end phases and :math:`\pi/2` to every
    interior phase.

    Args:
        encoding (LCUBlockEncoding): Block-encoding descriptor to transform.
            Captured at build time, so each descriptor needs its own kernel.

    Returns:
        QKernel: A qkernel with signature ``(signal: Vector[Qubit], system:
            Vector[Qubit], phi: Vector[Float]) -> tuple[Vector[Qubit],
            Vector[Qubit]]``. ``phi`` must be bound at compile time because its
            length fixes the circuit structure.

    Raises:
        TypeError: If ``encoding`` is not an :class:`LCUBlockEncoding`.

    Example:
        >>> import qamomile.circuit as qmc
        >>> from qamomile.circuit.algorithm import qsvt_alternation
        >>> encoding = qmc.ising_z_block_encoding({(0,): 1.0}, 1)
        >>> alternation = qsvt_alternation(encoding)
    """
    if not isinstance(encoding, LCUBlockEncoding):
        raise TypeError(
            f"encoding must be an LCUBlockEncoding; got {type(encoding).__name__}."
        )
    projector_phase = qsvt_projector_phase(encoding.num_signal_qubits)

    @qmc.qkernel
    def uncorrected(
        signal: qmc.Vector[qmc.Qubit],
        system: qmc.Vector[qmc.Qubit],
        phi: qmc.Vector[qmc.Float],
    ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
        """Apply the bare alternation of encodings and projector phases.

        Args:
            signal (qmc.Vector[qmc.Qubit]): Signal register of the encoding.
            system (qmc.Vector[qmc.Qubit]): System register of the encoding.
            phi (qmc.Vector[qmc.Float]): Reflection-convention phases, even
                length.

        Returns:
            tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]: Signal and
                system registers after the alternation.
        """
        length = phi.shape[0]
        signal = projector_phase(signal, phi[0])
        for i in range(0, length - 2, 2):
            signal, system = encoding.unitary(signal, system)
            signal = projector_phase(signal, phi[i + 1])
            signal, system = qmc.inverse(encoding.unitary)(signal, system)
            signal = projector_phase(signal, phi[i + 2])
        signal, system = encoding.unitary(signal, system)
        signal = projector_phase(signal, phi[length - 1])
        return signal, system

    @qmc.qkernel
    def alternation(
        signal: qmc.Vector[qmc.Qubit],
        system: qmc.Vector[qmc.Qubit],
        phi: qmc.Vector[qmc.Float],
    ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
        """Apply the QSVT alternation defined by the phase sequence.

        Args:
            signal (qmc.Vector[qmc.Qubit]): Signal register of the encoding.
            system (qmc.Vector[qmc.Qubit]): System register of the encoding.
            phi (qmc.Vector[qmc.Float]): Reflection-convention phases, even
                length.

        Returns:
            tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]: Signal and
                system registers after the alternation.
        """
        # Fix the alternation's convention factor so that the good block reads
        # P(A/alpha) + i*(junk): with phases converted from the Wx convention,
        # the raw alternation encodes i**(len(phi)) times that combination.
        # Downstream consumers (the Hadamard test below) read the real part, so
        # the factor must go.
        corrected = qmc.global_phase(uncorrected, (-_HALF_PI) * phi.shape[0])
        return corrected(signal, system, phi)

    return alternation


def eigenstate_filter_projector(encoding: LCUBlockEncoding) -> QKernel:
    r"""Build the Lin & Tong projector kernel for one block encoding.

    A Hadamard test on one extra qubit turns the alternation block :math:`R`
    into :math:`(I - R)/2`, held in the block selected by all-zero projector
    *and* signal qubits. With phases approximating
    :math:`R \approx \mathrm{sign}(A/\alpha)` and ``encoding`` block-encoding
    :math:`H - \mu I`, that is the projector :math:`P_{<\mu}` onto the
    eigenspace of :math:`H` below :math:`\mu`.

    Args:
        encoding (LCUBlockEncoding): Block-encoding descriptor whose sign
            function is projected. Captured at build time.

    Returns:
        QKernel: A qkernel with signature ``(proj: Vector[Qubit], signal:
            Vector[Qubit], system: Vector[Qubit], phi: Vector[Float]) ->
            tuple[Vector[Qubit], Vector[Qubit], Vector[Qubit]]``. ``proj`` is a
            one-qubit register.

    Raises:
        TypeError: If ``encoding`` is not an :class:`LCUBlockEncoding`.
    """
    alternation = qsvt_alternation(encoding)

    @qmc.qkernel
    def projector(
        proj: qmc.Vector[qmc.Qubit],
        signal: qmc.Vector[qmc.Qubit],
        system: qmc.Vector[qmc.Qubit],
        phi: qmc.Vector[qmc.Float],
    ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
        """Apply the Hadamard-test projector built from the QSVT reflection.

        Args:
            proj (qmc.Vector[qmc.Qubit]): One-qubit Hadamard-test register.
            signal (qmc.Vector[qmc.Qubit]): Signal register of the encoding.
            system (qmc.Vector[qmc.Qubit]): System register of the encoding.
            phi (qmc.Vector[qmc.Float]): Reflection-convention phases.

        Returns:
            tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit],
                qmc.Vector[qmc.Qubit]]: Projector, signal, and system registers.
        """
        proj[0] = qmc.h(proj[0])
        proj[0] = qmc.z(proj[0])
        proj[0], signal, system = qmc.control(alternation)(proj[0], signal, system, phi)
        proj[0] = qmc.h(proj[0])
        return proj, signal, system

    return projector


def eigenstate_filter_probe(encoding: LCUBlockEncoding) -> QKernel:
    r"""Build the sampling circuit of the Lin & Tong ground-energy search.

    Prepares the uniform superposition :math:`\lvert\varphi_0\rangle =
    H^{\otimes n}\lvert 0\rangle` on the system register, applies
    :func:`eigenstate_filter_projector`, and measures every register. The
    fraction of shots whose projector and signal bits are all zero estimates
    :math:`\lVert P_{<\mu}\lvert\varphi_0\rangle\rVert^2`; the system bits of
    those shots are samples of the projected state.

    Args:
        encoding (LCUBlockEncoding): Block-encoding descriptor of the shifted
            operator :math:`H - \mu I`. Register widths are read from it.

    Returns:
        QKernel: A qkernel with signature ``(phi: Vector[Float]) ->
            tuple[Vector[Bit], Vector[Bit], Vector[Bit]]`` returning the
            projector, signal, and system measurements in that order.

    Raises:
        TypeError: If ``encoding`` is not an :class:`LCUBlockEncoding`.
    """
    projector = eigenstate_filter_projector(encoding)
    num_signal_qubits = encoding.num_signal_qubits
    num_system_qubits = encoding.num_system_qubits

    @qmc.qkernel
    def probe(
        phi: qmc.Vector[qmc.Float],
    ) -> tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]:
        """Prepare the uniform superposition, filter it, and measure.

        Args:
            phi (qmc.Vector[qmc.Float]): Reflection-convention phases.

        Returns:
            tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit],
                qmc.Vector[qmc.Bit]]: Projector, signal, and system
                measurement outcomes.
        """
        proj = qmc.qubit_array(1, "proj")
        signal = qmc.qubit_array(num_signal_qubits, "signal")
        system = qmc.qubit_array(num_system_qubits, "system")
        for i in range(num_system_qubits):
            system[i] = qmc.h(system[i])
        proj, signal, system = projector(proj, signal, system, phi)
        return qmc.measure(proj), qmc.measure(signal), qmc.measure(system)

    return probe
