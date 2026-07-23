r"""Build Lin & Tong eigenstate-filtering kernels on top of :func:`qmc.qsvt`.

The quantum singular value transformation itself is Qamomile's stdlib primitive
:func:`~qamomile.circuit.qsvt`: given a block encoding whose good block is
:math:`A/\alpha` and reflection-convention phases :math:`\Phi`, it applies the
alternation ``R(phi_0), U, R(phi_1), U^dagger, ...`` with
:math:`R(\phi) = e^{i\phi(2\Pi - I)}` and encodes :math:`P(A/\alpha)` in the
same block. This module does not reimplement that sequence; it only wraps it.

Taking :math:`P \approx \mathrm{sign}` turns the transformation into a reflection
:math:`R` about the eigenspace below the shift. One Hadamard-test qubit turns
that reflection into the projector :math:`P_{<\mu} = (I - R)/2` of Lin & Tong
(2020, arXiv:2002.12508). Sampling the probe kernel below and post-selecting on
all-zero ancillas estimates
:math:`\lVert P_{<\mu}\lvert\varphi_0\rangle\rVert^2`, the single scalar a
classical binary search over :math:`\mu` consumes to locate the ground energy.

The builders take an :class:`~qamomile.circuit.LCUBlockEncoding` and close over
it, because a controlled call (``qmc.control``) cannot forward a descriptor
argument to the wrapped kernel. Any producer satisfying the descriptor contract
— Ising-Z, general Pauli, or a nested LCU composition — can be passed in.
"""

from __future__ import annotations

import qamomile.circuit as qmc
from qamomile.circuit.frontend.qkernel import QKernel
from qamomile.circuit.stdlib.block_encoding import LCUBlockEncoding


def eigenstate_filter_projector(encoding: LCUBlockEncoding) -> QKernel:
    r"""Build the Lin & Tong projector kernel for one block encoding.

    A Hadamard test on one extra qubit turns the QSVT reflection block
    :math:`R` — produced by :func:`~qamomile.circuit.qsvt` from the reflection
    phases — into :math:`(I - R)/2`, held in the block selected by all-zero
    projector *and* signal qubits. With phases approximating
    :math:`R \approx \mathrm{sign}(A/\alpha)` and ``encoding`` block-encoding
    :math:`H - \mu I`, that is the projector :math:`P_{<\mu}` onto the
    eigenspace of :math:`H` below :math:`\mu`.

    Args:
        encoding (LCUBlockEncoding): Block-encoding descriptor whose sign
            function is projected. Captured at build time, so each descriptor
            needs its own kernel.

    Returns:
        QKernel: A qkernel with signature ``(proj: Vector[Qubit], signal:
            Vector[Qubit], system: Vector[Qubit], phi: Vector[Float]) ->
            tuple[Vector[Qubit], Vector[Qubit], Vector[Qubit]]``. ``proj`` is a
            one-qubit register.

    Raises:
        TypeError: If ``encoding`` is not an :class:`LCUBlockEncoding`.
    """
    if not isinstance(encoding, LCUBlockEncoding):
        raise TypeError(
            f"encoding must be an LCUBlockEncoding; got {type(encoding).__name__}."
        )

    @qmc.qkernel
    def reflector(
        signal: qmc.Vector[qmc.Qubit],
        system: qmc.Vector[qmc.Qubit],
        phi: qmc.Vector[qmc.Float],
    ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
        """Apply the QSVT reflection defined by the phase sequence.

        Args:
            signal (qmc.Vector[qmc.Qubit]): Signal register of the encoding.
            system (qmc.Vector[qmc.Qubit]): System register of the encoding.
            phi (qmc.Vector[qmc.Float]): Reflection-convention phases. Bound at
                compile time; the length fixes the alternation depth.

        Returns:
            tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]: Signal and
                system registers after the transformation.
        """
        signal, system = qmc.qsvt(signal, system, phi, encoding)
        return signal, system

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
        proj[0], signal, system = qmc.control(reflector)(proj[0], signal, system, phi)
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
        for i in qmc.range(num_system_qubits):
            system[i] = qmc.h(system[i])
        proj, signal, system = projector(proj, signal, system, phi)
        return qmc.measure(proj), qmc.measure(signal), qmc.measure(system)

    return probe
