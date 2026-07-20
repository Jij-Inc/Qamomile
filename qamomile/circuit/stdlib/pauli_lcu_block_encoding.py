"""Build block encodings from complex Pauli linear combinations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from qamomile.circuit.frontend.composite_gate import configure_composite
from qamomile.circuit.frontend.handle import Qubit, Vector
from qamomile.circuit.frontend.operation.global_phase import global_phase
from qamomile.circuit.frontend.operation.qubit_gates import x, y, z
from qamomile.circuit.frontend.qkernel import QKernel, qkernel
from qamomile.circuit.ir.operation.callable import CallPolicy
from qamomile.circuit.stdlib.lcu_block_encoding import (
    LCUBlockEncoding,
    _BlockEncodingUnitary,
    _build_lcu_block_encoding_unitary,
    _coefficient_phase,
    _identity_vector,
    _lcu_num_signal_qubits,
    _register_lcu_block_encoding_static_binding,
)
from qamomile.linalg import PauliLCU, PauliLCUTerm
from qamomile.observable import Pauli, PauliOperator


@dataclass(frozen=True, slots=True, eq=False)
class PauliLCUBlockEncoding(LCUBlockEncoding):
    """Identify an LCU block encoding produced from a Pauli decomposition.

    The subtype adds no qkernel-visible fields. Reusable qkernels should
    annotate encoding arguments with :class:`LCUBlockEncoding`; the Pauli
    subtype remains available for producer-specific host-side code and
    backward-compatible serialized templates.

    Args:
        unitary (QKernel): QKernel implementing the exact block-encoding
            unitary with the static ``(signal, system)`` ABI.
        normalization (float): Finite positive block normalization.
        num_signal_qubits (int): Concrete positive width of the complete
            signal register.
        num_system_qubits (int): Concrete positive width of the ordered system
            register.

    Raises:
        TypeError: If an inherited descriptor field has an invalid type or the
            unitary ABI is invalid.
        ValueError: If normalization or a register width is non-positive or
            non-finite.
    """


def pauli_lcu_block_encoding(lcu: PauliLCU) -> PauliLCUBlockEncoding:
    r"""Create a static exact block encoding of a complex Pauli LCU.

    For a nonzero decomposition

    .. math::

        A = \sum_j c_j P_j, \qquad \alpha = \sum_j |c_j|,

    the descriptor's qkernel ``unitary`` implements ``U`` and satisfies

    .. math::

        (\langle 0|^{\otimes a} \otimes I) U
        (|0\rangle^{\otimes a} \otimes I) = A / \alpha.

    Multi-term encodings use real-amplitude PREPARE weights
    ``sqrt(abs(c_j) / alpha)`` and SELECT cases
    ``exp(1j * arg(c_j)) * P_j``. The identity Pauli word is a normal case, so
    its coefficient phase is retained. The zero operator uses one signal
    qubit and an ``X`` gate, giving an exact zero all-zero block with
    normalization ``1.0``; its ``PauliLCU.alpha`` remains ``0.0``.

    The retained Pauli LCU is square, static, and encoded exactly. When
    :meth:`PauliLCU.from_matrix` truncated coefficients, its source-to-retained
    error remains available as ``lcu.truncation_error_bound`` and is not an
    error in this unitary. The unitary accepts arbitrary signal states, returns
    the same signal and system wires in the same order, supports
    :func:`~qamomile.circuit.inverse`, and allocates no hidden source-level
    logical workspace. Backend-only decomposition scratch is permitted only
    when resource-accounted and exactly uncomputed for all inputs, including
    under inverse and control.

    Args:
        lcu (PauliLCU): Immutable retained Pauli decomposition. It must
            describe at least one system qubit.

    Returns:
        PauliLCUBlockEncoding: Frozen non-callable descriptor. Allocate its
            registers with ``num_signal_qubits`` and ``num_system_qubits``,
            then invoke ``unitary(signal, system)``.

    Raises:
        TypeError: If ``lcu`` is not a ``PauliLCU``.
        ValueError: If ``lcu`` represents a scalar zero-qubit system.

    Example:
        >>> import numpy as np
        >>> import qamomile.circuit as qmc
        >>> from qamomile.linalg import PauliLCU
        >>> lcu = PauliLCU.from_matrix(np.array([[0, 1], [0, 0]], complex))
        >>> encoding = qmc.pauli_lcu_block_encoding(lcu)
        >>> @qmc.qkernel
        ... def circuit() -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
        ...     signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
        ...     system = qmc.qubit_array(encoding.num_system_qubits, "system")
        ...     return encoding.unitary(signal, system)
    """
    if not isinstance(lcu, PauliLCU):
        raise TypeError("lcu must be a PauliLCU.")
    if lcu.num_qubits == 0:
        raise ValueError(
            "pauli_lcu_block_encoding requires at least one system qubit; "
            "a 1 x 1 scalar matrix has no Qamomile system register."
        )

    coefficients = tuple(term.coefficient for term in lcu.terms)
    cases = tuple(_build_phased_pauli_case(term) for term in lcu.terms)
    unitary = _build_lcu_block_encoding_unitary(
        cases,
        coefficients,
        lcu.num_qubits,
        description="Pauli LCU block encoding",
        preparation_name="mottonen_amplitude_encoding",
    )
    unitary = _configure_block_encoding(unitary, lcu)
    return PauliLCUBlockEncoding(
        unitary=unitary,
        normalization=lcu.alpha if lcu.num_terms else 1.0,
        num_signal_qubits=_lcu_num_signal_qubits(lcu.num_terms),
        num_system_qubits=lcu.num_qubits,
    )


def _build_phased_pauli_case(
    term: PauliLCUTerm,
) -> QKernel[..., Vector[Qubit]]:
    """Build one explicit QKernel SELECT case for a complex Pauli term.

    Args:
        term (PauliLCUTerm): Nonzero Pauli LCU term.

    Returns:
        QKernel[..., Vector[Qubit]]: QKernel implementing
            ``exp(1j * arg(coefficient)) * P``.
    """
    operators = term.operators
    phase = _coefficient_phase(term.coefficient)

    if not phase:

        @qkernel
        def real_case(system: Vector[Qubit]) -> Vector[Qubit]:
            """Apply one real-positive sparse Pauli word.

            Args:
                system (Vector[Qubit]): Shared SELECT target register.

            Returns:
                Vector[Qubit]: Transformed target register.
            """
            _apply_pauli_word(system, operators)
            return system

        return real_case

    @qkernel
    def phased_case(system: Vector[Qubit]) -> Vector[Qubit]:
        """Apply one phased sparse Pauli word.

        Args:
            system (Vector[Qubit]): Shared SELECT target register.

        Returns:
            Vector[Qubit]: Transformed target register.
        """
        _apply_pauli_word(system, operators)
        return global_phase(_identity_vector, phase)(system)

    return phased_case


def _apply_pauli_word(
    system: Vector[Qubit],
    operators: tuple[PauliOperator, ...],
) -> None:
    """Emit one sparse Pauli word into a target register.

    Args:
        system (Vector[Qubit]): Target register mutated in place.
        operators (tuple[PauliOperator, ...]): Canonical sparse Pauli word.

    Raises:
        ValueError: If an operator contains an unsupported Pauli value.
    """
    for operator in operators:
        if operator.pauli is Pauli.X:
            system[operator.index] = x(system[operator.index])
        elif operator.pauli is Pauli.Y:
            system[operator.index] = y(system[operator.index])
        elif operator.pauli is Pauli.Z:
            system[operator.index] = z(system[operator.index])
        else:
            raise ValueError(f"Unsupported sparse Pauli value: {operator.pauli!r}.")


def _configure_block_encoding(
    unitary: _BlockEncodingUnitary,
    lcu: PauliLCU,
) -> _BlockEncodingUnitary:
    """Attach stable compiler identity and semantic LCU metadata.

    Args:
        unitary (_BlockEncodingUnitary): Composite qkernel to configure.
        lcu (PauliLCU): Decomposition determining the composite unitary.

    Returns:
        _BlockEncodingUnitary: The same configured qkernel.
    """
    semantic_arguments = _semantic_arguments(lcu)
    digest_input = json.dumps(
        semantic_arguments,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(digest_input).hexdigest()[:16]
    configure_composite(
        unitary,
        name="pauli_lcu_block_encoding",
        namespace=f"qamomile.stdlib.pauli_lcu.{digest}",
        policy=CallPolicy.PRESERVE_BOX,
        semantic_arguments=semantic_arguments,
    )
    return unitary


def _semantic_arguments(lcu: PauliLCU) -> dict[str, Any]:
    """Return serializer-safe arguments that determine the encoded unitary.

    Args:
        lcu (PauliLCU): Pauli decomposition to serialize.

    Returns:
        dict[str, Any]: Stable semantic argument mapping.
    """
    return {
        "num_qubits": lcu.num_qubits,
        "signal_qubits": _lcu_num_signal_qubits(lcu.num_terms),
        "terms": [
            {
                "coefficient": [
                    float(term.coefficient.real),
                    float(term.coefficient.imag),
                ],
                "operators": [
                    [operator.pauli.name, int(operator.index)]
                    for operator in term.operators
                ],
            }
            for term in lcu.terms
        ],
    }


_register_lcu_block_encoding_static_binding(
    PauliLCUBlockEncoding,
    "qamomile.stdlib.pauli_lcu_block_encoding",
)


__all__ = [
    "PauliLCUBlockEncoding",
    "pauli_lcu_block_encoding",
]
