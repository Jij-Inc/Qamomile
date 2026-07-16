"""Build block encodings from complex Pauli linear combinations."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np

from qamomile.circuit.frontend.composite_gate import (
    composite_gate,
    configure_composite,
)
from qamomile.circuit.frontend.constructors import uint
from qamomile.circuit.frontend.handle import Qubit, Vector
from qamomile.circuit.frontend.handle.utils import get_size
from qamomile.circuit.frontend.operation.global_phase import global_phase
from qamomile.circuit.frontend.operation.inverse import inverse
from qamomile.circuit.frontend.operation.qubit_gates import x, y, z
from qamomile.circuit.frontend.operation.select import select
from qamomile.circuit.frontend.qkernel import QKernel, qkernel
from qamomile.circuit.frontend.qkernel_build import build_specialized_block
from qamomile.circuit.ir.operation.callable import (
    CallableImplementation,
    CallPolicy,
    CallTransform,
)
from qamomile.circuit.stdlib.state_preparation.mottonen_amplitude_encoding import (
    _mottonen_composite,
)
from qamomile.linalg import PauliLCU, PauliLCUTerm
from qamomile.observable import Pauli, PauliOperator

BlockEncodingKernel = QKernel[
    ...,
    tuple[Vector[Qubit], Vector[Qubit]],
]


@qkernel
def _identity_vector(qubits: Vector[Qubit]) -> Vector[Qubit]:
    """Return a quantum register unchanged.

    Args:
        qubits (Vector[Qubit]): Register to preserve.

    Returns:
        Vector[Qubit]: The same logical register.
    """
    return qubits


def pauli_lcu_num_selection_qubits(lcu: PauliLCU) -> int:
    """Return the selection-register width required by the block encoding.

    At least one qubit is retained for the zero- and single-term paths so the
    returned block-encoding qkernel has one stable two-register signature.

    Args:
        lcu (PauliLCU): Pauli linear combination to encode.

    Returns:
        int: Required number of selection qubits.

    Raises:
        TypeError: If ``lcu`` is not a ``PauliLCU``.
    """
    if not isinstance(lcu, PauliLCU):
        raise TypeError("lcu must be a PauliLCU.")
    if lcu.num_terms <= 1:
        return 1
    return (lcu.num_terms - 1).bit_length()


def pauli_lcu_block_encoding(lcu: PauliLCU) -> BlockEncodingKernel:
    r"""Create a QKernel-backed block encoding of a complex Pauli LCU.

    For a nonzero decomposition

    .. math::

        A = \sum_j c_j P_j, \qquad \alpha = \sum_j |c_j|,

    the returned unitary ``U`` satisfies

    .. math::

        (\langle 0|^{\otimes a} \otimes I) U
        (|0\rangle^{\otimes a} \otimes I) = A / \alpha.

    Multi-term encodings use real-amplitude PREPARE weights
    ``sqrt(abs(c_j) / alpha)`` and SELECT cases
    ``exp(1j * arg(c_j)) * P_j``. The identity Pauli word is a normal case, so
    its coefficient phase is retained. The zero operator uses one selection
    qubit and an ``X`` gate, giving an exact zero top-left block with
    normalization ``1.0``; its ``PauliLCU.alpha`` remains ``0.0``.

    Args:
        lcu (PauliLCU): Immutable Pauli decomposition. It must describe at
            least one system qubit. Allocate the selection register with
            :func:`pauli_lcu_num_selection_qubits` and the system register with
            ``lcu.num_qubits``.

    Returns:
        BlockEncodingKernel: Named composite qkernel with signature
            ``(selection, system) -> (selection, system)``.

    Raises:
        TypeError: If ``lcu`` is not a ``PauliLCU``.
        ValueError: If ``lcu`` represents a scalar zero-qubit system.

    Example:
        >>> import numpy as np
        >>> import qamomile.circuit as qmc
        >>> from qamomile.linalg import PauliLCU
        >>> lcu = PauliLCU.from_matrix(np.array([[0, 1], [0, 0]], complex))
        >>> block = qmc.pauli_lcu_block_encoding(lcu)
        >>> @qmc.qkernel
        ... def circuit() -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
        ...     selection = qmc.qubit_array(
        ...         qmc.pauli_lcu_num_selection_qubits(lcu), "selection"
        ...     )
        ...     system = qmc.qubit_array(lcu.num_qubits, "system")
        ...     return block(selection, system)
    """
    if not isinstance(lcu, PauliLCU):
        raise TypeError("lcu must be a PauliLCU.")
    if lcu.num_qubits == 0:
        raise ValueError(
            "pauli_lcu_block_encoding requires at least one system qubit; "
            "a 1 x 1 scalar matrix has no Qamomile system register."
        )

    if lcu.num_terms == 0:
        kernel = _build_zero_encoding(lcu)
        inverse_kernel = _build_zero_encoding(lcu)
    elif lcu.num_terms == 1:
        kernel = _build_single_term_encoding(lcu)
        inverse_kernel = _build_single_term_encoding(lcu, phase_sign=-1.0)
    else:
        kernel = _build_multi_term_encoding(lcu)
        inverse_kernel = _build_multi_term_encoding(lcu, phase_sign=-1.0)
    return _configure_block_encoding(kernel, inverse_kernel, lcu)


def _build_zero_encoding(lcu: PauliLCU) -> BlockEncodingKernel:
    """Build the exact zero block using one signal-qubit bit flip.

    Args:
        lcu (PauliLCU): Zero Pauli linear combination.

    Returns:
        BlockEncodingKernel: Zero block-encoding qkernel.
    """
    selection_width = pauli_lcu_num_selection_qubits(lcu)

    @composite_gate(name="pauli_lcu_block_encoding")
    def kernel(
        selection: Vector[Qubit],
        system: Vector[Qubit],
    ) -> tuple[Vector[Qubit], Vector[Qubit]]:
        """Apply the zero block-encoding body.

        Args:
            selection (Vector[Qubit]): One-qubit selection register.
            system (Vector[Qubit]): System register.

        Returns:
            tuple[Vector[Qubit], Vector[Qubit]]: Updated selection and system
                registers.
        """
        _validate_register_widths(selection, system, selection_width, lcu.num_qubits)
        selection[0] = x(selection[0])
        return selection, system

    return kernel


def _build_single_term_encoding(
    lcu: PauliLCU,
    *,
    phase_sign: float = 1.0,
) -> BlockEncodingKernel:
    """Build an unconditional phased-Pauli encoding for one term.

    Args:
        lcu (PauliLCU): One-term Pauli linear combination.
        phase_sign (float): Multiplier for the coefficient phase. Use ``-1``
            for the explicit inverse implementation. Defaults to ``1``.

    Returns:
        BlockEncodingKernel: Single-term block-encoding qkernel.
    """
    selection_width = pauli_lcu_num_selection_qubits(lcu)
    term = lcu.terms[0]
    phase = _coefficient_phase(term.coefficient, phase_sign)

    @composite_gate(name="pauli_lcu_block_encoding")
    def kernel(
        selection: Vector[Qubit],
        system: Vector[Qubit],
    ) -> tuple[Vector[Qubit], Vector[Qubit]]:
        """Apply the single phased-Pauli body.

        Args:
            selection (Vector[Qubit]): One-qubit selection register, preserved
                unchanged.
            system (Vector[Qubit]): System register receiving the Pauli word.

        Returns:
            tuple[Vector[Qubit], Vector[Qubit]]: Preserved selection register
                and transformed system register.
        """
        _validate_register_widths(selection, system, selection_width, lcu.num_qubits)
        _apply_pauli_word(system, term.operators)
        system = global_phase(_identity_vector, phase)(system)
        return selection, system

    return kernel


def _build_multi_term_encoding(
    lcu: PauliLCU,
    *,
    phase_sign: float = 1.0,
) -> BlockEncodingKernel:
    """Build a PREPARE-SELECT-PREPARE-dagger encoding.

    Args:
        lcu (PauliLCU): Pauli linear combination containing at least two
            terms.
        phase_sign (float): Multiplier for every SELECT coefficient phase.
            Use ``-1`` for the explicit inverse implementation. Defaults to
            ``1``.

    Returns:
        BlockEncodingKernel: Multi-term block-encoding qkernel.
    """
    selection_width = pauli_lcu_num_selection_qubits(lcu)
    amplitudes = np.zeros(1 << selection_width, dtype=np.float64)
    alpha = lcu.alpha
    sqrt_alpha = math.sqrt(alpha)
    for index, term in enumerate(lcu.terms):
        amplitudes[index] = math.sqrt(abs(term.coefficient)) / sqrt_alpha

    preparation, required_width = _mottonen_composite(amplitudes)
    if required_width != selection_width:
        raise RuntimeError(
            "Möttönen preparation width disagrees with the LCU selection width."
        )
    unprepare = inverse(preparation)
    # A constant UInt keeps the unspecialized composite's selection Vector
    # traceable while lowering still resolves the exact fixed LCU width.
    selector = select(
        tuple(
            _build_phased_pauli_case(term, phase_sign=phase_sign) for term in lcu.terms
        ),
        num_index_qubits=uint(selection_width),
    )

    @composite_gate(name="pauli_lcu_block_encoding")
    def kernel(
        selection: Vector[Qubit],
        system: Vector[Qubit],
    ) -> tuple[Vector[Qubit], Vector[Qubit]]:
        """Apply the multi-term Pauli LCU block encoding.

        Args:
            selection (Vector[Qubit]): All-zero selection register.
            system (Vector[Qubit]): Arbitrary system register.

        Returns:
            tuple[Vector[Qubit], Vector[Qubit]]: Updated selection and system
                registers.
        """
        _validate_register_widths(selection, system, selection_width, lcu.num_qubits)
        selection = preparation(selection)
        selection, system = selector(selection, system)
        selection = unprepare(selection)
        return selection, system

    return kernel


def _build_phased_pauli_case(
    term: PauliLCUTerm,
    *,
    phase_sign: float = 1.0,
) -> QKernel[..., Vector[Qubit]]:
    """Build one explicit QKernel SELECT case for a complex Pauli term.

    Args:
        term (PauliLCUTerm): Nonzero Pauli LCU term.
        phase_sign (float): Multiplier for the coefficient phase. Defaults to
            ``1``.

    Returns:
        QKernel[..., Vector[Qubit]]: QKernel implementing
            ``exp(1j * arg(coefficient)) * P``.
    """
    operators = term.operators
    phase = _coefficient_phase(term.coefficient, phase_sign)

    @qkernel
    def case(system: Vector[Qubit]) -> Vector[Qubit]:
        """Apply one phased sparse Pauli word.

        Args:
            system (Vector[Qubit]): Shared SELECT target register.

        Returns:
            Vector[Qubit]: Transformed target register.
        """
        _apply_pauli_word(system, operators)
        return global_phase(_identity_vector, phase)(system)

    return case


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


def _coefficient_phase(coefficient: complex, phase_sign: float) -> float:
    """Return a coefficient phase with signed zero canonicalized.

    Args:
        coefficient (complex): Nonzero complex Pauli coefficient.
        phase_sign (float): Direction multiplier, normally ``1`` or ``-1``.

    Returns:
        float: Signed coefficient phase, using positive zero for either
            direction when the coefficient is positive real.
    """
    phase = phase_sign * math.atan2(coefficient.imag, coefficient.real)
    return phase if phase else 0.0


def _validate_register_widths(
    selection: Vector[Qubit],
    system: Vector[Qubit],
    expected_selection: int,
    expected_system: int,
) -> None:
    """Validate concrete register widths while allowing symbolic cache traces.

    Args:
        selection (Vector[Qubit]): Selection register.
        system (Vector[Qubit]): System register.
        expected_selection (int): Required selection width.
        expected_system (int): Required system width.

    Raises:
        TypeError: If either argument is not a vector register.
        ValueError: If a concrete register width differs from its requirement.
    """
    _validate_register_width(selection, expected_selection, "selection")
    _validate_register_width(system, expected_system, "system")


def _validate_register_width(
    register: Vector[Qubit],
    expected: int,
    name: str,
) -> None:
    """Validate one concrete vector width and defer symbolic-width checks.

    Args:
        register (Vector[Qubit]): Register to inspect.
        expected (int): Required width.
        name (str): Register name used in diagnostics.

    Raises:
        TypeError: If ``register`` is not a vector.
        ValueError: If its concrete width differs from ``expected``.
    """
    try:
        actual = get_size(register)
    except ValueError:
        return
    if actual != expected:
        unit = "qubit" if expected == 1 else "qubits"
        raise ValueError(
            f"Pauli LCU block encoding requires {expected} {name} {unit}, got {actual}."
        )


def _configure_block_encoding(
    kernel: BlockEncodingKernel,
    inverse_kernel: BlockEncodingKernel,
    lcu: PauliLCU,
) -> BlockEncodingKernel:
    """Attach stable compiler identity and semantic LCU metadata.

    Args:
        kernel (BlockEncodingKernel): Composite qkernel to configure.
        inverse_kernel (BlockEncodingKernel): Explicit inverse body whose
            SELECT phases have been conjugated before tracing.
        lcu (PauliLCU): Decomposition determining the composite unitary.

    Returns:
        BlockEncodingKernel: The same configured qkernel.
    """
    semantic_arguments = _semantic_arguments(lcu)
    digest_input = json.dumps(
        semantic_arguments,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(digest_input).hexdigest()[:16]
    inverse_body = build_specialized_block(
        inverse_kernel,
        parameters=[],
        bindings={},
        qubit_sizes={
            "selection": pauli_lcu_num_selection_qubits(lcu),
            "system": lcu.num_qubits,
        },
    )
    configure_composite(
        kernel,
        name="pauli_lcu_block_encoding",
        namespace=f"qamomile.stdlib.pauli_lcu.{digest}",
        policy=CallPolicy.PRESERVE_BOX,
        implementations=(
            CallableImplementation(
                transform=CallTransform.INVERSE,
                body=inverse_body,
            ),
        ),
        semantic_arguments=semantic_arguments,
    )
    return kernel


def _semantic_arguments(lcu: PauliLCU) -> dict[str, Any]:
    """Return serializer-safe arguments that determine the encoded unitary.

    Args:
        lcu (PauliLCU): Pauli decomposition to serialize.

    Returns:
        dict[str, Any]: Stable semantic argument mapping.
    """
    return {
        "num_qubits": lcu.num_qubits,
        "selection_qubits": pauli_lcu_num_selection_qubits(lcu),
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


__all__ = [
    "pauli_lcu_block_encoding",
    "pauli_lcu_num_selection_qubits",
]
