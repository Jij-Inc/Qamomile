r"""Bundle an amplification state preparation with its register widths.

Amplitude amplification reflects about the state produced from all-zero
registers, so the preparation ``A`` is part of the algorithm rather than
something applied beforehand. :class:`AmplificationPreparation` is the
descriptor for that operator: it carries the kernel together with the signal
and system widths it expects, the same way
:class:`~qamomile.circuit.stdlib.LCUBlockEncoding` carries its ``unitary``
alongside its normalization and widths.

The descriptor is registered for static qkernel binding, so a template can be
written, serialized, and shipped before any concrete preparation exists::

    @qmc.qkernel
    def template(target: qmc.AmplificationPreparation) -> qmc.Vector[qmc.Bit]:
        signal = qmc.qubit_array(target.num_signal_qubits, "signal")
        system = qmc.qubit_array(target.num_system_qubits, "system")
        signal, _ = qmc.amplitude_amplification(signal, system, target, schedule)
        return qmc.measure(signal)

Binding a descriptor built from a block encoding *plus* a system preparation
is the capability this adds: a bare ``LCUBlockEncoding`` slot can only carry
the encoding, never the composed operator.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, cast

from qamomile.circuit.frontend.handle import Qubit, Vector
from qamomile.circuit.frontend.handle.primitives import UInt
from qamomile.circuit.frontend.qkernel import QKernel, qkernel
from qamomile.circuit.frontend.static_binding import (
    StaticBindingFieldSpec,
    StaticBindingMemberSpec,
    StaticBindingSpec,
    register_static_binding,
)

from ..block_encoding import LCUBlockEncoding
from ..block_encoding.lcu import (
    _validate_positive_integer,
    _with_block_encoding_resource_contract,
)

_AmplificationPreparationKernel = QKernel[..., tuple[Vector[Qubit], Vector[Qubit]]]

_PREPARATION_ABI_MESSAGE = (
    "preparation must have signature "
    "(signal: Vector[Qubit], system: Vector[Qubit]) -> "
    "tuple[Vector[Qubit], Vector[Qubit]]."
)


def _has_preparation_abi(candidate: object) -> bool:
    """Return whether a callable exposes the ``(signal, system)`` ABI.

    The check reads introspection attributes rather than testing for
    ``QKernel``: inside a traced qkernel a statically bound member arrives as a
    static-binding member kernel, which is not a ``QKernel`` yet exposes the
    same attributes.

    Args:
        candidate (object): Callable to inspect.

    Returns:
        bool: Whether the callable matches the block-encoding ABI.
    """
    signature = getattr(candidate, "signature", None)
    input_types = getattr(candidate, "input_types", None)
    output_types = getattr(candidate, "output_types", None)
    if signature is None or input_types is None or output_types is None:
        return False
    parameters = tuple(signature.parameters.values())
    return (
        tuple(parameter.name for parameter in parameters) == ("signal", "system")
        and all(
            parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
            and parameter.default is inspect.Parameter.empty
            for parameter in parameters
        )
        and input_types == {"signal": Vector[Qubit], "system": Vector[Qubit]}
        and list(output_types) == [Vector[Qubit], Vector[Qubit]]
    )


def _validate_state_preparation(state_preparation: object) -> Any:
    """Validate a callable against the ``(signal, system)`` preparation ABI.

    Args:
        state_preparation (object): Candidate preparation callable.

    Returns:
        Any: The validated callable, unchanged.

    Raises:
        TypeError: If the callable does not expose the required ABI.
    """
    if not callable(state_preparation) or not _has_preparation_abi(state_preparation):
        raise TypeError(
            "state_preparation must be callable with signature "
            "(signal: Vector[Qubit], system: Vector[Qubit]) -> "
            "tuple[Vector[Qubit], Vector[Qubit]]."
        )
    return state_preparation


@dataclass(frozen=True, slots=True, eq=False)
class AmplificationPreparation:
    r"""Describe a state preparation together with its register widths.

    The preparation maps the all-zero registers to the state being amplified.
    Its all-zero signal subspace is the good subspace, so
    ``num_signal_qubits`` fixes what "success" means and
    ``num_system_qubits`` fixes the rest of the state.

    The kernel must be invertible by :func:`~qamomile.circuit.inverse`, which
    amplitude amplification applies once per sequence: its body may not
    allocate qubits, measure, recurse, or use ``if`` / ``while`` /
    ``qmc.items`` control flow, and any ``qmc.range`` loop needs
    trace-time-resolvable bounds. Every ``LCUBlockEncoding.unitary`` satisfies
    this.

    Descriptor comparison and hashing use object identity rather than field
    values.

    Args:
        preparation (QKernel): Kernel with signature
            ``(signal: Vector[Qubit], system: Vector[Qubit]) ->
            tuple[Vector[Qubit], Vector[Qubit]]``.
        num_signal_qubits (int): Positive width of the signal register.
        num_system_qubits (int): Positive width of the system register.

    Raises:
        TypeError: If ``preparation`` is not a ``QKernel`` with the required
            ABI, or a width is not an integer.
        ValueError: If a width is not positive.

    Example:
        >>> import qamomile.circuit as qmc
        >>> from qamomile.circuit.stdlib import amplification_preparation
        >>> def build(encoding):
        ...     return amplification_preparation(encoding)
    """

    preparation: _AmplificationPreparationKernel
    num_signal_qubits: int
    num_system_qubits: int

    def __post_init__(self) -> None:
        """Validate the ABI and normalize the declared register widths.

        Raises:
            TypeError: If ``preparation`` is not a ``QKernel`` with the exact
                static positional ABI, or a width is not an integer.
            ValueError: If a width is not positive.
        """
        if not isinstance(self.preparation, QKernel):
            raise TypeError("preparation must be a QKernel.")
        if not _has_preparation_abi(self.preparation):
            raise TypeError(_PREPARATION_ABI_MESSAGE)

        object.__setattr__(
            self,
            "num_signal_qubits",
            _validate_positive_integer(self.num_signal_qubits, "num_signal_qubits"),
        )
        object.__setattr__(
            self,
            "num_system_qubits",
            _validate_positive_integer(self.num_system_qubits, "num_system_qubits"),
        )
        object.__setattr__(
            self,
            "preparation",
            _with_block_encoding_resource_contract(
                self.preparation,
                signal_width=self.num_signal_qubits,
                system_width=self.num_system_qubits,
            ),
        )


def _register_amplification_preparation_static_binding(
    annotation: type[AmplificationPreparation],
    type_key: str,
) -> None:
    """Register one preparation descriptor type for static qkernel binding.

    Args:
        annotation (type[AmplificationPreparation]): Descriptor class accepted
            by the binding slot.
        type_key (str): Stable serialization key for qkernels annotated with
            ``annotation``.

    Raises:
        TypeError: If the annotation or generated adapter contract is invalid.
        ValueError: If the annotation or type key is already registered.
    """
    register_static_binding(
        StaticBindingSpec(
            annotation=annotation,
            type_key=type_key,
            fields={
                "num_signal_qubits": StaticBindingFieldSpec(
                    handle_type=UInt,
                    getter=lambda preparation: preparation.num_signal_qubits,
                ),
                "num_system_qubits": StaticBindingFieldSpec(
                    handle_type=UInt,
                    getter=lambda preparation: preparation.num_system_qubits,
                ),
            },
            members={
                "preparation": StaticBindingMemberSpec(
                    input_types={
                        "signal": Vector[Qubit],
                        "system": Vector[Qubit],
                    },
                    output_types=(Vector[Qubit], Vector[Qubit]),
                    return_annotation=tuple[Vector[Qubit], Vector[Qubit]],
                    getter=lambda preparation: preparation.preparation,
                    qubit_width_fields={
                        "signal": "num_signal_qubits",
                        "system": "num_system_qubits",
                    },
                ),
            },
        )
    )


_register_amplification_preparation_static_binding(
    AmplificationPreparation,
    "qamomile.stdlib.amplification_preparation",
)


def amplification_preparation(
    encoding: LCUBlockEncoding,
    system_preparation: Any | None = None,
) -> AmplificationPreparation:
    """Build an amplification preparation from a block encoding.

    Amplitude amplification reflects about the state produced from all-zero
    registers, so a non-trivial system input must be produced by a kernel that
    becomes part of the preparation rather than applied beforehand. This
    factory composes ``system_preparation`` with ``encoding.unitary`` and
    records the encoding's register widths.

    This is a host-side constructor. It cannot be called inside a traced
    qkernel on a statically bound ``LCUBlockEncoding`` parameter, because the
    binding proxy is not an ``LCUBlockEncoding`` instance; build the descriptor
    before tracing and bind it instead.

    Args:
        encoding (LCUBlockEncoding): Block encoding whose all-zero signal
            outcome is the success event.
        system_preparation (Any | None): Kernel mapping an all-zero system
            register to the input state, with signature
            ``(system: Vector[Qubit]) -> Vector[Qubit]``. Must be invertible by
            :func:`~qamomile.circuit.inverse`. Defaults to ``None``, meaning
            the system starts in the all-zero state and the encoding unitary is
            used directly.

    Returns:
        AmplificationPreparation: Descriptor wrapping the composed preparation.

    Raises:
        TypeError: If ``encoding`` is not an ``LCUBlockEncoding`` or
            ``system_preparation`` is not callable.

    Example:
        >>> import qamomile.circuit as qmc
        >>> from qamomile.circuit.stdlib import amplification_preparation
        >>> @qmc.qkernel
        ... def uniform(system: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
        ...     return qmc.h(system)
        >>> def build(encoding):
        ...     return amplification_preparation(encoding, uniform)
    """
    if not isinstance(encoding, LCUBlockEncoding):
        raise TypeError("encoding must be an LCUBlockEncoding.")

    if system_preparation is None:
        return AmplificationPreparation(
            preparation=encoding.unitary,
            num_signal_qubits=encoding.num_signal_qubits,
            num_system_qubits=encoding.num_system_qubits,
        )
    if not callable(system_preparation):
        raise TypeError("system_preparation must be callable.")

    # The None case is resolved here rather than inside the kernel body: the
    # qkernel transformer would rewrite a body-level `if` into traced control
    # flow and trace both branches.
    prepare_system = cast(Callable[..., Any], system_preparation)
    encoding_unitary = cast(Callable[..., Any], encoding.unitary)

    @qkernel
    def prepare(
        signal: Vector[Qubit],
        system: Vector[Qubit],
    ) -> tuple[Vector[Qubit], Vector[Qubit]]:
        """Prepare the system state and apply the block encoding.

        Args:
            signal (Vector[Qubit]): Block-encoding signal register.
            system (Vector[Qubit]): System register in the all-zero state on
                entry.

        Returns:
            tuple[Vector[Qubit], Vector[Qubit]]: Signal and system registers
                after preparation.
        """
        system = prepare_system(system)
        signal, system = encoding_unitary(signal, system)
        return signal, system

    # The name is diagnostic only; stable generated semantic identity remains
    # deferred until the compiler can derive it from the owned callable body.
    prepare.name = "amplification_preparation"
    return AmplificationPreparation(
        preparation=prepare,
        num_signal_qubits=encoding.num_signal_qubits,
        num_system_qubits=encoding.num_system_qubits,
    )


__all__ = [
    "AmplificationPreparation",
    "amplification_preparation",
]
