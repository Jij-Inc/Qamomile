"""Construction-independent contracts for block encodings."""

from __future__ import annotations

import math
from dataclasses import dataclass

import qamomile as qm
from qamomile.effects import CallableKind
from qamomile.types import (
    FunctionType,
    QuantumType,
    infer_declared_function_application,
    logical_width,
    strip_stage,
)


@dataclass(frozen=True, slots=True)
class BlockEncoding:
    """Describe one block-encoding unitary without a runtime registry.

    The callable acts on a single joint state whose leading components are the
    complete all-zero signal register and whose remaining components are the
    ordered system register. The descriptor is host-side composition data. A
    bound source parameter projects it to a closed unary function reference,
    so no descriptor or Python object reaches QCore.

    Args:
        unitary (qm.QKernel): Exact unary unitary over the joint signal/system state.
        normalization (float): Positive finite ``alpha`` for an encoded block
            ``A / alpha``.
        signal_width (int): Positive width of the complete projected signal state.
        system_width (int): Positive width of the encoded system state.
        error_bound (float): Asserted absolute spectral-norm reconstruction
            error for the encoded matrix, excluding device noise. Producer
            factories must certify a nonzero value before setting it.

    Raises:
        TypeError: If a field has the wrong host type or callable ABI.
        ValueError: If a scalar is outside its valid domain or the callable
            width disagrees with the descriptor.
    """

    unitary: qm.QKernel
    normalization: float
    signal_width: int
    system_width: int
    error_bound: float = 0.0

    def __class_getitem__(cls, widths: object) -> FunctionType:
        """Describe the source callable carried by a block encoding.

        Args:
            widths (object): Positive signal and system widths.

        Returns:
            FunctionType: Unary coherent callable over the joint register.

        Raises:
            TypeError: If two integer widths are not supplied.
            ValueError: If either width is nonpositive.
        """

        if not isinstance(widths, tuple) or len(widths) != 2:
            raise TypeError("BlockEncoding expects signal and system widths")
        signal = _positive_width(widths[0], "signal_width")
        system = _positive_width(widths[1], "system_width")
        state = qm.Q[qm.Bits[signal + system]]
        return FunctionType((state,), state)

    def __post_init__(self) -> None:
        """Validate the descriptor against the callable's derived QCore ABI.

        Raises:
            TypeError: If a host field or callable ABI is invalid.
            ValueError: If a width or normalization is outside its domain.
        """

        if isinstance(self.normalization, bool) or not isinstance(
            self.normalization, (int, float)
        ):
            raise TypeError("normalization must be a real scalar")
        normalization = float(self.normalization)
        if not math.isfinite(normalization) or normalization <= 0.0:
            raise ValueError("normalization must be positive and finite")
        if isinstance(self.error_bound, bool) or not isinstance(
            self.error_bound, (int, float)
        ):
            raise TypeError("error_bound must be a real scalar")
        error_bound = float(self.error_bound)
        if not math.isfinite(error_bound) or error_bound < 0.0:
            raise ValueError("error_bound must be nonnegative and finite")
        signal_width = _positive_width(self.signal_width, "signal_width")
        system_width = _positive_width(self.system_width, "system_width")

        total_width = signal_width + system_width
        _validate_unary_unitary(
            self.unitary,
            total_width,
            label="block-encoding unitary",
        )

        object.__setattr__(self, "normalization", normalization)
        object.__setattr__(self, "signal_width", signal_width)
        object.__setattr__(self, "system_width", system_width)
        object.__setattr__(self, "error_bound", error_bound)

    @property
    def total_width(self) -> int:
        """Return the joint signal/system width."""

        return self.signal_width + self.system_width


def projected_block_encoding(
    prepare: qm.QKernel,
    oracle: qm.QKernel,
    *,
    signal_width: int,
    system_width: int,
    normalization: float,
) -> BlockEncoding:
    r"""Construct a block encoding by preparing a projected signal state.

    The returned unitary is
    ``(PREPARE adjoint tensor I) · oracle · (PREPARE tensor I)``. Projecting
    the leading ``signal_width`` components onto zero therefore selects the
    block induced by the prepared signal state. Sparse-access, walk, and
    application-specific oracle constructions all fit this one functional
    contract; they do not need producer descriptor subclasses or compiler
    support.

    Args:
        prepare: Exact unary preparation over the complete signal state.
        oracle: Exact unary oracle over the joint signal/system state.
        signal_width: Positive projected signal width.
        system_width: Positive encoded system width.
        normalization: Positive finite scale of the selected block.

    Returns:
        A construction-independent exact block-encoding contract.

    Raises:
        TypeError: If a callable or scalar has the wrong host type or ABI.
        ValueError: If a width, normalization, or callable width is invalid.
    """

    checked_signal_width = _positive_width(signal_width, "signal_width")
    checked_system_width = _positive_width(system_width, "system_width")
    checked_prepare = _validate_unary_unitary(
        prepare,
        checked_signal_width,
        label="signal preparation",
    )
    checked_oracle = _validate_unary_unitary(
        oracle,
        checked_signal_width + checked_system_width,
        label="projected oracle",
    )
    total_width = checked_signal_width + checked_system_width

    @qm.qkernel
    def encoded(
        state: qm.Q[qm.Bits[total_width]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[total_width]]:  # type: ignore[valid-type]
        """Apply the prepared projected-unitary block encoding.

        Args:
            state (qm.Q[qm.Bits[total_width]]): Signal and system state.

        Returns:
            qm.Q[qm.Bits[total_width]]: Encoded signal and system state.
        """

        signal, system = qm.split_at(state, checked_signal_width)
        signal = checked_prepare(signal)
        state = checked_oracle(qm.concat(signal, system))
        signal, system = qm.split_at(state, checked_signal_width)
        signal = qm.adjoint(checked_prepare, signal)
        return qm.concat(signal, system)

    return BlockEncoding(
        encoded,
        normalization=normalization,
        signal_width=checked_signal_width,
        system_width=checked_system_width,
    )


def pad_block_encoding(block: BlockEncoding, *, signal_width: int) -> BlockEncoding:
    """Add leading unused signal qubits to a block encoding.

    The additional signal register is acted on by identity. Its all-zero
    projection therefore preserves the encoded matrix, normalization, and
    error bound while allowing differently sized blocks to share one algorithm
    signature. The padded unitary remains a first-class deferred callable.

    Args:
        block (BlockEncoding): Existing block encoding.
        signal_width (int): Desired complete signal width, at least the
            block's current signal width.

    Returns:
        BlockEncoding: Equivalent encoding with the requested signal width.

    Raises:
        TypeError: If the descriptor or requested width has the wrong type.
        ValueError: If the requested width is smaller than the current width.
    """

    if not isinstance(block, BlockEncoding):
        raise TypeError("block must be a BlockEncoding")
    checked_width = _positive_width(signal_width, "signal_width")
    if checked_width < block.signal_width:
        raise ValueError("signal_width cannot be smaller than the existing width")
    if checked_width == block.signal_width:
        return block
    padding_width = checked_width - block.signal_width
    total_width = checked_width + block.system_width
    original = block.unitary

    @qm.qkernel
    def padded(
        state: qm.Q[qm.Bits[total_width]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[total_width]]:  # type: ignore[valid-type]
        """Apply the original block after the leading signal padding.

        Args:
            state (qm.Q[qm.Bits[total_width]]): Padded signal and system.

        Returns:
            qm.Q[qm.Bits[total_width]]: Encoded state with untouched padding.
        """

        padding, remainder = qm.split_at(state, padding_width)
        return qm.concat(padding, original(remainder))

    return BlockEncoding(
        padded,
        block.normalization,
        checked_width,
        block.system_width,
        error_bound=block.error_bound,
    )


def _positive_width(value: object, name: str) -> int:
    """Return one validated positive host-side width."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_unary_unitary(
    unitary: object,
    width: int,
    *,
    label: str,
) -> qm.QKernel:
    """Validate one exact unary unitary against its derived QCore ABI.

    Args:
        unitary (object): Candidate source qkernel.
        width (int): Required coherent input and output width.
        label (str): Argument label used in diagnostics.

    Returns:
        qm.QKernel: The validated source qkernel.

    Raises:
        TypeError: If the candidate is not an exactly unitary unary qkernel.
        ValueError: If its dependent ABI cannot instantiate at ``width``.
    """

    if not isinstance(unitary, qm.QKernel):
        raise TypeError(f"{label} must be a Qamomile QKernel")
    function = unitary.module().function(unitary.function_name)
    if len(function.parameters) != 1:
        raise TypeError(f"{label} must accept one state")
    parameter_type = strip_stage(function.parameters[0].value.type)
    result_type = strip_stage(function.result_type)
    if not isinstance(parameter_type, QuantumType) or not isinstance(
        result_type, QuantumType
    ):
        raise TypeError(f"{label} must map one Q[T] value to one Q[T] value")
    try:
        instantiated_result = infer_declared_function_application(
            FunctionType((parameter_type,), result_type, ("state",)),
            (qm.Q[qm.Bits[width]],),
            (None,),
        )
    except ValueError as error:
        raise ValueError(f"{label} must have coherent width {width}") from error
    if logical_width(strip_stage(instantiated_result)) != width:
        raise ValueError(f"{label} must have coherent width {width}")
    if function.kind is not CallableKind.UNITARY:
        raise TypeError(f"{label} body must be exactly unitary")
    return unitary


__all__ = ["BlockEncoding", "pad_block_encoding", "projected_block_encoding"]
