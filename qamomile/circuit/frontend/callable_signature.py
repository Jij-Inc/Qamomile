"""Frontend signature helpers for callable-style operations."""

from __future__ import annotations

import dataclasses
from typing import Any

from qamomile.circuit.frontend.func_to_block import (
    handle_type_map,
    is_array_type,
)
from qamomile.circuit.frontend.handle import Qubit
from qamomile.circuit.ir.operation.operation import ParamHint, Signature


def _array_element_type(param_type: Any) -> Any:
    """Return the element type from a frontend array annotation.

    Args:
        param_type (Any): A frontend annotation such as ``Vector[Qubit]``.

    Returns:
        Any: The element type when present, otherwise ``None``.
    """
    if hasattr(param_type, "__args__") and param_type.__args__:
        return param_type.__args__[0]
    return getattr(param_type, "element_type", None)


@dataclasses.dataclass(frozen=True)
class CallableSignature:
    """Describe frontend input and output handle types for an opaque callable.

    This class is intentionally a small frontend helper. It lets users write
    signature-shaped APIs such as ``opaque(name, signature=...)`` without
    exposing the compiler-facing ``CallableDef`` model.

    Args:
        inputs (list[Any]): Frontend handle annotations accepted by the
            callable.
        outputs (list[Any]): Frontend handle annotations produced by the
            callable.
    """

    inputs: list[Any]
    outputs: list[Any]

    def to_ir_signature(self) -> Signature:
        """Convert the frontend signature into an IR operation signature.

        Returns:
            Signature: Best-effort IR signature using operation parameter
            hints.

        Raises:
            TypeError: If any frontend type cannot be mapped to an IR value
            type.
        """
        return Signature(
            operands=[
                ParamHint(name=f"arg_{i}", type=handle_type_map(param_type))
                for i, param_type in enumerate(self.inputs)
            ],
            results=[
                ParamHint(name=f"result_{i}", type=handle_type_map(param_type))
                for i, param_type in enumerate(self.outputs)
            ],
        )

    def scalar_qubit_input_count(self) -> int | None:
        """Return scalar-qubit arity when the signature is scalar-only.

        Returns:
            int | None: Number of scalar ``Qubit`` inputs, or ``None`` when
            the signature contains a vector register.
        """
        count = 0
        for param_type in self.inputs:
            if is_array_type(param_type) and _array_element_type(param_type) is Qubit:
                return None
            if param_type is Qubit:
                count += 1
        return count

    def accepts_single_qubit_vector(self) -> bool:
        """Return whether this signature is a one-vector quantum callable.

        Returns:
            bool: ``True`` when both input and output are exactly one
            ``Vector[Qubit]``-style annotation.
        """
        if len(self.inputs) != 1 or len(self.outputs) != 1:
            return False
        return (
            is_array_type(self.inputs[0])
            and _array_element_type(self.inputs[0]) is Qubit
            and is_array_type(self.outputs[0])
            and _array_element_type(self.outputs[0]) is Qubit
        )
