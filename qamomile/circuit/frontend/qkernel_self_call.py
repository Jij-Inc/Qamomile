"""Self-recursive qkernel invocation helpers."""

from __future__ import annotations

from typing import Any

from qamomile.circuit.frontend.func_to_block import handle_type_map, is_array_type
from qamomile.circuit.frontend.qkernel_callable import (
    qkernel_callable_attrs,
    qkernel_callable_def,
    qkernel_callable_ref,
)
from qamomile.circuit.frontend.qkernel_utils import match_output_to_input
from qamomile.circuit.ir.operation.callable import (
    CallableDef,
    CallPolicy,
    InvokeOperation,
    signature_from_values,
)
from qamomile.circuit.ir.value import Value
from qamomile.circuit.transpiler.errors import FrontendTransformError


def emit_self_call_forward_ref(
    kernel: Any,
    inputs_map: dict[str, Value],
) -> InvokeOperation:
    """Emit a forward-reference invocation for a self-recursive qkernel call.

    Args:
        kernel (Any): QKernel-like object currently building its block.
        inputs_map (dict[str, Value]): Actual argument Values keyed by
            parameter name.

    Returns:
        InvokeOperation: Inline invocation whose definition body is
        back-patched after the enclosing block is constructed.

    Raises:
        FrontendTransformError: If an unmatched array output cannot be
            synthesized for the forward reference.
    """
    label_args = list(kernel.signature.parameters)
    operands = [inputs_map[label] for label in label_args]
    input_types_list = [kernel.input_types[label] for label in label_args]

    claimed = [False] * len(operands)
    results: list[Value] = []
    for i, out_type in enumerate(kernel.output_types):
        matched_idx = match_output_to_input(out_type, input_types_list, claimed)
        if matched_idx is not None:
            claimed[matched_idx] = True
            results.append(operands[matched_idx].next_version())
            continue

        ir_type = handle_type_map(out_type)
        if is_array_type(out_type):
            raise FrontendTransformError(
                f"Self-recursive @qkernel '{kernel.name}' has an "
                f"array output at position {i} with no matching "
                f"array input of the same type.  Forward-ref "
                f"emission cannot synthesize a symbolic shape "
                f"without a matching input; restructure the "
                f"signature so the quantum register is both "
                f"input and output, or remove the self-recursion."
            )
        results.append(Value(type=ir_type, name=f"{kernel.name}_result_{i}"))

    op = InvokeOperation(
        operands=operands,
        results=results,
        target=qkernel_callable_ref(kernel.name),
        attrs=qkernel_callable_attrs(),
        definition=CallableDef(
            ref=qkernel_callable_ref(kernel.name),
            signature=signature_from_values(
                operands,
                results,
                operand_names=label_args,
            ),
            default_policy=CallPolicy.INLINE,
            attrs=qkernel_callable_attrs(),
        ),
    )
    kernel._pending_self_calls.append(op)
    return op


def finalize_pending_self_calls(kernel: Any) -> None:
    """Back-patch forward-reference self-calls after block construction.

    Args:
        kernel (Any): QKernel-like object with ``_pending_self_calls`` and a
            constructed ``_block``.
    """
    if not kernel._pending_self_calls:
        return
    assert kernel._block is not None

    definition = qkernel_callable_def(kernel.name, kernel._block)
    for op in kernel._pending_self_calls:
        op.definition = definition
        op.body = kernel._block

    kernel._pending_self_calls = []
