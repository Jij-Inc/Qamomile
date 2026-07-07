"""QKernel helpers for compiler-facing callable invocation objects."""

from __future__ import annotations

from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.callable import (
    CallableDef,
    CallableRef,
    CallPolicy,
    InvokeOperation,
    signature_from_block,
)
from qamomile.circuit.ir.value import Value


def block_call_operands_and_results(
    block: Block,
    inputs_map: dict[str, Value],
) -> tuple[list[Value], list[Value]]:
    """Map a callee block's ports to call operands and result values.

    Args:
        block (Block): Callee body whose labels, inputs, and outputs define
            the call contract.
        inputs_map (dict[str, Value]): Actual argument values keyed by callee
            label.

    Returns:
        tuple[list[Value], list[Value]]: Call operands in callee-label order
        and result values with pass-through inputs advanced to their next SSA
        version.
    """
    inputs = [inputs_map[label] for label in block.label_args]
    dummy_inputs = {
        value.logical_id: idx for idx, value in enumerate(block.input_values)
    }

    results: list[Value] = []
    for dummy_return in block.output_values:
        if dummy_return.logical_id in dummy_inputs:
            input_idx = dummy_inputs[dummy_return.logical_id]
            results.append(inputs[input_idx].next_version())
        else:
            results.append(dummy_return)

    return inputs, results


def qkernel_callable_attrs() -> dict[str, str]:
    """Return standard attrs for qkernel callable invocations.

    Returns:
        dict[str, str]: Serializer-friendly attrs marking an inline qkernel
        call.
    """
    return {"kind": "qkernel", "default_policy": CallPolicy.INLINE.name}


def qkernel_callable_ref(name: str) -> CallableRef:
    """Return the compiler-facing callable reference for a qkernel.

    Args:
        name (str): User-visible qkernel name.

    Returns:
        CallableRef: Stable reference used by ``InvokeOperation`` call sites.
    """
    return CallableRef(namespace="user.qkernel", name=name)


def qkernel_callable_def(name: str, block: Block) -> CallableDef:
    """Build the inline-by-default callable definition for a qkernel block.

    Args:
        name (str): User-visible qkernel name.
        block (Block): Implementation body for the qkernel.

    Returns:
        CallableDef: Compiler-facing definition with ``INLINE`` policy.
    """
    return CallableDef(
        ref=qkernel_callable_ref(name),
        signature=signature_from_block(block),
        body=block,
        default_policy=CallPolicy.INLINE,
        attrs=qkernel_callable_attrs(),
    )


def qkernel_invoke_block(
    name: str,
    block: Block,
    inputs_map: dict[str, Value],
) -> InvokeOperation:
    """Create an ``InvokeOperation`` for a qkernel call.

    Args:
        name (str): User-visible qkernel name.
        block (Block): Callee body referenced by the callable definition.
        inputs_map (dict[str, Value]): Actual argument values keyed by callee
            label.

    Returns:
        InvokeOperation: Inline-by-default qkernel invocation.
    """
    inputs, results = block_call_operands_and_results(block, inputs_map)

    return InvokeOperation(
        operands=inputs,
        results=results,
        target=qkernel_callable_ref(name),
        attrs=qkernel_callable_attrs(),
        definition=qkernel_callable_def(name, block),
    )
