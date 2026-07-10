"""QKernel helpers for compiler-facing callable invocation objects."""

from __future__ import annotations

from typing import Any

from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.callable import (
    CallableDef,
    CallableRef,
    CallPolicy,
    InvokeOperation,
    signature_from_block,
)
from qamomile.circuit.ir.value import Value


def _qkernel_callable_name(kernel: Any) -> str:
    """Return a stable display name for a qkernel-like object.

    Args:
        kernel (Any): QKernel or an object being validated as qkernel-like.

    Returns:
        str: Explicit callable name, qkernel name, or type-name fallback.
    """
    return str(
        getattr(kernel, "_callable_name", None)
        or getattr(kernel, "name", type(kernel).__name__)
    )


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


def qkernel_callable_attrs(kernel: Any) -> dict[str, Any]:
    """Return compiler attrs for a qkernel invocation.

    Composite metadata lives directly on ``QKernel``. This helper is the
    single translation point from that frontend state into serializer-safe IR
    attributes, so direct, controlled, and inverse calls share one identity.

    Args:
        kernel (Any): QKernel-like object carrying callable metadata.

    Returns:
        dict[str, Any]: Serializer-friendly callable attributes.
    """
    kind = getattr(kernel, "_callable_kind", "qkernel")
    policy = getattr(kernel, "_callable_policy", CallPolicy.INLINE)
    attrs: dict[str, Any] = {
        "kind": kind,
        "default_policy": policy.name,
    }
    if kind == "composite":
        gate_type = getattr(kernel, "_callable_gate_type", None)
        attrs.update(
            {
                "gate_type": getattr(gate_type, "name", "CUSTOM"),
                "custom_name": _qkernel_callable_name(kernel),
                "num_control_qubits": 0,
                "num_target_qubits": 0,
                "strategy_name": None,
            }
        )
        estimate_kind = getattr(kernel, "_default_estimate_kind", None)
        if estimate_kind is not None:
            attrs["default_estimate_kind"] = estimate_kind
    return attrs


def qkernel_callable_ref(kernel: Any) -> CallableRef:
    """Return the compiler-facing callable reference for a qkernel.

    Args:
        kernel (Any): QKernel-like object carrying callable metadata.

    Returns:
        CallableRef: Stable reference used by ``InvokeOperation`` call sites.
    """
    kind = getattr(kernel, "_callable_kind", "qkernel")
    name = _qkernel_callable_name(kernel)
    namespace = "user.composite" if kind == "composite" else "user.qkernel"
    if getattr(kernel, "_callable_namespace", None) is not None:
        namespace = kernel._callable_namespace
    return CallableRef(namespace=namespace, name=name)


def qkernel_callable_def(kernel: Any, block: Block) -> CallableDef:
    """Build the inline-by-default callable definition for a qkernel block.

    Args:
        kernel (Any): QKernel-like object carrying callable metadata.
        block (Block): Implementation body for the qkernel.

    Returns:
        CallableDef: Compiler-facing definition for the qkernel.
    """
    attrs = qkernel_callable_attrs(kernel)
    return CallableDef(
        ref=qkernel_callable_ref(kernel),
        signature=signature_from_block(block),
        body=block,
        implementations=list(getattr(kernel, "_callable_implementations", ())),
        resource_models=list(getattr(kernel, "_callable_resource_models", ())),
        default_policy=getattr(kernel, "_callable_policy", CallPolicy.INLINE),
        attrs=attrs,
    )


def qkernel_invoke_block(
    kernel: Any,
    block: Block,
    inputs_map: dict[str, Value],
) -> InvokeOperation:
    """Create an ``InvokeOperation`` for a qkernel call.

    Args:
        kernel (Any): QKernel-like object carrying callable metadata.
        block (Block): Callee body referenced by the callable definition.
        inputs_map (dict[str, Value]): Actual argument values keyed by callee
            label.

    Returns:
        InvokeOperation: Inline-by-default qkernel invocation.
    """
    inputs, results = block_call_operands_and_results(block, inputs_map)
    attrs = qkernel_callable_attrs(kernel)

    return InvokeOperation(
        operands=inputs,
        results=results,
        target=qkernel_callable_ref(kernel),
        attrs=attrs,
        definition=qkernel_callable_def(kernel, block),
    )
