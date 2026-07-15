"""QKernel helpers for compiler-facing callable invocation objects."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.callable import (
    CallableDef,
    CallableRef,
    CallPolicy,
    InvokeOperation,
    block_call_operands_and_results,
    signature_from_block,
)
from qamomile.circuit.ir.value import ValueLike


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
    preserved_attrs = getattr(kernel, "_callable_attrs_override", None)
    if isinstance(preserved_attrs, dict):
        return dict(preserved_attrs)

    kind = getattr(kernel, "_callable_kind", "qkernel")
    policy = getattr(kernel, "_callable_policy", CallPolicy.INLINE)
    attrs: dict[str, Any] = {
        "kind": kind,
        "default_policy": policy.name,
        "origin_qualified": getattr(kernel, "_callable_namespace", None) is None,
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
        semantic_arguments = getattr(kernel, "_callable_semantic_arguments", {})
        if semantic_arguments:
            attrs["semantic_arguments"] = dict(semantic_arguments)
    return attrs


def qkernel_callable_ref(kernel: Any) -> CallableRef:
    """Return the compiler-facing callable reference for a qkernel.

    Args:
        kernel (Any): QKernel-like object carrying callable metadata.

    Returns:
        CallableRef: Stable reference used by ``InvokeOperation`` call sites.
    """
    preserved_ref = getattr(kernel, "_callable_ref_override", None)
    if isinstance(preserved_ref, CallableRef):
        return preserved_ref

    kind = getattr(kernel, "_callable_kind", "qkernel")
    name = _qkernel_callable_name(kernel)
    namespace = getattr(kernel, "_callable_namespace", None)
    if namespace is None:
        raw_func = getattr(kernel, "raw_func", None)
        module = getattr(raw_func, "__module__", type(kernel).__module__)
        qualname = getattr(raw_func, "__qualname__", name)
        code = getattr(raw_func, "__code__", None)
        first_line = getattr(code, "co_firstlineno", 0)
        family = "composite" if kind == "composite" else "qkernel"
        namespace = f"user.{family}.{module}.{qualname}:{first_line}"
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
        default_policy=getattr(kernel, "_callable_policy", CallPolicy.INLINE),
        attrs=attrs,
    )


def qkernel_invoke_block(
    kernel: Any,
    block: Block,
    inputs_map: Mapping[str, ValueLike],
) -> InvokeOperation:
    """Create an ``InvokeOperation`` for a qkernel call.

    Args:
        kernel (Any): QKernel-like object carrying callable metadata.
        block (Block): Callee body referenced by the callable definition.
        inputs_map (Mapping[str, ValueLike]): Actual argument values keyed by
            callee label.

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
