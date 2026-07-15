"""Shared helpers for qkernel invocation and tracing."""

from __future__ import annotations

from typing import Any

from qamomile.circuit.frontend.constructors import bit, float_, uint
from qamomile.circuit.frontend.func_to_block import is_array_type
from qamomile.circuit.frontend.handle import Qubit
from qamomile.circuit.frontend.handle.array import Vector
from qamomile.circuit.frontend.handle.handle import _describe_consume_sites
from qamomile.circuit.frontend.handle.primitives import Bit, Float, Handle, UInt
from qamomile.circuit.ir.types.primitives import UIntType
from qamomile.circuit.ir.value import ArrayValue, Value
from qamomile.circuit.transpiler.errors import QubitConsumedError


def get_array_element_type(param_type: Any) -> type | None:
    """Extract the element type from an array type annotation.

    Args:
        param_type (Any): Frontend annotation such as ``Vector[Qubit]``.

    Returns:
        type | None: Element type when present, otherwise ``None``.
    """
    if hasattr(param_type, "__args__") and param_type.__args__:
        return param_type.__args__[0]
    return getattr(param_type, "element_type", None)


def promote_literal_to_handle(value: Any, expected_type: Any) -> Any:
    """Promote a Python literal to a scalar handle for qkernel calls.

    Args:
        value (Any): Argument value supplied at a qkernel call site.
        expected_type (Any): Callee annotation used to decide whether a
            scalar literal can be wrapped as ``UInt``, ``Float``, or ``Bit``.

    Returns:
        Any: A freshly-created scalar handle when a promotion rule applies,
        otherwise ``value`` unchanged.
    """
    if isinstance(value, Handle):
        return value
    is_bool = isinstance(value, bool)
    if expected_type is UInt:
        if isinstance(value, int) and not is_bool:
            return uint(value)
    elif expected_type is Float:
        if isinstance(value, float) or (isinstance(value, int) and not is_bool):
            return float_(float(value))
    elif expected_type is Bit:
        if is_bool:
            return bit(value)
    return value


def quantum_handle_display_name(handle: Handle) -> str:
    """Return a human-readable name for a quantum handle.

    Args:
        handle (Handle): Handle to name.

    Returns:
        str: Non-empty display name for diagnostics.
    """
    return handle.name or handle.value.name or f"qubit_{handle.id[:8]}"


def reject_aliased_quantum_args(
    kernel_name: str,
    arguments: dict[str, Any],
) -> None:
    """Reject duplicate live quantum handles at a qkernel call boundary.

    Args:
        kernel_name (str): Name of the called qkernel for diagnostics.
        arguments (dict[str, Any]): Bound call arguments keyed by parameter
            name.

    Raises:
        QubitConsumedError: If two quantum arguments are backed by the same
            live IR value.
    """
    seen: dict[str, str] = {}
    for name, handle in arguments.items():
        if not isinstance(handle, Handle) or not handle._should_enforce_linear():
            continue
        backing_uuid = handle.value.uuid
        first_name = seen.get(backing_uuid)
        if first_name is not None:
            display_name = quantum_handle_display_name(handle)
            raise QubitConsumedError(
                f"Arguments '{first_name}' and '{name}' of "
                f"'QKernel[{kernel_name}]' are backed by the same qubit "
                f"register ('{display_name}').\n\n"
                f"Affine type rule: Each qubit handle can be passed to a "
                f"kernel call at most once; binding one register to two "
                f"parameters would alias both onto the same physical "
                f"qubits.\n\n"
                f"Fix: pass disjoint registers, e.g.:\n"
                f"  x, y = {kernel_name}(q[0:2], q[2:4])  # disjoint slices",
                handle_name=display_name,
                operation_name=f"QKernel[{kernel_name}]",
            )
        seen[backing_uuid] = name


def reject_consumed_view_arg(kernel_name: str, handle: Handle) -> None:
    """Reject an already-consumed vector view passed to a qkernel call.

    Args:
        kernel_name (str): Name of the called qkernel for diagnostics.
        handle (Handle): View argument to check.

    Raises:
        QubitConsumedError: If ``handle`` was already consumed.
    """
    if not handle._consumed:
        return
    display_name = quantum_handle_display_name(handle)
    first_use, reuse, consumed_at = _describe_consume_sites(
        handle, f"QKernel[{kernel_name}]"
    )
    raise QubitConsumedError(
        f"Qubit view '{display_name}' was already consumed by "
        f"{first_use} and cannot be used again in {reuse}.\n\n"
        f"Affine type rule: Each qubit handle can only be used once. "
        f"After a gate operation, reassign the result to use the new "
        f"handle.\n\n"
        f"Fix:\n"
        f"  v = qm.h(v)  # Reassign to capture the new handle\n"
        f"  {kernel_name}(v)  # Pass the reassigned handle",
        handle_name=display_name,
        operation_name=f"QKernel[{kernel_name}]",
        first_use_location=consumed_at or handle._consumed_by,
    )


def const_int(value: Value | None) -> int | None:
    """Return a compile-time integer constant from an IR value.

    Args:
        value (Value | None): IR value that may carry a constant.

    Returns:
        int | None: Plain integer constant, or ``None`` when unavailable.
    """
    if value is None:
        return None
    const = value.get_const()
    if isinstance(const, bool) or not isinstance(const, int):
        return None
    return const


def is_valid_array_extent(value: Value | None) -> bool:
    """Return whether a value is a well-formed array extent.

    Args:
        value (Value | None): Candidate scalar extent value.

    Returns:
        bool: ``True`` for a scalar ``UInt`` whose constant payload, when
        present, is a non-negative plain integer.
    """
    if (
        not isinstance(value, Value)
        or isinstance(value, ArrayValue)
        or not isinstance(value.type, UIntType)
    ):
        return False
    if not value.is_constant():
        return True
    extent = const_int(value)
    return extent is not None and extent >= 0


def array_extents_equal(left: Value, right: Value) -> bool:
    """Return whether two well-formed array extents are statically equal.

    Args:
        left (Value): First scalar ``UInt`` extent.
        right (Value): Second scalar ``UInt`` extent.

    Returns:
        bool: ``True`` for one SSA extent or equal non-negative constants.
    """
    if not is_valid_array_extent(left) or not is_valid_array_extent(right):
        return False
    if left.uuid == right.uuid:
        return True
    left_value = const_int(left)
    right_value = const_int(right)
    return left_value is not None and left_value == right_value


def _full_reslice_terminal(value: ArrayValue) -> tuple[ArrayValue, bool] | None:
    """Return the resource reached through a prefix of exact full slices.

    Args:
        value (ArrayValue): Array value whose full-slice ancestry is inspected.

    Returns:
        tuple[ArrayValue, bool] | None: Terminal array and whether at least one
            exact full slice was traversed, or ``None`` for a cyclic chain.
    """
    current = value
    saw_slice = False
    seen: set[int] = set()
    while current.slice_of is not None:
        current_id = id(current)
        if current_id in seen:
            return None
        seen.add(current_id)
        parent = current.slice_of
        if (
            current.type != parent.type
            or len(current.shape) != 1
            or len(parent.shape) != 1
            or const_int(current.slice_start) != 0
            or const_int(current.slice_step) != 1
            or not is_valid_array_extent(current.slice_start)
            or not is_valid_array_extent(current.slice_step)
            or not array_extents_equal(current.shape[0], parent.shape[0])
        ):
            break
        saw_slice = True
        current = parent
    return current, saw_slice


def array_resource_identity(value: ArrayValue) -> str | None:
    """Return the canonical logical identity of an array resource.

    Args:
        value (ArrayValue): Array whose exact full-slice prefix is ignored.

    Returns:
        str | None: Terminal logical identity, or None for a cyclic chain.
    """
    terminal = _full_reslice_terminal(value)
    return terminal[0].logical_id if terminal is not None else None


def array_resources_equal(left: ArrayValue, right: ArrayValue) -> bool:
    """Return whether arrays denote the same whole logical resource.

    Exact full re-slices are transparent, while partial or strided views are
    distinct resources. This lets control-flow merges preserve identity for a
    direct value and ``value[:]`` as well as for two sibling full re-slices.

    Args:
        left (ArrayValue): First array resource.
        right (ArrayValue): Second array resource.

    Returns:
        bool: True when both arrays reach one compatible logical resource.
    """
    if (
        left.type != right.type
        or len(left.shape) != len(right.shape)
        or len(left.shape) != 1
        or not array_extents_equal(left.shape[0], right.shape[0])
    ):
        return False
    left_terminal = _full_reslice_terminal(left)
    right_terminal = _full_reslice_terminal(right)
    if left_terminal is None or right_terminal is None:
        return False
    left_resource = left_terminal[0]
    right_resource = right_terminal[0]
    return (
        left_resource.logical_id == right_resource.logical_id
        and left_resource.type == right_resource.type
        and len(left_resource.shape) == len(right_resource.shape) == 1
        and array_extents_equal(
            left_resource.shape[0],
            right_resource.shape[0],
        )
    )


def is_full_reslice_of_input(
    output: ArrayValue,
    formal_input: ArrayValue,
) -> bool:
    """Check whether an output is only full-sliced from a formal input.

    Args:
        output (ArrayValue): Callee output array value.
        formal_input (ArrayValue): Callee formal input array value.

    Returns:
        bool: ``True`` when every slice from ``output`` back to
        ``formal_input`` is ``0:len:1`` with equal concrete lengths or the
        same symbolic length identity.
    """
    if (
        output.type != formal_input.type
        or len(output.shape) != 1
        or len(formal_input.shape) != 1
    ):
        return False
    terminal = _full_reslice_terminal(output)
    return (
        terminal is not None
        and terminal[1]
        and array_resources_equal(terminal[0], formal_input)
    )


def view_result_value_for_full_reslice(
    result_value: ArrayValue,
    input_view: Vector[Any],
) -> ArrayValue:
    """Build the caller-side array value for a full re-sliced view output.

    Args:
        result_value (ArrayValue): Caller-local output materialized from the
            callee result.
        input_view (Vector[Any]): Caller-side view argument being preserved.

    Returns:
        ArrayValue: Fresh SSA version preserving caller-side slice metadata.
    """
    source = input_view.value
    return ArrayValue(
        type=result_value.type,
        name=result_value.name,
        version=source.version + 1,
        metadata=result_value.metadata,
        logical_id=source.logical_id,
        shape=source.shape,
        slice_of=source.slice_of,
        slice_start=source.slice_start,
        slice_step=source.slice_step,
    )


def handle_types_equal(left: Any, right: Any) -> bool:
    """Compare two handle type annotations.

    Args:
        left (Any): First annotation.
        right (Any): Second annotation.

    Returns:
        bool: ``True`` when origins and generic arguments match.
    """
    left_cls = getattr(left, "__origin__", left)
    right_cls = getattr(right, "__origin__", right)
    if left_cls is not right_cls:
        return False
    return getattr(left, "__args__", ()) == getattr(right, "__args__", ())


def match_output_to_input(
    output_type: Any,
    input_types: list[Any],
    claimed: list[bool],
) -> int | None:
    """Return the first unclaimed input whose handle type matches output.

    Args:
        output_type (Any): Output annotation to match.
        input_types (list[Any]): Input annotations in positional order.
        claimed (list[bool]): Flags for already matched input positions.

    Returns:
        int | None: Matching input index, or ``None``.
    """
    for idx, input_type in enumerate(input_types):
        if claimed[idx]:
            continue
        if handle_types_equal(input_type, output_type):
            return idx
    return None


def quantum_param_names(input_types: dict[str, type]) -> set[str]:
    """Return parameter names whose frontend type is quantum.

    Args:
        input_types (dict[str, type]): QKernel input annotations keyed by
            parameter name.

    Returns:
        set[str]: Names annotated as ``Qubit`` or an array of ``Qubit``.
    """
    quantum_names: set[str] = set()
    for name, param_type in input_types.items():
        if param_type is Qubit:
            quantum_names.add(name)
        elif is_array_type(param_type):
            elem = get_array_element_type(param_type)
            if elem is Qubit:
                quantum_names.add(name)
    return quantum_names
