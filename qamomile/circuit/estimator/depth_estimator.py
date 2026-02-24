"""Circuit depth estimation for quantum circuits.

This module estimates circuit depth by analyzing operation dependencies.
Depth is expressed as SymPy expressions for parametric problem sizes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import sympy as sp
from sympy import Sum

from qamomile.circuit.frontend.handle import Handle
from qamomile.circuit.ir.operation.arithmetic_operations import (
    BinOp,
    BinOpKind,
    CompOp,
    CompOpKind,
)
from qamomile.circuit.ir.operation.call_block_ops import CallBlockOperation
from qamomile.circuit.ir.operation.composite_gate import (
    CompositeGateOperation,
    ResourceMetadata,
)
from qamomile.circuit.ir.operation.control_flow import (
    ForOperation,
    IfOperation,
    WhileOperation,
)
from qamomile.circuit.ir.operation.gate import (
    ControlledUOperation,
    GateOperation,
)
from qamomile.circuit.ir.operation.operation import Operation
from qamomile.circuit.ir.value import Value

if TYPE_CHECKING:
    from qamomile.circuit.ir.block import Block
    from qamomile.circuit.ir.block_value import BlockValue


# Loop context: [(var_name, start, stop, step), ...]
LoopContext = list[tuple[str, sp.Expr, sp.Expr, sp.Expr]]


@dataclass
class CircuitDepth:
    """Circuit depth breakdown for a quantum circuit.

    All depths are SymPy expressions that may contain symbols
    for parametric problem sizes.

    Attributes:
        total_depth: Total circuit depth (all gates)
        t_depth: Depth of T gates only (critical for fault tolerance)
        two_qubit_depth: Depth of two-qubit gates (often the bottleneck)
    """

    total_depth: sp.Expr
    t_depth: sp.Expr
    two_qubit_depth: sp.Expr
    multi_qubit_depth: sp.Expr
    rotation_depth: sp.Expr

    def __add__(self, other: CircuitDepth) -> CircuitDepth:
        """Add two circuit depths (sequential composition)."""
        return CircuitDepth(
            total_depth=self.total_depth + other.total_depth,
            t_depth=self.t_depth + other.t_depth,
            two_qubit_depth=self.two_qubit_depth + other.two_qubit_depth,
            multi_qubit_depth=self.multi_qubit_depth + other.multi_qubit_depth,
            rotation_depth=self.rotation_depth + other.rotation_depth,
        )

    def __mul__(self, factor: sp.Expr | int) -> CircuitDepth:
        """Multiply circuit depth by a factor."""
        factor_expr = sp.Integer(factor) if isinstance(factor, int) else factor
        return CircuitDepth(
            total_depth=self.total_depth * factor_expr,
            t_depth=self.t_depth * factor_expr,
            two_qubit_depth=self.two_qubit_depth * factor_expr,
            multi_qubit_depth=self.multi_qubit_depth * factor_expr,
            rotation_depth=self.rotation_depth * factor_expr,
        )

    __rmul__ = __mul__

    def max(self, other: CircuitDepth) -> CircuitDepth:
        """Element-wise maximum of two circuit depths (parallel composition)."""
        return CircuitDepth(
            total_depth=sp.Max(self.total_depth, other.total_depth),
            t_depth=sp.Max(self.t_depth, other.t_depth),
            two_qubit_depth=sp.Max(self.two_qubit_depth, other.two_qubit_depth),
            multi_qubit_depth=sp.Max(self.multi_qubit_depth, other.multi_qubit_depth),
            rotation_depth=sp.Max(self.rotation_depth, other.rotation_depth),
        )

    def simplify(self) -> CircuitDepth:
        """Simplify all SymPy expressions."""
        return CircuitDepth(
            total_depth=sp.simplify(self.total_depth),
            t_depth=sp.simplify(self.t_depth),
            two_qubit_depth=sp.simplify(self.two_qubit_depth),
            multi_qubit_depth=sp.simplify(self.multi_qubit_depth),
            rotation_depth=sp.simplify(self.rotation_depth),
        )

    @staticmethod
    def zero() -> CircuitDepth:
        """Return a zero circuit depth."""
        return CircuitDepth(
            total_depth=sp.Integer(0),
            t_depth=sp.Integer(0),
            two_qubit_depth=sp.Integer(0),
            multi_qubit_depth=sp.Integer(0),
            rotation_depth=sp.Integer(0),
        )


# Gate categorization
T_GATES = {"t", "tdg"}
TWO_QUBIT_GATES = {"cx", "cy", "cz", "swap", "cp", "crx", "cry", "crz", "rzz"}
ROTATION_GATES = {"rx", "ry", "rz", "p", "cp", "crx", "cry", "crz", "rzz"}


SINGLE_QUBIT_GATES = {
    "h",
    "x",
    "y",
    "z",
    "s",
    "sdg",
    "t",
    "tdg",
    "rx",
    "ry",
    "rz",
    "p",
    "u",
    "u1",
    "u2",
    "u3",
}


def _estimate_gate_depth(
    op: GateOperation, is_controlled: bool = False
) -> CircuitDepth:
    """Estimate depth for a single gate operation.

    Args:
        op: The gate operation
        is_controlled: Whether this gate is inside a ControlledUOperation

    Returns:
        Circuit depths for this operation
    """
    # Get gate name from enum
    gate_name = op.gate_type.name.lower() if op.gate_type else "unknown"

    if is_controlled:
        is_single_qubit_base = gate_name in SINGLE_QUBIT_GATES
        is_two_qubit_base = gate_name in TWO_QUBIT_GATES

        # Controlled T/Tdg are 2q gates, not simple T gates
        t_depth = sp.Integer(0)
        two_qubit_depth = sp.Integer(1) if is_single_qubit_base else sp.Integer(0)
        multi_qubit_depth = sp.Integer(1) if is_two_qubit_base else sp.Integer(0)
        rotation_depth = sp.Integer(1) if gate_name in ROTATION_GATES else sp.Integer(0)
    else:
        is_t_gate = gate_name in T_GATES
        is_two_qubit = gate_name in TWO_QUBIT_GATES
        is_rotation = gate_name in ROTATION_GATES

        t_depth = sp.Integer(1) if is_t_gate else sp.Integer(0)
        two_qubit_depth = sp.Integer(1) if is_two_qubit else sp.Integer(0)
        multi_qubit_depth = sp.Integer(0)
        rotation_depth = sp.Integer(1) if is_rotation else sp.Integer(0)

    return CircuitDepth(
        total_depth=sp.Integer(1),
        t_depth=t_depth,
        two_qubit_depth=two_qubit_depth,
        multi_qubit_depth=multi_qubit_depth,
        rotation_depth=rotation_depth,
    )


def _extract_depth_from_metadata(meta: ResourceMetadata) -> CircuitDepth:
    """Extract CircuitDepth from ResourceMetadata.

    Args:
        meta: ResourceMetadata containing depth information

    Returns:
        CircuitDepth extracted from metadata
    """
    custom = meta.custom_metadata

    # Extract depth
    total_depth = custom.get("depth", custom.get("total_depth", 0))

    # Extract T-depth
    t_depth = custom.get("t_depth", 0)

    # Extract two-qubit depth
    two_qubit_depth = custom.get("two_qubit_depth", 0)

    # Extract rotation depth
    rotation_depth = custom.get("rotation_depth", 0)

    return CircuitDepth(
        total_depth=sp.Integer(total_depth),
        t_depth=sp.Integer(t_depth),
        two_qubit_depth=sp.Integer(two_qubit_depth),
        multi_qubit_depth=sp.Integer(0),
        rotation_depth=sp.Integer(rotation_depth),
    )


def _estimate_composite_gate_depth(
    op: CompositeGateOperation,
    block: Block | None,
    loop_context: LoopContext,
    call_context: dict[str, Any],
    loop_var_symbols: dict[str, sp.Symbol],
    is_controlled: bool = False,
) -> CircuitDepth:
    """Estimate depth of a CompositeGateOperation.

    Priority order:
    1. Use resource_metadata if available
    2. Use implementation (decomposition) if available
    3. Compute from known gate types (QFT, IQFT, QPE)
    4. Raise error if none available

    Args:
        op: CompositeGateOperation to estimate
        block: Current block for context
        loop_context: Loop context for symbolic evaluation
        call_context: Call context for parameter resolution
        loop_var_symbols: Loop variable symbols

    Returns:
        CircuitDepth for the composite gate

    Raises:
        ValueError: If no resource information or implementation available
    """
    from qamomile.circuit.ir.block_value import BlockValue
    from qamomile.circuit.ir.operation.composite_gate import CompositeGateType

    # Priority 1: Use resource_metadata
    if op.resource_metadata is not None:
        return _extract_depth_from_metadata(op.resource_metadata)

    # Priority 2: Use implementation (decomposition)
    if op.has_implementation and op.implementation is not None:
        impl_block = op.implementation
        if isinstance(impl_block, BlockValue):
            # Build call context for implementation parameters
            new_call_context = call_context.copy()

            # Map operands to implementation inputs
            for formal_idx, formal_input in enumerate(impl_block.input_values):
                if formal_idx + 1 < len(op.operands):
                    actual_arg = op.operands[formal_idx + 1]
                    resolved_arg = value_to_expr(
                        actual_arg, block, call_context, loop_var_symbols
                    )
                    new_call_context[formal_input.uuid] = resolved_arg

            # Recursively estimate depth in implementation
            return _estimate_sequential_depth(
                impl_block.operations,
                block=impl_block,
                loop_context=loop_context,
                call_context=new_call_context,
                loop_var_symbols=loop_var_symbols,
                is_controlled=is_controlled,
            )

    # Priority 3: Compute from known gate types
    if op.gate_type in (CompositeGateType.QFT, CompositeGateType.IQFT):
        # QFT/IQFT: Approximate depth is O(n)
        # Determine n from the number of target qubit operands
        target_qubits = op.target_qubits
        n_qubits = len(target_qubits)

        if n_qubits > 0:
            # We have concrete qubits, so use the count
            n = sp.Integer(n_qubits)
        else:
            # No operands - infer size from context (same as gate_counter)
            n = None

            if block is not None:
                # Look for QInitOperation that creates qubit arrays
                from qamomile.circuit.ir.operation.operation import QInitOperation
                from qamomile.circuit.ir.value import ArrayValue

                for block_op in block.operations:
                    if isinstance(block_op, QInitOperation):
                        if block_op.results and isinstance(
                            block_op.results[0], ArrayValue
                        ):
                            array = block_op.results[0]
                            if (
                                array.shape and array.name == "counting"
                            ):  # QPE uses "counting" array
                                n = value_to_expr(
                                    array.shape[0],
                                    block,
                                    call_context,
                                    loop_var_symbols,
                                )
                                break

            # Fallback: try num_target_qubits field
            if (
                n is None
                and isinstance(op.num_target_qubits, int)
                and op.num_target_qubits > 0
            ):
                n = sp.Integer(op.num_target_qubits)

            # Last resort: symbolic n
            if n is None:
                n = sp.Symbol("n", integer=True, positive=True)

        return CircuitDepth(
            total_depth=n,
            t_depth=sp.Integer(0),
            two_qubit_depth=n * (n - 1) / 2,  # Sequential estimate
            multi_qubit_depth=sp.Integer(0),
            rotation_depth=n * (n - 1) / 2,  # CP gates are rotation gates
        )

    # Priority 4: Error if no resource info or implementation
    gate_name = op.custom_name or op.gate_type.value
    raise ValueError(
        f"Cannot estimate depth for CompositeGateOperation '{gate_name}': "
        f"No resource_metadata or implementation available."
    )


def value_to_expr(
    v: Any,
    block: Block | None = None,
    call_context: dict[str, Any] | None = None,
    loop_var_symbols: dict[str, sp.Symbol] | None = None,
) -> sp.Expr:
    """Convert Value to SymPy expression, tracing operations if needed.

    Args:
        v: Value or constant to convert
        block: Optional Block for operation tracing
        call_context: Optional mapping from Value UUIDs to actual argument Values or SymPy expressions
        loop_var_symbols: Optional mapping from Value names to SymPy loop variable symbols

    Returns:
        SymPy expression representing the value
    """
    from qamomile.circuit.ir.value import Value

    # If already a SymPy expression, return as-is
    if isinstance(v, sp.Basic):
        return v

    if not isinstance(v, Value):
        if isinstance(v, int):
            return sp.Integer(v)
        elif isinstance(v, float):
            return sp.Float(v)
        elif isinstance(v, bool):
            return sp.Integer(1 if v else 0)
        else:
            return sp.Symbol(str(v), integer=True, positive=True)

    # Check if this value is mapped in call_context
    if call_context and v.uuid in call_context:
        resolved_val = call_context[v.uuid]
        return value_to_expr(resolved_val, block, call_context, loop_var_symbols)

    # Use proper Value API
    if v.is_constant():
        const_val = v.get_const()
        if isinstance(const_val, bool):
            return sp.Integer(1 if const_val else 0)
        elif isinstance(const_val, float):
            return sp.Float(const_val)
        else:
            return sp.Integer(int(const_val))
    elif v.is_parameter():
        return sp.Symbol(v.parameter_name(), integer=True, positive=True)
    else:
        # Check if this is a loop variable
        if loop_var_symbols and v.name in loop_var_symbols:
            return loop_var_symbols[v.name]

        # Computed value - trace if block provided
        if block is not None:
            traced = _trace_value_operation(
                v,
                block,
                visited=set(),
                call_context=call_context,
                loop_var_symbols=loop_var_symbols,
            )
            if traced is not None:
                return traced
        return sp.Symbol(v.name, integer=True, positive=True)  # Fallback


def _trace_value_operation(
    v: Value,
    block: Block,
    visited: set,
    call_context: dict[str, Any] | None = None,
    loop_var_symbols: dict[str, sp.Symbol] | None = None,
) -> sp.Expr | None:
    """Trace backward through operations to find what produced this Value.

    Args:
        v: Value to trace
        block: Block containing the operations
        visited: Set of visited Value IDs to prevent infinite recursion
        call_context: Optional mapping from Value UUIDs to actual argument Values
        loop_var_symbols: Optional mapping from Value names to SymPy loop variable symbols

    Returns:
        SymPy expression if successfully traced, None otherwise
    """
    if id(v) in visited:
        return None
    visited.add(id(v))

    for op in block.operations:
        if not hasattr(op, "results") or not op.results:
            continue

        if op.results[0] == v:
            if isinstance(op, BinOp):
                left = value_to_expr(
                    op.operands[0], block, call_context, loop_var_symbols
                )
                right = value_to_expr(
                    op.operands[1], block, call_context, loop_var_symbols
                )

                if op.kind == BinOpKind.ADD:
                    return left + right
                elif op.kind == BinOpKind.SUB:
                    return left - right
                elif op.kind == BinOpKind.MUL:
                    return left * right
                elif op.kind == BinOpKind.DIV:
                    return left / right
                elif op.kind == BinOpKind.FLOORDIV:
                    return sp.floor(left / right)
                elif op.kind == BinOpKind.POW:
                    return left**right

            elif isinstance(op, CompOp):
                left = value_to_expr(
                    op.operands[0], block, call_context, loop_var_symbols
                )
                right = value_to_expr(
                    op.operands[1], block, call_context, loop_var_symbols
                )

                if op.kind == CompOpKind.EQ:
                    return sp.Eq(left, right)
                elif op.kind == CompOpKind.NEQ:
                    return sp.Ne(left, right)
                elif op.kind == CompOpKind.LT:
                    return sp.Lt(left, right)
                elif op.kind == CompOpKind.LE:
                    return sp.Le(left, right)
                elif op.kind == CompOpKind.GT:
                    return sp.Gt(left, right)
                elif op.kind == CompOpKind.GE:
                    return sp.Ge(left, right)

    return None


def _find_loop_variable_values(operations: list[Operation], loop_var_name: str) -> list:
    """Find Values that directly represent the loop variable (not derived expressions).

    Only finds Values whose name exactly matches the loop variable name.
    Does NOT include computed expressions like 2**i or i+1.

    Args:
        operations: List of operations to search
        loop_var_name: Name of the loop variable (e.g., "i", "j")

    Returns:
        List of Value objects that directly represent this loop variable
    """
    from qamomile.circuit.ir.value import Value

    result = []
    seen_uuids = set()

    def add_value(v: Value):
        if v.uuid not in seen_uuids:
            seen_uuids.add(v.uuid)
            result.append(v)

    def check_operation(op: Operation):
        if hasattr(op, "operands"):
            for operand in op.operands:
                if isinstance(operand, Value):
                    # Only match exact loop variable name
                    if operand.name == loop_var_name:
                        add_value(operand)

    for op in operations:
        check_operation(op)
        # Recursively check nested operations
        if isinstance(op, ForOperation):
            for inner_op in op.operations:
                check_operation(inner_op)
        elif isinstance(op, (IfOperation, WhileOperation)):
            if hasattr(op, "operations"):
                for inner_op in op.operations:
                    check_operation(inner_op)

    return result


def _apply_sum_to_depth(
    depth: CircuitDepth, loop_var: sp.Symbol, start: sp.Expr, stop: sp.Expr
) -> CircuitDepth:
    """Apply SymPy Sum to all fields of a CircuitDepth."""
    return CircuitDepth(
        total_depth=Sum(depth.total_depth, (loop_var, start, stop - 1)).doit(),
        t_depth=Sum(depth.t_depth, (loop_var, start, stop - 1)).doit(),
        two_qubit_depth=Sum(depth.two_qubit_depth, (loop_var, start, stop - 1)).doit(),
        multi_qubit_depth=Sum(
            depth.multi_qubit_depth, (loop_var, start, stop - 1)
        ).doit(),
        rotation_depth=Sum(depth.rotation_depth, (loop_var, start, stop - 1)).doit(),
    )


def _resolve_power_expr(
    power: Any,
    block: Block | None,
    call_context: dict[str, Any] | None,
    loop_var_symbols: dict[str, sp.Symbol] | None,
) -> sp.Expr | int:
    """Resolve power value to SymPy expression or int.

    Args:
        power: Power value (int, SymPy expression, Handle, or Value).
        block: Current block for context.
        call_context: Call context for parameter resolution.
        loop_var_symbols: Loop variable symbols.

    Returns:
        SymPy expression or int representing the power.
    """
    if isinstance(power, (int, sp.Basic)):
        return power

    # Handle is a frontend handle - convert via its value
    if isinstance(power, Handle):
        return value_to_expr(power.value, block, call_context, loop_var_symbols)

    # Value object - convert directly
    if hasattr(power, "value"):
        return value_to_expr(power, block, call_context, loop_var_symbols)

    return power


def _estimate_sequential_depth(
    operations: list[Operation],
    block: Block | None = None,
    loop_context: LoopContext | None = None,
    call_context: dict[str, Any] | None = None,
    loop_var_symbols: dict[str, sp.Symbol] | None = None,
    is_controlled: bool = False,
) -> CircuitDepth:
    """Estimate depth assuming sequential execution.

    This is a conservative estimate that assumes no parallelization.
    For actual parallel depth, would need full dependency analysis.

    Args:
        operations: List of operations
        block: Optional Block for value tracing
        loop_context: Stack of outer loop variables for nested loop analysis
        call_context: Optional mapping from Value UUIDs to actual argument Values
        loop_var_symbols: Optional mapping from Value names to SymPy loop variable symbols

    Returns:
        Sequential circuit depth
    """
    if loop_context is None:
        loop_context = []
    if call_context is None:
        call_context = {}
    if loop_var_symbols is None:
        loop_var_symbols = {}

    depth = CircuitDepth.zero()

    for op in operations:
        match op:
            case GateOperation():
                depth = depth + _estimate_gate_depth(op, is_controlled=is_controlled)

            case ForOperation():
                # Get loop bounds: for i in range(start, stop, step)
                if len(op.operands) >= 2:
                    start = op.operands[0]
                    stop = op.operands[1]
                    step = op.operands[2] if len(op.operands) >= 3 else None

                    # Build loop variable symbols mapping FIRST (needed for tracing loop bounds)
                    new_loop_var_symbols = loop_var_symbols.copy()
                    loop_var_symbol = sp.Symbol(
                        op.loop_var, integer=True, positive=True
                    )

                    # Find all Values that represent this loop variable
                    loop_var_values = _find_loop_variable_values(
                        op.operations, op.loop_var
                    )
                    for lv in loop_var_values:
                        new_loop_var_symbols[lv.name] = loop_var_symbol

                    # Also map the loop variable name itself
                    new_loop_var_symbols[op.loop_var] = loop_var_symbol

                    # Create a temporary "local block" to trace values defined in this loop's operations
                    local_block = type(
                        "LocalBlock", (), {"operations": op.operations}
                    )()

                    # Convert values to SymPy expressions, tracing in local block if value not in outer block
                    # Try outer block first, then local block
                    def convert_with_local_trace(v):
                        # First try with outer block
                        if block is not None:
                            result = value_to_expr(
                                v, block, call_context, new_loop_var_symbols
                            )
                            # If we got a symbol, try local block too
                            from qamomile.circuit.ir.value import Value

                            if (
                                isinstance(v, Value)
                                and not v.is_constant()
                                and not v.is_parameter()
                            ):
                                local_result = value_to_expr(
                                    v, local_block, call_context, new_loop_var_symbols
                                )
                                # Use local result if it's not just the symbol name
                                if local_result != sp.Symbol(
                                    v.name, integer=True, positive=True
                                ):
                                    return local_result
                            return result
                        else:
                            return value_to_expr(
                                v, local_block, call_context, new_loop_var_symbols
                            )

                    start_expr = convert_with_local_trace(start)
                    stop_expr = convert_with_local_trace(stop)
                    step_expr = (
                        convert_with_local_trace(step) if step else sp.Integer(1)
                    )

                    # Add current loop to context for inner analysis
                    new_context = loop_context + [
                        (op.loop_var, start_expr, stop_expr, step_expr)
                    ]

                    # Recursively estimate inner operations with enhanced context
                    # Pass local_block so inner operations can trace values defined in this scope
                    inner_depth = _estimate_sequential_depth(
                        op.operations,
                        block=local_block,  # Use local block for tracing
                        loop_context=new_context,
                        call_context=call_context,
                        loop_var_symbols=new_loop_var_symbols,
                        is_controlled=is_controlled,
                    )

                    # Check if inner depth contains the current loop variable
                    # This happens when inner loops have bounds that depend on this variable
                    if loop_var_symbol in inner_depth.total_depth.free_symbols:
                        # The inner depth varies with our loop variable
                        # We need to sum it over all loop iterations
                        # Sum(inner_depth, (loop_var, start, stop-1))
                        depth_expr = _apply_sum_to_depth(
                            inner_depth, loop_var_symbol, start_expr, stop_expr
                        )
                    else:
                        # Normal case: inner depth is constant per iteration
                        # Multiply by number of iterations
                        iterations = (stop_expr - start_expr) / step_expr
                        depth_expr = inner_depth * iterations

                    depth = depth + depth_expr
                else:
                    # Fallback
                    iterations = sp.Symbol("iter", integer=True, positive=True)
                    inner_depth = _estimate_sequential_depth(
                        op.operations,
                        block=None,
                        loop_context=loop_context,
                        call_context=call_context,
                        loop_var_symbols=loop_var_symbols,
                        is_controlled=is_controlled,
                    )
                    depth = depth + (inner_depth * iterations)

            case WhileOperation():
                # Conservative: assume constant iterations
                inner_depth = _estimate_sequential_depth(
                    op.operations,
                    block=None,
                    loop_context=loop_context,
                    call_context=call_context,
                    loop_var_symbols=loop_var_symbols,
                    is_controlled=is_controlled,
                )
                iterations = sp.Symbol("|while|", integer=True, positive=True)
                depth = depth + (inner_depth * iterations)

            case IfOperation():
                # Take maximum of both branches
                true_depth = _estimate_sequential_depth(
                    op.true_operations,
                    block=None,
                    loop_context=loop_context,
                    call_context=call_context,
                    loop_var_symbols=loop_var_symbols,
                    is_controlled=is_controlled,
                )
                false_depth = _estimate_sequential_depth(
                    op.false_operations,
                    block=None,
                    loop_context=loop_context,
                    call_context=call_context,
                    loop_var_symbols=loop_var_symbols,
                    is_controlled=is_controlled,
                )
                depth = depth + true_depth.max(false_depth)

            case CallBlockOperation():
                # Recursively estimate depth in called block
                from qamomile.circuit.ir.block_value import BlockValue
                from qamomile.circuit.ir.value import ArrayValue

                called_block = op.operands[0]
                if isinstance(called_block, BlockValue):
                    # Build call context: map formal parameters to actual arguments
                    new_call_context = call_context.copy()

                    for formal_idx, formal_input in enumerate(
                        called_block.input_values
                    ):
                        if formal_idx + 1 < len(op.operands):
                            actual_arg = op.operands[formal_idx + 1]

                            # Resolve actual_arg to SymPy expression in the OUTER block
                            # so that computed values (like 2**i) can be traced
                            resolved_arg = value_to_expr(
                                actual_arg, block, call_context, loop_var_symbols
                            )
                            new_call_context[formal_input.uuid] = resolved_arg

                            # If both are arrays, also map dimensions
                            if isinstance(actual_arg, ArrayValue) and isinstance(
                                formal_input, ArrayValue
                            ):
                                for dim_formal, dim_actual in zip(
                                    formal_input.shape, actual_arg.shape
                                ):
                                    resolved_dim = value_to_expr(
                                        dim_actual,
                                        block,
                                        call_context,
                                        loop_var_symbols,
                                    )
                                    new_call_context[dim_formal.uuid] = resolved_dim

                    # Recursively estimate with context
                    inner_depth = _estimate_sequential_depth(
                        called_block.operations,
                        block=called_block,
                        loop_context=loop_context,
                        call_context=new_call_context,
                        loop_var_symbols=loop_var_symbols,
                        is_controlled=is_controlled,
                    )
                    depth = depth + inner_depth

            case ControlledUOperation():
                # Recursively estimate depth in the unitary block
                from qamomile.circuit.ir.block_value import BlockValue
                from qamomile.circuit.ir.value import ArrayValue

                controlled_block = op.block
                if isinstance(controlled_block, BlockValue):
                    # Build call context: map formal parameters to actual arguments
                    new_call_context = call_context.copy()

                    # Get target operands (skip BlockValue and control qubits)
                    target_operands = op.target_operands

                    # Map each formal input to corresponding target operand
                    for formal_idx, formal_input in enumerate(
                        controlled_block.input_values
                    ):
                        if formal_idx < len(target_operands):
                            actual_arg = target_operands[formal_idx]

                            # Resolve actual_arg to SymPy expression in the OUTER block
                            # so that computed values (like 2**i) can be traced
                            resolved_arg = value_to_expr(
                                actual_arg, block, call_context, loop_var_symbols
                            )
                            new_call_context[formal_input.uuid] = resolved_arg

                            # If both are arrays, also map dimensions
                            if isinstance(actual_arg, ArrayValue) and isinstance(
                                formal_input, ArrayValue
                            ):
                                for dim_formal, dim_actual in zip(
                                    formal_input.shape, actual_arg.shape
                                ):
                                    resolved_dim = value_to_expr(
                                        dim_actual,
                                        block,
                                        call_context,
                                        loop_var_symbols,
                                    )
                                    new_call_context[dim_formal.uuid] = resolved_dim

                    # Recursively estimate with preserved context
                    # is_controlled=True promotes depth classifications
                    inner_depth = _estimate_sequential_depth(
                        controlled_block.operations,
                        block=controlled_block,
                        loop_context=loop_context,
                        call_context=new_call_context,
                        loop_var_symbols=loop_var_symbols,
                        is_controlled=True,
                    )

                    # Multiply by power if U^k is applied
                    power_expr = _resolve_power_expr(
                        op.power, block, call_context, loop_var_symbols
                    )
                    if power_expr != 1:
                        inner_depth = inner_depth * power_expr

                    depth = depth + inner_depth

            case CompositeGateOperation():
                composite_depth = _estimate_composite_gate_depth(
                    op,
                    block,
                    loop_context,
                    call_context,
                    loop_var_symbols,
                    is_controlled=is_controlled,
                )
                depth = depth + composite_depth

            case _:
                continue

    return depth.simplify()


def estimate_depth(block: BlockValue | Block | list[Operation]) -> CircuitDepth:
    """Estimate circuit depth.

    This function provides a conservative (upper bound) estimate of circuit depth
    by assuming sequential execution. For actual parallelizable depth, a full
    dependency analysis would be required.

    Depth is expressed using SymPy expressions and may contain symbols for
    parametric problem sizes (e.g., loop bounds).

    Supports:
    - GateOperation: Contributes depth 1
    - ForOperation: Multiplies inner depth by iterations
    - IfOperation: Takes maximum of branches
    - CallBlockOperation: Recursively estimates called blocks
    - ControlledUOperation: Recursively estimates unitary block

    Args:
        block: BlockValue, Block, or list of Operations to analyze

    Returns:
        CircuitDepth with total_depth, t_depth, two_qubit_depth

    Example:
        >>> from qamomile.circuit.estimator import estimate_depth
        >>> depth = estimate_depth(my_circuit.block)
        >>> print(depth.total_depth)  # e.g., "n + 5"
        >>> print(depth.t_depth)  # e.g., "n"

    Note:
        This is a sequential depth estimate (upper bound). Actual depth
        may be lower due to gate parallelization, which requires full
        dependency graph analysis.
    """
    from qamomile.circuit.ir.block import Block
    from qamomile.circuit.ir.block_value import BlockValue

    # Extract Block reference and operations
    block_ref = None
    if isinstance(block, (BlockValue, Block)):
        # Both BlockValue and Block have operations attribute
        # Pass either directly as they both have the needed structure
        block_ref = block
        ops = block.operations
    else:
        ops = block

    return _estimate_sequential_depth(
        ops, block_ref, loop_context=[], call_context={}, loop_var_symbols={}
    )
