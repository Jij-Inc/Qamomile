"""Circuit depth estimation for quantum circuits.

This module estimates circuit depth by analyzing operation dependencies.
Depth is expressed as SymPy expressions for parametric problem sizes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import sympy as sp
from sympy import Sum

from qamomile.circuit.ir.operation.arithmetic_operations import (
    BinOp,
    BinOpKind,
    CompOp,
    CompOpKind,
    CondOp,
    NotOp,
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
    MeasureOperation,
    MeasureVectorOperation,
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

    def substitute(
        self, subs_dict: dict[sp.Symbol, int | sp.Expr]
    ) -> CircuitDepth:
        """Substitute symbols with concrete values in all fields."""
        return CircuitDepth(
            total_depth=self.total_depth.subs(subs_dict),
            t_depth=self.t_depth.subs(subs_dict),
            two_qubit_depth=self.two_qubit_depth.subs(subs_dict),
            multi_qubit_depth=self.multi_qubit_depth.subs(subs_dict),
            rotation_depth=self.rotation_depth.subs(subs_dict),
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
MULTI_QUBIT_GATES = {"toffoli"}
_GATE_BASE_QUBITS: dict[str, int] = {"toffoli": 3}


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
    op: GateOperation, num_controls: int | sp.Expr = 0
) -> CircuitDepth:
    """Estimate depth for a single gate operation.

    Args:
        op: The gate operation
        num_controls: Number of control qubits (0 means no control)

    Returns:
        Circuit depths for this operation
    """
    # Get gate name from enum
    gate_name = op.gate_type.name.lower() if op.gate_type else "unknown"

    # Safe symbolic check: "has controls?"
    _has_controls = not (isinstance(num_controls, int) and num_controls == 0)

    if _has_controls:
        is_single_qubit_base = gate_name in SINGLE_QUBIT_GATES
        is_two_qubit_base = gate_name in TWO_QUBIT_GATES

        # Controlled T/Tdg are 2q gates, not simple T gates
        t_depth = sp.Integer(0)

        if isinstance(num_controls, int):
            # Concrete num_controls: classify based on total qubit count
            if is_single_qubit_base:
                total_qubits = num_controls + 1
            elif is_two_qubit_base:
                total_qubits = num_controls + 2
            else:
                total_qubits = num_controls + _GATE_BASE_QUBITS.get(gate_name, 1)
            two_qubit_depth = sp.Integer(1) if total_qubits == 2 else sp.Integer(0)
            multi_qubit_depth = sp.Integer(1) if total_qubits > 2 else sp.Integer(0)
        else:
            # Symbolic num_controls: use Piecewise so subs() resolves correctly
            if is_single_qubit_base:
                total_qubits = num_controls + 1
            elif is_two_qubit_base:
                total_qubits = num_controls + 2
            else:
                total_qubits = num_controls + _GATE_BASE_QUBITS.get(gate_name, 1)
            is_multi_qubit_base = gate_name in MULTI_QUBIT_GATES
            if is_single_qubit_base or is_two_qubit_base or is_multi_qubit_base:
                two_qubit_depth = sp.Piecewise(
                    (sp.Integer(1), sp.Eq(total_qubits, 2)),
                    (sp.Integer(0), True),
                )
                multi_qubit_depth = sp.Piecewise(
                    (sp.Integer(1), total_qubits > 2),
                    (sp.Integer(0), True),
                )
            else:
                two_qubit_depth = sp.Integer(0)
                multi_qubit_depth = sp.Integer(0)

        rotation_depth = sp.Integer(1) if gate_name in ROTATION_GATES else sp.Integer(0)
    else:
        is_t_gate = gate_name in T_GATES
        is_two_qubit = gate_name in TWO_QUBIT_GATES
        is_multi_qubit = gate_name in MULTI_QUBIT_GATES
        is_rotation = gate_name in ROTATION_GATES

        t_depth = sp.Integer(1) if is_t_gate else sp.Integer(0)
        two_qubit_depth = sp.Integer(1) if is_two_qubit else sp.Integer(0)
        multi_qubit_depth = sp.Integer(1) if is_multi_qubit else sp.Integer(0)
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
    num_controls: int | sp.Expr = 0,
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
        depth = _extract_depth_from_metadata(op.resource_metadata)
        # Controlled stubs with no explicit depth: minimum 1 two-qubit gate layer
        if (
            op.num_control_qubits > 0
            and depth.total_depth == 0
            and depth.two_qubit_depth == 0
        ):
            depth = CircuitDepth(
                total_depth=sp.Integer(1),
                t_depth=depth.t_depth,
                two_qubit_depth=sp.Integer(1),
                multi_qubit_depth=depth.multi_qubit_depth,
                rotation_depth=depth.rotation_depth,
            )
        return depth

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
            return _compute_sequential_depth(
                impl_block.operations,
                block=impl_block,
                loop_context=loop_context,
                call_context=new_call_context,
                loop_var_symbols=loop_var_symbols,
                num_controls=num_controls,
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
    depth: CircuitDepth,
    loop_var: sp.Symbol,
    start: sp.Expr,
    stop: sp.Expr,
    step: sp.Expr = sp.Integer(1),
) -> CircuitDepth:
    """Apply SymPy Sum to all fields of a CircuitDepth.

    For forward loops (step > 0): Sum over (loop_var, start, stop-1)
    For reverse loops (step < 0): Sum over (loop_var, stop+1, start)
    """
    is_negative_step = False
    try:
        is_negative_step = bool(step < 0)
    except TypeError:
        pass  # symbolic step, assume positive

    if is_negative_step:
        lower = stop + 1
        upper = start
    else:
        lower = start
        upper = stop - 1

    return CircuitDepth(
        total_depth=Sum(depth.total_depth, (loop_var, lower, upper)).doit(),
        t_depth=Sum(depth.t_depth, (loop_var, lower, upper)).doit(),
        two_qubit_depth=Sum(depth.two_qubit_depth, (loop_var, lower, upper)).doit(),
        multi_qubit_depth=Sum(
            depth.multi_qubit_depth, (loop_var, lower, upper)
        ).doit(),
        rotation_depth=Sum(depth.rotation_depth, (loop_var, lower, upper)).doit(),
    )



# Type alias for per-qubit depth tracking
QubitDepthMap = dict[str, CircuitDepth]


def _get_max_depth(qubit_depths: QubitDepthMap) -> CircuitDepth:
    """Get maximum depth across all qubits."""
    if not qubit_depths:
        return CircuitDepth.zero()
    depths = iter(qubit_depths.values())
    result = next(depths)
    for depth in depths:
        result = result.max(depth)
    return result


def _collect_gate_qubit_names(operations: list[Operation]) -> set[str]:
    """Collect qubit names from GateOperation operands in operation list."""
    names: set[str] = set()
    for op in operations:
        if isinstance(op, GateOperation):
            for v in op.operands:
                names.add(v.name)
        elif isinstance(op, ForOperation):
            names.update(_collect_gate_qubit_names(op.operations))
        elif isinstance(op, IfOperation):
            names.update(_collect_gate_qubit_names(op.true_operations))
            names.update(_collect_gate_qubit_names(op.false_operations))
    return names


def _collect_all_qubit_names(operations: list[Operation]) -> set[str]:
    """Collect qubit names from all operation types (for IfOperation branches)."""
    names: set[str] = set()
    for op in operations:
        if isinstance(op, GateOperation):
            for v in op.operands:
                names.add(v.name)
        elif isinstance(op, CallBlockOperation):
            for v in op.operands[1:]:
                if (
                    hasattr(v, "type")
                    and hasattr(v.type, "is_quantum")
                    and v.type.is_quantum()
                ):
                    names.add(v.name)
        elif isinstance(op, ControlledUOperation):
            for v in op.operands[1:]:
                if hasattr(v, "name"):
                    names.add(v.name)
        elif isinstance(op, CompositeGateOperation):
            for v in op.operands:
                if (
                    hasattr(v, "name")
                    and hasattr(v, "type")
                    and v.type.is_quantum()
                ):
                    names.add(v.name)
        elif isinstance(op, MeasureOperation):
            names.add(op.operands[0].name)
        elif isinstance(op, MeasureVectorOperation):
            names.add(op.operands[0].name)
        elif isinstance(op, ForOperation):
            names.update(_collect_all_qubit_names(op.operations))
        elif isinstance(op, WhileOperation):
            names.update(_collect_all_qubit_names(op.operations))
        elif isinstance(op, IfOperation):
            names.update(_collect_all_qubit_names(op.true_operations))
            names.update(_collect_all_qubit_names(op.false_operations))
    return names


def _build_qubit_name_map(
    called_block: BlockValue,
    op_operands: list,
    offset: int = 1,
) -> dict[str, str]:
    """Build formal_name -> actual_name mapping for qubit arguments."""
    qubit_name_map: dict[str, str] = {}
    for formal_idx, formal_input in enumerate(called_block.input_values):
        if formal_idx + offset < len(op_operands):
            actual_arg = op_operands[formal_idx + offset]
            if (
                hasattr(formal_input, "type")
                and hasattr(formal_input.type, "is_quantum")
                and formal_input.type.is_quantum()
            ):
                qubit_name_map[formal_input.name] = actual_arg.name
    return qubit_name_map


def _map_depths_to_local(
    qubit_name_map: dict[str, str],
    qubit_depths: QubitDepthMap,
) -> QubitDepthMap:
    """Create local depth map with formal names from actual depths."""
    local_depths: QubitDepthMap = {}
    for formal_name, actual_name in qubit_name_map.items():
        if actual_name in qubit_depths:
            local_depths[formal_name] = qubit_depths[actual_name]
        # ArrayValue prefix matching
        for key, val in qubit_depths.items():
            if key.startswith(actual_name + "["):
                suffix = key[len(actual_name) :]
                local_depths[formal_name + suffix] = val
    return local_depths


def _write_back_depths(
    qubit_name_map: dict[str, str],
    local_depths: QubitDepthMap,
    qubit_depths: QubitDepthMap,
) -> None:
    """Write back local depths to caller's depth map."""
    mapped_formals = set(qubit_name_map.keys())
    for formal_name, actual_name in qubit_name_map.items():
        if formal_name in local_depths:
            qubit_depths[actual_name] = local_depths[formal_name]
        for key, val in local_depths.items():
            if key.startswith(formal_name + "["):
                suffix = key[len(formal_name) :]
                qubit_depths[actual_name + suffix] = val
    # Propagate newly allocated qubits
    for qid, depth in local_depths.items():
        base = qid.split("[")[0]
        if base not in mapped_formals:
            qubit_depths[qid] = depth


# ============================================================
# Concrete simulation helpers (for _handle_for_parallel)
# ============================================================


def _concretize_qubit_name(name: str, var_env: dict[str, int]) -> str:
    """Replace symbolic index expressions in qubit names with concrete values.

    Args:
        name: Qubit name like 'qs[j]', 'qs[uint_tmp]', 'target'
        var_env: Mapping from variable names to concrete integer values

    Returns:
        Concretized name like 'qs[3]', 'target'
    """
    m = re.match(r"(\w+)\[(.+)\]", name)
    if not m:
        return name

    array_name = m.group(1)
    index_expr_str = m.group(2)

    # Direct lookup
    if index_expr_str in var_env:
        return f"{array_name}[{var_env[index_expr_str]}]"

    # Try to evaluate as SymPy expression
    try:
        local_symbols = {k: sp.Symbol(k) for k in var_env}
        expr = sp.sympify(index_expr_str, locals=local_symbols)
        val = expr.subs({sp.Symbol(k): v for k, v in var_env.items()})
        if val.is_number:
            return f"{array_name}[{int(val)}]"
    except (sp.SympifyError, TypeError, ValueError):
        pass

    return name


def _eval_value_concrete(
    v: Any,
    block: Any,
    call_context: dict[str, Any],
    var_env: dict[str, int],
) -> int | None:
    """Evaluate a Value to a concrete integer using var_env and call_context.

    Returns None if the value cannot be fully resolved to an integer.
    """
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, sp.Basic):
        result = v.subs({sp.Symbol(k): val for k, val in var_env.items()})
        if result.is_number:
            return int(result)
        return None

    if not isinstance(v, Value):
        return None

    # Check call_context by UUID
    if call_context and v.uuid in call_context:
        resolved = call_context[v.uuid]
        return _eval_value_concrete(resolved, block, call_context, var_env)

    # Check constant
    if v.is_constant():
        c = v.get_const()
        if c is not None:
            return int(c)

    # Check var_env by UUID first (avoids name collisions when multiple
    # BinOps share the same result name like 'uint_tmp')
    if v.uuid in var_env:
        return var_env[v.uuid]

    # Check var_env by name
    if v.name in var_env:
        return var_env[v.name]

    # Check parameter
    if v.is_parameter():
        pname = v.parameter_name()
        if pname and pname in var_env:
            return var_env[pname]

    # Try to trace through operations
    if block is not None:
        expr = value_to_expr(v, block, call_context, {})
        if isinstance(expr, sp.Basic):
            result = expr.subs({sp.Symbol(k): val for k, val in var_env.items()})
            if result.is_number:
                return int(result)

    return None



def _concretize_qubit_operand(
    v: Any,
    block: Any,
    call_context: dict[str, Any],
    var_env: dict[str, int],
) -> str:
    """Resolve a qubit operand Value to a concrete qubit name.

    Uses the Value's element_indices to trace indices through the IR
    by UUID, avoiding name collisions when multiple BinOps share the
    same result name (e.g. 'uint_tmp').
    """
    if (
        hasattr(v, "parent_array")
        and v.parent_array is not None
        and hasattr(v, "element_indices")
        and v.element_indices
    ):
        array_name = v.parent_array.name
        idx_val = _eval_value_concrete(
            v.element_indices[0], block, call_context, var_env
        )
        if idx_val is not None:
            return f"{array_name}[{idx_val}]"

    # Fallback to string-based resolution
    return _concretize_qubit_name(v.name, var_env)


def _simulate_parallel_depth_concrete(
    operations: list[Operation],
    qubit_depths: dict[str, CircuitDepth],
    block: Any,
    call_context: dict[str, Any],
    var_env: dict[str, int],
    num_controls: int | sp.Expr = 0,
    value_depths: dict[str, CircuitDepth] | None = None,
) -> None:
    """Simulate parallel depth with fully concrete variable values.

    Like _estimate_parallel_depth but resolves all loop variables and
    qubit indices to concrete integers. No symbolic manipulation.

    Args:
        operations: List of operations to simulate
        qubit_depths: Mutable per-qubit depth map (modified in place)
        block: Block for value tracing
        call_context: Call context for parameter resolution
        var_env: Mapping from all variable names to concrete integer values
        num_controls: Number of control qubits (0 means no control)
        value_depths: Mutable mapping from Value UUID to depth (for measurement tracking)
    """
    from qamomile.circuit.ir.block_value import BlockValue
    from qamomile.circuit.ir.value import ArrayValue

    for op in operations:
        if isinstance(op, GateOperation):
            gate_depth = _estimate_gate_depth(op, num_controls=num_controls)
            qubit_ids = [
                _concretize_qubit_operand(v, block, call_context, var_env)
                for v in op.operands
            ]
            if qubit_ids:
                max_current = CircuitDepth.zero()
                for qid in qubit_ids:
                    max_current = max_current.max(
                        qubit_depths.get(qid, CircuitDepth.zero())
                    )
                new_depth = max_current + gate_depth
                for qid in qubit_ids:
                    qubit_depths[qid] = new_depth

        elif isinstance(op, BinOp):
            # Track arithmetic results so qubit names like q[uint_tmp] resolve
            if op.results:
                result_val = op.results[0]
                result_name = result_val.name
                result_uuid = result_val.uuid
                left = _eval_value_concrete(
                    op.operands[0], block, call_context, var_env
                )
                right = _eval_value_concrete(
                    op.operands[1], block, call_context, var_env
                )
                if left is not None and right is not None:
                    computed: int = 0
                    if op.kind == BinOpKind.ADD:
                        computed = left + right
                    elif op.kind == BinOpKind.SUB:
                        computed = left - right
                    elif op.kind == BinOpKind.MUL:
                        computed = left * right
                    elif op.kind == BinOpKind.FLOORDIV:
                        computed = left // right if right != 0 else 0
                    elif op.kind == BinOpKind.DIV:
                        computed = left // right if right != 0 else 0
                    elif op.kind == BinOpKind.POW:
                        computed = left**right
                    # Store by both name (backward compat) and UUID
                    # (avoids collisions when multiple BinOps share a name)
                    var_env[result_name] = computed
                    var_env[result_uuid] = computed

        elif isinstance(op, MeasureOperation):
            qubit_id = _concretize_qubit_operand(
                op.operands[0], block, call_context, var_env
            )
            current = qubit_depths.get(qubit_id, CircuitDepth.zero())
            measure_unit = CircuitDepth(
                total_depth=sp.Integer(1),
                t_depth=sp.Integer(0),
                two_qubit_depth=sp.Integer(0),
                multi_qubit_depth=sp.Integer(0),
                rotation_depth=sp.Integer(0),
            )
            new_depth = current + measure_unit
            qubit_depths[qubit_id] = new_depth
            if value_depths is not None and op.results:
                value_depths[op.results[0].uuid] = new_depth

        elif isinstance(op, MeasureVectorOperation):
            arr = op.operands[0]
            arr_name = _concretize_qubit_name(arr.name, var_env)
            matched_qids = [
                k
                for k in qubit_depths
                if k.startswith(arr_name + "[") or k == arr_name
            ]
            if not matched_qids:
                matched_qids = [arr_name]

            max_current = CircuitDepth.zero()
            for qid in matched_qids:
                max_current = max_current.max(
                    qubit_depths.get(qid, CircuitDepth.zero())
                )
            measure_unit = CircuitDepth(
                total_depth=sp.Integer(1),
                t_depth=sp.Integer(0),
                two_qubit_depth=sp.Integer(0),
                multi_qubit_depth=sp.Integer(0),
                rotation_depth=sp.Integer(0),
            )
            new_depth = max_current + measure_unit
            for qid in matched_qids:
                qubit_depths[qid] = new_depth
            if value_depths is not None and op.results:
                value_depths[op.results[0].uuid] = new_depth

        elif isinstance(op, NotOp):
            if value_depths is not None:
                input_depth = value_depths.get(
                    op.input.uuid, CircuitDepth.zero()
                )
                value_depths[op.output.uuid] = input_depth

        elif isinstance(op, CondOp):
            if value_depths is not None:
                lhs_depth = value_depths.get(
                    op.operands[0].uuid, CircuitDepth.zero()
                )
                rhs_depth = value_depths.get(
                    op.operands[1].uuid, CircuitDepth.zero()
                )
                value_depths[op.results[0].uuid] = lhs_depth.max(rhs_depth)

        elif isinstance(op, CompOp):
            if value_depths is not None:
                lhs_depth = value_depths.get(
                    op.operands[0].uuid, CircuitDepth.zero()
                )
                rhs_depth = value_depths.get(
                    op.operands[1].uuid, CircuitDepth.zero()
                )
                value_depths[op.results[0].uuid] = lhs_depth.max(rhs_depth)

        elif isinstance(op, IfOperation):
            condition_depth = CircuitDepth.zero()
            if value_depths is not None:
                condition_depth = value_depths.get(
                    op.condition.uuid, CircuitDepth.zero()
                )

            true_depths = qubit_depths.copy()
            false_depths = qubit_depths.copy()

            branch_qubits = _collect_all_qubit_names(
                op.true_operations
            ) | _collect_gate_qubit_names(op.false_operations)
            for qid in branch_qubits:
                current = qubit_depths.get(qid, CircuitDepth.zero())
                true_depths[qid] = current.max(condition_depth)
                false_depths[qid] = current.max(condition_depth)

            _simulate_parallel_depth_concrete(
                op.true_operations,
                true_depths,
                block=block,
                call_context=call_context,
                var_env=var_env,
                num_controls=num_controls,
                value_depths=value_depths,
            )
            _simulate_parallel_depth_concrete(
                op.false_operations,
                false_depths,
                block=block,
                call_context=call_context,
                var_env=var_env,
                num_controls=num_controls,
                value_depths=value_depths,
            )

            all_qids = set(true_depths) | set(false_depths)
            for qid in all_qids:
                qubit_depths[qid] = true_depths.get(
                    qid, CircuitDepth.zero()
                ).max(false_depths.get(qid, CircuitDepth.zero()))

            # PhiOp depth propagation
            for phi_op in op.phi_ops:
                tv_depth = qubit_depths.get(
                    phi_op.true_value.name, CircuitDepth.zero()
                )
                fv_depth = qubit_depths.get(
                    phi_op.false_value.name, CircuitDepth.zero()
                )
                qubit_depths[phi_op.output.name] = tv_depth.max(fv_depth)

        elif isinstance(op, ForOperation):
            if len(op.operands) < 2:
                continue
            start_val = _eval_value_concrete(
                op.operands[0], block, call_context, var_env
            )
            stop_val = _eval_value_concrete(
                op.operands[1], block, call_context, var_env
            )
            step_val = (
                _eval_value_concrete(
                    op.operands[2], block, call_context, var_env
                )
                if len(op.operands) >= 3
                else 1
            )
            if start_val is None or stop_val is None or step_val is None:
                continue
            for loop_val in range(start_val, stop_val, step_val):
                inner_env = var_env.copy()
                inner_env[op.loop_var] = loop_val
                _simulate_parallel_depth_concrete(
                    op.operations,
                    qubit_depths,
                    block=block,
                    call_context=call_context,
                    var_env=inner_env,
                    num_controls=num_controls,
                    value_depths=value_depths,
                )

        elif isinstance(op, CallBlockOperation):
            called_block = op.operands[0]
            if not isinstance(called_block, BlockValue):
                continue

            new_call_context = call_context.copy()
            for formal_idx, formal_input in enumerate(
                called_block.input_values
            ):
                if formal_idx + 1 < len(op.operands):
                    actual_arg = op.operands[formal_idx + 1]
                    concrete_val = _eval_value_concrete(
                        actual_arg, block, call_context, var_env
                    )
                    if concrete_val is not None:
                        new_call_context[formal_input.uuid] = sp.Integer(
                            concrete_val
                        )
                    else:
                        resolved = value_to_expr(
                            actual_arg, block, call_context, {}
                        )
                        new_call_context[formal_input.uuid] = resolved

                    if isinstance(actual_arg, ArrayValue) and isinstance(
                        formal_input, ArrayValue
                    ):
                        for dim_formal, dim_actual in zip(
                            formal_input.shape, actual_arg.shape
                        ):
                            dim_val = _eval_value_concrete(
                                dim_actual, block, call_context, var_env
                            )
                            if dim_val is not None:
                                new_call_context[dim_formal.uuid] = sp.Integer(
                                    dim_val
                                )
                            else:
                                new_call_context[dim_formal.uuid] = (
                                    value_to_expr(
                                        dim_actual, block, call_context, {}
                                    )
                                )

            # Build qubit name mapping with concretized names
            qubit_name_map = {}
            for formal_idx, formal_input in enumerate(
                called_block.input_values
            ):
                if formal_idx + 1 < len(op.operands):
                    actual_arg = op.operands[formal_idx + 1]
                    if (
                        hasattr(formal_input, "type")
                        and hasattr(formal_input.type, "is_quantum")
                        and formal_input.type.is_quantum()
                    ):
                        concrete_name = _concretize_qubit_operand(
                            actual_arg, block, call_context, var_env
                        )
                        qubit_name_map[formal_input.name] = concrete_name
            local_depths = _map_depths_to_local(qubit_name_map, qubit_depths)

            _simulate_parallel_depth_concrete(
                called_block.operations,
                local_depths,
                block=called_block,
                call_context=new_call_context,
                var_env=var_env,
                num_controls=num_controls,
                value_depths=value_depths,
            )

            _write_back_depths(qubit_name_map, local_depths, qubit_depths)

        elif isinstance(op, ControlledUOperation):
            if op.is_symbolic_num_controls:
                nc = _eval_value_concrete(
                    op.num_controls, block, call_context, var_env
                )
                if nc is None:
                    # Cannot resolve to concrete int; conservative fallback
                    all_qubit_names = [v.name for v in op.operands[1:]]
                    max_current = CircuitDepth.zero()
                    for qid in all_qubit_names:
                        max_current = max_current.max(
                            qubit_depths.get(qid, CircuitDepth.zero())
                        )
                    gate_depth = CircuitDepth(
                        total_depth=sp.Integer(1),
                        t_depth=sp.Integer(0),
                        two_qubit_depth=sp.Integer(0),
                        multi_qubit_depth=sp.Integer(1),
                        rotation_depth=sp.Integer(0),
                    )
                    new_depth = max_current + gate_depth
                    for qid in all_qubit_names:
                        qubit_depths[qid] = new_depth
                    continue
            else:
                nc = op.num_controls

            controlled_block = op.block
            if not isinstance(controlled_block, BlockValue):
                continue

            new_call_context = call_context.copy()

            if op.has_index_spec:
                # index_spec mode: skip qubit formal inputs,
                # map only non-qubit parameters from operands[2:]
                param_operands = op.operands[2:]
                param_idx = 0
                for formal_input in controlled_block.input_values:
                    if not formal_input.type.is_quantum():
                        if param_idx < len(param_operands):
                            actual_arg = param_operands[param_idx]
                            concrete_val = _eval_value_concrete(
                                actual_arg, block, call_context, var_env
                            )
                            if concrete_val is not None:
                                new_call_context[
                                    formal_input.uuid
                                ] = sp.Integer(concrete_val)
                            else:
                                new_call_context[
                                    formal_input.uuid
                                ] = value_to_expr(
                                    actual_arg, block, call_context, {}
                                )
                            param_idx += 1

                # Get qubit names from Vector elements
                vector = op.operands[1]
                vector_name = vector.name
                if vector.shape:
                    vs = _eval_value_concrete(
                        vector.shape[0], block, call_context, var_env
                    )
                    if vs is not None:
                        all_qubit_names = [
                            f"{vector_name}[{i}]" for i in range(vs)
                        ]
                    else:
                        all_qubit_names = [vector_name]
                else:
                    all_qubit_names = [vector_name]
            else:
                target_operands = op.target_operands
                for formal_idx, formal_input in enumerate(
                    controlled_block.input_values
                ):
                    if formal_idx < len(target_operands):
                        actual_arg = target_operands[formal_idx]
                        concrete_val = _eval_value_concrete(
                            actual_arg, block, call_context, var_env
                        )
                        if concrete_val is not None:
                            new_call_context[
                                formal_input.uuid
                            ] = sp.Integer(concrete_val)
                        else:
                            new_call_context[
                                formal_input.uuid
                            ] = value_to_expr(
                                actual_arg, block, call_context, {}
                            )
                        if isinstance(actual_arg, ArrayValue) and isinstance(
                            formal_input, ArrayValue
                        ):
                            for dim_formal, dim_actual in zip(
                                formal_input.shape, actual_arg.shape
                            ):
                                dim_val = _eval_value_concrete(
                                    dim_actual, block, call_context, var_env
                                )
                                if dim_val is not None:
                                    new_call_context[
                                        dim_formal.uuid
                                    ] = sp.Integer(dim_val)

                # Get control + target qubit names
                ctrl_names = [
                    _concretize_qubit_operand(v, block, call_context, var_env)
                    for v in op.control_operands
                ]
                tgt_names = [
                    _concretize_qubit_operand(v, block, call_context, var_env)
                    for v in op.target_operands
                ]
                all_qubit_names = ctrl_names + tgt_names

            # Compute inner depth
            inner_depths: dict[str, CircuitDepth] = {}
            _simulate_parallel_depth_concrete(
                controlled_block.operations,
                inner_depths,
                block=controlled_block,
                call_context=new_call_context,
                var_env=var_env,
                num_controls=nc,
            )
            inner_depth = _get_max_depth(inner_depths)

            # Update all involved qubits
            max_current = CircuitDepth.zero()
            for qid in all_qubit_names:
                max_current = max_current.max(
                    qubit_depths.get(qid, CircuitDepth.zero())
                )
            new_depth = max_current + inner_depth
            for qid in all_qubit_names:
                qubit_depths[qid] = new_depth

        elif isinstance(op, CompositeGateOperation):
            composite_depth = _estimate_composite_gate_depth(
                op, block, [], call_context, {}, num_controls=num_controls
            )
            # Substitute var_env values into composite depth
            subs_dict = {sp.Symbol(k): v for k, v in var_env.items()}
            composite_depth = composite_depth.substitute(subs_dict)

            # Expand array operand names to element-level qubit IDs
            raw_names = [
                _concretize_qubit_name(v.name, var_env)
                for v in op.operands
                if hasattr(v, "name")
            ]
            qubit_ids: list[str] = []
            for name in raw_names:
                matched = [
                    k for k in qubit_depths
                    if k.startswith(name + "[")
                ]
                if matched:
                    qubit_ids.extend(matched)
                else:
                    qubit_ids.append(name)

            max_current = CircuitDepth.zero()
            for qid in qubit_ids:
                max_current = max_current.max(
                    qubit_depths.get(qid, CircuitDepth.zero())
                )
            new_depth = max_current + composite_depth
            for qid in qubit_ids:
                qubit_depths[qid] = new_depth


# ============================================================
# Interpolation
# ============================================================


def _interpolate_depth(
    samples: dict[int, CircuitDepth],
    sym: sp.Symbol,
) -> CircuitDepth:
    """Interpolate symbolic CircuitDepth from concrete samples.

    Strategy per field:
    1. Try polynomial interpolation with leave-one-out verification
    2. Try exponential base-2 + polynomial: a*2^n + b*n + c
    3. Fallback to full polynomial interpolation

    Args:
        samples: Mapping from concrete parameter value to CircuitDepth
        sym: The SymPy symbol to use in the result expression

    Returns:
        CircuitDepth with symbolic expressions
    """
    sorted_ns = sorted(samples.keys())

    def interpolate_field(field_getter: Any) -> sp.Expr:
        vals = {n_val: int(field_getter(samples[n_val])) for n_val in sorted_ns}

        if all(v == 0 for v in vals.values()):
            return sp.Integer(0)

        pts = [(n_val, vals[n_val]) for n_val in sorted_ns]

        # Step 1: polynomial (leave-one-out)
        if len(pts) >= 4:
            train_pts = pts[:-1]
            test_n, test_v = pts[-1]
            poly = sp.interpolate(train_pts, sym)
            if sp.simplify(poly.subs(sym, test_n) - test_v) == 0:
                return sp.simplify(poly)

        # Step 2: exponential base-2 + degree-1 polynomial: a*2^n + b*n + c
        if len(pts) >= 4:
            a_s, b_s, c_s = sp.symbols("_a _b _c")
            n1, v1 = pts[0]
            n2, v2 = pts[1]
            n3, v3 = pts[2]
            eqs = [
                a_s * 2**n1 + b_s * n1 + c_s - v1,
                a_s * 2**n2 + b_s * n2 + c_s - v2,
                a_s * 2**n3 + b_s * n3 + c_s - v3,
            ]
            sol = sp.solve(eqs, [a_s, b_s, c_s])
            if sol and all(
                isinstance(sol[s], sp.Rational) for s in [a_s, b_s, c_s]
            ):
                expr = sol[a_s] * 2**sym + sol[b_s] * sym + sol[c_s]
                if all(
                    sp.simplify(expr.subs(sym, ni) - vi) == 0
                    for ni, vi in pts
                ):
                    return sp.simplify(expr)

        # Step 3: fallback to full polynomial
        return sp.simplify(sp.interpolate(pts, sym))

    return CircuitDepth(
        total_depth=interpolate_field(lambda d: d.total_depth),
        t_depth=interpolate_field(lambda d: d.t_depth),
        two_qubit_depth=interpolate_field(lambda d: d.two_qubit_depth),
        multi_qubit_depth=interpolate_field(lambda d: d.multi_qubit_depth),
        rotation_depth=interpolate_field(lambda d: d.rotation_depth),
    )


def _has_binop_name_collisions(operations: list[Operation]) -> bool:
    """Check if the operations have multiple BinOps with the same result name.

    When BinOps share result names (e.g. 'uint_tmp'), symbolic qubit names
    like 'qs[uint_tmp]' become ambiguous and the symbolic depth tracker
    can't distinguish which concrete qubit is meant. In this case, full-block
    concrete sampling is needed.
    """
    seen: set[str] = set()
    for op in operations:
        if isinstance(op, BinOp) and op.results:
            name = op.results[0].name
            if name in seen:
                return True
            seen.add(name)
    return False


def _scan_parametric_for_loops(
    operations: list[Operation],
    block: Any,
    call_context: dict[str, Any],
    loop_var_symbols: dict[str, sp.Symbol],
    loop_context: LoopContext,
) -> tuple[sp.Symbol, list[tuple[sp.Expr, sp.Expr, sp.Expr]]] | None:
    """Scan top-level operations for ForOperations with parametric bounds.

    Returns (parametric_symbol, loop_bounds_list) or None if no parametric
    for-loops found.
    """
    param_syms_found: set[sp.Symbol] = set()
    loop_bounds: list[tuple[sp.Expr, sp.Expr, sp.Expr]] = []

    for op in operations:
        if not isinstance(op, ForOperation) or len(op.operands) < 2:
            continue
        try:
            (
                loop_var_symbol,
                start_expr,
                stop_expr,
                step_expr,
                _,
                _,
                _,
            ) = _prepare_for_loop(
                op, block, call_context, loop_var_symbols, loop_context
            )
        except Exception:
            continue

        all_bound_syms = (
            start_expr.free_symbols | stop_expr.free_symbols | step_expr.free_symbols
        )
        parametric_syms = (
            all_bound_syms - {loop_var_symbol} - set(loop_var_symbols.values())
        )
        if parametric_syms:
            param_syms_found.update(parametric_syms)
            loop_bounds.append((start_expr, stop_expr, step_expr))

    if not param_syms_found:
        return None

    param_sym = sorted(param_syms_found, key=str)[0]
    return param_sym, loop_bounds


def _compute_valid_sample_points_for_block(
    param_sym: sp.Symbol,
    loop_bounds: list[tuple[sp.Expr, sp.Expr, sp.Expr]],
    min_samples: int = 7,
) -> list[int]:
    """Compute sample points where all parametric loops have non-zero iterations."""
    valid: list[int] = []
    for n_val in range(2, 30):
        subs = {param_sym: n_val}
        all_ok = True
        for start_expr, stop_expr, step_expr in loop_bounds:
            try:
                cs = int(start_expr.subs(subs))
                ce = int(stop_expr.subs(subs))
                ct = int(step_expr.subs(subs))
            except (TypeError, ValueError):
                all_ok = False
                break
            if len(range(cs, ce, ct)) == 0:
                all_ok = False
                break
        if all_ok:
            valid.append(n_val)
            if len(valid) >= min_samples:
                break
    return valid


def _estimate_by_full_block_sampling(
    operations: list[Operation],
    qubit_depths: QubitDepthMap,
    block: Any,
    call_context: dict[str, Any],
    param_sym: sp.Symbol,
    valid_points: list[int],
    num_controls: int | sp.Expr,
) -> None:
    """Estimate depth by concretely simulating the entire operations block.

    For each sample value of the parametric symbol, simulates all operations
    concretely (resolving symbolic qubit names to concrete ones) and records
    the max depth. Then interpolates to get a symbolic expression.

    This avoids qubit name aliasing issues that arise when tracking symbolic
    depths through parametric for-loops.
    """
    entry_depth = _get_max_depth(qubit_depths)
    body_qubits = _collect_all_qubit_names(operations)
    samples: dict[int, CircuitDepth] = {}

    for n_val in valid_points:
        subs = {param_sym: n_val}
        var_env: dict[str, int] = {str(param_sym): n_val}

        # Concretize call_context
        concrete_call_ctx: dict[str, Any] = {}
        for uuid, expr in call_context.items():
            if isinstance(expr, sp.Basic) and expr.free_symbols:
                concrete_call_ctx[uuid] = expr.subs(subs)
            else:
                concrete_call_ctx[uuid] = expr

        inner_depths: dict[str, CircuitDepth] = {}
        _simulate_parallel_depth_concrete(
            operations,
            inner_depths,
            block=block,
            call_context=concrete_call_ctx,
            var_env=var_env,
            num_controls=num_controls,
        )

        samples[n_val] = _get_max_depth(inner_depths)

    if not samples:
        return

    # Use last point as verification
    verify_n = valid_points[-1]
    verify_sample = samples.pop(verify_n, None)

    interpolated = _interpolate_depth(samples, param_sym)

    if verify_sample is not None:
        verify_check = interpolated.substitute({param_sym: verify_n})
        mismatch = any(
            sp.simplify(
                getattr(verify_check, field) - getattr(verify_sample, field)
            )
            != 0
            for field in [
                "total_depth",
                "t_depth",
                "two_qubit_depth",
                "multi_qubit_depth",
                "rotation_depth",
            ]
        )
        if mismatch:
            samples[verify_n] = verify_sample
            interpolated = _interpolate_depth(samples, param_sym)

    # Update qubit_depths
    new_depth = entry_depth + interpolated
    for qid in set(qubit_depths) | body_qubits:
        qubit_depths[qid] = new_depth


def _estimate_parallel_depth(
    operations: list[Operation],
    qubit_depths: QubitDepthMap,
    block: Block | None = None,
    loop_context: LoopContext | None = None,
    call_context: dict[str, Any] | None = None,
    loop_var_symbols: dict[str, sp.Symbol] | None = None,
    num_controls: int | sp.Expr = 0,
    value_depths: dict[str, CircuitDepth] | None = None,
) -> None:
    """Estimate parallel depth by tracking per-qubit depths.

    Modifies qubit_depths in place. Each gate is placed at the earliest
    time step where all its qubits are available.

    Args:
        operations: List of operations
        qubit_depths: Mutable mapping from qubit name to current depth
        block: Optional Block for value tracing
        loop_context: Stack of outer loop variables for nested loop analysis
        call_context: Optional mapping from Value UUIDs to actual argument Values
        loop_var_symbols: Optional mapping from Value names to SymPy loop variable symbols
        num_controls: Number of control qubits (0 means no control)
        value_depths: Mutable mapping from Value UUID to depth (for measurement tracking)
    """
    if loop_context is None:
        loop_context = []
    if call_context is None:
        call_context = {}
    if loop_var_symbols is None:
        loop_var_symbols = {}

    # Use full-block concrete sampling only when BinOp name collisions
    # cause qubit name aliasing (e.g. multiple BinOps produce 'uint_tmp',
    # making 'qs[uint_tmp]' ambiguous in symbolic depth tracking).
    if _has_binop_name_collisions(operations):
        param_result = _scan_parametric_for_loops(
            operations, block, call_context, loop_var_symbols, loop_context
        )
        if param_result is not None:
            param_sym, loop_bounds = param_result
            valid_points = _compute_valid_sample_points_for_block(
                param_sym, loop_bounds
            )
            if valid_points:
                _estimate_by_full_block_sampling(
                    operations,
                    qubit_depths,
                    block,
                    call_context,
                    param_sym,
                    valid_points,
                    num_controls,
                )
                return

    _MEASURE_UNIT = CircuitDepth(
        total_depth=sp.Integer(1),
        t_depth=sp.Integer(0),
        two_qubit_depth=sp.Integer(0),
        multi_qubit_depth=sp.Integer(0),
        rotation_depth=sp.Integer(0),
    )

    for op in operations:
        match op:
            case GateOperation():
                gate_depth = _estimate_gate_depth(op, num_controls=num_controls)
                qubit_ids = [v.name for v in op.operands]
                if qubit_ids:
                    # Max depth among all involved qubits
                    max_current = CircuitDepth.zero()
                    for qid in qubit_ids:
                        max_current = max_current.max(
                            qubit_depths.get(qid, CircuitDepth.zero())
                        )
                    new_depth = max_current + gate_depth
                    for qid in qubit_ids:
                        qubit_depths[qid] = new_depth

            case MeasureOperation():
                qubit_id = op.operands[0].name
                current = qubit_depths.get(qubit_id, CircuitDepth.zero())
                new_depth = current + _MEASURE_UNIT
                qubit_depths[qubit_id] = new_depth
                if value_depths is not None and op.results:
                    value_depths[op.results[0].uuid] = new_depth

            case MeasureVectorOperation():
                arr = op.operands[0]
                arr_name = arr.name
                matched_qids = [
                    k
                    for k in qubit_depths
                    if k.startswith(arr_name + "[") or k == arr_name
                ]
                if not matched_qids:
                    matched_qids = [arr_name]

                max_current = CircuitDepth.zero()
                for qid in matched_qids:
                    max_current = max_current.max(
                        qubit_depths.get(qid, CircuitDepth.zero())
                    )
                new_depth = max_current + _MEASURE_UNIT
                for qid in matched_qids:
                    qubit_depths[qid] = new_depth
                if value_depths is not None and op.results:
                    value_depths[op.results[0].uuid] = new_depth

            case NotOp():
                if value_depths is not None:
                    input_depth = value_depths.get(
                        op.input.uuid, CircuitDepth.zero()
                    )
                    value_depths[op.output.uuid] = input_depth

            case CondOp():
                if value_depths is not None:
                    lhs_depth = value_depths.get(
                        op.operands[0].uuid, CircuitDepth.zero()
                    )
                    rhs_depth = value_depths.get(
                        op.operands[1].uuid, CircuitDepth.zero()
                    )
                    value_depths[op.results[0].uuid] = lhs_depth.max(rhs_depth)

            case CompOp():
                if value_depths is not None:
                    lhs_depth = value_depths.get(
                        op.operands[0].uuid, CircuitDepth.zero()
                    )
                    rhs_depth = value_depths.get(
                        op.operands[1].uuid, CircuitDepth.zero()
                    )
                    value_depths[op.results[0].uuid] = lhs_depth.max(rhs_depth)

            case ForOperation():
                _handle_for_parallel(
                    op,
                    qubit_depths,
                    block,
                    loop_context,
                    call_context,
                    loop_var_symbols,
                    num_controls,
                    value_depths=value_depths,
                )

            case WhileOperation():
                # Conservative: sequential depth as increase to all qubits
                inner_depth = _compute_sequential_depth(
                    op.operations,
                    block=None,
                    loop_context=loop_context,
                    call_context=call_context,
                    loop_var_symbols=loop_var_symbols,
                    num_controls=num_controls,
                )
                iterations = sp.Symbol("|while|", integer=True, positive=True)
                increase = inner_depth * iterations
                current_max = _get_max_depth(qubit_depths)
                new_max = current_max + increase
                body_qubits = _collect_gate_qubit_names(op.operations)
                for qid in set(qubit_depths) | body_qubits:
                    qubit_depths[qid] = new_max

            case IfOperation():
                # Get condition depth from value_depths
                condition_depth = CircuitDepth.zero()
                if value_depths is not None:
                    condition_depth = value_depths.get(
                        op.condition.uuid, CircuitDepth.zero()
                    )

                true_depths = qubit_depths.copy()
                false_depths = qubit_depths.copy()

                # Bump branch qubit depths to condition_depth
                branch_qubits = _collect_all_qubit_names(
                    op.true_operations
                ) | _collect_gate_qubit_names(op.false_operations)
                for qid in branch_qubits:
                    current = qubit_depths.get(qid, CircuitDepth.zero())
                    bumped = current.max(condition_depth)
                    true_depths[qid] = bumped
                    false_depths[qid] = bumped

                _estimate_parallel_depth(
                    op.true_operations,
                    true_depths,
                    block=None,
                    loop_context=loop_context,
                    call_context=call_context,
                    loop_var_symbols=loop_var_symbols,
                    num_controls=num_controls,
                    value_depths=value_depths,
                )
                _estimate_parallel_depth(
                    op.false_operations,
                    false_depths,
                    block=None,
                    loop_context=loop_context,
                    call_context=call_context,
                    loop_var_symbols=loop_var_symbols,
                    num_controls=num_controls,
                    value_depths=value_depths,
                )
                all_qids = set(true_depths) | set(false_depths)
                for qid in all_qids:
                    qubit_depths[qid] = true_depths.get(
                        qid, CircuitDepth.zero()
                    ).max(false_depths.get(qid, CircuitDepth.zero()))

                # PhiOp depth propagation
                for phi_op in op.phi_ops:
                    tv_name = phi_op.true_value.name
                    fv_name = phi_op.false_value.name
                    out_name = phi_op.output.name
                    tv_depth = qubit_depths.get(tv_name, CircuitDepth.zero())
                    fv_depth = qubit_depths.get(fv_name, CircuitDepth.zero())
                    qubit_depths[out_name] = tv_depth.max(fv_depth)

            case CallBlockOperation():
                _handle_call_block_parallel(
                    op,
                    qubit_depths,
                    block,
                    loop_context,
                    call_context,
                    loop_var_symbols,
                    num_controls,
                    value_depths=value_depths,
                )

            case ControlledUOperation():
                _handle_controlled_u_parallel(
                    op,
                    qubit_depths,
                    block,
                    loop_context,
                    call_context,
                    loop_var_symbols,
                    value_depths=value_depths,
                )

            case CompositeGateOperation():
                composite_depth = _estimate_composite_gate_depth(
                    op,
                    block,
                    loop_context,
                    call_context,
                    loop_var_symbols,
                    num_controls=num_controls,
                )
                # Expand array operand names to element-level qubit IDs
                raw_names = [v.name for v in op.operands if hasattr(v, "name")]
                qubit_ids: list[str] = []
                for name in raw_names:
                    matched = [
                        k for k in qubit_depths
                        if k.startswith(name + "[")
                    ]
                    if matched:
                        qubit_ids.extend(matched)
                    else:
                        qubit_ids.append(name)

                if qubit_ids:
                    max_current = CircuitDepth.zero()
                    for qid in qubit_ids:
                        max_current = max_current.max(
                            qubit_depths.get(qid, CircuitDepth.zero())
                        )
                    new_depth = max_current + composite_depth
                    for qid in qubit_ids:
                        qubit_depths[qid] = new_depth

            case _:
                continue


def _prepare_for_loop(
    op: ForOperation,
    block: Block | None,
    call_context: dict[str, Any],
    loop_var_symbols: dict[str, sp.Symbol],
    loop_context: LoopContext,
) -> tuple[sp.Symbol, sp.Expr, sp.Expr, sp.Expr, LoopContext, dict[str, sp.Symbol], Any]:
    """Extract and prepare loop variable, bounds, and context for ForOperation.

    Returns:
        (loop_var_symbol, start_expr, stop_expr, step_expr,
         new_context, new_loop_var_symbols, local_block)
    """
    start = op.operands[0]
    stop = op.operands[1]
    step = op.operands[2] if len(op.operands) >= 3 else None

    new_loop_var_symbols = loop_var_symbols.copy()
    loop_var_symbol = sp.Symbol(op.loop_var, integer=True, positive=True)

    loop_var_values = _find_loop_variable_values(op.operations, op.loop_var)
    for lv in loop_var_values:
        new_loop_var_symbols[lv.name] = loop_var_symbol
    new_loop_var_symbols[op.loop_var] = loop_var_symbol

    local_block = type("LocalBlock", (), {"operations": op.operations})()

    def convert_with_local_trace(v):
        if block is not None:
            result = value_to_expr(v, block, call_context, new_loop_var_symbols)
            if isinstance(v, Value) and not v.is_constant() and not v.is_parameter():
                local_result = value_to_expr(
                    v, local_block, call_context, new_loop_var_symbols
                )
                if local_result != sp.Symbol(v.name, integer=True, positive=True):
                    return local_result
            return result
        else:
            return value_to_expr(v, local_block, call_context, new_loop_var_symbols)

    start_expr = convert_with_local_trace(start)
    stop_expr = convert_with_local_trace(stop)
    step_expr = convert_with_local_trace(step) if step else sp.Integer(1)

    new_context = loop_context + [(op.loop_var, start_expr, stop_expr, step_expr)]

    return (
        loop_var_symbol,
        start_expr,
        stop_expr,
        step_expr,
        new_context,
        new_loop_var_symbols,
        local_block,
    )


def _handle_for_parallel(
    op: ForOperation,
    qubit_depths: QubitDepthMap,
    block: Block | None,
    loop_context: LoopContext,
    call_context: dict[str, Any],
    loop_var_symbols: dict[str, sp.Symbol],
    num_controls: int | sp.Expr,
    value_depths: dict[str, CircuitDepth] | None = None,
) -> None:
    """Handle ForOperation for parallel depth estimation.

    Uses concrete value simulation + interpolation for accurate
    nested loop depth estimation.
    """
    if len(op.operands) < 2:
        # Fallback: unknown loop bounds
        increase = _compute_sequential_depth(
            op.operations,
            block=None,
            loop_context=loop_context,
            call_context=call_context,
            loop_var_symbols=loop_var_symbols,
            num_controls=num_controls,
        )
        iterations = sp.Symbol("iter", integer=True, positive=True)
        current_max = _get_max_depth(qubit_depths)
        new_max = current_max + increase * iterations
        body_qubits = _collect_gate_qubit_names(op.operations)
        for qid in set(qubit_depths) | body_qubits:
            qubit_depths[qid] = new_max
        return

    (
        loop_var_symbol,
        start_expr,
        stop_expr,
        step_expr,
        new_context,
        new_loop_var_symbols,
        local_block,
    ) = _prepare_for_loop(op, block, call_context, loop_var_symbols, loop_context)

    body_qubits = _collect_gate_qubit_names(op.operations)

    # Collect free parametric symbols from loop bounds
    all_bound_syms = (
        start_expr.free_symbols | stop_expr.free_symbols | step_expr.free_symbols
    )
    parametric_syms = all_bound_syms - {loop_var_symbol} - set(
        loop_var_symbols.values()
    )

    sim_block = local_block if local_block is not None else block

    if not parametric_syms:
        # Bounds are fully concrete -> direct simulation
        try:
            start_val = int(start_expr)
            stop_val = int(stop_expr)
            step_val = int(step_expr)
        except (TypeError, ValueError):
            # Cannot convert to int (still symbolic) -> conservative fallback
            increase = _compute_sequential_depth(
                op.operations,
                block=sim_block,
                loop_context=new_context,
                call_context=call_context,
                loop_var_symbols=new_loop_var_symbols,
                num_controls=num_controls,
            )
            iterations = (stop_expr - start_expr) / step_expr
            current_max = _get_max_depth(qubit_depths)
            new_max = current_max + increase * iterations
            for qid in set(qubit_depths) | body_qubits:
                qubit_depths[qid] = new_max
            return

        var_env: dict[str, int] = {}
        for loop_val in range(start_val, stop_val, step_val):
            inner_env = var_env.copy()
            inner_env[op.loop_var] = loop_val
            _simulate_parallel_depth_concrete(
                op.operations,
                qubit_depths,
                block=sim_block,
                call_context=call_context,
                var_env=inner_env,
                num_controls=num_controls,
                value_depths=value_depths,
            )
        return

    # Bounds contain parametric symbols -> sample and interpolate
    param_sym = sorted(parametric_syms, key=str)[0]

    _MIN_SAMPLES = 6  # Minimum training samples for interpolation

    entry_depth = _get_max_depth(qubit_depths)
    samples: dict[int, CircuitDepth] = {}

    # Dynamically select sample points (skip zero-iteration points)
    valid_points: list[int] = []
    for n_val in range(2, 21):
        subs = {param_sym: n_val}
        try:
            cs = int(start_expr.subs(subs))
            ce = int(stop_expr.subs(subs))
            ct = int(step_expr.subs(subs))
        except (TypeError, ValueError):
            continue
        if len(range(cs, ce, ct)) == 0:
            continue  # Skip zero-iteration points
        valid_points.append(n_val)
        if len(valid_points) >= _MIN_SAMPLES + 1:  # training + verification
            break

    if not valid_points:
        # All sample points have zero iterations -> loop does not contribute depth
        return

    if len(valid_points) < 3:
        # Too few valid samples -> conservative fallback
        increase = _compute_sequential_depth(
            op.operations,
            block=sim_block,
            loop_context=new_context,
            call_context=call_context,
            loop_var_symbols=new_loop_var_symbols,
            num_controls=num_controls,
        )
        iterations = (stop_expr - start_expr) / step_expr
        current_max = _get_max_depth(qubit_depths)
        new_max = current_max + increase * iterations
        for qid in set(qubit_depths) | body_qubits:
            qubit_depths[qid] = new_max
        return

    for n_val in valid_points:
        subs = {param_sym: n_val}
        try:
            concrete_start = int(start_expr.subs(subs))
            concrete_stop = int(stop_expr.subs(subs))
            concrete_step = int(step_expr.subs(subs))
        except (TypeError, ValueError):
            continue

        var_env = {str(param_sym): n_val}

        # Concretize call_context
        concrete_call_ctx: dict[str, Any] = {}
        for uuid, expr in call_context.items():
            if isinstance(expr, sp.Basic) and expr.free_symbols:
                concrete_call_ctx[uuid] = expr.subs(subs)
            else:
                concrete_call_ctx[uuid] = expr

        inner_depths: dict[str, CircuitDepth] = {}
        for loop_val in range(concrete_start, concrete_stop, concrete_step):
            inner_env = var_env.copy()
            inner_env[op.loop_var] = loop_val
            _simulate_parallel_depth_concrete(
                op.operations,
                inner_depths,
                block=sim_block,
                call_context=concrete_call_ctx,
                var_env=inner_env,
                num_controls=num_controls,
            )

        samples[n_val] = _get_max_depth(inner_depths)

    if not samples:
        # Could not evaluate any sample point -> conservative fallback
        increase = _compute_sequential_depth(
            op.operations,
            block=sim_block,
            loop_context=new_context,
            call_context=call_context,
            loop_var_symbols=new_loop_var_symbols,
            num_controls=num_controls,
        )
        iterations = (stop_expr - start_expr) / step_expr
        current_max = _get_max_depth(qubit_depths)
        new_max = current_max + increase * iterations
        for qid in set(qubit_depths) | body_qubits:
            qubit_depths[qid] = new_max
        return

    # Separate verification point (last valid point)
    verify_n = valid_points[-1]
    verify_sample = samples.pop(verify_n, None)

    # Interpolate from sample points
    interpolated = _interpolate_depth(samples, param_sym)

    # Verify at verification point
    if verify_sample is not None:
        verify_check = interpolated.substitute({param_sym: verify_n})
        mismatch = any(
            sp.simplify(
                getattr(verify_check, field) - getattr(verify_sample, field)
            )
            != 0
            for field in [
                "total_depth",
                "t_depth",
                "two_qubit_depth",
                "multi_qubit_depth",
                "rotation_depth",
            ]
        )
        if mismatch:
            # Verification failed - include verify point in interpolation
            samples[verify_n] = verify_sample
            interpolated = _interpolate_depth(samples, param_sym)

    # Update qubit_depths
    new_depth = entry_depth + interpolated
    for qid in set(qubit_depths) | body_qubits:
        qubit_depths[qid] = new_depth


def _handle_call_block_parallel(
    op: CallBlockOperation,
    qubit_depths: QubitDepthMap,
    block: Block | None,
    loop_context: LoopContext,
    call_context: dict[str, Any],
    loop_var_symbols: dict[str, sp.Symbol],
    num_controls: int | sp.Expr,
    value_depths: dict[str, CircuitDepth] | None = None,
) -> None:
    """Handle CallBlockOperation for parallel depth estimation."""
    from qamomile.circuit.ir.block_value import BlockValue
    from qamomile.circuit.ir.value import ArrayValue

    called_block = op.operands[0]
    if not isinstance(called_block, BlockValue):
        return

    new_call_context = call_context.copy()
    for formal_idx, formal_input in enumerate(called_block.input_values):
        if formal_idx + 1 < len(op.operands):
            actual_arg = op.operands[formal_idx + 1]
            resolved_arg = value_to_expr(
                actual_arg, block, call_context, loop_var_symbols
            )
            new_call_context[formal_input.uuid] = resolved_arg

            if isinstance(actual_arg, ArrayValue) and isinstance(
                formal_input, ArrayValue
            ):
                for dim_formal, dim_actual in zip(
                    formal_input.shape, actual_arg.shape
                ):
                    resolved_dim = value_to_expr(
                        dim_actual, block, call_context, loop_var_symbols
                    )
                    new_call_context[dim_formal.uuid] = resolved_dim

    # Build qubit name mapping and use local depth map
    qubit_name_map = _build_qubit_name_map(called_block, op.operands, offset=1)
    local_depths = _map_depths_to_local(qubit_name_map, qubit_depths)

    _estimate_parallel_depth(
        called_block.operations,
        local_depths,
        block=called_block,
        loop_context=loop_context,
        call_context=new_call_context,
        loop_var_symbols=loop_var_symbols,
        num_controls=num_controls,
        value_depths=value_depths,
    )

    _write_back_depths(qubit_name_map, local_depths, qubit_depths)


def _handle_controlled_u_parallel(
    op: ControlledUOperation,
    qubit_depths: QubitDepthMap,
    block: Block | None,
    loop_context: LoopContext,
    call_context: dict[str, Any],
    loop_var_symbols: dict[str, sp.Symbol],
    value_depths: dict[str, CircuitDepth] | None = None,
) -> None:
    """Handle ControlledUOperation for parallel depth estimation."""
    from qamomile.circuit.ir.block_value import BlockValue
    from qamomile.circuit.ir.value import ArrayValue

    if op.is_symbolic_num_controls:
        nc_expr = value_to_expr(op.num_controls, block, call_context, loop_var_symbols)
    else:
        nc_expr = op.num_controls

    controlled_block = op.block
    if not isinstance(controlled_block, BlockValue):
        return

    new_call_context = call_context.copy()

    if op.has_index_spec:
        # index_spec mode: skip qubit formal inputs,
        # map only non-qubit parameters from operands[2:]
        param_operands = op.operands[2:]
        param_idx = 0
        for formal_input in controlled_block.input_values:
            if not formal_input.type.is_quantum():
                if param_idx < len(param_operands):
                    actual_arg = param_operands[param_idx]
                    resolved_arg = value_to_expr(
                        actual_arg, block, call_context, loop_var_symbols
                    )
                    new_call_context[formal_input.uuid] = resolved_arg
                    if isinstance(actual_arg, ArrayValue) and isinstance(
                        formal_input, ArrayValue
                    ):
                        for dim_formal, dim_actual in zip(
                            formal_input.shape, actual_arg.shape
                        ):
                            resolved_dim = value_to_expr(
                                dim_actual, block, call_context, loop_var_symbols
                            )
                            new_call_context[dim_formal.uuid] = resolved_dim
                    param_idx += 1

        # Use Vector name for symbolic qubit tracking
        vector = op.operands[1]
        all_qubit_names = [vector.name]
    else:
        target_operands = op.target_operands

        for formal_idx, formal_input in enumerate(controlled_block.input_values):
            if formal_idx < len(target_operands):
                actual_arg = target_operands[formal_idx]
                resolved_arg = value_to_expr(
                    actual_arg, block, call_context, loop_var_symbols
                )
                new_call_context[formal_input.uuid] = resolved_arg

                if isinstance(actual_arg, ArrayValue) and isinstance(
                    formal_input, ArrayValue
                ):
                    for dim_formal, dim_actual in zip(
                        formal_input.shape, actual_arg.shape
                    ):
                        resolved_dim = value_to_expr(
                            dim_actual, block, call_context, loop_var_symbols
                        )
                        new_call_context[dim_formal.uuid] = resolved_dim

        # Get all qubits involved (control + target)
        all_qubit_names = [v.name for v in op.operands if hasattr(v, "name")]

    # Compute inner depth with resolved num_controls
    inner_depths: QubitDepthMap = {}
    _estimate_parallel_depth(
        controlled_block.operations,
        inner_depths,
        block=controlled_block,
        loop_context=loop_context,
        call_context=new_call_context,
        loop_var_symbols=loop_var_symbols,
        num_controls=nc_expr,
        value_depths=value_depths,
    )

    inner_depth = _get_max_depth(inner_depths)

    # All involved qubits get the max depth
    max_current = CircuitDepth.zero()
    for qid in all_qubit_names:
        max_current = max_current.max(
            qubit_depths.get(qid, CircuitDepth.zero())
        )
    new_depth = max_current + inner_depth
    for qid in all_qubit_names:
        qubit_depths[qid] = new_depth


def _compute_sequential_depth(
    operations: list[Operation],
    block: Block | None = None,
    loop_context: LoopContext | None = None,
    call_context: dict[str, Any] | None = None,
    loop_var_symbols: dict[str, sp.Symbol] | None = None,
    num_controls: int | sp.Expr = 0,
) -> CircuitDepth:
    """Compute sequential (sum-of-all-gates) depth. Used as fallback for loops."""
    if loop_context is None:
        loop_context = []
    if call_context is None:
        call_context = {}
    if loop_var_symbols is None:
        loop_var_symbols = {}

    depth = CircuitDepth.zero()

    _MEASURE_UNIT = CircuitDepth(
        total_depth=sp.Integer(1),
        t_depth=sp.Integer(0),
        two_qubit_depth=sp.Integer(0),
        multi_qubit_depth=sp.Integer(0),
        rotation_depth=sp.Integer(0),
    )

    for op in operations:
        match op:
            case GateOperation():
                depth = depth + _estimate_gate_depth(op, num_controls=num_controls)

            case MeasureOperation():
                depth = depth + _MEASURE_UNIT

            case MeasureVectorOperation():
                depth = depth + _MEASURE_UNIT

            case ForOperation():
                if len(op.operands) >= 2:
                    (
                        loop_var_symbol,
                        start_expr,
                        stop_expr,
                        step_expr,
                        new_context,
                        new_loop_var_symbols,
                        local_block,
                    ) = _prepare_for_loop(
                        op, block, call_context, loop_var_symbols, loop_context
                    )

                    inner_depth = _compute_sequential_depth(
                        op.operations,
                        block=local_block,
                        loop_context=new_context,
                        call_context=call_context,
                        loop_var_symbols=new_loop_var_symbols,
                        num_controls=num_controls,
                    )

                    all_depth_free = inner_depth.total_depth.free_symbols
                    all_depth_free = all_depth_free | inner_depth.two_qubit_depth.free_symbols
                    all_depth_free = all_depth_free | inner_depth.multi_qubit_depth.free_symbols
                    if loop_var_symbol in all_depth_free:
                        depth_expr = _apply_sum_to_depth(
                            inner_depth, loop_var_symbol, start_expr, stop_expr,
                            step_expr,
                        )
                    else:
                        iterations = (stop_expr - start_expr) / step_expr
                        depth_expr = inner_depth * iterations

                    depth = depth + depth_expr

            case IfOperation():
                true_depth = _compute_sequential_depth(
                    op.true_operations,
                    block=None,
                    loop_context=loop_context,
                    call_context=call_context,
                    loop_var_symbols=loop_var_symbols,
                    num_controls=num_controls,
                )
                false_depth = _compute_sequential_depth(
                    op.false_operations,
                    block=None,
                    loop_context=loop_context,
                    call_context=call_context,
                    loop_var_symbols=loop_var_symbols,
                    num_controls=num_controls,
                )
                depth = depth + true_depth.max(false_depth)

            case CallBlockOperation():
                from qamomile.circuit.ir.block_value import BlockValue
                from qamomile.circuit.ir.value import ArrayValue

                called_block = op.operands[0]
                if isinstance(called_block, BlockValue):
                    new_call_context = call_context.copy()
                    for formal_idx, formal_input in enumerate(
                        called_block.input_values
                    ):
                        if formal_idx + 1 < len(op.operands):
                            actual_arg = op.operands[formal_idx + 1]
                            resolved_arg = value_to_expr(
                                actual_arg, block, call_context, loop_var_symbols
                            )
                            new_call_context[formal_input.uuid] = resolved_arg
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

                    inner_depth = _compute_sequential_depth(
                        called_block.operations,
                        block=called_block,
                        loop_context=loop_context,
                        call_context=new_call_context,
                        loop_var_symbols=loop_var_symbols,
                        num_controls=num_controls,
                    )
                    depth = depth + inner_depth

            case ControlledUOperation():
                from qamomile.circuit.ir.block_value import BlockValue
                from qamomile.circuit.ir.value import ArrayValue

                if op.is_symbolic_num_controls:
                    nc_expr = value_to_expr(op.num_controls, block, call_context, loop_var_symbols)
                else:
                    nc_expr = op.num_controls

                controlled_block = op.block
                if isinstance(controlled_block, BlockValue):
                    new_call_context = call_context.copy()

                    if op.has_index_spec:
                        # index_spec mode: skip qubit formal inputs,
                        # map only non-qubit parameters
                        param_operands = op.operands[2:]
                        param_idx = 0
                        for formal_input in controlled_block.input_values:
                            if not formal_input.type.is_quantum():
                                if param_idx < len(param_operands):
                                    actual_arg = param_operands[param_idx]
                                    resolved_arg = value_to_expr(
                                        actual_arg,
                                        block,
                                        call_context,
                                        loop_var_symbols,
                                    )
                                    new_call_context[
                                        formal_input.uuid
                                    ] = resolved_arg
                                    if isinstance(
                                        actual_arg, ArrayValue
                                    ) and isinstance(
                                        formal_input, ArrayValue
                                    ):
                                        for dim_formal, dim_actual in zip(
                                            formal_input.shape,
                                            actual_arg.shape,
                                        ):
                                            resolved_dim = value_to_expr(
                                                dim_actual,
                                                block,
                                                call_context,
                                                loop_var_symbols,
                                            )
                                            new_call_context[
                                                dim_formal.uuid
                                            ] = resolved_dim
                                    param_idx += 1
                    else:
                        target_operands = op.target_operands
                        for formal_idx, formal_input in enumerate(
                            controlled_block.input_values
                        ):
                            if formal_idx < len(target_operands):
                                actual_arg = target_operands[formal_idx]
                                resolved_arg = value_to_expr(
                                    actual_arg,
                                    block,
                                    call_context,
                                    loop_var_symbols,
                                )
                                new_call_context[
                                    formal_input.uuid
                                ] = resolved_arg
                                if isinstance(
                                    actual_arg, ArrayValue
                                ) and isinstance(formal_input, ArrayValue):
                                    for dim_formal, dim_actual in zip(
                                        formal_input.shape,
                                        actual_arg.shape,
                                    ):
                                        resolved_dim = value_to_expr(
                                            dim_actual,
                                            block,
                                            call_context,
                                            loop_var_symbols,
                                        )
                                        new_call_context[
                                            dim_formal.uuid
                                        ] = resolved_dim

                    inner_depth = _compute_sequential_depth(
                        controlled_block.operations,
                        block=controlled_block,
                        loop_context=loop_context,
                        call_context=new_call_context,
                        loop_var_symbols=loop_var_symbols,
                        num_controls=nc_expr,
                    )
                    depth = depth + inner_depth

            case CompositeGateOperation():
                composite_depth = _estimate_composite_gate_depth(
                    op,
                    block,
                    loop_context,
                    call_context,
                    loop_var_symbols,
                    num_controls=num_controls,
                )
                depth = depth + composite_depth

            case _:
                continue

    return depth.simplify()


def estimate_depth(block: BlockValue | Block | list[Operation]) -> CircuitDepth:
    """Estimate circuit depth using parallel (DAG critical path) analysis.

    Computes the minimum circuit depth considering gate-level parallelism.
    Gates on independent qubits can execute in parallel. The result is
    the longest dependency chain through the circuit.

    For loops where qubit access patterns can be analyzed:
    - If all iterations use distinct qubits: depth = single iteration depth
    - If iterations share qubits: falls back to sequential (conservative) depth

    Depth is expressed using SymPy expressions and may contain symbols for
    parametric problem sizes (e.g., loop bounds).

    Args:
        block: BlockValue, Block, or list of Operations to analyze

    Returns:
        CircuitDepth with total_depth, t_depth, two_qubit_depth, etc.

    Example:
        >>> from qamomile.circuit.estimator import estimate_depth
        >>> depth = estimate_depth(my_circuit.block)
        >>> print(depth.total_depth)  # e.g., "n + 5"
    """
    from qamomile.circuit.ir.block import Block
    from qamomile.circuit.ir.block_value import BlockValue

    block_ref = None
    if isinstance(block, (BlockValue, Block)):
        block_ref = block
        ops = block.operations
    else:
        ops = block

    qubit_depths: QubitDepthMap = {}
    value_depths: dict[str, CircuitDepth] = {}
    _estimate_parallel_depth(
        ops, qubit_depths, block_ref,
        loop_context=[], call_context={}, loop_var_symbols={},
        value_depths=value_depths,
    )
    return _get_max_depth(qubit_depths).simplify()
