"""Gate counting for quantum circuits.

This module provides algebraic gate counting using SymPy expressions,
allowing resource estimates to depend on problem size parameters.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import sympy as sp
from sympy import Sum

from qamomile.circuit.estimator._utils import _strip_nonneg_max

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
    ForItemsOperation,
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
class GateCount:
    """Gate count breakdown for a quantum circuit.

    All counts are SymPy expressions that may contain symbols
    for parametric problem sizes.

    Attributes:
        total: Total number of gates
        single_qubit: Number of single-qubit gates
        two_qubit: Number of two-qubit gates
        t_gates: Number of T gates (critical for fault tolerance)
        clifford_gates: Number of Clifford gates (H, S, CNOT, CZ, etc.)
    """

    total: sp.Expr
    single_qubit: sp.Expr
    two_qubit: sp.Expr
    multi_qubit: sp.Expr
    t_gates: sp.Expr
    clifford_gates: sp.Expr
    rotation_gates: sp.Expr
    oracle_calls: dict[str, sp.Expr] = field(default_factory=dict)
    oracle_queries: dict[str, sp.Expr] = field(default_factory=dict)

    def __add__(self, other: GateCount) -> GateCount:
        """Add two gate counts together."""
        merged_oracle = dict(self.oracle_calls)
        for name, count in other.oracle_calls.items():
            if name in merged_oracle:
                merged_oracle[name] = merged_oracle[name] + count
            else:
                merged_oracle[name] = count
        merged_queries = dict(self.oracle_queries)
        for name, count in other.oracle_queries.items():
            if name in merged_queries:
                merged_queries[name] = merged_queries[name] + count
            else:
                merged_queries[name] = count
        return GateCount(
            total=self.total + other.total,
            single_qubit=self.single_qubit + other.single_qubit,
            two_qubit=self.two_qubit + other.two_qubit,
            multi_qubit=self.multi_qubit + other.multi_qubit,
            t_gates=self.t_gates + other.t_gates,
            clifford_gates=self.clifford_gates + other.clifford_gates,
            rotation_gates=self.rotation_gates + other.rotation_gates,
            oracle_calls=merged_oracle,
            oracle_queries=merged_queries,
        )

    def __mul__(self, factor: sp.Expr | int) -> GateCount:
        """Multiply gate counts by a factor."""
        factor_expr = sp.Integer(factor) if isinstance(factor, int) else factor
        return GateCount(
            total=self.total * factor_expr,
            single_qubit=self.single_qubit * factor_expr,
            two_qubit=self.two_qubit * factor_expr,
            multi_qubit=self.multi_qubit * factor_expr,
            t_gates=self.t_gates * factor_expr,
            clifford_gates=self.clifford_gates * factor_expr,
            rotation_gates=self.rotation_gates * factor_expr,
            oracle_calls={
                name: count * factor_expr
                for name, count in self.oracle_calls.items()
            },
            oracle_queries={
                name: count * factor_expr
                for name, count in self.oracle_queries.items()
            },
        )

    __rmul__ = __mul__

    def max(self, other: GateCount) -> GateCount:
        """Element-wise maximum of two gate counts."""
        def _merge_max(
            a: dict[str, sp.Expr], b: dict[str, sp.Expr]
        ) -> dict[str, sp.Expr]:
            merged: dict[str, sp.Expr] = {}
            for key in set(a.keys()) | set(b.keys()):
                if key in a and key in b:
                    merged[key] = sp.Max(a[key], b[key])
                elif key in a:
                    merged[key] = a[key]
                else:
                    merged[key] = b[key]
            return merged

        return GateCount(
            total=sp.Max(self.total, other.total),
            single_qubit=sp.Max(self.single_qubit, other.single_qubit),
            two_qubit=sp.Max(self.two_qubit, other.two_qubit),
            multi_qubit=sp.Max(self.multi_qubit, other.multi_qubit),
            t_gates=sp.Max(self.t_gates, other.t_gates),
            clifford_gates=sp.Max(self.clifford_gates, other.clifford_gates),
            rotation_gates=sp.Max(self.rotation_gates, other.rotation_gates),
            oracle_calls=_merge_max(self.oracle_calls, other.oracle_calls),
            oracle_queries=_merge_max(self.oracle_queries, other.oracle_queries),
        )

    def simplify(self) -> GateCount:
        """Simplify all SymPy expressions."""
        return GateCount(
            total=_strip_nonneg_max(sp.simplify(self.total)),
            single_qubit=_strip_nonneg_max(sp.simplify(self.single_qubit)),
            two_qubit=_strip_nonneg_max(sp.simplify(self.two_qubit)),
            multi_qubit=_strip_nonneg_max(sp.simplify(self.multi_qubit)),
            t_gates=_strip_nonneg_max(sp.simplify(self.t_gates)),
            clifford_gates=_strip_nonneg_max(sp.simplify(self.clifford_gates)),
            rotation_gates=_strip_nonneg_max(sp.simplify(self.rotation_gates)),
            oracle_calls={
                name: _strip_nonneg_max(sp.simplify(count))
                for name, count in self.oracle_calls.items()
            },
            oracle_queries={
                name: _strip_nonneg_max(sp.simplify(count))
                for name, count in self.oracle_queries.items()
            },
        )

    @staticmethod
    def zero() -> GateCount:
        """Return a zero gate count."""
        return GateCount(
            total=sp.Integer(0),
            single_qubit=sp.Integer(0),
            two_qubit=sp.Integer(0),
            multi_qubit=sp.Integer(0),
            t_gates=sp.Integer(0),
            clifford_gates=sp.Integer(0),
            rotation_gates=sp.Integer(0),
        )


# Gate categorization
CLIFFORD_GATES = {"h", "x", "y", "z", "s", "sdg", "cx", "cy", "cz", "swap"}
T_GATES = {"t", "tdg"}
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
TWO_QUBIT_GATES = {"cx", "cy", "cz", "swap", "cp", "crx", "cry", "crz", "rzz"}
ROTATION_GATES = {"rx", "ry", "rz", "p", "cp", "crx", "cry", "crz", "rzz"}
MULTI_QUBIT_GATES = {"toffoli"}
_GATE_BASE_QUBITS: dict[str, int] = {"toffoli": 3}


_CONTROLLED_CLIFFORD_GATES = {"x", "y", "z"}


def _count_gate_operation(
    op: GateOperation, num_controls: int | sp.Expr = 0
) -> GateCount:
    """Count gates for a single gate operation.

    Args:
        op: The gate operation to count
        num_controls: Number of control qubits wrapping this gate (0 = not controlled).
            Can be int (concrete) or sp.Expr (symbolic).

    Returns:
        Gate counts for this operation
    """
    # Get gate name from enum
    gate_name = op.gate_type.name.lower() if op.gate_type else "unknown"

    # Safe symbolic check: "has controls?"
    _has_controls = not (isinstance(num_controls, int) and num_controls == 0)

    if _has_controls:
        is_single_qubit_base = gate_name in SINGLE_QUBIT_GATES
        is_two_qubit_base = gate_name in TWO_QUBIT_GATES

        single = sp.Integer(0)

        if isinstance(num_controls, int):
            # Concrete num_controls: classify based on total qubit count
            if is_single_qubit_base:
                total_qubits = num_controls + 1
            elif is_two_qubit_base:
                total_qubits = num_controls + 2
            else:
                total_qubits = num_controls + _GATE_BASE_QUBITS.get(gate_name, 1)
            two = sp.Integer(1) if total_qubits == 2 else sp.Integer(0)
            multi = sp.Integer(1) if total_qubits > 2 else sp.Integer(0)
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
                two = sp.Piecewise(
                    (sp.Integer(1), sp.Eq(total_qubits, 2)),
                    (sp.Integer(0), True),
                )
                multi = sp.Piecewise(
                    (sp.Integer(1), total_qubits > 2),
                    (sp.Integer(0), True),
                )
            else:
                two = sp.Integer(0)
                multi = sp.Integer(0)

        # Controlled T/Tdg are 2q non-Clifford gates, not simple T gates
        t_count = sp.Integer(0)
        # Only CX, CY, CZ (num_controls=1) are standard Clifford gates
        # CCX, CCZ (num_controls>=2) are NOT Clifford
        if gate_name in _CONTROLLED_CLIFFORD_GATES:
            if isinstance(num_controls, int):
                clifford = sp.Integer(1) if num_controls == 1 else sp.Integer(0)
            else:
                clifford = sp.Piecewise(
                    (sp.Integer(1), sp.Eq(num_controls, 1)),
                    (sp.Integer(0), True),
                )
        else:
            clifford = sp.Integer(0)
        rotation = sp.Integer(1) if gate_name in ROTATION_GATES else sp.Integer(0)
    else:
        is_clifford = gate_name in CLIFFORD_GATES
        is_t_gate = gate_name in T_GATES
        is_single_qubit = gate_name in SINGLE_QUBIT_GATES
        is_two_qubit = gate_name in TWO_QUBIT_GATES
        is_multi_qubit = gate_name in MULTI_QUBIT_GATES
        is_rotation = gate_name in ROTATION_GATES

        single = sp.Integer(1) if is_single_qubit else sp.Integer(0)
        two = sp.Integer(1) if is_two_qubit else sp.Integer(0)
        multi = sp.Integer(1) if is_multi_qubit else sp.Integer(0)
        t_count = sp.Integer(1) if is_t_gate else sp.Integer(0)
        clifford = sp.Integer(1) if is_clifford else sp.Integer(0)
        rotation = sp.Integer(1) if is_rotation else sp.Integer(0)

    return GateCount(
        total=sp.Integer(1),
        single_qubit=single,
        two_qubit=two,
        multi_qubit=multi,
        t_gates=t_count,
        clifford_gates=clifford,
        rotation_gates=rotation,
    )


def _extract_gate_count_from_metadata(meta: ResourceMetadata) -> GateCount:
    """Extract GateCount from ResourceMetadata typed fields.

    Reads the typed gate-count fields directly. Falls back to zero for any
    field that is None. If total_gates is not set, computes it as
    the sum of single_qubit + two_qubit + multi_qubit.

    Emits a UserWarning if total_gates is set but some gate category
    fields are None and the known sub-total is less than total_gates.

    Args:
        meta: ResourceMetadata containing gate information

    Returns:
        GateCount extracted from metadata
    """
    single_qubit = meta.single_qubit_gates or 0
    two_qubit = meta.two_qubit_gates or 0
    multi_qubit = meta.multi_qubit_gates or 0
    t_gates = meta.t_gates or 0
    clifford = meta.clifford_gates or 0
    rotation = meta.rotation_gates or 0

    # Warn if total_gates is set but sub-categories have None gaps
    if meta.total_gates:
        none_fields = [
            name
            for name, val in [
                ("single_qubit_gates", meta.single_qubit_gates),
                ("two_qubit_gates", meta.two_qubit_gates),
                ("multi_qubit_gates", meta.multi_qubit_gates),
            ]
            if val is None
        ]
        if none_fields:
            non_none_sum = single_qubit + two_qubit + multi_qubit
            if non_none_sum < meta.total_gates:
                warnings.warn(
                    f"ResourceMetadata has total_gates={meta.total_gates} "
                    f"but {', '.join(none_fields)} "
                    f"{'is' if len(none_fields) == 1 else 'are'} "
                    f"unspecified (None) and treated as 0. "
                    f"The known sub-total ({non_none_sum}) is less than "
                    f"total_gates ({meta.total_gates}). "
                    f"Set gate category fields explicitly for accurate "
                    f"sub-category counts.",
                    UserWarning,
                    stacklevel=2,
                )

    total = meta.total_gates
    if not total:
        total = single_qubit + two_qubit + multi_qubit

    return GateCount(
        total=sp.Integer(total),
        single_qubit=sp.Integer(single_qubit),
        two_qubit=sp.Integer(two_qubit),
        multi_qubit=sp.Integer(multi_qubit),
        t_gates=sp.Integer(t_gates),
        clifford_gates=sp.Integer(clifford),
        rotation_gates=sp.Integer(rotation),
    )


def _count_composite_gate(
    op: CompositeGateOperation,
    block: Block | None,
    loop_context: LoopContext,
    call_context: dict[str, Any],
    loop_var_symbols: dict[str, sp.Symbol],
    num_controls: int | sp.Expr = 0,
    parent_blocks: list | None = None,
) -> GateCount:
    """Count gates in a CompositeGateOperation.

    Priority order:
    1. Use resource_metadata if available
    2. Use implementation (decomposition) if available
    3. Compute from known gate types (QFT, IQFT, QPE)
    4. Raise error if none available

    Args:
        op: CompositeGateOperation to count
        block: Current block for context
        loop_context: Loop context for symbolic evaluation
        call_context: Call context for parameter resolution
        loop_var_symbols: Loop variable symbols

    Returns:
        GateCount for the composite gate

    Raises:
        ValueError: If no resource information or implementation available
    """
    from qamomile.circuit.ir.block_value import BlockValue
    from qamomile.circuit.ir.operation.composite_gate import CompositeGateType

    # Priority 1: Use resource_metadata
    if op.resource_metadata is not None:
        gate_count = _extract_gate_count_from_metadata(op.resource_metadata)
        # Record oracle call for stub gates (gates without implementation)
        if not op.has_implementation:
            gate_name = op.custom_name or op.gate_type.value
            gate_count.oracle_calls[gate_name] = sp.Integer(1)
            qc = op.resource_metadata.query_complexity
            if qc is not None:
                gate_count.oracle_queries[gate_name] = sp.Integer(qc)
        return gate_count

    # Priority 2: Use implementation (decomposition)
    if op.has_implementation and op.implementation is not None:
        impl_block = op.implementation
        if isinstance(impl_block, BlockValue):
            # Build call context for implementation parameters
            new_call_context = call_context.copy()

            # Map operands to implementation inputs
            # operands[0] is BlockValue, operands[1:] are qubits/params
            for formal_idx, formal_input in enumerate(impl_block.input_values):
                if formal_idx + 1 < len(op.operands):
                    actual_arg = op.operands[formal_idx + 1]
                    resolved_arg = value_to_expr(
                        actual_arg, block, call_context, loop_var_symbols,
                        parent_blocks=parent_blocks,
                    )
                    new_call_context[formal_input.uuid] = resolved_arg

            # Recursively count gates in implementation
            return _count_from_operations(
                impl_block.operations,
                block=impl_block,
                parent_blocks=[],
                loop_context=loop_context,
                call_context=new_call_context,
                loop_var_symbols=loop_var_symbols,
                num_controls=num_controls,
            )

    # Priority 3: Compute from known gate types
    if op.gate_type in (CompositeGateType.QFT, CompositeGateType.IQFT):
        # QFT/IQFT: n H gates + n(n-1)/2 CP gates + n//2 SWAP gates
        # Determine n from the number of target qubit operands
        target_qubits = op.target_qubits
        n_qubits = len(target_qubits)

        if n_qubits > 0:
            # We have concrete qubits - check if they came from an array with symbolic size
            # Try to trace back to the parent array dimension
            from qamomile.circuit.ir.value import ArrayValue

            n = None
            first_qubit = target_qubits[0]

            # Check if this qubit has array metadata
            if hasattr(first_qubit, "parent_array") and first_qubit.parent_array:
                parent = first_qubit.parent_array
                if isinstance(parent, ArrayValue) and parent.shape:
                    # Get the symbolic dimension from the parent array
                    n = value_to_expr(
                        parent.shape[0], block, call_context, loop_var_symbols,
                        parent_blocks=parent_blocks,
                    )

            # If we couldn't trace to parent array, use the concrete count
            if n is None:
                n = sp.Integer(n_qubits)
        else:
            # No operands - this happens when QPE is used with symbolic array sizes
            # The IQFT was created with n=0 because the size wasn't known at build time
            # We need to infer the size from context - look for array allocations in the block
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
                                    parent_blocks=parent_blocks,
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

        h_gates = n
        cp_gates = n * (n - 1) / 2
        swap_gates = n // 2
        total = h_gates + cp_gates + swap_gates

        return GateCount(
            total=total,
            single_qubit=h_gates,
            two_qubit=cp_gates + swap_gates,
            multi_qubit=sp.Integer(0),
            t_gates=sp.Integer(0),
            clifford_gates=h_gates + swap_gates,  # H and SWAP are Clifford
            rotation_gates=cp_gates,  # CP gates are rotation gates
        )

    # Priority 4: Error if no resource info or implementation
    gate_name = op.custom_name or op.gate_type.value
    raise ValueError(
        f"Cannot estimate resources for CompositeGateOperation '{gate_name}': "
        f"No resource_metadata or implementation available. "
        f"Please provide either resource_metadata or set has_implementation=True."
    )


def value_to_expr(
    v: Any,
    block: Block | None = None,
    call_context: dict[str, Any] | None = None,
    loop_var_symbols: dict[str, sp.Symbol] | None = None,
    parent_blocks: list | None = None,
) -> sp.Expr:
    """Convert Value to SymPy expression, tracing operations if needed.

    Args:
        v: Value or constant to convert
        block: Optional Block for operation tracing
        call_context: Optional mapping from Value UUIDs to actual argument Values or SymPy expressions
        loop_var_symbols: Optional mapping from Value names to SymPy loop variable symbols
        parent_blocks: Optional list of ancestor blocks for scope chain resolution

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
        return value_to_expr(resolved_val, block, call_context, loop_var_symbols,
                             parent_blocks=parent_blocks)

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
                parent_blocks=parent_blocks,
            )
            if traced is not None:
                return traced

        # Search parent blocks for scope chain resolution
        if parent_blocks:
            for pb in reversed(parent_blocks):
                traced = _trace_value_operation(
                    v,
                    pb,
                    visited=set(),
                    call_context=call_context,
                    loop_var_symbols=loop_var_symbols,
                    parent_blocks=parent_blocks,
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
    parent_blocks: list | None = None,
) -> sp.Expr | None:
    """Trace backward through operations to find what produced this Value.

    Args:
        v: Value to trace
        block: Block containing the operations
        visited: Set of visited Value IDs to prevent infinite recursion
        call_context: Optional mapping from Value UUIDs to actual argument Values
        loop_var_symbols: Optional mapping from Value names to SymPy loop variable symbols
        parent_blocks: Optional list of ancestor blocks for scope chain resolution

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
                    op.operands[0], block, call_context, loop_var_symbols,
                    parent_blocks=parent_blocks,
                )
                right = value_to_expr(
                    op.operands[1], block, call_context, loop_var_symbols,
                    parent_blocks=parent_blocks,
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
                    quotient = sp.simplify(left / right)
                    # For symbolic integer expressions (e.g., 2**m / 2**i = 2**(m-i)),
                    # avoid wrapping in floor() to enable symbolic simplification.
                    # This is safe because quantum circuit IR uses FLOORDIV between
                    # integer types where the result is typically exact.
                    if isinstance(quotient, (sp.Integer, sp.Symbol, sp.Pow)):
                        return quotient
                    return sp.floor(left / right)
                elif op.kind == BinOpKind.POW:
                    return left**right

            elif isinstance(op, CompOp):
                left = value_to_expr(
                    op.operands[0], block, call_context, loop_var_symbols,
                    parent_blocks=parent_blocks,
                )
                right = value_to_expr(
                    op.operands[1], block, call_context, loop_var_symbols,
                    parent_blocks=parent_blocks,
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


def _apply_sum_to_count(
    count: GateCount,
    loop_var: sp.Symbol,
    start: sp.Expr,
    stop: sp.Expr,
    step: sp.Expr = sp.Integer(1),
) -> GateCount:
    """Apply SymPy Sum to all fields of a GateCount.

    For forward loops (step > 0): Sum over (loop_var, start, stop-1)
    For reverse loops (step < 0): Sum over (loop_var, stop+1, start)
    """
    # Determine correct bounds based on step direction
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

    return GateCount(
        total=Sum(count.total, (loop_var, lower, upper)).doit(),
        single_qubit=Sum(count.single_qubit, (loop_var, lower, upper)).doit(),
        two_qubit=Sum(count.two_qubit, (loop_var, lower, upper)).doit(),
        multi_qubit=Sum(count.multi_qubit, (loop_var, lower, upper)).doit(),
        t_gates=Sum(count.t_gates, (loop_var, lower, upper)).doit(),
        clifford_gates=Sum(count.clifford_gates, (loop_var, lower, upper)).doit(),
        rotation_gates=Sum(count.rotation_gates, (loop_var, lower, upper)).doit(),
        oracle_calls={
            name: Sum(val, (loop_var, lower, upper)).doit()
            for name, val in count.oracle_calls.items()
        },
        oracle_queries={
            name: Sum(val, (loop_var, lower, upper)).doit()
            for name, val in count.oracle_queries.items()
        },
    )



def _count_from_operations(
    operations: list[Operation],
    block: Block | None = None,
    parent_blocks: list | None = None,
    loop_context: LoopContext | None = None,
    call_context: dict[str, Any] | None = None,
    loop_var_symbols: dict[str, sp.Symbol] | None = None,
    num_controls: int | sp.Expr = 0,
) -> GateCount:
    """Count gates from a list of operations.

    Args:
        operations: List of operations to analyze
        block: Optional Block for value tracing
        parent_blocks: Optional list of ancestor blocks for scope chain resolution
        loop_context: Stack of outer loop variables for nested loop analysis
        call_context: Optional mapping from Value UUIDs to actual argument Values
        loop_var_symbols: Optional mapping from Value names to SymPy loop variable symbols

    Returns:
        Total gate counts as SymPy expressions
    """
    if loop_context is None:
        loop_context = []
    if call_context is None:
        call_context = {}
    if loop_var_symbols is None:
        loop_var_symbols = {}

    count = GateCount.zero()

    for op in operations:
        match op:
            case GateOperation():
                count = count + _count_gate_operation(op, num_controls=num_controls)

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

                    # Build parent block chain for scope resolution
                    new_parent_blocks = list(parent_blocks or [])
                    if block is not None:
                        new_parent_blocks.append(block)

                    # Convert values to SymPy expressions, tracing in local block if value not in outer block
                    # Try outer block first, then local block
                    def convert_with_local_trace(v):
                        # First try with outer block
                        if block is not None:
                            result = value_to_expr(
                                v, block, call_context, new_loop_var_symbols,
                                parent_blocks=parent_blocks,
                            )
                            # If we got a symbol, try local block too
                            from qamomile.circuit.ir.value import Value

                            if (
                                isinstance(v, Value)
                                and not v.is_constant()
                                and not v.is_parameter()
                            ):
                                local_result = value_to_expr(
                                    v, local_block, call_context, new_loop_var_symbols,
                                    parent_blocks=new_parent_blocks,
                                )
                                # Use local result if it's not just the symbol name
                                if local_result != sp.Symbol(
                                    v.name, integer=True, positive=True
                                ):
                                    return local_result
                            return result
                        else:
                            return value_to_expr(
                                v, local_block, call_context, new_loop_var_symbols,
                                parent_blocks=new_parent_blocks,
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

                    # Recursively count inner operations with enhanced context
                    # Pass local_block so inner operations can trace values defined in this scope
                    # Pass parent_blocks so ancestor scope values remain accessible
                    inner_count = _count_from_operations(
                        op.operations,
                        block=local_block,
                        parent_blocks=new_parent_blocks,
                        loop_context=new_context,
                        call_context=call_context,
                        loop_var_symbols=new_loop_var_symbols,
                        num_controls=num_controls,
                    )

                    # Check if inner count contains the current loop variable
                    # This happens when inner loops have bounds that depend on this variable
                    # Also check oracle_calls (stubs have total=0 but oracle_calls
                    # may depend on the loop variable)
                    all_free = inner_count.total.free_symbols
                    all_free = all_free | inner_count.two_qubit.free_symbols
                    all_free = all_free | inner_count.multi_qubit.free_symbols
                    for val in inner_count.oracle_calls.values():
                        all_free = all_free | val.free_symbols
                    for val in inner_count.oracle_queries.values():
                        all_free = all_free | val.free_symbols
                    if loop_var_symbol in all_free:
                        # The inner count varies with our loop variable
                        # We need to sum it over all loop iterations
                        # Sum(inner_count, (loop_var, start, stop-1))
                        count_expr = _apply_sum_to_count(
                            inner_count,
                            loop_var_symbol,
                            start_expr,
                            stop_expr,
                            step_expr,
                        )
                    else:
                        # Normal case: inner count is constant per iteration
                        # Multiply by number of iterations
                        iterations = (stop_expr - start_expr) / step_expr
                        count_expr = inner_count * iterations

                    count = count + count_expr
                else:
                    # Fallback
                    fallback_parent_blocks = list(parent_blocks or [])
                    if block is not None:
                        fallback_parent_blocks.append(block)
                    iterations = sp.Symbol("iter", integer=True, positive=True)
                    inner_count = _count_from_operations(
                        op.operations,
                        block=None,
                        parent_blocks=fallback_parent_blocks,
                        loop_context=loop_context,
                        call_context=call_context,
                        loop_var_symbols=loop_var_symbols,
                        num_controls=num_controls,
                    )
                    count = count + (inner_count * iterations)

            case WhileOperation():
                # Conservative: assume constant iterations
                # In practice, would need bounds analysis
                while_parent_blocks = list(parent_blocks or [])
                if block is not None:
                    while_parent_blocks.append(block)
                inner_count = _count_from_operations(
                    op.operations,
                    block=None,
                    parent_blocks=while_parent_blocks,
                    loop_context=loop_context,
                    call_context=call_context,
                    loop_var_symbols=loop_var_symbols,
                    num_controls=num_controls,
                )
                iterations = sp.Symbol("|while|", integer=True, positive=True)
                count = count + (inner_count * iterations)

            case IfOperation():
                # Take maximum of both branches
                if_parent_blocks = list(parent_blocks or [])
                if block is not None:
                    if_parent_blocks.append(block)
                true_count = _count_from_operations(
                    op.true_operations,
                    block=None,
                    parent_blocks=if_parent_blocks,
                    loop_context=loop_context,
                    call_context=call_context,
                    loop_var_symbols=loop_var_symbols,
                    num_controls=num_controls,
                )
                false_count = _count_from_operations(
                    op.false_operations,
                    block=None,
                    parent_blocks=if_parent_blocks,
                    loop_context=loop_context,
                    call_context=call_context,
                    loop_var_symbols=loop_var_symbols,
                    num_controls=num_controls,
                )
                count = count + true_count.max(false_count)

            case ForItemsOperation():
                items_parent_blocks = list(parent_blocks or [])
                if block is not None:
                    items_parent_blocks.append(block)
                inner_count = _count_from_operations(
                    op.operations,
                    block=None,
                    parent_blocks=items_parent_blocks,
                    loop_context=loop_context,
                    call_context=call_context,
                    loop_var_symbols=loop_var_symbols,
                    num_controls=num_controls,
                )
                dict_operand = op.operands[0]
                if (
                    hasattr(dict_operand, "is_parameter")
                    and dict_operand.is_parameter()
                ):
                    dict_name = dict_operand.parameter_name() or dict_operand.name
                else:
                    dict_name = dict_operand.name
                cardinality_sym = sp.Symbol(
                    f"|{dict_name}|", integer=True, positive=True
                )
                count = count + inner_count * cardinality_sym

            case CallBlockOperation():
                # Recursively count gates in called block
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
                                actual_arg, block, call_context, loop_var_symbols,
                                parent_blocks=parent_blocks,
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
                                        parent_blocks=parent_blocks,
                                    )
                                    new_call_context[dim_formal.uuid] = resolved_dim

                    # Recursively count with context
                    inner_count = _count_from_operations(
                        called_block.operations,
                        block=called_block,
                        parent_blocks=[],
                        loop_context=loop_context,
                        call_context=new_call_context,
                        loop_var_symbols=loop_var_symbols,
                        num_controls=num_controls,
                    )
                    count = count + inner_count

            case ControlledUOperation():
                # Treat ControlledUOperation as an opaque gate:
                # total=1, rotation=0, clifford=0
                # Only classify by qubit width (two_qubit / multi_qubit)
                from qamomile.circuit.ir.block_value import BlockValue

                # Resolve num_controls
                if op.is_symbolic_num_controls:
                    nc = value_to_expr(
                        op.num_controls, block, call_context, loop_var_symbols,
                        parent_blocks=parent_blocks,
                    )
                else:
                    nc = op.num_controls

                # Determine number of target qubits
                controlled_block = op.block
                if isinstance(controlled_block, BlockValue):
                    num_targets = sum(
                        1 for inp in controlled_block.input_values
                        if inp.type.is_quantum()
                    )
                else:
                    target_ops = op.target_operands if hasattr(op, 'target_operands') else []
                    num_targets = len(target_ops) if target_ops else 1

                # Classify by qubit width
                if isinstance(nc, int):
                    total_qubits = nc + num_targets
                    two = sp.Integer(1) if total_qubits == 2 else sp.Integer(0)
                    multi = sp.Integer(1) if total_qubits > 2 else sp.Integer(0)
                else:
                    total_qubits = nc + num_targets
                    two = sp.Piecewise(
                        (sp.Integer(1), sp.Eq(total_qubits, 2)),
                        (sp.Integer(0), True),
                    )
                    multi = sp.Piecewise(
                        (sp.Integer(1), total_qubits > 2),
                        (sp.Integer(0), True),
                    )

                gate_count = GateCount(
                    total=sp.Integer(1),
                    single_qubit=sp.Integer(0),
                    two_qubit=two,
                    multi_qubit=multi,
                    t_gates=sp.Integer(0),
                    clifford_gates=sp.Integer(0),
                    rotation_gates=sp.Integer(0),
                )
                count = count + gate_count

            case CompositeGateOperation():
                composite_count = _count_composite_gate(
                    op,
                    block,
                    loop_context,
                    call_context,
                    loop_var_symbols,
                    num_controls=num_controls,
                    parent_blocks=parent_blocks,
                )
                count = count + composite_count

            case _:
                continue

    return count.simplify()


def count_gates(block: BlockValue | Block | list[Operation]) -> GateCount:
    """Count gates in a quantum circuit.

    This function analyzes operations and returns algebraic gate counts
    using SymPy expressions. Counts may contain symbols for parametric
    problem sizes (e.g., loop bounds, array dimensions).

    Supports:
    - GateOperation: Single gate counts
    - ForOperation: Multiplies inner count by iterations
    - IfOperation: Takes maximum of branches
    - CallBlockOperation: Recursively counts called blocks
    - ControlledUOperation: Recursively counts unitary block

    Args:
        block: BlockValue, Block, or list of Operations to analyze

    Returns:
        GateCount with total, single_qubit, two_qubit, t_gates, clifford_gates

    Example:
        >>> from qamomile.circuit.estimator import count_gates
        >>> count = count_gates(my_circuit.block)
        >>> print(count.total)  # e.g., "2*n + 5"
        >>> print(count.t_gates)  # e.g., "n"
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

    return _count_from_operations(
        ops, block_ref, parent_blocks=[], loop_context=[], call_context={},
        loop_var_symbols={},
    )
