"""Circuit depth estimation for quantum circuits.

This module estimates circuit depth by analyzing operation dependencies.
Depth is expressed as SymPy expressions for parametric problem sizes.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import sympy as sp
from sympy import Sum

from qamomile.circuit.ir.operation.arithmetic_operations import (
    CompOp,
    CondOp,
    NotOp,
)
from qamomile.circuit.ir.operation.call_block_ops import CallBlockOperation
from qamomile.circuit.ir.operation.composite_gate import CompositeGateOperation
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    IfOperation,
    WhileOperation,
)
from qamomile.circuit.ir.operation.gate import (
    ControlledUOperation,
    GateOperation,
    MeasureOperation,
    MeasureQFixedOperation,
    MeasureVectorOperation,
)
from qamomile.circuit.ir.operation.operation import Operation

from ._catalog import (
    classify_controlled_u,
    classify_gate,
    extract_depth_from_metadata,
    gate_count_to_depth,
    qft_iqft_depth,
)
from ._engine import (
    _LocalBlock,
    build_for_items_scope,
    build_for_loop_scope,
    build_if_scopes,
    build_while_scope,
    resolve_composite_gate,
    resolve_controlled_u,
    resolve_for_items_cardinality,
)
from ._loop_executor import (
    find_parametric_symbols,
    interpolate_scalar,
    symbolic_iterations,
)
from ._resolver import ExprResolver, UnresolvedValueError
from ._utils import _strip_nonneg_max

if TYPE_CHECKING:
    from qamomile.circuit.ir.block import Block
    from qamomile.circuit.ir.block_value import BlockValue


class _UnresolvableForOpError(Exception):
    """Raised when concrete simulation encounters unresolvable ForOperation bounds."""


# ------------------------------------------------------------------ #
#  CircuitDepth dataclass (public API — unchanged)                    #
# ------------------------------------------------------------------ #


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
            total_depth=_strip_nonneg_max(sp.simplify(self.total_depth)),
            t_depth=_strip_nonneg_max(sp.simplify(self.t_depth)),
            two_qubit_depth=_strip_nonneg_max(sp.simplify(self.two_qubit_depth)),
            multi_qubit_depth=_strip_nonneg_max(sp.simplify(self.multi_qubit_depth)),
            rotation_depth=_strip_nonneg_max(sp.simplify(self.rotation_depth)),
        )

    def substitute(self, subs_dict: dict[sp.Symbol, int | sp.Expr]) -> CircuitDepth:
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

    @staticmethod
    def apply_gate_to_qubits(
        qubit_depths: dict[str, CircuitDepth],
        qubit_ids: list[str],
        gate_depth: CircuitDepth,
    ) -> None:
        """Apply gate depth to qubits with per-field selective propagation.

        For each depth field independently:
        - If the gate contributes (not definitely zero): max across all
          involved qubits + gate contribution, assigned to all qubits.
        - If the gate does NOT contribute (definitely zero): each qubit
          retains its own current value for that field.
        """
        from dataclasses import fields as dc_fields

        if not qubit_ids:
            return

        field_names = [f.name for f in dc_fields(CircuitDepth)]
        current = [qubit_depths.get(qid, CircuitDepth.zero()) for qid in qubit_ids]

        propagated: dict[str, sp.Expr] = {}
        no_prop: set[str] = set()

        for fname in field_names:
            gate_contrib = getattr(gate_depth, fname)
            if gate_contrib.is_zero is True:
                no_prop.add(fname)
            else:
                max_val = getattr(current[0], fname)
                for cd in current[1:]:
                    max_val = sp.Max(max_val, getattr(cd, fname))
                propagated[fname] = max_val + gate_contrib

        for i, qid in enumerate(qubit_ids):
            new_fields = {}
            for fname in field_names:
                if fname in no_prop:
                    new_fields[fname] = getattr(current[i], fname)
                else:
                    new_fields[fname] = propagated[fname]
            qubit_depths[qid] = CircuitDepth(**new_fields)


# ------------------------------------------------------------------ #
#  Per-qubit depth tracking                                           #
# ------------------------------------------------------------------ #

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


_MEASURE_UNIT = CircuitDepth(
    total_depth=sp.Integer(1),
    t_depth=sp.Integer(0),
    two_qubit_depth=sp.Integer(0),
    multi_qubit_depth=sp.Integer(0),
    rotation_depth=sp.Integer(0),
)


# ------------------------------------------------------------------ #
#  Qubit name helpers                                                 #
# ------------------------------------------------------------------ #


def _qubit_key(v: Any, resolver: ExprResolver) -> str:
    """Normalized qubit key. Array indices resolved via ExprResolver."""
    if (
        hasattr(v, "parent_array")
        and v.parent_array is not None
        and hasattr(v, "element_indices")
        and v.element_indices
    ):
        idx_parts = []
        for idx in v.element_indices:
            expr = resolver.resolve(idx)
            idx_parts.append(str(expr))
        return f"{v.parent_array.name}[{','.join(idx_parts)}]"
    return v.name


def _collect_all_qubit_names(
    operations: list[Operation],
    resolver: ExprResolver | None = None,
) -> set[str]:
    """Collect qubit names from all operation types."""
    names: set[str] = set()

    def _name(v: Any) -> str:
        if resolver is not None:
            return _qubit_key(v, resolver)
        if hasattr(v, "parent_array") and v.parent_array is not None:
            return v.parent_array.name
        return v.name

    for op in operations:
        if isinstance(op, GateOperation):
            for v in op.operands:
                names.add(_name(v))
        elif isinstance(op, CallBlockOperation):
            for v in op.operands[1:]:
                if (
                    hasattr(v, "type")
                    and hasattr(v.type, "is_quantum")
                    and v.type.is_quantum()
                ):
                    if hasattr(v, "parent_array") and v.parent_array is not None:
                        names.add(v.parent_array.name)
                    else:
                        names.add(v.name)
        elif isinstance(op, ControlledUOperation):
            for v in op.operands[1:]:
                if hasattr(v, "name"):
                    names.add(_name(v))
        elif isinstance(op, CompositeGateOperation):
            for v in op.operands:
                if hasattr(v, "name") and hasattr(v, "type") and v.type.is_quantum():
                    names.add(_name(v))
        elif isinstance(op, MeasureOperation):
            names.add(_name(op.operands[0]))
        elif isinstance(op, (MeasureVectorOperation, MeasureQFixedOperation)):
            names.add(op.operands[0].name)
        elif isinstance(op, ForOperation):
            names.update(_collect_all_qubit_names(op.operations, resolver))
        elif isinstance(op, WhileOperation):
            names.update(_collect_all_qubit_names(op.operations, resolver))
        elif isinstance(op, IfOperation):
            names.update(_collect_all_qubit_names(op.true_operations, resolver))
            names.update(_collect_all_qubit_names(op.false_operations, resolver))
        elif isinstance(op, ForItemsOperation):
            names.update(_collect_all_qubit_names(op.operations, resolver))
    return names


# ------------------------------------------------------------------ #
#  Call-boundary helpers                                              #
# ------------------------------------------------------------------ #


def _build_qubit_name_map(
    called_block: Any,
    op_operands: list,
    resolver: ExprResolver,
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
                qubit_name_map[formal_input.name] = _qubit_key(actual_arg, resolver)
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
        else:
            base = actual_name.split("[")[0]
            if base != actual_name and base in qubit_depths:
                local_depths[formal_name] = qubit_depths[base]
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
    for qid, depth in local_depths.items():
        base = qid.split("[")[0]
        if base not in mapped_formals:
            qubit_depths[qid] = depth


# ------------------------------------------------------------------ #
#  Dict resolution helper                                             #
# ------------------------------------------------------------------ #


def _resolve_dict_entries_for_depth(
    dict_value: Any,
    dict_bindings: dict[str, Any],
) -> list[tuple[Any, Any]] | None:
    """Resolve DictValue to concrete (key, value) pairs for depth estimation."""
    if hasattr(dict_value, "params") and "bound_data" in dict_value.params:
        bound = dict_value.params["bound_data"]
        if isinstance(bound, dict):
            return list(bound.items())
        elif hasattr(bound, "items"):
            return list(bound.items())
        return bound

    if hasattr(dict_value, "is_parameter") and dict_value.is_parameter():
        param_name = dict_value.parameter_name()
        if param_name and param_name in dict_bindings:
            bound = dict_bindings[param_name]
            if isinstance(bound, dict):
                return list(bound.items())
            elif hasattr(bound, "items"):
                return list(bound.items())
            return bound

    if hasattr(dict_value, "name") and dict_value.name in dict_bindings:
        bound = dict_bindings[dict_value.name]
        if isinstance(bound, dict):
            return list(bound.items())
        elif hasattr(bound, "items"):
            return list(bound.items())
        return bound

    return None


# ------------------------------------------------------------------ #
#  SymPy Sum helper                                                   #
# ------------------------------------------------------------------ #


def _apply_sum_to_depth(
    depth: CircuitDepth,
    loop_var: sp.Symbol,
    start: sp.Expr,
    stop: sp.Expr,
    step: sp.Expr = sp.Integer(1),
) -> CircuitDepth:
    """Apply SymPy Sum to all fields of a CircuitDepth."""
    is_negative_step = False
    try:
        is_negative_step = bool(step < 0)
    except TypeError:
        pass

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
        multi_qubit_depth=Sum(depth.multi_qubit_depth, (loop_var, lower, upper)).doit(),
        rotation_depth=Sum(depth.rotation_depth, (loop_var, lower, upper)).doit(),
    )


# ------------------------------------------------------------------ #
#  Interpolation helper                                               #
# ------------------------------------------------------------------ #


def _interpolate_depth_from_samples(
    samples: dict[int, CircuitDepth],
    sym: sp.Symbol,
) -> CircuitDepth:
    """Interpolate symbolic CircuitDepth from concrete samples."""
    from dataclasses import fields as dc_fields

    field_names = [f.name for f in dc_fields(CircuitDepth)]
    sorted_ns = sorted(samples.keys())

    result = {}
    for fname in field_names:
        pts = [(n, getattr(samples[n], fname)) for n in sorted_ns]
        result[fname] = interpolate_scalar(pts, sym)

    return CircuitDepth(**result)


# ------------------------------------------------------------------ #
#  Shared operation helpers                                           #
# ------------------------------------------------------------------ #


def _concretize_resolver(
    resolver: ExprResolver,
    symbol: sp.Symbol,
    value: int,
) -> ExprResolver:
    """Create resolver with a parametric symbol substituted to a concrete value.

    Maps the parameter UUID → concrete value in context (step 3 of
    ExprResolver._resolve) so that ``resolve_concrete()`` finds the value
    before reaching the ``is_parameter()`` check at step 5.
    """
    from qamomile.circuit.ir.value import ArrayValue

    subs = {symbol: value}
    ctx = {
        uuid: (expr.subs(subs) if isinstance(expr, sp.Basic) else expr)
        for uuid, expr in resolver._context.items()
    }
    sym_name = str(symbol)
    concrete_val = sp.Integer(value)

    # Map parameter UUIDs matching the symbol name to the concrete value.
    # This covers block.input_values and parent_blocks (for _LocalBlock).
    def _scan_input_values(block: Any) -> None:
        if not hasattr(block, "input_values"):
            return
        for iv in block.input_values:
            if hasattr(iv, "is_parameter") and iv.is_parameter():
                pname = iv.parameter_name() or iv.name
                if pname == sym_name:
                    ctx[iv.uuid] = concrete_val
                    # Also map array shape dimension UUIDs
                    if isinstance(iv, ArrayValue):
                        for dim in iv.shape:
                            if hasattr(dim, "uuid"):
                                ctx[dim.uuid] = concrete_val

    _scan_input_values(resolver._block)
    for pb in resolver._parent_blocks:
        _scan_input_values(pb)

    lvn = dict(resolver._loop_var_names)
    lvn[sym_name] = concrete_val
    return ExprResolver(
        block=resolver._block,
        context=ctx,
        loop_var_names=lvn,
        parent_blocks=list(resolver._parent_blocks),
    )


def _expand_array_qubit_ids(
    raw_names: list[str],
    qubit_depths: QubitDepthMap,
) -> list[str]:
    """Expand array base names to element-level qubit IDs if present."""
    qubit_ids: list[str] = []
    for name in raw_names:
        matched = [k for k in qubit_depths if k.startswith(name + "[")]
        if matched:
            qubit_ids.extend(matched)
        else:
            qubit_ids.append(name)
    return qubit_ids


def _handle_composite_depth(
    op: CompositeGateOperation,
    resolver: ExprResolver,
    num_controls: int | sp.Expr,
) -> CircuitDepth:
    """Resolve CompositeGateOperation to CircuitDepth."""
    res = resolve_composite_gate(op, resolver)

    if res.kind == "metadata":
        return extract_depth_from_metadata(res.metadata, res.num_control_qubits)

    if res.kind == "implementation":
        return _compute_sequential_depth(
            res.impl_block.operations,
            res.impl_resolver,
            num_controls,
        )

    if res.kind == "qft_iqft":
        return qft_iqft_depth(res.n_qubits)

    raise ValueError(res.error_message)


def _get_controlled_u_qubit_names(
    op: ControlledUOperation,
    resolver: ExprResolver,
) -> list[str]:
    """Get all qubit names involved in a ControlledUOperation."""
    if op.has_index_spec:
        vector = op.operands[1]
        vector_name = vector.name
        if vector.shape:
            shape_expr = resolver.resolve(vector.shape[0])
            if shape_expr.is_number:
                return [f"{vector_name}[{i}]" for i in range(int(shape_expr))]
            return [vector_name]
        return [vector_name]

    return [_qubit_key(v, resolver) for v in op.control_operands] + [
        _qubit_key(v, resolver) for v in op.target_operands
    ]


def _handle_controlled_u_depth(
    op: ControlledUOperation,
    resolver: ExprResolver,
) -> CircuitDepth:
    """Compute CircuitDepth for a ControlledUOperation."""
    nc, num_targets = resolve_controlled_u(op, resolver)
    # Convert sp.Integer to int for clean classification
    if isinstance(nc, sp.Integer):
        nc = int(nc)
    return gate_count_to_depth(classify_controlled_u(nc, num_targets))


# ------------------------------------------------------------------ #
#  Concrete parallel depth simulation                                 #
# ------------------------------------------------------------------ #


def _simulate_parallel_depth_concrete(
    operations: list[Operation],
    qubit_depths: QubitDepthMap,
    resolver: ExprResolver,
    num_controls: int | sp.Expr = 0,
    value_depths: dict[str, CircuitDepth] | None = None,
    *,
    dict_bindings: dict[str, Any] | None = None,
) -> None:
    """Simulate parallel depth with fully concrete variable values.

    Uses ExprResolver for value tracing (no explicit BinOp handling needed).
    For inner ForOperations with unresolvable bounds, raises
    _UnresolvableForOpError.
    """
    from qamomile.circuit.ir.block_value import BlockValue

    for op in operations:
        if isinstance(op, GateOperation):
            gate_depth = gate_count_to_depth(
                classify_gate(op, num_controls=num_controls)
            )
            qubit_ids = [_qubit_key(v, resolver) for v in op.operands]
            CircuitDepth.apply_gate_to_qubits(qubit_depths, qubit_ids, gate_depth)

        elif isinstance(op, MeasureOperation):
            qubit_id = _qubit_key(op.operands[0], resolver)
            current = qubit_depths.get(qubit_id, CircuitDepth.zero())
            new_depth = current + _MEASURE_UNIT
            qubit_depths[qubit_id] = new_depth
            if value_depths is not None and op.results:
                value_depths[op.results[0].uuid] = new_depth

        elif isinstance(op, MeasureVectorOperation):
            arr_name = op.operands[0].name
            matched_qids = [
                k for k in qubit_depths if k.startswith(arr_name + "[") or k == arr_name
            ]
            if not matched_qids:
                matched_qids = [arr_name]
            CircuitDepth.apply_gate_to_qubits(qubit_depths, matched_qids, _MEASURE_UNIT)
            if value_depths is not None and op.results:
                vd = CircuitDepth.zero()
                for qid in matched_qids:
                    vd = vd.max(qubit_depths[qid])
                value_depths[op.results[0].uuid] = vd

        elif isinstance(op, MeasureQFixedOperation):
            qfixed_name = op.operands[0].name
            matched_qids = [
                k
                for k in qubit_depths
                if k.startswith(qfixed_name + "[") or k == qfixed_name
            ]
            if not matched_qids:
                matched_qids = [qfixed_name]
            CircuitDepth.apply_gate_to_qubits(qubit_depths, matched_qids, _MEASURE_UNIT)
            if value_depths is not None and op.results:
                vd = CircuitDepth.zero()
                for qid in matched_qids:
                    vd = vd.max(qubit_depths[qid])
                value_depths[op.results[0].uuid] = vd

        elif isinstance(op, NotOp):
            if value_depths is not None:
                input_depth = value_depths.get(op.input.uuid, CircuitDepth.zero())
                value_depths[op.output.uuid] = input_depth

        elif isinstance(op, CondOp):
            if value_depths is not None:
                lhs_depth = value_depths.get(op.operands[0].uuid, CircuitDepth.zero())
                rhs_depth = value_depths.get(op.operands[1].uuid, CircuitDepth.zero())
                value_depths[op.results[0].uuid] = lhs_depth.max(rhs_depth)

        elif isinstance(op, CompOp):
            if value_depths is not None:
                lhs_depth = value_depths.get(op.operands[0].uuid, CircuitDepth.zero())
                rhs_depth = value_depths.get(op.operands[1].uuid, CircuitDepth.zero())
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
            ) | _collect_all_qubit_names(op.false_operations)
            for qid in branch_qubits:
                current = qubit_depths.get(qid, CircuitDepth.zero())
                true_depths[qid] = current.max(condition_depth)
                false_depths[qid] = current.max(condition_depth)

            true_child, false_child = build_if_scopes(op, resolver)
            _simulate_parallel_depth_concrete(
                op.true_operations,
                true_depths,
                true_child,
                num_controls,
                value_depths,
                dict_bindings=dict_bindings,
            )
            _simulate_parallel_depth_concrete(
                op.false_operations,
                false_depths,
                false_child,
                num_controls,
                value_depths,
                dict_bindings=dict_bindings,
            )

            all_qids = set(true_depths) | set(false_depths)
            for qid in all_qids:
                qubit_depths[qid] = true_depths.get(qid, CircuitDepth.zero()).max(
                    false_depths.get(qid, CircuitDepth.zero())
                )

            for phi_op in op.phi_ops:
                tv_depth = qubit_depths.get(phi_op.true_value.name, CircuitDepth.zero())
                fv_depth = qubit_depths.get(
                    phi_op.false_value.name, CircuitDepth.zero()
                )
                qubit_depths[phi_op.output.name] = tv_depth.max(fv_depth)

        elif isinstance(op, ForItemsOperation):
            if not op.operands:
                continue

            dict_operand = op.operands[0]
            entries = _resolve_dict_entries_for_depth(dict_operand, dict_bindings or {})

            if entries is not None:
                for key, value in entries:
                    extra_lvn: dict[str, Any] = {}
                    if len(op.key_vars) > 1:
                        for ki, kv_name in enumerate(op.key_vars):
                            kval = key[ki] if hasattr(key, "__getitem__") else key
                            extra_lvn[kv_name] = sp.Integer(int(kval))
                    elif len(op.key_vars) == 1:
                        extra_lvn[op.key_vars[0]] = sp.Integer(int(key))
                    if op.value_var and isinstance(value, (int, float)):
                        extra_lvn[op.value_var] = sp.Integer(int(value))

                    local_block = _LocalBlock(op.operations)
                    child = resolver.child_scope(
                        inner_block=local_block,
                        extra_loop_vars=extra_lvn,
                    )
                    _simulate_parallel_depth_concrete(
                        op.operations,
                        qubit_depths,
                        child,
                        num_controls,
                        value_depths,
                        dict_bindings=dict_bindings,
                    )
            else:
                child = build_for_items_scope(op, resolver)
                inner_depth = _compute_sequential_depth(
                    op.operations,
                    child,
                    num_controls,
                )
                cardinality = resolve_for_items_cardinality(op)
                total_depth = inner_depth * cardinality
                body_qubits = _collect_all_qubit_names(op.operations)
                current_max = _get_max_depth(qubit_depths)
                new_max = current_max + total_depth
                for qid in set(qubit_depths) | body_qubits:
                    qubit_depths[qid] = new_max
                if (
                    hasattr(dict_operand, "is_parameter")
                    and dict_operand.is_parameter()
                ):
                    dict_name = dict_operand.parameter_name() or dict_operand.name
                else:
                    dict_name = dict_operand.name
                warnings.warn(
                    f"Depth for ForItemsOperation over '{dict_name}' is a "
                    f"sequential upper bound (multiplied by |{dict_name}|). "
                    f"True depth depends on runtime data.",
                    stacklevel=2,
                )

        elif isinstance(op, WhileOperation):
            child, trip_count = build_while_scope(op, resolver)
            inner_depth = _compute_sequential_depth(
                op.operations,
                child,
                num_controls,
            )
            increase = inner_depth * trip_count
            body_qubits = _collect_all_qubit_names(op.operations)
            current_max = _get_max_depth(qubit_depths)
            new_max = current_max + increase
            for qid in set(qubit_depths) | body_qubits:
                qubit_depths[qid] = new_max
            warnings.warn(
                "Depth for WhileOperation is a sequential upper bound "
                "(multiplied by |while|). True iteration count is unknown.",
                stacklevel=2,
            )

        elif isinstance(op, ForOperation):
            if len(op.operands) < 2:
                continue
            try:
                start_val = resolver.resolve_concrete(op.operands[0])
                stop_val = resolver.resolve_concrete(op.operands[1])
                step_val = (
                    resolver.resolve_concrete(op.operands[2])
                    if len(op.operands) >= 3
                    else 1
                )
            except UnresolvedValueError:
                raise _UnresolvableForOpError()
            local_block = _LocalBlock(op.operations)
            for loop_val in range(start_val, stop_val, step_val):
                child = resolver.child_scope(
                    inner_block=local_block,
                    extra_loop_vars={op.loop_var: sp.Integer(loop_val)},
                )
                _simulate_parallel_depth_concrete(
                    op.operations,
                    qubit_depths,
                    child,
                    num_controls,
                    value_depths,
                    dict_bindings=dict_bindings,
                )

        elif isinstance(op, CallBlockOperation):
            called_block = op.operands[0]
            if not isinstance(called_block, BlockValue):
                continue

            child = resolver.call_child_scope(op)
            qubit_name_map = _build_qubit_name_map(
                called_block,
                op.operands,
                resolver,
            )
            local_depths = _map_depths_to_local(qubit_name_map, qubit_depths)

            _simulate_parallel_depth_concrete(
                called_block.operations,
                local_depths,
                child,
                num_controls,
                value_depths,
                dict_bindings=dict_bindings,
            )

            _write_back_depths(qubit_name_map, local_depths, qubit_depths)

        elif isinstance(op, ControlledUOperation):
            gate_depth = _handle_controlled_u_depth(op, resolver)
            all_qubit_names = _get_controlled_u_qubit_names(op, resolver)
            CircuitDepth.apply_gate_to_qubits(qubit_depths, all_qubit_names, gate_depth)

        elif isinstance(op, CompositeGateOperation):
            composite_depth = _handle_composite_depth(
                op,
                resolver,
                num_controls,
            )
            raw_names = [
                _qubit_key(v, resolver)
                for v in op.operands
                if hasattr(v, "name") and hasattr(v, "type") and v.type.is_quantum()
            ]
            qubit_ids = _expand_array_qubit_ids(raw_names, qubit_depths)
            CircuitDepth.apply_gate_to_qubits(qubit_depths, qubit_ids, composite_depth)


# ------------------------------------------------------------------ #
#  Full-block sampling for parametric loops                           #
# ------------------------------------------------------------------ #


def _find_parametric_for_symbols(
    operations: list[Operation],
    resolver: ExprResolver,
) -> set[sp.Symbol]:
    """Scan operations for ForOperations with parametric bounds.

    Returns the set of parametric symbols found in any ForOperation bounds.
    """
    syms: set[sp.Symbol] = set()
    for op in operations:
        if not isinstance(op, ForOperation) or len(op.operands) < 2:
            continue
        child, start, stop, step, loop_sym = build_for_loop_scope(op, resolver)
        parametric = find_parametric_symbols(start, stop, step)
        parametric -= {loop_sym}
        parametric -= set(resolver.loop_var_names.values())
        syms |= parametric
    return syms


def _needs_full_block_sampling(
    operations: list[Operation],
    resolver: ExprResolver,
) -> sp.Symbol | None:
    """Check if operations need full-block sampling.

    Returns the parametric symbol if full-block sampling is needed
    (parametric ForOps coexist with other gate ops), None otherwise.
    """
    param_syms = _find_parametric_for_symbols(operations, resolver)
    if not param_syms:
        return None

    # Check if there are other gate-like operations alongside parametric loops
    has_other_ops = any(
        isinstance(
            op,
            (
                GateOperation,
                CallBlockOperation,
                CompositeGateOperation,
                ControlledUOperation,
            ),
        )
        for op in operations
    )
    if not has_other_ops:
        # Only parametric loops — per-loop sampling is fine
        return None

    return sorted(param_syms, key=str)[0]


def _sample_full_block_depth(
    operations: list[Operation],
    qubit_depths: QubitDepthMap,
    resolver: ExprResolver,
    param_sym: sp.Symbol,
    num_controls: int | sp.Expr = 0,
    value_depths: dict[str, CircuitDepth] | None = None,
) -> bool:
    """Sample the entire block at multiple parameter values and interpolate.

    This preserves per-qubit depth information across operations within
    each sample, avoiding depth information loss that occurs when
    parametric ForOperations are sampled independently.

    Returns True if sampling succeeded, False if caller should fall back
    to per-operation estimation.
    """
    _MIN_SAMPLES = 6

    # Collect valid sample points — only include n values where ALL
    # parametric ForOperations have non-zero iterations (skip degenerate cases)
    valid_points: list[int] = []
    for n_val in range(2, 21):
        sample_resolver = _concretize_resolver(resolver, param_sym, n_val)
        all_valid = True
        for op in operations:
            if not isinstance(op, ForOperation) or len(op.operands) < 2:
                continue
            try:
                _, start, stop, step, _ = build_for_loop_scope(
                    op,
                    sample_resolver,
                )
                s, e, t = int(start), int(stop), int(step)
                if len(range(s, e, t)) == 0:
                    all_valid = False
                    break
            except (TypeError, ValueError, UnresolvedValueError):
                continue
        if not all_valid:
            continue
        valid_points.append(n_val)
        if len(valid_points) >= _MIN_SAMPLES + 1:
            break

    if len(valid_points) < 3:
        return False

    entry_depth = _get_max_depth(qubit_depths)
    samples: dict[int, CircuitDepth] = {}

    for n_val in valid_points:
        sample_resolver = _concretize_resolver(resolver, param_sym, n_val)
        local_depths: QubitDepthMap = {}

        try:
            _simulate_parallel_depth_concrete(
                operations,
                local_depths,
                sample_resolver,
                num_controls,
            )
        except _UnresolvableForOpError:
            continue

        samples[n_val] = _get_max_depth(local_depths)

    if not samples:
        return False

    # Verification
    verify_n = max(samples.keys())
    verify_sample = samples.pop(verify_n)

    if not samples:
        samples[verify_n] = verify_sample
        interpolated = _interpolate_depth_from_samples(samples, param_sym)
    else:
        interpolated = _interpolate_depth_from_samples(samples, param_sym)

        verify_check = interpolated.substitute({param_sym: verify_n})
        mismatch = any(
            sp.simplify(getattr(verify_check, field) - getattr(verify_sample, field))
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
            interpolated = _interpolate_depth_from_samples(samples, param_sym)

    new_depth = entry_depth + interpolated
    body_qubits = _collect_all_qubit_names(operations)
    for qid in set(qubit_depths) | body_qubits:
        qubit_depths[qid] = new_depth

    return True


# ------------------------------------------------------------------ #
#  Symbolic parallel depth estimation                                 #
# ------------------------------------------------------------------ #


def _estimate_parallel_depth(
    operations: list[Operation],
    qubit_depths: QubitDepthMap,
    resolver: ExprResolver,
    num_controls: int | sp.Expr = 0,
    value_depths: dict[str, CircuitDepth] | None = None,
) -> None:
    """Estimate parallel depth by tracking per-qubit depths (symbolic path).

    Modifies qubit_depths in place. Each gate is placed at the earliest
    time step where all its qubits are available.
    """
    # Check if full-block sampling is needed
    param_sym = _needs_full_block_sampling(operations, resolver)
    if param_sym is not None:
        if _sample_full_block_depth(
            operations,
            qubit_depths,
            resolver,
            param_sym,
            num_controls,
            value_depths,
        ):
            return
        # Full-block sampling failed — fall through to per-op estimation

    for op in operations:
        match op:
            case GateOperation():
                gate_depth = gate_count_to_depth(
                    classify_gate(op, num_controls=num_controls)
                )
                qubit_ids = [_qubit_key(v, resolver) for v in op.operands]
                CircuitDepth.apply_gate_to_qubits(qubit_depths, qubit_ids, gate_depth)

            case MeasureOperation():
                qubit_id = _qubit_key(op.operands[0], resolver)
                current = qubit_depths.get(qubit_id, CircuitDepth.zero())
                new_depth = current + _MEASURE_UNIT
                qubit_depths[qubit_id] = new_depth
                if value_depths is not None and op.results:
                    value_depths[op.results[0].uuid] = new_depth

            case MeasureVectorOperation():
                arr_name = op.operands[0].name
                matched_qids = [
                    k
                    for k in qubit_depths
                    if k.startswith(arr_name + "[") or k == arr_name
                ]
                if not matched_qids:
                    matched_qids = [arr_name]
                CircuitDepth.apply_gate_to_qubits(
                    qubit_depths, matched_qids, _MEASURE_UNIT
                )
                if value_depths is not None and op.results:
                    vd = CircuitDepth.zero()
                    for qid in matched_qids:
                        vd = vd.max(qubit_depths[qid])
                    value_depths[op.results[0].uuid] = vd

            case MeasureQFixedOperation():
                qfixed_name = op.operands[0].name
                matched_qids = [
                    k
                    for k in qubit_depths
                    if k.startswith(qfixed_name + "[") or k == qfixed_name
                ]
                if not matched_qids:
                    matched_qids = [qfixed_name]
                CircuitDepth.apply_gate_to_qubits(
                    qubit_depths, matched_qids, _MEASURE_UNIT
                )
                if value_depths is not None and op.results:
                    vd = CircuitDepth.zero()
                    for qid in matched_qids:
                        vd = vd.max(qubit_depths[qid])
                    value_depths[op.results[0].uuid] = vd

            case NotOp():
                if value_depths is not None:
                    input_depth = value_depths.get(op.input.uuid, CircuitDepth.zero())
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
                    resolver,
                    num_controls,
                    value_depths,
                )

            case WhileOperation():
                child, trip_count = build_while_scope(op, resolver)
                inner_depth = _compute_sequential_depth(
                    op.operations,
                    child,
                    num_controls,
                )
                increase = inner_depth * trip_count
                current_max = _get_max_depth(qubit_depths)
                new_max = current_max + increase
                body_qubits = _collect_all_qubit_names(op.operations, resolver)
                for qid in set(qubit_depths) | body_qubits:
                    qubit_depths[qid] = new_max

            case ForItemsOperation():
                child = build_for_items_scope(op, resolver)
                inner_depth = _compute_sequential_depth(
                    op.operations,
                    child,
                    num_controls,
                )
                cardinality = resolve_for_items_cardinality(op)
                total_depth = inner_depth * cardinality
                body_qubits = _collect_all_qubit_names(op.operations, resolver)
                current_max = _get_max_depth(qubit_depths)
                new_max = current_max + total_depth
                for qid in set(qubit_depths) | body_qubits:
                    qubit_depths[qid] = new_max

            case IfOperation():
                condition_depth = CircuitDepth.zero()
                if value_depths is not None:
                    condition_depth = value_depths.get(
                        op.condition.uuid, CircuitDepth.zero()
                    )

                true_depths = qubit_depths.copy()
                false_depths = qubit_depths.copy()

                branch_qubits = _collect_all_qubit_names(
                    op.true_operations, resolver
                ) | _collect_all_qubit_names(op.false_operations, resolver)
                for qid in branch_qubits:
                    current = qubit_depths.get(qid, CircuitDepth.zero())
                    bumped = current.max(condition_depth)
                    true_depths[qid] = bumped
                    false_depths[qid] = bumped

                true_child, false_child = build_if_scopes(op, resolver)
                _estimate_parallel_depth(
                    op.true_operations,
                    true_depths,
                    true_child,
                    num_controls,
                    value_depths,
                )
                _estimate_parallel_depth(
                    op.false_operations,
                    false_depths,
                    false_child,
                    num_controls,
                    value_depths,
                )

                all_qids = set(true_depths) | set(false_depths)
                for qid in all_qids:
                    qubit_depths[qid] = true_depths.get(qid, CircuitDepth.zero()).max(
                        false_depths.get(qid, CircuitDepth.zero())
                    )

                for phi_op in op.phi_ops:
                    tv_name = _qubit_key(phi_op.true_value, resolver)
                    fv_name = _qubit_key(phi_op.false_value, resolver)
                    out_name = _qubit_key(phi_op.output, resolver)
                    tv_depth = qubit_depths.get(tv_name, CircuitDepth.zero())
                    fv_depth = qubit_depths.get(fv_name, CircuitDepth.zero())
                    qubit_depths[out_name] = tv_depth.max(fv_depth)

            case CallBlockOperation():
                _handle_call_block_parallel(
                    op,
                    qubit_depths,
                    resolver,
                    num_controls,
                    value_depths,
                )

            case ControlledUOperation():
                gate_depth = _handle_controlled_u_depth(op, resolver)
                all_qubit_names = _get_controlled_u_qubit_names(op, resolver)
                CircuitDepth.apply_gate_to_qubits(
                    qubit_depths, all_qubit_names, gate_depth
                )

            case CompositeGateOperation():
                composite_depth = _handle_composite_depth(
                    op,
                    resolver,
                    num_controls,
                )
                raw_names = [
                    _qubit_key(v, resolver) for v in op.operands if hasattr(v, "name")
                ]
                qubit_ids = _expand_array_qubit_ids(raw_names, qubit_depths)
                CircuitDepth.apply_gate_to_qubits(
                    qubit_depths, qubit_ids, composite_depth
                )

            case _:
                continue


# ------------------------------------------------------------------ #
#  ForOperation handler (parallel depth)                              #
# ------------------------------------------------------------------ #


def _handle_for_parallel(
    op: ForOperation,
    qubit_depths: QubitDepthMap,
    resolver: ExprResolver,
    num_controls: int | sp.Expr,
    value_depths: dict[str, CircuitDepth] | None = None,
) -> None:
    """Handle ForOperation for parallel depth estimation.

    Uses concrete simulation + interpolation for accurate nested loop
    depth estimation.
    """
    if len(op.operands) < 2:
        increase = _compute_sequential_depth(
            op.operations,
            resolver,
            num_controls,
        )
        iterations = sp.Symbol("iter", integer=True, positive=True)
        current_max = _get_max_depth(qubit_depths)
        new_max = current_max + increase * iterations
        body_qubits = _collect_all_qubit_names(op.operations)
        for qid in set(qubit_depths) | body_qubits:
            qubit_depths[qid] = new_max
        return

    child, start, stop, step, loop_sym = build_for_loop_scope(op, resolver)
    body_qubits = _collect_all_qubit_names(op.operations)

    # Check for parametric symbols in bounds
    parametric_syms = find_parametric_symbols(start, stop, step)
    parametric_syms -= {loop_sym}
    parametric_syms -= set(resolver.loop_var_names.values())

    if not parametric_syms:
        # Bounds are fully concrete -> direct simulation
        try:
            start_val = int(start)
            stop_val = int(stop)
            step_val = int(step)
        except (TypeError, ValueError):
            # Still symbolic -> multiply fallback
            inner_depth = _compute_sequential_depth(
                op.operations,
                child,
                num_controls,
            )
            iterations = symbolic_iterations(start, stop, step)
            current_max = _get_max_depth(qubit_depths)
            new_max = current_max + inner_depth * iterations
            for qid in set(qubit_depths) | body_qubits:
                qubit_depths[qid] = new_max
            return

        local_block = _LocalBlock(op.operations)
        try:
            for loop_val in range(start_val, stop_val, step_val):
                iter_child = resolver.child_scope(
                    inner_block=local_block,
                    extra_loop_vars={op.loop_var: sp.Integer(loop_val)},
                )
                _simulate_parallel_depth_concrete(
                    op.operations,
                    qubit_depths,
                    iter_child,
                    num_controls,
                    value_depths,
                )
        except _UnresolvableForOpError:
            per_iter_depths: QubitDepthMap = {}
            _estimate_parallel_depth(
                op.operations,
                per_iter_depths,
                child,
                num_controls,
            )
            increase = _get_max_depth(per_iter_depths)
            iterations = (stop - start) / step
            current_max = _get_max_depth(qubit_depths)
            new_max = current_max + increase * iterations
            for qid in set(qubit_depths) | body_qubits:
                qubit_depths[qid] = new_max
        return

    # Bounds contain parametric symbols -> sample and interpolate
    param_sym = sorted(parametric_syms, key=str)[0]
    _MIN_SAMPLES = 6

    entry_depth = _get_max_depth(qubit_depths)
    samples: dict[int, CircuitDepth] = {}

    # Collect valid sample points
    valid_points: list[int] = []
    for n_val in range(2, 21):
        subs = {param_sym: n_val}
        try:
            cs = int(start.subs(subs))
            ce = int(stop.subs(subs))
            ct = int(step.subs(subs))
        except (TypeError, ValueError):
            continue
        if len(range(cs, ce, ct)) == 0:
            continue
        valid_points.append(n_val)
        if len(valid_points) >= _MIN_SAMPLES + 1:
            break

    if not valid_points:
        return

    if len(valid_points) < 3:
        inner_depth = _compute_sequential_depth(
            op.operations,
            child,
            num_controls,
        )
        iterations = symbolic_iterations(start, stop, step)
        current_max = _get_max_depth(qubit_depths)
        new_max = current_max + inner_depth * iterations
        for qid in set(qubit_depths) | body_qubits:
            qubit_depths[qid] = new_max
        return

    # Sample by concrete simulation
    for n_val in valid_points:
        subs = {param_sym: n_val}
        try:
            cs = int(start.subs(subs))
            ce = int(stop.subs(subs))
            ct = int(step.subs(subs))
        except (TypeError, ValueError):
            continue

        sample_resolver = _concretize_resolver(resolver, param_sym, n_val)
        local_block = _LocalBlock(op.operations)

        inner_depths: QubitDepthMap = {}
        try:
            for loop_val in range(cs, ce, ct):
                iter_child = sample_resolver.child_scope(
                    inner_block=local_block,
                    extra_loop_vars={op.loop_var: sp.Integer(loop_val)},
                )
                _simulate_parallel_depth_concrete(
                    op.operations,
                    inner_depths,
                    iter_child,
                    num_controls,
                )
        except _UnresolvableForOpError:
            # Build symbolic child from sample resolver for per-iteration est.
            sym_child, _, _, _, _ = build_for_loop_scope(
                op,
                sample_resolver,
            )
            per_iter_depths: QubitDepthMap = {}
            _estimate_parallel_depth(
                op.operations,
                per_iter_depths,
                sym_child,
                num_controls,
            )
            per_iter_increase = _get_max_depth(per_iter_depths)
            n_iterations = len(range(cs, ce, ct))
            samples[n_val] = per_iter_increase * n_iterations
            continue

        samples[n_val] = _get_max_depth(inner_depths)

    if not samples:
        inner_depth = _compute_sequential_depth(
            op.operations,
            child,
            num_controls,
        )
        iterations = symbolic_iterations(start, stop, step)
        current_max = _get_max_depth(qubit_depths)
        new_max = current_max + inner_depth * iterations
        for qid in set(qubit_depths) | body_qubits:
            qubit_depths[qid] = new_max
        return

    # Verification
    verify_n = valid_points[-1]
    verify_sample = samples.pop(verify_n, None)

    interpolated = _interpolate_depth_from_samples(samples, param_sym)

    if verify_sample is not None:
        verify_check = interpolated.substitute({param_sym: verify_n})
        mismatch = any(
            sp.simplify(getattr(verify_check, field) - getattr(verify_sample, field))
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
            interpolated = _interpolate_depth_from_samples(samples, param_sym)

    new_depth = entry_depth + interpolated
    for qid in set(qubit_depths) | body_qubits:
        qubit_depths[qid] = new_depth


# ------------------------------------------------------------------ #
#  CallBlockOperation handler (parallel depth)                        #
# ------------------------------------------------------------------ #


def _handle_call_block_parallel(
    op: CallBlockOperation,
    qubit_depths: QubitDepthMap,
    resolver: ExprResolver,
    num_controls: int | sp.Expr,
    value_depths: dict[str, CircuitDepth] | None = None,
) -> None:
    """Handle CallBlockOperation for parallel depth estimation."""
    from qamomile.circuit.ir.block_value import BlockValue

    called_block = op.operands[0]
    if not isinstance(called_block, BlockValue):
        return

    child = resolver.call_child_scope(op)
    qubit_name_map = _build_qubit_name_map(
        called_block,
        op.operands,
        resolver,
    )
    local_depths = _map_depths_to_local(qubit_name_map, qubit_depths)

    _estimate_parallel_depth(
        called_block.operations,
        local_depths,
        child,
        num_controls,
        value_depths,
    )

    _write_back_depths(qubit_name_map, local_depths, qubit_depths)


# ------------------------------------------------------------------ #
#  Sequential depth (fallback for loops)                              #
# ------------------------------------------------------------------ #


def _compute_sequential_depth(
    operations: list[Operation],
    resolver: ExprResolver,
    num_controls: int | sp.Expr = 0,
) -> CircuitDepth:
    """Compute sequential (sum-of-all-gates) depth. Used as fallback."""
    depth = CircuitDepth.zero()

    for op in operations:
        match op:
            case GateOperation():
                depth = depth + gate_count_to_depth(
                    classify_gate(op, num_controls=num_controls)
                )

            case (
                MeasureOperation() | MeasureVectorOperation() | MeasureQFixedOperation()
            ):
                depth = depth + _MEASURE_UNIT

            case ForItemsOperation():
                child = build_for_items_scope(op, resolver)
                inner_depth = _compute_sequential_depth(
                    op.operations,
                    child,
                    num_controls,
                )
                cardinality = resolve_for_items_cardinality(op)
                depth = depth + inner_depth * cardinality

            case ForOperation():
                if len(op.operands) >= 2:
                    child, start, stop, step, loop_sym = build_for_loop_scope(
                        op, resolver
                    )
                    inner_depth = _compute_sequential_depth(
                        op.operations,
                        child,
                        num_controls,
                    )
                    all_free = (
                        inner_depth.total_depth.free_symbols
                        | inner_depth.two_qubit_depth.free_symbols
                        | inner_depth.multi_qubit_depth.free_symbols
                    )
                    if loop_sym in all_free:
                        depth = depth + _apply_sum_to_depth(
                            inner_depth,
                            loop_sym,
                            start,
                            stop,
                            step,
                        )
                    else:
                        iterations = symbolic_iterations(start, stop, step)
                        depth = depth + inner_depth * iterations

            case IfOperation():
                true_child, false_child = build_if_scopes(op, resolver)
                true_depth = _compute_sequential_depth(
                    op.true_operations,
                    true_child,
                    num_controls,
                )
                false_depth = _compute_sequential_depth(
                    op.false_operations,
                    false_child,
                    num_controls,
                )
                depth = depth + true_depth.max(false_depth)

            case CallBlockOperation():
                from qamomile.circuit.ir.block_value import BlockValue

                called_block = op.operands[0]
                if isinstance(called_block, BlockValue):
                    child = resolver.call_child_scope(op)
                    inner_depth = _compute_sequential_depth(
                        called_block.operations,
                        child,
                        num_controls,
                    )
                    depth = depth + inner_depth

            case ControlledUOperation():
                depth = depth + _handle_controlled_u_depth(op, resolver)

            case CompositeGateOperation():
                depth = depth + _handle_composite_depth(
                    op,
                    resolver,
                    num_controls,
                )

            case _:
                continue

    return depth.simplify()


# ------------------------------------------------------------------ #
#  Entry point helpers                                                #
# ------------------------------------------------------------------ #


def _build_resolver_from_bindings(
    block: Any,
    bindings: dict[str, Any],
) -> tuple[ExprResolver, dict[str, Any]]:
    """Build ExprResolver and dict_bindings from user-provided bindings."""
    from qamomile.circuit.ir.block_value import BlockValue
    from qamomile.circuit.ir.value import ArrayValue, DictValue

    scalar_bindings: dict[str, int] = {}
    dict_bindings: dict[str, Any] = {}
    for key, val in bindings.items():
        if isinstance(val, dict):
            dict_bindings[key] = val
        elif isinstance(val, (int, float)):
            scalar_bindings[key] = int(val)

    context: dict[str, sp.Expr] = {}
    if isinstance(block, BlockValue):
        for formal in block.input_values:
            pname = None
            if hasattr(formal, "is_parameter") and formal.is_parameter():
                pname = formal.parameter_name()
            if pname is None:
                pname = formal.name

            if isinstance(formal, DictValue):
                continue
            if pname in scalar_bindings:
                context[formal.uuid] = sp.Integer(scalar_bindings[pname])
                if isinstance(formal, ArrayValue):
                    for dim in formal.shape:
                        dim_pname = None
                        if hasattr(dim, "is_parameter") and dim.is_parameter():
                            dim_pname = dim.parameter_name()
                        if dim_pname and dim_pname in scalar_bindings:
                            context[dim.uuid] = sp.Integer(scalar_bindings[dim_pname])

    resolver = ExprResolver(block=block, context=context)
    return resolver, dict_bindings


# ------------------------------------------------------------------ #
#  Public entry point                                                 #
# ------------------------------------------------------------------ #


def estimate_depth(
    block: BlockValue | Block | list[Operation],
    *,
    bindings: dict[str, Any] | None = None,
) -> CircuitDepth:
    """Estimate circuit depth using parallel (DAG critical path) analysis.

    Computes the minimum circuit depth considering gate-level parallelism.
    Gates on independent qubits can execute in parallel. The result is
    the longest dependency chain through the circuit.

    When ``bindings`` is provided with concrete dict values,
    ForItemsOperation loops are unrolled with per-qubit depth tracking.

    Args:
        block: BlockValue, Block, or list of Operations to analyze
        bindings: Optional concrete parameter bindings (scalars and dicts).

    Returns:
        CircuitDepth with total_depth, t_depth, two_qubit_depth, etc.
    """
    from qamomile.circuit.ir.block import Block
    from qamomile.circuit.ir.block_value import BlockValue

    block_ref = None
    if isinstance(block, (BlockValue, Block)):
        block_ref = block
        ops = block.operations
    else:
        ops = block

    # Concrete path: when bindings provided
    if bindings is not None and block_ref is not None:
        resolver, dict_bindings = _build_resolver_from_bindings(block_ref, bindings)
        try:
            qubit_depths: QubitDepthMap = {}
            value_depths: dict[str, CircuitDepth] = {}
            _simulate_parallel_depth_concrete(
                ops,
                qubit_depths,
                resolver,
                value_depths=value_depths,
                dict_bindings=dict_bindings,
            )
            return _get_max_depth(qubit_depths).simplify()
        except _UnresolvableForOpError:
            pass  # Fall through to symbolic path

    # Symbolic path
    resolver = ExprResolver(block=block_ref)
    qubit_depths = {}
    value_depths = {}
    _estimate_parallel_depth(
        ops,
        qubit_depths,
        resolver,
        value_depths=value_depths,
    )
    return _get_max_depth(qubit_depths).simplify()
