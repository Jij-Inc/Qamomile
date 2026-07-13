"""Circuit analysis: IR inspection, value resolution, and label generation.

This module provides CircuitAnalyzer, which handles all IR-level analysis
for the circuit visualization pipeline. It has no matplotlib dependency.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.arithmetic_operations import (
    BinOp,
    BinOpKind,
    CompOp,
    CompOpKind,
    CondOp,
    CondOpKind,
    NotOp,
)
from qamomile.circuit.ir.operation.callable import CallPolicy, InvokeOperation
from qamomile.circuit.ir.operation.cast import CastOperation
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    IfOperation,
    WhileOperation,
)
from qamomile.circuit.ir.operation.expval import ExpvalOp
from qamomile.circuit.ir.operation.gate import (
    ControlledUOperation,
    GateOperation,
    GateOperationType,
    MeasureOperation,
    MeasureQFixedOperation,
    MeasureVectorOperation,
)
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.operation.return_operation import ReturnOperation
from qamomile.circuit.ir.types.primitives import QubitType
from qamomile.circuit.ir.value import ArrayValue, DictValue, Value, ValueBase, ValueLike
from qamomile.circuit.transpiler.block_parameter_binding import (
    align_formal_operands,
)

from .geometry import compute_border_padding
from .style import CircuitStyle
from .types import _TEX_SYMBOLS
from .visual_ir import (
    VFoldedBlock,
    VFoldedKind,
    VGate,
    VGateKind,
    VInlineBlock,
    VisualCircuit,
    VisualNode,
    VSkip,
    VUnfoldedKind,
    VUnfoldedSequence,
)

if TYPE_CHECKING:
    from qamomile.circuit.ir.block import Block


_INTERNAL_TMP_NAMES: frozenset[str] = frozenset({"uint_tmp", "float_tmp", "bit_tmp"})


# String markers appended to a node's scope path for the true / false branch of
# an ``IfOperation`` (see ``_build_vif`` and the layout engine's unfolded-if
# placement). They are the only non-integer scope-path elements any node key
# carries, so their presence uniquely identifies a node nested inside an
# if/else branch — used to keep mid-circuit measurements from terminating a
# wire that the other branch never measured.
_IF_BRANCH_SCOPE_KEYS: frozenset[str] = frozenset({"true", "false"})


# Folded loops do not materialize visual nodes for every iteration, but a
# small, concrete loop still needs value replay so RegionArg results used by
# later gates have their final value.  Cap that analysis to keep drawing a
# deliberately huge folded loop cheap.
_MAX_FOLDED_VALUE_REPLAY_ITERATIONS = 1000
_UNRESOLVED_ENV_VALUE = object()


# Single source of truth for the TeX rendering of every built-in
# gate the drawer knows how to label, keyed by ``GateOperationType``.
# Used directly by the inline-gate label path
# (``CircuitAnalyzer._get_gate_label``) and indirectly — via
# ``_BUILTIN_TEX_LABELS`` below — by the controlled-U-box path,
# which keys off ``block.name`` (the string assigned by built-in
# factories) instead of the enum.
_TEX_LABELS_BY_GATE_TYPE: dict[GateOperationType, str] = {
    # Parametric gates
    GateOperationType.RX: r"$R_x$",
    GateOperationType.RY: r"$R_y$",
    GateOperationType.RZ: r"$R_z$",
    GateOperationType.RZZ: r"$R_{zz}$",
    GateOperationType.CP: r"$CP$",
    GateOperationType.P: r"$P$",
    # Non-parametric gates
    GateOperationType.H: r"$H$",
    GateOperationType.X: r"$X$",
    GateOperationType.Y: r"$Y$",
    GateOperationType.Z: r"$Z$",
    GateOperationType.T: r"$T$",
    GateOperationType.TDG: r"$T^{\dagger}$",
    GateOperationType.S: r"$S$",
    GateOperationType.SDG: r"$S^{\dagger}$",
    GateOperationType.CX: r"$CX$",
    GateOperationType.CZ: r"$CZ$",
    GateOperationType.SWAP: r"$SWAP$",
    GateOperationType.TOFFOLI: r"$CCX$",
}


# ``block.name`` strings assigned by built-in gate factories.  For
# most gates this is the lowercase enum name (``RX`` → ``"rx"``),
# but a couple of factories use a different historical name
# (``GateOperationType.TOFFOLI`` is exposed as ``qmc.ccx`` and
# stores ``block.name = "ccx"``).  Drift risk is now limited to
# this one mapping rather than to two parallel TeX tables.
_GATE_TYPE_BUILTIN_NAMES: dict[GateOperationType, str] = {
    GateOperationType.RX: "rx",
    GateOperationType.RY: "ry",
    GateOperationType.RZ: "rz",
    GateOperationType.RZZ: "rzz",
    GateOperationType.P: "p",
    GateOperationType.CP: "cp",
    GateOperationType.H: "h",
    GateOperationType.X: "x",
    GateOperationType.Y: "y",
    GateOperationType.Z: "z",
    GateOperationType.T: "t",
    GateOperationType.TDG: "tdg",
    GateOperationType.S: "s",
    GateOperationType.SDG: "sdg",
    GateOperationType.CX: "cx",
    GateOperationType.CZ: "cz",
    GateOperationType.SWAP: "swap",
    GateOperationType.TOFFOLI: "ccx",
}


# Derived from the two tables above.  Used by the controlled-U-box
# rendering path: when the wrapped block of a
# ``ControlledUOperation`` was a built-in gate (e.g.
# ``qmc.control(qmc.rx)`` gives ``block.name = "rx"``) this maps
# the block name back to the same TeX label inline gates use.
_BUILTIN_TEX_LABELS: dict[str, str] = {
    _GATE_TYPE_BUILTIN_NAMES[gate_type]: tex
    for gate_type, tex in _TEX_LABELS_BY_GATE_TYPE.items()
    if gate_type in _GATE_TYPE_BUILTIN_NAMES
}


# Prefix that ``ControlledGate.__call__`` attaches to a wrapped
# kernel's classical-parameter Value names (e.g. ``ctrl_param_theta``
# for an inner ``theta`` parameter).  The drawer strips it so the
# rendered label reads ``_phase(theta=π/2)`` rather than
# ``_phase(ctrl_param_theta=π/2)``.
_CTRL_PARAM_PREFIX: str = "ctrl_param_"


# Infix symbols for spelling out an ``IfOperation`` condition (see
# ``CircuitAnalyzer._format_condition_expr``).  A symbolic ``if`` whose
# condition is not compile-time resolved is drawn with its predicate
# rendered as source, e.g. ``if flag == 1:`` rather than ``if cond:``.
# ``BinOpKind.MIN`` is intentionally absent — it has no infix form and
# falls back to the anonymous-condition label.
_COMP_OP_SYMBOLS: dict[CompOpKind, str] = {
    CompOpKind.EQ: "==",
    CompOpKind.NEQ: "!=",
    CompOpKind.LT: "<",
    CompOpKind.LE: "<=",
    CompOpKind.GT: ">",
    CompOpKind.GE: ">=",
}
_COND_OP_SYMBOLS: dict[CondOpKind, str] = {
    CondOpKind.AND: "and",
    CondOpKind.OR: "or",
}
_BIN_OP_SYMBOLS: dict[BinOpKind, str] = {
    BinOpKind.ADD: "+",
    BinOpKind.SUB: "-",
    BinOpKind.MUL: "*",
    BinOpKind.DIV: "/",
    BinOpKind.FLOORDIV: "//",
    BinOpKind.MOD: "%",
    BinOpKind.POW: "**",
}


class CircuitAnalyzer:
    """Analyzes IR blocks for circuit visualization.

    Handles qubit mapping, value resolution, label generation,
    and width estimation. Has no matplotlib dependency.
    """

    @staticmethod
    def _is_internal_temp_name(name: str | None) -> bool:
        """Return True if `name` is an IR-internal placeholder.

        These default names (``uint_tmp`` / ``float_tmp`` / ``bit_tmp``
        from `qamomile.circuit.frontend.handle.primitives`) label
        anonymous Values produced by BinOp and handle construction.
        They are implementation detail and must never reach user-facing
        rendered labels.

        Args:
            name: Candidate display string.

        Returns:
            True if `name` equals one of the reserved placeholders.
        """
        return name in _INTERNAL_TMP_NAMES

    def __init__(
        self,
        graph: "Block",
        style: CircuitStyle,
        inline: bool = False,
        fold_loops: bool = True,
        expand_composite: bool = False,
        inline_depth: int | None = None,
        fold_ifs: bool = False,
    ):
        """Initialize the visualization analyzer.

        Args:
            graph (Block): Computation graph to analyze for rendering.
            style (CircuitStyle): Visual style configuration.
            inline (bool): Whether to expand inline callable contents.
            fold_loops (bool): Whether to render loop operations as folded
                summary blocks instead of materialized iterations.
            expand_composite (bool): Whether to expand composite gates.
            inline_depth (int | None): Maximum nesting depth for inline
                expansion, or None for unlimited depth.
            fold_ifs (bool): Whether to render IfOperation nodes as folded
                summary blocks instead of side-by-side branches.
        """
        self.graph = graph
        self.style = style
        self.inline = inline
        self.fold_loops = fold_loops
        self.expand_composite = expand_composite
        self.inline_depth = inline_depth
        self.fold_ifs = fold_ifs

    def _environment_value(
        self,
        value: Value,
        param_values: dict,
        operations: list[Operation] | None = None,
    ) -> object:
        """Resolve one scalar Value for the visualization value environment.

        Args:
            value (Value): Scalar IR value to resolve.
            param_values (dict): Current visualization value environment.
            operations (list[Operation] | None): Scope containing the value's
                producer. Defaults to None, which lets ``_evaluate_value``
                inspect the top-level graph.

        Returns:
            object: A concrete scalar or symbolic string when resolution
                succeeds, otherwise the private unresolved sentinel.
        """
        if value.logical_id in param_values:
            return param_values[value.logical_id]

        evaluated = self._evaluate_value(value, param_values, operations)
        if evaluated is not None:
            return evaluated

        if value.is_parameter():
            parameter_name = value.parameter_name() or value.name
            if parameter_name and not self._is_internal_temp_name(parameter_name):
                return parameter_name

        if operations is not None:
            symbolic = self._resolve_binop_as_symbolic(value, param_values, operations)
            if symbolic is not None:
                return symbolic
        return _UNRESOLVED_ENV_VALUE

    def _seed_region_arguments(
        self,
        op: ForOperation | ForItemsOperation | WhileOperation,
        param_values: dict,
    ) -> dict:
        """Create a loop environment with each RegionArg bound to its init.

        Args:
            op (ForOperation | ForItemsOperation | WhileOperation): Loop whose
                body region is about to execute.
            param_values (dict): Enclosing visualization value environment.

        Returns:
            dict: A child environment in which every resolvable
                ``RegionArg.block_arg`` denotes its initializer.
        """
        loop_values = dict(param_values)
        for region_arg in op.region_args:
            resolved = self._environment_value(region_arg.init, loop_values)
            if resolved is _UNRESOLVED_ENV_VALUE:
                loop_values.pop(region_arg.block_arg.logical_id, None)
            else:
                loop_values[region_arg.block_arg.logical_id] = resolved
        return loop_values

    def _advance_region_arguments(
        self,
        op: ForOperation | ForItemsOperation | WhileOperation,
        body_values: dict,
        loop_values: dict,
    ) -> None:
        """Advance all RegionArgs from one body's yielded values in parallel.

        Args:
            op (ForOperation | ForItemsOperation | WhileOperation): Loop whose
                body just completed one materialized iteration.
            body_values (dict): Value environment after evaluating that body.
            loop_values (dict): Persistent loop environment to update for the
                next iteration.
        """
        resolved_yields = [
            self._environment_value(region_arg.yielded, body_values, op.operations)
            for region_arg in op.region_args
        ]
        for region_arg, resolved in zip(op.region_args, resolved_yields, strict=True):
            if resolved is _UNRESOLVED_ENV_VALUE:
                # Never retain the previous iteration's value after an
                # unresolved yield: a symbolic label is safer than displaying
                # a stale concrete angle or index.
                loop_values.pop(region_arg.block_arg.logical_id, None)
            else:
                loop_values[region_arg.block_arg.logical_id] = resolved

    def _publish_region_results(
        self,
        op: ForOperation | ForItemsOperation | WhileOperation,
        loop_values: dict,
        param_values: dict,
    ) -> None:
        """Publish final RegionArg states under their post-loop result IDs.

        Args:
            op (ForOperation | ForItemsOperation | WhileOperation): Completed
                loop operation.
            loop_values (dict): Persistent loop environment after its final
                materialized iteration, or immediately after seeding for a
                zero-trip loop.
            param_values (dict): Enclosing environment to receive results.
        """
        for region_arg in op.region_args:
            resolved = loop_values.get(
                region_arg.block_arg.logical_id, _UNRESOLVED_ENV_VALUE
            )
            if resolved is _UNRESOLVED_ENV_VALUE:
                param_values.pop(region_arg.result.logical_id, None)
            else:
                param_values[region_arg.result.logical_id] = resolved

    @staticmethod
    def _bind_loop_value(
        param_values: dict,
        name: str,
        formal: Value | None,
        value: object,
    ) -> None:
        """Bind a range/items formal by display key and logical identity.

        Args:
            param_values (dict): Environment to mutate.
            name (str): Loop-variable display name used by legacy expression
                formatting.
            formal (Value | None): Identity-bearing loop formal, when present.
            value (object): Concrete entry value for this iteration.
        """
        param_values[f"_loop_{name}"] = value
        if formal is not None:
            param_values[formal.logical_id] = value

    def _bind_for_items_entry(
        self,
        op: ForItemsOperation,
        entry_key: object,
        entry_value: object,
        param_values: dict,
    ) -> None:
        """Bind one materialized ForItems key/value pair into an environment.

        Args:
            op (ForItemsOperation): Items loop owning the key/value formals.
            entry_key (object): Materialized scalar, tuple, or IR tuple key.
            entry_value (object): Materialized scalar or IR value entry.
            param_values (dict): Iteration environment to mutate.
        """
        if hasattr(entry_key, "elements"):
            key_elements = tuple(entry_key.elements)
        elif isinstance(entry_key, tuple):
            key_elements = entry_key
        else:
            key_elements = (entry_key,)

        key_formals = op.key_var_values or ()
        if (
            op.key_is_vector
            and len(key_formals) == 1
            and isinstance(key_formals[0], ArrayValue)
        ):
            formal = key_formals[0]
            resolved_elements: list[object] = []
            for element in key_elements:
                resolved = (
                    self._environment_value(element, param_values)
                    if isinstance(element, Value)
                    else element
                )
                if resolved is _UNRESOLVED_ENV_VALUE:
                    break
                resolved_elements.append(resolved)
            else:
                resolved_key = tuple(resolved_elements)
                if op.key_vars:
                    self._bind_loop_value(
                        param_values,
                        op.key_vars[0],
                        formal,
                        resolved_key,
                    )
                param_values[f"_array_data_{formal.logical_id}"] = resolved_key
                if formal.shape:
                    param_values[formal.shape[0].logical_id] = len(resolved_key)
        else:
            for index, (name, element) in enumerate(zip(op.key_vars, key_elements)):
                formal = key_formals[index] if index < len(key_formals) else None
                if isinstance(element, Value):
                    resolved = self._environment_value(element, param_values)
                else:
                    resolved = element
                if resolved is not _UNRESOLVED_ENV_VALUE:
                    self._bind_loop_value(param_values, name, formal, resolved)

        if isinstance(entry_value, Value):
            resolved_value = self._environment_value(entry_value, param_values)
        else:
            resolved_value = entry_value
        if resolved_value is not _UNRESOLVED_ENV_VALUE:
            self._bind_loop_value(
                param_values,
                op.value_var,
                op.value_var_value,
                resolved_value,
            )

    def _should_inline_at_depth(self, depth: int) -> bool:
        """Return whether legacy call/control blocks expand at this depth."""
        return self.inline and (self.inline_depth is None or depth < self.inline_depth)

    def _should_inline_invoke_at_depth(self, op: InvokeOperation, depth: int) -> bool:
        """Return whether an InvokeOperation should expand visually.

        Args:
            op (InvokeOperation): Invocation being analyzed.
            depth (int): Current nested visualization depth.

        Returns:
            bool: True when the invocation has a body and the current drawing
                options request expansion for that callable class.
        """
        if not isinstance(op.effective_body(), Block):
            return False
        if op.default_policy is CallPolicy.INLINE:
            return self._should_inline_at_depth(depth)
        return self.expand_composite

    @staticmethod
    def _invoke_box_kind(op: InvokeOperation) -> VGateKind:
        """Return the visual box kind for a non-expanded invocation.

        Args:
            op (InvokeOperation): Invocation rendered as a summary box.

        Returns:
            VGateKind: ``BLOCK_BOX`` for inline-by-default qkernel helpers and
            ``COMPOSITE_BOX`` for preserve-box composite/oracle callables.
        """
        if op.default_policy is CallPolicy.INLINE:
            return VGateKind.BLOCK_BOX
        return VGateKind.COMPOSITE_BOX

    def _invoke_actual_inputs(
        self,
        op: InvokeOperation,
        block_value: Block,
    ) -> list[ValueBase]:
        """Return invoke operands aligned to a body block's formal inputs.

        Args:
            op (InvokeOperation): Invocation whose operands should be aligned.
            block_value (Block): Embedded callable body.

        Returns:
            list[ValueBase]: Actual inputs ordered to match
                ``block_value.input_values``.
        """
        if op.attrs.get("kind") == "composite":
            return self._align_actuals_to_formals(
                block_value.input_values,
                quantum_actuals=list(op.target_qubits),
                classical_actuals=list(op.parameters),
            )
        return list(op.operands)

    @staticmethod
    def _invoke_qubit_operands(op: InvokeOperation) -> list[Value]:
        """Return invoke operands that should occupy quantum wires.

        Args:
            op (InvokeOperation): Invocation to inspect.

        Returns:
            list[Value]: Quantum operands, using composite arity metadata when
                present and falling back to all quantum operands otherwise.
        """
        if op.attrs.get("kind") == "composite":
            return [
                v
                for v in list(op.control_qubits) + list(op.target_qubits)
                if isinstance(v.type, QubitType)
            ]
        return [v for v in op.operands if isinstance(v.type, QubitType)]

    def build_qubit_map(
        self, graph: "Block"
    ) -> tuple[dict[str, int], dict[int, str], int]:
        """Build mapping from qubit logical_id to wire indices.

        In SSA form, each operation creates new Values via next_version(),
        which preserves logical_id. This means all versions of a qubit share
        the same logical_id, so we only need logical_id-based tracking.

        Args:
            graph: Computation block.

        Returns:
            Tuple of (qubit_map, qubit_names, num_qubits).
        """
        qubit_map: dict[str, int] = {}
        qubit_names: dict[int, str] = {}
        next_idx = 0

        def map_block_results(
            operands: list[Value],
            results: list[Value],
            logical_id_remap: dict[str, str],
            param_values: dict | None = None,
        ) -> None:
            """Map block output logical_ids to the same wire indices as their inputs.

            For each (operand, result) pair that is a qubit, resolves the operand's
            logical_id via logical_id_remap, looks up its wire index in qubit_map,
            and assigns the same index to the result's logical_id. If the operand
            is not yet registered, both are assigned a new wire index.

            Args:
                operands: Input values passed to the block (actual arguments).
                results: Output values returned from the block.
                logical_id_remap: Mapping from formal-parameter logical_ids to
                    actual-argument logical_ids.
                param_values: Parameter values for resolving symbolic indices.
            """
            nonlocal next_idx
            for operand, result in zip(operands, results):
                if not isinstance(result.type, QubitType):
                    continue
                lid = self._resolve_array_element_lid(
                    operand, qubit_map, logical_id_remap, param_values
                )
                qubit_idx = qubit_map.get(lid)
                if qubit_idx is not None:
                    qubit_map[result.logical_id] = qubit_idx
                    # When both operand and result are Vector[Qubit]
                    # ArrayValues (e.g. a controlled-U whose
                    # sub-kernel target is a whole ``Vector[Qubit]``,
                    # or a sub-kernel call that returns ``q`` after
                    # mutating it), copy every per-element key from
                    # the operand's lid to the result's lid.  Without
                    # this, downstream ops that walk
                    # ``parent_array.logical_id -> {lid}_[i]`` (most
                    # visibly: ``qmc.iqft`` expanded inline to
                    # per-element CP / H gates whose ``parent_array``
                    # is the post-controlled-U next-version
                    # ``ArrayValue``) fall through the resolver's
                    # element-key lookup and the CompositeGate /
                    # ControlledU dispatch fresh-allocates a phantom
                    # wire per element.
                    if isinstance(operand, ArrayValue) and isinstance(
                        result, ArrayValue
                    ):
                        operand_lid = logical_id_remap.get(
                            operand.logical_id, operand.logical_id
                        )
                        prefix = f"{operand_lid}_["
                        prefix_len = len(prefix)
                        for key, idx in list(qubit_map.items()):
                            if key.startswith(prefix) and key.endswith("]"):
                                suffix = key[prefix_len:]
                                new_key = f"{result.logical_id}_[{suffix}"
                                if new_key not in qubit_map:
                                    qubit_map[new_key] = idx
                else:
                    # Guard: symbolic array element whose parent is known
                    # → don't create a new wire; let layout resolve dynamically
                    if (
                        hasattr(operand, "parent_array")
                        and operand.parent_array is not None
                        and hasattr(operand, "element_indices")
                        and operand.element_indices
                        and not operand.element_indices[0].is_constant()
                    ):
                        parent_lid = logical_id_remap.get(
                            operand.parent_array.logical_id,
                            operand.parent_array.logical_id,
                        )
                        if parent_lid in qubit_map:
                            qubit_map[result.logical_id] = qubit_map[parent_lid]
                            continue

                    qubit_map[operand.logical_id] = next_idx
                    qubit_map[result.logical_id] = next_idx
                    next_idx += 1

        def map_callable_outputs(
            body_outputs: Sequence[ValueLike],
            call_results: Sequence[ValueLike],
            logical_id_remap: dict[str, str],
        ) -> None:
            """Alias caller-local invocation results to callee output wires.

            Caller-local result values have fresh logical IDs, including for
            quantum arrays allocated inside the callee. Map those results to
            the wires registered while walking the callee body so later nested
            calls do not allocate phantom wires for the fresh result IDs.

            Args:
                body_outputs (Sequence[ValueLike]): Callee outputs in signature
                    order.
                call_results (Sequence[ValueLike]): Caller-local invocation
                    results in the same order.
                logical_id_remap (dict[str, str]): Formal-to-actual logical-ID
                    remapping active inside the callee.

            Raises:
                ValueError: If the callee output count differs from the call
                    result count.
            """
            for body_output, call_result in zip(
                body_outputs,
                call_results,
                strict=True,
            ):
                if not isinstance(call_result.type, QubitType):
                    continue
                source_lid = logical_id_remap.get(
                    body_output.logical_id,
                    body_output.logical_id,
                )
                if source_lid in qubit_map:
                    qubit_map[call_result.logical_id] = qubit_map[source_lid]
                if isinstance(body_output, ArrayValue) and isinstance(
                    call_result, ArrayValue
                ):
                    source_prefix = f"{source_lid}_["
                    prefix_length = len(source_prefix)
                    for key, wire in list(qubit_map.items()):
                        if key.startswith(source_prefix) and key.endswith("]"):
                            suffix = key[prefix_length:]
                            qubit_map[f"{call_result.logical_id}_[{suffix}"] = wire

        def build_chains(
            ops: list[Operation],
            logical_id_remap: dict[str, str] | None = None,
            depth: int = 0,
            param_values: dict | None = None,
        ) -> None:
            """Register qubit logical_ids by walking operations recursively.

            For QInitOperation, registers new qubits (scalar or array elements).
            For inlined calls (inline=True), builds a logical_id_remap from
            block formal parameters to actual arguments, then recurses into the
            block body. For CastOperation, propagates the source qubit's wire
            index to the cast result.

            GateOperation and similar are no-ops because next_version() preserves
            logical_id.

            Args:
                ops: List of operations to process.
                logical_id_remap: Mapping from block formal-parameter logical_ids
                    to actual-argument logical_ids. Only non-empty when recursing
                    into inlined callable bodies.
                depth: Current nesting depth for inline_depth checking.
                param_values: Parameter values for resolving symbolic indices.
            """
            nonlocal next_idx
            if logical_id_remap is None:
                logical_id_remap = {}
            if param_values is None:
                param_values = {}

            for op in ops:
                if isinstance(op, QInitOperation):
                    qubits = op.results[0]

                    # If the qubits is an array, we need to register each element separately.
                    if isinstance(qubits, ArrayValue):
                        # If this assertion fails without adding a new quantum array type, there are some bugs.
                        # If you added a new quantum array type, please also update the visualization logic here.
                        assertion_message = f"[FOR DEVELOPER] QInitOperation produced non-qubit array type: {qubits.type}"
                        assert isinstance(qubits.type, QubitType), assertion_message

                        if len(qubits.shape) == 1:  # 1D array (Vector)
                            size_value = qubits.shape[0]
                            array_size = None
                            if size_value.is_constant():
                                array_size = size_value.get_const()
                            elif param_values:
                                # Try resolving from param_values (e.g. inlined block)
                                pv = param_values.get(size_value.logical_id)
                                if pv is not None and isinstance(pv, (int, float)):
                                    array_size = int(pv)
                            if array_size is None:
                                raise ValueError(
                                    f"Cannot visualize circuit: qubit array "
                                    f"'{qubits.name}' has symbolic size. "
                                    f"Please provide concrete values for all "
                                    f"size parameters when calling draw()."
                                )

                            # create_bound_input stores qubit-array sizes as ints.
                            # If this assertion fails without adding a new quantum array type, there are some bugs.
                            assertion_message = f"[FOR DEVELOPER] Qubit array '{qubits.name}' has non-integer size: {array_size}"
                            assert isinstance(array_size, int), assertion_message

                            for i in range(array_size):
                                element_key = f"{qubits.logical_id}_[{i}]"
                                if element_key not in qubit_map:
                                    qubit_map[element_key] = next_idx
                                    qubit_names[next_idx] = f"{qubits.name}[{i}]"
                                    next_idx += 1
                            # Store base index for array
                            qubit_map[qubits.logical_id] = qubit_map.get(
                                f"{qubits.logical_id}_[0]", next_idx
                            )
                            continue
                        # TODO: 2D, 3D arrays if needed.
                        else:
                            raise NotImplementedError(
                                f"Cannot visualize circuit: qubit array "
                                f"'{qubits.name}' has unsupported rank "
                                f"{len(qubits.shape)}."
                            )
                    # If the qubits is actually just a qubit, then register the qubit.
                    # This else branch relied on the specification that QInitOperation is always producing qubits.
                    else:
                        if qubits.logical_id not in qubit_map:
                            qubit_map[qubits.logical_id] = next_idx
                            qubit_names[next_idx] = qubits.name
                            next_idx += 1

                elif isinstance(op, InvokeOperation):
                    if self._should_inline_invoke_at_depth(op, depth):
                        block_value = op.effective_body()
                        assert isinstance(block_value, Block)
                        actual_inputs = self._invoke_actual_inputs(op, block_value)
                        new_remap, child_param_values = (
                            self._build_block_value_mappings(
                                block_value,
                                actual_inputs,
                                dict(logical_id_remap),
                                param_values,
                                qubit_map=qubit_map,
                            )
                        )
                        build_chains(
                            block_value.operations,
                            new_remap,
                            depth + 1,
                            child_param_values,
                        )
                        map_callable_outputs(
                            block_value.output_values,
                            op.results,
                            new_remap,
                        )
                        qubit_operands = self._invoke_qubit_operands(op)
                        qubit_results = [
                            v for v in op.results if isinstance(v.type, QubitType)
                        ]
                        map_block_results(
                            qubit_operands,
                            qubit_results,
                            logical_id_remap,
                            param_values,
                        )
                    else:
                        qubit_operands = self._invoke_qubit_operands(op)
                        qubit_results = [
                            v for v in op.results if isinstance(v.type, QubitType)
                        ]
                        map_block_results(
                            qubit_operands,
                            qubit_results,
                            logical_id_remap,
                            param_values,
                        )

                elif isinstance(op, CastOperation):
                    assert op.operands and op.results, (
                        "[FOR DEVELOPER] CastOperation must have operands and results. "
                        "If this assertion fails, there is a bug in the IR construction."
                    )
                    source = op.operands[0]
                    result = op.results[0]
                    source_lid = logical_id_remap.get(
                        source.logical_id, source.logical_id
                    )
                    qubit_idx = qubit_map.get(source_lid)
                    if qubit_idx is not None:
                        qubit_map[result.logical_id] = qubit_idx
                    else:
                        resolved = self._resolve_array_element_lid(
                            source, qubit_map, logical_id_remap, param_values
                        )
                        resolved_idx = qubit_map.get(resolved)
                        if resolved_idx is not None:
                            qubit_map[result.logical_id] = resolved_idx

                elif isinstance(
                    op, ControlledUOperation
                ) and self._should_inline_at_depth(depth):
                    block_value = op.block
                    if isinstance(block_value, Block):
                        new_remap = dict(logical_id_remap)
                        # ``op.target_operands`` lays out quantum sub-args
                        # first then classical (frontend ``_build_operands``
                        # builds them in that order), while
                        # ``block_value.input_values`` is in the wrapped
                        # kernel's signature-declared order (qubit and
                        # classical possibly interleaved).  Filter both
                        # to their quantum / classical halves before
                        # zipping so a signature like
                        # ``def sub(theta, q: Vector[Qubit])`` does not
                        # mis-pair the ``theta`` formal with the ``q``
                        # actual.
                        quantum_formals = [
                            iv
                            for iv in block_value.input_values
                            if isinstance(iv.type, QubitType)
                        ]
                        classical_formals = [
                            iv
                            for iv in block_value.input_values
                            if not isinstance(iv.type, QubitType)
                        ]
                        quantum_actuals = [
                            v
                            for v in op.target_operands
                            if isinstance(v.type, QubitType)
                        ]
                        classical_actuals = [
                            v
                            for v in op.target_operands
                            if not isinstance(v.type, QubitType)
                        ]
                        for dummy_input, actual_input in zip(
                            quantum_formals, quantum_actuals
                        ):
                            actual_lid = self._resolve_array_element_lid(
                                actual_input,
                                qubit_map,
                                logical_id_remap,
                                param_values,
                            )
                            new_remap[dummy_input.logical_id] = actual_lid
                            if actual_lid not in qubit_map:
                                qubit_map[actual_lid] = next_idx
                                next_idx += 1
                        # Build child_param_values for non-qubit inputs.
                        child_param_values = dict(param_values) if param_values else {}
                        for dummy_input, actual_input in zip(
                            classical_formals, classical_actuals
                        ):
                            c = actual_input.get_const()
                            if c is not None:
                                child_param_values[dummy_input.logical_id] = c
                            elif (
                                param_values and actual_input.logical_id in param_values
                            ):
                                child_param_values[dummy_input.logical_id] = (
                                    param_values[actual_input.logical_id]
                                )
                        build_chains(
                            block_value.operations,
                            new_remap,
                            depth + 1,
                            child_param_values,
                        )
                    qubit_operands = [
                        v
                        for v in list(op.control_operands) + list(op.target_operands)
                        if isinstance(v.type, QubitType)
                    ]
                    qubit_results = [
                        v for v in op.results if isinstance(v.type, QubitType)
                    ]
                    map_block_results(
                        qubit_operands, qubit_results, logical_id_remap, param_values
                    )

                elif (
                    isinstance(op, InverseBlockOperation)
                    and self.expand_composite
                    and op.implementation_block is not None
                ):
                    block_value = op.implementation_block
                    new_remap = dict(logical_id_remap)
                    actual_inputs = self._align_actuals_to_formals(
                        block_value.input_values,
                        quantum_actuals=list(op.target_qubits),
                        classical_actuals=list(op.parameters),
                    )
                    new_remap, child_param_values = self._build_block_value_mappings(
                        block_value,
                        actual_inputs,
                        new_remap,
                        param_values,
                        qubit_map=qubit_map,
                    )
                    build_chains(
                        block_value.operations,
                        new_remap,
                        depth + 1,
                        child_param_values,
                    )
                    qubit_operands = list(op.control_qubits) + list(op.target_qubits)
                    qubit_results = [
                        v for v in op.results if isinstance(v.type, QubitType)
                    ]
                    map_block_results(
                        qubit_operands, qubit_results, logical_id_remap, param_values
                    )

                elif isinstance(op, ForOperation):
                    start, stop, step = self._evaluate_loop_range(op, param_values)
                    if stop is not None and (not self.fold_loops or self.inline):
                        loop_values = self._seed_region_arguments(op, param_values)
                        for iter_value in range(start, stop, step):
                            child_pv = dict(loop_values)
                            self._bind_loop_value(
                                child_pv,
                                op.loop_var,
                                op.loop_var_value,
                                iter_value,
                            )
                            # Pre-evaluate body BinOps for this iteration
                            # so resolvers that consult ``param_values``
                            # (notably ``_resolve_view_chain_to_root``
                            # for slice bounds emitted via ``_uint_min``)
                            # find a concrete answer instead of bailing
                            # and falling through to fresh-wire allocation
                            # in ``map_block_results``.  Mirrors what
                            # ``_build_vfor`` already does for the visual
                            # IR build.
                            self._evaluate_loop_body_intermediates(
                                op.operations, child_pv
                            )
                            build_chains(
                                op.operations,
                                logical_id_remap,
                                depth + 1,
                                child_pv,
                            )
                            self._advance_region_arguments(op, child_pv, loop_values)
                        self._publish_region_results(op, loop_values, param_values)
                    else:
                        child_pv = self._seed_region_arguments(op, param_values)
                        build_chains(
                            op.operations,
                            logical_id_remap,
                            depth + 1,
                            child_pv,
                        )
                        self._replay_for_value_flow(
                            op,
                            param_values,
                            logical_id_remap,
                            _MAX_FOLDED_VALUE_REPLAY_ITERATIONS,
                        )

                elif isinstance(op, WhileOperation):
                    child_pv = self._seed_region_arguments(op, param_values)
                    build_chains(op.operations, logical_id_remap, depth + 1, child_pv)

                elif isinstance(op, IfOperation):
                    true_values = dict(param_values)
                    build_chains(
                        op.true_operations,
                        logical_id_remap,
                        depth + 1,
                        true_values,
                    )
                    false_values = dict(param_values)
                    build_chains(
                        op.false_operations,
                        logical_id_remap,
                        depth + 1,
                        false_values,
                    )
                    self._publish_if_results(
                        op, true_values, false_values, param_values, ops
                    )

                elif isinstance(op, ForItemsOperation):
                    dict_value = op.operands[0] if op.operands else None
                    materialized = (
                        self._materialize_dict_entries(
                            dict_value, param_values, logical_id_remap
                        )
                        if dict_value is not None
                        else None
                    )
                    if materialized is not None and (
                        not self.fold_loops or self.inline
                    ):
                        loop_values = self._seed_region_arguments(op, param_values)
                        for entry_key, entry_value in materialized:
                            child_pv = dict(loop_values)
                            self._bind_for_items_entry(
                                op, entry_key, entry_value, child_pv
                            )
                            build_chains(
                                op.operations,
                                logical_id_remap,
                                depth + 1,
                                child_pv,
                            )
                            self._advance_region_arguments(op, child_pv, loop_values)
                        self._publish_region_results(op, loop_values, param_values)
                    else:
                        child_pv = self._seed_region_arguments(op, param_values)
                        build_chains(
                            op.operations,
                            logical_id_remap,
                            depth + 1,
                            child_pv,
                        )
                        self._replay_for_items_value_flow(
                            op,
                            param_values,
                            logical_id_remap,
                            _MAX_FOLDED_VALUE_REPLAY_ITERATIONS,
                        )

                elif isinstance(op, (BinOp, CompOp, CondOp, NotOp)):
                    self._record_classical_result(op, param_values, ops)

                # GateOperation, non-expanded InvokeOperation:
                # No-op — next_version() preserves logical_id

        build_chains(graph.operations, depth=0, param_values={})
        return qubit_map, qubit_names, next_idx

    # ------------------------------------------------------------------
    # Visual IR construction
    # ------------------------------------------------------------------

    def build_visual_ir(
        self,
        graph: "Block",
        qubit_map: dict[str, int],
        qubit_names: dict[int, str],
        num_qubits: int,
    ) -> VisualCircuit:
        """Build a Visual IR tree from the IR block.

        Walks all operations, resolving labels, qubit indices, and widths
        into pre-computed VisualNode dataclasses. The resulting VisualCircuit
        can be consumed by Layout and Renderer without any Analyzer access.

        Args:
            graph: IR computation block.
            qubit_map: Mapping from logical_id to wire index.
            qubit_names: Mapping from wire index to display name.
            num_qubits: Total number of qubit wires.

        Returns:
            VisualCircuit containing the VisualNode tree.
        """
        # Pre-evaluate top-level intermediate BinOps so that callable
        # arguments derived from them (e.g. final_base = reps * 2 * n)
        # resolve to numeric or symbolic strings instead of leaking the
        # IR-internal placeholder name "uint_tmp" downstream.
        top_param_values: dict = {}
        self._evaluate_loop_body_intermediates(graph.operations, top_param_values)
        children = self._build_visual_nodes(
            graph.operations, qubit_map, {}, top_param_values, depth=0, scope_path=()
        )
        return VisualCircuit(
            children=children,
            qubit_map=qubit_map,
            qubit_names=qubit_names,
            num_qubits=num_qubits,
            output_names=getattr(graph, "output_names", None) or [],
        )

    def _build_visual_nodes(
        self,
        ops: list[Operation],
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict,
        depth: int,
        scope_path: tuple,
    ) -> list[VisualNode]:
        """Recursively build VisualNode list from IR operations.

        This is the core dispatch method that replaces both Layout's
        _measure_operations and Renderer's draw_ops dispatch.
        """
        result: list[VisualNode] = []

        for op in ops:
            node_key = (*scope_path, id(op))

            if isinstance(op, QInitOperation):
                result.append(VSkip(node_key=node_key))
                continue

            if isinstance(op, CastOperation):
                result.append(VSkip(node_key=node_key))
                continue

            if isinstance(op, (BinOp, CompOp, CondOp, NotOp)):
                self._record_classical_result(op, param_values, ops)
                continue  # No visual representation

            if isinstance(op, ExpvalOp):
                node = self._build_vexpval(
                    op, node_key, qubit_map, logical_id_remap, param_values
                )
                result.append(node)
                continue

            if isinstance(op, WhileOperation):
                raise NotImplementedError(
                    "Circuit visualization does not yet support WhileOperation. "
                    "This feature will be added in a future release."
                )

            if isinstance(op, IfOperation):
                node = self._build_vif(
                    op,
                    node_key,
                    qubit_map,
                    logical_id_remap,
                    param_values,
                    depth,
                    scope_path,
                    body_operations=ops,
                )
                result.append(node)
                continue

            if isinstance(op, ForItemsOperation):
                node = self._build_vfor_items(
                    op,
                    node_key,
                    qubit_map,
                    logical_id_remap,
                    param_values,
                    depth,
                    scope_path,
                )
                result.append(node)
                continue

            if isinstance(op, ForOperation):
                node = self._build_vfor(
                    op,
                    node_key,
                    qubit_map,
                    logical_id_remap,
                    param_values,
                    depth,
                    scope_path,
                )
                result.append(node)
                continue

            if isinstance(op, InvokeOperation) and self._should_inline_invoke_at_depth(
                op, depth
            ):
                node = self._build_vinline_block(
                    op,
                    node_key,
                    qubit_map,
                    logical_id_remap,
                    param_values,
                    depth,
                    scope_path,
                )
                result.append(node)
                continue

            if isinstance(op, ControlledUOperation) and self._should_inline_at_depth(
                depth
            ):
                node = self._build_vinline_block(
                    op,
                    node_key,
                    qubit_map,
                    logical_id_remap,
                    param_values,
                    depth,
                    scope_path,
                )
                result.append(node)
                continue

            if (
                isinstance(op, InverseBlockOperation)
                and self.expand_composite
                and op.implementation_block is not None
            ):
                node = self._build_vinline_block(
                    op,
                    node_key,
                    qubit_map,
                    logical_id_remap,
                    param_values,
                    depth,
                    scope_path,
                )
                result.append(node)
                continue

            # Generic: GateOperation, callable/control/inverse boxes,
            # MeasureOperation, MeasureVectorOperation, MeasureQFixedOperation
            if isinstance(
                op,
                (
                    GateOperation,
                    MeasureOperation,
                    MeasureVectorOperation,
                    MeasureQFixedOperation,
                    InverseBlockOperation,
                    ControlledUOperation,
                    InvokeOperation,
                ),
            ):
                node = self._build_vgate(
                    op, node_key, qubit_map, logical_id_remap, param_values
                )
                result.append(node)

        return result

    @staticmethod
    def _node_key_in_if_branch(node_key: tuple) -> bool:
        """Whether a node sits inside an if/else branch scope.

        If-branch scopes are tagged with the ``"true"`` / ``"false"`` string
        markers in the node key (see ``_build_vif`` and the layout engine's
        unfolded-if placement). No other scope kind uses string markers, so
        their presence uniquely identifies an if-branch nesting.

        Args:
            node_key (tuple): The node's scope key, of the form
                ``(*scope_path, id(op))``.

        Returns:
            bool: True if any element of ``node_key`` is an if-branch marker,
                meaning the node is nested inside a true/false branch.
        """
        return any(
            isinstance(part, str) and part in _IF_BRANCH_SCOPE_KEYS for part in node_key
        )

    def _build_vgate(
        self,
        op: Operation,
        node_key: tuple,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict,
    ) -> VGate:
        """Build a VGate node for gates, measurements, and block boxes."""
        if isinstance(op, GateOperation):
            label, has_param = self._get_gate_label(op, param_values)
            estimated_width = self._estimate_gate_width(op, param_values)
            qubit_indices: list[int] = []
            for operand in op.operands:
                indices = self._resolve_operand_to_qubit_indices(
                    operand, qubit_map, logical_id_remap, param_values
                )
                if indices is not None:
                    qubit_indices.extend(indices)
            return VGate(
                node_key=node_key,
                label=label,
                qubit_indices=qubit_indices,
                estimated_width=estimated_width,
                kind=VGateKind.GATE,
                gate_type=op.gate_type,
                has_param=has_param,
            )

        if isinstance(op, MeasureOperation):
            gate_width = self.style.gate_width
            qubit_indices = []
            if op.operands:
                indices = self._resolve_operand_to_qubit_indices(
                    op.operands[0], qubit_map, logical_id_remap, param_values
                )
                if indices is not None:
                    qubit_indices.extend(indices)
            return VGate(
                node_key=node_key,
                label="M",
                qubit_indices=qubit_indices,
                estimated_width=gate_width,
                kind=VGateKind.MEASURE,
                terminates_wire=not self._node_key_in_if_branch(node_key),
            )

        if isinstance(op, MeasureVectorOperation):
            gate_width = self.style.gate_width
            qubit_indices = []
            if op.operands:
                indices = self._resolve_operand_to_qubit_indices(
                    op.operands[0], qubit_map, logical_id_remap, param_values
                )
                if indices is not None:
                    qubit_indices = indices
            return VGate(
                node_key=node_key,
                label="M",
                qubit_indices=qubit_indices,
                estimated_width=gate_width,
                kind=VGateKind.MEASURE_VECTOR,
                terminates_wire=not self._node_key_in_if_branch(node_key),
            )

        if isinstance(op, MeasureQFixedOperation):
            # ``MeasureQFixedOperation`` is the HYBRID measurement that
            # ``plan`` later splits into ``MeasureVectorOperation +
            # DecodeQFixedOperation``.  The visualizer works on the
            # pre-plan IR so it sees the unsplit form and must resolve
            # the carrier qubits itself.  The operand is the QFixed
            # ``Value`` produced by the upstream ``CastOperation``,
            # which attaches a ``CastMetadata`` whose
            # ``qubit_logical_ids`` field enumerates the source qubits
            # in carrier order.  Each entry is of the form
            # ``f"{root_logical_id}_{idx}"`` (without brackets);
            # ``QInitOperation`` registers root qubits in
            # ``qubit_map`` under ``f"{root_logical_id}_[{idx}]"``
            # (with brackets).  We bridge the two encodings here so
            # the carrier wires resolve precisely — including the
            # non-contiguous indices a slice-source cast produces
            # (e.g. ``cast(q[1::2], QFixed)`` covers root ``{1, 3}``).
            qubit_indices = []
            if op.operands:
                qubit_indices = self._resolve_qfixed_carrier_indices(
                    op.operands[0], qubit_map, logical_id_remap
                )
            return VGate(
                node_key=node_key,
                label="M",
                qubit_indices=qubit_indices,
                estimated_width=self.style.gate_width,
                kind=VGateKind.MEASURE_VECTOR,
                terminates_wire=not self._node_key_in_if_branch(node_key),
            )

        if isinstance(op, InvokeOperation):
            label = op.name.upper()
            box_width = self._estimate_block_label_box_width(label)
            qubit_indices = []
            for operand in self._invoke_qubit_operands(op):
                indices = self._resolve_operand_to_qubit_indices(
                    operand, qubit_map, logical_id_remap, param_values
                )
                if indices is not None:
                    qubit_indices.extend(indices)
            if not qubit_indices:
                for result_val in op.results:
                    if isinstance(result_val.type, QubitType):
                        indices = self._resolve_operand_to_qubit_indices(
                            result_val, qubit_map, logical_id_remap, param_values
                        )
                        if indices is not None:
                            qubit_indices.extend(indices)
            return VGate(
                node_key=node_key,
                label=label,
                qubit_indices=qubit_indices,
                estimated_width=box_width,
                kind=self._invoke_box_kind(op),
                box_width=box_width,
            )

        if isinstance(op, InverseBlockOperation):
            base_name = op.source_block.name if op.source_block is not None else op.name
            label = f"{base_name.upper()}^-1"
            box_width = self._estimate_block_label_box_width(label)
            qubit_indices = []
            for qval in list(op.control_qubits) + list(op.target_qubits):
                indices = self._resolve_operand_to_qubit_indices(
                    qval, qubit_map, logical_id_remap, param_values
                )
                if indices is not None:
                    qubit_indices.extend(indices)
            return VGate(
                node_key=node_key,
                label=label,
                qubit_indices=qubit_indices,
                estimated_width=box_width,
                kind=VGateKind.COMPOSITE_BOX,
                box_width=box_width,
            )

        if isinstance(op, ControlledUOperation):
            raw_name = getattr(op.block, "name", "U") or "U"
            # Built-in gates wrapped via ``qmc.control(qmc.rx)`` etc.
            # carry their lowercase gate name on ``block.name``; render
            # them with the same TeX label the inline-gate path uses
            # (``$R_x$`` instead of ``rx``).  User kernels fall through
            # unchanged so ``_phase`` stays ``_phase``.
            u_name = _BUILTIN_TEX_LABELS.get(raw_name, raw_name)
            power_val = self._resolve_controlled_u_power(op, param_values)
            # Combine the two recent improvements to the controlled-U box:
            # main (PR #409) now recognises a small set of wrapped gates
            # (X/Z/CX/CZ/SWAP/TOFFOLI with power=1) and renders them with
            # a dedicated native symbol instead of a labelled box; this
            # branch added a parameter-suffix label (``_phase($\theta$=...)``,
            # ``$R_x$(angle=...)``) and a separate ``VGate.power`` field
            # so the renderer can draw an outer ``pow=N`` wrapper box for
            # ``power > 1``.  Both behaviours coexist: when the dedicated
            # symbol applies the label is unused (the wrapped primitives
            # have no classical parameters so ``param_suffix`` is empty
            # anyway), and otherwise the label keeps its full callable
            # name + bound classical parameters.
            controlled_gate_type = self._controlled_u_single_gate_type(op, power_val)
            # Append a parameter suffix when the wrapped block has
            # classical parameters bound at the call site.  Each entry
            # is ``name=value`` where ``name`` is the wrapped kernel's
            # own parameter name (the leading ``ctrl_param_`` Value
            # prefix is stripped) and ``value`` is either the resolved
            # numeric constant or the symbolic name (Greek letters get
            # TeX rendering via ``_format_symbolic_param``).  When
            # ``power_val > 1`` the renderer draws an outer ``pow=N``
            # wrapper box around the inner controlled-U rectangle.
            param_suffix = self._format_controlled_param_suffix(op, param_values)
            label = f"{u_name}{param_suffix}"
            if controlled_gate_type is not None:
                box_width = self.style.gate_width
            else:
                box_width = self._estimate_label_box_width(label)
                if power_val > 1:
                    # Reserve a little extra horizontal space so the
                    # outer wrapper box does not clip into adjacent
                    # gates.
                    box_width += 2 * self.style.power_wrapper_margin
            # Control qubits first, then target qubits
            control_indices: list[int] = []
            for qval in op.control_operands:
                indices = self._resolve_operand_to_qubit_indices(
                    qval, qubit_map, logical_id_remap, param_values
                )
                if indices is not None:
                    control_indices.extend(indices)
            # Symbolic-mode subset selection: when the operation
            # carries a ``control_indices`` list, only those pool
            # slots are wired in as active controls.  The remaining
            # slots are pass-through wires and must not appear as
            # control dots in the diagram.  When every index resolves
            # to a concrete integer (literal or via ``bindings``) we
            # filter the flat control-qubit list; otherwise (e.g. a
            # ``UInt`` expression without a binding) we leave the
            # full list as a safe fallback so the picture still
            # reads sensibly.
            ctrl_idx_values = getattr(op, "control_indices", None)
            if ctrl_idx_values is not None and control_indices:
                resolved_positions: list[int] = []
                all_resolved = True
                for idx_v in ctrl_idx_values:
                    ev = self._evaluate_value(idx_v, param_values)
                    if isinstance(ev, (int, float)):
                        resolved_positions.append(int(ev))
                    else:
                        all_resolved = False
                        break
                if all_resolved and all(
                    0 <= p < len(control_indices) for p in resolved_positions
                ):
                    control_indices = [control_indices[p] for p in resolved_positions]
            target_indices: list[int] = []
            for qval in op.target_operands:
                indices = self._resolve_operand_to_qubit_indices(
                    qval, qubit_map, logical_id_remap, param_values
                )
                if indices is not None:
                    target_indices.extend(indices)
            return VGate(
                node_key=node_key,
                label=label,
                qubit_indices=control_indices + target_indices,
                estimated_width=box_width,
                kind=VGateKind.CONTROLLED_U_BOX,
                gate_type=controlled_gate_type,
                box_width=box_width,
                control_count=len(control_indices),
                power=power_val,
            )

        raise TypeError(
            f"Unsupported operation type for _build_vgate: {type(op).__name__}"
        )

    def _controlled_u_single_gate_type(
        self, op: ControlledUOperation, power: int
    ) -> GateOperationType | None:
        """Return the wrapped gate type when controlled-U is a single gate.

        Args:
            op (ControlledUOperation): Controlled operation whose wrapped
                block may contain one concrete `GateOperation`.
            power (int): Resolved controlled-U power. Values greater than one
                are left in box form so the exponent remains visible.

        Returns:
            GateOperationType | None: Wrapped gate type when the controlled
                block is exactly one gate plus an optional return operation;
                otherwise None.
        """
        if power != 1:
            return None
        block = op.block
        if block is None:
            return None
        body_ops = [
            body_op
            for body_op in block.operations
            if not isinstance(body_op, ReturnOperation)
        ]
        if len(body_ops) != 1 or not isinstance(body_ops[0], GateOperation):
            return None
        gate_type = body_ops[0].gate_type
        if gate_type not in {
            GateOperationType.X,
            GateOperationType.Z,
            GateOperationType.CX,
            GateOperationType.CZ,
            GateOperationType.SWAP,
            GateOperationType.TOFFOLI,
        }:
            return None
        return gate_type

    def _build_vexpval(
        self,
        op: ExpvalOp,
        node_key: tuple,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict,
    ) -> VGate:
        """Build a VGate node for an ExpvalOp."""
        label = "<H>"
        box_width = self._estimate_label_box_width(label)
        qubit_indices: list[int] = []
        if op.operands:
            indices = self._resolve_operand_to_qubit_indices(
                op.operands[0], qubit_map, logical_id_remap, param_values
            )
            if indices is not None:
                qubit_indices = indices
        return VGate(
            node_key=node_key,
            label=label,
            qubit_indices=qubit_indices,
            estimated_width=box_width,
            kind=VGateKind.EXPVAL,
            box_width=box_width,
        )

    def _build_vinline_block(
        self,
        op: ControlledUOperation | InverseBlockOperation | InvokeOperation,
        node_key: tuple,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict,
        depth: int,
        scope_path: tuple,
    ) -> VInlineBlock:
        """Build a VInlineBlock node for an inlined block operation."""
        # Extract block_value, affected_qubits, and actual_inputs based on op type
        if isinstance(op, ControlledUOperation):
            block_value = op.block
            assert isinstance(block_value, Block)
            control_qubit_indices: list[int] = []
            for operand in op.control_operands:
                indices = self._resolve_operand_to_qubit_indices(
                    operand, qubit_map, logical_id_remap, param_values
                )
                if indices is not None:
                    control_qubit_indices.extend(indices)
            affected_qubits = list(control_qubit_indices)
            for operand in op.target_operands:
                indices = self._resolve_operand_to_qubit_indices(
                    operand, qubit_map, logical_id_remap, param_values
                )
                if indices is not None:
                    affected_qubits.extend(indices)
            # ``op.target_operands`` lays out quantum sub-args first
            # then classical (frontend ``_build_operands`` order),
            # while ``block_value.input_values`` is in the wrapped
            # kernel's signature-declared order (quantum and
            # classical possibly interleaved).  Reorder the actuals
            # to match the formal order before
            # :meth:`_build_block_value_mappings` zips them, so a
            # sub-signature like ``def sub(theta, q: Vector[Qubit])``
            # does not mis-pair ``theta`` with the ``q`` actual.
            actual_inputs = self._align_actuals_to_formals(
                block_value.input_values,
                quantum_actuals=[
                    v for v in op.target_operands if isinstance(v.type, QubitType)
                ],
                classical_actuals=[
                    v for v in op.target_operands if not isinstance(v.type, QubitType)
                ],
            )
            u_name = getattr(block_value, "name", "U") or "U"
            block_name = u_name
        elif isinstance(op, InvokeOperation):
            block_value = op.effective_body()
            assert isinstance(block_value, Block)
            affected_qubits = []
            for operand in self._invoke_qubit_operands(op):
                indices = self._resolve_operand_to_qubit_indices(
                    operand, qubit_map, logical_id_remap, param_values
                )
                if indices is not None:
                    affected_qubits.extend(indices)
            actual_inputs = self._invoke_actual_inputs(op, block_value)
            block_name = op.name
        elif isinstance(op, InverseBlockOperation):
            block_value = op.implementation_block
            assert isinstance(block_value, Block)
            affected_qubits = []
            for operand in list(op.control_qubits) + list(op.target_qubits):
                indices = self._resolve_operand_to_qubit_indices(
                    operand, qubit_map, logical_id_remap, param_values
                )
                if indices is not None:
                    affected_qubits.extend(indices)
            actual_inputs = self._align_actuals_to_formals(
                block_value.input_values,
                quantum_actuals=list(op.target_qubits),
                classical_actuals=list(op.parameters),
            )
            block_name = op.name
        else:
            raise TypeError(f"Unsupported inline block type: {type(op).__name__}")

        new_logical_id_remap, child_param_values = self._build_block_value_mappings(
            block_value,
            actual_inputs,
            logical_id_remap,
            param_values,
            qubit_map=qubit_map,
        )
        # Inline expansion sees actual arguments, so callee-local compile-time IFs
        # can be lowered before building child visual nodes.
        from qamomile.circuit.transpiler.passes.compile_time_if_lowering import (
            CompileTimeIfLoweringPass,
        )

        block_value = CompileTimeIfLoweringPass(bindings=child_param_values).run(
            block_value
        )

        self._evaluate_loop_body_intermediates(
            block_value.operations, child_param_values
        )

        # Include qubits created inside the block body via QInitOperation
        for body_op in block_value.operations:
            if isinstance(body_op, QInitOperation):
                result_val = body_op.results[0]
                resolved = self._resolve_operand_to_qubit_indices(
                    result_val, qubit_map, logical_id_remap, param_values
                )
                if resolved is not None:
                    for idx in resolved:
                        if idx not in affected_qubits:
                            affected_qubits.append(idx)

        max_gate_width = self._max_block_gate_width(
            block_value.operations, child_param_values
        )

        children = self._build_visual_nodes(
            block_value.operations,
            qubit_map,
            new_logical_id_remap,
            child_param_values,
            depth + 1,
            node_key,
        )

        # Update max_gate_width from children
        children_max = self._max_visual_gate_width(children)
        max_gate_width = max(max_gate_width, children_max)

        border_padding = compute_border_padding(self.style, depth)
        content_width = self._compute_visual_content_width(
            children, max_gate_width, depth
        )
        power = (
            self._resolve_controlled_u_power(op, param_values)
            if isinstance(op, ControlledUOperation)
            else 1
        )
        # Inner box label width (block name only, no power text)
        label_width = self._estimate_label_box_width(block_name)
        if power > 1:
            # Account for outer wrapper box: margin on each side + pow label width
            margin = self.style.power_wrapper_margin
            pow_label_width = self._estimate_label_box_width(f"pow={power}")
            final_width = max(
                label_width + 2 * margin,
                pow_label_width + 2 * margin,
                content_width + 2 * margin,
            )
        else:
            final_width = max(label_width, content_width)

        ctrl_indices = (
            control_qubit_indices if isinstance(op, ControlledUOperation) else []
        )
        return VInlineBlock(
            node_key=node_key,
            label=block_name,
            children=children,
            affected_qubits=affected_qubits,
            control_qubit_indices=ctrl_indices,
            power=power,
            depth=depth,
            border_padding=border_padding,
            max_gate_width=max_gate_width,
            label_width=label_width,
            content_width=content_width,
            final_width=final_width,
        )

    def _build_vfor(
        self,
        op: ForOperation,
        node_key: tuple,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict,
        depth: int,
        scope_path: tuple,
    ) -> VFoldedBlock | VUnfoldedSequence | VSkip:
        """Build a Visual IR node for a ForOperation."""
        start_val, stop_val_raw, step_val = self._evaluate_loop_range(op, param_values)

        if self._is_zero_iteration_loop(start_val, stop_val_raw, step_val):
            loop_values = self._seed_region_arguments(op, param_values)
            self._publish_region_results(op, loop_values, param_values)
            return VSkip(node_key=node_key)

        affected_qubits, affected_qubits_precise = self._analyze_loop_affected_qubits(
            op, qubit_map, logical_id_remap, param_values
        )

        # Folded mode: fold_loops=True or symbolic stop
        if self.fold_loops or stop_val_raw is None:
            header, body_lines, folded_width = self._compute_folded_for_info(
                op, param_values
            )
            self._replay_for_value_flow(
                op,
                param_values,
                logical_id_remap,
                _MAX_FOLDED_VALUE_REPLAY_ITERATIONS,
            )
            return VFoldedBlock(
                node_key=node_key,
                header_label=header,
                body_lines=body_lines,
                affected_qubits=affected_qubits,
                folded_width=folded_width,
                kind=VFoldedKind.FOR,
                affected_qubits_precise=affected_qubits_precise,
            )

        # Unfolded: measure each iteration
        num_iterations = self._compute_loop_iterations(
            start_val, stop_val_raw, step_val
        )
        iterations: list[list[VisualNode]] = []
        iteration_widths: list[float] = []
        loop_values = self._seed_region_arguments(op, param_values)

        for iteration in range(num_iterations):
            iter_value = start_val + iteration * step_val
            child_param_values = dict(loop_values)
            self._bind_loop_value(
                child_param_values,
                op.loop_var,
                op.loop_var_value,
                iter_value,
            )

            self._evaluate_loop_body_intermediates(op.operations, child_param_values)

            children = self._build_visual_nodes(
                op.operations,
                qubit_map,
                logical_id_remap,
                child_param_values,
                depth + 1,
                (*node_key, iteration),
            )
            iter_width = self._sum_visual_widths(children)
            iterations.append(children)
            iteration_widths.append(iter_width)
            self._advance_region_arguments(op, child_param_values, loop_values)

        self._publish_region_results(op, loop_values, param_values)

        return VUnfoldedSequence(
            node_key=node_key,
            iterations=iterations,
            affected_qubits=affected_qubits,
            kind=VUnfoldedKind.FOR,
            iteration_widths=iteration_widths,
            affected_qubits_precise=affected_qubits_precise,
        )

    def _build_vwhile(
        self,
        op: WhileOperation,
        node_key: tuple,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict,
    ) -> VFoldedBlock:
        """Build a VFoldedBlock for a WhileOperation (always folded).

        Note: This method is implemented for future use but currently not called,
        because ``_build_visual_nodes`` raises ``NotImplementedError`` for
        ``WhileOperation``.  It will be connected once While-loop visualization
        is fully enabled.
        """
        affected_qubits, affected_qubits_precise = self._analyze_loop_affected_qubits(
            op, qubit_map, logical_id_remap, param_values
        )
        header = "while cond:"
        local_param_values = dict(param_values)
        self._evaluate_loop_body_intermediates(op.operations, local_param_values)
        body_lines: list[str] = []
        for body_op in op.operations:
            expr = self._format_operation_as_expression(
                body_op,
                set(),
                body_operations=op.operations,
                param_values=local_param_values,
            )
            if expr:
                body_lines.extend(expr.split("\n"))
        # Truncate to 3 lines
        if len(body_lines) > 3:
            body_lines = body_lines[:3] + ["..."]

        folded_width = self._compute_folded_text_width(header, body_lines)

        return VFoldedBlock(
            node_key=node_key,
            header_label=header,
            body_lines=body_lines,
            affected_qubits=affected_qubits,
            folded_width=folded_width,
            kind=VFoldedKind.WHILE,
            affected_qubits_precise=affected_qubits_precise,
        )

    def _build_vif(
        self,
        op: IfOperation,
        node_key: tuple,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict,
        depth: int,
        scope_path: tuple,
        body_operations: list[Operation] | None = None,
    ) -> VFoldedBlock | VUnfoldedSequence:
        """Build a Visual IR node for an IfOperation.

        Compile-time resolvable ``if``s are already lowered away by
        ``CompileTimeIfLoweringPass`` before visual analysis runs, so this method
        only handles conditions that survive — measurement-backed runtime
        ``if`` and symbolic (unbound) classical ``if``. When ``fold_ifs`` is
        enabled it returns a single ``VFoldedBlock`` summarizing the true branch
        and, when present, the else branch; otherwise it returns a
        ``VUnfoldedSequence`` carrying the surviving branch alternatives for
        side-by-side rendering.

        Args:
            op (IfOperation): The if/else operation to visualize.
            node_key (tuple): Stable identity key for layout/render lookup.
            qubit_map (dict[str, int]): Mapping from logical_id to wire index.
            logical_id_remap (dict[str, str]): Mapping from formal-parameter
                logical_ids to actual-argument logical_ids in scope.
            param_values (dict): Resolved loop/parameter values in scope.
            depth (int): Current nesting depth for child expansion.
            scope_path (tuple): Path of enclosing scope keys for child node keys.
            body_operations (list[Operation] | None): Operations of the scope
                enclosing ``op``, used to spell the condition predicate (e.g.
                ``flag == 1``). Defaults to None, which falls back to the
                anonymous ``if cond:`` label.

        Returns:
            VFoldedBlock | VUnfoldedSequence: A folded summary box when
                ``fold_ifs`` is set, otherwise an unfolded sequence containing
                the true branch and, when present, the false branch.
        """
        affected_qubits, affected_qubits_precise = self._collect_if_affected_qubits(
            op, qubit_map, logical_id_remap, param_values
        )

        condition_expr = self._format_condition_expr(
            op.condition, body_operations, param_values
        )
        condition_label = f"if {condition_expr}:" if condition_expr else "if cond:"
        condition_measure_info = self._condition_measure_info(
            op.condition,
            body_operations,
            qubit_map,
            logical_id_remap,
            param_values,
            scope_path,
        )
        condition_measure_node_key: tuple | None = None
        condition_measure_qubit_indices: list[int] = []
        if condition_measure_info is not None:
            condition_measure_node_key, condition_measure_qubit_indices = (
                condition_measure_info
            )
        if not affected_qubits and condition_measure_qubit_indices:
            affected_qubits = list(dict.fromkeys(condition_measure_qubit_indices))
            affected_qubits_precise = True
        elif not affected_qubits and qubit_map:
            # Empty symbolic IFs still need a display wire for their branch box.
            affected_qubits = [min(qubit_map.values())]
            affected_qubits_precise = False

        if self.fold_ifs:
            true_values = dict(param_values)
            self._replay_value_flow(op.true_operations, true_values, logical_id_remap)
            false_values = dict(param_values)
            self._replay_value_flow(op.false_operations, false_values, logical_id_remap)
            self._publish_if_results(
                op,
                true_values,
                false_values,
                param_values,
                body_operations or [],
            )
            body_lines = self._format_folded_body_lines(
                op.true_operations, param_values
            )
            if op.false_operations:
                if not body_lines:
                    body_lines.append("pass")
                false_lines = self._format_folded_body_lines(
                    op.false_operations, param_values
                )
                if false_lines:
                    body_lines.append(f"else: {false_lines[0].lstrip()}")
                    body_lines.extend(false_lines[1:])
                else:
                    body_lines.append("else: pass")
            if len(body_lines) > 3:
                body_lines = body_lines[:3] + ["..."]

            folded_width = self._compute_folded_text_width(condition_label, body_lines)

            return VFoldedBlock(
                node_key=node_key,
                header_label=condition_label,
                body_lines=body_lines,
                affected_qubits=affected_qubits,
                folded_width=folded_width,
                kind=VFoldedKind.IF,
                affected_qubits_precise=affected_qubits_precise,
                condition_measure_node_key=condition_measure_node_key,
                condition_measure_qubit_indices=condition_measure_qubit_indices,
            )

        # Unfolded: build both branches
        true_param_values = dict(param_values)
        self._evaluate_loop_body_intermediates(op.true_operations, true_param_values)
        true_children = self._build_visual_nodes(
            op.true_operations,
            qubit_map,
            logical_id_remap,
            true_param_values,
            depth + 1,
            (*node_key, "true"),
        )
        true_width = self._sum_visual_widths(true_children)

        false_param_values = dict(param_values)
        self._evaluate_loop_body_intermediates(op.false_operations, false_param_values)
        false_children = self._build_visual_nodes(
            op.false_operations,
            qubit_map,
            logical_id_remap,
            false_param_values,
            depth + 1,
            (*node_key, "false"),
        )
        false_width = self._sum_visual_widths(false_children)

        self._publish_if_results(
            op,
            true_param_values,
            false_param_values,
            param_values,
            body_operations or [],
        )

        iterations = [true_children]
        iteration_widths = [true_width]
        branch_labels = [condition_label]
        if false_children:
            iterations.append(false_children)
            iteration_widths.append(false_width)
            branch_labels.append("else:")

        branch_label_widths = [
            self._estimate_label_box_width(label) for label in branch_labels
        ]

        return VUnfoldedSequence(
            node_key=node_key,
            iterations=iterations,
            affected_qubits=affected_qubits,
            kind=VUnfoldedKind.IF,
            iteration_widths=iteration_widths,
            condition_label=condition_label,
            affected_qubits_precise=affected_qubits_precise,
            condition_label_width=branch_label_widths[0],
            branch_label_widths=branch_label_widths,
            condition_measure_node_key=condition_measure_node_key,
            condition_measure_qubit_indices=condition_measure_qubit_indices,
        )

    def _condition_measure_info(
        self,
        value: ValueBase | None,
        body_operations: list[Operation] | None,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict,
        scope_path: tuple,
    ) -> tuple[tuple, list[int]] | None:
        """Return the direct measurement node that produced an IF condition.

        Only a condition value produced directly by a measurement operation is
        treated as measurement-derived. Compound predicates such as
        ``not bit`` or ``bit and flag`` are produced by classical operations and
        intentionally return None.

        Args:
            value (ValueBase | None): The IF condition value to inspect.
            body_operations (list[Operation] | None): Operations in the scope
                enclosing the IF operation.
            qubit_map (dict[str, int]): Mapping from logical_id to wire index.
            logical_id_remap (dict[str, str]): Mapping from formal-parameter
                logical_ids to actual-argument logical_ids in scope.
            param_values (dict): Resolved loop/parameter values in scope.
            scope_path (tuple): Path of enclosing scope keys.

        Returns:
            tuple[tuple, list[int]] | None: The producer node key and measured
                qubit indices, or None when the condition is not produced
                directly by a measurement operation.
        """
        if value is None or not body_operations:
            return None

        producer = self._find_direct_measure_producer_op(value, body_operations)
        if producer is None:
            return None

        qubit_indices = self._measure_condition_qubit_indices(
            value, producer, qubit_map, logical_id_remap, param_values
        )
        return (*scope_path, id(producer)), qubit_indices

    def _find_direct_measure_producer_op(
        self,
        value: ValueBase,
        body_operations: list[Operation],
    ) -> MeasureOperation | MeasureVectorOperation | MeasureQFixedOperation | None:
        """Find a measurement operation that directly produced ``value``.

        Args:
            value (ValueBase): The condition value to match against
                measurement results.
            body_operations (list[Operation]): Operations in the enclosing
                scope.

        Returns:
            MeasureOperation | MeasureVectorOperation | MeasureQFixedOperation | None:
                The direct measurement producer, or None if ``value`` is
                produced by another operation.
        """
        target_uuid = getattr(value, "uuid", None)
        parent_array = getattr(value, "parent_array", None)
        parent_uuid = getattr(parent_array, "uuid", None)
        for candidate in body_operations:
            if not isinstance(
                candidate,
                (MeasureOperation, MeasureVectorOperation, MeasureQFixedOperation),
            ):
                continue
            for result in candidate.results:
                result_uuid = getattr(result, "uuid", None)
                if target_uuid is not None and result_uuid == target_uuid:
                    return candidate
                if parent_uuid is not None and result_uuid == parent_uuid:
                    return candidate
        return None

    def _measure_condition_qubit_indices(
        self,
        value: ValueBase,
        op: MeasureOperation | MeasureVectorOperation | MeasureQFixedOperation,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict,
    ) -> list[int]:
        """Resolve the measurement wires used by an IF condition value.

        Args:
            value (ValueBase): IF condition value produced by ``op``.
            op (MeasureOperation | MeasureVectorOperation | MeasureQFixedOperation):
                Measurement operation that directly produced ``value``.
            qubit_map (dict[str, int]): Mapping from logical_id to wire index.
            logical_id_remap (dict[str, str]): Mapping from formal-parameter
                logical_ids to actual-argument logical_ids in scope.
            param_values (dict): Resolved loop/parameter values in scope.

        Returns:
            list[int]: Wire indices relevant to the condition. For a vector
                measurement element such as ``bits[1]``, this narrows to the
                corresponding measured qubit when the element index resolves.
        """
        if isinstance(op, MeasureVectorOperation):
            wire = self._measure_vector_condition_wire(
                value, op, qubit_map, logical_id_remap, param_values
            )
            if wire is not None:
                return [wire]
        return self._measure_qubit_indices(
            op, qubit_map, logical_id_remap, param_values
        )

    def _measure_vector_condition_wire(
        self,
        value: ValueBase,
        op: MeasureVectorOperation,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict,
    ) -> int | None:
        """Resolve ``measure(qs)[i]`` to the measured ``qs[i]`` wire.

        Args:
            value (ValueBase): IF condition value, possibly an element of the
                vector measurement result.
            op (MeasureVectorOperation): Vector measurement producer.
            qubit_map (dict[str, int]): Mapping from logical_id to wire index.
            logical_id_remap (dict[str, str]): Mapping from formal-parameter
                logical_ids to actual-argument logical_ids in scope.
            param_values (dict): Resolved loop/parameter values in scope.

        Returns:
            int | None: The corresponding measured qubit wire, or None when
                the condition is not a resolvable vector element.
        """
        if not op.operands:
            return None
        index = self._condition_array_element_index(value, param_values)
        if index is None:
            return None
        return self._resolve_array_operand_index_to_qubit(
            op.operands[0], index, qubit_map, logical_id_remap, param_values
        )

    def _condition_array_element_index(
        self,
        value: ValueBase,
        param_values: dict,
    ) -> int | None:
        """Resolve the first array-element index of a condition value.

        Args:
            value (ValueBase): Condition value to inspect.
            param_values (dict): Resolved loop/parameter values in scope.

        Returns:
            int | None: The concrete element index, or None when ``value`` is
                not a resolvable array element.
        """
        if not isinstance(value, Value):
            return None
        if not (hasattr(value, "parent_array") and value.parent_array is not None):
            return None
        if not (hasattr(value, "element_indices") and value.element_indices):
            return None

        index_value = value.element_indices[0]
        if index_value.is_constant():
            index = index_value.get_const()
        else:
            index = self._evaluate_value(index_value, param_values)
        if index is None:
            return None
        return int(index)

    def _resolve_array_operand_index_to_qubit(
        self,
        operand: Value,
        index: int,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict,
    ) -> int | None:
        """Resolve one element of a measured qubit array operand.

        Args:
            operand (Value): Quantum array operand of ``MeasureVectorOperation``.
            index (int): Concrete element index selected from the measurement
                result.
            qubit_map (dict[str, int]): Mapping from logical_id to wire index.
            logical_id_remap (dict[str, str]): Mapping from formal-parameter
                logical_ids to actual-argument logical_ids in scope.
            param_values (dict): Resolved loop/parameter values in scope.

        Returns:
            int | None: The qubit wire corresponding to ``operand[index]``, or
                None when the array operand cannot be resolved.
        """
        if not isinstance(operand, ArrayValue):
            return None
        resolved_lid = logical_id_remap.get(operand.logical_id, operand.logical_id)
        if getattr(operand, "slice_of", None) is not None:
            chain = self._resolve_view_chain_to_root(operand, param_values)
            if chain is None:
                return None
            root_av, start, step = chain
            root_lid = logical_id_remap.get(root_av.logical_id, root_av.logical_id)
            root_index = start + step * index
            root_element_key = f"{root_lid}_[{root_index}]"
            if root_element_key in qubit_map:
                return qubit_map[root_element_key]
            if root_lid in qubit_map:
                return qubit_map[root_lid] + root_index
            return None

        element_key = f"{resolved_lid}_[{index}]"
        if element_key in qubit_map:
            return qubit_map[element_key]
        if resolved_lid in qubit_map:
            return qubit_map[resolved_lid] + index
        return None

    def _measure_qubit_indices(
        self,
        op: MeasureOperation | MeasureVectorOperation | MeasureQFixedOperation,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict,
    ) -> list[int]:
        """Resolve the qubit wires touched by a measurement operation.

        Args:
            op (MeasureOperation | MeasureVectorOperation | MeasureQFixedOperation):
                Measurement operation whose quantum operand should be resolved.
            qubit_map (dict[str, int]): Mapping from logical_id to wire index.
            logical_id_remap (dict[str, str]): Mapping from formal-parameter
                logical_ids to actual-argument logical_ids in scope.
            param_values (dict): Resolved loop/parameter values in scope.

        Returns:
            list[int]: Wire indices touched by ``op``. Empty when the operand
                cannot be resolved.
        """
        if not op.operands:
            return []
        if isinstance(op, MeasureQFixedOperation):
            return self._resolve_qfixed_carrier_indices(
                op.operands[0], qubit_map, logical_id_remap
            )
        indices = self._resolve_operand_to_qubit_indices(
            op.operands[0], qubit_map, logical_id_remap, param_values
        )
        return indices or []

    def _format_folded_body_lines(
        self,
        operations: list[Operation],
        param_values: dict,
    ) -> list[str]:
        """Format operations for a folded control-flow summary body.

        Args:
            operations (list[Operation]): Branch or loop body operations to
                summarize.
            param_values (dict): Resolved loop/parameter values in scope.

        Returns:
            list[str]: Human-readable operation expressions in order.
        """
        local_param_values = dict(param_values)
        self._evaluate_loop_body_intermediates(operations, local_param_values)
        body_lines: list[str] = []
        for body_op in operations:
            expr = self._format_operation_as_expression(
                body_op,
                set(),
                body_operations=operations,
                param_values=local_param_values,
            )
            if expr:
                body_lines.extend(expr.split("\n"))
        return body_lines

    def _format_condition_expr(
        self,
        value: ValueBase | None,
        body_operations: list[Operation] | None,
        param_values: dict | None,
        depth: int = 0,
    ) -> str | None:
        """Spell an IfOperation condition as source-like text.

        Walks the classical producer chain (CompOp / CondOp / NotOp / BinOp)
        that computes ``value`` so a symbolic ``if`` renders as, e.g.,
        ``flag == 1`` instead of an anonymous placeholder. A value with a
        meaningful name (a measurement bit such as ``q0_measured``, or a bound
        parameter) short-circuits to that name; a constant to its literal.

        Args:
            value (ValueBase | None): The condition value
                (``IfOperation.condition``) or a sub-operand reached during
                recursion. None yields None.
            body_operations (list[Operation] | None): Operations of the scope
                enclosing the ``if``, searched to find the producer of
                ``value``. None disables producer lookup (name/const only).
            param_values (dict | None): Resolved loop/parameter values used to
                fold a symbolic operand to a constant when possible.
            depth (int): Current recursion depth; guards against runaway or
                cyclic producer chains. Defaults to 0.

        Returns:
            str | None: A human-readable predicate (e.g. ``"flag == 1"``,
                ``"not done"``, ``"i < n"``), or None when ``value`` cannot be
                described from name, constant, or a recognized producer.
        """
        if value is None:
            return None

        # A meaningful name (measurement bit, bound parameter) wins.
        name = getattr(value, "name", None)
        if name and not self._is_internal_temp_name(name):
            return name

        # Constant literal, possibly via param_values resolution.
        if isinstance(value, Value):
            const = self._evaluate_value(value, param_values or {})
            if const is not None:
                return str(const)

        if depth >= 4 or not body_operations:
            return None

        producer = self._find_producer_op(value, body_operations)
        if producer is None:
            return None

        def sub(operand: ValueBase | None) -> str:
            """Format a binary operand.

            Args:
                operand (ValueBase | None): Operand to format recursively.

            Returns:
                str: Formatted operand, or ``?`` when unknown.
            """
            return (
                self._format_condition_expr(
                    operand, body_operations, param_values, depth + 1
                )
                or "?"
            )

        if isinstance(producer, CompOp) and producer.kind in _COMP_OP_SYMBOLS:
            symbol = _COMP_OP_SYMBOLS[producer.kind]
            return f"{sub(producer.operands[0])} {symbol} {sub(producer.operands[1])}"
        if isinstance(producer, CondOp) and producer.kind in _COND_OP_SYMBOLS:
            symbol = _COND_OP_SYMBOLS[producer.kind]
            return f"{sub(producer.operands[0])} {symbol} {sub(producer.operands[1])}"
        if isinstance(producer, NotOp):
            return f"not {sub(producer.operands[0])}"
        if isinstance(producer, BinOp) and producer.kind in _BIN_OP_SYMBOLS:
            symbol = _BIN_OP_SYMBOLS[producer.kind]
            return f"{sub(producer.operands[0])} {symbol} {sub(producer.operands[1])}"
        return None

    def _find_producer_op(
        self,
        value: ValueBase,
        body_operations: list[Operation],
    ) -> Operation | None:
        """Find the operation in scope whose result is ``value``.

        Args:
            value (ValueBase): The value whose producing operation is sought.
            body_operations (list[Operation]): Operations of the scope to scan.

        Returns:
            Operation | None: The first operation whose results include a value
                with the same UUID as ``value``, or None if no producer is in
                ``body_operations`` (e.g. ``value`` is a block input).
        """
        target = getattr(value, "uuid", None)
        if target is None:
            return None
        for candidate in body_operations:
            for result in candidate.results:
                if getattr(result, "uuid", None) == target:
                    return candidate
        return None

    def _build_vfor_items(
        self,
        op: ForItemsOperation,
        node_key: tuple,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict,
        depth: int,
        scope_path: tuple,
    ) -> VFoldedBlock | VUnfoldedSequence | VSkip:
        """Build a Visual IR node for a ForItemsOperation."""
        affected_qubits, affected_qubits_precise = self._analyze_loop_affected_qubits(
            op, qubit_map, logical_id_remap, param_values
        )

        # Build label
        key_str = ", ".join(op.key_vars) if op.key_vars else "k"
        if len(op.key_vars) > 1:
            key_str = f"({key_str})"
        dict_value = op.operands[0] if op.operands else None
        if dict_value is None:
            dict_name = "dict"
        else:
            dict_name = getattr(dict_value, "name", None)
            if dict_name is None or self._is_internal_temp_name(dict_name):
                dict_name = "dict"
        header = f"for {key_str}, {op.value_var} in {dict_name}"

        # Materialize entries
        materialized = (
            self._materialize_dict_entries(dict_value, param_values, logical_id_remap)
            if dict_value is not None
            else None
        )

        if self.fold_loops or materialized is None:
            loop_vars = set(op.key_vars) | {op.value_var}
            body_lines: list[str] = []
            for body_op in op.operations:
                expr = self._format_operation_as_expression(
                    body_op,
                    loop_vars,
                    param_values=param_values,
                    body_operations=op.operations,
                )
                if expr:
                    body_lines.extend(expr.split("\n"))

            folded_width = self._compute_folded_text_width(header, body_lines)
            self._replay_for_items_value_flow(
                op,
                param_values,
                logical_id_remap,
                _MAX_FOLDED_VALUE_REPLAY_ITERATIONS,
            )
            return VFoldedBlock(
                node_key=node_key,
                header_label=header,
                body_lines=body_lines,
                affected_qubits=affected_qubits,
                folded_width=folded_width,
                kind=VFoldedKind.FOR_ITEMS,
                affected_qubits_precise=affected_qubits_precise,
            )

        entries = materialized
        if not entries:
            loop_values = self._seed_region_arguments(op, param_values)
            self._publish_region_results(op, loop_values, param_values)
            return VSkip(node_key=node_key)

        # Unfolded: iterate materialized entries
        iterations: list[list[VisualNode]] = []
        iteration_widths: list[float] = []
        loop_values = self._seed_region_arguments(op, param_values)

        for entry_key, entry_value in entries:
            child_param_values = dict(loop_values)
            self._bind_for_items_entry(op, entry_key, entry_value, child_param_values)

            self._evaluate_loop_body_intermediates(op.operations, child_param_values)

            children = self._build_visual_nodes(
                op.operations,
                qubit_map,
                logical_id_remap,
                child_param_values,
                depth + 1,
                (*node_key, len(iterations)),
            )
            iter_width = self._sum_visual_widths(children)
            iterations.append(children)
            iteration_widths.append(iter_width)
            self._advance_region_arguments(op, child_param_values, loop_values)

        self._publish_region_results(op, loop_values, param_values)

        return VUnfoldedSequence(
            node_key=node_key,
            iterations=iterations,
            affected_qubits=affected_qubits,
            kind=VUnfoldedKind.FOR_ITEMS,
            iteration_widths=iteration_widths,
            affected_qubits_precise=affected_qubits_precise,
        )

    def _compute_folded_for_info(
        self, op: ForOperation, param_values: dict
    ) -> tuple[str, list[str], float]:
        """Compute header, body lines, and folded width for a ForOperation."""
        range_str = self._format_range_str(op, set(), param_values)
        header = f"for {op.loop_var} in {range_str}"

        # Evaluate intermediate BinOp results so that nested expressions
        # (e.g., a = omega + 1; b = a / 2) resolve to symbolic names
        # instead of exposing internal temporaries like "float_tmp".
        local_param_values = dict(param_values)
        self._evaluate_loop_body_intermediates(op.operations, local_param_values)

        body_lines: list[str] = []
        for body_op in op.operations:
            expr = self._format_operation_as_expression(
                body_op,
                {op.loop_var},
                param_values=local_param_values,
                body_operations=op.operations,
            )
            if expr:
                body_lines.extend(expr.split("\n"))

        folded_width = self._compute_folded_text_width(header, body_lines)
        return header, body_lines, folded_width

    def _compute_folded_text_width(self, header: str, body_lines: list[str]) -> float:
        """Compute the width needed for a folded block's text content."""
        scale = self.style.subfont_size / self.style.font_size
        header_width = len(header) * self.style.char_width_bold * scale
        max_body_chars = max((len(line) for line in body_lines), default=0)
        body_width = max_body_chars * self.style.char_width_monospace * scale
        text_width = max(header_width, body_width)
        margin = 0.15  # Extra margin for folded blocks
        return max(
            self.style.folded_loop_width,
            text_width + 2 * self.style.text_padding + margin,
        )

    def _sum_visual_widths(self, children: list[VisualNode]) -> float:
        """Sum widths of VisualNode children with inter-element gaps."""
        gap = self.style.gate_gap
        total = 0.0
        count = 0

        for child in children:
            if isinstance(child, VSkip):
                continue
            elif isinstance(child, VGate):
                total += child.estimated_width
            elif isinstance(child, VInlineBlock):
                total += child.final_width
            elif isinstance(child, VFoldedBlock):
                total += child.folded_width
            elif isinstance(child, VUnfoldedSequence):
                total += sum(child.iteration_widths)
                n_iters = len(child.iterations)
                if n_iters > 1:
                    total += gap * (n_iters - 1)
            count += 1

        if count > 1:
            total += gap * (count - 1)

        return total

    def _max_visual_gate_width(self, children: list[VisualNode]) -> float:
        """Find the maximum VGate.estimated_width in the VisualNode tree."""
        max_w = 0.0
        for child in children:
            if isinstance(child, VGate):
                max_w = max(max_w, child.estimated_width)
            elif isinstance(child, VInlineBlock):
                max_w = max(max_w, self._max_visual_gate_width(child.children))
            elif isinstance(child, VUnfoldedSequence):
                for iter_children in child.iterations:
                    max_w = max(max_w, self._max_visual_gate_width(iter_children))
        return max_w

    def _compute_visual_content_width(
        self,
        children: list[VisualNode],
        max_gate_width: float,
        depth: int,
    ) -> float:
        """Compute the total content width of VisualNode children inside a block."""
        border_padding = compute_border_padding(self.style, depth)
        border_extent = (
            max_gate_width / 2 + border_padding + self.style.gate_text_padding
        )
        total = self._sum_visual_widths(children)
        total += 2 * border_extent
        return total

    @staticmethod
    def _is_zero_iteration_loop(
        start_val: int, stop_val_raw: int | None, step_val: int
    ) -> bool:
        """Check if a loop has zero iterations.

        Returns True only when stop is concrete and the range is empty.

        Args:
            start_val: Loop start value.
            stop_val_raw: Loop stop value, or None if symbolic.
            step_val: Loop step value.

        Returns:
            True if the loop would produce zero iterations.
        """
        if stop_val_raw is None:
            return False
        if step_val > 0 and start_val >= stop_val_raw:
            return True
        if step_val < 0 and start_val <= stop_val_raw:
            return True
        return False

    @staticmethod
    def _qubit_bearing_operands(op: Operation) -> list[Value] | None:
        """Return the operands of `op` that may reference qubits.

        Used by the loop- and if-affect analyzers to pick out operands
        to feed into ``_resolve_operand_to_affected_qubits``. Returns
        None for control-flow operations, which the caller must handle
        via recursion rather than operand resolution.

        Args:
            op: IR operation to inspect.

        Returns:
            A list of operand Values to resolve, or None when the op is
            a control-flow construct (For/While/If/ForItems) that the
            caller handles separately.
        """
        if isinstance(op, (GateOperation, ControlledUOperation)):
            return list(op.operands)
        if isinstance(op, InvokeOperation):
            if op.attrs.get("kind") == "composite":
                return list(op.control_qubits) + list(op.target_qubits)
            return list(op.operands)
        if isinstance(op, InverseBlockOperation):
            return list(op.control_qubits) + list(op.target_qubits)
        if isinstance(op, (MeasureOperation, MeasureVectorOperation)):
            return list(op.operands[:1])
        return None

    def _analyze_loop_affected_qubits(
        self,
        op: ForOperation | WhileOperation | ForItemsOperation,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str] | None = None,
        param_values: dict | None = None,
    ) -> tuple[list[int], bool]:
        """Analyze which qubits are affected by a loop operation.

        Args:
            op: Loop operation (ForOperation, WhileOperation, or ForItemsOperation).
            qubit_map: Mapping from logical_id to qubit index.
            logical_id_remap: Mapping from dummy logical_ids to actual logical_ids.
            param_values: Parameter values for resolving loop range and indices.

        Returns:
            Tuple ``(indices, is_precise)`` where ``indices`` is the list
            of affected qubit wire indices and ``is_precise`` is True
            when every operand was resolved during the precise
            iteration walk (including any nested control-flow
            recursion). False when the conservative fallback was used,
            meaning the result may over-approximate.
        """
        if logical_id_remap is None:
            logical_id_remap = {}
        if param_values is None:
            param_values = {}

        # Try precise iteration-based analysis when loop range is known (ForOperation only)
        if isinstance(op, ForOperation):
            start_val, stop_val_raw, step_val = self._evaluate_loop_range(
                op, param_values
            )
            if (
                stop_val_raw is not None
                and step_val != 0
                and not self._is_zero_iteration_loop(start_val, stop_val_raw, step_val)
            ):
                num_iters = len(range(int(start_val), int(stop_val_raw), int(step_val)))
                if 0 < num_iters <= 100:
                    precise_affected: set[int] = set()
                    all_resolved = True
                    loop_values = self._seed_region_arguments(op, param_values)
                    for iter_val in range(
                        int(start_val), int(stop_val_raw), int(step_val)
                    ):
                        iter_params = dict(loop_values)
                        self._bind_loop_value(
                            iter_params,
                            op.loop_var,
                            op.loop_var_value,
                            iter_val,
                        )
                        self._replay_value_flow(op.operations, iter_params)
                        for inner_op in op.operations:
                            if isinstance(
                                inner_op,
                                (ForOperation, WhileOperation, ForItemsOperation),
                            ):
                                nested, nested_precise = (
                                    self._analyze_loop_affected_qubits(
                                        inner_op,
                                        qubit_map,
                                        logical_id_remap,
                                        iter_params,
                                    )
                                )
                                precise_affected.update(nested)
                                if not nested_precise:
                                    all_resolved = False
                                continue
                            if isinstance(inner_op, IfOperation):
                                nested, nested_precise = (
                                    self._collect_if_affected_qubits(
                                        inner_op,
                                        qubit_map,
                                        logical_id_remap,
                                        iter_params,
                                    )
                                )
                                precise_affected.update(nested)
                                if not nested_precise:
                                    all_resolved = False
                                continue
                            operands = self._qubit_bearing_operands(inner_op)
                            if operands is None:
                                # Unknown op type — degrade to the recursive
                                # fallback to avoid silently dropping qubits.
                                all_resolved = False
                                continue
                            for operand in operands:
                                indices = self._resolve_operand_to_affected_qubits(
                                    operand,
                                    qubit_map,
                                    logical_id_remap,
                                    iter_params,
                                )
                                if indices is None:
                                    if isinstance(operand.type, QubitType):
                                        all_resolved = False
                                else:
                                    precise_affected.update(indices)
                        self._advance_region_arguments(op, iter_params, loop_values)
                    if all_resolved and precise_affected:
                        return list(precise_affected), True

        # Fallback to conservative analysis
        affected: set[int] = set()

        def add_affected(operand: Value) -> None:
            """Resolve `operand` and merge its wire indices into `affected`."""
            indices = self._resolve_operand_to_affected_qubits(
                operand, qubit_map, logical_id_remap, param_values
            )
            if indices is not None:
                affected.update(indices)

        def collect_from_ops(ops: list[Operation]) -> None:
            """Recursively collect qubit indices from all operations into `affected` set."""
            for inner_op in ops:
                if isinstance(
                    inner_op,
                    (ForOperation, WhileOperation, ForItemsOperation),
                ):
                    collect_from_ops(inner_op.operations)
                    continue
                if isinstance(inner_op, IfOperation):
                    collect_from_ops(inner_op.true_operations)
                    collect_from_ops(inner_op.false_operations)
                    continue
                operands = self._qubit_bearing_operands(inner_op)
                if operands is None:
                    continue
                for operand in operands:
                    add_affected(operand)

        collect_from_ops(op.operations)
        return list(affected), False

    def _collect_if_affected_qubits(
        self,
        op: IfOperation,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str] | None = None,
        param_values: dict | None = None,
    ) -> tuple[list[int], bool]:
        """Collect qubit indices affected by an IfOperation.

        Recursively walks both true and false branches to collect all
        qubit indices that are operands of any operation.

        Args:
            op: IfOperation to analyze.
            qubit_map: Mapping from logical_id to qubit wire index.
            logical_id_remap: Mapping from dummy logical_ids to actual logical_ids.
            param_values: Parameter values for resolving expressions.

        Returns:
            Tuple ``(indices, is_precise)``. ``is_precise`` is True
            when every operand — including those inside nested
            control flow — resolved cleanly.
        """
        if logical_id_remap is None:
            logical_id_remap = {}
        if param_values is None:
            param_values = {}

        affected: set[int] = set()
        is_precise = True

        def collect_qubits(ops: list[Operation]) -> None:
            nonlocal is_precise
            for inner_op in ops:
                if isinstance(
                    inner_op,
                    (ForOperation, WhileOperation, ForItemsOperation),
                ):
                    collect_qubits(inner_op.operations)
                    continue
                if isinstance(inner_op, IfOperation):
                    collect_qubits(inner_op.true_operations)
                    collect_qubits(inner_op.false_operations)
                    continue
                operands = self._qubit_bearing_operands(inner_op)
                if operands is None:
                    continue
                for operand in operands:
                    indices = self._resolve_operand_to_affected_qubits(
                        operand, qubit_map, logical_id_remap, param_values
                    )
                    if indices is None:
                        if isinstance(operand.type, QubitType):
                            is_precise = False
                    else:
                        affected.update(indices)

        collect_qubits(op.true_operations)
        collect_qubits(op.false_operations)
        return list(affected), is_precise

    def _format_controlled_param_suffix(
        self,
        op: ControlledUOperation,
        param_values: dict,
    ) -> str:
        """Render the ``(p1=v1, p2=v2, ...)`` suffix for a controlled-U box.

        Walks ``op.param_operands`` (the classical parameters bound to
        the wrapped block at the call site) and returns the formatted
        suffix that follows the wrapped-callable name on the box label.
        The leading ``ctrl_param_`` prefix that
        ``ControlledGate.__call__`` attaches to each Value name is
        stripped so the rendered label uses the wrapped kernel's own
        parameter name.  Bound numeric constants are formatted via
        :meth:`_format_parameter`; unresolved symbolic parameters fall
        back to :meth:`_format_symbolic_param` so Greek-letter names
        like ``theta`` render in TeX.

        Args:
            op (ControlledUOperation): The controlled-U op whose
                classical parameters should be rendered.
            param_values (dict): The same ``param_values`` mapping the
                surrounding ``_build_vgate`` call uses to resolve
                Values inside inline blocks.

        Returns:
            str: A string of the form ``"(name=value, ...)"`` when
            ``op.param_operands`` is non-empty, or ``""`` when the
            wrapped block takes no classical parameters.
        """
        params = list(getattr(op, "param_operands", ()) or ())
        if not params:
            return ""
        parts: list[str] = []
        for v in params:
            # Drop the ctrl_param_ prefix to recover the wrapped
            # kernel's own parameter name.
            raw = v.name or ""  # type: ignore[unreachable]
            if raw.startswith(_CTRL_PARAM_PREFIX):  # type: ignore[unreachable]
                raw = raw[len(_CTRL_PARAM_PREFIX) :]  # type: ignore[index]
            pname_raw = raw or "?"
            # Route the parameter name through the same symbolic
            # formatter used by inline gates so Greek-letter names
            # (``theta``, ``alpha``, ``beta``, ...) render in TeX
            # rather than as ASCII text.
            pname_disp = self._format_symbolic_param(pname_raw)

            # Resolve the value: bound constant first, then logical_id
            # remap, then a generic _evaluate_value, then the symbolic
            # name as a final fallback.
            const_val = v.get_const() if hasattr(v, "get_const") else None
            if isinstance(const_val, (int, float)):
                pval = self._format_parameter(const_val)
            elif param_values and v.logical_id in param_values:  # type: ignore[operator]
                resolved = param_values[v.logical_id]
                if isinstance(resolved, (int, float)):
                    pval = self._format_parameter(resolved)
                elif isinstance(resolved, str):
                    pval = self._format_symbolic_param(resolved)
                else:
                    pval = pname_disp
            else:
                evaluated = self._evaluate_value(v, param_values or {})
                if isinstance(evaluated, (int, float)):
                    pval = self._format_parameter(evaluated)
                else:
                    pval = pname_disp

            parts.append(f"{pname_disp}={pval}")
        return f"({', '.join(parts)})"

    def _resolve_controlled_u_power(
        self,
        op: ControlledUOperation,
        param_values: dict,
    ) -> int:
        """Resolve ControlledUOperation.power to a concrete integer.

        Handles int literals, Value expressions (e.g., 2**k from QPE),
        and init_value fallback.

        Args:
            op: ControlledUOperation whose power to resolve.
            param_values: Parameter values for evaluating Value expressions.

        Returns:
            Resolved power as int (minimum 1).
        """
        if isinstance(op.power, int):
            return op.power
        if isinstance(op.power, Value):
            evaluated = self._evaluate_value(op.power, param_values)
            if isinstance(evaluated, (int, float)):
                return int(evaluated)
            init = getattr(op.power, "init_value", None)
            if isinstance(init, int):
                return init
        return 1

    def _evaluate_value(
        self,
        value: Value,
        param_values: dict,
        operations: list[Operation] | None = None,
    ) -> int | float | None:
        """Evaluate a Value to a concrete number using param_values and BinOp resolution.

        Handles constants, named parameters, and binary arithmetic expressions.

        Args:
            value: IR Value to evaluate.
            param_values: Mapping from logical_ids or parameter names to concrete values.
            operations: Operation list to search for defining BinOps.
                Falls back to self.graph.operations if None.

        Returns:
            Concrete numeric value, or None if unresolvable.
        """
        # 1. Direct constant
        if value.is_constant():
            return value.get_const()

        # 2. Named parameter
        if value.is_parameter():
            name = value.parameter_name()
            if name and name in param_values:
                param_value = param_values[name]
                if isinstance(param_value, (int, float)):
                    return param_value

        # 3. Check param_values by logical_id
        vid = value.logical_id
        if vid in param_values:
            param_value = param_values[vid]
            if isinstance(param_value, (int, float)):
                return param_value

        # 3.5. Loop variable name-based lookup
        if value.name:
            loop_key = f"_loop_{value.name}"
            if loop_key in param_values:
                param_value = param_values[loop_key]
                if isinstance(param_value, (int, float)):
                    return param_value

        # 3.6. Array element access (e.g. edges[idx, 0])
        if hasattr(value, "parent_array") and value.parent_array is not None:
            parent = value.parent_array
            const_array = parent.get_const_array()
            if const_array is None:
                const_array = param_values.get(f"_array_data_{parent.logical_id}")
            if (
                const_array is not None
                and hasattr(value, "element_indices")
                and value.element_indices
            ):
                indices = []
                for idx_val in value.element_indices:
                    resolved_idx = self._evaluate_value(
                        idx_val, param_values, operations
                    )
                    if resolved_idx is None:
                        break
                    indices.append(int(resolved_idx))
                else:
                    try:
                        result = const_array
                        for idx in indices:
                            result = result[idx]
                        return result
                    except (IndexError, TypeError):
                        pass

        # 4. Look for a defining scalar classical operation in this scope.
        if operations is None:
            operations = getattr(self, "graph", None)
            operations = operations.operations if operations else []

        for op in operations:
            if (
                isinstance(op, (BinOp, CompOp, CondOp, NotOp))
                and op.results
                and op.results[0].uuid == value.uuid
            ):
                return self._evaluate_classical_operation(op, param_values, operations)

        return None

    def _evaluate_classical_operation(
        self,
        op: BinOp | CompOp | CondOp | NotOp,
        param_values: dict,
        operations: list[Operation],
    ) -> int | float | bool | None:
        """Evaluate one scalar classical operation in a visualization scope.

        Args:
            op (BinOp | CompOp | CondOp | NotOp): Operation to evaluate.
            param_values (dict): Current visualization value environment.
            operations (list[Operation]): Scope used to resolve operand
                producers recursively.

        Returns:
            int | float | bool | None: Concrete scalar result, or None when an
                operand remains symbolic or the operation is not foldable.
        """
        from qamomile.circuit.transpiler.passes.eval_utils import (
            FoldPolicy,
            fold_classical_op,
        )

        return fold_classical_op(
            op,
            lambda operand: self._evaluate_value(operand, param_values, operations),
            parameters=set(),
            policy=FoldPolicy.COMPILE_TIME,
        )

    def _record_classical_result(
        self,
        op: BinOp | CompOp | CondOp | NotOp,
        param_values: dict,
        operations: list[Operation],
    ) -> None:
        """Record a classical operation result in the current environment.

        Args:
            op (BinOp | CompOp | CondOp | NotOp): Operation whose result should
                become visible to later operations.
            param_values (dict): Environment to mutate.
            operations (list[Operation]): Enclosing operation scope.
        """
        if not op.results:
            return
        result_value = op.results[0]
        resolved = self._evaluate_classical_operation(op, param_values, operations)
        if resolved is not None:
            param_values[result_value.logical_id] = resolved
            return
        if isinstance(op, BinOp):
            symbolic = self._build_symbolic_binop(op, param_values)
            if symbolic is not None:
                param_values[result_value.logical_id] = symbolic
                return
        param_values.pop(result_value.logical_id, None)

    def _publish_if_results(
        self,
        op: IfOperation,
        true_values: dict,
        false_values: dict,
        param_values: dict,
        enclosing_operations: list[Operation],
    ) -> None:
        """Publish every determinable If merge into the parent environment.

        Args:
            op (IfOperation): Branch operation whose merges should be
                interpreted.
            true_values (dict): Environment after replaying the true branch.
            false_values (dict): Environment after replaying the false branch.
            param_values (dict): Parent environment to receive merge results.
            enclosing_operations (list[Operation]): Scope containing the
                condition producer.
        """
        condition = self._evaluate_value(
            op.condition, param_values, enclosing_operations
        )
        selected_true = bool(condition) if condition is not None else None

        for merge in op.iter_merges():
            true_value = self._environment_value(
                merge.true_value, true_values, op.true_operations
            )
            false_value = self._environment_value(
                merge.false_value, false_values, op.false_operations
            )
            if selected_true is True:
                resolved = true_value
            elif selected_true is False:
                resolved = false_value
            elif (
                true_value is not _UNRESOLVED_ENV_VALUE
                and false_value is not _UNRESOLVED_ENV_VALUE
                and true_value == false_value
            ):
                resolved = true_value
            else:
                resolved = _UNRESOLVED_ENV_VALUE

            if resolved is _UNRESOLVED_ENV_VALUE:
                param_values.pop(merge.result.logical_id, None)
            else:
                param_values[merge.result.logical_id] = resolved

    def _replay_if_value_flow(
        self,
        op: IfOperation,
        param_values: dict,
        enclosing_operations: list[Operation],
        logical_id_remap: dict[str, str],
    ) -> None:
        """Replay branch-local scalar flow and publish resolvable merges.

        Args:
            op (IfOperation): Branch operation to replay.
            param_values (dict): Parent visualization value environment.
            enclosing_operations (list[Operation]): Scope containing ``op``.
            logical_id_remap (dict[str, str]): Inline formal-to-actual mapping
                used when nested items loops materialize bound dictionaries.
        """
        true_values = dict(param_values)
        self._replay_value_flow(op.true_operations, true_values, logical_id_remap)
        false_values = dict(param_values)
        self._replay_value_flow(op.false_operations, false_values, logical_id_remap)
        self._publish_if_results(
            op,
            true_values,
            false_values,
            param_values,
            enclosing_operations,
        )

    def _replay_for_value_flow(
        self,
        op: ForOperation,
        param_values: dict,
        logical_id_remap: dict[str, str],
        max_iterations: int | None,
    ) -> bool:
        """Replay a concrete range loop's scalar RegionArg recurrence.

        Args:
            op (ForOperation): Range loop to replay.
            param_values (dict): Parent visualization value environment.
            logical_id_remap (dict[str, str]): Inline formal-to-actual mapping
                forwarded to nested items loops.
            max_iterations (int | None): Maximum number of iterations to
                replay, or None for no limit.

        Returns:
            bool: True when the complete loop was replayed and its results
                were published (trivially so for a loop that carries
                nothing), otherwise False for a symbolic or over-budget
                loop.
        """
        if not op.region_args:
            # Nothing to publish: every replayed value lands in a
            # per-iteration scratch environment that is discarded, and
            # nested carries only escape through THIS loop's region
            # results. Skipping avoids the multiplicative replay cost
            # for deeply nested concrete loops that carry nothing.
            return True
        start, stop, step = self._evaluate_loop_range(op, param_values)
        if stop is None:
            return False
        iteration_range = range(start, stop, step)
        if max_iterations is not None:
            try:
                if len(iteration_range) > max_iterations:
                    return False
            except OverflowError:
                return False

        loop_values = self._seed_region_arguments(op, param_values)
        for iteration_value in iteration_range:
            body_values = dict(loop_values)
            self._bind_loop_value(
                body_values,
                op.loop_var,
                op.loop_var_value,
                iteration_value,
            )
            self._replay_value_flow(op.operations, body_values, logical_id_remap)
            self._advance_region_arguments(op, body_values, loop_values)
        self._publish_region_results(op, loop_values, param_values)
        return True

    def _replay_for_items_value_flow(
        self,
        op: ForItemsOperation,
        param_values: dict,
        logical_id_remap: dict[str, str],
        max_iterations: int | None,
    ) -> bool:
        """Replay a materialized items loop's scalar RegionArg recurrence.

        Args:
            op (ForItemsOperation): Items loop to replay.
            param_values (dict): Parent visualization value environment.
            logical_id_remap (dict[str, str]): Inline formal-to-actual mapping
                used to find forwarded bound dictionary data.
            max_iterations (int | None): Maximum number of entries to replay,
                or None for no limit.

        Returns:
            bool: True when every entry was replayed and results were
                published (trivially so for a loop that carries nothing),
                otherwise False for an unbound or over-budget dictionary.
        """
        if not op.region_args:
            # Mirror _replay_for_value_flow: a carry-less loop publishes
            # nothing, so the replay is pure discarded scratch work.
            return True
        dict_value = op.operands[0] if op.operands else None
        if dict_value is None:
            return False
        entries = self._materialize_dict_entries(
            dict_value, param_values, logical_id_remap
        )
        if entries is None:
            return False
        if max_iterations is not None and len(entries) > max_iterations:
            return False

        loop_values = self._seed_region_arguments(op, param_values)
        for entry_key, entry_value in entries:
            body_values = dict(loop_values)
            self._bind_for_items_entry(op, entry_key, entry_value, body_values)
            self._replay_value_flow(op.operations, body_values, logical_id_remap)
            self._advance_region_arguments(op, body_values, loop_values)
        self._publish_region_results(op, loop_values, param_values)
        return True

    def _replay_value_flow(
        self,
        operations: list[Operation],
        param_values: dict,
        logical_id_remap: dict[str, str] | None = None,
    ) -> None:
        """Replay scalar value flow through nested static control operations.

        Args:
            operations (list[Operation]): Ordered IR operations to replay.
            param_values (dict): Visualization value environment to mutate.
            logical_id_remap (dict[str, str] | None): Inline formal-to-actual
                mapping used for bound dictionaries. Defaults to an empty map.
        """
        logical_id_remap = logical_id_remap or {}
        for operation in operations:
            if isinstance(operation, (BinOp, CompOp, CondOp, NotOp)):
                self._record_classical_result(operation, param_values, operations)
            elif isinstance(operation, ForOperation):
                self._replay_for_value_flow(
                    operation,
                    param_values,
                    logical_id_remap,
                    _MAX_FOLDED_VALUE_REPLAY_ITERATIONS,
                )
            elif isinstance(operation, ForItemsOperation):
                self._replay_for_items_value_flow(
                    operation,
                    param_values,
                    logical_id_remap,
                    _MAX_FOLDED_VALUE_REPLAY_ITERATIONS,
                )
            elif isinstance(operation, IfOperation):
                self._replay_if_value_flow(
                    operation, param_values, operations, logical_id_remap
                )

    def _resolve_symbolic_array_name(
        self, value: Value, param_values: dict
    ) -> str | None:
        """Try to resolve array element indices for symbolic display.

        For array elements like phis[i] where the array is symbolic but
        indices can be resolved (e.g., loop variable i=0), produces
        "phis[0]" instead of "phis[i]".

        Args:
            value: IR Value that may be an array element.
            param_values: Mapping from logical_ids/names to concrete values.

        Returns:
            Resolved name string (e.g., "phis[0]"), or None if not applicable.
        """
        if not (
            hasattr(value, "parent_array")
            and value.parent_array is not None
            and hasattr(value, "element_indices")
            and value.element_indices
        ):
            return None

        parent_lid = value.parent_array.logical_id
        if parent_lid in param_values and isinstance(param_values[parent_lid], str):
            parent_name = param_values[parent_lid]
        else:
            parent_name = value.parent_array.name or "params"
        resolved_indices = []
        for idx_val in value.element_indices:
            resolved = self._evaluate_value(idx_val, param_values)
            if resolved is not None:
                resolved_indices.append(
                    str(int(resolved))
                    if isinstance(resolved, float) and resolved == int(resolved)
                    else str(resolved)
                )
            else:
                return None

        return f"{parent_name}[{','.join(resolved_indices)}]"

    @staticmethod
    def _safe_int_str(val: int | float | None) -> str:
        """Convert a numeric value to an integer string, handling non-integer floats.

        Args:
            val: Numeric value to convert, or None.

        Returns:
            Integer string representation, or "?" if val is None.
        """
        if val is None:
            return "?"
        try:
            return str(int(val))
        except (ValueError, OverflowError):
            return str(val)

    def _format_range_str(
        self, op: ForOperation, loop_vars: set[str], param_values: dict
    ) -> str:
        """Format range string for ForOperation with algebraic fallback.

        Tries concrete resolution first. If stop value is unresolvable,
        falls back to algebraic expression using _format_value_as_expression.
        """
        start_val, stop_val_raw, step_val = self._evaluate_loop_range(op, param_values)
        start_str = self._safe_int_str(start_val)
        step_str = self._safe_int_str(step_val)

        if stop_val_raw is not None:
            stop_str = self._safe_int_str(stop_val_raw)
        elif len(op.operands) > 1:
            stop_str = self._format_value_as_expression(op.operands[1], loop_vars)
        else:
            stop_str = "?"

        if start_val == 0 and step_val == 1:
            return f"qm.range({stop_str})"
        elif step_val == 1:
            return f"qm.range({start_str}, {stop_str})"
        else:
            return f"qm.range({start_str}, {stop_str}, {step_str})"

    def _format_binop_operand(self, value: Value, param_values: dict) -> str | None:
        """Format a BinOp operand as a symbolic string.

        Resolution order: ``param_values[logical_id]`` → ``_loop_<name>``
        fallback for ForOperation/ForItemsOperation loop variables →
        constant → array element access (``arr[i,j]``, recursing on each
        index so loop-bound indices substitute) → named parameter →
        display-name fallback (rejecting IR-internal placeholders).

        The array-element branch is checked *before* ``is_parameter()``
        because an element access on a parameter array reports
        ``is_parameter()`` True and ``parameter_name() == "gammas[layer]"``
        — i.e. the parameter name is the pre-formatted string with the
        unsubstituted loop variable baked in. Returning that verbatim
        would leak ``layer`` into the rendered expression even though
        ForOperation unrolling has already provided the concrete index
        via ``_loop_layer``. Trying the recursive array-element formatter
        first lets the index resolution see those concrete values
        (``gammas[0]``, ``gammas[1]``).

        Args:
            value (Value): IR Value used as a BinOp operand (lhs or rhs).
            param_values (dict): Mapping from ``logical_id`` (or ``_loop_<var>``
                / parameter name) to a resolved entry — either a numeric value
                or a previously-built symbolic string. Numeric values are
                formatted as integers when float-equal to an integer
                (``2.0`` → ``"2"``); otherwise as ``str(value)``.

        Returns:
            str | None: Human-readable string for the operand, or ``None``
                when the operand cannot be resolved by any of the rules
                above (caller treats ``None`` as "give up on this BinOp").
        """
        # Check param_values first (may contain number or symbolic string)
        if value.logical_id in param_values:
            pv = param_values[value.logical_id]
            if isinstance(pv, (int, float)):
                if isinstance(pv, float) and pv == int(pv):
                    return str(int(pv))
                return str(pv)
            if isinstance(pv, str):
                return pv
        # Loop variable: ForOperation/ForItemsOperation unrolling writes
        # `_loop_<var>` entries keyed by the variable *name* rather than the
        # operand's logical_id, so loop-bound operands like ``Jij`` are not
        # found by the logical_id lookup above. Mirror the same fallback
        # ``_evaluate_value`` already uses so concrete entry values (e.g.
        # ``0.5``) substitute into rendered angle expressions instead of
        # leaking the loop-variable name (``Jij*gamma``).
        if value.name:
            loop_key = f"_loop_{value.name}"
            if loop_key in param_values:
                pv = param_values[loop_key]
                if isinstance(pv, (int, float)):
                    if isinstance(pv, float) and pv == int(pv):
                        return str(int(pv))
                    return str(pv)
                if isinstance(pv, str):
                    return pv
        # Constant
        const = value.get_const()
        if const is not None:
            if isinstance(const, float) and const == int(const):
                return str(int(const))
            return str(const)
        # Array element (e.g., weights[e]). Checked before is_parameter()
        # so unrolled loop indices substitute (`gammas[layer]` →
        # `gammas[0]`) instead of leaking via the pre-formatted
        # parameter_name string `value.parameter_name()` returns for
        # element accesses on parameter arrays.
        if (
            hasattr(value, "parent_array")
            and value.parent_array is not None
            and hasattr(value, "element_indices")
            and value.element_indices
        ):
            array_name = value.parent_array.name or "arr"
            idx_parts = []
            for idx_val in value.element_indices:
                idx = self._format_binop_operand(idx_val, param_values) or "?"
                idx_parts.append(idx)
            return f"{array_name}[{','.join(idx_parts)}]"
        # Named parameter
        if value.is_parameter():
            name = value.parameter_name() or value.name
            if name and not self._is_internal_temp_name(name):
                return name

        # Fallback: name (refuse IR-internal placeholders so that
        # "uint_tmp"/"float_tmp"/"bit_tmp" never leak into rendered
        # expressions).
        if (
            hasattr(value, "name")
            and value.name
            and not self._is_internal_temp_name(value.name)
        ):
            return value.name
        return None

    def _resolve_binop_as_symbolic(
        self,
        value: Value,
        param_values: dict,
        operations: list[Operation] | None = None,
    ) -> str | None:
        """Find the defining BinOp for a value and build a symbolic expression string."""
        # Search provided operations first (e.g., loop body)
        if operations is not None:
            for op in operations:
                if (
                    isinstance(op, BinOp)
                    and op.results
                    and id(op.results[0]) == id(value)
                ):
                    return self._build_symbolic_binop(op, param_values)
        # Fallback: search top-level graph
        graph = getattr(self, "graph", None)
        if graph is None:
            return None
        for op in graph.operations:
            if isinstance(op, BinOp) and op.results and id(op.results[0]) == id(value):
                return self._build_symbolic_binop(op, param_values)
        return None

    def _build_symbolic_binop(self, binop: BinOp, param_values: dict) -> str | None:
        """Build a symbolic string for a BinOp expression.

        Returns expressions like "2*gamma", "theta+1", "gamma" (when multiplied by 1).
        Returns None if either operand is unresolvable.
        """
        lhs_str = self._format_binop_operand(binop.lhs, param_values)
        rhs_str = self._format_binop_operand(binop.rhs, param_values)
        if lhs_str is None or rhs_str is None:
            return None

        # Simplify common cases
        if binop.kind == BinOpKind.MUL:
            if lhs_str == "1":
                return rhs_str
            if rhs_str == "1":
                return lhs_str
            if lhs_str == "0" or rhs_str == "0":
                return "0"
            # Prefer coefficient form: "2*gamma" rather than "gamma*2"
            return f"{lhs_str}*{rhs_str}"
        elif binop.kind == BinOpKind.ADD:
            if lhs_str == "0":
                return rhs_str
            if rhs_str == "0":
                return lhs_str
            return f"{lhs_str}+{rhs_str}"
        elif binop.kind == BinOpKind.SUB:
            if rhs_str == "0":
                return lhs_str
            return f"{lhs_str}-{rhs_str}"
        elif binop.kind == BinOpKind.DIV:
            if rhs_str == "1":
                return lhs_str
            return f"{lhs_str}/{rhs_str}"
        elif binop.kind == BinOpKind.MIN:
            return f"min({lhs_str},{rhs_str})"
        else:
            _extra_ops: dict[BinOpKind, str] = {
                BinOpKind.FLOORDIV: "//",
                BinOpKind.MOD: "%",
                BinOpKind.POW: "**",
            }
            op_sym = _extra_ops.get(binop.kind, "?") if binop.kind is not None else "?"
            return f"{lhs_str}{op_sym}{rhs_str}"

    def _evaluate_loop_body_intermediates(
        self,
        operations: list[Operation],
        param_values: dict,
    ) -> None:
        """Pre-evaluate intermediate scalar results in a loop body.

        Scans ``BinOp``, ``CompOp``, ``CondOp``, and ``NotOp`` operations and
        stores evaluated results by value ID in ``param_values``. This allows
        subsequent index and branch resolution to find concrete values.

        When a ``BinOp`` cannot be fully resolved numerically (e.g., one
        operand is a symbolic parameter), a symbolic string expression is
        stored instead (e.g., ``"2*gamma"`` or ``"theta+1"``).

        Args:
            operations: List of operations in the loop body.
            param_values: Mutable mapping updated in-place with resolved
                intermediate values keyed by logical_id.
        """
        for op in operations:
            if isinstance(op, (BinOp, CompOp, CondOp, NotOp)):
                self._record_classical_result(op, param_values, operations)

        # Also resolve array element access intermediates (e.g. i = edges[idx, 0])
        for op in operations:
            if not isinstance(
                op,
                (
                    GateOperation,
                    ControlledUOperation,
                    InvokeOperation,
                    InverseBlockOperation,
                ),
            ):
                continue
            operand_list = op.operands
            for operand in operand_list:
                if hasattr(operand, "element_indices") and operand.element_indices:
                    for idx_val in operand.element_indices:
                        if (
                            hasattr(idx_val, "parent_array")
                            and idx_val.parent_array is not None
                        ):
                            resolved = self._evaluate_value(
                                idx_val, param_values, operations
                            )
                            if resolved is not None:
                                param_values[idx_val.logical_id] = resolved

    def _resolve_array_element_lid(
        self,
        value: Value,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict | None = None,
    ) -> str:
        """Resolve a Value's logical_id to a canonical qubit_map key.

        For array element Values whose logical_id is a random UUID (created by
        Vector.__getitem__), construct the canonical key format
        ``f"{parent_lid}_[{idx}]"`` and register an alias if found.
        When param_values is provided, symbolic indices (e.g. loop variables)
        can be resolved via _evaluate_value.

        For *view* elements (``value.parent_array.slice_of is not None``)
        the ``parent_array`` is the view's own ``ArrayValue`` and is **not**
        registered in ``qubit_map`` — only the root register is.  In that
        case the chain is walked via :meth:`_resolve_view_chain_to_root`
        so the view-local ``element_indices[0]`` is composed with the
        chain's affine map to a root-space index, and the canonical key
        is built against the root register.  Without this, inlined
        sub-kernels called with ``view[i]`` would allocate a fresh wire
        instead of mapping to the root register's wire (mirrors the same
        composition done by :meth:`_resolve_parent_array_element`).
        """
        lid = logical_id_remap.get(value.logical_id, value.logical_id)
        # For symbolic array elements, skip cached UUID lookup —
        # the cached qubit_map entry may be stale from a different loop
        # iteration (e.g., map_block_results registers result.lid which
        # shares operand.lid via SSA).
        is_symbolic_array_element = (
            hasattr(value, "parent_array")
            and value.parent_array is not None
            and hasattr(value, "element_indices")
            and value.element_indices
            and not value.element_indices[0].is_constant()
        )
        if lid in qubit_map and not is_symbolic_array_element:
            return lid

        # Whole slice view (``ArrayValue`` with ``slice_of``, no
        # ``parent_array``): callers like ``map_block_results`` need a
        # canonical key that resolves to an existing root-space wire.
        # Without this branch the lid falls through and the caller
        # allocates a phantom wire for a value that actually aliases
        # root slots (visible as ghost wires past ``num_qubits`` in
        # rendered circuits).  Alias ``lid`` to the root's first
        # slice-covered element's wire and return the *array* lid
        # itself (NOT the element key) — callers (e.g. inline-block
        # remap consumers, ControlledU / callable-box paths) keep
        # using the returned string as a logical-id key for further
        # ``f"{lid}_[{i}]"`` element-key construction; returning an
        # element key here would yield malformed keys like
        # ``f"q.lid_[0]_[{i}]"``.
        if (
            isinstance(value, ArrayValue)
            and getattr(value, "slice_of", None) is not None
        ):
            resolved = self._resolve_view_chain_to_root(value, param_values)
            if resolved is not None:
                root_av, start, _step = resolved
                root_lid = logical_id_remap.get(root_av.logical_id, root_av.logical_id)
                root_elem_key = f"{root_lid}_[{start}]"
                if root_elem_key in qubit_map:
                    qubit_map[lid] = qubit_map[root_elem_key]
                    return lid
                if root_lid in qubit_map:
                    qubit_map[lid] = qubit_map[root_lid] + start
                    return lid

        if hasattr(value, "parent_array") and value.parent_array is not None:
            parent_array = value.parent_array
            # When the parent is a slice view, compose back to the root
            # register so the canonical key references the wire that
            # actually lives in ``qubit_map``.  The view itself was
            # never registered as a wire owner.
            chain_start = 0
            chain_step = 1
            chain_failed = False
            if getattr(parent_array, "slice_of", None) is not None:
                resolved = self._resolve_view_chain_to_root(parent_array, param_values)
                if resolved is None:
                    # Symbolic ``slice_start`` / ``slice_step`` along the
                    # chain — fall back to the original ``parent_lid``
                    # path; the caller's broader resolution may still
                    # find a match through a different alias.
                    chain_failed = True
                else:
                    root_av, chain_start, chain_step = resolved
                    parent_array = root_av

            if chain_failed:
                parent_lid = logical_id_remap.get(
                    value.parent_array.logical_id, value.parent_array.logical_id
                )
            else:
                parent_lid = logical_id_remap.get(
                    parent_array.logical_id, parent_array.logical_id
                )

            if hasattr(value, "element_indices") and value.element_indices:
                idx_value = value.element_indices[0]
                if idx_value.is_constant():
                    idx = idx_value.get_const()
                    if idx is not None:
                        root_idx = chain_start + chain_step * int(idx)
                        element_key = f"{parent_lid}_[{root_idx}]"
                        if element_key in qubit_map:
                            qubit_map[lid] = qubit_map[element_key]
                            return element_key
                elif param_values:
                    idx = self._evaluate_value(idx_value, param_values)
                    if idx is not None:
                        root_idx = chain_start + chain_step * int(idx)
                        element_key = f"{parent_lid}_[{root_idx}]"
                        if element_key in qubit_map:
                            # Don't cache — symbolic index may resolve
                            # differently across loop iterations
                            return element_key
        return lid

    def _resolve_array_size(
        self,
        array_value: Value,
        resolved_lid: str,
        qubit_map: dict[str, int],
        param_values: dict,
    ) -> int | None:
        """Resolve the length of an `ArrayValue`.

        Tries `shape[0].get_const()`, then `_evaluate_value` against
        `param_values`, and finally a scan of `qubit_map` keys
        matching `"{resolved_lid}_[{i}]"`.

        Args:
            array_value: The array whose length to resolve. Expected
                to expose a `shape` attribute.
            resolved_lid: Logical id used to index `qubit_map` for the
                element-key scan fallback.
            qubit_map: Mapping from logical_id to qubit wire index.
                Used only for the scan fallback.
            param_values: Parameter values for evaluating a symbolic
                shape.

        Returns:
            The array length as an int, or None if no strategy
            succeeds.
        """
        if hasattr(array_value, "shape") and array_value.shape:
            size_value = array_value.shape[0]
            if size_value.is_constant():
                c = size_value.get_const()
                if isinstance(c, int):
                    return c
            else:
                ev = self._evaluate_value(size_value, param_values)
                if isinstance(ev, (int, float)):
                    return int(ev)
        count = 0
        while f"{resolved_lid}_[{count}]" in qubit_map:
            count += 1
        if count > 0:
            return count
        return None

    def _resolve_qfixed_carrier_indices(
        self,
        operand: Value,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
    ) -> list[int]:
        """Resolve a QFixed measurement operand to its carrier wire indices.

        The operand is the QFixed ``Value`` produced by ``CastOperation``;
        its ``metadata.cast.qubit_logical_ids`` carries one entry per
        carrier qubit in measurement order, formatted as
        ``f"{root_logical_id}_{idx}"``.  ``QInitOperation`` registers
        each root element in ``qubit_map`` as ``f"{root_logical_id}_[{idx}]"``.
        This helper bridges the two encodings: it parses each cast
        carrier string, applies ``logical_id_remap`` to the root prefix
        (in case the cast lives in an inlined block), reformats with
        brackets, and looks up the result in ``qubit_map``.

        Args:
            operand (Value): The QFixed-typed operand of a
                ``MeasureQFixedOperation`` (i.e. the result of a prior
                ``CastOperation``).
            qubit_map (dict[str, int]): Mapping from logical_id-derived
                keys to qubit wire indices.
            logical_id_remap (dict[str, str]): Mapping from formal-
                parameter logical_ids to actual-argument logical_ids,
                used when the cast occurs inside an inlined block.

        Returns:
            list[int]: The wire indices the QFixed measurement targets,
                in carrier (most-significant-bit-first) order.  Empty
                when the operand carries no cast metadata or when no
                carrier entry resolves to a known wire.
        """
        cast_meta = getattr(operand.metadata, "cast", None)
        if cast_meta is None or not cast_meta.qubit_logical_ids:
            return []
        indices: list[int] = []
        for carrier_key in cast_meta.qubit_logical_ids:
            # Carrier key format: ``f"{root_logical_id}_{idx}"`` (no
            # brackets).  Split on the *last* ``_`` to recover the root
            # prefix and idx; the root logical_id itself may contain
            # underscores (UUIDs use them), so split from the right.
            try:
                root_lid, idx_str = carrier_key.rsplit("_", 1)
                idx_int = int(idx_str)
            except (ValueError, AttributeError):
                continue
            remapped_lid = logical_id_remap.get(root_lid, root_lid)
            bracket_key = f"{remapped_lid}_[{idx_int}]"
            wire = qubit_map.get(bracket_key)
            if wire is not None:
                indices.append(wire)
        return indices

    def _resolve_view_chain_to_root(
        self,
        array_value: "ArrayValue",
        param_values: dict | None = None,
    ) -> tuple["ArrayValue", int, int] | None:
        """Walk the ``slice_of`` chain to the root array, composing the affine map.

        For a sliced ``ArrayValue`` (i.e. one with ``slice_of is not None``),
        the element-index in operand metadata is *view-local*: ``view[i]``
        on ``q[0::2]`` records ``element_indices[0] = i`` against
        ``parent_array = view_av``, not against ``q``.  Translating to a
        physical wire index requires walking the ``slice_of`` chain to
        the root and composing the start / step pairs of every hop.

        At each hop ``cur = slice_of(child)`` the parent-space index of
        ``child[i]`` is ``cur.slice_start + cur.slice_step * i``.
        Composing through a chain accumulates as
        ``(start, step) -> (cur.slice_start + cur.slice_step * start,
        cur.slice_step * step)`` until ``cur.slice_of is None`` (root).

        When ``param_values`` is supplied, a non-constant ``slice_start``
        or ``slice_step`` is still resolvable as long as
        :meth:`_evaluate_value` can fold it (e.g. the frontend's
        ``_uint_min(0, stop)`` clamp returns a symbolic ``BinOp`` even
        when the user wrote ``q[0:stop]``; under a loop unrolling where
        ``stop`` is bound to a concrete integer, the BinOp folds to ``0``
        and the chain walk succeeds).

        Args:
            array_value (ArrayValue): A possibly-sliced ArrayValue.  When
                ``slice_of is None`` the function returns the identity
                transform ``(array_value, 0, 1)``.
            param_values (dict | None): Optional mapping consulted by
                :meth:`_evaluate_value` when ``slice_start`` /
                ``slice_step`` is not directly constant.  Pass the
                per-iteration ``child_param_values`` from
                :meth:`_build_vfor` (or any caller-side
                ``param_values``) to enable this fallback.  Defaults to
                ``None``, which preserves the strict ``is_constant()``
                behaviour for callers that do not have a parameter
                context.

        Returns:
            tuple[ArrayValue, int, int] | None: ``(root_av, start, step)``
                so that the root-space index of a view-local ``idx`` is
                ``start + step * idx``.  ``None`` when any
                ``slice_start`` / ``slice_step`` along the chain is
                symbolic (cannot compose the affine map at draw time).
        """
        cur = array_value
        start = 0
        step = 1
        while getattr(cur, "slice_of", None) is not None:
            slice_start = cur.slice_start
            slice_step = cur.slice_step
            if slice_start is None or slice_step is None:
                return None
            s_int = self._resolve_slice_bound(slice_start, param_values)
            st_int = self._resolve_slice_bound(slice_step, param_values)
            if s_int is None or st_int is None:
                return None
            start = s_int + st_int * start
            step = st_int * step
            cur = cur.slice_of
            assert cur is not None, (
                "[FOR DEVELOPER] slice_of must point at a parent ArrayValue while "
                "walking a slice chain."
            )
        return cur, start, step

    def _resolve_slice_bound(
        self,
        bound: Value,
        param_values: dict | None,
    ) -> int | None:
        """Resolve a ``slice_start`` / ``slice_step`` ``Value`` to ``int``.

        Tries ``is_constant()`` first (the historical path), and falls
        back to :meth:`_evaluate_value` against ``param_values`` so
        clamps emitted by the frontend's ``_uint_min`` helper still
        resolve under a loop unrolling that fixes the symbolic operands.

        Args:
            bound (Value): The slice bound Value to resolve.
            param_values (dict | None): Parameter values for the
                fallback ``_evaluate_value`` lookup; ``None`` disables
                the fallback and keeps the legacy ``is_constant()``-only
                behaviour.

        Returns:
            int | None: The resolved integer, or ``None`` when neither
                path produces a usable value.
        """
        if bound.is_constant():
            c = bound.get_const()
            if c is None:
                return None
            try:
                return int(c)
            except (TypeError, ValueError):
                return None
        if param_values is None:
            return None
        ev = self._evaluate_value(bound, param_values)
        if ev is None:
            return None
        try:
            return int(ev)
        except (TypeError, ValueError):
            return None

    def _resolve_parent_array_element(
        self,
        operand: Value,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict,
    ) -> int | None:
        """Resolve a `q[i]` operand to its wire index when `i` is concrete.

        Handles both direct ``q[i]`` and ``view[i]`` where ``view`` is a
        slice of ``q``: when ``parent_array.slice_of`` is non-None, the
        chain is walked via :meth:`_resolve_view_chain_to_root` so that
        view-local ``i`` is composed with the chain's affine map to a
        root-space index.

        Args:
            operand: IR Value expected to have `parent_array` and
                `element_indices`. Operands without a `parent_array`
                return None.
            qubit_map: Mapping from logical_id to qubit wire index.
            logical_id_remap: Remap from dummy logical_ids to actual
                logical_ids.
            param_values: Parameter values for evaluating a symbolic
                `element_indices[0]`.

        Returns:
            The wire index for the resolved element.  When the parent
            is a slice view, the view-local index is composed with the
            ``slice_of`` chain back to the root register.  ``None`` when
            the parent / root is not in ``qubit_map``,
            ``element_indices`` is absent, or any index / slice
            metadata in the chain is symbolic and cannot be evaluated.
        """
        if not (hasattr(operand, "parent_array") and operand.parent_array is not None):
            return None
        parent_array = operand.parent_array
        if not (hasattr(operand, "element_indices") and operand.element_indices):
            return None
        idx_value = operand.element_indices[0]
        if idx_value.is_constant():
            idx = idx_value.get_const()
        else:
            idx = self._evaluate_value(idx_value, param_values)
        if idx is None:
            return None
        idx_int = int(idx)
        # Slice view → walk to root, compose affine map.  Mirror the
        # direct-element branch's ``element_key`` preference: when the
        # root has per-element entries (an ordinary register, or a
        # callee parameter slice-aliased to non-contiguous root wires
        # by ``build_qubit_map``'s sub-kernel handling), the formula
        # ``base + start + step * idx`` is only correct for contiguous
        # wires.  Look up the canonical ``f"{root_lid}_[{root_idx}]"``
        # key first and fall back to the formula only when no
        # per-element entry exists.
        if getattr(parent_array, "slice_of", None) is not None:
            resolved = self._resolve_view_chain_to_root(parent_array, param_values)
            if resolved is None:
                return None
            root_av, start, step = resolved
            root_lid = logical_id_remap.get(root_av.logical_id, root_av.logical_id)
            root_idx = start + step * idx_int
            root_elem_key = f"{root_lid}_[{root_idx}]"
            if root_elem_key in qubit_map:
                return qubit_map[root_elem_key]
            if root_lid not in qubit_map:
                return None
            return qubit_map[root_lid] + root_idx
        # Direct array element.  Prefer the canonical per-element key
        # (``f"{parent_lid}_[{idx}]"``) which is populated both by
        # ``QInitOperation`` for ordinary registers and by the slice-
        # view sub-kernel-argument handling in ``build_qubit_map``.
        # The formula fallback (``base + idx``) only fires for arrays
        # without per-element entries, where wires are contiguous by
        # construction.
        parent_lid = logical_id_remap.get(
            parent_array.logical_id, parent_array.logical_id
        )
        element_key = f"{parent_lid}_[{idx_int}]"
        if element_key in qubit_map:
            return qubit_map[element_key]
        if parent_lid not in qubit_map:
            return None
        return qubit_map[parent_lid] + idx_int

    def _resolve_non_element_operand(
        self,
        operand: Value,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict,
    ) -> list[int] | None:
        """Resolve a non-element operand to qubit wire indices.

        Dispatches the three non-`parent_array` operand shapes: a
        whole `ArrayValue` with `shape`, a synthetic `ArrayValue`
        from `expval()` tuple input, and a direct `logical_id`
        lookup.

        Args:
            operand: IR Value that is not a `parent_array` element
                access.
            qubit_map: Mapping from logical_id to qubit wire index.
            logical_id_remap: Remap from dummy logical_ids to actual
                logical_ids.
            param_values: Parameter values for evaluating symbolic
                shapes.

        Returns:
            None if the operand is unresolvable (logical_id not in
            `qubit_map`, or shape symbolic and unevaluable). An
            empty list if the operand is resolved but touches zero
            qubits (e.g. concrete `shape[0] == 0`). Otherwise, a
            list of wire indices.
        """
        resolved_lid = logical_id_remap.get(operand.logical_id, operand.logical_id)

        # Slice view operand → walk the ``slice_of`` chain to the root
        # register and translate each element to a root-space wire via
        # ``start + step * i``.  Without this, the broadcast gate /
        # call-block / measure paths would treat a slice view as a
        # contiguous fresh-wire block (the formula ``base + k``), which
        # is correct only for an ordinary register.
        if (
            isinstance(operand, ArrayValue)
            and getattr(operand, "slice_of", None) is not None
        ):
            chain = self._resolve_view_chain_to_root(operand, param_values)
            if chain is not None:
                root_av, start, step = chain
                root_lid = logical_id_remap.get(root_av.logical_id, root_av.logical_id)
                size = self._resolve_array_size(
                    operand, resolved_lid, qubit_map, param_values
                )
                if size is not None:
                    # Loop-with-``else``: the ``else`` block runs only
                    # when the loop completes without a break.  A
                    # successful resolution (including an empty slice
                    # where ``size == 0``) returns the accumulated
                    # ``wires`` deterministically — matching the
                    # function contract that documents ``[]`` as the
                    # "zero qubits" outcome.  A missing-key break
                    # falls through to the other resolution branches.
                    wires: list[int] = []
                    for i in range(size):
                        root_idx = start + step * i
                        root_elem_key = f"{root_lid}_[{root_idx}]"
                        if root_elem_key in qubit_map:
                            wires.append(qubit_map[root_elem_key])
                        elif root_lid in qubit_map:
                            wires.append(qubit_map[root_lid] + root_idx)
                        else:
                            break
                    else:
                        return wires

        if (
            isinstance(operand, ArrayValue)
            and hasattr(operand, "shape")
            and operand.shape
            and resolved_lid in qubit_map
        ):
            size = self._resolve_array_size(
                operand, resolved_lid, qubit_map, param_values
            )
            if size is not None:
                # Prefer per-element keys (``f"{resolved_lid}_[{k}]"``)
                # when present.  ``QInitOperation`` populates them for
                # ordinary registers and ``build_qubit_map``'s slice-
                # view sub-kernel-argument handling populates them
                # with non-contiguous wires when a slice view is
                # passed as a helper qkernel argument.  The ``base +
                # k`` formula is only correct for genuinely contiguous
                # arrays where every wire is consecutive, so use the
                # element-key lookup first and fall back to the
                # formula only when no per-element entry exists.
                element_keyed_wires: list[int] = []
                for k in range(size):
                    elem_key = f"{resolved_lid}_[{k}]"
                    if elem_key in qubit_map:
                        element_keyed_wires.append(qubit_map[elem_key])
                    else:
                        element_keyed_wires = []
                        break
                if element_keyed_wires:
                    return element_keyed_wires
                base_idx = qubit_map[resolved_lid]
                return [base_idx + k for k in range(size)]
            return None

        if isinstance(operand, ArrayValue) and operand.get_element_uuids():
            return [
                qubit_map[uuid]
                for uuid in operand.get_element_uuids()
                if uuid in qubit_map
            ]

        if resolved_lid in qubit_map:
            return [qubit_map[resolved_lid]]

        return None

    def _resolve_operand_to_qubit_indices(
        self,
        operand: Value,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str] | None = None,
        param_values: dict | None = None,
    ) -> list[int] | None:
        """Resolve an operand to qubit wire indices for single-gate placement.

        Entry point for gate-building callers. When `operand` is
        `q[i]` with a symbolic `i` that cannot be evaluated, the
        result collapses to the parent array's base wire index —
        a single-qubit approximation that keeps the gate on one wire
        visually. Loop-affect analysis that needs to expand that
        case to the full array should call
        `_resolve_operand_to_affected_qubits` instead.

        Args:
            operand: IR Value representing a qubit or qubit-array
                operand.
            qubit_map: Mapping from logical_id to qubit wire index.
            logical_id_remap: Optional remap from dummy logical_ids
                (used during callable inlining) to actual
                logical_ids. Defaults to None (empty remap).
            param_values: Optional parameter values for evaluating
                computed indices and shapes. Defaults to None.

        Returns:
            None if the operand is unresolvable. An empty list if
            the operand resolves to zero qubits. Otherwise, a list
            of wire indices.
        """
        if logical_id_remap is None:
            logical_id_remap = {}
        if param_values is None:
            param_values = {}

        idx = self._resolve_parent_array_element(
            operand, qubit_map, logical_id_remap, param_values
        )
        if idx is not None:
            return [idx]

        if hasattr(operand, "parent_array") and operand.parent_array is not None:
            parent_lid = logical_id_remap.get(
                operand.parent_array.logical_id, operand.parent_array.logical_id
            )
            if parent_lid in qubit_map:
                return [qubit_map[parent_lid]]

        return self._resolve_non_element_operand(
            operand, qubit_map, logical_id_remap, param_values
        )

    def _resolve_operand_to_affected_qubits(
        self,
        operand: Value,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str] | None = None,
        param_values: dict | None = None,
    ) -> list[int] | None:
        """Resolve an operand to all wires it may touch across iterations.

        Entry point for loop-affect analysis. When `operand` is
        `q[i]` with a symbolic `i` that cannot be evaluated, the
        result expands to every element of the parent array — the
        correct semantics for "across all iterations this loop
        touches every element of `q`". If the parent array's size
        cannot be resolved either, returns None so callers can
        propagate `all_resolved = False`.

        Args:
            operand: IR Value representing a qubit or qubit-array
                operand.
            qubit_map: Mapping from logical_id to qubit wire index.
            logical_id_remap: Optional remap from dummy logical_ids
                to actual logical_ids. Defaults to None.
            param_values: Optional parameter values for evaluating
                computed indices and shapes. Defaults to None.

        Returns:
            None if the operand is unresolvable. An empty list if
            the operand resolves to zero qubits. Otherwise, a list
            of wire indices.
        """
        if logical_id_remap is None:
            logical_id_remap = {}
        if param_values is None:
            param_values = {}

        idx = self._resolve_parent_array_element(
            operand, qubit_map, logical_id_remap, param_values
        )
        if idx is not None:
            return [idx]

        if hasattr(operand, "parent_array") and operand.parent_array is not None:
            parent_lid = logical_id_remap.get(
                operand.parent_array.logical_id, operand.parent_array.logical_id
            )
            if parent_lid in qubit_map:
                base_idx = qubit_map[parent_lid]
                size = self._resolve_array_size(
                    operand.parent_array, parent_lid, qubit_map, param_values
                )
                if size is not None:
                    return [base_idx + k for k in range(size)]
                return None

        return self._resolve_non_element_operand(
            operand, qubit_map, logical_id_remap, param_values
        )

    def _evaluate_loop_range(
        self, op: ForOperation, param_values: dict
    ) -> tuple[int, int | None, int]:
        """Evaluate ForOperation start/stop/step to concrete values.

        Args:
            op: ForOperation whose range to evaluate.
            param_values: Parameter values for resolving symbolic operands.

        Returns:
            (start_val, stop_val_or_None, step_val) where stop_val is None
            if unresolvable (symbolic).

        Raises:
            ValueError: If step evaluates to zero.
        """
        start_val = (
            self._evaluate_value(op.operands[0], param_values) if op.operands else None
        )
        stop_val = (
            self._evaluate_value(op.operands[1], param_values)
            if len(op.operands) > 1
            else None
        )
        step_val = (
            self._evaluate_value(op.operands[2], param_values)
            if len(op.operands) > 2
            else None
        )

        if start_val is None:
            start_val = 0
        if step_val is None:
            step_val = 1
        if step_val == 0:
            raise ValueError("ForOperation step must not be zero")

        return (
            int(start_val),
            stop_val if stop_val is None else int(stop_val),
            int(step_val),
        )

    def _compute_loop_iterations(
        self, start_val: int, stop_val: int, step_val: int
    ) -> int:
        """Compute number of loop iterations from resolved range values.

        Args:
            start_val: Loop start value.
            stop_val: Loop stop value (must be concrete).
            step_val: Loop step value (non-zero).

        Returns:
            Number of iterations.
        """
        if step_val > 0:
            return (stop_val - start_val + step_val - 1) // step_val
        return (start_val - stop_val - step_val - 1) // (-step_val)

    def _align_actuals_to_formals(
        self,
        formals: Sequence[ValueBase],
        *,
        quantum_actuals: Sequence[ValueBase],
        classical_actuals: Sequence[ValueBase],
    ) -> list[ValueBase]:
        """Reorder split actual operands to match formal signature order.

        :meth:`_build_block_value_mappings` zips a callee block's
        ``input_values`` against the caller's ``actual_inputs``
        positionally three times.  Callers that already have their
        actuals split into quantum and classical pools (typically
        ``ControlledUOperation.target_operands`` partitioned by
        ``QubitType`` or composite/inverse target qubits plus
        ``parameters``) must therefore weave them back into the
        formals' declared order before handing them off, otherwise
        a sub-signature like ``def sub(theta, q: Vector[Qubit])``
        ends up with the ``theta`` formal paired with the ``q``
        actual (and vice versa) and the inner block's symbolic
        ``q.shape[0]`` is never bound to the caller's actual
        ``Vector`` shape.

        Args:
            formals (Sequence[ValueBase]): The callee block's formal
                ``input_values``, in signature-declared order.
            quantum_actuals (Sequence[ValueBase]): Quantum actuals
                in their original order (e.g. quantum-typed entries
                of ``op.target_operands``).
            classical_actuals (Sequence[ValueBase]): Classical
                actuals in their original order (e.g. classical-
                typed entries of ``op.target_operands``, or
                ``op.parameters`` for composites).

        Returns:
            list[ValueBase]: One actual per formal, in the formals'
                declared order.  Surplus actuals in either pool are
                intentionally dropped (never appended after the
                formal-aligned region) -- appending them would
                silently mis-pair them with later formals downstream.
                A shortfall (running out of actuals in either pool
                before consuming every formal) returns a list
                shorter than ``formals``; the caller's downstream
                ``zip`` then truncates, matching the existing
                behaviour for an unaligned actual list that was
                already too short.
        """
        return align_formal_operands(
            formals,
            quantum_actuals,
            classical_actuals,
        )

    def _build_block_value_mappings(
        self,
        block_value: Block,
        actual_inputs: Sequence[ValueBase],
        logical_id_remap: dict[str, str],
        param_values: dict[str, object],
        qubit_map: dict[str, int] | None = None,
    ) -> tuple[dict[str, str], dict[str, object]]:
        """Build value remaps and forwarded bindings for a nested block.

        Maps a callee block's formal inputs to the caller's actual
        operands so inline drawing can resolve qubit wires, scalar
        parameters, bound array data, and bound dict data recursively.

        Args:
            block_value (Block): Callee block whose input mappings to build.
            actual_inputs (Sequence[ValueBase]): Actual operands passed to
                the callee block.
            logical_id_remap (dict[str, str]): Current logical-id remapping.
                Copied before child-specific entries are added.
            param_values (dict[str, object]): Current parameter and bound-data
                values. Copied before child-specific entries are added.
            qubit_map (dict[str, int] | None): Qubit wire map for resolving
                array element indices. Defaults to None when wire resolution
                is not needed.

        Returns:
            tuple[dict[str, str], dict[str, object]]: Child logical-id remap
                and child parameter values for recursive inline processing.
        """
        new_logical_id_remap = dict(logical_id_remap)

        for dummy_input, actual_input in zip(block_value.input_values, actual_inputs):
            # For Qubit array elements, resolve through _resolve_array_element_lid
            # to get the canonical parent_[idx] key instead of the raw UUID
            if (
                qubit_map is not None
                and isinstance(actual_input, Value)
                and hasattr(actual_input, "parent_array")
                and actual_input.parent_array is not None
                and isinstance(actual_input.type, QubitType)
            ):
                resolved_lid = self._resolve_array_element_lid(
                    actual_input, qubit_map, logical_id_remap, param_values
                )
                new_logical_id_remap[dummy_input.logical_id] = resolved_lid
            else:
                actual_lid = logical_id_remap.get(
                    actual_input.logical_id, actual_input.logical_id
                )
                new_logical_id_remap[dummy_input.logical_id] = actual_lid
                if (
                    qubit_map is not None
                    and isinstance(dummy_input, ArrayValue)
                    and isinstance(actual_input, ArrayValue)
                    and isinstance(dummy_input.type, QubitType)
                    and isinstance(actual_input.type, QubitType)
                ):
                    wires = self._resolve_operand_to_qubit_indices(
                        actual_input,
                        qubit_map,
                        logical_id_remap,
                        param_values,
                    )
                    if wires:
                        qubit_map.setdefault(actual_lid, wires[0])
                        for idx, wire in enumerate(wires):
                            qubit_map[f"{actual_lid}_[{idx}]"] = wire
                            qubit_map[f"{dummy_input.logical_id}_[{idx}]"] = wire

        child_param_values = dict(param_values)
        for dummy_input, actual_input in zip(block_value.input_values, actual_inputs):
            # The IR guarantees that operands implement ValueBase, so this assertion should always pass.
            # If it fails, there is a bug in the IR construction.
            assertion_message = (
                "[FOR DEVELOPER] Operation.operands elements must be ValueBase instances. "
                "If this assertion fails, there is a bug in the IR construction."
            )
            assert isinstance(actual_input, ValueBase), assertion_message
            actual_lid = logical_id_remap.get(
                actual_input.logical_id, actual_input.logical_id
            )
            const = actual_input.get_const()
            if const is not None:
                child_param_values[dummy_input.logical_id] = const
            elif actual_input.logical_id in param_values or actual_lid in param_values:
                pv = param_values.get(
                    actual_input.logical_id, param_values.get(actual_lid)
                )
                if isinstance(pv, (int, float)):
                    # Numeric value: store directly
                    child_param_values[dummy_input.logical_id] = pv
                elif (
                    isinstance(actual_input, Value)
                    and hasattr(actual_input, "parent_array")
                    and actual_input.parent_array is not None
                    and not isinstance(actual_input.type, QubitType)
                ):
                    # Non-qubit array element with non-numeric value:
                    # try resolving the array index before storing the raw string
                    evaluated = self._evaluate_value(actual_input, param_values)
                    if evaluated is not None and isinstance(evaluated, (int, float)):
                        child_param_values[dummy_input.logical_id] = evaluated
                    else:
                        resolved = self._resolve_symbolic_array_name(
                            actual_input, param_values
                        )
                        child_param_values[dummy_input.logical_id] = (
                            resolved if resolved is not None else pv
                        )
                else:
                    child_param_values[dummy_input.logical_id] = pv
            elif (
                isinstance(actual_input, Value)
                and hasattr(actual_input, "parent_array")
                and actual_input.parent_array is not None
                and not isinstance(actual_input.type, QubitType)
            ):
                # Non-qubit array element not in param_values:
                # try evaluating or resolving symbolic name
                evaluated = self._evaluate_value(actual_input, param_values)
                if evaluated is not None and isinstance(evaluated, (int, float)):
                    child_param_values[dummy_input.logical_id] = evaluated
                else:
                    resolved = self._resolve_symbolic_array_name(
                        actual_input, param_values
                    )
                    if resolved is not None:
                        child_param_values[dummy_input.logical_id] = resolved
                    elif actual_input.is_parameter():
                        child_param_values[dummy_input.logical_id] = (
                            actual_input.parameter_name() or actual_input.name
                        )
            elif actual_input.is_parameter():
                child_param_values[dummy_input.logical_id] = (
                    actual_input.parameter_name() or actual_input.name
                )
            elif isinstance(actual_input, Value):
                # actual_input is a BinOp result (or similar unresolved
                # non-parameter Value). Try numeric evaluation via
                # graph.operations first — after top-level
                # pre-evaluation in build_visual_ir this usually
                # succeeds.
                evaluated = self._evaluate_value(actual_input, param_values)
                if evaluated is not None and isinstance(evaluated, (int, float)):
                    child_param_values[dummy_input.logical_id] = evaluated
                else:
                    # Fall back to a symbolic expression so downstream
                    # renders e.g. "reps*2*n" instead of "uint_tmp".
                    # `_format_value_as_expression` is called with empty
                    # `loop_vars` and `operations=None` here, so it may
                    # legitimately return its unresolved sentinel "?"
                    # for values that a richer downstream context could
                    # still resolve. Treat "?" the same as a temp-name
                    # miss and forward the logical_id so the downstream
                    # renderer gets a second chance with its own context.
                    expr = self._format_value_as_expression(actual_input, set(), None)
                    if expr and expr != "?" and not self._is_internal_temp_name(expr):
                        child_param_values[dummy_input.logical_id] = expr
                    else:
                        # Last resort: forward the logical_id so
                        # downstream lookups can still chase the
                        # defining BinOp via logical_id_remap.
                        new_logical_id_remap[dummy_input.logical_id] = (
                            actual_input.logical_id
                        )

        # Propagate ArrayValue shape dimensions and const_array data
        for dummy_input, actual_input in zip(block_value.input_values, actual_inputs):
            if isinstance(actual_input, ArrayValue) and isinstance(
                dummy_input, ArrayValue
            ):
                if dummy_input.shape and actual_input.shape:
                    for dummy_dim, actual_dim in zip(
                        dummy_input.shape, actual_input.shape
                    ):
                        const = actual_dim.get_const()
                        if const is not None:
                            child_param_values[dummy_dim.logical_id] = const
                        elif actual_dim.logical_id in param_values:
                            child_param_values[dummy_dim.logical_id] = param_values[
                                actual_dim.logical_id
                            ]
                const_array = actual_input.get_const_array()
                if const_array is None:
                    actual_lid = logical_id_remap.get(
                        actual_input.logical_id, actual_input.logical_id
                    )
                    const_array = param_values.get(
                        f"_array_data_{actual_input.logical_id}",
                        param_values.get(f"_array_data_{actual_lid}"),
                    )
                if const_array is not None:
                    child_param_values[f"_array_data_{dummy_input.logical_id}"] = (
                        const_array
                    )
            elif isinstance(actual_input, DictValue) and hasattr(
                dummy_input, "logical_id"
            ):
                bound_data = None
                if actual_input.metadata.dict_runtime is not None:
                    bound_data = list(actual_input.get_bound_data_items())
                else:
                    actual_lid = logical_id_remap.get(
                        actual_input.logical_id, actual_input.logical_id
                    )
                    bound_data = param_values.get(
                        f"_dict_data_{actual_input.logical_id}",
                        param_values.get(f"_dict_data_{actual_lid}"),
                    )
                if bound_data is not None:
                    child_param_values[f"_dict_data_{dummy_input.logical_id}"] = (
                        bound_data
                    )

        return new_logical_id_remap, child_param_values

    def _estimate_gate_width(
        self, op: GateOperation, param_values: dict | None = None
    ) -> float:
        """Estimate gate box width from label text without matplotlib axes.

        Uses character-based estimation suitable for layout phase
        (before figure creation). Also serves as floor for drawing phase.

        Args:
            op: GateOperation whose width to estimate.
            param_values: Parameter values for resolving gate labels.

        Returns:
            Estimated gate width in data coordinate units.
        """
        label, has_param = self._get_gate_label(op, param_values)
        if not has_param:
            return self.style.gate_width

        # Strip TeX $ delimiters (not rendered)
        visual = label.replace("$", "")
        # TeX commands like \theta render as ~1 wide character
        visual = re.sub(r"\\[a-zA-Z]+", "X", visual)
        effective_len = len(visual)
        # Use style settings for width calculation
        char_width = self.style.char_width_gate  # 0.14 default, configurable
        text_width = effective_len * char_width
        padding = self.style.text_padding  # match drawing phase
        return max(self.style.gate_width, text_width + 2 * padding)

    def _estimate_label_box_width(self, label: str) -> float:
        """Estimate box width for a text label (blocks, composite gates, controlled-U).

        Strips TeX formatting for visual length estimation.

        Args:
            label: Text label to estimate width for.

        Returns:
            Estimated box width in data coordinate units.
        """
        visual_label = label.replace("$", "")
        visual_label = re.sub(r"\\[a-zA-Z]+", "X", visual_label)
        text_width = len(visual_label) * self.style.char_width_gate
        return max(text_width + 2 * self.style.text_padding, self.style.gate_width)

    def _estimate_block_label_box_width(self, label: str) -> float:
        """Estimate box width for a callable-block label.

        Uses ``char_width_block`` (wider than ``char_width_gate``) because
        block labels often contain longer text with parenthesized parameters
        (e.g., ``mixer(omegas[0])``).

        Args:
            label: Text label to estimate width for.

        Returns:
            Estimated box width in data coordinate units.
        """
        visual_label = label.replace("$", "")
        visual_label = re.sub(r"\\[a-zA-Z]+", "X", visual_label)
        text_width = len(visual_label) * self.style.char_width_block
        return max(text_width + 2 * self.style.text_padding, self.style.gate_width)

    @staticmethod
    def _get_child_op_lists(op: Operation) -> list[list[Operation]]:
        """Return child operation lists for control-flow operations."""
        if isinstance(op, (ForOperation, WhileOperation, ForItemsOperation)):
            return [op.operations]
        if isinstance(op, IfOperation):
            return [op.true_operations, op.false_operations]
        return []

    def _max_block_gate_width(
        self, operations: list[Operation], param_values: dict | None = None
    ) -> float:
        """Find the maximum estimated gate width among GateOperations in a block.

        Args:
            operations: List of operations to scan.
            param_values: Parameter values for gate width estimation.

        Returns:
            Maximum gate width, at least style.gate_width.
        """
        max_width = self.style.gate_width
        for op in operations:
            if isinstance(op, GateOperation):
                max_width = max(max_width, self._estimate_gate_width(op, param_values))
            else:
                for child_ops in self._get_child_op_lists(op):
                    max_width = max(
                        max_width,
                        self._max_block_gate_width(child_ops, param_values),
                    )
        return max_width

    # ------------------------------------------------------------------
    # Measure Phase: compute widths without assigning coordinates
    # ------------------------------------------------------------------

    @staticmethod
    def _materialize_dict_entries(
        dict_value: ValueBase,
        param_values: dict | None = None,
        logical_id_remap: dict[str, str] | None = None,
    ) -> Sequence[tuple] | None:
        """Extract entries from a DictValue from either IR or runtime metadata.

        A ``DictValue`` carries entries in one of two complementary
        encodings depending on how it was constructed:

        * **IR-level**: ``DictValue.entries`` is a tuple of
          ``(TupleValue | Value, Value)`` pairs — IR Values, not
          plain Python objects. Used when the dict was built inside a
          kernel (e.g. literal ``{i: x[i]}`` against symbolic
          parameters).
        * **Runtime metadata**: ``DictValue.metadata.dict_runtime.bound_data``
          holds a tuple of ``(raw_key, raw_value)`` Python pairs. Used
          when the dict was bound at ``transpile`` / ``draw`` time via
          ``bindings={"d": {0: 1.0, ...}}``.

        The runtime-metadata branch distinguishes "bound but empty" by
        checking ``metadata.dict_runtime is not None`` directly (so a
        dict bound to ``{}`` returns ``[]`` and renders as ``VSkip``).
        The IR-entries branch only fires when ``DictValue.entries`` is
        truthy: an empty IR ``entries`` tuple is treated as the
        "unbound parameter placeholder" case (every kernel-parameter
        ``Dict`` is constructed with ``entries=()``) and falls through
        to ``None``, which the caller renders as a folded box. There
        is currently no way for a kernel to declare a literal empty
        ``DictValue`` whose contents are knowable to be empty without
        also setting ``dict_runtime`` — if that ever becomes possible
        the IR-entries branch needs the same "is the entries field
        explicitly empty?" check the runtime branch already does.

        Callers (`_build_vfor_items`, `build_qubit_map`'s ForItems
        branch) handle the union shape by sniffing each pair —
        ``hasattr(entry_key, "elements")`` for ``TupleValue``,
        ``hasattr(entry_key, "get_const")`` for scalar IR Values,
        otherwise treating the pair as raw Python.

        Args:
            dict_value (ValueBase): The DictValue (or compatible Value)
                whose entries should be materialized. Should carry
                IR-level ``entries`` or runtime ``dict_runtime``
                metadata; otherwise treated as truly unbound.
            param_values (dict | None): Optional nested visualization
                context. Used to forward bound dict data through
                inlined helper calls whose formal parameter does not
                itself carry runtime metadata.
            logical_id_remap (dict[str, str] | None): Optional remap
                from formal logical IDs to actual logical IDs. Used
                with ``param_values`` when resolving forwarded bound
                dict data.

        Returns:
            Sequence[tuple] | None: A sequence of ``(key, value)``
                pairs whose element types depend on the encoding above
                (IR Values for the IR-level branch, raw Python objects
                for the runtime branch). Returns ``[]`` for a dict
                that is bound to an empty mapping so callers can
                render zero iterations as a ``VSkip`` rather than a
                folded box. Returns ``None`` only when the dict is
                truly unbound — no IR-level entries AND no runtime
                metadata.
        """
        # IR-level entries: a tuple of (TupleValue | Value, Value)
        # pairs. Returned as-is so the caller can resolve each operand
        # against `param_values` and pull symbolic / constant info as
        # appropriate.
        if hasattr(dict_value, "entries") and dict_value.entries:
            return dict_value.entries

        # Runtime metadata: a tuple of (raw_key, raw_value) Python
        # pairs. Distinguish "bound but empty" from "never bound" — a
        # truthy check on get_bound_data_items() would conflate the
        # two (both return the empty tuple), which made ForItems over
        # an empty bound Dict render as a folded box even when
        # fold_loops=False.
        if (
            isinstance(dict_value, DictValue)
            and dict_value.metadata.dict_runtime is not None
        ):
            return list(dict_value.get_bound_data_items())

        if param_values is not None:
            logical_id_remap = logical_id_remap or {}
            resolved_lid = logical_id_remap.get(
                dict_value.logical_id, dict_value.logical_id
            )
            for key in (
                f"_dict_data_{dict_value.logical_id}",
                f"_dict_data_{resolved_lid}",
            ):
                if key in param_values:
                    return list(param_values[key])

        return None

    def _get_gate_params_for_expression(
        self,
        op: GateOperation,
        loop_vars: set[str] | None = None,
        body_operations: list | None = None,
        param_values: dict | None = None,
    ) -> str | None:
        """Get gate parameters as a string for expression format.

        Args:
            op: GateOperation.
            loop_vars: Set of loop variable names in scope (e.g., {"i", "j"}).
            body_operations: Operations list for resolving index expressions.
            param_values: Parameter values for resolving symbolic expressions.

        Returns:
            Parameter string (e.g., "0.5") or None if no parameters.
        """
        # Parameterized gates have additional operands beyond qubits
        # Check for common parameterized gates
        param_gates = {
            GateOperationType.RX,
            GateOperationType.RY,
            GateOperationType.RZ,
            GateOperationType.RZZ,
            GateOperationType.P,
            GateOperationType.CP,
        }

        if op.gate_type not in param_gates:
            return None

        # Check theta attribute first (rotation gates store angle here)
        if hasattr(op, "theta") and op.theta is not None:
            theta = op.theta
            if isinstance(theta, (int, float)):
                return self._format_parameter(theta)
            elif isinstance(theta, Value):
                const = theta.get_const()
                if const is not None:
                    return self._format_parameter(const)
                elif hasattr(theta, "parent_array") and theta.parent_array is not None:
                    array_name = theta.parent_array.name or "params"
                    idx_str = self._resolve_index_expression(
                        theta, loop_vars or set(), body_operations
                    )
                    return self._format_symbolic_param(f"{array_name}[{idx_str}]")
                elif theta.is_parameter():
                    param_name = theta.parameter_name() or "θ"
                    return self._format_symbolic_param(param_name)
                else:
                    # BinOp result (e.g., gamma * wij)
                    symbolic = self._resolve_binop_as_symbolic(
                        theta, param_values or {}, body_operations
                    )
                    if symbolic is not None and not self._is_internal_temp_name(
                        symbolic
                    ):
                        return self._format_symbolic_expression(symbolic)
                    if (
                        hasattr(theta, "name")
                        and theta.name
                        and not self._is_internal_temp_name(theta.name)
                    ):
                        return self._format_symbolic_param(theta.name)

        return None

    def _format_param_for_expression(
        self,
        operand: Value,
        loop_vars: set[str],
        param_values: dict | None = None,
        body_operations: list | None = None,
    ) -> str | None:
        """Format a non-qubit parameter operand for folded body expressions.

        Applies numeric evaluation and TeX symbolic formatting consistently
        for ControlledUOperation, InvokeOperation, and InverseBlockOperation
        expression branches.

        Args:
            operand: Non-qubit Value to format.
            loop_vars: Loop variable names in scope.
            param_values: Parameter values for evaluation.
            body_operations: Operations list for BinOp resolution.

        Returns:
            Formatted string (possibly TeX), or None if unresolvable.
        """
        # Try numeric evaluation first
        if param_values is not None:
            evaluated = self._evaluate_value(operand, param_values, body_operations)
            if evaluated is not None and isinstance(evaluated, (int, float)):
                return self._format_parameter(evaluated)

        # Defining BinOp in scope — expand recursively before consulting
        # pre-computed symbolic strings. Keeps kernel parameters
        # symbolic (renders "r*2*n" rather than "r*2*2" because
        # _format_value_as_expression does not consult param_values for
        # numeric folding).
        if self._operand_is_binop_result(operand, body_operations):
            expr = self._format_value_as_expression(operand, loop_vars, body_operations)
            if expr and not self._is_internal_temp_name(expr):
                return self._format_symbolic_param(expr)

        # Symbolic string from pre-evaluation (e.g., _build_symbolic_binop
        # storing "reps*2*n" keyed by logical_id). Reached only when the
        # operand lacks a visible defining BinOp — typical for top-level
        # intermediates that were pre-evaluated during build_visual_ir.
        if param_values is not None and operand.logical_id in param_values:
            pv = param_values[operand.logical_id]
            if isinstance(pv, str) and not self._is_internal_temp_name(pv):
                return self._format_symbolic_param(pv)

        # Array element: resolve index and apply TeX formatting
        if hasattr(operand, "parent_array") and operand.parent_array is not None:
            arr = operand.parent_array.name or "params"
            idx = self._resolve_index_expression(operand, loop_vars, body_operations)
            return self._format_symbolic_param(f"{arr}[{idx}]")

        # Named parameter: apply TeX formatting. Skip IR-internal
        # placeholders to avoid leaking "uint_tmp" etc.
        if (
            hasattr(operand, "name")
            and operand.name
            and not self._is_internal_temp_name(operand.name)
        ):
            return self._format_symbolic_param(operand.name)

        return None

    def _operand_is_binop_result(
        self,
        operand: Value,
        operations: list | None,
    ) -> bool:
        """Return True if `operand` is the result Value of a BinOp in scope.

        Args:
            operand: IR Value to test.
            operations: Operations list to search. Falls back to
                ``self.graph.operations`` when None.

        Returns:
            True if a matching BinOp is found, else False.
        """
        ops = operations
        if ops is None:
            graph = getattr(self, "graph", None)
            ops = graph.operations if graph else []
        for op in ops:
            if (
                isinstance(op, BinOp)
                and op.results
                and id(op.results[0]) == id(operand)
            ):
                return True
        return False

    def _format_operation_as_expression(
        self,
        op: Operation,
        loop_vars: set[str],
        indent: int = 0,
        max_depth: int = 2,
        param_values: dict | None = None,
        body_operations: list | None = None,
    ) -> str | None:
        """Convert an operation to expression format string.

        Args:
            op: Operation to format.
            loop_vars: Set of loop variable names in scope (e.g., {"i", "j"}).
            indent: Indentation level (2 spaces per level).
            max_depth: Maximum nesting depth for recursive formatting.
            param_values: Parameter values for resolving symbolic expressions.

        Returns:
            Expression string (e.g., "q[i],q[j] = cx(q[i],q[j])") or None.
        """
        prefix = "  " * indent

        if isinstance(op, GateOperation):
            if op.gate_type is None:
                return None
            gate_name = op.gate_type.name.lower()

            if not op.operands:
                return None

            # Collect qubit operand strings. Rotation gates (rx/ry/rz
            # etc.) store their angle as the last operand via
            # GateOperation.rotation(); exclude it here by filtering on
            # QubitType so the result LHS and argument list show only
            # qubit operands.
            qubit_strs = []
            for operand in op.operands:
                if not isinstance(operand.type, QubitType):
                    continue
                if (
                    hasattr(operand, "parent_array")
                    and operand.parent_array is not None
                ):
                    array_name = operand.parent_array.name or "qubits"
                    idx_str = self._resolve_index_expression(
                        operand, loop_vars, body_operations
                    )
                    qubit_strs.append(f"{array_name}[{idx_str}]")
                elif hasattr(operand, "name") and operand.name:
                    qubit_strs.append(operand.name)

            if not qubit_strs:
                return None

            params = self._get_gate_params_for_expression(
                op, loop_vars, body_operations, param_values=param_values
            )
            result_str = ",".join(qubit_strs)
            args_str = ",".join(qubit_strs)
            if params:
                args_str += f", {params}"
            return f"{prefix}{result_str} = {gate_name}({args_str})"

        elif isinstance(op, (MeasureOperation, MeasureVectorOperation)):
            if not op.operands:
                return None

            operand = op.operands[0]
            if hasattr(operand, "parent_array") and operand.parent_array is not None:
                array_name = operand.parent_array.name or "qubits"
                idx_str = self._resolve_index_expression(
                    operand, loop_vars, body_operations
                )
                return f"{prefix}measure({array_name}[{idx_str}])"
            elif hasattr(operand, "name") and operand.name:
                return f"{prefix}measure({operand.name})"
            return f"{prefix}measure(...)"

        elif isinstance(op, InvokeOperation):
            block_name = op.name or "callable"
            qubit_parts: list[str] = []
            param_parts: list[str] = []
            if op.attrs.get("kind") == "composite":
                qubit_operands = list(op.control_qubits) + list(op.target_qubits)
                param_operands = list(op.parameters)
            else:
                qubit_operands = [
                    v for v in op.operands if isinstance(v.type, QubitType)
                ]
                param_operands = [
                    v for v in op.operands if not isinstance(v.type, QubitType)
                ]
            for operand in qubit_operands:
                s = None
                if (
                    hasattr(operand, "parent_array")
                    and operand.parent_array is not None
                ):
                    arr = operand.parent_array.name or "qubits"
                    idx = self._resolve_index_expression(
                        operand, loop_vars, body_operations
                    )
                    s = f"{arr}[{idx}]"
                elif hasattr(operand, "name") and operand.name:
                    s = operand.name
                if s is not None:
                    qubit_parts.append(s)
            for operand in param_operands:
                s = self._format_param_for_expression(
                    operand, loop_vars, param_values, body_operations
                )
                if s is not None:
                    param_parts.append(s)

            if not qubit_parts:
                return f"{prefix}{block_name}(...)"
            result_str = ",".join(qubit_parts)
            args_str = ",".join(qubit_parts + param_parts)
            return f"{prefix}{result_str} = {block_name}({args_str})"

        elif isinstance(op, ControlledUOperation):
            block_value = op.block
            block_name = (
                block_value.name if isinstance(block_value, Block) else "U"
            ) or "U"

            def _qubit_str(v: Value) -> str | None:
                if hasattr(v, "parent_array") and v.parent_array is not None:
                    arr = v.parent_array.name or "qubits"
                    idx = self._resolve_index_expression(v, loop_vars, body_operations)
                    return f"{arr}[{idx}]"
                if hasattr(v, "name") and v.name:
                    return v.name
                return None

            all_operands = list(op.control_operands) + list(op.target_operands)
            qubit_parts = [
                s
                for v in all_operands
                if isinstance(v.type, QubitType) and (s := _qubit_str(v)) is not None
            ]
            param_parts = [
                s
                for v in all_operands
                if not isinstance(v.type, QubitType)
                and (
                    s := self._format_param_for_expression(
                        v, loop_vars, param_values, body_operations
                    )
                )
                is not None
            ]
            if not qubit_parts:
                return f"{prefix}{block_name}(...)"
            result_str = ",".join(qubit_parts)
            args_str = ",".join(qubit_parts + param_parts)
            return f"{prefix}{result_str} = {block_name}({args_str})"

        elif isinstance(op, InverseBlockOperation):
            block_name = op.name or "composite"
            qubit_parts: list[str] = []
            param_parts: list[str] = []
            for qval in list(op.control_qubits) + list(op.target_qubits):
                s = None
                if hasattr(qval, "parent_array") and qval.parent_array is not None:
                    arr = qval.parent_array.name or "qubits"
                    idx = self._resolve_index_expression(
                        qval, loop_vars, body_operations
                    )
                    s = f"{arr}[{idx}]"
                elif hasattr(qval, "name") and qval.name:
                    s = qval.name
                if s is not None:
                    qubit_parts.append(s)
            for pval in op.parameters:
                s = self._format_param_for_expression(
                    pval, loop_vars, param_values, body_operations
                )
                if s is not None:
                    param_parts.append(s)
            if not qubit_parts:
                return f"{prefix}{block_name}(...)"
            result_str = ",".join(qubit_parts)
            args_str = ",".join(qubit_parts + param_parts)
            return f"{prefix}{result_str} = {block_name}({args_str})"

        elif isinstance(op, ForOperation):
            if max_depth <= 0:
                return f"{prefix}..."

            range_str = self._format_range_str(op, loop_vars, param_values or {})
            header = f"{prefix}for {op.loop_var} in {range_str}:"
            inner_vars = loop_vars | {op.loop_var}

            lines = [header]
            for body_op in op.operations:
                expr = self._format_operation_as_expression(
                    body_op,
                    inner_vars,
                    indent + 1,
                    max_depth - 1,
                    param_values,
                    body_operations=op.operations,
                )
                if expr:
                    lines.append(expr)
            return "\n".join(lines)

        elif isinstance(op, WhileOperation):
            return f"{prefix}while ...:"

        elif isinstance(op, IfOperation):
            return f"{prefix}if ...:"

        elif isinstance(op, ForItemsOperation):
            key_str = ", ".join(op.key_vars) if op.key_vars else "k"
            return f"{prefix}for {key_str}, {op.value_var} in ...:"

        return None

    def _resolve_index_expression(
        self,
        operand: Value,
        loop_vars: set[str],
        operations: list | None = None,
    ) -> str:
        """Resolve an operand's element index to a human-readable string.

        Args:
            operand: A Value that may be an array element.
            loop_vars: Set of loop variable names in scope.
            operations: Operations list to search for BinOp definitions.

        Returns:
            Index expression string (e.g., "i", "i+1", "1").
        """
        if not hasattr(operand, "element_indices") or not operand.element_indices:
            return next(iter(loop_vars)) if loop_vars else "?"

        idx_value = operand.element_indices[0]

        # Constant index (e.g., j = 1; qubits[j])
        const = idx_value.get_const()
        if const is not None:
            return (
                str(int(const))
                if isinstance(const, float) and const == int(const)
                else str(const)
            )

        # Try to find the defining BinOp operation for this value
        return self._format_value_as_expression(idx_value, loop_vars, operations)

    def _format_value_as_expression(
        self,
        value: Value,
        loop_vars: set[str],
        operations: list | None = None,
    ) -> str:
        """Format a Value as a human-readable expression string.

        Recursively resolves BinOp chains to produce expressions like "i+1", "2*i".

        Args:
            value: IR Value to format.
            loop_vars: Set of loop variable names in scope.
            operations: Operations list to search for BinOp definitions.
                If None, falls back to self.graph.operations.

        Returns:
            Human-readable expression string.
        """
        # Constant
        if value.is_constant():
            c = value.get_const()
            if c is not None:
                return str(int(c)) if isinstance(c, float) and c == int(c) else str(c)

        # Loop variable
        if value.is_parameter():
            name = value.parameter_name()
            if name:
                return name

        # Check if this value is a loop variable by name
        if hasattr(value, "name") and value.name in loop_vars:
            return value.name

        # Search for the defining BinOp in the provided or graph-level operations
        if operations is None:
            graph = getattr(self, "graph", None)
            operations = graph.operations if graph else []
        for op in operations:
            if isinstance(op, BinOp) and op.results and id(op.results[0]) == id(value):
                lhs_str = self._format_value_as_expression(
                    op.lhs, loop_vars, operations
                )
                rhs_str = self._format_value_as_expression(
                    op.rhs, loop_vars, operations
                )
                if op.kind == BinOpKind.MIN:
                    return f"min({lhs_str},{rhs_str})"
                _binop_symbols: dict[BinOpKind, str] = {
                    BinOpKind.ADD: "+",
                    BinOpKind.SUB: "-",
                    BinOpKind.MUL: "*",
                    BinOpKind.FLOORDIV: "//",
                    BinOpKind.MOD: "%",
                    BinOpKind.POW: "**",
                }
                op_symbol = (
                    _binop_symbols.get(op.kind, "?") if op.kind is not None else "?"
                )
                return f"{lhs_str}{op_symbol}{rhs_str}"

        # Array element access (e.g., edges[_e, 0])
        if hasattr(value, "parent_array") and value.parent_array is not None:
            array_name = value.parent_array.name or "arr"
            if hasattr(value, "element_indices") and value.element_indices:
                idx_parts = []
                for idx in value.element_indices:
                    idx_parts.append(
                        self._format_value_as_expression(idx, loop_vars, operations)
                    )
                return f"{array_name}[{','.join(idx_parts)}]"
            return array_name

        # Named value fallback (e.g., edges_dim0 for unbound array
        # shapes). Refuse IR-internal placeholders to avoid leaking
        # "uint_tmp" etc. to rendered labels.
        if (
            hasattr(value, "name")
            and value.name
            and not self._is_internal_temp_name(value.name)
        ):
            return value.name

        return next(iter(loop_vars)) if loop_vars else "?"

    def _format_parameter(self, value: float | int) -> str:
        """Format a parameter value to fit in gate box.

        Args:
            value: Numeric parameter value.

        Returns:
            Formatted string:
            - Very small/large numbers: scientific notation (1e-3, 2e5)
            - Normal numbers: 1-2 decimal places
            - Max length: ~5 characters
        """
        if isinstance(value, int):
            if value == 0:
                return "0"
        elif math.isclose(value, 0.0, abs_tol=1e-15):
            return "0"

        abs_val = abs(value)

        # Very large or very small: use scientific notation
        if abs_val >= 1000 or abs_val < 0.01:
            return f"{value:.0e}"  # e.g., "1e3", "-2e-4"

        # Normal range: truncate decimals
        if abs_val >= 10:
            return f"{value:.1f}"  # e.g., "12.3"
        return f"{value:.2f}"  # e.g., "0.79"

    @staticmethod
    def _extract_greek_prefix(name: str) -> tuple[str, str] | None:
        """Find the longest Greek letter prefix in *name*.

        Returns ``(tex_symbol, remainder)`` or ``None``.
        Only matches when there IS a remaining suffix (not exact match).
        """
        best: tuple[str, str] | None = None
        for symbol in _TEX_SYMBOLS:
            if name.startswith(symbol) and len(symbol) < len(name):
                if best is None or len(symbol) > len(best[0]):
                    best = (symbol, name[len(symbol) :])
        return best

    def _format_symbolic_param(self, name: str) -> str:
        """Format symbolic parameter name for display.

        Greek-letter names get TeX rendering ($\\theta$).
        Names with a Greek prefix get the prefix converted (phis → ${\\phi}s$).
        Non-Greek names with underscores/brackets get subscript notation.
        Other names are returned as plain text.

        Args:
            name: Parameter name.

        Returns:
            Formatted string suitable for matplotlib text rendering.
        """
        if "\\" in name:
            # Already a TeX command like \theta, \phi
            return f"${name}$"

        # Handle array subscript notation (e.g., "phi[0]", "phis[i]")
        bracket_pos = name.find("[")
        if bracket_pos > 0:
            base = name[:bracket_pos]
            rest = name[bracket_pos:]  # e.g., "[i]" or "[0]"
            if base in _TEX_SYMBOLS:
                return f"$\\{base}{rest}$"
            prefix = self._extract_greek_prefix(base)
            if prefix:
                symbol, suffix = prefix
                return f"${{\\{symbol}}}{suffix}{rest}$"
            return f"$\\mathrm{{{base}}}{rest}$"

        # Handle underscore notation (e.g., "theta_2", "x_2", "phis_0")
        if "_" in name:
            parts = name.split("_")
            # Build nested subscripts: beta_a_b_c → \beta_{a_{b_{c}}}
            subscript = parts[-1]
            for i in range(len(parts) - 2, 0, -1):
                subscript = f"{parts[i]}_{{{subscript}}}"

            base = parts[0]
            if base in _TEX_SYMBOLS:
                return f"$\\{base}_{{{subscript}}}$"
            prefix = self._extract_greek_prefix(base)
            if prefix:
                symbol, suffix = prefix
                return f"${{\\{symbol}}}{suffix}_{{{subscript}}}$"
            return f"$\\mathrm{{{base}}}_{{{subscript}}}$"

        # Simple name
        if name in _TEX_SYMBOLS:
            return f"$\\{name}$"

        # Simple name with Greek prefix (e.g., "phis" → ${\phi}s$)
        prefix = self._extract_greek_prefix(name)
        if prefix:
            symbol, suffix = prefix
            return f"${{\\{symbol}}}{suffix}$"

        return name  # plain text, e.g., "params"

    def _format_symbolic_expression(self, expr: str) -> str:
        """Format a symbolic expression (e.g., 'theta/2') as a TeX math string.

        Splits on arithmetic operators and converts each operand individually,
        applying Greek letter TeX conversion where appropriate.
        """
        parts = re.split(r"([+\-*/])", expr)
        tex_parts: list[str] = []
        for part in parts:
            stripped = part.strip()
            if stripped in "+-*/":
                tex_parts.append(stripped)
            elif stripped:
                if stripped in _TEX_SYMBOLS:
                    tex_parts.append(f"\\{stripped}")
                else:
                    tex_parts.append(stripped)
        return "$" + "".join(tex_parts) + "$"

    def _get_gate_label(
        self, op: GateOperation, param_values: dict | None = None
    ) -> tuple[str, bool]:
        """Get display label for a gate.

        Args:
            op: Gate operation.
            param_values: Mapping from logical_ids to resolved parameter values,
                used for inline block expansion.

        Returns:
            Tuple of (label_string, has_parameter).
        """
        # TeX-style gate names — sourced from the module-level
        # ``_TEX_LABELS_BY_GATE_TYPE`` so the controlled-U-box
        # rendering path (``_BUILTIN_TEX_LABELS``) and this inline
        # path stay in sync.
        base_label = (
            _TEX_LABELS_BY_GATE_TYPE.get(op.gate_type, str(op.gate_type))
            if op.gate_type is not None
            else "?"
        )

        # Add parameter display if the gate has a theta parameter
        if op.theta is not None:
            theta = op.theta
            if isinstance(theta, (int, float)):
                param_str = self._format_parameter(theta)
            elif isinstance(theta, Value):
                # Check if the Value has a bound constant
                const_val = theta.get_const()
                if const_val is not None:
                    # Use the bound constant value
                    param_str = self._format_parameter(const_val)
                elif theta.is_parameter():
                    # Check if resolved through param_values (inline block expansion)
                    if param_values and theta.logical_id in param_values:
                        resolved = param_values[theta.logical_id]
                        if isinstance(resolved, (int, float)):
                            param_str = self._format_parameter(resolved)
                        else:
                            param_str = self._format_symbolic_param(str(resolved))
                    else:
                        # Try evaluating (handles array element access like phis[i])
                        evaluated = self._evaluate_value(theta, param_values or {})
                        if evaluated is not None and isinstance(
                            evaluated, (int, float)
                        ):
                            param_str = self._format_parameter(evaluated)
                        else:
                            # Try resolving array indices (e.g., phis[i] → phis[0])
                            resolved_name = self._resolve_symbolic_array_name(
                                theta, param_values or {}
                            )
                            if resolved_name is not None:
                                param_str = self._format_symbolic_param(resolved_name)
                            else:
                                name = theta.parameter_name() or theta.name
                                if self._is_internal_temp_name(name):
                                    name = "?"
                                param_str = self._format_symbolic_param(name or "?")
                else:
                    # Generic Value — try evaluating (handles array elements, BinOps)
                    evaluated = self._evaluate_value(theta, param_values or {})
                    if evaluated is not None and isinstance(evaluated, (int, float)):
                        param_str = self._format_parameter(evaluated)
                    elif (
                        param_values
                        and theta.logical_id in param_values
                        and isinstance(param_values[theta.logical_id], str)
                        and not self._is_internal_temp_name(
                            param_values[theta.logical_id]
                        )
                    ):
                        param_str = self._format_symbolic_param(
                            param_values[theta.logical_id]
                        )
                    else:
                        resolved_name = self._resolve_symbolic_array_name(
                            theta, param_values or {}
                        )
                        if resolved_name is not None:
                            param_str = self._format_symbolic_param(resolved_name)
                        else:
                            symbolic = self._resolve_binop_as_symbolic(
                                theta, param_values or {}
                            )
                            if symbolic is not None and not self._is_internal_temp_name(
                                symbolic
                            ):
                                param_str = self._format_symbolic_expression(symbolic)
                            else:
                                fallback = theta.name
                                if fallback is None or self._is_internal_temp_name(
                                    fallback
                                ):
                                    fallback = "?"
                                param_str = self._format_symbolic_param(fallback)
            else:
                # Unknown type, convert to string (defensive fallback)
                param_str = str(theta)  # type: ignore[unreachable]

            return f"{base_label}({param_str})", True

        return base_label, False
