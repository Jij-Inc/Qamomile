"""Circuit analysis: IR inspection, value resolution, and label generation.

This module provides CircuitAnalyzer, which handles all IR-level analysis
for the circuit visualization pipeline without depending on renderer state.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import cast

from qamomile.circuit.ir.block import Block, BlockKind
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
    HasNestedOps,
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
from qamomile.circuit.ir.value import (
    ArrayValue,
    DictValue,
    TupleValue,
    Value,
    ValueBase,
    ValueLike,
    split_indexed_identifier,
)
from qamomile.circuit.transpiler.block_parameter_binding import (
    align_formal_operands,
)

from .geometry import compute_border_padding
from .style import CircuitStyle
from .text_metrics import measure_text_width
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

_INTERNAL_TMP_NAMES: frozenset[str] = frozenset({"uint_tmp", "float_tmp", "bit_tmp"})


# String markers appended to a node's scope path for the true / false branch
# of an ``IfOperation`` (see ``_build_vif``) and for the body of a
# ``WhileOperation`` (see ``_build_vwhile``). They are the only non-integer
# scope-path elements any node key carries, so their presence uniquely
# identifies a node nested inside a conditional scope — used to keep
# mid-circuit measurements from terminating a wire: an if branch's
# measurement may not run (the other branch never measured), and a while
# body's measurement re-runs every iteration, so in both cases the wire must
# continue past the box.
_CONDITIONAL_SCOPE_KEYS: frozenset[str] = frozenset({"true", "false", "body"})

# Reserved logical-ID remap entry carrying the current nested callable scope.
# Real logical IDs are UUID-like values, so this descriptive sentinel cannot
# collide with an IR value identity.
_CALL_SCOPE_REMAP_KEY = "__qamomile_visual_call_scope__"

# Carry recursive callable expansion state through the existing lexical
# parameter mapping.  This keeps qubit-map and Visual-IR construction
# deterministic even when callers use separate ``CircuitAnalyzer`` instances.
_CALL_STACK_CONTEXT_KEY: object = object()
_CALL_BUDGET_CONTEXT_KEY: object = object()
_UNRESOLVED_SLICE_CONTEXT_KEY: object = object()
_CONSERVATIVE_ALIAS_TOKEN = "?"

# Match the transpiler's recursive unroll safety limit.  Visualization degrades
# the final recursive use to a box instead of raising after the limit.
_MAX_RECURSIVE_INLINE_STATES = 64
_MAX_RECURSIVE_INLINE_EXPANSIONS = 256


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
            name (str | None): Candidate display string.

        Returns:
            bool: True if `name` equals one of the reserved placeholders.
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
        fold_whiles: bool = False,
    ) -> None:
        """Initialize the visualization analyzer.

        Args:
            graph (Block): Computation graph to analyze for rendering.
            style (CircuitStyle): Visual style configuration.
            inline (bool): Whether to expand inline callable contents.
            fold_loops (bool): Whether to render ``for`` loop operations as
                folded summary blocks instead of materialized iterations. Does
                not affect ``while`` loops, which are governed by
                ``fold_whiles``.
            expand_composite (bool): Whether to expand composite gates.
            inline_depth (int | None): Maximum nesting depth for inline
                expansion, or None for unlimited depth.
            fold_ifs (bool): Whether to render IfOperation nodes as folded
                summary blocks instead of side-by-side branches.
            fold_whiles (bool): Whether to render WhileOperation nodes as
                folded summary blocks instead of an expanded body-in-box.
                Defaults to False, mirroring ``fold_ifs``: a ``while`` is
                measurement-backed control flow and reads best with its body
                shown, so it expands by default.
        """
        self.graph = graph
        self.style = style
        self.inline = inline
        self.fold_loops = fold_loops
        self.expand_composite = expand_composite
        self.inline_depth = inline_depth
        self.fold_ifs = fold_ifs
        self.fold_whiles = fold_whiles
        self._scoped_callable_wire_index: dict[str, list[int]] = {}

    def _should_inline_at_depth(self, depth: int) -> bool:
        """Return whether legacy call/control blocks expand at this depth.

        Args:
            depth (int): Current nested visualization depth.

        Returns:
            bool: True when inline expansion is enabled at ``depth``.
        """
        return self.inline and self._within_expansion_depth(depth)

    def _within_expansion_depth(self, depth: int) -> bool:
        """Return whether a nested body may expand at the given depth.

        Args:
            depth (int): Current nested visualization depth.

        Returns:
            bool: True when no expansion limit is configured or ``depth`` is
                below that limit.
        """
        return self.inline_depth is None or depth < self.inline_depth

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
        return self.expand_composite and self._within_expansion_depth(depth)

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

    @classmethod
    def _structural_input_pairs(
        cls,
        formal: ValueBase,
        actual: ValueBase,
    ) -> list[tuple[ValueBase, ValueBase]]:
        """Pair a callable input and its structurally bound tuple elements.

        Dictionary entries are intentionally not paired. A symbolic formal
        dictionary may have no entries while its bound actual dictionary is
        populated, so positional entry pairing would disagree with the inline
        pass and invent an invalid correspondence.

        Args:
            formal (ValueBase): Callee-side formal value.
            actual (ValueBase): Structurally corresponding caller value.

        Returns:
            list[tuple[ValueBase, ValueBase]]: Root pair followed by matching
                tuple-element pairs in depth-first order. Tuple children are
                omitted when their arities differ.
        """
        pairs = [(formal, actual)]
        if not isinstance(formal, TupleValue) or not isinstance(actual, TupleValue):
            return pairs
        if len(formal.elements) != len(actual.elements):
            return pairs
        for formal_element, actual_element in zip(
            formal.elements,
            actual.elements,
            strict=True,
        ):
            pairs.extend(cls._structural_input_pairs(formal_element, actual_element))
        return pairs

    @classmethod
    def _quantum_value_leaves(cls, value: ValueBase) -> list[Value]:
        """Return concrete quantum leaves from one structural IR value.

        Structural tuple and dictionary roots can report ``is_quantum()``
        when they contain qubits, but they are not physical wire owners. Only
        their scalar or array leaves may be sent to wire resolvers.

        Args:
            value (ValueBase): Scalar, array, tuple, or dictionary IR value.

        Returns:
            list[Value]: Quantum scalar and array leaves in structural order.
        """
        if isinstance(value, TupleValue):
            return [
                leaf
                for element in value.elements
                for leaf in cls._quantum_value_leaves(element)
            ]
        if isinstance(value, DictValue):
            return [
                leaf
                for key, entry_value in value.entries
                for item in (key, entry_value)
                for leaf in cls._quantum_value_leaves(item)
            ]
        if isinstance(value, Value) and value.type.is_quantum():
            return [value]
        return []

    @classmethod
    def _flatten_quantum_values(
        cls,
        values: Sequence[ValueBase],
    ) -> list[Value]:
        """Flatten quantum leaves from a sequence of structural values.

        Args:
            values (Sequence[ValueBase]): Values to traverse in order.

        Returns:
            list[Value]: Quantum scalar and array leaves in encounter order.
        """
        return [leaf for value in values for leaf in cls._quantum_value_leaves(value)]

    @classmethod
    def _structural_value_nodes(cls, value: ValueBase) -> list[ValueBase]:
        """Return a structural value root and all tuple/dictionary children.

        Args:
            value (ValueBase): Scalar or structural IR value to traverse.

        Returns:
            list[ValueBase]: Root followed by nested tuple/dictionary values in
                depth-first order.
        """
        nodes = [value]
        if isinstance(value, TupleValue):
            for element in value.elements:
                nodes.extend(cls._structural_value_nodes(element))
        elif isinstance(value, DictValue):
            for key, entry_value in value.entries:
                nodes.extend(cls._structural_value_nodes(key))
                nodes.extend(cls._structural_value_nodes(entry_value))
        return nodes

    @classmethod
    def _clear_callable_local_bindings(
        cls,
        block_value: Block,
        child_param_values: dict[str, object],
    ) -> None:
        """Clear stale body-local values inherited from a recursive frame.

        Recursive uses share one immutable body object, so operation-result
        logical IDs repeat in every frame. Only formal bindings should flow
        into the next frame; cached BinOp or merge results must be recomputed
        from those new formals.

        Args:
            block_value (Block): Callable body whose local definitions to
                clear recursively.
            child_param_values (dict[str, object]): Mutable child lexical
                bindings copied from the caller frame.
        """
        formal_logical_ids = {
            node.logical_id
            for formal in block_value.input_values
            for node in cls._structural_value_nodes(formal)
        }

        def clear_operations(operations: Sequence[Operation]) -> None:
            """Clear result bindings from one nested operation list.

            Args:
                operations (Sequence[Operation]): Operations whose result
                    identities should be removed from the child scope.
            """
            for operation in operations:
                for result in operation.results:
                    if not isinstance(result, ValueBase):
                        continue
                    for node in cls._structural_value_nodes(result):
                        child_param_values.pop(node.uuid, None)
                        if node.logical_id not in formal_logical_ids:
                            child_param_values.pop(node.logical_id, None)
                if isinstance(operation, HasNestedOps):
                    for nested_operations in operation.nested_op_lists():
                        clear_operations(nested_operations)

        clear_operations(block_value.operations)

    @staticmethod
    def _nested_call_scope(
        logical_id_remap: dict[str, str],
        op: Operation,
        call_path: tuple[int | str, ...],
    ) -> str:
        """Build a stable analyzer-local scope key for one callable use.

        Args:
            logical_id_remap (dict[str, str]): Remapping active at the call
                site, including any enclosing callable scope.
            op (Operation): Callable operation at this scope boundary.
            call_path (tuple[int | str, ...]): Deterministic operation path
                within the current callable or control-flow tree.

        Returns:
            str: Parent-qualified scope key stable across qubit-map and Visual
                IR construction for the same operation graph.
        """
        result_uuids = [
            result.uuid for result in op.results if isinstance(result, ValueBase)
        ]
        if result_uuids:
            token = f"{type(op).__name__}:results:{','.join(result_uuids)}"
        else:
            # Resultless invocations have no caller-local value identity, and
            # multiple call sites may intentionally share one CallableDef.
            # Their deterministic traversal path remains stable when
            # compile-time IF lowering clones the operation graph.
            path = "/".join(str(part) for part in call_path)
            token = f"{type(op).__name__}:path:{path}"
        parent_scope = logical_id_remap.get(_CALL_SCOPE_REMAP_KEY)
        if parent_scope:
            return f"{parent_scope}/{token}"
        return token

    @classmethod
    def _body_owned_quantum_logical_ids(cls, block_value: Block) -> set[str]:
        """Collect quantum logical IDs defined inside one callable body.

        Input-derived next versions preserve their formal logical IDs and are
        excluded. Results created by QInit, nested Invoke, control flow, or any
        other operation remain body-owned and need a call-scope namespace when
        the same body object is expanded at multiple call sites.

        Args:
            block_value (Block): Callable body whose definitions to inspect.

        Returns:
            set[str]: Body-owned quantum leaf logical IDs.
        """
        formal_ids = {
            leaf.logical_id
            for formal in block_value.input_values
            for leaf in cls._quantum_value_leaves(formal)
        }
        result_ids: set[str] = set()

        def collect(operations: list[Operation]) -> None:
            """Collect quantum result IDs through nested control-flow regions.

            Args:
                operations (list[Operation]): Operations to inspect.
            """
            for operation in operations:
                for result in operation.results:
                    if isinstance(result, ValueBase):
                        for leaf in cls._quantum_value_leaves(result):
                            result_ids.add(leaf.logical_id)
                            source_logical_id = leaf.get_cast_source_logical_id()
                            if source_logical_id is not None:
                                result_ids.add(source_logical_id)
                            for carrier_key in leaf.get_cast_qubit_logical_ids() or ():
                                root_and_index = split_indexed_identifier(carrier_key)
                                if root_and_index is not None:
                                    result_ids.add(root_and_index[0])
                if isinstance(operation, HasNestedOps):
                    for nested_operations in operation.nested_op_lists():
                        collect(nested_operations)

        collect(block_value.operations)
        return result_ids - formal_ids

    @staticmethod
    def _lower_callable_compile_time_ifs(
        block_value: Block,
        param_values: dict[str, object],
    ) -> Block:
        """Lower resolvable compile-time branches in one callable body.

        Args:
            block_value (Block): Callable body to prepare for visualization.
            param_values (dict[str, object]): Child-scope bindings used to
                choose compile-time branches.

        Returns:
            Block: Body containing only the selected compile-time branches,
                while runtime conditions remain as IfOperation nodes. An
                already analyzed body is returned unchanged because lowering
                must precede dependency analysis.

        Raises:
            ValidationError: If compile-time branch lowering rejects the body.
        """
        if block_value.kind is BlockKind.ANALYZED:
            return block_value

        # Keep the optional lowering stack out of visualization import time.
        # The drawer needs this pass only when it expands a callable body.
        from qamomile.circuit.transpiler.passes.compile_time_if_lowering import (
            CompileTimeIfLoweringPass,
        )

        binding_view = {
            key: value
            for key, value in cast(dict[object, object], param_values).items()
            if isinstance(key, str)
        }
        return CompileTimeIfLoweringPass(bindings=binding_view).run(block_value)

    def _callable_actual_inputs(
        self,
        op: ControlledUOperation | InverseBlockOperation | InvokeOperation,
        block_value: Block,
    ) -> list[ValueBase]:
        """Return callable operands aligned to a body block's formal inputs.

        Args:
            op (ControlledUOperation | InverseBlockOperation | InvokeOperation):
                Callable operation whose operands should be aligned.
            block_value (Block): Embedded callable body.

        Returns:
            list[ValueBase]: Actual inputs ordered to match
                ``block_value.input_values``.

        Raises:
            TypeError: If ``op`` is not a supported callable operation.
        """
        if isinstance(op, InvokeOperation):
            if op.attrs.get("kind") not in {"composite", "oracle"}:
                return list(op.operands)
            quantum_actuals = list(op.target_qubits)
            if self._invoke_body_owns_controls(op, block_value):
                # A transform-specific controlled body includes its control
                # ports in the signature. When ``effective_body()`` falls
                # back to the direct body instead, the controls remain an
                # outer visualization concern and must not be bound to it.
                quantum_actuals = list(op.control_qubits) + quantum_actuals
            classical_actuals = list(op.parameters)
        elif isinstance(op, ControlledUOperation):
            quantum_actuals = [
                value for value in op.target_operands if value.type.is_quantum()
            ]
            classical_actuals = [
                value for value in op.target_operands if not value.type.is_quantum()
            ]
        elif isinstance(op, InverseBlockOperation):
            quantum_actuals = list(op.target_qubits)
            classical_actuals = list(op.parameters)
        else:
            raise TypeError(f"Unsupported callable type: {type(op).__name__}")

        return self._align_actuals_to_formals(
            block_value.input_values,
            quantum_actuals=quantum_actuals,
            classical_actuals=classical_actuals,
        )

    def _prepare_callable_body(
        self,
        op: ControlledUOperation | InverseBlockOperation | InvokeOperation,
        block_value: Block,
        logical_id_remap: dict[str, str],
        param_values: dict[str, object],
        qubit_map: dict[str, int],
        call_path: tuple[int | str, ...],
    ) -> tuple[Block, dict[str, str], dict[str, object], bool]:
        """Prepare one callable body for consistent map and visual traversal.

        Args:
            op (ControlledUOperation | InverseBlockOperation | InvokeOperation):
                Callable operation owning the body.
            block_value (Block): Selected implementation body.
            logical_id_remap (dict[str, str]): Enclosing logical-ID remapping.
            param_values (dict[str, object]): Enclosing parameter bindings.
            qubit_map (dict[str, int]): Logical-ID-to-wire mapping used for
                slice-view alias resolution.
            call_path (tuple[int | str, ...]): Deterministic operation path of
                this call site.

        Returns:
            tuple[Block, dict[str, str], dict[str, object], bool]: Prepared
                body, child logical-ID remapping, child parameter bindings,
                and whether recursive inline expansion may enter this body.

        Raises:
            TypeError: If ``op`` is not a supported callable operation.
            ValidationError: If compile-time branch lowering rejects the body.
        """
        actual_inputs = self._callable_actual_inputs(op, block_value)
        child_remap, child_param_values = self._build_block_value_mappings(
            block_value,
            actual_inputs,
            logical_id_remap,
            param_values,
            qubit_map=qubit_map,
            call_scope=self._nested_call_scope(
                logical_id_remap,
                op,
                call_path,
            ),
        )
        self._clear_callable_local_bindings(block_value, child_param_values)
        may_expand = self._enter_callable_expansion_state(
            block_value,
            actual_inputs,
            child_param_values,
        )
        prepared_block = self._lower_callable_compile_time_ifs(
            block_value,
            child_param_values,
        )
        self._evaluate_loop_body_intermediates(
            prepared_block.operations,
            child_param_values,
        )
        return prepared_block, child_remap, child_param_values, may_expand

    def _enter_callable_expansion_state(
        self,
        block_value: Block,
        actual_inputs: Sequence[ValueBase],
        child_param_values: dict[str, object],
    ) -> bool:
        """Record one callable state and guard recursive inline expansion.

        A self-recursive body may be expanded again when at least one concrete
        classical value changes (for example ``k=3, 2, 1, 0``), even if an
        unrelated angle remains symbolic. States without any changing concrete
        component cannot prove progress, so their recursive use remains a
        summary box. A hard limit also prevents a concrete but increasing
        driver from exhausting Python's recursion stack.

        Args:
            block_value (Block): Original callable body before local lowering.
            actual_inputs (Sequence[ValueBase]): Caller values aligned with
                ``block_value.input_values``.
            child_param_values (dict[str, object]): Mutable child lexical
                bindings that also carry the analyzer-local expansion stack.

        Returns:
            bool: True when this body may be expanded inline, or False when a
                recursive cycle or safety limit requires a summary box.
        """
        context = cast(dict[object, object], child_param_values)
        stack_value = context.get(_CALL_STACK_CONTEXT_KEY, ())
        stack = stack_value if isinstance(stack_value, tuple) else ()

        signature_parts: list[tuple[str, str]] = []
        concrete_parts: list[tuple[str, str]] = []
        for formal, actual in zip(block_value.input_values, actual_inputs):
            for formal_value, _ in self._structural_input_pairs(formal, actual):
                if isinstance(formal_value, (TupleValue, DictValue)) or (
                    formal_value.type.is_quantum()
                ):
                    continue
                bound = child_param_values.get(formal_value.logical_id)
                signature_part = (formal_value.logical_id, repr(bound))
                signature_parts.append(signature_part)
                if isinstance(bound, (bool, int, float, complex)):
                    concrete_parts.append(signature_part)

        concrete_signature = tuple(concrete_parts)
        state = (id(block_value), tuple(signature_parts), concrete_signature)
        same_body_states = [entry for entry in stack if entry[0] == id(block_value)]
        concrete_progress = bool(concrete_signature) and (
            not same_body_states or concrete_signature != same_body_states[-1][2]
        )
        recursive_reentry = bool(same_body_states)
        budget_value = context.get(_CALL_BUDGET_CONTEXT_KEY)
        if not recursive_reentry or not (
            isinstance(budget_value, list)
            and len(budget_value) == 1
            and isinstance(budget_value[0], int)
        ):
            budget_value = [_MAX_RECURSIVE_INLINE_EXPANSIONS]
            context[_CALL_BUDGET_CONTEXT_KEY] = budget_value
        must_stop = (
            state in stack
            or (recursive_reentry and not concrete_progress)
            or len(same_body_states) >= _MAX_RECURSIVE_INLINE_STATES
            or (recursive_reentry and budget_value[0] <= 0)
        )
        if must_stop:
            return False
        if recursive_reentry:
            budget_value[0] -= 1
        context[_CALL_STACK_CONTEXT_KEY] = (*stack, state)
        return True

    @staticmethod
    def _invoke_body_owns_controls(op: InvokeOperation, block_value: Block) -> bool:
        """Return whether a selected invoke body implements its controls.

        Args:
            op (InvokeOperation): Invocation whose selected body is inspected.
            block_value (Block): Body returned by ``effective_body()``.

        Returns:
            bool: True when a transform-specific implementation body owns the
                invocation's control ports. False for a direct-body fallback,
                where controls remain outside the body.
        """
        implementation = op.implementation_for()
        return bool(
            op.num_control_qubits
            and implementation is not None
            and implementation.body is block_value
        )

    def _invoke_body_results(
        self,
        op: InvokeOperation,
        block_value: Block,
    ) -> Sequence[ValueLike]:
        """Return caller results corresponding to one selected invoke body.

        A controlled invocation whose selected implementation falls back to
        the direct body keeps its control values outside that body. Its result
        list therefore has one leading caller-local result per control that
        has no matching entry in ``block_value.output_values``.

        Args:
            op (InvokeOperation): Invocation whose results should be aligned.
            block_value (Block): Body selected by ``op.effective_body()``.

        Returns:
            Sequence[ValueLike]: Caller-local results aligned with the selected
                body's output values. An invalid arity is left unchanged so
                the strict structural pairing reports it to the caller.
        """
        results = list(op.results)
        if self._invoke_body_owns_controls(op, block_value):
            return results
        control_count = op.num_control_qubits
        if len(results) == len(block_value.output_values) + control_count:
            return results[control_count:]
        return results

    def _array_wire_aliases(
        self,
        array_value: ArrayValue,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict[str, object],
    ) -> tuple[list[int], dict[int, int]]:
        """Resolve conservative and exact aliases through an array view chain.

        Conservative aliases use marker keys because they describe possible
        wires for unresolved semantic positions. Numeric keys remain reserved
        for positions whose physical wire is known exactly. For a derived
        slice, exact ancestor indices are transformed into view-local indices;
        if any selected ancestor position remains unknown, its marker wires
        stay conservative instead of becoming positional aliases.

        Args:
            array_value (ArrayValue): Array whose semantic width bounds exact
                alias lookup.
            qubit_map (dict[str, int]): Logical-ID-to-wire registry.
            logical_id_remap (dict[str, str]): Active callable logical-ID
                remapping, including body-owned call scopes.
            param_values (dict[str, object]): Bindings used to resolve the
                array width and slice transforms.

        Returns:
            tuple[list[int], dict[int, int]]: Conservative candidate wires in
                marker order and exact semantic-index-to-wire aliases.
        """
        initial_size = self._resolve_declared_array_size(
            array_value,
            param_values,
        )
        if initial_size == 0:
            return [], {}

        exact_aliases: dict[int, int] = {}
        possible_wires: list[int] = []
        possible_wire_set: set[int] = set()
        fallback_conservative: list[int] = []
        fallback_exact: dict[int, int] = {}
        current = array_value
        ancestor_offset = 0
        ancestor_step = 1
        mapping_known = True
        seen_arrays: set[int] = set()
        prefer_ancestor = False
        if array_value.slice_of is not None:
            slice_start = array_value.slice_start
            slice_step = array_value.slice_step
            prefer_ancestor = (
                slice_start is not None
                and slice_step is not None
                and self._resolve_slice_bound(slice_start, param_values) is not None
                and self._resolve_slice_bound(slice_step, param_values) is not None
            )

        while True:
            current_identity = id(current)
            if current_identity in seen_arrays:
                if possible_wires or exact_aliases:
                    return possible_wires, exact_aliases
                return fallback_conservative, fallback_exact
            seen_arrays.add(current_identity)

            current_lid = logical_id_remap.get(
                current.logical_id,
                current.logical_id,
            )
            current_size = self._resolve_declared_array_size(
                current,
                param_values,
            )
            current_conservative: list[int] = []
            while True:
                marker_key = (
                    f"{current_lid}_[{_CONSERVATIVE_ALIAS_TOKEN}"
                    f"{len(current_conservative)}]"
                )
                if marker_key not in qubit_map:
                    break
                current_conservative.append(qubit_map[marker_key])

            current_exact: dict[int, int] = {}
            if current_size is not None:
                for index in range(current_size):
                    element_key = f"{current_lid}_[{index}]"
                    if element_key in qubit_map:
                        current_exact[index] = qubit_map[element_key]
            else:
                exact_key_pattern = re.compile(rf"^{re.escape(current_lid)}_\[(\d+)\]$")
                for element_key, wire in qubit_map.items():
                    match = exact_key_pattern.match(element_key)
                    if match is not None:
                        current_exact[int(match.group(1))] = wire

            if current is array_value and prefer_ancestor:
                fallback_conservative = current_conservative
                fallback_exact = current_exact
                current_conservative = []
                current_exact = {}

            if mapping_known and initial_size is not None:
                for local_index in range(initial_size):
                    ancestor_index = ancestor_offset + ancestor_step * local_index
                    if ancestor_index in current_exact:
                        exact_aliases.setdefault(
                            local_index,
                            current_exact[ancestor_index],
                        )
                if current_conservative:
                    if len(exact_aliases) == initial_size:
                        return [], exact_aliases
                    return current_conservative, exact_aliases
                if len(exact_aliases) == initial_size:
                    return [], exact_aliases
            elif mapping_known:
                exact_aliases.update(current_exact)
                if current_conservative:
                    return current_conservative, exact_aliases
            else:
                for wire in [*current_conservative, *current_exact.values()]:
                    if wire not in possible_wire_set:
                        possible_wire_set.add(wire)
                        possible_wires.append(wire)
                aliases_cover_current = bool(current_conservative) or (
                    current_size is not None and len(current_exact) == current_size
                )
                if aliases_cover_current:
                    return possible_wires, exact_aliases

            parent = current.slice_of
            if parent is None:
                if mapping_known:
                    if exact_aliases:
                        return [], exact_aliases
                    return fallback_conservative, fallback_exact
                if possible_wires or exact_aliases:
                    return possible_wires, exact_aliases
                return fallback_conservative, fallback_exact

            slice_start = current.slice_start
            slice_step = current.slice_step
            start = (
                self._resolve_slice_bound(slice_start, param_values)
                if slice_start is not None
                else None
            )
            step = (
                self._resolve_slice_bound(slice_step, param_values)
                if slice_step is not None
                else None
            )
            if mapping_known and start is not None and step is not None:
                ancestor_offset = start + step * ancestor_offset
                ancestor_step = step * ancestor_step
            else:
                mapping_known = False
            current = parent

    def _alias_invoke_output_value(
        self,
        body_output: ValueLike,
        call_result: ValueLike,
        logical_id_remap: dict[str, str],
        qubit_map: dict[str, int],
        param_values: dict[str, object],
    ) -> None:
        """Alias quantum leaves in one invocation output pair.

        Args:
            body_output (ValueLike): Callee-side output value or structural
                container.
            call_result (ValueLike): Matching caller-local result value.
            logical_id_remap (dict[str, str]): Formal-to-actual logical-ID
                remapping active inside the callee.
            qubit_map (dict[str, int]): Mutable logical-ID-to-wire map.
            param_values (dict[str, object]): Bindings used to resolve array
                indices, views, and shapes.

        Raises:
            ValueError: If tuple, dictionary, or quantum leaf structures do
                not match recursively.
        """
        if isinstance(body_output, TupleValue) or isinstance(call_result, TupleValue):
            if not isinstance(body_output, TupleValue) or not isinstance(
                call_result, TupleValue
            ):
                raise ValueError("Invoke tuple output does not match its caller result")
            for body_element, call_element in zip(
                body_output.elements,
                call_result.elements,
                strict=True,
            ):
                self._alias_invoke_output_value(
                    body_element,
                    call_element,
                    logical_id_remap,
                    qubit_map,
                    param_values,
                )
            return

        if isinstance(body_output, DictValue) or isinstance(call_result, DictValue):
            if not isinstance(body_output, DictValue) or not isinstance(
                call_result, DictValue
            ):
                raise ValueError(
                    "Invoke dictionary output does not match its caller result"
                )
            if body_output.logical_id in logical_id_remap:
                # Formal dict inputs are bound at the root only: a symbolic
                # empty formal may correspond to a populated actual dict. The
                # caller result already preserves that actual's entry graph,
                # so positional entry aliasing here would invent a mapping
                # that the inline pass deliberately does not make.
                return
            for body_entry, call_entry in zip(
                body_output.entries,
                call_result.entries,
                strict=True,
            ):
                self._alias_invoke_output_value(
                    body_entry[0],
                    call_entry[0],
                    logical_id_remap,
                    qubit_map,
                    param_values,
                )
                self._alias_invoke_output_value(
                    body_entry[1],
                    call_entry[1],
                    logical_id_remap,
                    qubit_map,
                    param_values,
                )
            return

        if body_output.type.is_quantum() != call_result.type.is_quantum():
            raise ValueError("Invoke quantum output does not match its caller result")
        if not call_result.type.is_quantum():
            return
        if isinstance(body_output, ArrayValue) != isinstance(call_result, ArrayValue):
            raise ValueError("Invoke array output does not match its caller result")

        source_wires = self._resolve_operand_to_qubit_indices(
            body_output,
            qubit_map,
            logical_id_remap,
            param_values,
        )
        if not source_wires:
            return

        result_lid = logical_id_remap.get(
            call_result.logical_id,
            call_result.logical_id,
        )
        qubit_map[result_lid] = source_wires[0]
        if not isinstance(call_result, ArrayValue):
            return

        assert isinstance(body_output, ArrayValue), (
            "[FOR DEVELOPER] A quantum array call result must match an "
            "ArrayValue body output."
        )
        conservative_wires, exact_aliases = self._array_wire_aliases(
            body_output,
            qubit_map,
            logical_id_remap,
            param_values,
        )
        slice_context_value = cast(dict[object, object], param_values).get(
            _UNRESOLVED_SLICE_CONTEXT_KEY
        )
        source_is_unresolved_slice = bool(conservative_wires) or (
            isinstance(slice_context_value, dict)
            and isinstance(
                slice_context_value.get(body_output.logical_id),
                ArrayValue,
            )
        )
        if source_is_unresolved_slice:
            candidate_wires = conservative_wires or source_wires
            for index, wire in enumerate(candidate_wires):
                marker_key = f"{result_lid}_[{_CONSERVATIVE_ALIAS_TOKEN}{index}]"
                qubit_map[marker_key] = wire

            result_size = self._resolve_declared_array_size(
                call_result,
                param_values,
            )
            for semantic_index, wire in exact_aliases.items():
                if result_size is None or semantic_index < result_size:
                    qubit_map[f"{result_lid}_[{semantic_index}]"] = wire
            return

        for index, wire in enumerate(source_wires):
            qubit_map[f"{result_lid}_[{index}]"] = wire

    def _map_invoke_body_outputs(
        self,
        body_outputs: Sequence[ValueLike],
        call_results: Sequence[ValueLike],
        logical_id_remap: dict[str, str],
        qubit_map: dict[str, int],
        param_values: dict[str, object],
    ) -> None:
        """Alias caller-local invocation results to callee output wires.

        Args:
            body_outputs (Sequence[ValueLike]): Callee outputs in signature
                order.
            call_results (Sequence[ValueLike]): Caller-local invocation
                results aligned to ``body_outputs``.
            logical_id_remap (dict[str, str]): Formal-to-actual logical-ID
                remapping active inside the callee.
            qubit_map (dict[str, int]): Mutable logical-ID-to-wire map.
            param_values (dict[str, object]): Bindings used to resolve output
                array indices, views, and shapes.

        Raises:
            ValueError: If output counts or nested value structures differ.
        """
        for body_output, call_result in zip(
            body_outputs,
            call_results,
            strict=True,
        ):
            self._alias_invoke_output_value(
                body_output,
                call_result,
                logical_id_remap,
                qubit_map,
                param_values,
            )

    @classmethod
    def _invoke_qubit_operands(
        cls,
        op: InvokeOperation,
    ) -> list[Value]:
        """Return invoke operand leaves that occupy quantum wires.

        Args:
            op (InvokeOperation): Invocation to inspect.

        Returns:
            list[Value]: Quantum leaves, using composite arity metadata when
                present and falling back to all invocation operands otherwise.
        """
        if op.attrs.get("kind") in {"composite", "oracle"}:
            values = list(op.control_qubits) + list(op.target_qubits)
        else:
            values = list(op.operands)
        return cls._flatten_quantum_values(values)

    @classmethod
    def _invoke_qubit_results(
        cls,
        op: InvokeOperation,
    ) -> list[Value]:
        """Return invoke result leaves that occupy quantum wires.

        Args:
            op (InvokeOperation): Invocation to inspect.

        Returns:
            list[Value]: Quantum scalar and array result leaves in structural
                order.
        """
        return cls._flatten_quantum_values(op.results)

    def build_qubit_map(
        self, graph: "Block"
    ) -> tuple[dict[str, int], dict[int, str], int]:
        """Build mapping from qubit logical_id to wire indices.

        In SSA form, each operation creates new Values via next_version(),
        which preserves logical_id. This means all versions of a qubit share
        the same logical_id, so we only need logical_id-based tracking.

        Args:
            graph (Block): Computation block.

        Returns:
            tuple[dict[str, int], dict[int, str], int]: Qubit logical-ID map,
                display-name map, and total number of wires.
        """
        qubit_map: dict[str, int] = {}
        qubit_names: dict[int, str] = {}
        next_idx = 0

        def allocate_wire(logical_id: str, display_name: str) -> int:
            """Allocate one named physical wire unless it already exists.

            Args:
                logical_id (str): Registry key for the physical wire.
                display_name (str): User-facing wire label.

            Returns:
                int: Existing or newly allocated wire index.
            """
            nonlocal next_idx
            if logical_id in qubit_map:
                return qubit_map[logical_id]
            wire = next_idx
            qubit_map[logical_id] = wire
            qubit_names[wire] = display_name
            next_idx += 1
            return wire

        def register_qubit_value(
            qubits: Value,
            param_values: dict | None = None,
            logical_id_remap: dict[str, str] | None = None,
        ) -> None:
            """Register a newly allocated scalar or vector quantum value.

            Args:
                qubits (Value): Quantum value that owns new physical wires.
                param_values (dict | None): Bindings used to resolve a vector
                    shape. Defaults to None.
                logical_id_remap (dict[str, str] | None): Mapping that scopes
                    body-owned logical IDs to this call site. Defaults to None.

            Raises:
                AssertionError: If an ArrayValue is not qubit-typed.
                NotImplementedError: If a quantum array has rank other than
                    one.
                ValueError: If a one-dimensional array size remains symbolic.
            """
            remap = logical_id_remap or {}
            resolved_lid = remap.get(qubits.logical_id, qubits.logical_id)
            if not isinstance(qubits, ArrayValue):
                allocate_wire(resolved_lid, qubits.name)
                return

            assert isinstance(qubits.type, QubitType), (
                "[FOR DEVELOPER] A quantum ArrayValue must have QubitType."
            )
            if len(qubits.shape) != 1:
                raise NotImplementedError(
                    "Cannot visualize circuit: qubit array "
                    f"'{qubits.name}' has unsupported rank {len(qubits.shape)}."
                )
            array_size = self._resolve_array_size(
                qubits,
                resolved_lid,
                qubit_map,
                param_values or {},
            )
            if array_size is None:
                raise ValueError(
                    "Cannot visualize circuit: qubit array "
                    f"'{qubits.name}' has symbolic size. Please provide "
                    "concrete values for all size parameters when calling draw()."
                )
            for index in range(array_size):
                element_key = f"{resolved_lid}_[{index}]"
                allocate_wire(element_key, f"{qubits.name}[{index}]")
            if array_size:
                qubit_map[resolved_lid] = qubit_map[f"{resolved_lid}_[0]"]

        def map_block_results(
            operands: list[Value],
            results: list[Value],
            logical_id_remap: dict[str, str],
            param_values: dict | None = None,
        ) -> None:
            """Alias quantum block results to already registered input wires.

            Scalar results alias their operand's existing wire, while array
            results preserve resolved operand aliases. An opaque array result
            that is wider than its operand allocates only the missing suffix,
            with a display name for every new physical wire. Empty arrays
            alias no wires, and unresolved slice candidates are recorded as a
            conservative footprint rather than an independent phantom register.

            Args:
                operands (list[Value]): Input values passed to the block
                    (actual arguments).
                results (list[Value]): Output values returned from the block.
                logical_id_remap (dict[str, str]): Mapping from formal-parameter
                    logical_ids to
                    actual-argument logical_ids.
                param_values (dict | None): Parameter values for resolving
                    symbolic indices. Defaults to None.
            """
            for operand, result in zip(operands, results):
                if not result.type.is_quantum():
                    continue
                result_lid = logical_id_remap.get(
                    result.logical_id,
                    result.logical_id,
                )
                resolved_wires = self._resolve_operand_to_qubit_indices(
                    operand,
                    qubit_map,
                    logical_id_remap,
                    param_values or {},
                )
                if resolved_wires is None:
                    continue
                if not resolved_wires:
                    continue

                qubit_map[result_lid] = resolved_wires[0]
                if isinstance(result, ArrayValue):
                    param_context = param_values or {}
                    operand_size = None
                    conservative_wires: list[int] = []
                    exact_aliases: dict[int, int] = {}
                    if isinstance(operand, ArrayValue):
                        operand_lid = logical_id_remap.get(
                            operand.logical_id,
                            operand.logical_id,
                        )
                        operand_size = self._resolve_array_size(
                            operand,
                            operand_lid,
                            qubit_map,
                            param_context,
                        )
                        conservative_wires, exact_aliases = self._array_wire_aliases(
                            operand,
                            qubit_map,
                            logical_id_remap,
                            param_context,
                        )
                    slice_context_value = cast(
                        dict[object, object],
                        param_context,
                    ).get(_UNRESOLVED_SLICE_CONTEXT_KEY)
                    forwarded_unresolved_slice = isinstance(
                        slice_context_value,
                        dict,
                    ) and isinstance(
                        slice_context_value.get(operand.logical_id),
                        ArrayValue,
                    )
                    unresolved_slice = (
                        bool(conservative_wires)
                        or forwarded_unresolved_slice
                        or (
                            isinstance(operand, ArrayValue)
                            and operand.slice_of is not None
                            and self._resolve_view_chain_to_root(
                                operand,
                                param_context,
                            )
                            is None
                        )
                    )
                    result_size = self._resolve_array_size(
                        result,
                        result_lid,
                        qubit_map,
                        param_context,
                    )
                    if unresolved_slice:
                        candidate_wires = conservative_wires or resolved_wires
                        for index, wire in enumerate(candidate_wires):
                            marker_key = (
                                f"{result_lid}_[{_CONSERVATIVE_ALIAS_TOKEN}{index}]"
                            )
                            qubit_map[marker_key] = wire
                        for semantic_index, wire in exact_aliases.items():
                            if result_size is None or semantic_index < result_size:
                                result_key = f"{result_lid}_[{semantic_index}]"
                                qubit_map[result_key] = wire
                        if operand_size is not None and result_size is not None:
                            for semantic_index in range(operand_size, result_size):
                                result_key = f"{result_lid}_[{semantic_index}]"
                                result_name = result.name or "qubits"
                                allocate_wire(
                                    result_key,
                                    f"{result_name}[{semantic_index}]",
                                )
                        continue

                    alias_count = (
                        len(resolved_wires)
                        if result_size is None
                        else min(result_size, len(resolved_wires))
                    )
                    alias_wires = resolved_wires[:alias_count]
                    for index, wire in enumerate(alias_wires):
                        qubit_map[f"{result_lid}_[{index}]"] = wire

        def map_callable_output_provenance(
            block_value: Block,
            call_results: Sequence[ValueLike],
            logical_id_remap: dict[str, str],
            param_values: dict[str, object],
        ) -> None:
            """Map only operations that can define boxed callable outputs.

            Boxed callables do not materialize every body-local allocation,
            but their public outputs still need body-scope shape bindings and
            positional provenance from the operation that produced them. This
            targeted walk avoids exposing unrelated temporary wires while
            preserving dynamic widening and local QInit results.

            Args:
                block_value (Block): Boxed callable body to inspect.
                call_results (Sequence[ValueLike]): Caller-local results
                    aligned with the selected body outputs.
                logical_id_remap (dict[str, str]): Child callable remapping.
                param_values (dict[str, object]): Child bindings with evaluated
                    body-local shape expressions.

            Raises:
                ValueError: If body outputs and caller results have different
                    numbers of quantum leaves.
            """
            body_output_leaves = self._flatten_quantum_values(block_value.output_values)
            call_result_leaves = self._flatten_quantum_values(call_results)
            if len(body_output_leaves) != len(call_result_leaves):
                raise ValueError("Invoke quantum outputs do not match caller results")
            caller_output_by_body_lid = {
                body_output.logical_id: call_result
                for body_output, call_result in zip(
                    body_output_leaves,
                    call_result_leaves,
                    strict=True,
                )
            }
            output_logical_ids: set[str] = set()
            for output in body_output_leaves:
                root_output = output
                if isinstance(root_output, ArrayValue):
                    seen_arrays: set[int] = set()
                    while root_output.slice_of is not None:
                        root_identity = id(root_output)
                        if root_identity in seen_arrays:
                            break
                        seen_arrays.add(root_identity)
                        root_output = root_output.slice_of
                output_logical_ids.add(root_output.logical_id)

            def map_cast_output(
                operation: CastOperation,
                quantum_results: list[Value],
            ) -> None:
                """Map one boxed cast result and its public carrier wires.

                Args:
                    operation (CastOperation): Output-producing cast to map.
                    quantum_results (list[Value]): Quantum leaves produced by
                        the cast.
                """
                assert operation.operands and quantum_results, (
                    "[FOR DEVELOPER] A quantum CastOperation must have a "
                    "source and result."
                )
                source = operation.operands[0]
                cast_result = quantum_results[0]
                map_block_results(
                    [source],
                    [cast_result],
                    logical_id_remap,
                    param_values,
                )
                caller_result = caller_output_by_body_lid.get(cast_result.logical_id)
                if caller_result is None:
                    return

                source_wires = (
                    self._resolve_operand_to_qubit_indices(
                        source,
                        qubit_map,
                        logical_id_remap,
                        param_values,
                    )
                    or []
                )
                conservative_wires: list[int] = []
                exact_aliases: dict[int, int] = {}
                if isinstance(source, ArrayValue):
                    conservative_wires, exact_aliases = self._array_wire_aliases(
                        source,
                        qubit_map,
                        logical_id_remap,
                        param_values,
                    )
                carrier_wires: list[int] = []
                for carrier_position, carrier_key in enumerate(
                    caller_result.get_cast_qubit_logical_ids() or ()
                ):
                    root_and_index = split_indexed_identifier(carrier_key)
                    if root_and_index is None:
                        continue
                    root_lid, index_text = root_and_index
                    remapped_root_lid = logical_id_remap.get(root_lid, root_lid)
                    bracket_key = f"{remapped_root_lid}_[{int(index_text)}]"
                    wire = qubit_map.get(bracket_key)
                    if conservative_wires:
                        for marker_position, marker_wire in enumerate(
                            conservative_wires
                        ):
                            marker_key = (
                                f"{remapped_root_lid}_"
                                f"[{_CONSERVATIVE_ALIAS_TOKEN}{marker_position}]"
                            )
                            qubit_map[marker_key] = marker_wire
                        exact_wire = exact_aliases.get(carrier_position)
                        if exact_wire is not None:
                            qubit_map[bracket_key] = exact_wire
                        continue
                    if wire is None and carrier_position < len(source_wires):
                        wire = source_wires[carrier_position]
                        qubit_map[bracket_key] = wire
                    if wire is None:
                        source_name = source.name or "qubits"
                        wire = allocate_wire(
                            bracket_key,
                            f"{source_name}[{carrier_position}]",
                        )
                    carrier_wires.append(wire)

                if conservative_wires:
                    carrier_wires = list(conservative_wires)
                    seen_carrier_wires = set(carrier_wires)
                    for semantic_index in sorted(exact_aliases):
                        exact_wire = exact_aliases[semantic_index]
                        if exact_wire not in seen_carrier_wires:
                            seen_carrier_wires.add(exact_wire)
                            carrier_wires.append(exact_wire)

                if carrier_wires:
                    result_lid = logical_id_remap.get(
                        cast_result.logical_id,
                        cast_result.logical_id,
                    )
                    qubit_map[result_lid] = carrier_wires[0]

            def visit(operations: Sequence[Operation]) -> None:
                """Map matching producers in one nested operation list.

                Args:
                    operations (Sequence[Operation]): Body operations to scan.
                """
                for operation in operations:
                    quantum_results = self._flatten_quantum_values(operation.results)
                    result_ids = {result.logical_id for result in quantum_results}
                    if result_ids & output_logical_ids:
                        if isinstance(operation, InvokeOperation):
                            map_block_results(
                                self._invoke_qubit_operands(operation),
                                self._invoke_qubit_results(operation),
                                logical_id_remap,
                                param_values,
                            )
                        elif isinstance(operation, ControlledUOperation):
                            operands = [
                                value
                                for value in [
                                    *operation.control_operands,
                                    *operation.target_operands,
                                ]
                                if value.type.is_quantum()
                            ]
                            results = [
                                value
                                for value in operation.results
                                if value.type.is_quantum()
                            ]
                            map_block_results(
                                operands,
                                results,
                                logical_id_remap,
                                param_values,
                            )
                        elif isinstance(operation, InverseBlockOperation):
                            map_block_results(
                                [*operation.control_qubits, *operation.target_qubits],
                                [
                                    value
                                    for value in operation.results
                                    if value.type.is_quantum()
                                ],
                                logical_id_remap,
                                param_values,
                            )
                        elif isinstance(operation, IfOperation):
                            for merge in operation.iter_merges():
                                if merge.result.logical_id not in output_logical_ids:
                                    continue
                                true_wires = self._resolve_operand_to_qubit_indices(
                                    merge.true_value,
                                    qubit_map,
                                    logical_id_remap,
                                    param_values,
                                )
                                false_wires = self._resolve_operand_to_qubit_indices(
                                    merge.false_value,
                                    qubit_map,
                                    logical_id_remap,
                                    param_values,
                                )
                                if true_wires and true_wires == false_wires:
                                    map_block_results(
                                        [merge.true_value],
                                        [merge.result],
                                        logical_id_remap,
                                        param_values,
                                    )
                        elif isinstance(operation, CastOperation):
                            map_cast_output(operation, quantum_results)
                    if isinstance(operation, HasNestedOps):
                        for nested_operations in operation.nested_op_lists():
                            visit(nested_operations)

            visit(block_value.operations)

        def build_chains(
            ops: list[Operation],
            logical_id_remap: dict[str, str] | None = None,
            depth: int = 0,
            param_values: dict | None = None,
            analysis_path: tuple[int | str, ...] = (),
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
                ops (list[Operation]): List of operations to process.
                logical_id_remap (dict[str, str] | None): Mapping from block
                    formal-parameter logical_ids to actual-argument logical_ids.
                    Only non-empty when recursing into inlined callable bodies.
                    Defaults to None.
                depth (int): Current nesting depth for inline-depth checking.
                    Defaults to zero.
                param_values (dict | None): Parameter values for resolving
                    symbolic indices. Defaults to None.
                analysis_path (tuple[int | str, ...]): Deterministic path of
                    the enclosing operation tree. Defaults to the graph root.
            """
            nonlocal next_idx
            if logical_id_remap is None:
                logical_id_remap = {}
            if param_values is None:
                param_values = {}

            for operation_index, op in enumerate(ops):
                operation_path = (*analysis_path, operation_index)
                if isinstance(op, QInitOperation):
                    register_qubit_value(
                        op.results[0],
                        param_values,
                        logical_id_remap,
                    )

                elif isinstance(op, InvokeOperation):
                    block_value = op.effective_body()
                    result_param_values = param_values
                    if isinstance(block_value, Block):
                        prepared_block, new_remap, child_param_values, may_expand = (
                            self._prepare_callable_body(
                                op,
                                block_value,
                                logical_id_remap,
                                param_values,
                                qubit_map,
                                operation_path,
                            )
                        )
                        result_param_values = child_param_values
                        body_results = self._invoke_body_results(op, block_value)
                        expands_body = (
                            may_expand
                            and self._should_inline_invoke_at_depth(op, depth)
                        )
                        if expands_body:
                            build_chains(
                                prepared_block.operations,
                                new_remap,
                                depth + 1,
                                child_param_values,
                                (),
                            )
                        else:
                            map_callable_output_provenance(
                                prepared_block,
                                body_results,
                                new_remap,
                                child_param_values,
                            )
                        self._map_invoke_body_outputs(
                            prepared_block.output_values,
                            body_results,
                            new_remap,
                            qubit_map,
                            child_param_values,
                        )
                        if (
                            op.num_control_qubits
                            and not self._invoke_body_owns_controls(op, block_value)
                        ):
                            control_operands = self._flatten_quantum_values(
                                op.control_qubits
                            )
                            control_results = self._flatten_quantum_values(
                                op.results[: op.num_control_qubits]
                            )
                            map_block_results(
                                control_operands,
                                control_results,
                                logical_id_remap,
                                param_values,
                            )
                    else:
                        # Opaque callables expose no body-output provenance.
                        # Preserve the legacy positional approximation as the
                        # only available way to forward their quantum results.
                        qubit_operands = self._invoke_qubit_operands(op)
                        qubit_results = self._invoke_qubit_results(op)
                        map_block_results(
                            qubit_operands,
                            qubit_results,
                            logical_id_remap,
                            param_values,
                        )
                    for result in self._invoke_qubit_results(op):
                        resolved_wires = self._resolve_operand_to_qubit_indices(
                            result,
                            qubit_map,
                            logical_id_remap,
                            result_param_values,
                        )
                        if resolved_wires is None:
                            register_qubit_value(
                                result,
                                result_param_values,
                                logical_id_remap,
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
                    result_lid = logical_id_remap.get(
                        result.logical_id,
                        result.logical_id,
                    )
                    qubit_idx = qubit_map.get(source_lid)
                    if qubit_idx is not None:
                        qubit_map[result_lid] = qubit_idx
                    else:
                        resolved = self._resolve_array_element_lid(
                            source, qubit_map, logical_id_remap, param_values
                        )
                        resolved_idx = qubit_map.get(resolved)
                        if resolved_idx is not None:
                            qubit_map[result_lid] = resolved_idx

                elif isinstance(
                    op, ControlledUOperation
                ) and self._should_inline_at_depth(depth):
                    block_value = op.block
                    if isinstance(block_value, Block):
                        prepared_block, new_remap, child_param_values, may_expand = (
                            self._prepare_callable_body(
                                op,
                                block_value,
                                logical_id_remap,
                                param_values,
                                qubit_map,
                                operation_path,
                            )
                        )
                        if may_expand:
                            build_chains(
                                prepared_block.operations,
                                new_remap,
                                depth + 1,
                                child_param_values,
                                (),
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
                    prepared_block, new_remap, child_param_values, may_expand = (
                        self._prepare_callable_body(
                            op,
                            block_value,
                            logical_id_remap,
                            param_values,
                            qubit_map,
                            operation_path,
                        )
                    )
                    if may_expand:
                        build_chains(
                            prepared_block.operations,
                            new_remap,
                            depth + 1,
                            child_param_values,
                            (),
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
                    if self._is_zero_iteration_loop(start, stop, step):
                        self._publish_region_arg_results(
                            op,
                            self._initial_region_arg_state(op, param_values),
                            param_values,
                        )
                        continue
                    if stop is not None and (not self.fold_loops or self.inline):
                        region_state = self._initial_region_arg_state(op, param_values)
                        for iter_value in range(start, stop, step):
                            child_pv = dict(param_values)
                            child_pv.update(region_state)
                            child_pv[f"_loop_{op.loop_var}"] = iter_value
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
                                (*operation_path, "body"),
                            )
                            region_state = self._advance_region_arg_state(
                                op,
                                child_pv,
                                region_state,
                            )
                        self._publish_region_arg_results(
                            op,
                            region_state,
                            param_values,
                        )
                    else:
                        child_pv = dict(param_values)
                        child_pv.update(
                            self._initial_region_arg_state(op, param_values)
                        )
                        build_chains(
                            op.operations,
                            logical_id_remap,
                            depth + 1,
                            child_pv,
                            (*operation_path, "body"),
                        )
                        if stop is not None and step != 0:
                            self._simulate_for_region_args(
                                op,
                                range(start, stop, step),
                                param_values,
                            )

                elif isinstance(op, WhileOperation):
                    build_chains(
                        op.operations,
                        logical_id_remap,
                        depth + 1,
                        param_values,
                        (*operation_path, "body"),
                    )

                elif isinstance(op, IfOperation):
                    build_chains(
                        op.true_operations,
                        logical_id_remap,
                        depth + 1,
                        param_values,
                        (*operation_path, "true"),
                    )
                    build_chains(
                        op.false_operations,
                        logical_id_remap,
                        depth + 1,
                        param_values,
                        (*operation_path, "false"),
                    )
                    # An if-else that rebinds a qubit produces a phi-merged
                    # output whose ``logical_id`` is fresh (e.g. ``q1_phi_0``),
                    # distinct from the pre-branch qubit's logical_id.  Map each
                    # quantum merge result onto the same wire as its branch
                    # value so operations after the if-else that read the
                    # merged qubit (a trailing ``qmc.measure(q)``, a later
                    # gate) resolve to the right wire instead of an empty
                    # ``qubit_indices`` list and silently vanish from the
                    # drawing.
                    for merge in op.iter_merges():
                        if not isinstance(merge.result.type, QubitType):
                            continue
                        map_block_results(
                            [merge.true_value],
                            [merge.result],
                            logical_id_remap,
                            param_values,
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
                    if materialized == []:
                        self._publish_region_arg_results(
                            op,
                            self._initial_region_arg_state(op, param_values),
                            param_values,
                        )
                        continue
                    if materialized is not None and (
                        not self.fold_loops or self.inline
                    ):
                        region_state = self._initial_region_arg_state(op, param_values)
                        for entry_key, entry_value in materialized:
                            child_pv = dict(param_values)
                            child_pv.update(region_state)
                            self._bind_for_items_iteration(
                                op,
                                entry_key,
                                entry_value,
                                child_pv,
                            )
                            self._evaluate_loop_body_intermediates(
                                op.operations,
                                child_pv,
                            )
                            build_chains(
                                op.operations,
                                logical_id_remap,
                                depth + 1,
                                child_pv,
                                (*operation_path, "body"),
                            )
                            region_state = self._advance_region_arg_state(
                                op,
                                child_pv,
                                region_state,
                            )
                        self._publish_region_arg_results(
                            op,
                            region_state,
                            param_values,
                        )
                    else:
                        child_pv = dict(param_values)
                        child_pv.update(
                            self._initial_region_arg_state(op, param_values)
                        )
                        build_chains(
                            op.operations,
                            logical_id_remap,
                            depth + 1,
                            child_pv,
                            (*operation_path, "body"),
                        )
                        if materialized is not None:
                            self._simulate_for_items_region_args(
                                op,
                                materialized,
                                param_values,
                            )

                # GateOperation, non-expanded InvokeOperation:
                # No-op — next_version() preserves logical_id

        # A standalone callable ``Block`` may receive physical qubits directly
        # instead of allocating them with QInitOperation. Seed those public
        # input wires before walking operations so every registered wire has a
        # display name and callable result aliasing never needs to allocate.
        for graph_input in graph.input_values:
            for quantum_input in self._quantum_value_leaves(graph_input):
                register_qubit_value(quantum_input)

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
            graph (Block): IR computation block.
            qubit_map (dict[str, int]): Mapping from logical_id to wire index.
            qubit_names (dict[int, str]): Mapping from wire index to display
                name.
            num_qubits (int): Total number of qubit wires.

        Returns:
            VisualCircuit: Circuit containing the VisualNode tree.
        """
        self._scoped_callable_wire_index = self._build_scoped_callable_wire_index(
            qubit_map
        )
        # Pre-evaluate top-level intermediate BinOps so that callable
        # arguments derived from them (e.g. final_base = reps * 2 * n)
        # resolve to numeric or symbolic strings instead of leaking the
        # IR-internal placeholder name "uint_tmp" downstream.
        top_param_values: dict = {}
        self._evaluate_loop_body_intermediates(graph.operations, top_param_values)
        children = self._build_visual_nodes(
            graph.operations,
            qubit_map,
            {},
            top_param_values,
            depth=0,
            scope_path=(),
            analysis_path=(),
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
        analysis_path: tuple[int | str, ...],
    ) -> list[VisualNode]:
        """Recursively build VisualNode list from IR operations.

        This is the core dispatch method that replaces both Layout's
        _measure_operations and Renderer's draw_ops dispatch.

        Args:
            ops (list[Operation]): Operations to convert into visual nodes.
            qubit_map (dict[str, int]): Mapping from logical IDs to wire indices.
            logical_id_remap (dict[str, str]): Mapping from block-local logical
                IDs to their caller-visible logical IDs.
            param_values (dict): Parameter values available in the current
                visualization scope.
            depth (int): Current nested visualization depth.
            scope_path (tuple): Stable path used to construct child node keys.
            analysis_path (tuple[int | str, ...]): Deterministic path used to
                identify resultless callable sites across analysis passes.

        Returns:
            list[VisualNode]: Visual nodes corresponding to ``ops`` in order.
        """
        result: list[VisualNode] = []

        for operation_index, op in enumerate(ops):
            node_key = (*scope_path, id(op))
            operation_path = (*analysis_path, operation_index)

            if isinstance(op, QInitOperation):
                result.append(VSkip(node_key=node_key))
                continue

            if isinstance(op, CastOperation):
                result.append(VSkip(node_key=node_key))
                continue

            if isinstance(op, BinOp):
                continue  # No visual representation

            if isinstance(op, ExpvalOp):
                node = self._build_vexpval(
                    op, node_key, qubit_map, logical_id_remap, param_values
                )
                result.append(node)
                continue

            if isinstance(op, WhileOperation):
                node = self._build_vwhile(
                    op,
                    node_key,
                    qubit_map,
                    logical_id_remap,
                    param_values,
                    depth,
                    scope_path,
                    operation_path,
                    body_operations=ops,
                )
                result.append(node)
                continue

            if isinstance(op, IfOperation):
                node = self._build_vif(
                    op,
                    node_key,
                    qubit_map,
                    logical_id_remap,
                    param_values,
                    depth,
                    scope_path,
                    operation_path,
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
                    operation_path,
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
                    operation_path,
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
                    operation_path,
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
                    operation_path,
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
                    operation_path,
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
    def _node_key_in_conditional_scope(node_key: tuple) -> bool:
        """Whether a node sits inside an if/else branch or while-body scope.

        Conditional scopes are tagged with the ``"true"`` / ``"false"``
        string markers (``_build_vif``) or the ``"body"`` marker
        (``_build_vwhile``) in the node key. No other scope kind uses string
        markers, so their presence uniquely identifies a conditional nesting.

        Args:
            node_key (tuple): The node's scope key, of the form
                ``(*scope_path, id(op))``.

        Returns:
            bool: True if any element of ``node_key`` is a conditional-scope
                marker, meaning the node is nested inside an if branch or a
                while body.
        """
        return any(
            isinstance(part, str) and part in _CONDITIONAL_SCOPE_KEYS
            for part in node_key
        )

    def _build_vgate(
        self,
        op: Operation,
        node_key: tuple,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict,
    ) -> VGate:
        """Build a visual node for a gate-like IR operation.

        Args:
            op (Operation): Gate, measurement, block, composite, inverse, or
                controlled operation to convert.
            node_key (tuple): Stable identity key used by layout and rendering.
            qubit_map (dict[str, int]): Mapping from logical qubit identifiers
                to display-wire indices.
            logical_id_remap (dict[str, str]): Mapping from callee-local logical
                identifiers to caller identifiers for inline expansion.
            param_values (dict): Concrete or symbolic parameter values available
                in the current analysis scope.

        Returns:
            VGate: Pre-resolved visual gate node with display metadata and
                layout dimensions.

        Raises:
            TypeError: If ``op`` is not a supported gate-like operation.
        """
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
                terminates_wire=not self._node_key_in_conditional_scope(node_key),
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
                terminates_wire=not self._node_key_in_conditional_scope(node_key),
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
                terminates_wire=not self._node_key_in_conditional_scope(node_key),
            )

        if isinstance(op, InvokeOperation):
            label = op.name.upper()
            box_width = self._estimate_block_label_box_width(label)
            control_indices: list[int] = []
            for operand in self._flatten_quantum_values(op.control_qubits):
                indices = self._resolve_operand_to_qubit_indices(
                    operand, qubit_map, logical_id_remap, param_values
                )
                if indices is not None:
                    control_indices.extend(indices)
            target_indices: list[int] = []
            target_operands = (
                self._flatten_quantum_values(op.target_qubits)
                if op.num_control_qubits
                or op.attrs.get("kind") in {"composite", "oracle"}
                else self._invoke_qubit_operands(op)
            )
            for operand in target_operands:
                indices = self._resolve_operand_to_qubit_indices(
                    operand, qubit_map, logical_id_remap, param_values
                )
                if indices is not None:
                    target_indices.extend(indices)
            qubit_indices = control_indices + target_indices
            for result_val in self._invoke_qubit_results(op):
                indices = self._resolve_operand_to_qubit_indices(
                    result_val, qubit_map, logical_id_remap, param_values
                )
                if indices is not None:
                    for index in indices:
                        if index not in qubit_indices:
                            qubit_indices.append(index)
            return VGate(
                node_key=node_key,
                label=label,
                qubit_indices=qubit_indices,
                estimated_width=box_width,
                kind=(
                    VGateKind.CONTROLLED_U_BOX
                    if control_indices
                    else self._invoke_box_kind(op)
                ),
                box_width=box_width,
                control_count=len(control_indices),
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
                inner_box_width = self.style.gate_width
            else:
                inner_box_width = self._estimate_label_box_width(label)
            # ``box_width`` is consumed by the renderer for the inner target
            # rectangle.  Layout instead reserves ``estimated_width``, which
            # must include the powered wrapper's margin on both sides.  Keeping
            # these dimensions separate prevents the renderer from applying
            # the wrapper margin twice to the visible patch.
            estimated_width = inner_box_width
            if power_val > 1:
                estimated_width = max(
                    inner_box_width + 2 * self.style.power_wrapper_margin,
                    self._estimate_label_box_width(f"pow={power_val}"),
                )
            # Control qubits first, then target qubits.
            control_indices = self._resolve_controlled_control_wires(
                op,
                qubit_map,
                logical_id_remap,
                param_values,
            )
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
                estimated_width=estimated_width,
                kind=VGateKind.CONTROLLED_U_BOX,
                gate_type=controlled_gate_type,
                box_width=inner_box_width,
                control_count=len(control_indices),
                power=power_val,
            )

        raise TypeError(
            f"Unsupported operation type for _build_vgate: {type(op).__name__}"
        )

    def _resolve_controlled_control_wires(
        self,
        op: ControlledUOperation,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict,
    ) -> list[int]:
        """Resolve the active control wires of a controlled operation.

        Args:
            op (ControlledUOperation): Controlled operation to resolve.
            qubit_map (dict[str, int]): Mapping from logical IDs to wire indices.
            logical_id_remap (dict[str, str]): Caller-to-callee logical-ID map.
            param_values (dict): Bindings used to evaluate symbolic indices.

        Returns:
            list[int]: Active control wire indices in control-operand order.
        """
        wires, _ = self._resolve_controlled_control_footprint(
            op,
            qubit_map,
            logical_id_remap,
            param_values,
        )
        return wires

    def _resolve_controlled_control_footprint(
        self,
        op: ControlledUOperation,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict,
    ) -> tuple[list[int], bool]:
        """Resolve the active control wires of a controlled operation.

        Symbolic controlled operations may select a subset of a larger control
        pool with ``control_indices``. Only selected slots receive control dots
        or contribute to an inline block's control footprint. Exact aliases are
        selected by semantic array index. An unresolved slot expands only to
        that array's conservative candidates, so a marker union is never
        mistaken for positional array order. If a selection index or semantic
        width remains unresolved, the complete pool is retained as fallback.

        Args:
            op (ControlledUOperation): Controlled operation to resolve.
            qubit_map (dict[str, int]): Mapping from logical IDs to wire indices.
            logical_id_remap (dict[str, str]): Caller-to-callee logical-ID map.
            param_values (dict): Bindings used to evaluate symbolic indices.

        Returns:
            tuple[list[int], bool]: Active control wire indices in operand
                order and whether every returned wire is a definite control
                rather than one member of a conservative candidate union.
        """
        control_wires: list[int] = []
        semantic_candidates: list[list[int]] = []
        semantic_candidate_precision: list[bool] = []
        semantic_mapping_complete = True
        all_control_wires_precise = True
        slice_context_value = cast(dict[object, object], param_values).get(
            _UNRESOLVED_SLICE_CONTEXT_KEY
        )
        slice_context = (
            slice_context_value if isinstance(slice_context_value, dict) else {}
        )

        for control_operand in op.control_operands:
            indices = self._resolve_operand_to_qubit_indices(
                control_operand,
                qubit_map,
                logical_id_remap,
                param_values,
            )
            if indices is None:
                semantic_mapping_complete = False
                all_control_wires_precise = False
                continue
            control_wires.extend(indices)

            if not isinstance(control_operand, ArrayValue):
                semantic_candidates.extend([[wire] for wire in indices])
                semantic_candidate_precision.extend([True] * len(indices))
                continue

            alias_source = slice_context.get(control_operand.logical_id)
            if not isinstance(alias_source, ArrayValue):
                alias_source = control_operand
            conservative_wires, exact_aliases = self._array_wire_aliases(
                alias_source,
                qubit_map,
                logical_id_remap,
                param_values,
            )
            semantic_size = self._resolve_declared_array_size(
                control_operand,
                param_values,
            )
            if semantic_size is None:
                semantic_mapping_complete = False
                all_control_wires_precise = False
                continue
            if conservative_wires:
                all_control_wires_precise = False
                for semantic_index in range(semantic_size):
                    exact_wire = exact_aliases.get(semantic_index)
                    if exact_wire is not None:
                        semantic_candidates.append([exact_wire])
                        semantic_candidate_precision.append(True)
                    else:
                        semantic_candidates.append(list(conservative_wires))
                        semantic_candidate_precision.append(False)
                continue
            if len(exact_aliases) == semantic_size:
                semantic_candidates.extend(
                    [[exact_aliases[index]] for index in range(semantic_size)]
                )
                semantic_candidate_precision.extend([True] * semantic_size)
                continue
            if len(indices) == semantic_size:
                semantic_candidates.extend([[wire] for wire in indices])
                semantic_candidate_precision.extend([True] * semantic_size)
                continue
            semantic_mapping_complete = False
            all_control_wires_precise = False

        selected_values = getattr(op, "control_indices", None)
        if selected_values is None or not control_wires:
            return control_wires, all_control_wires_precise
        if not semantic_mapping_complete:
            return control_wires, False

        selected_positions: list[int] = []
        for selected_value in selected_values:
            evaluated = self._evaluate_value(selected_value, param_values)
            if not isinstance(evaluated, (int, float)):
                return control_wires, False
            selected_positions.append(int(evaluated))
        if not all(
            0 <= position < len(semantic_candidates) for position in selected_positions
        ):
            return control_wires, False

        selected_wires: list[int] = []
        selected_wire_set: set[int] = set()
        for position in selected_positions:
            for wire in semantic_candidates[position]:
                if wire not in selected_wire_set:
                    selected_wire_set.add(wire)
                    selected_wires.append(wire)
        selected_wires_precise = all(
            semantic_candidate_precision[position] for position in selected_positions
        )
        return selected_wires, selected_wires_precise

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
        """Build a VGate node for an ExpvalOp.

        Args:
            op (ExpvalOp): Expectation-value operation to visualize.
            node_key (tuple): Stable identity key for layout and rendering.
            qubit_map (dict[str, int]): Mapping from logical IDs to wire indices.
            logical_id_remap (dict[str, str]): Mapping from block-local logical
                IDs to caller-visible logical IDs.
            param_values (dict): Parameter values available in the current
                visualization scope.

        Returns:
            VGate: Expectation-value gate node with resolved qubit indices.
        """
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
        analysis_path: tuple[int | str, ...],
    ) -> VInlineBlock | VGate:
        """Build an inline callable node or a recursion-safe summary box.

        Args:
            op (ControlledUOperation | InverseBlockOperation | InvokeOperation):
                Block-like operation whose implementation should be expanded.
            node_key (tuple): Stable identity key for layout and rendering.
            qubit_map (dict[str, int]): Mapping from logical IDs to wire indices.
            logical_id_remap (dict[str, str]): Mapping from block-local logical
                IDs to caller-visible logical IDs.
            param_values (dict): Parameter values available in the caller scope.
            depth (int): Current nested visualization depth.
            scope_path (tuple): Path of enclosing scope keys.
            analysis_path (tuple[int | str, ...]): Deterministic operation
                path of this callable site.

        Returns:
            VInlineBlock | VGate: Expanded callable geometry, or a summary
                gate when recursive expansion cannot prove finite progress.

        Raises:
            AssertionError: If the operation has no concrete implementation
                block despite having been selected for inline expansion.
            TypeError: If ``op`` is not a supported block-like operation.
        """
        # Extract block_value, affected_qubits, and actual_inputs based on op type
        if isinstance(op, ControlledUOperation):
            block_value = op.block
            assert isinstance(block_value, Block)
            control_qubit_indices = self._resolve_controlled_control_wires(
                op,
                qubit_map,
                logical_id_remap,
                param_values,
            )
            affected_qubits = list(control_qubit_indices)
            for operand in op.target_operands:
                indices = self._resolve_operand_to_qubit_indices(
                    operand, qubit_map, logical_id_remap, param_values
                )
                if indices is not None:
                    affected_qubits.extend(indices)
            u_name = getattr(block_value, "name", "U") or "U"
            block_name = u_name
        elif isinstance(op, InvokeOperation):
            block_value = op.effective_body()
            assert isinstance(block_value, Block)
            body_owns_controls = self._invoke_body_owns_controls(op, block_value)
            control_qubit_indices = []
            if not body_owns_controls:
                for operand in self._flatten_quantum_values(op.control_qubits):
                    indices = self._resolve_operand_to_qubit_indices(
                        operand, qubit_map, logical_id_remap, param_values
                    )
                    if indices is not None:
                        control_qubit_indices.extend(indices)
            affected_qubits = list(control_qubit_indices)
            body_operands = self._invoke_qubit_operands(op)
            if op.num_control_qubits and not body_owns_controls:
                body_operands = self._flatten_quantum_values(op.target_qubits)
            for operand in body_operands:
                indices = self._resolve_operand_to_qubit_indices(
                    operand, qubit_map, logical_id_remap, param_values
                )
                if indices is not None:
                    for index in indices:
                        if index not in affected_qubits:
                            affected_qubits.append(index)
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
            block_name = op.name
        else:
            raise TypeError(f"Unsupported inline block type: {type(op).__name__}")

        (
            block_value,
            new_logical_id_remap,
            child_param_values,
            may_expand,
        ) = self._prepare_callable_body(
            op,
            block_value,
            logical_id_remap,
            param_values,
            qubit_map,
            analysis_path,
        )
        if not may_expand:
            return self._build_vgate(
                op,
                node_key,
                qubit_map,
                logical_id_remap,
                param_values,
            )

        # Include qubits created inside the block body via QInitOperation
        for body_op in block_value.operations:
            if isinstance(body_op, QInitOperation):
                result_val = body_op.results[0]
                resolved = self._resolve_operand_to_qubit_indices(
                    result_val,
                    qubit_map,
                    new_logical_id_remap,
                    child_param_values,
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
            (),
        )
        affected_qubits = self._merge_visual_footprint(affected_qubits, children)

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
            control_qubit_indices
            if isinstance(op, (ControlledUOperation, InvokeOperation))
            else []
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

    def _initial_region_arg_state(
        self,
        op: ForOperation | ForItemsOperation,
        param_values: dict,
    ) -> dict[str, int | float]:
        """Resolve loop-carried scalar values entering iteration zero.

        Args:
            op (ForOperation | ForItemsOperation): Loop whose explicit region
                arguments should be initialized.
            param_values (dict): Values available in the enclosing scope.

        Returns:
            dict[str, int | float]: Concrete carried values keyed by each
                ``RegionArg.block_arg.logical_id``. Unresolved symbolic values
                are omitted and remain symbolic in the drawing.
        """
        state: dict[str, int | float] = {}
        for region_arg in op.region_args:
            resolved = self._evaluate_value(region_arg.init, param_values)
            if isinstance(resolved, (int, float)):
                state[region_arg.block_arg.logical_id] = resolved
        return state

    def _advance_region_arg_state(
        self,
        op: ForOperation | ForItemsOperation,
        iteration_values: dict,
        previous_state: dict[str, int | float],
    ) -> dict[str, int | float]:
        """Resolve loop-carried values yielded by one materialized iteration.

        Args:
            op (ForOperation | ForItemsOperation): Loop whose body just ran.
            iteration_values (dict): Parameter values used for that iteration,
                including evaluated body intermediates.
            previous_state (dict[str, int | float]): State entering the
                iteration, used when a yielded value remains unchanged.

        Returns:
            dict[str, int | float]: State to bind to the next iteration.
        """
        next_state: dict[str, int | float] = {}
        for region_arg in op.region_args:
            resolved = self._evaluate_value(
                region_arg.yielded,
                iteration_values,
                op.operations,
            )
            if isinstance(resolved, (int, float)):
                next_state[region_arg.block_arg.logical_id] = resolved
                continue
            if region_arg.yielded.uuid == region_arg.block_arg.uuid:
                previous = previous_state.get(region_arg.block_arg.logical_id)
                if previous is not None:
                    next_state[region_arg.block_arg.logical_id] = previous
        return next_state

    @staticmethod
    def _publish_region_arg_results(
        op: ForOperation | ForItemsOperation,
        state: dict[str, int | float],
        param_values: dict,
    ) -> None:
        """Publish final carried values under their post-loop result IDs.

        Args:
            op (ForOperation | ForItemsOperation): Completed loop operation.
            state (dict[str, int | float]): State after its final iteration.
            param_values (dict): Enclosing values mapping to update in place.
        """
        for region_arg in op.region_args:
            resolved = state.get(region_arg.block_arg.logical_id)
            if resolved is not None:
                param_values[region_arg.result.logical_id] = resolved

    def _simulate_for_region_args(
        self,
        op: ForOperation,
        iteration_values: Sequence[int],
        param_values: dict,
    ) -> None:
        """Evaluate and publish static ``for`` loop region arguments.

        Args:
            op (ForOperation): Loop whose carried scalars should be evaluated.
            iteration_values (Sequence[int]): Concrete loop-variable values.
            param_values (dict): Enclosing values mapping to read and update.
        """
        state = self._initial_region_arg_state(op, param_values)
        for iteration_value in iteration_values:
            child_values = dict(param_values)
            child_values.update(state)
            child_values[f"_loop_{op.loop_var}"] = iteration_value
            self._evaluate_loop_body_intermediates(op.operations, child_values)
            state = self._advance_region_arg_state(op, child_values, state)
        self._publish_region_arg_results(op, state, param_values)

    @staticmethod
    def _bind_for_items_iteration(
        op: ForItemsOperation,
        entry_key: object,
        entry_value: object,
        param_values: dict,
    ) -> None:
        """Bind one materialized ``items`` entry for analyzer evaluation.

        Args:
            op (ForItemsOperation): Loop declaring the key/value variables.
            entry_key (object): Materialized key or tuple-like key value.
            entry_value (object): Materialized dictionary value.
            param_values (dict): Iteration-local mapping to update in place.
        """
        if hasattr(entry_key, "elements"):
            for key_var, element in zip(op.key_vars, entry_key.elements):
                value = element.get_const() if hasattr(element, "get_const") else None
                if value is not None:
                    param_values[f"_loop_{key_var}"] = value
        elif isinstance(entry_key, tuple):
            for key_var, element in zip(op.key_vars, entry_key):
                param_values[f"_loop_{key_var}"] = element
        elif op.key_vars:
            value = (
                entry_key.get_const() if hasattr(entry_key, "get_const") else entry_key
            )
            if value is not None:
                param_values[f"_loop_{op.key_vars[0]}"] = value

        value = (
            entry_value.get_const()
            if hasattr(entry_value, "get_const")
            else entry_value
        )
        if value is not None:
            param_values[f"_loop_{op.value_var}"] = value

    def _simulate_for_items_region_args(
        self,
        op: ForItemsOperation,
        entries: Sequence[tuple[object, object]],
        param_values: dict,
    ) -> None:
        """Evaluate and publish static ``for-items`` region arguments.

        Args:
            op (ForItemsOperation): Loop whose carried scalars should be
                evaluated.
            entries (Sequence[tuple[object, object]]): Materialized key/value
                entries in iteration order.
            param_values (dict): Enclosing values mapping to read and update.
        """
        state = self._initial_region_arg_state(op, param_values)
        for entry_key, entry_value in entries:
            child_values = dict(param_values)
            child_values.update(state)
            self._bind_for_items_iteration(
                op,
                entry_key,
                entry_value,
                child_values,
            )
            self._evaluate_loop_body_intermediates(op.operations, child_values)
            state = self._advance_region_arg_state(op, child_values, state)
        self._publish_region_arg_results(op, state, param_values)

    def _build_for_iteration_nodes(
        self,
        op: ForOperation,
        iteration_values: Sequence[int],
        node_key: tuple,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict,
        depth: int,
        analysis_path: tuple[int | str, ...],
    ) -> list[list[VisualNode]]:
        """Build visual children for concrete iterations of one for loop.

        Args:
            op (ForOperation): Loop whose body nodes should be materialized.
            iteration_values (Sequence[int]): Concrete loop-variable values in
                execution order.
            node_key (tuple): Parent visual identity used for child keys.
            qubit_map (dict[str, int]): Logical-ID-to-wire mapping.
            logical_id_remap (dict[str, str]): Callable-local value remapping.
            param_values (dict): Enclosing parameter values, updated with final
                region-argument results.
            depth (int): Current visual nesting depth.
            analysis_path (tuple[int | str, ...]): Deterministic path of the
                enclosing loop.

        Returns:
            list[list[VisualNode]]: Visual children grouped by iteration.
        """
        iterations: list[list[VisualNode]] = []
        region_state = self._initial_region_arg_state(op, param_values)
        for iteration, iter_value in enumerate(iteration_values):
            child_param_values = dict(param_values)
            child_param_values.update(region_state)
            child_param_values[f"_loop_{op.loop_var}"] = iter_value
            self._evaluate_loop_body_intermediates(
                op.operations,
                child_param_values,
            )
            iterations.append(
                self._build_visual_nodes(
                    op.operations,
                    qubit_map,
                    logical_id_remap,
                    child_param_values,
                    depth + 1,
                    (*node_key, iteration),
                    (*analysis_path, "body"),
                )
            )
            region_state = self._advance_region_arg_state(
                op,
                child_param_values,
                region_state,
            )
        self._publish_region_arg_results(op, region_state, param_values)
        return iterations

    def _build_for_items_iteration_nodes(
        self,
        op: ForItemsOperation,
        entries: Sequence[tuple[object, object]],
        node_key: tuple,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict,
        depth: int,
        analysis_path: tuple[int | str, ...],
    ) -> list[list[VisualNode]]:
        """Build visual children for materialized dictionary iterations.

        Args:
            op (ForItemsOperation): Dictionary loop whose body to materialize.
            entries (Sequence[tuple[object, object]]): Concrete key/value pairs
                in iteration order.
            node_key (tuple): Parent visual identity used for child keys.
            qubit_map (dict[str, int]): Logical-ID-to-wire mapping.
            logical_id_remap (dict[str, str]): Callable-local value remapping.
            param_values (dict): Enclosing parameter values, updated with final
                region-argument results.
            depth (int): Current visual nesting depth.
            analysis_path (tuple[int | str, ...]): Deterministic path of the
                enclosing loop.

        Returns:
            list[list[VisualNode]]: Visual children grouped by dictionary item.
        """
        iterations: list[list[VisualNode]] = []
        region_state = self._initial_region_arg_state(op, param_values)
        for entry_key, entry_value in entries:
            child_param_values = dict(param_values)
            child_param_values.update(region_state)
            self._bind_for_items_iteration(
                op,
                entry_key,
                entry_value,
                child_param_values,
            )
            self._evaluate_loop_body_intermediates(
                op.operations,
                child_param_values,
            )
            iterations.append(
                self._build_visual_nodes(
                    op.operations,
                    qubit_map,
                    logical_id_remap,
                    child_param_values,
                    depth + 1,
                    (*node_key, len(iterations)),
                    (*analysis_path, "body"),
                )
            )
            region_state = self._advance_region_arg_state(
                op,
                child_param_values,
                region_state,
            )
        self._publish_region_arg_results(op, region_state, param_values)
        return iterations

    def _build_vfor(
        self,
        op: ForOperation,
        node_key: tuple,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict,
        depth: int,
        scope_path: tuple,
        analysis_path: tuple[int | str, ...],
    ) -> VFoldedBlock | VUnfoldedSequence | VSkip:
        """Build a Visual IR node for a ForOperation.

        Args:
            op (ForOperation): Loop operation to visualize.
            node_key (tuple): Stable identity key for layout and rendering.
            qubit_map (dict[str, int]): Mapping from logical IDs to wire indices.
            logical_id_remap (dict[str, str]): Mapping from block-local logical
                IDs to caller-visible logical IDs.
            param_values (dict): Parameter values available in the current
                visualization scope.
            depth (int): Current nested visualization depth.
            scope_path (tuple): Path of enclosing scope keys.
            analysis_path (tuple[int | str, ...]): Deterministic operation
                path of this loop.

        Returns:
            VFoldedBlock | VUnfoldedSequence | VSkip: Folded loop summary,
                materialized iteration sequence, or an empty-loop marker.
        """
        start_val, stop_val_raw, step_val = self._evaluate_loop_range(op, param_values)

        if self._is_zero_iteration_loop(start_val, stop_val_raw, step_val):
            state = self._initial_region_arg_state(op, param_values)
            self._publish_region_arg_results(op, state, param_values)
            return VSkip(node_key=node_key)

        affected_qubits, affected_qubits_precise = self._analyze_loop_affected_qubits(
            op, qubit_map, logical_id_remap, param_values
        )
        affected_qubits = list(
            dict.fromkeys(
                [
                    *affected_qubits,
                    *self._collect_scoped_callable_qubits(
                        op.operations,
                        logical_id_remap,
                        (*analysis_path, "body"),
                    ),
                ]
            )
        )

        # Folded mode: fold_loops=True or symbolic stop
        if self.fold_loops or stop_val_raw is None:
            header, body_lines, folded_width = self._compute_folded_for_info(
                op, param_values
            )
            if stop_val_raw is not None and step_val != 0:
                self._simulate_for_region_args(
                    op,
                    range(start_val, stop_val_raw, step_val),
                    param_values,
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

        iterations = self._build_for_iteration_nodes(
            op,
            range(start_val, stop_val_raw, step_val),
            node_key,
            qubit_map,
            logical_id_remap,
            param_values,
            depth,
            analysis_path,
        )
        iteration_widths = [
            self._sum_visual_widths(children) for children in iterations
        ]

        affected_qubits = self._merge_visual_footprint(
            affected_qubits,
            [child for iteration in iterations for child in iteration],
        )

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
        depth: int,
        scope_path: tuple,
        analysis_path: tuple[int | str, ...] = (),
        body_operations: list[Operation] | None = None,
    ) -> VFoldedBlock | VUnfoldedSequence:
        """Build a Visual IR node for a WhileOperation.

        A ``while`` loop is measurement-backed (its condition is a ``Bit``
        produced by ``qmc.measure``) and cannot be unrolled at compile time,
        so it is drawn like a single-branch ``if``: the loop body is shown
        once inside a ``while <cond>:`` box, connected to the measurement that
        produces the condition. This is the default. When ``fold_whiles`` is
        enabled the body collapses to a compact summary box instead, matching
        the folded ``for`` rendering.

        Args:
            op (WhileOperation): The while loop to visualize.
            node_key (tuple): Stable identity key for layout/render lookup.
            qubit_map (dict[str, int]): Mapping from logical_id to wire index.
            logical_id_remap (dict[str, str]): Mapping from formal-parameter
                logical_ids to actual-argument logical_ids in scope.
            param_values (dict): Resolved loop/parameter values in scope.
            depth (int): Current nesting depth for child expansion.
            scope_path (tuple): Path of enclosing scope keys for child node keys.
            analysis_path (tuple[int | str, ...]): Deterministic operation
                path of this loop. Defaults to the current root.
            body_operations (list[Operation] | None): Operations of the scope
                enclosing ``op``, used to spell the condition predicate and
                locate the measurement that feeds it. Defaults to None, which
                falls back to the anonymous ``while cond:`` label with no
                measurement connector.

        Returns:
            VFoldedBlock | VUnfoldedSequence: A folded summary box when
                ``fold_whiles`` is set, otherwise an unfolded sequence carrying
                the loop body as its single iteration.
        """
        affected_qubits, affected_qubits_precise = self._analyze_loop_affected_qubits(
            op, qubit_map, logical_id_remap, param_values
        )
        affected_qubits = list(
            dict.fromkeys(
                [
                    *affected_qubits,
                    *self._collect_scoped_callable_qubits(
                        op.operations,
                        logical_id_remap,
                        (*analysis_path, "body"),
                    ),
                ]
            )
        )

        condition = op.operands[0] if op.operands else None
        condition_expr = self._format_condition_expr(
            condition, body_operations, param_values
        )
        condition_label = (
            f"while {condition_expr}:" if condition_expr else "while cond:"
        )
        condition_measure_node_key: tuple | None = None
        condition_measure_qubit_indices: list[int] = []
        condition_measure_info = self._condition_measure_info(
            condition,
            body_operations,
            qubit_map,
            logical_id_remap,
            param_values,
            scope_path,
        )
        if condition_measure_info is not None:
            condition_measure_node_key, condition_measure_qubit_indices = (
                condition_measure_info
            )
        if not affected_qubits and condition_measure_qubit_indices:
            affected_qubits = list(dict.fromkeys(condition_measure_qubit_indices))
            affected_qubits_precise = len(condition_measure_qubit_indices) == 1

        body_param_values = dict(param_values)
        self._evaluate_loop_body_intermediates(op.operations, body_param_values)

        if self.fold_whiles:
            if not affected_qubits and qubit_map:
                # An empty-bodied while still needs a display wire for its box.
                affected_qubits = [min(qubit_map.values())]
                affected_qubits_precise = False
            body_lines: list[str] = []
            for body_op in op.operations:
                expr = self._format_operation_as_expression(
                    body_op,
                    set(),
                    body_operations=op.operations,
                    param_values=body_param_values,
                )
                if expr:
                    body_lines.extend(expr.split("\n"))
            if len(body_lines) > 3:
                body_lines = body_lines[:3] + ["..."]

            folded_width = self._compute_folded_text_width(condition_label, body_lines)

            return VFoldedBlock(
                node_key=node_key,
                header_label=condition_label,
                body_lines=body_lines,
                affected_qubits=affected_qubits,
                folded_width=folded_width,
                kind=VFoldedKind.WHILE,
                affected_qubits_precise=affected_qubits_precise,
                condition_measure_node_key=condition_measure_node_key,
                condition_measure_qubit_indices=condition_measure_qubit_indices,
            )

        body_children = self._build_visual_nodes(
            op.operations,
            qubit_map,
            logical_id_remap,
            body_param_values,
            depth + 1,
            (*node_key, "body"),
            (*analysis_path, "body"),
        )
        affected_qubits = self._merge_visual_footprint(
            affected_qubits,
            body_children,
        )

        if not affected_qubits and qubit_map:
            affected_qubits = [min(qubit_map.values())]
            affected_qubits_precise = False
        body_width = self._sum_visual_widths(body_children)
        branch_label_widths = [self._estimate_label_box_width(condition_label)]

        return VUnfoldedSequence(
            node_key=node_key,
            iterations=[body_children],
            affected_qubits=affected_qubits,
            kind=VUnfoldedKind.WHILE,
            iteration_widths=[body_width],
            condition_label=condition_label,
            affected_qubits_precise=affected_qubits_precise,
            condition_label_width=branch_label_widths[0],
            branch_label_widths=branch_label_widths,
            condition_measure_node_key=condition_measure_node_key,
            condition_measure_qubit_indices=condition_measure_qubit_indices,
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
        analysis_path: tuple[int | str, ...] = (),
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
            analysis_path (tuple[int | str, ...]): Deterministic operation
                path of this conditional. Defaults to the current root.
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
        affected_qubits = list(
            dict.fromkeys(
                [
                    *affected_qubits,
                    *self._collect_scoped_callable_qubits(
                        op.true_operations,
                        logical_id_remap,
                        (*analysis_path, "true"),
                    ),
                    *self._collect_scoped_callable_qubits(
                        op.false_operations,
                        logical_id_remap,
                        (*analysis_path, "false"),
                    ),
                ]
            )
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
            affected_qubits_precise = len(condition_measure_qubit_indices) == 1

        if self.fold_ifs:
            if not affected_qubits and qubit_map:
                # Empty symbolic IFs still need a display wire for their branch box.
                affected_qubits = [min(qubit_map.values())]
                affected_qubits_precise = False
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

        true_param_values = dict(param_values)
        self._evaluate_loop_body_intermediates(op.true_operations, true_param_values)
        true_children = self._build_visual_nodes(
            op.true_operations,
            qubit_map,
            logical_id_remap,
            true_param_values,
            depth + 1,
            (*node_key, "true"),
            (*analysis_path, "true"),
        )
        false_param_values = dict(param_values)
        self._evaluate_loop_body_intermediates(op.false_operations, false_param_values)
        false_children = self._build_visual_nodes(
            op.false_operations,
            qubit_map,
            logical_id_remap,
            false_param_values,
            depth + 1,
            (*node_key, "false"),
            (*analysis_path, "false"),
        )
        affected_qubits = self._merge_visual_footprint(
            affected_qubits,
            [*true_children, *false_children],
        )

        true_width = self._sum_visual_widths(true_children)
        false_width = self._sum_visual_widths(false_children)

        iterations = [true_children]
        iteration_widths = [true_width]
        branch_labels = [condition_label]
        if false_children:
            iterations.append(false_children)
            iteration_widths.append(false_width)
            branch_labels.append("else:")

        if not affected_qubits and qubit_map:
            affected_qubits = [min(qubit_map.values())]
            affected_qubits_precise = False

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
            wires = self._measure_vector_condition_wires(
                value, op, qubit_map, logical_id_remap, param_values
            )
            if wires is not None:
                return wires
        return self._measure_qubit_indices(
            op, qubit_map, logical_id_remap, param_values
        )

    def _measure_vector_condition_wires(
        self,
        value: ValueBase,
        op: MeasureVectorOperation,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict,
    ) -> list[int] | None:
        """Resolve ``measure(qs)[i]`` to exact or candidate source wires.

        Args:
            value (ValueBase): IF condition value, possibly an element of the
                vector measurement result.
            op (MeasureVectorOperation): Vector measurement producer.
            qubit_map (dict[str, int]): Mapping from logical_id to wire index.
            logical_id_remap (dict[str, str]): Mapping from formal-parameter
                logical_ids to actual-argument logical_ids in scope.
            param_values (dict): Resolved loop/parameter values in scope.

        Returns:
            list[int] | None: The exact measured wire, or every conservative
                candidate when semantic slot ``i`` is unresolved. None when
                the condition is not a resolvable vector element.
        """
        if not op.operands:
            return None
        index = self._condition_array_element_index(value, param_values)
        if index is None:
            return None
        return self._resolve_array_operand_index_to_qubits(
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

    def _resolve_array_operand_index_to_qubits(
        self,
        operand: Value,
        index: int,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict,
    ) -> list[int] | None:
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
            list[int] | None: Exact or conservative qubit wires corresponding
                to ``operand[index]``, or None when the array cannot resolve.
        """
        if not isinstance(operand, ArrayValue):
            return None
        conservative_wires, exact_aliases = self._array_wire_aliases(
            operand,
            qubit_map,
            logical_id_remap,
            param_values,
        )
        if index in exact_aliases:
            return [exact_aliases[index]]
        if conservative_wires:
            return conservative_wires

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
                return [qubit_map[root_element_key]]
            if root_lid in qubit_map:
                return [qubit_map[root_lid] + root_index]
            return None

        element_key = f"{resolved_lid}_[{index}]"
        if element_key in qubit_map:
            return [qubit_map[element_key]]
        if resolved_lid in qubit_map:
            return [qubit_map[resolved_lid] + index]
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
        analysis_path: tuple[int | str, ...],
    ) -> VFoldedBlock | VUnfoldedSequence | VSkip:
        """Build a Visual IR node for a ForItemsOperation.

        Args:
            op (ForItemsOperation): Dictionary-items loop to visualize.
            node_key (tuple): Stable identity key for layout and rendering.
            qubit_map (dict[str, int]): Mapping from logical IDs to wire indices.
            logical_id_remap (dict[str, str]): Mapping from block-local logical
                IDs to caller-visible logical IDs.
            param_values (dict): Parameter values available in the current
                visualization scope.
            depth (int): Current nested visualization depth.
            scope_path (tuple): Path of enclosing scope keys.
            analysis_path (tuple[int | str, ...]): Deterministic operation
                path of this loop.

        Returns:
            VFoldedBlock | VUnfoldedSequence | VSkip: Folded loop summary,
                materialized item sequence, or an empty-loop marker.
        """
        affected_qubits, affected_qubits_precise = self._analyze_loop_affected_qubits(
            op, qubit_map, logical_id_remap, param_values
        )
        affected_qubits = list(
            dict.fromkeys(
                [
                    *affected_qubits,
                    *self._collect_scoped_callable_qubits(
                        op.operations,
                        logical_id_remap,
                        (*analysis_path, "body"),
                    ),
                ]
            )
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
        if materialized == []:
            state = self._initial_region_arg_state(op, param_values)
            self._publish_region_arg_results(op, state, param_values)
            return VSkip(node_key=node_key)

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
            if materialized is not None:
                self._simulate_for_items_region_args(
                    op,
                    materialized,
                    param_values,
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

        iterations = self._build_for_items_iteration_nodes(
            op,
            entries,
            node_key,
            qubit_map,
            logical_id_remap,
            param_values,
            depth,
            analysis_path,
        )
        iteration_widths = [
            self._sum_visual_widths(children) for children in iterations
        ]

        affected_qubits = self._merge_visual_footprint(
            affected_qubits,
            [child for iteration in iterations for child in iteration],
        )

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
        """Compute header, body lines, and folded width for a ForOperation.

        Args:
            op (ForOperation): Loop whose folded presentation to compute.
            param_values (dict): Parameter values used to resolve loop bounds
                and body expressions.

        Returns:
            tuple[str, list[str], float]: Header text, formatted body lines,
                and required folded-box width.
        """
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
        """Compute the width needed for a folded block's text content.

        Args:
            header (str): Bold header drawn at the top of the folded block.
            body_lines (list[str]): Body lines drawn below or with the header.

        Returns:
            float: Minimum folded-box width in layout units.
        """
        header_width = measure_text_width(
            header,
            font_size=self.style.subfont_size,
            font_weight="bold",
            fallback_char_width=self.style.char_width_bold,
        )
        body_width = max(
            (
                max(
                    measure_text_width(
                        line,
                        font_size=self.style.subfont_size,
                        fallback_char_width=self.style.char_width_gate,
                    ),
                    measure_text_width(
                        line,
                        font_size=self.style.subfont_size,
                        font_family="monospace",
                        fallback_char_width=self.style.char_width_monospace,
                    ),
                )
                for line in body_lines
            ),
            default=0.0,
        )
        text_width = max(header_width, body_width)
        margin = 0.15  # Extra margin for folded blocks
        return max(
            self.style.folded_loop_width,
            text_width + 2 * self.style.text_padding + margin,
        )

    def _sum_visual_widths(self, children: list[VisualNode]) -> float:
        """Sum widths of VisualNode children with inter-element gaps.

        Args:
            children (list[VisualNode]): Visual nodes whose widths to combine.

        Returns:
            float: Total horizontal content width in layout units.
        """
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
        """Find the maximum VGate.estimated_width in the VisualNode tree.

        Args:
            children (list[VisualNode]): Visual-node subtree to inspect.

        Returns:
            float: Largest gate width in the subtree, or zero when absent.
        """
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

    @staticmethod
    def _visual_node_qubits(node: VisualNode) -> list[int]:
        """Return the wire footprint advertised by one Visual IR node.

        Args:
            node (VisualNode): Visual node whose physical wires to inspect.

        Returns:
            list[int]: Gate wires or the affected-wire footprint of a nested
                block/control-flow node. VSkip nodes return an empty list.
        """
        if isinstance(node, VGate):
            return list(node.qubit_indices)
        if isinstance(node, (VInlineBlock, VFoldedBlock, VUnfoldedSequence)):
            return list(node.affected_qubits)
        return []

    @classmethod
    def _merge_visual_footprint(
        cls,
        affected_qubits: Sequence[int],
        children: Sequence[VisualNode],
    ) -> list[int]:
        """Merge child-node wires into an enclosing visual footprint.

        Args:
            affected_qubits (Sequence[int]): Wires found by operand analysis.
            children (Sequence[VisualNode]): Materialized child nodes whose
                advertised footprints may include callable-local allocations.

        Returns:
            list[int]: Unique wire indices in first-encounter order.
        """
        merged = list(dict.fromkeys(affected_qubits))
        for child in children:
            for index in cls._visual_node_qubits(child):
                if index not in merged:
                    merged.append(index)
        return merged

    @staticmethod
    def _build_scoped_callable_wire_index(
        qubit_map: dict[str, int],
    ) -> dict[str, list[int]]:
        """Index callable-local wires by every enclosing call scope.

        Args:
            qubit_map (dict[str, int]): Logical-ID-to-wire mapping containing
                scoped callable-local keys.

        Returns:
            dict[str, list[int]]: Scope keys mapped to unique descendant wires
                in qubit-map encounter order.
        """
        indexed_wires: dict[str, dict[int, None]] = {}
        for logical_id, wire in qubit_map.items():
            _, separator, scoped_suffix = logical_id.partition("@")
            if not separator:
                continue
            registered_scope = re.sub(
                r"_\[(?:\?\d+|\d+)\]$",
                "",
                scoped_suffix,
            )
            scope_prefixes = [registered_scope]
            slash_index = registered_scope.find("/")
            while slash_index >= 0:
                scope_prefixes.append(registered_scope[:slash_index])
                slash_index = registered_scope.find("/", slash_index + 1)
            for scope in scope_prefixes:
                indexed_wires.setdefault(scope, {})[wire] = None
        return {scope: list(wires) for scope, wires in indexed_wires.items()}

    def _collect_scoped_callable_qubits(
        self,
        operations: Sequence[Operation],
        logical_id_remap: dict[str, str],
        analysis_path: tuple[int | str, ...],
    ) -> list[int]:
        """Collect registered callable-local wires from an operation tree.

        Args:
            operations (Sequence[Operation]): Operations to inspect recursively.
            logical_id_remap (dict[str, str]): Enclosing logical-ID remapping.
            analysis_path (tuple[int | str, ...]): Deterministic path of the
                enclosing operation list.

        Returns:
            list[int]: Unique callable-local wires in first-encounter order.
        """
        call_scopes: list[str] = []
        seen_scopes: set[str] = set()

        def collect_scopes(
            nested_operations: Sequence[Operation],
            nested_path: tuple[int | str, ...],
        ) -> None:
            """Collect deterministic scopes from one nested operation list.

            Args:
                nested_operations (Sequence[Operation]): Operations to inspect.
                nested_path (tuple[int | str, ...]): Path of the operation
                    list within the surrounding control-flow tree.
            """
            for operation_index, operation in enumerate(nested_operations):
                operation_path = (*nested_path, operation_index)
                if isinstance(
                    operation,
                    (ControlledUOperation, InverseBlockOperation, InvokeOperation),
                ):
                    call_scope = self._nested_call_scope(
                        logical_id_remap,
                        operation,
                        operation_path,
                    )
                    if call_scope not in seen_scopes:
                        seen_scopes.add(call_scope)
                        call_scopes.append(call_scope)
                if isinstance(operation, IfOperation):
                    collect_scopes(
                        operation.true_operations,
                        (*operation_path, "true"),
                    )
                    collect_scopes(
                        operation.false_operations,
                        (*operation_path, "false"),
                    )
                elif isinstance(
                    operation,
                    (ForOperation, WhileOperation, ForItemsOperation),
                ):
                    collect_scopes(
                        operation.operations,
                        (*operation_path, "body"),
                    )

        collect_scopes(operations, analysis_path)
        wires: list[int] = []
        seen_wires: set[int] = set()
        for call_scope in call_scopes:
            for wire in self._scoped_callable_wire_index.get(call_scope, []):
                if wire not in seen_wires:
                    seen_wires.add(wire)
                    wires.append(wire)
        return wires

    def _compute_visual_content_width(
        self,
        children: list[VisualNode],
        max_gate_width: float,
        depth: int,
    ) -> float:
        """Compute the total content width of VisualNode children inside a block.

        Args:
            children (list[VisualNode]): Visual nodes contained by the block.
            max_gate_width (float): Largest gate width in the child subtree.
            depth (int): Nested block depth used to determine border padding.

        Returns:
            float: Total block content width including border extents.
        """
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
            start_val (int): Loop start value.
            stop_val_raw (int | None): Loop stop value, or None if symbolic.
            step_val (int): Loop step value.

        Returns:
            bool: True if the loop would produce zero iterations.
        """
        if stop_val_raw is None:
            return False
        if step_val > 0 and start_val >= stop_val_raw:
            return True
        if step_val < 0 and start_val <= stop_val_raw:
            return True
        return False

    def _qubit_bearing_operands(self, op: Operation) -> list[Value] | None:
        """Return the values of `op` that may reference affected qubits.

        Used by the loop- and if-affect analyzers to pick out operands or
        callable results to feed into ``_resolve_operand_to_affected_qubits``.
        Returns
        None for control-flow operations, which the caller must handle
        via recursion rather than operand resolution.

        Args:
            op (Operation): IR operation to inspect.

        Returns:
            list[Value] | None: Operand/result values to resolve, or None when
                the op is a control-flow construct (For/While/If/ForItems)
                that the caller handles separately.
        """
        if isinstance(op, QInitOperation):
            return self._flatten_quantum_values(op.results)
        if isinstance(op, GateOperation):
            return list(op.operands)
        if isinstance(op, CastOperation):
            return [operand for operand in op.operands if operand.type.is_quantum()]
        if isinstance(op, InvokeOperation):
            return self._invoke_qubit_operands(op) + self._invoke_qubit_results(op)
        if isinstance(op, InverseBlockOperation):
            return list(op.control_qubits) + list(op.target_qubits)
        if isinstance(op, (MeasureOperation, MeasureVectorOperation)):
            return list(op.operands[:1])
        if isinstance(op, ExpvalOp):
            return [op.qubits]
        return None

    def _operand_wire_resolution_is_precise(
        self,
        operand: Value,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict,
    ) -> bool:
        """Return whether an operand maps to exact physical wires.

        Args:
            operand (Value): Quantum scalar, element, or array operand.
            qubit_map (dict[str, int]): Logical-ID-to-wire registry.
            logical_id_remap (dict[str, str]): Active callable remapping.
            param_values (dict): Bindings used to resolve indices and views.

        Returns:
            bool: False when the resolved footprint contains conservative
                candidates for an unknown semantic position.
        """
        if isinstance(operand, ArrayValue):
            conservative_wires, _ = self._array_wire_aliases(
                operand,
                qubit_map,
                logical_id_remap,
                param_values,
            )
            return not conservative_wires

        parent = operand.parent_array
        if not isinstance(parent, ArrayValue) or not operand.element_indices:
            return True
        index_value = operand.element_indices[0]
        if index_value.is_constant():
            index = index_value.get_const()
        else:
            index = self._evaluate_value(index_value, param_values)
        if index is None:
            return False
        conservative_wires, exact_aliases = self._array_wire_aliases(
            parent,
            qubit_map,
            logical_id_remap,
            param_values,
        )
        return not conservative_wires or int(index) in exact_aliases

    def _resolve_operation_affected_qubits(
        self,
        op: Operation,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict,
    ) -> tuple[list[int], bool]:
        """Resolve one non-control-flow operation's complete wire footprint.

        Operations with nonstandard quantum provenance are handled here rather
        than forced through their raw operands. Controlled operations use only
        selected controls, QFixed measurement follows cast carrier metadata,
        and ordinary operations share affected-operand resolution.

        Args:
            op (Operation): Non-control-flow operation to inspect.
            qubit_map (dict[str, int]): Logical-ID-to-wire registry.
            logical_id_remap (dict[str, str]): Active callable remapping.
            param_values (dict): Bindings used for indices and control
                selections.

        Returns:
            tuple[list[int], bool]: Unique affected wires and whether every
                quantum operand resolved precisely.
        """
        if isinstance(op, ControlledUOperation):
            affected, all_resolved = self._resolve_controlled_control_footprint(
                op,
                qubit_map,
                logical_id_remap,
                param_values,
            )
            for operand in op.target_operands:
                if not operand.type.is_quantum():
                    continue
                indices = self._resolve_operand_to_affected_qubits(
                    operand,
                    qubit_map,
                    logical_id_remap,
                    param_values,
                )
                if indices is None:
                    all_resolved = False
                    continue
                affected.extend(indices)
                if not self._operand_wire_resolution_is_precise(
                    operand,
                    qubit_map,
                    logical_id_remap,
                    param_values,
                ):
                    all_resolved = False
            return list(dict.fromkeys(affected)), all_resolved

        if isinstance(op, MeasureQFixedOperation):
            if not op.operands:
                return [], False
            carriers = self._resolve_qfixed_carrier_indices(
                op.operands[0],
                qubit_map,
                logical_id_remap,
            )
            return carriers, len(carriers) == op.num_bits

        operands = self._qubit_bearing_operands(op)
        if operands is None:
            if isinstance(op, ReturnOperation):
                return [], True
            quantum_values = self._flatten_quantum_values([*op.operands, *op.results])
            return [], not quantum_values

        affected: list[int] = []
        all_resolved = True
        for operand in operands:
            indices = self._resolve_operand_to_affected_qubits(
                operand,
                qubit_map,
                logical_id_remap,
                param_values,
            )
            if indices is None:
                if operand.type.is_quantum():
                    all_resolved = False
                continue
            affected.extend(indices)
            if operand.type.is_quantum() and not (
                self._operand_wire_resolution_is_precise(
                    operand,
                    qubit_map,
                    logical_id_remap,
                    param_values,
                )
            ):
                all_resolved = False
        return list(dict.fromkeys(affected)), all_resolved

    def _analyze_loop_affected_qubits(
        self,
        op: ForOperation | WhileOperation | ForItemsOperation,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str] | None = None,
        param_values: dict | None = None,
    ) -> tuple[list[int], bool]:
        """Analyze which qubits are affected by a loop operation.

        Args:
            op (ForOperation | WhileOperation | ForItemsOperation): Loop
                operation to analyze.
            qubit_map (dict[str, int]): Mapping from logical_id to qubit index.
            logical_id_remap (dict[str, str] | None): Mapping from dummy
                logical_ids to actual logical_ids. Defaults to None.
            param_values (dict | None): Parameter values for resolving loop
                ranges and indices. Defaults to None.

        Returns:
            tuple[list[int], bool]: Pair ``(indices, is_precise)`` where
                ``indices`` is the list
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
            if self._is_zero_iteration_loop(start_val, stop_val_raw, step_val):
                return [], True
            if stop_val_raw is not None and step_val != 0:
                num_iters = len(range(int(start_val), int(stop_val_raw), int(step_val)))
                if 0 < num_iters <= 100:
                    precise_affected: set[int] = set()
                    all_resolved = True
                    region_state = self._initial_region_arg_state(op, param_values)
                    for iter_val in range(
                        int(start_val), int(stop_val_raw), int(step_val)
                    ):
                        iter_params = dict(param_values)
                        iter_params.update(region_state)
                        iter_params[f"_loop_{op.loop_var}"] = iter_val
                        self._evaluate_loop_body_intermediates(
                            op.operations, iter_params
                        )
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
                            indices, operation_resolved = (
                                self._resolve_operation_affected_qubits(
                                    inner_op,
                                    qubit_map,
                                    logical_id_remap,
                                    iter_params,
                                )
                            )
                            precise_affected.update(indices)
                            if not operation_resolved:
                                all_resolved = False
                        region_state = self._advance_region_arg_state(
                            op,
                            iter_params,
                            region_state,
                        )
                    if all_resolved and precise_affected:
                        return list(precise_affected), True

        # Fallback to conservative analysis
        affected: set[int] = set()
        fallback_precise = True

        def collect_from_ops(ops: list[Operation]) -> None:
            """Recursively collect qubit indices from a list of operations.

            Args:
                ops (list[Operation]): Operations whose qubit operands to
                    collect into ``affected``.
            """
            nonlocal fallback_precise
            for inner_op in ops:
                if isinstance(
                    inner_op,
                    (ForOperation, WhileOperation, ForItemsOperation),
                ):
                    nested, nested_precise = self._analyze_loop_affected_qubits(
                        inner_op,
                        qubit_map,
                        logical_id_remap,
                        param_values,
                    )
                    affected.update(nested)
                    if not nested_precise:
                        fallback_precise = False
                    continue
                if isinstance(inner_op, IfOperation):
                    nested, nested_precise = self._collect_if_affected_qubits(
                        inner_op,
                        qubit_map,
                        logical_id_remap,
                        param_values,
                    )
                    affected.update(nested)
                    if not nested_precise:
                        fallback_precise = False
                    continue
                indices, operation_precise = self._resolve_operation_affected_qubits(
                    inner_op,
                    qubit_map,
                    logical_id_remap,
                    param_values,
                )
                affected.update(indices)
                if not operation_precise:
                    fallback_precise = False

        collect_from_ops(op.operations)
        return list(affected), fallback_precise

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
            op (IfOperation): Conditional operation to analyze.
            qubit_map (dict[str, int]): Mapping from logical_id to qubit wire
                index.
            logical_id_remap (dict[str, str] | None): Mapping from dummy
                logical_ids to actual logical_ids. Defaults to None.
            param_values (dict | None): Parameter values for resolving
                expressions. Defaults to None.

        Returns:
            tuple[list[int], bool]: Pair ``(indices, is_precise)``.
                ``is_precise`` is True when every operand — including those
                inside nested control flow — resolved cleanly.
        """
        if logical_id_remap is None:
            logical_id_remap = {}
        if param_values is None:
            param_values = {}

        affected: set[int] = set()
        is_precise = True

        def collect_qubits(ops: list[Operation]) -> None:
            """Collect affected wires from one nested operation list.

            Args:
                ops (list[Operation]): Operations whose quantum operands and
                    nested control-flow regions should be inspected.
            """
            nonlocal is_precise
            for inner_op in ops:
                if isinstance(
                    inner_op,
                    (ForOperation, WhileOperation, ForItemsOperation),
                ):
                    nested, nested_precise = self._analyze_loop_affected_qubits(
                        inner_op,
                        qubit_map,
                        logical_id_remap,
                        param_values,
                    )
                    affected.update(nested)
                    if not nested_precise:
                        is_precise = False
                    continue
                if isinstance(inner_op, IfOperation):
                    collect_qubits(inner_op.true_operations)
                    collect_qubits(inner_op.false_operations)
                    continue
                indices, operation_resolved = self._resolve_operation_affected_qubits(
                    inner_op,
                    qubit_map,
                    logical_id_remap,
                    param_values,
                )
                affected.update(indices)
                if not operation_resolved:
                    is_precise = False

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
            op (ControlledUOperation): Controlled operation whose power to
                resolve.
            param_values (dict): Parameter values for evaluating Value
                expressions.

        Returns:
            int: Resolved power, with one as the fallback minimum.
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
            value (Value): IR value to evaluate.
            param_values (dict): Mapping from logical IDs or parameter names to
                concrete values.
            operations (list[Operation] | None): Operation list to search for
                defining BinOps.
                Falls back to self.graph.operations if None.

        Returns:
            int | float | None: Concrete numeric value, or None if
                unresolvable.
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

        # 4. Look for defining BinOp in operations
        if operations is None:
            operations = getattr(self, "graph", None)
            operations = operations.operations if operations else []

        for op in operations:
            if isinstance(op, BinOp) and op.results and id(op.results[0]) == id(value):
                lhs_val = self._evaluate_value(op.lhs, param_values, operations)
                rhs_val = self._evaluate_value(op.rhs, param_values, operations)
                if lhs_val is not None and rhs_val is not None:
                    if op.kind == BinOpKind.ADD:
                        return lhs_val + rhs_val
                    elif op.kind == BinOpKind.SUB:
                        return lhs_val - rhs_val
                    elif op.kind == BinOpKind.MUL:
                        return lhs_val * rhs_val
                    elif op.kind == BinOpKind.FLOORDIV:
                        return lhs_val // rhs_val if rhs_val != 0 else None
                    elif op.kind == BinOpKind.MOD:
                        return lhs_val % rhs_val if rhs_val != 0 else None
                    elif op.kind == BinOpKind.DIV:
                        return lhs_val / rhs_val if rhs_val != 0 else None
                    elif op.kind == BinOpKind.POW:
                        result = lhs_val**rhs_val
                        if isinstance(rhs_val, (int, float)) and rhs_val < 0:
                            return result
                        return int(result)
                    elif op.kind == BinOpKind.MIN:
                        return lhs_val if lhs_val <= rhs_val else rhs_val
                return None

        return None

    def _resolve_symbolic_array_name(
        self, value: Value, param_values: dict
    ) -> str | None:
        """Try to resolve array element indices for symbolic display.

        For array elements like phis[i] where the array is symbolic but
        indices can be resolved (e.g., loop variable i=0), produces
        "phis[0]" instead of "phis[i]".

        Args:
            value (Value): IR value that may be an array element.
            param_values (dict): Mapping from logical IDs or names to concrete
                values.

        Returns:
            str | None: Resolved name string (e.g., ``"phis[0]"``), or None
                if not applicable.
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
            val (int | float | None): Numeric value to convert, or None.

        Returns:
            str: Integer string representation, or ``"?"`` if val is None.
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

        Args:
            op (ForOperation): Loop whose range expression to format.
            loop_vars (set[str]): Loop-variable names available to symbolic
                expression formatting.
            param_values (dict): Parameter values used to resolve range bounds.

        Returns:
            str: Human-readable ``qm.range(...)`` expression.
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
        """Find a value's defining BinOp and build a symbolic expression.

        Args:
            value (Value): Value whose defining operation to locate.
            param_values (dict): Parameter values used to resolve operands.
            operations (list[Operation] | None): Preferred operation scope to
                search before the top-level graph. Defaults to None.

        Returns:
            str | None: Symbolic expression, or None when no defining BinOp or
                resolvable operands are available.
        """
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

        Args:
            binop (BinOp): Binary operation to format.
            param_values (dict): Parameter values used to resolve operands.

        Returns:
            str | None: Simplified symbolic expression, or None when either
                operand cannot be resolved.
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
        """Pre-evaluate intermediate BinOp results in loop body.

        Scans loop body operations for BinOp (e.g., j = i + 1) and stores
        evaluated results by value ID in param_values. This allows subsequent
        index resolution to find the concrete values.

        When a BinOp cannot be fully resolved numerically (e.g., one operand
        is a symbolic parameter), a symbolic string expression is stored
        instead (e.g., "2*gamma", "theta+1").

        Args:
            operations (list[Operation]): List of operations in the loop body.
            param_values (dict): Mutable mapping updated in-place with resolved
                intermediate values keyed by logical_id.
        """
        for op in operations:
            if isinstance(op, BinOp) and op.results:
                result_value = op.results[0]
                resolved = self._evaluate_value(result_value, param_values, operations)
                if resolved is not None:
                    param_values[result_value.logical_id] = resolved
                else:
                    # Try building a symbolic string expression
                    symbolic = self._build_symbolic_binop(op, param_values)
                    if symbolic is not None:
                        param_values[result_value.logical_id] = symbolic

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

        Args:
            value (Value): Value whose logical ID or array-element alias to
                canonicalize.
            qubit_map (dict[str, int]): Mapping from canonical logical-ID keys
                to wire indices; discovered aliases are registered in place.
            logical_id_remap (dict[str, str]): Mapping from block-local logical
                IDs to caller-visible logical IDs.
            param_values (dict | None): Parameter values used to resolve
                symbolic array indices. Defaults to None.

        Returns:
            str: Canonical key suitable for lookup in ``qubit_map``.
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
        is_view_array_element = (
            value.parent_array is not None
            and isinstance(value.parent_array, ArrayValue)
            and value.parent_array.slice_of is not None
        )
        is_array_view = isinstance(value, ArrayValue) and value.slice_of is not None
        if (
            lid in qubit_map
            and not is_symbolic_array_element
            and not is_view_array_element
            and not is_array_view
        ):
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
        if is_array_view:
            assert isinstance(value, ArrayValue)
            conservative_wires, exact_aliases = self._array_wire_aliases(
                value,
                qubit_map,
                logical_id_remap,
                param_values or {},
            )
            if conservative_wires or exact_aliases:
                ordered_exact_wires = [
                    exact_aliases[index] for index in sorted(exact_aliases)
                ]
                first_wire = (conservative_wires or ordered_exact_wires)[0]
                qubit_map[lid] = first_wire
                for index, wire in enumerate(conservative_wires):
                    marker_key = f"{lid}_[{_CONSERVATIVE_ALIAS_TOKEN}{index}]"
                    qubit_map[marker_key] = wire
                for index, wire in exact_aliases.items():
                    qubit_map[f"{lid}_[{index}]"] = wire
                return lid

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

    def _resolve_declared_array_size(
        self,
        array_value: Value,
        param_values: dict,
    ) -> int | None:
        """Resolve an array's declared first-dimension size.

        Unlike :meth:`_resolve_array_size`, this method never infers width from
        wire aliases. Keeping declaration resolution separate prevents a
        sparse exact alias such as ``_[2]`` from being mistaken for a
        one-element array when ``_[0]`` also happens to exist.

        Args:
            array_value (Value): Value expected to expose an array ``shape``.
            param_values (dict): Bindings used to evaluate a symbolic shape.

        Returns:
            int | None: Declared first-dimension size, or None when the shape
                remains unresolved.
        """
        if not hasattr(array_value, "shape") or not array_value.shape:
            return None
        size_value = array_value.shape[0]
        if size_value.is_constant():
            size = size_value.get_const()
            if isinstance(size, int):
                return size
            return None
        evaluated = self._evaluate_value(size_value, param_values)
        if isinstance(evaluated, (int, float)):
            return int(evaluated)
        return None

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
            array_value (Value): The array whose length to resolve. Expected
                to expose a `shape` attribute.
            resolved_lid (str): Logical ID used to index `qubit_map` for the
                element-key scan fallback.
            qubit_map (dict[str, int]): Mapping from logical_id to qubit wire
                index.
                Used only for the scan fallback.
            param_values (dict): Parameter values for evaluating a symbolic
                shape.

        Returns:
            int | None: The array length, or None if no strategy
            succeeds.
        """
        declared_size = self._resolve_declared_array_size(
            array_value,
            param_values,
        )
        if declared_size is not None:
            return declared_size
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
            list[int]: The wire indices the QFixed measurement targets. Exact
                carriers keep most-significant-bit-first order; an unresolved
                carrier contributes every conservative marker candidate once.
                Empty when the operand carries no cast metadata or no carrier
                entry resolves to a known wire.
        """
        cast_meta = getattr(operand.metadata, "cast", None)
        if cast_meta is None or not cast_meta.qubit_logical_ids:
            return []
        indices: list[int] = []
        seen_indices: set[int] = set()
        for carrier_key in cast_meta.qubit_logical_ids:
            # Carrier key format: ``f"{root_logical_id}_{idx}"`` (no
            # brackets).  Split on the *last* ``_`` to recover the root
            # prefix and idx; the root logical_id itself may contain
            # underscores (UUIDs use them), so split from the right.
            root_and_index = split_indexed_identifier(carrier_key)
            if root_and_index is None:
                continue
            root_lid, idx_str = root_and_index
            idx_int = int(idx_str)
            remapped_lid = logical_id_remap.get(root_lid, root_lid)
            bracket_key = f"{remapped_lid}_[{idx_int}]"
            wire = qubit_map.get(bracket_key)
            if wire is not None:
                if wire not in seen_indices:
                    seen_indices.add(wire)
                    indices.append(wire)
                continue

            marker_index = 0
            while True:
                marker_key = (
                    f"{remapped_lid}_[{_CONSERVATIVE_ALIAS_TOKEN}{marker_index}]"
                )
                marker_wire = qubit_map.get(marker_key)
                if marker_wire is None:
                    break
                if marker_wire not in seen_indices:
                    seen_indices.add(marker_wire)
                    indices.append(marker_wire)
                marker_index += 1
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
            operand (Value): IR value expected to have `parent_array` and
                `element_indices`. Operands without a `parent_array`
                return None.
            qubit_map (dict[str, int]): Mapping from logical_id to qubit wire
                index.
            logical_id_remap (dict[str, str]): Remap from dummy logical_ids to actual
                logical_ids.
            param_values (dict): Parameter values for evaluating a symbolic
                `element_indices[0]`.

        Returns:
            int | None: Wire index for the resolved element. When the parent
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
        parent_lid = logical_id_remap.get(
            parent_array.logical_id,
            parent_array.logical_id,
        )
        if isinstance(parent_array, ArrayValue):
            conservative_wires, exact_aliases = self._array_wire_aliases(
                parent_array,
                qubit_map,
                logical_id_remap,
                param_values,
            )
            if idx_int in exact_aliases:
                return exact_aliases[idx_int]
            if conservative_wires:
                # The semantic element is not tied to one physical candidate.
                # Let the caller choose its documented conservative fallback.
                return None
        local_element_key = f"{parent_lid}_[{idx_int}]"
        if local_element_key in qubit_map:
            return qubit_map[local_element_key]
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
            operand (Value): IR value that is not a `parent_array` element
                access.
            qubit_map (dict[str, int]): Mapping from logical_id to qubit wire
                index.
            logical_id_remap (dict[str, str]): Remap from dummy logical_ids to actual
                logical_ids.
            param_values (dict): Parameter values for evaluating symbolic
                shapes.

        Returns:
            list[int] | None: None if the operand is unresolvable (logical_id not in
            `qubit_map`, or shape symbolic and unevaluable). An
            empty list if the operand is resolved but touches zero
            qubits (e.g. concrete `shape[0] == 0`). Otherwise, a
            list of wire indices.
        """
        resolved_lid = logical_id_remap.get(operand.logical_id, operand.logical_id)

        if operand.get_cast_qubit_logical_ids():
            carrier_wires = self._resolve_qfixed_carrier_indices(
                operand,
                qubit_map,
                logical_id_remap,
            )
            if carrier_wires:
                return carrier_wires

        slice_context_value = cast(dict[object, object], param_values).get(
            _UNRESOLVED_SLICE_CONTEXT_KEY
        )
        if isinstance(slice_context_value, dict):
            actual_slice = slice_context_value.get(operand.logical_id)
            if isinstance(actual_slice, ArrayValue) and actual_slice is not operand:
                forwarded_params = dict(param_values)
                cast(dict[object, object], forwarded_params).pop(
                    _UNRESOLVED_SLICE_CONTEXT_KEY,
                    None,
                )
                return self._resolve_non_element_operand(
                    actual_slice,
                    qubit_map,
                    logical_id_remap,
                    forwarded_params,
                )

        if isinstance(operand, ArrayValue):
            conservative_wires, exact_aliases = self._array_wire_aliases(
                operand,
                qubit_map,
                logical_id_remap,
                param_values,
            )
            if conservative_wires:
                resolved_wires = list(conservative_wires)
                seen_wires = set(resolved_wires)
                for semantic_index in sorted(exact_aliases):
                    semantic_wire = exact_aliases[semantic_index]
                    if semantic_wire not in seen_wires:
                        seen_wires.add(semantic_wire)
                        resolved_wires.append(semantic_wire)
                return resolved_wires
            semantic_size = self._resolve_array_size(
                operand,
                resolved_lid,
                qubit_map,
                param_values,
            )
            if semantic_size is not None and len(exact_aliases) == semantic_size:
                return [exact_aliases[index] for index in range(semantic_size)]

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
            else:
                # A caller may already have supplied explicit view-local wire
                # aliases even though this view's affine bounds remain
                # symbolic in the callee. Prefer those exact aliases first.
                size = self._resolve_array_size(
                    operand,
                    resolved_lid,
                    qubit_map,
                    param_values,
                )
                if size is not None:
                    local_wires: list[int] = []
                    while (
                        local_wire := qubit_map.get(
                            f"{resolved_lid}_[{len(local_wires)}]"
                        )
                    ) is not None:
                        local_wires.append(local_wire)
                    if len(local_wires) >= size:
                        return local_wires

                # Without concrete slice bounds the exact subset is unknown,
                # but every possible element still aliases the existing root
                # register. Conservatively show all root wires rather than
                # allocating a fake independent register for the view.
                root = operand
                seen_arrays: set[int] = set()
                while root.slice_of is not None:
                    root_identity = id(root)
                    if root_identity in seen_arrays:
                        return None
                    seen_arrays.add(root_identity)
                    root = root.slice_of
                return self._resolve_non_element_operand(
                    root,
                    qubit_map,
                    logical_id_remap,
                    param_values,
                )

        if isinstance(operand, ArrayValue) and operand.shape:
            size = self._resolve_array_size(
                operand, resolved_lid, qubit_map, param_values
            )
            if size == 0:
                return []
            if size is not None and resolved_lid in qubit_map:
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
                keyed_wires = [
                    qubit_map.get(f"{resolved_lid}_[{index}]") for index in range(size)
                ]
                if all(wire is not None for wire in keyed_wires):
                    return [wire for wire in keyed_wires if wire is not None]
                if any(wire is not None for wire in keyed_wires):
                    # A partial result alias must allocate its missing suffix;
                    # arithmetic base+index could otherwise capture an
                    # unrelated physical wire that happens to be adjacent.
                    return None
                base_idx = qubit_map[resolved_lid]
                contiguous_wires = [base_idx + k for k in range(size)]
                registered_wires = set(qubit_map.values())
                if all(wire in registered_wires for wire in contiguous_wires):
                    return contiguous_wires
                # A partially aliased opaque result may be wider than its
                # input. Returning arithmetic indices here would advertise
                # wires that the registry has never allocated. Let the
                # Invoke result pass register the missing physical wires.
                return None
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
            operand (Value): IR value representing a qubit or qubit-array
                operand.
            qubit_map (dict[str, int]): Mapping from logical_id to qubit wire
                index.
            logical_id_remap (dict[str, str] | None): Optional remap from dummy
                logical_ids
                (used during callable inlining) to actual
                logical_ids. Defaults to None (empty remap).
            param_values (dict | None): Optional parameter values for evaluating
                computed indices and shapes. Defaults to None.

        Returns:
            list[int] | None: None if the operand is unresolvable. An empty list if
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
            parent_wires = self._resolve_non_element_operand(
                operand.parent_array,
                qubit_map,
                logical_id_remap,
                param_values,
            )
            if parent_wires is not None:
                return parent_wires[:1]
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
            operand (Value): IR value representing a qubit or qubit-array
                operand.
            qubit_map (dict[str, int]): Mapping from logical_id to qubit wire
                index.
            logical_id_remap (dict[str, str] | None): Optional remap from dummy
                logical_ids
                to actual logical_ids. Defaults to None.
            param_values (dict | None): Optional parameter values for evaluating
                computed indices and shapes. Defaults to None.

        Returns:
            list[int] | None: None if the operand is unresolvable. An empty list if
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
            parent_wires = self._resolve_non_element_operand(
                operand.parent_array,
                qubit_map,
                logical_id_remap,
                param_values,
            )
            if parent_wires is not None:
                return parent_wires
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
            op (ForOperation): Loop whose range to evaluate.
            param_values (dict): Parameter values for resolving symbolic
                operands.

        Returns:
            tuple[int, int | None, int]: Start, optional stop, and step values.
                The stop value is None when it remains symbolic.

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
        call_scope: str | None = None,
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
            call_scope (str | None): Analyzer-local namespace for values owned
                by this callable body. Defaults to None when no callable scope
                is being entered.

        Returns:
            tuple[dict[str, str], dict[str, object]]: Child logical-id remap
                and child parameter values for recursive inline processing.
        """
        new_logical_id_remap = dict(logical_id_remap)
        if call_scope is not None:
            new_logical_id_remap[_CALL_SCOPE_REMAP_KEY] = call_scope
            for logical_id in self._body_owned_quantum_logical_ids(block_value):
                new_logical_id_remap[logical_id] = f"{logical_id}@{call_scope}"
        input_pairs = [
            structural_pair
            for formal, actual in zip(block_value.input_values, actual_inputs)
            for structural_pair in self._structural_input_pairs(formal, actual)
        ]
        unresolved_slice_inputs: dict[str, ArrayValue] = {}
        parent_slice_context_value = cast(dict[object, object], param_values).get(
            _UNRESOLVED_SLICE_CONTEXT_KEY
        )
        parent_slice_context = (
            parent_slice_context_value
            if isinstance(parent_slice_context_value, dict)
            else {}
        )

        for dummy_input, actual_input in input_pairs:
            # For Qubit array elements, resolve through _resolve_array_element_lid
            # to get the canonical parent_[idx] key instead of the raw UUID
            if (
                qubit_map is not None
                and isinstance(actual_input, Value)
                and hasattr(actual_input, "parent_array")
                and actual_input.parent_array is not None
                and actual_input.type.is_quantum()
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
                    and dummy_input.type.is_quantum()
                    and actual_input.type.is_quantum()
                ):
                    forwarded_unresolved_slice = parent_slice_context.get(
                        actual_input.logical_id
                    )
                    conservative_wires, exact_aliases = self._array_wire_aliases(
                        actual_input,
                        qubit_map,
                        logical_id_remap,
                        param_values,
                    )
                    direct_unresolved_slice = (
                        actual_input.slice_of is not None
                        and self._resolve_view_chain_to_root(
                            actual_input,
                            param_values,
                        )
                        is None
                    )
                    if conservative_wires:
                        qubit_map.setdefault(actual_lid, conservative_wires[0])
                        qubit_map.setdefault(
                            dummy_input.logical_id,
                            conservative_wires[0],
                        )
                        alias_lids = (actual_lid, dummy_input.logical_id)
                        for alias_lid in alias_lids:
                            for index, wire in enumerate(conservative_wires):
                                marker_suffix = f"{_CONSERVATIVE_ALIAS_TOKEN}{index}"
                                qubit_map[f"{alias_lid}_[{marker_suffix}]"] = wire
                            for semantic_index, wire in exact_aliases.items():
                                element_key = f"{alias_lid}_[{semantic_index}]"
                                qubit_map[element_key] = wire
                    elif isinstance(forwarded_unresolved_slice, ArrayValue):
                        unresolved_slice_inputs[dummy_input.logical_id] = (
                            forwarded_unresolved_slice
                        )
                    elif direct_unresolved_slice:
                        unresolved_slice_inputs[dummy_input.logical_id] = actual_input
                    else:
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
        context = cast(dict[object, object], child_param_values)
        inherited_slice_context = context.get(_UNRESOLVED_SLICE_CONTEXT_KEY)
        slice_context = (
            dict(inherited_slice_context)
            if isinstance(inherited_slice_context, dict)
            else {}
        )
        for formal in block_value.input_values:
            for formal_node in self._structural_value_nodes(formal):
                slice_context.pop(formal_node.logical_id, None)
        slice_context.update(unresolved_slice_inputs)
        if slice_context:
            context[_UNRESOLVED_SLICE_CONTEXT_KEY] = slice_context
        else:
            context.pop(_UNRESOLVED_SLICE_CONTEXT_KEY, None)
        cleared_formal_uuids: set[str] = set()

        def clear_formal_bindings(formal_value: ValueBase) -> None:
            """Remove inherited bindings shadowed by one callee formal graph.

            Args:
                formal_value (ValueBase): Formal scalar or structural value
                    whose own IDs, parameter name, and child values to clear.
            """
            if formal_value.uuid in cleared_formal_uuids:
                return
            cleared_formal_uuids.add(formal_value.uuid)
            child_param_values.pop(formal_value.uuid, None)
            child_param_values.pop(formal_value.logical_id, None)
            if formal_value.is_parameter():
                parameter_name = formal_value.parameter_name()
                if parameter_name:
                    child_param_values.pop(parameter_name, None)

            if isinstance(formal_value, TupleValue):
                children: Sequence[ValueBase] = formal_value.elements
            elif isinstance(formal_value, DictValue):
                children = tuple(
                    child
                    for key, entry_value in formal_value.entries
                    for child in (key, entry_value)
                )
            else:
                value_children: list[ValueBase] = []
                if isinstance(formal_value, Value):
                    if formal_value.parent_array is not None:
                        value_children.append(formal_value.parent_array)
                    value_children.extend(formal_value.element_indices)
                if isinstance(formal_value, ArrayValue):
                    value_children.extend(formal_value.shape)
                    for view_value in (
                        formal_value.slice_of,
                        formal_value.slice_start,
                        formal_value.slice_step,
                    ):
                        if view_value is not None:
                            value_children.append(view_value)
                children = value_children
            for child in children:
                clear_formal_bindings(child)

        for formal_input in block_value.input_values:
            clear_formal_bindings(formal_input)

        def bind_child_value(dummy_value: ValueBase, value: object) -> None:
            """Publish one forwarded value under every callee lookup key.

            Args:
                dummy_value (ValueBase): Callee-side formal or shape value.
                value (object): Concrete or symbolic caller-side value.
            """
            child_param_values[dummy_value.logical_id] = value
            # Symbolic strings are display aliases (for example ``"theta"``
            # or ``"items[i]"``), not compile-time values. Publishing one by
            # parameter name would make CompileTimeIfLoweringPass treat it as
            # concrete and silently choose a branch.
            if dummy_value.is_parameter() and not isinstance(
                value,
                (str, ValueBase),
            ):
                parameter_name = dummy_value.parameter_name()
                if parameter_name:
                    child_param_values[parameter_name] = value

        for dummy_input, actual_input in input_pairs:
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
                bind_child_value(dummy_input, const)
            elif actual_input.logical_id in param_values or actual_lid in param_values:
                pv = param_values.get(
                    actual_input.logical_id, param_values.get(actual_lid)
                )
                if isinstance(pv, (int, float)):
                    # Numeric value: store directly
                    bind_child_value(dummy_input, pv)
                elif (
                    isinstance(actual_input, Value)
                    and hasattr(actual_input, "parent_array")
                    and actual_input.parent_array is not None
                    and not actual_input.type.is_quantum()
                ):
                    # Non-qubit array element with non-numeric value:
                    # try resolving the array index before storing the raw string
                    evaluated = self._evaluate_value(actual_input, param_values)
                    if evaluated is not None and isinstance(evaluated, (int, float)):
                        bind_child_value(dummy_input, evaluated)
                    else:
                        resolved = self._resolve_symbolic_array_name(
                            actual_input, param_values
                        )
                        bind_child_value(
                            dummy_input,
                            resolved if resolved is not None else pv,
                        )
                else:
                    bind_child_value(dummy_input, pv)
            elif (
                isinstance(actual_input, Value)
                and hasattr(actual_input, "parent_array")
                and actual_input.parent_array is not None
                and not actual_input.type.is_quantum()
            ):
                # Non-qubit array element not in param_values:
                # try evaluating or resolving symbolic name
                evaluated = self._evaluate_value(actual_input, param_values)
                if evaluated is not None and isinstance(evaluated, (int, float)):
                    bind_child_value(dummy_input, evaluated)
                else:
                    resolved = self._resolve_symbolic_array_name(
                        actual_input, param_values
                    )
                    if resolved is not None:
                        bind_child_value(dummy_input, resolved)
                    elif actual_input.is_parameter():
                        bind_child_value(
                            dummy_input,
                            actual_input.parameter_name() or actual_input.name,
                        )
            elif actual_input.is_parameter():
                bind_child_value(
                    dummy_input,
                    actual_input.parameter_name() or actual_input.name,
                )
            elif isinstance(actual_input, Value):
                # actual_input is a BinOp result (or similar unresolved
                # non-parameter Value). Try numeric evaluation via
                # graph.operations first — after top-level
                # pre-evaluation in build_visual_ir this usually
                # succeeds.
                evaluated = self._evaluate_value(actual_input, param_values)
                if evaluated is not None and isinstance(evaluated, (int, float)):
                    bind_child_value(dummy_input, evaluated)
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
                        bind_child_value(dummy_input, expr)
                    else:
                        # Last resort: forward the logical_id so
                        # downstream lookups can still chase the
                        # defining BinOp via logical_id_remap.
                        new_logical_id_remap[dummy_input.logical_id] = (
                            actual_input.logical_id
                        )

        # Propagate ArrayValue shape dimensions and const_array data
        for dummy_input, actual_input in input_pairs:
            if isinstance(actual_input, ArrayValue) and isinstance(
                dummy_input, ArrayValue
            ):
                if dummy_input.shape and actual_input.shape:
                    for dummy_dim, actual_dim in zip(
                        dummy_input.shape, actual_input.shape
                    ):
                        const = actual_dim.get_const()
                        if const is not None:
                            bind_child_value(dummy_dim, const)
                        elif actual_dim.logical_id in param_values:
                            bind_child_value(
                                dummy_dim,
                                param_values[actual_dim.logical_id],
                            )
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
        """Measure a gate box width before figure creation.

        Args:
            op (GateOperation): Gate whose rendered label determines the width.
            param_values (dict | None): Parameter values used to resolve the
                label. Defaults to None.

        Returns:
            float: Gate width in layout coordinate units.
        """
        label, has_param = self._get_gate_label(op, param_values)
        if not has_param:
            return self.style.gate_width

        text_width = measure_text_width(
            label,
            font_size=self.style.font_size,
            fallback_char_width=self.style.char_width_gate,
        )
        padding = self.style.text_padding
        return max(self.style.gate_width, text_width + 2 * padding)

    def _estimate_label_box_width(self, label: str) -> float:
        """Measure a general text-label box width.

        Args:
            label (str): Text label rendered inside a gate or block box.

        Returns:
            float: Box width in layout coordinate units.
        """
        regular_width = measure_text_width(
            label,
            font_size=self.style.font_size,
            fallback_char_width=self.style.char_width_gate,
        )
        header_width = measure_text_width(
            label,
            font_size=self.style.subfont_size,
            font_weight="bold",
            fallback_char_width=self.style.char_width_bold,
        )
        text_width = max(regular_width, header_width)
        return max(text_width + 2 * self.style.text_padding, self.style.gate_width)

    def _estimate_block_label_box_width(self, label: str) -> float:
        """Estimate box width for a callable-block label.

        Uses ``char_width_block`` (wider than ``char_width_gate``) because
        block labels often contain longer text with parenthesized parameters
        (e.g., ``mixer(omegas[0])``).

        Args:
            label (str): Callable-block label rendered at the regular gate size.

        Returns:
            float: Box width in layout coordinate units.
        """
        text_width = measure_text_width(
            label,
            font_size=self.style.font_size,
            fallback_char_width=self.style.char_width_block,
        )
        return max(text_width + 2 * self.style.text_padding, self.style.gate_width)

    @staticmethod
    def _get_child_op_lists(op: Operation) -> list[list[Operation]]:
        """Return child operation lists for control-flow operations.

        Args:
            op (Operation): Operation whose nested regions to inspect.

        Returns:
            list[list[Operation]]: Child operation lists, or an empty list for
                an operation without nested regions.
        """
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
            operations (list[Operation]): Operations to scan recursively.
            param_values (dict | None): Parameter values for gate-width
                estimation. Defaults to None.

        Returns:
            float: Maximum gate width, at least ``style.gate_width``.
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
            op (GateOperation): Gate whose parameter to format.
            loop_vars (set[str] | None): Loop variable names in scope (e.g.,
                ``{"i", "j"}``). Defaults to None.
            body_operations (list | None): Operations used to resolve index
                expressions. Defaults to None.
            param_values (dict | None): Parameter values used to resolve
                symbolic expressions. Defaults to None.

        Returns:
            str | None: Parameter string (e.g., ``"0.5"``), or None when the
                gate has no displayable parameter.
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
            operand (Value): Non-qubit value to format.
            loop_vars (set[str]): Loop variable names in scope.
            param_values (dict | None): Parameter values for evaluation.
                Defaults to None.
            body_operations (list | None): Operations used for BinOp
                resolution. Defaults to None.

        Returns:
            str | None: Formatted string, possibly containing TeX, or None if
                unresolvable.
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
            operand (Value): IR value to test.
            operations (list | None): Operations to search. Falls back to
                ``self.graph.operations`` when None.

        Returns:
            bool: True if a matching BinOp is found, else False.
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
            op (Operation): Operation to format.
            loop_vars (set[str]): Loop variable names in scope (e.g.,
                ``{"i", "j"}``).
            indent (int): Indentation level, using two spaces per level.
                Defaults to zero.
            max_depth (int): Maximum nesting depth for recursive formatting.
                Defaults to two.
            param_values (dict | None): Parameter values used to resolve
                symbolic expressions. Defaults to None.
            body_operations (list | None): Enclosing operations used for
                producer lookup. Defaults to None.

        Returns:
            str | None: Expression string (e.g.,
                ``"q[i],q[j] = cx(q[i],q[j])"``), or None.
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

        elif isinstance(
            op,
            (MeasureOperation, MeasureVectorOperation, MeasureQFixedOperation),
        ):
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

        elif isinstance(op, ExpvalOp):
            qubits = op.qubits
            if qubits.parent_array is not None:
                array_name = qubits.parent_array.name or "qubits"
                index = self._resolve_index_expression(
                    qubits,
                    loop_vars,
                    body_operations,
                )
                qubit_expression = f"{array_name}[{index}]"
            else:
                qubit_expression = qubits.name or "qubits"
            observable_expression = op.observable.name or "observable"
            result_expression = op.output.name or "expval"
            return (
                f"{prefix}{result_expression} = "
                f"expval({qubit_expression}, {observable_expression})"
            )

        elif isinstance(op, InvokeOperation):
            block_name = op.name or "callable"
            qubit_parts: list[str] = []
            param_parts: list[str] = []
            if op.attrs.get("kind") == "composite":
                qubit_operands = list(op.control_qubits) + list(op.target_qubits)
                param_operands = list(op.parameters)
            else:
                qubit_operands = [v for v in op.operands if v.type.is_quantum()]
                param_operands = [v for v in op.operands if not v.type.is_quantum()]
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
                """Format one controlled-call qubit operand.

                Args:
                    v (Value): Quantum operand to format.

                Returns:
                    str | None: Array-element expression or value name, or
                    ``None`` when no display name can be derived.
                """
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
            operand (Value): Value that may be an array element.
            loop_vars (set[str]): Loop variable names in scope.
            operations (list | None): Operations to search for BinOp
                definitions. Defaults to None.

        Returns:
            str: Index expression string (e.g., ``"i"``, ``"i+1"``, or
                ``"1"``).
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
            value (Value): IR value to format.
            loop_vars (set[str]): Loop variable names in scope.
            operations (list | None): Operations to search for BinOp
                definitions.
                If None, falls back to self.graph.operations.

        Returns:
            str: Human-readable expression string.
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
            value (float | int): Numeric parameter value.

        Returns:
            str: Compact display string using scientific notation for very
                small or large magnitudes and one or two decimal places
                otherwise.
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

        Args:
            name (str): Candidate symbolic parameter name.

        Returns:
            tuple[str, str] | None: Greek-symbol prefix and remaining suffix,
                or None when no non-exact Greek prefix matches.
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
            name (str): Parameter name.

        Returns:
            str: Formatted string suitable for matplotlib text rendering.
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

        Args:
            expr (str): Symbolic arithmetic expression to format.

        Returns:
            str: Expression wrapped as a TeX math string.
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
            op (GateOperation): Gate operation to label.
            param_values (dict | None): Mapping from logical IDs to resolved
                parameter values, used for inline block expansion. Defaults to
                None.

        Returns:
            tuple[str, bool]: Display label and whether it includes a
                parameter.
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
