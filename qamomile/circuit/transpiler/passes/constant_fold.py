"""Constant folding pass for compile-time expression evaluation."""

from __future__ import annotations

import dataclasses
from typing import Any, cast

from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation import (
    Operation,
    ReleaseSliceViewOperation,
    SliceArrayOperation,
)
from qamomile.circuit.ir.operation.arithmetic_operations import BinOp, BinOpKind
from qamomile.circuit.ir.operation.classical_ops import StoreArrayElementOperation
from qamomile.circuit.ir.operation.gate import (
    ConcreteControlledU,
    ControlledUOperation,
    SymbolicControlledU,
)
from qamomile.circuit.ir.value import ArrayValue, Value, ValueBase
from qamomile.circuit.transpiler.errors import ValidationError
from qamomile.circuit.transpiler.value_resolver import (
    ValueResolver as UnifiedValueResolver,
)

from . import Pass
from .control_flow_visitor import OperationTransformer


class ConstantFoldingPass(Pass[Block, Block]):
    """Evaluates constant expressions at compile time.

    This pass folds BinOp operations when all operands are constants
    or bound parameters, eliminating unnecessary classical operations
    that would otherwise split quantum segments.

    Example:
        Before (with bindings={"phase": 0.5}):
            BinOp(phase * 2) -> classical segment split

        After:
            Constant 1.0 -> no segment split
    """

    def __init__(
        self,
        bindings: dict[str, Any] | None = None,
        *,
        strip_slice_ops: bool = True,
    ):
        """Create a constant-folding pass.

        Args:
            bindings: Compile-time parameter bindings used when folding
                BinOps that reference declared parameters.
            strip_slice_ops: When ``True`` (default), removes
                ``SliceArrayOperation`` nodes after folding. Set to
                ``False`` when a downstream pass — notably
                ``SliceBorrowCheckPass`` — still needs to observe
                slice declaration points in program order to decide
                view liveness. A separate strip pass must then run
                after the linearity check so segmentation still sees
                a pure quantum-op stream.
        """
        self._bindings = bindings or {}
        self._strip_slice_ops = strip_slice_ops

    @property
    def name(self) -> str:
        return "constant_fold"

    def run(self, input: Block) -> Block:
        """Run constant folding on the block."""
        # HIERARCHICAL is accepted during the self-recursion unroll loop;
        # unresolved inline callable invocations are passed through untouched.
        if input.kind not in (BlockKind.AFFINE, BlockKind.HIERARCHICAL):
            raise ValidationError(
                f"ConstantFoldingPass expects AFFINE or HIERARCHICAL "
                f"block, got {input.kind}",
            )

        # Track folded values: uuid -> constant Value
        folded_values: dict[str, Value] = {}

        # Block-output UUIDs: a folded store whose result is returned must
        # stay in the IR so the classical executor materializes the value
        # at runtime (folding records compile-time metadata only).
        output_uuids = {v.uuid for v in input.output_values if isinstance(v, ValueBase)}

        # Process operations
        new_ops = self._fold_operations(input.operations, folded_values, output_uuids)

        return dataclasses.replace(input, operations=new_ops)

    def _fold_operations(
        self,
        operations: list[Operation],
        folded_values: dict[str, Value],
        output_uuids: set[str] | None = None,
    ) -> list[Operation]:
        """Fold constant BinOps and top-level classical element stores.

        Constant ``BinOp``s are removed from the stream and recorded in
        ``folded_values`` for operand/result substitution.  Top-level
        ``StoreArrayElementOperation``s whose inputs are all compile-time
        resolvable are folded into updated ``const_array`` metadata on
        the result; the folded store is stripped unless its result is a
        block output (returned arrays keep the runtime store so the
        executor materializes the final contents).  All other operations
        get folded values substituted into their operands and results.

        Args:
            operations (list[Operation]): Operations to process.
                Recurses through control-flow bodies; stores inside a
                loop body never fold (they execute once per iteration).
            folded_values (dict[str, Value]): Map from folded value
                UUIDs to their constant replacements.  Updated in place.
            output_uuids (set[str] | None): UUIDs of block output
                values; a folded store producing one of these stays in
                the IR.  Defaults to None (no outputs).

        Returns:
            list[Operation]: The transformed operation list.
        """
        outer_self = self
        block_output_uuids = output_uuids or set()

        class ConstantFoldingTransformer(OperationTransformer):
            def __init__(self) -> None:
                """Initialize the transformer with zero control-flow depth."""
                self._nesting_depth = 0

            def _transform_control_flow(self, op: Operation) -> Operation:
                """Recurse into control-flow bodies, tracking nesting depth.

                Args:
                    op (Operation): The (possibly control-flow) operation.

                Returns:
                    Operation: The operation with nested bodies transformed.
                """
                self._nesting_depth += 1
                try:
                    return super()._transform_control_flow(op)
                finally:
                    self._nesting_depth -= 1

            def transform_operation(self, op: Operation) -> Operation | None:
                if isinstance(op, BinOp):
                    folded = outer_self._try_fold_binop(op, folded_values)
                    if folded is not None:
                        # BinOp was folded to a constant - remove it
                        # Just record the mapping for later substitution
                        folded_values[op.results[0].uuid] = folded
                        return None

                # Classical array element stores fold only at the top
                # level: inside a loop body the store executes once per
                # iteration (possibly zero times), so a single folded
                # contents snapshot would be wrong.
                if (
                    isinstance(op, StoreArrayElementOperation)
                    and self._nesting_depth == 0
                ):
                    substituted = cast(
                        StoreArrayElementOperation,
                        outer_self._substitute_folded_operands(op, folded_values),
                    )
                    folded_array = outer_self._try_fold_store_array_element(
                        substituted, folded_values
                    )
                    if folded_array is not None:
                        folded_values[op.results[0].uuid] = folded_array
                        if op.results[0].uuid in block_output_uuids:
                            # Returned arrays still need a runtime store so
                            # the executor can materialize the output; the
                            # substituted result carries the folded
                            # contents for compile-time readers.
                            return outer_self._substitute_folded_results(
                                substituted, folded_values
                            )
                        return None
                    return outer_self._substitute_folded_results(
                        substituted, folded_values
                    )

                # SliceArrayOperation is purely declarative: its result
                # (a sliced ``ArrayValue``) already carries all required
                # metadata (slice_of/slice_start/slice_step) for the
                # emit-time resolver, and downstream ops reference that
                # result directly.  Strip the op so segmentation sees a
                # pure quantum-segment sequence without a classical op
                # interleaved in the middle.  When ``strip_slice_ops``
                # is ``False``, keep the op so ``SliceBorrowCheckPass``
                # can use its position as the view's declaration point;
                # a later strip pass must remove it before segmentation.
                if isinstance(op, SliceArrayOperation):
                    if outer_self._strip_slice_ops:
                        return None
                    # Fold the result's slice metadata too so the
                    # linearity check can inspect the result directly.
                    # Default operand substitution only walks
                    # ``op.operands``; the result ArrayValue's
                    # ``slice_start`` / ``slice_step`` are separate
                    # references that need the same folding applied.
                    return outer_self._substitute_slice_op_result(op, folded_values)

                # ReleaseSliceViewOperation is the symmetric counterpart
                # of SliceArrayOperation: a declarative marker that tells
                # SliceBorrowCheckPass to drop the view's borrow.  The
                # same strip / keep policy applies — when
                # ``strip_slice_ops`` is True the marker is removed (it
                # carries no information needed downstream), otherwise
                # it survives to be observed by the linearity check and
                # finally removed by StripSliceArrayOpsPass.
                if isinstance(op, ReleaseSliceViewOperation):
                    if outer_self._strip_slice_ops:
                        return None
                    return op

                # Substitute folded values in operands and results.
                # Results must also be folded so that, e.g.,
                # MeasureVectorOperation's clbit-result ArrayValue has its
                # shape folded after the BinOps that computed the slice
                # length have been eliminated.  Without this the allocator
                # sees a symbolic shape and emits zero clbits.
                op_after_operands = outer_self._substitute_folded_operands(
                    op, folded_values
                )
                return outer_self._substitute_folded_results(
                    op_after_operands, folded_values
                )

        transformer = ConstantFoldingTransformer()
        return transformer.transform_operations(operations)

    def _try_fold_binop(
        self,
        op: BinOp,
        folded_values: dict[str, Value],
    ) -> Value | None:
        """Try to fold a BinOp to a constant. Returns None if not foldable."""
        if len(op.operands) != 2:
            return None

        left = self._resolve_value(op.operands[0], folded_values)
        right = self._resolve_value(op.operands[1], folded_values)

        if left is None or right is None:
            return None

        # Both operands are constants, evaluate
        result_value = self._evaluate_binop(op.kind, left, right)
        if result_value is None:
            return None

        # Create constant Value with same uuid for substitution
        result_type = op.results[0].type
        return Value(
            type=result_type,
            name=f"folded_{op.results[0].name}",
            uuid=op.results[0].uuid,  # Keep same UUID for substitution
        ).with_const(result_value)

    def _try_fold_store_array_element(
        self,
        op: StoreArrayElementOperation,
        folded_values: dict[str, Value],
    ) -> ArrayValue | None:
        """Try to fold a classical element store into constant array contents.

        A store folds when the source array's contents, the index, and the
        stored value are all compile-time resolvable (const metadata,
        bindings, or previously folded values).  The folded result is the
        op's result ``ArrayValue`` (same UUID) carrying the updated
        ``const_array`` metadata, so downstream element reads resolve
        against the post-store contents.

        Args:
            op (StoreArrayElementOperation): The store with operands
                already substituted from ``folded_values``.
            folded_values (dict[str, Value]): Map from folded value UUIDs
                to their constant replacements.

        Returns:
            ArrayValue | None: The result value with updated
                ``const_array`` metadata, or ``None`` when any input is
                not compile-time resolvable (the store then executes at
                runtime) or the constant index is out of range (left for
                the runtime path to reject loudly).
        """
        if len(op.operands) != 3:
            # Multi-dimensional stores are rejected at the frontend; leave
            # any programmatically constructed op for runtime rejection.
            return None

        array_value = op.operands[0]
        if not isinstance(array_value, ArrayValue):
            return None

        container = array_value.get_const_array()
        if container is None and getattr(array_value, "version", 0) == 0:
            # Name-keyed bindings (and parameter-name bindings) describe
            # the *initial* (version-0) contents of a kernel argument.  A
            # later SSA version — produced by a prior store that did NOT
            # fold, so its result carries no ``const_array`` — must not
            # fold against that stale pre-store snapshot.  Leaving
            # ``container`` unresolved keeps the store as a correct
            # runtime operation.  Mirrors the version guard in
            # ``value_resolver._resolve_array_element``.
            name = array_value.name
            if name and name in self._bindings:
                container = self._bindings[name]
            elif array_value.is_parameter():
                param_name = array_value.parameter_name()
                if param_name and param_name in self._bindings:
                    container = self._bindings[param_name]
        if container is None:
            return None

        stored = self._resolve_value(op.operands[1], folded_values)
        index = self._resolve_value(op.operands[2], folded_values)
        if stored is None or index is None:
            return None

        try:
            concrete_index = int(index)
            elements = list(container)
        except (TypeError, ValueError):
            return None
        if not 0 <= concrete_index < len(elements):
            return None

        elements[concrete_index] = stored
        result = op.results[0]
        if not isinstance(result, ArrayValue):
            return None
        return result.with_array_runtime_metadata(const_array=tuple(elements))

    def _resolve_value(
        self,
        value: Value,
        folded_values: dict[str, Value],
    ) -> float | int | None:
        """Resolve a Value to a constant, or None if not resolvable."""
        return UnifiedValueResolver(
            context=folded_values, bindings=self._bindings
        ).resolve(value)

    def _substitute_slice_op_result(
        self,
        op: SliceArrayOperation,
        folded_values: dict[str, Value],
    ) -> SliceArrayOperation:
        """Return a SliceArrayOperation whose result and operands are folded.

        Default operand substitution only walks ``op.operands``.  The
        ``SliceArrayOperation`` result is a sliced ``ArrayValue``
        whose ``slice_start`` / ``slice_step`` / ``slice_of`` fields
        reference the same Values as the operands do; without
        explicitly folding those fields the result retains symbolic
        BinOp values even after operand folding.  When the result is
        still in the IR (``strip_slice_ops=False``),
        ``SliceBorrowCheckPass`` inspects the result directly to
        decide view coverage, so it must see folded bounds.

        Args:
            op: The slice declaration to fold.
            folded_values: Map from BinOp uuid to its const-folded
                replacement ``Value``.

        Returns:
            A new ``SliceArrayOperation`` with both operands and the
            result's slice metadata substituted.  Returns ``op``
            unchanged if no folding was needed.
        """
        from qamomile.circuit.ir.value import ArrayValue

        new_op = cast(
            SliceArrayOperation,
            self._substitute_folded_operands(op, folded_values),
        )
        if not new_op.results or not isinstance(new_op.results[0], ArrayValue):
            return new_op
        result = new_op.results[0]
        folded_result = self._substitute_in_value(result, folded_values)
        if folded_result is result or not isinstance(folded_result, ArrayValue):
            return new_op
        return cast(
            SliceArrayOperation,
            dataclasses.replace(new_op, results=[folded_result]),
        )

    @staticmethod
    def _evaluate_binop(
        kind: BinOpKind | None,
        left: float | int,
        right: float | int,
    ) -> float | int | None:
        """Evaluate a binary operation on two constants."""
        from qamomile.circuit.transpiler.passes.eval_utils import (
            evaluate_binop_values,
        )

        return evaluate_binop_values(kind, left, right)

    def _substitute_in_value(
        self,
        v: Value,
        folded_values: dict[str, Value],
    ) -> Value:
        """Recursively substitute folded constants in a Value.

        Walks ``element_indices`` to arbitrary depth so that nested
        array accesses like ``q[indices[uint_tmp]]`` have their
        innermost symbolic index resolved when it is foldable.  Also
        propagates folded values into ArrayValue-specific fields —
        ``parent_array`` (recursively), ``shape``, and slice metadata
        (``slice_of`` / ``slice_start`` / ``slice_step``) — so that a
        sliced ArrayValue whose bounds were BinOp results has those
        bounds resolved to constants before emit.
        """
        from qamomile.circuit.ir.value import ArrayValue

        if v.uuid in folded_values:
            return folded_values[v.uuid]

        changed = False

        new_element_indices: tuple[Value, ...] = v.element_indices
        if v.element_indices:
            new_indices: list[Value] = []
            for idx in v.element_indices:
                if isinstance(idx, Value):
                    new_idx = self._substitute_in_value(idx, folded_values)
                    if new_idx is not idx:
                        changed = True
                    new_indices.append(new_idx)
                else:
                    new_indices.append(idx)
            new_element_indices = tuple(new_indices)

        # Chase parent_array so sliced ArrayValues reached indirectly
        # (via element Value -> parent_array) also get their slice
        # metadata folded.
        new_parent_array = v.parent_array
        if v.parent_array is not None:
            sub_parent = self._substitute_in_value(v.parent_array, folded_values)
            if isinstance(sub_parent, ArrayValue) and sub_parent is not v.parent_array:
                new_parent_array = sub_parent
                changed = True

        if isinstance(v, ArrayValue):
            new_shape: tuple[Value, ...] = v.shape
            if v.shape:
                new_shape_list: list[Value] = []
                for dim in v.shape:
                    sub_dim = self._substitute_in_value(dim, folded_values)
                    if sub_dim is not dim:
                        changed = True
                    new_shape_list.append(sub_dim)
                new_shape = tuple(new_shape_list)

            new_slice_of = v.slice_of
            if v.slice_of is not None:
                sub_slice_of = self._substitute_in_value(v.slice_of, folded_values)
                if (
                    isinstance(sub_slice_of, ArrayValue)
                    and sub_slice_of is not v.slice_of
                ):
                    new_slice_of = sub_slice_of
                    changed = True

            new_slice_start = v.slice_start
            if v.slice_start is not None:
                sub_slice_start = self._substitute_in_value(
                    v.slice_start, folded_values
                )
                if sub_slice_start is not v.slice_start:
                    new_slice_start = sub_slice_start
                    changed = True

            new_slice_step = v.slice_step
            if v.slice_step is not None:
                sub_slice_step = self._substitute_in_value(v.slice_step, folded_values)
                if sub_slice_step is not v.slice_step:
                    new_slice_step = sub_slice_step
                    changed = True

            if changed:
                return dataclasses.replace(
                    v,
                    element_indices=new_element_indices,
                    parent_array=new_parent_array,
                    shape=new_shape,
                    slice_of=new_slice_of,
                    slice_start=new_slice_start,
                    slice_step=new_slice_step,
                )
            return v

        if changed:
            return dataclasses.replace(
                v,
                element_indices=new_element_indices,
                parent_array=new_parent_array,
            )
        return v

    def _substitute_folded_results(
        self,
        op: Operation,
        folded_values: dict[str, Value],
    ) -> Operation:
        """Substitute folded constants in an operation's result Values.

        Complements :meth:`_substitute_folded_operands`: that method
        folds the inputs to an operation while this one folds the
        outputs.  Both are needed for operations whose result
        ``ArrayValue`` carries a ``shape`` derived from BinOp nodes
        that have since been folded.  The canonical case is
        ``MeasureVectorOperation``: its clbit-result array has
        ``shape=(slice_length,)`` where ``slice_length`` was a
        symbolic expression; after the corresponding BinOps are
        removed and their UUIDs land in ``folded_values``, the result
        shape still references the original symbolic ``Value`` unless
        this pass walks it.

        Note: ``SliceArrayOperation`` is **not** handled here because
        it returns early from
        :meth:`_substitute_slice_op_result` which already covers
        both operands and results.

        Args:
            op: Operation whose results to fold.
            folded_values: UUID → folded ``Value`` map built by the pass.

        Returns:
            A new ``Operation`` with folded results, or ``op`` unchanged
            if no result field required substitution.
        """
        if not op.results:
            return op

        new_results: list[Any] = []
        changed = False
        for r in op.results:
            if isinstance(r, Value):
                new_r = self._substitute_in_value(r, folded_values)
                if new_r is not r:
                    changed = True
                new_results.append(new_r)
            else:
                new_results.append(r)

        if changed:
            return dataclasses.replace(op, results=new_results)
        return op

    def _substitute_folded_operands(
        self,
        op: Operation,
        folded_values: dict[str, Value],
    ) -> Operation:
        """Substitute folded constant values in operation operands.

        Also propagates folded values into ``element_indices`` of Value
        operands (recursively, so nested array accesses like
        ``q[indices[uint_tmp]]`` are fully resolved).

        For ``ControlledUOperation``, also folds ``num_controls`` and
        ``control_indices`` fields.
        """
        new_operands: list[Any] = []
        changed = False

        for operand in op.operands:
            if isinstance(operand, ValueBase) and operand.uuid in folded_values:
                new_operands.append(folded_values[operand.uuid])
                changed = True
            elif isinstance(operand, Value):
                new_operand = self._substitute_in_value(operand, folded_values)
                if new_operand is not operand:
                    new_operands.append(new_operand)
                    changed = True
                else:
                    new_operands.append(operand)
            else:
                new_operands.append(operand)

        result_op = dataclasses.replace(op, operands=new_operands) if changed else op

        # Fold ControlledUOperation-specific dataclass fields per subclass.
        if isinstance(result_op, ControlledUOperation):
            extra_kwargs: dict[str, Any] = {}

            # Fold power (shared across all subclasses).
            if isinstance(result_op.power, Value):
                new_power = self._resolve_power_field_value(
                    result_op.power, folded_values
                )
                if new_power is not result_op.power:
                    extra_kwargs["power"] = new_power
                    changed = True

            if isinstance(result_op, SymbolicControlledU):
                # Always fold the ``control_indices`` Value list first
                # so the IR carries constant ints when the bindings
                # supply them.
                if result_op.control_indices is not None:
                    new_ci = self._fold_value_list(
                        list(result_op.control_indices), folded_values
                    )
                    if new_ci is not None:
                        extra_kwargs["control_indices"] = tuple(new_ci)
                        changed = True
                # Fold num_controls: Value -> int.  If resolved to int,
                # consider promoting to ConcreteControlledU.
                new_nc = self._resolve_field_value(
                    result_op.num_controls, folded_values
                )
                if new_nc is not result_op.num_controls:
                    if (
                        isinstance(new_nc, int)
                        and result_op.control_indices is None
                        and result_op.num_control_args == 1
                    ):
                        # Promote to ConcreteControlledU.
                        #
                        # ``SymbolicControlledU`` carries the controls as a
                        # single ``Vector[Qubit]`` operand (``operands[0]``)
                        # plus individual target / param operands.  The
                        # promoted ``ConcreteControlledU`` expects the
                        # operand layout ``[ctrl_0, ..., ctrl_{nc-1}, tgt_0,
                        # ..., params...]`` with one ``Value`` per control
                        # qubit.  Expand both operands and results so the
                        # layout matches the concrete subclass; without
                        # this step ``control_operands`` aliases the first
                        # target into the control slice and the emit path
                        # produces a partial-arity controlled gate.
                        #
                        # Skipped when ``control_indices`` is non-``None``
                        # because the pass-through semantics of non-selected
                        # pool elements cannot be represented in the
                        # promoted ``ConcreteControlledU`` operand layout
                        # (no scalar slot stands for "this slot is part of
                        # the control register but is not actually a
                        # control on this op"); the
                        # ``emit_controlled_u_with_symbolic_indices`` emit
                        # path consumes the un-promoted form directly.
                        current_operands = (
                            new_operands if changed else list(result_op.operands)
                        )
                        current_results = list(result_op.results)
                        new_operands, new_results = (
                            self._expand_symbolic_controlled_operands(
                                current_operands, current_results, new_nc
                            )
                        )
                        changed = True
                        power = extra_kwargs.get("power", result_op.power)
                        result_op = ConcreteControlledU(
                            operands=new_operands,
                            results=new_results,
                            num_controls=new_nc,
                            power=power,
                            block=result_op.block,
                            callable_ref=result_op.callable_ref,
                        )
                        extra_kwargs = {}  # Already applied
                    else:
                        if isinstance(new_nc, int):
                            # ``SymbolicControlledU.num_controls`` is
                            # contractually a ``Value`` with a UUID
                            # (serialize, if-lowering, and the IR
                            # walkers depend on it).  When the
                            # constant fold resolves it to an ``int``
                            # but promotion to ``ConcreteControlledU``
                            # is blocked (e.g. ``control_indices``
                            # is set), bind the constant onto a fresh
                            # copy of the original ``UInt`` ``Value``
                            # so the IR shape stays
                            # ``Value(..., const_value=<int>)`` rather
                            # than a bare ``int``.
                            new_nc = result_op.num_controls.with_const(new_nc)
                        extra_kwargs["num_controls"] = new_nc
                        changed = True
            # ConcreteControlledU: num_controls is already int, nothing to fold.

            if extra_kwargs:
                result_op = dataclasses.replace(result_op, **extra_kwargs)

        # theta is now part of operands, so the operands substitution
        # above already handles it.  No GateOperation-specific code needed.

        if changed:
            return dataclasses.replace(result_op, operands=new_operands)
        return result_op

    def _resolve_field_value(
        self,
        value: Value,
        folded_values: dict[str, Value],
    ) -> Value | int:
        """Resolve a Value in a ControlledUOperation field.

        Returns a concrete ``int`` if resolvable, or the original
        ``Value`` unchanged.
        """
        if value.uuid in folded_values:
            folded = folded_values[value.uuid]
            const = folded.get_const()
            if const is not None:
                return int(const)
            return folded

        resolved = self._resolve_value(value, folded_values)
        if resolved is not None:
            return int(resolved)

        return value

    def _resolve_power_field_value(
        self,
        value: Value,
        folded_values: dict[str, Value],
    ) -> Value | int:
        """Resolve a ``ControlledUOperation.power`` Value to a concrete ``int``.

        Unlike :meth:`_resolve_field_value`, this uses strict-integer
        semantics: ``bool`` and non-integer ``float`` constants are
        rejected via :meth:`_strict_int_cast` instead of being silently
        coerced through ``int(...)``.

        Args:
            value: The symbolic ``Value`` stored in the ``power`` field.
            folded_values: UUID → folded ``Value`` map built by the pass.

        Returns:
            A concrete ``int`` if *value* can be resolved, or the original
            ``Value`` unchanged.
        """
        if value.uuid in folded_values:
            folded = folded_values[value.uuid]
            const = folded.get_const()
            if const is not None:
                return self._strict_int_cast(const)
            return folded

        resolved = self._resolve_value(value, folded_values)
        if resolved is not None:
            return self._strict_int_cast(resolved)

        return value

    @staticmethod
    def _strict_int_cast(value: object) -> int:
        """Cast *value* to ``int`` with strict validation for power fields.

        Only true integer values (or whole ``float`` like ``4.0``) are
        accepted.  ``bool``, non-integer ``float``, and non-positive
        integers are rejected.

        Args:
            value: The resolved constant to cast.

        Returns:
            A validated positive ``int``.

        Raises:
            ValueError: If *value* is ``bool``, a non-integer ``float``,
                a non-``int`` type, or ``<= 0``.
        """
        if isinstance(value, bool):
            raise ValueError(
                f"ControlledU power must be a positive integer, got bool ({value})."
            )
        if isinstance(value, float):
            if value != int(value):
                raise ValueError(
                    f"ControlledU power must be an integer, "
                    f"got non-integer float {value}."
                )
            value = int(value)
        if not isinstance(value, int):
            raise ValueError(
                f"ControlledU power must be an integer, got {type(value).__name__}."
            )
        if value <= 0:
            raise ValueError(
                f"ControlledU power must be strictly positive, got {value}."
            )
        return value

    def _expand_symbolic_controlled_operands(
        self,
        operands: list[Any],
        results: list[Any],
        num_controls: int,
    ) -> tuple[list[Any], list[Any]]:
        """Expand a ``SymbolicControlledU`` operand/result layout into the
        ``ConcreteControlledU`` per-qubit layout.

        ``SymbolicControlledU`` carries the controls as one
        ``Vector[Qubit]`` operand at index 0; the matching result is a
        new-version ``ArrayValue`` at result index 0.  The promoted
        ``ConcreteControlledU`` expects ``num_controls`` individual qubit
        ``Value`` operands (and matching results) followed by the target
        and parameter slots.  This helper builds those per-qubit
        ``Value``\\ s, anchoring them to the original ``ArrayValue``\\ s
        via ``parent_array`` so the emit-time qubit resolver finds the
        physical qubit through the existing slice / array-element path.

        Args:
            operands (list[Any]): Operand list with ``operands[0]`` set to
                the control ``ArrayValue``; the remaining entries are the
                target qubits and classical parameters in their original
                order.
            results (list[Any]): Result list whose first entry is the
                next-version control ``ArrayValue``; the remaining entries
                are the target output qubits in their original order.
            num_controls (int): Concrete (folded) number of control
                qubits.  The control vector's length is assumed to match.

        Returns:
            tuple[list[Any], list[Any]]: A pair ``(new_operands,
            new_results)`` carrying the expanded per-qubit layout.

        Raises:
            ValidationError: If the operand or result layout does not
                match the ``SymbolicControlledU`` contract (missing
                control ``ArrayValue`` at index 0), or if the control
                ``Vector``'s length resolves to a concrete value that
                disagrees with *num_controls*.
        """
        from qamomile.circuit.ir.types.primitives import QubitType, UIntType
        from qamomile.circuit.ir.value import ArrayValue

        if not operands or not isinstance(operands[0], ArrayValue):
            raise ValidationError(
                "SymbolicControlledU expected an ArrayValue at operands[0] "
                "(the control Vector), but the layout did not match.  This "
                "is a compiler bug; the SymbolicControlledU contract was "
                "violated upstream of ConstantFoldingPass."
            )
        if not results or not isinstance(results[0], ArrayValue):
            raise ValidationError(
                "SymbolicControlledU expected an ArrayValue at results[0] "
                "(the next-version control Vector), but the layout did "
                "not match.  This is a compiler bug; the "
                "SymbolicControlledU contract was violated upstream of "
                "ConstantFoldingPass."
            )

        ctrl_vector_in = operands[0]
        ctrl_vector_out = results[0]

        # Verify the control Vector's length matches num_controls when both
        # are resolvable to concrete integers.  Without this check, a
        # mismatch silently uses the first num_controls elements of an
        # over-sized Vector (rest of the Vector is dropped from the circuit)
        # or, for an under-sized Vector, triggers a downstream allocator
        # AssertionError whose surface message ("QInit was not allocated"
        # — see ``_allocate_qubit_list``) hides the real cause.
        if ctrl_vector_in.shape:
            shape_const = ctrl_vector_in.shape[0].get_const()
            if shape_const is not None:
                vector_len = int(shape_const)
                if vector_len != num_controls:
                    raise ValidationError(
                        f"SymbolicControlledU: control Vector "
                        f"'{ctrl_vector_in.name}' has length {vector_len}, "
                        f"but num_controls resolves to {num_controls}.  The "
                        f"Vector passed as the first argument to a "
                        f"symbolic-num_controls controlled gate must hold "
                        f"exactly num_controls qubits."
                    )

        tail_operands = list(operands[1:])
        tail_results = list(results[1:])

        ctrl_operands: list[Any] = []
        ctrl_results: list[Any] = []
        for i in range(num_controls):
            idx_value = Value(
                type=UIntType(),
                name=f"const_{i}",
            ).with_const(i)
            ctrl_operands.append(
                Value(
                    type=QubitType(),
                    name=f"{ctrl_vector_in.name}[{i}]",
                    parent_array=ctrl_vector_in,
                    element_indices=(idx_value,),
                )
            )
            ctrl_results.append(
                Value(
                    type=QubitType(),
                    name=f"{ctrl_vector_out.name}[{i}]",
                    parent_array=ctrl_vector_out,
                    element_indices=(idx_value,),
                )
            )

        return ctrl_operands + tail_operands, ctrl_results + tail_results

    def _fold_value_list(
        self,
        values: list[Value],
        folded_values: dict[str, Value],
    ) -> list[Value] | None:
        """Fold a list of Values (e.g. target_indices).

        Returns a new list if any element changed, or ``None`` if unchanged.
        """
        new_values: list[Value] = []
        list_changed = False
        for v in values:
            if v.uuid in folded_values:
                new_values.append(folded_values[v.uuid])
                list_changed = True
            else:
                resolved = self._resolve_value(v, folded_values)
                if resolved is not None:
                    new_values.append(
                        Value(
                            type=v.type,
                            name=f"folded_{v.name}",
                            uuid=v.uuid,
                        ).with_const(int(resolved))
                    )
                    list_changed = True
                else:
                    new_values.append(v)
        return new_values if list_changed else None
