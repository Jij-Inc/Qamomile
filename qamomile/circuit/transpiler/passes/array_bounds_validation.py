"""Reject reachable compile-time array accesses outside resolved extents."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Sequence
from typing import Any, cast

import numpy as np

from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.arithmetic_operations import (
    BinOp,
    CompOp,
    CondOp,
    NotOp,
    UnaryMathOp,
)
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    HasNestedOps,
    IfOperation,
    genuine_input_values,
)
from qamomile.circuit.ir.operation.gate import ControlledUOperation
from qamomile.circuit.ir.operation.select import SelectOperation
from qamomile.circuit.ir.operation.slice_array import SliceArrayOperation
from qamomile.circuit.ir.value import (
    ArrayValue,
    DictValue,
    TupleValue,
    Value,
    ValueBase,
)
from qamomile.circuit.transpiler.block_parameter_binding import pair_block_operands
from qamomile.circuit.transpiler.errors import ValidationError
from qamomile.circuit.transpiler.passes import Pass
from qamomile.circuit.transpiler.passes.control_flow_reachability import (
    MAX_STATIC_REPLAY_TRIPS,
    constant_integer,
    reachable_nested_regions,
    same_exact_typed_constant,
    static_for_items_entries,
    static_for_range,
)
from qamomile.circuit.transpiler.passes.eval_utils import (
    FoldPolicy,
    fold_classical_op,
)
from qamomile.circuit.transpiler.passes.value_mapping import ValueSubstitutor
from qamomile.circuit.transpiler.value_resolver import ValueResolver


def _root_array(array: ArrayValue) -> ArrayValue:
    """Return the root container beneath a possibly nested array view.

    Args:
        array (ArrayValue): Root array or sliced view.

    Returns:
        ArrayValue: Last acyclic parent in the ``slice_of`` chain.
    """
    current = array
    seen: set[str] = set()
    while current.slice_of is not None and current.uuid not in seen:
        seen.add(current.uuid)
        current = current.slice_of
    return current


class ArrayBoundsValidationPass(Pass[Block, Block]):
    """Reject reachable element accesses and views outside array bounds.

    This pass runs after partial evaluation has resolved binding-dependent
    slice extents and before declarative slice operations are stripped. It
    deliberately skips statically zero-trip loop bodies so an unreachable
    access does not become a false-positive compilation error. Exact loop
    replay is capped by ``MAX_STATIC_REPLAY_TRIPS``; the conservative fallback
    validates one reachable body instance, including first-iteration constants
    when available, and never publishes speculative final results.
    """

    _MAX_REPLAY_TRIPS = MAX_STATIC_REPLAY_TRIPS

    @property
    def name(self) -> str:
        """Return the stable pass identifier.

        Returns:
            str: The name ``"array_bounds_validation"``.
        """
        return "array_bounds_validation"

    def run(self, input: Block) -> Block:
        """Validate reachable array element operands in one semantic block.

        Args:
            input (Block): Post-partial-evaluation affine or hierarchical
                block whose concrete array extents should be checked.

        Returns:
            Block: ``input`` unchanged when every reachable access and view is
                valid or still symbolic.

        Raises:
            ValidationError: If ``input`` has an unsupported block kind, a
                reachable constant index is outside a resolved array extent,
                or a concrete view descriptor exceeds its physical root.
        """
        if input.kind not in (BlockKind.AFFINE, BlockKind.HIERARCHICAL):
            raise ValidationError(
                "ArrayBoundsValidationPass expects AFFINE or HIERARCHICAL "
                f"block, got {input.kind}",
            )
        self._remaining_replay_trips = self._MAX_REPLAY_TRIPS
        self._walk_block(
            input,
            owned_blocks_on_path=frozenset({id(input)}),
        )
        return input

    def _walk_block(
        self,
        block: Block,
        *,
        substitutor: ValueSubstitutor | None = None,
        owned_blocks_on_path: frozenset[int],
    ) -> None:
        """Validate reachable operations and public outputs in one block.

        Args:
            block (Block): Entry or operation-owned block to inspect.
            substitutor (ValueSubstitutor | None): Formal-to-actual mapping for
                an operation-owned block. Defaults to ``None`` in the entry
                block.
            owned_blocks_on_path (frozenset[int]): Block object identities on
                the active recursion path.

        Raises:
            ValidationError: If a reachable operation input or block output
                has a concrete out-of-bounds array index.
        """
        substitutors = () if substitutor is None else (substitutor,)
        known_values: dict[str, ValueBase] = {}
        self._walk_operations(
            block.operations,
            substitutors=substitutors,
            known_values=known_values,
            owned_blocks_on_path=owned_blocks_on_path,
        )
        self._validate_substituted_values(
            block.output_values,
            substitutors,
            known_values,
        )

    def _walk_operations(
        self,
        operations: list[Operation],
        *,
        substitutors: tuple[ValueSubstitutor, ...],
        known_values: dict[str, ValueBase],
        owned_blocks_on_path: frozenset[int],
    ) -> None:
        """Walk reachable operations and validate their direct operands.

        Args:
            operations (list[Operation]): Operations in the current reachable
                control-flow region.
            substitutors (tuple[ValueSubstitutor, ...]): Outer-scope
                formal-to-actual and iteration substitutions, in application
                order.
            known_values (dict[str, ValueBase]): Constants learned while
                replaying the current reachable operation sequence.
            owned_blocks_on_path (frozenset[int]): Block object identities on
                the active recursion path.

        Raises:
            ValidationError: If a reachable operand has a concrete
                out-of-bounds array index.
        """
        for operation in operations:
            current = self._substitute_operation(
                operation,
                substitutors,
                known_values,
            )
            if isinstance(current, IfOperation):
                self._walk_if_operation(
                    current,
                    substitutors=substitutors,
                    known_values=known_values,
                    owned_blocks_on_path=owned_blocks_on_path,
                )
                continue
            if isinstance(current, ForOperation):
                self._walk_for_operation(
                    current,
                    substitutors=substitutors,
                    known_values=known_values,
                    owned_blocks_on_path=owned_blocks_on_path,
                )
                continue
            if isinstance(current, ForItemsOperation):
                self._walk_for_items_operation(
                    current,
                    substitutors=substitutors,
                    known_values=known_values,
                    owned_blocks_on_path=owned_blocks_on_path,
                )
                continue

            self._validate_values(genuine_input_values(current))
            if isinstance(current, SliceArrayOperation):
                for result in current.results:
                    if isinstance(result, ArrayValue):
                        self._validate_array_view(result)
            if isinstance(current, HasNestedOps):
                for region in reachable_nested_regions(current):
                    child_values = dict(known_values)
                    self._validate_substituted_values(
                        region.captures,
                        substitutors,
                        child_values,
                    )
                    self._walk_operations(
                        list(region.operations),
                        substitutors=substitutors,
                        known_values=child_values,
                        owned_blocks_on_path=owned_blocks_on_path,
                    )
                    self._validate_substituted_values(
                        region.yields,
                        substitutors,
                        child_values,
                    )
            self._walk_owned_blocks(current, owned_blocks_on_path)
            self._record_folded_result(
                operation,
                current,
                known_values,
            )

    def _walk_if_operation(
        self,
        operation: IfOperation,
        *,
        substitutors: tuple[ValueSubstitutor, ...],
        known_values: dict[str, ValueBase],
        owned_blocks_on_path: frozenset[int],
    ) -> None:
        """Validate only branches and yields reachable from one conditional.

        Args:
            operation (IfOperation): Conditional whose condition, branch
                yields, and nested operations should be checked.
            substitutors (tuple[ValueSubstitutor, ...]): Outer-scope
                substitutions active for this conditional.
            known_values (dict[str, ValueBase]): Constants learned in the
                enclosing operation sequence.
            owned_blocks_on_path (frozenset[int]): Block object identities on
                the active recursion path.

        Raises:
            ValidationError: If a reachable array access is out of bounds.
        """
        self._validate_values(operation.operands)
        self._validate_values(rebind.before for rebind in operation.branch_rebinds)

        regions = operation.nested_regions()
        condition = self._resolve_replay_value(operation.condition, (), {})
        if isinstance(condition, Value) and condition.is_constant():
            branch_index = 0 if bool(condition.get_const()) else 1
            region = regions[branch_index]
            branch_values = dict(known_values)
            self._validate_substituted_values(
                region.captures,
                substitutors,
                branch_values,
            )
            self._walk_operations(
                list(region.operations),
                substitutors=substitutors,
                known_values=branch_values,
                owned_blocks_on_path=owned_blocks_on_path,
            )
            self._validate_substituted_values(
                region.yields,
                substitutors,
                branch_values,
            )
            for merge in operation.iter_merges():
                known_values[merge.result.uuid] = self._resolve_replay_value(
                    merge.select(branch_index == 0),
                    substitutors,
                    branch_values,
                )
            return

        branch_values_list: list[dict[str, ValueBase]] = []
        for region in regions:
            branch_values = dict(known_values)
            self._validate_substituted_values(
                region.captures,
                substitutors,
                branch_values,
            )
            self._walk_operations(
                list(region.operations),
                substitutors=substitutors,
                known_values=branch_values,
                owned_blocks_on_path=owned_blocks_on_path,
            )
            self._validate_substituted_values(
                region.yields,
                substitutors,
                branch_values,
            )
            branch_values_list.append(branch_values)

        for merge in operation.iter_merges():
            true_value = self._resolve_replay_value(
                merge.true_value,
                substitutors,
                branch_values_list[0],
            )
            false_value = self._resolve_replay_value(
                merge.false_value,
                substitutors,
                branch_values_list[1],
            )
            if true_value.uuid == false_value.uuid:
                known_values[merge.result.uuid] = true_value
                continue
            if (
                isinstance(true_value, Value)
                and isinstance(false_value, Value)
                and same_exact_typed_constant(true_value, false_value)
            ):
                replacement = self._constant_replacement(
                    merge.result,
                    true_value.get_const(),
                )
                if replacement is not None:
                    known_values[merge.result.uuid] = replacement

    def _walk_for_operation(
        self,
        operation: ForOperation,
        *,
        substitutors: tuple[ValueSubstitutor, ...],
        known_values: dict[str, ValueBase],
        owned_blocks_on_path: frozenset[int],
    ) -> None:
        """Validate a counted loop without inspecting a zero-trip body.

        Loop bounds, carried initializers, and pre-loop rebind values are read
        even when the loop is empty. Body operations and yielded/rebound body
        values are reachable only when the loop may execute.

        Args:
            operation (ForOperation): Counted loop to inspect.
            substitutors (tuple[ValueSubstitutor, ...]): Outer-scope
                substitutions active for this loop.
            known_values (dict[str, ValueBase]): Constants learned in the
                enclosing operation sequence.
            owned_blocks_on_path (frozenset[int]): Block object identities on
                the active recursion path.

        Raises:
            ValidationError: If a reachable array access is out of bounds.
        """
        self._validate_values(operation.operands)
        self._validate_values(region_arg.init for region_arg in operation.region_args)
        self._validate_values(
            rebind.before for rebind in operation.loop_carried_rebinds
        )
        resolved_bounds = [
            cast(Value, self._resolve_replay_value(bound, (), {}))
            for bound in operation.operands[:3]
        ]
        resolved_operation = dataclasses.replace(
            operation,
            operands=[*resolved_bounds, *operation.operands[3:]],
        )
        iteration_range = static_for_range(resolved_operation)
        iteration_bindings: Iterable[dict[str, ValueBase]] | None = None
        trip_count: int | None = None
        if iteration_range is not None:
            try:
                trip_count = len(iteration_range)
            except OverflowError:
                trip_count = None
            if trip_count is not None:
                if operation.loop_var_value is not None:
                    iteration_bindings = (
                        {
                            operation.loop_var_value.uuid: (
                                operation.loop_var_value.with_const(loop_value)
                            )
                        }
                        for loop_value in iteration_range
                    )
                else:
                    iteration_bindings = ({} for _ in iteration_range)
        self._walk_loop_region(
            operation,
            iteration_bindings=iteration_bindings,
            trip_count=trip_count,
            substitutors=substitutors,
            known_values=known_values,
            owned_blocks_on_path=owned_blocks_on_path,
        )

    def _walk_for_items_operation(
        self,
        operation: ForItemsOperation,
        *,
        substitutors: tuple[ValueSubstitutor, ...],
        known_values: dict[str, ValueBase],
        owned_blocks_on_path: frozenset[int],
    ) -> None:
        """Validate an items loop with exact replay when entries are bound.

        Args:
            operation (ForItemsOperation): Items loop to inspect.
            substitutors (tuple[ValueSubstitutor, ...]): Outer-scope
                substitutions active for this loop.
            known_values (dict[str, ValueBase]): Constants learned in the
                enclosing operation sequence.
            owned_blocks_on_path (frozenset[int]): Block object identities on
                the active recursion path.

        Raises:
            ValidationError: If a reachable array access is out of bounds.
        """
        self._validate_values(operation.operands)
        self._validate_values(region_arg.init for region_arg in operation.region_args)
        self._validate_values(
            rebind.before for rebind in operation.loop_carried_rebinds
        )
        entries = static_for_items_entries(operation)
        iteration_bindings = (
            None
            if entries is None
            else (
                self._for_items_iteration_bindings(operation, key, value)
                for key, value in entries
            )
        )
        self._walk_loop_region(
            operation,
            iteration_bindings=iteration_bindings,
            trip_count=None if entries is None else len(entries),
            substitutors=substitutors,
            known_values=known_values,
            owned_blocks_on_path=owned_blocks_on_path,
        )

    def _walk_loop_region(
        self,
        operation: ForOperation | ForItemsOperation,
        *,
        iteration_bindings: Iterable[dict[str, ValueBase]] | None,
        trip_count: int | None,
        substitutors: tuple[ValueSubstitutor, ...],
        known_values: dict[str, ValueBase],
        owned_blocks_on_path: frozenset[int],
    ) -> None:
        """Validate and replay one counted or items loop region.

        Args:
            operation (ForOperation | ForItemsOperation): Loop to inspect.
            iteration_bindings (Iterable[dict[str, ValueBase]] | None): Exact
                induction or item values for each reachable iteration, or
                ``None`` when the trip sequence remains symbolic.
            trip_count (int | None): Exact cardinality aligned with
                ``iteration_bindings``, or ``None`` when unresolved.
            substitutors (tuple[ValueSubstitutor, ...]): Outer-scope
                substitutions active for the loop.
            known_values (dict[str, ValueBase]): Constants learned in the
                enclosing operation sequence; loop results are published here.
            owned_blocks_on_path (frozenset[int]): Block object identities on
                the active recursion path.

        Raises:
            ValidationError: If a reachable array access is out of bounds.
        """
        region = operation.nested_regions()[0]
        initial_values = [
            self._resolve_replay_value(region_arg.init, substitutors, known_values)
            for region_arg in operation.region_args
        ]
        if trip_count == 0:
            for region_arg, initial in zip(
                operation.region_args,
                initial_values,
                strict=True,
            ):
                known_values[region_arg.result.uuid] = initial
            return

        over_budget = (
            iteration_bindings is not None
            and trip_count is not None
            and trip_count > self._remaining_replay_trips
        )
        if over_budget:
            assert iteration_bindings is not None
            first_bindings = next(iter(iteration_bindings))
            self._replay_loop_iteration(
                operation,
                bindings=first_bindings,
                carried=initial_values,
                substitutors=substitutors,
                known_values=known_values,
                owned_blocks_on_path=owned_blocks_on_path,
            )
            self._validate_values(
                rebind.after for rebind in operation.loop_carried_rebinds
            )
            return
        if iteration_bindings is not None and trip_count is not None:
            self._remaining_replay_trips -= trip_count

        if iteration_bindings is None:
            child_values = dict(known_values)
            self._validate_substituted_values(
                region.captures,
                substitutors,
                child_values,
            )
            self._walk_operations(
                list(region.operations),
                substitutors=substitutors,
                known_values=child_values,
                owned_blocks_on_path=owned_blocks_on_path,
            )
            self._validate_substituted_values(
                region.yields,
                substitutors,
                child_values,
            )
            self._validate_values(
                rebind.after for rebind in operation.loop_carried_rebinds
            )
            return

        carried = initial_values
        for bindings in iteration_bindings:
            carried = self._replay_loop_iteration(
                operation,
                bindings=bindings,
                carried=carried,
                substitutors=substitutors,
                known_values=known_values,
                owned_blocks_on_path=owned_blocks_on_path,
            )

        self._validate_values(rebind.after for rebind in operation.loop_carried_rebinds)
        for region_arg, final_value in zip(
            operation.region_args,
            carried,
            strict=True,
        ):
            known_values[region_arg.result.uuid] = final_value

    def _replay_loop_iteration(
        self,
        operation: ForOperation | ForItemsOperation,
        *,
        bindings: dict[str, ValueBase],
        carried: Sequence[ValueBase],
        substitutors: tuple[ValueSubstitutor, ...],
        known_values: dict[str, ValueBase],
        owned_blocks_on_path: frozenset[int],
    ) -> list[ValueBase]:
        """Validate one concrete loop iteration and resolve its carried yields.

        Args:
            operation (ForOperation | ForItemsOperation): Loop owning the body
                and carried region arguments.
            bindings (dict[str, ValueBase]): Concrete induction or item values
                for this iteration.
            carried (Sequence[ValueBase]): Values entering the carried region
                arguments for this iteration.
            substitutors (tuple[ValueSubstitutor, ...]): Outer-scope
                substitutions active for the loop.
            known_values (dict[str, ValueBase]): Constants learned before the
                loop.
            owned_blocks_on_path (frozenset[int]): Block object identities on
                the active recursion path.

        Returns:
            list[ValueBase]: Resolved values yielded to the next iteration.

        Raises:
            ValidationError: If this reachable iteration contains an
                out-of-bounds array access.
        """
        region = operation.nested_regions()[0]
        iteration_values = dict(known_values)
        iteration_values.update(bindings)
        iteration_values.update(
            (region_arg.block_arg.uuid, value)
            for region_arg, value in zip(
                operation.region_args,
                carried,
                strict=True,
            )
        )
        self._validate_substituted_values(
            region.captures,
            substitutors,
            iteration_values,
        )
        self._walk_operations(
            list(region.operations),
            substitutors=substitutors,
            known_values=iteration_values,
            owned_blocks_on_path=owned_blocks_on_path,
        )
        self._validate_substituted_values(
            region.yields,
            substitutors,
            iteration_values,
        )
        return [
            self._resolve_replay_value(
                region_arg.yielded,
                substitutors,
                iteration_values,
            )
            for region_arg in operation.region_args
        ]

    def _for_items_iteration_bindings(
        self,
        operation: ForItemsOperation,
        key: Any,
        value: Any,
    ) -> dict[str, ValueBase]:
        """Build substitutions for one statically bound item.

        Args:
            operation (ForItemsOperation): Items loop owning the iteration
                identities.
            key (Any): Concrete mapping key for this iteration.
            value (Any): Concrete mapping value for this iteration.

        Returns:
            dict[str, ValueBase]: UUID-keyed replacements for scalar key/value
                formals and one-dimensional vector-key contents and length.
        """
        bindings: dict[str, ValueBase] = {}
        key_values = operation.key_var_values or ()
        if (
            operation.key_is_vector
            and len(key_values) == 1
            and isinstance(key_values[0], ArrayValue)
            and isinstance(key, (tuple, list))
        ):
            identity = key_values[0]
            bound_key = identity.with_array_runtime_metadata(const_array=tuple(key))
            bindings[identity.uuid] = bound_key
            if identity.shape:
                bindings[identity.shape[0].uuid] = identity.shape[0].with_const(
                    len(key)
                )
        elif not operation.key_is_vector:
            key_parts: tuple[Any, ...]
            if len(key_values) == 1:
                key_parts = (key,)
            elif isinstance(key, (tuple, list)) and len(key) == len(key_values):
                key_parts = tuple(key)
            else:
                key_parts = ()
            for identity, item in zip(key_values, key_parts, strict=False):
                replacement = self._constant_replacement(identity, item)
                if replacement is not None:
                    bindings[identity.uuid] = replacement
        if operation.value_var_value is not None:
            replacement = self._constant_replacement(
                operation.value_var_value,
                value,
            )
            if replacement is not None:
                bindings[operation.value_var_value.uuid] = replacement
        return bindings

    def _substitute_operation(
        self,
        operation: Operation,
        substitutors: tuple[ValueSubstitutor, ...],
        known_values: dict[str, ValueBase],
    ) -> Operation:
        """Apply outer and replay substitutions to one operation.

        Args:
            operation (Operation): Operation to rewrite for validation.
            substitutors (tuple[ValueSubstitutor, ...]): Stable outer-scope
                substitutions in application order.
            known_values (dict[str, ValueBase]): Mutable replay values known at
                this operation's program point.

        Returns:
            Operation: Rewritten operation used only by this validation pass.
        """
        current = operation
        for substitutor in substitutors:
            current = substitutor.substitute_operation(current)
        return ValueSubstitutor(
            known_values,
            transitive=True,
        ).substitute_operation(current)

    def _resolve_replay_value(
        self,
        value: ValueBase,
        substitutors: tuple[ValueSubstitutor, ...],
        known_values: dict[str, ValueBase],
    ) -> ValueBase:
        """Substitute and concretize one value in a replay environment.

        Args:
            value (ValueBase): Value to resolve.
            substitutors (tuple[ValueSubstitutor, ...]): Stable outer-scope
                substitutions in application order.
            known_values (dict[str, ValueBase]): Replay values known at the
                current program point.

        Returns:
            ValueBase: Substituted value, carrying constant metadata when its
                scalar payload is compile-time resolvable.
        """
        current = value
        for substitutor in substitutors:
            current = substitutor.substitute_value(current)
        current = ValueSubstitutor(
            known_values,
            transitive=True,
        ).substitute_value(current)
        if not isinstance(current, Value):
            return current
        resolved = ValueResolver().resolve(current)
        replacement = self._constant_replacement(current, resolved)
        return current if replacement is None else replacement

    def _validate_substituted_values(
        self,
        values: Iterable[ValueBase],
        substitutors: tuple[ValueSubstitutor, ...],
        known_values: dict[str, ValueBase],
    ) -> None:
        """Validate values after applying a replay environment.

        Args:
            values (Iterable[ValueBase]): Values to rewrite and inspect.
            substitutors (tuple[ValueSubstitutor, ...]): Stable outer-scope
                substitutions in application order.
            known_values (dict[str, ValueBase]): Replay values known at the
                current program point.

        Raises:
            ValidationError: If a substituted array access is out of bounds.
        """
        self._validate_values(
            self._resolve_replay_value(value, substitutors, known_values)
            for value in values
        )

    def _record_folded_result(
        self,
        original: Operation,
        current: Operation,
        known_values: dict[str, ValueBase],
    ) -> None:
        """Record one replayable classical operation result.

        Args:
            original (Operation): Operation before validation substitutions.
            current (Operation): Operation after validation substitutions.
            known_values (dict[str, ValueBase]): Replay environment to update.

        Returns:
            None: Mutates ``known_values`` when ``current`` folds to a scalar.
        """
        if not isinstance(current, (BinOp, CompOp, CondOp, NotOp, UnaryMathOp)):
            return
        if not original.results or not current.results:
            return
        folded = fold_classical_op(
            current,
            ValueResolver().resolve,
            set(),
            FoldPolicy.COMPILE_TIME,
        )
        replacement = self._constant_replacement(current.results[0], folded)
        if replacement is None:
            return
        known_values[original.results[0].uuid] = replacement
        known_values[current.results[0].uuid] = replacement

    def _constant_replacement(
        self,
        template: ValueBase,
        payload: Any,
    ) -> Value | None:
        """Attach one supported Python scalar to an IR value identity.

        Args:
            template (ValueBase): Value whose identity and type should be
                preserved.
            payload (Any): Candidate resolved scalar payload.

        Returns:
            Value | None: Constant replacement, or ``None`` for structured,
                missing, or unsupported payloads.
        """
        if not isinstance(template, Value) or payload is None:
            return None
        if isinstance(payload, np.generic):
            payload = payload.item()
        if not isinstance(payload, (bool, int, float)):
            return None
        return template.with_const(payload)

    def _walk_owned_blocks(
        self,
        operation: Operation,
        owned_blocks_on_path: frozenset[int],
    ) -> None:
        """Validate SELECT cases and controlled-unitary bodies at their call site.

        Operation-owned blocks have independent value namespaces. Their formal
        inputs are therefore substituted with the owning operation's actual
        operands before bounds and reachability are inspected. Each shared
        block is revisited for each call site, while the path-local identity
        guard prevents malformed cyclic block graphs from recursing forever.

        Args:
            operation (Operation): Operation that may own nested blocks.
            owned_blocks_on_path (frozenset[int]): Block object identities on
                the active recursion path.

        Raises:
            ValidationError: If a reachable owned-body access is out of bounds.
        """
        owned_blocks: list[tuple[Block, Sequence[ValueBase]]] = []
        if isinstance(operation, SelectOperation):
            actuals = [*operation.target_operands, *operation.param_operands]
            owned_blocks.extend((block, actuals) for block in operation.case_blocks)
        elif (
            isinstance(operation, ControlledUOperation) and operation.block is not None
        ):
            targets = [
                value for value in operation.target_operands if value.type.is_quantum()
            ]
            actuals = [*targets, *operation.param_operands]
            owned_blocks.append((operation.block, actuals))

        for block, actuals in owned_blocks:
            block_id = id(block)
            if block_id in owned_blocks_on_path:
                continue
            mapping: dict[str, ValueBase] = {}
            for formal, actual in pair_block_operands(block, actuals):
                mapping[formal.uuid] = actual
                if isinstance(formal, ArrayValue) and isinstance(actual, ArrayValue):
                    mapping.update(
                        (formal_dimension.uuid, actual_dimension)
                        for formal_dimension, actual_dimension in zip(
                            formal.shape,
                            actual.shape,
                            strict=False,
                        )
                    )
            self._walk_block(
                block,
                substitutor=ValueSubstitutor(mapping, transitive=True),
                owned_blocks_on_path=owned_blocks_on_path | {block_id},
            )

    def _validate_array_view(self, view: ArrayValue) -> None:
        """Validate the concrete physical coverage declared by one view.

        A serialized template can acquire concrete parent and view lengths only
        after binding. When the view descriptor then claims slots beyond its
        parent, a loop-local symbolic index can otherwise hide the invalid
        access until backend emission. Checking both affine endpoints is
        sufficient because supported slice strides are constant and positive;
        checking both also keeps malformed non-positive strides conservative.

        Args:
            view (ArrayValue): Slice result whose declared coverage should fit
                every resolved ancestor extent.

        Raises:
            ValidationError: If a concrete non-empty view extends outside a
                concrete ancestor extent.
        """
        if view.slice_of is None or not view.shape:
            return
        length = constant_integer(view.shape[0])
        if length is None or length <= 0:
            return
        self._validate_leading_coordinate(0, view)
        if length > 1:
            self._validate_leading_coordinate(length - 1, view)

    def _validate_values(self, values: Iterable[ValueBase]) -> None:
        """Validate array-element accesses in one iterable of IR values.

        Args:
            values (Iterable[ValueBase]): Direct operation inputs or semantic
                control-flow values.

        Raises:
            ValidationError: If a value is a concrete out-of-bounds access.
        """
        for value in values:
            self._validate_element_access(value)

    def _validate_element_access(
        self,
        value: ValueBase,
        seen: set[str] | None = None,
    ) -> None:
        """Validate array accesses recursively embedded in one IR value.

        Tuple elements, dictionary entries, array shapes, slice metadata,
        parent arrays, and element indices all carry semantic value references
        outside ordinary operation operands. For the leading coordinate of a
        sliced vector, each view-local index is validated before applying that
        view's affine map to its parent. This prevents a zero-length view from
        borrowing an otherwise valid slot in the root array.

        Args:
            value (ValueBase): Operation operand or structured boundary value
                that may contain array element accesses.
            seen (set[str] | None): Value UUIDs already inspected during this
                recursive metadata walk. Defaults to a fresh set.

        Raises:
            ValidationError: If a concrete index falls outside a concrete
                immediate-view or root-array extent.
        """
        if seen is None:
            seen = set()
        if value.uuid in seen:
            return
        seen.add(value.uuid)

        if isinstance(value, TupleValue):
            for element in value.elements:
                self._validate_element_access(element, seen)
            return
        if isinstance(value, DictValue):
            for key, entry_value in value.entries:
                self._validate_element_access(key, seen)
                self._validate_element_access(entry_value, seen)
            return
        if not isinstance(value, Value):
            return

        if isinstance(value, ArrayValue):
            for dimension in value.shape:
                self._validate_element_access(dimension, seen)
            if value.slice_of is not None:
                self._validate_element_access(value.slice_of, seen)
            if value.slice_start is not None:
                self._validate_element_access(value.slice_start, seen)
            if value.slice_step is not None:
                self._validate_element_access(value.slice_step, seen)
            self._validate_array_view(value)

        parent = value.parent_array
        if parent is not None:
            self._validate_element_access(parent, seen)
        for index_value in value.element_indices:
            self._validate_element_access(index_value, seen)
        if parent is None or not value.element_indices:
            return

        root = _root_array(parent)
        leading_index = self._resolve_integer_index(value.element_indices[0])
        if leading_index is not None:
            self._validate_leading_coordinate(leading_index, parent)

        for dimension, index_value in enumerate(value.element_indices[1:], start=1):
            index = self._resolve_integer_index(index_value)
            if index is None:
                continue
            self._validate_coordinate(
                index,
                parent,
                dimension=dimension,
                accessed_parent=parent,
                root=root,
            )

    @staticmethod
    def _resolve_integer_index(value: Value) -> int | None:
        """Resolve an integer index from constants or bound array metadata.

        Args:
            value (Value): Scalar index value to resolve.

        Returns:
            int | None: Concrete non-boolean integer, or ``None`` when the
                index remains symbolic or non-integral.
        """
        constant = constant_integer(value)
        if constant is not None:
            return constant
        resolved = ValueResolver().resolve(value)
        if isinstance(resolved, (bool, np.bool_)) or not isinstance(
            resolved,
            (int, np.integer),
        ):
            return None
        return int(resolved)

    def _validate_leading_coordinate(
        self,
        index: int,
        parent: ArrayValue,
    ) -> None:
        """Validate and map one leading coordinate through a view chain.

        Args:
            index (int): Concrete coordinate in ``parent`` space.
            parent (ArrayValue): Immediate array or view being accessed.

        Raises:
            ValidationError: If the coordinate falls outside a resolved local
                or ancestor extent.
        """
        root = _root_array(parent)
        current = parent
        current_index = index
        seen: set[str] = set()
        while current.uuid not in seen:
            seen.add(current.uuid)
            self._validate_coordinate(
                current_index,
                current,
                dimension=0,
                accessed_parent=parent,
                root=root,
            )
            if current.slice_of is None:
                break
            start = constant_integer(current.slice_start)
            step = constant_integer(current.slice_step)
            if start is None or step is None:
                break
            current_index = start + step * current_index
            current = current.slice_of

    def _validate_coordinate(
        self,
        index: int,
        bounded_array: ArrayValue,
        *,
        dimension: int,
        accessed_parent: ArrayValue,
        root: ArrayValue,
    ) -> None:
        """Reject one concrete coordinate outside its resolved extent.

        Args:
            index (int): Concrete coordinate in ``bounded_array`` space.
            bounded_array (ArrayValue): Array or view whose dimension bounds
                ``index``.
            dimension (int): Zero-based dimension being checked.
            accessed_parent (ArrayValue): Immediate parent named by the source
                element access.
            root (ArrayValue): Root array beneath ``accessed_parent``.

        Raises:
            ValidationError: If ``index`` is negative or no smaller than the
                resolved dimension extent.
        """
        if dimension >= len(bounded_array.shape):
            return

        root_name = root.name or "<anonymous>"
        parent_name = accessed_parent.name or root_name
        if accessed_parent.slice_of is None:
            subject = f"array '{root_name}'"
        else:
            subject = f"array view '{parent_name}' of root array '{root_name}'"
        if index < 0:
            raise ValidationError(
                f"Index {index} is out of range for {subject} at dimension "
                f"{dimension}. Array indices must be non-negative.",
                value_name=root_name,
            )

        extent_value = bounded_array.shape[dimension]
        extent = constant_integer(extent_value)
        if extent is None or index < extent:
            return

        extent_name = extent_value.name or f"dimension {dimension}"
        view_context = ""
        if (
            accessed_parent.slice_of is not None
            and bounded_array.uuid != accessed_parent.uuid
            and accessed_parent.shape
        ):
            view_extent_value = accessed_parent.shape[0]
            view_extent = constant_integer(view_extent_value)
            if view_extent is not None:
                view_extent_name = view_extent_value.name or "dimension 0"
                view_context = (
                    f"The view extent '{view_extent_name}' resolved to {view_extent}. "
                )
        requirement = (
            f"The extent '{extent_name}' must resolve to at least "
            f"{index + 1} for this access."
        )
        raise ValidationError(
            f"Index {index} is out of range for {subject} at dimension "
            f"{dimension}: extent '{extent_name}' resolved to {extent}. "
            f"{view_context}{requirement}",
            value_name=root_name,
        )


__all__ = ["ArrayBoundsValidationPass"]
