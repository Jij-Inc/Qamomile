"""Provide shared state and scheduling for IR resource interpretation."""

from __future__ import annotations

import dataclasses
import itertools
import math
from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias, cast

import sympy as sp

from qamomile.circuit.estimator._call_liveness import (
    _block_input_allocations,
)
from qamomile.circuit.estimator._constants import (
    _ZERO,
)
from qamomile.circuit.estimator._constraints import (
    _block_input_constraints,
    _block_output_constraints,
)
from qamomile.circuit.estimator._estimate import ResourceEstimate
from qamomile.circuit.estimator._estimate_validation import _with_constraints
from qamomile.circuit.estimator._inputs import (
    _root_input_binding_context,
)
from qamomile.circuit.estimator._interpreter_dataflow import (
    _ResourceInlineBoundaryOperation,
)
from qamomile.circuit.estimator._interpreter_state import (
    _MAX_RUNTIME_DOMAIN_ALTERNATIVES,
    _InterpreterState,
)
from qamomile.circuit.estimator._resolver import (
    ExprResolver,
)
from qamomile.circuit.estimator._resource_algebra import (
    _wrap_trace,
)
from qamomile.circuit.estimator._resource_base import (
    _is_concrete_integer,
    _symbol_display_name,
)
from qamomile.circuit.estimator._resource_constraints import (
    _ResourceConstraint,
)
from qamomile.circuit.estimator._symbol_discovery import (
    _free_symbols,
    _trace_guard_free_symbols,
)
from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.dataflow import (
    has_legacy_scalar_bit_rebinds,
    walk_operations,
)
from qamomile.circuit.ir.operation.callable import (
    InvokeOperation,
)
from qamomile.circuit.ir.operation.control_flow import (
    HasNestedOps,
    WhileOperation,
)
from qamomile.circuit.ir.operation.gate import (
    ControlledUOperation,
)
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.operation.operation import (
    Operation,
)
from qamomile.circuit.ir.operation.select import SelectOperation
from qamomile.circuit.ir.types.primitives import BitType
from qamomile.circuit.ir.value import (
    ArrayValue,
    ValueBase,
    ValueLike,
)
from qamomile.circuit.transpiler.errors import ValidationError
from qamomile.circuit.transpiler.passes.analyze import (
    reject_loop_carried_classical_rebinds,
)
from qamomile.circuit.transpiler.passes.compile_time_if_lowering import (
    lower_compile_time_ifs_preserving_loop_conditions,
)
from qamomile.circuit.transpiler.passes.inline import (
    InlinePass,
    _has_any_inline_call,
)

# Circuit transpilation performs one initial inline pass before its 64-round
# recursion loop. The validation copy starts from the original Block, so it
# needs one additional round to cover the same supported concrete depth.
_MAX_LOOP_VALIDATION_INLINE_DEPTH = 65


_CONCRETE_REGION_REPLAY_LIMIT = 64


_InlineArrayStateBinding: TypeAlias = tuple[
    ArrayValue,
    tuple[ArrayValue, ...],
]
_InlineArrayStateBindings: TypeAlias = tuple[_InlineArrayStateBinding, ...]


class _InterpreterCore(_InterpreterState):
    """Add estimator entry, validation, and inline-view services."""

    def estimate(self, block_or_ops: Block | Sequence[Operation]) -> ResourceEstimate:
        """Estimate resources for a block or operation sequence.

        Args:
            block_or_ops (Block | Sequence[Operation]): IR block or operations.

        Returns:
            ResourceEstimate: Estimated logical resources.

        Raises:
            QubitConsumedError: If compiler-equivalent inlining finds duplicate
                quantum operands at one call site.
            ValueError: If a selected inline body violates its invocation
                contract.
            NotImplementedError: If the IR contains legacy scalar Bit state
                that cannot flow correctly between loop iterations.
        """
        self._run_state = self._new_run_state()
        if isinstance(block_or_ops, Block):
            self._validate_legacy_scalar_bit_rebinds(block_or_ops)
            block_or_ops = self._resource_estimation_view(block_or_ops)
            # Only genuine classical parameters may decide a branch — a
            # measurement bit is never a param slot, so this prevents a runtime
            # ``if bit:`` from being specialized by a same-named value.
            if self._run_state.condition_values:
                slot_names = {slot.name for slot in block_or_ops.param_slots}
                condition_values = self._run_state.condition_values
                self._run_state.condition_values = {
                    name: value
                    for name, value in condition_values.items()
                    if name in slot_names
                }
            resolver = ExprResolver(
                block=block_or_ops,
                context=_root_input_binding_context(block_or_ops, self.bindings),
            )
            input_allocations = _block_input_allocations(block_or_ops, resolver)
            body = self.eval_operations(
                block_or_ops.operations,
                resolver,
                initial_allocations=input_allocations,
            )
            body = _with_constraints(
                body,
                *_block_input_constraints(block_or_ops, resolver),
                *_block_output_constraints(
                    block_or_ops,
                    resolver,
                    proven_cache=self._run_state.array_constraint_proven,
                ),
            )
            input_qubits = sum(input_allocations.values(), _ZERO)
            width = dataclasses.replace(
                body.width,
                input_qubits=input_qubits,
                peak_qubits=input_qubits + body.width.peak_qubits,
            )
            return dataclasses.replace(
                body,
                width=width,
                trace=(
                    _wrap_trace(block_or_ops.name or "qkernel", body.trace)
                    if self.config.trace
                    else None
                ),
            )
        sequence_block = Block(
            name="operation_sequence",
            operations=list(block_or_ops),
            kind=BlockKind.HIERARCHICAL,
        )
        self._validate_legacy_scalar_bit_rebinds(sequence_block)
        sequence_view = self._resource_estimation_view(sequence_block)
        resolver = ExprResolver(
            block=sequence_view,
            context=_root_input_binding_context(block_or_ops, self.bindings),
        )
        return self.eval_operations(sequence_view.operations, resolver)

    def _resource_estimation_view(self, block: Block) -> Block:
        """Build the compiler-equivalent block used for interpretation.

        Ordinary direct qkernel calls are compiler-level organization rather
        than resource boundaries. Expanding them before scheduling preserves
        the readiness of individual measurement results and quantum wires, so
        extracting code into a helper does not change the estimate. Explicit
        quantum-width contracts remain as zero-work validation markers at the
        original call sites.

        Args:
            block (Block): Hierarchical root block to interpret.

        Returns:
            Block: Non-mutating view with eligible direct calls inlined.

        Raises:
            QubitConsumedError: If inlining detects duplicate quantum actuals.
            ValueError: If a selected body violates the invocation contract.
        """
        return InlinePass(
            body_selector=self._resource_estimation_inline_body,
            inline_prefix_factory=self._resource_estimation_inline_prefix,
            preserve_bound_quantum_identity=True,
            preserve_bound_array_identity=True,
        ).run(block)

    def _resource_estimation_inline_body(
        self,
        operation: InvokeOperation,
    ) -> Block | None:
        """Select the strategy-specific body for resource inlining.

        Args:
            operation (InvokeOperation): Direct inline-policy invocation being
                considered for expansion.

        Returns:
            Block | None: Strategy-selected body, or ``None`` when no inline
                body is available.

        Raises:
            ValueError: If the selected body violates the invocation contract.
        """
        return operation.select_body(strategy=self._strategy_for(operation)).body

    def _resource_estimation_inline_prefix(
        self,
        operation: InvokeOperation,
        array_state_bindings: _InlineArrayStateBindings,
    ) -> tuple[Operation, ...]:
        """Preserve resource-only state while dissolving a call boundary.

        Args:
            operation (InvokeOperation): Caller-substituted invocation whose
                selected body is about to be inlined.
            array_state_bindings (_InlineArrayStateBindings): Caller arrays
                paired with cloned callee entries that need call-time snapshots.

        Returns:
            tuple[Operation, ...]: One zero-work boundary marker when the call
                declares a quantum width or carries array state, otherwise an
                empty tuple.
        """
        callable_attrs = {
            **(operation.definition.attrs if operation.definition is not None else {}),
            **operation.attrs,
        }
        legacy_width = callable_attrs.get("num_target_qubits")
        has_width_contract = callable_attrs.get("resource_contract") is not None or (
            type(legacy_width) is int and legacy_width > 0
        )
        if not has_width_contract and not array_state_bindings:
            return ()
        return (
            _ResourceInlineBoundaryOperation(
                constraint_operands=tuple(operation.target_qubits),
                resource_operands=tuple(
                    operation.operands[operation.num_body_external_control_qubits :]
                ),
                callable_attrs=callable_attrs,
                source=operation.custom_name,
                array_state_bindings=array_state_bindings,
            ),
        )

    def resolve_finite_runtime_constraints(
        self,
        estimate: ResourceEstimate,
    ) -> ResourceEstimate:
        """Expand structural constraints over finite runtime alternatives.

        Runtime indices can be scheduled conservatively on their whole owner
        while still having a small, known value domain such as a measured Bit.
        Every alternative must satisfy the structural constraint, after which
        no internal outcome symbol needs to remain in the public payload.

        Args:
            estimate (ResourceEstimate): Specialized estimate to rewrite.

        Returns:
            ResourceEstimate: Estimate whose finitely bounded runtime
                constraints have been expanded and validated.

        Raises:
            ValueError: If any runtime alternative violates a constraint.
        """
        expanded_constraints: list[_ResourceConstraint] = []
        for constraint in estimate._constraints:
            expressions: list[sp.Basic] = [
                sp.sympify(constraint.expression),
                sp.sympify(constraint.active_when),
            ]
            if constraint.expected is not None:
                expressions.append(sp.sympify(constraint.expected))
            for loop_range in constraint.ranges:
                expressions.extend(
                    sp.sympify(value)
                    for value in (
                        loop_range.start,
                        loop_range.step,
                        loop_range.iterations,
                    )
                )
            symbols = set().union(
                *(expression.free_symbols for expression in expressions)
            )
            runtime_symbols = sorted(
                (
                    symbol
                    for symbol in symbols
                    if isinstance(symbol, sp.Symbol)
                    and symbol in self._run_state.runtime_value_domains
                ),
                key=sp.default_sort_key,
            )
            if not runtime_symbols:
                expanded_constraints.append(constraint)
                continue
            domains = [
                self._run_state.runtime_value_domains[symbol]
                for symbol in runtime_symbols
            ]
            if any(domain is None for domain in domains):
                expanded_constraints.append(constraint)
                continue
            finite_domains = cast(list[tuple[sp.Expr, ...]], domains)
            if (
                math.prod(len(domain) for domain in finite_domains)
                > _MAX_RUNTIME_DOMAIN_ALTERNATIVES
            ):
                expanded_constraints.append(constraint)
                continue
            for values in itertools.product(*finite_domains):
                replacements: dict[sp.Symbol, sp.Expr] = dict(
                    zip(runtime_symbols, values, strict=True)
                )
                expanded_constraints.append(
                    constraint.mapped(
                        lambda expression, replacements=replacements: cast(
                            sp.Expr,
                            sp.sympify(expression).subs(
                                list(replacements.items()),
                                simultaneous=True,
                            ),
                        )
                    )
                )
        return dataclasses.replace(
            estimate,
            _constraints=tuple(expanded_constraints),
        )

    def validate_no_internal_resource_symbols(
        self,
        estimate: ResourceEstimate,
    ) -> None:
        """Reject internal runtime values that escaped into public algebra.

        Internal observation outcomes and unresolved loop carries are semantic
        execution state, not user inputs. Hiding them from ``parameters``
        would leave an unsubstitutable expression, while publishing them would
        let a user choose an outcome and silently underestimate a runtime
        worst case. The estimator therefore fails closed when such a value
        affects any serialized metric, constraint, metadata guard, or trace.

        Args:
            estimate (ResourceEstimate): Final specialized estimate to check.

        Raises:
            NotImplementedError: If an internal runtime or unresolved carry
                symbol remains in the public estimate payload.
        """
        internal_symbols = (
            self._run_state.runtime_observation_symbols
            | self._run_state.unresolved_resource_symbols
        )
        escaped = (
            _free_symbols(estimate) | _trace_guard_free_symbols(estimate.trace)
        ) & internal_symbols
        if not escaped:
            return
        names = ", ".join(sorted(_symbol_display_name(symbol) for symbol in escaped))
        raise NotImplementedError(
            "Resource estimation cannot bound a runtime-derived or unresolved "
            f"loop-carried value used by resource-sensitive structure ({names}). "
            "Provide concrete structural inputs or rewrite the continuation so "
            "its resource cost does not depend on that runtime value."
        )

    def _validate_legacy_scalar_bit_rebinds(
        self,
        block_or_operations: Block | Sequence[Operation],
        *,
        output_values: Sequence[ValueLike] | None = None,
        local_bindings: Mapping[str, Any] | None = None,
    ) -> None:
        """Reuse compiler loop-state validation before interpreting a body.

        Args:
            block_or_operations (Block | Sequence[Operation]): Semantic body
                to validate. A Block is inlined into a temporary affine view
                so only callee inputs that the selected body actually reads
                count as loop back-edge reads.
            output_values (Sequence[ValueLike] | None): Body outputs used for
                post-loop liveness when a bare operation sequence is passed.
                A Block always supplies its own outputs. Defaults to ``None``.
            local_bindings (Mapping[str, Any] | None): Resolved callee-formal
                values added to root resource inputs. Defaults to ``None``.

        Raises:
            NotImplementedError: If a legacy scalar Bit rebind cannot be
                represented across the loop boundary.
        """
        operations = (
            block_or_operations.operations
            if isinstance(block_or_operations, Block)
            else block_or_operations
        )
        if not self._has_reachable_legacy_scalar_bit_rebinds(operations):
            return
        validation_bindings = {
            name: self._compiler_validation_binding(value)
            for name, value in self.bindings.items()
        }
        validation_bindings.update(
            {
                name: self._compiler_validation_binding(value)
                for name, value in self._run_state.condition_values.items()
            }
        )
        if local_bindings is not None:
            validation_bindings.update(
                {
                    name: self._compiler_validation_binding(value)
                    for name, value in local_bindings.items()
                }
            )
        validation_outputs = output_values
        if isinstance(block_or_operations, Block):
            validation_block = self._loop_validation_view(
                block_or_operations,
                validation_bindings,
            )
            operations = validation_block.operations
            validation_outputs = validation_block.output_values
        if not has_legacy_scalar_bit_rebinds(operations):
            return
        try:
            reject_loop_carried_classical_rebinds(
                list(operations),
                bindings=validation_bindings,
                output_values=(
                    list(validation_outputs) if validation_outputs is not None else None
                ),
            )
        except ValidationError as exc:
            raise NotImplementedError(str(exc)) from exc

    def _has_reachable_legacy_scalar_bit_rebinds(
        self,
        operations: Sequence[Operation],
        *,
        visited_blocks: set[int] | None = None,
    ) -> bool:
        """Return whether selected callable bodies need loop-state validation.

        The compiler-equivalent validation view performs fixed-point inlining
        so it can distinguish used and unused scalar Bit inputs. Building that
        view is unnecessary for the overwhelmingly common case with no legacy
        Bit rebinds, and can expand an unrelated recursive algorithm before
        resource interpretation reaches its callable boundary. This cheap,
        cycle-safe scan follows only the bodies selected by estimator policy.

        Args:
            operations (Sequence[Operation]): Operations in the current scope.
            visited_blocks (set[int] | None): Callable block identities already
                scanned on this path. Defaults to ``None``.

        Returns:
            bool: Whether compiler loop-state validation is required.

        Raises:
            ValueError: If a strategy-selected body violates the invocation
                contract.
        """
        if has_legacy_scalar_bit_rebinds(operations):
            return True
        visited = visited_blocks if visited_blocks is not None else set()

        def body_has_rebind(block: Block | None) -> bool:
            """Scan one callable block without following recursive cycles.

            Args:
                block (Block | None): Selected callable body, when available.

            Returns:
                bool: Whether the body graph contains a legacy Bit rebind.

            Raises:
                ValueError: If a nested strategy-selected body violates its
                    invocation contract.
            """
            if not isinstance(block, Block):
                return False
            identity = id(block)
            if identity in visited:
                return False
            visited.add(identity)
            return self._has_reachable_legacy_scalar_bit_rebinds(
                block.operations,
                visited_blocks=visited,
            )

        for operation in walk_operations(operations):
            if isinstance(operation, InvokeOperation):
                selection = operation.select_body(
                    strategy=self._strategy_for(operation)
                )
                if body_has_rebind(selection.body):
                    return True
            elif isinstance(operation, ControlledUOperation):
                if body_has_rebind(operation.block):
                    return True
            elif isinstance(operation, InverseBlockOperation):
                if body_has_rebind(operation.implementation_block):
                    return True
            elif isinstance(operation, SelectOperation):
                if any(body_has_rebind(block) for block in operation.case_blocks):
                    return True
        return False

    def _loop_validation_view(
        self,
        block: Block,
        bindings: dict[str, Any],
    ) -> Block:
        """Build the compiler-equivalent view used by loop-state validation.

        Resource interpretation intentionally preserves callable boundaries,
        but the compiler validates loop-carried state after inline-policy calls
        have been expanded. This temporary view follows the same selected-body
        and compile-time specialization rules without changing the Block that
        resource evaluation traverses. Repeating inline then specialization
        also handles concrete self-recursion up to the compiler's supported
        unroll depth. Callable bodies without a scalar Bit interface or legacy
        Bit carry stay boxed because their internal structure cannot affect
        this validation.

        Args:
            block (Block): Hierarchical semantic body to validate.
            bindings (dict[str, Any]): Concrete compiler-domain bindings used
                for branch specialization.

        Returns:
            Block: Non-mutating validation view. A recursive call that does not
                converge within the supported depth remains boxed and is
                handled conservatively by the shared validator.

        Raises:
            ValueError: If a strategy-selected body violates the invocation
                input or output contract.
            QubitConsumedError: If inlining detects duplicate quantum actuals.
            ValidationError: If compile-time specialization encounters invalid
                IR.
        """
        inline = InlinePass(body_selector=self._loop_validation_inline_body)
        view = self._loop_validation_pruned_block(block)
        view = lower_compile_time_ifs_preserving_loop_conditions(view, bindings)
        for _ in range(_MAX_LOOP_VALIDATION_INLINE_DEPTH):
            view = inline.run(view)
            if view.kind is not BlockKind.HIERARCHICAL:
                return view
            view = lower_compile_time_ifs_preserving_loop_conditions(
                view,
                bindings,
            )
            if not _has_any_inline_call(
                view.operations,
                self._loop_validation_inline_body,
            ):
                # Lowering the final base-case branch can remove the last
                # invocation while the copied block still carries its stale
                # HIERARCHICAL kind. Refresh it exactly as recursion unrolling
                # does so the supported depth boundary stays identical.
                return inline.run(view)
        name = block.name or "qkernel"
        raise ValueError(
            f"Recursive resource validation for '{name}' did not reach a "
            "base case within the supported inline depth. Supply a concrete "
            "recursion-driving value in inputs, or replace the "
            "recursion with a bounded loop."
        )

    def _loop_validation_inline_body(
        self,
        operation: InvokeOperation,
    ) -> Block | None:
        """Select a body relevant to legacy scalar Bit validation.

        Callables without scalar Bit inputs, outputs, or internal legacy
        carries are opaque to this validation-only view. Their internal
        quantum and classical structure cannot change whether a caller's
        legacy scalar Bit crosses a loop back edge.

        Args:
            operation (InvokeOperation): Inline-policy invocation being copied
                into the validation view.

        Returns:
            Block | None: Strategy-selected body, or ``None`` when unavailable
                or irrelevant to legacy scalar Bit validation.

        Raises:
            ValueError: If the selected body violates the invocation contract.
        """
        selection = operation.select_body(strategy=self._strategy_for(operation))
        body = selection.body
        if not isinstance(body, Block):
            return None
        if not self._loop_validation_body_is_relevant(
            body,
            (*selection.operands, *operation.results),
        ):
            return None
        return self._loop_validation_pruned_block(body)

    def _loop_validation_body_is_relevant(
        self,
        body: Block | None,
        interface_values: Sequence[ValueBase],
    ) -> bool:
        """Return whether a callable body can affect scalar Bit validation.

        Args:
            body (Block | None): Callable implementation body, when available.
            interface_values (Sequence[ValueBase]): Call-site operands and
                results visible to the surrounding validation scope.

        Returns:
            bool: Whether the body must be expanded for validation.

        Raises:
            ValueError: If a nested strategy-selected body violates its
                invocation contract.
        """
        if not isinstance(body, Block):
            return False
        return any(
            isinstance(value.type, BitType) for value in interface_values
        ) or self._has_reachable_legacy_scalar_bit_rebinds(body.operations)

    def _loop_validation_pruned_block(self, block: Block) -> Block:
        """Build a validation-only block with irrelevant bodies opaque.

        Args:
            block (Block): Semantic block whose operation-owned callable
                bodies should be projected.

        Returns:
            Block: Copy containing only callable internals relevant to legacy
            scalar Bit validation.

        Raises:
            ValueError: If a nested strategy-selected body violates its
                invocation contract.
        """
        identity = id(block)
        cached = self._run_state.loop_validation_pruned_blocks.get(identity)
        if cached is not None:
            return cached
        projected = dataclasses.replace(
            block,
            operations=self._loop_validation_pruned_operations(block.operations),
        )
        self._run_state.loop_validation_pruned_blocks[identity] = projected
        return projected

    def _loop_validation_pruned_operations(
        self,
        operations: Sequence[Operation],
    ) -> list[Operation]:
        """Project operation-owned bodies for scalar Bit validation.

        Args:
            operations (Sequence[Operation]): Operations in one lexical scope.

        Returns:
            list[Operation]: Non-mutating validation projection.

        Raises:
            ValueError: If a nested strategy-selected body violates its
                invocation contract.
        """

        def project_owned_body(
            body: Block | None,
            interface_values: Sequence[ValueBase],
        ) -> Block | None:
            """Keep a relevant operation-owned body or replace it with a stub.

            Args:
                body (Block | None): Controlled, inverse, or SELECT body.
                interface_values (Sequence[ValueBase]): Owner operands and
                    results visible to the surrounding validation scope.

            Returns:
                Block | None: Recursively projected body or an affine stub.

            Raises:
                ValueError: If a nested strategy-selected body violates its
                    invocation contract.
            """
            if not isinstance(body, Block):
                return None
            if self._loop_validation_body_is_relevant(body, interface_values):
                return self._loop_validation_pruned_block(body)
            return dataclasses.replace(
                body,
                operations=[],
                kind=BlockKind.AFFINE,
            )

        projected: list[Operation] = []
        for operation in operations:
            interface_values = (*operation.operands, *operation.results)
            if isinstance(operation, ControlledUOperation):
                projected.append(
                    dataclasses.replace(
                        operation,
                        block=project_owned_body(
                            operation.block,
                            interface_values,
                        ),
                    )
                )
            elif isinstance(operation, InverseBlockOperation):
                projected.append(
                    dataclasses.replace(
                        operation,
                        source_block=project_owned_body(
                            operation.source_block,
                            interface_values,
                        ),
                        implementation_block=project_owned_body(
                            operation.implementation_block,
                            interface_values,
                        ),
                    )
                )
            elif isinstance(operation, SelectOperation):
                projected.append(
                    dataclasses.replace(
                        operation,
                        case_blocks=[
                            cast(
                                Block,
                                project_owned_body(case, interface_values),
                            )
                            for case in operation.case_blocks
                        ],
                    )
                )
            elif isinstance(operation, HasNestedOps):
                regions = tuple(
                    dataclasses.replace(
                        region,
                        operations=tuple(
                            self._loop_validation_pruned_operations(
                                region.operations,
                            )
                        ),
                    )
                    for region in operation.nested_regions()
                )
                projected.append(operation.rebuild_regions(regions))
            else:
                projected.append(operation)
        return projected

    @staticmethod
    def _compiler_validation_binding(value: Any) -> Any:
        """Convert a concrete SymPy scalar to the compiler binding domain.

        Args:
            value (Any): Resource-estimator input or resolved callee value.

        Returns:
            Any: Equivalent Python scalar when concrete, otherwise the
                original binding object.
        """
        if not isinstance(value, sp.Expr) or not value.is_number:
            return value
        if _is_concrete_integer(value):
            return int(value)
        if value.is_real is True:
            return float(value)
        return value

    def _reserve_while_trip_count_names(
        self,
        operations: Sequence[Operation],
        resolver: ExprResolver,
    ) -> None:
        """Reserve deterministic names for every structurally reachable while.

        The interpreter may evaluate a loop body first as a symbolic probe and
        again with resolved carry values, or skip that body after input
        specialization. Preorder reservation makes both paths use the same
        public trip-count symbol names.

        Args:
            operations (Sequence[Operation]): Root operations whose nested
                control-flow and callable bodies should be indexed.
            resolver (ExprResolver): Resolver carrying the current structural
                call-site path.
        """

        def visit(
            body_operations: Sequence[Operation],
            structural_scope: tuple[tuple[int, int], ...],
            active_call_bodies: frozenset[int],
        ) -> None:
            """Visit one operation sequence in deterministic lexical order.

            Args:
                body_operations (Sequence[Operation]): Operations to scan.
                structural_scope (tuple[tuple[int, int], ...]): Callable path
                    containing this operation sequence.
                active_call_bodies (frozenset[int]): Callable block identities
                    already entered on this lexical path.
            """
            for operation in body_operations:
                operation_id = id(operation)
                operation_key = (structural_scope, operation_id)
                cached = self._run_state.while_name_scan_operations.get(operation_key)
                if cached is operation:
                    continue
                self._run_state.while_name_scan_operations[operation_key] = operation
                if isinstance(operation, WhileOperation):
                    ordinal = len(self._run_state.while_trip_count_names) + 1
                    name = "|while|" if ordinal == 1 else f"|while[{ordinal}]|"
                    self._run_state.while_trip_count_names[operation_key] = (
                        operation,
                        name,
                    )

                if isinstance(operation, InvokeOperation):
                    body, _realized_transform = operation.body_for_transform(
                        strategy=self._strategy_for(operation)
                    )
                    if isinstance(body, Block) and id(body) not in active_call_bodies:
                        visit(
                            body.operations,
                            (*structural_scope, (operation_id, id(body))),
                            active_call_bodies | {id(body)},
                        )
                elif isinstance(operation, ControlledUOperation):
                    if (
                        isinstance(operation.block, Block)
                        and id(operation.block) not in active_call_bodies
                    ):
                        visit(
                            operation.block.operations,
                            (
                                *structural_scope,
                                (operation_id, id(operation.block)),
                            ),
                            active_call_bodies | {id(operation.block)},
                        )
                elif isinstance(operation, SelectOperation):
                    for case in operation.case_blocks:
                        if id(case) in active_call_bodies:
                            continue
                        visit(
                            case.operations,
                            (*structural_scope, (operation_id, id(case))),
                            active_call_bodies | {id(case)},
                        )
                elif isinstance(operation, InverseBlockOperation):
                    if (
                        isinstance(operation.implementation_block, Block)
                        and id(operation.implementation_block) not in active_call_bodies
                    ):
                        visit(
                            operation.implementation_block.operations,
                            (
                                *structural_scope,
                                (operation_id, id(operation.implementation_block)),
                            ),
                            active_call_bodies | {id(operation.implementation_block)},
                        )
                elif isinstance(operation, HasNestedOps):
                    for nested in operation.nested_op_lists():
                        visit(nested, structural_scope, active_call_bodies)

        visit(operations, resolver.structural_scope, frozenset())

    def eval_block(self, block: Block, resolver: ExprResolver) -> ResourceEstimate:
        """Evaluate a block body.

        Args:
            block (Block): Block to evaluate.
            resolver (ExprResolver): Resolver scoped to ``block``.

        Returns:
            ResourceEstimate: Estimated resources for the body.
        """
        estimate = self.eval_operations(
            block.operations,
            resolver,
            initial_allocations=_block_input_allocations(block, resolver),
        )
        return _with_constraints(
            estimate,
            *_block_input_constraints(block, resolver),
        )

    @staticmethod
    def _bind_resource_inline_array_states(
        operation: _ResourceInlineBoundaryOperation,
        resolver: ExprResolver,
    ) -> None:
        """Snapshot caller arrays into one inlined callable's entry values.

        Every caller state is captured before any callee entry is rebound.
        This preserves simultaneous call-argument semantics even when future
        IR permits two classical formals to alias one array lineage.

        Args:
            operation (_ResourceInlineBoundaryOperation): Zero-work call
                boundary carrying caller-to-callee array bindings.
            resolver (ExprResolver): Resolver at the original call position.
        """
        snapshots = tuple(
            (entries, resolver.snapshot_array_state(actual))
            for actual, entries in operation.array_state_bindings
        )
        for entries, state in snapshots:
            for entry in entries:
                resolver.bind_array_state(entry, state)
