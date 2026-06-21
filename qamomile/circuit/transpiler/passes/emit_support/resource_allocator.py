"""Resource allocation helpers for emission."""

from __future__ import annotations

import numbers
from typing import TYPE_CHECKING, Any

from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.arithmetic_operations import PhiOp
from qamomile.circuit.ir.operation.cast import CastOperation
from qamomile.circuit.ir.operation.composite_gate import (
    CompositeGateOperation,
)
from qamomile.circuit.ir.operation.control_flow import (
    HasNestedOps,
    IfOperation,
    WhileOperation,
)
from qamomile.circuit.ir.operation.gate import (
    ConcreteControlledU,
    ControlledUOperation,
    GateOperation,
    MeasureOperation,
    MeasureQFixedOperation,
    MeasureVectorOperation,
    SymbolicControlledU,
)
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp
from qamomile.circuit.ir.value import ArrayValue, Value, resolve_root_qubit_address
from qamomile.circuit.transpiler.passes.emit_support.condition_resolution import (
    map_phi_outputs,
    remap_static_phi_outputs,
    resolve_condition_address_detailed,
    resolve_if_condition,
)
from qamomile.circuit.transpiler.passes.emit_support.qubit_address import (
    ClbitMap,
    QubitAddress,
    QubitMap,
)
from qamomile.circuit.transpiler.passes.emit_support.value_resolver import (
    ValueResolver,
    resolve_qubit_key,
)

if TYPE_CHECKING:
    from qamomile.circuit.ir.value import Value


class ResourceAllocator:
    """Allocates qubit and classical bit indices from operations.

    This class handles the first pass of circuit emission: determining
    how many physical qubits and classical bits are needed and mapping
    Value UUIDs to their physical indices.

    New physical indices are assigned via monotonic counters
    (``_next_qubit_index`` / ``_next_clbit_index``) so that alias
    entries — which reuse an existing physical index — never inflate
    the counter.  Using ``len(map)`` would cause sparse (gapped)
    physical indices because alias keys increase the map size without
    adding new physical resources.
    """

    def __init__(self, resolver: ValueResolver | None = None) -> None:
        """Initialize allocator state.

        Args:
            resolver (ValueResolver | None): Emit value resolver that carries
                runtime parameter names and binding lookup rules. Defaults to
                None, which creates a resolver without runtime parameters.
        """
        self._next_qubit_index: int = 0
        self._next_clbit_index: int = 0
        self._resolver = resolver or ValueResolver()

    @staticmethod
    def _coerce_nonnegative_integral_size(value: Any) -> int | None:
        """Coerce a non-negative, non-boolean integral value to a Python int.

        Args:
            value (Any): Candidate structural size value resolved from IR
                constants, compile-time bindings, or bound array elements.

        Returns:
            int | None: The coerced integer size, or None when ``value`` is
                negative, not an integral numeric value, or is a boolean.
        """
        # Python bool is an Integral subclass, but it is never a valid size.
        # NumPy bool scalars are rejected because they are not Integral values.
        if isinstance(value, bool):
            return None
        if isinstance(value, numbers.Integral):
            size = int(value)
            return size if size >= 0 else None
        return None

    def allocate(
        self,
        operations: list[Operation],
        bindings: dict[str, Any] | None = None,
        initial_qubit_map: QubitMap | None = None,
        initial_clbit_map: ClbitMap | None = None,
    ) -> tuple[QubitMap, ClbitMap]:
        """Allocate qubit and clbit indices for all operations.

        Args:
            operations (list[Operation]): Operations to allocate resources
                for.
            bindings (dict[str, Any] | None): Optional variable bindings
                for resolving dynamic sizes. Defaults to None (treated as
                an empty mapping).
            initial_qubit_map (QubitMap | None): Optional pre-populated
                qubit address mapping. Used by callers that need to seed
                the allocator with bindings established outside the
                operation list — for instance, the inner-block emitter
                in ``blockvalue_to_gate`` pre-allocates ``Vector[Qubit]``
                input elements (the inner block has no ``QInitOperation``
                for inputs, so per-element entries must be supplied here
                or the assertion in ``_allocate_gate`` fires). The map is
                copied; allocation continues from
                ``max(values) + 1`` so new ``QInitOperation`` allocations
                inside ``operations`` do not collide. Defaults to None
                (treated as empty).
            initial_clbit_map (ClbitMap | None): Optional pre-populated
                clbit address mapping. Same semantics as
                ``initial_qubit_map`` but for classical bits. Defaults to
                None.

        Returns:
            tuple[QubitMap, ClbitMap]: ``(qubit_map, clbit_map)`` where
                each maps ``QubitAddress`` to a physical index. If an
                initial map was supplied, its entries are preserved
                verbatim in the returned map.
        """
        qubit_map: QubitMap = dict(initial_qubit_map) if initial_qubit_map else {}
        clbit_map: ClbitMap = dict(initial_clbit_map) if initial_clbit_map else {}
        self._next_qubit_index = max(qubit_map.values(), default=-1) + 1
        self._next_clbit_index = max(clbit_map.values(), default=-1) + 1
        self._allocate_recursive(operations, qubit_map, clbit_map, bindings or {})
        return qubit_map, clbit_map

    def _allocate_recursive(
        self,
        operations: list[Operation],
        qubit_map: QubitMap,
        clbit_map: ClbitMap,
        bindings: dict[str, Any],
    ) -> None:
        """Recursively allocate resources from operations."""
        for op in operations:
            if isinstance(op, QInitOperation):
                result = op.results[0]
                if isinstance(result, ArrayValue):
                    # Allocate physical qubits for array elements using
                    # QubitAddress(array_uuid, i) keys.  At this stage only
                    # these composite keys are registered; the individual
                    # element Values (which carry their own UUIDs) are created
                    # dynamically during frontend tracing and their UUID
                    # mapping is deferred to _allocate_gate / _allocate_qubit_list.
                    if result.shape:
                        size_val = result.shape[0]
                        size = self._resolve_size(size_val, bindings)
                        if size is None:
                            from qamomile.circuit.transpiler.errors import EmitError

                            raise EmitError(
                                "Cannot resolve array size for qubit allocation. "
                                "Structural UInt parameters must be bound at transpile time."
                            )
                        for i in range(size):
                            qubit_addr = QubitAddress(result.uuid, i)
                            if qubit_addr not in qubit_map:
                                qubit_map[qubit_addr] = self._next_qubit_index
                                self._next_qubit_index += 1
                    continue
                scalar_addr = QubitAddress(result.uuid)
                if scalar_addr not in qubit_map:
                    qubit_map[scalar_addr] = self._next_qubit_index
                    self._next_qubit_index += 1

            elif isinstance(op, MeasureOperation):
                result = op.results[0]
                clbit_addr = QubitAddress(result.uuid)
                if clbit_addr not in clbit_map:
                    clbit_map[clbit_addr] = self._next_clbit_index
                    self._next_clbit_index += 1

            elif isinstance(op, MeasureVectorOperation):
                result = op.results[0]
                if isinstance(result, ArrayValue) and result.shape:
                    size_val = result.shape[0]
                    size = self._resolve_size(size_val, bindings)
                    if size is not None:
                        for i in range(size):
                            clbit_addr = QubitAddress(result.uuid, i)
                            if clbit_addr not in clbit_map:
                                clbit_map[clbit_addr] = self._next_clbit_index
                                self._next_clbit_index += 1

            elif isinstance(op, MeasureQFixedOperation):
                qfixed = op.operands[0]
                qubit_uuids = qfixed.get_qfixed_qubit_uuids()
                result = op.results[0]
                for i, qubit_uuid in enumerate(qubit_uuids):
                    clbit_addr = QubitAddress(result.uuid, i)
                    if clbit_addr not in clbit_map:
                        clbit_map[clbit_addr] = self._next_clbit_index
                        self._next_clbit_index += 1

            elif isinstance(op, GateOperation):
                self._allocate_gate(op, qubit_map)

            elif isinstance(op, IfOperation):
                resolved = resolve_if_condition(op.condition, bindings)
                if resolved is not None:
                    # Compile-time constant: only allocate the selected
                    # branch and remap phi outputs directly.
                    selected = op.true_operations if resolved else op.false_operations
                    self._allocate_recursive(selected, qubit_map, clbit_map, bindings)
                    self._remap_static_phi_outputs(
                        op.phi_ops, resolved, qubit_map, clbit_map
                    )
                else:
                    self._allocate_recursive(
                        op.true_operations, qubit_map, clbit_map, bindings
                    )
                    self._allocate_recursive(
                        op.false_operations, qubit_map, clbit_map, bindings
                    )
                    self._allocate_phi_ops(op.phi_ops, qubit_map, clbit_map)

            elif isinstance(op, WhileOperation):
                # WhileOperation operands:
                #   operands[0]: initial condition (always present)
                #   operands[1]: loop-carried condition (present only when the
                #                body reassigns the condition variable)
                # No other operand count is valid.
                if len(op.operands) == 1:
                    # Invariant condition: the condition variable is not
                    # reassigned inside the loop body.  No loop-carried
                    # clbit aliasing is needed; just allocate the body.
                    self._allocate_recursive(
                        op.operations, qubit_map, clbit_map, bindings
                    )
                elif len(op.operands) == 2:
                    initial_cond = op.operands[0]
                    loop_carried = op.operands[1]
                    init_val = (
                        initial_cond.value
                        if hasattr(initial_cond, "value")
                        else initial_cond
                    )
                    init_addr = self._condition_source_address(init_val, bindings)

                    # Save the canonical clbit for the initial condition
                    # BEFORE body allocation.  An if-only (no else) inside
                    # the loop body produces a PhiOp whose false_val is the
                    # pre-if while-condition value.  map_phi_outputs will
                    # redirect that false_val UUID to the true-branch clbit,
                    # overwriting clbit_map[init_addr] and making the
                    # post-body mismatch detection ineffective.
                    saved_init_clbit = clbit_map.get(init_addr)

                    # Allocate the loop body so that IfOperation phi
                    # mappings inside the body are fully resolved.
                    self._allocate_recursive(
                        op.operations, qubit_map, clbit_map, bindings
                    )

                    # Restore the canonical clbit for init_addr if it was
                    # overwritten during body allocation.
                    if saved_init_clbit is not None:
                        clbit_map[init_addr] = saved_init_clbit

                    carried_val = (
                        loop_carried.value
                        if hasattr(loop_carried, "value")
                        else loop_carried
                    )
                    carried_addr = self._condition_source_address(carried_val, bindings)
                    carried_clbit = clbit_map.get(carried_addr)

                    # Alias the loop-carried condition to the initial
                    # while-condition clbit.  After body allocation the
                    # loop-carried UUID may point to a different clbit
                    # (e.g. a phi-merged measurement from an if-else).
                    # We recursively trace IfOperation phi_ops and map all
                    # upstream branch-measurement UUIDs to the canonical clbit.
                    if (
                        saved_init_clbit is not None
                        and carried_clbit is not None
                        and saved_init_clbit != carried_clbit
                    ):
                        clbit_map[carried_addr] = saved_init_clbit
                        self._alias_loop_carried_clbits(
                            op.operations,
                            carried_addr,
                            saved_init_clbit,
                            clbit_map,
                        )
                    elif saved_init_clbit is not None and carried_addr not in clbit_map:
                        clbit_map[carried_addr] = saved_init_clbit
                else:
                    assert False, (
                        "[FOR DEVELOPER] WhileOperation must have exactly 2 "
                        "operands to reach this branch, but got "
                        f"{len(op.operands)}. This indicates a bug in the "
                        "WhileOperation construction."
                    )

            elif isinstance(op, HasNestedOps):
                # Generic recursion for For/ForItems: recurse into all nested bodies.
                for body in op.nested_op_lists():
                    self._allocate_recursive(body, qubit_map, clbit_map, bindings)

            elif isinstance(op, PauliEvolveOp):
                self._allocate_pauli_evolve(op, qubit_map)

            elif isinstance(op, (CompositeGateOperation, InverseBlockOperation)):
                self._allocate_composite(op, qubit_map)

            elif isinstance(op, ControlledUOperation):
                self._allocate_controlled_u(op, qubit_map)

            elif isinstance(op, CastOperation):
                self._allocate_cast(op, qubit_map)

    @staticmethod
    def _condition_source_address(
        value: Any,
        bindings: dict[str, Any],
    ) -> QubitAddress:
        """Resolve a while / phi condition source to its ``clbit_map`` key.

        A measured ``Vector[Bit]`` element (``s[i]`` from
        ``s = qmc.measure(register)``) registers its clbit under
        ``QubitAddress(root_array.uuid, index)``, not the element's own
        UUID. Resolving the source the same way the emit-time condition
        lookup does keeps the loop-carried / phi clbit alias consistent
        with where the clbit was actually allocated. The allocator has no
        ``ValueResolver``, so only constant indices / slice bounds resolve;
        an unresolved element falls back to its scalar UUID (which is not
        registered, so the caller's ``clbit_map`` lookup misses and the
        aliasing is correctly skipped rather than pointed at a wrong slot).

        Args:
            value (Any): The condition / phi source — an IR ``Value`` or, in
                degenerate cases, a non-Value (handled via ``str``).
            bindings (dict[str, Any]): Active bindings for constant folding.

        Returns:
            QubitAddress: The address the source's classical bit is
                registered under.
        """
        if isinstance(value, Value):
            address, _ = resolve_condition_address_detailed(value, bindings, None)
            return address
        return QubitAddress(str(value))

    def _alias_loop_carried_clbits(
        self,
        operations: list[Operation],
        target_addr: QubitAddress,
        canonical_clbit: int,
        clbit_map: ClbitMap,
    ) -> None:
        """Recursively trace PhiOp sources and alias them to *canonical_clbit*.

        When a while loop body contains an if-else with measurements in
        both branches, the phi-merged result (the loop-carried condition)
        and all its upstream branch-measurement UUIDs must write to the
        same classical bit as the initial while condition.
        """
        for op in operations:
            if isinstance(op, HasNestedOps) and not isinstance(op, IfOperation):
                for body in op.nested_op_lists():
                    self._alias_loop_carried_clbits(
                        body, target_addr, canonical_clbit, clbit_map
                    )
                continue
            if not isinstance(op, IfOperation):
                continue
            for phi in op.phi_ops:
                if not isinstance(phi, PhiOp):
                    continue
                output = phi.results[0]
                if output.uuid != target_addr.uuid:
                    continue
                true_val = phi.operands[1]
                false_val = phi.operands[2]
                true_addr = self._condition_source_address(true_val, {})
                false_addr = self._condition_source_address(false_val, {})
                if true_addr in clbit_map:
                    clbit_map[true_addr] = canonical_clbit
                if false_addr in clbit_map:
                    clbit_map[false_addr] = canonical_clbit
                # Recurse into branches for nested if-else
                self._alias_loop_carried_clbits(
                    op.true_operations, true_addr, canonical_clbit, clbit_map
                )
                self._alias_loop_carried_clbits(
                    op.false_operations, false_addr, canonical_clbit, clbit_map
                )

    def _allocate_phi_ops(
        self,
        phi_ops: list,
        qubit_map: QubitMap,
        clbit_map: ClbitMap,
    ) -> None:
        """Register phi output UUIDs via the shared ``map_phi_outputs`` utility."""
        map_phi_outputs(phi_ops, qubit_map, clbit_map)

    def _remap_static_phi_outputs(
        self,
        phi_ops: list,
        condition_value: bool,
        qubit_map: QubitMap,
        clbit_map: ClbitMap,
    ) -> None:
        """Remap phi outputs for a compile-time constant ``IfOperation``.

        Delegates to the module-level ``remap_static_phi_outputs`` helper,
        which is shared with the emit pass to ensure scalar and array
        quantum phi outputs are handled identically at both allocation
        and emission time.

        Args:
            phi_ops: Phi operations from the ``IfOperation``.
            condition_value: The resolved boolean condition.
            qubit_map: QubitAddress-to-physical-qubit mapping (mutated in place).
            clbit_map: QubitAddress-to-physical-clbit mapping (mutated in place).
        """
        remap_static_phi_outputs(phi_ops, condition_value, qubit_map, clbit_map)

    def _resolve_size(
        self,
        size_val: "Value",
        bindings: dict[str, Any],
    ) -> int | None:
        """Resolve a size Value to a concrete integer.

        Args:
            size_val (Value): IR value used as a qubit-array size.
                Constants, bound scalar values, shape-dimension values, and
                array-element values are supported when compile-time concrete.
            bindings (dict[str, Any]): Compile-time bindings available to the
                emit pass, keyed by parameter names or value UUIDs.

        Returns:
            int | None: Concrete integer size, or None when the value cannot
                be resolved at allocation time.
        """
        import re

        if size_val.is_constant():
            return self._coerce_nonnegative_integral_size(size_val.get_const())

        # Array element (e.g., sizes[0]): delegate to the emit value resolver
        # so bound containers and VectorView slices follow the same lookup
        # rules as other emit-time value resolution paths.  Resolver refusal
        # is final here; symbolic array-element sizes must stay unresolved.
        if size_val.parent_array is not None and size_val.element_indices:
            return self._coerce_nonnegative_integral_size(
                self._resolver.resolve_bound_value(size_val, bindings)
            )

        # Check by name, then uuid in bindings
        if size_val.name and size_val.name in bindings:
            bound = bindings[size_val.name]
            if (size := self._coerce_nonnegative_integral_size(bound)) is not None:
                return size
            if hasattr(bound, "__len__"):
                return len(bound)

        if size_val.uuid in bindings:
            bound = bindings[size_val.uuid]
            if (size := self._coerce_nonnegative_integral_size(bound)) is not None:
                return size

        # Dimension naming pattern (e.g., "hi_dim0" -> array "hi", dimension 0).
        # Handles cases where parent_array is None after inlining.
        if size_val.name:
            match = re.match(r"^(.+)_dim(\d+)$", size_val.name)
            if match:
                array_name = match.group(1)
                dim_index = int(match.group(2))
                if array_name in bindings:
                    array_data = bindings[array_name]
                    if hasattr(array_data, "shape"):
                        if dim_index < len(array_data.shape):
                            return int(array_data.shape[dim_index])
                    elif dim_index == 0 and hasattr(array_data, "__len__"):
                        return len(array_data)

        return None

    def _allocate_gate(
        self,
        op: GateOperation,
        qubit_map: QubitMap,
    ) -> None:
        """Allocate resources for a GateOperation."""
        # GateOperation represents a unitary gate: qubit count is preserved.
        qubit_ops = op.qubit_operands
        assert len(op.results) == len(qubit_ops), (
            f"GateOperation must have equal qubit operands and results, "
            f"got {len(qubit_ops)} qubit operands and {len(op.results)} results."
        )

        # Phase 1: Register all operands in qubit_map.
        # Element Values are created dynamically during frontend tracing
        # (handle/array.py _get_element), so their UUIDs are unknown at
        # QInitOperation time.  Here we lazily map each element UUID to
        # the physical qubit already allocated under the root parent's
        # QubitAddress key — the chain walk in
        # ``_resolve_root_qubit_address`` handles sliced views so that
        # e.g. ``view[0]`` for ``view = q[1:3]`` resolves to the
        # physical qubit for ``q[1]``.
        for operand in qubit_ops:
            operand_addr = QubitAddress(operand.uuid)
            if operand_addr not in qubit_map:
                chain_addr = self._resolve_root_qubit_address(operand)
                if chain_addr is not None:
                    assert chain_addr in qubit_map, (
                        f"Array element key {str(chain_addr)!r} not found in qubit_map. "
                        f"This indicates a bug in the transpiler pipeline: "
                        f"QInitOperation for the parent array was not processed "
                        f"before this GateOperation."
                    )
                    qubit_map[operand_addr] = qubit_map[chain_addr]
                elif operand.parent_array is None or not operand.element_indices:
                    # Scalar qubit: allocate new index.
                    # This path is used for @qkernel input parameters created with
                    # emit_init=False (func_to_block.py), which have no QInitOperation.
                    # ResourceAllocator.allocate() receives only operations, not the
                    # block's input_values, so these qubits are first registered here.
                    qubit_map[operand_addr] = self._next_qubit_index
                    self._next_qubit_index += 1
                # Non-constant indices (symbolic loop vars) are resolved
                # at emit time via ValueResolver.resolve_qubit_index_detailed.

        # Phase 2: Map each result to its corresponding qubit operand (1:1)
        for i, result in enumerate(op.results):
            result_addr = QubitAddress(result.uuid)
            if result_addr not in qubit_map:
                operand = qubit_ops[i]
                operand_addr = QubitAddress(operand.uuid)
                if operand_addr in qubit_map:
                    qubit_map[result_addr] = qubit_map[operand_addr]

    def _allocate_qubit_list(
        self,
        all_qubits: list["Value"],
        results: list["Value"],
        qubit_map: QubitMap,
    ) -> None:
        """Allocate qubits for a list of qubit Values and their results.

        For view elements (operand whose ``parent_array.slice_of`` is
        set), the root-space ``QubitAddress`` is derived by walking the
        slice chain and composing the affine map.  No new physical
        qubit is allocated for a view element — the root parent's
        pre-existing physical qubit is reused — which prevents the
        previous bug where e.g. ``qft(q[1::2])`` inflated the
        circuit's qubit count beyond the true number of physical
        qubits because every view element was mistakenly registered
        under its view-local ``(view_uuid, i)`` key.
        """
        for qubit in all_qubits:
            chain_addr = self._resolve_root_qubit_address(qubit)
            if chain_addr is not None:
                qubit_addr = chain_addr
                if qubit_addr not in qubit_map:
                    # Element key missing under the root parent — this
                    # means the QInitOperation for the root was never
                    # allocated.  Treat it as a hard bug rather than
                    # silently allocating a fresh index (the prior
                    # behaviour), because doing so inflates the
                    # circuit's qubit count when a view is passed to
                    # a composite gate.
                    raise AssertionError(
                        f"Root qubit address '{str(qubit_addr)}' not found in qubit_map "
                        f"when allocating for element '{qubit.uuid}'. "
                        "The root array's QInitOperation must be allocated first."
                    )
            elif isinstance(qubit, ArrayValue):
                # Whole ``Vector[Qubit]`` operand (not a per-element
                # access): its per-element addresses are already in
                # ``qubit_map`` (placed there by the upstream
                # ``QInitOperation`` or by a prior allocator that
                # produced this array as its result, see the
                # ArrayValue→ArrayValue copy in the result loop
                # below).  Do **not** fall through to the scalar
                # fresh-allocate branch -- doing so would allocate a
                # single wire for the whole vector and leave the
                # element keys missing, which then trips
                # ``_resolve_root_qubit_address`` on a downstream op
                # that addresses elements of this vector (e.g.
                # ``qmc.iqft`` after ``ctrl_qft(c, q_out, coef)``).
                qubit_addr = None
            else:
                qubit_addr, is_array = resolve_qubit_key(qubit)
                if qubit_addr is None:
                    continue
                if qubit_addr not in qubit_map:
                    # Scalar qubit or symbolic-index element: fall back
                    # to the legacy allocation behaviour so qkernel
                    # input parameters (emit_init=False) and symbolic
                    # loop-var indices keep working.
                    qubit_map[qubit_addr] = self._next_qubit_index
                    self._next_qubit_index += 1

            if qubit_addr is not None:
                scalar_addr = QubitAddress(qubit.uuid)
                if scalar_addr not in qubit_map:
                    qubit_map[scalar_addr] = qubit_map[qubit_addr]

        for i, result in enumerate(results):
            result_addr = QubitAddress(result.uuid)
            if i < len(all_qubits):
                operand = all_qubits[i]
                # ArrayValue input → ArrayValue result: alias every
                # per-element address from the input's UUID to the
                # result's UUID.  Mirrors :meth:`_allocate_pauli_evolve`
                # and the ``SymbolicControlledU`` control-prefix branch
                # in :meth:`_allocate_controlled_u`.  Without this copy,
                # a ``ConcreteControlledU`` with a ``Vector[Qubit]``
                # sub-kernel argument leaves the next-version vector's
                # element keys unmapped, and any downstream op that
                # walks ``parent_array.uuid -> (uuid, i)`` (e.g.
                # ``qmc.iqft`` expanded into per-element CP / H gates,
                # or a ``MeasureVectorOperation`` on the result)
                # trips the ``_resolve_root_qubit_address`` assertion.
                if isinstance(operand, ArrayValue) and isinstance(result, ArrayValue):
                    copied = False
                    for addr, idx in list(qubit_map.items()):
                        if addr.matches_array(operand.uuid):
                            new_addr = QubitAddress(result.uuid, addr.element_index)
                            if new_addr not in qubit_map:
                                qubit_map[new_addr] = idx
                                copied = True
                    if copied and result_addr not in qubit_map:
                        first_elem = QubitAddress(result.uuid, 0)
                        if first_elem in qubit_map:
                            qubit_map[result_addr] = qubit_map[first_elem]
                    continue

                chain_addr = self._resolve_root_qubit_address(operand)
                if chain_addr is not None:
                    qubit_addr = chain_addr
                else:
                    qubit_addr, _ = resolve_qubit_key(operand)
                if qubit_addr is not None and qubit_addr in qubit_map:
                    physical = qubit_map[qubit_addr]
                    if result_addr not in qubit_map:
                        qubit_map[result_addr] = physical
                    # If the result is itself an array element, also
                    # register the (parent_array.uuid, idx) key so that
                    # downstream operations referencing the result's
                    # parent ``ArrayValue`` (e.g. ``MeasureVectorOperation``
                    # on a controlled-U's next-version control vector)
                    # can resolve each element through the same path
                    # used for QInit-allocated arrays.
                    result_chain_addr = self._resolve_root_qubit_address(result)
                    if (
                        result_chain_addr is not None
                        and result_chain_addr not in qubit_map
                    ):
                        qubit_map[result_chain_addr] = physical
                elif qubit_addr is not None and result_addr not in qubit_map:
                    raise AssertionError(
                        f"Missing qubit address '{str(qubit_addr)}' in qubit_map when "
                        f"allocating result '{result.uuid}'. "
                        "This indicates a bug in operand allocation."
                    )

    def _resolve_root_qubit_address(
        self,
        operand: "Value",
    ) -> QubitAddress | None:
        """Walk the slice_of chain and return the root-space QubitAddress.

        Thin wrapper over :func:`resolve_root_qubit_address` (shared with the
        frontend's ``expval`` lowering) that wraps the resolved
        ``(root_uuid, index)`` pair in a ``QubitAddress``.

        Args:
            operand (Value): The qubit operand to resolve; expected to be an
                array element with a constant index.

        Returns:
            QubitAddress | None: ``QubitAddress(root_uuid, index)`` for a
                resolvable array element, or ``None`` when the operand is not an
                array element, its index is non-constant, or the slice chain has
                a non-constant ``slice_start`` / ``slice_step`` (deferred to the
                emit-time resolver, which has bindings available).
        """
        resolved = resolve_root_qubit_address(operand)
        if resolved is None:
            return None
        root_uuid, idx = resolved
        return QubitAddress(root_uuid, idx)

    def _allocate_pauli_evolve(
        self,
        op: PauliEvolveOp,
        qubit_map: QubitMap,
    ) -> None:
        """Allocate resources for a PauliEvolveOp.

        Maps result array elements to the same physical qubits as
        the input array (identity mapping -- same qubits, new SSA values).
        """
        input_qubits = op.qubits
        result_qubits = op.evolved_qubits
        for addr, idx in list(qubit_map.items()):
            if addr.matches_array(input_qubits.uuid):
                result_addr = QubitAddress(result_qubits.uuid, addr.element_index)
                if result_addr not in qubit_map:
                    qubit_map[result_addr] = idx

    def _allocate_composite(
        self,
        op: CompositeGateOperation | InverseBlockOperation,
        qubit_map: QubitMap,
    ) -> None:
        """Allocate resources for a composite-like quantum operation."""
        all_qubits = op.control_qubits + op.target_qubits
        self._allocate_qubit_list(all_qubits, list(op.results), qubit_map)

    def _allocate_controlled_u(
        self,
        op: ControlledUOperation,
        qubit_map: QubitMap,
    ) -> None:
        """Allocate resources for a ControlledUOperation."""
        if isinstance(op, SymbolicControlledU):
            from qamomile.circuit.ir.value import ArrayValue

            # Three shapes can land here without having been promoted to
            # ``ConcreteControlledU`` by ``ConstantFoldingPass``:
            #
            #   * single-pool + ``control_indices``: one ``ArrayValue``
            #     control operand whose pass-through slots cannot be
            #     represented in ``ConcreteControlledU``'s scalar layout.
            #   * multi-arg control prefix (``num_control_args > 1``):
            #     a heterogeneous mix of scalar ``Value`` and
            #     ``ArrayValue`` operands whose qubit-count sum equals
            #     ``num_controls``.
            #   * single-pool with no ``control_indices`` but a
            #     ``num_controls`` that depends on a loop variable
            #     (``num_controls = n - 1 - k`` inside a ``qmc.range``):
            #     constant folding cannot resolve the loop variable so
            #     the promotion never fires; each unrolled iteration
            #     instead arrives at the emit pass with a fully
            #     resolvable ``num_controls``.
            #
            # All three flow through the same per-operand allocation:
            # each input operand keeps its physical mapping onto the
            # corresponding result operand.  Whether ``num_controls``
            # ultimately resolves is the emit pass's responsibility;
            # the allocator only needs to thread per-element addresses
            # through.
            for i in range(op.num_control_args):
                src = op.operands[i]
                dst = op.results[i]
                if isinstance(src, ArrayValue):
                    for addr, idx in list(qubit_map.items()):
                        if addr.matches_array(src.uuid):
                            result_addr = QubitAddress(dst.uuid, addr.element_index)
                            if result_addr not in qubit_map:
                                qubit_map[result_addr] = idx
                else:
                    src_addr = QubitAddress(src.uuid)
                    if src_addr not in qubit_map:
                        # Scalar control whose UUID is first introduced
                        # at this ``SymbolicControlledU`` -- typically a
                        # top-level ``@qkernel`` ``Qubit`` input
                        # (``emit_init=False``, so no ``QInitOperation``
                        # pre-registers it).  ``_allocate_qubit_list``
                        # already handles the same edge case for the
                        # concrete-controlled-U path (line ~527); mirror
                        # the fresh-slot allocation here so emit does
                        # not later trip on a missing scalar mapping
                        # for the control prefix.
                        qubit_map[src_addr] = self._next_qubit_index
                        self._next_qubit_index += 1
                    dst_addr = QubitAddress(dst.uuid)
                    if dst_addr not in qubit_map:
                        qubit_map[dst_addr] = qubit_map[src_addr]
            sub_quantum_operands = [
                v for v in op.operands[op.num_control_args :] if v.type.is_quantum()
            ]
            sub_quantum_results = [
                r for r in op.results[op.num_control_args :] if r.type.is_quantum()
            ]
            if sub_quantum_operands:
                self._allocate_qubit_list(
                    sub_quantum_operands, sub_quantum_results, qubit_map
                )
            return

        assert isinstance(op, ConcreteControlledU)
        control_qubits = list(op.control_operands)
        target_qubits = [v for v in op.target_operands if v.type.is_quantum()]
        all_qubits = control_qubits + target_qubits
        self._allocate_qubit_list(all_qubits, list(op.results), qubit_map)

    @staticmethod
    def _parse_composite_key(key: str) -> QubitAddress:
        """Parse a legacy composite key string into a QubitAddress.

        Delegates to ``QubitAddress.from_composite_key``.
        """
        return QubitAddress.from_composite_key(key)

    def _allocate_cast(
        self,
        op: CastOperation,
        qubit_map: QubitMap,
    ) -> None:
        """Allocate resources for a CastOperation (alias mapping)."""
        resolved = 0
        for i, qubit_uuid in enumerate(op.qubit_mapping):
            qubit_addr = self._parse_composite_key(qubit_uuid)
            if qubit_addr not in qubit_map:
                # Fallback: try as plain scalar UUID (the qubit_mapping may
                # store element UUIDs that were registered via _allocate_gate)
                qubit_addr = QubitAddress(qubit_uuid)
            if qubit_addr in qubit_map:
                result_element_addr = QubitAddress(op.results[0].uuid, i)
                qubit_map[result_element_addr] = qubit_map[qubit_addr]
                result_base_addr = QubitAddress(op.results[0].uuid)
                if result_base_addr not in qubit_map:
                    qubit_map[result_base_addr] = qubit_map[qubit_addr]
                resolved += 1
        total = len(op.qubit_mapping)
        if total > 0 and resolved < total:
            import warnings

            warnings.warn(
                f"CastOperation: {total - resolved}/{total} carrier qubits "
                f"unresolved in qubit_map. "
                f"Downstream measurements may be silently dropped.",
                stacklevel=2,
            )
