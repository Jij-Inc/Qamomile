"""Base classes for emit pass implementation.

This module provides backend-agnostic helper classes for resource allocation,
value resolution, and loop analysis. These are used by StandardEmitPass to
implement the emission logic without backend-specific code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from qamomile.circuit.transpiler.errors import ResolutionFailureReason

if TYPE_CHECKING:
    from qamomile.circuit.ir.value import Value


@dataclass
class QubitResolutionResult:
    """Result of attempting to resolve a qubit index."""

    success: bool
    index: int | None = None
    failure_reason: ResolutionFailureReason | None = None
    failure_details: str = ""


from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.operation.gate import (
    GateOperation,
    MeasureOperation,
    MeasureVectorOperation,
    MeasureQFixedOperation,
    ControlledUOperation,
)
from qamomile.circuit.ir.operation.composite_gate import CompositeGateOperation
from qamomile.circuit.ir.operation.cast import CastOperation
from qamomile.circuit.ir.operation.control_flow import (
    ForOperation,
    ForItemsOperation,
    IfOperation,
    WhileOperation,
)
from qamomile.circuit.ir.operation.arithmetic_operations import BinOp, PhiOp
from qamomile.circuit.ir.types.primitives import BitType
from qamomile.circuit.ir.value import ArrayValue


def resolve_if_condition(
    condition: Any,
    bindings: dict[str, Any],
) -> bool | None:
    """Resolve an if-condition to a compile-time boolean.

    Checks whether the condition can be statically evaluated at emit time.
    Plain Python values (int/bool captured by ``@qkernel`` from closure),
    constant-folded Values, and Values resolvable via ``bindings`` are all
    treated as compile-time constants.

    Args:
        condition: The condition from ``IfOperation`` (plain value or
            ``Value`` object).
        bindings: Current variable bindings (uuid → value and/or
            name → value).

    Returns:
        ``True``/``False`` for compile-time resolvable conditions,
        ``None`` for runtime conditions that must be dispatched to the
        backend's conditional branching protocol.
    """
    # Plain Python int/bool captured by @qkernel AST transformer
    if not hasattr(condition, "uuid"):
        return bool(condition)

    # Value with is_constant() (e.g., from constant folding)
    if hasattr(condition, "is_constant") and condition.is_constant():
        return bool(condition.get_const())

    # Bound by UUID
    if condition.uuid in bindings:
        return bool(bindings[condition.uuid])

    # Bound by name
    if hasattr(condition, "name") and condition.name and condition.name in bindings:
        return bool(bindings[condition.name])

    # Const in params
    if (
        hasattr(condition, "params")
        and condition.params
        and "const" in condition.params
    ):
        return bool(condition.params["const"])

    # Runtime condition — cannot resolve at compile time
    return None


def remap_static_phi_outputs(
    phi_ops: list[Operation],
    condition_value: bool,
    qubit_map: dict[str, int],
    clbit_map: dict[str, int],
) -> None:
    """Remap phi outputs for a compile-time constant ``IfOperation``.

    When the condition is statically known, the dead branch is never
    allocated.  Phi outputs are aliased directly to the selected
    branch's source value, bypassing the two-branch merge validation
    in ``map_phi_outputs``.

    This is the shared implementation used by both the resource allocator
    (during allocation) and the emit pass (during emission) to ensure
    scalar and array quantum phi outputs are handled identically.

    Args:
        phi_ops: Phi operations from the ``IfOperation``.
        condition_value: The resolved boolean condition.
        qubit_map: UUID-to-physical-qubit mapping (mutated in place).
        clbit_map: UUID-to-physical-clbit mapping (mutated in place).
    """
    for phi in phi_ops:
        if not isinstance(phi, PhiOp):
            continue
        output = phi.results[0]
        # operands: [condition, true_val, false_val]
        selected_val = phi.operands[1] if condition_value else phi.operands[2]

        if output.uuid in qubit_map or output.uuid in clbit_map:
            continue

        if output.type.is_quantum():
            if isinstance(output, ArrayValue):
                # Copy composite element keys from selected branch
                prefix = (
                    f"{selected_val.uuid}_"
                    if isinstance(selected_val, ArrayValue)
                    else ""
                )
                for key, phys_idx in list(qubit_map.items()):
                    if prefix and key.startswith(prefix):
                        suffix = key[len(prefix) :]
                        out_key = f"{output.uuid}_{suffix}"
                        if out_key not in qubit_map:
                            qubit_map[out_key] = phys_idx
            else:
                # Scalar qubit: alias to selected branch's physical qubit
                key, _ = ResourceAllocator._resolve_qubit_key(selected_val)
                phys = qubit_map.get(
                    selected_val.uuid,
                    qubit_map.get(key) if key is not None else None,
                )
                if phys is not None:
                    qubit_map[output.uuid] = phys
        else:
            # Classical bit: alias to selected branch
            if selected_val.uuid in clbit_map:
                clbit_map[output.uuid] = clbit_map[selected_val.uuid]


def map_phi_outputs(
    phi_ops: list[Operation],
    qubit_map: dict[str, int],
    clbit_map: dict[str, int],
    resolve_scalar_qubit: Any = None,
) -> None:
    """Register phi output UUIDs to the same physical resources as their source operands.

    After an if-else block, PhiOp merges values from both branches.
    The phi output is a new Value with a new UUID that must map to the
    same physical qubit (or classical bit) as the branch values.

    For ArrayValue phi outputs (e.g., qubit arrays), this copies all
    composite element keys ``{source_uuid}_{i}`` →
    ``{output_uuid}_{i}`` so that subsequent element accesses on the
    phi output resolve to the correct physical qubits.

    Args:
        phi_ops: List of phi operations from an IfOperation.
        qubit_map: Mapping from UUID/composite key to physical qubit index.
        clbit_map: Mapping from UUID to physical classical bit index.
        resolve_scalar_qubit: Optional callback ``(source, qubit_map) -> int | None``
            for resolving scalar qubit values that are not directly in
            ``qubit_map`` (e.g., array element resolution).  When *None*,
            ``ResourceAllocator._resolve_qubit_key`` is used as fallback.

    Raises:
        EmitError: When a quantum PhiOp would merge different or partially
            unresolved physical qubit resources across branches.  This means
            the qubit identity depends on the runtime branch condition and
            cannot be statically resolved.
    """
    for phi in phi_ops:
        if not isinstance(phi, PhiOp):
            continue
        output = phi.results[0]
        true_val = phi.operands[1]
        false_val = phi.operands[2]

        if output.uuid in qubit_map or output.uuid in clbit_map:
            continue

        # Quantum types: map phi output to same physical qubit.
        # Both branches must resolve to identical physical resources;
        # mismatches indicate a quantum phi merge that cannot be
        # statically resolved (the qubit identity depends on runtime).
        if output.type.is_quantum():
            if isinstance(output, ArrayValue):
                # ArrayValue: collect suffix→phys_idx for each branch
                true_prefix = (
                    f"{true_val.uuid}_" if isinstance(true_val, ArrayValue) else ""
                )
                false_prefix = (
                    f"{false_val.uuid}_" if isinstance(false_val, ArrayValue) else ""
                )
                true_mapping: dict[str, int] = {}
                false_mapping: dict[str, int] = {}
                for key, phys_idx in qubit_map.items():
                    if true_prefix and key.startswith(true_prefix):
                        true_mapping[key[len(true_prefix) :]] = phys_idx
                    if false_prefix and key.startswith(false_prefix):
                        false_mapping[key[len(false_prefix) :]] = phys_idx

                if true_mapping or false_mapping:
                    all_suffixes = set(true_mapping) | set(false_mapping)
                    for suffix in sorted(all_suffixes):
                        true_idx = true_mapping.get(suffix)
                        false_idx = false_mapping.get(suffix)
                        if true_idx is None or false_idx is None:
                            from qamomile.circuit.transpiler.errors import EmitError

                            raise EmitError(
                                "Quantum PhiOp merge requires identical physical "
                                "resources across branches",
                                operation="PhiOp",
                            )
                        if true_idx != false_idx:
                            from qamomile.circuit.transpiler.errors import EmitError

                            raise EmitError(
                                "Quantum PhiOp merge requires identical physical "
                                "resources across branches",
                                operation="PhiOp",
                            )
                        out_key = f"{output.uuid}_{suffix}"
                        if out_key not in qubit_map:
                            qubit_map[out_key] = true_idx
            else:
                # Scalar qubit phi output: resolve both branches
                def _resolve_one(source: Any) -> int | None:
                    if source.uuid in qubit_map:
                        return qubit_map[source.uuid]
                    if resolve_scalar_qubit is not None:
                        return resolve_scalar_qubit(source, qubit_map)
                    key, _ = ResourceAllocator._resolve_qubit_key(source)
                    return qubit_map.get(key) if key is not None else None

                true_phys = _resolve_one(true_val)
                false_phys = _resolve_one(false_val)

                if true_phys is not None and false_phys is not None:
                    if true_phys != false_phys:
                        from qamomile.circuit.transpiler.errors import EmitError

                        raise EmitError(
                            "Quantum PhiOp merge requires identical physical "
                            "resources across branches",
                            operation="PhiOp",
                        )
                    qubit_map[output.uuid] = true_phys
                elif true_phys is not None or false_phys is not None:
                    from qamomile.circuit.transpiler.errors import EmitError

                    raise EmitError(
                        "Quantum PhiOp merge requires identical physical "
                        "resources across branches",
                        operation="PhiOp",
                    )

        # Classical bit types: consolidate both branches to the same
        # physical clbit.  Under Qiskit's ``if_test`` only one branch
        # executes, so both branches' measurements must target the same
        # physical clbit — otherwise the phi output always reads the
        # first-found branch (which was always the true branch).
        elif isinstance(output.type, BitType):
            if isinstance(output, ArrayValue):
                # ArrayValue BitType: consolidate per-element keys
                true_src = true_val if isinstance(true_val, ArrayValue) else None
                false_src = false_val if isinstance(false_val, ArrayValue) else None
                primary = true_src or false_src
                secondary = false_src if true_src is not None else true_src

                if primary is not None:
                    for key, phys_idx in list(clbit_map.items()):
                        prefix = f"{primary.uuid}_"
                        if key.startswith(prefix):
                            suffix = key[len(prefix) :]
                            # Map phi output element to primary's clbit
                            out_key = f"{output.uuid}_{suffix}"
                            if out_key not in clbit_map:
                                clbit_map[out_key] = phys_idx
                            # Redirect secondary branch element to same clbit
                            if secondary is not None:
                                sec_key = f"{secondary.uuid}_{suffix}"
                                if sec_key in clbit_map:
                                    clbit_map[sec_key] = phys_idx
            else:
                # Scalar BitType: pick the first available clbit as
                # canonical, redirect the other branch to it.
                #
                # NOTE: For if-only (no else), false_val is the pre-if
                # value of the variable.  When this runs inside a while
                # loop body, false_val.uuid may be the while-condition
                # UUID.  The redirect below will overwrite that UUID's
                # canonical clbit.  The WhileOperation allocation branch
                # compensates by saving/restoring init_uuid's clbit
                # around body allocation.
                true_clbit = clbit_map.get(true_val.uuid)
                false_clbit = clbit_map.get(false_val.uuid)

                if true_clbit is not None:
                    clbit_map[output.uuid] = true_clbit
                    # Redirect false branch to write to same physical clbit
                    if false_clbit is not None and false_clbit != true_clbit:
                        clbit_map[false_val.uuid] = true_clbit
                elif false_clbit is not None:
                    clbit_map[output.uuid] = false_clbit


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

    def __init__(self) -> None:
        self._next_qubit_index: int = 0
        self._next_clbit_index: int = 0

    def allocate(
        self,
        operations: list[Operation],
        bindings: dict[str, Any] | None = None,
    ) -> tuple[dict[str, int], dict[str, int]]:
        """Allocate qubit and clbit indices for all operations.

        Args:
            operations: List of operations to allocate resources for
            bindings: Optional variable bindings for resolving dynamic sizes

        Returns:
            Tuple of (qubit_map, clbit_map) where each maps UUID to index
        """
        qubit_map: dict[str, int] = {}
        clbit_map: dict[str, int] = {}
        self._next_qubit_index = 0
        self._next_clbit_index = 0
        self._allocate_recursive(operations, qubit_map, clbit_map, bindings or {})
        return qubit_map, clbit_map

    def _allocate_recursive(
        self,
        operations: list[Operation],
        qubit_map: dict[str, int],
        clbit_map: dict[str, int],
        bindings: dict[str, Any],
    ) -> None:
        """Recursively allocate resources from operations."""
        for op in operations:
            if isinstance(op, QInitOperation):
                result = op.results[0]
                if isinstance(result, ArrayValue):
                    # Allocate physical qubits for array elements using
                    # {array_uuid}_{i} keys.  At this stage only these
                    # composite keys are registered; the individual element
                    # Values (which carry their own UUIDs) are created
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
                            qubit_id = f"{result.uuid}_{i}"
                            if qubit_id not in qubit_map:
                                qubit_map[qubit_id] = self._next_qubit_index
                                self._next_qubit_index += 1
                    continue
                if result.uuid not in qubit_map:
                    qubit_map[result.uuid] = self._next_qubit_index
                    self._next_qubit_index += 1

            elif isinstance(op, MeasureOperation):
                result = op.results[0]
                if result.uuid not in clbit_map:
                    clbit_map[result.uuid] = self._next_clbit_index
                    self._next_clbit_index += 1

            elif isinstance(op, MeasureVectorOperation):
                result = op.results[0]
                if isinstance(result, ArrayValue) and result.shape:
                    size_val = result.shape[0]
                    size = self._resolve_size(size_val, bindings)
                    if size is not None:
                        for i in range(size):
                            clbit_id = f"{result.uuid}_{i}"
                            if clbit_id not in clbit_map:
                                clbit_map[clbit_id] = self._next_clbit_index
                                self._next_clbit_index += 1

            elif isinstance(op, MeasureQFixedOperation):
                qfixed = op.operands[0]
                qubit_uuids = qfixed.params.get("qubit_values", [])
                result = op.results[0]
                for i, qubit_uuid in enumerate(qubit_uuids):
                    clbit_id = f"{result.uuid}_{i}"
                    if clbit_id not in clbit_map:
                        clbit_map[clbit_id] = self._next_clbit_index
                        self._next_clbit_index += 1

            elif isinstance(op, GateOperation):
                self._allocate_gate(op, qubit_map)

            elif isinstance(op, ForOperation):
                self._allocate_recursive(op.operations, qubit_map, clbit_map, bindings)

            elif isinstance(op, ForItemsOperation):
                # ForItemsOperation allocates resources for its loop body
                self._allocate_recursive(op.operations, qubit_map, clbit_map, bindings)

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
                    init_uuid = (
                        init_val.uuid if hasattr(init_val, "uuid") else str(init_val)
                    )

                    # Save the canonical clbit for the initial condition
                    # BEFORE body allocation.  An if-only (no else) inside
                    # the loop body produces a PhiOp whose false_val is the
                    # pre-if while-condition value.  map_phi_outputs will
                    # redirect that false_val UUID to the true-branch clbit,
                    # overwriting clbit_map[init_uuid] and making the
                    # post-body mismatch detection ineffective.
                    saved_init_clbit = clbit_map.get(init_uuid)

                    # Allocate the loop body so that IfOperation phi
                    # mappings inside the body are fully resolved.
                    self._allocate_recursive(
                        op.operations, qubit_map, clbit_map, bindings
                    )

                    # Restore the canonical clbit for init_uuid if it was
                    # overwritten during body allocation.
                    if saved_init_clbit is not None:
                        clbit_map[init_uuid] = saved_init_clbit

                    carried_val = (
                        loop_carried.value
                        if hasattr(loop_carried, "value")
                        else loop_carried
                    )
                    carried_uuid = (
                        carried_val.uuid
                        if hasattr(carried_val, "uuid")
                        else str(carried_val)
                    )
                    carried_clbit = clbit_map.get(carried_uuid)

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
                        clbit_map[carried_uuid] = saved_init_clbit
                        self._alias_loop_carried_clbits(
                            op.operations,
                            carried_uuid,
                            saved_init_clbit,
                            clbit_map,
                        )
                    elif saved_init_clbit is not None and carried_uuid not in clbit_map:
                        clbit_map[carried_uuid] = saved_init_clbit
                else:
                    assert False, (
                        "[FOR DEVELOPER] WhileOperation must have exactly 2 "
                        "operands to reach this branch, but got "
                        f"{len(op.operands)}. This indicates a bug in the "
                        "WhileOperation construction."
                    )

            elif isinstance(op, CompositeGateOperation):
                self._allocate_composite(op, qubit_map)

            elif isinstance(op, ControlledUOperation):
                self._allocate_controlled_u(op, qubit_map)

            elif isinstance(op, CastOperation):
                self._allocate_cast(op, qubit_map)

    def _alias_loop_carried_clbits(
        self,
        operations: list[Operation],
        target_uuid: str,
        canonical_clbit: int,
        clbit_map: dict[str, int],
    ) -> None:
        """Recursively trace PhiOp sources and alias them to *canonical_clbit*.

        When a while loop body contains an if-else with measurements in
        both branches, the phi-merged result (the loop-carried condition)
        and all its upstream branch-measurement UUIDs must write to the
        same classical bit as the initial while condition.
        """
        for op in operations:
            if not isinstance(op, IfOperation):
                continue
            for phi in op.phi_ops:
                if not isinstance(phi, PhiOp):
                    continue
                output = phi.results[0]
                if output.uuid != target_uuid:
                    continue
                true_val = phi.operands[1]
                false_val = phi.operands[2]
                if true_val.uuid in clbit_map:
                    clbit_map[true_val.uuid] = canonical_clbit
                if false_val.uuid in clbit_map:
                    clbit_map[false_val.uuid] = canonical_clbit
                # Recurse into branches for nested if-else
                self._alias_loop_carried_clbits(
                    op.true_operations, true_val.uuid, canonical_clbit, clbit_map
                )
                self._alias_loop_carried_clbits(
                    op.false_operations, false_val.uuid, canonical_clbit, clbit_map
                )

    def _allocate_phi_ops(
        self,
        phi_ops: list[Operation],
        qubit_map: dict[str, int],
        clbit_map: dict[str, int],
    ) -> None:
        """Register phi output UUIDs via the shared ``map_phi_outputs`` utility."""
        map_phi_outputs(phi_ops, qubit_map, clbit_map)

    def _remap_static_phi_outputs(
        self,
        phi_ops: list[Operation],
        condition_value: bool,
        qubit_map: dict[str, int],
        clbit_map: dict[str, int],
    ) -> None:
        """Remap phi outputs for a compile-time constant ``IfOperation``.

        Delegates to the module-level ``remap_static_phi_outputs`` helper,
        which is shared with the emit pass to ensure scalar and array
        quantum phi outputs are handled identically at both allocation
        and emission time.

        Args:
            phi_ops: Phi operations from the ``IfOperation``.
            condition_value: The resolved boolean condition.
            qubit_map: UUID-to-physical-qubit mapping (mutated in place).
            clbit_map: UUID-to-physical-clbit mapping (mutated in place).
        """
        remap_static_phi_outputs(phi_ops, condition_value, qubit_map, clbit_map)

    def _resolve_size(
        self,
        size_val: Any,
        bindings: dict[str, Any],
    ) -> int | None:
        """Resolve a size value to a concrete integer.

        Args:
            size_val: The size value (may be constant or symbolic)
            bindings: Variable bindings for resolution

        Returns:
            Resolved size as int, or None if cannot resolve
        """
        import re

        # Check for constant in params
        if hasattr(size_val, "params") and "const" in size_val.params:
            return int(size_val.params["const"])

        # Check if it's a Value with parent_array (e.g., hi.shape[0])
        if hasattr(size_val, "parent_array") and size_val.parent_array is not None:
            array_name = size_val.parent_array.name
            if array_name in bindings:
                array_data = bindings[array_name]
                # Get the shape/length of the bound array
                if hasattr(array_data, "__len__"):
                    return len(array_data)

        # Check by name in bindings
        if hasattr(size_val, "name") and size_val.name in bindings:
            bound = bindings[size_val.name]
            if isinstance(bound, (int, float)):
                return int(bound)
            if hasattr(bound, "__len__"):
                return len(bound)

        # Check by uuid in bindings
        if hasattr(size_val, "uuid") and size_val.uuid in bindings:
            bound = bindings[size_val.uuid]
            if isinstance(bound, (int, float)):
                return int(bound)

        # Check for dimension naming pattern (e.g., "hi_dim0" -> array "hi", dimension 0)
        # This handles cases where parent_array is None after inlining
        if hasattr(size_val, "name") and size_val.name:
            match = re.match(r"^(.+)_dim(\d+)$", size_val.name)
            if match:
                array_name = match.group(1)
                dim_index = int(match.group(2))
                if array_name in bindings:
                    array_data = bindings[array_name]
                    # Get shape at specified dimension
                    if hasattr(array_data, "shape"):
                        # numpy array or similar
                        if dim_index < len(array_data.shape):
                            return int(array_data.shape[dim_index])
                    elif dim_index == 0 and hasattr(array_data, "__len__"):
                        # For 1D sequences, dim0 is length
                        return len(array_data)

        return None

    def _allocate_gate(
        self,
        op: GateOperation,
        qubit_map: dict[str, int],
    ) -> None:
        """Allocate resources for a GateOperation."""
        # GateOperation represents a unitary gate: qubit count is preserved.
        assert len(op.results) == len(op.operands), (
            f"GateOperation must have equal operands and results, "
            f"got {len(op.operands)} operands and {len(op.results)} results."
        )

        # Phase 1: Register all operands in qubit_map.
        # Element Values are created dynamically during frontend tracing
        # (handle/array.py _get_element), so their UUIDs are unknown at
        # QInitOperation time.  Here we lazily map each element UUID to
        # the physical qubit already allocated under the {parent_uuid}_{idx} key.
        for operand in op.operands:
            if operand.uuid not in qubit_map:
                if operand.parent_array is not None and operand.element_indices:
                    # Array element: resolve via parent_array key
                    parent_uuid = operand.parent_array.uuid
                    idx_value = operand.element_indices[0]
                    if idx_value.is_constant():
                        idx = int(idx_value.get_const())
                        key = f"{parent_uuid}_{idx}"
                        assert key in qubit_map, (
                            f"Array element key {key!r} not found in qubit_map. "
                            f"This indicates a bug in the transpiler pipeline: "
                            f"QInitOperation for the parent array was not processed "
                            f"before this GateOperation."
                        )
                        qubit_map[operand.uuid] = qubit_map[key]
                    # Non-constant indices (symbolic loop vars) are resolved
                    # at emit time via ValueResolver.resolve_qubit_index_detailed.
                else:
                    # Scalar qubit: allocate new index.
                    # This path is used for @qkernel input parameters created with
                    # emit_init=False (func_to_block.py), which have no QInitOperation.
                    # ResourceAllocator.allocate() receives only operations, not the
                    # block's input_values, so these qubits are first registered here.
                    qubit_map[operand.uuid] = self._next_qubit_index
                    self._next_qubit_index += 1

        # Phase 2: Map each result to its corresponding operand (1:1)
        for i, result in enumerate(op.results):
            if result.uuid not in qubit_map:
                operand = op.operands[i]
                if operand.uuid in qubit_map:
                    qubit_map[result.uuid] = qubit_map[operand.uuid]

    @staticmethod
    def _resolve_qubit_key(qubit: "Value") -> tuple[str | None, bool]:
        """Resolve a qubit Value to its allocation key.

        Returns:
            (key, is_array_element) where key is the qubit_map key
            or None if the index is non-constant.
        """
        if qubit.parent_array is not None and qubit.element_indices:
            parent_uuid = qubit.parent_array.uuid
            idx_value = qubit.element_indices[0]
            if idx_value.is_constant():
                idx = int(idx_value.get_const())
                return f"{parent_uuid}_{idx}", True
            return None, True
        return qubit.uuid, False

    def _allocate_qubit_list(
        self,
        all_qubits: list["Value"],
        results: list["Value"],
        qubit_map: dict[str, int],
    ) -> None:
        """Allocate qubits for a list of qubit Values and their results."""
        for qubit in all_qubits:
            qubit_key, is_array = self._resolve_qubit_key(qubit)
            if qubit_key is not None:
                if qubit_key not in qubit_map:
                    # New allocation for qubits not yet registered.
                    # For scalar qubits, this handles @qkernel input parameters
                    # (emit_init=False) that have no preceding QInitOperation.
                    # For array elements, this handles first-seen element keys.
                    qubit_map[qubit_key] = self._next_qubit_index
                    self._next_qubit_index += 1
                if qubit.uuid not in qubit_map:
                    qubit_map[qubit.uuid] = qubit_map[qubit_key]

        for i, result in enumerate(results):
            if result.uuid not in qubit_map and i < len(all_qubits):
                qubit_key, is_array = self._resolve_qubit_key(all_qubits[i])
                if qubit_key is not None:
                    # Alias: result must map to the same physical index as its
                    # corresponding operand.  If qubit_key is somehow missing
                    # (should not happen after the operand loop above), this is
                    # a bug — raise explicitly rather than silently allocating.
                    if qubit_key in qubit_map:
                        qubit_map[result.uuid] = qubit_map[qubit_key]
                    else:
                        raise AssertionError(
                            f"Missing qubit_key '{qubit_key}' in qubit_map when "
                            f"allocating result '{result.uuid}'. "
                            "This indicates a bug in operand allocation."
                        )

    def _allocate_composite(
        self,
        op: CompositeGateOperation,
        qubit_map: dict[str, int],
    ) -> None:
        """Allocate resources for a CompositeGateOperation."""
        all_qubits = op.control_qubits + op.target_qubits
        self._allocate_qubit_list(all_qubits, list(op.results), qubit_map)

    def _allocate_controlled_u(
        self,
        op: ControlledUOperation,
        qubit_map: dict[str, int],
    ) -> None:
        """Allocate resources for a ControlledUOperation."""
        if op.has_index_spec:
            # Vector already allocated by QInitOperation.
            # Map result ArrayValue to same physical qubits.
            vector_operand = op.operands[1]
            vector_result = op.results[0]
            for key, idx in list(qubit_map.items()):
                if key.startswith(f"{vector_operand.uuid}_"):
                    suffix = key[len(f"{vector_operand.uuid}_") :]
                    result_key = f"{vector_result.uuid}_{suffix}"
                    if result_key not in qubit_map:
                        qubit_map[result_key] = idx
            return

        if op.is_symbolic_num_controls:
            from qamomile.circuit.transpiler.errors import EmitError

            raise EmitError(
                "Cannot transpile ControlledUOperation with symbolic num_controls. "
                "Bind parameters to concrete values before transpilation.",
                operation="ControlledUOperation",
            )
        control_qubits = list(op.control_operands)
        target_qubits = [
            v for v in op.target_operands if hasattr(v, "type") and v.type.is_quantum()
        ]
        all_qubits = control_qubits + target_qubits
        self._allocate_qubit_list(all_qubits, list(op.results), qubit_map)

    def _allocate_cast(
        self,
        op: CastOperation,
        qubit_map: dict[str, int],
    ) -> None:
        """Allocate resources for a CastOperation (alias mapping)."""
        resolved = 0
        for i, qubit_uuid in enumerate(op.qubit_mapping):
            if qubit_uuid in qubit_map:
                result_element_id = f"{op.results[0].uuid}_{i}"
                qubit_map[result_element_id] = qubit_map[qubit_uuid]
                if op.results[0].uuid not in qubit_map:
                    qubit_map[op.results[0].uuid] = qubit_map[qubit_uuid]
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


class ValueResolver:
    """Resolves Value objects to concrete indices or values.

    Handles the mapping from IR Value objects to physical qubit indices,
    classical values, and parameter names during emission.
    """

    def __init__(self, parameters: set[str] | None = None):
        """Initialize the resolver.

        Args:
            parameters: Set of parameter names to preserve as symbolic
        """
        self.parameters = parameters or set()

    def resolve_qubit_index(
        self,
        v: "Value",
        qubit_map: dict[str, int],
        bindings: dict[str, Any],
    ) -> int | None:
        """Resolve a Value to a physical qubit index.

        Args:
            v: The Value to resolve
            qubit_map: Mapping from UUID to qubit index
            bindings: Current variable bindings

        Returns:
            Physical qubit index, or None if cannot resolve
        """
        result = self.resolve_qubit_index_detailed(v, qubit_map, bindings)
        return result.index if result.success else None

    def resolve_qubit_index_detailed(
        self,
        v: "Value",
        qubit_map: dict[str, int],
        bindings: dict[str, Any],
    ) -> QubitResolutionResult:
        """Resolve a Value to a physical qubit index with detailed failure info.

        Args:
            v: The Value to resolve
            qubit_map: Mapping from UUID to qubit index
            bindings: Current variable bindings

        Returns:
            QubitResolutionResult with success status and either index or failure details
        """
        # Check array element first
        if v.parent_array is not None and v.element_indices:
            parent_uuid = v.parent_array.uuid
            idx_value = v.element_indices[0]

            idx = None
            if idx_value.is_constant():
                idx = int(idx_value.get_const())
            elif idx_value.name in bindings:
                bound_val = bindings[idx_value.name]
                if isinstance(bound_val, (int, float)):
                    idx = int(bound_val)
                else:
                    return QubitResolutionResult(
                        success=False,
                        failure_reason=ResolutionFailureReason.INDEX_NOT_NUMERIC,
                        failure_details=(
                            f"Index '{idx_value.name}' resolved to non-numeric type: "
                            f"{type(bound_val).__name__}"
                        ),
                    )
            elif idx_value.uuid in bindings:
                bound_val = bindings[idx_value.uuid]
                if isinstance(bound_val, (int, float)):
                    idx = int(bound_val)
                else:
                    return QubitResolutionResult(
                        success=False,
                        failure_reason=ResolutionFailureReason.INDEX_NOT_NUMERIC,
                        failure_details=(
                            f"Index (uuid: {idx_value.uuid[:8]}...) resolved to "
                            f"non-numeric type: {type(bound_val).__name__}"
                        ),
                    )
            elif idx_value.parent_array is not None:
                # Nested array access (e.g., edges[e, 0])
                nested_result = self._resolve_array_element_value(idx_value, bindings)
                if nested_result is None:
                    array_name = idx_value.parent_array.name
                    return QubitResolutionResult(
                        success=False,
                        failure_reason=ResolutionFailureReason.NESTED_ARRAY_RESOLUTION_FAILED,
                        failure_details=(
                            f"Nested array access '{array_name}[...]' could not be resolved. "
                            f"Array '{array_name}' may not be in bindings."
                        ),
                    )
                idx = int(nested_result)
            else:
                return QubitResolutionResult(
                    success=False,
                    failure_reason=ResolutionFailureReason.SYMBOLIC_INDEX_NOT_BOUND,
                    failure_details=(
                        f"Index variable '{idx_value.name}' is not bound. "
                        f"Neither name nor uuid found in bindings."
                    ),
                )

            if idx is not None:
                array_qubit_id = f"{parent_uuid}_{idx}"
                if array_qubit_id in qubit_map:
                    return QubitResolutionResult(
                        success=True, index=qubit_map[array_qubit_id]
                    )
                else:
                    return QubitResolutionResult(
                        success=False,
                        failure_reason=ResolutionFailureReason.ARRAY_ELEMENT_NOT_IN_QUBIT_MAP,
                        failure_details=(
                            f"Computed qubit ID '{array_qubit_id}' not found in qubit_map. "
                            f"Index {idx} may be out of bounds for array '{v.parent_array.name}'."
                        ),
                    )

        # Direct UUID lookup
        if v.uuid in qubit_map:
            return QubitResolutionResult(success=True, index=qubit_map[v.uuid])

        return QubitResolutionResult(
            success=False,
            failure_reason=ResolutionFailureReason.DIRECT_UUID_NOT_FOUND,
            failure_details=(
                f"Value uuid '{v.uuid[:8]}...' not found in qubit_map "
                f"and is not an array element."
            ),
        )

    def resolve_classical_value(
        self,
        value: "Value",
        bindings: dict[str, Any],
    ) -> Any:
        """Resolve a classical Value to a concrete Python value.

        Args:
            value: The Value to resolve
            bindings: Current variable bindings

        Returns:
            Resolved value (int, float, etc.), or None if cannot resolve
        """
        if value.is_constant():
            return value.get_const()

        if value.uuid in bindings:
            return bindings[value.uuid]

        if value.name in bindings:
            return bindings[value.name]

        if value.params and "const" in value.params:
            return value.params["const"]

        if value.parent_array is not None and value.element_indices:
            return self._resolve_array_element_value(value, bindings)

        return None

    def _resolve_array_element_value(
        self,
        v: "Value",
        bindings: dict[str, Any],
    ) -> int | float | None:
        """Resolve an array element to a concrete value.

        Args:
            v: The array element Value
            bindings: Current variable bindings

        Returns:
            Resolved numeric value, or None if cannot resolve
        """
        if v.parent_array is None or not v.element_indices:
            return None

        array_name = v.parent_array.name
        if array_name not in bindings:
            return None

        array_data = bindings[array_name]

        indices = []
        for idx in v.element_indices:
            idx_val = self.resolve_int_value(idx, bindings)
            if idx_val is None:
                return None
            indices.append(idx_val)

        try:
            import numbers

            result = array_data
            for idx in indices:
                result = result[idx]
            # Check for numeric types including numpy integers/floats
            if isinstance(result, numbers.Real):
                return result
            return None
        except (IndexError, TypeError, KeyError):
            return None

    def resolve_int_value(
        self,
        val: Any,
        bindings: dict[str, Any],
    ) -> int | None:
        """Resolve a value to an integer (for loop bounds, etc.).

        Args:
            val: The value to resolve
            bindings: Current variable bindings

        Returns:
            Integer value, or None if cannot resolve
        """
        from qamomile.circuit.ir.value import Value

        if isinstance(val, (int, float)):
            return int(val)
        elif isinstance(val, Value):
            if val.is_constant():
                return int(val.get_const())
            elif val.is_parameter():
                param_name = val.parameter_name()
                if param_name and param_name in bindings:
                    bound_val = bindings[param_name]
                    if isinstance(bound_val, (int, float)):
                        return int(bound_val)
                    return None
            elif val.uuid in bindings:
                bound_val = bindings[val.uuid]
                if isinstance(bound_val, (int, float)):
                    return int(bound_val)
                return None
            elif val.name in bindings:
                bound_val = bindings[val.name]
                if isinstance(bound_val, (int, float)):
                    return int(bound_val)
                return None
        return 0

    def get_parameter_key(
        self,
        value: "Value",
        bindings: dict[str, Any],
    ) -> str | None:
        """Get parameter key if this value should be a symbolic parameter.

        Args:
            value: The Value to check
            bindings: Current bindings (for resolving array indices)

        Returns:
            Parameter key (e.g., "gammas[0]") if symbolic, None otherwise
        """
        if value.name in self.parameters:
            return value.name

        if value.parent_array is not None:
            parent_name = value.parent_array.name
            if parent_name in self.parameters:
                if value.element_indices and len(value.element_indices) > 0:
                    idx_value = value.element_indices[0]
                    idx = None
                    if idx_value.is_constant():
                        idx = int(idx_value.get_const())
                    elif idx_value.name in bindings:
                        bound_val = bindings[idx_value.name]
                        if isinstance(bound_val, (int, float)):
                            idx = int(bound_val)
                    elif idx_value.uuid in bindings:
                        bound_val = bindings[idx_value.uuid]
                        if isinstance(bound_val, (int, float)):
                            idx = int(bound_val)

                    if idx is not None:
                        return f"{parent_name}[{idx}]"

        return None


class LoopAnalyzer:
    """Analyzes loop structures to determine emission strategy.

    Determines whether loops should use native backend control flow
    or be unrolled at emission time.
    """

    def should_unroll(
        self,
        op: ForOperation,
        bindings: dict[str, Any],
    ) -> bool:
        """Determine if a ForOperation should be unrolled.

        Args:
            op: The ForOperation to analyze
            bindings: Current variable bindings

        Returns:
            True if loop should be unrolled, False for native emission
        """
        # Check for dynamic nested loops
        if self._has_dynamic_nested_loop(op.operations, bindings, op.loop_var):
            return True

        # Check for array element access
        if self._has_array_element_access(op.operations, op.loop_var):
            return True

        # Check for BinOps that depend on the loop variable.
        # These produce values used as array indices or gate angles
        # and require concrete loop variable values to evaluate.
        if self._has_loop_var_binop(op.operations, op.loop_var):
            return True

        return False

    def _has_loop_var_binop(
        self,
        operations: list[Operation],
        loop_var: str,
    ) -> bool:
        """Check if operations contain BinOps that reference the loop variable.

        When a BinOp depends on the loop variable, its result cannot be
        evaluated with native loop control flow (the variable is symbolic).
        The loop must be unrolled so the BinOp can be evaluated with
        concrete iteration values.

        Args:
            operations: List of IR operations to inspect.
            loop_var: Name of the enclosing loop variable to detect.

        Returns:
            True if any BinOp operand directly references *loop_var*.
        """
        from qamomile.circuit.ir.value import Value

        for op in operations:
            if isinstance(op, BinOp):
                for operand in op.operands:
                    if isinstance(operand, Value) and operand.name == loop_var:
                        return True
            elif isinstance(op, ForOperation):
                if self._has_loop_var_binop(op.operations, loop_var):
                    return True
            elif isinstance(op, IfOperation):
                if self._has_loop_var_binop(
                    op.true_operations, loop_var
                ) or self._has_loop_var_binop(op.false_operations, loop_var):
                    return True
            elif isinstance(op, WhileOperation):
                if self._has_loop_var_binop(op.operations, loop_var):
                    return True
            elif isinstance(op, ForItemsOperation):
                # ForItemsOperation is unrolled for its own iteration, but its
                # body may contain BinOps referencing the *outer* loop variable.
                # Those BinOps still need concrete values, so we must recurse.
                if self._has_loop_var_binop(op.operations, loop_var):
                    return True
            # No action for other operation types (GateOperation, CastOperation, etc.)
            # — only BinOps and control-flow containers can carry loop-var dependencies.
        return False

    def _has_dynamic_nested_loop(
        self,
        operations: list[Operation],
        bindings: dict[str, Any],
        parent_loop_var: str,
    ) -> bool:
        """Check if operations contain nested loops depending on parent loop variable."""
        for op in operations:
            if isinstance(op, ForOperation):
                for bound_val in op.operands[:3]:
                    if hasattr(bound_val, "name"):
                        if bound_val.name == parent_loop_var:
                            return True
                        if bound_val.name in bindings:
                            bound = bindings[bound_val.name]
                            if not isinstance(bound, (int, float)):
                                return True
                if self._has_dynamic_nested_loop(
                    op.operations, bindings, parent_loop_var
                ):
                    return True
        return False

    def _has_array_element_access(
        self,
        operations: list[Operation],
        loop_var: str,
    ) -> bool:
        """Check if operations access array elements using loop variable."""
        from qamomile.circuit.ir.value import Value as _Value

        for op in operations:
            if isinstance(op, GateOperation):
                for v in op.operands:
                    if v.parent_array is not None and v.element_indices:
                        for idx in v.element_indices:
                            if self._index_depends_on_loop_var(idx, loop_var):
                                return True

                if isinstance(op.theta, _Value):
                    if op.theta.parent_array is not None and op.theta.element_indices:
                        for idx in op.theta.element_indices:
                            if self._index_depends_on_loop_var(idx, loop_var):
                                return True

            elif isinstance(op, BinOp):
                for operand in [op.lhs, op.rhs]:
                    if operand.parent_array is not None and operand.element_indices:
                        for idx in operand.element_indices:
                            if self._index_depends_on_loop_var(idx, loop_var):
                                return True

            elif isinstance(op, ControlledUOperation):
                for v in op.operands:
                    if isinstance(v, _Value):
                        if v.parent_array is not None and v.element_indices:
                            for idx in v.element_indices:
                                if self._index_depends_on_loop_var(idx, loop_var):
                                    return True

            elif isinstance(op, ForOperation):
                if self._has_array_element_access(op.operations, loop_var):
                    return True

        return False

    def _index_depends_on_loop_var(self, idx: "Value", loop_var: str) -> bool:
        """Check if an index depends on the loop variable."""
        if idx.name == loop_var:
            return True

        if idx.parent_array is not None and idx.element_indices:
            for sub_idx in idx.element_indices:
                if self._index_depends_on_loop_var(sub_idx, loop_var):
                    return True

        return False


class CompositeDecomposer:
    """Decomposes composite gates into primitive operations.

    Provides algorithms for QFT, IQFT, and QPE decomposition that
    backends can use for fallback when native implementations are
    unavailable.
    """

    @staticmethod
    def qft_structure(n: int) -> list[tuple[str, tuple[int, ...], float | None]]:
        """Generate QFT gate sequence.

        Args:
            n: Number of qubits

        Returns:
            List of (gate_name, qubit_indices, angle) tuples
        """
        import math

        gates = []
        for i in range(n):
            gates.append(("h", (i,), None))
            for j in range(i + 1, n):
                k = j - i
                angle = math.pi / (2**k)
                gates.append(("cp", (j, i), angle))

        # Swaps for bit order reversal
        for i in range(n // 2):
            gates.append(("swap", (i, n - 1 - i), None))

        return gates

    @staticmethod
    def iqft_structure(n: int) -> list[tuple[str, tuple[int, ...], float | None]]:
        """Generate inverse QFT gate sequence.

        Args:
            n: Number of qubits

        Returns:
            List of (gate_name, qubit_indices, angle) tuples
        """
        import math

        gates = []
        for i in range(n - 1, -1, -1):
            gates.append(("h", (i,), None))
            for j in range(i - 1, -1, -1):
                k = i - j
                angle = -math.pi / (2**k)
                gates.append(("cp", (j, i), angle))

        # Swaps for bit order reversal
        for i in range(n // 2):
            gates.append(("swap", (i, n - 1 - i), None))

        return gates
