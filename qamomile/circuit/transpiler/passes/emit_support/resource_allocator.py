"""Resource allocation helpers for emission."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.arithmetic_operations import PhiOp
from qamomile.circuit.ir.operation.cast import CastOperation
from qamomile.circuit.ir.operation.composite_gate import CompositeGateOperation
from qamomile.circuit.ir.operation.control_flow import (
    HasNestedOps,
    IfOperation,
    WhileOperation,
)
from qamomile.circuit.ir.operation.gate import (
    ConcreteControlledU,
    ControlledUOperation,
    GateOperation,
    IndexSpecControlledU,
    MeasureOperation,
    MeasureQFixedOperation,
    MeasureVectorOperation,
    SymbolicControlledU,
)
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp
from qamomile.circuit.ir.value import ArrayValue
from qamomile.circuit.transpiler.passes.emit_support.condition_resolution import (
    map_phi_outputs,
    remap_static_phi_outputs,
    resolve_if_condition,
)
from qamomile.circuit.transpiler.passes.emit_support.qubit_address import (
    ClbitMap,
    QubitAddress,
    QubitMap,
)
from qamomile.circuit.transpiler.passes.emit_support.value_resolver import (
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

    def __init__(self) -> None:
        self._next_qubit_index: int = 0
        self._next_clbit_index: int = 0

    def allocate(
        self,
        operations: list[Operation],
        bindings: dict[str, Any] | None = None,
    ) -> tuple[QubitMap, ClbitMap]:
        """Allocate qubit and clbit indices for all operations.

        Args:
            operations: List of operations to allocate resources for
            bindings: Optional variable bindings for resolving dynamic sizes

        Returns:
            Tuple of (qubit_map, clbit_map) where each maps
            QubitAddress to physical index
        """
        qubit_map: QubitMap = {}
        clbit_map: ClbitMap = {}
        self._next_qubit_index = 0
        self._next_clbit_index = 0
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
                    init_uuid = (
                        init_val.uuid if hasattr(init_val, "uuid") else str(init_val)
                    )
                    init_addr = QubitAddress(init_uuid)

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
                    carried_uuid = (
                        carried_val.uuid
                        if hasattr(carried_val, "uuid")
                        else str(carried_val)
                    )
                    carried_addr = QubitAddress(carried_uuid)
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

            elif isinstance(op, CompositeGateOperation):
                self._allocate_composite(op, qubit_map)

            elif isinstance(op, ControlledUOperation):
                self._allocate_controlled_u(op, qubit_map)

            elif isinstance(op, CastOperation):
                self._allocate_cast(op, qubit_map)

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
                true_addr = QubitAddress(true_val.uuid)
                false_addr = QubitAddress(false_val.uuid)
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
        """Resolve a size Value to a concrete integer."""
        import re

        if size_val.is_constant():
            return int(size_val.get_const())

        # Value with parent_array (e.g., hi.shape[0])
        if size_val.parent_array is not None:
            array_name = size_val.parent_array.name
            if array_name in bindings:
                array_data = bindings[array_name]
                if hasattr(array_data, "__len__"):
                    return len(array_data)

        # Check by name, then uuid in bindings
        if size_val.name and size_val.name in bindings:
            bound = bindings[size_val.name]
            if isinstance(bound, (int, float)):
                return int(bound)
            if hasattr(bound, "__len__"):
                return len(bound)

        if size_val.uuid in bindings:
            bound = bindings[size_val.uuid]
            if isinstance(bound, (int, float)):
                return int(bound)

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
        # the physical qubit already allocated under the QubitAddress(parent_uuid, idx) key.
        for operand in qubit_ops:
            operand_addr = QubitAddress(operand.uuid)
            if operand_addr not in qubit_map:
                if operand.parent_array is not None and operand.element_indices:
                    # Array element: resolve via parent_array key
                    parent_uuid = operand.parent_array.uuid
                    idx_value = operand.element_indices[0]
                    if idx_value.is_constant():
                        idx = int(idx_value.get_const())
                        key = QubitAddress(parent_uuid, idx)
                        assert key in qubit_map, (
                            f"Array element key {str(key)!r} not found in qubit_map. "
                            f"This indicates a bug in the transpiler pipeline: "
                            f"QInitOperation for the parent array was not processed "
                            f"before this GateOperation."
                        )
                        qubit_map[operand_addr] = qubit_map[key]
                    # Non-constant indices (symbolic loop vars) are resolved
                    # at emit time via ValueResolver.resolve_qubit_index_detailed.
                else:
                    # Scalar qubit: allocate new index.
                    # This path is used for @qkernel input parameters created with
                    # emit_init=False (func_to_block.py), which have no QInitOperation.
                    # ResourceAllocator.allocate() receives only operations, not the
                    # block's input_values, so these qubits are first registered here.
                    qubit_map[operand_addr] = self._next_qubit_index
                    self._next_qubit_index += 1

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
        """Allocate qubits for a list of qubit Values and their results."""
        for qubit in all_qubits:
            qubit_addr, is_array = resolve_qubit_key(qubit)
            if qubit_addr is not None:
                if qubit_addr not in qubit_map:
                    # New allocation for qubits not yet registered.
                    # For scalar qubits, this handles @qkernel input parameters
                    # (emit_init=False) that have no preceding QInitOperation.
                    # For array elements, this handles first-seen element keys.
                    qubit_map[qubit_addr] = self._next_qubit_index
                    self._next_qubit_index += 1
                scalar_addr = QubitAddress(qubit.uuid)
                if scalar_addr not in qubit_map:
                    qubit_map[scalar_addr] = qubit_map[qubit_addr]

        for i, result in enumerate(results):
            result_addr = QubitAddress(result.uuid)
            if result_addr not in qubit_map and i < len(all_qubits):
                qubit_addr, is_array = resolve_qubit_key(all_qubits[i])
                if qubit_addr is not None:
                    # Alias: result must map to the same physical index as its
                    # corresponding operand.  If qubit_addr is somehow missing
                    # (should not happen after the operand loop above), this is
                    # a bug -- raise explicitly rather than silently allocating.
                    if qubit_addr in qubit_map:
                        qubit_map[result_addr] = qubit_map[qubit_addr]
                    else:
                        raise AssertionError(
                            f"Missing qubit address '{str(qubit_addr)}' in qubit_map when "
                            f"allocating result '{result.uuid}'. "
                            "This indicates a bug in operand allocation."
                        )

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
        op: CompositeGateOperation,
        qubit_map: QubitMap,
    ) -> None:
        """Allocate resources for a CompositeGateOperation."""
        all_qubits = op.control_qubits + op.target_qubits
        self._allocate_qubit_list(all_qubits, list(op.results), qubit_map)

    def _allocate_controlled_u(
        self,
        op: ControlledUOperation,
        qubit_map: QubitMap,
    ) -> None:
        """Allocate resources for a ControlledUOperation."""
        if isinstance(op, IndexSpecControlledU):
            # Vector already allocated by QInitOperation.
            # Map result ArrayValue to same physical qubits.
            vector_operand = op.operands[0]
            vector_result = op.results[0]
            for addr, idx in list(qubit_map.items()):
                if addr.matches_array(vector_operand.uuid):
                    result_addr = QubitAddress(vector_result.uuid, addr.element_index)
                    if result_addr not in qubit_map:
                        qubit_map[result_addr] = idx
            return

        if isinstance(op, SymbolicControlledU):
            from qamomile.circuit.transpiler.errors import EmitError

            raise EmitError(
                "Cannot transpile ControlledUOperation with symbolic num_controls. "
                "Bind parameters to concrete values before transpilation.",
                operation="ControlledUOperation",
            )
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
