"""Helpers for compile-time condition resolution and merge remapping."""

from __future__ import annotations

from typing import Any

from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.control_flow import HasNestedOps, IfOperation
from qamomile.circuit.ir.operation.gate import (
    MeasureOperation,
    MeasureVectorOperation,
    ProjectOperation,
)
from qamomile.circuit.ir.types.primitives import BitType
from qamomile.circuit.ir.value import (
    ArrayValue,
    Value,
    ValueBase,
    array_physical_region,
)

from .physical_index_map import (
    array_element_mapping,
    map_array_element_aliases,
)
from .qubit_address import ClbitMap, QubitAddress, QubitMap
from .value_resolver import ValueResolver, resolve_qubit_key


def _array_value_mapping(
    source: ArrayValue,
    index_map: QubitMap | ClbitMap,
) -> dict[int, int]:
    """Resolve an array value's local elements to physical indices.

    Direct aliases under ``source.uuid`` are authoritative. When a sliced
    view has not been registered under its own UUID, supplement them from its
    statically known root-space coverage. This is what lets a root register
    and ``root[:]`` participate in a runtime merge without looking divergent
    merely because only the root owns allocation keys.

    Args:
        source (ArrayValue): Root array or sliced view to resolve.
        index_map (QubitMap | ClbitMap): Physical resource map to read.

    Returns:
        dict[int, int]: Source-local element indices mapped to physical
            indices. Unresolved symbolic coverage contributes no fallback
            entries; any direct aliases remain present.
    """
    mapping = array_element_mapping(source.uuid, index_map)
    region = array_physical_region(source)
    if region is None:
        return mapping

    root = source
    while root.slice_of is not None:
        root = root.slice_of
    for local_index, root_index in enumerate(region[1]):
        if local_index in mapping:
            continue
        physical = index_map.get(QubitAddress(root.uuid, root_index))
        if physical is not None:
            mapping[local_index] = physical
    return mapping


def _value_reference_chain(value: ValueBase) -> list[str]:
    """Collect a value and its structural array ancestors in proximity order.

    Args:
        value (ValueBase): Scalar value, array, element, or sliced view.

    Returns:
        list[str]: UUIDs beginning with ``value`` and followed by its parent
            array / slice ancestors, without duplicates.
    """
    references: list[str] = []
    seen: set[str] = set()
    pending: list[ValueBase] = [value]
    while pending:
        current = pending.pop()
        if current.uuid in seen:
            continue
        seen.add(current.uuid)
        references.append(current.uuid)
        parent = getattr(current, "parent_array", None)
        if isinstance(parent, ValueBase):
            pending.append(parent)
        slice_of = getattr(current, "slice_of", None)
        if isinstance(slice_of, ValueBase):
            pending.append(slice_of)
    return references


def _reference_measurement_provenance(
    reference_uuid: str,
    operations: list[Operation],
    seen: set[str],
    root_operations: list[Operation] | None = None,
) -> bool | None:
    """Classify one reference's producer within an outer branch tree.

    A nested ``IfOperation`` result is fresh only when both of its runtime
    sources recursively originate at measurements in the same outer branch.
    Merely being defined by the nested if is insufficient: both sources may be
    pre-existing clbits from before the outer branch.

    Args:
        reference_uuid (str): UUID whose producer should be located.
        operations (list[Operation]): Current operation list being searched.
        seen (set[str]): Merge-result UUIDs already followed on this path.
        root_operations (list[Operation] | None): Complete outer branch tree
            used when recursively resolving a nested merge's sources. Defaults
            to ``operations``.

    Returns:
        bool | None: True for branch-local measurement provenance, False for a
            nested merge with any external source, or None when no relevant
            producer exists in this tree.
    """
    root = operations if root_operations is None else root_operations
    for operation in operations:
        if isinstance(operation, IfOperation):
            for merge in operation.iter_merges():
                if merge.result.uuid != reference_uuid:
                    continue
                if reference_uuid in seen:
                    return False
                next_seen = {*seen, reference_uuid}
                return _merge_source_is_branch_local_measurement(
                    merge.true_value,
                    root,
                    next_seen,
                ) and _merge_source_is_branch_local_measurement(
                    merge.false_value,
                    root,
                    next_seen,
                )

        if isinstance(operation, (MeasureOperation, MeasureVectorOperation)) and any(
            result.uuid == reference_uuid for result in operation.results
        ):
            return True

        if (
            isinstance(operation, ProjectOperation)
            and len(operation.results) > 1
            and operation.results[1].uuid == reference_uuid
        ):
            return True

        if isinstance(operation, HasNestedOps):
            for body in operation.nested_op_lists():
                provenance = _reference_measurement_provenance(
                    reference_uuid,
                    body,
                    seen,
                    root,
                )
                if provenance is not None:
                    return provenance
    return None


def _merge_source_is_branch_local_measurement(
    source: Value,
    operations: list[Operation],
    seen: set[str] | None = None,
) -> bool:
    """Return whether every source path starts at a branch-local measurement.

    Args:
        source (Value): Merge source selected by the branch.
        operations (list[Operation]): Complete operation tree of that branch.
        seen (set[str] | None): Nested merge results already followed while
            checking provenance. Defaults to None.

    Returns:
        bool: True only when the source, measured-vector parent, or nested
            merge recursively resolves exclusively to measurements executed
            inside this branch.
    """
    active = set() if seen is None else seen
    for reference_uuid in _value_reference_chain(source):
        provenance = _reference_measurement_provenance(
            reference_uuid,
            operations,
            active,
        )
        if provenance is not None:
            return provenance
    return False


def resolve_condition_address_detailed(
    condition: Value,
    bindings: dict[str, Any],
    resolver: ValueResolver | None = None,
) -> tuple[QubitAddress, bool]:
    """Resolve a condition / source ``Value`` to its ``clbit_map`` key.

    Single source of truth (shared by ``control_flow_emission.emit_if`` /
    ``emit_while``, the Qiskit / CUDA-Q backends, ``ResourceAllocator``'s
    loop-carried / merge aliasing, and the merge-output mapping helpers here)
    for turning a runtime control-flow condition — or a merge source Value —
    into the address its classical bit is registered under.

    Scalar measurement results register under ``QubitAddress(bit.uuid)``,
    while a ``Vector[Bit]`` element ``s[i]`` (``s = qmc.measure(register)``)
    registers under ``QubitAddress(root_array.uuid, root_index)``. The
    element index and every ``slice_start`` / ``slice_step`` along the
    parent's ``slice_of`` chain are resolved the same way — taken directly
    when constant, otherwise folded through ``bindings`` via ``resolver`` —
    and composed into the root index via ``root_index = start + step *
    view_local_index`` repeated along the chain.

    Args:
        condition (Value): The condition operand or merge source Value.
        bindings (dict[str, Any]): Active emit-time bindings used to fold
            symbolic indices / slice bounds. May be empty for callers that
            only have constant addresses to resolve (e.g. the allocator's
            pre-emit pass).
        resolver (ValueResolver | None): Resolver exposing
            ``resolve_int_value``; ``None`` restricts resolution to
            constants. Defaults to None.

    Returns:
        tuple[QubitAddress, bool]: ``(address, resolved_as_element)``.
            ``resolved_as_element`` is ``True`` only when ``condition`` is a
            ``Vector[Bit]`` element whose index (and any slice bounds)
            resolved to concrete ints, giving the root-space
            ``QubitAddress(root.uuid, root_index)``. It is ``False`` for a
            plain scalar, or when an element's index / slice bound could
            not be resolved (the scalar UUID fallback). Callers that mutate
            ``clbit_map`` should trust the address as a vector key only when
            this flag is ``True``, to avoid aliasing an unresolved element
            onto an unrelated scalar slot.
    """

    def _resolve_int(value: Value | None) -> int | None:
        """Resolve an index / slice-bound Value to a concrete int or None.

        Args:
            value (Value | None): The index or slice-bound Value, or None.

        Returns:
            int | None: The concrete integer when constant or resolvable
                through ``bindings`` via ``resolver``; ``None`` otherwise.
        """
        if value is None:
            return None
        if value.is_constant():
            return int(value.get_const())
        if resolver is not None:
            return resolver.resolve_int_value(value, bindings)
        return None

    if condition.parent_array is None or not condition.element_indices:
        return QubitAddress(condition.uuid), False
    idx = _resolve_int(condition.element_indices[0])
    if idx is None:
        return QubitAddress(condition.uuid), False
    parent = condition.parent_array
    while parent.slice_of is not None:
        start = _resolve_int(parent.slice_start)
        step = _resolve_int(parent.slice_step)
        if start is None or step is None:
            return QubitAddress(condition.uuid), False
        idx = start + step * idx
        parent = parent.slice_of
    return QubitAddress(parent.uuid, idx), True


def _is_unresolved_bit_element(value: Any, resolved_as_element: bool) -> bool:
    """Whether a merge source is a not-yet-resolvable ``Vector[Bit]`` element.

    A measured ``Vector[Bit]`` element (``bits[j]``) whose index is still
    symbolic — e.g. a loop variable that only gets a concrete value once
    the loop is unrolled at emit time — cannot be mapped to its root clbit
    during resource allocation (which sees the rolled loop with no index
    binding). ``resolve_condition_address_detailed`` reports this as
    ``resolved_as_element=False`` while ``parent_array`` / ``element_indices``
    are present. Merge-output registration must be deferred for such a
    source so the emit pass (with the loop variable bound) registers it
    against the correct per-iteration clbit instead of the element's own
    unregistered UUID.

    Args:
        value (Any): The merge source Value.
        resolved_as_element (bool): The ``resolved_as_element`` flag from
            :func:`resolve_condition_address_detailed`.

    Returns:
        bool: ``True`` when ``value`` is a ``Vector`` element whose index
            did not resolve to a concrete root address.
    """
    return (
        getattr(value, "parent_array", None) is not None
        and bool(getattr(value, "element_indices", None))
        and not resolved_as_element
    )


def _validate_runtime_bit_merge_partitions(
    if_op: IfOperation,
    clbit_map: ClbitMap,
    bindings: dict[str, Any],
    resolver: ValueResolver | None,
) -> None:
    """Validate that all runtime Bit merges preserve source alias partitions.

    The clbit implementation realizes a runtime merge by making corresponding
    sources on both branches share one physical clbit. This is sound only when
    equality between source slots is identical on both sides. For example,
    ``(t, t)`` cannot be merged with ``(f1, f2)``: aliasing both slots would
    collapse ``f1`` and ``f2``, so the later measurement would overwrite the
    earlier output.

    Args:
        if_op (IfOperation): Runtime if whose Bit merge slots are validated.
        clbit_map (ClbitMap): Current source-address-to-physical-clbit map.
        bindings (dict[str, Any]): Active bindings used to resolve symbolic
            measured-vector element indices.
        resolver (ValueResolver | None): Resolver for non-constant element
            indices, or None when only constant addresses are available.

    Raises:
        EmitError: If one branch reuses a physical source across merge slots
            whose sources are distinct on the opposite branch.
    """
    true_to_false: dict[int, int] = {}
    false_to_true: dict[int, int] = {}

    for merge in if_op.iter_merges():
        if not isinstance(merge.result.type, BitType):
            continue

        physical_pairs: list[tuple[int | None, int | None]] = []
        if isinstance(merge.result, ArrayValue):
            if not isinstance(merge.true_value, ArrayValue) or not isinstance(
                merge.false_value, ArrayValue
            ):
                continue
            true_mapping = _array_value_mapping(merge.true_value, clbit_map)
            false_mapping = _array_value_mapping(merge.false_value, clbit_map)
            for index in sorted(set(true_mapping) | set(false_mapping)):
                physical_pairs.append(
                    (true_mapping.get(index), false_mapping.get(index))
                )
        else:
            true_addr, true_resolved = resolve_condition_address_detailed(
                merge.true_value, bindings, resolver
            )
            false_addr, false_resolved = resolve_condition_address_detailed(
                merge.false_value, bindings, resolver
            )
            if _is_unresolved_bit_element(
                merge.true_value, true_resolved
            ) or _is_unresolved_bit_element(merge.false_value, false_resolved):
                continue
            physical_pairs.append((clbit_map.get(true_addr), clbit_map.get(false_addr)))

        for true_clbit, false_clbit in physical_pairs:
            if true_clbit is None or false_clbit is None:
                continue
            previous_false = true_to_false.setdefault(true_clbit, false_clbit)
            previous_true = false_to_true.setdefault(false_clbit, true_clbit)
            if previous_false != false_clbit or previous_true != true_clbit:
                from qamomile.circuit.transpiler.errors import EmitError

                raise EmitError(
                    "Runtime if-merge of measured Bits has an incompatible "
                    "clbit alias pattern across multiple outputs; one branch "
                    "reuses a source where the other branch has distinct "
                    "values. The clbit-aliasing model cannot preserve both "
                    "outputs. Use distinct measurements with the same alias "
                    "pattern in both branches, or make the condition "
                    "compile-time.",
                    operation="IfOperation",
                )


def _coerce_to_bool(value: Any) -> bool | None:
    """Coerce a Python scalar to bool; return None for non-scalar values.

    A backend-specific runtime expression (e.g. ``qiskit.circuit.classical.expr.Expr``)
    may be stored in ``bindings`` for the same UUID slot as a compile-time
    Python bool. This guard ensures we don't accidentally call ``bool()`` on
    such an object — that would either raise or return a misleading truthy
    value.

    Args:
        value: The value found in bindings (or the condition's constant).

    Returns:
        ``True`` / ``False`` for ``bool``/``int`` inputs, ``None`` otherwise.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    return None


def resolve_if_condition(
    condition: Any,
    bindings: dict[str, Any],
) -> bool | None:
    """Resolve an if-condition to a compile-time boolean."""
    if not hasattr(condition, "uuid"):
        return _coerce_to_bool(condition)
    if hasattr(condition, "is_constant") and condition.is_constant():
        return _coerce_to_bool(condition.get_const())
    if condition.uuid in bindings:
        return _coerce_to_bool(bindings[condition.uuid])
    # Resolve against ``bindings`` only via UUID (above) or the sanctioned
    # parameter-name provenance (below) — never the display ``Value.name``. A
    # bare-name lookup would mis-resolve an inlined callee-local condition that
    # happened to share a name with a caller binding key, silently pruning a
    # live branch.
    if hasattr(condition, "is_parameter") and condition.is_parameter():
        param_name = condition.parameter_name()
        if param_name and param_name in bindings:
            return _coerce_to_bool(bindings[param_name])
    return None


def remap_static_merge_outputs(
    if_op: IfOperation,
    condition_value: bool,
    qubit_map: QubitMap,
    clbit_map: ClbitMap,
    bindings: dict[str, Any] | None = None,
    resolver: ValueResolver | None = None,
) -> None:
    """Remap merge outputs for a compile-time constant ``IfOperation``.

    Registers each merged output under the physical resource of the
    branch selected by the resolved condition, so downstream operations
    referencing the merge output resolve to the surviving branch's qubit /
    clbit. Outputs already registered are left untouched.

    Args:
        if_op (IfOperation): The if-else whose condition resolved at
            compile time; merges are read through ``iter_merges``.
        condition_value (bool): The resolved condition, selecting which
            branch's source the outputs alias.
        qubit_map (QubitMap): Address-to-physical-qubit map, mutated in
            place.
        clbit_map (ClbitMap): Address-to-physical-clbit map, mutated in
            place.
        bindings (dict[str, Any] | None): Active bindings used to fold a
            merge source's symbolic ``Vector[Bit]`` element index / slice
            bounds (e.g. an unrolled loop variable). Defaults to None
            (empty), restricting element resolution to constant indices.
        resolver (ValueResolver | None): Resolver used with ``bindings``
            to fold non-constant element indices; ``None`` restricts to
            constants. Defaults to None.
    """
    for merge in if_op.iter_merges():
        output = merge.result
        selected_val = merge.select(condition_value)

        # Scalar Bit merges whose selected source is a loop-indexed
        # measured element are re-pointed per unrolled iteration inside
        # the branch below (same rationale as ``map_merge_outputs``: one
        # merge-output UUID is reused across iterations), so they are
        # exempt from the "already registered → skip" short-circuit.
        is_scalar_bit = isinstance(output.type, BitType) and not isinstance(
            output, ArrayValue
        )
        if not is_scalar_bit and (
            QubitAddress(output.uuid) in qubit_map
            or QubitAddress(output.uuid) in clbit_map
        ):
            continue

        if output.type.is_quantum():
            if isinstance(output, ArrayValue):
                if isinstance(selected_val, ArrayValue):
                    map_array_element_aliases(
                        output.uuid,
                        _array_value_mapping(selected_val, qubit_map),
                        qubit_map,
                    )
            else:
                key, _ = resolve_qubit_key(selected_val)
                scalar_addr = QubitAddress(selected_val.uuid)
                phys = qubit_map.get(scalar_addr)
                if phys is None and key is not None:
                    phys = qubit_map.get(key)
                if phys is not None:
                    qubit_map[QubitAddress(output.uuid)] = phys
        else:
            # Scalar Bit source: resolve vector-element sources (``s[i]``
            # from a measured ``Vector[Bit]``) to their root clbit key, not
            # the element's own UUID (which is not registered in clbit_map).
            # A loop-indexed element needs the current bindings / resolver
            # to fold its index; passing empty ``{}`` would leave it at the
            # unregistered element UUID and mis-alias the merge output.
            src_addr, src_resolved = resolve_condition_address_detailed(
                selected_val, bindings or {}, resolver
            )
            # Defer a still-symbolic loop-indexed element to emit time
            # (see the runtime counterpart in ``map_merge_outputs``).
            if _is_unresolved_bit_element(selected_val, src_resolved):
                continue
            # A resolved element source (``src_resolved``) is a
            # loop-indexed ``bits[j]`` folded to a concrete clbit for THIS
            # unrolled iteration, so its merge output must be re-pointed
            # every iteration; a plain-scalar source keeps
            # once-registration to avoid clobbering.
            if not src_resolved and QubitAddress(output.uuid) in clbit_map:
                continue
            if src_addr in clbit_map:
                clbit_map[QubitAddress(output.uuid)] = clbit_map[src_addr]


def map_merge_outputs(
    if_op: IfOperation,
    qubit_map: QubitMap,
    clbit_map: ClbitMap,
    resolve_scalar_qubit: Any = None,
    bindings: dict[str, Any] | None = None,
    resolver: ValueResolver | None = None,
    reject_runtime_bit_mux: bool = False,
    allowed_mixed_bit_outputs: frozenset[str] | set[str] | None = None,
) -> None:
    """Register merge output UUIDs to the same physical resources as their sources.

    Runtime-``if`` counterpart of :func:`remap_static_merge_outputs`: both
    branches may execute, so quantum merges are validated to sit on
    identical physical resources, and classical Bit merges consolidate
    both branches' clbits only when the complete source-alias partition is
    representable without overwriting an independently live value. Outputs
    already registered are left untouched.

    Args:
        if_op (IfOperation): The runtime if-else whose merged outputs need
            physical-resource registration; merges are read through
            ``iter_merges``.
        qubit_map (QubitMap): Address-to-physical-qubit map, mutated in
            place.
        clbit_map (ClbitMap): Address-to-physical-clbit map, mutated in
            place.
        resolve_scalar_qubit (Any): Optional callable
            ``(source, qubit_map) -> int | None`` used to resolve scalar
            qubit sources that are not directly registered (e.g. array
            elements). Defaults to None, restricting resolution to direct
            and root-array lookups.
        bindings (dict[str, Any] | None): Active bindings used to fold a
            classical ``Bit`` merge source's symbolic ``Vector[Bit]``
            element index / slice bounds (e.g. an unrolled loop variable).
            Defaults to None (empty), restricting element resolution to
            constant indices.
        resolver (ValueResolver | None): Resolver used with ``bindings``
            to fold non-constant element indices; ``None`` restricts to
            constants. Defaults to None.
        reject_runtime_bit_mux (bool): When True, raise ``EmitError`` for a
            runtime ``Bit`` merge that requires an at-runtime mux, mixes a
            pre-existing source with a branch-local measurement, or has
            incompatible source-sharing partitions across multiple outputs.
            Two all-branch-local source sets remain eligible only when their
            alias partitions match. Defaults to False.
        allowed_mixed_bit_outputs (frozenset[str] | set[str] | None): Merge
            result UUIDs whose pre-existing source is proven dead after the
            merge. This includes safe ordinary-if updates and phi-like loop
            condition remeasurements. Defaults to None.

    Raises:
        EmitError: If a quantum merge's branches resolve to different
            physical resources (or only one branch resolves), which a
            single-execution branch cannot realize; or, when
            ``reject_runtime_bit_mux`` is True, if a runtime ``Bit`` merge
            multiplexes two distinct pre-existing measured clbits/regions or
            has a non-identity branch source with no physical clbit
            (unrepresentable in the clbit-aliasing model), mixes external and
            branch-local provenance, or would collapse distinct merge slots.
    """
    active_bindings = bindings or {}
    mixed_output_allowlist = allowed_mixed_bit_outputs or frozenset()
    if reject_runtime_bit_mux:
        _validate_runtime_bit_merge_partitions(
            if_op,
            clbit_map,
            active_bindings,
            resolver,
        )

    for merge in if_op.iter_merges():
        output = merge.result
        true_val = merge.true_value
        false_val = merge.false_value

        # Scalar Bit merges whose source is a loop-indexed measured
        # element are re-pointed per unrolled iteration inside the branch
        # below (the loop body reuses one merge-output UUID across
        # iterations, so the clbit registration must be overwritten each
        # iteration), so they are exempt from the "already registered →
        # skip" short-circuit. Every other output keeps once-registration.
        is_scalar_bit = isinstance(output.type, BitType) and not isinstance(
            output, ArrayValue
        )
        if not is_scalar_bit and (
            QubitAddress(output.uuid) in qubit_map
            or QubitAddress(output.uuid) in clbit_map
        ):
            continue

        if output.type.is_quantum():
            if isinstance(output, ArrayValue):
                true_mapping = (
                    _array_value_mapping(true_val, qubit_map)
                    if isinstance(true_val, ArrayValue)
                    else {}
                )
                false_mapping = (
                    _array_value_mapping(false_val, qubit_map)
                    if isinstance(false_val, ArrayValue)
                    else {}
                )

                if true_mapping or false_mapping:
                    all_indices = set(true_mapping) | set(false_mapping)
                    output_mapping: dict[int, int] = {}
                    for idx in sorted(all_indices):
                        true_idx = true_mapping.get(idx)
                        false_idx = false_mapping.get(idx)
                        if (
                            true_idx is None
                            or false_idx is None
                            or true_idx != false_idx
                        ):
                            from qamomile.circuit.transpiler.errors import EmitError

                            raise EmitError(
                                "Quantum if-merge requires identical physical "
                                "resources across branches",
                                operation="IfOperation",
                            )
                        output_mapping[idx] = true_idx
                    map_array_element_aliases(output.uuid, output_mapping, qubit_map)
            else:

                def _resolve_one(source: Any) -> int | None:
                    src_addr = QubitAddress(source.uuid)
                    if src_addr in qubit_map:
                        return qubit_map[src_addr]
                    if resolve_scalar_qubit is not None:
                        return resolve_scalar_qubit(source, qubit_map)
                    key, _ = resolve_qubit_key(source)
                    if key is not None:
                        return qubit_map.get(key)
                    return None

                true_phys = _resolve_one(true_val)
                false_phys = _resolve_one(false_val)

                if true_phys is not None and false_phys is not None:
                    if true_phys != false_phys:
                        from qamomile.circuit.transpiler.errors import EmitError

                        raise EmitError(
                            "Quantum if-merge requires identical physical "
                            "resources across branches",
                            operation="IfOperation",
                        )
                    qubit_map[QubitAddress(output.uuid)] = true_phys
                elif true_phys is not None or false_phys is not None:
                    from qamomile.circuit.transpiler.errors import EmitError

                    raise EmitError(
                        "Quantum if-merge requires identical physical "
                        "resources across branches",
                        operation="IfOperation",
                    )

        elif isinstance(output.type, BitType):
            true_is_local_measurement = _merge_source_is_branch_local_measurement(
                true_val, if_op.true_operations
            )
            false_is_local_measurement = _merge_source_is_branch_local_measurement(
                false_val, if_op.false_operations
            )
            if (
                reject_runtime_bit_mux
                and true_val.uuid != false_val.uuid
                and true_is_local_measurement != false_is_local_measurement
                and output.uuid not in mixed_output_allowlist
            ):
                from qamomile.circuit.transpiler.errors import EmitError

                raise EmitError(
                    "Runtime if-merge of measured Bits mixes a pre-existing "
                    "clbit with a branch-local measurement; sharing the "
                    "pre-existing clbit would overwrite that independently "
                    "live value when the measuring branch executes. Measure "
                    "the merged bit inside both branches, or make the "
                    "condition compile-time.",
                    operation="IfOperation",
                )

            if isinstance(output, ArrayValue):
                true_src = true_val if isinstance(true_val, ArrayValue) else None
                false_src = false_val if isinstance(false_val, ArrayValue) else None
                primary = true_src or false_src
                secondary = false_src if true_src is not None else true_src

                if primary is not None:
                    primary_mapping = _array_value_mapping(primary, clbit_map)
                    secondary_mapping = (
                        _array_value_mapping(secondary, clbit_map)
                        if secondary is not None
                        else {}
                    )
                    if (
                        reject_runtime_bit_mux
                        and true_src is not None
                        and false_src is not None
                        and set(primary_mapping) != set(secondary_mapping)
                    ):
                        from qamomile.circuit.transpiler.errors import EmitError

                        raise EmitError(
                            "Runtime if-merge of measured Bit vectors requires "
                            "equal, fully resolved local clbit domains on both "
                            "branches; different vector lengths cannot share "
                            "one merge result.",
                            operation="IfOperation",
                        )
                    if (
                        reject_runtime_bit_mux
                        and true_src is not None
                        and false_src is not None
                        and not true_is_local_measurement
                        and not false_is_local_measurement
                        and primary_mapping
                        and secondary_mapping
                        and primary_mapping != secondary_mapping
                    ):
                        from qamomile.circuit.transpiler.errors import EmitError

                        raise EmitError(
                            "Runtime if-merge of measured Bits selects between "
                            "two distinct pre-existing clbit regions; the "
                            "clbit-aliasing model cannot multiplex two "
                            "already-measured values at runtime. Measure the "
                            "merged bits inside each branch, or make the "
                            "condition compile-time.",
                            operation="IfOperation",
                        )
                    map_array_element_aliases(
                        output.uuid,
                        primary_mapping,
                        clbit_map,
                        map_base_address=False,
                    )
                    if secondary is not None:
                        for element_index, phys_idx in primary_mapping.items():
                            sec_addr = QubitAddress(secondary.uuid, element_index)
                            if sec_addr in clbit_map:
                                clbit_map[sec_addr] = phys_idx
            else:
                # Scalar Bit merge: resolve vector-element sources to their
                # root clbit key so a measured ``Vector[Bit]`` element merged
                # through a merge (``if sel: bit = s[0] else: bit = t[0]``)
                # maps to the right clbit instead of the element's own UUID.
                # A loop-indexed element (``bits[j]``) needs the current
                # bindings / resolver to fold ``j``; passing empty ``{}``
                # leaves it at the unregistered element UUID and mis-aliases
                # the merge output to the wrong clbit.
                true_addr, true_resolved = resolve_condition_address_detailed(
                    true_val, active_bindings, resolver
                )
                false_addr, false_resolved = resolve_condition_address_detailed(
                    false_val, active_bindings, resolver
                )
                # If a source is still a symbolic ``Vector[Bit]`` element
                # (rolled loop at allocation time), its clbit is only known
                # once the loop is unrolled at emit; skip so emit registers
                # the correct per-iteration clbit rather than a stale
                # false-branch fallback.
                if _is_unresolved_bit_element(
                    true_val, true_resolved
                ) or _is_unresolved_bit_element(false_val, false_resolved):
                    continue

                # A resolved element source (``true_resolved`` /
                # ``false_resolved``) means a loop-indexed ``bits[j]`` that
                # folded to a concrete clbit for THIS unrolled iteration.
                # Such a merge output must be re-pointed every iteration.
                # A plain-scalar merge (both flags False) is stable, so it
                # keeps once-registration to avoid clobbering.
                per_iteration = true_resolved or false_resolved
                if not per_iteration and QubitAddress(output.uuid) in clbit_map:
                    continue
                true_clbit = clbit_map.get(true_addr)
                false_clbit = clbit_map.get(false_addr)

                if (
                    reject_runtime_bit_mux
                    and true_val.uuid != false_val.uuid
                    and (true_clbit is None or false_clbit is None)
                ):
                    from qamomile.circuit.transpiler.errors import EmitError

                    raise EmitError(
                        "Runtime if-merge of a measured Bit has a branch "
                        "source without a physical clbit; the clbit-aliasing "
                        "model cannot materialize a constant or symbolic Bit "
                        "assignment on only one runtime branch. Measure the "
                        "merged bit inside each branch, or make the condition "
                        "compile-time.",
                        operation="IfOperation",
                    )

                # A runtime scalar-Bit merge whose two sources resolve to
                # two distinct already-registered clbits is an at-runtime
                # multiplexing of two pre-existing measured bits. The
                # aliasing below binds the merged bit to the true-branch
                # clbit unconditionally, so it would silently return the
                # wrong value whenever the condition selects the false
                # branch. Branch-local fresh measurements are representable:
                # only one branch executes, so its write can target the shared
                # output clbit. They are detected structurally and exempted.
                # Constant-index and whole-vector pre-measured muxes are
                # rejected during allocation, before aliasing erases their
                # distinct source maps; this emit-time check covers sources
                # such as loop-indexed elements that only resolve now.
                if (
                    reject_runtime_bit_mux
                    and not true_is_local_measurement
                    and not false_is_local_measurement
                    and true_clbit is not None
                    and false_clbit is not None
                    and true_clbit != false_clbit
                ):
                    from qamomile.circuit.transpiler.errors import EmitError

                    raise EmitError(
                        "Runtime if-merge of a measured Bit selects between "
                        f"two distinct pre-existing clbits (true={true_clbit}, "
                        f"false={false_clbit}); the clbit-aliasing model "
                        "cannot multiplex two already-measured bits at "
                        "runtime. Measure the merged bit inside each branch, "
                        "or make the condition compile-time.",
                        operation="IfOperation",
                    )

                if true_clbit is not None:
                    clbit_map[QubitAddress(output.uuid)] = true_clbit
                    if false_clbit is not None and false_clbit != true_clbit:
                        clbit_map[false_addr] = true_clbit
                elif false_clbit is not None:
                    clbit_map[QubitAddress(output.uuid)] = false_clbit
