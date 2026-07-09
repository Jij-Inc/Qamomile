"""Helpers for compile-time condition resolution and merge remapping."""

from __future__ import annotations

from typing import Any

from qamomile.circuit.ir.operation.control_flow import IfOperation
from qamomile.circuit.ir.types.primitives import BitType
from qamomile.circuit.ir.value import ArrayValue, Value

from .physical_index_map import (
    array_element_mapping,
    array_element_mappings,
    copy_array_element_aliases,
    map_array_element_aliases,
)
from .qubit_address import ClbitMap, QubitAddress, QubitMap
from .value_resolver import ValueResolver, resolve_qubit_key


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
    if hasattr(condition, "name") and condition.name and condition.name in bindings:
        return _coerce_to_bool(bindings[condition.name])
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
                    copy_array_element_aliases(
                        selected_val.uuid, output.uuid, qubit_map
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
) -> None:
    """Register merge output UUIDs to the same physical resources as their sources.

    Runtime-``if`` counterpart of :func:`remap_static_merge_outputs`: both
    branches may execute, so quantum merges are validated to sit on
    identical physical resources, and classical Bit merges consolidate
    both branches' clbits onto one physical slot. Outputs already
    registered are left untouched.

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
            runtime scalar ``Bit`` merge whose two branch sources resolve to
            two distinct, already-registered clbits. That shape is an
            at-runtime multiplexing of two pre-existing measured bits, which
            the clbit-aliasing model cannot represent: it would silently
            bind the merged bit to the true-branch source regardless of the
            condition. Enable only at emit time (``register_merge_outputs``),
            where representable merges — branch-local fresh measurements and
            while-loop-carried conditions — have already been aliased onto a
            single shared clbit and therefore resolve to equal clbits. Leave
            False at resource-allocation time, where those representable
            merges still hold distinct clbits and would be wrongly rejected.
            Defaults to False.

    Raises:
        EmitError: If a quantum merge's branches resolve to different
            physical resources (or only one branch resolves), which a
            single-execution branch cannot realize; or, when
            ``reject_runtime_bit_mux`` is True, if a runtime scalar ``Bit``
            merge multiplexes two distinct pre-existing measured clbits
            (unrepresentable in the clbit-aliasing model).
    """
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
                array_sources = {
                    source.uuid
                    for source in (true_val, false_val)
                    if isinstance(source, ArrayValue)
                }
                mappings = array_element_mappings(array_sources, qubit_map)
                true_mapping = mappings.get(true_val.uuid, {})
                false_mapping = mappings.get(false_val.uuid, {})

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
            if isinstance(output, ArrayValue):
                true_src = true_val if isinstance(true_val, ArrayValue) else None
                false_src = false_val if isinstance(false_val, ArrayValue) else None
                primary = true_src or false_src
                secondary = false_src if true_src is not None else true_src

                if primary is not None:
                    primary_mapping = array_element_mapping(primary.uuid, clbit_map)
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
                    true_val, bindings or {}, resolver
                )
                false_addr, false_resolved = resolve_condition_address_detailed(
                    false_val, bindings or {}, resolver
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

                # A runtime scalar-Bit merge whose two sources resolve to
                # two distinct already-registered clbits is an at-runtime
                # multiplexing of two pre-existing measured bits. The
                # aliasing below binds the merged bit to the true-branch
                # clbit unconditionally, so it would silently return the
                # wrong value whenever the condition selects the false
                # branch. This fires precisely and NEVER on a representable
                # merge: branch-local fresh measurements and
                # while-loop-carried conditions have, by emit time, already
                # been aliased onto one shared clbit at allocation, so they
                # resolve to EQUAL clbits here. The still-distinct pair is
                # the loop-indexed pre-measured mux, whose sources were
                # symbolic at allocation and never got that aliasing — the
                # narrow shape that was silently miscompiling. This is a
                # precision guard, not a completeness one: a constant-index
                # or whole-vector pre-measured mux is aliased at allocation
                # and skips this check, but it does not reach a silent
                # result either — it is rejected loudly by segmentation or
                # the classical executor. Catching every unrepresentable
                # shape with one clear error here would need the
                # pre-aliasing source identities and is left as a follow-up
                # robustness nicety. The guard is enabled at emit time only;
                # at allocation the representable merges still hold distinct
                # clbits and must not be rejected.
                if (
                    reject_runtime_bit_mux
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
