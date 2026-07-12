"""Value resolution helpers for emission."""

from __future__ import annotations

import numbers
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from qamomile.circuit.ir.value import Value, resolve_root_qubit_address
from qamomile.circuit.transpiler.errors import EmitError, ResolutionFailureReason
from qamomile.circuit.transpiler.passes.emit_support.qubit_address import (
    QubitAddress,
    QubitMap,
)

if TYPE_CHECKING:
    from qamomile.circuit.ir.value import ArrayValue


@dataclass
class QubitResolutionResult:
    """Result of attempting to resolve a qubit index."""

    success: bool
    index: int | None = None
    failure_reason: ResolutionFailureReason | None = None
    failure_details: str = ""


def resolve_qubit_key(qubit: "Value") -> tuple[QubitAddress | None, bool]:
    """Resolve a qubit Value to its allocation key.

    Args:
        qubit (Value): Qubit value to resolve. Array elements with a
            constant index are resolved through their full ``slice_of``
            chain to the root array address.

    Returns:
        tuple[QubitAddress | None, bool]: ``(address, is_array_element)``.
            ``address`` is the root-space address for a resolvable array
            element, ``None`` for an array element whose index or slice
            bounds are symbolic at this binding-free stage, and a scalar
            UUID address for non-array qubits. A negative constant index
            (or an out-of-contract slice bound) also yields ``None``:
            ``resolve_root_qubit_address`` refuses it rather than letting it
            silently address a wrong root slot.
    """
    if qubit.parent_array is not None and qubit.element_indices:
        idx_value = qubit.element_indices[0]
        if idx_value.is_constant():
            resolved = resolve_root_qubit_address(qubit)
            if resolved is not None:
                root_uuid, idx = resolved
                return QubitAddress(root_uuid, idx), True
        return None, True
    return QubitAddress(qubit.uuid), False


class ValueResolver:
    """Resolves Value objects to concrete indices or values."""

    def __init__(self, parameters: set[str] | None = None):
        """Create an emit-time resolver.

        Args:
            parameters (set[str] | None): Names preserved as backend runtime
                parameters. Defaults to None.
        """
        self.parameters = parameters or set()

    def resolve_qubit_index(
        self,
        v: "Value",
        qubit_map: QubitMap,
        bindings: dict[str, Any],
    ) -> int | None:
        """Resolve a qubit value to its physical index.

        Args:
            v (Value): Scalar or indexed qubit value.
            qubit_map (QubitMap): Physical resource mapping.
            bindings (dict[str, Any]): Active UUID/parameter bindings.

        Returns:
            int | None: Physical index, or ``None`` when unresolved.
        """
        result = self.resolve_qubit_index_detailed(v, qubit_map, bindings)
        return result.index if result.success else None

    def resolve_qubit_index_detailed(
        self,
        v: "Value",
        qubit_map: QubitMap,
        bindings: dict[str, Any],
    ) -> QubitResolutionResult:
        """Resolve a Value to a physical qubit index with detailed failure info.

        Args:
            v (Value): The qubit Value to resolve. May be a scalar qubit or
                an array element (possibly through a sliced view chain).
            qubit_map (QubitMap): Mapping from ``QubitAddress`` to physical
                qubit indices built by the resource allocator.
            bindings (dict[str, Any]): Active emit-time bindings used to
                resolve symbolic element indices and slice bounds.

        Returns:
            QubitResolutionResult: Success with the physical index, or a
                failure carrying a ``ResolutionFailureReason`` — including
                ``NEGATIVE_INDEX`` when a resolved element index is negative
                or a resolved slice bound violates the frontend contract
                (non-negative start, positive step); composing those through
                the affine map would silently address a wrong root slot.
        """
        if v.parent_array is not None and v.element_indices:
            parent_uuid = v.parent_array.uuid
            idx_value = v.element_indices[0]

            idx = None
            if idx_value.is_constant():
                idx = int(idx_value.get_const())
            elif idx_value.uuid in bindings:
                # UUID is unique; prefer it so BinOp results (which all
                # share the generic name "uint_tmp" and therefore collide
                # in bindings-by-name) resolve to the correct iteration
                # value.  Falling through to name-lookup here produced
                # "duplicate qubit" errors for patterns like
                # ``q[2*i], q[2*i+1] = qmc.cx(...)`` inside a for loop.
                idx = self._resolve_numeric_index(bindings[idx_value.uuid])
                if idx is None:
                    bound_val = bindings[idx_value.uuid]
                    return QubitResolutionResult(
                        success=False,
                        failure_reason=ResolutionFailureReason.INDEX_NOT_NUMERIC,
                        failure_details=(
                            f"Index (uuid: {idx_value.uuid[:8]}...) resolved to "
                            f"non-numeric type: {type(bound_val).__name__}"
                        ),
                    )
            elif (
                idx_value.is_parameter()
                and (parameter_name := idx_value.parameter_name()) in bindings
            ):
                idx = self._resolve_numeric_index(bindings[parameter_name])
                if idx is None:
                    bound_val = bindings[parameter_name]
                    return QubitResolutionResult(
                        success=False,
                        failure_reason=ResolutionFailureReason.INDEX_NOT_NUMERIC,
                        failure_details=(
                            f"Index parameter '{parameter_name}' resolved to "
                            f"non-numeric type: "
                            f"{type(bound_val).__name__}"
                        ),
                    )
            elif idx_value.parent_array is not None:
                nested_result = self.resolve_classical_value(idx_value, bindings)
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
                raw = self.lookup_in_bindings(idx_value, bindings)
                if raw is not None:
                    idx = self._resolve_numeric_index(raw)
                    if idx is None:
                        return QubitResolutionResult(
                            success=False,
                            failure_reason=ResolutionFailureReason.INDEX_NOT_NUMERIC,
                            failure_details=(
                                f"Index '{idx_value.name}' (uuid: "
                                f"{idx_value.uuid[:8]}...) resolved to "
                                f"non-numeric type: {type(raw).__name__}"
                            ),
                        )
            if idx is None:
                if idx_value.parent_array is not None:
                    nested_result = self.resolve_classical_value(idx_value, bindings)
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
                            f"Index variable '{idx_value.name}' (uuid: "
                            f"{idx_value.uuid[:8]}...) is not bound."
                        ),
                    )

            if idx is not None:
                # A negative local index must fail before the slice-chain
                # walk: the affine composition ``start + step * idx`` can
                # turn it into a valid-but-wrong non-negative root index
                # (e.g. ``view[-1]`` for ``view = q[1:3]`` would silently
                # address ``q[0]`` instead of ``q[2]``).
                if idx < 0:
                    return QubitResolutionResult(
                        success=False,
                        failure_reason=ResolutionFailureReason.NEGATIVE_INDEX,
                        failure_details=(
                            f"Index {idx} for array '{v.parent_array.name}' "
                            f"is negative; Python-style negative indexing "
                            f"is not supported."
                        ),
                    )
                # Walk any slice_of chain attached to the parent so
                # sliced views resolve to their root parent's physical
                # qubit.  For ordinary (non-sliced) arrays the while
                # loop's condition is immediately false, so the path
                # remains a direct qubit_map lookup with the same cost
                # as before.
                parent = v.parent_array
                while (
                    parent.slice_of is not None
                    and parent.slice_start is not None
                    and parent.slice_step is not None
                ):
                    start_val = self.resolve_classical_value(
                        parent.slice_start, bindings
                    )
                    step_val = self.resolve_classical_value(parent.slice_step, bindings)
                    if start_val is None or step_val is None:
                        return QubitResolutionResult(
                            success=False,
                            failure_reason=ResolutionFailureReason.SYMBOLIC_INDEX_NOT_BOUND,
                            failure_details=(
                                f"Slice bounds for view over '{parent.slice_of.name}' "
                                f"could not be resolved; start={parent.slice_start.name}, "
                                f"step={parent.slice_step.name}."
                            ),
                        )
                    # Symbolic slice bounds resolved from bindings must
                    # satisfy the same contract the frontend enforces for
                    # constant bounds: non-negative start, positive step.
                    if int(start_val) < 0 or int(step_val) <= 0:
                        return QubitResolutionResult(
                            success=False,
                            failure_reason=ResolutionFailureReason.NEGATIVE_INDEX,
                            failure_details=(
                                f"Slice bounds for view over "
                                f"'{parent.slice_of.name}' resolved to "
                                f"start={int(start_val)}, step={int(step_val)}; "
                                f"start must be non-negative and step positive."
                            ),
                        )
                    idx = int(start_val) + int(step_val) * idx
                    parent = parent.slice_of
                parent_uuid = parent.uuid

                array_qubit_addr = QubitAddress(parent_uuid, idx)
                if array_qubit_addr in qubit_map:
                    return QubitResolutionResult(
                        success=True, index=qubit_map[array_qubit_addr]
                    )
                return QubitResolutionResult(
                    success=False,
                    failure_reason=ResolutionFailureReason.ARRAY_ELEMENT_NOT_IN_QUBIT_MAP,
                    failure_details=(
                        f"Computed qubit ID '{array_qubit_addr}' not found in qubit_map. "
                        f"Index {idx} may be out of bounds for array '{v.parent_array.name}'."
                    ),
                )

        scalar_addr = QubitAddress(v.uuid)
        if scalar_addr in qubit_map:
            return QubitResolutionResult(success=True, index=qubit_map[scalar_addr])

        return QubitResolutionResult(
            success=False,
            failure_reason=ResolutionFailureReason.DIRECT_UUID_NOT_FOUND,
            failure_details=(
                f"Value uuid '{v.uuid[:8]}...' not found in qubit_map "
                f"and is not an array element."
            ),
        )

    def resolve_slice_chain(
        self,
        av: "ArrayValue",
        bindings: dict[str, Any],
        *,
        operation: str = "emit",
    ) -> tuple["ArrayValue", int, int]:
        """Walk an ArrayValue's slice_of chain and return root + affine map.

        Composes the nested affine maps so that a view-local index ``i``
        corresponds to the root-space index ``start + step * i``. For a
        root ArrayValue (no ``slice_of`` chain), returns ``(av, 0, 1)``
        so callers can always apply the same formula uniformly.

        Args:
            av: The possibly-sliced ArrayValue whose root and affine
                mapping should be resolved.
            bindings: Compile-time parameter bindings; required when
                any ``slice_start`` / ``slice_step`` in the chain is
                symbolic.
            operation: Human-readable name of the enclosing emit
                operation. Used only in the ``EmitError`` message when
                a symbolic slice bound cannot be resolved. Defaults
                to ``"emit"``.

        Returns:
            Tuple ``(root_array, start, step)`` where ``root_array`` is
            the underlying non-sliced ArrayValue, and ``start``/``step``
            are Python ``int`` values satisfying
            ``view[i] == root_array[start + step * i]``.

        Raises:
            EmitError: If any ``slice_start`` or ``slice_step`` in the
                chain resolves to a non-numeric or unbound value, or to
                bounds violating the frontend contract (negative start or
                non-positive step).

        Example:
            >>> # For ``view = q[1::2]`` where ``q`` has 4 qubits:
            >>> resolver.resolve_slice_chain(view_av, bindings={})
            (q_av, 1, 2)
        """
        start = 0
        step = 1
        cur = av
        while (
            cur.slice_of is not None
            and cur.slice_start is not None
            and cur.slice_step is not None
        ):
            sub_start = self.resolve_classical_value(cur.slice_start, bindings)
            sub_step = self.resolve_classical_value(cur.slice_step, bindings)
            if not isinstance(sub_start, (int, float)) or not isinstance(
                sub_step, (int, float)
            ):
                raise EmitError(
                    f"Cannot resolve slice bounds for view of "
                    f"'{cur.slice_of.name}': start={cur.slice_start}, "
                    f"step={cur.slice_step}. Slice views require concrete "
                    f"start/step at emit time.",
                    operation=operation,
                )
            # Symbolic slice bounds resolved from bindings must satisfy
            # the same contract the frontend enforces for constant bounds
            # (non-negative start, positive step); composing a negative
            # start or non-positive step would silently remap view
            # elements onto wrong root slots.
            if int(sub_start) < 0 or int(sub_step) <= 0:
                raise EmitError(
                    f"Invalid slice bounds for view of "
                    f"'{cur.slice_of.name}': start={int(sub_start)}, "
                    f"step={int(sub_step)}. Slice start must be "
                    f"non-negative and step positive.",
                    operation=operation,
                )
            # Compose: if current (start, step) maps view-local i to
            # cur-local k = start + step * i, and cur-local k maps to
            # parent k' = sub_start + sub_step * k, the new map is
            # parent k' = (sub_start + sub_step * start) + (sub_step * step) * i.
            start = int(sub_start) + int(sub_step) * start
            step = int(sub_step) * step
            cur = cur.slice_of
        return cur, start, step

    # ------------------------------------------------------------------
    # Unified bindings lookup
    # ------------------------------------------------------------------

    def lookup_in_bindings(
        self,
        value: "Value",
        bindings: dict[str, Any],
        *,
        index_array: bool = False,
    ) -> Any:
        """Canonical resolution chain for a Value against ``bindings``.

        All other resolver methods (``resolve_bound_value``,
        ``resolve_classical_value``, ``resolve_int_value``,
        ``resolve_operand_for_binding``) wrap this single chain. Centralizing
        precedence here prevents the historical drift where one resolver
        checked ``is_parameter`` before UUID and another checked it after,
        which manifested as obscure binding failures when name-keyed writes
        were dropped from the emit pass.

        Resolution order (each step returns immediately on a hit):

        1. ``value`` is already a concrete Python scalar (no ``uuid``).
        2. ``value.uuid`` is in ``bindings`` — return that. This is where
           emit-time-computed intermediates (``evaluate_binop`` /
           ``evaluate_classical_predicate`` results) and merge aliases live.
        3. A compile-time array or dict payload stored in typed metadata.
        4. ``value.is_constant()`` — return ``value.get_const()``.
        5. ``value.is_parameter()`` and its parameter name is in
           ``bindings`` — return that.
        6. (When ``index_array=True``) ``value`` is an array element with
           a resolvable parent in ``bindings`` — index into it.
        7. A non-empty display name is present in ``bindings``. This is a
           compatibility bridge for programmatically constructed Values and
           older transforms that do not carry parameter metadata; UUID and
           explicit parameter provenance always take precedence.

        Compiler-created temporaries are anonymous and compiler-internal
        writers use UUID keys, so the compatibility bridge cannot merge two
        same-named temporaries. Kernel parameters should normally resolve via
        explicit parameter metadata, while loop variables and intermediates
        remain UUID-keyed.

        Args:
            value (Value | Any): The IR Value (or already-concrete Python
                scalar) to
                resolve.
            bindings (dict[str, Any]): The active bindings dict.
            index_array (bool): When True, also resolve array-element accesses
                via ``parent_array`` indexing. Off by default because not
                all callers want to index into bound containers.

        Returns:
            Any: The resolved Python value, or ``None`` if no step matched.
        """
        # 1. Already concrete (Python scalar that was passed through).
        if not hasattr(value, "uuid"):
            return value
        # 2. UUID-keyed entries (intermediates, loop variables, runtime
        #    expressions, merge aliases, and nested-block formal aliases).
        if value.uuid in bindings:
            return bindings[value.uuid]
        # 3. Compile-time-bound containers carry their payload in typed
        # metadata. Check these before ``is_constant``: an empty DictValue is
        # vacuously constant but has no scalar ``get_const()`` payload.
        # Resolve the whole container here as well as individual elements so
        # nested/inverse block parameter binding never has to recover it from
        # the display-only ArrayValue name.
        get_const_array = getattr(value, "get_const_array", None)
        if callable(get_const_array):
            const_array = get_const_array()
            if const_array is not None:
                return const_array
        metadata = getattr(value, "metadata", None)
        if getattr(metadata, "dict_runtime", None) is not None:
            get_bound_data = getattr(value, "get_bound_data", None)
            if callable(get_bound_data):
                return get_bound_data()
        # 4. IR scalar constant.
        if isinstance(value, Value) and value.is_constant():
            return value.get_const()
        # 5. Kernel parameter — keyed by parameter name (the only legitimate
        #    name-keyed path; users supply parameters by name at the public API
        #    boundary).
        if hasattr(value, "is_parameter") and value.is_parameter():
            param_name = value.parameter_name()
            if param_name and param_name in bindings:
                return bindings[param_name]
        # 6. Array-element access via parent_array (opt-in).
        if (
            index_array
            and getattr(value, "parent_array", None) is not None
            and getattr(value, "element_indices", None)
        ):
            resolved = self._index_into_array(value, bindings)
            if resolved is not None:
                return resolved
        # 7. Compatibility bridge for older/programmatic IR. Keep this last:
        # a public binding named like an internal constant/container must not
        # shadow typed metadata or structural element resolution.
        name = getattr(value, "name", None)
        if name and name in bindings:
            return bindings[name]
        return None

    def resolve_operand_for_binding(
        self,
        operand: "Value",
        bindings: dict[str, Any],
    ) -> Any:
        """Resolve an operand to a concrete value for block parameter binding.

        Used when calling a sub-block (e.g. a controlled-U body): each
        param operand at the call site must resolve to a value to seed the
        callee's parameter bindings.

        Args:
            operand (Value): Actual call-site operand.
            bindings (dict[str, Any]): Bindings visible at the call site.

        Returns:
            Any: Resolved Python/backend value, or ``None`` when symbolic.
        """
        return self.lookup_in_bindings(operand, bindings, index_array=True)

    def bind_block_params(
        self,
        block_value: Any,
        param_operands: list["Value"],
        bindings: dict[str, Any],
    ) -> dict[str, Any]:
        """Create UUID-safe local bindings for a nested block's parameters.

        Args:
            block_value (Any): Nested block whose formal inputs are bound.
            param_operands (list[Value]): Actual classical operands in formal
                input order.
            bindings (dict[str, Any]): Bindings visible at the call site.

        Returns:
            dict[str, Any]: A copy of ``bindings`` extended with each resolved
                formal's UUID and, for declared parameters, semantic parameter
                name.
        """
        local_bindings = bindings.copy()
        if not hasattr(block_value, "input_values"):
            return local_bindings
        param_inputs = [
            iv
            for iv in block_value.input_values
            if hasattr(iv, "type") and (iv.type.is_classical() or iv.type.is_object())
        ]
        for i, operand in enumerate(param_operands):
            if i >= len(param_inputs):
                break
            resolved = self.resolve_operand_for_binding(operand, bindings)
            if resolved is not None:
                formal = param_inputs[i]
                local_bindings[formal.uuid] = resolved
                if hasattr(formal, "is_parameter") and formal.is_parameter():
                    parameter_name = formal.parameter_name()
                    if parameter_name:
                        local_bindings[parameter_name] = resolved
        return local_bindings

    def resolve_bound_value(
        self,
        value: "Value",
        bindings: dict[str, Any],
    ) -> Any:
        """Resolve a Value to its raw bound Python object.

        Wraps :meth:`lookup_in_bindings` with ``index_array=True`` so that
        ``arr[i]`` accesses against a bound container resolve to the
        element. Does **not** coerce the result — callers that need a
        numeric scalar should go through :meth:`resolve_classical_value`.

        Args:
            value (Value): Value to resolve.
            bindings (dict[str, Any]): Active UUID/parameter bindings.

        Returns:
            Any: Raw bound object, or ``None`` when unresolved.
        """
        return self.lookup_in_bindings(value, bindings, index_array=True)

    def resolve_classical_value(
        self,
        value: "Value",
        bindings: dict[str, Any],
    ) -> Any:
        """Resolve a classical Value to a concrete Python value.

        Numeric bindings are normalized to native Python scalars
        regardless of whether they come from a direct binding or from
        array-element indexing, so downstream ``isinstance(x, (int,
        float))`` checks are stable when callers bind ``np.pi/4`` or
        the like. ``bool`` is preserved (not coerced to ``int``).
        Non-numeric values (Hamiltonians, strings, dict values, …)
        pass through unchanged.

        Args:
            value (Value): Classical value to resolve.
            bindings (dict[str, Any]): Active UUID/parameter bindings.

        Returns:
            Any: Resolved native scalar/object, or ``None`` when unresolved.
        """
        raw = self.resolve_bound_value(value, bindings)
        if raw is None or isinstance(raw, bool):
            return raw
        coerced = self._resolve_numeric_value(raw)
        return coerced if coerced is not None else raw

    def _index_into_array(
        self,
        v: "Value",
        bindings: dict[str, Any],
    ) -> Any | None:
        """Index into a bound array container at the operand's element indices.

        Refuses to index when the parent array's declared parameter name is
        in ``self.parameters``. That short-circuit is the same invariant
        ``fold_classical_op(... EMIT_RESPECT_PARAMS)`` enforces at the
        op level: a runtime parameter array's "value" is symbolic, and
        any concrete data the user supplied alongside is a placeholder
        — silently indexing into the placeholder is exactly what
        produced the silent miscompilation in Issue #354 B-series. This
        guard is defense-in-depth: even if a future caller forgets the
        op-level guard, array indexing for runtime parameters returns
        ``None`` here.

        Args:
            v (Value): The element Value to resolve. Must carry a
                ``parent_array``; scalar Values resolve to ``None``.
            bindings (dict[str, Any]): The active emit-time bindings.

        Returns:
            Any | None: The resolved element, or ``None`` when the access
                is genuinely symbolic at emit time (unresolved indices or
                slice bounds, a runtime parameter array, or a container
                that is absent from compile-time data).

        Raises:
            EmitError: If every index resolved to a concrete int and the
                root array's container holds compile-time data, but the
                access fails against that data — either an out-of-range
                index (``IndexError``; e.g. a loop-unrolled ``theta[i]``
                with ``i >= len(theta)``) or a shape mismatch (``KeyError``
                /``TypeError``; the bound value is not an int-indexable
                array of the expected shape). Falling back to ``None`` in
                these cases would let the element reach symbolic-parameter
                creation, where every failing access silently shared one
                phantom runtime parameter (or a silent ``0.0``).
        """
        parent = v.parent_array
        # Only element Values carry a parent array; scalar Values cannot be
        # indexed through this helper.
        if parent is None:
            return None
        # Resolve a possibly sliced view element to the root array and
        # concrete indices; symbolic indices or slice bounds intentionally
        # fall back to unresolved.
        location = self._resolve_array_element_location(
            parent, v.element_indices, bindings
        )
        if location is None:
            return None
        root_parent, indices = location
        # Runtime parameter arrays are symbolic even if placeholder data is
        # present in bindings; never index them at emit time. Prefer typed
        # parameter metadata, with a non-empty-name fallback for legacy or
        # hand-built ArrayValues whose runtime-parameter contract is supplied
        # only through ``ValueResolver(parameters=...)``.
        parameter_name = root_parent.parameter_name()
        if (parameter_name is not None and parameter_name in self.parameters) or (
            parameter_name is None
            and bool(root_parent.name)
            and root_parent.name in self.parameters
        ):
            return None
        # Prefer const_array metadata, then explicit bindings for the root
        # container. A missing container means the element stays unresolved.
        container = self._resolve_array_container(root_parent, bindings)
        if container is None:
            return None

        # Descend through nested containers one index at a time; the final
        # value of ``container`` is the resolved element, not the parent array.
        for i in indices:
            try:
                container = container[i]
            except IndexError as exc:
                # Every guard above already passed: the indices are
                # concrete ints, the array is not a runtime parameter,
                # and the container holds known compile-time data. The
                # access is therefore genuinely out of range, not
                # symbolic — fail fast instead of falling through to
                # phantom-parameter creation.
                try:
                    length = len(container)
                except TypeError:
                    # Some objects define ``__len__`` but still raise on
                    # ``len()`` (e.g. 0-d NumPy arrays). Fall back to an
                    # unqualified message rather than masking the real
                    # out-of-range error with a length-formatting failure.
                    length = None
                length_note = f" of length {length}" if length is not None else ""
                raise EmitError(
                    f"Index {i} is out of range for compile-time bound "
                    f"array '{root_parent.name}'{length_note}. Bind data "
                    f"covering every index reached at emit time (e.g. "
                    f"every unrolled loop iteration), or declare "
                    f"'{root_parent.name}' in `parameters` to keep its "
                    f"elements symbolic.",
                    operation="array element resolution",
                ) from exc
            except (KeyError, TypeError) as exc:
                # The same guards that make an IndexError here a genuine
                # out-of-range access also make a KeyError / TypeError a
                # genuine shape mismatch: the root array is not a runtime
                # parameter, its container is resolved compile-time data,
                # and every index is a concrete non-negative int. A
                # KeyError (the container is a dict) or TypeError (the
                # container is a scalar over-indexed by a deeper index, or
                # an otherwise non-indexable bound object) therefore means
                # the bound data is not an int-indexable array of the
                # expected shape. Returning None would fall through to the
                # same phantom-parameter / silent-0.0 hazard as the
                # out-of-range case, so fail fast instead.
                raise EmitError(
                    f"Compile-time bound array '{root_parent.name}' could "
                    f"not be indexed at index {i}: the bound value is not "
                    f"an indexable array of the expected shape (got "
                    f"{type(container).__name__}). Bind a correctly shaped "
                    f"array for '{root_parent.name}', or declare it in "
                    f"`parameters` to keep its elements symbolic.",
                    operation="array element resolution",
                ) from exc
        return container

    def _resolve_array_element_location(
        self,
        parent: "ArrayValue",
        element_indices: tuple["Value", ...],
        bindings: dict[str, Any],
    ) -> tuple["ArrayValue", tuple[int, ...]] | None:
        """Resolve an array element access to a root array and indices.

        Slice affine composition is applied to the leading index because
        Qamomile VectorView slices are one-dimensional Vector slices.

        Args:
            parent (ArrayValue): The immediate parent array on the
                element Value. This may be a sliced view.
            element_indices (tuple[Value, ...]): The element's indices in
                the immediate parent array's local coordinate system.
            bindings (dict[str, Any]): The active emit-time bindings.

        Returns:
            tuple[ArrayValue, tuple[int, ...]] | None: The root array and
                concrete indices into its container, or ``None`` when any
                index or slice bound is unresolved, when a resolved index
                is negative (Python-style wrapping is refused), or when a
                resolved slice bound violates the frontend contract
                (non-negative start, positive step).
        """
        resolved_indices: list[int] = []
        # Resolve local element indices first. Any symbolic index keeps the
        # whole element access unresolved at emit time.
        for idx in element_indices:
            i = self.resolve_int_value(idx, bindings)
            if i is None:
                # Symbolic element indices stay unresolved at emit time.
                return None
            if i < 0:
                # Negative indices must not reach Python container
                # indexing (where they would silently wrap) or the slice
                # affine composition below (where they would silently
                # address a wrong root slot).
                return None
            resolved_indices.append(i)

        # No explicit element index means callers are asking for the array
        # container itself, so there is no slice-local coordinate to compose.
        if not resolved_indices:
            return parent, ()

        root_index = resolved_indices[0]
        cur = parent
        # Compose each VectorView affine map back to the root container:
        # root_index = slice_start + slice_step * local_index.
        while cur.slice_of is not None:
            # A sliced view must carry both affine-map operands. If either is
            # absent, keep this access unresolved instead of inventing bounds.
            if cur.slice_start is None or cur.slice_step is None:
                return None
            start = self.resolve_int_value(cur.slice_start, bindings)
            step = self.resolve_int_value(cur.slice_step, bindings)
            if start is None or step is None:
                # Symbolic slice bounds stay unresolved at emit time.
                return None
            if start < 0 or step <= 0:
                # Bounds resolved from bindings must satisfy the same
                # contract the frontend enforces for constant bounds
                # (non-negative start, positive step); anything else
                # would compose a wrong root index.
                return None
            root_index = start + step * root_index
            cur = cur.slice_of

        return cur, (root_index, *resolved_indices[1:])

    def _resolve_array_container(
        self,
        parent: "ArrayValue",
        bindings: dict[str, Any],
    ) -> Any:
        """Resolve an array parent to a concrete container for indexing.

        Args:
            parent (ArrayValue): The parent array whose element is being
                resolved.
            bindings (dict[str, Any]): The active emit-time bindings.

        Returns:
            Any: The concrete array-like container, or ``None`` if the
                parent is unresolved.
        """
        # Emit-local loop bindings are UUID-scoped. Prefer them so nested
        # ``for_items`` keys with the same display name cannot shadow an
        # outer key that remains live in the inner body. Public caller
        # bindings normally have no UUID entry and therefore still resolve
        # through the stable parameter name below.
        container = bindings.get(parent.uuid)
        if container is not None:
            return container
        # Compile-time literal metadata is the next authoritative source.
        container = parent.get_const_array()
        if container is not None:
            return container
        if parent.is_parameter():
            parameter_name = parent.parameter_name()
            if parameter_name and parameter_name in bindings:
                return bindings[parameter_name]
        # Compatibility with programmatically constructed ArrayValues and
        # legacy transforms that have not preserved parameter metadata. The
        # UUID path above remains authoritative for compiler-internal arrays.
        if parent.name:
            return bindings.get(parent.name)
        return None

    def resolve_int_value(
        self,
        val: Any,
        bindings: dict[str, Any],
    ) -> int | None:
        """Resolve a value to an integer, or ``None`` when unresolvable.

        Unresolvable symbolic Values **must** return ``None``. The previous
        ``return 0`` fallback caused silent loop elision when parameter
        shape dims (``gamma_dim0``) reached this resolver without being
        folded into constants — downstream loop-bound resolution saw 0 and
        quietly emitted an empty loop. Returning ``None`` propagates the
        failure to ``emit_for_unrolled``, which converts it into a hard
        compile error.
        """
        raw = self.lookup_in_bindings(val, bindings, index_array=True)
        # Symbolic values, including array elements with symbolic indices,
        # must remain unresolved so emit callers can raise or fall back.
        if raw is None:
            return None
        return self._resolve_numeric_index(raw)

    def get_parameter_key(
        self,
        value: "Value",
        bindings: dict[str, Any],
    ) -> str | None:
        """Get the backend parameter key for a symbolic scalar.

        Args:
            value (Value): Scalar value or array element to identify.
            bindings (dict[str, Any]): Active bindings used to resolve
                element indices and slice offsets.

        Returns:
            str | None: Scalar name or root-array indexed key, or ``None``
                when an array element's root location is unresolved.
        """
        parameter_name = (
            value.parameter_name()
            if hasattr(value, "is_parameter") and value.is_parameter()
            else None
        )
        if parameter_name in self.parameters:
            return parameter_name

        if value.parent_array is not None:
            location = self._resolve_array_element_location(
                value.parent_array, value.element_indices, bindings
            )
            if location is None:
                return None
            root, indices = location
            root_name = root.parameter_name() if root.is_parameter() else None
            if root_name in self.parameters and indices:
                suffix = "".join(f"[{index}]" for index in indices)
                return f"{root_name}{suffix}"

        return None

    def _resolve_numeric_index(self, value: Any) -> int | None:
        """Resolve a bound numeric scalar to a Python int."""
        numeric = self._resolve_numeric_value(value)
        if numeric is None:
            return None
        return int(numeric)

    def _resolve_numeric_value(self, value: Any) -> int | float | None:
        """Normalize Python and NumPy numeric scalars to Python scalars."""
        if isinstance(value, numbers.Integral):
            return int(value)
        if isinstance(value, numbers.Real):
            return float(value)
        if hasattr(value, "item"):
            try:
                item = value.item()
            except (TypeError, ValueError):
                return None
            if isinstance(item, numbers.Integral):
                return int(item)
            if isinstance(item, numbers.Real):
                return float(item)
        return None
