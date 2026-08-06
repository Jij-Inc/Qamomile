"""Build dictionary-items loop evaluation contexts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import sympy as sp

from qamomile.circuit.estimator._estimate import (
    ResourceEstimate,
)
from qamomile.circuit.estimator._gate_models import _sympify_resource_value
from qamomile.circuit.estimator._interpreter_branches import _BranchInterpreter
from qamomile.circuit.estimator._resolver import (
    UnresolvedValueError,
)
from qamomile.circuit.estimator._scopes import (
    _typed_value_symbol,
)
from qamomile.circuit.estimator._symbol_discovery import _free_symbols
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    HasNestedOps,
)
from qamomile.circuit.ir.operation.operation import Operation
from qamomile.circuit.ir.value import (
    ArrayValue,
    Value,
    ValueBase,
    resolve_root_array_index,
)
from qamomile.circuit.transpiler.passes.control_flow_reachability import (
    static_for_items_entries,
)


class _ForItemsContextInterpreter(_BranchInterpreter):
    """Add dictionary entry resolution and per-entry binding services."""

    def _for_items_entries(
        self,
        operation: ForItemsOperation,
    ) -> tuple[tuple[Any, Any], ...] | None:
        """Return concrete entries available to an items loop.

        Args:
            operation (ForItemsOperation): Items loop to inspect.

        Returns:
            tuple[tuple[Any, Any], ...] | None: Bound entries, or ``None`` when
            the dictionary remains symbolic.
        """
        if not operation.operands:
            return ()
        operand = operation.operands[0]
        parameter_name = getattr(operand, "parameter_name", lambda: None)()
        bound = self.bindings.get(parameter_name) if parameter_name else None
        if isinstance(bound, Mapping):
            return tuple(bound.items())
        return static_for_items_entries(operation)

    def _symbolic_for_items_context(
        self,
        operation: ForItemsOperation,
    ) -> tuple[dict[str, sp.Expr], frozenset[sp.Symbol]]:
        """Build identity-bearing symbolic bindings for an items loop.

        Args:
            operation (ForItemsOperation): Items loop to bind.

        Returns:
            tuple[dict[str, sp.Expr], frozenset[sp.Symbol]]: UUID-keyed
                symbolic context and every symbol derived from the current
                key/value entry.

        Raises:
            UnresolvedValueError: If key or value formal identities are absent.
        """
        if operation.key_var_values is None:
            raise UnresolvedValueError(
                "?",
                "ForItemsOperation is missing key_var_values identities.",
            )
        if operation.value_var_value is None:
            raise UnresolvedValueError(
                "?",
                "ForItemsOperation is missing value_var_value identity.",
            )

        context: dict[str, sp.Expr] = {}
        iteration_symbols: set[sp.Symbol] = set()
        array_symbols: dict[str, sp.Symbol] = {}

        for index, key_value in enumerate(operation.key_var_values):
            name = (
                operation.key_vars[index] if index < len(operation.key_vars) else "key"
            )
            symbol = _typed_value_symbol(key_value, name, fresh=True)
            context[key_value.uuid] = symbol
            iteration_symbols.add(symbol)
            if isinstance(key_value, ArrayValue):
                array_symbols[key_value.uuid] = symbol
                for axis, dimension in enumerate(key_value.shape):
                    dimension_symbol = sp.Dummy(
                        f"{name}_dim{axis}",
                        integer=True,
                        nonnegative=True,
                    )
                    context[dimension.uuid] = dimension_symbol
                    iteration_symbols.add(dimension_symbol)

        value_symbol = _typed_value_symbol(
            operation.value_var_value,
            operation.value_var,
            fresh=True,
        )
        context[operation.value_var_value.uuid] = value_symbol
        iteration_symbols.add(value_symbol)

        def array_iteration_symbol(array: ArrayValue) -> sp.Symbol | None:
            """Find the item symbol behind an array value or view.

            Args:
                array (ArrayValue): Array whose slice ancestry is inspected.

            Returns:
                sp.Symbol | None: Matching item symbol, if any.
            """
            current: ArrayValue | None = array
            visited: set[str] = set()
            while current is not None and current.uuid not in visited:
                visited.add(current.uuid)
                if current.uuid in array_symbols:
                    return array_symbols[current.uuid]
                current = current.slice_of
            return None

        def register_array_alias(value: ValueBase) -> None:
            """Map vector-key elements and views to their item dependency.

            Args:
                value (ValueBase): Referenced IR value to inspect.
            """
            if isinstance(value, Value) and value.parent_array is not None:
                symbol = array_iteration_symbol(value.parent_array)
                if symbol is not None:
                    context[value.uuid] = symbol
            if isinstance(value, ArrayValue):
                for dimension in value.shape:
                    register_array_alias(dimension)
                if value.slice_start is not None:
                    register_array_alias(value.slice_start)
                if value.slice_step is not None:
                    register_array_alias(value.slice_step)

        def walk(operations: list[Operation]) -> None:
            """Register vector-key aliases throughout nested body operations.

            Args:
                operations (list[Operation]): Operations in the current scope.
            """
            for nested_operation in operations:
                for value in (
                    *nested_operation.all_input_values(),
                    *nested_operation.results,
                ):
                    if isinstance(value, ValueBase):
                        register_array_alias(value)
                if isinstance(nested_operation, HasNestedOps):
                    for nested in nested_operation.nested_op_lists():
                        walk(nested)

        walk(operation.operations)
        return context, frozenset(iteration_symbols)

    def _ensure_for_items_resource_independent(
        self,
        estimate: ResourceEstimate,
        iteration_symbols: frozenset[sp.Symbol],
    ) -> None:
        """Reject symbolic item loops whose resources or constraints vary.

        Args:
            estimate (ResourceEstimate): One symbolic body evaluation.
            iteration_symbols (frozenset[sp.Symbol]): Identity-bearing symbols
                assigned to current key/value formals and their derived array
                values.

        Raises:
            NotImplementedError: If any resource metric or structural
                requirement depends on a current dictionary key or value.
        """
        if _free_symbols(estimate) & iteration_symbols:
            raise NotImplementedError(
                "Resource estimation does not support a symbolic "
                "ForItemsOperation body whose resource use or structural "
                "requirements depend on the current item key or value. "
                "Supply a concrete dictionary or make the body "
                "entry-independent."
            )

    def _concrete_for_items_context(
        self,
        operation: ForItemsOperation,
        key: Any,
        value: Any,
    ) -> dict[str, sp.Expr]:
        """Build concrete key and value bindings for one item iteration.

        Args:
            operation (ForItemsOperation): Items loop to bind.
            key (Any): Current dictionary key.
            value (Any): Current dictionary value.

        Returns:
            dict[str, sp.Expr]: UUID-keyed scalar iteration context.

        Raises:
            ValueError: If a tuple-key value does not match the declared key
                arity.
        """
        context: dict[str, sp.Expr] = {}
        key_values = list(operation.key_var_values or ())
        if (
            operation.key_is_vector
            and len(key_values) == 1
            and isinstance(key_values[0], ArrayValue)
        ):
            self._bind_concrete_vector_key(
                operation,
                key_values[0],
                key,
                context,
            )
        elif len(key_values) > 1:
            if (
                not isinstance(key, Sequence)
                or isinstance(key, (str, bytes))
                or len(key) != len(key_values)
            ):
                raise ValueError(
                    "ForItems tuple key must contain exactly "
                    f"{len(key_values)} element(s), got {key!r}."
                )
            for ir_value, concrete in zip(key_values, key, strict=True):
                context[ir_value.uuid] = _sympify_resource_value(
                    concrete, ir_value.name
                )
        elif key_values:
            context[key_values[0].uuid] = _sympify_resource_value(
                key,
                key_values[0].name,
            )
        if operation.value_var_value is not None:
            context[operation.value_var_value.uuid] = _sympify_resource_value(
                value,
                operation.value_var,
            )
        return context

    def _bind_concrete_vector_key(
        self,
        operation: ForItemsOperation,
        formal: ArrayValue,
        key: Any,
        context: dict[str, sp.Expr],
    ) -> None:
        """Bind a concrete Vector dictionary key into body Value identities.

        Args:
            operation (ForItemsOperation): Items loop whose body references the
                vector key.
            formal (ArrayValue): Vector-key region formal.
            key (Any): Concrete dictionary key for one iteration.
            context (dict[str, sp.Expr]): UUID-keyed context updated in place.

        Raises:
            ValueError: If the concrete key is not a sequence or a constant
                element access is out of bounds.
            NotImplementedError: If the body dynamically indexes the current
                vector key; exact per-entry resource evaluation for that shape
                is not implemented.
        """
        if not isinstance(key, Sequence) or isinstance(key, (str, bytes)):
            raise ValueError(
                "A concrete Dict[Vector, ...] key must be a sequence of scalar values."
            )
        elements = tuple(key)
        if formal.shape:
            context[formal.shape[0].uuid] = sp.Integer(len(elements))

        seen: set[int] = set()

        def belongs_to_formal(array: ArrayValue) -> bool:
            """Return whether an array is the formal or one of its views.

            Args:
                array (ArrayValue): Candidate key array or view.

            Returns:
                bool: True when its slice ancestry reaches the vector formal.
            """
            current: ArrayValue | None = array
            visited: set[int] = set()
            while current is not None and id(current) not in visited:
                visited.add(id(current))
                if current.logical_id == formal.logical_id:
                    return True
                current = current.slice_of
            return False

        def register_value(value: ValueBase) -> None:
            """Bind concrete key elements reachable through one IR value.

            Args:
                value (ValueBase): Referenced body value to inspect
                    recursively.

            Raises:
                ValueError: If a constant key index is out of bounds.
                NotImplementedError: If a current-key element has a dynamic
                    index or unresolved view mapping.
            """
            if id(value) in seen:
                return
            seen.add(id(value))

            for index in getattr(value, "element_indices", ()):
                register_value(index)
            parent = getattr(value, "parent_array", None)
            if isinstance(parent, ArrayValue):
                register_value(parent)

            if (
                isinstance(value, Value)
                and isinstance(parent, ArrayValue)
                and belongs_to_formal(parent)
            ):
                if len(value.element_indices) != 1:
                    raise NotImplementedError(
                        "Resource estimation supports only one-dimensional "
                        "constant indexing of a concrete ForItems Vector key."
                    )
                index_value = value.element_indices[0]
                if not index_value.is_constant():
                    raise NotImplementedError(
                        "Resource estimation cannot exactly evaluate a "
                        "dynamically indexed concrete ForItems Vector key."
                    )
                local_index = index_value.get_const()
                if isinstance(local_index, bool) or not isinstance(local_index, int):
                    raise NotImplementedError(
                        "Resource estimation requires integer indexing for a "
                        "concrete ForItems Vector key."
                    )
                resolved = resolve_root_array_index(parent, local_index)
                if resolved is None or resolved[0].logical_id != formal.logical_id:
                    raise NotImplementedError(
                        "Resource estimation could not resolve a concrete "
                        "ForItems Vector-key view to its formal index space."
                    )
                root_index = resolved[1]
                if not 0 <= root_index < len(elements):
                    raise ValueError(
                        f"ForItems Vector-key index {root_index} is out of "
                        f"bounds for a key of length {len(elements)}."
                    )
                context[value.uuid] = _sympify_resource_value(
                    elements[root_index],
                    value.name,
                )

            if isinstance(value, ArrayValue):
                for dimension in value.shape:
                    register_value(dimension)
                if value.slice_start is not None:
                    register_value(value.slice_start)
                if value.slice_step is not None:
                    register_value(value.slice_step)

        def walk(operations: list[Operation]) -> None:
            """Visit nested loop-body operations for key references.

            Args:
                operations (list[Operation]): Operations to inspect.
            """
            for body_operation in operations:
                for value in (
                    *body_operation.all_input_values(),
                    *body_operation.results,
                ):
                    if isinstance(value, ValueBase):
                        register_value(value)
                if isinstance(body_operation, HasNestedOps):
                    for nested in body_operation.nested_op_lists():
                        walk(nested)

        walk(operation.operations)
