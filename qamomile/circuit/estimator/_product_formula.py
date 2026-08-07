"""Interpret product-formula resource contracts on callable boundaries."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import sympy as sp

import qamomile.observable as qm_o
from qamomile.circuit.estimator._constants import _ONE, _ZERO
from qamomile.circuit.estimator._estimate import ResourceEstimate
from qamomile.circuit.estimator._estimate_validation import _with_constraints
from qamomile.circuit.estimator._gate_models import _pauli_terms_share_local_basis
from qamomile.circuit.estimator._resolver import ExprResolver
from qamomile.circuit.estimator._resolver_indices import (
    _resolve_concrete_array_payload,
)
from qamomile.circuit.estimator._resource_base import ApproximationStatus
from qamomile.circuit.estimator._resource_constraints import _ResourceConstraint
from qamomile.circuit.estimator._resource_types import (
    ResourceAssumption,
    ResourceTraceNode,
)
from qamomile.circuit.ir._resource_contract import product_formula_contract
from qamomile.circuit.ir.value import ArrayValue, ValueBase
from qamomile.circuit.transpiler.passes.emit_support.value_resolver import (
    ValueResolver,
)
from qamomile.observable.hamiltonian import PAULI_TERM_ZERO_ATOL


def _bound_hamiltonian_components(
    operand: ValueBase,
    bindings: Mapping[str, Any],
    *,
    resolver: ExprResolver,
    specialize: Callable[[sp.Expr], sp.Expr],
    source: str,
) -> tuple[qm_o.Hamiltonian, ...] | None:
    """Resolve a product formula's compile-time Hamiltonian vector.

    Args:
        operand (ValueBase): Callable operand declared as the Hamiltonian
            vector by the resource contract.
        bindings (Mapping[str, Any]): Root build-time object bindings.
        resolver (ExprResolver): Resolver for structural expressions in the
            operand's current callable scope.
        specialize (Callable[[sp.Expr], sp.Expr]): Function applying concrete
            estimator inputs to resolved expressions.
        source (str): Callable name used in malformed-input diagnostics.

    Returns:
        tuple[qm_o.Hamiltonian, ...] | None: Concrete components, or ``None``
        when the vector remains unresolved.

    Raises:
        ValueError: If a resolved payload is not a sequence of at least two
            Hamiltonians.
    """
    if isinstance(operand, ArrayValue):
        payload: Any = _resolve_concrete_array_payload(
            operand,
            bindings,
            resolve_expression=resolver.resolve,
            specialize=specialize,
            source=f"{source} product-formula Hamiltonian",
        )
    else:
        payload = ValueResolver().resolve_bound_value(
            cast(Any, operand),
            dict(bindings),
        )
    if payload is None:
        candidates = [operand.uuid]
        parameter_name = operand.parameter_name()
        if parameter_name is not None:
            candidates.append(parameter_name)
        if operand.name:
            candidates.append(operand.name)
        for candidate in candidates:
            if candidate in bindings:
                payload = bindings[candidate]
                break
    if isinstance(payload, ArrayValue):
        payload = payload.get_const_array()
    if payload is None:
        return None
    if isinstance(payload, (str, bytes, Mapping)):
        raise ValueError(
            f"{source} product-formula Hamiltonian operand must resolve to "
            "a flat iterable of Hamiltonians."
        )
    try:
        components = tuple(payload)
    except TypeError as error:
        raise ValueError(
            f"{source} product-formula Hamiltonian operand must resolve to "
            "a flat iterable of Hamiltonians."
        ) from error
    if len(components) < 2:
        raise ValueError(
            f"{source} product-formula Hamiltonian must contain at least "
            f"2 components, got {len(components)}."
        )
    if not all(isinstance(component, qm_o.Hamiltonian) for component in components):
        raise ValueError(
            f"{source} product-formula Hamiltonian operand contains a "
            "non-Hamiltonian component."
        )
    return cast(tuple[qm_o.Hamiltonian, ...], components)


def _components_commute(components: Sequence[qm_o.Hamiltonian]) -> bool:
    """Return whether all component Hamiltonians commute pairwise.

    Components drawn from one local tensor-product basis take a linear fast
    path. Mixed-basis components use the complete commutator instead of
    requiring every Pauli-string pair to commute, because nonzero pair
    contributions can cancel.

    Args:
        components (Sequence[qm_o.Hamiltonian]): Concrete Hamiltonians in
            product-formula order.

    Returns:
        bool: Whether every pair has a zero commutator.
    """
    active_terms = (
        operators
        for component in components
        for operators, coefficient in component
        if operators and abs(complex(coefficient)) >= PAULI_TERM_ZERO_ATOL
    )
    if _pauli_terms_share_local_basis(active_terms):
        return True

    for index, left in enumerate(components):
        for right in components[index + 1 :]:
            commutator = qm_o.commutator(left, right)
            if (
                len(commutator)
                or abs(complex(commutator.constant)) >= PAULI_TERM_ZERO_ATOL
            ):
                return False
    return True


def _product_formula_constraints(
    order: sp.Expr,
    steps: sp.Expr,
) -> tuple[_ResourceConstraint, ...]:
    """Build persistent Suzuki–Trotter order and step requirements.

    Args:
        order (sp.Expr): Formula order expression.
        steps (sp.Expr): Finite step-count expression.

    Returns:
        tuple[_ResourceConstraint, ...]: Constraints retained through later
        symbolic substitution.
    """
    return (
        _ResourceConstraint(
            expression=order,
            minimum=1,
            label="Suzuki-Trotter order",
        ),
        _ResourceConstraint(
            expression=order,
            minimum=None,
            expected=_ONE,
            label="odd Suzuki-Trotter order",
            active_when=sp.Ne(sp.Mod(order, 2), _ZERO),
        ),
        _ResourceConstraint(
            expression=steps,
            minimum=1,
            label="Suzuki-Trotter step count",
        ),
    )


def _require_concrete_product_formula_structure(
    attrs: Mapping[str, Any],
    operands: Sequence[ValueBase],
    *,
    bindings: Mapping[str, Any],
    resolver: ExprResolver | None = None,
    specialize: Callable[[sp.Expr], sp.Expr] | None = None,
    source: str,
) -> bool:
    """Reject a product formula whose recursive structure is unresolved.

    Suzuki order selects recursive callable bodies. Resource-estimation
    preparation must know it before inlining those bodies; otherwise symbolic
    self-recursion cannot reach a base case.

    Args:
        attrs (Mapping[str, Any]): Merged callable resource attributes.
        operands (Sequence[ValueBase]): Untransformed callable operands.
        bindings (Mapping[str, Any]): Active UUID and public-name bindings.
        resolver (ExprResolver | None): Caller-scope expression resolver used
            for computed structural operands. Defaults to ``None``.
        specialize (Callable[[sp.Expr], sp.Expr] | None): Concrete-input
            substitution applied after expression resolution. Defaults to
            ``None``.
        source (str): Callable name used in the diagnostic.

    Returns:
        bool: Whether a product-formula contract was present.

    Raises:
        ValueError: If the declared Suzuki order is not a compile-time
            constant on the current call site.
    """
    contract = product_formula_contract(
        attrs,
        source=source,
        operand_count=len(operands),
    )
    if contract is None:
        return False
    order = operands[contract.order_operand]
    concrete_order: int | None = None
    if resolver is not None:
        resolved_order = resolver.resolve(order)
        if specialize is not None:
            resolved_order = specialize(resolved_order)
        if resolved_order.is_number and resolved_order.is_integer is True:
            concrete_order = int(resolved_order)
    if concrete_order is None:
        concrete_order = ValueResolver().resolve_int_value(order, dict(bindings))
    if concrete_order is not None:
        if concrete_order < 1:
            raise ValueError(
                f"Suzuki-Trotter order must be at least 1, got {concrete_order}."
            )
        if concrete_order != 1 and concrete_order % 2:
            raise ValueError(
                f"odd Suzuki-Trotter order must equal 1, got {concrete_order}."
            )
        return True
    name = order.parameter_name() or order.name or "order"
    raise ValueError(
        f"{source} resource estimation requires product-formula order "
        f"'{name}' as a concrete input because it determines recursive "
        "Suzuki circuit structure."
    )


def _apply_product_formula_contract(
    estimate: ResourceEstimate,
    attrs: Mapping[str, Any],
    operands: Sequence[ValueBase],
    resolver: ExprResolver,
    *,
    bindings: Mapping[str, Any],
    specialize: Callable[[sp.Expr], sp.Expr],
    source: str,
) -> ResourceEstimate:
    """Apply one callable's declared product-formula resource semantics.

    This changes only constraints and approximation metadata. The callable
    body's gate, depth, and width resources remain structurally derived from
    the actual Suzuki–Trotter circuit.

    Args:
        estimate (ResourceEstimate): Resources derived from the callable body.
        attrs (Mapping[str, Any]): Merged callable resource attributes.
        operands (Sequence[ValueBase]): Untransformed callable operands in ABI
            order, excluding coherent controls added by a later transform.
        resolver (ExprResolver): Resolver for the operands' current scope.
        bindings (Mapping[str, Any]): Root build-time object bindings.
        specialize (Callable[[sp.Expr], sp.Expr]): Function applying concrete
            scalar estimator inputs to a resolved expression.
        source (str): Callable name used in diagnostics and assumptions.

    Returns:
        ResourceEstimate: Estimate with persistent validity constraints and,
        for noncommuting components at nonzero time, approximation metadata.

    Raises:
        ValueError: If the contract is malformed, its operands are invalid, or
            concrete order/step values violate Suzuki–Trotter requirements.
    """
    contract = product_formula_contract(
        attrs,
        source=source,
        operand_count=len(operands),
    )
    if contract is None:
        return estimate

    order = specialize(resolver.resolve(operands[contract.order_operand]))
    time = specialize(resolver.resolve(operands[contract.time_operand]))
    steps = specialize(resolver.resolve(operands[contract.steps_operand]))
    constrained = _with_constraints(
        estimate,
        *_product_formula_constraints(order, steps),
    )
    components = _bound_hamiltonian_components(
        operands[contract.hamiltonian_operand],
        bindings,
        resolver=resolver,
        specialize=specialize,
        source=source,
    )
    if time.is_zero is True or (
        components is not None and _components_commute(components)
    ):
        return constrained

    if components is None:
        message = (
            "Hamiltonian commutation is unresolved; the finite-step "
            "Suzuki-Trotter construction is treated as an approximation "
            f"(order={sp.sstr(order)}, steps={sp.sstr(steps)})."
        )
    else:
        message = (
            "Noncommuting component Hamiltonians use a finite-step "
            f"Suzuki-Trotter product formula (order={sp.sstr(order)}, "
            f"steps={sp.sstr(steps)})."
        )
    assumption = ResourceAssumption(message, source=source)
    active_when = sp.Ne(time, _ZERO)
    modeled = constrained._with_metadata(
        assumptions=(assumption,),
        approximation=ApproximationStatus.APPROXIMATE,
        active_when=active_when,
    )
    assumption_trace = ResourceTraceNode(
        name=source,
        source_kind="model",
        assumptions=(assumption,),
        active_when=active_when,
    )
    if modeled.trace is None:
        return dataclasses.replace(modeled, trace=assumption_trace)
    return dataclasses.replace(
        modeled,
        trace=dataclasses.replace(
            modeled.trace,
            children=(*modeled.trace.children, assumption_trace),
        ),
    )


__all__ = [
    "_apply_product_formula_contract",
    "_require_concrete_product_formula_structure",
]
