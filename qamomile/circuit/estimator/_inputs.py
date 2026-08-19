"""Partition, validate, and specialize resource-estimation inputs."""

from __future__ import annotations

import numbers
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any, cast

import sympy as sp

import qamomile.circuit.estimator._input_contract as _input_contract
from qamomile.circuit.estimator._estimate import (
    ResourceEstimate,
)
from qamomile.circuit.estimator._resource_base import (
    _is_concrete_integer,
    _symbol_display_name,
)
from qamomile.circuit.estimator._resource_expressions import (
    _safe_constraint_substitute,
    _substitute_resource_expr,
)
from qamomile.circuit.estimator._resource_types import (
    ResourceAssumption,
)
from qamomile.circuit.estimator._symbol_discovery import _free_symbols
from qamomile.circuit.estimator._symbolic import (
    _canonicalize_concrete_integer,
    _normalize_resource_scalar,
)
from qamomile.circuit.frontend.func_to_block import is_array_type
from qamomile.circuit.frontend.handle import Observable
from qamomile.circuit.frontend.handle.primitives import Bit, Qubit
from qamomile.circuit.frontend.qkernel_inputs import (
    is_parameterizable_type,
    validate_bound_input_value,
)
from qamomile.circuit.frontend.qkernel_utils import get_array_element_type
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.dataflow import (
    walk_operations,
)
from qamomile.circuit.ir.operation.operation import (
    Operation,
)
from qamomile.circuit.ir.types.primitives import (
    BitType,
    FloatType,
    UIntType,
)
from qamomile.circuit.ir.value import (
    ArrayValue,
    ValueBase,
)
from qamomile.observable import Hamiltonian


def _validate_explicit_estimation_inputs(
    kernel: Any,
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate qkernel inputs once before build/estimation partitioning.

    Structural values must not bypass validation merely because they are
    consumed while tracing, while scalar and numeric-array inputs keep their
    existing post-interpretation symbolic specialization semantics. Raw IR
    targets do not expose frontend annotations and therefore continue through
    the IR contract validation path.

    Args:
        kernel (Any): QKernel, block, or operation sequence being estimated.
        inputs (Mapping[str, Any]): Explicit user inputs.

    Returns:
        dict[str, Any]: A defensive copy of the validated input mapping.

    Raises:
        TypeError: If a structural input has the wrong Python type or contains
            a value incompatible with its qkernel annotation.
        ValueError: If a structural value violates a finite-domain or
            container-shape contract.
    """
    values = dict(inputs)
    input_types = getattr(kernel, "input_types", None)
    if not isinstance(input_types, dict):
        _validate_raw_estimation_inputs(kernel, values)
        return values

    for name, value in values.items():
        if name not in input_types:
            # Generated array-dimension aliases are only known after tracing;
            # preserve them for the existing strict post-build contract.
            continue
        input_type = input_types[name]
        if input_type is Qubit or (
            is_array_type(input_type) and get_array_element_type(input_type) is Qubit
        ):
            # Quantum vectors accept estimator-only integer widths and
            # shape-only providers, neither of which is a frontend binding.
            # Their dedicated contract runs after the symbolic block exists.
            continue
        if is_parameterizable_type(input_type):
            # Scalar/numeric-array values deliberately remain symbolic until
            # after interpretation and use _apply_inputs plus shape expansion.
            continue
        if input_type is Observable:
            if not isinstance(value, Hamiltonian):
                raise TypeError(
                    f"resource input '{name}' expects a Hamiltonian, got "
                    f"{type(value).__name__} ({value!r})."
                )
            continue
        if (
            is_array_type(input_type)
            and get_array_element_type(input_type) is Observable
        ):
            if (
                isinstance(value, (str, bytes))
                or not isinstance(value, Sequence)
                and getattr(value, "shape", None) is None
            ):
                raise TypeError(
                    f"resource input '{name}' expects a sequence of "
                    f"Hamiltonians, got {type(value).__name__} ({value!r})."
                )
            for index, item in enumerate(value):
                if not isinstance(item, Hamiltonian):
                    raise TypeError(
                        f"resource input '{name}' element {index} expects a "
                        f"Hamiltonian, got {type(item).__name__} ({item!r})."
                    )
            continue
        validate_bound_input_value(input_type, name, value)
    return values


def _validate_raw_estimation_inputs(
    kernel: Any,
    inputs: Mapping[str, Any],
) -> None:
    """Validate structural inputs recoverable from a raw IR target.

    Raw blocks do not preserve frontend annotations, but their input values
    still expose enough IR type and shape information to reject invalid Bit
    bindings and malformed quantum-array widths before input partitioning can
    remove those values from the later specialization path.

    Args:
        kernel (Any): Raw block or operation sequence being estimated.
        inputs (Mapping[str, Any]): Explicit user inputs.

    Raises:
        TypeError: If a raw Bit input is not boolean or integral.
        ValueError: If a raw Bit is outside ``0..1`` or a quantum-array input
            has an invalid width, shape, or rank.
    """
    declared_inputs = _input_contract._declared_ir_inputs(kernel)
    for name, value in inputs.items():
        declared = declared_inputs.get(name)
        if declared is None:
            continue
        if isinstance(declared, ArrayValue) and declared.type.is_quantum():
            _validate_raw_quantum_array_input(name, declared, value)
            continue
        if isinstance(declared.type, BitType):
            validate_bound_input_value(Bit, name, value)


def _validate_raw_quantum_array_input(
    name: str,
    declared: ArrayValue,
    value: Any,
) -> None:
    """Validate one raw quantum-array width or concrete shape provider.

    Args:
        name (str): Public raw-IR input name used in diagnostics.
        declared (ArrayValue): Declared quantum-array IR input.
        value (Any): Integer width or array-like shape provider.

    Raises:
        ValueError: If ``value`` is not a valid width or has the wrong rank.
    """
    if (
        len(declared.shape) == 1
        and not isinstance(value, bool)
        and isinstance(value, numbers.Integral)
    ):
        width = int(value)
        if width < 0:
            raise ValueError(
                f"quantum array input '{name}' requires a non-negative integer "
                f"width, got {width}."
            )
        return
    if len(declared.shape) == 1 and isinstance(value, (bool, numbers.Real)):
        raise ValueError(
            f"quantum array input '{name}' requires an integer width or an "
            "array-like value."
        )
    shape = _input_contract._concrete_input_shape(value)
    if not shape and not _input_contract._is_array_like_input(value):
        raise ValueError(
            f"quantum array input '{name}' requires an integer width or an "
            f"array-like value; got {type(value).__name__} ({value!r})."
        )
    if len(shape) != len(declared.shape):
        raise ValueError(
            f"array input '{name}' has rank {len(shape)}, but the qkernel "
            f"declares rank {len(declared.shape)}."
        )


def _substitute_bindings(
    estimate: ResourceEstimate,
    bindings: Mapping[str, Any],
) -> ResourceEstimate:
    """Apply scalar and dictionary-cardinality bindings.

    Args:
        estimate (ResourceEstimate): Estimate to rewrite.
        bindings (Mapping[str, Any]): User bindings.

    Returns:
        ResourceEstimate: Rewritten estimate.
    """
    values: dict[str, int | float] = {}
    for key, value in bindings.items():
        if isinstance(value, dict):
            values[f"|{key}|"] = len(value)
        elif isinstance(value, (int, float)):
            values[key] = value
    if not values:
        return estimate
    applicable = {
        name: value for name, value in values.items() if name in estimate.parameters
    }
    if not applicable:
        return estimate
    return estimate.substitute(**applicable)


def _partition_estimation_inputs(
    kernel: Any,
    inputs: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Partition one user input mapping by its role during estimation.

    Scalar and numeric-array qkernel inputs remain symbolic until after
    interpretation. Structural values such as observables and dictionaries are
    supplied while tracing. A raw ``Block`` or operation list is already
    traced: recognized structural values are supplied directly during abstract
    interpretation, while UInt/Float scalars and arrays stay symbolic until
    the compact estimate is built. Unknown names remain post-interpretation
    inputs so strict validation can report them as typos.

    Args:
        kernel (Any): QKernel, block, or operation sequence being estimated.
        inputs (Mapping[str, Any] | None): User-provided input values.

    Returns:
        tuple[dict[str, Any], dict[str, Any]]: Build-time structural inputs and
        post-interpretation estimation inputs.
    """
    values = dict(inputs or {})
    input_types = getattr(kernel, "input_types", None)
    if not isinstance(input_types, dict):
        declared_inputs = _input_contract._declared_ir_inputs(kernel)
        build_inputs = {
            name: value
            for name, value in values.items()
            if name in declared_inputs
            and _raw_input_requires_interpretation(declared_inputs[name])
        }
        estimation_inputs = {
            name: value for name, value in values.items() if name not in build_inputs
        }
        return build_inputs, estimation_inputs

    def is_quantum_port(input_type: Any) -> bool:
        """Return whether an annotation denotes a quantum input port.

        Args:
            input_type (Any): Resolved qkernel input annotation.

        Returns:
            bool: Whether the annotation is a qubit or qubit vector.
        """
        if input_type is Qubit:
            return True
        return is_array_type(input_type) and get_array_element_type(input_type) is Qubit

    build_inputs = {
        name: value
        for name, value in values.items()
        if name in input_types
        and not is_parameterizable_type(input_types[name])
        and not is_quantum_port(input_types[name])
    }
    estimation_inputs = {
        name: value for name, value in values.items() if name not in build_inputs
    }
    return build_inputs, estimation_inputs


def _raw_input_requires_interpretation(value: ValueBase) -> bool:
    """Return whether a raw-IR input must be bound before interpretation.

    Numeric UInt/Float scalars and arrays stay symbolic until the compact
    estimate has been built, matching qkernel estimation and preventing a
    concrete problem size from unrolling a large region loop. Object/container
    values and quantum-array widths instead affect operation meaning or shape
    and must be available to the interpreter.

    Args:
        value (ValueBase): Declared raw-IR input value.

    Returns:
        bool: Whether the value belongs in interpreter bindings.
    """
    if isinstance(value, ArrayValue) and value.type.is_quantum():
        return True
    return not isinstance(value.type, (UIntType, FloatType))


def _root_input_binding_context(
    block_or_ops: Block | Sequence[Operation],
    bindings: Mapping[str, Any],
) -> dict[str, sp.Expr]:
    """Map concrete raw-IR inputs to the resolver's root value identities.

    Scalar values bind the matching parameter UUID directly. Array payloads
    bind only their dimension UUIDs; resource estimation does not interpret
    arbitrary classical array contents as scalar expressions. The public
    ``Block`` labels are authoritative when available, while a raw operation
    list relies on each operand's parameter metadata.

    Args:
        block_or_ops (Block | Sequence[Operation]): Raw IR estimator target.
        bindings (Mapping[str, Any]): Concrete inputs supplied for abstract
            interpretation.

    Returns:
        dict[str, sp.Expr]: IR value UUIDs mapped to concrete or symbolic SymPy
            expressions.

    Raises:
        ValueError: If a concrete array input has a rank different from its IR
            declaration.
    """
    scalar_bindings = _scalar_values(bindings)
    scalar_bindings.update(
        {
            name: cast(sp.Expr, value)
            for name, value in bindings.items()
            if isinstance(value, sp.Basic) and not value.is_number
        }
    )
    context: dict[str, sp.Expr] = {}

    def bind_value(name: str, value: ValueBase) -> None:
        """Bind one named public value into the root resolver context.

        Args:
            name (str): Public input name used in ``bindings``.
            value (ValueBase): Matching IR input or parameter operand.

        Raises:
            ValueError: If an array payload rank differs from ``value.shape``.
        """
        if name not in bindings:
            return
        if isinstance(value, ArrayValue):
            supplied = bindings[name]
            if (
                value.type.is_quantum()
                and len(value.shape) == 1
                and not isinstance(supplied, bool)
                and isinstance(supplied, numbers.Integral)
            ):
                shape = (int(supplied),)
            else:
                if (
                    value.type.is_quantum()
                    and len(value.shape) == 1
                    and isinstance(supplied, (bool, numbers.Real))
                ):
                    raise ValueError(
                        f"quantum array input '{name}' requires an integer "
                        "width or an array-like value."
                    )
                shape = _input_contract._concrete_input_shape(supplied)
            if shape and len(shape) != len(value.shape):
                raise ValueError(
                    f"array input '{name}' has rank {len(shape)}, but the "
                    f"qkernel declares rank {len(value.shape)}."
                )
            for dimension, size in zip(value.shape, shape):
                if dimension.is_constant():
                    expected = dimension.get_const()
                    if not _input_contract._input_values_equal(size, expected):
                        raise ValueError(
                            f"array input '{name}' dimension '{dimension.name}' "
                            f"is fixed at {expected}, but the supplied shape "
                            f"has size {size}."
                        )
                    continue
                context[dimension.uuid] = sp.Integer(size)
            return
        if name in scalar_bindings:
            context[value.uuid] = scalar_bindings[name]

    if isinstance(block_or_ops, Block):
        for name, value in zip(block_or_ops.label_args, block_or_ops.input_values):
            bind_value(name, value)
        for name, value in block_or_ops.parameters.items():
            bind_value(name, value)
        return context

    for operation in walk_operations(block_or_ops):
        for operand in operation.all_input_values():
            parameter_name = operand.parameter_name()
            if parameter_name is not None:
                bind_value(parameter_name, operand)
    return context


def _estimator_parameters(
    kernel: Any,
    kwargs: Mapping[str, Any],
) -> list[str] | None:
    """Choose symbolic classical parameters for resource estimation.

    Resource estimation is symbolic-first: every parameterizable classical
    argument not supplied as a structural build binding remains symbolic,
    including arguments with Python defaults. Estimation inputs are substituted
    only after interpretation.

    Args:
        kernel (Any): QKernel being built.
        kwargs (Mapping[str, Any]): Compile-time build bindings.

    Returns:
        list[str] | None: Parameter list used to build the estimator IR.
    """
    input_types = getattr(kernel, "input_types", None)
    if not isinstance(input_types, dict):
        return None
    return [
        name
        for name, input_type in input_types.items()
        if name not in kwargs and is_parameterizable_type(input_type)
    ]


def _scalar_values(values: Mapping[str, Any]) -> dict[str, sp.Expr]:
    """Keep numeric scalars for branches and physical dependency resolution.

    Accepts Python and NumPy numeric scalars (values registered as
    ``numbers.Real`` plus ``decimal.Decimal``; NumPy scalars are normalized via
    ``.item()``) and SymPy numbers. Dicts, Hamiltonians, and symbolic-expression
    substitution values are dropped: only concrete numbers can decide a branch
    or select one physical array/control index during the initial scheduling
    pass. Integer-valued numbers are represented as exact SymPy integers so
    accepted float bounds follow the same scheduling path as Python integers.

    Args:
        values (Mapping[str, Any]): Concrete input values.

    Returns:
        dict[str, sp.Expr]: Name -> numeric SymPy value.
    """
    out: dict[str, sp.Expr] = {}
    for name, value in values.items():
        if isinstance(value, bool):
            out[name] = sp.Integer(int(value))
        elif isinstance(value, (numbers.Real, Decimal)):
            # Normalize NumPy scalars (np.int64, np.float64, ...) to a Python
            # scalar before sympifying.
            scalar = value.item() if hasattr(value, "item") else value
            normalized = cast(sp.Expr, sp.sympify(scalar))
            out[name] = _canonicalize_concrete_integer(normalized)
        elif isinstance(value, sp.Basic) and value.is_number:
            out[name] = _canonicalize_concrete_integer(cast(sp.Expr, value))
    return out


def _apply_inputs(
    estimate: ResourceEstimate,
    inputs: Mapping[str, Any],
    *,
    contract_names: frozenset[str] | None = None,
    input_types: Mapping[str, Any] | None = None,
    branch_condition_names: set[str] | None = None,
    consumed_input_names: frozenset[str] | None = None,
) -> ResourceEstimate:
    """Specialize a symbolic estimate with qkernel input values.

    Unlike :func:`_substitute_bindings`, substitution values may be SymPy
    expressions (e.g. an optimal-iteration formula). Each name is classified:

    - a **free symbol** of the estimate is substituted;
    - a name used for a **branch or physical dependency decision** is silently
      accepted; the initial interpretation already accounts for it;
    - an array input whose **shape specialized dimension symbols** is accepted
      without claiming that the whole input was ignored;
    - any other **declared kernel argument** not appearing in the estimate (e.g.
      a rotation angle) is a no-op, recorded as an assumption so it stays
      auditable;
    - a name that is **none of these** is a typo and raises.

    Args:
        estimate (ResourceEstimate): Symbolic estimate to rewrite.
        inputs (Mapping[str, Any]): Input values keyed by parameter name. Values
            may be numbers or SymPy expressions.
        contract_names (frozenset[str] | None): Declared kernel argument names.
            ``None`` (raw op sequence, no contract) keeps every name strict.
        input_types (Mapping[str, Any] | None): Recoverable scalar input types
            used to distinguish Bit booleans from invalid UInt/Float booleans.
            Defaults to ``None``.
        branch_condition_names (set[str] | None): Names that participated in a
            compile-time branch or dependency-scheduling decision during
            interpretation. Defaults to ``None``.
        consumed_input_names (frozenset[str] | None): Declared inputs whose
            array shape already specialized dimension symbols. Defaults to
            ``None``.

    Returns:
        ResourceEstimate: Estimate with the inputs applied.

    Raises:
        ValueError: If an input name is neither a free symbol of the
            estimate nor a declared kernel argument, or a negative input is
            supplied for a nonnegative resource symbol.
        TypeError: If a boolean is supplied for a non-Bit scalar input, or if
            a string is supplied where a scalar value or explicit SymPy
            expression is required.
    """
    # SymPy treats same-named symbols with different assumptions as distinct, so
    # a name can map to more than one symbol object; substitute every match.
    symbols_by_name: dict[str, list[sp.Symbol]] = {}
    for symbol in _free_symbols(estimate):
        # Identity-fresh internal fallbacks are not qkernel inputs even when a
        # frontend-generated display name collides with a declared argument.
        # Users may still specialize them explicitly through
        # ``ResourceEstimate.substitute`` after estimation.
        if isinstance(symbol, sp.Dummy):
            continue
        symbols_by_name.setdefault(_symbol_display_name(symbol), []).append(symbol)
    known = contract_names or frozenset()
    referenced = set(branch_condition_names or ()) | set(consumed_input_names or ())
    declared_types = input_types or {}
    unknown = [
        name
        for name in inputs
        if name not in symbols_by_name and name not in known and name not in referenced
    ]
    if unknown:
        available = ", ".join(sorted(set(symbols_by_name) | set(known))) or "(none)"
        raise ValueError(
            f"input names {sorted(unknown)} are neither free symbols of "
            f"the estimate nor kernel arguments; available: {available}. Use "
            "the qkernel's declared input names."
        )
    subs: dict[sp.Symbol, sp.Expr] = {}
    ignored: list[str] = []
    for name, value in inputs.items():
        if isinstance(value, (str, bytes)):
            raise TypeError(
                f"resource input '{name}' requires a numeric scalar or explicit "
                f"SymPy expression, got {type(value).__name__} ({value!r})."
            )
        if isinstance(value, bool) and (
            name in declared_types and not isinstance(declared_types[name], BitType)
        ):
            raise TypeError(
                f"resource input '{name}' expects "
                f"{type(declared_types[name]).__name__}, got bool ({value!r})."
            )
        if name not in symbols_by_name:
            # Not a free symbol: either initial branch/dependency
            # interpretation already consumed it (silent), or it genuinely
            # affects nothing (recorded as an ignored no-op).
            if name not in referenced:
                ignored.append(name)
            continue
        _validate_finite_source_domain_input(
            estimate,
            symbols_by_name[name],
            value,
        )
        sympified = _normalize_resource_scalar(
            value,
            label=f"resource input '{name}'",
            allow_symbolic=True,
            allow_bool=(
                name not in declared_types or isinstance(declared_types[name], BitType)
            ),
        )
        if (
            sympified.is_number
            and any(symbol.is_integer is True for symbol in symbols_by_name[name])
            and not _is_concrete_integer(cast(sp.Expr, sympified))
        ):
            raise ValueError(
                f"Cannot apply non-integer value {value!r} to integer "
                f"resource parameter '{name}'."
            )
        if any(symbol.is_integer is True for symbol in symbols_by_name[name]):
            sympified = _canonicalize_concrete_integer(sympified)
        if sympified.is_negative is True and any(
            symbol.is_nonnegative is True for symbol in symbols_by_name[name]
        ):
            raise ValueError(
                f"Cannot apply negative value {value!r} to nonnegative "
                f"resource parameter '{name}'."
            )
        for symbol in symbols_by_name[name]:
            subs[symbol] = sympified
    # Simultaneous substitution: the only way a requested name survives is that
    # the caller's own value reintroduced it (a legitimate shift like n -> n+1),
    # never a partially-applied replacement.
    substituted = estimate._map_expr(
        lambda expr: _substitute_resource_expr(expr, subs),
        constraint_fn=lambda expr: _safe_constraint_substitute(expr, subs),
        dependency_fn=lambda expr: _substitute_resource_expr(expr, subs),
    )
    if ignored:
        note = ResourceAssumption(
            "input(s) "
            + ", ".join(repr(name) for name in sorted(ignored))
            + " do not affect any resource metric; ignored",
        )
        substituted = substituted._with_metadata(assumptions=(note,))
    return substituted


def _validate_finite_source_domain_input(
    estimate: ResourceEstimate,
    symbols: Sequence[sp.Symbol],
    value: object,
) -> None:
    """Preserve source-operation diagnostics for finite-domain inputs.

    Unary mathematical operations retain their own finite-domain constraints.
    Validate those constraints before the generic public-scalar guard so an
    invalid concrete input reports the operation whose domain was violated.
    Other structural inputs still use the generic finite-real diagnostic.

    Args:
        estimate (ResourceEstimate): Estimate carrying source-domain
            constraints.
        symbols (Sequence[sp.Symbol]): Internal symbols represented by the
            public input name.
        value (object): Candidate input value.

    Raises:
        ValueError: If a concrete value violates a retained finite-domain
            constraint.
    """
    if isinstance(value, (str, bytes)):
        return
    scalar = value.item() if hasattr(value, "item") else value
    try:
        normalized = sp.sympify(scalar)
    except (TypeError, ValueError, sp.SympifyError):
        return
    if not isinstance(normalized, sp.Expr) or normalized.is_number is not True:
        return
    substitutions = {symbol: normalized for symbol in symbols}
    symbol_set = set(symbols)
    for constraint in estimate._constraints:
        if not constraint.finite or constraint.expression not in symbol_set:
            continue
        constraint.mapped(lambda expr: _safe_constraint_substitute(expr, substitutions))
