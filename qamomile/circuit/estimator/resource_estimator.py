"""Expose the public resource-estimation API."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import sympy as sp

from qamomile.circuit.estimator._input_contract import (
    _contract_names,
    _expand_array_shape_inputs,
    _root_callable_resource_attrs,
    _root_callable_shape_inputs,
    _scalar_input_types,
)
from qamomile.circuit.estimator._resolver import (
    ExprResolver,
)
from qamomile.circuit.estimator._resource_base import (
    ApproximationStatus,
    ControlDecomposition,
    EstimateDerivation,
    EstimateQuality,
)
from qamomile.circuit.estimator._resource_types import (
    CallResources,
    DepthResources,
    GateResources,
    MeasurementResources,
    ResetResources,
    ResourceAssumption,
    ResourceTraceNode,
    WidthResources,
)
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.operation import (
    Operation,
)
from qamomile.circuit.ir.value import ArrayValue, Value, ValueBase

if TYPE_CHECKING:
    from qamomile.circuit.frontend.qkernel import QKernel

from qamomile.circuit.estimator._classical_provenance import (
    _LoopMayTaint as _LoopMayTaint,
)
from qamomile.circuit.estimator._config import (
    _DEFAULT_CONTROL_DECOMPOSITION,
    UnknownResourcePolicy,
    _ResourceEstimatorConfig,
)
from qamomile.circuit.estimator._constraints import (
    _quantum_operand_width_constraints,
)
from qamomile.circuit.estimator._control_model import (
    _EstimatorControlBatchProfile as _EstimatorControlBatchProfile,
)
from qamomile.circuit.estimator._estimate import ResourceEstimate
from qamomile.circuit.estimator._estimate_composition import (
    _SequentialEstimateComposer as _SequentialEstimateComposer,
)
from qamomile.circuit.estimator._estimate_domain import _DomainRewritePolicy
from qamomile.circuit.estimator._estimate_provenance import (
    _DEFER_RESOURCE_SYMBOL_METADATA,
)
from qamomile.circuit.estimator._estimate_validation import _with_constraints
from qamomile.circuit.estimator._inputs import (
    _apply_inputs,
    _estimator_parameters,
    _partition_estimation_inputs,
    _root_input_binding_context,
    _scalar_values,
    _substitute_bindings,
    _validate_explicit_estimation_inputs,
)
from qamomile.circuit.estimator._interpreter import ResourceInterpreter
from qamomile.circuit.estimator._interpreter_dataflow import (
    _ResourceInlineBoundaryOperation as _ResourceInlineBoundaryOperation,
)
from qamomile.circuit.estimator._opaque import (
    OpaqueCostContext,
    _OpaqueInvocationTransform as _OpaqueInvocationTransform,
)
from qamomile.circuit.estimator._product_formula import (
    _apply_product_formula_contract,
    _require_concrete_product_formula_structure,
)
from qamomile.circuit.estimator._resource_constraints import (
    _mark_root_domain_constraints,
)
from qamomile.circuit.estimator._symbolic import (
    _CappedRangeSum as _CappedRangeSum,
)

__all__ = [
    "ApproximationStatus",
    "CallResources",
    "ControlDecomposition",
    "DepthResources",
    "EstimateDerivation",
    "EstimateQuality",
    "GateResources",
    "OpaqueCostContext",
    "MeasurementResources",
    "ResetResources",
    "ResourceAssumption",
    "ResourceEstimate",
    "ResourceEstimator",
    "ResourceInterpreter",
    "ResourceTraceNode",
    "UnknownResourcePolicy",
    "WidthResources",
    "estimate_resources",
]


def _root_formal_resource_symbols(
    block: Block,
    resolver: ExprResolver,
) -> dict[sp.Symbol, str]:
    """Resolve exact root-formal symbol identities and stable input names.

    Args:
        block (Block): Root qkernel block whose interface is authoritative.
        resolver (ExprResolver): Resolver configured for the root interface.

    Returns:
        dict[sp.Symbol, str]: Exact symbolic identities mapped to formal names.
    """
    slot_names = tuple(dict.fromkeys(slot.name for slot in block.param_slots))
    values_by_name: dict[str, ValueBase] = {
        value.name: value
        for value in (*block.input_values, *block.parameters.values())
        if isinstance(value, (Value, ArrayValue)) and value.name
    }
    symbols: dict[sp.Symbol, str] = {}
    for name in slot_names:
        value = values_by_name.get(name)
        if isinstance(value, Value):
            expression = resolver.resolve(value)
            if isinstance(expression, sp.Symbol):
                symbols[expression] = name
    input_names = (
        {
            value.uuid: name
            for name, value in zip(block.label_args, block.input_values, strict=True)
        }
        if len(block.label_args) == len(block.input_values)
        else {}
    )
    for value in block.input_values:
        if not isinstance(value, ArrayValue):
            continue
        formal_name = input_names.get(value.uuid) or value.name or "input"
        for dimension in value.shape:
            expression = resolver.resolve(dimension)
            for symbol in cast(set[sp.Symbol], expression.free_symbols):
                symbols[symbol] = formal_name
    return symbols


class ResourceEstimator:
    """Estimate algorithmic resources for qkernels and IR blocks.

    Args:
        strategies (dict[str, str] | None): Strategy overrides by callable
            name. Defaults to ``None``.
        trace (bool): Whether to keep explanation traces. Defaults to
            ``False``.
        simplify (bool): Whether to simplify final expressions, including
            simplification over valid qkernel input conditions. Defaults to
            ``True``.
        unknown_policy (str | UnknownResourcePolicy): Handling for unknown
            bodyless callables. Defaults to ``ERROR``.
        control_decomposition (str | ControlDecomposition): Coherent-control
            decomposition. Defaults to ``CLEAN_ANCILLA_TOFFOLI``.

    Raises:
        ValueError: If ``unknown_policy`` or ``control_decomposition`` is
            unknown.
    """

    def __init__(
        self,
        *,
        strategies: dict[str, str] | None = None,
        trace: bool = False,
        simplify: bool = True,
        unknown_policy: str | UnknownResourcePolicy = UnknownResourcePolicy.ERROR,
        control_decomposition: str
        | ControlDecomposition = _DEFAULT_CONTROL_DECOMPOSITION,
    ) -> None:
        """Initialize a resource estimator.

        Args:
            strategies (dict[str, str] | None): Strategy overrides by callable
                name. Defaults to ``None``.
            trace (bool): Whether to keep explanation traces. Defaults to
                ``False``.
            simplify (bool): Whether to simplify the final estimate, including
                simplification under valid qkernel input conditions. Consumed
                conditions remain visible in ``ResourceEstimate.assumptions``.
                Set to ``False`` to preserve the unconditional symbolic
                formulas; calling ``ResourceEstimate.simplify()`` later
                explicitly enables the domain-aware pass. Defaults to ``True``.
            unknown_policy (str | UnknownResourcePolicy): Handling for unknown
                bodyless callables. Defaults to ``ERROR``.
            control_decomposition (str | ControlDecomposition):
                Coherent-control decomposition. Defaults to
                ``CLEAN_ANCILLA_TOFFOLI``.

        Raises:
            ValueError: If ``unknown_policy`` or ``control_decomposition`` is
                unknown.
        """
        try:
            normalized_unknown_policy = UnknownResourcePolicy(unknown_policy)
        except ValueError as error:
            valid = ", ".join(member.value for member in UnknownResourcePolicy)
            raise ValueError(
                f"unknown resource policy {unknown_policy!r}; expected one of: {valid}"
            ) from error
        try:
            normalized_control_decomposition = ControlDecomposition(
                control_decomposition
            )
        except ValueError as error:
            valid = ", ".join(member.value for member in ControlDecomposition)
            raise ValueError(
                "unknown control decomposition "
                f"{control_decomposition!r}; expected one of: {valid}"
            ) from error
        self.config = _ResourceEstimatorConfig(
            strategies=dict(strategies or {}),
            trace=trace,
            simplify=simplify,
            unknown_policy=normalized_unknown_policy,
            control_decomposition=normalized_control_decomposition,
        )

    def estimate(
        self,
        kernel: "QKernel[Any, Any] | Block | Sequence[Operation]",
        *,
        inputs: dict[str, Any] | None = None,
        strategies: dict[str, str] | None = None,
    ) -> ResourceEstimate:
        """Estimate algorithmic resources for a qkernel, block, or operations.

        Args:
            kernel (QKernel[Any, Any] | Block | Sequence[Operation]): Object to
                estimate. QKernel-like objects are built before traversal.
            inputs (dict[str, Any] | None): QKernel input values used to
                specialize the symbolic estimate without constructing a
                problem-sized circuit. Exact one-dimensional root quantum-port
                widths declared by callable resource metadata are inferred
                when omitted. Defaults to ``None``.
            strategies (dict[str, str] | None): Per-call override merged over
                estimator-level strategies. Defaults to ``None``.

        Returns:
            ResourceEstimate: Algorithmic resource estimate.

        Raises:
            RuntimeError: If a fixed or callback-provided opaque cost contains
                public metrics or metadata that disagree with retained
                canonical provenance.
            ValueError: If an input name is unknown, a callable resource
                contract is malformed or violated, or a structural resource
                requirement fails.
            TypeError: If ``kernel`` is not a supported estimator input.
            NotImplementedError: If the input IR contains a construct not
                supported by resource estimation.
        """
        defer_token = _DEFER_RESOURCE_SYMBOL_METADATA.set(True)
        try:
            estimate = self._estimate_deferred(
                kernel,
                inputs=inputs,
                strategies=strategies,
            )
        finally:
            _DEFER_RESOURCE_SYMBOL_METADATA.reset(defer_token)
        estimate._refresh_symbol_metadata()
        return estimate

    def _estimate_deferred(
        self,
        kernel: "QKernel[Any, Any] | Block | Sequence[Operation]",
        *,
        inputs: dict[str, Any] | None = None,
        strategies: dict[str, str] | None = None,
    ) -> ResourceEstimate:
        """Estimate resources while deferring public symbol derivation.

        The public :meth:`estimate` wrapper activates the deferral context and
        refreshes aliases and parameters exactly once on the final result.

        Args:
            kernel (QKernel[Any, Any] | Block | Sequence[Operation]): Object to
                estimate. QKernel-like objects are built before traversal.
            inputs (dict[str, Any] | None): Values used to specialize symbolic
                qkernel inputs. Defaults to ``None``.
            strategies (dict[str, str] | None): Per-call strategy overrides.
                Defaults to ``None``.

        Returns:
            ResourceEstimate: Estimate awaiting one final public-symbol
            metadata refresh.

        Raises:
            ValueError: If an input, callable resource contract, or structural
                requirement is invalid.
            TypeError: If ``kernel`` is not a supported estimator input.
            NotImplementedError: If the input contains an unsupported
                construct.
        """
        explicit_inputs = _validate_explicit_estimation_inputs(
            kernel,
            inputs or {},
        )
        root_callable_attrs = _root_callable_resource_attrs(kernel)
        build_inputs, estimation_inputs = _partition_estimation_inputs(
            kernel,
            explicit_inputs,
        )
        block_or_ops = self._coerce_input(
            kernel,
            build_inputs,
            estimation_inputs,
        )
        config = dataclasses.replace(
            self.config,
            strategies={**self.config.strategies, **dict(strategies or {})},
        )
        condition_values = _scalar_values({**build_inputs, **estimation_inputs})
        interpreter = ResourceInterpreter(
            config=config,
            bindings=build_inputs,
            condition_values=condition_values,
        )
        if isinstance(block_or_ops, Block):
            _require_concrete_product_formula_structure(
                root_callable_attrs,
                block_or_ops.input_values,
                bindings={**build_inputs, **condition_values},
                resolver=ExprResolver(
                    block=block_or_ops,
                    context=_root_input_binding_context(
                        block_or_ops,
                        build_inputs,
                    ),
                ),
                specialize=lambda expression: interpreter._apply_condition_values(
                    expression,
                    record_usage=False,
                ),
                source=getattr(kernel, "name", None) or block_or_ops.name or "qkernel",
            )
        estimate = interpreter.estimate(block_or_ops)
        # Boundary liveness is interpreter-local metadata. A root estimate can
        # later be reused as an opaque definition cost, where caller owner
        # identities would be meaningless and unserializable. Drop it before
        # public symbol discovery so private size expressions cannot introduce
        # public parameters either.
        estimate = dataclasses.replace(
            estimate,
            _output_sizes={},
            _input_sizes={},
            _has_output_summary=False,
        )
        root_source = getattr(kernel, "name", None) or (
            block_or_ops.name if isinstance(block_or_ops, Block) else "qkernel"
        )
        if isinstance(block_or_ops, Block):
            root_resolver = ExprResolver(
                block=block_or_ops,
                context=_root_input_binding_context(
                    block_or_ops,
                    build_inputs,
                ),
            )
            estimate = _apply_product_formula_contract(
                estimate,
                root_callable_attrs,
                block_or_ops.input_values,
                root_resolver,
                bindings=build_inputs,
                specialize=lambda expression: interpreter._apply_condition_values(
                    expression,
                    record_usage=False,
                ),
                source=root_source,
            )
            estimate = _with_constraints(
                estimate,
                *_quantum_operand_width_constraints(
                    root_callable_attrs,
                    block_or_ops.input_values,
                    root_resolver,
                    source=root_source,
                ),
            )
        # Runtime-domain alternatives may contain ordinary qkernel symbols.
        # Expand them before applying user inputs so the normal substitution
        # and constraint validation path specializes every alternative rather
        # than reintroducing an already supplied symbol afterward.
        estimate = interpreter.resolve_finite_runtime_constraints(estimate)
        if isinstance(block_or_ops, Block):
            formal_resolver = ExprResolver(
                block=block_or_ops,
                context=_root_input_binding_context(
                    block_or_ops,
                    build_inputs,
                ),
            )
            formal_symbols = _root_formal_resource_symbols(
                block_or_ops,
                formal_resolver,
            )
            estimate = dataclasses.replace(
                estimate,
                _constraints=_mark_root_domain_constraints(
                    estimate._constraints,
                    formal_symbols,
                ),
            )
        if build_inputs:
            estimate._refresh_symbol_metadata()
            estimate = _substitute_bindings(estimate, build_inputs)
        inferred_shape_inputs = _root_callable_shape_inputs(
            root_callable_attrs,
            block_or_ops,
            explicit_inputs,
            source=root_source,
        )
        effective_estimation_inputs = {
            **inferred_shape_inputs,
            **estimation_inputs,
        }
        expanded_estimation_inputs, shape_input_names = _expand_array_shape_inputs(
            block_or_ops,
            effective_estimation_inputs,
        )
        if effective_estimation_inputs:
            estimate = _apply_inputs(
                estimate,
                expanded_estimation_inputs,
                contract_names=_contract_names(block_or_ops),
                input_types=_scalar_input_types(block_or_ops),
                branch_condition_names=interpreter.branch_condition_names,
                consumed_input_names=shape_input_names,
            )
        if not config.trace:
            # Large recursive call bodies can produce an explanation tree
            # deeper than Python's recursion limit.  A disabled trace is not
            # observable, so discard it before symbolic mapping/simplification.
            estimate = dataclasses.replace(estimate, trace=None)
        interpreter.validate_no_internal_resource_symbols(estimate)
        if config.simplify:
            estimate = estimate.simplify()
        else:
            estimate = dataclasses.replace(
                estimate,
                _domain_rewrite_policy=_DomainRewritePolicy.DISABLED,
            )
        estimate = dataclasses.replace(
            estimate,
            control_decomposition=config.control_decomposition,
        )
        interpreter.validate_no_internal_resource_symbols(estimate)
        return estimate

    def _coerce_input(
        self,
        kernel: "QKernel[Any, Any] | Block | Sequence[Operation]",
        build_inputs: dict[str, Any],
        estimation_inputs: dict[str, Any],
    ) -> Block | Sequence[Operation]:
        """Coerce a supported input into an IR block or operation list.

        Parameterizable inputs stay symbolic while non-parameterizable inputs
        are supplied during tracing. This keeps problem sizes scalable without
        requiring users to distinguish build-time from estimation-time inputs.

        Args:
            kernel (QKernel[Any, Any] | Block | Sequence[Operation]): Input
                object.
            build_inputs (dict[str, Any]): Structural values supplied while
                tracing the qkernel.
            estimation_inputs (dict[str, Any]): Parameterizable values kept
                symbolic until after interpretation.

        Returns:
            Block | Sequence[Operation]: IR object ready for interpretation.

        Raises:
            TypeError: If a concrete build-time input cannot be converted to
                its declared qkernel type.
            ValueError: If a concrete build-time input violates its declared
                domain or structural contract.
        """
        if isinstance(kernel, Block):
            return kernel
        if isinstance(kernel, Sequence):
            return kernel
        build = getattr(kernel, "build", None)
        if callable(build):
            parameters = _estimator_parameters(kernel, build_inputs)
            try:
                return build(parameters=parameters, **build_inputs)
            except (TypeError, ValueError) as error:
                names = ", ".join(repr(name) for name in sorted(build_inputs))
                if not names:
                    raise
                raise type(error)(
                    f"resource estimation could not bind input(s) {names}: {error}"
                ) from error
        block = getattr(kernel, "block", None)
        if isinstance(block, Block):
            return block
        raise TypeError(
            "ResourceEstimator.estimate() expects a QKernel, Block, or "
            "sequence of Operation objects."
        )


def estimate_resources(
    kernel: "QKernel[Any, Any] | Block | Sequence[Operation]",
    *,
    inputs: dict[str, Any] | None = None,
    strategies: dict[str, str] | None = None,
    trace: bool = False,
    unknown_policy: str | UnknownResourcePolicy = UnknownResourcePolicy.ERROR,
    control_decomposition: str | ControlDecomposition = _DEFAULT_CONTROL_DECOMPOSITION,
) -> ResourceEstimate:
    """Estimate algorithmic resources using the default estimator facade.

    Args:
        kernel (QKernel[Any, Any] | Block | Sequence[Operation]): QKernel,
            block, or operation sequence to estimate.
        inputs (dict[str, Any] | None): QKernel input values used to specialize
            the symbolic estimate without building a problem-sized circuit.
            Exact one-dimensional root quantum-port widths declared by
            callable resource metadata are inferred when omitted. Defaults to
            ``None``.
        strategies (dict[str, str] | None): Strategy overrides by callable
            name. Defaults to ``None``.
        trace (bool): Whether to retain the explanation tree. Defaults to
            ``False``.
        unknown_policy (str | UnknownResourcePolicy): Unknown callable
            handling. Defaults to ``ERROR``.
        control_decomposition (str | ControlDecomposition): Coherent-control
            decomposition. Defaults to ``CLEAN_ANCILLA_TOFFOLI``.

    Returns:
        ResourceEstimate: Algorithmic resource estimate.

    Raises:
        RuntimeError: If a fixed or callback-provided opaque cost contains
            public metrics or metadata that disagree with retained canonical
            provenance.
        ValueError: If the input specialization, estimator configuration,
            callable resource contract, or structural requirements are invalid.
        TypeError: If ``kernel`` is not a supported estimator input.
        NotImplementedError: If the input IR contains a construct not
            supported by resource estimation.

    Example:
        >>> import qamomile.circuit as qmc
        >>> @qmc.qkernel
        ... def repeated_h(n: qmc.UInt) -> qmc.Qubit:
        ...     q = qmc.qubit("q")
        ...     for _ in qmc.range(n):
        ...         q = qmc.h(q)
        ...     return q
        >>> symbolic = estimate_resources(repeated_h)
        >>> str(symbolic.gates.total)
        'n'
        >>> estimate_resources(repeated_h, inputs={"n": 8}).gates.total
        8
    """
    estimator = ResourceEstimator(
        strategies=strategies,
        trace=trace,
        unknown_policy=unknown_policy,
        control_decomposition=control_decomposition,
    )
    return estimator.estimate(
        kernel,
        inputs=inputs,
    )
