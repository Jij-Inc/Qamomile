"""Describe and validate opaque callable resource costs."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any

import sympy as sp

from qamomile.circuit.estimator._config import (
    _DEFAULT_CONTROL_DECOMPOSITION,
    _DEFAULT_GATE_BASIS,
)
from qamomile.circuit.estimator._constants import (
    _ONE,
    _ZERO,
)
from qamomile.circuit.estimator._estimate import ResourceEstimate
from qamomile.circuit.estimator._estimate_provenance import (
    _estimate_has_basis_sensitive_resources,
    _precisions_match,
)
from qamomile.circuit.estimator._resource_base import (
    ControlDecomposition,
    GateBasis,
    ResourceExpr,
)
from qamomile.circuit.estimator._resource_expressions import (
    _expr,
    _safe_simplify,
)
from qamomile.circuit.estimator._resource_types import (
    WidthResources,
)
from qamomile.circuit.estimator._symbol_discovery import _serialization_expressions


def _estimate_uses_unresolved_functions(
    estimate: ResourceEstimate,
    functions: Iterable[Any],
) -> bool:
    """Return whether resource algebra retains an unresolved carry function.

    Comparing one concrete function application is insufficient after loop
    summation because SymPy can rewrite ``carry(k)`` to ``carry(0)`` or another
    application of the same undefined function. Matching the function identity
    catches every such rewrite without confusing independent same-name carries.

    Args:
        estimate (ResourceEstimate): Estimate whose serialized algebra is
            inspected.
        functions (Iterable[Any]): Identity-distinct undefined SymPy functions
            created for unsupported loop-carried recurrences.

    Returns:
        bool: Whether any resource or guarded metadata expression contains an
            application of one of the supplied functions.
    """
    unresolved = tuple(functions)
    return bool(unresolved) and any(
        expression.has(*unresolved)
        for expression in _serialization_expressions(estimate)
        if isinstance(expression, sp.Basic)
    )


def _validate_opaque_cost_provenance(
    estimate: ResourceEstimate,
    *,
    name: str,
    basis: GateBasis,
    control_decomposition: ControlDecomposition,
    precision: float,
) -> None:
    """Validate a fixed or callback-produced opaque cost model.

    Args:
        estimate (ResourceEstimate): Opaque cost result to validate.
        name (str): User-facing callable name for diagnostics.
        basis (GateBasis): Basis requested by the active estimator.
        control_decomposition (ControlDecomposition): Coherent-control
            decomposition requested by the active estimator.
        precision (float): Clifford+T synthesis precision requested by the
            active estimator.

    Raises:
        ValueError: If basis-sensitive cost metrics use incompatible basis or
            precision provenance.
    """
    if not _estimate_has_basis_sensitive_resources(estimate):
        return
    if estimate.basis is not basis:
        raise ValueError(
            f"Opaque cost for '{name}' uses basis {estimate.basis.value!r}, "
            f"but the estimator uses {basis.value!r}."
        )
    if estimate.control_decomposition is not control_decomposition:
        raise ValueError(
            f"Opaque cost for '{name}' uses control decomposition "
            f"{estimate.control_decomposition.value!r}, but the estimator "
            f"uses {control_decomposition.value!r}."
        )
    if basis is GateBasis.CLIFFORD_T and not _precisions_match(
        estimate.precision,
        precision,
    ):
        raise ValueError(
            f"Opaque cost for '{name}' uses precision "
            f"{estimate.precision!r}, but the estimator uses {precision!r}."
        )


@dataclasses.dataclass(frozen=True, slots=True)
class OpaqueCostContext:
    """Describe the base Oracle definition requested from a cost callback.

    The callback models one ordinary application of the definition. Its
    result therefore includes ``definition_control_qubits`` but never controls
    added by a later ``qmc.control`` call or inherited from an enclosing
    controlled qkernel. The estimator applies those call-site transforms after
    the callback returns. The callback result is a complete definition-level
    contract and must include any phase-relevant work that those later
    coherent controls need to transform.

    Args:
        callable_name (str): Human-readable callable name.
        target_shapes (Mapping[str, tuple[ResourceExpr, ...]]): Definition
            target shapes keyed by formal operand name. A scalar qubit has an
            empty shape tuple.
        definition_control_qubits (int): Controls declared by the Oracle
            definition and already included in the callback's base cost.
        strategy (str | None): Selected base resource strategy. Defaults to
            ``None``.
        basis (GateBasis): Requested output gate basis. Defaults to
            ``LOGICAL``.
        control_decomposition (ControlDecomposition): Requested coherent
            control model. Defaults to ``CLEAN_ANCILLA_TOFFOLI``.
        precision (float | None): Clifford+T synthesis precision, or ``None``
            for other bases. Defaults to ``None``.
    """

    callable_name: str
    target_shapes: Mapping[str, tuple[ResourceExpr, ...]]
    definition_control_qubits: int
    strategy: str | None = None
    basis: GateBasis = _DEFAULT_GATE_BASIS
    control_decomposition: ControlDecomposition = _DEFAULT_CONTROL_DECOMPOSITION
    precision: float | None = None

    def __post_init__(self) -> None:
        """Validate and freeze definition-level callback inputs.

        Raises:
            TypeError: If the callable name, target names, target shapes, or
                definition control count have invalid Python types.
            ValueError: If the definition control count is negative.
        """
        if not isinstance(self.callable_name, str):
            raise TypeError("callable_name must be a string.")
        if isinstance(self.definition_control_qubits, bool) or not isinstance(
            self.definition_control_qubits, int
        ):
            raise TypeError("definition_control_qubits must be a plain Python int.")
        if self.definition_control_qubits < 0:
            raise ValueError("definition_control_qubits must be nonnegative.")
        normalized_shapes: dict[str, tuple[ResourceExpr, ...]] = {}
        for name, shape in self.target_shapes.items():
            if not isinstance(name, str):
                raise TypeError("target shape names must be strings.")
            if not isinstance(shape, tuple):
                raise TypeError(f"target shape for {name!r} must be a tuple.")
            normalized_shapes[name] = tuple(_expr(dimension) for dimension in shape)
        object.__setattr__(
            self,
            "target_shapes",
            MappingProxyType(normalized_shapes),
        )

    @property
    def target_qubits(self) -> ResourceExpr:
        """Return the flattened width of all definition targets.

        Returns:
            ResourceExpr: Sum of scalar targets and products of array
            dimensions.
        """
        return _safe_simplify(
            _expr(
                sum(
                    (
                        sp.prod(shape) if shape else _ONE
                        for shape in self.target_shapes.values()
                    ),
                    _ZERO,
                )
            )
        )


def _opaque_call_relative_width(
    width: WidthResources,
) -> tuple[WidthResources, ResourceExpr]:
    """Remove a standalone estimate's caller-owned width baseline.

    Explicit opaque costs may be copied from a root qkernel estimate, whose
    ``peak_qubits`` includes ``input_qubits``. At an Invoke boundary those
    operands are already live in the caller, so only the peak above that
    baseline belongs to the call body. Any residual peak not categorized as
    allocation or clean/dirty ancilla is retained as anonymous allocation. A
    partial declaration that omits ``peak_qubits`` is normalized so its peak
    cannot be smaller than its declared allocation and ancilla workspace.

    Args:
        width (WidthResources): Definition-level standalone width.

    Returns:
        tuple[WidthResources, ResourceExpr]: Body-relative width for one opaque
            invocation and the anonymous workspace added to its allocation
            count.
    """
    categorized_workspace = (
        width.allocated_qubits + width.clean_ancilla_qubits + width.dirty_ancilla_qubits
    )
    relative_peak = _safe_simplify(
        sp.Max(
            _ZERO,
            width.peak_qubits - width.input_qubits,
            categorized_workspace,
        )
    )
    anonymous_workspace = _safe_simplify(
        sp.Max(_ZERO, relative_peak - categorized_workspace)
    )
    relative_width = dataclasses.replace(
        width,
        input_qubits=_ZERO,
        allocated_qubits=width.allocated_qubits + anonymous_workspace,
        peak_qubits=relative_peak,
    )
    return relative_width, anonymous_workspace


@dataclasses.dataclass(frozen=True, slots=True)
class _OpaqueInvocationTransform:
    """Carry call-site transforms kept out of ``OpaqueCostContext``.

    Args:
        declared_controls (int): Controls represented by the base definition.
        added_controls (int): Controls prepended by a frontend transform.
        inherited_controls (ResourceExpr): Controls inherited from enclosing
            controlled bodies.
        inverse (bool): Whether to invert the base definition.
        control_value (int | None): Combined LSB-first activation value for
            added then declared local controls.
    """

    declared_controls: int
    added_controls: int
    inherited_controls: ResourceExpr
    inverse: bool
    control_value: int | None

    @property
    def external_controls(self) -> ResourceExpr:
        """Return controls the estimator applies to the base cost.

        Returns:
            ResourceExpr: Added plus inherited controls.
        """
        return self.inherited_controls + self.added_controls

    @property
    def local_controls(self) -> int:
        """Return the controls normalized by this Oracle invocation.

        Returns:
            int: Added plus definition-declared controls.
        """
        return self.added_controls + self.declared_controls

    @property
    def zero_controls(self) -> int:
        """Return zero-valued controls in the combined local condition.

        Returns:
            int: Number of local controls activated by zero.
        """
        return int(
            _zero_control_count(
                self.local_controls,
                self.control_value,
            )
        )


def _zero_control_count(
    num_controls: int,
    control_value: int | None,
) -> int:
    """Return the number of zero bits in a concrete control pattern.

    Args:
        num_controls (int): Width of the control register.
        control_value (int | None): LSB-first activation value, or ``None``
            for the ordinary all-ones pattern.

    Returns:
        int: Controls that require X bracketing.
    """
    if control_value is None:
        return 0
    return num_controls - control_value.bit_count()
