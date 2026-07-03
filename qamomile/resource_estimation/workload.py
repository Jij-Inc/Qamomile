"""Provide the declarative base for Hamiltonian algorithm workloads.

An algorithm workload pairs a problem-level Hamiltonian summary with
algorithm parameters and exposes derived quantities symbolically. This
module owns everything that is common to all such workloads — field
validation, precision-budget accounting, resource-value composition, and
serialization — so a new algorithm only declares its fields, its validation
kinds, and its derived expressions.

A minimal new workload looks like::

    from qamomile.resource_estimation._common import SympyLike, as_expr

    @dataclass(frozen=True)
    class MyWorkload(HamiltonianWorkloadMixin):
        hamiltonian: PauliHamiltonianResource
        my_rounds: SympyLike
        representation_error: SympyLike = 0
        description: str = ""

        _POSITIVE_FIELDS = ("my_rounds",)

        def _own_resource_values(self) -> dict[str, sp.Expr]:
            return {"my_rounds": as_expr(self.my_rounds, "my_rounds")}

Everything else (``__post_init__`` validation via
``_validate_workload_fields``, ``algorithmic_precision``,
``resource_values``, ``resource_values_for_precision``, ``to_dict``) comes
from the mixin.
"""

from __future__ import annotations

import enum
from dataclasses import fields
from typing import TYPE_CHECKING, Any, ClassVar

import sympy as sp

from qamomile.resource_estimation._common import (
    _as_expr,
    _SympyLike,
    _validate_nonnegative,
    _validate_positive,
)
from qamomile.resource_estimation.hamiltonian import PauliHamiltonianResource


class HamiltonianWorkloadMixin:
    """Share validation, precision, and serialization across workloads.

    Subclasses are frozen dataclasses that declare a ``hamiltonian`` field
    (a :class:`PauliHamiltonianResource`) and a ``representation_error``
    field, list their numeric fields in the class-level validation tuples,
    and implement :meth:`_own_resource_values`. The mixin itself declares no
    dataclass fields, so subclasses keep full control over field order and
    defaults.

    Attributes:
        _POSITIVE_FIELDS (ClassVar[tuple[str, ...]]): Names of fields that
            must be provably positive when decidable.
        _NONNEGATIVE_FIELDS (ClassVar[tuple[str, ...]]): Names of fields
            that must be provably nonnegative when decidable.
        _OPTIONAL_POSITIVE_FIELDS (ClassVar[tuple[str, ...]]): Names of
            fields that must be positive when not None.

    Example:
        >>> from qamomile.resource_estimation import HamiltonianQPEWorkload
        >>> issubclass(HamiltonianQPEWorkload, HamiltonianWorkloadMixin)
        True
    """

    _POSITIVE_FIELDS: ClassVar[tuple[str, ...]] = ()
    _NONNEGATIVE_FIELDS: ClassVar[tuple[str, ...]] = ()
    _OPTIONAL_POSITIVE_FIELDS: ClassVar[tuple[str, ...]] = ()

    if TYPE_CHECKING:
        # Declared by the dataclass subclasses; hidden from the dataclass
        # machinery at runtime so the mixin contributes no fields.
        hamiltonian: PauliHamiltonianResource
        representation_error: _SympyLike

    def __post_init__(self) -> None:
        """Validate declared workload fields after dataclass construction.

        Raises:
            TypeError: If ``hamiltonian`` is not a
                ``PauliHamiltonianResource`` or a declared field cannot be
                converted to a SymPy expression.
            ValueError: If a declared positive field is provably non-positive
                or a declared nonnegative field is provably negative.
        """
        self._validate_workload_fields()

    def _validate_workload_fields(self) -> None:
        """Validate the Hamiltonian and the declared numeric fields.

        Subclasses that override ``__post_init__`` for extra checks should
        call this first.

        Raises:
            TypeError: If ``hamiltonian`` is not a
                ``PauliHamiltonianResource`` or a declared field cannot be
                converted to a SymPy expression.
            ValueError: If a declared positive field is provably non-positive
                or a declared nonnegative field is provably negative.
        """
        if not hasattr(self, "hamiltonian") or not isinstance(
            self.hamiltonian, PauliHamiltonianResource
        ):
            raise TypeError("hamiltonian must be a PauliHamiltonianResource.")
        if not hasattr(self, "representation_error"):
            raise TypeError(
                f"{type(self).__name__} must declare a representation_error "
                "field (use a default of 0 when the algorithm has no "
                "representation-error budget)."
            )
        for name in self._POSITIVE_FIELDS:
            _validate_positive(_as_expr(getattr(self, name), name), name)
        for name in self._OPTIONAL_POSITIVE_FIELDS:
            value = getattr(self, name)
            if value is not None:
                _validate_positive(_as_expr(value, name), name)
        for name in self._NONNEGATIVE_FIELDS:
            _validate_nonnegative(_as_expr(getattr(self, name), name), name)

    def algorithmic_precision(self, precision: _SympyLike) -> sp.Expr:
        """Return precision remaining after representation error.

        Args:
            precision (sp.Expr | int | float): Total target energy precision
                budget.

        Returns:
            sp.Expr: Precision budget available to phase estimation after
                subtracting ``representation_error``.

        Raises:
            ValueError: If ``precision`` is non-positive or if the remaining
                algorithmic precision is provably non-positive.
            TypeError: If ``precision`` cannot be converted into a SymPy
                expression.
        """
        precision_expr = _as_expr(precision, "precision")
        _validate_positive(precision_expr, "precision")
        remaining = sp.simplify(
            precision_expr - _as_expr(self.representation_error, "representation_error")
        )
        _validate_positive(remaining, "algorithmic_precision")
        return remaining

    def _own_resource_values(self) -> dict[str, sp.Expr]:
        """Return the algorithm-specific resource values of the workload.

        Returns:
            dict[str, sp.Expr]: Algorithm-level values keyed by quantity
                names, merged over the Hamiltonian summary values by
                :meth:`resource_values`.

        Raises:
            NotImplementedError: If the subclass does not implement it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _own_resource_values()."
        )

    def resource_values(self) -> dict[str, sp.Expr]:
        """Return canonical resource values exposed by the workload.

        Returns:
            dict[str, sp.Expr]: Hamiltonian summary values plus the
                algorithm-specific values from :meth:`_own_resource_values`
                (algorithm values win on key collision).
        """
        values = self.hamiltonian.resource_values()
        values.update(self._own_resource_values())
        return values

    def resource_values_for_precision(
        self,
        precision: _SympyLike,
    ) -> dict[str, sp.Expr]:
        """Return canonical resource values for one target precision.

        Args:
            precision (sp.Expr | int | float): Total target energy precision
                budget used to derive the algorithmic precision available to
                phase estimation.

        Returns:
            dict[str, sp.Expr]: Workload resource values plus
                ``target_precision`` and ``algorithmic_precision``.

        Raises:
            ValueError: If ``precision`` is non-positive or if
                ``representation_error`` leaves no positive precision for
                phase estimation.
            TypeError: If ``precision`` cannot be converted into a SymPy
                expression.
        """
        precision_expr = _as_expr(precision, "precision")
        _validate_positive(precision_expr, "precision")
        values = self.resource_values()
        values["target_precision"] = precision_expr
        values["algorithmic_precision"] = self.algorithmic_precision(precision_expr)
        return values

    def _dict_overrides(self) -> dict[str, Any]:
        """Return per-field serialization overrides for :meth:`to_dict`.

        Returns:
            dict[str, Any]: Field-name-to-serialized-value entries that
                replace the generic formatting. Defaults to no overrides.
        """
        return {}

    def to_dict(self) -> dict[str, Any]:
        """Serialize the workload to a JSON-friendly dictionary.

        Fields serialize in declaration order: the Hamiltonian summary nests
        via its own ``to_dict``, strings pass through, None stays None,
        enums serialize to their values, and numeric or symbolic fields
        serialize as strings. Subclasses adjust individual entries via
        :meth:`_dict_overrides`.

        Returns:
            dict[str, Any]: String-valued workload metadata.

        Raises:
            TypeError: If a numeric field cannot be converted to a SymPy
                expression.
        """
        overrides = self._dict_overrides()
        serialized: dict[str, Any] = {}
        for field_info in fields(self):  # type: ignore[arg-type]
            name = field_info.name
            if name in overrides:
                serialized[name] = overrides[name]
                continue
            value = getattr(self, name)
            if isinstance(value, PauliHamiltonianResource):
                serialized[name] = value.to_dict()
            elif value is None or isinstance(value, bool):
                serialized[name] = value
            elif isinstance(value, enum.Enum):
                serialized[name] = value.value
            elif isinstance(value, str):
                serialized[name] = value
            else:
                serialized[name] = str(_as_expr(value, name))
        return serialized
