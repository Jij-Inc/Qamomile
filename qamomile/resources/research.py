"""Notebook views and explicit portable models over the native resource visitor.

Views never change source facts. Models add circuit coordinates; their recipes
are implemented and counted in Rust. No Python QCore or decomposition walker
lives here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from html import escape
from typing import Mapping, cast

from qamomile._native import compiler as native
from qamomile.errors import CapabilityError
from qamomile.ir import Module, Program
from qamomile.lowering import RealizationOptions

from .expr import Number
from .metrics import (
    CIRCUIT_OUTPUT_TRACE_DISTANCE_BOUND,
    MetricDefinition,
    algorithmic_call_count,
)
from .report import ResourceStudy
from .values import EvidenceAuthority, ResourceValue, _verified_evidence

_FRONTIER = "qamomile.algorithmic.frontier."
_PORTABLE = "qamomile.circuit.portable."


class GateBasis(str, Enum):
    """Select an exact alphabet with arbitrary rotations left unsynthesized."""

    CLIFFORD_TOFFOLI_ROTATIONS = "clifford_toffoli_rotations"
    CLIFFORD_T_ROTATIONS = "clifford_t_rotations"


def _selection_names(
    study: ResourceStudy, expand: tuple[object, ...] | None
) -> tuple[str, ...] | None:
    """Resolve display selections through the existing compiler name boundary.

    Args:
        study (ResourceStudy): Study retaining authoritative source.
        expand (tuple[object, ...] | None): Callable objects/names, or all bodies.

    Returns:
        tuple[str, ...] | None: Canonical qualified symbols, or all bodies.

    Raises:
        TypeError: If a selection is not a callable identity.
        ValueError: If the compiler rejects an absent or ambiguous name.
    """

    if expand is None:
        return None
    if not expand:
        return ()
    if isinstance(cast(object, expand), (str, bytes)) or len(expand) > 4096:
        raise ValueError("expand must be a collection of at most 4096 callables")
    source = study._semantic_source()
    module = (
        Module(source.module_name, source.functions, source.constants)
        if isinstance(source, Program)
        else source
    )
    names = []
    for item in expand:
        name = item if isinstance(item, str) else getattr(item, "symbol", None)
        if not isinstance(name, str):
            raise TypeError("expand accepts callable objects or names")
        names.append(module.function(name).symbol)
    if len(names) != len(set(names)):
        raise ValueError("duplicate expanded callable")
    return tuple(sorted(names))


def _observe(
    study: ResourceStudy,
    expanded: tuple[str, ...] | None,
    basis: GateBasis | None = None,
    realizations: RealizationOptions = RealizationOptions(),
) -> dict[str, ResourceValue]:
    """Project native observation facts without replacing the study's metrics.

    Args:
        study (ResourceStudy): Source study, possibly carrying external rules.
        expanded (tuple[str, ...] | None): Canonical expanded callees, or all.
        basis (GateBasis | None): Circuit alphabet, or observation only.
        realizations (RealizationOptions): Implementations used by projection.

    Returns:
        dict[str, ResourceValue]: Compiler-derived observation/circuit facts.

    Raises:
        CapabilityError: If native observation is unavailable or unsupported.
        TypeError: If the native result violates its ABI.
        ValueError: If the selected request cannot be verified.
    """

    from qamomile.serialization import encode_analysis_request

    from ._rule_wire import encode_resource_analysis_request
    from .analysis import (
        _native_extra_metric_definition,
        _native_resource_value,
        _native_symbolic_logical_counts,
    )

    source = study._semantic_source()
    if isinstance(source, Program):
        payload = realizations._apply(source.to_bytes(), boundary="program")
    else:
        realized = Module.from_bytes(
            realizations._apply(source.to_bytes(), boundary="module")
        )
        payload = encode_analysis_request(realized, study._bundle.entrypoint)
    # Source rules stay in the original study. Observing a boundary needs no
    # price; an external portable realization remains explicitly unresolved.
    request = encode_resource_analysis_request(payload, ())
    projection = _native_symbolic_logical_counts(
        request,
        observation=True,
        expanded=expanded,
        basis=None if basis is None else basis.value,
        realizations=realizations.builtins,
    )
    return {
        identifier: _native_resource_value(
            _native_extra_metric_definition(identifier, ()), cost
        ).evaluate({})
        for identifier, cost in projection.extra_metrics.items()
    }


@dataclass(frozen=True, slots=True)
class ResourceBreakdown:
    """Hold a non-additive query view and a non-overlapping operation frontier.

    Args:
        study (ResourceStudy): Original immutable study.
        expanded (tuple[str, ...] | None): Selected expanded callees, or all.
        values (tuple[ResourceValue, ...]): Native observation facts.
        evaluated (bool): Whether display substitutions have been applied.
    """

    study: ResourceStudy
    expanded: tuple[str, ...] | None
    values: tuple[ResourceValue, ...]
    evaluated: bool = False

    @classmethod
    def from_study(
        cls, study: ResourceStudy, expand: tuple[object, ...] | None = ()
    ) -> ResourceBreakdown:
        """Observe a source without changing any original cost facts.

        Args:
            study (ResourceStudy): Authoritative source and existing costs.
            expand (tuple[object, ...] | None): Bodies to open; None opens all.

        Returns:
            ResourceBreakdown: Immutable notebook/text view.

        Raises:
            ValueError: If a callable selection is invalid.
            TypeError: If a selection or native result is malformed.
            CapabilityError: If the native projection is unavailable.
        """

        expanded = _selection_names(study, expand)
        return cls(study, expanded, tuple(_observe(study, expanded).values()))

    @property
    def rows(self) -> tuple[ResourceValue, ...]:
        """Return disjoint frontier counts, excluding diagnostic parent queries.

        Returns:
            tuple[ResourceValue, ...]: Operation or collapsed-call occurrences.
        """

        return tuple(
            value
            for value in self.values
            if value.metric.identifier.startswith(_FRONTIER)
        )

    def expand(self, *callables: object) -> ResourceBreakdown:
        """Open selected callables without changing the cost model.

        Args:
            *callables (object): Callable objects or names to open. With no
                arguments, open all currently visible callable boundaries.

        Returns:
            ResourceBreakdown: New view; the original remains unchanged.

        Raises:
            ValueError: If a callable selection is invalid or the view has
                already received display substitutions.
            TypeError: If a selection is malformed.
            CapabilityError: If native observation fails.
        """

        if self.evaluated:
            raise ValueError(
                "expand before evaluate; display substitutions do not specialize source"
            )
        if not callables:
            callables = tuple(
                value.metric.identifier.removeprefix(_FRONTIER).split(".", 3)[3]
                for value in self.rows
                if value.metric.identifier.startswith(f"{_FRONTIER}call.")
            )
            callables = tuple(dict.fromkeys(callables))
        selected = _selection_names(self.study, callables)
        expanded = (
            None
            if self.expanded is None
            else tuple(sorted(set((*self.expanded, *(selected or ())))))
        )
        return self.from_study(self.study, expanded)

    def call_count(self, callable: object) -> ResourceValue:
        """Return all invocations of a callable, separately from frontier costs.

        Args:
            callable (object): Callable object or resolvable name.

        Returns:
            ResourceValue: Invocation count across transform contexts.

        Raises:
            KeyError: If the callable has no observed invocation.
            ValueError: If the callable identity is absent or ambiguous.
            TypeError: If the identity is malformed.
        """

        names = _selection_names(self.study, (callable,))
        assert names is not None
        identifier = algorithmic_call_count(names[0]).identifier
        for value in self.values:
            if value.metric.identifier == identifier:
                return value
        raise KeyError(f"no observed invocation of {names[0]}")

    def evaluate(self, **values: Number) -> ResourceBreakdown:
        """Substitute display parameters, without changing source or realizations.

        Args:
            **values (Number): Unambiguous symbolic parameter substitutions.

        Returns:
            ResourceBreakdown: Evaluated observation rows over the same source.

        Raises:
            ValueError: If substitutions violate symbol domains or are ambiguous.
        """

        return replace(
            self,
            values=tuple(value.evaluate(values) for value in self.values),
            evaluated=True,
        )

    def __str__(self) -> str:
        """Render operation occurrences without inventing an elementary total.

        Returns:
            str: Stable text with abstraction and unresolved costs identified.
        """

        lines = ["Source operation occurrences (not an elementary gate total)"]
        lines.extend(
            f"{value.metric.identifier.removeprefix(_FRONTIER)}: {value}"
            for value in self.rows
        )
        portable = [
            value
            for _, value in self.study.metrics
            if value.metric.identifier.startswith(_PORTABLE)
        ]
        if portable:
            lines.append("Portable circuit costs (independent of row expansion)")
            lines.append(
                "T counts exclude rotation synthesis; depth uses a serial upper bound."
            )
            lines.extend(
                f"{value.metric.identifier.removeprefix(_PORTABLE)}: {value}"
                for value in portable
            )
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """Render the same escaped facts in notebook frontends.

        Returns:
            str: An escaped preformatted text view.
        """

        return f"<pre>{escape(str(self))}</pre>"


@dataclass(frozen=True, slots=True)
class GateCostModel:
    """Estimate gate costs using a gate basis and fixed decomposition recipes.

    This backend-neutral model does not construct an executable circuit. It
    prices supported recipes and clean workspace; unsupported operations remain
    named unknowns. By default, semantic modular multiplication uses the
    ``mod_mul.schoolbook_ripple``, ``mod_mul_add.schoolbook_ripple``, and
    ``mod_pow.square_and_multiply`` baselines.

    Args:
        basis (GateBasis | str): Alphabet; rotations remain unsynthesized.
        realizations (RealizationOptions): Built-in or user-authored operation
            implementations. Every source-backed callable body is analyzed.

    Raises:
        ValueError: If the basis is unknown.
    """

    basis: GateBasis | str = GateBasis.CLIFFORD_TOFFOLI_ROTATIONS
    realizations: RealizationOptions = RealizationOptions(
        {
            "quantum.mod_mul": "mod_mul.schoolbook_ripple",
            "quantum.mod_pow": "mod_pow.square_and_multiply",
        }
    )

    def __post_init__(self) -> None:
        """Freeze the alphabet and realization selection.

        Raises:
            ValueError: If the alphabet is unknown.
            TypeError: If realizations do not use the public option type.
        """

        object.__setattr__(self, "basis", GateBasis(self.basis))
        if not isinstance(self.realizations, RealizationOptions):
            raise TypeError("realizations must be RealizationOptions")

    @property
    def identity(self) -> str:
        """Return the selected portable alphabet identity.

        Returns:
            str: Model derivation label; full selections are retained in evidence.
        """

        return f"qamomile.portable.{GateBasis(self.basis).value}.v1"

    def project(self, study: ResourceStudy) -> Mapping[str, ResourceValue]:
        """Add circuit facts without replacing the source's logical facts.

        Args:
            study (ResourceStudy): Source study retaining verified QCore.

        Returns:
            Mapping[str, ResourceValue]: Basis counts, serial depth enclosure,
                and additional recipe scratch. External implementations stay unknown.

        Raises:
            ValueError: If a selection is invalid or the study was display-evaluated.
            TypeError: If a selection or native projection is malformed.
            CapabilityError: If native pricing is unavailable.
        """

        if "display-evaluation" in study.derivations:
            raise ValueError(
                "project before evaluate; specialize the source to change the program"
            )
        expanded = None
        evidence = _verified_evidence(
            EvidenceAuthority.STRUCTURALLY_DERIVED,
            self.identity,
            details=(
                ("expanded", "all"),
                ("realizations", self.realizations.fingerprint),
                ("rotation_synthesis", "not applied"),
                ("depth", "serial upper bound; no routing"),
            ),
        )
        assumptions = tuple(
            f"user asserts {row.implementation_symbol} implements {row.target_symbol}"
            for row in self.realizations._custom
        )
        return {
            identifier: replace(
                value,
                evidence=(*value.evidence, evidence),
                assumptions=tuple(dict.fromkeys((*value.assumptions, *assumptions))),
            )
            for identifier, value in _observe(
                study,
                expanded,
                GateBasis(self.basis),
                self.realizations,
            ).items()
            if identifier.startswith(_PORTABLE)
            or identifier == CIRCUIT_OUTPUT_TRACE_DISTANCE_BOUND.identifier
        }

    def recipe(
        self, opcode: str, *, controls: int = 0
    ) -> tuple[tuple[str, tuple[int, ...], str, int, int], ...] | None:
        """Inspect the exact ordered recipe used by the native counter.

        Args:
            opcode (str): Semantic primitive name, for example ``quantum.h``.
            controls (int): Number of additional positive controls.

        Returns:
            tuple | None: Steps containing opcode, local lanes, angle kind,
                numerator and denominator; None means unsupported. Angle kinds
                are none, input-angle multiple, or pi multiple.

        Raises:
            CapabilityError: If native recipe inspection is unavailable.
            ValueError: If controls is negative or not an integer.
            TypeError: If the native recipe violates its ABI.
        """

        if (
            isinstance(controls, bool)
            or not isinstance(controls, int)
            or not 0 <= controls <= 4096
        ):
            raise ValueError("controls must be an integer between 0 and 4096")
        inspect = getattr(native, "inspect_portable_gate_recipe", None)
        if not callable(inspect):
            raise CapabilityError("Q4260: native portable recipes are unavailable")
        rows = inspect(opcode, controls, GateBasis(self.basis).value)
        if rows is None:
            return None
        if not isinstance(rows, list):
            raise TypeError("native portable recipe violates its ABI")
        for row in rows:
            if not isinstance(row, tuple) or len(row) != 5:
                raise TypeError("native portable recipe step violates its ABI")
            name, lanes, kind, numerator, denominator = row
            if (
                not isinstance(name, str)
                or not name.startswith("quantum.")
                or not isinstance(lanes, bytes)
                or not lanes
                or any(type(lane) is not int or not 0 <= lane < 4 for lane in lanes)
                or kind not in {"none", "input", "pi"}
                or type(numerator) is not int
                or type(denominator) is not int
                or denominator <= 0
            ):
                raise TypeError("native portable recipe step violates its ABI")
        return tuple(
            (name, tuple(lanes), kind, numerator, denominator)
            for name, lanes, kind, numerator, denominator in rows
        )


def compare(
    studies: Mapping[str, ResourceStudy], *, metrics: tuple[MetricDefinition, ...]
) -> str:
    """Render evidence-bearing costs side by side without asserting a winner.

    Args:
        studies (Mapping[str, ResourceStudy]): User-labeled alternative studies.
        metrics (tuple[MetricDefinition, ...]): Explicit shared metric definitions.

    Returns:
        str: Comparison including coordinate, units, models, and missing values.

    Raises:
        ValueError: If no studies or metrics are selected.
        TypeError: If inputs are not named studies or metric definitions.
    """

    if not studies or not metrics:
        raise ValueError("comparison needs studies and explicit metrics")
    if any(
        not isinstance(name, str) or not isinstance(study, ResourceStudy)
        for name, study in studies.items()
    ):
        raise TypeError("compare expects named ResourceStudy values")
    if any(not isinstance(metric, MetricDefinition) for metric in metrics):
        raise TypeError("metrics must contain MetricDefinition values")
    lines = [
        "Conditional cost comparison; task/accuracy equivalence is not established."
    ]
    if any(metric.identifier.startswith(_PORTABLE) for metric in metrics):
        lines.append(
            "T counts exclude rotation synthesis; depth uses a serial upper bound."
        )
    for name, study in studies.items():
        models = (
            ", ".join(item for item in study.derivations if item.startswith("model:"))
            or "source only"
        )
        lines.append(f"{name}: {models}")
    for metric in metrics:
        lines.append(
            f"{metric.identifier} [{metric.coordinate.value}; {metric.unit.name}]"
        )
        for name, study in studies.items():
            value = study.values.get(metric.identifier)
            if value is not None and value.metric != metric:
                raise ValueError("comparison metric definitions disagree")
            lines.append(f"  {name}: {value if value is not None else 'not reported'}")
            if value is not None:
                lines.extend(
                    f"    assumes: {assumption}" for assumption in value.assumptions
                )
    return "\n".join(lines)
