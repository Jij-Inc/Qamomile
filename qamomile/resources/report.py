"""Immutable public resource studies and structural explanations."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from qamomile.ir import Module, Program

    from .research import ResourceBreakdown

from .bundle import AnalysisBundle
from .expr import Number, Symbol, _root_free_symbols, resolve_substitutions
from .metrics import MetricDefinition
from .values import ResourceValue


@dataclass(frozen=True, slots=True)
class ResourceContribution:
    """One source-addressable local contribution to a logical metric."""

    owner: str
    source: str
    operation: str
    value: ResourceValue


@dataclass(frozen=True, slots=True)
class ResourceTraceNode:
    """A deterministic structural explanation tree over retained QCore facts."""

    name: str
    kind: str
    source: str = ""
    children: tuple[ResourceTraceNode, ...] = ()

    def render(self, indent: int = 0) -> str:
        """Render this trace and its children as stable plain text."""

        location = f" at {self.source}" if self.source else ""
        lines = [f"{' ' * indent}{self.name} [{self.kind}]{location}"]
        for child in self.children:
            lines.append(child.render(indent + 2))
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class ResourceStudy:
    """Symbolic resource facts for one authoritative Module or Program."""

    metrics: tuple[tuple[str, ResourceValue], ...]
    _bundle: AnalysisBundle = field(repr=False)
    derivations: tuple[str, ...] = ()

    def breakdown(self, *, expand: tuple[object, ...] | None = ()) -> ResourceBreakdown:
        """Inspect call boundaries without changing source or model costs.

        Args:
            expand (tuple[object, ...] | None): Callables to open, or None for all.

        Returns:
            ResourceBreakdown: Native symbolic occurrence counts and query facts.

        Raises:
            ValueError: If a callable name is absent or ambiguous.
            TypeError: If a selection or native result is malformed.
            CapabilityError: If native observation is unavailable.
        """

        from .research import ResourceBreakdown

        if "display-evaluation" in self.derivations:
            raise ValueError(
                "expand before evaluate; a display substitution is not specialization"
            )
        return ResourceBreakdown.from_study(self, expand)

    def unresolved(
        self, metric: str | MetricDefinition | None = None
    ) -> tuple[ResourceValue, ...]:
        """Return named unresolved facts, optionally for a single metric.

        Args:
            metric (str | MetricDefinition | None): Selected metric, or all.

        Returns:
            tuple[ResourceValue, ...]: Facts retaining unresolved contributors.

        Raises:
            KeyError: If the selected metric is absent.
        """

        values = (self.metric(metric),) if metric is not None else self.values.values()
        return tuple(value for value in values if value.unknown_contributors)

    def require(
        self, *metrics: str | MetricDefinition, coverage: str = "finite_upper"
    ) -> ResourceStudy:
        """Require only the cost coverage needed by a research question.

        Args:
            *metrics (str | MetricDefinition): Required metrics.
            coverage (str): exact, finite_upper, or finite_enclosure.

        Returns:
            ResourceStudy: This unchanged study when every requirement holds.

        Raises:
            ValueError: If coverage is invalid, no metrics are supplied, or a
                required fact lacks the requested bound.
            KeyError: If a metric is absent.
        """

        from .values import Relation

        if coverage not in {"exact", "finite_upper", "finite_enclosure"} or not metrics:
            raise ValueError("require needs metrics and a supported coverage")
        for metric in metrics:
            value = self.metric(metric)
            lower, upper = value.finite_enclosure()
            sufficient = (
                value.relation is Relation.EQUALITY
                if coverage == "exact"
                else upper is not None
                if coverage == "finite_upper"
                else lower is not None and upper is not None
            )
            if not sufficient:
                raise ValueError(f"{value.metric.identifier} lacks {coverage}: {value}")
        return self

    def __post_init__(self) -> None:
        """Require one canonical mapping from metric identity to resource fact."""

        names = tuple(name for name, _ in self.metrics)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise ValueError("resource study metrics are not canonical")
        if any(name != value.metric.identifier for name, value in self.metrics):
            raise ValueError("resource metric key disagrees with its definition")

    @classmethod
    def _create(
        cls,
        metrics: Mapping[str, ResourceValue],
        bundle: AnalysisBundle,
        *,
        derivations: tuple[str, ...] = (),
    ) -> ResourceStudy:
        """Create an internal study from an ordinary metric mapping."""

        return cls(tuple(sorted(metrics.items())), bundle, derivations)

    @property
    def values(self) -> Mapping[str, ResourceValue]:
        """Return the study's sole read-only metric mapping."""

        return MappingProxyType(dict(self.metrics))

    @property
    def semantic_digest(self) -> str:
        """Return the selected QCore semantic identity retained by this study."""

        return self._bundle.qcore_digest

    def _semantic_source(self) -> Module | Program:
        """Decode the retained Module or Program used for compilation.

        Returns:
            object: Verified Module or Program retained by this study.

        Raises:
            SerializationError: If the retained payload is invalid.
            ValueError: If its selected entrypoint identity disagrees.
        """

        return self._bundle.semantic_source()

    def metric(self, metric: str | MetricDefinition) -> ResourceValue:
        """Return one metric from the authoritative study.

        Args:
            metric (str | MetricDefinition): Stable metric identifier or its
                complete definition.

        Returns:
            ResourceValue: The requested mathematical resource fact.

        Raises:
            KeyError: If the metric is absent.
        """

        identifier = _metric_identifier(metric)
        try:
            return self.values[identifier]
        except KeyError as error:
            raise KeyError(f"resource study has no metric {identifier!r}") from error

    @property
    def digest(self) -> str:
        """Return deterministic report identity, not authenticity or validity."""

        from ._wire import encode_study, study_digest

        return study_digest(encode_study(self))

    @property
    def iteration_symbols(self) -> tuple[Symbol, ...]:
        """Return scoped while body-execution counts appearing in this study.

        Returns:
            tuple[Symbol, ...]: Symbols ordered by their qualified identity.
        """
        symbols = {
            symbol.identity: symbol
            for _, value in self.metrics
            for expression in (value.expression, value.lower, value.upper)
            if expression is not None
            for symbol in _root_free_symbols(expression)
            if symbol.origin == "while_iterations"
        }
        return tuple(symbols[identity] for identity in sorted(symbols))

    def assume_iterations(self, values: Mapping[Symbol | str, int]) -> ResourceStudy:
        """Evaluate a scenario conditional on user-supplied while counts.

        Args:
            values (Mapping[Symbol | str, int]): Iteration symbols or qualified
                identities mapped to nonnegative body-execution counts.

        Returns:
            ResourceStudy: A new study recording the scenario assumptions.

        Raises:
            ValueError: If a symbol is absent or a count is negative.
            TypeError: If a count is not an integer.
            CapabilityError: If the native assumption API is unavailable.
        """
        return self._iteration_assumptions(values, upper_bound=False)

    def bound_iterations(self, values: Mapping[Symbol | str, int]) -> ResourceStudy:
        """Derive resource upper bounds conditional on asserted iteration limits.

        Args:
            values (Mapping[Symbol | str, int]): Iteration symbols or qualified
                identities mapped to nonnegative upper limits per invocation.

        Returns:
            ResourceStudy: A new study retaining user assumptions and upper bounds;
                unsupported monotonic expressions retain explicit unknowns.

        Raises:
            ValueError: If a symbol is absent or a count is negative.
            TypeError: If a count is not an integer.
            CapabilityError: If the native assumption API is unavailable.
        """
        return self._iteration_assumptions(values, upper_bound=True)

    def _iteration_assumptions(
        self, values: Mapping[Symbol | str, int], *, upper_bound: bool
    ) -> ResourceStudy:
        """Project an iteration assumption request to its native semantic owner.

        Args:
            values (Mapping[Symbol | str, int]): Qualified loop counts.
            upper_bound (bool): Interpret counts as upper limits when true.

        Returns:
            ResourceStudy: The validated native result.

        Raises:
            TypeError: If binding keys or counts have unsupported types.
            ValueError: If the native owner rejects a binding.
            CapabilityError: If the native API is unavailable.
        """
        from qamomile._native import compiler
        from qamomile.errors import CapabilityError

        apply = getattr(compiler, "assume_resource_iterations", None)
        if not callable(apply):
            raise CapabilityError("Q4257: native iteration assumptions are unavailable")
        bindings = []
        for symbol, count in values.items():
            if (
                not isinstance(symbol, (Symbol, str))
                or isinstance(count, bool)
                or not isinstance(count, int)
            ):
                raise TypeError(
                    "iteration assumptions require symbols and integer counts"
                )
            bindings.append(
                (symbol.identity if isinstance(symbol, Symbol) else symbol, str(count))
            )
        result = apply(self.to_bytes(), sorted(bindings), upper_bound)
        if not isinstance(result, bytes):
            raise TypeError(
                "Q4257: native iteration assumption result violated the ABI"
            )
        return ResourceStudy.from_bytes(result)

    def evaluate(self, **values: Number) -> ResourceStudy:
        """Substitute report formulas without pretending to re-specialize QCore."""

        expressions = tuple(
            expression
            for _, value in self.metrics
            for expression in (value.expression, value.lower, value.upper)
            if expression is not None
        )
        substitutions = resolve_substitutions(expressions, values)
        return replace(
            self,
            metrics=tuple(
                (name, value._substitute(substitutions)) for name, value in self.metrics
            ),
            derivations=(*self.derivations, "display-evaluation"),
        )

    def to_bytes(self) -> bytes:
        """Serialize and integrity-seal the complete study deterministically."""

        from ._wire import encode_study

        return encode_study(self)

    @classmethod
    def from_bytes(cls, payload: bytes | bytearray | memoryview) -> ResourceStudy:
        """Decode, integrity-check, and validate a persisted resource study."""

        from ._wire import decode_study

        return decode_study(bytes(payload))

    def with_metrics(
        self,
        metrics: Mapping[str, ResourceValue],
        *,
        derivation: str,
    ) -> ResourceStudy:
        """Append lower-coordinate facts without overwriting logical metrics."""

        merged = dict(self.metrics)
        overlap = merged.keys() & metrics.keys()
        if overlap:
            raise ValueError(
                "projection cannot overwrite earlier resource coordinates: "
                + ", ".join(sorted(overlap))
            )
        merged.update(metrics)
        return ResourceStudy._create(
            merged,
            self._bundle,
            derivations=(*self.derivations, derivation),
        )

    def contributions(
        self,
        metric: str | MetricDefinition,
    ) -> tuple[ResourceContribution, ...]:
        """Return retained source-local contribution enclosures for one metric."""

        definition = _metric_definition(self, metric)
        contributions: list[ResourceContribution] = []
        for fact in self._bundle.facts:
            for identifier, value in fact.contributions:
                if identifier == definition.identifier:
                    contributions.append(
                        ResourceContribution(
                            fact.owner,
                            fact.source,
                            fact.operation,
                            value,
                        )
                    )
        return tuple(contributions)

    def hotspots(
        self,
        metric: str | MetricDefinition,
        *,
        limit: int = 10,
    ) -> tuple[ResourceContribution, ...]:
        """Rank local contributions through exact native endpoint ordering."""

        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("hotspot limit must be a positive integer")
        from qamomile._native import compiler

        from ._rule_wire import encode_resource_expression_graph

        rank = getattr(compiler, "rank_resource_hotspots", None)
        if not callable(rank):
            raise RuntimeError(
                "Q4243: the required native resource-hotspot ranking is unavailable"
            )
        contributions = self.contributions(metric)
        candidates: list[tuple[bytes | None, str, str]] = []
        for contribution in contributions:
            _, upper = contribution.value.finite_enclosure()
            candidates.append(
                (
                    (
                        encode_resource_expression_graph(upper)
                        if upper is not None
                        else None
                    ),
                    contribution.owner,
                    contribution.source,
                )
            )
        expected = min(limit, len(contributions))
        projection = rank(candidates, expected)
        if (
            not isinstance(projection, list)
            or len(projection) != expected
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < len(contributions)
                for index in projection
            )
            or len(set(projection)) != len(projection)
        ):
            raise TypeError("Q4243: native resource-hotspot ranking violated the ABI")
        return tuple(contributions[index] for index in projection)

    @property
    def trace(self) -> ResourceTraceNode:
        """Return the retained callable/operation explanation hierarchy."""

        by_owner: dict[str, list[ResourceTraceNode]] = {}
        for fact in self._bundle.facts:
            by_owner.setdefault(fact.owner, []).append(
                ResourceTraceNode(
                    fact.operation,
                    fact.kind,
                    fact.source,
                )
            )
        callables = tuple(
            ResourceTraceNode(owner, "callable", children=tuple(children))
            for owner, children in sorted(by_owner.items())
        )
        return ResourceTraceNode(
            self._bundle.entrypoint,
            "resource-study",
            children=callables,
        )

    def explain(self, metric: str | MetricDefinition | None = None) -> str:
        """Render an aggregate fact and its relevant source contributions.

        Args:
            metric (str | MetricDefinition | None): Metric to explain. If
                omitted, render the complete retained operation hierarchy.

        Returns:
            str: Stable text containing the aggregate relation and only the
                source-local facts that contribute to the selected metric.
        """

        if metric is None:
            return self.trace.render()
        definition = _metric_definition(self, metric)
        value = self.metric(definition)
        lines = [f"{definition.identifier}: {value}"]
        for contribution in self.contributions(definition):
            location = f" at {contribution.source}" if contribution.source else ""
            lines.append(
                f"  {contribution.owner}::{contribution.operation}{location}: "
                f"{contribution.value}"
            )
        return "\n".join(lines)


def _metric_identifier(metric: str | MetricDefinition) -> str:
    """Normalize a metric definition or identifier for public lookup."""

    if isinstance(metric, MetricDefinition):
        return metric.identifier
    if not isinstance(metric, str):
        raise TypeError("resource metric must be an identifier or MetricDefinition")
    return metric


def _metric_definition(
    study: ResourceStudy,
    metric: str | MetricDefinition,
) -> MetricDefinition:
    """Resolve a complete definition from a public metric argument."""

    if isinstance(metric, MetricDefinition):
        return metric
    identifier = _metric_identifier(metric)
    value = study.values.get(identifier)
    if value is not None:
        return value.metric
    raise KeyError(f"resource study has no metric {identifier!r}")


__all__ = [
    "ResourceContribution",
    "ResourceStudy",
    "ResourceTraceNode",
]
