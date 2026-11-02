"""Hierarchical symbolic logical-resource analysis over verified QCore."""

from __future__ import annotations

import struct
from collections.abc import Callable
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Any, cast

from qamomile._native import compiler as native
from qamomile.errors import CapabilityError, Diagnostic, VerificationError
from qamomile.ir import (
    Module,
    Program,
)

from .bundle import AnalysisBundle, AnalysisFact
from .expr import (
    ONE,
    Domain,
    Expr,
    Predicate,
    Symbol,
    _binder_attributes,
    literal,
)
from .metrics import (
    CIRCUIT_OUTPUT_TRACE_DISTANCE_BOUND,
    LOGICAL_ALLOCATED_QUBITS,
    LOGICAL_CIRCUIT_QUBITS,
    LOGICAL_CLEANUP_WORK,
    LOGICAL_CLIFFORD_DEPTH,
    LOGICAL_CLIFFORD_GATES,
    LOGICAL_GATE_DEPTH,
    LOGICAL_GATES,
    LOGICAL_INPUT_QUBITS,
    LOGICAL_MEASUREMENT_DEPTH,
    LOGICAL_MEASUREMENTS,
    LOGICAL_MULTI_QUBIT_GATES,
    LOGICAL_NON_CLIFFORD_DEPTH,
    LOGICAL_NON_CLIFFORD_GATES,
    LOGICAL_PEAK_QUBITS,
    LOGICAL_RESET_DEPTH,
    LOGICAL_RESETS,
    LOGICAL_ROTATION_DEPTH,
    LOGICAL_ROTATION_GATES,
    LOGICAL_SINGLE_QUBIT_GATES,
    LOGICAL_SPAN,
    LOGICAL_T_DEPTH,
    LOGICAL_T_GATES,
    LOGICAL_TOFFOLI_DEPTH,
    LOGICAL_TOFFOLI_GATES,
    LOGICAL_TWO_QUBIT_GATES,
    LOGICAL_WORK,
    Composition,
    Coordinate,
    MetricDefinition,
    algorithmic_call_count,
    algorithmic_query_count,
    logical_operation_count,
    portable_metric,
)
from .report import ResourceStudy
from .rules import ResourceRule
from .values import (
    Evidence,
    EvidenceAuthority,
    Relation,
    ResourceValue,
    _verified_evidence,
)

_ANALYSIS_VERSION = "qamomile.logical-analysis.v1"
_STRUCTURAL_EVIDENCE = (
    _verified_evidence(
        EvidenceAuthority.STRUCTURALLY_DERIVED,
        "qamomile.qcore.metric-algebra.v1",
    ),
)


@dataclass(frozen=True, slots=True)
class _Cost:
    """A nonnegative additive value with optional finite upper endpoint."""

    lower: Expr
    upper: Expr | None
    exact: bool
    relation: Relation
    dependency_span_relation: Relation
    unknown: tuple[str, ...] = ()
    evidence: tuple[Evidence, ...] = ()
    assumptions: tuple[str, ...] = ()

    @classmethod
    def zero(cls) -> _Cost:
        """Return the additive identity."""

        return cls(
            literal(0),
            literal(0),
            True,
            Relation.EQUALITY,
            Relation.EQUALITY,
        )


@dataclass(frozen=True, slots=True)
class _NativeLogicalProjection:
    """Hold validated logical facts returned by the native compiler.

    Args:
        work (_Cost): Total primitive quantum-operation work.
        dependency_span (_Cost | None): Scalar dependency-span enclosure,
            including when the complete scheduled depth family is unavailable.
        opcodes (dict[str, _Cost]): Counts keyed by canonical quantum opcode.
        extra_metrics (dict[str, _Cost]): Declarative rule metrics outside the
            structural logical projection.
        families (dict[str, _Cost] | None): Complete logical gate-family counts,
            or `None` when native classification is intentionally incomplete.
        measurements (_Cost | None): Per-qubit measurement events when proved.
        resets (_Cost | None): Per-qubit reset events when proved.
        input_qubits (_Cost | None): Caller-supplied logical width when proved.
        allocated_qubits (_Cost | None): Statically reserved logical width when
            proved.
        circuit_qubits (_Cost | None): Input plus allocated logical width when
            proved.
        peak_qubits (_Cost | None): Maximum simultaneously live logical width
            when proved.
        depths (dict[str, _Cost] | None): Dependency span and scheduled depth
            families when proved.
        cleanup_work (_Cost): Work proven to discharge cleanup obligations.
        unknown (tuple[str, ...]): Sorted, unique unresolved contributors from
            every compiler-owned count in this projection.
        facts (tuple[AnalysisFact, ...]): Canonical source-local contribution
            facts from the same native symbolic traversal.
    """

    work: _Cost
    dependency_span: _Cost | None
    opcodes: dict[str, _Cost]
    extra_metrics: dict[str, _Cost]
    families: dict[str, _Cost] | None
    measurements: _Cost | None
    resets: _Cost | None
    input_qubits: _Cost | None
    allocated_qubits: _Cost | None
    circuit_qubits: _Cost | None
    peak_qubits: _Cost | None
    depths: dict[str, _Cost] | None
    cleanup_work: _Cost
    unknown: tuple[str, ...]
    facts: tuple[AnalysisFact, ...]


def _native_extra_metric_definition(
    identifier: str,
    rules: tuple[ResourceRule, ...],
) -> MetricDefinition:
    """Resolve one native extra-metric identity to its public definition.

    Args:
        identifier (str): Canonical identifier returned by native analysis.
        rules (tuple[ResourceRule, ...]): Rules declared for the same request.

    Returns:
        MetricDefinition: The declared or structurally defined metric.

    Raises:
        TypeError: If native analysis returns an undeclared metric.
        ValueError: If a structural callable identity is malformed.
    """

    for rule in rules:
        if rule.metric.identifier == identifier:
            return rule.metric
    call_prefix = "qamomile.algorithmic.call_count."
    if identifier.startswith(call_prefix):
        return algorithmic_call_count(identifier.removeprefix(call_prefix))
    query_prefix = "qamomile.algorithmic.query_count."
    if identifier.startswith(query_prefix):
        return algorithmic_query_count(identifier.removeprefix(query_prefix))
    if identifier.startswith("qamomile.algorithmic.frontier."):
        from .expr import SEMANTIC_OPS

        return MetricDefinition(
            identifier, Coordinate.ALGORITHMIC, Domain.NATURAL, SEMANTIC_OPS
        )
    if identifier.startswith("qamomile.circuit.portable."):
        return portable_metric(identifier.removeprefix("qamomile.circuit.portable."))
    if identifier == CIRCUIT_OUTPUT_TRACE_DISTANCE_BOUND.identifier:
        return CIRCUIT_OUTPUT_TRACE_DISTANCE_BOUND
    raise TypeError("native symbolic analysis returned an undeclared extra metric")


def _gate_family_metrics() -> dict[str, MetricDefinition]:
    """Return the canonical logical gate-family metric mapping."""

    return {
        "gates": LOGICAL_GATES,
        "single_qubit_gates": LOGICAL_SINGLE_QUBIT_GATES,
        "two_qubit_gates": LOGICAL_TWO_QUBIT_GATES,
        "multi_qubit_gates": LOGICAL_MULTI_QUBIT_GATES,
        "clifford_gates": LOGICAL_CLIFFORD_GATES,
        "rotation_gates": LOGICAL_ROTATION_GATES,
        "t_gates": LOGICAL_T_GATES,
        "toffoli_gates": LOGICAL_TOFFOLI_GATES,
        "non_clifford_gates": LOGICAL_NON_CLIFFORD_GATES,
    }


def _rule_cost(rule: ResourceRule) -> _Cost:
    """Convert one declarative relation into the internal enclosure algebra."""

    evidence = (
        Evidence(
            rule.authority,
            rule.provenance,
            details=(
                ("resource_rule", rule.identity),
                ("callable", cast(str, rule.callable)),
            ),
        ),
    )
    assumptions = tuple(
        dict.fromkeys(
            (*rule.assumptions, *(f"validity: {item}" for item in rule.validity_domain))
        )
    )
    if rule.relation is Relation.EQUALITY:
        return _Cost(
            rule.expression,
            rule.expression,
            True,
            Relation.EQUALITY,
            Relation.EQUALITY,
            evidence=evidence,
            assumptions=assumptions,
        )
    if rule.relation is Relation.LOWER_BOUND:
        return _Cost(
            rule.expression,
            None,
            False,
            Relation.LOWER_BOUND,
            Relation.LOWER_BOUND,
            evidence=evidence,
            assumptions=assumptions,
        )
    return _Cost(
        literal(0),
        rule.expression,
        False,
        Relation.ENCLOSURE,
        Relation.UPPER_BOUND,
        evidence=evidence,
        assumptions=assumptions,
    )


def _fact_contributions(
    *,
    work: _Cost,
    measurements: _Cost | None,
    resets: _Cost | None,
    cleanup: _Cost,
    operations: tuple[tuple[str, _Cost], ...],
    gate_families: tuple[tuple[str, _Cost], ...],
    extra_metrics: tuple[tuple[MetricDefinition, _Cost], ...],
) -> tuple[tuple[str, ResourceValue], ...]:
    """Project source-local contribution enclosures into typed values."""

    costs: dict[str, tuple[MetricDefinition, _Cost]] = {
        LOGICAL_WORK.identifier: (LOGICAL_WORK, work),
        LOGICAL_CLEANUP_WORK.identifier: (LOGICAL_CLEANUP_WORK, cleanup),
    }
    if measurements is not None:
        costs[LOGICAL_MEASUREMENTS.identifier] = (LOGICAL_MEASUREMENTS, measurements)
    if resets is not None:
        costs[LOGICAL_RESETS.identifier] = (LOGICAL_RESETS, resets)
    family_metrics = _gate_family_metrics()
    for family, cost in gate_families:
        metric = family_metrics[family]
        costs[metric.identifier] = (metric, cost)
    for operation_name, cost in operations:
        metric = logical_operation_count(operation_name)
        costs[metric.identifier] = (metric, cost)
    for metric, cost in extra_metrics:
        costs[metric.identifier] = (metric, cost)
    return tuple(
        (
            identifier,
            replace(
                _native_resource_value(metric, cost),
                evidence=(),
                assumptions=(),
            ),
        )
        for identifier, (metric, cost) in sorted(costs.items())
    )


_NATIVE_DEPTH_METRICS = {
    "span": LOGICAL_SPAN,
    "gates": LOGICAL_GATE_DEPTH,
    "clifford_gates": LOGICAL_CLIFFORD_DEPTH,
    "rotation_gates": LOGICAL_ROTATION_DEPTH,
    "t_gates": LOGICAL_T_DEPTH,
    "toffoli_gates": LOGICAL_TOFFOLI_DEPTH,
    "non_clifford_gates": LOGICAL_NON_CLIFFORD_DEPTH,
    "measurements": LOGICAL_MEASUREMENT_DEPTH,
    "resets": LOGICAL_RESET_DEPTH,
}


def logical(
    source: object,
    *,
    rules: ResourceRule | tuple[ResourceRule, ...] | list[ResourceRule] = (),
    strict: bool = False,
) -> ResourceStudy:
    """Analyze a program family or closed program without target assumptions.

    Args:
        source (object): ``ProgramFamily``, ``QKernel``, ``Module``, or
            ``Program`` to analyze.
        rules (ResourceRule | tuple[ResourceRule, ...] | list[ResourceRule]):
            Declarative resource contracts for bodyless callables.
        strict (bool): Raise when any operation, transform, call, or loop
            contribution remains unknown.

    Returns:
        ResourceStudy: An immutable symbolic logical study derived from the
        supplied body.

    Raises:
        TypeError: If ``source`` is not a Qamomile semantic program value.
        ValueError: If an entrypoint is ambiguous or strict coverage fails.
        CapabilityError: If the native resource compiler is unavailable or
            rejects the verified program.
        VerificationError: If the input is not a valid semantic QCore artifact.
    """

    return _logical_native(source, rules=rules, strict=strict)


def _logical_native(
    source: object,
    *,
    rules: ResourceRule | tuple[ResourceRule, ...] | list[ResourceRule],
    strict: bool,
) -> ResourceStudy:
    """Build one study from the authoritative native analyzer.

    Args:
        source (object): Semantic program value accepted by :func:`logical`.
        rules (ResourceRule | tuple[ResourceRule, ...] | list[ResourceRule]):
            Declarative resource contracts for bodyless callables.
        strict (bool): Raise when a contribution remains unknown.

    Returns:
        ResourceStudy: An immutable symbolic logical study.

    Raises:
        TypeError: If the source or native result violates its public ABI.
        ValueError: If source selection, a rule, or strict coverage is invalid.
        CapabilityError: If the selected native engine is unavailable or cannot
            analyze the verified program.
        VerificationError: If the native engine rejects an invalid semantic
            artifact.
    """

    module, entrypoint, closed = _resolve_source(source)
    normalized_rules = _normalize_resource_rules(rules)
    if closed:
        if not isinstance(module, Program):
            raise TypeError("closed resource source is not a Program")
        native_payload = (
            module.to_bytes() if normalized_rules else module._native_program()
        )
    else:
        from qamomile.serialization import encode_analysis_request

        if not isinstance(module, Module):
            raise TypeError("open resource source is not a Module")
        native_payload = encode_analysis_request(module, entrypoint)
    if normalized_rules:
        from ._rule_wire import encode_resource_analysis_request

        native_payload = encode_resource_analysis_request(
            native_payload,
            normalized_rules,
        )
    native_projection = (
        _native_symbolic_logical_counts(
            native_payload,
            resource_request=True,
            rules=normalized_rules,
        )
        if normalized_rules
        else _native_symbolic_logical_counts(
            native_payload,
            open_family=not closed,
        )
    )
    facts = native_projection.facts
    unknown = native_projection.unknown
    if strict and unknown:
        raise ValueError(
            "strict resource analysis has unknown contributors: " + ", ".join(unknown)
        )
    metrics = {}
    work = native_projection.work
    opcodes = native_projection.opcodes
    families = native_projection.families
    metrics[LOGICAL_WORK.identifier] = _native_resource_value(LOGICAL_WORK, work)
    for opcode, count in sorted(opcodes.items()):
        metric = logical_operation_count(opcode)
        metrics[metric.identifier] = _native_resource_value(metric, count)
    for identifier, cost in native_projection.extra_metrics.items():
        metric = _native_extra_metric_definition(identifier, normalized_rules)
        metrics[identifier] = _native_resource_value(metric, cost)
    if families is not None:
        for family, metric in _gate_family_metrics().items():
            metrics[metric.identifier] = _native_resource_value(
                metric,
                families.get(family, _Cost.zero()),
            )
    if native_projection.measurements is not None:
        metrics[LOGICAL_MEASUREMENTS.identifier] = _native_resource_value(
            LOGICAL_MEASUREMENTS,
            native_projection.measurements,
        )
    if native_projection.resets is not None:
        metrics[LOGICAL_RESETS.identifier] = _native_resource_value(
            LOGICAL_RESETS,
            native_projection.resets,
        )
    if native_projection.input_qubits is not None:
        metrics[LOGICAL_INPUT_QUBITS.identifier] = _native_resource_value(
            LOGICAL_INPUT_QUBITS,
            native_projection.input_qubits,
        )
    if native_projection.allocated_qubits is not None:
        metrics[LOGICAL_ALLOCATED_QUBITS.identifier] = _native_resource_value(
            LOGICAL_ALLOCATED_QUBITS,
            native_projection.allocated_qubits,
        )
    if native_projection.circuit_qubits is not None:
        metrics[LOGICAL_CIRCUIT_QUBITS.identifier] = _native_resource_value(
            LOGICAL_CIRCUIT_QUBITS,
            native_projection.circuit_qubits,
        )
    if native_projection.peak_qubits is not None:
        metrics[LOGICAL_PEAK_QUBITS.identifier] = _native_resource_value(
            LOGICAL_PEAK_QUBITS,
            native_projection.peak_qubits,
        )
    if native_projection.dependency_span is not None:
        metrics[LOGICAL_SPAN.identifier] = _native_resource_value(
            LOGICAL_SPAN,
            native_projection.dependency_span,
        )
    if native_projection.depths is not None:
        for family, cost in native_projection.depths.items():
            metric = _NATIVE_DEPTH_METRICS[family]
            metrics[metric.identifier] = _native_resource_value(metric, cost)
    metrics[LOGICAL_CLEANUP_WORK.identifier] = _native_resource_value(
        LOGICAL_CLEANUP_WORK,
        native_projection.cleanup_work,
    )
    bundle = AnalysisBundle.from_source(
        module,
        entrypoint=entrypoint,
        analysis_version=_ANALYSIS_VERSION,
        facts=facts,
    )
    return ResourceStudy._create(
        metrics,
        bundle,
        derivations=(
            "qcore-structured-analysis",
            "cleanup:closed" if closed else "cleanup:source-obligations",
            *(("native-transform-mode-composition",) if normalized_rules else ()),
            "native-symbolic-logical-counts",
            "native-source-local-facts",
            *(f"resource-rule:{rule.identity}" for rule in normalized_rules),
        ),
    )


def _native_symbolic_logical_counts(
    payload: Any,
    *,
    open_family: bool = False,
    resource_request: bool = False,
    rules: tuple[ResourceRule, ...] = (),
    observation: bool = False,
    expanded: tuple[str, ...] | None = None,
    basis: str | None = None,
    realizations: tuple[str, ...] = (),
) -> _NativeLogicalProjection:
    """Read native symbolic logical counts through the closed expression ABI.

    Args:
        payload (Any): Verified native owner or canonical selected-program bytes from the
            verified Python lifecycle. The artifact may be a closed
            `Program` or an analysis-only open program family.
        open_family (bool): Whether ``payload`` selects an open program family
            rather than a closed executable program.
        resource_request (bool): Whether ``payload`` is the canonical wrapper
            carrying declarative external resource rules.
        rules (tuple[ResourceRule, ...]): The same identity-ordered checked rules
            encoded in a resource-analysis request.
        observation (bool): Use the native call-boundary observation projection.
        expanded (tuple[str, ...] | None): Expanded callable identities, or all.
        basis (str | None): Explicit portable alphabet, or no circuit pricing.
        realizations (tuple[str, ...]): Compiler-owned operation implementations.

    Returns:
        _NativeLogicalProjection: Validated symbolic native facts.

    Raises:
        CapabilityError: If the native compiler is unavailable, lacks the
            required typed entrypoint, or rejects the verified program.
        VerificationError: If the payload is not valid semantic QCore.
        TypeError: If an installed native extension violates its result ABI.
        ValueError: If the extension returns an invalid resource expression.
    """

    if observation:
        entrypoint = "inspect_symbolic_resource_observation"
    elif resource_request:
        entrypoint = "inspect_symbolic_resource_analysis_request"
    elif open_family:
        entrypoint = "inspect_symbolic_program_family_counts"
    else:
        entrypoint = "inspect_symbolic_logical_counts"
    inspect = getattr(native, entrypoint, None)
    if not callable(inspect):
        raise CapabilityError(
            Diagnostic(
                "Q4214",
                f"the native compiler does not provide {entrypoint}",
            )
        )
    try:
        result = (
            inspect(payload, expanded, basis, list(realizations))
            if observation
            else inspect(payload)
        )
    except ValueError as error:
        raise _native_resource_error(str(error)) from error
    expressions = _decode_native_resource_expressions(
        getattr(result, "expressions", None)
    )
    raw_unknown = getattr(result, "unknown", None)
    if (
        not isinstance(raw_unknown, list)
        or any(not isinstance(reason, str) or not reason for reason in raw_unknown)
        or raw_unknown != sorted(set(raw_unknown))
    ):
        raise TypeError("native symbolic unknown contributors violate the Python ABI")
    rule_index = {rule.identity: rule for rule in rules}

    def decode_count(value: object) -> _Cost:
        """Decode one count with its native-composed rule usage."""

        return _decode_native_symbolic_count(value, expressions, rule_index)

    facts = _decode_native_analysis_facts(
        getattr(result, "facts", None),
        decode_count,
        rules,
    )

    work = decode_count(getattr(result, "work", None))
    raw_dependency_span = getattr(result, "dependency_span", None)
    dependency_span = (
        None if raw_dependency_span is None else decode_count(raw_dependency_span)
    )
    raw_opcodes = getattr(result, "opcodes", None)
    if not isinstance(raw_opcodes, list):
        raise TypeError("native symbolic opcode counts violate the Python ABI")
    opcodes: dict[str, _Cost] = {}
    for item in raw_opcodes:
        opcode = getattr(item, "opcode", None)
        if not isinstance(opcode, str) or not opcode or opcode in opcodes:
            raise TypeError("native symbolic opcode counts violate the Python ABI")
        opcodes[opcode] = decode_count(getattr(item, "count", None))
    raw_extra_metrics = getattr(result, "extra_metrics", None)
    if not isinstance(raw_extra_metrics, list):
        raise TypeError("native symbolic extra metrics violate the Python ABI")
    extra_metrics: dict[str, _Cost] = {}
    for item in raw_extra_metrics:
        identifier = getattr(item, "identifier", None)
        if (
            not isinstance(identifier, str)
            or "." not in identifier
            or identifier in extra_metrics
        ):
            raise TypeError("native symbolic extra metrics violate the Python ABI")
        extra_metrics[identifier] = decode_count(getattr(item, "count", None))
    raw_families = getattr(result, "families", None)
    if raw_families is not None and not isinstance(raw_families, list):
        raise TypeError("native symbolic family counts violate the Python ABI")
    families: dict[str, _Cost] = {}
    for item in raw_families or ():
        family = getattr(item, "family", None)
        if (
            not isinstance(family, str)
            or family not in _gate_family_metrics()
            or family in families
        ):
            raise TypeError("native symbolic family counts violate the Python ABI")
        families[family] = decode_count(getattr(item, "count", None))
    raw_measurements = getattr(result, "measurements", None)
    raw_resets = getattr(result, "resets", None)
    measurements = None if raw_measurements is None else decode_count(raw_measurements)
    resets = None if raw_resets is None else decode_count(raw_resets)
    missing = object()
    raw_input_qubits = getattr(result, "input_qubits", missing)
    raw_allocated_qubits = getattr(result, "allocated_qubits", missing)
    raw_circuit_qubits = getattr(result, "circuit_qubits", missing)
    raw_peak_qubits = getattr(result, "peak_qubits", missing)
    raw_depths = getattr(result, "depths", missing)
    raw_cleanup_work = getattr(result, "cleanup_work", missing)
    if (
        raw_input_qubits is missing
        or raw_allocated_qubits is missing
        or raw_circuit_qubits is missing
        or raw_peak_qubits is missing
        or raw_depths is missing
        or raw_cleanup_work is missing
    ):
        raise TypeError("native symbolic result violates the Python ABI")
    input_qubits = None if raw_input_qubits is None else decode_count(raw_input_qubits)
    allocated_qubits = (
        None if raw_allocated_qubits is None else decode_count(raw_allocated_qubits)
    )
    circuit_qubits = (
        None if raw_circuit_qubits is None else decode_count(raw_circuit_qubits)
    )
    peak_qubits = None if raw_peak_qubits is None else decode_count(raw_peak_qubits)
    if raw_depths is not None and not isinstance(raw_depths, list):
        raise TypeError("native symbolic depths violate the Python ABI")
    depths: dict[str, _Cost] = {}
    for item in raw_depths or ():
        depth_family = getattr(item, "family", None)
        if (
            not isinstance(depth_family, str)
            or depth_family not in _NATIVE_DEPTH_METRICS
            or depth_family in depths
        ):
            raise TypeError("native symbolic depths violate the Python ABI")
        depths[depth_family] = decode_count(getattr(item, "count", None))
    if raw_depths is not None and depths.keys() != _NATIVE_DEPTH_METRICS.keys():
        raise TypeError("native symbolic depths violate the Python ABI")
    cleanup_work = decode_count(raw_cleanup_work)
    return _NativeLogicalProjection(
        work=work,
        dependency_span=dependency_span,
        opcodes=dict(sorted(opcodes.items())),
        extra_metrics=dict(sorted(extra_metrics.items())),
        families=(None if raw_families is None else dict(sorted(families.items()))),
        measurements=measurements,
        resets=resets,
        input_qubits=input_qubits,
        allocated_qubits=allocated_qubits,
        circuit_qubits=circuit_qubits,
        peak_qubits=peak_qubits,
        depths=None if raw_depths is None else dict(sorted(depths.items())),
        cleanup_work=cleanup_work,
        unknown=tuple(raw_unknown),
        facts=facts,
    )


def _native_resource_error(message: str) -> Exception:
    """Map one stable native diagnostic to the matching public error family.

    Args:
        message (str): Native user-facing diagnostic text.

    Returns:
        Exception: A public failure carrying the original stable code.
    """

    code, separator, detail = message.partition(":")
    if separator and len(code) == 5 and code.startswith("Q") and code[1:].isdigit():
        if code.startswith("Q422"):
            return ValueError(message)
        if code.startswith("Q11"):
            return VerificationError(Diagnostic(code, detail.strip()))
        return CapabilityError(Diagnostic(code, detail.strip()))
    return CapabilityError(
        Diagnostic("Q4214", f"native symbolic resource analysis failed: {message}")
    )


def _decode_native_analysis_facts(
    raw_facts: object,
    decode_count: Callable[[object], _Cost],
    rules: tuple[ResourceRule, ...],
) -> tuple[AnalysisFact, ...]:
    """Decode canonical source-local facts from one native symbolic result.

    Args:
        raw_facts (object): Candidate fact sequence returned by PyO3.
        decode_count (Callable[[object], _Cost]): Decoder sharing the result's
            validated expression table and rule identity map.
        rules (tuple[ResourceRule, ...]): Checked rules from the same request.

    Returns:
        tuple[AnalysisFact, ...]: Facts ready for the persistent analysis bundle.

    Raises:
        TypeError: If fact identity, named counts, or metric declarations violate
            the native ABI.
    """

    if not isinstance(raw_facts, list):
        raise TypeError("native symbolic analysis facts violate the Python ABI")
    metric_definitions: dict[str, MetricDefinition] = {}
    for rule in rules:
        previous = metric_definitions.setdefault(rule.metric.identifier, rule.metric)
        if previous != rule.metric:
            raise TypeError("resource rules disagree on one metric definition")

    def named_counts(
        items: object,
        attribute: str,
        label: str,
    ) -> dict[str, _Cost]:
        """Decode one canonical named-count sequence."""

        if not isinstance(items, list):
            raise TypeError(f"native symbolic fact {label} violate the Python ABI")
        result: dict[str, _Cost] = {}
        for item in items:
            name = getattr(item, attribute, None)
            if not isinstance(name, str) or not name or name in result:
                raise TypeError(f"native symbolic fact {label} violate the Python ABI")
            result[name] = decode_count(getattr(item, "count", None))
        return result

    facts: list[AnalysisFact] = []
    identities: set[tuple[str, str, str]] = set()
    for raw in raw_facts:
        owner = getattr(raw, "owner", None)
        source = getattr(raw, "source", None)
        opcode = getattr(raw, "opcode", None)
        if (
            not isinstance(owner, str)
            or not owner
            or not isinstance(source, str)
            or not source
            or not isinstance(opcode, str)
            or not opcode
        ):
            raise TypeError("native symbolic fact identity violates the Python ABI")
        identity = (owner, source, opcode)
        if identity in identities:
            raise TypeError(
                "native symbolic facts contain a duplicate lexical identity"
            )
        identities.add(identity)
        operations = named_counts(getattr(raw, "opcodes", None), "opcode", "opcodes")
        extras = named_counts(
            getattr(raw, "extra_metrics", None),
            "identifier",
            "extra metrics",
        )
        raw_families = getattr(raw, "families", None)
        families = (
            {}
            if raw_families is None
            else named_counts(raw_families, "family", "families")
        )
        if any(family not in _gate_family_metrics() for family in families):
            raise TypeError("native symbolic fact family violates the Python ABI")
        extra_metrics: list[tuple[MetricDefinition, _Cost]] = []
        for identifier, cost in sorted(extras.items()):
            metric = metric_definitions.get(identifier)
            if metric is None:
                try:
                    metric = _native_extra_metric_definition(identifier, rules)
                except TypeError as error:
                    raise TypeError(
                        "native symbolic fact names an undeclared extra metric"
                    ) from error
            extra_metrics.append((metric, cost))
        raw_measurements = getattr(raw, "measurements", None)
        raw_resets = getattr(raw, "resets", None)
        facts.append(
            AnalysisFact(
                "operation",
                owner,
                source,
                opcode,
                _fact_contributions(
                    work=decode_count(getattr(raw, "work", None)),
                    measurements=(
                        None
                        if raw_measurements is None
                        else decode_count(raw_measurements)
                    ),
                    resets=(None if raw_resets is None else decode_count(raw_resets)),
                    cleanup=decode_count(getattr(raw, "cleanup_work", None)),
                    operations=tuple(sorted(operations.items())),
                    gate_families=tuple(sorted(families.items())),
                    extra_metrics=tuple(extra_metrics),
                ),
            )
        )
    return tuple(facts)


def _decode_native_resource_expressions(
    nodes: object,
) -> tuple[Expr | Predicate, ...]:
    """Reconstruct a topologically ordered native resource-expression DAG.

    Args:
        nodes (object): Sequence of typed expression-node projections returned
            by the required native compiler.

    Returns:
        tuple[Expr | Predicate, ...]: Validated Python expression and predicate
            nodes indexed by native node ID.

    Raises:
        TypeError: If a node or reference violates the bridge ABI.
        ValueError: If a node violates the closed expression algebra.
    """

    if not isinstance(nodes, list):
        raise TypeError("native symbolic expressions violate the Python ABI")
    expressions: list[Expr | Predicate] = []
    for node in nodes:
        kind = getattr(node, "kind", None)
        domain_value = getattr(node, "domain", None)
        raw_arguments = getattr(node, "arguments", None)
        raw_predicates = getattr(node, "predicates", [])
        if (
            not isinstance(kind, str)
            or not isinstance(domain_value, str)
            or not isinstance(raw_arguments, list)
            or not isinstance(raw_predicates, list)
        ):
            raise TypeError("native symbolic expression node violates the Python ABI")
        arguments = tuple(
            _native_expression_reference(identifier, expressions)
            for identifier in raw_arguments
        )
        predicates = tuple(
            _native_predicate_reference(identifier, expressions)
            for identifier in raw_predicates
        )
        domain = Domain(domain_value)
        expression = _decode_native_resource_node(
            node,
            kind,
            domain,
            arguments,
            predicates,
            expressions,
        )
        if isinstance(expression, Predicate) and domain is not Domain.BOOLEAN:
            raise ValueError("native symbolic predicate domain is inconsistent")
        expressions.append(expression)
    return tuple(expressions)


def _native_expression_reference(
    identifier: object,
    expressions: list[Expr | Predicate],
) -> Expr:
    """Resolve one backward-only native DAG reference.

    Args:
        identifier (object): Candidate unsigned expression-node identifier.
        expressions (list[Expr | Predicate]): Previously decoded topological nodes.

    Returns:
        Expr: The referenced earlier expression.

    Raises:
        TypeError: If the identifier is boolean, nonintegral, or forward-facing.
    """

    if (
        isinstance(identifier, bool)
        or not isinstance(identifier, int)
        or identifier < 0
        or identifier >= len(expressions)
    ):
        raise TypeError("native symbolic expression has an invalid DAG reference")
    expression = expressions[identifier]
    if not isinstance(expression, Expr):
        raise TypeError("native symbolic expression references a predicate node")
    return expression


def _native_predicate_reference(
    identifier: object,
    expressions: list[Expr | Predicate],
) -> Predicate:
    """Resolve one backward-only native predicate-node reference."""

    if (
        isinstance(identifier, bool)
        or not isinstance(identifier, int)
        or identifier < 0
        or identifier >= len(expressions)
    ):
        raise TypeError("native symbolic predicate has an invalid DAG reference")
    predicate = expressions[identifier]
    if not isinstance(predicate, Predicate):
        raise TypeError("native symbolic predicate references an expression node")
    return predicate


def _decode_native_resource_node(
    node: object,
    kind: str,
    domain: Domain,
    arguments: tuple[Expr, ...],
    predicates: tuple[Predicate, ...],
    expressions: list[Expr | Predicate],
) -> Expr | Predicate:
    """Decode one allowlisted native resource-expression constructor.

    Args:
        node (object): Typed native node projection.
        kind (str): Closed constructor spelling.
        domain (Domain): Declared result domain.
        arguments (tuple[Expr, ...]): Resolved operand expressions.
        predicates (tuple[Predicate, ...]): Resolved predicate operands.
        expressions (list[Expr | Predicate]): Earlier DAG nodes used to resolve
            a binder.

    Returns:
        Expr | Predicate: One validated node in the Python resource algebra.

    Raises:
        TypeError: If constructor metadata violates the bridge ABI.
        ValueError: If the constructor is unknown or semantically invalid.
    """

    if kind in {"true", "false"}:
        if arguments or predicates:
            raise TypeError("native symbolic predicate has unexpected operands")
        return Predicate._from_native(kind, (), ())
    if kind in {"and", "or"}:
        if arguments or not predicates:
            raise TypeError("native symbolic connective violates the Python ABI")
        return Predicate._from_native(kind, (), predicates)
    if kind == "not":
        if arguments or len(predicates) != 1:
            raise TypeError("native symbolic negation violates the Python ABI")
        return Predicate._from_native("not", (), predicates)
    if kind in {
        "equal",
        "not_equal",
        "less",
        "less_equal",
        "greater",
        "greater_equal",
        "divisible",
    }:
        if len(arguments) != 2 or predicates:
            raise TypeError("native symbolic comparison violates the Python ABI")
        return Predicate._from_native(kind, arguments, ())
    if kind == "literal":
        if arguments:
            raise TypeError("native symbolic literal has unexpected operands")
        literal_kind = getattr(node, "literal_kind", None)
        if literal_kind == "integer":
            raw = getattr(node, "literal", None)
            if not isinstance(raw, str):
                raise TypeError(
                    "native symbolic integer literal violates the Python ABI"
                )
            value = int(raw)
            if str(value) != raw:
                raise ValueError("native symbolic integer literal is not canonical")
            return Expr._from_native("literal", domain, ONE, value=value)
        if literal_kind == "rational":
            numerator = getattr(node, "rational_numerator", None)
            denominator = getattr(node, "rational_denominator", None)
            if not isinstance(numerator, str) or not isinstance(denominator, str):
                raise TypeError(
                    "native symbolic rational literal violates the Python ABI"
                )
            value = Fraction(int(numerator), int(denominator))
            if (
                str(value.numerator) != numerator
                or str(value.denominator) != denominator
            ):
                raise ValueError("native symbolic rational literal is not canonical")
            return Expr._from_native("literal", domain, ONE, value=value)
        if literal_kind == "real":
            bits = getattr(node, "real_bits", None)
            if (
                isinstance(bits, bool)
                or not isinstance(bits, int)
                or bits < 0
                or bits >= 1 << 64
            ):
                raise TypeError("native symbolic real literal violates the Python ABI")
            value = struct.unpack(">d", bits.to_bytes(8, "big"))[0]
            return Expr._from_native("literal", domain, ONE, value=value)
        if literal_kind is not None:
            raise TypeError("native symbolic literal violates the Python ABI")
        raise TypeError("native symbolic literal has no payload discriminator")
    if kind == "symbol":
        if arguments:
            raise TypeError("native symbolic symbol has unexpected operands")
        fields = (
            getattr(node, "symbol_scope", None),
            getattr(node, "symbol_name", None),
            getattr(node, "symbol_domain", None),
            getattr(node, "symbol_origin", None),
            getattr(node, "symbol_constraints", None),
        )
        scope, name, symbol_domain, origin, constraints = fields
        if (
            not isinstance(scope, str)
            or not isinstance(name, str)
            or not isinstance(symbol_domain, str)
            or not isinstance(origin, str)
            or not isinstance(constraints, list)
            or any(not isinstance(item, str) for item in constraints)
        ):
            raise TypeError("native symbolic declaration violates the Python ABI")
        symbol = Symbol._from_native(
            scope,
            name,
            Domain(symbol_domain),
            ONE,
            origin,
            tuple(constraints),
        )
        return Expr._from_native("symbol", domain, ONE, symbol=symbol)
    if kind in {"add", "multiply", "minimum", "maximum"}:
        if len(arguments) < 2:
            raise TypeError(
                f"native symbolic {kind} expression has insufficient operands"
            )
        return Expr._from_native(kind, domain, ONE, arguments)
    if kind == "power" and len(arguments) == 2:
        return Expr._from_native(kind, domain, ONE, arguments)
    if kind == "absolute" and len(arguments) == 1:
        return Expr._from_native(kind, domain, ONE, arguments)
    if kind == "floor" and len(arguments) == 1:
        return Expr._from_native(kind, domain, ONE, arguments)
    if kind == "ceiling" and len(arguments) == 1:
        return Expr._from_native(kind, domain, ONE, arguments)
    if kind == "floor_divide" and len(arguments) == 2:
        return Expr._from_native(kind, domain, ONE, arguments)
    if kind == "ceiling_divide" and len(arguments) == 2:
        return Expr._from_native(kind, domain, ONE, arguments)
    if kind in {"exp", "log2", "square_root"} and len(arguments) == 1:
        return Expr._from_native(kind, domain, ONE, arguments)
    if kind == "log" and len(arguments) in {1, 2}:
        return Expr._from_native(kind, domain, ONE, arguments)
    if kind == "piecewise" and len(arguments) == len(predicates) + 1:
        return Expr._from_native(
            kind,
            domain,
            ONE,
            arguments,
            predicates=predicates,
        )
    if kind == "contracted_function":
        name = getattr(node, "function_name", None)
        if not isinstance(name, str):
            raise TypeError("native symbolic function violates the Python ABI")
        return Expr._from_native(
            kind,
            domain,
            ONE,
            arguments,
            attributes=(("name", name), ("version", "1")),
        )
    if kind in {"recurrence", "recurrence_equation"}:
        name = getattr(node, "function_name", None)
        if not isinstance(name, str):
            raise TypeError("native symbolic recurrence violates the Python ABI")
        return Expr._from_native(
            kind,
            domain,
            ONE,
            arguments,
            attributes=(("name", name), ("version", "1")),
            predicates=predicates,
        )
    if kind == "recurrence_system":
        return Expr._from_native(kind, domain, ONE, arguments)
    if kind == "bounded_sum" and len(arguments) == 3:
        binder_id = getattr(node, "binder", None)
        binder_expression = _native_expression_reference(binder_id, expressions)
        if binder_expression.symbol is None:
            raise TypeError("native symbolic bounded sum has no binder declaration")
        binder = binder_expression.symbol
        return Expr._from_native(
            kind,
            domain,
            ONE,
            arguments,
            attributes=_binder_attributes(binder),
        )
    if kind == "bounded_product" and len(arguments) == 3:
        binder_id = getattr(node, "binder", None)
        binder_expression = _native_expression_reference(binder_id, expressions)
        if binder_expression.symbol is None:
            raise TypeError("native symbolic bounded product has no binder declaration")
        binder = binder_expression.symbol
        return Expr._from_native(
            kind,
            domain,
            ONE,
            arguments,
            attributes=_binder_attributes(binder),
        )
    raise ValueError(f"unknown native resource expression constructor {kind!r}")


def _decode_native_symbolic_count(
    value: object,
    expressions: tuple[Expr | Predicate, ...],
    rules: dict[str, ResourceRule] | None = None,
) -> _Cost:
    """Decode one symbolic lower and optional upper endpoint.

    Args:
        value (object): Typed native symbolic-count projection.
        expressions (tuple[Expr | Predicate, ...]): Validated node table.
        rules (dict[str, ResourceRule] | None): Checked rules indexed by the
            identities encoded in the same native analysis request.

    Returns:
        _Cost: Internal nonnegative resource enclosure.

    Raises:
        TypeError: If endpoint identifiers or exactness violate the bridge ABI.
    """

    lower_id = getattr(value, "lower", None)
    upper_id = getattr(value, "upper", None)
    exact = getattr(value, "exact", None)
    unknown = getattr(value, "unknown", None)
    rule_identities = getattr(value, "rule_identities", None)
    relation_name = getattr(value, "relation", None)
    dependency_span_relation_name = getattr(
        value,
        "dependency_span_relation",
        None,
    )
    table = list(expressions)
    lower = _native_expression_reference(lower_id, table)
    upper = None if upper_id is None else _native_expression_reference(upper_id, table)
    if (
        not isinstance(exact, bool)
        or (exact and upper_id != lower_id)
        or not isinstance(unknown, list)
        or any(not isinstance(reason, str) or not reason for reason in unknown)
        or not isinstance(rule_identities, list)
        or any(
            not isinstance(identity, str) or not identity
            for identity in rule_identities
        )
        or rule_identities != sorted(set(rule_identities))
    ):
        raise TypeError("native symbolic count violates the Python ABI")
    try:
        relation = Relation(relation_name)
        dependency_span_relation = Relation(dependency_span_relation_name)
    except (TypeError, ValueError) as error:
        raise TypeError("native symbolic count violates the Python ABI") from error
    supported_relations = {
        Relation.EQUALITY,
        Relation.LOWER_BOUND,
        Relation.UPPER_BOUND,
        Relation.ENCLOSURE,
        Relation.PARTIAL_ENCLOSURE,
    }
    if (
        relation not in supported_relations
        or dependency_span_relation not in supported_relations
    ):
        raise TypeError("native symbolic count violates the Python ABI")
    evidence: list[Evidence] = []
    assumptions: list[str] = []
    if rules is not None:
        for identity in rule_identities:
            rule = rules.get(identity)
            if rule is None:
                raise TypeError(
                    "native symbolic count names an undeclared resource rule"
                )
            metadata = _rule_cost(rule)
            evidence.extend(metadata.evidence)
            assumptions.extend(metadata.assumptions)
    return _Cost(
        lower,
        upper,
        exact,
        relation,
        dependency_span_relation,
        tuple(dict.fromkeys(unknown)),
        tuple(dict.fromkeys(evidence)),
        tuple(dict.fromkeys(assumptions)),
    )


def _resolve_source(
    source: object,
) -> tuple[Module | Program, str, bool]:
    """Normalize public program values without importing the source module."""

    if isinstance(source, Program):
        return (
            source,
            source.entrypoint,
            True,
        )
    if isinstance(source, Module):
        if len(source.functions) != 1:
            raise ValueError("logical(Module) is ambiguous; use module[entrypoint]")
        return source, source.functions[0].symbol, False
    module = getattr(source, "module", None)
    entrypoint = getattr(source, "entrypoint", None)
    if isinstance(module, Module) and isinstance(entrypoint, str):
        return module, entrypoint, False
    module_factory = getattr(source, "module", None)
    function_name = getattr(source, "function_name", None)
    if callable(module_factory) and isinstance(function_name, str):
        module = module_factory()
        if isinstance(module, Module):
            return module, function_name, False
    raise TypeError("logical expects ProgramFamily, QKernel, Module, or Program")


def _normalize_resource_rules(
    rules: ResourceRule | tuple[ResourceRule, ...] | list[ResourceRule],
) -> tuple[ResourceRule, ...]:
    """Freeze the public rule collection in deterministic identity order."""

    values = (rules,) if isinstance(rules, ResourceRule) else tuple(rules)
    if any(not isinstance(rule, ResourceRule) for rule in values):
        raise TypeError("rules= accepts ResourceRule values")
    return tuple(sorted(values, key=lambda rule: rule.identity))


def _resource_value(
    metric: MetricDefinition,
    cost: _Cost,
    *,
    relation: Relation,
) -> ResourceValue:
    """Project native endpoints using their compiler-owned bound relation."""

    lower = _native_endpoint(cost.lower, metric)
    upper = (
        None
        if cost.upper is None
        else lower
        if cost.upper is cost.lower
        else _native_endpoint(cost.upper, metric)
    )
    evidence = tuple(dict.fromkeys((*_STRUCTURAL_EVIDENCE, *cost.evidence)))
    if relation is Relation.EQUALITY:
        return ResourceValue.equality(
            metric,
            lower,
            evidence=evidence,
            assumptions=cost.assumptions,
        )
    if relation is Relation.UPPER_BOUND:
        if upper is None:
            raise TypeError("native symbolic count relation violates the Python ABI")
        return ResourceValue(
            metric,
            Relation.UPPER_BOUND,
            expression=upper,
            evidence=evidence,
            assumptions=cost.assumptions,
        )
    if relation is Relation.LOWER_BOUND:
        return ResourceValue.lower_bound(
            metric,
            lower,
            evidence=evidence,
            assumptions=cost.assumptions,
        )
    if relation is Relation.PARTIAL_ENCLOSURE:
        return ResourceValue.partial(
            metric,
            lower=lower,
            upper=upper,
            unknown=cost.unknown,
            evidence=evidence,
            assumptions=cost.assumptions,
        )
    if relation is Relation.ENCLOSURE:
        if upper is None:
            raise TypeError("native symbolic count relation violates the Python ABI")
        return ResourceValue.enclosure(
            metric,
            lower,
            upper,
            evidence=evidence,
            assumptions=cost.assumptions,
        )
    raise TypeError("native symbolic count relation violates the Python ABI")


def _native_endpoint(expression: Expr, metric: MetricDefinition) -> Expr:
    """Attach a unit and retain a closed native result as one typed literal."""

    numeric = expression.numeric()
    if numeric is not None:
        return literal(numeric, metric.unit, expression.domain)
    return _with_unit(expression, metric)


def _native_resource_value(
    metric: MetricDefinition,
    cost: _Cost,
) -> ResourceValue:
    """Project native endpoints and their natively composed rule provenance."""

    return _resource_value(
        metric,
        cost,
        relation=(
            cost.dependency_span_relation
            if metric.composition is Composition.DEPENDENCY_SPAN
            else cost.relation
        ),
    )


def _with_unit(expression: Expr, metric: MetricDefinition) -> Expr:
    """Attach a metric unit through the native resource-expression transform."""

    from ._rule_wire import (
        decode_resource_expression_graph,
        encode_resource_expression_graph,
    )

    retag = getattr(native, "retag_resource_expression_unit", None)
    if not callable(retag):
        raise RuntimeError(
            "Q4243: native resource-expression unit projection is unavailable"
        )
    payload = retag(
        encode_resource_expression_graph(expression),
        metric.unit.name,
    )
    if not isinstance(payload, bytes):
        raise TypeError(
            "Q4243: native resource-expression unit projection returned malformed data"
        )
    try:
        result = decode_resource_expression_graph(payload)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "Q4243: native resource-expression unit projection returned malformed data"
        ) from error
    if not isinstance(result, Expr):
        raise TypeError(
            "Q4243: native resource-expression unit projection returned a predicate"
        )
    return result


__all__ = ["logical"]
