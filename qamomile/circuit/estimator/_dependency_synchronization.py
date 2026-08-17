"""Represent grouped synchronized-entry scheduling certificates."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable

import sympy as sp
from sympy.logic.boolalg import Boolean

from qamomile.circuit.estimator._dependency_indices import WireKey
from qamomile.circuit.estimator._resource_expressions import _boolean_condition


@dataclasses.dataclass(frozen=True)
class _SynchronizedEntryCertificate:
    """Keep one synchronized-entry premise grouped by physical footprint.

    ``coverage`` is the complete caller-visible wire set that one trusted
    uniform event must write to discharge the premise. ``frontier`` is the
    subset whose arbitrary prior readiness is safe because the first serial
    gate consumes it. Keeping both sets in one immutable value prevents a
    frontier from one loop from exempting a late wire in another loop.

    Args:
        coverage (frozenset[WireKey]): Complete physical footprint protected
            by this certificate.
        frontier (frozenset[WireKey]): Exact physical scalar inputs consumed
            by the first serial gate.
        active_when (Boolean): Guard under which the premise is required.
    """

    coverage: frozenset[WireKey]
    frontier: frozenset[WireKey]
    active_when: Boolean

    def __post_init__(self) -> None:
        """Normalize immutable sets and the Boolean activation guard."""
        object.__setattr__(self, "coverage", frozenset(self.coverage))
        object.__setattr__(self, "frontier", frozenset(self.frontier))
        object.__setattr__(
            self,
            "active_when",
            _boolean_condition(self.active_when),
        )

    def when(self, condition: sp.Basic) -> _SynchronizedEntryCertificate | None:
        """Guard this certificate by an additional condition.

        Args:
            condition (sp.Basic): Additional execution predicate.

        Returns:
            _SynchronizedEntryCertificate | None: Guarded certificate, or
                ``None`` when the combined guard is identically false.
        """
        active = _boolean_condition(
            sp.And(_boolean_condition(condition), self.active_when)
        )
        if active is sp.false:
            return None
        return dataclasses.replace(self, active_when=active)

    def mapped(
        self,
        key_fn: Callable[[WireKey], WireKey],
        guard_fn: Callable[[sp.Basic], sp.Basic],
        *,
        preserve_frontier: bool = True,
    ) -> _SynchronizedEntryCertificate | None:
        """Rewrite physical keys and the activation guard.

        Args:
            key_fn (Callable[[WireKey], WireKey]): Callable mapping one
                physical key to one key.
            guard_fn (Callable[[sp.Basic], sp.Basic]): Callable rewriting the
                Boolean activation guard.
            preserve_frontier (bool): Whether exact scalar frontier keys may
                survive the mapping. Defaults to ``True``.

        Returns:
            _SynchronizedEntryCertificate | None: Rewritten certificate, or
                ``None`` when its guard becomes false.
        """
        active = _boolean_condition(guard_fn(self.active_when))
        if active is sp.false:
            return None
        return _SynchronizedEntryCertificate(
            coverage=frozenset(key_fn(key) for key in self.coverage),
            frontier=(
                frozenset(key_fn(key) for key in self.frontier)
                if preserve_frontier
                else frozenset()
            ),
            active_when=active,
        )

    def without_frontier(self) -> _SynchronizedEntryCertificate:
        """Return a fail-closed certificate with no safe entry frontier.

        Returns:
            _SynchronizedEntryCertificate: Certificate retaining its full
                reset coverage and activation guard.
        """
        if not self.frontier:
            return self
        return dataclasses.replace(self, frontier=frozenset())


def _merge_synchronized_entry_certificates(
    *certificate_groups: Iterable[_SynchronizedEntryCertificate],
) -> tuple[_SynchronizedEntryCertificate, ...]:
    """Merge nonvacuous certificates without flattening distinct groups.

    Args:
        *certificate_groups (Iterable[_SynchronizedEntryCertificate]): Ordered
            certificate collections to combine.

    Returns:
        tuple[_SynchronizedEntryCertificate, ...]: Stable grouped
            certificates whose guards are not identically false. Certificates
            with identical coverage and frontier are coalesced by disjoining
            their activation guards.
    """
    keys: list[tuple[frozenset[WireKey], frozenset[WireKey]]] = []
    merged: dict[
        tuple[frozenset[WireKey], frozenset[WireKey]],
        _SynchronizedEntryCertificate,
    ] = {}
    for certificates in certificate_groups:
        for certificate in certificates:
            if not certificate.coverage or certificate.active_when is sp.false:
                continue
            key = (certificate.coverage, certificate.frontier)
            existing = merged.get(key)
            if existing is None:
                keys.append(key)
                merged[key] = certificate
                continue
            merged[key] = dataclasses.replace(
                existing,
                active_when=_boolean_condition(
                    sp.Or(existing.active_when, certificate.active_when)
                ),
            )
    return tuple(merged[key] for key in keys)


def _guard_synchronized_entry_certificates(
    certificates: Iterable[_SynchronizedEntryCertificate],
    condition: sp.Basic,
) -> tuple[_SynchronizedEntryCertificate, ...]:
    """Apply one execution guard to grouped entry certificates.

    Args:
        certificates (Iterable[_SynchronizedEntryCertificate]): Certificates
            to guard independently.
        condition (sp.Basic): Additional execution predicate.

    Returns:
        tuple[_SynchronizedEntryCertificate, ...]: Nonfalse guarded
            certificates in their original order.
    """
    guarded: list[_SynchronizedEntryCertificate] = []
    for certificate in certificates:
        mapped = certificate.when(condition)
        if mapped is not None:
            guarded.append(mapped)
    return _merge_synchronized_entry_certificates(guarded)


def _map_synchronized_entry_certificates(
    certificates: Iterable[_SynchronizedEntryCertificate],
    key_fn: Callable[[WireKey], WireKey],
    guard_fn: Callable[[sp.Basic], sp.Basic],
    *,
    preserve_frontier: bool = True,
) -> tuple[_SynchronizedEntryCertificate, ...]:
    """Rewrite grouped entry certificates through an exact key mapping.

    Args:
        certificates (Iterable[_SynchronizedEntryCertificate]): Certificates
            to rewrite independently.
        key_fn (Callable[[WireKey], WireKey]): Callable mapping one physical
            key to one key.
        guard_fn (Callable[[sp.Basic], sp.Basic]): Callable rewriting Boolean
            activation guards.
        preserve_frontier (bool): Whether the caller has proven that every
            frontier scalar has exactly one root-scalar image. Defaults to
            ``True``.

    Returns:
        tuple[_SynchronizedEntryCertificate, ...]: Rewritten nonfalse
            certificates in their original order.
    """
    mapped_certificates: list[_SynchronizedEntryCertificate] = []
    for certificate in certificates:
        mapped = certificate.mapped(
            key_fn,
            guard_fn,
            preserve_frontier=preserve_frontier,
        )
        if mapped is not None and mapped.coverage:
            mapped_certificates.append(mapped)
    return _merge_synchronized_entry_certificates(mapped_certificates)


def _clear_synchronized_entry_frontiers(
    certificates: Iterable[_SynchronizedEntryCertificate],
) -> tuple[_SynchronizedEntryCertificate, ...]:
    """Clear every safe frontier while retaining grouped reset coverage.

    Args:
        certificates (Iterable[_SynchronizedEntryCertificate]): Certificates
            crossing a temporal transform that invalidates their frontier.

    Returns:
        tuple[_SynchronizedEntryCertificate, ...]: Fail-closed certificates
            with unchanged coverage and activation guards.
    """
    return _merge_synchronized_entry_certificates(
        (
            certificate.without_frontier()
            for certificate in certificates
            if certificate.coverage and certificate.active_when is not sp.false
        )
    )


__all__: list[str] = []
