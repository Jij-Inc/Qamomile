"""Tests for the classical-provenance scheduler contract."""

from __future__ import annotations

import sympy as sp

from qamomile.circuit.estimator._dependency_footprints import (
    _classical_dependency_footprint,
    _classical_dependency_key,
)


def test_source_tokens_define_scheduler_reads_and_writes() -> None:
    """The scheduler uses only resolved observation-source tokens."""
    read_flag = sp.Symbol("read_flag", boolean=True)
    write_flag = sp.Symbol("write_flag", boolean=True)

    reads, writes, active = _classical_dependency_footprint(
        {"measurement_root": read_flag},
        {"derived_measurement": write_flag},
    )

    assert reads == frozenset({_classical_dependency_key("measurement_root")})
    assert writes == frozenset({_classical_dependency_key("derived_measurement")})
    assert sp.simplify_logic(sp.Equivalent(active, sp.Or(read_flag, write_flag)))


def test_empty_source_tokens_create_no_scheduler_dependency() -> None:
    """Empty provenance maps mean that no classical token participates."""

    reads, writes, active = _classical_dependency_footprint(
        {},
        {},
    )

    assert reads == frozenset()
    assert writes == frozenset()
    assert active is sp.false


def test_false_source_guards_do_not_create_dependencies() -> None:
    """Statically inactive source tokens do not constrain scheduling."""

    reads, writes, active = _classical_dependency_footprint(
        {"measurement_root": sp.false},
        {"derived_measurement": sp.false},
    )

    assert reads == frozenset()
    assert writes == frozenset()
    assert active is sp.false


def test_explicit_written_tokens_publish_only_observation_roots() -> None:
    """A newly produced source becomes ready without publishing its alias."""

    reads, writes, active = _classical_dependency_footprint(
        {},
        {"measurement_root": sp.true},
    )

    assert reads == frozenset()
    assert writes == frozenset({_classical_dependency_key("measurement_root")})
    assert active is sp.true
