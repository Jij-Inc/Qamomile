"""Compose measurement-derived classical-value provenance."""

from __future__ import annotations

from collections.abc import Mapping

import sympy as sp
from sympy.logic.boolalg import Boolean

from qamomile.circuit.estimator._resource_expressions import _boolean_condition


def _merge_measurement_taint_conditions(
    *mappings: Mapping[str, Boolean],
) -> dict[str, Boolean]:
    """Merge guarded measurement provenance by SSA value identity.

    Args:
        *mappings (Mapping[str, Boolean]): UUID-keyed provenance conditions.

    Returns:
        dict[str, Boolean]: Conditions combined with Boolean OR, excluding
            values whose merged condition is definitely false.
    """
    merged: dict[str, Boolean] = {}
    for mapping in mappings:
        for uuid, condition in mapping.items():
            combined = _boolean_condition(sp.Or(merged.get(uuid, sp.false), condition))
            if combined is sp.false:
                merged.pop(uuid, None)
            else:
                merged[uuid] = combined
    return merged


def _guard_measurement_taint_conditions(
    mapping: Mapping[str, Boolean],
    condition: sp.Basic,
) -> dict[str, Boolean]:
    """Conjoin one activation condition with every taint entry.

    Args:
        mapping (Mapping[str, Boolean]): UUID-keyed provenance conditions.
        condition (sp.Basic): Additional activation predicate.

    Returns:
        dict[str, Boolean]: Guarded non-false provenance entries.
    """
    active = _boolean_condition(condition)
    return {
        uuid: guarded
        for uuid, value in mapping.items()
        if (guarded := _boolean_condition(sp.And(active, value))) is not sp.false
    }


def _conditional_measurement_taint_conditions(
    when_true: Mapping[str, Boolean],
    when_false: Mapping[str, Boolean],
    condition: sp.Basic,
) -> dict[str, Boolean]:
    """Select guarded provenance from two compile-time branches.

    Args:
        when_true (Mapping[str, Boolean]): True-branch provenance.
        when_false (Mapping[str, Boolean]): False-branch provenance.
        condition (sp.Basic): Branch selection predicate.

    Returns:
        dict[str, Boolean]: Branch-guarded union of both mappings.
    """
    predicate = _boolean_condition(condition)
    return _merge_measurement_taint_conditions(
        _guard_measurement_taint_conditions(when_true, predicate),
        _guard_measurement_taint_conditions(when_false, sp.Not(predicate)),
    )
