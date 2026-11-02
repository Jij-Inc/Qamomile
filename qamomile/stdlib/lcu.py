# ruff: noqa: F821 -- dependent widths are closed into generated source.

"""Exact linear-combination-of-unitaries block encodings."""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence
from dataclasses import dataclass

import qamomile as qm
from qamomile._native import compiler as native

from ._structure import identity
from .block_encoding import (
    BlockEncoding,
    _positive_width,
    _validate_unary_unitary,
)
from .state_preparation import amplitude_encoding


@dataclass(frozen=True, slots=True)
class _LcuPlan:
    """Hold one checked Python projection of a native LCU plan."""

    terms: tuple[tuple[complex, qm.QKernel], ...]
    normalization: float
    signal_width: int
    case_count: int
    amplitudes: tuple[float, ...]
    phases: tuple[float, ...]


def lcu_block_encoding(
    terms: Sequence[tuple[complex, qm.QKernel]],
    *,
    system_width: int,
) -> BlockEncoding:
    r"""Encode a finite linear combination using PREPARE and SELECT.

    For ``A = sum_j coefficient_j U_j``, the returned unitary is
    ``PREPARE adjoint · SELECT · PREPARE`` and has leading block
    ``A / alpha``, where ``alpha = sum_j |coefficient_j|``. Complex
    coefficient phases are closed into their selected unitaries. Equal
    callables are combined before choosing the selector width, and unused
    selector states are padded by identity.

    The zero operator has normalization one and uses an exact signal flip, so
    its projected leading block is zero. No LCU descriptor, coefficient, or
    Python callback reaches QCore: the returned :class:`BlockEncoding` closes
    to ordinary state preparation, total select, adjoint, and product reshape.

    Args:
        terms: Ordered ``(coefficient, unary unitary)`` pairs.
        system_width: Positive coherent width shared by every term.

    Returns:
        A construction-independent exact block-encoding contract.

    Raises:
        TypeError: If the container, coefficient, callable, or width has the
            wrong type, or the required native planner is unavailable.
        ValueError: If a coefficient is nonfinite, a callable has the wrong
            width, exact coefficient or normalization arithmetic exceeds the
            binary64 domain, or bounded plan materialization is exceeded.
    """

    width = _positive_width(system_width, "system_width")
    plan = _native_lcu_plan(terms, width)
    if not plan.terms:
        return BlockEncoding(
            _zero_encoding(width),
            normalization=1.0,
            signal_width=1,
            system_width=width,
        )

    preparation = amplitude_encoding(
        plan.amplitudes + (0.0,) * (plan.case_count - len(plan.terms))
    )
    identity_case = identity
    cases = tuple(
        _with_phase(unitary, width=width, phase=phase)
        for (_, unitary), phase in zip(plan.terms, plan.phases, strict=True)
    ) + (identity_case,) * (plan.case_count - len(plan.terms))
    signal_width = plan.signal_width
    total_width = signal_width + width

    @qm.qkernel
    def encoded(
        state: qm.Q[qm.Bits[total_width]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[total_width]]:  # type: ignore[valid-type]
        """Apply the configured linear-combination block encoding.

        Args:
            state (qm.Q[qm.Bits[total_width]]): Signal and system state.

        Returns:
            qm.Q[qm.Bits[total_width]]: Encoded signal and system state.
        """

        signal, system = qm.split_at(state, signal_width)
        signal = preparation(signal)
        signal, system = qm.multiplex(signal, cases, system)
        signal = qm.adjoint(preparation, signal)
        return qm.concat(signal, system)

    return BlockEncoding(
        encoded,
        normalization=plan.normalization,
        signal_width=signal_width,
        system_width=width,
    )


def _native_lcu_plan(
    terms: object,
    system_width: int,
) -> _LcuPlan:
    """Validate callable contracts and project one native coefficient plan."""

    if isinstance(terms, (str, bytes)) or not isinstance(terms, Sequence):
        raise TypeError("terms must be a sequence of (coefficient, unitary) pairs")
    input_limit, case_limit = _native_lcu_budgets()
    if len(terms) > input_limit:
        raise ValueError(f"Q7612: LCU input exceeds the {input_limit}-term budget")
    groups: dict[qm.QKernel, int] = {}
    unitaries: list[qm.QKernel] = []
    rows: list[tuple[int, float, float]] = []
    for term in terms:
        if not isinstance(term, (tuple, list)) or len(term) != 2:
            raise TypeError("each LCU term must be a coefficient/callable pair")
        coefficient, unitary = term
        if isinstance(coefficient, bool) or not isinstance(
            coefficient, numbers.Complex
        ):
            raise TypeError("each LCU coefficient must be a complex scalar")
        value = complex(coefficient)
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ValueError("each LCU coefficient must be finite")
        checked = _validate_unary_unitary(
            unitary,
            system_width,
            label="LCU term",
        )
        try:
            group = groups[checked]
        except KeyError:
            group = len(unitaries)
            groups[checked] = group
            unitaries.append(checked)
        rows.append((group, value.real, value.imag))
    derive = getattr(native, "plan_lcu_coefficients", None)
    if not callable(derive):
        raise TypeError("Q7610: native LCU coefficient planning is unavailable")
    plan = derive(len(unitaries), rows)
    return _decode_native_lcu_plan(plan, tuple(unitaries), case_limit)


def _native_lcu_budgets() -> tuple[int, int]:
    """Return compiler-owned LCU input and selector limits."""

    input_limit = getattr(native, "MAX_LCU_INPUT_TERMS", None)
    case_limit = getattr(native, "MAX_LCU_CASE_COUNT", None)
    if (
        type(input_limit) is not int
        or input_limit <= 0
        or type(case_limit) is not int
        or case_limit < 2
        or case_limit & (case_limit - 1)
    ):
        raise TypeError("Q7610: native LCU coefficient planning is unavailable")
    return input_limit, case_limit


def _decode_native_lcu_plan(
    plan: object,
    unitaries: tuple[qm.QKernel, ...],
    case_limit: int,
) -> _LcuPlan:
    """Validate one transient native LCU coefficient projection."""

    raw_terms = getattr(plan, "terms", None)
    normalization = getattr(plan, "normalization", None)
    signal_width = getattr(plan, "signal_width", None)
    case_count = getattr(plan, "case_count", None)
    if (
        not isinstance(raw_terms, list)
        or not isinstance(normalization, float)
        or not math.isfinite(normalization)
        or normalization <= 0.0
        or type(signal_width) is not int
        or signal_width <= 0
        or type(case_count) is not int
        or case_count < 2
        or case_count > case_limit
        or case_count & (case_count - 1)
        or signal_width != case_count.bit_length() - 1
    ):
        raise TypeError("Q7618: malformed native LCU coefficient plan")
    canonical: list[tuple[complex, qm.QKernel]] = []
    amplitudes: list[float] = []
    phases: list[float] = []
    groups: list[int] = []
    for row in raw_terms:
        if (
            not isinstance(row, tuple)
            or len(row) != 5
            or type(row[0]) is not int
            or not 0 <= row[0] < len(unitaries)
            or any(
                type(value) is not float or not math.isfinite(value)
                for value in row[1:]
            )
            or (row[1] == 0.0 and row[2] == 0.0)
            or row[3] <= 0.0
        ):
            raise TypeError("Q7618: malformed native LCU coefficient plan")
        groups.append(row[0])
        canonical.append((complex(row[1], row[2]), unitaries[row[0]]))
        amplitudes.append(row[3])
        phases.append(row[4])
    if groups != sorted(set(groups)):
        raise TypeError("Q7618: malformed native LCU coefficient plan")
    expected_width = max(1, (len(canonical) - 1).bit_length())
    if signal_width != expected_width or case_count < len(canonical):
        raise TypeError("Q7618: malformed native LCU coefficient plan")
    if not canonical and normalization != 1.0:
        raise TypeError("Q7618: malformed native LCU coefficient plan")
    return _LcuPlan(
        tuple(canonical),
        normalization,
        signal_width,
        case_count,
        tuple(amplitudes),
        tuple(phases),
    )


def _with_phase(
    unitary: qm.QKernel,
    *,
    width: int,
    phase: float,
) -> qm.QKernel:
    """Close a scalar coefficient phase into one selected unitary."""

    if not phase:
        return unitary

    @qm.qkernel
    def phased(
        state: qm.Q[qm.Bits[width]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
        """Apply the configured unitary and coefficient phase.

        Args:
            state (qm.Q[qm.Bits[width]]): Owned system state.

        Returns:
            qm.Q[qm.Bits[width]]: Transformed state with the configured phase.
        """

        state = unitary(state)
        return qm.global_phase(state, phase)

    return phased


def _zero_encoding(system_width: int) -> qm.QKernel:
    """Construct a unitary whose all-zero signal block is exactly zero."""

    total_width = system_width + 1

    @qm.qkernel
    def zero(
        state: qm.Q[qm.Bits[total_width]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[total_width]]:  # type: ignore[valid-type]
        """Move the signal state outside the all-zero block.

        Args:
            state (qm.Q[qm.Bits[total_width]]): Signal and system state.

        Returns:
            qm.Q[qm.Bits[total_width]]: State with its signal component flipped.
        """

        signal, system = qm.split_at(state, 1)
        signal = qm.x(signal)
        return qm.concat(signal, system)

    return zero


__all__ = ["lcu_block_encoding"]
