r"""Phase schedules for amplitude amplification, as frozen descriptors.

An amplification sequence is determined entirely by its interleaved reflection
phases ``(good_1, zero_1, good_2, zero_2, ...)``. These descriptors bundle
those phases with the analytic contract they satisfy, so a query count, its
schedule, and its success-probability formula travel together instead of being
recomputed by convention at each call site.

:class:`AmplificationSchedule` is the general container and accepts any phase
list. :class:`StandardAmplificationSchedule` pins every phase to ``pi``, the
Brassard-Hoyer-Mosca-Tapp rotation whose success probability is
``sin^2((2k + 1) theta)``. :class:`FixedPointAmplificationSchedule` carries the
Yoder-Low-Chuang schedule, which reaches at least ``1 - delta^2`` for every
initial success probability above a known bound, without overshooting.

Every schedule answers :meth:`AmplificationSchedule.success_probability`, so a
caller can predict the outcome of a sequence before emitting a single gate.

Schedules are deliberately **not** registered for static binding. Static
binding fields accept only ``UInt`` and ``Float`` scalars, so a phase tuple
cannot be a field; and a static-bound ``UInt`` reaches a traced body as a
symbolic handle that cannot be iterated, so a bound query count would be
useless to the emitter. Schedules stay host-side Python objects that the
emitter unrolls at trace time.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

_HARD_REFLECTION_PHASE = math.pi
"""Phase that turns a generalized reflection into ``I - 2 * Pi``."""


def _validate_probability(value: object, name: str) -> float:
    """Return a probability in ``(0, 1]`` as a Python float.

    Args:
        value (object): Candidate probability.
        name (str): Parameter name used in error messages.

    Returns:
        float: The validated probability.

    Raises:
        TypeError: If ``value`` is a boolean or not a real number.
        ValueError: If ``value`` is outside ``(0, 1]`` or not finite.
    """
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real number, not bool.")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a real number.")
    probability = float(value)
    if not math.isfinite(probability):
        raise ValueError(f"{name} must be finite.")
    if not 0.0 < probability <= 1.0:
        raise ValueError(f"{name} must lie in (0, 1], got {probability}.")
    return probability


def _validate_failure_tolerance(value: object) -> float:
    """Return a failure tolerance in ``(0, 1)`` as a Python float.

    Args:
        value (object): Candidate failure tolerance ``delta``.

    Returns:
        float: The validated failure tolerance.

    Raises:
        TypeError: If ``value`` is a boolean or not a real number.
        ValueError: If ``value`` is outside ``(0, 1)`` or not finite.
    """
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("failure_tolerance must be a real number, not bool.")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError("failure_tolerance must be a real number.")
    tolerance = float(value)
    if not math.isfinite(tolerance):
        raise ValueError("failure_tolerance must be finite.")
    if not 0.0 < tolerance < 1.0:
        raise ValueError(f"failure_tolerance must lie in (0, 1), got {tolerance}.")
    return tolerance


def _validate_odd_query_count(value: object) -> int:
    """Return a positive odd query count as a Python int.

    Args:
        value (object): Candidate query count ``L``.

    Returns:
        int: The validated query count.

    Raises:
        TypeError: If ``value`` is a boolean or not an integer.
        ValueError: If ``value`` is not positive or not odd.
    """
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("query_count must be an integer, not bool.")
    if not isinstance(value, (int, np.integer)):
        raise TypeError("query_count must be an integer.")
    count = int(value)
    if count <= 0:
        raise ValueError(f"query_count must be positive, got {count}.")
    if count % 2 == 0:
        raise ValueError(f"query_count must be odd, got {count}.")
    return count


def _validate_phases(phases: object) -> tuple[float, ...]:
    """Validate an interleaved amplification phase sequence.

    Args:
        phases (object): Candidate sequence of reflection phases. A Python
            sequence or a one-dimensional NumPy array of real numbers.

    Returns:
        tuple[float, ...]: The phases as Python floats.

    Raises:
        TypeError: If ``phases`` is not a sequence of real numbers.
        ValueError: If the sequence has an odd length; each amplification round
            consumes one good-subspace phase and one zero-state phase, or if a
            NumPy array is not one-dimensional.
    """
    if isinstance(phases, np.ndarray):
        if phases.ndim != 1:
            raise ValueError(
                f"phases must be one-dimensional, got shape {phases.shape}."
            )
        phases = phases.tolist()
    if isinstance(phases, (str, bytes)) or not isinstance(phases, Sequence):
        raise TypeError("phases must be a sequence of real numbers.")
    resolved: list[float] = []
    for phase in phases:
        if isinstance(phase, (bool, np.bool_)):
            raise TypeError("phases must contain real numbers, not bool.")
        if not isinstance(phase, (int, float, np.integer, np.floating)):
            raise TypeError("phases must contain real numbers.")
        resolved.append(float(phase))
    if len(resolved) % 2 != 0:
        raise ValueError(
            f"phases must have even length, got {len(resolved)}. "
            "Each round uses one good-subspace phase and one zero-state phase."
        )
    return tuple(resolved)


def _chebyshev_t(order: int, value: float) -> float:
    """Return the first-kind Chebyshev polynomial ``T_order(value)``.

    Uses the trigonometric form on ``[-1, 1]`` and the hyperbolic continuation
    outside it. The hyperbolic branch is reached whenever the success
    probability falls below the schedule's admissible lower bound.

    Args:
        order (int): Polynomial order, a non-negative integer.
        value (float): Real argument.

    Returns:
        float: ``T_order(value)``.
    """
    if abs(value) <= 1.0:
        return math.cos(order * math.acos(value))
    if value > 1.0:
        return math.cosh(order * math.acosh(value))
    return (-1.0) ** order * math.cosh(order * math.acosh(-value))


def _fpaa_gamma(query_count: int, failure_tolerance: float) -> float:
    """Return the fixed-point schedule parameter ``gamma``.

    ``gamma = 1 / T_{1 / L}(1 / delta)``, evaluated as
    ``1 / cosh(acosh(1 / delta) / L)`` since ``1 / delta > 1``.

    Args:
        query_count (int): Odd query count ``L``.
        failure_tolerance (float): Target failure tolerance ``delta``.

    Returns:
        float: The schedule parameter ``gamma`` in ``(0, 1)``.
    """
    return 1.0 / math.cosh(math.acosh(1.0 / failure_tolerance) / query_count)


def _fpaa_query_count(
    success_probability_bound: float,
    failure_tolerance: float,
) -> int:
    """Return the smallest odd query count meeting the fixed-point guarantee.

    The Yoder-Low-Chuang schedule certifies success probability at least
    ``1 - delta^2`` for every initial success probability ``a >= lambda``
    exactly when ``1 - gamma(L, delta)^2 <= lambda``. Solving for ``L`` gives
    ``L >= acosh(1 / delta) / acosh(1 / sqrt(1 - lambda))``, rounded up to the
    next odd integer. This is the ``O(log(2 / delta) / sqrt(lambda))`` bound.

    Args:
        success_probability_bound (float): Validated lower bound ``lambda``.
        failure_tolerance (float): Validated failure tolerance ``delta``.

    Returns:
        int: Smallest admissible odd query count ``L``.
    """
    if success_probability_bound >= 1.0:
        # A certain success needs no amplification; the sequence is just A.
        return 1
    exact = math.acosh(1.0 / failure_tolerance) / math.acosh(
        1.0 / math.sqrt(1.0 - success_probability_bound)
    )
    count = max(1, math.ceil(exact))
    if count % 2 == 0:
        count += 1
    return count


def _fpaa_phases(query_count: int, failure_tolerance: float) -> tuple[float, ...]:
    """Return the Yoder-Low-Chuang phases for one query count.

    For ``L = 2l + 1`` queries the schedule is built from

    ``kappa_j = 2 * arccot(tan(2 pi j / L) * sqrt(1 - gamma^2))``, ``j = 1..l``

    with ``arccot`` taken on ``(0, pi)``. Round ``j`` uses
    ``-kappa_{l - j + 1}`` for the good-subspace reflection and ``-kappa_j``
    for the zero-state reflection. ``tan`` is never singular because ``L`` is
    odd.

    Args:
        query_count (int): Validated odd query count ``L``.
        failure_tolerance (float): Validated failure tolerance ``delta``.

    Returns:
        tuple[float, ...]: ``L - 1`` interleaved phases; empty when ``L == 1``.
    """
    half_count = (query_count - 1) // 2
    if half_count == 0:
        return ()

    gamma = _fpaa_gamma(query_count, failure_tolerance)
    scale = math.sqrt(max(0.0, 1.0 - gamma * gamma))
    kappa = [
        2.0 * math.atan2(1.0, math.tan(2.0 * math.pi * j / query_count) * scale)
        for j in range(1, half_count + 1)
    ]

    phases: list[float] = []
    for index in range(half_count):
        phases.append(-kappa[half_count - 1 - index])
        phases.append(-kappa[index])
    return tuple(phases)


@dataclass(frozen=True, slots=True, eq=False)
class AmplificationSchedule:
    r"""Describe one amplitude-amplification sequence by its reflection phases.

    Each amplification round applies a good-subspace reflection followed by a
    zero-state reflection, so ``phases`` holds an even number of entries
    interleaved as ``(good_1, zero_1, good_2, zero_2, ...)``. A sequence of
    ``l`` rounds costs ``L = 2l + 1`` applications of the state preparation and
    its inverse, so :attr:`query_count` is always odd.

    This base class accepts any phase list and is the right type for a
    hand-designed schedule. The subclasses add the analytic contract of a known
    schedule family.

    Descriptor comparison and hashing use object identity rather than field
    values, so two structurally identical schedules are distinct objects.

    Args:
        phases (tuple[float, ...]): Interleaved reflection phases in radians.
            Any even-length sequence or one-dimensional NumPy array is
            accepted and normalized to a tuple of Python floats.

    Raises:
        TypeError: If ``phases`` is not a sequence of real numbers.
        ValueError: If ``phases`` has an odd length.

    Example:
        >>> import math
        >>> from qamomile.circuit.stdlib import AmplificationSchedule
        >>> schedule = AmplificationSchedule((math.pi, math.pi))
        >>> schedule.query_count
        3
    """

    phases: tuple[float, ...]

    def __post_init__(self) -> None:
        """Normalize and validate the phase sequence.

        Raises:
            TypeError: If ``phases`` is not a sequence of real numbers.
            ValueError: If ``phases`` has an odd length.
        """
        object.__setattr__(self, "phases", _validate_phases(self.phases))

    @property
    def num_rounds(self) -> int:
        """Return the number of amplification rounds.

        Returns:
            int: Half the phase count.
        """
        return len(self.phases) // 2

    @property
    def query_count(self) -> int:
        """Return the number of state-preparation queries ``L = 2l + 1``.

        Returns:
            int: Odd query count, one more than the phase count.
        """
        return len(self.phases) + 1

    def success_probability(self, initial_success_probability: float) -> float:
        r"""Return the success probability this schedule produces.

        Amplification acts inside the two-dimensional subspace spanned by the
        good and bad components of the prepared state, so the whole sequence is
        evaluated exactly by tracking two amplitudes. With
        ``sin^2(theta) = initial_success_probability`` and
        ``v = (sin theta, cos theta)``, the good-subspace reflection acts as
        ``diag(exp(i * good), 1)`` and the conjugated zero-state reflection as
        ``I - (1 - exp(i * zero)) v v^T``.

        Subclasses override this with their closed form. Those overrides are
        fast paths for the identical quantity, and the test suite pins them
        against this implementation.

        Args:
            initial_success_probability (float): Success probability ``a`` of
                the bare state preparation, in ``(0, 1]``.

        Returns:
            float: Success probability after the sequence. Not clamped, so a
                caller can see an out-of-range value caused by an inconsistent
                hand-written schedule.

        Raises:
            TypeError: If ``initial_success_probability`` is a boolean or not a
                real number.
            ValueError: If ``initial_success_probability`` is outside
                ``(0, 1]``.
        """
        probability = _validate_probability(
            initial_success_probability,
            "initial_success_probability",
        )
        angle = math.asin(math.sqrt(probability))
        good = math.sin(angle)
        bad = math.cos(angle)
        amplitude_good = complex(good)
        amplitude_bad = complex(bad)
        for index in range(self.num_rounds):
            amplitude_good *= cmath.exp(1j * self.phases[2 * index])
            overlap = good * amplitude_good + bad * amplitude_bad
            factor = (1.0 - cmath.exp(1j * self.phases[2 * index + 1])) * overlap
            amplitude_good -= factor * good
            amplitude_bad -= factor * bad
        return abs(amplitude_good) ** 2


@dataclass(frozen=True, slots=True, eq=False)
class StandardAmplificationSchedule(AmplificationSchedule):
    r"""Describe the Brassard-Hoyer-Mosca-Tapp schedule of hard reflections.

    Every phase is ``pi``, so each round applies ``A S_0 A^dagger S_good`` —
    the Grover iterate up to a global sign. After ``k`` rounds the success
    probability is exactly ``sin^2((2k + 1) theta)`` with
    ``sin^2(theta) = a``. That rotation overshoots past its peak, so ``k`` must
    be chosen from a known ``a`` via
    :func:`amplitude_amplification_iteration_count`. When only a lower bound on
    ``a`` is available, use :class:`FixedPointAmplificationSchedule` instead.

    This subclass adds no fields; the round count is derived from the phases.

    Args:
        phases (tuple[float, ...]): Interleaved reflection phases. Every entry
            must be ``math.pi``.

    Raises:
        TypeError: If ``phases`` is not a sequence of real numbers.
        ValueError: If ``phases`` has an odd length or any entry differs from
            ``math.pi``.

    Example:
        >>> from qamomile.circuit.stdlib import standard_amplification_schedule
        >>> standard_amplification_schedule(3).iterations
        3
    """

    def __post_init__(self) -> None:
        """Validate the phases and pin them to hard reflections.

        Raises:
            TypeError: If ``phases`` is not a sequence of real numbers.
            ValueError: If ``phases`` has an odd length or any entry differs
                from ``math.pi``.
        """
        # Explicit unbound base call: zero-argument ``super()`` does not work
        # inside a ``slots=True`` dataclass subclass, because the decorator
        # rebuilds the class after the method's ``__class__`` cell is bound.
        AmplificationSchedule.__post_init__(self)
        if any(phase != _HARD_REFLECTION_PHASE for phase in self.phases):
            raise ValueError(
                "StandardAmplificationSchedule phases must all equal math.pi. "
                "Use AmplificationSchedule for a general phase list."
            )

    @property
    def iterations(self) -> int:
        """Return the number of Grover iterations ``k``.

        Returns:
            int: Round count, identical to :attr:`num_rounds`.
        """
        return self.num_rounds

    def success_probability(self, initial_success_probability: float) -> float:
        """Return ``sin^2((2k + 1) theta)`` for this schedule.

        Args:
            initial_success_probability (float): Success probability ``a`` of
                the bare state preparation, in ``(0, 1]``.

        Returns:
            float: Success probability after ``k`` rounds.

        Raises:
            TypeError: If ``initial_success_probability`` is a boolean or not a
                real number.
            ValueError: If ``initial_success_probability`` is outside
                ``(0, 1]``.
        """
        probability = _validate_probability(
            initial_success_probability,
            "initial_success_probability",
        )
        angle = math.asin(math.sqrt(probability))
        return math.sin((2 * self.iterations + 1) * angle) ** 2


@dataclass(frozen=True, slots=True, eq=False)
class FixedPointAmplificationSchedule(AmplificationSchedule):
    r"""Describe the Yoder-Low-Chuang fixed-point schedule.

    The phases are the phase-matched sequence that turns the amplification
    response into a Chebyshev plateau: for every initial success probability
    at or above :attr:`admissible_success_probability_bound`, the output
    reaches at least :attr:`guaranteed_success_probability`, and it never
    overshoots as that probability grows.

    The phases must be exactly the schedule derived from
    :attr:`query_count` and ``failure_tolerance``; the constructor recomputes
    and checks them, so the closed form in :meth:`success_probability` always
    describes the phases actually carried.

    Args:
        phases (tuple[float, ...]): Interleaved reflection phases.
        failure_tolerance (float): Target failure tolerance ``delta`` in
            ``(0, 1)``.

    Raises:
        TypeError: If a field has an invalid runtime type.
        ValueError: If ``phases`` has an odd length, ``failure_tolerance`` is
            outside ``(0, 1)``, or the phases do not match the schedule derived
            from the query count and tolerance.

    Example:
        >>> from qamomile.circuit.stdlib import fixed_point_amplification_schedule
        >>> schedule = fixed_point_amplification_schedule(0.05, 0.3)
        >>> schedule.query_count
        9
    """

    failure_tolerance: float

    def __post_init__(self) -> None:
        """Validate the phases, the tolerance, and their mutual consistency.

        Raises:
            TypeError: If a field has an invalid runtime type.
            ValueError: If a field is out of range or the phases do not match
                the derived schedule.
        """
        # Explicit unbound base call; see StandardAmplificationSchedule.
        AmplificationSchedule.__post_init__(self)
        object.__setattr__(
            self,
            "failure_tolerance",
            _validate_failure_tolerance(self.failure_tolerance),
        )
        expected = _fpaa_phases(self.query_count, self.failure_tolerance)
        if len(expected) != len(self.phases) or any(
            not math.isclose(actual, target, rel_tol=1e-9, abs_tol=1e-12)
            for actual, target in zip(self.phases, expected)
        ):
            raise ValueError(
                "FixedPointAmplificationSchedule phases must match the "
                "Yoder-Low-Chuang schedule for query_count "
                f"{self.query_count} and failure_tolerance "
                f"{self.failure_tolerance}. Build one with "
                "fixed_point_amplification_schedule, or use "
                "AmplificationSchedule for a general phase list."
            )

    @property
    def guaranteed_success_probability(self) -> float:
        """Return the guaranteed output success probability ``1 - delta^2``.

        Returns:
            float: Lower bound met for every admissible initial probability.
        """
        return 1.0 - self.failure_tolerance**2

    @property
    def admissible_success_probability_bound(self) -> float:
        """Return the smallest initial probability this schedule covers.

        This is ``1 - gamma(L, delta)^2``, the exact admissibility threshold.
        It is at or below the bound requested from
        :func:`fixed_point_amplification_schedule`, because the query count is
        rounded up to the next odd integer.

        Returns:
            float: Smallest initial success probability meeting the guarantee.
        """
        return 1.0 - _fpaa_gamma(self.query_count, self.failure_tolerance) ** 2

    def success_probability(self, initial_success_probability: float) -> float:
        """Return ``1 - delta^2 * T_L(sqrt(1 - a) / gamma)^2`` for this schedule.

        Args:
            initial_success_probability (float): Success probability ``a`` of
                the bare state preparation, in ``(0, 1]``.

        Returns:
            float: Success probability after the sequence, at least
                :attr:`guaranteed_success_probability` whenever ``a`` is at or
                above :attr:`admissible_success_probability_bound`.

        Raises:
            TypeError: If ``initial_success_probability`` is a boolean or not a
                real number.
            ValueError: If ``initial_success_probability`` is outside
                ``(0, 1]``.
        """
        probability = _validate_probability(
            initial_success_probability,
            "initial_success_probability",
        )
        gamma = _fpaa_gamma(self.query_count, self.failure_tolerance)
        argument = math.sqrt(max(0.0, 1.0 - probability)) / gamma
        return (
            1.0
            - self.failure_tolerance**2 * _chebyshev_t(self.query_count, argument) ** 2
        )


def amplitude_amplification_iteration_count(success_probability: float) -> int:
    """Return the iteration count that maximizes standard amplitude amplification.

    With ``sin^2(theta) = success_probability``, ``k`` iterations give success
    probability ``sin^2((2k + 1) theta)``. The maximizing integer is
    ``round(pi / (4 theta) - 1/2)``, clamped at zero.

    Args:
        success_probability (float): Initial success probability ``a`` in
            ``(0, 1]``.

    Returns:
        int: Optimal non-negative iteration count.

    Raises:
        TypeError: If ``success_probability`` is a boolean or not a real
            number.
        ValueError: If ``success_probability`` is outside ``(0, 1]``.

    Example:
        >>> amplitude_amplification_iteration_count(0.05)
        3
    """
    probability = _validate_probability(success_probability, "success_probability")
    angle = math.asin(math.sqrt(probability))
    return max(0, round(math.pi / (4.0 * angle) - 0.5))


def standard_amplification_schedule(
    iterations: int,
) -> StandardAmplificationSchedule:
    """Build the standard schedule for a given number of Grover iterations.

    Args:
        iterations (int): Number of amplification rounds ``k``, non-negative.
            Zero gives an empty schedule, so the sequence is the bare state
            preparation.

    Returns:
        StandardAmplificationSchedule: Schedule of ``2 * iterations`` hard
            reflections.

    Raises:
        TypeError: If ``iterations`` is a boolean or not an integer.
        ValueError: If ``iterations`` is negative.

    Example:
        >>> standard_amplification_schedule(2).query_count
        5
    """
    if isinstance(iterations, (bool, np.bool_)):
        raise TypeError("iterations must be an integer, not bool.")
    if not isinstance(iterations, (int, np.integer)):
        raise TypeError("iterations must be an integer.")
    count = int(iterations)
    if count < 0:
        raise ValueError(f"iterations must be non-negative, got {count}.")
    return StandardAmplificationSchedule(
        phases=(_HARD_REFLECTION_PHASE,) * (2 * count),
    )


def fixed_point_amplification_schedule(
    success_probability_bound: float | None = None,
    failure_tolerance: float = 0.1,
    query_count: int | None = None,
) -> FixedPointAmplificationSchedule:
    """Build the Yoder-Low-Chuang fixed-point schedule.

    With the default query count, the resulting schedule reaches success
    probability at least ``1 - failure_tolerance^2`` for every initial success
    probability at or above ``success_probability_bound``, using
    ``O(log(2 / delta) / sqrt(lambda))`` queries. This is the schedule to use
    for block-encoding post-selection, where the exact success amplitude is
    generically unknown but a bound follows from the subnormalization.

    Args:
        success_probability_bound (float | None): Known lower bound ``lambda``
            on the initial success probability, in ``(0, 1]``. Required unless
            ``query_count`` is given.
        failure_tolerance (float): Target failure tolerance ``delta`` in
            ``(0, 1)``. Defaults to ``0.1``, i.e. success probability at least
            ``0.99``.
        query_count (int | None): Explicit odd query count ``L``. Overrides the
            count derived from ``success_probability_bound``; a smaller ``L``
            weakens the guarantee. Defaults to ``None``.

    Returns:
        FixedPointAmplificationSchedule: Schedule of ``query_count - 1``
            phases.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If neither ``success_probability_bound`` nor
            ``query_count`` is given, or an argument is out of range.

    Example:
        >>> schedule = fixed_point_amplification_schedule(0.05, 0.1)
        >>> schedule.query_count
        15
        >>> round(schedule.guaranteed_success_probability, 2)
        0.99
    """
    tolerance = _validate_failure_tolerance(failure_tolerance)
    if query_count is None:
        if success_probability_bound is None:
            raise ValueError(
                "Provide success_probability_bound or query_count "
                "to build a fixed-point phase schedule."
            )
        bound = _validate_probability(
            success_probability_bound,
            "success_probability_bound",
        )
        resolved_count = _fpaa_query_count(bound, tolerance)
    else:
        resolved_count = _validate_odd_query_count(query_count)
    return FixedPointAmplificationSchedule(
        phases=_fpaa_phases(resolved_count, tolerance),
        failure_tolerance=tolerance,
    )


__all__ = [
    "AmplificationSchedule",
    "FixedPointAmplificationSchedule",
    "StandardAmplificationSchedule",
    "amplitude_amplification_iteration_count",
    "fixed_point_amplification_schedule",
    "standard_amplification_schedule",
]
