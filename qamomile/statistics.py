"""Statistics of measured classical values, without inferring quantum observables.

All reductions operate on frequency histograms through Rust. Readouts describe
classical fields or bit parities; the caller owns their relationship to a
measurement basis. Standard errors assume independent, identically distributed
shots and are not confidence bounds or estimates of device bias.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TypeVar

from ._native import compiler as native
from .compilation import ResultABI, SampleResult
from .errors import CapabilityError
from .types import Bits

Path = tuple[int | str, ...]
_EstimateT = TypeVar("_EstimateT", bound="Estimate")


def _require(name: str):
    """Require a callable native statistics operation without a fallback.

    Args:
        name (str): Native bridge operation name.

    Returns:
        Callable: Required native operation.

    Raises:
        CapabilityError: If the installed native compiler lacks the operation.
    """
    operation = getattr(native, name, None)
    if not callable(operation):
        raise CapabilityError(
            f"Q7404: native statistics operation {name!r} is unavailable"
        )
    return operation


def _path(path: Path) -> list[tuple[int | None, str | None]]:
    """Encode explicit nonnegative indices and record field names.

    Args:
        path (Path): Path into a decoded sampled value.

    Returns:
        list: Tagged native path components.

    Raises:
        TypeError: If a component is not an integer index or field name.
        ValueError: If an index is negative or the path exceeds its native budget.
    """
    if not isinstance(path, tuple):
        raise TypeError("path must be a tuple of indices or field names")
    if len(path) > native.MAX_STATISTICS_PATH:
        raise ValueError("Q7402: statistics path exceeds the processing budget")
    result: list[tuple[int | None, str | None]] = []
    for component in path:
        if type(component) is int:
            if component < 0:
                raise ValueError("statistics path indices must be nonnegative")
            result.append((component, None))
        elif type(component) is str:
            result.append((None, component))
        else:
            raise TypeError("statistics paths require integer indices or field names")
    return result


@dataclass(frozen=True, slots=True)
class Value:
    """Select one numeric field from a sampled classical value.

    Bool and Bit values are interpreted as zero or one. Whole bit strings are
    never implicitly interpreted as integers or Pauli eigenvalues.

    Args:
        path (Path): Nonnegative tuple indices or record field names; empty
            selects the entire scalar result.

    Raises:
        TypeError: If the path has malformed components.
        ValueError: If the path exceeds native limits or contains negative indices.
    """

    path: Path = ()
    _native: native.StatisticsReadout = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Retain the native checked numeric readout."""
        operation = _require("statistics_readout")
        object.__setattr__(self, "_native", operation(_path(self.path)))


@dataclass(frozen=True, slots=True)
class Parity:
    """Read a product of signs from selected coordinates of a decoded bit tuple.

    A zero bit contributes +1 and a one bit contributes -1. This is a classical
    parity, not a claim about which quantum observable was measured.

    Args:
        indices (tuple[int, ...]): Distinct nonnegative logical coordinates.
            Empty denotes constant +1.
        path (Path): Path to the bit tuple; empty selects the full result.

    Raises:
        TypeError: If indices or path components have invalid types.
        ValueError: If indices repeat, are negative, or exceed native budgets.
    """

    indices: tuple[int, ...]
    path: Path = ()
    _native: native.StatisticsReadout = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Retain the native checked parity readout."""
        operation = _require("statistics_readout")
        if not isinstance(self.indices, tuple):
            raise TypeError("parity indices must be a tuple of integers")
        if len(self.indices) > native.MAX_STATISTICS_TERMS:
            raise ValueError("Q7402: parity exceeds the processing budget")
        if any(type(i) is not int for i in self.indices):
            raise TypeError("parity indices must be a tuple of integers")
        if any(i < 0 for i in self.indices):
            raise ValueError("parity indices must be nonnegative")
        object.__setattr__(
            self, "_native", operation(_path(self.path), list(self.indices))
        )


@dataclass(frozen=True, slots=True, init=False)
class Estimate:
    """An immutable estimate and its plug-in standard error.

    Obtain instances through statistics functions. Independence is a caller
    assumption, not inferred from seeds or backend names. Zero estimated error
    does not certify the true expectation. Arithmetic is deliberately explicit.

    Attributes:
        value (float): Estimated scalar value.
        standard_error (float | None): Estimated uncertainty of the mean; None
            when a contributing sample has fewer than two observations.
        shots (int): Total observations used across contributing inputs.
    """

    _native: native.StatisticsEstimate = field(repr=False)

    def __init__(self) -> None:
        """Require construction through native statistics functions.

        Raises:
            TypeError: Always; use mean or combine_independent.
        """
        raise TypeError("estimates are created by statistics functions")

    @classmethod
    def _from_native(cls: type[_EstimateT], result: object) -> _EstimateT:
        """Project one checked native statistical result.

        Args:
            result (object): Native reduction result.

        Returns:
            Estimate: Immutable projection of the native owner.

        Raises:
            CapabilityError: If the native result violates the facade contract.
        """
        if not isinstance(result, native.StatisticsEstimate):
            raise CapabilityError("Q7404: malformed native statistical estimate")
        instance = object.__new__(cls)
        object.__setattr__(instance, "_native", result)
        return instance

    @classmethod
    def _from_summary(
        cls, value: float, standard_error: float | None, shots: int
    ) -> Estimate:
        """Import normalized backend uncertainty without claiming raw samples.

        Args:
            value (float): Finite estimated mean.
            standard_error (float | None): Normalized error of the mean, or unknown.
            shots (int): Positive actual observation count.

        Returns:
            Estimate: Rust-validated summary, with no individual-shot variance.

        Raises:
            ValueError: If the summary is invalid or exceeds native bounds.
            CapabilityError: If native summary import is unavailable.
        """
        operation = _require("statistics_from_summary")
        return cls._from_native(operation(value, standard_error, str(shots)))

    @property
    def value(self) -> float:
        """Return the estimated scalar value."""
        return self._native.value

    @property
    def standard_error(self) -> float | None:
        """Return the estimated error of the mean, not the spread of outcomes."""
        return self._native.standard_error

    @property
    def shots(self) -> int:
        """Return the total observation count without expanding the histogram."""
        return int(self._native.shots)


@dataclass(frozen=True, slots=True, init=False)
class MeanEstimate(Estimate):
    """A sample mean with the unbiased variance of individual outcomes.

    Attributes:
        sample_variance (float | None): Variance with denominator N-1, or None
            for one observation. Inherited standard_error uses sqrt(variance/N).
    """

    @property
    def sample_variance(self) -> float | None:
        """Return unbiased shot variance, distinct from uncertainty of the mean."""
        return self._native.sample_variance


def mean(samples: SampleResult, *, path: Path = ()) -> MeanEstimate:
    """Estimate the mean of one decoded numeric scalar or selected field.

    Args:
        samples (SampleResult): Histogram of identically distributed independent shots.
        path (Path): Field selection; empty requires a scalar result.

    Returns:
        MeanEstimate: Mean, unbiased shot variance, standard error, and count.

    Raises:
        TypeError: If input or path has an invalid Python shape.
        ValueError: If the selected value is not numeric, its statistics exceed
            binary64 range, or native work limits are exceeded.
        CapabilityError: If native statistics support is unavailable.
    """
    return linear_combination(samples, [(1.0, Value(path))])


def parity_mean(
    samples: SampleResult, *, indices: tuple[int, ...], path: Path = ()
) -> MeanEstimate:
    """Estimate the mean of an explicitly selected classical bit parity.

    Args:
        samples (SampleResult): Joint histogram in logical source order.
        indices (tuple[int, ...]): Distinct coordinates of the decoded bit tuple.
        path (Path): Path to that tuple, including inside nested return values.

    Returns:
        MeanEstimate: Parity mean and its sampling uncertainty.

    Raises:
        TypeError: If the Python arguments are malformed.
        ValueError: If selected coordinates are not bits, repeat, are out of
            range, or native budgets are exceeded.
        CapabilityError: If native statistics support is unavailable.
    """
    return linear_combination(samples, [(1.0, Parity(indices, path))])


def linear_combination(
    samples: SampleResult,
    terms: Sequence[tuple[float, Value | Parity]],
    *,
    offset: float = 0.0,
) -> MeanEstimate:
    """Estimate a linear score within one joint sample, preserving covariance.

    Each term is (coefficient, readout). The score is evaluated for each distinct
    outcome before frequency-weighted aggregation. No covariance matrix or
    expanded shot array is constructed. Integer inputs are converted to binary64.

    Args:
        samples (SampleResult): Joint observations from one measurement program.
        terms (Sequence[tuple[float, Value | Parity]]): Explicit numeric or parity
            readouts of this same result, with finite real coefficients.
        offset (float): Finite constant added to each score.

    Returns:
        MeanEstimate: Mean and uncertainty including within-shot correlations.

    Raises:
        TypeError: If samples, coefficients, or readout objects are malformed.
        ValueError: If readouts disagree with decoded values, statistics exceed
            the finite binary64 range, or native processing limits are exceeded.
        CapabilityError: If native statistics support is unavailable.
    """
    operation = _require("statistics_mean")
    if not isinstance(samples, SampleResult):
        raise TypeError("statistics require a SampleResult")
    if len(terms) > native.MAX_STATISTICS_TERMS:
        raise ValueError("Q7402: statistics terms exceed the processing budget")
    rows = []
    for coefficient, readout in terms:
        if not isinstance(readout, (Value, Parity)):
            raise TypeError("statistics terms require Value or Parity readouts")
        rows.append((_coefficient(coefficient), readout._native))
    abi = samples.result_abi
    if abi is None:
        abi = ResultABI(Bits[samples.width], tuple(range(samples.width)), samples.width)
    result = operation(
        samples._native_summary, abi._to_bytes(), rows, _coefficient(offset)
    )
    return MeanEstimate._from_native(result)


def combine_independent(
    estimates: Sequence[Estimate],
    *,
    coefficients: Sequence[float] | None = None,
    offset: float = 0.0,
) -> Estimate:
    """Combine estimates from explicitly independent observations.

    Reusing the same shots for different terms requires linear_combination
    instead. Equal seeds do not establish independence. A zero coefficient does
    not propagate unknown uncertainty, but all supplied shot counts are reported.
    Empty input returns the known offset with zero shots and zero standard error.

    Args:
        estimates (Sequence[Estimate]): Estimates from disjoint independent runs.
        coefficients (Sequence[float] | None): Finite weights, defaulting to one.
        offset (float): Finite constant contribution.

    Returns:
        Estimate: Weighted estimate with independent mean variances added.
            No single-shot variance is invented for different populations.

    Raises:
        TypeError: If estimates or coefficients have invalid Python shapes.
        ValueError: If counts disagree, estimates repeat, statistics exceed
            binary64 range, or native budgets are exceeded.
        CapabilityError: If native statistics support is unavailable.
    """
    operation = _require("statistics_combine_independent")
    if len(estimates) > native.MAX_STATISTICS_TERMS:
        raise ValueError("Q7402: estimates exceed the processing budget")
    weights = (1.0,) * len(estimates) if coefficients is None else coefficients
    if len(weights) != len(estimates):
        raise ValueError("coefficients must match the number of estimates")
    rows = []
    for coefficient, estimate in zip(weights, estimates, strict=True):
        if not isinstance(estimate, Estimate):
            raise TypeError("combine_independent requires statistical estimates")
        rows.append((_coefficient(coefficient), estimate._native))
    if len({id(estimate._native) for estimate in estimates}) != len(estimates):
        raise ValueError("the same estimate cannot be declared independent twice")
    return Estimate._from_native(operation(rows, _coefficient(offset)))


def _coefficient(value: float) -> float:
    """Normalize one real coefficient at the Python ABI boundary.

    Args:
        value (float): Python real scalar.

    Returns:
        float: Binary64 coefficient, with finiteness checked by Rust.

    Raises:
        TypeError: If the scalar is not a Python int or float, or is Boolean.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("statistics coefficients must be real numbers")
    return float(value)


__all__ = [
    "Estimate",
    "MeanEstimate",
    "Value",
    "Parity",
    "mean",
    "parity_mean",
    "linear_combination",
    "combine_independent",
]
