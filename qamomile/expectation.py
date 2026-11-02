"""Backend-neutral expectation estimation from functional state programs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from .compilation import (
    ExecutionEndpoint,
)
from .ir import Program
from .observable import Hamiltonian, PauliWord
from .source import ProgramFamily, QKernel, _close_program


@dataclass(frozen=True, slots=True)
class PauliEstimate:
    """One sampled Pauli-word mean and unbiased sample variance (None for N=1)."""

    word: PauliWord
    coefficient: float
    mean: float
    variance: float | None
    shots: int


@dataclass(frozen=True, slots=True)
class PauliReadout:
    """One signed Z-product readout after Clifford diagonalization."""

    term_index: int
    word: PauliWord
    negative: bool


@dataclass(frozen=True, slots=True)
class PauliMeasurementGroup:
    """One inspected shared-measurement basis, allocation, and variance."""

    basis: PauliWord | None
    term_indices: tuple[int, ...]
    variance: float | None
    shots: int
    seed: int
    readouts: tuple[PauliReadout, ...] = ()
    diagonalizing_gates: int = 0
    strategy: Literal["independent", "qwc", "commuting"] = "qwc"

    @property
    def standard_error(self) -> float | None:
        """Return this group's covariance-aware standard-error contribution."""

        return None if self.variance is None else (self.variance / self.shots) ** 0.5


@dataclass(frozen=True, slots=True)
class ExpectationResult:
    """An observable expectation with explicit budget and measurement uncertainty.

    Attributes:
        value (float): Estimated Hamiltonian expectation.
        standard_error (float | None): Normalized error of the mean, if available.
        backend (str): Execution endpoint identity.
        shots (int): Actual total device shots, independent of detail availability.
        terms (tuple[PauliEstimate, ...] | None): Term statistics, or unavailable.
        groups (tuple[PauliMeasurementGroup, ...] | None): Group statistics, or unavailable.
        method (str): Selected estimation implementation.
    """

    value: float
    standard_error: float | None
    backend: str
    shots: int
    terms: tuple[PauliEstimate, ...] | None = None
    groups: tuple[PauliMeasurementGroup, ...] | None = None
    method: str = "sampling"


def expectation(
    source: Program | ProgramFamily | QKernel,
    observable: Hamiltonian,
    *,
    on: ExecutionEndpoint,
    parameters: Mapping[str, object] | None = None,
    shots: int = 1_000,
    seed: int = 0,
    grouping: Literal["independent", "qwc", "commuting"] = "independent",
    **bindings: object,
) -> ExpectationResult:
    """Estimate once using the same measurement path as Estimator.

    Args:
        source (Program | ProgramFamily | QKernel): Classical-input state
            preparation returning a fixed-width Q[Bits[N]].
        observable (Hamiltonian): Hermitian Pauli sum on the returned bits.
        on (ExecutionEndpoint): Execution endpoint.
        parameters (Mapping[str, object] | None): Runtime Param bindings.
        shots (int): Total budget across all measurement groups, for every strategy.
        seed (int): Portable seed; zero uses endpoint-default randomness.
        grouping (str): Independent, QWC, or generally commuting measurements.
        **bindings (object): Const bindings for the state preparation.

    Returns:
        ExpectationResult: Estimate and unbiased sampling uncertainty.

    Raises:
        TypeCheckError: If the state-preparation contract is invalid.
        ExecutionError: If the budget, native plan, or execution is invalid.
        CapabilityError: If required native support is unavailable.
    """
    from .measurement import Estimator

    return Estimator(
        _close_program(source, bindings), observable, grouping=grouping
    ).estimate(on=on, parameters=parameters, shots=shots, seed=seed)


__all__ = [
    "ExpectationResult",
    "PauliEstimate",
    "PauliMeasurementGroup",
    "PauliReadout",
    "expectation",
]
