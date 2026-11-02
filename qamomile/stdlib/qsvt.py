# ruff: noqa: F821 -- dependent widths are closed into generated source.

"""Functional quantum singular value transformation."""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, cast

import qamomile as qm
from qamomile._native import compiler as native

from .block_encoding import BlockEncoding, _positive_width
from .lcu import lcu_block_encoding


@dataclass(frozen=True, slots=True)
class QSVTPhases:
    """Canonical projector-convention phases and polynomial coordinates.

    Every instance is normalized to the projector convention consumed by
    :func:`qsvt`. The optional polynomial coordinates describe the producer's
    intended polynomial. Checked approximation evidence lives on
    :class:`CertifiedQSVTPolynomial`; raw phase coordinates deliberately carry
    no unqualified error claim.

    Args:
        phases (tuple[float, ...]): Nonempty finite projector phases.
        odd (bool | None): Whether the intended polynomial is odd, or ``None``
            when parity is not asserted.
        domain (tuple[float, float] | None): Optional increasing approximation
            interval.
        scale (float): Finite nonzero output scale.

    Raises:
        TypeError: If a coordinate has the wrong host type.
        ValueError: If a numeric coordinate is empty, nonfinite, or invalid.
    """

    phases: tuple[float, ...]
    odd: bool | None = None
    domain: tuple[float, float] | None = None
    scale: float = 1.0

    def __post_init__(self) -> None:
        """Validate and freeze the canonical coordinates.

        Raises:
            TypeError: If a coordinate has the wrong host type.
            ValueError: If a numeric coordinate is empty, nonfinite, or invalid.
        """

        phases = _finite_reals(self.phases, name="phases")
        if not phases:
            raise ValueError("QSVT requires at least one phase")
        if self.odd is not None and not isinstance(self.odd, bool):
            raise TypeError("odd must be bool or None")
        domain = self.domain
        if domain is not None:
            if not isinstance(domain, tuple) or len(domain) != 2:
                raise TypeError("domain must be a two-element tuple or None")
            lower, upper = _finite_reals(domain, name="domain")
            if lower >= upper:
                raise ValueError("domain lower bound must be less than upper bound")
            domain = (lower, upper)
        scale = _finite_real(self.scale, name="scale")
        if not scale:
            raise ValueError("scale must be nonzero")
        object.__setattr__(self, "phases", phases)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "scale", scale)

    @classmethod
    def from_qsp_wx(
        cls,
        phases: Sequence[float],
        *,
        odd: bool | None = None,
        domain: tuple[float, float] | None = None,
        scale: float = 1.0,
    ) -> QSVTPhases:
        """Convert QSP ``Wx`` phases into the canonical projector convention.

        Args:
            phases (Sequence[float]): Nonempty finite ``Wx`` phase sequence.
            odd (bool | None): Optional polynomial parity coordinate.
            domain (tuple[float, float] | None): Optional polynomial domain.
            scale (float): Finite nonzero output scale.

        Returns:
            QSVTPhases: Canonical projector phases with the same coordinates.

        Raises:
            TypeError: If a coordinate has the wrong host type.
            ValueError: If a numeric coordinate is empty, nonfinite, or invalid.
        """

        converted = _convert_qsp_wx_phases(phases)
        return cls(converted, odd=odd, domain=domain, scale=scale)


class QSVTSuccess(StrEnum):
    """How observing the signal subspace certifies an operator-function result."""

    HERALDED = "heralded"


class _NativeQsvtPolynomialCertificate(Protocol):
    """Structural projection returned by the required native boundary."""

    chebyshev_coefficients: list[float]
    qsp_phases: list[float]
    projector_phases: list[float]
    degree: int
    odd: bool
    uniform_error_bound: float
    target_magnitude_bound: float
    realized_magnitude_bound: float
    queries: int
    projector_rotations: int
    success: str


class _NativeReciprocalQsvtCertificate(Protocol):
    """Structural reciprocal projection returned by the native boundary."""

    normalization: float
    minimum_singular_value: float
    normalized_gap: float
    reciprocal_scale: float
    operator_scale: float
    approximation_error_bound: float
    total_error_bound: float
    polynomial: _NativeQsvtPolynomialCertificate


@dataclass(frozen=True, slots=True)
class _DecodedReciprocalSVT:
    """Validated host projection before source composition."""

    polynomial: CertifiedQSVTPolynomial
    normalization: float
    minimum_singular_value: float
    normalized_gap: float
    reciprocal_scale: float
    operator_scale: float
    approximation_error_bound: float
    total_error_bound: float


@dataclass(frozen=True, slots=True, init=False)
class CertifiedQSVTPolynomial:
    """Record one Rust-replayed certificate for a QSP real component.

    The target uses ascending first-kind Chebyshev coefficients. ``phases`` are
    already in Qamomile's projector convention, while ``qsp_phases`` retain the
    symmetric ``Wx`` candidate replayed by Rust. The error and magnitude bounds
    hold uniformly for the response's real component on ``domain``; the full
    projected response may also contain its unitary complementary polynomial.
    Instances are minted only by :func:`synthesize_qsvt_polynomial`;
    :func:`qsvt` replays the certificate before consuming it.
    """

    chebyshev_coefficients: tuple[float, ...]
    phases: QSVTPhases
    qsp_phases: tuple[float, ...]
    degree: int
    odd: bool
    domain: tuple[float, float]
    uniform_error_bound: float
    target_magnitude_bound: float
    realized_magnitude_bound: float
    queries: int
    projector_rotations: int
    success: QSVTSuccess

    def __init__(self) -> None:
        """Reject unchecked construction outside the native synthesis boundary."""

        raise TypeError("use synthesize_qsvt_polynomial() to mint a certificate")

    @classmethod
    def _from_projection(
        cls,
        *,
        chebyshev_coefficients: tuple[float, ...],
        phases: QSVTPhases,
        qsp_phases: tuple[float, ...],
        degree: int,
        odd: bool,
        uniform_error_bound: float,
        target_magnitude_bound: float,
        realized_magnitude_bound: float,
        queries: int,
        projector_rotations: int,
        success: QSVTSuccess,
    ) -> CertifiedQSVTPolynomial:
        """Construct one already-validated native projection."""

        result = object.__new__(cls)
        values = {
            "chebyshev_coefficients": chebyshev_coefficients,
            "phases": phases,
            "qsp_phases": qsp_phases,
            "degree": degree,
            "odd": odd,
            "domain": (-1.0, 1.0),
            "uniform_error_bound": uniform_error_bound,
            "target_magnitude_bound": target_magnitude_bound,
            "realized_magnitude_bound": realized_magnitude_bound,
            "queries": queries,
            "projector_rotations": projector_rotations,
            "success": success,
        }
        for name, value in values.items():
            object.__setattr__(result, name, value)
        return result


@dataclass(frozen=True, slots=True, init=False)
class CertifiedReciprocalSVT:
    """Expose one Rust-certified reciprocal singular-value block encoding.

    ``block_encoding`` and its ``transform`` unitary realize an approximation
    to ``operator_scale / sigma`` for singular values ``sigma`` in the promised
    gapped domain. One additional signal qubit extracts the Hermitian part of
    the certified QSP response, so postselecting the complete signal register
    applies the real reciprocal polynomial. This remains a singular-value
    reciprocal, not an unqualified matrix inverse.

    Attributes:
        transform (qm.QKernel): Unitary of the reciprocal block encoding.
        block_encoding (BlockEncoding): Approximate scaled-reciprocal encoding.
        polynomial (CertifiedQSVTPolynomial): Replayed polynomial certificate.
        normalization (float): Normalization of the input block encoding.
        minimum_singular_value (float): Promised positive physical gap.
        normalized_gap (float): Gap divided by the input normalization.
        spectral_domain (tuple[tuple[float, float], tuple[float, float]]):
            Certified normalized spectral intervals.
        reciprocal_scale (float): Scale of the reciprocal polynomial.
        operator_scale (float): Physical scaled-inverse factor.
        approximation_error_bound (float): Reciprocal approximation error.
        total_error_bound (float): Combined approximation and realization error.
        queries (int): Input-encoding queries in one raw QSP branch.
        projector_rotations (int): Projector rotations in one raw QSP branch.
        success (QSVTSuccess): Complete-signal heralding rule.

    Raises:
        TypeError: If constructed directly instead of through
            :func:`reciprocal_svt`.
    """

    transform: qm.QKernel
    block_encoding: BlockEncoding
    polynomial: CertifiedQSVTPolynomial
    normalization: float
    minimum_singular_value: float
    normalized_gap: float
    spectral_domain: tuple[tuple[float, float], tuple[float, float]]
    reciprocal_scale: float
    operator_scale: float
    approximation_error_bound: float
    total_error_bound: float
    queries: int
    projector_rotations: int
    success: QSVTSuccess

    def __init__(self) -> None:
        """Reject unchecked construction outside the native producer.

        Raises:
            TypeError: Always, because :func:`reciprocal_svt` is the only
                certificate constructor.
        """

        raise TypeError("use reciprocal_svt() to mint a certificate")

    @classmethod
    def _from_projection(
        cls,
        *,
        transform: qm.QKernel,
        block_encoding: BlockEncoding,
        polynomial: CertifiedQSVTPolynomial,
        normalization: float,
        minimum_singular_value: float,
        normalized_gap: float,
        reciprocal_scale: float,
        operator_scale: float,
        approximation_error_bound: float,
        total_error_bound: float,
    ) -> CertifiedReciprocalSVT:
        """Construct one already-validated native projection.

        Args:
            transform (qm.QKernel): Unitary of the reciprocal block encoding.
            block_encoding (BlockEncoding): Complete reciprocal block encoding.
            polynomial (CertifiedQSVTPolynomial): Replayed polynomial evidence.
            normalization (float): Input block-encoding normalization.
            minimum_singular_value (float): Promised physical spectral gap.
            normalized_gap (float): Gap divided by the normalization.
            reciprocal_scale (float): Scale of the normalized reciprocal.
            operator_scale (float): Scale of the physical inverse operator.
            approximation_error_bound (float): Reciprocal approximation error.
            total_error_bound (float): Combined certified error.

        Returns:
            CertifiedReciprocalSVT: Frozen checked host projection.
        """

        result = object.__new__(cls)
        values = {
            "transform": transform,
            "block_encoding": block_encoding,
            "polynomial": polynomial,
            "normalization": normalization,
            "minimum_singular_value": minimum_singular_value,
            "normalized_gap": normalized_gap,
            "spectral_domain": ((-1.0, -normalized_gap), (normalized_gap, 1.0)),
            "reciprocal_scale": reciprocal_scale,
            "operator_scale": operator_scale,
            "approximation_error_bound": approximation_error_bound,
            "total_error_bound": total_error_bound,
            "queries": polynomial.queries,
            "projector_rotations": polynomial.projector_rotations,
            "success": polynomial.success,
        }
        for name, value in values.items():
            object.__setattr__(result, name, value)
        return result


def synthesize_qsvt_polynomial(
    chebyshev_coefficients: Sequence[float],
    *,
    max_error: float = 1e-8,
    max_iterations: int = 512,
) -> CertifiedQSVTPolynomial:
    """Synthesize and certify one target polynomial as a QSP real component.

    The coefficient at position ``j`` multiplies the first-kind Chebyshev
    polynomial ``T_j``. A single leading Chebyshev term has a direct analytic
    realization. General bounded targets use Rust's symmetric damped-Newton
    solver inside a conservative coefficient-L1 region and are independently
    replayed before a certificate is returned.

    Args:
        chebyshev_coefficients (Sequence[float]): Nonempty finite ascending
            Chebyshev coordinates whose nonzero terms have the degree parity.
        max_error (float): Requested positive uniform error bound on ``[-1, 1]``.
        max_iterations (int): Positive bounded solver iteration limit.

    Returns:
        CertifiedQSVTPolynomial: Immutable real-component certificate accepted
            directly by :func:`qsvt`.

    Raises:
        TypeError: If host values have the wrong shape or native synthesis is
            unavailable or violates its projection ABI.
        ValueError: If Rust rejects the polynomial, guarantee, or work budget.
    """

    coefficients = _finite_reals(
        chebyshev_coefficients,
        name="Chebyshev coefficients",
    )
    tolerance = _finite_real(max_error, name="max_error")
    if tolerance <= 0.0:
        raise ValueError("max_error must be positive")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise TypeError("max_iterations must be an integer")
    iteration_limit = getattr(native, "MAX_QSVT_SYNTHESIS_ITERATIONS", None)
    synthesize = getattr(native, "synthesize_qsvt_polynomial", None)
    if (
        type(iteration_limit) is not int
        or iteration_limit <= 0
        or not callable(synthesize)
    ):
        raise TypeError("Q7700: native QSVT polynomial synthesis is unavailable")
    if not 1 <= max_iterations <= iteration_limit:
        raise ValueError(
            f"Q7706: QSVT synthesis iterations must be between 1 and {iteration_limit}"
        )
    try:
        projection = synthesize(list(coefficients), tolerance, max_iterations)
    except ValueError as error:
        raise ValueError(str(error)) from error
    return _decode_polynomial_certificate(
        projection,
        expected_coefficients=coefficients,
        allowed_error=tolerance,
    )


def reciprocal_svt(
    encoding: BlockEncoding,
    *,
    minimum_singular_value: float,
    max_error: float = 1e-8,
    max_degree: int = 63,
    max_iterations: int = 512,
) -> CertifiedReciprocalSVT:
    """Synthesize a certified reciprocal transform over a gapped spectrum.

    Rust chooses an odd minimax reciprocal polynomial, synthesizes symmetric
    QSP phases, and independently replays both claims. An equal LCU of the QSP
    response and its adjoint extracts the certified real component with one
    additional signal qubit. The returned transform remains ordinary Qamomile
    source, so simulation, transpilation, execution, persistence, and resource
    estimation see the same linked program.

    Args:
        encoding (BlockEncoding): Exact block encoding with physical
            normalization ``alpha``.
            A nonzero reconstruction error requires a separate perturbation
            bound and is rejected by this reciprocal certificate.
        minimum_singular_value (float): Positive physical gap no larger than
            ``alpha``.
        max_error (float): Positive combined approximation and realization error.
        max_degree (int): Positive degree budget; the producer selects an odd
            degree.
        max_iterations (int): Positive bounded phase-synthesis iteration budget.

    Returns:
        CertifiedReciprocalSVT: Reciprocal block encoding plus replayable error,
            scale, and per-QSP-branch cost evidence.

    Raises:
        TypeError: If a host value has the wrong type or native support is absent.
        ValueError: If Rust rejects the domain, tolerance, budgets, or proof.
    """

    if not isinstance(encoding, BlockEncoding):
        raise TypeError("encoding must be a BlockEncoding")
    if encoding.error_bound > 0.0:
        raise ValueError(
            "reciprocal_svt requires an exact block encoding; "
            "nonzero error_bound is not covered by its certificate"
        )
    minimum = _finite_real(
        minimum_singular_value,
        name="minimum_singular_value",
    )
    tolerance = _finite_real(max_error, name="max_error")
    if minimum <= 0.0 or minimum > encoding.normalization:
        raise ValueError(
            "minimum_singular_value must be positive and no larger than normalization"
        )
    if tolerance <= 0.0:
        raise ValueError("max_error must be positive")
    if isinstance(max_degree, bool) or not isinstance(max_degree, int):
        raise TypeError("max_degree must be an integer")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise TypeError("max_iterations must be an integer")
    degree_limit = getattr(native, "MAX_RECIPROCAL_QSVT_DEGREE", None)
    iteration_limit = getattr(native, "MAX_QSVT_SYNTHESIS_ITERATIONS", None)
    synthesize = getattr(native, "synthesize_reciprocal_qsvt", None)
    if (
        type(degree_limit) is not int
        or degree_limit <= 0
        or type(iteration_limit) is not int
        or iteration_limit <= 0
        or not callable(synthesize)
    ):
        raise TypeError("Q7720: native reciprocal SVT synthesis is unavailable")
    if not 1 <= max_degree <= degree_limit:
        raise ValueError(
            f"Q7723: reciprocal SVT degree must be between 1 and {degree_limit}"
        )
    if not 1 <= max_iterations <= iteration_limit:
        raise ValueError(
            f"Q7706: QSVT synthesis iterations must be between 1 and {iteration_limit}"
        )
    try:
        projection = synthesize(
            encoding.normalization,
            minimum,
            tolerance,
            max_degree,
            max_iterations,
        )
    except ValueError as error:
        raise ValueError(str(error)) from error
    decoded = _decode_reciprocal_certificate(
        projection,
        expected_normalization=encoding.normalization,
        expected_minimum=minimum,
        allowed_error=tolerance,
    )
    replayed = _replay_reciprocal_certificate(decoded, allowed_error=tolerance)
    raw_transform = qsvt(encoding, polynomial=replayed.polynomial)
    reciprocal_encoding = _real_polynomial_block_encoding(
        raw_transform,
        source_signal_width=encoding.signal_width,
        system_width=encoding.system_width,
        error_bound=replayed.total_error_bound,
    )
    return CertifiedReciprocalSVT._from_projection(
        transform=reciprocal_encoding.unitary,
        block_encoding=reciprocal_encoding,
        polynomial=replayed.polynomial,
        normalization=replayed.normalization,
        minimum_singular_value=replayed.minimum_singular_value,
        normalized_gap=replayed.normalized_gap,
        reciprocal_scale=replayed.reciprocal_scale,
        operator_scale=replayed.operator_scale,
        approximation_error_bound=replayed.approximation_error_bound,
        total_error_bound=replayed.total_error_bound,
    )


def _real_polynomial_block_encoding(
    raw_transform: qm.QKernel,
    *,
    source_signal_width: int,
    system_width: int,
    error_bound: float,
) -> BlockEncoding:
    """Extract the real certified polynomial from one complex QSP response.

    Symmetric QSP phase synthesis certifies the real part of the projected
    response. An equal LCU of the response and its adjoint realizes
    ``(U + U adjoint) / 2``. The LCU selector and original signal register are
    then one complete signal state for the returned block encoding.

    Args:
        raw_transform (qm.QKernel): QSP response over source signal and system.
        source_signal_width (int): Width of the original projected signal state.
        system_width (int): Width of the transformed system register.
        error_bound (float): Certified target-polynomial error to retain.

    Returns:
        BlockEncoding: Real-polynomial encoding with one additional leading
        signal qubit.

    Raises:
        TypeError: If the transform ABI or a numeric coordinate is invalid.
        ValueError: If a width or the error bound is outside its valid domain.
    """

    checked_signal_width = _positive_width(source_signal_width, "source_signal_width")
    checked_system_width = _positive_width(system_width, "system_width")
    transformed_width = checked_signal_width + checked_system_width

    @qm.qkernel
    def adjoint_response(
        state: qm.Q[qm.Bits[transformed_width]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[transformed_width]]:  # type: ignore[valid-type]
        """Apply the adjoint QSP response.

        Args:
            state (qm.Q[qm.Bits[transformed_width]]): Source signal and system.

        Returns:
            qm.Q[qm.Bits[transformed_width]]: State after the adjoint response.
        """

        return qm.adjoint(raw_transform, state)

    hermitian_part = lcu_block_encoding(
        ((0.5, raw_transform), (0.5, adjoint_response)),
        system_width=transformed_width,
    )
    return BlockEncoding(
        hermitian_part.unitary,
        normalization=1.0,
        signal_width=hermitian_part.signal_width + checked_signal_width,
        system_width=checked_system_width,
        error_bound=error_bound,
    )


@qm.qkernel
def _scalar_phase(
    state: qm.Q[qm.Bits[n]],  # type: ignore[valid-type]
    angle: qm.Real,  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[n]]:  # type: ignore[valid-type]
    """Apply a scalar phase as a controllable unary unitary."""

    return qm.global_phase(state, angle)


def qsvt(
    encoding: BlockEncoding,
    *,
    phases: Sequence[float] | QSVTPhases | None = None,
    polynomial: CertifiedQSVTPolynomial | None = None,
) -> qm.QKernel:
    r"""Compose QSVT from an exact block encoding and projector phases.

    For ``phases = (phi_0, ..., phi_d)``, the returned unary transform applies
    ``R(phi_0), U, R(phi_1), U adjoint, ...`` where
    ``R(phi) = exp(i phi (2 Pi - I))`` and ``Pi`` projects the complete signal
    register onto all zero. Phase synthesis and polynomial-domain validation
    remain separate classical functions; this helper implements only the exact
    quantum sequence.

    Composition happens at decoration time. The resulting ``QKernel`` is
    ordinary linked source containing calls, coherent control, global phase,
    and the common adjoint transform. QCore gains no QSVT instruction,
    descriptor registry, inverse implementation, or resource-estimator rule.

    Args:
        encoding (BlockEncoding): Exact construction-independent block-encoding
            contract.
        phases (Sequence[float] | QSVTPhases | None): Nonempty finite
            projector-phase sequence, or canonical phase coordinates produced
            by :class:`QSVTPhases`. Mutually exclusive with ``polynomial``.
        polynomial (CertifiedQSVTPolynomial | None): Rust-certified
            real-component synthesis result. Its certificate is replayed before
            projector phases are consumed; it does not assert that the
            response's complementary component is zero.

    Returns:
        qm.QKernel: Unary transform over the joint signal/system state.

    Raises:
        TypeError: If ``encoding`` or a phase has the wrong host type.
        ValueError: If the phase sequence is empty or contains a nonfinite
            value.
    """

    if not isinstance(encoding, BlockEncoding):
        raise TypeError("encoding must be a BlockEncoding")
    if (phases is None) == (polynomial is None):
        raise TypeError("provide exactly one of phases or polynomial")
    if polynomial is not None:
        if not isinstance(polynomial, CertifiedQSVTPolynomial):
            raise TypeError("polynomial must be a CertifiedQSVTPolynomial")
        frozen_phases = _replay_polynomial_certificate(polynomial)
    else:
        frozen_phases = _validate_phases(phases)
    total_width = encoding.total_width
    signal_width = encoding.signal_width
    unitary = encoding.unitary
    rotation = _projector_rotation(
        total_width=total_width,
        signal_width=signal_width,
    )
    first_phase = frozen_phases[0]
    steps = tuple(
        (position % 2 != 0, phase)
        for position, phase in enumerate(frozen_phases[1:], start=1)
    )

    if not steps:

        @qm.qkernel
        def single_phase(
            state: qm.Q[qm.Bits[total_width]],  # type: ignore[valid-type]
        ) -> qm.Q[qm.Bits[total_width]]:  # type: ignore[valid-type]
            """Apply the sole configured projector phase.

            Args:
                state (qm.Q[qm.Bits[total_width]]): Signal and system state.

            Returns:
                qm.Q[qm.Bits[total_width]]: Phase-rotated state.
            """

            return rotation(state, first_phase)

        return single_phase

    @qm.qkernel
    def transform(
        state: qm.Q[qm.Bits[total_width]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[total_width]]:  # type: ignore[valid-type]
        """Apply the configured QSVT phase sequence.

        Args:
            state (qm.Q[qm.Bits[total_width]]): Signal and system state.

        Returns:
            qm.Q[qm.Bits[total_width]]: QSVT-transformed state.
        """

        state = rotation(state, first_phase)
        for step in steps:
            if step[0]:
                state = unitary(state)
            else:
                state = qm.adjoint(unitary, state)
            state = rotation(state, step[1])
        return state

    return transform


def qsvt_from_phases(
    encoding: BlockEncoding,
    *,
    degree: int,
) -> qm.QKernel:
    r"""Construct a QSVT transform with a runtime phase vector.

    The returned function accepts the joint signal/system state followed by
    ``Vector[Real, degree + 1]``. It is therefore compiled once and can receive
    phases through ``Param[Vector[...]]`` at execution time. Indexing follows
    the same rank-aware array rule as ordinary Matrix access; no phase-specific
    container or QSVT IR operation is introduced.

    Args:
        encoding: Exact construction-independent block-encoding contract.
        degree: Nonnegative polynomial degree. The phase-vector length is one
            greater than this value.

    Returns:
        A binary transform over the joint state and runtime phase vector.

    Raises:
        TypeError: If ``encoding`` is not a block encoding or ``degree`` is not
            an integer.
        ValueError: If ``degree`` is negative.
    """

    if not isinstance(encoding, BlockEncoding):
        raise TypeError("encoding must be a BlockEncoding")
    if isinstance(degree, bool) or not isinstance(degree, int):
        raise TypeError("degree must be an integer")
    if degree < 0:
        raise ValueError("degree must be nonnegative")
    phase_count = degree + 1
    total_width = encoding.total_width
    unitary = encoding.unitary
    rotation = _projector_rotation(
        total_width=total_width,
        signal_width=encoding.signal_width,
    )

    @qm.qkernel
    def transform(
        state: qm.Q[qm.Bits[total_width]],  # type: ignore[valid-type]
        phases: qm.Vector[qm.Real, phase_count],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[total_width]]:  # type: ignore[valid-type]
        """Apply QSVT with runtime projector phases.

        Args:
            state (qm.Q[qm.Bits[total_width]]): Signal and system state.
            phases (qm.Vector[qm.Real, phase_count]): Runtime phase vector.

        Returns:
            qm.Q[qm.Bits[total_width]]: QSVT-transformed state.
        """

        state = rotation(state, phases[0])
        for position in range(1, phase_count):
            if position % 2 == 1:
                state = unitary(state)
            else:
                state = qm.adjoint(unitary, state)
            state = rotation(state, phases[position])
        return state

    return transform


def qsvt_application(
    *,
    signal_width: int,
    system_width: int,
    degree: int,
) -> qm.QKernel:
    """Declare a QSVT step whose block is selected when its caller is bound.

    The returned helper depends only on register shape and polynomial degree.
    Pass it a ``BlockEncoding`` inside another qkernel; the caller can then be
    compiled with different encoded blocks without redefining the algorithm.

    Args:
        signal_width (int): Positive projected signal width.
        system_width (int): Positive encoded system width.
        degree (int): Nonnegative degree; runtime phases have length degree + 1.

    Returns:
        qm.QKernel: Higher-order source step over block, state, and phases.

    Raises:
        TypeError: If a width or degree is not an integer.
        ValueError: If a width is nonpositive or degree is negative.
    """

    signal_width = _positive_width(signal_width, "signal_width")
    system_width = _positive_width(system_width, "system_width")
    if isinstance(degree, bool) or not isinstance(degree, int):
        raise TypeError("degree must be an integer")
    if degree < 0:
        raise ValueError("degree must be nonnegative")
    total_width = signal_width + system_width
    phase_count = degree + 1
    rotation = _projector_rotation(
        total_width=total_width,
        signal_width=signal_width,
    )

    @qm.qkernel
    def apply(
        block: BlockEncoding[signal_width, system_width],  # type: ignore[valid-type, type-arg]
        state: qm.Q[qm.Bits[total_width]],  # type: ignore[valid-type]
        phases: qm.Vector[qm.Real, phase_count],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[total_width]]:  # type: ignore[valid-type]
        """Apply runtime projector phases around a later-selected block.

        Args:
            block (BlockEncoding[signal_width, system_width]): Encoded unitary.
            state (qm.Q[qm.Bits[total_width]]): Joint coherent register.
            phases (qm.Vector[qm.Real, phase_count]): Projector phases.

        Returns:
            qm.Q[qm.Bits[total_width]]: Transformed joint register.
        """

        state = rotation(state, phases[0])
        for position in range(1, phase_count):
            if position % 2 == 1:
                state = block(state)  # type: ignore[operator]
            else:
                state = qm.adjoint(block, state)
            state = rotation(state, phases[position])
        return state

    return apply


def _projector_rotation(
    *,
    total_width: int,
    signal_width: int,
) -> qm.QKernel:
    """Construct one ancilla-free all-zero signal projector rotation."""

    @qm.qkernel
    def rotate(
        state: qm.Q[qm.Bits[total_width]],  # type: ignore[valid-type]
        phase: qm.Real,  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[total_width]]:  # type: ignore[valid-type]
        """Rotate the all-zero signal subspace by one phase.

        Args:
            state (qm.Q[qm.Bits[total_width]]): Signal and system state.
            phase (qm.Real): Projector rotation phase in radians.

        Returns:
            qm.Q[qm.Bits[total_width]]: Phase-rotated state.
        """

        signal, system = qm.split_at(state, signal_width)
        signal, system = qm.control(
            _scalar_phase,
            signal,
            system,
            2.0 * phase,
            control_state=0,
        )
        state = qm.concat(signal, system)
        return qm.global_phase(state, -phase)

    return rotate


def _validate_phases(phases: object) -> tuple[float, ...]:
    """Freeze one nonempty finite real phase sequence."""

    if isinstance(phases, QSVTPhases):
        return phases.phases
    if isinstance(phases, (str, bytes)) or not isinstance(phases, Sequence):
        raise TypeError("phases must be a sequence of real scalars")
    result = _finite_reals(phases, name="each QSVT phase")
    if not result:
        raise ValueError("QSVT requires at least one phase")
    return result


def _replay_polynomial_certificate(
    certificate: CertifiedQSVTPolynomial,
) -> tuple[float, ...]:
    """Recheck one certificate through Rust before lowering consumes it."""

    verify = getattr(native, "verify_qsvt_polynomial", None)
    if not callable(verify):
        raise TypeError("Q7700: native QSVT polynomial verification is unavailable")
    try:
        coefficients = list(certificate.chebyshev_coefficients)
        qsp_phases = list(certificate.qsp_phases)
        claimed_error = max(
            certificate.uniform_error_bound,
            float.fromhex("0x0.0000000000001p-1022"),
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise TypeError("Q7711: malformed QSVT polynomial certificate") from error
    try:
        projection = verify(coefficients, qsp_phases, claimed_error)
    except TypeError as error:
        raise TypeError("Q7711: malformed QSVT polynomial certificate") from error
    except ValueError as error:
        raise ValueError(str(error)) from error
    replayed = _decode_polynomial_certificate(
        projection,
        expected_coefficients=certificate.chebyshev_coefficients,
        allowed_error=claimed_error,
    )
    try:
        agrees = replayed == certificate
    except (AttributeError, TypeError, ValueError):
        agrees = False
    if not agrees:
        raise TypeError(
            "Q7711: native QSVT certificate replay disagrees with its value"
        )
    return replayed.phases.phases


def _replay_reciprocal_certificate(
    certificate: _DecodedReciprocalSVT,
    *,
    allowed_error: float,
) -> _DecodedReciprocalSVT:
    """Recheck one reciprocal certificate through Rust before composition."""

    verify = getattr(native, "verify_reciprocal_qsvt", None)
    if not callable(verify):
        raise TypeError("Q7720: native reciprocal SVT verification is unavailable")
    try:
        projection = verify(
            certificate.normalization,
            certificate.minimum_singular_value,
            certificate.reciprocal_scale,
            list(certificate.polynomial.chebyshev_coefficients),
            list(certificate.polynomial.qsp_phases),
            allowed_error,
        )
    except TypeError as error:
        raise TypeError("Q7726: malformed reciprocal SVT certificate") from error
    except ValueError as error:
        raise ValueError(str(error)) from error
    replayed = _decode_reciprocal_certificate(
        projection,
        expected_normalization=certificate.normalization,
        expected_minimum=certificate.minimum_singular_value,
        allowed_error=allowed_error,
    )
    if replayed != certificate:
        raise TypeError("Q7726: native reciprocal SVT replay disagrees with its value")
    return replayed


def _decode_reciprocal_certificate(
    projection: object,
    *,
    expected_normalization: float,
    expected_minimum: float,
    allowed_error: float,
) -> _DecodedReciprocalSVT:
    """Validate and freeze one complete native reciprocal projection."""

    candidate = cast(_NativeReciprocalQsvtCertificate, projection)
    try:
        normalization = candidate.normalization
        minimum = candidate.minimum_singular_value
        normalized_gap = candidate.normalized_gap
        reciprocal_scale = candidate.reciprocal_scale
        operator_scale = candidate.operator_scale
        approximation_error = candidate.approximation_error_bound
        total_error = candidate.total_error_bound
        polynomial_projection = candidate.polynomial
        raw_coefficients = polynomial_projection.chebyshev_coefficients
    except (AttributeError, TypeError, ValueError) as error:
        raise TypeError("Q7726: malformed native reciprocal SVT certificate") from error
    scalars = (
        normalization,
        minimum,
        normalized_gap,
        reciprocal_scale,
        operator_scale,
        approximation_error,
        total_error,
    )
    if (
        any(type(value) is not float or not math.isfinite(value) for value in scalars)
        or normalization != expected_normalization
        or minimum != expected_minimum
        or normalized_gap != minimum / normalization
        or not 0.0 < normalized_gap <= 1.0
        or reciprocal_scale <= 0.0
        or operator_scale != reciprocal_scale * normalization
        or approximation_error < 0.0
        or not approximation_error <= total_error <= allowed_error
        or not isinstance(raw_coefficients, list)
    ):
        raise TypeError("Q7726: malformed native reciprocal SVT certificate")
    polynomial = _decode_polynomial_certificate(
        polynomial_projection,
        expected_coefficients=tuple(raw_coefficients),
        allowed_error=allowed_error - approximation_error,
    )
    if not polynomial.odd:
        raise TypeError("Q7726: reciprocal SVT polynomial must be odd")
    return _DecodedReciprocalSVT(
        polynomial=polynomial,
        normalization=normalization,
        minimum_singular_value=minimum,
        normalized_gap=normalized_gap,
        reciprocal_scale=reciprocal_scale,
        operator_scale=operator_scale,
        approximation_error_bound=approximation_error,
        total_error_bound=total_error,
    )


def _decode_polynomial_certificate(
    projection: object,
    *,
    expected_coefficients: tuple[float, ...],
    allowed_error: float,
) -> CertifiedQSVTPolynomial:
    """Validate and freeze one complete native certificate projection."""

    candidate = cast(_NativeQsvtPolynomialCertificate, projection)
    try:
        raw_coefficients = candidate.chebyshev_coefficients
        raw_qsp_phases = candidate.qsp_phases
        raw_projector_phases = candidate.projector_phases
        degree = candidate.degree
        odd = candidate.odd
        uniform_error_bound = candidate.uniform_error_bound
        target_magnitude_bound = candidate.target_magnitude_bound
        realized_magnitude_bound = candidate.realized_magnitude_bound
        queries = candidate.queries
        projector_rotations = candidate.projector_rotations
        success = candidate.success
    except (AttributeError, TypeError, ValueError) as error:
        raise TypeError(
            "Q7711: malformed native QSVT polynomial certificate"
        ) from error
    if (
        not isinstance(raw_coefficients, list)
        or tuple(raw_coefficients) != expected_coefficients
        or not isinstance(raw_qsp_phases, list)
        or not isinstance(raw_projector_phases, list)
        or len(raw_qsp_phases) != len(expected_coefficients)
        or len(raw_projector_phases) != len(expected_coefficients)
        or any(
            type(value) is not float
            or not math.isfinite(value)
            or not -math.pi <= value < math.pi
            for value in raw_qsp_phases
        )
        or any(
            type(value) is not float
            or not math.isfinite(value)
            or not -math.pi <= value < math.pi
            for value in raw_projector_phases
        )
        or any(
            left != right
            for left, right in zip(
                raw_qsp_phases, reversed(raw_qsp_phases), strict=True
            )
        )
        or type(degree) is not int
        or degree != len(expected_coefficients) - 1
        or type(odd) is not bool
        or odd is not (degree % 2 == 1)
        or type(uniform_error_bound) is not float
        or not math.isfinite(uniform_error_bound)
        or not 0.0 <= uniform_error_bound <= allowed_error
        or type(target_magnitude_bound) is not float
        or not math.isfinite(target_magnitude_bound)
        or target_magnitude_bound < 0.0
        or type(realized_magnitude_bound) is not float
        or not math.isfinite(realized_magnitude_bound)
        or not 0.0 <= realized_magnitude_bound <= 1.0
        or type(queries) is not int
        or queries != degree
        or type(projector_rotations) is not int
        or projector_rotations != degree + 1
        or success != QSVTSuccess.HERALDED.value
    ):
        raise TypeError("Q7711: malformed native QSVT polynomial certificate")
    qsp_phases = tuple(raw_qsp_phases)
    projector_phases = tuple(raw_projector_phases)
    phases = QSVTPhases(
        projector_phases,
        odd=odd,
        domain=(-1.0, 1.0),
    )
    return CertifiedQSVTPolynomial._from_projection(
        chebyshev_coefficients=expected_coefficients,
        phases=phases,
        qsp_phases=qsp_phases,
        degree=degree,
        odd=odd,
        uniform_error_bound=uniform_error_bound,
        target_magnitude_bound=target_magnitude_bound,
        realized_magnitude_bound=realized_magnitude_bound,
        queries=queries,
        projector_rotations=projector_rotations,
        success=QSVTSuccess(success),
    )


def _convert_qsp_wx_phases(phases: object) -> tuple[float, ...]:
    """Validate host coordinates and project the native phase transform."""

    if isinstance(phases, (str, bytes)) or not isinstance(phases, Sequence):
        raise TypeError("phases must be a sequence of real scalars")
    source = _finite_reals(phases, name="phases")
    if not source:
        raise ValueError("QSVT requires at least one phase")
    limit = getattr(native, "MAX_QSVT_PHASES", None)
    convert = getattr(native, "convert_qsp_wx_phases", None)
    if type(limit) is not int or limit <= 0 or not callable(convert):
        raise TypeError("Q7620: native QSVT phase conversion is unavailable")
    if len(source) > limit:
        raise ValueError(f"Q7622: QSVT phase count exceeds the {limit}-entry budget")
    projection = convert(source)
    if (
        not isinstance(projection, list)
        or len(projection) != len(source)
        or any(
            type(value) is not float
            or not math.isfinite(value)
            or not -math.pi <= value < math.pi
            for value in projection
        )
    ):
        raise TypeError("Q7625: malformed native QSVT phase projection")
    return tuple(projection)


def _finite_reals(values: Sequence[object], *, name: str) -> tuple[float, ...]:
    """Freeze a sequence of finite non-boolean real scalars."""

    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of real scalars")
    return tuple(_finite_real(value, name=name) for value in values)


def _finite_real(value: object, *, name: str) -> float:
    """Normalize one finite non-boolean real scalar."""

    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


__all__ = [
    "CertifiedReciprocalSVT",
    "CertifiedQSVTPolynomial",
    "QSVTPhases",
    "QSVTSuccess",
    "qsvt",
    "qsvt_from_phases",
    "reciprocal_svt",
    "synthesize_qsvt_polynomial",
]
