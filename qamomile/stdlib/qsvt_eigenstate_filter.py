# ruff: noqa: F821 -- dependent widths are closed into generated source.

"""Real-polynomial eigenstate filters composed from certified QSVT source.

The selected input operator is assumed Hermitian. Phase candidates may come
from the external ``pyqsp`` producer, but Rust independently replays
their polynomial and owns their conversion to projector coordinates. A raw
phase sequence makes no sign-approximation claim. The coherent real-part
extraction is retained because a QSP real-component certificate does not
eliminate the response's complementary imaginary polynomial.
"""

from __future__ import annotations

import contextlib
import io
import logging
import math
from collections.abc import Sequence

import numpy as np

import qamomile as qm
from qamomile._native import compiler as native

from .block_encoding import BlockEncoding
from .qsvt import (
    CertifiedQSVTPolynomial,
    QSVTPhases,
    _decode_polynomial_certificate,
    _finite_real,
    _real_polynomial_block_encoding,
    qsvt,
)
from .signals import Signal, sample
from .variational import uniform_superposition

DEFAULT_POLYNOMIAL_DEGREE = 61
DEFAULT_TRANSITION_WIDTH = 20.0
DEFAULT_POLYNOMIAL_SCALE = 1.10
_LOGGER = logging.getLogger(__name__)


def synthesize_sign_filter(
    *,
    degree: int = DEFAULT_POLYNOMIAL_DEGREE,
    delta: float = DEFAULT_TRANSITION_WIDTH,
    scale: float = DEFAULT_POLYNOMIAL_SCALE,
    max_error: float = 1e-6,
) -> CertifiedQSVTPolynomial:
    """Produce an odd sign-like polynomial and certify its QSP realization.

    ``pyqsp`` proposes a bounded Chebyshev approximation to ``erf(delta*x)``.
    Rust replays the full candidate before returning its real-component
    certificate; no sampled magnitude check substitutes for that replay.
    Positive ``scale`` selects lower energies when consumed by the projector.
    ``delta`` controls the transition near zero, without a certified spectral
    gap or an independent uniform approximation claim for the discontinuous
    sign function.

    Args:
        degree (int): Positive odd degree, at most the native degree budget.
        delta (float): Positive finite transition sharpness, default 20.
        scale (float): Finite nonzero multiplier of the proposed polynomial.
            Positive values retain energies below the threshold; negative
            values retain energies above it. Unrealizable targets are rejected.
        max_error (float): Positive uniform polynomial-realization tolerance.

    Returns:
        CertifiedQSVTPolynomial: Independently replayed real-component target.

    Raises:
        TypeError: If host types or the required native boundary are invalid.
        ValueError: If the degree, coordinates, or phase certificate is invalid.
    """
    if type(degree) is not int:
        raise TypeError("degree must be an integer")
    limit = getattr(native, "MAX_CERTIFIED_QSVT_DEGREE", None)
    verify = getattr(native, "verify_qsvt_polynomial", None)
    if type(limit) is not int or limit <= 0 or not callable(verify):
        raise TypeError("Q7700: native QSVT polynomial verification is unavailable")
    if not 1 <= degree <= limit or degree % 2 != 1:
        raise ValueError(f"degree must be positive and odd, and at most {limit}")
    sharpness = _finite_real(delta, name="delta")
    multiplier = _finite_real(scale, name="scale")
    tolerance = _finite_real(max_error, name="max_error")
    if sharpness <= 0.0:
        raise ValueError("delta must be positive")
    if multiplier == 0.0:
        raise ValueError("scale must be nonzero")
    if tolerance <= 0.0:
        raise ValueError("max_error must be positive")
    from pyqsp.angle_sequence import QuantumSignalProcessingPhases
    from pyqsp.poly import PolySign

    _LOGGER.info(
        "synthesizing sign filter: degree=%s, delta=%s, scale=%s",
        degree,
        sharpness,
        multiplier,
    )
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            generated, _ = PolySign().generate(
                degree=degree,
                delta=sharpness,
                ensure_bounded=True,
                return_scale=True,
                chebyshev_basis=True,
            )
            coefficients = tuple(
                multiplier * float(value) for value in np.asarray(generated).ravel()
            )
            proposed, _, _ = QuantumSignalProcessingPhases(
                list(coefficients), method="sym_qsp", chebyshev_basis=True
            )
    finally:
        for line in output.getvalue().splitlines():
            if line.strip():
                _LOGGER.debug("%s", line)
    # The external producer encodes its target in the imaginary Wx component.
    # These candidate endpoint offsets select its positive real component;
    # Rust independently checks that claim and converts the phase convention.
    candidate = [float(value) for value in proposed]
    candidate[0] -= math.pi / 4.0
    candidate[-1] -= math.pi / 4.0
    projection = verify(list(coefficients), candidate, tolerance)
    return _decode_polynomial_certificate(
        projection,
        expected_coefficients=coefficients,
        allowed_error=tolerance,
    )


def qsvt_filter_projector(
    encoding: BlockEncoding,
    *,
    phases: Sequence[float] | QSVTPhases | None = None,
    polynomial: CertifiedQSVTPolynomial | None = None,
) -> BlockEncoding:
    """Encode the filtered Hermitian block ``(I - Re(QSVT)) / 2``.

    The source's selected operator must be Hermitian for a certified polynomial
    to describe its eigenvalues. Its sign-like response selects eigenvalues
    below zero; shift the original Hamiltonian by the desired threshold first.
    One leading extraction qubit removes the complementary imaginary response
    and another performs the projector Hadamard test. Raw phases instead define
    the exact Hermitian part of the chosen QSVT block, with no sign guarantee.

    Args:
        encoding (BlockEncoding): Exact selected encoding of the Hermitian input.
        phases (Sequence[float] | QSVTPhases | None): Explicit projector phases,
            mutually exclusive with ``polynomial``.
        polynomial (CertifiedQSVTPolynomial | None): Replayed real-component
            target, mutually exclusive with ``phases``.

    Returns:
        BlockEncoding: Normalization-one filter with two additional signal
        qubits. Its error is half the supplied polynomial realization error.

    Raises:
        TypeError: If a host value or QSVT phase contract is invalid.
        ValueError: If the source has nonzero reconstruction error or the
            supplied phases or certificate are invalid.
    """
    if not isinstance(encoding, BlockEncoding):
        raise TypeError("encoding must be a BlockEncoding")
    if encoding.error_bound != 0.0:
        raise ValueError("qsvt_filter_projector requires an exact block encoding")
    raw = qsvt(encoding, phases=phases, polynomial=polynomial)
    error = polynomial.uniform_error_bound if polynomial is not None else 0.0
    real = _real_polynomial_block_encoding(
        raw,
        source_signal_width=encoding.signal_width,
        system_width=encoding.system_width,
        error_bound=error,
    )
    response = real.unitary
    width = real.total_width + 1

    @qm.qkernel
    def projector(state: qm.Q[qm.Bits[width]]) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
        """Retain the coherent filtered block and all of its success qubits.

        Args:
            state (qm.Q[qm.Bits[width]]): Projector, extraction, and source state.

        Returns:
            qm.Q[qm.Bits[width]]: Complete Hadamard-test witness.
        """
        probe, workspace = qm.split_at(state, 1)
        probe = qm.h(probe)
        probe = qm.z(probe)
        probe, workspace = qm.control(response, probe, workspace)
        probe = qm.h(probe)
        return qm.concat(probe, workspace)

    return BlockEncoding(
        projector,
        normalization=1.0,
        signal_width=encoding.signal_width + 2,
        system_width=encoding.system_width,
        error_bound=error / 2.0,
    )


def qsvt_filter_probe(
    encoding: BlockEncoding,
    *,
    phases: Sequence[float] | QSVTPhases | None = None,
    polynomial: CertifiedQSVTPolynomial | None = None,
) -> qm.QKernel:
    """Sample a uniform input after the real-polynomial eigenstate filter.

    Args:
        encoding (BlockEncoding): Exact selected Hermitian input encoding.
        phases (Sequence[float] | QSVTPhases | None): Explicit projector phases,
            mutually exclusive with ``polynomial``.
        polynomial (CertifiedQSVTPolynomial | None): Replayed real-component
            target, mutually exclusive with ``phases``.

    Returns:
        qm.QKernel: Nullary sampler returning ``(Bool, Bits[system_width])``.
        Success means every projector, extraction, and source signal bit is zero.

    Raises:
        TypeError: If a host value or selected-process contract is invalid.
        ValueError: If source errors, phases, or native layout budgets are invalid.
    """
    filtered = qsvt_filter_projector(encoding, phases=phases, polynomial=polynomial)
    system_width = encoding.system_width

    @qm.qkernel
    def uniform(
        state: qm.Q[qm.Bits[system_width]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[system_width]]:  # type: ignore[valid-type]
        """Prepare the fixed-width uniform logical input.

        Args:
            state (qm.Q[qm.Bits[system_width]]): Fresh zero system register.

        Returns:
            qm.Q[qm.Bits[system_width]]: Uniform system superposition.
        """
        return uniform_superposition(state, system_width)

    return sample(Signal.from_block_encoding(filtered), source=uniform)


__all__ = [
    "qsvt_filter_probe",
    "qsvt_filter_projector",
    "synthesize_sign_filter",
]
