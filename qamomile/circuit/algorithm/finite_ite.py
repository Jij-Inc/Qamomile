r"""Build finite imaginary-time evolution (FinITE) circuits.

FinITE (*Finite Imaginary-Time Evolution for Polynomial Unconstrained Binary
Optimization*, arXiv:2604.27482) prepares the normalized imaginary-time state
of a diagonal Pauli-Z cost Hamiltonian
:math:`\hat H = \sum_{\mu \neq 0} x_\mu \sigma_\mu` by block-encoding
:math:`m(\beta) e^{-\beta \hat H}` with :math:`m(\beta) = e^{-\beta W}` and
post-selecting every ancilla on :math:`|0\rangle`.

Because the Pauli-Z strings pairwise commute, the imaginary-time operator
factorizes exactly, with no product-formula error:

.. math::

    e^{-\beta \hat H}
    = \prod_\mu \Bigl[
        \cosh(\beta |x_\mu|) I
        - \operatorname{sgn}(x_\mu) \sinh(\beta |x_\mu|) \sigma_\mu
      \Bigr].

Each bracket is a two-term linear combination of unitaries with coefficient
one-norm :math:`e^{\beta |x_\mu|}`, so one Ising-Z block encoding per Pauli
term suffices; :func:`~qamomile.circuit.product_block_encoding` chains them on
a shared system register, giving total normalization
:math:`\alpha_{\text{total}} = e^{\beta W}`.

The identity component of the Pauli expansion is deliberately excluded: it
cancels in the normalized post-selected state and would only inflate ``W``.
Callers pass identity-free coefficients and apply the scalar classically.

The problem-level driver, which turns a PUBO instance into these arguments and
decodes the post-selected shots, is
:class:`~qamomile.optimization.finite_ite.FinITEConverter`.

Only the block-encoding stage of the paper is implemented. Fixed-point
amplitude amplification of the post-selection success probability is not
included, so the observed acceptance rate is the raw
:math:`P_{\mathrm{LCU}}(\beta)`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import qamomile.circuit as qmc
from qamomile.circuit.frontend.qkernel import QKernel
from qamomile.circuit.stdlib.block_encoding import (
    IsingZBlockEncoding,
    LCUBlockEncoding,
    ising_z_block_encoding,
    product_block_encoding,
)

__all__ = [
    "finite_ite_block_encoding",
    "finite_ite_state",
]


def finite_ite_block_encoding(
    coefficients: Mapping[tuple[int, ...], float],
    beta: float,
    num_system_qubits: int,
) -> LCUBlockEncoding:
    r"""Block-encode the scaled FinITE propagator of a diagonal Hamiltonian.

    Builds one two-term factor per Pauli-Z term and chains them with
    :func:`~qamomile.circuit.product_block_encoding`, encoding
    :math:`m(\beta) e^{-\beta \hat H}` with :math:`W = \sum_{\mu \neq 0}|x_\mu|`.
    The returned normalization is ``exp(beta * W)``, and the construction uses
    one ancilla per term.

    Terms are processed in sorted-word order so the emitted circuit is
    deterministic for a given coefficient mapping. An identity entry ``()`` is
    rejected rather than silently absorbed: keeping it would inflate ``W``
    without changing the normalized post-selected state, so the caller must
    account for it classically.

    Args:
        coefficients (Mapping[tuple[int, ...], float]): Real, identity-free
            Ising-Z coefficients keyed by products of zero-based system-qubit
            indices. Must be nonempty.
        beta (float): Imaginary time. Must be finite and non-negative.
        num_system_qubits (int): Positive ordered system-register width.

    Returns:
        LCUBlockEncoding: Descriptor with ``num_signal_qubits`` equal to the
            number of terms and ``normalization`` equal to ``exp(beta * W)``.

    Raises:
        TypeError: If ``coefficients`` is not a mapping.
        ValueError: If ``coefficients`` is empty, contains the identity word
            ``()``, holds a non-finite coefficient, ``beta`` is negative or
            non-finite, or ``num_system_qubits`` is non-positive.

    Example:
        >>> import math
        >>> from qamomile.circuit.algorithm import finite_ite_block_encoding
        >>> encoding = finite_ite_block_encoding(
        ...     {(0,): 1.0, (0, 1): -1.0}, 0.5, 2
        ... )
        >>> encoding.num_signal_qubits
        2
        >>> math.isclose(encoding.normalization, math.exp(0.5 * 2.0))
        True
    """
    if not isinstance(coefficients, Mapping):
        raise TypeError("coefficients must be a mapping.")
    if () in coefficients:
        raise ValueError(
            "coefficients must be identity-free; apply the constant term "
            "classically as exp(-beta * c) instead of block-encoding it."
        )
    if not coefficients:
        raise ValueError("coefficients must contain at least one Pauli term.")
    _validate_beta(beta)

    return product_block_encoding(
        [
            _term_block_encoding(
                word,
                float(coefficients[word]),
                beta,
                num_system_qubits,
            )
            for word in sorted(coefficients)
        ]
    )


def finite_ite_state(
    encoding: LCUBlockEncoding,
) -> QKernel[..., tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]]:
    r"""Build the FinITE sampling circuit on the uniform initial state.

    Prepares :math:`|\psi_0\rangle = |+\rangle^{\otimes n}` on the system
    register, applies ``encoding``, and measures both registers. A shot is an
    LCU success exactly when every signal bit is zero; the surviving system
    bitstrings are distributed as the normalized imaginary-time state, and the
    success rate estimates :math:`P_{\mathrm{LCU}}(\beta)`.

    Both registers are measured, so post-selection is a purely classical
    filter over the returned pairs. Signal bits come first, matching the
    register order used by the other probe kernels in this package.

    The imaginary time is already baked into ``encoding``: the hyperbolic
    amplitudes are fixed when the descriptor is built, so sweeping
    :math:`\beta` means building one encoding and transpiling per value.

    Args:
        encoding (LCUBlockEncoding): Block-encoding descriptor of
            :math:`m(\beta) e^{-\beta \hat H}`, normally from
            :func:`finite_ite_block_encoding`. Register widths are read from
            it.

    Returns:
        QKernel[..., tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]]: Kernel
            with no classical inputs returning ``(signal_bits, system_bits)``.

    Raises:
        TypeError: If ``encoding`` is not an :class:`LCUBlockEncoding`.

    Example:
        >>> from qamomile.circuit.algorithm import (
        ...     finite_ite_block_encoding,
        ...     finite_ite_state,
        ... )
        >>> encoding = finite_ite_block_encoding(
        ...     {(0,): 1.0, (0, 1): -1.0}, 0.5, 2
        ... )
        >>> finite_ite_state(encoding).output_types
        [Vector[Bit], Vector[Bit]]
    """
    if not isinstance(encoding, LCUBlockEncoding):
        raise TypeError("encoding must be an LCUBlockEncoding.")

    num_signal_qubits = encoding.num_signal_qubits
    num_system_qubits = encoding.num_system_qubits

    @qmc.qkernel
    def finite_ite_sampling() -> tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]:
        """Sample the post-selected FinITE state and its success flag.

        Returns:
            tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]: Signal bits,
                all zero on LCU success, and the system bits.
        """
        signal = qmc.qubit_array(num_signal_qubits, "signal")
        system = qmc.qubit_array(num_system_qubits, "system")
        system = qmc.h(system)
        signal, system = encoding.unitary(signal, system)
        return qmc.measure(signal), qmc.measure(system)

    return finite_ite_sampling


def _term_block_encoding(
    word: tuple[int, ...],
    coefficient: float,
    beta: float,
    num_system_qubits: int,
) -> IsingZBlockEncoding:
    r"""Block-encode one FinITE factor of the imaginary-time product.

    Builds the two-term Ising-Z encoding of
    :math:`e^{-\beta x \sigma_{\text{word}}}`, namely
    ``cosh(beta * |x|) I - sgn(x) sinh(beta * |x|) Z_word``, whose coefficient
    one-norm is ``exp(beta * abs(x))``. At ``beta == 0`` the hyperbolic sine
    vanishes and the term is dropped, leaving an identity block; the signal
    register keeps its single pass-through qubit either way, so every term
    contributes the same width.

    Args:
        word (tuple[int, ...]): Zero-based system-qubit indices of the Pauli-Z
            string.
        coefficient (float): Real Pauli coefficient ``x`` of the term.
        beta (float): Imaginary time. Must be finite and non-negative.
        num_system_qubits (int): Positive ordered system-register width.

    Returns:
        IsingZBlockEncoding: Descriptor whose ``normalization`` equals
            ``exp(beta * abs(coefficient))``.

    Raises:
        ValueError: If ``coefficient`` is non-finite, or ``num_system_qubits``
            is non-positive.
    """
    if not math.isfinite(coefficient):
        raise ValueError("coefficient must be finite.")

    angle = beta * abs(coefficient)
    return ising_z_block_encoding(
        {
            (): math.cosh(angle),
            word: -math.copysign(math.sinh(angle), coefficient),
        },
        num_system_qubits,
    )


def _validate_beta(beta: float) -> None:
    """Reject imaginary times outside the supported range.

    Args:
        beta (float): Candidate imaginary time.

    Raises:
        ValueError: If ``beta`` is non-finite or negative.
    """
    if not math.isfinite(beta):
        raise ValueError("beta must be finite.")
    if beta < 0.0:
        raise ValueError("beta must be non-negative.")
