r"""Convert PUBO problems into FinITE imaginary-time evolution circuits.

FinITE (*Finite Imaginary-Time Evolution for Polynomial Unconstrained Binary
Optimization*, arXiv:2604.27482) prepares the normalized imaginary-time state
:math:`e^{-\beta \hat H}|\psi_0\rangle / \lVert \cdot \rVert` by block-encoding
:math:`m(\beta) e^{-\beta \hat H}` with a termwise linear combination of
unitaries and post-selecting every ancilla on :math:`|0\rangle`. Because
:math:`e^{-\beta \hat H}` exponentially reweights low energies, the surviving
shots concentrate on the optimum.

:class:`FinITEConverter` turns a PUBO instance into the arguments of
:func:`~qamomile.circuit.algorithm.finite_ite.finite_ite_block_encoding`,
feeds the resulting descriptor to
:func:`~qamomile.circuit.algorithm.finite_ite.finite_ite_state`, and decodes
the post-selected shots. The circuit construction itself lives in the
algorithm layer; this module only supplies problem data and interprets
results.

Only the block-encoding stage is implemented. The paper's second stage, fixed
point amplitude amplification of the post-selection success probability, is not
included, so the acceptance rate observed here is the raw
:math:`P_{\mathrm{LCU}}(\beta)`.
"""

from __future__ import annotations

import math

import ommx.v1

import qamomile.observable as qm_o
from qamomile.circuit.algorithm.finite_ite import (
    finite_ite_block_encoding,
    finite_ite_state,
)
from qamomile.circuit.stdlib.block_encoding import LCUBlockEncoding
from qamomile.circuit.transpiler.executable import ExecutableProgram
from qamomile.circuit.transpiler.job import SampleResult
from qamomile.circuit.transpiler.transpiler import Transpiler
from qamomile.optimization.binary_model import BinarySampleSet

from .converter import MathematicalProblemConverter

__all__ = [
    "FinITEConverter",
    "finite_ite_beta_threshold",
]


def finite_ite_beta_threshold(
    target_fidelity: float,
    *,
    spectral_gap: float,
    ground_overlap: float,
) -> float:
    r"""Compute the sufficient imaginary time for a target fidelity.

    Corollary 2 of arXiv:2604.27482, obtained by inverting the gap-based bound
    :math:`F_g(\beta) \ge \gamma_0 / (\gamma_0 + (1 - \gamma_0)e^{-2\beta\Delta})`:

    .. math::

        \beta^\star(\bar F) = \max\left\{
            0,\ \frac{1}{2\Delta}
            \log \frac{\bar F (1 - \gamma_0)}{\gamma_0 (1 - \bar F)}
        \right\}.

    Taking :math:`\beta \ge \beta^\star` guarantees :math:`F_g(\beta) \ge \bar F`.
    The condition is only sufficient, so the fidelity actually reached at
    :math:`\beta^\star` is usually higher.

    The gap and the overlap are estimates the caller must supply; neither is
    available for free on a generic instance.

    Args:
        target_fidelity (float): Target ground-subspace fidelity
            :math:`\bar F`, in ``[0, 1)``.
        spectral_gap (float): Positive estimate of the spectral gap
            :math:`\Delta` above the ground subspace.
        ground_overlap (float): Estimate of the initial ground-subspace
            overlap :math:`\gamma_0`, in ``[0, 1]``.

    Returns:
        float: The threshold :math:`\beta^\star`. Returns ``0.0`` when the
            initial state already meets the target
            (``target_fidelity <= ground_overlap``), and ``math.inf`` when
            ``ground_overlap`` is zero, since no finite imaginary time can
            populate a subspace with no initial overlap.

    Raises:
        ValueError: If ``target_fidelity`` is outside ``[0, 1)``,
            ``ground_overlap`` is outside ``[0, 1]``, or ``spectral_gap`` is
            non-positive or non-finite.

    Example:
        >>> round(finite_ite_beta_threshold(
        ...     0.99, spectral_gap=2.0, ground_overlap=0.125), 6)
        0.898189
    """
    if not 0.0 <= target_fidelity < 1.0:
        raise ValueError("target_fidelity must lie in [0, 1).")
    if not 0.0 <= ground_overlap <= 1.0:
        raise ValueError("ground_overlap must lie in [0, 1].")
    if not math.isfinite(spectral_gap) or spectral_gap <= 0.0:
        raise ValueError("spectral_gap must be finite and positive.")

    if target_fidelity <= ground_overlap:
        return 0.0
    if ground_overlap <= 0.0:
        return math.inf

    ratio = (target_fidelity * (1.0 - ground_overlap)) / (
        ground_overlap * (1.0 - target_fidelity)
    )
    return max(0.0, math.log(ratio) / (2.0 * spectral_gap))


class FinITEConverter(MathematicalProblemConverter):
    r"""Convert a PUBO problem into a FinITE imaginary-time evolution circuit.

    Block-encodes :math:`m(\beta) e^{-\beta \hat H}` with one two-term LCU
    ancilla per Pauli-Z term of the cost Hamiltonian and post-selects on all
    ancillas reading zero. Higher-order (HUBO) terms are block-encoded
    directly, without quadratization.

    The imaginary time :math:`\beta` is supplied per :meth:`transpile` call and
    is a compile-time value: the hyperbolic amplitudes of each LCU block are
    baked into the emitted circuit, so sweeping :math:`\beta` means transpiling
    once per value. :func:`finite_ite_beta_threshold` gives a sufficient
    :math:`\beta` when estimates of the gap and initial overlap are available.

    The model constant is left out of the LCU: it cancels in the normalized
    post-selected state and would only inflate :attr:`weight_norm`, shrinking
    the success probability for nothing. It remains available as
    ``self.spin_model.constant`` and on :meth:`get_cost_hamiltonian`.

    Attributes:
        weight_norm (float): One-norm :math:`W = \sum_{\mu \neq 0}|x_\mu|` of
            the non-identity coefficients, which sets the subnormalization
            :math:`m(\beta) = e^{-\beta W}`.

    Example:
        >>> from qamomile.optimization.binary_model import BinaryModel
        >>> model = BinaryModel.from_higher_ising({(0,): 1.0, (0, 1): -1.0})
        >>> converter = FinITEConverter(model)
        >>> converter.weight_norm
        2.0
        >>> converter.num_ancilla_bits
        2
    """

    def __post_init__(self) -> None:
        """Derive the identity-free Pauli terms and their one-norm.

        Raises:
            ValueError: If the model has no non-identity term, leaving nothing
                to block-encode.
        """
        self._terms: dict[tuple[int, ...], float] = {
            word: float(value)
            for word, value in self.spin_model.coefficients.items()
            if word
        }
        if not self._terms:
            raise ValueError(
                "FinITE requires at least one non-identity Pauli term; the "
                "model reduces to a constant."
            )
        self.weight_norm = math.fsum(abs(value) for value in self._terms.values())

    def get_cost_hamiltonian(self) -> qm_o.Hamiltonian:
        """Construct the diagonal Ising cost Hamiltonian of the problem.

        Built from the same identity-free terms the circuit encodes, so the
        returned operator cannot drift from the emitted block encoding. The
        constant is carried on ``Hamiltonian.constant`` rather than as an
        identity term, matching where it sits in the spin model.

        Returns:
            qm_o.Hamiltonian: Pauli-Z cost Hamiltonian including the constant.
        """
        hamiltonian = qm_o.Hamiltonian()
        for word, coefficient in self._terms.items():
            hamiltonian.add_term(
                tuple(qm_o.PauliOperator(qm_o.Pauli.Z, index) for index in word),
                coefficient,
            )
        hamiltonian.constant = self.spin_model.constant
        return hamiltonian

    @property
    def num_ancilla_bits(self) -> int:
        """Return the number of LCU ancilla qubits.

        Returns:
            int: One signal qubit per non-identity Pauli term.
        """
        return len(self._terms)

    def block_encoding(self, beta: float) -> LCUBlockEncoding:
        r"""Build the block encoding of :math:`m(\beta)e^{-\beta \hat H}`.

        Thin wrapper over
        :func:`~qamomile.circuit.algorithm.finite_ite.finite_ite_block_encoding`
        that supplies this problem's identity-free Pauli terms.

        Args:
            beta (float): Imaginary time. Must be finite and non-negative.

        Returns:
            LCUBlockEncoding: Descriptor with one signal qubit per Pauli term,
                normalization :math:`e^{\beta W}`, reusable as a child of
                another block-encoding composition.

        Raises:
            ValueError: If ``beta`` is negative or non-finite.
        """
        return finite_ite_block_encoding(self._terms, beta, self.spin_model.num_bits)

    def transpile(
        self,
        transpiler: Transpiler,
        *,
        beta: float,
    ) -> ExecutableProgram:
        r"""Transpile the FinITE sampling circuit at one imaginary time.

        The emitted circuit measures the ancillas as well as the system, so a
        shot is an LCU success exactly when every ancilla bit is zero;
        :meth:`decode` applies that filter.

        Args:
            transpiler (Transpiler): Backend transpiler to compile with.
            beta (float): Imaginary time. Must be finite and non-negative.
                Baked into the circuit, so each value needs its own call.

        Returns:
            ExecutableProgram: Compiled program returning
                ``(signal_bits, system_bits)`` per shot, with no runtime
                parameters.

        Raises:
            ValueError: If ``beta`` is negative or non-finite.
        """
        return transpiler.transpile(finite_ite_state(self.block_encoding(beta)))

    @staticmethod
    def _postselect(
        samples: SampleResult[tuple[list[int], list[int]]],
    ) -> tuple[list[tuple[list[int], int]], int]:
        """Keep only the shots whose ancillas all measured zero.

        Args:
            samples (SampleResult[tuple[list[int], list[int]]]): Raw results as
                ``(signal_bits, system_bits)`` pairs.

        Returns:
            tuple[list[tuple[list[int], int]], int]: The surviving
                ``(system_bits, count)`` entries and the total kept shots.

        Raises:
            ValueError: If a result value is not a two-element tuple.
        """
        kept: list[tuple[list[int], int]] = []
        kept_shots = 0
        for value, count in samples.results:
            if not isinstance(value, tuple) or len(value) != 2:
                raise ValueError(
                    "Expected FinITE results as (signal, system) tuples; "
                    f"got {value!r}."
                )
            signal_bits, system_bits = value
            if any(signal_bits):
                continue
            kept.append((list(system_bits), count))
            kept_shots += count
        return kept, kept_shots

    def success_probability(
        self,
        samples: SampleResult[tuple[list[int], list[int]]],
    ) -> float:
        r"""Estimate :math:`P_{\mathrm{LCU}}(\beta)` from measured shots.

        The acceptance rate falls exponentially with :math:`\beta`, so a small
        value means most shots were discarded, not that the algorithm failed.

        Args:
            samples (SampleResult[tuple[list[int], list[int]]]): Raw results
                from ``ExecutableProgram.sample(...).result()``.

        Returns:
            float: Fraction of shots with every ancilla bit zero, or ``0.0``
                when no shots were executed.

        Raises:
            ValueError: If a result value is not a two-element tuple.
        """
        _, kept_shots = self._postselect(samples)
        if samples.shots <= 0:
            return 0.0
        return kept_shots / samples.shots

    def decode_to_binary_sampleset(  # type: ignore[override]
        self,
        samples: SampleResult[tuple[list[int], list[int]]],
    ) -> BinarySampleSet:
        """Post-select the shots and decode the survivors.

        Failed shots are dropped, not reweighted; use
        :meth:`success_probability` to see how many survived. An empty sample
        set is returned when nothing passes, which is a legitimate outcome at
        large ``beta`` with too few shots.

        Args:
            samples (SampleResult[tuple[list[int], list[int]]]): Raw results as
                ``(signal_bits, system_bits)`` pairs.

        Returns:
            BinarySampleSet: Decoded post-selected samples in the model's
                original vartype.

        Raises:
            ValueError: If a result value is not a two-element tuple.
        """
        kept, kept_shots = self._postselect(samples)
        return super().decode_to_binary_sampleset(
            SampleResult(results=kept, shots=kept_shots)
        )

    def decode(  # type: ignore[override]
        self,
        samples: SampleResult[tuple[list[int], list[int]]],
    ) -> BinarySampleSet | ommx.v1.SampleSet:
        """Decode post-selected measurement results.

        Behaves like :meth:`MathematicalProblemConverter.decode`, with
        post-selection applied first through
        :meth:`decode_to_binary_sampleset`.

        Args:
            samples (SampleResult[tuple[list[int], list[int]]]): Raw results as
                ``(signal_bits, system_bits)`` pairs.

        Returns:
            BinarySampleSet | ommx.v1.SampleSet: An ``ommx.v1.SampleSet`` when
                this converter was built from an ``ommx.v1.Instance``,
                otherwise a :class:`BinarySampleSet`.

        Raises:
            ValueError: If a result value is not a two-element tuple.
        """
        return super().decode(samples)  # type: ignore[arg-type]
