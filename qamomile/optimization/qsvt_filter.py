r"""Build QSVT eigenstate-filtering circuits for combinatorial problems.

The converter compiles a QUBO/HUBO problem into the sampling circuit of the
Lin & Tong (2020, arXiv:2002.12508) ground-energy algorithm: the Ising cost
Hamiltonian is block encoded, shifted by :math:`-\mu I`, filtered with a QSVT
approximation of the sign function, and probed with one Hadamard-test qubit.
The fraction of shots with all ancillas zero estimates
:math:`\lVert P_{<\mu}\lvert\varphi_0\rangle\rVert^2`, which is the predicate a
classical binary search over :math:`\mu` needs. The search itself stays outside
the library — the converter only emits quantum programs.
"""

from __future__ import annotations

import math

import numpy as np

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile._utils import is_close_zero
from qamomile.circuit.algorithm.qsvt_filter import eigenstate_filter_probe
from qamomile.circuit.stdlib.block_encoding import LCUBlockEncoding
from qamomile.circuit.transpiler.executable import ExecutableProgram
from qamomile.circuit.transpiler.job import SampleResult
from qamomile.circuit.transpiler.transpiler import Transpiler

from .binary_model import BinarySampleSet
from .converter import MathematicalProblemConverter

DEFAULT_POLYNOMIAL_DEGREE = 61
"""Degree of the sign-function approximation used when none is given."""

DEFAULT_TRANSITION_WIDTH = 20
"""``pyqsp`` ``delta`` parameter controlling the sharpness of the step."""

DEFAULT_POLYNOMIAL_SCALE = -1.10
"""Rescaling applied to the sign polynomial before phase extraction.

The sign fixes the filter's orientation: the negative default keeps the
eigenstates *below* the threshold. The magnitude sharpens the step.
"""


class QSVTFilterConverter(MathematicalProblemConverter):
    r"""Converter for QSVT eigenstate filtering (Lin & Tong ground-energy search).

    The problem's Ising Hamiltonian :math:`H` is block encoded once with
    :func:`~qamomile.circuit.ising_z_block_encoding`. Each threshold
    :math:`\mu` then composes a second LCU for :math:`H - \mu I` and compiles
    its own probe circuit, because the LCU amplitudes — and hence :math:`\mu` —
    are baked in at compile time.

    A typical binary-search step therefore looks like: :meth:`transpile` for the
    current :math:`\mu`, sample the program, read :meth:`success_probability`,
    and — once the search has converged — :meth:`decode` the same result to get
    the filtered states themselves.

    Attributes:
        encoding (LCUBlockEncoding): Block encoding of the unshifted cost
            Hamiltonian, built during construction.
        normalization (float): Its subnormalization :math:`\alpha`, which
            bounds the spectrum: every eigenvalue lies in
            :math:`[-\alpha, \alpha]`, so it sets the binary-search grid.
        num_ancilla_bits (int): Width of the ancilla block (projector plus
            signal) of the most recent :meth:`transpile` call. Only present
            after the first call.

    Example:
        >>> from qamomile.optimization.binary_model import BinaryModel
        >>> from qamomile.optimization.qsvt_filter import QSVTFilterConverter
        >>> model = BinaryModel.from_higher_ising({(0,): 1.0, (0, 1): -1.0})
        >>> converter = QSVTFilterConverter(model)
        >>> converter.normalization
        2.0
    """

    def __post_init__(self) -> None:
        """Block encode the spin model's Hamiltonian once, up front."""
        coefficients: dict[tuple[int, ...], float] = dict(self.spin_model.coefficients)
        if not is_close_zero(self.spin_model.constant):
            coefficients[()] = self.spin_model.constant
        self.encoding = qmc.ising_z_block_encoding(
            coefficients, self.spin_model.num_bits
        )
        self.normalization = self.encoding.normalization
        self._phase_cache: dict[tuple[int, float, float], list[float]] = {}

    def get_cost_hamiltonian(self) -> qm_o.Hamiltonian:
        """Report that this converter exposes no cost Hamiltonian.

        The Ising cost Hamiltonian is block encoded internally (see
        :meth:`__post_init__`) and consumed by the QSVT sign filter; it is
        never handed back as a Pauli observable. Following the base-class
        convention for converters that do not use a cost Hamiltonian, this
        always raises.

        Returns:
            qm_o.Hamiltonian: Never returns.

        Raises:
            NotImplementedError: Always; the converter does not expose a cost
                Hamiltonian.
        """
        raise NotImplementedError(
            "QSVTFilterConverter does not expose a cost Hamiltonian; the Ising "
            "operator is block encoded internally for the QSVT sign filter."
        )

    def _shifted_encoding(self, mu: float) -> LCUBlockEncoding:
        r"""Compose a block encoding of :math:`H - \mu I`.

        Written as an LCU *of block encodings*, so the Hamiltonian is decomposed
        once and every threshold reuses it:
        :math:`H - \mu I = 1 \cdot H + (-\mu) \cdot I`, with normalization
        :math:`\alpha' = \alpha + \lvert\mu\rvert`.

        Args:
            mu (float): Energy threshold to subtract, in the units of the
                problem's Ising Hamiltonian.

        Returns:
            LCUBlockEncoding: Descriptor of the shifted operator, with its own
                flat signal register.
        """
        identity = qmc.identity_block_encoding(self.encoding.num_system_qubits)
        return qmc.lcu_block_encoding(
            [
                qmc.LCUBlockEncodingTerm(1.0, self.encoding),
                qmc.LCUBlockEncodingTerm(-float(mu), identity),
            ]
        )

    def _qsp_phases(
        self,
        degree: int = DEFAULT_POLYNOMIAL_DEGREE,
        delta: float = DEFAULT_TRANSITION_WIDTH,
        scale: float = DEFAULT_POLYNOMIAL_SCALE,
    ) -> list[float]:
        r"""Compute reflection-convention phases approximating the sign function.

        ``pyqsp`` builds an odd polynomial approximation of
        :math:`\mathrm{sign}(x)` and returns its phases in the Wx (signal
        rotation) convention. They are converted to the projector-rotation
        convention :func:`~qamomile.circuit.qsvt` consumes,
        :math:`R(\phi) = e^{i\phi(2\Pi - I)}`: the raw ``sym_qsp`` output is
        first turned into a genuine Wx sequence by adding :math:`\pi/4` to the
        two end phases, then mapped by adding :math:`(2d - 1)\pi/4` to the first
        phase, subtracting :math:`\pi/2` from every interior phase and
        :math:`\pi/4` from the last, and wrapping into :math:`(-\pi, \pi]`.
        Results are cached per ``(degree, delta, scale)``; they do not depend on
        :math:`\mu`.

        Args:
            degree (int): Odd degree of the sign approximation. Higher degree
                sharpens the step and lengthens the circuit linearly.
            delta (float): ``pyqsp`` transition-width parameter: the larger it
                is, the steeper the step near the origin.
            scale (float): Factor applied to the polynomial coefficients before
                phase extraction. Its sign fixes the filter's orientation — the
                negative default keeps the eigenstates *below* the threshold —
                and its magnitude sharpens the step, ``pyqsp``'s
                ``ensure_bounded`` leaving headroom under the QSP bound
                :math:`\lvert p \rvert \le 1`. Magnitudes much above ``1.2``
                push the polynomial outside that bound and the phases become
                meaningless.

        Returns:
            list[float]: ``degree + 1`` phases in the reflection convention,
                ready to bind as ``phi``.

        Raises:
            ImportError: If ``pyqsp`` is not installed.
            ValueError: If ``degree`` is not a positive odd integer.
        """
        if not isinstance(degree, int) or degree < 1 or degree % 2 == 0:
            raise ValueError(f"degree must be a positive odd int; got {degree!r}.")

        key = (degree, float(delta), float(scale))
        cached = self._phase_cache.get(key)
        if cached is not None:
            return list(cached)

        try:
            from pyqsp.angle_sequence import QuantumSignalProcessingPhases
            from pyqsp.poly import PolySign
        except ImportError as error:  # pragma: no cover - depends on install
            raise ImportError(
                "QSVTFilterConverter._qsp_phases requires pyqsp. Install it with "
                "`pip install 'qamomile[qsvt]'`, or pass precomputed phases via "
                "the `phi` argument of transpile()."
            ) from error

        generated, _ = PolySign().generate(
            degree=degree,
            delta=delta,
            ensure_bounded=True,
            return_scale=True,
            chebyshev_basis=True,
        )
        coefficients = [scale * float(c) for c in np.asarray(generated).ravel()]
        wx_phases, _, _ = QuantumSignalProcessingPhases(
            coefficients, method="sym_qsp", chebyshev_basis=True
        )

        # sym_qsp raw output -> genuine Wx sequence.
        wx = np.asarray(wx_phases, dtype=float)
        wx = wx.copy()
        wx[0] += math.pi / 4
        wx[-1] += math.pi / 4

        # Genuine Wx -> projector-rotation (reflection) convention.
        degree_out = len(wx) - 1
        reflection = np.empty_like(wx)
        reflection[0] = wx[0] + (2 * degree_out - 1) * math.pi / 4
        reflection[1:-1] = wx[1:-1] - math.pi / 2
        reflection[-1] = wx[-1] - math.pi / 4
        reflection = (reflection + math.pi) % (2 * math.pi) - math.pi

        phases = [float(phase) for phase in reflection]
        self._phase_cache[key] = list(phases)
        return phases

    def transpile(
        self,
        transpiler: Transpiler,
        *,
        mu: float,
        degree: int = DEFAULT_POLYNOMIAL_DEGREE,
        delta: float = DEFAULT_TRANSITION_WIDTH,
        scale: float = DEFAULT_POLYNOMIAL_SCALE,
        phi: list[float] | None = None,
    ) -> ExecutableProgram:
        r"""Transpile the probe circuit for one energy threshold.

        The emitted circuit prepares the uniform superposition on the system
        register, applies :math:`P_{<\mu}` through the QSVT sign filter, and
        measures the projector, signal, and system registers in that order.

        Both ``mu`` and ``phi`` are compile-time data: the threshold sets the
        LCU amplitudes and the phase count sets the circuit structure, so each
        threshold needs its own ``transpile`` call.

        Args:
            transpiler (Transpiler): Backend transpiler to use.
            mu (float): Energy threshold. The filter keeps eigenstates of the
                cost Hamiltonian with energy below ``mu``.
            degree (int): Degree of the sign approximation, passed to
                :meth:`_qsp_phases`. Ignored when ``phi`` is given.
            delta (float): Transition-width parameter, passed to
                :meth:`_qsp_phases`. Ignored when ``phi`` is given.
            scale (float): Polynomial rescaling, passed to :meth:`_qsp_phases`.
                Ignored when ``phi`` is given.
            phi (list[float] | None): Precomputed reflection-convention phases.
                Defaults to None, meaning they are computed with ``pyqsp``.
                Must have even length.

        Returns:
            ExecutableProgram: Compiled probe circuit for this threshold.

        Raises:
            ValueError: If ``phi`` has odd or fewer than two entries.
            ImportError: If ``phi`` is omitted and ``pyqsp`` is not installed.
        """
        phases = (
            list(phi) if phi is not None else self._qsp_phases(degree, delta, scale)
        )
        if len(phases) < 2 or len(phases) % 2 != 0:
            raise ValueError(
                "phi must hold an even number of at least two phases "
                f"(odd polynomial degree); got {len(phases)}."
            )

        encoding = self._shifted_encoding(mu)
        self.num_ancilla_bits = 1 + encoding.num_signal_qubits
        return transpiler.transpile(
            eigenstate_filter_probe(encoding),
            bindings={"phi": phases},
        )

    @staticmethod
    def _postselect(
        samples: SampleResult[tuple[list[int], list[int], list[int]]],
    ) -> tuple[list[tuple[list[int], int]], int]:
        """Keep the shots whose projector and signal registers measured zero.

        Args:
            samples (SampleResult[tuple[list[int], list[int], list[int]]]): Raw
                probe results, each value holding the projector, signal, and
                system bits in that order.

        Returns:
            tuple[list[tuple[list[int], int]], int]: The surviving
                ``(system_bits, count)`` pairs and the number of shots they
                account for.

        Raises:
            ValueError: If a result value is not a three-register tuple.
        """
        kept: list[tuple[list[int], int]] = []
        kept_shots = 0
        for value, count in samples.results:
            if not isinstance(value, tuple) or len(value) != 3:
                raise ValueError(
                    "Expected probe results as (projector, signal, system) "
                    f"tuples; got {value!r}."
                )
            projector_bits, signal_bits, system_bits = value
            if any(projector_bits) or any(signal_bits):
                continue
            kept.append((list(system_bits), count))
            kept_shots += count
        return kept, kept_shots

    def success_probability(
        self,
        samples: SampleResult[tuple[list[int], list[int], list[int]]],
    ) -> float:
        r"""Estimate :math:`\lVert P_{<\mu}\lvert\varphi_0\rangle\rVert^2`.

        This is the quantity the Lin & Tong binary search thresholds: it is
        bounded away from zero exactly when an eigenstate below :math:`\mu`
        carries weight in the uniform superposition.

        Args:
            samples (SampleResult[tuple[list[int], list[int], list[int]]]): Raw
                probe results for one threshold.

        Returns:
            float: Fraction of shots with all ancilla bits zero, or ``0.0``
                when no shots were taken.

        Raises:
            ValueError: If a result value is not a three-register tuple.
        """
        _, kept_shots = self._postselect(samples)
        if samples.shots <= 0:
            return 0.0
        return kept_shots / samples.shots

    def decode_to_binary_sampleset(
        self,
        samples: SampleResult[tuple[list[int], list[int], list[int]]],  # type: ignore[override]
    ) -> BinarySampleSet:
        """Decode the post-selected system measurements into problem samples.

        Shots whose projector or signal bits are non-zero missed the filtered
        block and carry no information about the low-energy subspace, so they
        are dropped before the usual SPIN/BINARY decoding runs. Use
        :meth:`success_probability` on the same result to recover how many were
        discarded.

        Args:
            samples (SampleResult[tuple[list[int], list[int], list[int]]]): Raw
                probe results, each value holding the projector, signal, and
                system bits in that order.

        Returns:
            BinarySampleSet: Post-selected samples in the converter's original
                vartype, with energies from the problem's model.

        Raises:
            ValueError: If a result value is not a three-register tuple.
        """
        kept, kept_shots = self._postselect(samples)
        return super().decode_to_binary_sampleset(
            SampleResult(results=kept, shots=kept_shots)
        )
