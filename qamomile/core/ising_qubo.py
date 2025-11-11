from __future__ import annotations
import dataclasses

from qamomile.core.higher_ising_model import HigherIsingModel
from qamomile.core.converters.utils import is_close_zero


@dataclasses.dataclass
class IsingModel(HigherIsingModel):
    """Ising model as a special case of Higher Ising Model.

    This class represents a quadratic Ising model, which is a special case of
    the Higher Ising Model where all terms are at most quadratic.

    The model internally uses HigherIsingModel's coefficients representation,
    but provides quad and linear properties for backward compatibility.
    """

    def __init__(
        self,
        quad: dict[tuple[int, int], float],
        linear: dict[int, float],
        constant: float,
        index_map: dict[int, int] | None = None,
    ):
        """Initialize IsingModel.

        Args:
            quad: Quadratic coefficients J_ij
            linear: Linear coefficients h_i
            constant: Constant term
            index_map: Mapping from Ising indices to QUBO indices
        """
        # Convert quad and linear to coefficients format for parent class
        coefficients: dict[tuple[int, ...], float] = {}

        # Add quadratic terms
        for (i, j), value in quad.items():
            key = tuple(sorted([i, j]))
            coefficients[key] = value

        # Add linear terms
        for i, value in linear.items():
            coefficients[(i,)] = value

        # Initialize parent class
        super().__init__(coefficients=coefficients, constant=constant)

        # Override index_map if provided, otherwise use default behavior
        if index_map is not None:
            self.index_map = index_map
        else:
            self.index_map = {i: i for i in linear.keys()}
            for i, j in quad.keys():
                self.index_map[i] = i
                self.index_map[j] = j

    @property
    def quad(self) -> dict[tuple[int, int], float]:
        """Extract quadratic coefficients from parent's coefficients."""
        quad = {}
        for key, value in self.coefficients.items():
            if len(key) == 2:
                quad[key] = value
        return quad

    @property
    def linear(self) -> dict[int, float]:
        """Extract linear coefficients from parent's coefficients."""
        linear = {}
        for key, value in self.coefficients.items():
            if len(key) == 1:
                linear[key[0]] = value
        return linear

    @classmethod
    def from_qubo(
        cls, qubo: dict[tuple[int, int], float], constant: float = 0.0, simplify=False
    ) -> "IsingModel":
        r"""Converts a Quadratic Unconstrained Binary Optimization (QUBO) problem to an equivalent Ising model.

        QUBO:
            .. math::
                \sum_{ij} Q_{ij} x_i x_j,~\text{s.t.}~x_i \in \{0, 1\}

        Ising model:
            .. math::
                \sum_{ij} J_{ij} z_i z_j + \sum_i h_i z_i, ~\text{s.t.}~z_i \in \{-1, 1\}

        Correspondence:
            .. math::
                x_i = \frac{1 - z_i}{2}
            where :math:`(x_i \in \{0, 1\})` and :math:`(z_i \in \{-1, 1\})`.

        This transformation is derived from the conventions used to describe the eigenstates and eigenvalues of the Pauli Z operator in quantum computing. Specifically, the eigenstates |0⟩ and |1⟩ of the Pauli Z operator correspond to the eigenvalues +1 and -1, respectively:

        .. math::
            Z|0\rangle = |0\rangle, \quad Z|1\rangle = -|1\rangle

        This relationship is leveraged to map the binary variables \(x_i\) in QUBO to the spin variables \(z_i\) in the Ising model.

        Examples:
            >>> qubo = {(0, 0): 1.0, (0, 1): 2.0, (1, 1): 3.0}
            >>> ising = IsingModel.from_qubo(qubo)
            >>> binary = [1, 0]
            >>> spin = [-1, 1]
            >>> qubo_energy = calc_qubo_energy(qubo, binary)
            >>> assert qubo_energy == ising.calc_energy(spin)

            >>> qubo = {(0, 1): 2, (0, 0): -1, (1, 1): -1}
            >>> ising = IsingModel.from_qubo(qubo)
            >>> assert ising.constant == -0.5
            >>> assert ising.linear == {}
            >>> assert ising.quad == {(0, 1): 0.5}

        """
        ising_J: dict[tuple[int, int], float] = {}
        ising_h: dict[int, float] = {}
        constant = constant
        for (i, j), value in qubo.items():
            if i != j:
                ising_J[(i, j)] = value / 4.0 + ising_J.get((i, j), 0.0)
                constant += value / 4.0
            else:
                constant += value / 2.0
            ising_h[i] = -value / 4.0 + ising_h.get(i, 0.0)
            ising_h[j] = -value / 4.0 + ising_h.get(j, 0.0)

        if simplify:
            _J, _h = {}, {}
            for i, hi in ising_h.items():
                if not is_close_zero(hi):
                    _h[i] = hi
            for (i, j), Jij in ising_J.items():
                if not is_close_zero(Jij):
                    _J[(i, j)] = Jij
            ising_J = _J
            ising_h = _h
            return cls(ising_J, ising_h, constant)
        return cls(ising_J, ising_h, constant)
