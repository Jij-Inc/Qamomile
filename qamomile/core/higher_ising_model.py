from __future__ import annotations
import itertools
import dataclasses

import numpy as np


@dataclasses.dataclass
class HigherIsingModel:
    """A model for accepting HUBO problems."""

    coefficients: dict[tuple[int, ...], float]
    constant: float
    index_map: dict[int, int] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialise the index map."""
        if len(self.index_map) == 0:
            # Iterate over the keys of its coefficients
            # and set the position to the key of the index map and the key to the value of the index map.
            unique_indices = {idx for key in self.coefficients.keys() for idx in key}
            for index in unique_indices:
                self.index_map[index] = index

    @property
    def num_bits(self) -> int:
        """Returns the number of variables in the model,
        which is the number of bits since those variables are supposedly binary.

        Returns:
            int: Number of variables in the model.
        """
        unique_indices = set().union(*self.coefficients)
        return len(unique_indices)

    def calc_energy(self, state: list[int]) -> float:
        """Calculate the energy of the state.

        Args:
            state (list[int]): A list of spin values (+1 or -1) representing the state of each variable.

        Raises:
            ValueError: If any element in state is not close to +1 or -1.

        Returns:
            float: The calculated energy of the given state.

        Examples:
            >>> higher_ising = HigherIsingModel({(0, 1): 2.0, (0,): 4.0, (1,): 5.0}, 6.0)
            >>> higher_ising.calc_energy([1, -1])
            3.0

        """
        # Validate the given state.
        if not np.allclose(np.abs(state), 1.0):
            raise ValueError(
                "All elements in state must be close to +1 or -1 since it is a spin."
            )

        # Initialise the energy with the constant term.
        energy = self.constant

        # Iterate over the keys of its coefficients and calculate the energy with the given state.
        for indices, coeff in self.coefficients.items():
            term = coeff
            for idx in indices:
                term *= state[idx]
            energy += term

        return energy

    def normalize_by_abs_max(self) -> None:
        r"""Normalize coefficients by the absolute maximum value.

        The coefficients for normalized is defined as:

        .. math::
            W = \max(|w_{i_0, \dots, i_k}|)

        where w are coefficients and their subscriptions imply a term to be applied.
        We normalize the Ising Hamiltonian as

        .. math::
            \tilde{H} = \frac{1}{W} \left( C + \sum_i w_i Z_i + \cdots + \sum_{i_0, \dots, i_k} w_{i_0, \dots, i_k} Z_{i_0}\dots Z_{i_k} \right)

        """
        # Skip normalization if there are no coefficients.
        if not self.coefficients:
            return

        # Get the maximum absolute value of the coefficients.
        max_abs = max(abs(v) for v in self.coefficients.values())
        # Normalize by the maximum absolute value.
        self.normalize_by_factor(factor=max_abs)

    def normalize_by_rms(self):
        r"""Normalize coefficients by the root mean square.

        The coefficients for normalized is defined as:

        .. math::
            W = \sqrt{\frac{1}{\lvert E_k \rvert} \sum_{\{u_1, \dots, u_k\}} (w_{u_1,...,u_k}^{(k)})^2 + \cdots + \frac{1}{\lvert E_1 \rvert} \sum_u (w_u^{(1)})^2}

        where w are coefficients and their subscriptions imply a term to be applied.
        E_i are the number of i-th order terms.
        We normalize the Ising Hamiltonian as

        .. math::
            \tilde{H} = \frac{1}{W} \left( C + \sum_i w_i Z_i + \cdots + \sum_{i_0, \dots, i_k} w_{i_0, \dots, i_k} Z_{i_0}\dots Z_{i_k} \right)
        This method is proposed in :cite:`Sureshbabu2024parametersettingin`

        .. bibliography::
            :filter: docname in docnames

        """
        # Skip normalization if there are no coefficients.
        if not self.coefficients:
            return

        # Get square sum and count for each kind of term.
        counts = {}  # key: term order, value: (sum of squares, count)
        for indices, coeff in self.coefficients.items():
            order = len(indices)
            if order not in counts:
                counts[order] = [0.0, 0]

            # Add the square of the coefficient to the sum of squares.
            counts[order][0] += coeff**2
            # Increment the count of terms.
            counts[order][1] += 1

        # Compute the mean square for each kind of term.
        rms_components = 0.0
        for order, (sum_squares, count) in counts.items():
            if count > 0:  # This check is redundant but safe.
                mean_square = sum_squares / count
                rms_components += mean_square

        rms = np.sqrt(rms_components)

        # Normalize by the root mean square.
        self.normalize_by_factor(factor=rms)

    def normalize_by_factor(self, factor: float) -> None:
        r"""Normalize coefficients by a given factor.

        We normalize the Ising Hamiltonian as

        .. math::
            \tilde{H} = \frac{1}{W} \left( C + \sum_i w_i Z_i + \cdots + \sum_{i_0, \dots, i_k} w_{i_0, \dots, i_k} Z_{i_0}\dots Z_{i_k} \right)

        where W is the given normalization factor.

        Args:
            factor (float): the normalization factor
        """
        # Skip normalization if there are no coefficients.
        if not self.coefficients:
            return
        # Skip normalization if the factor is zero to avoid division by zero.
        if factor == 0:
            return

        # Normalise each coefficient and the constant term by dividing by the given factor.
        for key in self.coefficients:
            self.coefficients[key] /= factor
        self.constant /= factor

    @classmethod
    def from_hubo(
        cls, hubo: dict[tuple[int, ...], float], constant: float = 0.0, simplify=False
    ) -> HigherIsingModel:
        r"""Converts a Quadratic Unconstrained Binary Optimization (QUBO) problem to an equivalent Ising model.

        HUBO:
            .. math::
                \sum_{i} H_i x_i + \sum_{i, j} H_{i, j} x_i x_j + \sum_{i, j, k} H_{i, j, k} x_i x_j x_k,~\text{s.t.}~x_i \in \{0, 1\}

        Higher Ising model:
            .. math::
                \sum_{i} J_i z_i + \sum_{i, j} J_{i, j} z_i z_j + \sum_{i, j, k} J_{i, j, k} z_i z_j z_k, ~\text{s.t.}~z_i \in \{-1, 1\}

        Correspondence:
            .. math::
                x_i = \frac{1 - z_i}{2}
            where :math:`(x_i \in \{0, 1\})` and :math:`(z_i \in \{-1, 1\})`.

        This transformation is derived from the conventions used to describe the eigenstates and eigenvalues of the Pauli Z operator in quantum computing.
        Specifically, the eigenstates |0⟩ and |1⟩ of the Pauli Z operator correspond to the eigenvalues +1 and -1, respectively:

        .. math::
            Z|0\rangle = |0\rangle, \quad Z|1\rangle = -|1\rangle

        This relationship is leveraged to map the binary variables \(x_i\) in HUBO to the spin variables \(z_i\) in the Ising model.

        Args:
            hubo (dict[tuple[int, ...], float]): HUBO coefficients
            constant (float): Constant term in the HUBO
            simplify (bool): If True, remove coefficients close to zero (not yet implemented)

        Returns:
            HigherIsingModel: The equivalent Ising model

        """
        # Initialise the coefficients and constant term for the Higher Ising model.
        coefficients = {}
        ising_constant = constant  # Start with the input constant

        for indices, value in hubo.items():
            n = len(indices)
            base = value / (2**n)
            for r in range(n + 1):
                for subset in itertools.combinations(indices, r):
                    sign = (-1) ** r
                    if len(subset) == 0:  # Constant term
                        ising_constant += base * sign
                    else:
                        coefficient = tuple(sorted(subset))
                        coefficients[coefficient] = (
                            coefficients.get(coefficient, 0) + base * sign
                        )

        # Optionally simplify by removing near-zero coefficients
        if simplify:
            coefficients = {
                k: v for k, v in coefficients.items() if np.isclose(abs(v), 0.0)
            }

        return HigherIsingModel(coefficients=coefficients, constant=ising_constant)
