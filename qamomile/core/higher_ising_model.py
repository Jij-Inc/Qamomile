from __future__ import annotations
import itertools
import dataclasses


@dataclasses.dataclass
class HigherIsingModel:
    """A model for accepting HUBO problems."""

    coefficients: dict[tuple[int, ...], float]
    constant: float
    index_map: dict[int, int] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialise the index map."""
        # Iterate over the keys of its coefficients
        # and set the position to the key of the index map and the kye to the value of the index map.
        pass

    def num_bits(self) -> int:
        """Returns the number of variables in the model."""
        # Returns the number of unique keys.
        pass

    def calc_energy(self, state: list[int]) -> float:
        """Calculate the energy of the state.

        Examples:
            >>> higher_ising = HigherIsingModel({(0, 1): 2.0}, {0: 4.0, 1: 5.0}, 6.0)
            >>> higher_ising.calc_energy([1, -1])
            3.0

        """
        # Initialise the energy with the constant term.
        # Iterate over the keys of its coefficients and calculate the energy with the given state.
        pass

    def ising2hubo_index(self, ising_index: int) -> int:
        """Return the corresponding hubo index for the given ising index.

        Args:
            ising_index (int): the index in the Ising model

        Returns:
            int: the hubo index
        """
        return self.index_map[ising_index]

    def normalize_by_abs_max(self) -> None:
        r"""Normalize coefficients by the absolute maximum value.

        The coefficients for normalized is defined as:

        .. math::
            W = \max(|J_{ij}|, |h_i|)

        We normalize the Ising Hamiltonian as

        .. math::
            \tilde{H} = \frac{1}{W}\sum_{ij}J_{ij}Z_iZ_j + \frac{1}{W}\sum_ih_iZ_i + \frac{1}{W}C

        """
        # Skip normalization if there are no coefficients.
        # Get the maximum absolute value of the coefficients.
        # Normalise each coefficient and the constant term by dividing by the maximum absolute value.
        pass

    def normalize_by_rms(self):
        r"""Normalize coefficients by the root mean square.

        The coefficients for normalized is defined as:

        .. math::
            W = \sqrt{ \frac{1}{E_2}\sum(w_ij^2) + \frac{1}{E_1}\sum(w_i^2) }

        where w_ij are quadratic coefficients and w_i are linear coefficients.
        E_2 and E_1 are the number of quadratic and linear terms respectively.
        We normalize the Ising Hamiltonian as

        .. math::
            \tilde{H} = \frac{1}{W}\sum_{ij}J_{ij}Z_iZ_j + \frac{1}{W}\sum_ih_iZ_i + \frac{1}{W}C
        This method is proposed in :cite:`Sureshbabu2024parametersettingin`

        .. bibliography::
            :filter: docname in docnames

        """
        # Skip normalization if there are no coefficients.
        # Calculate the sum of squares of the coefficients.
        # Calculate the root mean square.
        # Normalise each coefficient and the constant term by dividing by the root mean square.
        pass

    def normalize_by_factor(self, factor: float) -> None:
        r"""Normalize coefficients by a given factor.

        We normalize the Ising Hamiltonian as

        .. math::
            \tilde{H} = \frac{1}{factor}\sum_{ij}J_{ij}Z_iZ_j + \frac{1}{factor}\sum_ih_iZ_i + \frac{1}{factor}C

        Args:
            factor (float): The normalization factor.
        """
        # Skip normalization if there are no coefficients.
        # Normalise each coefficient and the constant term by dividing by the given factor.
        pass

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

        """
        # Initialise the coefficients and constant term for the Higher Ising model.
        # Transform each term in the HUBO to the corresponding term in the Higher Ising model.
        pass
