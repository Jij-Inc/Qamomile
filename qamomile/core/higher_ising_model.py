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
        if len(self.index_map) == 0:
            # Iterate over the keys of its coefficients
            # and set the position to the key of the index map and the kye to the value of the index map.
            unique_indices = {idx for key in self.coefficients.keys() for idx in key}
            for key in unique_indices.keys():
                self.index_map[key] = key

    @property
    def num_bits(self) -> int:
        """Returns the number of variables in the model.

        Finds the maximum index across all terms in the model and returns max_index + 1.
        For example, if the model has terms with indices (0, 1, 5), num_bits will be 6.
        """
        if not self.coefficients:
            return 0

        max_index = -1
        for indices in self.coefficients.keys():
            if indices:  # Skip empty tuples (constant terms)
                max_index = max(max_index, max(indices))

        return max_index + 1

    def calc_energy(self, state: list[int]) -> float:
        """Calculate the energy of the state.

        Examples:
            >>> higher_ising = HigherIsingModel({(0, 1): 2.0, (0,): 4.0, (1,): 5.0}, 6.0)
            >>> higher_ising.calc_energy([1, -1])
            3.0

        """
        # Initialise the energy with the constant term.
        energy = self.constant

        # Iterate over the keys of its coefficients and calculate the energy with the given state.
        for indices, coeff in self.coefficients.items():
            term = coeff
            for idx in indices:
                term *= state[idx]
            energy += term

        return energy

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
        if not self.coefficients:
            return

        # Calculate the sum of squares of the coefficients.
        linear_sum_sq = 0.0
        nonlinear_sum_sq = 0.0
        linear_count = 0
        nonlinear_count = 0

        for indices, coeff in self.coefficients.items():
            if len(indices) == 1:
                linear_sum_sq += coeff**2
                linear_count += 1
            else:  # len(indices) >= 2
                nonlinear_sum_sq += coeff**2
                nonlinear_count += 1

        # Calculate the root mean square.
        rms_components = 0.0
        if linear_count > 0:
            rms_components += linear_sum_sq / linear_count
        if nonlinear_count > 0:
            rms_components += nonlinear_sum_sq / nonlinear_count
        rms = rms_components**0.5

        # Normalize by the root mean square.
        self.normalize_by_factor(factor=rms)

    def normalize_by_factor(self, factor: float) -> None:
        r"""Normalize coefficients by a given factor.

        We normalize the Ising Hamiltonian as

        .. math::
            \tilde{H} = \frac{1}{factor}\sum_{ij}J_{ij}Z_iZ_j + \frac{1}{factor}\sum_ih_iZ_i + \frac{1}{factor}C

        Args:
            factor (float): The normalization factor.
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
            coefficients = {k: v for k, v in coefficients.items() if abs(v) != 0.0}

        return HigherIsingModel(coefficients=coefficients, constant=ising_constant)
