import dataclasses
import typing as typ
import copy
import numpy as np
from ..udm import map_qubo, solve_qubo, qubo_result_to_networkx, QUBOResult


@dataclasses.dataclass
class IsingModel:
    quad: dict[tuple[int, int], float]
    linear: dict[int, float]
    constant: float
    index_map: typ.Optional[dict[int, int]] = None

    def num_bits(self) -> int:
        num_bits = max(self.linear.keys(), default=-1)
        num_bits = max(
            num_bits, max((max(pair) for pair in self.quad.keys()), default=num_bits)
        )
        return num_bits + 1

    def calc_energy(self, state: list[int]) -> float:
        """Calculates the energy of the state.

        Examples:
            >>> ising = IsingModel({(0, 1): 2.0}, {0: 4.0, 1: 5.0}, 6.0)
            >>> ising.calc_energy([1, -1])
            3.0

        """
        energy = self.constant
        for (i, j), value in self.quad.items():
            energy += value * state[i] * state[j]
        for i, value in self.linear.items():
            energy += value * state[i]
        return energy

    def ising2qubo_index(self, index: int) -> int:
        if self.index_map is None:
            return index
        return self.index_map[index]

    def normalize_by_abs_max(self):
        r"""Normalize coefficients by the absolute maximum value.

        The coefficients for normalized is defined as:

        .. math::
            W = \max(|J_{ij}|, |h_i|)

        We normalize the Ising Hamiltonian as
        
        .. math::
            \tilde{H} = \frac{1}{W}\sum_{ij}J_{ij}Z_iZ_j + \frac{1}{W}\sum_ih_iZ_i + \frac{1}{W}C

        """

        if not self.linear and not self.quad:
            return  # 係数が存在しない場合は正規化しない

        max_coeff = max(
            max((abs(value) for value in self.linear.values()), default=0),
            max((abs(value) for value in self.quad.values()), default=0),
        )

        if max_coeff == 0:
            return  # すべての係数が0の場合は正規化しない

        self.constant /= max_coeff
        for key in self.linear:
            self.linear[key] /= max_coeff
        for key in self.quad:
            self.quad[key] /= max_coeff

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
        if not self.linear and not self.quad:
            return  # 係数が存在しない場合は正規化しない

        # numpyのarrayに変換して二乗和を計算
        quad_coeffs = np.array(list(self.quad.values()))
        linear_coeffs = np.array(list(self.linear.values()))

        E2 = len(self.quad)
        E1 = len(self.linear)

        # np.sum(quad_coeffs ** 2)はnp.dot(quad_coeffs, quad_coeffs)より効率的
        quad_variance = np.sum(quad_coeffs**2) / E2 if E2 > 0 else 0
        linear_variance = np.sum(linear_coeffs**2) / E1 if E1 > 0 else 0

        normalization_factor = np.sqrt(quad_variance + linear_variance)

        if normalization_factor == 0:
            return  # すべての係数が0の場合は正規化しない

        # 正規化
        self.constant /= normalization_factor
        for key in self.linear:
            self.linear[key] /= normalization_factor
        for key in self.quad:
            self.quad[key] /= normalization_factor
            
    def to_unit_disk_graph(self, normalize: bool = True) -> 'UnitDiskGraph':
        """Convert the Ising model to a unit disk graph representation.
        
        Args:
            normalize: Whether to normalize the coefficients before conversion
            
        Returns:
            UnitDiskGraph object representing the Ising model
        """
        if normalize:
            # Create a copy of the model to avoid modifying the original
            model = IsingModel(
                quad=copy.deepcopy(self.quad),
                linear=copy.deepcopy(self.linear),
                constant=self.constant,
                index_map=copy.deepcopy(self.index_map) if self.index_map else None
            )
            model.normalize_by_abs_max()
        else:
            model = self
            
        return UnitDiskGraph(model)


@dataclasses.dataclass
class UnitDiskGraph:
    """
    A representation of an Ising model as a unit disk graph suitable for neutral atom quantum computers.
    
    This class wraps the UDM module to map QUBO/Ising problems to unit disk graphs.
    """
    ising_model: IsingModel
    _qubo_result: typ.Optional[QUBOResult] = None
    _networkx_graph: typ.Optional[object] = None
    _delta: typ.Optional[float] = None
    
    def __post_init__(self):
        """Initialize the unit disk graph mapping after construction."""
        self._create_mapping()
    
    def _create_mapping(self):
        """Create the unit disk graph mapping from the Ising model."""
        # Convert the Ising model to J and h matrices/vectors
        n = self.ising_model.num_bits()
        
        # Initialize J matrix and h vector
        J = np.zeros((n, n))
        h = np.zeros(n)
        
        # Fill in the J matrix from quad terms
        for (i, j), value in self.ising_model.quad.items():
            J[i, j] = value
            J[j, i] = value  # Make symmetric
        
        # Fill in the h vector from linear terms
        for i, value in self.ising_model.linear.items():
            h[i] = value
        
        # Calculate delta parameter for weight scaling
        self._delta = 1.5 * max(np.max(np.abs(h)), np.max(np.abs(J)))
        
        # Map the QUBO problem to a unit disk graph
        self._qubo_result = map_qubo(J, h, self._delta)
        
        # Convert to a NetworkX graph for visualization and analysis
        self._networkx_graph = qubo_result_to_networkx(self._qubo_result)
    
    def solve(self, use_brute_force: bool = False, binary_variables: bool = False) -> dict:
        """
        Solve the Ising model using the unit disk graph mapping.
        
        Args:
            use_brute_force: Whether to use brute force enumeration for small problems
            binary_variables: Whether to use {0,1} variables (True) or {-1,1} variables (False)
            
        Returns:
            Dictionary containing solution information including:
            - original_config: Configuration for the original Ising variables
            - energy: Energy of the solution
            - solution_method: Method used to find the solution ("brute_force" or "mwis")
        """
        # Convert to J and h matrices/vectors
        n = self.ising_model.num_bits()
        
        J = np.zeros((n, n))
        h = np.zeros(n)
        
        for (i, j), value in self.ising_model.quad.items():
            J[i, j] = value
            J[j, i] = value
        
        for i, value in self.ising_model.linear.items():
            h[i] = value
        
        # Solve the QUBO problem
        result = solve_qubo(
            J, h, 
            binary_variables=binary_variables,
            use_brute_force=use_brute_force,
            max_brute_force_size=20  # Adjust this value based on performance needs
        )
        
        return result
    
    @property
    def qubo_result(self) -> QUBOResult:
        """Get the QUBOResult object from the UDM module."""
        return self._qubo_result
    
    @property
    def networkx_graph(self) -> object:
        """Get the NetworkX graph representation."""
        return self._networkx_graph
    
    @property
    def pins(self) -> list:
        """Get the list of pins (indices of nodes corresponding to original variables)."""
        if self._qubo_result:
            return self._qubo_result.pins
        return []
    
    @property
    def nodes(self) -> list:
        """Get the list of nodes in the unit disk graph."""
        if self._qubo_result and hasattr(self._qubo_result.grid_graph, 'nodes'):
            return self._qubo_result.grid_graph.nodes
        return []
    
    @property
    def delta(self) -> float:
        """Get the delta parameter used for scaling the weights."""
        return self._delta


def calc_qubo_energy(qubo: dict[tuple[int, int], float], state: list[int]) -> float:
    """Calculates the energy of the state.

    Examples:
        >>> calc_qubo_energy({(0, 0): 1.0, (0, 1): 2.0, (1, 1): 3.0}, [1, 1])
        6.0
    """
    energy = 0.0
    for (i, j), value in qubo.items():
        energy += value * state[i] * state[j]
    return energy


def qubo_to_ising(
    qubo: dict[tuple[int, int], float], constant: float = 0.0, simplify=False
) -> IsingModel:
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
        >>> ising = qubo_to_ising(qubo)
        >>> binary = [1, 0]
        >>> spin = [-1, 1]
        >>> qubo_energy = calc_qubo_energy(qubo, binary)
        >>> assert qubo_energy == ising.calc_energy(spin)

        >>> qubo = {(0, 1): 2, (0, 0): -1, (1, 1): -1}
        >>> ising = qubo_to_ising(qubo)
        >>> assert ising.constant == -0.5
        >>> assert ising.linear == {}
        >>> assert ising.quad == {(0, 1): 0.5}

    """
    ising_J: dict[tuple[int, int], float] = {}
    ising_h: dict[int, float] = {}
    constant = 0.0
    for (i, j), value in qubo.items():
        if i != j:
            ising_J[(i, j)] = value / 4.0 + ising_J.get((i, j), 0.0)
            constant += value / 4.0
        else:
            constant += value / 2.0
        ising_h[i] = -value / 4.0 + ising_h.get(i, 0.0)
        ising_h[j] = -value / 4.0 + ising_h.get(j, 0.0)

    if simplify:
        index_map = {}
        _J, _h = {}, {}
        for i, hi in ising_h.items():
            if hi != 0.0:
                if i not in index_map:
                    index_map[i] = len(index_map)
                _h[index_map[i]] = hi
        for (i, j), Jij in ising_J.items():
            if Jij != 0.0:
                if i not in index_map:
                    index_map[i] = len(index_map)
                if j not in index_map:
                    index_map[j] = len(index_map)
                _J[(index_map[i], index_map[j])] = Jij
        ising_J = _J
        ising_h = _h
        return IsingModel(ising_J, ising_h, constant, index_map)
    return IsingModel(ising_J, ising_h, constant)