import dataclasses
import typing as typ


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
