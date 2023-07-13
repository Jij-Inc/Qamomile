import dataclasses


@dataclasses.dataclass
class IsingModel:
    quad: dict[tuple[int, int], float]
    linear: dict[int, float]
    constant: float

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


def qubo_to_ising(qubo: dict[tuple[int, int], float], simplify=True) -> IsingModel:
    """Converts a QUBO to an Ising model.

    QUBO: sum_{ij} Q_{ij} x_i x_j -> Ising: sum_{ij} J_{ij} z_i z_j + sum_i h_i z_i
    Correspondence - x_i = (1 - z_i) / 2, where x_i in {0, 1} and z_i in {-1, 1}

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
        ising_J = {ij: value for ij, value in ising_J.items() if value != 0.0}
        ising_h = {i: value for i, value in ising_h.items() if value != 0.0}
    return IsingModel(ising_J, ising_h, constant)
