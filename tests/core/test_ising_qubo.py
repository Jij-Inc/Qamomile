from qamomile.core.ising_qubo import qubo_to_ising, IsingModel


def test_onehot_conversion():
    qubo: dict[tuple[int, int], float] = {(0, 1): 2, (0, 0): -1, (1, 1): -1}
    ising = qubo_to_ising(qubo)
    assert ising.constant == -0.5
    assert ising.linear == {0:0,1:0}
    assert ising.quad == {(0, 1): 0.5}

    ising = qubo_to_ising(qubo, simplify=True)
    assert ising.constant == -0.5
    assert ising.linear == {}
    assert ising.quad == {(0, 1): 0.5}



def test_num_bits():
    ising = IsingModel(
        {(0, 1): 2.0, (0, 2): 1.0},
        {2: 5.0, 3: 2.0, 4: 1.0, 5: 1.0, 6: 1.0},
        6.0,
    )
    assert ising.num_bits() == 7

    ising = IsingModel({}, {0: 1.0, 1: 1.0, 2: 5.0, 3: 2.0}, 6.0)
    assert ising.num_bits() == 4

    ising = IsingModel(
        {(0, 1): 2.0, (0, 2): 1.0},
        {},
        6.0,
    )
    assert ising.num_bits() == 3

    ising = IsingModel(
        {},
        {},
        6.0,
    )
    assert ising.num_bits() == 0

    ising = IsingModel(
        {},
        {0: 1.0},
        6.0,
    )
    assert ising.num_bits() == 1
