from jijmodeling_transpiler_quantum.core.ising_qubo import qubo_to_ising


def test_onehot_conversion():
    qubo: dict[tuple[int, int], float] = {(0, 1): 2, (0, 0): -1, (1, 1): -1}
    ising = qubo_to_ising(qubo)
    assert ising.constant == -0.5
    assert ising.linear == {}
    assert ising.quad == {(0, 1): 0.5}
