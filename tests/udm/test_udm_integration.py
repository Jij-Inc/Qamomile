import pytest
import numpy as np
import jijmodeling as jm
from collections import OrderedDict
import ommx.v1
import qamomile.core.bitssample as qm_bs
from qamomile.core.converters.qaoa import QAOAConverter
from qamomile.udm import Ising_UnitDiskGraph
from qamomile.core.ising_qubo import IsingModel
from qamomile.udm.transpiler import UDMTranspiler


def Create_QUBO_problem():

    V = jm.Placeholder("V")
    E = jm.Placeholder("E", ndim=2)
    U = jm.Placeholder("U", ndim=2)

    n = jm.BinaryVar("n", shape=(V,))
    e = jm.Element("e", belong_to=E)

    problem = jm.Problem("QUBO_Hamiltonian")

    quadratic_term = jm.sum(e, U[e[0], e[1]] * n[e[0]] * n[e[1]])
    problem += quadratic_term
    return problem


def Create_instance():

    quad = {
        (0, 0): -1.2,
        (1, 1): -3.2,
        (2, 2): -1.2,
        (0, 1): 4.0,
        (0, 2): -2.0,
        (1, 2): 3.2,
    }

    V_val = 3
    E_val = np.array(list(quad.keys()), dtype=int)
    U_val = np.zeros((V_val, V_val))
    for (i, j), Jij in quad.items():
        U_val[i, j] = U_val[j, i] = Jij

    instance = {
        "V": V_val,
        "E": E_val,
        "U": U_val,
    }

    return instance


def test_udm_integration():
    """Test the UDM integration by creating and solving a simple Ising problem."""

    problem = Create_QUBO_problem()
    instance = Create_instance()
    compiled_instance = jm.Interpreter(instance).eval_problem(problem)
    udm_converter = QAOAConverter(compiled_instance)

    ising_model = udm_converter.ising_encode()
    assert isinstance(ising_model, IsingModel)

    udg = Ising_UnitDiskGraph(ising_model)
    assert len(udg.pins) == instance["V"]
    assert len(udg.nodes) == 26


def test_convert_result():

    problem = Create_QUBO_problem()
    instance = Create_instance()
    compiled_instance = jm.Interpreter(instance).eval_problem(problem)
    udm_converter = QAOAConverter(compiled_instance)
    ising_model = udm_converter.ising_encode()
    udg = Ising_UnitDiskGraph(ising_model)

    transpiler = UDMTranspiler(udg, instance["V"])
    mock_result = OrderedDict(
        [
            ("10010011111101011001110101", 501),
            ("11010101111110011001110101", 499),
        ]
    )
    result = transpiler.convert_result(mock_result)

    assert isinstance(result, qm_bs.BitsSampleSet)
    assert len(result.bitarrays) == 2
    assert result.total_samples() == 1000

    sampleset = udm_converter.decode(transpiler, mock_result)
    assert isinstance(sampleset, ommx.v1.SampleSet)
