import qiskit as qk
import qiskit.quantum_info as qk_ope
import numpy as np
import jijmodeling as jm
import jijmodeling_transpiler as jmt
from jijmodeling_transpiler_quantum.qiskit.qrao.qrao_space_efficient import (
    numbering_space_efficient_encode,
    qrac_space_efficient_encode_ising,
)
from jijmodeling_transpiler_quantum.core.ising_qubo import IsingModel
from jijmodeling_transpiler_quantum.qiskit.qrao.qrao31 import Pauli


def full_connect_3node_ising_model():
    linear = {0: 1, 1: 1, 2: 1}
    quad = {(0, 1): 1, (1, 2): 1, (0, 2): 1}
    return IsingModel(linear=linear, quad=quad, constant=0)


def test_numbering_space_efficient_encode():
    ising = full_connect_3node_ising_model()
    encode = numbering_space_efficient_encode(ising)
    answer = {0: (0, Pauli.X), 1: (0, Pauli.Y), 2: (1, Pauli.X)}
    assert encode == answer


def test_qrac_space_efficient_encode_ising():
    ising = full_connect_3node_ising_model()
    op, offset, _ = qrac_space_efficient_encode_ising(ising)
    answer_op = qk_ope.SparsePauliOp.from_list(
        [
            ("IX", np.sqrt(3)),
            ("IY", np.sqrt(3)),
            ("XI", np.sqrt(3)),
            ("IZ", np.sqrt(3)),
            ("XY", 3),
            ("XX", 3),
        ]
    )
    assert offset == 0
    assert op == answer_op
