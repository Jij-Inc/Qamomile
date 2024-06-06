from __future__ import annotations
from qamomile.core import IsingModel
from qamomile.core.compile import PauliOperator, PauliType, SubstitutedQuantumExpression

def compile_ising_model(ising_model: IsingModel) -> SubstitutedQuantumExpression:
    """Compile Ising model to a SubstitutedQuantumExpression.

    Args:
        ising_model (IsingModel): Ising model.

    Returns:
        SubstitutedQuantumExpression: Compiled Ising model.
    """
    coeff = {}
    for i, h in ising_model.linear.items():
        coeff[(PauliOperator(PauliType.Z, i),)] = h

    for (i, j), jij in ising_model.quad.items():
        coeff[(PauliOperator(PauliType.Z, i), PauliOperator(PauliType.Z, j))] = jij

    
    return SubstitutedQuantumExpression(coeff, ising_model.constant, 1)