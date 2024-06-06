from __future__ import annotations
from dataclasses import dataclass
import jijmodeling_transpiler.core as jtc
from qamomile.core import Hamiltonian


@dataclass
class CompiledHamiltonian:
    substituted_hamiltonian: jtc.SubstitutedQuantumExpression
    hamiltonian: Hamiltonian
    data: jtc.InstanceData
    var_map: jtc.VariableMap
    qubit_index_map: dict[tuple[int, ...], int]
    deci_var_shape: dict[str, tuple[int | None, ...]]
