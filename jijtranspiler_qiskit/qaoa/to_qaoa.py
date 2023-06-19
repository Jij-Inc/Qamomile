import qiskit as qk

import jijmodeling.transpiler as jmt
from jijtranspiler_qiskit.ising_qubo.ising_qubo import qubo_to_ising
from .ising_hamiltonian import to_ising_operator_from_qubo


class QiskitQAOAAnsatzBuilder:
    def __init__(self, pubo_builder: jmt.core.pubo.PuboBuilder, var_num: int):
        self.pubo_builder = pubo_builder
        self.var_num = var_num
    
    def get_qaoa_ansatz(
        self,
        p: int,
        multipliers=None,
        detail_parameters=None
    ):
        qubo, constant = self.pubo_builder.get_qubo_dict(multipliers=multipliers, detail_parameters=detail_parameters)
        ising_operator, ising_const = to_ising_operator_from_qubo(qubo, self.var_num)
        qaoa_ansatz = qk.circuit.library.QAOAAnsatz(ising_operator, reps=p)
        return qaoa_ansatz, ising_operator, ising_const + constant


def transpile_to_qaoa_ansatz(
    compiled_instance: jmt.core.CompiledInstance,
    normalize: bool = True,
    relax_method = jmt.core.pubo.RelaxationMethod.AugmentedLagrangian
) -> QiskitQAOAAnsatzBuilder:
    pubo_builder = jmt.core.pubo.transpile_to_pubo(compiled_instance, normalize, relax_method=relax_method)
    var_num = compiled_instance.var_map.var_num
    return QiskitQAOAAnsatzBuilder(pubo_builder, var_num)

