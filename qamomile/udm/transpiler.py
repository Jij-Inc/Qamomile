import typing as typ
from qamomile.core.transpiler import QuantumSDKTranspiler
import jijmodeling as jm
import qamomile.core.circuit as qm_c
import qamomile.core.operator as qm_o
import qamomile.core.bitssample as qm_bs
from qamomile.core.transpiler import QuantumSDKTranspiler
from .mwis_solver import map_config_back, Ising_UnitDiskGraph
import collections


class UDMTranspiler(QuantumSDKTranspiler[collections.OrderedDict]):
    """
    Transpiler to convert raw bitstring-count results from a quantum SDK
    into jijmodeling Solutions by mapping through the unit-disk-graph.
    """

    def __init__(self, udg: Ising_UnitDiskGraph, num_vars: int):
        """
        Args:
            udg: An Ising_UnitDiskGraph instance containing the QUBO mapping.
            jm_vars: Ordered list of jijmodeling.BinaryVar corresponding
                     to the original problem variables.
        """
        self.udg = udg
        self.num_vars = num_vars

    def convert_result(self, result: collections.OrderedDict) -> qm_bs.BitsSampleSet:
        """
        Convert the SDK-specific bitstring->count result into a list of
        jijmodeling Solution objects.

        Steps:
        1. Parse each bitstring into a list of ints (0/1).
        2. Map through the QUBOResult using map_config_back to recover
           the original binary assignments.
        3. Build a jijmodeling Solution with the variable assignments
           and the measurement count as weight.
        """
        sorted_counts = {
            k: v
            for k, v in sorted(result.items(), key=lambda item: item[1], reverse=True)
        }
        orig_counts: dict[str, int] = {}
        for bitstr, count in sorted_counts.items():
            # 1. raw measurement configuration
            cfg = [int(b) for b in bitstr]
            # 2. map back to original QUBO variables
            orig_assign = map_config_back(self.udg.qubo_result, cfg, binary=True)
            key = "".join(str(bit) for bit in orig_assign)
            orig_counts[key] = orig_counts.get(key, 0) + count

            # 2. Convert binary-string keys to integer indices for BitsSampleSet
            int_counts: dict[int, int] = {}
            for bstr, cnt in orig_counts.items():
                idx = int(bstr, 2)
                int_counts[idx] = cnt

        # 3. Build and return BitsSampleSet
        return qm_bs.BitsSampleSet.from_int_counts(int_counts, self.num_vars)

    def transpile_circuit(self, circuit: qm_c.QuantumCircuit):
        """
        No circuit-level changes needed; return the original circuit.
        """
        return circuit

    def transpile_hamiltonian(self, operator: qm_o.Hamiltonian):
        """
        No Hamiltonian-level changes needed; return the original operator.
        """
        return operator
