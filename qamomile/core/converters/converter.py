"""
qamomile/core/converter.py

This module defines the QuantumConverter abstract base class for converting between
different problem representations and quantum models in Qamomile.

The QuantumConverter class provides a framework for encoding classical optimization
problems into quantum representations (e.g., Ising models) and decoding quantum
computation results back into classical problem solutions.

Key Features:
- Conversion between classical optimization problems and Ising models.
- Abstract methods for generating cost Hamiltonians and decoding results.
- Integration with QuantumSDKTranspiler for SDK-specific result handling.

Usage:
Developers implementing specific quantum conversion strategies should subclass
QuantumConverter and implement the abstract methods. The class is designed to work
with jijmodeling for problem representation and various quantum SDKs through
the QuantumSDKTranspiler interface.

Example:
    class QAOAConverter(QuantumConverter):
        def get_cost_hamiltonian(self) -> qm_o.Hamiltonian:
            # Implementation for generating QAOA cost Hamiltonian
            ...

"""

import abc
import typing as typ

import numpy as np
import jijmodeling as jm
import jijmodeling_transpiler.core as jmt
import qamomile.core.bitssample as qm_bs
import qamomile.core.operator as qm_o
from qamomile.core.ising_qubo import IsingModel, qubo_to_ising
from qamomile.core.transpiler import QuantumSDKTranspiler

# Import necessary functions from jijmodeling_transpiler
from jijmodeling_transpiler.core.decode import dict_to_record
from jijmodeling_transpiler.core.pubo.binary_decode import binary_decode
from jijmodeling_transpiler.core.decode.evaluate import calc_expr, subs_expr


ResultType = typ.TypeVar("ResultType")


class QuantumConverter(abc.ABC):
    """
    Abstract base class for quantum problem converters in Qamomile.

    This class provides methods for encoding classical optimization problems
    into quantum representations (e.g., Ising models) and decoding quantum
    computation results back into classical problem solutions.

    Attributes:
        compiled_instance: The compiled instance of the optimization problem.
        pubo_builder: The PUBO (Polynomial Unconstrained Binary Optimization) builder.
        _ising (Optional[IsingModel]): Cached Ising model representation.

    Methods:
        get_ising: Retrieve or compute the Ising model representation.
        ising_encode: Encode the problem into an Ising model.
        get_cost_hamiltonian: Abstract method to get the cost Hamiltonian.
        decode: Decode quantum computation results into a SampleSet.
        decode_bits_to_sampleset: Abstract method to convert BitsSampleSet to SampleSet.
    """

    def __init__(
        self,
        compiled_instance,
        relax_method: jmt.pubo.RelaxationMethod = jmt.pubo.RelaxationMethod.AugmentedLagrangian,
    ):
        """
        Initialize the QuantumConverter.

        Args:
            compiled_instance: The compiled instance of the optimization problem.
            relax_method (jmt.pubo.RelaxationMethod): The relaxation method for PUBO conversion.
                Defaults to AugmentedLagrangian.
        """
        pubo_builder = jmt.pubo.transpile_to_pubo(
            compiled_instance, relax_method=relax_method
        )

        self.compiled_instance = compiled_instance
        self.pubo_builder = pubo_builder
        self.int2varlabel: dict[int, str] = {}

        self._ising: typ.Optional[IsingModel] = None

    def get_ising(self) -> IsingModel:
        """
        Get the Ising model representation of the problem.

        Returns:
            IsingModel: The Ising model representation.
        """
        if self._ising is None:
            self._ising = self.ising_encode()
        return self._ising

    def ising_encode(
        self,
        multipliers: typ.Optional[dict[str, float]] = None,
        detail_parameters: typ.Optional[
            dict[str, dict[tuple[int, ...], tuple[float, float]]]
        ] = None,
    ) -> IsingModel:
        """
        Encode the problem to an Ising model.

        This method converts the problem from QUBO (Quadratic Unconstrained Binary Optimization)
        to Ising model representation.

        Args:
            multipliers (Optional[dict[str, float]]): Multipliers for constraint terms.
            detail_parameters (Optional[dict[str, dict[tuple[int, ...], tuple[float, float]]]]):
                Detailed parameters for the encoding process.

        Returns:
            IsingModel: The encoded Ising model.
        """
        qubo, constant = self.pubo_builder.get_qubo_dict(
            multipliers=multipliers, detail_parameters=detail_parameters
        )
        ising = qubo_to_ising(qubo, simplify=False)
        ising.constant += constant

        var_map = self.compiled_instance.var_map.var_map
        inv_varmap = {}
        for var_label, var_indices in var_map.items():
            for subs, index in var_indices.items():
                inv_varmap[index] = var_label + "_{" + ",".join(map(str, subs)) + "}"
        self.int2varlabel = inv_varmap

        return ising

    @abc.abstractmethod
    def get_cost_hamiltonian(self) -> qm_o.Hamiltonian:
        """
        Abstract method to get the cost Hamiltonian for the quantum problem.

        This method should be implemented in subclasses to define how the
        cost Hamiltonian is constructed for specific quantum algorithms.

        Returns:
            qm_o.Hamiltonian: The cost Hamiltonian for the quantum problem.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError()

    def decode(
        self, transpiler: QuantumSDKTranspiler[ResultType], result: ResultType
    ) -> jm.experimental.SampleSet:
        """
        Decode quantum computation results into a SampleSet.

        This method uses the provided transpiler to convert SDK-specific results
        into a BitsSampleSet, then calls decode_bits_to_sampleset to produce
        the final SampleSet.

        Args:
            transpiler (QuantumSDKTranspiler[ResultType]): The transpiler for the specific quantum SDK.
            result (ResultType): The raw result from the quantum computation.

        Returns:
            jm.experimental.SampleSet: The decoded results as a SampleSet.
        """
        bitssampleset = transpiler.convert_result(result)
        return self.decode_bits_to_sampleset(bitssampleset)

    def decode_bits_to_sampleset(
        self, bitssampleset: qm_bs.BitsSampleSet
    ) -> jm.experimental.SampleSet:
        """
        Decode a BitArraySet to a SampleSet.

        This method converts the quantum computation results (bitstrings)
        into a format that represents solutions to the original optimization problem.

        Args:
            bitarray_set (qm_c.BitArraySet): The set of bitstring results from quantum computation.

        Returns:
            jm.experimental.SampleSet: The decoded results as a SampleSet.
        """
        ising = self.get_ising()
        num_occurrences = []
        samples = []

        # Convert bitstrings to samples
        for bitssample in bitssampleset.bitarrays:
            sample = {}
            for i, bit in enumerate(bitssample.bits):
                index = ising.ising2qubo_index(i)
                sample[index] = bit
            samples.append(sample)
            num_occurrences.append(bitssample.num_occurrences)

        # Decode samples using jijmodeling_transpiler
        sampleset = decode_from_dict_binary_result(
            samples, self.pubo_builder.binary_encoder, self.compiled_instance
        )

        # Update the number of occurrences
        record = jm.Record(sampleset.record.solution, num_occurrences=num_occurrences)
        sampleset.record = record

        return jm.experimental.from_old_sampleset(sampleset)


# Helper functions for decoding results
def decode_from_dict_binary_result(
    samples: typ.Iterable[dict[int, int | float]],
    binary_encoder,
    compiled_model: jmt.CompiledInstance,
) -> jm.SampleSet:
    """
    Decode binary results into a SampleSet.

    Args:
        samples: Iterable of sample dictionaries.
        binary_encoder: Binary encoder from jijmodeling_transpiler.
        compiled_model: Compiled instance of the optimization problem.

    Returns:
        jm.SampleSet: Decoded sample set.
    """
    inverse_varmap: dict[int, tuple[str, tuple[int, ...]]] = {}
    for label, values in compiled_model.var_map.var_map.items():
        for forall, index in values.items():
            inverse_varmap[index] = (label, forall)

    decoded_samples = binary_decode(samples, binary_encoder, inverse_varmap)

    record = dict_to_record(decoded_samples, compiled_model)

    evaluation = _evaluate(decoded_samples, compiled_model)

    return jm.SampleSet(
        record=record,
        evaluation=evaluation,
        measuring_time=jm.MeasuringTime(),
    )


def _evaluate(
    samples: typ.Iterable[dict[int, int | float]],
    compiled_model: jmt.CompiledInstance,
) -> jm.Evaluation:
    """
    Evaluate samples against the compiled model.

    This function calculates objective values, constraint violations,
    and penalty values for each sample.

    Args:
        samples: Iterable of sample dictionaries.
        compiled_model: Compiled instance of the optimization problem.

    Returns:
        jm.Evaluation: Evaluation results for the samples.
    """
    objectives: list[float] = []
    const_violation: dict[str, list[float]] = {
        label: [] for label in compiled_model.constraint.keys()
    }
    constraint_forall = {}
    constraint_values = []
    pena_violation: dict[str, list[float]] = {
        label: [] for label in compiled_model.penalty.keys()
    }

    for i, sample in enumerate(samples):
        sample = dict(sample)
        result = calc_expr(sample, compiled_model)
        objectives.append(result.objective)
        constraint_values.append({})

        # Process constraints
        for label, const_value in result.constraint.items():
            if i == 0:
                constraint_forall[label] = np.array(
                    [list(subs) for subs in const_value.keys()]
                )
            values = np.array(list(const_value.values()))
            constraint_condition = compiled_model.problem.constraints[label].condition
            if constraint_condition.kind == subs_expr.ConstraintKind.EQUAL:
                values = np.abs(values)
            else:
                feas = values <= 0
                values[feas] = 0.0
            const_violation[label].append(np.sum(values))
            constraint_values[i][label] = values

        # Process penalties
        for label, pena_values in result.penalty.items():
            pena_violation[label].append(sum(list(pena_values.values())))

    return jm.Evaluation(
        objective=np.array(objectives, dtype=np.float64),
        constraint_violations=const_violation,
        constraint_forall=constraint_forall,
        constraint_values=constraint_values,
        penalty=pena_violation,
    )
