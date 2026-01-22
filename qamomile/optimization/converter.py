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
    .. code::

        class QAOAConverter(QuantumConverter):
            def get_cost_hamiltonian(self) -> qm_o.Hamiltonian:
                # Implementation for generating QAOA cost Hamiltonian
                ...

"""

import abc
import copy
import enum
import typing as typ

import ommx.v1

import qamomile.core.bitssample as qm_bs
import qamomile.core.operator as qm_o
from qamomile.core.higher_ising_model import HigherIsingModel
from qamomile.core.ising_qubo import IsingModel
from qamomile.core.transpiler import QuantumSDKTranspiler

ResultType = typ.TypeVar("ResultType")


class RelaxationMethod(enum.Enum):
    """
    Enumeration for relaxation methods used in quantum problem conversion.

    Attributes:
        AugmentedLagrangian: Augmented Lagrangian method for PUBO conversion.
        SquaredPenalty: Squared penalty method for PUBO conversion.
    """

    AugmentedLagrangian = "AugmentedLagrangian"
    SquaredPenalty = "SquaredPenalty"


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
        instance: ommx.v1.Instance,
        relax_method: RelaxationMethod = RelaxationMethod.SquaredPenalty,
        normalize_ising: typ.Optional[typ.Literal["abs_max", "rms"]] = None,
    ):
        """
        Initialize the QuantumConverter.

        This method initializes the converter with the compiled instance of the optimization problem

        Args:
            instance (ommx.v1.Instance): an orginal instance to be converted.
            relax_method (RelaxationMethod): The relaxation method for PUBO conversion.
                Defaults to RelaxationMethod.SquaredPenalty.
            normalize_ising (Literal["abs_max", "rms"] | None): The normalization method for the Ising Hamiltonian.
                Available options:
                - "abs_max": Normalize by absolute maximum value
                - "rms": Normalize by root mean square
                Defaults to None.

        Raises:
            ValueError: If an unsupported relaxation method is provided.
            ValueError: If the instance is HUBO but the converter does not support HUBO.
        """
        self.original_instance: ommx.v1.Instance = instance

        # TODO: Support other relaxation methods.
        if relax_method != RelaxationMethod.SquaredPenalty:
            raise ValueError(
                "Relaxation method other than SquaredPenalty is not supported yet."
            )

        # If the problem is HUBO and the conveter does not support, then raise an error.
        if not (self.is_qubo or self._hubo_support):
            raise ValueError(f"The instance is HUBO, but {self} does not support HUBO.")

        self.int2varlabel: dict[int, str] = {}
        self.normalize_ising = normalize_ising

        self._ising: typ.Optional[IsingModel | HigherIsingModel] = None
        self._converted_instance: typ.Optional[ommx.v1.Instance] = None

    @property
    def is_qubo(self) -> bool:
        """Return if the instance is QUBO.
        If the degree is more than 2, it is not QUBO but HUBO.

        Returns:
            bool: if the instance is QUBO
        """
        return self.original_instance.objective.degree() <= 2

    @property
    @abc.abstractmethod
    def _hubo_support(self) -> bool:
        """Property to show if this class supports HUBO.

        Raises:
            NotImplementedError: if this property is not implemented in subclass

        Returns:
            bool: if this class supports HUBO
        """
        raise NotImplementedError()

    def instance_to_qubo(
        self,
        multipliers: typ.Optional[dict[str, float]] = None,
        detail_parameters: typ.Optional[dict[str, dict[tuple[int, ...], float]]] = None,
    ) -> tuple[dict[tuple[int, int], float], float]:
        """
        Convert the instance to QUBO format.

        This method converts the optimization problem instance into a QUBO (Quadratic Unconstrained Binary Optimization)
        representation, which is suitable for quantum computation.

        Args:
            multipliers (Optional[dict[str, float]]): Multipliers for constraint terms.
            detail_parameters (Optional[dict[str, dict[tuple[int, ...], float]]]):
                Detailed parameters for the encoding process.

        Note:
            $\min_x f(x)$~s.t. $g_{s, i}(x) = 0~\forall s, i$ is converted to
            $\min_x f(x) + \sum_{s \in \{\text{'const1'}, \cdots\}} A_s \sum_i \lambda_i g_i(x)$.

            where $A_s$ is the multiplier for constraint $s$ and $\lambda_i$ is the detailed parameter for constraint $s$ with subscripts $i$.

        Returns:
            tuple[dict[tuple[int, int], float], float]: A tuple containing the QUBO dictionary and the constant term.


        Example:
            .. code::
                imoprt jijmodeling as jm
                n = jm.Placeholder("n")
                x = jm.BinaryVar("x", shape=(n,))
                y = jm.BinaryVar("y")
                problem = jm.Problem("sample")
                i = jm.Element("i", (0, n))
                problem += jm.Constraint("const1", x[i] + y == 1, forall=i)
                intepreter = jm.Interpreter({"n": 3})
                multipliers = {"const1": 1.0}
                detail_parameters = {"const1": {(0,): 2.0}}
                qubo, constant = converter.instance_to_qubo(multipliers, detail_parameters)

        """
        penalty_weights = self.get_penalty_weights(
            multipliers=multipliers, detail_parameters=detail_parameters
        )
        instance_copy = copy.deepcopy(self.original_instance)
        qubo, constant = instance_copy.to_qubo(penalty_weights=penalty_weights)

        # Store the modified instance for later access to slack and log-encoded variables.
        self._converted_instance = instance_copy

        return qubo, constant

    def instance_to_hubo(
        self,
        multipliers: typ.Optional[dict[str, float]] = None,
        detail_parameters: typ.Optional[dict[str, dict[tuple[int, ...], float]]] = None,
    ) -> tuple[dict[tuple[int, ...], float], float]:
        """
        Convert the instance to HUBO format.

        This method converts the optimization problem instance into a HUBO (Higher-order Unconstrained Binary Optimization)
        representation, which is suitable for quantum computation. HUBO generalizes QUBO by allowing polynomial terms
        of arbitrary order (linear, quadratic, cubic, quartic, quintic, etc.).

        Args:
            multipliers (Optional[dict[str, float]]): Multipliers for constraint terms.
            detail_parameters (Optional[dict[str, dict[tuple[int, ...], float]]]):
                Detailed parameters for the encoding process.

        Note:
            $\min_x f(x)$~s.t. $g_{s, i}(x) = 0~\forall s, i$ is converted to
            $\min_x f(x) + \sum_{s \in \{\text{'const1'}, \cdots\}} A_s \sum_i \lambda_i g_i(x)$.

            where $A_s$ is the multiplier for constraint $s$ and $\lambda_i$ is the detailed parameter for constraint $s$ with subscripts $i$.

        Returns:
            tuple[dict[tuple[int, ...], float], float]: A tuple containing the HUBO dictionary and the constant term.
                The keys in the HUBO dictionary are tuples of variable indices with arbitrary length (1 for linear,
                2 for quadratic, 3 for cubic, etc.), and values are the corresponding coefficients.


        Example:
            .. code::
                imoprt jijmodeling as jm
                n = jm.Placeholder("n")
                x = jm.BinaryVar("x", shape=(n,))
                y = jm.BinaryVar("y")
                problem = jm.Problem("sample")
                i = jm.Element("i", (0, n))
                problem += jm.Constraint("const1", x[i] + y == 1, forall=i)
                intepreter = jm.Interpreter({"n": 3})
                multipliers = {"const1": 1.0}
                detail_parameters = {"const1": {(0,): 2.0}}
                hubo, constant = converter.instance_to_hubo(multipliers, detail_parameters)

        """
        penalty_weights = self.get_penalty_weights(
            multipliers=multipliers, detail_parameters=detail_parameters
        )
        instance_copy = copy.deepcopy(self.original_instance)
        hubo, constant = instance_copy.to_hubo(penalty_weights=penalty_weights)

        # Store the modified instance for later access to slack and log-encoded variables.
        self._converted_instance = instance_copy

        return hubo, constant

    def get_penalty_weights(
        self,
        multipliers: typ.Optional[dict[str, float]] = None,
        detail_parameters: typ.Optional[dict[str, dict[tuple[int, ...], float]]] = None,
    ) -> dict[int, float]:
        """Get penalty weights computed with the given parameters.

        Args:
            multipliers (Optional[dict[str, float]]): Multipliers for constraint terms.
            detail_parameters (Optional[dict[str, dict[tuple[int, ...], float]]]):
                Detailed parameters for the encoding process.

        Returns:
            dict[int, float]: a dictionary mapping constraint IDs to their penalty weights
        """
        _multipliers = multipliers if multipliers is not None else {}
        _parameters = detail_parameters if detail_parameters is not None else {}

        penalty_weights = {}
        for constraint in self.original_instance.constraints:
            name = constraint.name
            if name is not None and name in _multipliers:
                multiplier = _multipliers[name]
            else:
                multiplier = 1.0
            subscripts = tuple(constraint.subscripts)
            if name is not None and name in _parameters:
                multiplier *= _parameters[name].get(subscripts, 1.0)

            const_id = constraint.id
            penalty_weights[const_id] = multiplier

        return penalty_weights

    def get_ising(self) -> IsingModel | HigherIsingModel:
        """
        Get the Ising model representation of the problem.

        Returns:
            IsingModel | HigherIsingModel: The Ising model representation.

        """
        if self._ising is None:
            self._ising = self.ising_encode()
        return self._ising

    def ising_encode(
        self,
        multipliers: typ.Optional[dict[str, float]] = None,
        detail_parameters: typ.Optional[dict[str, dict[tuple[int, ...], float]]] = None,
        simplify: bool = True,
    ) -> IsingModel | HigherIsingModel:
        """
        Encode the problem to an Ising model.

        This method converts the problem from QUBO (Quadratic Unconstrained Binary Optimization)
        to Ising model representation.

        Args:
            multipliers (Optional[dict[str, float]]): Multipliers for constraint terms.
            detail_parameters (Optional[dict[str, dict[tuple[int, ...], float]]]):
                Detailed parameters for the encoding process.
            simplify (bool): Whether to simplify the Ising model by removing zero coefficients. Defaults to True.

        Raises:
            ValueError: If the instance is HUBO but HUBO is not supported.

        Returns:
            IsingModel | HigherIsingModel: The encoded Ising model.

        """
        # If the problem is QUBO, use instance_to_qubo.
        if self.is_qubo:
            qubo, constant = self.instance_to_qubo(multipliers, detail_parameters)
            ising = IsingModel.from_qubo(
                qubo=qubo, constant=constant, simplify=simplify
            )
        # If the problem is HUBO and the class supports HUBO, convert to HUBO first.
        elif self._hubo_support:
            hubo, constant = self.instance_to_hubo(multipliers, detail_parameters)
            ising = HigherIsingModel.from_hubo(
                hubo=hubo, constant=constant, simplify=simplify
            )
        else:
            raise ValueError("The instance is HUBO, but HUBO is not supported.")

        if isinstance(self.normalize_ising, str):
            if self.normalize_ising == "abs_max":
                ising.normalize_by_abs_max()
            elif self.normalize_ising == "rms":
                ising.normalize_by_rms()
            else:
                raise ValueError(
                    f"Invalid value for normalize_ising: {self.normalize_ising}"
                )

        # Use the converted instance's decision variables after to_qubo conversion
        # This handles log-encoded variables created during to_qubo
        for ising_index, problem_index in ising.index_map.items():
            # self.instance_to_qubo has guranteeed the converted instance is not None.
            deci_var = self._converted_instance.get_decision_variable_by_id(
                problem_index
            )
            var_name = deci_var.name if deci_var.name else "unnamed"
            subscripts = deci_var.subscripts if deci_var.subscripts else []
            self.int2varlabel[ising_index] = (
                var_name + "_{" + ",".join(map(str, subscripts)) + "}"
            )

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
    ) -> ommx.v1.SampleSet:
        """
        Decode quantum computation results into a SampleSet.

        This method uses the provided transpiler to convert SDK-specific results
        into a BitsSampleSet, then calls decode_bits_to_sampleset to produce
        the final SampleSet.

        Args:
            transpiler (QuantumSDKTranspiler[ResultType]): The transpiler for the specific quantum SDK.
            result (ResultType): The raw result from the quantum computation.

        Returns:
            ommx.v1.SampleSet: The decoded results as a SampleSet.
        """
        bitssampleset = transpiler.convert_result(result)
        return self.decode_bits_to_sampleset(bitssampleset)

    def decode_bits_to_sampleset(
        self, bitssampleset: qm_bs.BitsSampleSet
    ) -> ommx.v1.SampleSet:
        """
        Decode a BitArraySet to a SampleSet.

        This method converts the quantum computation results (bitstrings)
        into a format that represents solutions to the original optimization problem.

        Args:
            bitarray_set (qm_c.BitArraySet): The set of bitstring results from quantum computation.

        Returns:
            ommx.v1.SampleSet: The decoded results as a SampleSet.
        """
        ising = self.get_ising()

        # Create ommx.v1.Samples
        sample_id = 0
        samples = ommx.v1.Samples(entries=[])
        for bitssample in bitssampleset.bitarrays:
            sample = {}
            for i, bit in enumerate(bitssample.bits):
                index = ising.ising2original_index(i)
                sample[index] = bit
            state = ommx.v1.State(entries=sample)
            # `num_occurrences` is encoded into sample ID list.
            # For example, if `num_occurrences` is 2, there are two samples with the same state, thus two sample IDs are generated.
            ids = []
            for _ in range(bitssample.num_occurrences):
                ids.append(sample_id)
                sample_id += 1
            samples.append(sample_ids=ids, state=state)

        return self.original_instance.evaluate_samples(samples)
