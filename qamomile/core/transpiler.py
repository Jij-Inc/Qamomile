"""
This module defines the abstract base class for quantum SDK transpilers in Qamomile.

The QuantumSDKTranspiler class serves as a template for creating transpilers
that convert between Qamomile's internal representations and other quantum SDKs.
It ensures a consistent interface for circuit transpilation, Hamiltonian conversion,
and result interpretation across different quantum computing platforms.

Key Features:
- Abstract methods for converting quantum circuits, Hamiltonians, and measurement results.
- Generic typing to specify the expected result type from the target SDK.
- Designed to be subclassed for each specific quantum SDK integration.

Usage:
Developers implementing transpilers for specific quantum SDKs should subclass
QuantumSDKTranspiler and implement all abstract methods. The ResultType generic
parameter should be set to the appropriate result type of the target SDK.

Example:
    class QiskitTranspiler(QuantumSDKTranspiler[qiskit.result.Result]):
        def convert_result(self, result: qiskit.result.Result) -> qm.BitsSampleSet:
            # Implementation for converting Qiskit results to Qamomile's BitsSampleSet
            ...

        def transpile_circuit(self, circuit: qm_c.QuantumCircuit) -> qiskit.QuantumCircuit:
            # Implementation for converting Qamomile's QuantumCircuit to Qiskit's QuantumCircuit
            ...

        def transpile_hamiltonian(self, operator: qm_o.Hamiltonian) -> qiskit.quantum_info.Operator:
            # Implementation for converting Qamomile's Hamiltonian to Qiskit's Operator
            ...
"""

import abc
import typing as typ
import qamomile.core as qm
import qamomile.core.circuit as qm_c
import qamomile.core.operator as qm_o

# Define a type variable for the result type of the target SDK
ResultType = typ.TypeVar("ResultType")


class QuantumSDKTranspiler(typ.Generic[ResultType], abc.ABC):
    """
    Abstract base class for quantum SDK transpilers in Qamomile.

    This class defines the interface for transpilers that convert between
    Qamomile's internal representations and other quantum SDKs. It uses
    generic typing to specify the expected result type from the target SDK.

    Attributes:
        ResultType (TypeVar): A type variable representing the result type of the target SDK.

    Methods:
        convert_result: Convert SDK-specific result to Qamomile's BitsSampleSet.
        transpile_circuit: Convert Qamomile's QuantumCircuit to SDK-specific circuit.
        transpile_hamiltonian: Convert Qamomile's Hamiltonian to SDK-specific operator.

    Note:
        When subclassing, specify the concrete type for ResultType, e.g.,
        class QiskitTranspiler(QuantumSDKTranspiler[qiskit.result.Result]):
    """

    @abc.abstractmethod
    def convert_result(self, result: ResultType) -> qm.BitsSampleSet:
        """
        Convert the result from the target SDK to Qamomile's BitsSampleSet.

        This method should be implemented to interpret the measurement results
        from the target SDK and convert them into Qamomile's BitsSampleSet format.

        Args:
            result (ResultType): The measurement result from the target SDK.

        Returns:
            qm.BitsSampleSet: The converted result in Qamomile's format.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def transpile_circuit(self, circuit: qm_c.QuantumCircuit):
        """
        Transpile a Qamomile QuantumCircuit to the target SDK's circuit representation.

        This method should be implemented to convert Qamomile's internal
        QuantumCircuit representation to the equivalent circuit in the target SDK.

        Args:
            circuit (qm_c.QuantumCircuit): The Qamomile QuantumCircuit to be transpiled.

        Returns:
            The equivalent circuit in the target SDK's format.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError()

    def transpile_operators(self, operators: list[qm_o.Hamiltonian]):
        """
        Transpile a list of Qamomile Hamiltonians to the target SDK's operator representation.

        This method should be implemented to convert a list of Qamomile Hamiltonians
        to the equivalent operators in the target SDK.

        Args:
            operators (list[qm_o.Hamiltonian]): The list of Qamomile Hamiltonians to be transpiled.

        Returns:
            list: The equivalent operators in the target SDK's format.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        return [self.transpile_hamiltonian(op) for op in operators]

    @abc.abstractmethod
    def transpile_hamiltonian(self, operator: qm_o.Hamiltonian):
        """
        Transpile a Qamomile Hamiltonian to the target SDK's operator representation.

        This method should be implemented to convert Qamomile's internal
        Hamiltonian representation to the equivalent operator in the target SDK.

        Args:
            operator (qm_o.Hamiltonian): The Qamomile Hamiltonian to be transpiled.

        Returns:
            The equivalent operator in the target SDK's format.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError()
