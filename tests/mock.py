from qamomile.core.transpiler import QuantumSDKTranspiler


class UnsupportedBinaryOpeKind:
    """Mock class for an unsupported binary operation kind."""

    pass


class UnsupportedParam:
    """Mock class for an unsupported parameter type."""

    pass


class UnsupportedGate:
    """Mock class for an unsupported gate type."""

    pass


class InvalidTranspilerNoConvertResult(QuantumSDKTranspiler[int]):
    """Mock class for QuantumSDKTranspiler whose convert_result has not implemented."""

    def transpile_circuit(self, circuit):
        pass

    def transpile_hamiltonian(self, operator):
        pass


class InvalidTranspilerNoTranspileCircuit(QuantumSDKTranspiler[int]):
    """Mock class for QuantumSDKTranspiler whose transpile_circuit has not implemented."""

    def convert_result(self, result):
        pass

    def transpile_hamiltonian(self, operator):
        pass


class InvalidTranspilerNoTranspileHamiltonian(QuantumSDKTranspiler[int]):
    """Mock class for QuantumSDKTranspiler whose transpile_hamiltonian has not implemented."""

    def convert_result(self, result):
        pass

    def transpile_circuit(self, circuit):
        pass


class ValidTranspilerWithAllImplementations(QuantumSDKTranspiler[int]):
    """Mock class for QuantumSDKTranspiler with all methods implemented."""

    def convert_result(self, result):
        pass

    def transpile_circuit(self, circuit):
        pass

    def transpile_hamiltonian(self, operator):
        pass
