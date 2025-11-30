from abc import abstractmethod
from typing import Optional

import qamomile.core.circuit as qm_c
import qamomile.core.operator as qm_o

from .layer import Layer
from .parameter_context import ParameterContext


class ParameterizedLayer(Layer):
    """
    ParameterizedLayer is an abstract class that represents a layer with parameters in a quantum circuit.

    Attributes:
        num_params (int): The number of parameters for the layer.
        params (list[qm_c.ParameterExpression | float], optional): A list of parameter expressions or floats. If not provided, parameters will be generated.
        symbol (str): The symbol used to represent the parameters.
        layer_id (int): The identifier for the layer.

    """

    def __init__(
        self,
        num_params,
        params: Optional[list[qm_c.ParameterExpression | float]] = None,
        parameter_context: Optional[ParameterContext] = None,
        symbol="θ",
    ):
        self.num_params = num_params
        self.symbol = symbol
        self.parameter_context = parameter_context

        if params is None:
            self.params = self._generate_parameters()
        else:
            if len(params) != num_params:
                raise ValueError(
                    f"The number of parameters must be {num_params}: {len(params)}"
                )

            self.params = []
            for param in params:
                if isinstance(param, qm_c.ParameterExpression):
                    self.params.append(param)
                else:
                    # If user specified a number, convert it to Value
                    try:
                        self.params.append(qm_c.Value(float(param)))
                    except (ValueError, TypeError):
                        raise TypeError(f"Invalid parameter type: {type(param)}")

    def _generate_parameters(self):
        if self.parameter_context is None:
            # コンテキストがない場合は独自のパラメータを生成
            return [
                qm_c.Parameter(f"{self.symbol}_{i}") for i in range(self.num_params)
            ]
        else:
            # コンテキストを使用してパラメータを生成
            return [
                self.parameter_context.get_next_parameter(self.symbol)
                for _ in range(self.num_params)
            ]

    def set_parameter_context(
        self, context: ParameterContext, regenerate: bool = False
    ) -> None:
        """
        パラメータコンテキストを設定します。

        Args:
            context: 新しいパラメータコンテキスト
            regenerate: True の場合、パラメータを再生成します
        """
        self.parameter_context = context

        if regenerate or self.params is None:
            self.params = self._generate_parameters()

    @abstractmethod
    def get_circuit(self) -> qm_c.QuantumCircuit:
        pass


class RotationLayer(ParameterizedLayer):
    """
    RotationLayer class applies a specified rotation layer to a quantum circuit.

    Attributes:
        rotation_type (str): Type of rotation to apply ('rx', 'ry', 'rz').
        num_qubits (int): Number of qubits in the layer.
        params (list, optional): Parameters for the rotations. Defaults to None.
        symbol (str, optional): Symbol representing the parameters. Defaults to "θ".
        layer_id (int, optional): Identifier for the layer. Defaults to None.

    """

    SUPPORTED_ROTATIONS = ["rx", "ry", "rz"]

    def __init__(
        self,
        num_qubits: int,
        rotation_type: str,
        params=None,
        symbol="θ",
        parameter_context: Optional[ParameterContext] = None,
    ):
        if rotation_type not in self.SUPPORTED_ROTATIONS:
            raise ValueError(
                f"Unsupported rotation type: {rotation_type}. Supported types: {self.SUPPORTED_ROTATIONS}"
            )

        self.rotation_type = rotation_type
        self.num_qubits = num_qubits

        super().__init__(
            num_params=num_qubits,
            params=params,
            symbol=symbol,
            parameter_context=parameter_context,
        )

    def get_circuit(self) -> qm_c.QuantumCircuit:
        circuit = qm_c.QuantumCircuit(
            self.num_qubits, 0, name="RotationLayer: " + self.rotation_type
        )

        rotation_methods = {"rx": circuit.rx, "ry": circuit.ry, "rz": circuit.rz}

        apply_rotation = rotation_methods[self.rotation_type]

        for i in range(self.num_qubits):
            apply_rotation(self.params[i], i)

        return circuit


class CostLayer(ParameterizedLayer):
    """
    CostMixerLayer class applies a cost mixer layer to a quantum circuit.

    Attributes:
        num_qubits (int): Number of qubits in the layer.
        params (list, optional): Parameters for the cost mixer. Defaults to None.
        symbol (str, optional): Symbol representing the parameters. Defaults to "θ".
        layer_id (int, optional): Identifier for the layer. Defaults to None.

    """

    def __init__(
        self,
        hamiltonian: qm_o.Hamiltonian,
        symbol="γ",
        parameter_context: Optional[ParameterContext] = None,
    ):
        self.num_qubits = hamiltonian.num_qubits
        self.hamiltonian = hamiltonian

        super().__init__(
            num_params=1,
            symbol=symbol,
            parameter_context=parameter_context,
        )

    def get_circuit(self) -> qm_c.QuantumCircuit:
        circuit = qm_c.QuantumCircuit(self.num_qubits, 0, name="CostMixerLayer")

        circuit.exp_evolution(self.params[0], self.hamiltonian)

        return circuit


class MixerLayer(ParameterizedLayer):
    """A layer that applies a mixer operation to a quantum circuit."""

    SUPPORTED_MIXER_TYPES = ["x", "y", "z"]

    def __init__(
        self,
        num_qubits,
        symbol="β",
        parameter_context: Optional[ParameterContext] = None,
        mixer_type="x",
    ):
        if mixer_type not in self.SUPPORTED_MIXER_TYPES:
            raise ValueError(
                f"Unsupported mixer type: {mixer_type}. Supported types: {self.SUPPORTED_MIXER_TYPES}"
            )

        self.num_qubits = num_qubits
        self.mixer_type = mixer_type

        super().__init__(
            num_params=1, symbol=symbol, parameter_context=parameter_context
        )

    def get_circuit(self) -> qm_c.QuantumCircuit:
        """Apply the mixer layer to the given quantum circuit.

        Args:
            circuit (QuantumCircuit): The quantum circuit to which the layer will be applied.
        """
        circuit = qm_c.QuantumCircuit(self.num_qubits, 0, name="MixerLayer")

        rotation_methods = {"x": circuit.rx, "y": circuit.ry, "z": circuit.rz}
        apply_rotation = rotation_methods[self.mixer_type]

        for i in range(self.num_qubits):
            apply_rotation(self.params[0], i)

        return circuit
