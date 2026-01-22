import abc
import dataclasses

from qamomile.circuit.ir.operation.operation import Operation, OperationKind


class Runnable(abc.ABC):
    def __init__(self, operations: list) -> None:
        self._global_vars: dict[str, int | float | bool | list] = {}
        self.operations = operations

    @abc.abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError()

    def input_args(self) -> dict[str, int | float | bool | list]:
        return {}

    def set_global_vars(self, global_vars: dict[str, int | float | bool | list]):
        self._global_vars = global_vars

    def reset_global_vars(self):
        self._global_vars = {}

    def return_assingnments(
        self, return_values: list
    ) -> dict[str, int | float | bool | list]:
        return {}


class QuantumRunnable(Runnable):
    pass


class ClassicalRunnable(Runnable):
    pass


def separate_operations(
    operations: list[Operation],
    quantum_runnable_cls: type[QuantumRunnable],
    classical_runnable_cls: type[ClassicalRunnable],
) -> list[Runnable]:
    runnables: list[Runnable] = []
    current_runnable: list[Operation] = []
    current_context: OperationKind | None = None

    for ops in operations:
        if current_context != ops.operation_kind:
            # Switch Context
            current_runnable = [ops]

        current_runnable.append(ops)

        if current_context != ops.operation_kind:
            match ops.operation_kind:
                case OperationKind.QUANTUM:
                    runnables.append(quantum_runnable_cls(current_runnable))
                case OperationKind.CLASSICAL:
                    runnables.append(classical_runnable_cls(current_runnable))
            current_runnable = []

    return runnables


@dataclasses.dataclass
class Orchestrator:
    global_variables: dict[str, float | int | bool | list] = dataclasses.field(
        default_factory=dict
    )
    operations: list[Runnable] = dataclasses.field(default_factory=list)
    return_ids: list[tuple[str, int]] = dataclasses.field(default_factory=list)

    def run(self, **kwargs):
        self.global_variables = kwargs

        for operation in self.operations[:-1]:
            args = operation.input_args()
            operation.set_global_vars(self.global_variables)
            returns = operation.run(args)
            self.global_variables.update(operation.return_assingnments(returns))
            operation.reset_global_vars()

        return_values = [self.global_variables[ri[0]] for ri in self.return_ids]

        self.global_variables = {}

        return return_values
