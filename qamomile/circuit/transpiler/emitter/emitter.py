import typing
import abc

from qamomile.circuit.ir.block_value import BlockValue


T = typing.TypeVar("T")


class EmitResult(abc.ABC, typing.Generic[T]):
    @property
    @abc.abstractmethod
    def circuit(self) -> T:
        pass


class Emitter(abc.ABC, typing.Generic[T]):
    def __init__(
        self, block: BlockValue, bind: dict[str, int | float | list[int] | list[float]]
    ) -> None:
        self.block = block
        self.bind = bind

    @abc.abstractmethod
    def emit(self, block: BlockValue) -> EmitResult[T]:
        pass
