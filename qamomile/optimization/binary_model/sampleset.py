import dataclasses
from typing import Generic
import numpy as np

from .expr import VarType, VT


@dataclasses.dataclass
class BinarySampleSet(Generic[VT]):
    samples: list[dict[int, int]]
    num_occurrences: list[int]
    energy: list[float]
    vartype: VT = VarType.BINARY  # type: ignore

    def lowest(self) -> tuple[dict[int, int], float, int]:
        min_energy = min(self.energy)
        min_index = self.energy.index(min_energy)
        min_occurrences = self.num_occurrences[min_index]
        return self.samples[min_index], min_energy, min_occurrences

    def energy_mean(self) -> float:
        _e = np.array(self.energy)
        _n = np.array(self.num_occurrences)
        return float(_e @ _n / np.sum(_n))
