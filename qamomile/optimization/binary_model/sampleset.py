import dataclasses
from typing import Generic

import numpy as np
import ommx.v1

from .expr import VT, VarType


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

    def to_ommx_samples(self) -> ommx.v1.Samples:
        """Convert this BINARY sample set into an OMMX ``Samples`` container.

        Each unique sample state is appended once with a list of sample IDs of
        length ``num_occurrences``, so OMMX-side aggregation reflects the
        original shot counts without duplicating the underlying state. Used
        internally by :meth:`MathematicalProblemConverter.decode` to feed
        bitstrings into ``Instance.evaluate_samples``.

        Returns:
            ommx.v1.Samples: An OMMX Samples object with ``sum(num_occurrences)``
            sample IDs, where IDs sharing the same state are grouped together.

        Raises:
            ValueError: If ``vartype`` is not ``VarType.BINARY``. OMMX expects
                0/1 decision-variable values, so SPIN sample sets must be
                converted to BINARY before calling this method.

        Example:
            >>> ss = BinarySampleSet(
            ...     samples=[{0: 1, 1: 0}, {0: 0, 1: 1}],
            ...     num_occurrences=[3, 1],
            ...     energy=[0.0, 0.0],
            ...     vartype=VarType.BINARY,
            ... )
            >>> ommx_samples = ss.to_ommx_samples()
            >>> ommx_samples.num_samples()
            4
        """
        # Compare via the enum's str value: bound TypeVar VT can be narrowed
        # to the default (BINARY) by static type-checkers, which then flag
        # the guard's non-BINARY branch as unreachable. The str(...) cast
        # bypasses that narrowing without changing runtime semantics.
        if str(self.vartype) != str(VarType.BINARY):
            raise ValueError(
                "to_ommx_samples requires vartype=BINARY; got "
                f"{self.vartype}. Convert to BINARY first."
            )

        ommx_samples = ommx.v1.Samples({})
        next_id = 0
        for sample, occ in zip(self.samples, self.num_occurrences):
            if occ <= 0:
                continue
            sample_ids = list(range(next_id, next_id + occ))
            next_id += occ
            state = ommx.v1.State({idx: float(val) for idx, val in sample.items()})
            ommx_samples.append(sample_ids, state)
        return ommx_samples
