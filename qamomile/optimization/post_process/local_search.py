"""Local search post-processing for binary optimization models.

Provides first-improvement and best-improvement local search algorithms
that operate on :class:`BinaryModel` instances (both SPIN and BINARY).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from qamomile.optimization.binary_model.expr import VarType
from qamomile.optimization.binary_model.model import BinaryModel
from qamomile.optimization.binary_model.sampleset import BinarySampleSet


class LocalSearch:
    """Local search optimizer for :class:`BinaryModel`.

    Internally converts the model to SPIN representation (±1) for the search,
    then converts results back to the original vartype.

    Args:
        model: The binary model to optimize.
    """

    def __init__(self, model: BinaryModel) -> None:
        self._model = model
        self._spin_model = (
            model.change_vartype(VarType.SPIN)
            if model.vartype != VarType.SPIN
            else model
        )

        n = self._spin_model.num_bits
        self._quad = np.zeros((n, n))
        for (i, j), v in self._spin_model.quad.items():
            self._quad[i, j] = v
            self._quad[j, i] = v

        self._linear = np.zeros(n)
        for i, v in self._spin_model.linear.items():
            self._linear[i] = v

    def run(
        self,
        initial_state: list[int],
        max_iter: int = -1,
        method: str = "best_improvement",
    ) -> BinarySampleSet:
        """Run local search starting from *initial_state*.

        Args:
            initial_state: Variable values in the model's vartype domain
                (±1 for SPIN, 0/1 for BINARY).
            max_iter: Maximum iterations (-1 for unlimited).
            method: ``"best_improvement"`` or ``"first_improvement"``.

        Returns:
            A :class:`BinarySampleSet` containing the optimized state.

        Raises:
            ValueError: If *method* is not recognized, *initial_state* has
                wrong length or invalid values, or *max_iter* is less than -1.
        """
        if max_iter < -1:
            raise ValueError(
                f"max_iter must be -1 (unlimited) or non-negative, got {max_iter}."
            )

        if len(initial_state) != self._model.num_bits:
            raise ValueError(
                f"initial_state length {len(initial_state)} does not match "
                f"model size {self._model.num_bits}."
            )

        if self._model.vartype == VarType.SPIN:
            if not all(v in (1, -1) for v in initial_state):
                raise ValueError(
                    "All elements of initial_state must be +1 or -1 for SPIN models."
                )
        else:
            if not all(v in (0, 1) for v in initial_state):
                raise ValueError(
                    "All elements of initial_state must be 0 or 1 for BINARY models."
                )

        methods: dict[str, Callable[..., np.ndarray]] = {
            "best_improvement": self._best_improvement,
            "first_improvement": self._first_improvement,
        }
        if method not in methods:
            raise ValueError(
                f"Invalid method: {method}. Choose from {list(methods.keys())}."
            )

        spin_state = self._to_spin(np.asarray(initial_state))
        spin_state = self._search(methods[method], spin_state, max_iter)
        return self._to_sampleset(spin_state)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_spin(self, state: np.ndarray) -> np.ndarray:
        """Convert state from the model's vartype to SPIN (±1)."""
        if self._model.vartype == VarType.BINARY:
            return np.where(state == 0, 1, -1).astype(float)
        else:
            return state.astype(float)

    def _from_spin(self, spin_state: np.ndarray) -> list[int]:
        """Convert SPIN (±1) state back to the model's vartype."""
        if self._model.vartype == VarType.BINARY:
            return [0 if s > 0 else 1 for s in spin_state]
        else:
            return [int(s) for s in spin_state]

    def _search(
        self,
        step: Callable[..., np.ndarray],
        state: np.ndarray,
        max_iter: int,
    ) -> np.ndarray:
        """Run the local search loop until convergence or max_iter."""
        counter = 0
        while max_iter == -1 or counter < max_iter:
            prev = state.copy()
            state = step(state, self._quad, self._linear, len(state))
            if np.array_equal(prev, state):
                break
            counter += 1
        return state

    @staticmethod
    def _calc_e_diff(
        state: np.ndarray, quad: np.ndarray, linear: np.ndarray, idx: int
    ) -> float:
        """Calculate the energy difference when flipping bit *idx*."""
        return float(-2 * state[idx] * (quad[:, idx] @ state + linear[idx]))

    @staticmethod
    def _first_improvement(
        state: np.ndarray, quad: np.ndarray, linear: np.ndarray, n: int
    ) -> np.ndarray:
        """Sweep all bits and accept every flip that lowers energy.

        This is a greedy-sweep variant of first-improvement: each bit is
        examined once per call, and every improving flip is accepted
        immediately before moving to the next bit.
        """
        for i in range(n):
            if LocalSearch._calc_e_diff(state, quad, linear, i) < 0:
                state[i] = -state[i]
        return state

    @staticmethod
    def _best_improvement(
        state: np.ndarray, quad: np.ndarray, linear: np.ndarray, n: int
    ) -> np.ndarray:
        """Flip the single bit that gives the largest energy decrease."""
        deltas = np.array(
            [LocalSearch._calc_e_diff(state, quad, linear, i) for i in range(n)]
        )
        best = int(np.argmin(deltas))
        if deltas[best] < 0:
            state[best] = -state[best]
        return state

    def _to_sampleset(self, spin_state: np.ndarray) -> BinarySampleSet:
        """Convert final SPIN state to a BinarySampleSet in the original vartype."""
        result_values = self._from_spin(spin_state)
        energy = self._model.calc_energy(result_values)
        sample = {
            self._model.index_new_to_origin[i]: v for i, v in enumerate(result_values)
        }
        return BinarySampleSet(
            samples=[sample],
            num_occurrences=[1],
            energy=[energy],
            vartype=self._model.vartype,
        )
