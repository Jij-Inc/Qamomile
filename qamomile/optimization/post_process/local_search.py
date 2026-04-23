"""Local search post-processing for binary optimization models.

Provides first-improvement and best-improvement local search algorithms
that operate on :class:`BinaryModel` instances (both SPIN and BINARY).
"""

from __future__ import annotations

import enum
from typing import Protocol

import numpy as np
import ommx.v1

from qamomile.optimization.binary_model.expr import VarType
from qamomile.optimization.binary_model.model import BinaryModel
from qamomile.optimization.binary_model.sampleset import BinarySampleSet


class LocalSearchMethod(enum.StrEnum):
    """Strategy for picking which bit to flip at each local-search step.

    Both variants only accept strictly improving flips; the difference is
    in how the next flip is chosen.

    Attributes:
        BEST: Flip the single bit with the largest energy decrease.
        FIRST: Flip the first bit (in index order) whose flip lowers energy.
    """

    BEST = "best"
    FIRST = "first"


class _StepFn(Protocol):
    """Callable protocol for a single local-search step.

    A step mutates *state* in place if an improving flip is found and
    returns ``True``; otherwise it leaves *state* unchanged and returns
    ``False``. The boolean return lets the driver detect convergence
    without copying *state*.
    """

    def __call__(self, state: np.ndarray) -> bool: ...


class LocalSearch:
    """Local search **energy minimizer** for :class:`BinaryModel`.

    Greedily flips single bits that strictly lower the model's energy
    (``calc_energy``), so **lower is better** — the search terminates at a
    local minimum where no single-bit flip further decreases energy.

    Supports arbitrary-order Ising / HUBO models. Internally converts the
    model to SPIN representation (±1) for the search, then converts the
    result back to the original vartype.

    Accepts either a :class:`BinaryModel` directly or an
    :class:`ommx.v1.Instance`, matching the other optimization converters.
    An ommx instance is lowered to a BINARY model via
    :meth:`ommx.v1.Instance.to_hubo`, preserving higher-order terms so
    that HUBO objectives survive the handoff.

    Args:
        model: The problem to minimize. Either a :class:`BinaryModel`
            (any vartype, any order) or an :class:`ommx.v1.Instance`.

    Raises:
        TypeError: If *model* is neither a BinaryModel nor an
            ommx.v1.Instance.
    """

    def __init__(self, model: ommx.v1.Instance | BinaryModel) -> None:
        if isinstance(model, ommx.v1.Instance):
            hubo, constant = model.to_hubo()
            model = BinaryModel.from_hubo(hubo, constant=constant)
        elif not isinstance(model, BinaryModel):
            raise TypeError(
                "model must be an ommx.v1.Instance or BinaryModel, "
                f"got {type(model).__name__}."
            )

        self._model = model
        self._spin_model = (
            model.change_vartype(VarType.SPIN)
            if model.vartype != VarType.SPIN
            else model
        )

        n = self._spin_model.num_bits

        # For each index k, record (co_indices, coeff) for every Hamiltonian
        # term that contains k. co_indices excludes k itself (empty tuple for
        # linear terms). This unifies linear, quadratic, and higher-order
        # terms under one sparse per-index adjacency, so the energy-delta
        #     ΔE_k = -2 · s_k · Σ_{T ∋ k} coeff_T · ∏_{i∈T, i≠k} s_i
        # works at arbitrary order.
        self._terms: dict[int, list[tuple[tuple[int, ...], float]]] = {
            i: [] for i in range(n)
        }
        for i, coeff in self._spin_model.linear.items():
            self._terms[i].append(((), coeff))
        for (i, j), coeff in self._spin_model.quad.items():
            self._terms[i].append(((j,), coeff))
            self._terms[j].append(((i,), coeff))
        for inds, coeff in self._spin_model.higher.items():
            for k in inds:
                co = tuple(x for x in inds if x != k)
                self._terms[k].append((co, coeff))

    def run(
        self,
        initial_state: list[int],
        max_iter: int = -1,
        method: str | LocalSearchMethod = LocalSearchMethod.BEST,
    ) -> BinarySampleSet:
        """Run local search starting from *initial_state*.

        Minimizes the model's energy: only strictly energy-decreasing flips
        are accepted, so the returned state is a local minimum (**lower is
        better**), not necessarily the global optimum.

        Args:
            initial_state: Variable values in the model's vartype domain
                (±1 for SPIN, 0/1 for BINARY).
            max_iter: Maximum iterations (-1 for unlimited).
            method: Flip-selection strategy. Either a :class:`LocalSearchMethod`
                member or its string value (``"best"`` / ``"first"``).

        Returns:
            A :class:`BinarySampleSet` containing the energy-minimized state.

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

        try:
            method = LocalSearchMethod(method)
        except ValueError as e:
            valid = [m.value for m in LocalSearchMethod]
            raise ValueError(f"Invalid method: {method!r}. Choose from {valid}.") from e

        match method:
            case LocalSearchMethod.BEST:
                step_fn: _StepFn = self._best_improvement
            case LocalSearchMethod.FIRST:
                step_fn = self._first_improvement
            case _:
                assert False, "unreachable"

        spin_state = self._to_spin(np.asarray(initial_state))
        spin_state = self._search(step_fn, spin_state, max_iter)
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
        step_fn: _StepFn,
        state: np.ndarray,
        max_iter: int,
    ) -> np.ndarray:
        """Run the local search loop until convergence or max_iter.

        The step function mutates *state* in place and reports whether a
        flip occurred via its boolean return, avoiding a per-iteration
        array copy for the convergence check.
        """
        counter = 0
        while max_iter == -1 or counter < max_iter:
            flipped = step_fn(state)
            if not flipped:
                break
            counter += 1
        return state

    @staticmethod
    def _calc_e_diff(
        state: np.ndarray,
        terms: dict[int, list[tuple[tuple[int, ...], float]]],
        idx: int,
    ) -> float:
        """Calculate the energy difference when flipping bit *idx*.

        Uses the sparse per-index term list so cost is O(deg(idx)) per
        flip, where deg(idx) is the number of Hamiltonian terms that
        contain idx. Handles arbitrary order (linear / quadratic / HUBO)
        uniformly: for a term ``T ∋ idx`` with coefficient ``c`` and
        co-indices ``T \\ {idx}``, its contribution to ΔE is
        ``-2 · s_idx · c · ∏_{j ∈ co-indices} s_j``.
        """
        total = 0.0
        for co_indices, coeff in terms[idx]:
            prod = coeff
            for j in co_indices:
                prod *= state[j]
            total += prod
        return float(-2 * state[idx] * total)

    def _first_improvement(self, state: np.ndarray) -> bool:
        """Flip the first bit whose flip lowers energy.

        Scans bits in index order, flips the first one with a negative
        energy delta in place, and reports ``True``. If no flip improves
        energy, *state* is left unchanged and ``False`` is returned.
        """
        for i in range(len(state)):
            if self._calc_e_diff(state, self._terms, i) < 0:
                state[i] = -state[i]
                return True
        return False

    def _best_improvement(self, state: np.ndarray) -> bool:
        """Flip the single bit giving the largest energy decrease.

        Mutates *state* in place and returns ``True`` if a flip was
        applied; returns ``False`` (leaving *state* unchanged) when no
        single-bit flip lowers energy, including the empty-state case.
        """
        n = len(state)
        if n == 0:
            return False
        deltas = [self._calc_e_diff(state, self._terms, i) for i in range(n)]
        best = int(np.argmin(deltas))
        if deltas[best] < 0:
            state[best] = -state[best]
            return True
        return False

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
