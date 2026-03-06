from __future__ import annotations
from typing import Generic
import warnings

import numpy as np

from qamomile.circuit.transpiler.job import SampleResult
from qamomile.optimization.utils import is_close_zero
from .expr import BinaryExpr, VarType, VT
from .normalize import normalize_by_factor, normalize_by_abs_max, normalize_by_rms
from .sampleset import BinarySampleSet


class BinaryModel(Generic[VT]):
    def __init__(self, expr: BinaryExpr[VT]) -> None:
        self._expr = expr

        # Bidirectional index mapping
        index_new_to_origin: dict[int, int] = {}
        index_origin_to_new: dict[int, int] = {}
        unique_indices = set([i for inds in expr.coefficients.keys() for i in inds])
        for new_i, old_i in enumerate(sorted(unique_indices)):
            index_new_to_origin[new_i] = old_i
            index_origin_to_new[old_i] = new_i
        self.index_new_to_origin = index_new_to_origin
        self.index_origin_to_new = index_origin_to_new

        # internal coefficients with new indices
        self._linear: dict[int, float] = {}
        self._quad: dict[tuple[int, int], float] = {}
        self._higher: dict[tuple[int, ...], float] = {}
        self.constant: float = 0.0
        self.order = 0
        self._update_internal_coefficients()

    @property
    def num_bits(self) -> int:
        return len(self.index_new_to_origin)

    @property
    def vartype(self) -> VT:
        return self._expr.vartype

    def _update_internal_coefficients(self) -> None:
        self._linear = {}
        self._quad = {}
        self._higher = {}

        constant = self._expr.constant

        for inds, coeff in self._expr.coefficients.items():
            if is_close_zero(coeff):
                continue

            if len(set(inds)) != len(inds):
                normalized = tuple(sorted(set(inds)))
                warnings.warn(
                    "Duplicate variable indices in term "
                    f"{inds} were normalized to {normalized}.",
                    UserWarning,
                    stacklevel=4,
                )
                inds = normalized

            if len(inds) == 0:
                constant += coeff
                continue

            # inds contains original indices, map to new sequential indices
            new_inds = tuple(sorted(self.index_origin_to_new[i] for i in inds))
            if len(new_inds) == 1:
                idx = new_inds[0]
                self._linear[idx] = self._linear.get(idx, 0.0) + coeff
            elif len(new_inds) == 2:
                self._quad[new_inds] = self._quad.get(new_inds, 0.0) + coeff
            else:
                self._higher[new_inds] = self._higher.get(new_inds, 0.0) + coeff

        self.constant = constant
        if self._higher:
            self.order = max(len(inds) for inds in self._higher)
        elif self._quad:
            self.order = 2
        elif self._linear:
            self.order = 1
        else:
            self.order = 0

    @classmethod
    def from_qubo(
        cls,
        qubo: dict[tuple[int, int], float],
        constant: float = 0.0,
        simplify: bool = False,
    ) -> "BinaryModel":
        expr = BinaryExpr(vartype=VarType.BINARY, constant=constant, coefficients={})
        for (i, j), coeff in qubo.items():
            if i == j:
                expr.coefficients[(i,)] = expr.coefficients.get((i,), 0.0) + coeff
            else:
                key = tuple(sorted((i, j)))
                expr.coefficients[key] = expr.coefficients.get(key, 0.0) + coeff
        if simplify:
            expr.coefficients = {
                k: v for k, v in expr.coefficients.items() if not is_close_zero(v)
            }
        return cls(expr)  # type: ignore

    @classmethod
    def from_ising(
        cls,
        linear: dict[int, float],
        quad: dict[tuple[int, int], float],
        constant: float = 0.0,
        simplify: bool = False,
    ) -> "BinaryModel":
        expr = BinaryExpr(vartype=VarType.SPIN, constant=constant, coefficients={})
        for i, coeff in linear.items():
            expr.coefficients[(i,)] = expr.coefficients.get((i,), 0.0) + coeff
        for (i, j), coeff in quad.items():
            key = tuple(sorted((i, j)))
            expr.coefficients[key] = expr.coefficients.get(key, 0.0) + coeff
        if simplify:
            expr.coefficients = {
                k: v for k, v in expr.coefficients.items() if not is_close_zero(v)
            }
        return cls(expr)  # type: ignore

    @classmethod
    def from_hubo(
        cls,
        hubo: dict[tuple[int, ...], float],
        constant: float = 0.0,
        simplify: bool = False,
    ) -> "BinaryModel":
        """Create a BINARY BinaryModel from HUBO coefficients.

        Args:
            hubo: HUBO coefficients mapping index tuples to values.
                  Index tuples are sorted and deduplicated.
                  Duplicate indices in a single term (e.g., ``(0, 0, 2)``) emit
                  a warning and are normalized to unique indices (``(0, 2)``).
                  Empty tuples (``()``) are accumulated into the constant term.
            constant: Constant offset term.
            simplify: If True, remove near-zero coefficients.

        Returns:
            BinaryModel with BINARY vartype.
        """
        expr = BinaryExpr(vartype=VarType.BINARY, constant=constant, coefficients={})
        for indices, coeff in hubo.items():
            key = tuple(sorted(set(indices)))
            if len(key) != len(indices):
                warnings.warn(
                    "Duplicate variable indices in HUBO term "
                    f"{indices} were normalized to {key}.",
                    UserWarning,
                    stacklevel=2,
                )
            if len(key) == 0:
                expr.constant += coeff
                continue
            expr.coefficients[key] = expr.coefficients.get(key, 0.0) + coeff
        if simplify:
            expr.coefficients = {
                k: v for k, v in expr.coefficients.items() if not is_close_zero(v)
            }
        return cls(expr)  # type: ignore

    @property
    def linear(self) -> dict[int, float]:
        return self._linear

    @property
    def quad(self) -> dict[tuple[int, int], float]:
        return self._quad

    @property
    def higher(self) -> dict[tuple[int, ...], float]:
        return self._higher

    @property
    def coefficients(self) -> dict[tuple[int, ...], float]:
        """All coefficients as a single flat dictionary using sequential indices."""
        coeffs: dict[tuple[int, ...], float] = {}

        for i, coeff in self._linear.items():
            if not is_close_zero(coeff):
                coeffs[(i,)] = coeff
        for inds, coeff in self._quad.items():
            if not is_close_zero(coeff):
                coeffs[inds] = coeff
        for inds, coeff in self._higher.items():
            if not is_close_zero(coeff):
                coeffs[inds] = coeff

        return coeffs

    def calc_energy(self, state: list[int]) -> float:
        """Calculate the energy for a given variable assignment.

        Args:
            state: Variable values indexed by sequential (zero-origin) indices.
                   For SPIN: values must be +1 or -1.
                   For BINARY: values must be 0 or 1.

        Returns:
            The energy value.

        Raises:
            ValueError: If state values are invalid for the vartype.
        """
        if self.vartype == VarType.SPIN:
            if not np.allclose(np.abs(state), 1.0):
                raise ValueError(
                    "All elements must be close to +1 or -1 for SPIN vartype."
                )
        elif self.vartype == VarType.BINARY:
            if not all(v in (0, 1) for v in state):
                raise ValueError("All elements must be 0 or 1 for BINARY vartype.")

        energy = self.constant
        for inds, coeff in self.coefficients.items():
            term = coeff
            for idx in inds:
                term *= state[idx]
            energy += term
        return energy

    def change_vartype(self, vartype: VarType) -> "BinaryModel":
        if self._expr.vartype == vartype:
            return BinaryModel(self._expr.copy())

        # Carry over the original constant — it's vartype-independent.
        new_expr = BinaryExpr(
            vartype=vartype, constant=self._expr.constant, coefficients={}
        )

        # binary spin corerspondence is based on Z|0> = |0>, Z|1> = -|1>
        # that means
        # binary | spin
        #   0    |  1
        #   1    | -1
        if vartype == VarType.SPIN:
            # BINARY to SPIN: x = (1 - s) / 2
            for inds, coeff in self._expr.coefficients.items():
                # For efficiency, we handle up to quadratic terms directly
                # Note: This branch is currently dead code because BinaryExpr
                # stores constant terms in `constant`, not as coefficients[()].
                # Kept as a defensive guard for correctness.
                if len(inds) == 0:
                    term = BinaryExpr(
                        vartype=VarType.SPIN, constant=coeff, coefficients={}
                    )
                elif len(inds) == 1:
                    i = inds[0]
                    term = BinaryExpr(
                        vartype=VarType.SPIN,
                        constant=coeff * 0.5,
                        coefficients={(i,): -coeff * 0.5},
                    )
                elif len(inds) == 2:
                    i, j = inds
                    _coeff = coeff * 0.25
                    term = BinaryExpr(
                        vartype=VarType.SPIN,
                        constant=_coeff,
                        coefficients={(i,): -_coeff, (j,): -_coeff, (i, j): _coeff},
                    )
                else:
                    # For higher order terms, expand coeff * prod((1 - s_i) / 2)
                    term = BinaryExpr(
                        vartype=VarType.SPIN, constant=coeff, coefficients={}
                    )
                    for i in inds:
                        single_term = BinaryExpr(
                            vartype=VarType.SPIN,
                            constant=0.5,
                            coefficients={(i,): -0.5},
                        )
                        term *= single_term
                new_expr += term
        elif vartype == VarType.BINARY:
            # SPIN to BINARY: s = 1 - 2x
            for inds, coeff in self._expr.coefficients.items():
                # For efficiency, we handle up to quadratic terms directly
                # Note: This branch is currently dead code because BinaryExpr
                # stores constant terms in `constant`, not as coefficients[()].
                # Kept as a defensive guard for correctness.
                if len(inds) == 0:
                    term = BinaryExpr(
                        vartype=VarType.BINARY, constant=coeff, coefficients={}
                    )
                elif len(inds) == 1:
                    i = inds[0]
                    term = BinaryExpr(
                        vartype=VarType.BINARY,
                        constant=coeff,
                        coefficients={(i,): -2 * coeff},
                    )
                elif len(inds) == 2:
                    i, j = inds
                    _coeff = coeff
                    term = BinaryExpr(
                        vartype=VarType.BINARY,
                        constant=_coeff,
                        coefficients={
                            (i,): -2 * _coeff,
                            (j,): -2 * _coeff,
                            (i, j): 4 * _coeff,
                        },
                    )
                else:
                    # For higher order terms, expand coeff * prod(1 - 2*x_i)
                    term = BinaryExpr(
                        vartype=VarType.BINARY, constant=coeff, coefficients={}
                    )
                    for i in inds:
                        single_term = BinaryExpr(
                            vartype=VarType.BINARY,
                            constant=1.0,
                            coefficients={(i,): -2.0},
                        )
                        term *= single_term
                new_expr += term

        else:
            raise ValueError("Unsupported vartype conversion.")

        return BinaryModel(new_expr)

    def normalize_by_factor(
        self, factor: float, replace: bool = False
    ) -> BinaryModel[VT]:
        """Normalize the BinaryModel by a given factor.

        Args:
            factor (float): The normalization factor.

        Returns:
            BinaryModel[VT]: The normalized binary model.
        """
        normalized_expr = normalize_by_factor(self._expr, factor, replace=replace)
        if replace:
            self._expr = normalized_expr
            self._update_internal_coefficients()
            return self
        return BinaryModel(normalized_expr)

    def normalize_by_abs_max(self, replace: bool = False) -> BinaryModel[VT]:
        """Normalize the BinaryModel by its absolute maximum coefficient.

        Returns:
            BinaryModel[VT]: The normalized binary model.
        """
        normalized_expr = normalize_by_abs_max(self._expr, replace=replace)
        if replace:
            self._expr = normalized_expr
            self._update_internal_coefficients()
            return self
        return BinaryModel(normalized_expr)

    def normalize_by_rms(self, replace: bool = False) -> BinaryModel[VT]:
        """Normalize the BinaryModel by its root mean square.

        Returns:
            BinaryModel[VT]: The normalized binary model.
        """
        normalized_expr = normalize_by_rms(self._expr, replace=replace)
        if replace:
            self._expr = normalized_expr
            self._update_internal_coefficients()
            return self
        return BinaryModel(normalized_expr)

    def decode_from_sampleresult(
        self, result: SampleResult[list[int]]
    ) -> BinarySampleSet[VT]:
        """Decode quantum measurement results into samples with energies.

        Converts raw measurement bitstrings (0/1 from quantum hardware) into
        the model's variable domain and calculates energies.

        Args:
            result: Measurement results with sequential bit indices

        Returns:
            BinarySampleSet with samples in expression domain, using original indices

        Note:
            For SPIN models, conversion follows: measurement 0 → +1, measurement 1 → -1
            This matches the quantum convention Z|0⟩ = |0⟩, Z|1⟩ = -|1⟩
        """
        samples: list[dict[int, int]] = []
        num_occurrences: list[int] = []
        energies: list[float] = []

        for sample, n_occ in result.results:
            # Validate sample length
            if len(sample) != self.num_bits:
                raise ValueError(
                    f"Sample length {len(sample)} does not match model size {self.num_bits}"
                )

            # Step 1: Convert measurement bits to expression domain values
            # Measurements are always 0/1, but SPIN expressions need ±1
            expr_values: dict[int, int] = {}
            for new_idx, measurement_bit in enumerate(sample):
                if self._expr.vartype == VarType.SPIN:
                    # SPIN: 0 → +1, 1 → -1
                    expr_values[new_idx] = 1 - 2 * measurement_bit
                else:
                    # BINARY: 0 → 0, 1 → 1 (no conversion)
                    expr_values[new_idx] = measurement_bit

            # Step 2: Calculate energy using expression coefficients
            energy = self._expr.constant
            for inds, coeff in self._expr.coefficients.items():
                prod = 1
                for orig_idx in inds:
                    # Map original index to sequential index
                    new_idx = self.index_origin_to_new[orig_idx]
                    prod *= expr_values[new_idx]
                energy += coeff * prod

            # Step 3: Create output sample using original indices
            # Output values stay in expression domain
            output_sample = {
                self.index_new_to_origin[new_idx]: expr_values[new_idx]
                for new_idx in range(self.num_bits)
            }

            samples.append(output_sample)
            num_occurrences.append(n_occ)
            energies.append(energy)

        return BinarySampleSet(
            samples=samples,
            num_occurrences=num_occurrences,
            energy=energies,
            vartype=self.vartype,
        )
