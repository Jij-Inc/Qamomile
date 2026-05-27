"""Integer-to-binary encoding for constrained integer optimization.

Provides the :class:`BinaryIntegerEncoder` class which transforms
constrained integer optimization problems with linear equality
constraints into an equivalent binary instance.  The total
right-hand-side value of the encoded constraints is returned alongside
the binary instance so that downstream solvers can exploit the
constraint structure (e.g. particle-number conservation in FQAOA).
"""

from __future__ import annotations

import enum
import math
from collections import defaultdict

import ommx.v1


class IntegerEncodingMethod(enum.StrEnum):
    """Encoding scheme for integer-to-binary variable conversion.

    Determines how an integer variable ``z_l`` with range
    ``[lower, upper]`` is decomposed into binary decision variables.

    Attributes:
        UNARY: Unary (thermometer) encoding.  An integer ``z_l`` in
            ``[lower, upper]`` is encoded as
            ``z_l = lower + sum_{d=0}^{D-1} x_{l,d}`` where
            ``D = upper - lower``, requiring *D* binary variables.
    """

    UNARY = "unary"


class BinaryIntegerEncoder:
    """Encode integer decision variables as binary variables.

    Transforms an :class:`ommx.v1.Instance` containing integer decision
    variables with linear equality constraints into an equivalent binary
    instance.  The sum of the adjusted right-hand-side values across all
    encoded constraints (``constraint_rhs_total``) is returned together
    with the binary instance, enabling downstream algorithms to exploit
    the constraint structure directly.

    The encoding pipeline:

    1. Validate that all constraints are linear equalities.
    2. Validate that all decision variables are integer-typed with finite
       bounds.
    3. Replace each integer variable ``z_l`` with binary variables
       ``x_{l,d}`` using the chosen encoding scheme.
    4. Substitute the encoding into the objective and constraints.
    5. Derive ``constraint_rhs_total`` from the encoded constraints.

    Args:
        instance (ommx.v1.Instance): The integer optimization problem.
            All decision variables must be integer-typed with finite
            bounds.  All constraints must be linear equalities.
        encoding (str | IntegerEncodingMethod): The encoding scheme to
            use.  Currently only ``"unary"`` is supported.  Defaults to
            ``"unary"``.

    Raises:
        ValueError: If any constraint is not a linear equality, if any
            decision variable is not integer-typed with finite bounds, or
            if the encoding scheme is not recognized.

    Example:
        >>> encoder = BinaryIntegerEncoder(integer_instance, encoding="unary")
        >>> binary_instance, constraint_rhs_total = encoder.encode()
    """

    def __init__(
        self,
        instance: ommx.v1.Instance,
        encoding: str | IntegerEncodingMethod = IntegerEncodingMethod.UNARY,
    ) -> None:
        """Initialize the encoder.

        Args:
            instance (ommx.v1.Instance): The integer optimization problem.
            encoding (str | IntegerEncodingMethod): The encoding scheme.
                Defaults to ``"unary"``.

        Raises:
            ValueError: If validation of constraints or decision variables
                fails.
        """
        self._instance = ommx.v1.Instance.from_bytes(instance.to_bytes())
        self._encoding = IntegerEncodingMethod(encoding)
        self._var_info: dict[int, tuple[int, int]] = {}
        self._validate()

    @property
    def encoding(self) -> IntegerEncodingMethod:
        """Return the encoding scheme.

        Returns:
            IntegerEncodingMethod: The encoding scheme used by this encoder.
        """
        return self._encoding

    @property
    def instance(self) -> ommx.v1.Instance:
        """Return the deep-copied original instance.

        Returns:
            ommx.v1.Instance: A deep copy of the instance passed at
                construction time.
        """
        return self._instance

    def _validate(self) -> None:
        """Run all validation checks.

        Raises:
            ValueError: If any constraint or decision variable is invalid.
        """
        self._validate_constraints()
        self._validate_decision_variables()

    def _validate_constraints(self) -> None:
        """Validate that all constraints are linear equalities.

        Raises:
            ValueError: If any constraint is a non-equality or non-linear.
        """
        for constraint in self._instance.constraints:
            if constraint.equality != ommx.v1.Constraint.EQUAL_TO_ZERO:
                name = constraint.name or constraint.id
                raise ValueError(
                    f"Constraint '{name}' is not an equality constraint. "
                    "BinaryIntegerEncoder requires all constraints to be "
                    "equalities (== 0)."
                )
            if constraint.function.degree() > 1:
                name = constraint.name or constraint.id
                raise ValueError(
                    f"Constraint '{name}' is not linear "
                    f"(degree {constraint.function.degree()}). "
                    "BinaryIntegerEncoder requires all constraints to be "
                    "linear."
                )

    def _validate_decision_variables(self) -> None:
        """Validate that all variables are integer with finite bounds.

        Caches ``(lower, upper)`` bounds in :attr:`_var_info` as a
        side-effect.

        Raises:
            ValueError: If any decision variable is not integer-typed or
                has non-finite bounds.
        """
        for dv in self._instance.decision_variables:
            if dv.kind != ommx.v1.DecisionVariable.INTEGER:
                name = dv.name or dv.id
                raise ValueError(
                    f"Decision variable '{name}' has kind '{dv.kind}', "
                    "expected INTEGER. BinaryIntegerEncoder requires all "
                    "decision variables to be integer-typed."
                )
            lower = dv.bound.lower
            upper = dv.bound.upper
            if not (math.isfinite(lower) and math.isfinite(upper)):
                name = dv.name or dv.id
                raise ValueError(
                    f"Decision variable '{name}' has non-finite bounds "
                    f"[{lower}, {upper}]. BinaryIntegerEncoder requires "
                    "finite bounds on all decision variables."
                )
            self._var_info[dv.id] = (int(lower), int(upper))

    def encode(self) -> tuple[ommx.v1.Instance, int]:
        """Encode integer variables into binary variables.

        Dispatches to the encoding strategy specified at construction
        time.

        Returns:
            tuple[ommx.v1.Instance, int]: A tuple of the binary-encoded
                instance and the total right-hand-side value of the
                encoded equality constraints.

        Raises:
            ValueError: If the derived ``constraint_rhs_total`` is not a
                non-negative integer.
        """
        match self._encoding:
            case IntegerEncodingMethod.UNARY:
                return self._encode_unary()

    # ------------------------------------------------------------------
    # Unary encoding
    # ------------------------------------------------------------------

    def _encode_unary(self) -> tuple[ommx.v1.Instance, int]:
        """Perform unary encoding.

        For each integer ``z_l`` in ``[lower_l, upper_l]``, introduces
        ``D_l = upper_l - lower_l`` binary variables ``x_{l,d}`` such that
        ``z_l = lower_l + sum_{d=0}^{D_l-1} x_{l,d}``.

        Returns:
            tuple[ommx.v1.Instance, int]: The binary instance and
                constraint_rhs_total.

        Raises:
            ValueError: If the objective degree exceeds 2, or if the
                derived ``constraint_rhs_total`` is not a non-negative
                integer.
        """
        binary_dvs, binary_vars_for, id_to_dv = self._create_binary_variables_unary()

        new_obj = self._substitute_expression(
            self._instance.objective, binary_vars_for, id_to_dv
        )

        new_constraints, constraint_rhs_total = self._substitute_constraints_unary(
            binary_vars_for, id_to_dv
        )

        if not float(constraint_rhs_total).is_integer() or constraint_rhs_total < 0:
            raise ValueError(
                f"Derived constraint_rhs_total ({constraint_rhs_total}) is "
                "not a non-negative integer. The constraint structure may "
                "be incompatible with the encoding."
            )

        new_instance = ommx.v1.Instance.from_components(
            decision_variables=binary_dvs,
            objective=new_obj,
            constraints=new_constraints,
            sense=self._instance.sense,
        )
        return new_instance, int(constraint_rhs_total)

    def _create_binary_variables_unary(
        self,
    ) -> tuple[
        list[ommx.v1.DecisionVariable],
        dict[int, list[ommx.v1.DecisionVariable]],
        dict[int, ommx.v1.DecisionVariable],
    ]:
        """Create binary decision variables for unary encoding.

        Returns:
            tuple[list[ommx.v1.DecisionVariable], dict[int, list[ommx.v1.DecisionVariable]], dict[int, ommx.v1.DecisionVariable]]:
                A tuple of ``(binary_dvs, binary_vars_for, id_to_dv)``
                where *binary_dvs* is the flat list of all binary
                variables, *binary_vars_for* maps each original integer
                variable ID to its list of binary variables, and
                *id_to_dv* maps each new binary variable ID to the
                corresponding :class:`ommx.v1.DecisionVariable`.
        """
        binary_dvs: list[ommx.v1.DecisionVariable] = []
        binary_vars_for: dict[int, list[ommx.v1.DecisionVariable]] = {}
        id_to_dv: dict[int, ommx.v1.DecisionVariable] = {}
        next_id = 0

        for l_idx, orig_id in enumerate(sorted(self._var_info.keys())):
            lower, upper = self._var_info[orig_id]
            n_bits = upper - lower
            bits: list[ommx.v1.DecisionVariable] = []
            for d in range(n_bits):
                dv = ommx.v1.DecisionVariable.binary(
                    next_id,
                    name="x",
                    subscripts=[l_idx, d],
                )
                bits.append(dv)
                id_to_dv[next_id] = dv
                binary_dvs.append(dv)
                next_id += 1
            binary_vars_for[orig_id] = bits

        return binary_dvs, binary_vars_for, id_to_dv

    def _substitute_constraints_unary(
        self,
        binary_vars_for: dict[int, list[ommx.v1.DecisionVariable]],
        id_to_dv: dict[int, ommx.v1.DecisionVariable],
    ) -> tuple[list[ommx.v1.Constraint], float]:
        """Transform all constraints and compute total constraint RHS.

        Each linear equality ``sum_i a_i z_i + c = 0`` is rewritten
        over binary variables.  The adjusted RHS
        ``-(c + sum_i a_i * lower_i)`` of each constraint contributes
        to the total.

        Args:
            binary_vars_for (dict[int, list[ommx.v1.DecisionVariable]]):
                Mapping from original variable ID to binary variables.
            id_to_dv (dict[int, ommx.v1.DecisionVariable]): Mapping from
                binary variable ID to decision variable.

        Returns:
            tuple[list[ommx.v1.Constraint], float]: The transformed
                constraints and the total constraint RHS.
        """
        new_constraints: list[ommx.v1.Constraint] = []
        rhs_total = 0.0

        for c_idx, c in enumerate(self._instance.constraints):
            new_expr = self._substitute_expression(
                c.function, binary_vars_for, id_to_dv
            )
            new_c = (new_expr == 0).set_id(c_idx)
            if c.name:
                new_c = new_c.add_name(c.name)
            new_constraints.append(new_c)

            linear = c.function.as_linear()
            assert linear is not None
            adjusted_constant = float(linear.constant_term)
            for var_id, coeff in linear.linear_terms.items():
                adjusted_constant += coeff * self._var_info[var_id][0]
            rhs_total += -adjusted_constant

        return new_constraints, rhs_total

    # ------------------------------------------------------------------
    # Expression substitution
    # ------------------------------------------------------------------

    def _substitute_expression(
        self,
        func: ommx.v1.Function,
        binary_vars_for: dict[int, list[ommx.v1.DecisionVariable]],
        id_to_dv: dict[int, ommx.v1.DecisionVariable],
    ) -> float | ommx.v1.Linear | ommx.v1.Quadratic | ommx.v1.Polynomial:
        """Substitute integer variables in a function with binary encoding.

        Uses ``Function.terms`` to handle any polynomial degree in a
        single code path.  Each term ``c * z_{i1} * ... * z_{ik}`` is
        expanded via iterative partial-product multiplication against
        ``z_l = lower_l + sum x_{l,d}``.

        Args:
            func (ommx.v1.Function): The function to transform.
            binary_vars_for (dict[int, list[ommx.v1.DecisionVariable]]):
                Mapping from original variable ID to binary variables.
            id_to_dv (dict[int, ommx.v1.DecisionVariable]): Mapping from
                binary variable ID to decision variable.

        Returns:
            float | ommx.v1.Linear | ommx.v1.Quadratic | ommx.v1.Polynomial:
                The substituted expression.
        """
        expanded: dict[tuple[int, ...], float] = defaultdict(float)

        for var_ids, coeff in func.terms.items():
            partial = self._expand_term(var_ids, coeff, binary_vars_for)
            for key, val in partial.items():
                expanded[key] += val

        return self._build_expression(dict(expanded), id_to_dv)

    def _expand_term(
        self,
        var_ids: tuple[int, ...],
        coeff: float,
        binary_vars_for: dict[int, list[ommx.v1.DecisionVariable]],
    ) -> dict[tuple[int, ...], float]:
        """Expand a single polynomial term after substitution.

        For a term ``c * z_{i1} * ... * z_{ik}``, iteratively multiplies
        the running partial product by each factor
        ``(lower_{ij} + sum x_{ij,d})``.

        Args:
            var_ids (tuple[int, ...]): Original integer variable IDs in
                the monomial.  An empty tuple represents the constant
                term.
            coeff (float): Coefficient of the term.
            binary_vars_for (dict[int, list[ommx.v1.DecisionVariable]]):
                Mapping from original variable ID to binary variables.

        Returns:
            dict[tuple[int, ...], float]: Expanded terms keyed by tuples
                of binary variable IDs.
        """
        # Start with just the coefficient: {(): coeff}
        partial: dict[tuple[int, ...], float] = {(): coeff}

        for orig_id in var_ids:
            lower = self._var_info[orig_id][0]
            bits = binary_vars_for[orig_id]
            next_partial: dict[tuple[int, ...], float] = defaultdict(float)

            for key, val in partial.items():
                # Multiply by the constant part (lower)
                if lower != 0:
                    next_partial[key] += val * lower
                # Multiply by each binary variable x_{l,d}
                for dv in bits:
                    new_key = tuple(sorted(key + (dv.id,)))
                    next_partial[new_key] += val

            partial = dict(next_partial)

        return partial

    @staticmethod
    def _build_expression(
        expanded: dict[tuple[int, ...], float],
        id_to_dv: dict[int, ommx.v1.DecisionVariable],
    ) -> float | ommx.v1.Linear | ommx.v1.Quadratic | ommx.v1.Polynomial:
        """Build an ommx expression from expanded coefficient map.

        Constructs the expression incrementally using ommx arithmetic
        on :class:`ommx.v1.DecisionVariable` objects.

        Args:
            expanded (dict[tuple[int, ...], float]): Mapping from tuples
                of binary variable IDs to coefficients.  An empty tuple
                key represents the constant term.
            id_to_dv (dict[int, ommx.v1.DecisionVariable]): Mapping from
                binary variable ID to decision variable.

        Returns:
            float | ommx.v1.Linear | ommx.v1.Quadratic | ommx.v1.Polynomial:
                The built expression.
        """
        constant = expanded.pop((), 0.0)
        if not expanded:
            return constant

        expr: float | ommx.v1.Linear | ommx.v1.Quadratic | ommx.v1.Polynomial
        expr = constant

        for var_ids, coeff in expanded.items():
            term = coeff
            for vid in var_ids:
                term = term * id_to_dv[vid]
            expr = expr + term

        return expr
