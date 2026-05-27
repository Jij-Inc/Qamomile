"""Tests for BinaryIntegerEncoder and IntegerEncodingMethod."""

from __future__ import annotations

import numpy as np
import ommx.v1
import pytest

from qamomile.optimization.integer_encoding import (
    BinaryIntegerEncoder,
    IntegerEncodingMethod,
)


# =====================================================================
# Helpers
# =====================================================================


def _make_instance(
    dvs: list[ommx.v1.DecisionVariable],
    objective,
    constraints: list | None = None,
    sense=ommx.v1.Instance.MINIMIZE,
) -> ommx.v1.Instance:
    """Shorthand for creating an ommx Instance."""
    return ommx.v1.Instance.from_components(
        decision_variables=dvs,
        objective=objective,
        constraints=constraints or [],
        sense=sense,
    )


# =====================================================================
# 1. Validation tests
# =====================================================================


class TestValidation:
    """Validate that __init__ rejects invalid inputs."""

    def test_reject_inequality_constraint(self):
        """Non-equality constraint raises ValueError."""
        z = ommx.v1.DecisionVariable.integer(0, lower=0, upper=3)
        instance = _make_instance([z], z, [(z <= 2).set_id(0)])
        with pytest.raises(ValueError, match="not an equality constraint"):
            BinaryIntegerEncoder(instance)

    def test_reject_nonlinear_constraint(self):
        """Quadratic equality constraint raises ValueError."""
        z0 = ommx.v1.DecisionVariable.integer(0, lower=0, upper=3)
        z1 = ommx.v1.DecisionVariable.integer(1, lower=0, upper=3)
        instance = _make_instance([z0, z1], z0 + z1, [(z0 * z1 == 0).set_id(0)])
        with pytest.raises(ValueError, match="not linear"):
            BinaryIntegerEncoder(instance)

    def test_reject_binary_variable(self):
        """Binary (non-integer) decision variable raises ValueError."""
        x = ommx.v1.DecisionVariable.binary(0, name="x")
        instance = _make_instance([x], x)
        with pytest.raises(ValueError, match="expected INTEGER"):
            BinaryIntegerEncoder(instance)

    def test_reject_invalid_encoding(self):
        """Unrecognized encoding string raises ValueError."""
        z = ommx.v1.DecisionVariable.integer(0, lower=0, upper=2)
        instance = _make_instance([z], z)
        with pytest.raises(ValueError):
            BinaryIntegerEncoder(instance, encoding="onehot")


# =====================================================================
# 2. Properties
# =====================================================================


class TestProperties:
    """Verify read-only properties and deep-copy semantics."""

    def test_encoding_property(self):
        """The encoding property returns the method passed at init."""
        z = ommx.v1.DecisionVariable.integer(0, lower=0, upper=2)
        instance = _make_instance([z], z)
        enc = BinaryIntegerEncoder(instance, encoding="unary")
        assert enc.encoding == IntegerEncodingMethod.UNARY

    def test_caller_instance_not_mutated(self):
        """The original instance must not be modified by encode()."""
        z = ommx.v1.DecisionVariable.integer(0, lower=0, upper=3)
        instance = _make_instance([z], z, [(z == 2).set_id(0)])
        original_bytes = instance.to_bytes()

        enc = BinaryIntegerEncoder(instance)
        enc.encode()

        assert instance.to_bytes() == original_bytes


# =====================================================================
# 3. Unary encoding — binary variable generation
# =====================================================================


class TestBinaryVariableGeneration:
    """Verify the structure of generated binary variables."""

    def test_variable_count(self):
        """Number of binary variables = sum(upper - lower) for each integer."""
        z0 = ommx.v1.DecisionVariable.integer(0, lower=0, upper=3)
        z1 = ommx.v1.DecisionVariable.integer(1, lower=0, upper=2)
        instance = _make_instance([z0, z1], z0 + z1)
        binary_inst, _ = BinaryIntegerEncoder(instance).encode()
        assert len(binary_inst.decision_variables) == 3 + 2

    def test_subscript_structure(self):
        """Each binary variable has subscripts [l_idx, d]."""
        z0 = ommx.v1.DecisionVariable.integer(0, lower=0, upper=2)
        z1 = ommx.v1.DecisionVariable.integer(1, lower=0, upper=3)
        instance = _make_instance([z0, z1], z0 + z1)
        binary_inst, _ = BinaryIntegerEncoder(instance).encode()

        subscripts = [dv.subscripts for dv in binary_inst.decision_variables]
        assert subscripts == [
            [0, 0],
            [0, 1],  # z0: D=2
            [1, 0],
            [1, 1],
            [1, 2],  # z1: D=3
        ]

    def test_all_variables_are_binary(self):
        """Every generated decision variable has kind=BINARY."""
        z = ommx.v1.DecisionVariable.integer(0, lower=0, upper=4)
        instance = _make_instance([z], z)
        binary_inst, _ = BinaryIntegerEncoder(instance).encode()
        for dv in binary_inst.decision_variables:
            assert dv.kind == ommx.v1.DecisionVariable.BINARY


# =====================================================================
# 4. Linear objective substitution
# =====================================================================


class TestLinearObjective:
    """Verify encoding correctness for linear objectives."""

    def test_simple_linear(self):
        """z0 + z1 with lower=0 becomes sum of all binary variables."""
        z0 = ommx.v1.DecisionVariable.integer(0, lower=0, upper=2)
        z1 = ommx.v1.DecisionVariable.integer(1, lower=0, upper=2)
        instance = _make_instance([z0, z1], z0 + z1)
        binary_inst, _ = BinaryIntegerEncoder(instance).encode()

        obj = binary_inst.objective.as_linear()
        assert obj is not None
        assert obj.constant_term == 0.0
        for coeff in obj.linear_terms.values():
            assert coeff == pytest.approx(1.0)

    def test_weighted_linear(self):
        """3*z0 + 2*z1 produces correct coefficients."""
        z0 = ommx.v1.DecisionVariable.integer(0, lower=0, upper=2)
        z1 = ommx.v1.DecisionVariable.integer(1, lower=0, upper=2)
        instance = _make_instance([z0, z1], 3 * z0 + 2 * z1)
        binary_inst, _ = BinaryIntegerEncoder(instance).encode()

        obj = binary_inst.objective.as_linear()
        assert obj is not None
        # z0 → x0, x1 (coeff 3); z1 → x2, x3 (coeff 2)
        assert obj.linear_terms[0] == pytest.approx(3.0)
        assert obj.linear_terms[1] == pytest.approx(3.0)
        assert obj.linear_terms[2] == pytest.approx(2.0)
        assert obj.linear_terms[3] == pytest.approx(2.0)

    def test_nonzero_lower_bound_constant_shift(self):
        """Non-zero lower bounds produce a constant offset in the objective."""
        z = ommx.v1.DecisionVariable.integer(0, lower=2, upper=5)
        instance = _make_instance([z], z)
        binary_inst, _ = BinaryIntegerEncoder(instance).encode()

        obj = binary_inst.objective.as_linear()
        assert obj is not None
        # z = 2 + x0 + x1 + x2 → objective = x0 + x1 + x2 + 2
        assert obj.constant_term == pytest.approx(2.0)
        assert len(obj.linear_terms) == 3


# =====================================================================
# 5. Quadratic objective substitution
# =====================================================================


class TestQuadraticObjective:
    """Verify encoding correctness for quadratic objectives."""

    def test_product_expansion(self):
        """z0*z1 with lower=0 expands to all binary cross-products."""
        z0 = ommx.v1.DecisionVariable.integer(0, lower=0, upper=2)
        z1 = ommx.v1.DecisionVariable.integer(1, lower=0, upper=2)
        instance = _make_instance([z0, z1], z0 * z1)
        binary_inst, _ = BinaryIntegerEncoder(instance).encode()

        obj_func = binary_inst.objective
        assert obj_func.degree() == 2
        quad = obj_func.as_quadratic()
        assert quad is not None
        # (x0+x1)*(x2+x3) → 4 quadratic terms
        assert len(quad.quadratic_terms) == 4

    def test_quadratic_with_nonzero_lower(self):
        """Quadratic substitution with non-zero lower produces all three components."""
        z0 = ommx.v1.DecisionVariable.integer(0, lower=1, upper=3)
        z1 = ommx.v1.DecisionVariable.integer(1, lower=1, upper=3)
        # z0*z1 = (1+x0+x1)(1+x2+x3)
        #       = 1 + (x2+x3) + (x0+x1) + cross-products
        instance = _make_instance([z0, z1], z0 * z1)
        binary_inst, _ = BinaryIntegerEncoder(instance).encode()

        quad = binary_inst.objective.as_quadratic()
        assert quad is not None
        assert quad.constant_term == pytest.approx(1.0)
        assert len(quad.linear_terms) == 4
        assert len(quad.quadratic_terms) == 4


# =====================================================================
# 5b. Higher-order objective substitution
# =====================================================================


class TestHigherOrderObjective:
    """Verify encoding correctness for objectives of degree > 2."""

    def test_cubic_term_expansion(self):
        """z0*z1*z2 with lower=0 produces correct degree-3 terms."""
        z0 = ommx.v1.DecisionVariable.integer(0, lower=0, upper=2)
        z1 = ommx.v1.DecisionVariable.integer(1, lower=0, upper=2)
        z2 = ommx.v1.DecisionVariable.integer(2, lower=0, upper=2)
        instance = _make_instance([z0, z1, z2], z0 * z1 * z2)
        binary_inst, _ = BinaryIntegerEncoder(instance).encode()

        obj = binary_inst.objective
        assert obj.degree() == 3
        # (x0+x1)(x2+x3)(x4+x5) → 2*2*2 = 8 cubic terms
        cubic_terms = {k: v for k, v in obj.terms.items() if len(k) == 3}
        assert len(cubic_terms) == 8

    def test_cubic_with_nonzero_lower(self):
        """Cubic with non-zero lower produces constant, linear, quad, cubic."""
        z0 = ommx.v1.DecisionVariable.integer(0, lower=1, upper=2)
        z1 = ommx.v1.DecisionVariable.integer(1, lower=1, upper=2)
        z2 = ommx.v1.DecisionVariable.integer(2, lower=1, upper=2)
        # z0*z1*z2 = (1+x0)(1+x1)(1+x2) = 1 + x0+x1+x2 + x0x1+x0x2+x1x2 + x0x1x2
        instance = _make_instance([z0, z1, z2], z0 * z1 * z2)
        binary_inst, _ = BinaryIntegerEncoder(instance).encode()

        terms = binary_inst.objective.terms
        constant = terms.get((), 0.0)
        linear = {k: v for k, v in terms.items() if len(k) == 1}
        quad = {k: v for k, v in terms.items() if len(k) == 2}
        cubic = {k: v for k, v in terms.items() if len(k) == 3}

        assert constant == pytest.approx(1.0)
        assert len(linear) == 3
        assert len(quad) == 3
        assert len(cubic) == 1

    @pytest.mark.parametrize("seed", [0, 1, 42])
    def test_cubic_objective_equivalence(self, seed):
        """Encoded cubic objective evaluates to same value as original."""
        rng = np.random.default_rng(seed)
        upper = 2
        dvs = [
            ommx.v1.DecisionVariable.integer(i, lower=0, upper=upper) for i in range(3)
        ]
        c = float(rng.uniform(-3, 3))
        objective = c * dvs[0] * dvs[1] * dvs[2]
        instance = _make_instance(dvs, objective)

        binary_inst, _ = BinaryIntegerEncoder(instance).encode()

        for _ in range(10):
            z_vals = rng.integers(0, upper + 1, size=3)
            original_value = c * z_vals[0] * z_vals[1] * z_vals[2]

            bin_state = {}
            idx = 0
            for i in range(3):
                for d in range(upper):
                    bin_state[idx] = 1.0 if d < z_vals[i] else 0.0
                    idx += 1

            encoded_value = binary_inst.objective.evaluate(ommx.v1.State(bin_state))
            assert encoded_value == pytest.approx(original_value, abs=1e-10)

    @pytest.mark.parametrize("seed", [0, 1, 42])
    def test_mixed_degree_objective_equivalence(self, seed):
        """Objective with cubic + quadratic + linear + constant terms."""
        rng = np.random.default_rng(seed)
        upper = 2
        dvs = [
            ommx.v1.DecisionVariable.integer(i, lower=0, upper=upper) for i in range(3)
        ]
        c3 = float(rng.uniform(-2, 2))
        c2 = float(rng.uniform(-2, 2))
        c1 = float(rng.uniform(-2, 2))
        c0 = float(rng.uniform(-5, 5))
        objective = (
            c3 * dvs[0] * dvs[1] * dvs[2] + c2 * dvs[0] * dvs[1] + c1 * dvs[2] + c0
        )
        instance = _make_instance(dvs, objective)

        binary_inst, _ = BinaryIntegerEncoder(instance).encode()

        for _ in range(10):
            z_vals = rng.integers(0, upper + 1, size=3)
            original_value = (
                c3 * z_vals[0] * z_vals[1] * z_vals[2]
                + c2 * z_vals[0] * z_vals[1]
                + c1 * z_vals[2]
                + c0
            )

            bin_state = {}
            idx = 0
            for i in range(3):
                for d in range(upper):
                    bin_state[idx] = 1.0 if d < z_vals[i] else 0.0
                    idx += 1

            encoded_value = binary_inst.objective.evaluate(ommx.v1.State(bin_state))
            assert encoded_value == pytest.approx(original_value, abs=1e-10)


# =====================================================================
# 6. Constraint transformation and constraint_rhs_total
# =====================================================================


class TestConstraintTransformation:
    """Verify constraint rewriting and RHS total derivation."""

    def test_simple_sum_constraint(self):
        """sum(z) = M with lower=0 yields constraint_rhs_total = M."""
        z0 = ommx.v1.DecisionVariable.integer(0, lower=0, upper=3)
        z1 = ommx.v1.DecisionVariable.integer(1, lower=0, upper=3)
        instance = _make_instance([z0, z1], z0 + z1, [(z0 + z1 == 4).set_id(0)])
        _, rhs = BinaryIntegerEncoder(instance).encode()
        assert rhs == 4

    def test_nonzero_lower_adjusts_rhs(self):
        """Non-zero lower bounds shift the constraint RHS."""
        z0 = ommx.v1.DecisionVariable.integer(0, lower=1, upper=3)
        z1 = ommx.v1.DecisionVariable.integer(1, lower=1, upper=3)
        # z0+z1 = 4 → (1+sum x0d) + (1+sum x1d) = 4 → sum x = 2
        instance = _make_instance([z0, z1], z0 + z1, [(z0 + z1 == 4).set_id(0)])
        _, rhs = BinaryIntegerEncoder(instance).encode()
        assert rhs == 2

    def test_multiple_constraints(self):
        """RHS total is the sum across all constraints."""
        z0 = ommx.v1.DecisionVariable.integer(0, lower=0, upper=3)
        z1 = ommx.v1.DecisionVariable.integer(1, lower=0, upper=3)
        z2 = ommx.v1.DecisionVariable.integer(2, lower=0, upper=3)
        instance = _make_instance(
            [z0, z1, z2],
            z0 + z1 + z2,
            [
                (z0 + z1 == 3).set_id(0).add_name("c0"),
                (z1 + z2 == 2).set_id(1).add_name("c1"),
            ],
        )
        _, rhs = BinaryIntegerEncoder(instance).encode()
        assert rhs == 3 + 2

    def test_constraint_name_preserved(self):
        """Constraint names survive the encoding."""
        z = ommx.v1.DecisionVariable.integer(0, lower=0, upper=3)
        instance = _make_instance([z], z, [(z == 2).set_id(0).add_name("budget")])
        binary_inst, _ = BinaryIntegerEncoder(instance).encode()
        assert binary_inst.constraints[0].name == "budget"

    def test_no_constraints(self):
        """Unconstrained problem yields constraint_rhs_total = 0."""
        z = ommx.v1.DecisionVariable.integer(0, lower=0, upper=3)
        instance = _make_instance([z], z)
        _, rhs = BinaryIntegerEncoder(instance).encode()
        assert rhs == 0


# =====================================================================
# 7. Objective value equivalence
# =====================================================================


class TestObjectiveEquivalence:
    """Verify that the encoded objective reproduces the original values."""

    @pytest.mark.parametrize("seed", [0, 1, 42])
    def test_linear_objective_equivalence(self, seed):
        """Encoded linear objective evaluates to same value as original."""
        rng = np.random.default_rng(seed)
        n = 3
        upper = 4
        coeffs = rng.uniform(-5, 5, size=n)

        dvs = [
            ommx.v1.DecisionVariable.integer(i, lower=0, upper=upper) for i in range(n)
        ]
        objective = sum(float(c) * dv for c, dv in zip(coeffs, dvs))
        instance = _make_instance(dvs, objective)

        encoder = BinaryIntegerEncoder(instance)
        binary_inst, _ = encoder.encode()

        # Enumerate a few integer assignments and check equivalence
        for _ in range(10):
            z_vals = rng.integers(0, upper + 1, size=n)
            original_value = float(np.dot(coeffs, z_vals))

            # Build binary assignment (unary: x_{l,d}=1 for d < z_l)
            bin_state = {}
            idx = 0
            for i in range(n):
                for d in range(upper):
                    bin_state[idx] = 1.0 if d < z_vals[i] else 0.0
                    idx += 1

            encoded_value = binary_inst.objective.evaluate(ommx.v1.State(bin_state))
            assert encoded_value == pytest.approx(original_value, abs=1e-10)

    @pytest.mark.parametrize("seed", [0, 1, 42])
    def test_quadratic_objective_equivalence(self, seed):
        """Encoded quadratic objective evaluates to same value as original."""
        rng = np.random.default_rng(seed)
        upper = 3
        dvs = [
            ommx.v1.DecisionVariable.integer(i, lower=0, upper=upper) for i in range(2)
        ]
        q_coeff = float(rng.uniform(-3, 3))
        objective = q_coeff * dvs[0] * dvs[1]
        instance = _make_instance(dvs, objective)

        binary_inst, _ = BinaryIntegerEncoder(instance).encode()

        for _ in range(10):
            z_vals = rng.integers(0, upper + 1, size=2)
            original_value = q_coeff * z_vals[0] * z_vals[1]

            bin_state = {}
            idx = 0
            for i in range(2):
                for d in range(upper):
                    bin_state[idx] = 1.0 if d < z_vals[i] else 0.0
                    idx += 1

            encoded_value = binary_inst.objective.evaluate(ommx.v1.State(bin_state))
            assert encoded_value == pytest.approx(original_value, abs=1e-10)


# =====================================================================
# 8. Integration with FQAOAConverter
# =====================================================================


class TestFQAOAIntegration:
    """Verify the encode → FQAOAConverter pipeline."""

    def test_encode_to_fqaoa_converter(self):
        """Integer instance feeds directly into FQAOAConverter."""
        from qamomile.optimization.fqaoa import FQAOAConverter

        z0 = ommx.v1.DecisionVariable.integer(0, lower=0, upper=3)
        z1 = ommx.v1.DecisionVariable.integer(1, lower=0, upper=3)
        instance = _make_instance([z0, z1], z0 * z1, [(z0 + z1 == 3).set_id(0)])

        converter = FQAOAConverter(instance)

        assert converter.num_fermions == 3
        assert converter.num_qubits == 6
        hamiltonian = converter.get_cost_hamiltonian()
        assert len(hamiltonian.terms) > 0
