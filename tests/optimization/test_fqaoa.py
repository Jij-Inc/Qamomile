"""Tests for FQAOAConverter."""

from __future__ import annotations

import numpy as np
import ommx.v1
import pytest

from qamomile.optimization.binary_model import BinaryModel
from qamomile.optimization.fqaoa import FQAOAConverter
from qamomile.qiskit import QiskitTranspiler

# =====================================================================
# Helpers / fixtures
# =====================================================================


def _make_instance(dvs, objective, constraints=None, sense=ommx.v1.Instance.MINIMIZE):
    """Shorthand for creating an ommx Instance."""
    return ommx.v1.Instance.from_components(
        decision_variables=dvs,
        objective=objective,
        constraints=constraints or [],
        sense=sense,
    )


@pytest.fixture
def simple_integer_problem():
    """Integer problem: 4 integers in [0,2], quadratic objective, sum=4."""
    J = [
        [0.0, 0.4, 0.0, 0.0],
        [0.0, 0.0, 0.8, 0.0],
        [0.0, 0.0, 0.0, 0.3],
        [0.0, 0.0, 0.0, 0.0],
    ]
    n = 4
    dvs = [
        ommx.v1.DecisionVariable.integer(i, lower=0, upper=2, name=f"z{i}")
        for i in range(n)
    ]

    objective = sum(
        J[i][j] * dvs[i] * dvs[j] for i in range(n) for j in range(n) if J[i][j] != 0.0
    )

    constraint = (sum(dvs) == 4).set_id(0).add_name("constraint")

    return _make_instance(dvs, objective, [constraint])


# =====================================================================
# 1. Initialization
# =====================================================================


def test_initialization(simple_integer_problem):
    """Verify derived attributes after construction."""
    fqaoa_converter = FQAOAConverter(simple_integer_problem)

    assert fqaoa_converter.num_integers == 4
    assert fqaoa_converter.num_bits == 2
    assert fqaoa_converter.num_fermions == 4
    assert isinstance(fqaoa_converter.var_map, dict)
    assert isinstance(fqaoa_converter.spin_model, BinaryModel)
    assert fqaoa_converter.num_qubits == 8


def test_caller_instance_not_mutated():
    """The original ommx instance must not be modified by FQAOAConverter."""
    z0 = ommx.v1.DecisionVariable.integer(0, lower=0, upper=2)
    z1 = ommx.v1.DecisionVariable.integer(1, lower=0, upper=2)
    instance = _make_instance([z0, z1], z0 + z1, [(z0 + z1 == 2).set_id(0)])
    original_bytes = instance.to_bytes()

    FQAOAConverter(instance)

    assert instance.to_bytes() == original_bytes


# =====================================================================
# 2. num_fermions derivation
# =====================================================================


def test_num_fermions_with_nonzero_lower():
    """Non-zero lower bounds shift the constraint RHS and num_fermions."""
    z0 = ommx.v1.DecisionVariable.integer(0, lower=1, upper=3)
    z1 = ommx.v1.DecisionVariable.integer(1, lower=1, upper=3)
    # z0+z1=4 → (1+x0+x1)+(1+x2+x3)=4 → sum x = 2
    instance = _make_instance([z0, z1], z0 + z1, [(z0 + z1 == 4).set_id(0)])
    converter = FQAOAConverter(instance)
    assert converter.num_fermions == 2


# =====================================================================
# 3. Cyclic mapping
# =====================================================================


def test_cyclic_mapping(simple_integer_problem):
    """Verify the ring-driver variable map (l,d) → qubit index."""
    fqaoa_converter = FQAOAConverter(simple_integer_problem)

    assert fqaoa_converter.var_map == {
        (0, 0): 0,
        (1, 0): 1,
        (2, 0): 2,
        (3, 0): 3,
        (0, 1): 4,
        (1, 1): 5,
        (2, 1): 6,
        (3, 1): 7,
    }


# =====================================================================
# 4. get_cost_hamiltonian
# =====================================================================


def test_cost_hamiltonian_has_terms(simple_integer_problem):
    """Hamiltonian must have non-empty terms for a non-trivial problem."""
    converter = FQAOAConverter(simple_integer_problem)
    hamiltonian = converter.get_cost_hamiltonian()
    assert len(hamiltonian.terms) > 0


def test_cost_hamiltonian_constant_propagated():
    """The spin model constant must propagate to the Hamiltonian."""
    z0 = ommx.v1.DecisionVariable.integer(0, lower=0, upper=2)
    z1 = ommx.v1.DecisionVariable.integer(1, lower=0, upper=2)
    # 3*z0 + 2*z1 + 7 has a constant that shifts through SPIN conversion
    instance = _make_instance([z0, z1], 3 * z0 + 2 * z1 + 7, [(z0 + z1 == 2).set_id(0)])
    converter = FQAOAConverter(instance)
    hamiltonian = converter.get_cost_hamiltonian()
    assert hamiltonian.constant == converter.spin_model.constant


# =====================================================================
# 5. get_fermi_orbital
# =====================================================================


def test_fermi_orbital_shape(simple_integer_problem):
    """Orbital matrix must be (num_fermions, num_qubits)."""
    converter = FQAOAConverter(simple_integer_problem)
    orbital = converter.get_fermi_orbital()
    assert orbital.shape == (converter.num_fermions, converter.num_qubits)


def test_fermi_orbital_orthonormality_even():
    """Rows of the orbital matrix must be orthonormal (even num_fermions)."""
    z0 = ommx.v1.DecisionVariable.integer(0, lower=0, upper=3)
    z1 = ommx.v1.DecisionVariable.integer(1, lower=0, upper=3)
    # num_fermions=4 (even), num_qubits=6
    instance = _make_instance([z0, z1], z0 + z1, [(z0 + z1 == 4).set_id(0)])
    converter = FQAOAConverter(instance)
    assert converter.num_fermions % 2 == 0

    orbital = converter.get_fermi_orbital()
    gram = orbital @ orbital.T
    np.testing.assert_allclose(gram, np.eye(converter.num_fermions), atol=1e-12)


def test_fermi_orbital_orthonormality_odd():
    """Rows of the orbital matrix must be orthonormal (odd num_fermions)."""
    z0 = ommx.v1.DecisionVariable.integer(0, lower=0, upper=3)
    z1 = ommx.v1.DecisionVariable.integer(1, lower=0, upper=3)
    # num_fermions=3 (odd), num_qubits=6
    instance = _make_instance([z0, z1], z0 + z1, [(z0 + z1 == 3).set_id(0)])
    converter = FQAOAConverter(instance)
    assert converter.num_fermions % 2 == 1

    orbital = converter.get_fermi_orbital()
    gram = orbital @ orbital.T
    np.testing.assert_allclose(gram, np.eye(converter.num_fermions), atol=1e-12)


# =====================================================================
# 6. transpile — quadratic path
# =====================================================================


def test_transpile_quadratic(simple_integer_problem):
    """Quadratic-only problem produces a valid circuit with correct shape."""
    converter = FQAOAConverter(simple_integer_problem)
    transpiler = QiskitTranspiler()
    executable = converter.transpile(transpiler, p=2)
    circuit = executable.get_first_circuit()

    assert circuit.num_qubits == converter.num_qubits
    assert len(circuit.parameters) == 4  # 2 gammas + 2 betas


# =====================================================================
# 7. transpile — HUBO path
# =====================================================================


def test_transpile_hubo():
    """Higher-order objective triggers the HUBO path and produces a circuit."""
    z0 = ommx.v1.DecisionVariable.integer(0, lower=0, upper=2)
    z1 = ommx.v1.DecisionVariable.integer(1, lower=0, upper=2)
    z2 = ommx.v1.DecisionVariable.integer(2, lower=0, upper=2)
    # Cubic objective → after unary encoding produces higher-order terms
    instance = _make_instance(
        [z0, z1, z2],
        z0 * z1 * z2,
        [(z0 + z1 + z2 == 3).set_id(0)],
    )
    converter = FQAOAConverter(instance)
    assert converter.spin_model.higher  # confirm HUBO path is taken

    transpiler = QiskitTranspiler()
    executable = converter.transpile(transpiler, p=1)
    circuit = executable.get_first_circuit()

    assert circuit.num_qubits == converter.num_qubits
    assert len(circuit.parameters) == 2  # 1 gamma + 1 beta


# =====================================================================
# 8. Randomized parametrized tests
# =====================================================================


def _make_random_qubo_instance(n, D, seed):
    """Build a random QUBO integer instance with n vars in [0, D] and sum constraint."""
    rng = np.random.default_rng(seed)

    dvs = [
        ommx.v1.DecisionVariable.integer(i, lower=0, upper=D, name=f"z{i}")
        for i in range(n)
    ]

    linear_coeffs = rng.standard_normal(n)
    quad_coeffs = rng.standard_normal((n, n))

    objective = sum(float(linear_coeffs[i]) * dvs[i] for i in range(n))
    for i in range(n):
        for j in range(i + 1, n):
            objective = objective + float(quad_coeffs[i, j]) * dvs[i] * dvs[j]

    M = int(rng.integers(1, n * D))
    constraint = (sum(dvs) == M).set_id(0).add_name("sum")

    return _make_instance(dvs, objective, [constraint]), n, D, M


@pytest.mark.parametrize("n", [2, 3, 4])
@pytest.mark.parametrize("D", [2, 3])
@pytest.mark.parametrize("seed", [0, 1, 42])
class TestRandomizedQubo:
    """End-to-end tests with random QUBO instances of varying size."""

    def test_derived_attributes(self, n, D, seed):
        """num_integers, num_bits, num_qubits, num_fermions are consistent."""
        instance, _, _, M = _make_random_qubo_instance(n, D, seed)
        converter = FQAOAConverter(instance)

        assert converter.num_integers == n
        assert converter.num_bits == D
        assert converter.num_qubits == n * D
        assert converter.num_fermions == M

    def test_cost_hamiltonian(self, n, D, seed):
        """Cost Hamiltonian has non-empty terms."""
        instance, _, _, _ = _make_random_qubo_instance(n, D, seed)
        converter = FQAOAConverter(instance)
        hamiltonian = converter.get_cost_hamiltonian()
        assert len(hamiltonian.terms) > 0

    def test_fermi_orbital_orthonormality(self, n, D, seed):
        """Orbital rows are orthonormal for any (n, D, seed)."""
        instance, _, _, _ = _make_random_qubo_instance(n, D, seed)
        converter = FQAOAConverter(instance)
        orbital = converter.get_fermi_orbital()

        assert orbital.shape == (converter.num_fermions, converter.num_qubits)
        gram = orbital @ orbital.T
        np.testing.assert_allclose(gram, np.eye(converter.num_fermions), atol=1e-12)

    def test_transpile_produces_circuit(self, n, D, seed):
        """Transpile succeeds and circuit has correct qubit/parameter count."""
        instance, _, _, _ = _make_random_qubo_instance(n, D, seed)
        converter = FQAOAConverter(instance)
        transpiler = QiskitTranspiler()
        p = 2
        executable = converter.transpile(transpiler, p=p)
        circuit = executable.get_first_circuit()

        assert circuit.num_qubits == converter.num_qubits
        assert len(circuit.parameters) == 2 * p


# =====================================================================
# 9. Validation (rejection)
# =====================================================================


def test_reject_binary_instance():
    """Binary decision variables are rejected."""
    x = ommx.v1.DecisionVariable.binary(0, name="x")
    instance = _make_instance([x], x)
    with pytest.raises(ValueError, match="expected INTEGER"):
        FQAOAConverter(instance)


def test_reject_inequality_constraint():
    """Non-equality constraints are rejected."""
    z = ommx.v1.DecisionVariable.integer(0, lower=0, upper=3)
    instance = _make_instance([z], z, [(z <= 2).set_id(0)])
    with pytest.raises(ValueError, match="not an equality constraint"):
        FQAOAConverter(instance)


def test_reject_nonlinear_constraint():
    """Non-linear constraints are rejected."""
    z0 = ommx.v1.DecisionVariable.integer(0, lower=0, upper=3)
    z1 = ommx.v1.DecisionVariable.integer(1, lower=0, upper=3)
    instance = _make_instance([z0, z1], z0 + z1, [(z0 * z1 == 0).set_id(0)])
    with pytest.raises(ValueError, match="not linear"):
        FQAOAConverter(instance)
