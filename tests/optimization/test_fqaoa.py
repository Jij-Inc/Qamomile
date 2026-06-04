"""Tests for FQAOAConverter."""

from __future__ import annotations

import numpy as np
import ommx.v1
import pytest

import qamomile.circuit as qmc
from qamomile._utils import is_close_zero
from qamomile.circuit.algorithm.fqaoa import (
    fqaoa_state,
    hubo_cost_layer,
    hubo_fqaoa_state,
)
from qamomile.circuit.transpiler.job import SampleResult
from qamomile.optimization.binary_model import BinaryModel
from qamomile.optimization.fqaoa import FQAOAConverter
from qamomile.qiskit import QiskitTranspiler


@qmc.qkernel
def _fqaoa_expval(
    p: qmc.UInt,
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    n: qmc.UInt,
    num_fermions: qmc.UInt,
    givens_ij: qmc.Matrix[qmc.UInt],
    givens_theta: qmc.Vector[qmc.Float],
    hopping: qmc.Float,
    gammas: qmc.Vector[qmc.Float],
    betas: qmc.Vector[qmc.Float],
    obs: qmc.Observable,
) -> qmc.Float:
    """FQAOA state followed by expectation value measurement."""
    q = fqaoa_state(
        p=p,
        quad=quad,
        linear=linear,
        n=n,
        num_fermions=num_fermions,
        givens_ij=givens_ij,
        givens_theta=givens_theta,
        hopping=hopping,
        gammas=gammas,
        betas=betas,
    )
    return qmc.expval(q, obs)


@qmc.qkernel
def _hubo_fqaoa_sample(
    p: qmc.UInt,
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    higher: qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float],
    n: qmc.UInt,
    num_fermions: qmc.UInt,
    givens_ij: qmc.Matrix[qmc.UInt],
    givens_theta: qmc.Vector[qmc.Float],
    hopping: qmc.Float,
    gammas: qmc.Vector[qmc.Float],
    betas: qmc.Vector[qmc.Float],
) -> qmc.Vector[qmc.Bit]:
    """HUBO FQAOA state followed by computational-basis measurement."""
    q = hubo_fqaoa_state(
        p=p,
        quad=quad,
        linear=linear,
        higher=higher,
        n=n,
        num_fermions=num_fermions,
        givens_ij=givens_ij,
        givens_theta=givens_theta,
        hopping=hopping,
        gammas=gammas,
        betas=betas,
    )
    return qmc.measure(q)


@qmc.qkernel
def _hubo_fqaoa_expval(
    p: qmc.UInt,
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    higher: qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float],
    n: qmc.UInt,
    num_fermions: qmc.UInt,
    givens_ij: qmc.Matrix[qmc.UInt],
    givens_theta: qmc.Vector[qmc.Float],
    hopping: qmc.Float,
    gammas: qmc.Vector[qmc.Float],
    betas: qmc.Vector[qmc.Float],
    obs: qmc.Observable,
) -> qmc.Float:
    """HUBO FQAOA state followed by expectation value measurement."""
    q = hubo_fqaoa_state(
        p=p,
        quad=quad,
        linear=linear,
        higher=higher,
        n=n,
        num_fermions=num_fermions,
        givens_ij=givens_ij,
        givens_theta=givens_theta,
        hopping=hopping,
        gammas=gammas,
        betas=betas,
    )
    return qmc.expval(q, obs)


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


def _make_native_hubo_fqaoa_instance(seed: int) -> ommx.v1.Instance:
    """Build a native-binary HUBO FQAOA test instance.

    Args:
        seed (int): Seed used to generate deterministic objective
            coefficients.

    Returns:
        ommx.v1.Instance: Four-variable fixed-Hamming-weight binary instance
            with a cubic objective term.
    """
    rng = np.random.default_rng(seed)
    dvs = [ommx.v1.DecisionVariable.binary(i, name=f"x{i}") for i in range(4)]
    cubic = float(rng.uniform(0.4, 1.2))
    quadratic = float(rng.uniform(-0.7, 0.7))
    linear = float(rng.uniform(-0.5, 0.5))
    objective = cubic * dvs[0] * dvs[1] * dvs[2]
    objective = objective + quadratic * dvs[0] * dvs[3] + linear * dvs[2]
    constraint = (sum(dvs) == 2).set_id(0).add_name("cardinality")
    return _make_instance(dvs, objective, [constraint])


def _build_hubo_fqaoa_bindings(
    converter: FQAOAConverter,
    gammas: list[float],
    betas: list[float],
    obs: qmc.Observable | None = None,
) -> dict[str, object]:
    """Build compile-time bindings for HUBO FQAOA execution tests.

    Args:
        converter (FQAOAConverter): Converter whose prepared spin model and
            Givens decomposition should be bound.
        gammas (list[float]): Cost-layer angles.
        betas (list[float]): Mixer-layer angles.
        obs (qmc.Observable | None): Optional observable for expectation-value
            kernels. Defaults to None.

    Returns:
        dict[str, object]: Bindings accepted by ``_hubo_fqaoa_sample`` and
            ``_hubo_fqaoa_expval``.
    """
    unitary_rows = converter.get_fermi_orbital()
    givens_data = converter._givens_decomposition(unitary_rows)
    givens_ij, gtheta = converter._flatten_givens_data(givens_data)

    bindings: dict[str, object] = {
        "p": len(gammas),
        "quad": {
            ij: Jij
            for ij, Jij in converter.spin_model.quad.items()
            if not is_close_zero(Jij)
        },
        "linear": {
            i: hi
            for i, hi in converter.spin_model.linear.items()
            if not is_close_zero(hi)
        },
        "higher": converter.spin_model.higher,
        "n": converter.num_qubits,
        "num_fermions": converter.num_fermions,
        "givens_ij": givens_ij,
        "givens_theta": gtheta,
        "hopping": 1.0,
        "gammas": gammas,
        "betas": betas,
    }
    if obs is not None:
        bindings["obs"] = obs
    return bindings


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

    assert fqaoa_converter.input_layout == "unary_integer"
    assert fqaoa_converter.num_integers == 4
    assert fqaoa_converter.num_bits == 2
    assert fqaoa_converter.num_fermions == 4
    assert isinstance(fqaoa_converter.var_map, dict)
    assert fqaoa_converter.var_id_to_qubit == {idx: idx for idx in range(8)}
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
        (0, 1): 1,
        (1, 0): 2,
        (1, 1): 3,
        (2, 0): 4,
        (2, 1): 5,
        (3, 0): 6,
        (3, 1): 7,
    }


def test_unary_integer_var_map_handles_uneven_bounds():
    """Unary integer var_map follows actual qubit indices for uneven widths."""
    z0 = ommx.v1.DecisionVariable.integer(0, lower=0, upper=1)
    z1 = ommx.v1.DecisionVariable.integer(1, lower=0, upper=3)
    instance = _make_instance([z0, z1], z0 + z1, [(z0 + z1 == 2).set_id(0)])

    converter = FQAOAConverter(instance)

    assert converter.num_qubits == 4
    assert converter.var_map == {
        (0, 0): 0,
        (1, 0): 1,
        (1, 1): 2,
        (1, 2): 3,
    }


# =====================================================================
# 3b. Native binary inputs
# =====================================================================


def test_accept_native_binary_cardinality_instance():
    """Native binary fixed-Hamming-weight problems are accepted directly."""
    x0 = ommx.v1.DecisionVariable.binary(10, name="x0")
    x1 = ommx.v1.DecisionVariable.binary(2, name="x1")
    x2 = ommx.v1.DecisionVariable.binary(7, name="x2")
    instance = _make_instance(
        [x0, x1, x2],
        1.5 * x0 * x2 + x0,
        [(x0 + x1 + x2 == 2).set_id(0).add_name("cardinality")],
    )

    converter = FQAOAConverter(instance)

    assert converter.input_layout == "native_binary"
    assert converter.num_fermions == 2
    assert converter.num_qubits == 3
    assert converter.num_integers == 3
    assert converter.num_bits == 1
    assert converter.var_id_to_qubit == {2: 0, 7: 1, 10: 2}
    assert converter.var_map == {(10, 0): 2, (2, 0): 0, (7, 0): 1}


def test_native_binary_preserves_objective_absent_variables():
    """Binary variables absent from the objective still allocate qubits."""
    x0 = ommx.v1.DecisionVariable.binary(0, name="x0")
    x1 = ommx.v1.DecisionVariable.binary(1, name="x1")
    x2 = ommx.v1.DecisionVariable.binary(2, name="x2")
    instance = _make_instance(
        [x0, x1, x2],
        x0 * x2,
        [(x0 + x1 + x2 == 1).set_id(0).add_name("cardinality")],
    )

    converter = FQAOAConverter(instance)

    assert converter.num_qubits == 3
    assert converter.spin_model.num_bits == 3
    assert converter.var_id_to_qubit == {0: 0, 1: 1, 2: 2}


def test_native_binary_transpile_quadratic():
    """Native binary cardinality problems transpile through the quadratic path."""
    x0 = ommx.v1.DecisionVariable.binary(0, name="x0")
    x1 = ommx.v1.DecisionVariable.binary(1, name="x1")
    x2 = ommx.v1.DecisionVariable.binary(2, name="x2")
    instance = _make_instance(
        [x0, x1, x2],
        x0 * x1 + 0.25 * x2,
        [(x0 + x1 + x2 == 2).set_id(0).add_name("cardinality")],
    )
    converter = FQAOAConverter(instance)

    executable = converter.transpile(QiskitTranspiler(), p=1)
    circuit = executable.get_first_circuit()

    assert circuit.num_qubits == 3
    assert len(circuit.parameters) == 2


def test_reject_native_binary_single_variable_problem():
    """Single-variable problems are rejected before mixer construction."""
    x = ommx.v1.DecisionVariable.binary(0, name="x")
    instance = _make_instance([x], x, [(x == 1).set_id(0).add_name("cardinality")])
    with pytest.raises(ValueError, match="at least two binary variables"):
        FQAOAConverter(instance)


def test_native_binary_decode_uses_original_variable_ids():
    """Decoded native binary samples keep the original OMMX variable IDs."""
    x0 = ommx.v1.DecisionVariable.binary(10, name="x0")
    x1 = ommx.v1.DecisionVariable.binary(2, name="x1")
    x2 = ommx.v1.DecisionVariable.binary(7, name="x2")
    instance = _make_instance(
        [x0, x1, x2],
        x0 + 2.0 * x2,
        [(x0 + x1 + x2 == 2).set_id(0).add_name("cardinality")],
    )
    converter = FQAOAConverter(instance)

    sample_set = converter.decode(SampleResult(results=[([1, 0, 1], 3)], shots=3))
    assert isinstance(sample_set, ommx.v1.SampleSet)

    solution = sample_set.get(sample_set.sample_ids[0])
    assert solution.decision_variables_df.loc[2, "value"] == pytest.approx(1.0)
    assert solution.decision_variables_df.loc[7, "value"] == pytest.approx(0.0)
    assert solution.decision_variables_df.loc[10, "value"] == pytest.approx(1.0)


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


def test_hubo_cost_layer_uses_full_phase_angle():
    """Higher-order FQAOA terms use the same 2*coeff*gamma angle convention."""

    @qmc.qkernel
    def circuit(
        gamma: qmc.Float,
        quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
        linear: qmc.Dict[qmc.UInt, qmc.Float],
        higher: qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float],
    ) -> qmc.Vector[qmc.Bit]:
        q = qmc.qubit_array(3, "q")
        q = hubo_cost_layer(quad, linear, higher, q, gamma)
        return qmc.measure(q)

    transpiler = QiskitTranspiler()
    executable = transpiler.transpile(
        circuit,
        bindings={
            "gamma": 0.4,
            "quad": {},
            "linear": {},
            "higher": {(0, 1, 2): 1.25},
        },
    )
    circuit = executable.get_first_circuit()
    rz_angles = [
        float(instruction.operation.params[0])
        for instruction in circuit.data
        if instruction.operation.name == "rz"
    ]

    np.testing.assert_allclose(rz_angles, [1.0], atol=1e-12, rtol=0.0)


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
# 9. Cross-backend execution tests
# =====================================================================


def _make_small_fqaoa_instance():
    """Build a small instance for execution tests (2 vars in [0,2], sum=2)."""
    z0 = ommx.v1.DecisionVariable.integer(0, lower=0, upper=2, name="z0")
    z1 = ommx.v1.DecisionVariable.integer(1, lower=0, upper=2, name="z1")
    return _make_instance(
        [z0, z1],
        0.5 * z0 * z1 + z0 + z1,
        [(z0 + z1 == 2).set_id(0).add_name("sum")],
    )


class TestCrossBackendSampling:
    """Cross-backend sampling execution tests."""

    @pytest.mark.parametrize("seed", [0, 42])
    def test_sampling_produces_valid_bitstrings(self, seed):
        """Sampled bitstrings must have the correct length on all backends."""
        instance = _make_small_fqaoa_instance()
        converter = FQAOAConverter(instance)
        p = 1
        gammas = [0.5]
        betas = [0.3]
        n_qubits = converter.num_qubits

        results: dict[str, list[tuple]] = {}

        # Qiskit
        pytest.importorskip("qiskit")
        from qiskit.providers.basic_provider import BasicSimulator

        qiskit_tr = QiskitTranspiler()
        backend = BasicSimulator()
        backend.set_options(seed_simulator=seed)
        exe = converter.transpile(qiskit_tr, p=p)
        job = exe.sample(
            qiskit_tr.executor(backend=backend),
            shots=256,
            bindings={"gammas": gammas, "betas": betas},
        )
        results["qiskit"] = [bits for bits, _ in job.result().results]

        # QuriParts
        # NOTE: QuriParts sampling of the FQAOA circuit with *runtime*
        # gammas/betas hits a backend limitation — the nested Givens-prefixed
        # structure registers the runtime parameters in a sub-circuit scope
        # that the cost-layer RZZ emission cannot see ("Parameter gammas[0]
        # does not belong to this LinearMappedParametricQuantumCircuit").
        # QAOA (no Givens prefix) works, so this is FQAOA-specific. QuriParts
        # FQAOA execution is instead covered by TestCrossBackendExpval, which
        # binds the parameters at compile time. The known limitation is
        # detected and omitted so a different QuriParts failure still surfaces.
        try:
            pytest.importorskip("quri_parts")
            from qamomile.quri_parts import QuriPartsTranspiler

            qp_tr = QuriPartsTranspiler()
            try:
                exe = converter.transpile(qp_tr, p=p)
                job = exe.sample(
                    qp_tr.executor(),
                    shots=256,
                    bindings={"gammas": gammas, "betas": betas},
                )
                results["quri_parts"] = [bits for bits, _ in job.result().results]
            except ValueError as e:
                if "does not belong to this" not in str(e):
                    raise
        except pytest.skip.Exception:
            pass

        # CUDA-Q
        try:
            pytest.importorskip("cudaq")
            from qamomile.cudaq import CudaqTranspiler

            cudaq_tr = CudaqTranspiler()
            exe = converter.transpile(cudaq_tr, p=p)
            job = exe.sample(
                cudaq_tr.executor(),
                shots=256,
                bindings={"gammas": gammas, "betas": betas},
            )
            results["cudaq"] = [bits for bits, _ in job.result().results]
        except (pytest.skip.Exception, ImportError, RuntimeError):
            pass

        for backend_name, bitstrings in results.items():
            assert len(bitstrings) > 0, f"{backend_name}: no results"
            for bits in bitstrings:
                assert len(bits) == n_qubits, (
                    f"{backend_name}: expected {n_qubits} bits, got {len(bits)}"
                )


class TestCrossBackendExpval:
    """Cross-backend expectation value execution tests."""

    @pytest.mark.parametrize("seed", [0, 42])
    def test_expval_cross_backend_agreement(self, seed):
        """Expectation values must agree across backends within tolerance."""
        from qamomile._utils import is_close_zero

        instance = _make_small_fqaoa_instance()
        converter = FQAOAConverter(instance)
        hamiltonian = converter.get_cost_hamiltonian()

        rng = np.random.default_rng(seed)
        gammas_val = rng.uniform(-np.pi, np.pi, size=1).tolist()
        betas_val = rng.uniform(-np.pi, np.pi, size=1).tolist()

        unitary_rows = converter.get_fermi_orbital()
        givens_data = converter._givens_decomposition(unitary_rows)
        givens_ij, gtheta = converter._flatten_givens_data(givens_data)

        linear = {
            i: hi
            for i, hi in converter.spin_model.linear.items()
            if not is_close_zero(hi)
        }
        quad = {
            ij: Jij
            for ij, Jij in converter.spin_model.quad.items()
            if not is_close_zero(Jij)
        }

        bindings = {
            "p": 1,
            "quad": quad,
            "linear": linear,
            "n": converter.num_qubits,
            "num_fermions": converter.num_fermions,
            "givens_ij": givens_ij,
            "givens_theta": gtheta,
            "hopping": 1.0,
            "gammas": gammas_val,
            "betas": betas_val,
            "obs": hamiltonian,
        }

        expvals: dict[str, float] = {}

        # Qiskit
        pytest.importorskip("qiskit")
        qiskit_tr = QiskitTranspiler()
        exe = qiskit_tr.transpile(_fqaoa_expval, bindings=bindings)
        val = exe.run(qiskit_tr.executor()).result()
        expvals["qiskit"] = float(val)

        # QuriParts
        try:
            pytest.importorskip("quri_parts")
            from qamomile.quri_parts import QuriPartsTranspiler

            qp_tr = QuriPartsTranspiler()
            exe = qp_tr.transpile(_fqaoa_expval, bindings=bindings)
            val = exe.run(qp_tr.executor()).result()
            expvals["quri_parts"] = float(val)
        except pytest.skip.Exception:
            pass

        # CUDA-Q
        try:
            pytest.importorskip("cudaq")
            from qamomile.cudaq import CudaqTranspiler

            cudaq_tr = CudaqTranspiler()
            exe = cudaq_tr.transpile(_fqaoa_expval, bindings=bindings)
            val = exe.run(cudaq_tr.executor()).result()
            expvals["cudaq"] = float(val)
        except (pytest.skip.Exception, ImportError, RuntimeError):
            pass

        # All backends must agree
        vals = list(expvals.values())
        assert len(vals) >= 1
        for backend_name, v in expvals.items():
            np.testing.assert_allclose(
                v,
                vals[0],
                atol=1e-6,
                err_msg=f"{backend_name} disagrees with {list(expvals.keys())[0]}",
            )


class TestHuboCrossBackendExecution:
    """Cross-backend HUBO FQAOA execution tests."""

    @pytest.mark.parametrize("seed", [0, 42])
    def test_hubo_sampling_produces_valid_bitstrings(self, seed):
        """HUBO FQAOA sampling executes and returns n-bit samples."""
        instance = _make_native_hubo_fqaoa_instance(seed)
        converter = FQAOAConverter(instance)
        assert converter.spin_model.higher

        rng = np.random.default_rng(seed + 100)
        bindings = _build_hubo_fqaoa_bindings(
            converter,
            gammas=rng.uniform(-np.pi, np.pi, size=1).tolist(),
            betas=rng.uniform(-np.pi, np.pi, size=1).tolist(),
        )
        n_qubits = converter.num_qubits
        results: dict[str, list[tuple]] = {}

        pytest.importorskip("qiskit")
        qiskit_tr = QiskitTranspiler()
        exe = qiskit_tr.transpile(_hubo_fqaoa_sample, bindings=bindings)
        job = exe.sample(qiskit_tr.executor(), shots=128)
        results["qiskit"] = [bits for bits, _ in job.result().results]

        try:
            pytest.importorskip("quri_parts")
            from qamomile.quri_parts import QuriPartsTranspiler

            qp_tr = QuriPartsTranspiler()
            exe = qp_tr.transpile(_hubo_fqaoa_sample, bindings=bindings)
            job = exe.sample(qp_tr.executor(), shots=128)
            results["quri_parts"] = [bits for bits, _ in job.result().results]
        except (pytest.skip.Exception, ImportError, RuntimeError):
            pass

        try:
            pytest.importorskip("cudaq")
            from qamomile.cudaq import CudaqTranspiler

            cudaq_tr = CudaqTranspiler()
            exe = cudaq_tr.transpile(_hubo_fqaoa_sample, bindings=bindings)
            job = exe.sample(cudaq_tr.executor(), shots=128)
            results["cudaq"] = [bits for bits, _ in job.result().results]
        except (pytest.skip.Exception, ImportError, RuntimeError):
            pass

        assert len(results) >= 1
        for backend_name, bitstrings in results.items():
            assert len(bitstrings) > 0, f"{backend_name}: no results"
            for bits in bitstrings:
                assert len(bits) == n_qubits, (
                    f"{backend_name}: expected {n_qubits} bits, got {len(bits)}"
                )

    @pytest.mark.parametrize("seed", [0, 42])
    def test_hubo_expval_cross_backend_agreement(self, seed):
        """HUBO FQAOA expectation values agree across available backends."""
        instance = _make_native_hubo_fqaoa_instance(seed)
        converter = FQAOAConverter(instance)
        hamiltonian = converter.get_cost_hamiltonian()

        rng = np.random.default_rng(seed + 200)
        bindings = _build_hubo_fqaoa_bindings(
            converter,
            gammas=rng.uniform(-np.pi, np.pi, size=1).tolist(),
            betas=rng.uniform(-np.pi, np.pi, size=1).tolist(),
            obs=hamiltonian,
        )
        expvals: dict[str, float] = {}

        pytest.importorskip("qiskit")
        qiskit_tr = QiskitTranspiler()
        exe = qiskit_tr.transpile(_hubo_fqaoa_expval, bindings=bindings)
        expvals["qiskit"] = float(exe.run(qiskit_tr.executor()).result())

        try:
            pytest.importorskip("quri_parts")
            from qamomile.quri_parts import QuriPartsTranspiler

            qp_tr = QuriPartsTranspiler()
            exe = qp_tr.transpile(_hubo_fqaoa_expval, bindings=bindings)
            expvals["quri_parts"] = float(exe.run(qp_tr.executor()).result())
        except (pytest.skip.Exception, ImportError, RuntimeError):
            pass

        try:
            pytest.importorskip("cudaq")
            from qamomile.cudaq import CudaqTranspiler

            cudaq_tr = CudaqTranspiler()
            exe = cudaq_tr.transpile(_hubo_fqaoa_expval, bindings=bindings)
            expvals["cudaq"] = float(exe.run(cudaq_tr.executor()).result())
        except (pytest.skip.Exception, ImportError, RuntimeError):
            pass

        assert len(expvals) >= 1
        reference_backend = next(iter(expvals))
        reference = expvals[reference_backend]
        for backend_name, value in expvals.items():
            np.testing.assert_allclose(
                value,
                reference,
                atol=1e-6,
                rtol=1e-6,
                err_msg=f"{backend_name} disagrees with {reference_backend}",
            )


# =====================================================================
# 10. Validation (rejection)
# =====================================================================


def test_reject_binary_without_cardinality_constraint():
    """Binary decision variables need a fixed-Hamming-weight equality."""
    x = ommx.v1.DecisionVariable.binary(0, name="x")
    instance = _make_instance([x], x)
    with pytest.raises(ValueError, match="exactly one fixed-Hamming-weight"):
        FQAOAConverter(instance)


def test_reject_binary_partial_cardinality_constraint():
    """Binary cardinality constraints must include every decision variable."""
    x0 = ommx.v1.DecisionVariable.binary(0, name="x0")
    x1 = ommx.v1.DecisionVariable.binary(1, name="x1")
    x2 = ommx.v1.DecisionVariable.binary(2, name="x2")
    instance = _make_instance([x0, x1, x2], x0 + x1 + x2, [(x0 + x1 == 1).set_id(0)])
    with pytest.raises(ValueError, match="must include all binary decision variables"):
        FQAOAConverter(instance)


def test_reject_binary_weighted_cardinality_constraint():
    """Binary constraints with non-uniform coefficients are rejected."""
    x0 = ommx.v1.DecisionVariable.binary(0, name="x0")
    x1 = ommx.v1.DecisionVariable.binary(1, name="x1")
    instance = _make_instance([x0, x1], x0 + x1, [(x0 + 2.0 * x1 == 1).set_id(0)])
    with pytest.raises(ValueError, match="same non-zero coefficient"):
        FQAOAConverter(instance)


def test_reject_integer_without_cardinality_constraint():
    """Integer FQAOA also requires one encoded cardinality equality."""
    z = ommx.v1.DecisionVariable.integer(0, lower=0, upper=2)
    instance = _make_instance([z], z)
    with pytest.raises(ValueError, match="exactly one fixed-Hamming-weight"):
        FQAOAConverter(instance)


def test_reject_integer_weighted_cardinality_constraint():
    """Weighted integer constraints are rejected after unary encoding."""
    z0 = ommx.v1.DecisionVariable.integer(0, lower=0, upper=2)
    z1 = ommx.v1.DecisionVariable.integer(1, lower=0, upper=2)
    instance = _make_instance(
        [z0, z1],
        z0 + z1,
        [(2.0 * z0 + z1 == 2).set_id(0).add_name("weighted")],
    )
    with pytest.raises(ValueError, match="same non-zero coefficient"):
        FQAOAConverter(instance)


def test_reject_integer_multiple_cardinality_constraints():
    """Multiple integer equalities are incompatible with the global mixer."""
    z0 = ommx.v1.DecisionVariable.integer(0, lower=0, upper=2)
    z1 = ommx.v1.DecisionVariable.integer(1, lower=0, upper=2)
    instance = _make_instance(
        [z0, z1],
        z0 + z1,
        [
            (z0 == 1).set_id(0).add_name("left"),
            (z1 == 1).set_id(1).add_name("right"),
        ],
    )
    with pytest.raises(ValueError, match="exactly one fixed-Hamming-weight"):
        FQAOAConverter(instance)


def test_reject_mixed_integer_binary_instance():
    """Mixed integer and binary decision variables are rejected."""
    x = ommx.v1.DecisionVariable.binary(0, name="x")
    z = ommx.v1.DecisionVariable.integer(1, lower=0, upper=2, name="z")
    instance = _make_instance([x, z], x + z, [(x + z == 1).set_id(0)])
    with pytest.raises(ValueError, match="all INTEGER or all BINARY"):
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
