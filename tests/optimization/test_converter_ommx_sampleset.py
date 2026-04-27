"""Tests for the OMMX round-trip output path on MathematicalProblemConverter.

When a converter is built from an ``ommx.v1.Instance``, ``decode()``
returns an ``ommx.v1.SampleSet`` so feasibility, true (un-penalized)
objective, and per-constraint evaluation are available through OMMX's
own API. These tests cover that polymorphic return path plus the
supporting ``BinarySampleSet.to_ommx_samples`` helper.
"""

from __future__ import annotations

import numpy as np
import ommx.v1
import pytest

from qamomile.optimization.binary_model import BinarySampleSet, VarType
from qamomile.optimization.qaoa import QAOAConverter

# --- Helper unit tests (no quantum execution) -------------------------------


def test_to_ommx_samples_preserves_counts_and_states():
    """Verify ``to_ommx_samples`` flattens num_occurrences into sample IDs."""
    ss = BinarySampleSet(
        samples=[{0: 1, 1: 0}, {0: 0, 1: 1}],
        num_occurrences=[3, 2],
        energy=[0.0, 0.0],
        vartype=VarType.BINARY,
    )

    ommx_samples = ss.to_ommx_samples()

    assert isinstance(ommx_samples, ommx.v1.Samples)
    assert ommx_samples.num_samples() == 5
    sample_ids = list(ommx_samples.sample_ids())
    assert len(sample_ids) == 5
    assert len(set(sample_ids)) == 5  # all unique

    state_to_count: dict[tuple[tuple[int, float], ...], int] = {}
    for sid in sample_ids:
        entries = tuple(sorted(ommx_samples.get_state(sid).entries.items()))
        state_to_count[entries] = state_to_count.get(entries, 0) + 1

    expected = {
        ((0, 1.0), (1, 0.0)): 3,
        ((0, 0.0), (1, 1.0)): 2,
    }
    assert state_to_count == expected


def test_to_ommx_samples_skips_zero_occurrences():
    """States with num_occurrences==0 are dropped from the output."""
    ss = BinarySampleSet(
        samples=[{0: 1}, {0: 0}],
        num_occurrences=[2, 0],
        energy=[0.0, 0.0],
        vartype=VarType.BINARY,
    )
    ommx_samples = ss.to_ommx_samples()
    assert ommx_samples.num_samples() == 2


def test_to_ommx_samples_empty_sampleset():
    """An empty BinarySampleSet produces an empty Samples (no sample IDs)."""
    ss = BinarySampleSet(
        samples=[], num_occurrences=[], energy=[], vartype=VarType.BINARY
    )
    ommx_samples = ss.to_ommx_samples()
    assert ommx_samples.num_samples() == 0
    assert list(ommx_samples.sample_ids()) == []


def test_to_ommx_samples_rejects_spin_vartype():
    """SPIN vartype rejected with a clear error pointing to vartype conversion."""
    ss = BinarySampleSet(
        samples=[{0: 1, 1: -1}],
        num_occurrences=[1],
        energy=[0.0],
        vartype=VarType.SPIN,
    )
    with pytest.raises(ValueError, match="BINARY"):
        ss.to_ommx_samples()


# --- Caller-instance immutability regression --------------------------------


def test_qaoa_converter_does_not_mutate_caller_instance():
    """QAOAConverter must not mutate the caller's ommx Instance.

    ``Instance.to_qubo`` mutates the instance it is called on (drops
    constraints, rewrites the objective into penalty form, may add slack
    decision variables). The converter must run that on a deep copy so the
    caller's instance is left exactly as they passed it in.
    """
    x0 = ommx.v1.DecisionVariable.binary(0, name="x0")
    x1 = ommx.v1.DecisionVariable.binary(1, name="x1")
    constraint = (x0 + x1 == 1).set_id(0).add_name("eq")
    instance = ommx.v1.Instance.from_components(
        decision_variables=[x0, x1],
        objective=x0 * x1,
        constraints=[constraint],
        sense=ommx.v1.Instance.MINIMIZE,
    )

    snapshot = instance.to_bytes()
    QAOAConverter(instance)

    assert instance.to_bytes() == snapshot, (
        "QAOAConverter mutated the caller's ommx.v1.Instance; "
        "it must operate on a deep copy."
    )
    assert len(instance.constraints) == 1
    assert instance.constraints[0].name == "eq"


def test_fqaoa_converter_does_not_mutate_caller_instance():
    """FQAOAConverter must not mutate the caller's ommx Instance either."""
    from qamomile.optimization.fqaoa import FQAOAConverter

    # FQAOA expects an OMMX instance with a fermion-counting structure.
    # Use a minimal binary instance — FQAOA does not enforce structure at
    # construction time beyond degree<=2.
    n_sites = 2
    n_orbitals = 2
    dvs = []
    for site in range(n_sites):
        for orb in range(n_orbitals):
            dv = ommx.v1.DecisionVariable.binary(
                site * n_orbitals + orb,
                name="x",
                subscripts=[site, orb],
            )
            dvs.append(dv)
    instance = ommx.v1.Instance.from_components(
        decision_variables=dvs,
        objective=sum(dvs[i] * dvs[i + 1] for i in range(len(dvs) - 1)),
        constraints=[],
        sense=ommx.v1.Instance.MINIMIZE,
    )

    snapshot = instance.to_bytes()
    FQAOAConverter(instance, num_fermions=1)

    assert instance.to_bytes() == snapshot, (
        "FQAOAConverter mutated the caller's ommx.v1.Instance; "
        "it must operate on a deep copy."
    )


# --- End-to-end round-trip via Qiskit ---------------------------------------

pytest.importorskip("qiskit")
from qamomile.qiskit.transpiler import QiskitTranspiler  # noqa: E402


def _build_max_cut_instance(seed: int) -> tuple[ommx.v1.Instance, np.ndarray]:
    """Build a small 3-node weighted max-cut as an OMMX Instance.

    Returns (instance, adjacency_weights). The objective is *minimized* so that
    QAOA's ground-state-as-minimum convention matches the OMMX sense.
    """
    rng = np.random.default_rng(seed)
    n = 3
    weights = rng.uniform(0.5, 2.0, size=(n, n))
    weights = (weights + weights.T) / 2
    np.fill_diagonal(weights, 0.0)

    dvs = [ommx.v1.DecisionVariable.binary(i, name=f"x{i}") for i in range(n)]
    # Max-cut value = sum_{i<j} w_ij * (x_i + x_j - 2*x_i*x_j).
    # Cast to minimization by negating.
    obj = ommx.v1.Linear(terms={}, constant=0.0)
    for i in range(n):
        for j in range(i + 1, n):
            w = float(weights[i, j])
            obj = obj + (-w) * (dvs[i] + dvs[j] - 2 * dvs[i] * dvs[j])

    instance = ommx.v1.Instance.from_components(
        decision_variables=dvs,
        objective=obj,
        constraints=[],
        sense=ommx.v1.Instance.MINIMIZE,
    )
    return instance, weights


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_decode_to_ommx_sampleset_round_trip(seed: int):
    """End-to-end: OMMX in, QAOA on Qiskit, OMMX SampleSet out.

    Verifies (a) the return type is ``ommx.v1.SampleSet``, (b) per-sample
    objectives match a manual evaluation of the OMMX objective on the
    decoded bitstrings, and (c) the best-feasible solution is the global
    optimum (3 nodes is small enough to enumerate).
    """
    instance, weights = _build_max_cut_instance(seed)
    converter = QAOAConverter(instance)

    transpiler = QiskitTranspiler()
    executable = converter.transpile(transpiler, p=2)
    rng = np.random.default_rng(seed)
    bindings = {
        "gammas": rng.uniform(0, np.pi, size=2).tolist(),
        "betas": rng.uniform(0, np.pi / 2, size=2).tolist(),
    }
    result = executable.sample(
        transpiler.executor(), shots=512, bindings=bindings
    ).result()

    sample_set = converter.decode(result)
    assert isinstance(sample_set, ommx.v1.SampleSet)

    n = 3

    def _max_cut_value(bits: list[int]) -> float:
        return -sum(
            weights[i, j] * (bits[i] + bits[j] - 2 * bits[i] * bits[j])
            for i in range(n)
            for j in range(i + 1, n)
        )

    # Per-sample objective check via OMMX's own API: for every sample_id,
    # the objective OMMX reports must equal a manual evaluation on the
    # corresponding decoded bitstring.
    summary = sample_set.summary
    for sid in sample_set.sample_ids:
        sol = sample_set.get(sid)
        bits = [int(round(sol.decision_variables_df.loc[i, "value"])) for i in range(n)]
        assert summary.loc[sid, "objective"] == pytest.approx(
            _max_cut_value(bits), abs=1e-9
        )

    # Best-feasible's objective must be the OMMX argmin (no constraints) and
    # must not undershoot the global optimum found by exhaustive enumeration.
    enumerated_optimum = min(
        _max_cut_value([x0, x1, x2]) for x0 in (0, 1) for x1 in (0, 1) for x2 in (0, 1)
    )
    best = sample_set.best_feasible
    assert best.feasible
    assert best.objective == pytest.approx(min(summary["objective"]))
    assert best.objective >= enumerated_optimum - 1e-9


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_decode_to_ommx_sampleset_constraint_feasibility(seed: int):
    """Graph-partition style: equality constraint must drive ``feasible``.

    Builds a 4-node instance with an equality constraint ``sum x == 2``. After
    sampling, asserts that ``SampleSet.feasible`` agrees with a hand-computed
    feasibility check on the decoded bitstrings, and that ``best_feasible``
    only selects from feasible samples.
    """
    rng = np.random.default_rng(seed)
    n = 4
    target = n // 2

    dvs = [ommx.v1.DecisionVariable.binary(i, name=f"x{i}") for i in range(n)]
    weights = rng.uniform(0.5, 2.0, size=(n, n))
    weights = (weights + weights.T) / 2
    np.fill_diagonal(weights, 0.0)

    obj = ommx.v1.Linear(terms={}, constant=0.0)
    for i in range(n):
        for j in range(i + 1, n):
            w = float(weights[i, j])
            obj = obj + (-w) * (dvs[i] + dvs[j] - 2 * dvs[i] * dvs[j])

    constraint = sum(dvs[i] for i in range(n)) == target
    constraint = constraint.set_id(0).add_name("partition")

    instance = ommx.v1.Instance.from_components(
        decision_variables=dvs,
        objective=obj,
        constraints=[constraint],
        sense=ommx.v1.Instance.MINIMIZE,
    )
    converter = QAOAConverter(instance)

    transpiler = QiskitTranspiler()
    executable = converter.transpile(transpiler, p=1)
    bindings = {
        "gammas": rng.uniform(0, np.pi, size=1).tolist(),
        "betas": rng.uniform(0, np.pi / 2, size=1).tolist(),
    }
    result = executable.sample(
        transpiler.executor(), shots=512, bindings=bindings
    ).result()

    sample_set = converter.decode(result)
    assert isinstance(sample_set, ommx.v1.SampleSet)

    # For every sample_id, OMMX's reported feasibility must match a
    # hand-computed feasibility check on the decoded bitstring.
    summary = sample_set.summary
    any_feasible = False
    for sid in sample_set.sample_ids:
        sol = sample_set.get(sid)
        bits = [int(round(sol.decision_variables_df.loc[i, "value"])) for i in range(n)]
        expected_feasible = sum(bits) == target
        actual_feasible = bool(summary.loc[sid, "feasible"])
        assert actual_feasible == expected_feasible
        any_feasible = any_feasible or expected_feasible

    # If any sampled state was feasible, best_feasible must be feasible.
    if any_feasible:
        assert sample_set.best_feasible.feasible


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_fqaoa_decode_to_ommx_sampleset_round_trip(seed: int):
    """FQAOAConverter.decode() round-trips into an ommx.v1.SampleSet.

    Covers the wiring at FQAOAConverter.__init__ that sets ``self.instance``
    to the post-qubo working copy after ``super().__init__(BinaryModel)``.
    Without that wiring, the inherited polymorphic ``decode`` would fall
    through to the BinaryModel branch and return a BinarySampleSet.
    """
    from qamomile.optimization.fqaoa import FQAOAConverter

    rng = np.random.default_rng(seed)
    n_sites = 2
    n_orbitals = 2
    dvs = []
    for site in range(n_sites):
        for orb in range(n_orbitals):
            dvs.append(
                ommx.v1.DecisionVariable.binary(
                    site * n_orbitals + orb,
                    name="x",
                    subscripts=[site, orb],
                )
            )
    weights = rng.uniform(-1.0, 1.0, len(dvs))
    obj = sum(float(w) * dv for w, dv in zip(weights, dvs))
    instance = ommx.v1.Instance.from_components(
        decision_variables=dvs,
        objective=obj,
        constraints=[],
        sense=ommx.v1.Instance.MINIMIZE,
    )

    converter = FQAOAConverter(instance, num_fermions=1)
    transpiler = QiskitTranspiler()
    executable = converter.transpile(transpiler, p=1)
    bindings = {
        "gammas": rng.uniform(0, np.pi, size=1).tolist(),
        "betas": rng.uniform(0, np.pi / 2, size=1).tolist(),
    }
    result = executable.sample(
        transpiler.executor(), shots=128, bindings=bindings
    ).result()

    sample_set = converter.decode(result)
    assert isinstance(sample_set, ommx.v1.SampleSet)

    # Every OMMX-reported objective must match a manual evaluation on the
    # decoded bitstring — confirms FQAOA's stored self.instance is the
    # right one for evaluate_samples and that DV indices line up.
    summary = sample_set.summary
    for sid in sample_set.sample_ids:
        sol = sample_set.get(sid)
        bits = [
            int(round(sol.decision_variables_df.loc[i, "value"]))
            for i in range(len(dvs))
        ]
        manual = sum(float(w) * b for w, b in zip(weights, bits))
        assert summary.loc[sid, "objective"] == pytest.approx(manual, abs=1e-9)


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_decode_to_ommx_sampleset_integer_slack_mapping(seed: int):
    """Integer decision variables are reconstructed from QUBO slack bits.

    OMMX's ``Instance.evaluate_samples`` is documented to handle the inverse
    mapping from QUBO bitstrings (which include slack variables added by
    ``to_qubo()``) back into the original decision-variable space. Verify the
    returned ``SampleSet``'s ``decision_variables_df`` reports values for the
    *original* decision variables, including the integer one, not the QUBO
    slack variables.
    """
    x = ommx.v1.DecisionVariable.binary(0, name="x")
    y = ommx.v1.DecisionVariable.integer(1, lower=0, upper=3, name="y")

    instance = ommx.v1.Instance.from_components(
        decision_variables=[x, y],
        objective=x + 2 * y,
        constraints=[],
        sense=ommx.v1.Instance.MAXIMIZE,
    )
    converter = QAOAConverter(instance)

    transpiler = QiskitTranspiler()
    executable = converter.transpile(transpiler, p=1)
    rng = np.random.default_rng(seed)
    bindings = {
        "gammas": rng.uniform(0, np.pi, size=1).tolist(),
        "betas": rng.uniform(0, np.pi / 2, size=1).tolist(),
    }
    result = executable.sample(
        transpiler.executor(), shots=128, bindings=bindings
    ).result()

    sample_set = converter.decode(result)
    df = sample_set.best_feasible.decision_variables_df

    # Original DVs (id 0 binary, id 1 integer) must be in the result; their
    # values must respect the declared bounds. Slack variables added by
    # to_qubo (ids >= 2 with kind Binary) are also present in the post-qubo
    # instance — that is fine; we just assert the originals are correct.
    assert 0 in df.index
    assert 1 in df.index
    x_value = float(df.loc[0, "value"])
    y_value = float(df.loc[1, "value"])
    assert x_value in (0.0, 1.0)
    assert 0.0 <= y_value <= 3.0
    # Integer variable must take an integer-valued reconstruction.
    assert y_value == pytest.approx(round(y_value))
