"""Tests for PCE (Pauli Correlation Encoding) converter."""

import itertools
import math

import networkx as nx
import numpy as np
import ommx.v1
import pytest

import qamomile.observable as qm_o
from qamomile.optimization.binary_model import (
    BinaryExpr,
    BinaryModel,
    VarType,
    binary,
)
from qamomile.optimization.pce import PCEConverter, PCEEncoder, SignRounder


def _make_spin_model(num_vars: int) -> BinaryModel:
    """Build a tiny linear-only SPIN model with ``num_vars`` variables."""
    return BinaryModel.from_ising(
        linear={i: 1.0 for i in range(num_vars)},
        quad={},
        constant=0.0,
    )


class TestPCEEncoderMinNumQubits:
    """Tests for ``PCEEncoder.min_num_qubits`` capacity formula."""

    def test_k1_capacity(self):
        """k=1 → C(n, 1) * 3 = 3n; smallest n with 3n >= num_vars."""
        # num_vars = 1 → n = 1; num_vars = 3 → n = 1; num_vars = 4 → n = 2
        assert PCEEncoder.min_num_qubits(num_vars=1, k=1) == 1
        assert PCEEncoder.min_num_qubits(num_vars=3, k=1) == 1
        assert PCEEncoder.min_num_qubits(num_vars=4, k=1) == 2
        assert PCEEncoder.min_num_qubits(num_vars=6, k=1) == 2
        assert PCEEncoder.min_num_qubits(num_vars=7, k=1) == 3

    def test_k2_capacity(self):
        """k=2 → C(n, 2) * 9; n=2 fits 9, n=3 fits 27, n=4 fits 54."""
        assert PCEEncoder.min_num_qubits(num_vars=1, k=2) == 2
        assert PCEEncoder.min_num_qubits(num_vars=9, k=2) == 2
        assert PCEEncoder.min_num_qubits(num_vars=10, k=2) == 3
        assert PCEEncoder.min_num_qubits(num_vars=27, k=2) == 3
        assert PCEEncoder.min_num_qubits(num_vars=28, k=2) == 4

    def test_k3_capacity(self):
        """k=3 → C(n, 3) * 27; n=3 fits 27, n=4 fits 108."""
        assert PCEEncoder.min_num_qubits(num_vars=27, k=3) == 3
        assert PCEEncoder.min_num_qubits(num_vars=28, k=3) == 4

    def test_zero_vars_returns_k(self):
        """Encoding zero variables still requires ``k`` qubits."""
        assert PCEEncoder.min_num_qubits(num_vars=0, k=1) == 1
        assert PCEEncoder.min_num_qubits(num_vars=0, k=2) == 2
        assert PCEEncoder.min_num_qubits(num_vars=0, k=5) == 5

    def test_invalid_k(self):
        """``k < 1`` is rejected."""
        with pytest.raises(ValueError, match="k must be a positive integer"):
            PCEEncoder.min_num_qubits(num_vars=1, k=0)
        with pytest.raises(ValueError, match="k must be a positive integer"):
            PCEEncoder.min_num_qubits(num_vars=1, k=-1)

    def test_negative_num_vars(self):
        """Negative ``num_vars`` is rejected."""
        with pytest.raises(ValueError, match="num_vars must be non-negative"):
            PCEEncoder.min_num_qubits(num_vars=-1, k=2)


class TestPCEEncoderConstruction:
    """Tests for ``PCEEncoder`` construction and validation."""

    def test_invalid_k(self):
        """``k`` must be a positive integer."""
        spin = _make_spin_model(2)
        with pytest.raises(ValueError, match="k must be a positive integer"):
            PCEEncoder(spin, k=0)

    def test_non_spin_vartype(self):
        """``BinaryModel`` must be in SPIN vartype."""
        binary_model = BinaryModel.from_qubo({(0, 0): 1.0, (0, 1): 1.0})
        with pytest.raises(ValueError, match="SPIN"):
            PCEEncoder(binary_model, k=2)

    def test_hubo_rejection(self):
        """PCE rejects higher-order (HUBO) problems."""
        hubo = BinaryModel.from_hubo({(0, 1, 2): 1.0})
        spin = hubo.change_vartype(VarType.SPIN)
        with pytest.raises(ValueError, match="higher-order"):
            PCEEncoder(spin, k=2)


class TestPCEEncoderEnumeration:
    """Tests for the deterministic enumeration of k-body Pauli correlators."""

    def test_pauli_encoding_structure(self):
        """Each variable maps to a single-term Hamiltonian on ``num_qubits``."""
        spin = _make_spin_model(num_vars=5)
        encoder = PCEEncoder(spin, k=2)

        assert encoder.num_qubits == 2  # C(2, 2) * 9 = 9 ≥ 5
        assert len(encoder.pauli_encoding) == 5

        for i in range(5):
            ham = encoder.pauli_encoding[i]
            assert ham.num_qubits == encoder.num_qubits
            assert len(ham.terms) == 1
            ((operators, coeff),) = ham.terms.items()
            # k-body: each correlator acts on exactly ``k`` qubits.
            assert len(operators) == 2
            assert coeff == pytest.approx(1.0)
            # All three Pauli choices may appear in this position.
            for op in operators:
                assert op.pauli in (qm_o.Pauli.X, qm_o.Pauli.Y, qm_o.Pauli.Z)

    def test_correlators_are_distinct(self):
        """No two variables share the same Pauli string."""
        spin = _make_spin_model(num_vars=12)
        encoder = PCEEncoder(spin, k=2)

        seen = set()
        for ham in encoder.pauli_encoding.values():
            ((operators, _),) = ham.terms.items()
            key = tuple(sorted((op.index, op.pauli.value) for op in operators))
            assert key not in seen
            seen.add(key)

    def test_k1_enumeration_order(self):
        """k=1 enumerates X, Y, Z on qubit 0, then qubit 1, ..."""
        spin = _make_spin_model(num_vars=6)
        encoder = PCEEncoder(spin, k=1)

        # n=2: (qubit 0, X), (qubit 0, Y), (qubit 0, Z),
        #      (qubit 1, X), (qubit 1, Y), (qubit 1, Z)
        assert encoder.num_qubits == 2
        expected = [
            (0, qm_o.Pauli.X),
            (0, qm_o.Pauli.Y),
            (0, qm_o.Pauli.Z),
            (1, qm_o.Pauli.X),
            (1, qm_o.Pauli.Y),
            (1, qm_o.Pauli.Z),
        ]
        for i, (q, p) in enumerate(expected):
            ham = encoder.pauli_encoding[i]
            ((operators, _),) = ham.terms.items()
            assert len(operators) == 1
            assert operators[0].pauli == p
            assert operators[0].index == q

    def test_k2_enumeration_order(self):
        """k=2 iterates (qubit pairs in lex order) × (Pauli pairs in product order)."""
        spin = _make_spin_model(num_vars=9)
        encoder = PCEEncoder(spin, k=2)

        # n=2, qubit pair (0, 1); Pauli assignments in itertools.product order.
        assert encoder.num_qubits == 2
        paulis = (qm_o.Pauli.X, qm_o.Pauli.Y, qm_o.Pauli.Z)
        expected_assignments = list(itertools.product(paulis, repeat=2))

        for i, (p_a, p_b) in enumerate(expected_assignments):
            ham = encoder.pauli_encoding[i]
            ((operators, _),) = ham.terms.items()
            # Hamiltonian.add_term sorts operators by index*10+pauli.value;
            # qubits 0 and 1 are already in ascending order.
            assert operators[0].index == 0
            assert operators[1].index == 1
            assert operators[0].pauli == p_a
            assert operators[1].pauli == p_b


class TestPCEConverterInit:
    """Tests for ``PCEConverter`` construction."""

    def test_from_binary_model_binary(self):
        """A BINARY ``BinaryModel`` keeps BINARY as the output vartype."""
        x = binary(0)
        y = binary(1)
        z = binary(2)

        problem = BinaryExpr()
        problem += x * y
        problem += y * z

        model = BinaryModel(problem)
        converter = PCEConverter(model, k=2)

        assert converter.original_vartype == VarType.BINARY
        assert converter.spin_model.vartype == VarType.SPIN
        assert converter.spin_model.num_bits == 3
        assert converter.k == 2
        assert converter.num_qubits == PCEEncoder.min_num_qubits(3, 2)
        assert converter.instance is None

    def test_from_binary_model_spin(self):
        """A SPIN ``BinaryModel`` keeps SPIN as the output vartype."""
        ising = BinaryModel.from_ising(
            linear={0: 1.0, 1: 0.5}, quad={(0, 1): -0.3}, constant=0.0
        )
        converter = PCEConverter(ising, k=2)

        assert converter.original_vartype == VarType.SPIN
        assert converter.spin_model.vartype == VarType.SPIN

    def test_from_ommx_instance(self):
        """An ``ommx.v1.Instance`` is converted via ``to_qubo`` and reports BINARY.

        Verifies that PCEConverter follows the same caller-instance
        immutability contract as MathematicalProblemConverter:
        ``to_qubo`` is called on a deep copy stored as ``self.instance``,
        so the caller's instance is left exactly as they passed it.
        """
        jm = pytest.importorskip("jijmodeling")

        problem = jm.Problem("simple")

        @problem.update
        def _(p: jm.DecoratedProblem):
            x = p.BinaryVar()
            y = p.BinaryVar()
            z = p.BinaryVar()
            p += x * y
            p += -1 * y * z
            p += x

        instance = problem.eval({})
        snapshot = instance.to_bytes()
        converter = PCEConverter(instance, k=2)

        assert converter.instance is not None
        assert converter.original_vartype == VarType.BINARY
        assert converter.spin_model.vartype == VarType.SPIN
        assert converter.spin_model.num_bits == 3
        # Caller's instance must be byte-identical after construction —
        # PCEConverter operates on a deep copy of the OMMX instance.
        assert instance.to_bytes() == snapshot

    def test_invalid_instance_type(self):
        """Non-Instance / non-BinaryModel inputs raise ``TypeError``."""
        with pytest.raises(TypeError, match="ommx.v1.Instance or BinaryModel"):
            PCEConverter("not a model", k=2)  # type: ignore[arg-type]

    def test_invalid_k_propagates(self):
        """``k < 1`` propagates from the encoder as ``ValueError``."""
        ising = BinaryModel.from_ising(linear={0: 1.0}, quad={}, constant=0.0)
        with pytest.raises(ValueError, match="k must be a positive integer"):
            PCEConverter(ising, k=0)

    def test_hubo_rejection_via_converter(self):
        """HUBO problems are rejected when constructing the converter."""
        hubo = BinaryModel.from_hubo({(0, 1, 2): 1.0})
        with pytest.raises(ValueError, match="higher-order"):
            PCEConverter(hubo, k=2)


class TestPCEConverterAPI:
    """Tests for ``PCEConverter`` properties and forwarder methods."""

    def test_properties_match_encoder(self):
        """``num_qubits``, ``k``, ``pauli_encoding``, ``encoder`` agree with the encoder."""
        ising = BinaryModel.from_ising(
            linear={0: 1.0, 1: 1.0, 2: 1.0}, quad={}, constant=0.0
        )
        converter = PCEConverter(ising, k=2)

        assert converter.num_qubits == converter.encoder.num_qubits
        assert converter.k == converter.encoder.k
        assert converter.pauli_encoding is converter.encoder.pauli_encoding
        assert isinstance(converter.encoder, PCEEncoder)

    def test_get_encoded_pauli_list_order(self):
        """``get_encoded_pauli_list`` returns observables in variable-index order."""
        ising = BinaryModel.from_ising(
            linear={0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
            quad={},
            constant=0.0,
        )
        converter = PCEConverter(ising, k=2)
        pauli_list = converter.get_encoded_pauli_list()

        assert len(pauli_list) == ising.num_bits
        for i, ham in enumerate(pauli_list):
            assert ham is converter.pauli_encoding[i]
            assert ham.num_qubits == converter.num_qubits

    def test_min_num_qubits_static_forwarder(self):
        """``PCEConverter.min_num_qubits`` matches ``PCEEncoder.min_num_qubits``."""
        for num_vars, k in [(0, 2), (1, 1), (10, 2), (27, 3), (50, 2)]:
            assert PCEConverter.min_num_qubits(
                num_vars, k
            ) == PCEEncoder.min_num_qubits(num_vars, k)


class TestPCEConverterDecode:
    """Tests for ``PCEConverter.decode``."""

    def test_decode_returns_binary_for_binary_input(self):
        """A BINARY-input converter decodes back to BINARY samples."""
        x = binary(0)
        y = binary(1)
        z = binary(2)

        problem = BinaryExpr()
        problem += x * y
        problem += -1.0 * y * z

        model = BinaryModel(problem)
        converter = PCEConverter(model, k=2)

        # Spins [+1, -1, +1] → binaries [0, 1, 0]
        sampleset = converter.decode([0.7, -0.3, 0.5])

        assert sampleset.vartype == VarType.BINARY
        assert len(sampleset.samples) == 1
        assert sampleset.num_occurrences == [1]
        assert sampleset.samples[0] == {0: 0, 1: 1, 2: 0}
        assert len(sampleset.energy) == 1

    def test_decode_returns_spin_for_spin_input(self):
        """A SPIN-input converter decodes back to SPIN samples."""
        ising = BinaryModel.from_ising(
            linear={0: 1.0, 1: 1.0, 2: 1.0},
            quad={(0, 1): 1.0, (1, 2): 1.0},
            constant=0.0,
        )
        converter = PCEConverter(ising, k=2)

        sampleset = converter.decode([0.9, -0.4, 0.2])

        assert sampleset.vartype == VarType.SPIN
        assert sampleset.samples[0] == {0: 1, 1: -1, 2: 1}

    def test_decode_energy_matches_calc_energy(self):
        """Decoded energy equals ``spin_model.calc_energy`` of the rounded spins."""
        ising = BinaryModel.from_ising(
            linear={0: 1.0, 1: -2.0, 2: 0.5},
            quad={(0, 1): 1.5, (1, 2): -0.7},
            constant=3.0,
        )
        converter = PCEConverter(ising, k=2)

        expectations = [0.4, -0.8, 0.1]
        spins = SignRounder().round(expectations)
        expected_energy = converter.spin_model.calc_energy(spins)

        sampleset = converter.decode(expectations)
        assert sampleset.energy[0] == pytest.approx(expected_energy)

    def test_decode_length_mismatch_raises(self):
        """Mismatched ``expectations`` length raises ``ValueError``."""
        ising = BinaryModel.from_ising(
            linear={0: 1.0, 1: 1.0, 2: 1.0}, quad={}, constant=0.0
        )
        converter = PCEConverter(ising, k=2)

        with pytest.raises(ValueError, match="Expected 3 expectation values"):
            converter.decode([0.5, -0.3])
        with pytest.raises(ValueError, match="Expected 3 expectation values"):
            converter.decode([0.1, 0.2, 0.3, 0.4])

    def test_decode_non_contiguous_ids_spin(self):
        """Decoded sample keys match the original non-contiguous variable IDs (SPIN)."""
        # Variables 10, 20, 30 — BinaryModel remaps them to 0, 1, 2 internally.
        ising = BinaryModel.from_ising(
            linear={10: 1.0, 20: -1.0, 30: 0.5},
            quad={(10, 20): 1.0},
            constant=0.0,
        )
        converter = PCEConverter(ising, k=2)

        # Spins: [+1, -1, +1] via sign rounding of [0.7, -0.3, 0.5]
        sampleset = converter.decode([0.7, -0.3, 0.5])

        assert sampleset.vartype == VarType.SPIN
        assert sampleset.samples[0] == {10: 1, 20: -1, 30: 1}

    def test_decode_non_contiguous_ids_binary(self):
        """Decoded sample keys match the original non-contiguous variable IDs (BINARY)."""
        # Variables 10, 20, 30 in a BINARY model.
        x10 = binary(10)
        x20 = binary(20)
        x30 = binary(30)

        problem = BinaryExpr()
        problem += x10 * x20
        problem += -1.0 * x20 * x30

        model = BinaryModel(problem)
        converter = PCEConverter(model, k=2)

        # Expectations [0.7, -0.3, 0.5] → spins [+1, -1, +1] → binaries [0, 1, 0]
        sampleset = converter.decode([0.7, -0.3, 0.5])

        assert sampleset.vartype == VarType.BINARY
        assert sampleset.samples[0] == {10: 0, 20: 1, 30: 0}


class TestPCEConverterDecodeOMMX:
    """Tests for ``PCEConverter.decode`` on the OMMX-backed return path.

    When constructed from an ``ommx.v1.Instance``, ``decode`` should
    return an ``ommx.v1.SampleSet`` so feasibility, original objective,
    and per-constraint diagnostics are available through OMMX's API —
    matching ``MathematicalProblemConverter.decode``'s polymorphic
    behaviour.
    """

    @staticmethod
    def _build_ommx_instance() -> ommx.v1.Instance:
        """Build a small unconstrained 3-binary OMMX instance.

        Objective: ``x0 * x1 - x1 * x2 + x0`` (minimize).
        """
        dvs = [ommx.v1.DecisionVariable.binary(i, name=f"x{i}") for i in range(3)]
        obj = dvs[0] * dvs[1] + (-1.0) * dvs[1] * dvs[2] + dvs[0]
        return ommx.v1.Instance.from_components(
            decision_variables=dvs,
            objective=obj,
            constraints=[],
            sense=ommx.v1.Instance.MINIMIZE,
        )

    def test_decode_returns_ommx_sampleset_for_ommx_input(self):
        """OMMX input → ``decode`` returns ``ommx.v1.SampleSet``."""
        instance = self._build_ommx_instance()
        converter = PCEConverter(instance, k=2)

        # Spins [+1, -1, +1] → bits [0, 1, 0]
        sample_set = converter.decode([0.7, -0.3, 0.5])

        assert isinstance(sample_set, ommx.v1.SampleSet)
        # Single-sample decode → single sample id.
        assert len(sample_set.sample_ids) == 1
        sid = sample_set.sample_ids[0]
        sol = sample_set.get(sid)
        bits = [int(round(sol.decision_variables_df.loc[i, "value"])) for i in range(3)]
        assert bits == [0, 1, 0]

    def test_decode_objective_matches_manual_evaluation(self):
        """OMMX-reported objective equals a manual evaluation on the bits."""
        instance = self._build_ommx_instance()
        converter = PCEConverter(instance, k=2)

        # [+1, +1, -1] → bits [0, 0, 1]
        sample_set = converter.decode([0.5, 0.6, -0.7])
        sid = sample_set.sample_ids[0]
        bits = [
            int(round(sample_set.get(sid).decision_variables_df.loc[i, "value"]))
            for i in range(3)
        ]
        # Objective: x0*x1 - x1*x2 + x0
        manual = bits[0] * bits[1] + (-1.0) * bits[1] * bits[2] + bits[0]
        assert sample_set.summary.loc[sid, "objective"] == pytest.approx(manual)

    def test_decode_feasibility_with_constraint(self):
        """An OMMX equality constraint surfaces in ``SampleSet.feasible``."""
        n = 3
        dvs = [ommx.v1.DecisionVariable.binary(i, name=f"x{i}") for i in range(n)]
        obj = dvs[0] * dvs[1] + (-1.0) * dvs[1] * dvs[2]
        # x0 + x1 + x2 == 2 — feasibility is determined by the rounded bits.
        constraint = (sum(dvs) == 2).set_id(0).add_name("eq")
        instance = ommx.v1.Instance.from_components(
            decision_variables=dvs,
            objective=obj,
            constraints=[constraint],
            sense=ommx.v1.Instance.MINIMIZE,
        )
        converter = PCEConverter(instance, k=2)

        # [+1, -1, +1] → bits [0, 1, 0] — sum = 1, infeasible.
        infeasible_set = converter.decode([0.7, -0.3, 0.5])
        sid_inf = infeasible_set.sample_ids[0]
        assert not bool(infeasible_set.summary.loc[sid_inf, "feasible"])

        # [-1, -1, +1] → bits [1, 1, 0] — sum = 2, feasible.
        feasible_set = converter.decode([-0.4, -0.2, 0.9])
        sid_f = feasible_set.sample_ids[0]
        assert bool(feasible_set.summary.loc[sid_f, "feasible"])

    def test_decode_integer_variable_slack_round_trip(self):
        """Integer DVs are reconstructed from QUBO slack bits in the SampleSet.

        Verifies that the deep-copy + ``to_qubo`` retention pattern lets
        ``evaluate_samples`` map the slack bits added by ``to_qubo`` back to
        the original integer decision variable, matching
        ``MathematicalProblemConverter`` behaviour for integer DVs.
        """
        x = ommx.v1.DecisionVariable.binary(0, name="x")
        y = ommx.v1.DecisionVariable.integer(1, lower=0, upper=3, name="y")
        instance = ommx.v1.Instance.from_components(
            decision_variables=[x, y],
            objective=x + 2 * y,
            constraints=[],
            sense=ommx.v1.Instance.MAXIMIZE,
        )
        converter = PCEConverter(instance, k=2)

        # Use a constant +1 expectations vector — round to all-spin-up,
        # which maps to all-zero bits. The exact decoded values depend on
        # spin_model variable count; we just assert the original DVs are
        # reconstructed and respect their bounds.
        n_vars = converter.spin_model.num_bits
        sample_set = converter.decode([0.5] * n_vars)

        sid = sample_set.sample_ids[0]
        df = sample_set.get(sid).decision_variables_df

        # Original DVs (id 0 binary, id 1 integer) must appear with values
        # respecting their declared kinds and bounds. Slack DVs added by
        # to_qubo are also present in the post-qubo instance — that is
        # fine; we only assert the originals are correct.
        assert 0 in df.index
        assert 1 in df.index
        x_val = float(df.loc[0, "value"])
        y_val = float(df.loc[1, "value"])
        assert x_val in (0.0, 1.0)
        assert 0.0 <= y_val <= 3.0
        # Integer reconstruction must produce an integer-valued result.
        assert y_val == pytest.approx(round(y_val))


class TestPCEEndToEnd:
    """End-to-end workflow tests for PCE without quantum execution."""

    def test_full_decode_workflow(self):
        """Build encoding → mock expectations → decode produces a valid sample."""
        x = binary(0)
        y = binary(1)
        z = binary(2)

        problem = BinaryExpr()
        problem += -1.0 * x * y
        problem += -1.0 * y * z

        model = BinaryModel(problem)
        converter = PCEConverter(model, k=2)

        observables = converter.get_encoded_pauli_list()
        assert len(observables) == 3
        # Each observable is a single k=2 correlator with coefficient 1.
        for ham in observables:
            assert len(ham.terms) == 1
            ((operators, coeff),) = ham.terms.items()
            assert len(operators) == 2
            assert coeff == pytest.approx(1.0)

        # Mock expectations in [-1, 1] decode to a single BINARY sample.
        sampleset = converter.decode([0.9, -0.8, 0.7])
        assert sampleset.vartype == VarType.BINARY
        assert sampleset.samples[0] == {0: 0, 1: 1, 2: 0}

    @staticmethod
    def _build_pce_setup():
        """Build a PCE converter, observables, and a small ansatz kernel."""
        import qamomile.circuit as qmc

        ising = BinaryModel.from_ising(
            linear={0: 1.0, 1: 1.0, 2: 1.0},
            quad={(0, 1): 1.0},
            constant=0.0,
        )
        converter = PCEConverter(ising, k=2)
        n = converter.num_qubits  # 2
        observables = converter.get_encoded_pauli_list()

        @qmc.qkernel
        def ansatz(
            n: qmc.UInt,
            thetas: qmc.Vector[qmc.Float],
            P: qmc.Observable,
        ) -> qmc.Float:
            q = qmc.qubit_array(n, name="q")
            for i in qmc.range(n):
                q[i] = qmc.h(q[i])
                q[i] = qmc.ry(q[i], thetas[i])
            return qmc.expval(q, P)

        return converter, n, observables, ansatz

    def _check_transpile_runs(self, transpiler):
        """Transpile the PCE ansatz with ``transpiler`` and verify expval runs.

        Asserts that ``PCEConverter.transpile`` returns an executable for which
        the expectation-value path produces a finite real number for each
        encoded observable. Common to all backends in the supported matrix.
        """
        converter, n, observables, ansatz = self._build_pce_setup()

        executor = transpiler.executor()
        for P_i in observables:
            executable = converter.transpile(
                ansatz,
                transpiler,
                bindings={"n": n, "P": P_i},
                parameters=["thetas"],
            )
            result = executable.run(
                executor, bindings={"thetas": [0.0] * n}
            ).result()
            assert isinstance(result, float)
            assert math.isfinite(result)
            # ⟨ψ|P|ψ⟩ for any single Pauli string must lie in [-1, 1].
            assert -1.0 - 1e-6 <= result <= 1.0 + 1e-6

    def test_transpile_with_qiskit(self):
        """``PCEConverter.transpile`` produces a runnable executable on Qiskit."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        self._check_transpile_runs(QiskitTranspiler())

    def test_transpile_with_quri_parts(self):
        """``PCEConverter.transpile`` produces a runnable executable on QuriParts."""
        pytest.importorskip("quri_parts")
        from qamomile.quri_parts import QuriPartsTranspiler

        self._check_transpile_runs(QuriPartsTranspiler())

    def test_transpile_with_cudaq(self):
        """``PCEConverter.transpile`` produces a runnable executable on CUDA-Q."""
        pytest.importorskip("cudaq")
        from qamomile.cudaq import CudaqTranspiler

        self._check_transpile_runs(CudaqTranspiler())

    def test_cross_backend_expval_consistency(self):
        """The same ansatz/observable produces matching ⟨P⟩ across backends.

        Builds one ansatz, runs it through every available backend (Qiskit,
        QuriParts, CUDA-Q), and checks that all backends agree on the
        expectation values to within numerical tolerance. Backends that are
        not installed are silently skipped; if fewer than two backends are
        available, the test itself is skipped.
        """
        converter, n, observables, ansatz = self._build_pce_setup()

        from qamomile.circuit.transpiler.transpiler import Transpiler

        backends: list[tuple[str, Transpiler]] = []
        try:
            pytest.importorskip("qiskit")
            from qamomile.qiskit import QiskitTranspiler

            backends.append(("qiskit", QiskitTranspiler()))
        except pytest.skip.Exception:
            pass
        try:
            pytest.importorskip("quri_parts")
            from qamomile.quri_parts import QuriPartsTranspiler

            backends.append(("quri_parts", QuriPartsTranspiler()))
        except pytest.skip.Exception:
            pass
        try:
            pytest.importorskip("cudaq")
            from qamomile.cudaq import CudaqTranspiler

            backends.append(("cudaq", CudaqTranspiler()))
        except pytest.skip.Exception:
            pass

        if len(backends) < 2:
            pytest.skip("Need at least two installed backends to cross-check.")

        # Use a fixed, non-trivial ``thetas`` so each backend computes the
        # same physical expectation value.
        thetas = [0.3, 0.7]
        per_backend: dict[str, list[float]] = {}
        for name, transpiler in backends:
            executor = transpiler.executor()
            results: list[float] = []
            for P_i in observables:
                executable = converter.transpile(
                    ansatz,
                    transpiler,
                    bindings={"n": n, "P": P_i},
                    parameters=["thetas"],
                )
                results.append(
                    executable.run(executor, bindings={"thetas": thetas}).result()
                )
            per_backend[name] = results

        reference_name, reference = next(iter(per_backend.items()))
        for name, values in per_backend.items():
            if name == reference_name:
                continue
            assert np.allclose(values, reference, atol=1e-6), (
                f"{name} disagrees with {reference_name}: {values} vs {reference}"
            )


class TestPCERandomGraphs:
    """Property-based tests with random Erdős–Rényi graphs."""

    @pytest.mark.parametrize("seed", [42, 123, 456, 789, 1024])
    @pytest.mark.parametrize("k", [1, 2])
    def test_random_graph_invariants(self, seed, k):
        """Structural invariants on random Ising instances at varying ``k``."""
        rng = np.random.default_rng(seed)
        n = int(rng.integers(4, 12))
        G = nx.erdos_renyi_graph(n, 0.4, seed=seed)

        linear = {i: float(rng.uniform(-2, 2)) for i in range(n)}
        quad = {(u, v): float(rng.uniform(-2, 2)) for u, v in G.edges()}
        ising = BinaryModel.from_ising(linear=linear, quad=quad, constant=0.0)

        converter = PCEConverter(ising, k=k)

        # Capacity invariant.
        assert (
            math.comb(converter.num_qubits, k) * (3**k) >= ising.num_bits
        )
        if converter.num_qubits > k:
            assert (
                math.comb(converter.num_qubits - 1, k) * (3**k) < ising.num_bits
            )

        # Encoding shape.
        assert len(converter.encoder.pauli_encoding) == n
        pauli_list = converter.get_encoded_pauli_list()
        assert len(pauli_list) == n
        for ham in pauli_list:
            assert ham.num_qubits == converter.num_qubits
            assert len(ham.terms) == 1
            ((operators, coeff),) = ham.terms.items()
            assert len(operators) == k
            assert coeff == pytest.approx(1.0)

        # Decode round-trip.
        expectations = list(rng.uniform(-1.0, 1.0, size=n))
        sampleset = converter.decode(expectations)
        assert sampleset.vartype == VarType.SPIN  # input was SPIN
        spins = SignRounder().round(expectations)
        assert sampleset.energy[0] == pytest.approx(ising.calc_energy(spins))
