"""End-to-end tests for the Qamomile native compiler bridge."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace

import pytest

import qamomile as qm
from qamomile import _qamomile_native as native
from qamomile.circuit import lower_to_circuit
from qamomile.ir import Program
from qamomile.proto import hir_pb2 as hir_wire, qcore_pb2 as qcore_wire
from qamomile.serialization import (
    decode_literal,
    decode_target_adjoint,
    decode_type,
    encode_literal,
    encode_specialization_request,
    encode_type,
)
from qamomile.stdlib import (
    BlockEncoding,
    amplitude_encoding,
    fqaoa_state,
    inverse_qft,
    qaoa_state,
    qft,
    qsvt_from_phases,
    sparse_matrix_lookup,
)
from qamomile.types import as_type

MODULAR_CLEANUP_SOURCE = """
import qamomile as qm

@qm.qkernel
def scratch_step(state: qm.Q[qm.Bits[2]]) -> qm.Q[qm.Bits[2]]:
    state = qm.h(state, target=1)
    state = qm.cx(state, target=0, control=1)
    return state

@qm.qkernel
def modular_cleanup() -> qm.Bits[1]:
    scratch = qm.qubits(2)
    scratch = scratch_step(scratch)
    output = qm.prepare((1,))
    return qm.measure(output)
"""

MULTI_INPUT_CLEANUP_SOURCE = """
import qamomile as qm

@qm.qkernel
def combine(
    left: qm.Q[qm.Bits[1]],
    right: qm.Q[qm.Bits[1]],
) -> tuple[qm.Q[qm.Bits[1]], qm.Q[qm.Bits[1]]]:
    state = qm.concat(left, right)
    state = qm.h(state, target=0)
    return qm.split_at(state, 1)

@qm.qkernel
def multi_input_cleanup() -> qm.Bits[1]:
    left = qm.qubits(1)
    right = qm.qubits(1)
    pair = combine(left, right)
    output = qm.prepare((1,))
    return qm.measure(output)
"""


def lower_adjoint_calls(program: Program):
    """Decode the Rust-owned transient target-adjoint representation."""

    return decode_target_adjoint(
        bytes(native.lower_adjoint_calls_for_target(program.to_bytes()))
    )


@qm.qkernel
def _native_qsvt_encoded_x(state: qm.Q[qm.Bits[2]]) -> qm.Q[qm.Bits[2]]:
    """Provide one minimal block encoding for native runtime QSVT coverage."""

    return qm.x(state, target=1)


_NATIVE_RUNTIME_QSVT = qsvt_from_phases(
    BlockEncoding(_native_qsvt_encoded_x, 1.0, 1, 1),
    degree=2,
)
_NATIVE_STATIC_AMPLITUDE = amplitude_encoding((1.0, 1.0j))
_NATIVE_SPARSE_LOOKUP = sparse_matrix_lookup(
    ((0, 3), (2, 1)),
    ((1, 2), (3, 0)),
    column_width=2,
    value_width=2,
)


def test_native_bridge_owns_qcore_semantic_identity() -> None:
    """Module, selected-family, and program identities cross the thin bridge."""

    module = qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def answer() -> qm.Int:
    return 42
""",
        module_name="native_identity_bridge",
    )
    program = module.specialize("answer")
    module_payload = module.to_bytes()
    program_payload = program.to_bytes()

    assert native.module_semantic_digest(module_payload).hex() == module.semantic_digest
    assert (
        native.resolve_module_entrypoint(module_payload, "answer")
        == "native_identity_bridge.answer"
    )
    assert (
        native.selected_module_semantic_digest(
            module_payload,
            "answer",
        ).hex()
        == module["answer"].semantic_digest
    )
    assert (
        native.program_semantic_digest(program_payload).hex() == program.semantic_digest
    )
    materializer_payload = native.seal_launch_materializer_payload(
        "probe.v1",
        program_payload,
    )
    materializer = native.inspect_launch_materializer_payload(materializer_payload)
    assert materializer.target_format == "probe.v1"
    assert bytes(materializer.program) == program_payload
    with pytest.raises(ValueError, match="Q3001"):
        native.resolve_module_entrypoint(module_payload, "missing")
    with pytest.raises(ValueError, match="Q3001"):
        native.selected_module_semantic_digest(module_payload, "missing")
    with pytest.raises(ValueError, match="Q4011"):
        native.module_semantic_digest(program_payload)
    with pytest.raises(ValueError, match="Q4464"):
        native.seal_launch_materializer_payload("probe.v1", b"not-a-program")


def test_native_bridge_derives_lookup_schedules() -> None:
    """Rust owns address widths, sparse packing, and ascending X targets."""

    xor_plan = native.plan_xor_lookup(["0", "3", "1", "2"], None)
    assert xor_plan.address_width == 2
    assert xor_plan.value_width == 2
    assert xor_plan.flip_targets == [[], [0, 1], [0], [1]]

    sparse_plan = native.plan_sparse_lookup(
        [["0", "1"], ["1", "0"]],
        [["1", "2"], ["3", "0"]],
        "1",
        None,
    )
    assert sparse_plan.address_width == 2
    assert sparse_plan.column_width == 1
    assert sparse_plan.value_width == 2
    assert sparse_plan.flip_targets == [[1], [0, 1, 2], [0, 2], []]

    with pytest.raises(ValueError, match="Q7650"):
        native.plan_xor_lookup(["0", "1", "2"], None)
    with pytest.raises(ValueError, match="Q7658"):
        native.plan_sparse_lookup([["0"], ["2"]], [["0"], ["0"]], "1", None)


def test_native_bridge_derives_mottonen_amplitude_coordinates() -> None:
    """Rust returns one complete exact-phase state-preparation coordinate set."""

    width, ry, rz, phase = native.mottonen_amplitude_coordinates(
        [(1.0, 0.0), (0.0, 1.0)]
    )

    assert width == 1
    assert ry == pytest.approx([1.5707963267948966], abs=1e-15)
    assert rz == pytest.approx([1.5707963267948966], abs=1e-15)
    assert phase == pytest.approx(0.7853981633974483, abs=1e-15)

    with pytest.raises(ValueError, match="Q7504"):
        native.mottonen_amplitude_coordinates([(0.0, 0.0), (0.0, 0.0)])


def test_native_bridge_derives_uniformly_controlled_coordinates() -> None:
    """Rust exposes the shared normalized Walsh/Gray transform."""

    coordinates = native.uniformly_controlled_rotation_coordinates([1.0, 2.0, 4.0, 8.0])

    assert coordinates == pytest.approx([3.75, -1.25, 0.75, -2.25], abs=1e-15)
    with pytest.raises(ValueError, match="Q7511"):
        native.uniformly_controlled_rotation_coordinates([0.0, 1.0, 2.0])


def test_native_bridge_derives_gray_walk_schedules() -> None:
    """Rust owns the shared Gray control walk and Möttönen wire topology."""

    assert native.cyclic_gray_control_schedule(2) == [0, 1, 0, 1]
    assert native.mottonen_rotation_schedule(3) == [
        (1, 1, 2),
        (2, 1, 2),
        (3, 0, 2),
        (4, 0, 1),
        (5, 0, 2),
        (6, 0, 1),
    ]
    with pytest.raises(ValueError, match="Q7512"):
        native.cyclic_gray_control_schedule(21)
    with pytest.raises(ValueError, match="Q7502"):
        native.mottonen_rotation_schedule(21)


def test_native_bridge_derives_bounded_suzuki_schedule() -> None:
    """Rust returns adjacent-normalized product-formula term visits."""

    schedule = native.suzuki_product_formula_schedule(2, 4)

    assert len(schedule) == 11
    assert schedule == list(reversed(schedule))
    with pytest.raises(ValueError, match="Q7522"):
        native.suzuki_product_formula_schedule(2, 3)
    with pytest.raises(ValueError, match="Q7523"):
        native.suzuki_product_formula_schedule(500_001, 2)


def test_native_bridge_minimizes_sparse_polynomials() -> None:
    """Rust selects deterministic flips and returns native objective energy."""

    model = native.SparsePolynomial(
        "BINARY",
        [["0"], ["1"]],
        [-1.0, -3.0],
        0.0,
        None,
    )

    assert model.minimize([0, 0], "best", 1) == ([0, 1], -3.0, 1)
    assert model.minimize_assignment(
        ["0", "0"],
        "best",
        1,
        ["1", "0"],
    ) == ([0, 1], -3.0, 1)
    with pytest.raises(ValueError, match="Q7552"):
        model.minimize([0, 0], "other", None)
    with pytest.raises(ValueError, match="Q7554"):
        model.minimize([2, 0], "best", None)


def test_native_bridge_owns_sparse_polynomial_normal_forms() -> None:
    """Native models preserve arbitrary IDs, domain laws, conversion, and energy."""

    huge = format((1 << 20_000) + 3, "x")
    model = native.SparsePolynomial(
        "BINARY",
        [["9", "9", "2"], ["2", "9"], [], [huge]],
        [3.0, -1.0, 2.0, 0.25],
        0.5,
        ["c", "2", "9", huge],
    )

    assert model.domain == "BINARY"
    assert model.terms == [(["2", "9"], 2.0), ([huge], 0.25)]
    assert model.constant == 2.5
    assert model.variables == ["2", "9", "c", huge]
    assert model.energy([1, 1, 0, 1]) == 4.75
    assert model.normalize_assignment(
        ["1", "0", "1", "1"],
        [huge, "c", "9", "2"],
    ) == [1, 1, 0, 1]
    assert (
        model.energy_assignment(
            ["1", "0", "1", "1"],
            [huge, "c", "9", "2"],
        )
        == 4.75
    )
    hamiltonian, hermitian = model.hamiltonian()
    assert hermitian
    assert native.observable_is_pairwise_commuting(hamiltonian)
    spin = model.convert("SPIN")
    assert spin.domain == "SPIN"
    assert spin.variables == model.variables
    assert spin.energy([-1, -1, 1, -1]) == model.energy([1, 1, 0, 1])

    with pytest.raises(ValueError, match="Q7563"):
        native.SparsePolynomial("BINARY", [], [], 0.0, ["0", "0"])
    with pytest.raises(ValueError, match="Q7566"):
        native.SparsePolynomial(
            "BINARY",
            [[format(index, "x") for index in range(21)]],
            [1.0],
            0.0,
            None,
        ).convert("SPIN")
    with pytest.raises(ValueError, match="Q7567"):
        model.energy_assignment(["1", "1", "0"], ["2", "9", "c"])
    with pytest.raises(ValueError, match="Q7567"):
        model.energy_assignment(["1", "1", "0", "2"])
    with pytest.raises(ValueError, match="Q7567"):
        model.normalize_assignment(
            ["1", "1", "0", "1"],
            ["2", "2", "c", huge],
        )


def test_native_bridge_persists_one_canonical_optimization_problem() -> None:
    """Polynomial bytes and identity come from the native persistence owner."""

    model = native.SparsePolynomial(
        "BINARY",
        [["9", "2"], ["2"]],
        [1.5, -0.25],
        0.5,
        ["9", "2", "c"],
    )

    payload = bytes(model.to_bytes())
    restored = native.SparsePolynomial.from_bytes(payload)

    assert bytes(restored.to_bytes()) == payload
    assert restored.semantic_digest == model.semantic_digest
    assert restored.domain == "BINARY"
    assert restored.terms == model.terms
    assert restored.variables == model.variables
    envelope = qcore_wire.OptimizationProblemProto.FromString(payload)
    assert envelope.artifact_kind == "qamomile.optimization.polynomial"
    assert envelope.problem_digest.hex() == model.semantic_digest

    envelope.constant = 2.0
    with pytest.raises(ValueError, match="Q7574"):
        native.SparsePolynomial.from_bytes(
            envelope.SerializeToString(deterministic=True)
        )


def test_native_bridge_bulk_decodes_and_validates_polynomial_samples() -> None:
    """One native pass owns bit decoding, energy, and distribution statistics."""

    model = native.SparsePolynomial("SPIN", [["2"]], [1.0], 0.0, ["2"])
    summary = native.normalize_sample_counts(
        [([1], "1"), ([0], "3")],
        "1",
    )

    rows, total, mean, lowest = model.decode_samples(summary)

    assert rows == [
        (["2"], ["1"], 1.0, "3"),
        (["2"], ["-1"], -1.0, "1"),
    ]
    assert total == "4"
    assert mean == 0.5
    assert lowest == 1
    assert native.validate_decoded_samples(
        "BINARY",
        [(["0"], ["0"], 2.0, "2")],
    ) == ([(["0"], ["0"], 2.0, "2")], "2", 2.0, 0)
    assert native.validate_decoded_samples(
        None,
        [(["0"], ["-7"], 2.0, "2")],
    ) == ([(["0"], ["-7"], 2.0, "2")], "2", 2.0, 0)
    with pytest.raises(ValueError, match="Q7569.*domain"):
        native.validate_decoded_samples(
            "BINARY",
            [(["0"], ["-1"], 0.0, "1")],
        )
    with pytest.raises(ValueError, match="Q7569.*unique"):
        native.validate_decoded_samples(
            "SPIN",
            [(["0"], ["1"], 0.0, "1"), (["0"], ["1"], 0.0, "1")],
        )


def test_native_bridge_owns_pauli_correlation_planning_and_decoding() -> None:
    """PCE capacity, words, relaxed cost, and sign decode share one Rust plan."""

    model = native.SparsePolynomial(
        "SPIN",
        [["a"], ["a", "14"], ["14"]],
        [1.0, 0.5, -2.0],
        0.25,
        None,
    )
    plan = model.pauli_correlation_plan(2, None)

    assert native.minimum_pce_qubits(9, 2) == 2
    assert native.minimum_pce_qubits(10, 2) == 3
    assert plan.width == 2
    assert plan.words[:2] == [
        [(0, "X"), (1, "X")],
        [(0, "X"), (1, "Y")],
    ]
    assert plan.relaxed_cost([0.4, -0.5]) == pytest.approx(1.55)
    assert plan.decode([0.4, -0.5]) == ([1, -1], 2.75)
    with pytest.raises(ValueError, match="Q7574"):
        plan.relaxed_cost([0.4, 1.1])
    with pytest.raises(ValueError, match="Q7572"):
        model.pauli_correlation_plan(2, 1)


def test_native_bridge_owns_qrac_planning_and_decoding() -> None:
    """QRAC grouping, observables, relaxation, and decode share one Rust plan."""

    model = native.SparsePolynomial(
        "BINARY",
        [["a"], ["a", "14"]],
        [1.0, -2.0],
        0.0,
        None,
    )
    plan = model.qrac_plan("3,1")

    assert plan.width == 2
    assert plan.max_decode_rows == 1_000_000
    assert plan.max_decode_cells == 10_000_000
    assert plan.groups == [[0], [1]]
    assert len(plan.observables) == 2
    assert isinstance(plan.hamiltonian[0], bytes)
    assert plan.hamiltonian[1] is True
    assert plan.decode([[1, -1], [1, -1], [-1, 1]]) == [
        ([0, 1], 0.0, 2),
        ([1, 0], 1.0, 1),
    ]
    assert plan.sign_decode([0.2, -0.3]) == [([0, 1], 0.0, 1)]
    with pytest.raises(ValueError, match="Q7580"):
        model.qrac_plan("unknown")
    with pytest.raises(ValueError, match="Q7584"):
        plan.decode([[1, 0]])


def test_native_bridge_owns_fqaoa_planning_and_decoding() -> None:
    """FQAOA geometry, preparation, cost, and decode share one Rust plan."""

    model = native.SparsePolynomial(
        "BINARY",
        [["28"], ["7"], ["14"], ["28", "3"]],
        [1.0, 2.0, -1.0, 0.5],
        0.0,
        ["28", "7", "14", "3"],
    )
    plan = model.fqaoa_plan(
        ["28", "7", "14", "3"],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        2,
    )

    assert plan.num_sites == 2
    assert plan.num_orbitals == 2
    assert plan.variable_order == [3, 2, 1, 0]
    assert len(plan.givens_pairs) == len(plan.givens_angles)
    assert isinstance(plan.hamiltonian[0], bytes)
    assert plan.hamiltonian[1] is True
    assert plan.max_decode_rows == 1_000_000
    assert plan.max_decode_cells == 10_000_000
    assert plan.decode([([1, 0, 1, 0], 3)]) == [([0, 1, 0, 1], 3.0, 3)]
    with pytest.raises(ValueError, match="Q7593"):
        model.fqaoa_plan(["28"], [(0, 0), (1, 0)], 1)
    with pytest.raises(ValueError, match="rectangular"):
        model.fqaoa_plan(
            ["28", "7", "14", "3"],
            [(0, 0), (0, 2), (1, 0), (1, 1)],
            2,
        )


def test_native_bridge_owns_lcu_coefficient_planning() -> None:
    """LCU alias sums, normalization, padding, amplitudes, and phases share Rust."""

    residue = float.fromhex("0x0.0000000000001p-1022")
    plan = native.plan_lcu_coefficients(
        2,
        [
            (0, float.fromhex("0x1.fffffffffffffp+1023"), 0.0),
            (1, 0.0, -1.0),
            (0, -float.fromhex("0x1.fffffffffffffp+1023"), 0.0),
            (0, residue, 0.0),
        ],
    )

    assert plan.terms[0][:3] == (0, residue, 0.0)
    assert plan.terms[1][:3] == (1, 0.0, -1.0)
    assert plan.terms[1][4] == pytest.approx(-math.pi / 2)
    assert plan.normalization == 1.0
    assert plan.signal_width == 1
    assert plan.case_count == 2
    with pytest.raises(ValueError, match="Q7611"):
        native.plan_lcu_coefficients(2, [(0, 1.0, 0.0)])
    with pytest.raises(ValueError, match="Q7614"):
        native.plan_lcu_coefficients(
            1,
            [
                (0, float.fromhex("0x1.fffffffffffffp+1023"), 0.0),
                (0, float.fromhex("0x1.fffffffffffffp+1023"), 0.0),
            ],
        )


def test_native_bridge_converts_qsp_wx_phases() -> None:
    """QSP endpoint corrections and canonical wrapping are owned by Rust."""

    converted = native.convert_qsp_wx_phases([0.1, 0.2, 0.3, 0.4])
    expected = (
        0.1 + 5.0 * math.pi / 4.0,
        0.2 - math.pi / 2.0,
        0.3 - math.pi / 2.0,
        0.4 - math.pi / 4.0,
    )
    expected = tuple(
        (value + math.pi) % (2.0 * math.pi) - math.pi for value in expected
    )

    assert converted == pytest.approx(expected, abs=1e-15, rel=0.0)
    assert native.convert_qsp_wx_phases([0.25]) == pytest.approx(
        [0.25 - math.pi / 2.0],
        abs=1e-15,
        rel=0.0,
    )
    with pytest.raises(ValueError, match="Q7621"):
        native.convert_qsp_wx_phases([])
    with pytest.raises(ValueError, match="Q7623"):
        native.convert_qsp_wx_phases([math.inf])


def test_native_bridge_synthesizes_and_replays_qsvt_polynomials() -> None:
    """Rust owns bounded phase synthesis and independent certificate replay."""

    certificate = native.synthesize_qsvt_polynomial([0.0, 1.0], 1e-15, 1)

    assert certificate.chebyshev_coefficients == [0.0, 1.0]
    assert certificate.degree == certificate.queries == 1
    assert certificate.odd is True
    assert certificate.uniform_error_bound == 0.0
    assert certificate.target_magnitude_bound == 1.0
    assert certificate.realized_magnitude_bound == 1.0
    assert certificate.projector_rotations == 2
    assert certificate.success == "heralded"
    replayed = native.verify_qsvt_polynomial(
        certificate.chebyshev_coefficients,
        certificate.qsp_phases,
        float.fromhex("0x0.0000000000001p-1022"),
    )
    assert replayed.projector_phases == certificate.projector_phases

    with pytest.raises(ValueError, match="Q7702"):
        native.synthesize_qsvt_polynomial([0.1, 0.1], 1e-6, 32)
    corrupted = certificate.qsp_phases
    corrupted[0] += 0.01
    with pytest.raises(ValueError, match="Q7709"):
        native.verify_qsvt_polynomial([0.0, 1.0], corrupted, 1e-6)


def test_native_bridge_normalizes_periodic_shift_aliases() -> None:
    """Arbitrary-precision signed offsets share exact native coefficient sums."""

    residue = float.fromhex("0x0.0000000000001p-1022")
    terms = native.normalize_periodic_shift_terms(
        [2],
        [["-1"], ["3"], [format((1 << 20_000) + 3, "x")]],
        [
            (float.fromhex("0x1.fffffffffffffp+1023"), 0.0),
            (-float.fromhex("0x1.fffffffffffffp+1023"), 0.0),
            (residue, 0.0),
        ],
    )

    assert terms == [(["3"], residue, 0.0)]
    with pytest.raises(ValueError, match="Q7534"):
        native.normalize_periodic_shift_terms([1, 1], [["0"]], [(1.0, 0.0)])


def test_native_bridge_recognizes_bounded_periodic_matrix() -> None:
    """Dense periodic recognition returns the same canonical shift terms."""

    entries = [
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 0.0),
        (0.0, 0.0),
    ]

    assert native.decompose_periodic_shift_matrix([1], [2, 2], entries) == [
        (["1"], 1.0, 0.0)
    ]
    with pytest.raises(ValueError, match="Q7540"):
        native.decompose_periodic_shift_matrix(
            [1],
            [2, 2],
            [(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (2.0, 0.0)],
        )
    with pytest.raises(ValueError, match="Q7539"):
        native.decompose_periodic_shift_matrix([9], [512, 512], [])


def test_native_bridge_prepares_a_verified_pauli_measurement() -> None:
    """Rust owns observable basis lowering and returns a verified program."""

    source = """
import qamomile as qm

@qm.qkernel
def plus_state() -> qm.Q[qm.Bool]:
    return qm.h(qm.prepare(False), target=0)
"""
    state = qm.build_source(source, module_name="native_pauli").specialize("plus_state")
    payload = native.prepare_pauli_measurement(state.to_bytes(), ((0, "X"),))
    measured = qm.Program.from_bytes(bytes(payload))

    assert "quantum.h" in measured.inspect()
    assert "quantum.measure" in measured.inspect()
    assert qm.run(measured, on=qm.Reference(), shots=4, seed=7).result.counts == {
        (0,): 4
    }


def test_native_bridge_plans_qwc_measurements() -> None:
    """The native bridge exposes total-shot grouping and portable seeds."""

    observable = qm.Z(0) + qm.Z(1)
    plan = native.plan_qwc_pauli_expectation(observable._native_literal, "5", 7)

    assert plan == [([(0, "Z"), (1, "Z")], [0, 1], "5", plan[0][3])]
    assert 0 <= plan[0][3] < 1 << 32
    with pytest.raises(ValueError, match="Q7318"):
        native.plan_qwc_pauli_expectation((qm.X(0) + qm.Z(0))._native_literal, "1", 7)


def test_native_bridge_projects_the_closed_operation_catalog() -> None:
    """Rust exposes one source/effect/requirement vocabulary to Python."""

    operations, terminators = native.semantic_operation_catalog()
    by_opcode = {row[0]: row for row in operations}

    assert len(by_opcode) == len(operations)
    assert by_opcode["quantum.rx"] == (
        "quantum.rx",
        "quantum",
        False,
        ["quantum"],
        ["rotation"],
        (
            "rx",
            [("target", 0)],
            [
                "gates",
                "non_clifford_gates",
                "rotation_gates",
                "single_qubit_gates",
            ],
            True,
        ),
    )
    assert by_opcode["qcore.if"][2] is True
    assert by_opcode["classical.add"][4] == ["classical_compute"]
    assert terminators == [
        "return",
        "unreachable",
        "yield",
        "break",
        "continue",
        "external",
    ]


def test_native_bridge_derives_classical_result_types() -> None:
    """The source adapter consumes the verifier's exact local type equation."""

    def encoded(type_spec: qm.TypeSpec) -> bytes:
        value = qcore_wire.TypeProto()
        encode_type(type_spec, value)
        return value.SerializeToString(deterministic=True)

    payload = native.infer_classical_result_type(
        "classical.add",
        [encoded(qm.Const[qm.Int]), encoded(qm.Param[qm.Real])],
        [],
    )
    result = qcore_wire.TypeProto()
    result.ParseFromString(payload)

    assert decode_type(result) == qm.Param[qm.Real]
    with pytest.raises(ValueError, match="Q4504:.*numeric scalar"):
        native.infer_classical_result_type(
            "classical.add",
            [encoded(qm.Bool)],
            [],
        )


def test_native_bridge_normalizes_exact_aggregate_slices() -> None:
    """Slice direction, omitted bounds, and clamping have one native owner."""

    assert native.normalize_aggregate_slice(5, None, None, None) == [0, 1, 2, 3, 4]
    assert native.normalize_aggregate_slice(5, None, None, "-1") == [4, 3, 2, 1, 0]
    assert native.normalize_aggregate_slice(5, "-4", "-1", None) == [1, 2, 3]
    assert native.normalize_aggregate_slice(5, "99", "-99", "-2") == [4, 2, 0]
    with pytest.raises(ValueError, match="Q4513: aggregate slice step cannot be zero"):
        native.normalize_aggregate_slice(5, None, None, "0")


def test_native_bridge_owns_dimension_unification_and_assignability() -> None:
    """Python projects the compiler's scoped dimensions and recursive stages."""

    def encoded(type_spec: qm.TypeSpec) -> bytes:
        value = qcore_wire.TypeProto()
        encode_type(type_spec, value)
        return value.SerializeToString(deterministic=True)

    assert native.unify_type_dimensions(
        [encoded(qm.Param[qm.Bits["n"]])],
        [encoded(qm.Shot[qm.Bits[4]])],
        [],
    ) == [("n", 4, None)]
    assert (
        native.unify_type_dimensions(
            [encoded(qm.Bits["n"]), encoded(qm.Bits["n"])],
            [encoded(qm.Bits[4]), encoded(qm.Bits[5])],
            [],
        )
        is None
    )

    expected = encoded(qm.Param[tuple[qm.Int, qm.Int]])
    accepted = encoded(as_type(tuple[qm.Const[qm.Int], qm.Param[qm.Int]]))
    rejected = encoded(as_type(tuple[qm.Shot[qm.Int], qm.Param[qm.Int]]))
    assert native.is_type_assignable(expected, accepted)
    assert not native.is_type_assignable(expected, rejected)
    assert native.are_local_peer_types(
        encoded(qm.Const[qm.Bits[4]]),
        encoded(qm.Shot[qm.Bits[4]]),
    )
    assert not native.are_local_peer_types(
        encoded(qm.Const[qm.Bits[4]]),
        encoded(qm.Shot[qm.Bits[5]]),
    )

    literal = qcore_wire.LiteralProto()
    encode_literal((1, 2), literal)
    literal_bytes = literal.SerializeToString(deterministic=True)
    assert native.literal_conforms_type(
        encoded(qm.Param[qm.Vector[qm.Int, 2]]),
        literal_bytes,
    )
    assert not native.literal_conforms_type(
        encoded(qm.Param[qm.Vector[qm.Int, 3]]),
        literal_bytes,
    )
    assert native.infer_literal_type(literal_bytes) == encoded(
        as_type(tuple[qm.Int, qm.Int])
    )
    explicit = qcore_wire.LiteralProto()
    encode_literal("requires a declaration", explicit)
    assert (
        native.infer_literal_type(explicit.SerializeToString(deterministic=True))
        is None
    )

    scheme = encoded(  # noqa: F821
        qm.Param[tuple[qm.Bits["n"], qm.Bits["m"]]]  # noqa: F821
    )
    substituted = native.substitute_type_dimensions(
        scheme,
        [("n", 4, None), ("m", None, "remaining")],
    )
    projected = qcore_wire.TypeProto()
    projected.ParseFromString(substituted)
    assert (
        decode_type(projected)
        == (  # noqa: F821
            qm.Param[tuple[qm.Bits[4], qm.Bits["remaining"]]]  # noqa: F821
        )
    )
    assert native.type_symbolic_dimensions(substituted) == ["remaining"]

    assert native.inspect_type_shape(encoded(qm.Param[qm.Bits[4]])) == (
        4,
        None,
        encoded(qm.Param[qm.Bit]),
        None,
    )
    coherent = encoded(qm.Q[tuple[qm.Bits[2], qm.UFixed[1, 2]]])
    assert native.inspect_type_shape(coherent) == (None, [2, 3], None, None)
    aggregate = encoded(qm.Shot[tuple[qm.Int, qm.Bool]])
    assert native.inspect_type_shape(aggregate) == (
        None,
        None,
        None,
        [encoded(qm.Shot[qm.Int]), encoded(qm.Shot[qm.Bool])],
    )
    assert native.inspect_type_admissibility(
        encoded(as_type(tuple[qm.Bits["n"], qm.UFixed[0, 3]]))  # noqa: F821
    ) == (True, True, True)
    assert native.inspect_type_admissibility(encoded(qm.Real)) == (False, True, True)
    assert native.inspect_type_admissibility(encoded(qm.Q[qm.Bits["n"]])) == (
        False,
        False,
        False,
    )
    assert native.inspect_type_admissibility(encoded(qm.Param[qm.Int])) == (
        False,
        False,
        True,
    )
    assert native.common_type_stage([]) is None
    assert (
        native.common_type_stage(
            [
                encoded(qm.Const[qm.Int]),
                encoded(qm.Shot[qm.Bool]),
                encoded(qm.Param[qm.Real]),
            ]
        )
        == "Shot"
    )
    assert (
        native.common_type_stage([encoded(as_type(tuple[qm.Shot[qm.Int], qm.Bool]))])
        is None
    )


def test_native_bridge_derives_a_distinct_transient_adjoint_artifact() -> None:
    """Primitive callable inversion is Rust-owned and cannot masquerade as persistent QCore."""

    source = """
import qamomile as qm

@qm.qkernel
def flip_phase(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.s(qm.x(state))

@qm.qkernel
def undo(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.adjoint(flip_phase, state)
"""
    program = qm.build_source(source, module_name="native_adjoint").specialize("undo")
    payload = bytes(native.lower_adjoint_calls_for_target(program.to_bytes()))
    package = qcore_wire.QCorePackageProto()
    package.ParseFromString(payload)
    operations = [
        operation
        for function in package.functions
        for operation in function.body.operations
    ]
    opcodes = [operation.opcode for operation in operations]
    originals = [
        attribute.value.string_value
        for operation in operations
        if operation.opcode == "qcore.adjoint"
        for attribute in operation.attributes
        if attribute.name == "original"
    ]

    assert package.artifact_kind == "qamomile.target-adjoint"
    assert "qcore.adjoint_call" not in opcodes
    inverse = next(
        f
        for f in package.functions
        if f.symbol == "native_adjoint.flip_phase.__qamomile_adjoint"
    )
    assert [o.opcode for o in inverse.body.operations] == [
        "qcore.adjoint",
        "qcore.adjoint",
    ]
    entry = next(f for f in package.functions if f.symbol == package.entrypoint)
    assert [o.opcode for o in entry.body.operations] == ["qcore.call"]
    assert originals == ["quantum.s", "quantum.x"]
    header = native.inspect_target_adjoint(payload)
    assert header.verified is True
    assert header.entrypoint == "native_adjoint.undo"
    view = decode_target_adjoint(payload)
    assert not isinstance(view, Program)
    assert view.entry.symbol == "native_adjoint.undo"
    assert not hasattr(view, "to_bytes")
    with pytest.raises(ValueError, match="qamomile.specialization"):
        native.inspect_specialization(payload)
    request = encode_specialization_request(program)
    with pytest.raises(ValueError, match="qamomile.target-adjoint"):
        native.inspect_target_adjoint(request)
    with pytest.raises(qm.SerializationError, match="Q4011"):
        qm.Program.from_bytes(payload)


def test_native_adjoint_lowering_preserves_dependent_callable_boundary() -> None:
    """A concrete caller may invert a callable with a symbolic state width."""

    source = """
import qamomile as qm

@qm.qkernel
def rotate(
    state: qm.Q[qm.Bits[width]],
    theta: qm.Real,
) -> qm.Q[qm.Bits[width]]:
    return qm.ry(state, theta, target=1)

@qm.qkernel
def undo(state: qm.Q[qm.Bits[2]], theta: qm.Real) -> qm.Q[qm.Bits[2]]:
    return qm.adjoint(rotate, state, theta)
"""
    program = qm.build_source(
        source,
        module_name="native_dependent_adjoint",
    ).specialize("undo")

    target = lower_adjoint_calls(program)
    undo = next(
        function
        for function in target.functions
        if function.symbol == "native_dependent_adjoint.undo"
    )

    assert [operation.opcode for operation in undo.body.operations] == ["qcore.call"]
    assert undo.body.operations[0].results[0].type == qm.Q[qm.Bits[2]]
    inverse = next(
        f
        for f in target.functions
        if f.symbol == "native_dependent_adjoint.rotate.__qamomile_adjoint"
    )
    forward = next(
        f for f in target.functions if f.symbol == "native_dependent_adjoint.rotate"
    )
    assert inverse.parameters[0].value.type == forward.result_type
    assert (
        inverse.body.operations[-1].results[0].type == forward.parameters[0].value.type
    )
    assert [o.opcode for o in inverse.body.operations] == [
        "qcore.constant",
        "qcore.adjoint",
    ]


def test_native_adjoint_lowering_materializes_a_structured_conditional() -> None:
    """Rust explicitly inverts both branches without expanding the condition."""

    source = """
import qamomile as qm

@qm.qkernel
def conditional(
    state: qm.Q[qm.Bits[1]],
    enabled: qm.Param[qm.Bool],
) -> qm.Q[qm.Bits[1]]:
    if enabled:
        state = qm.t(qm.s(state))
    else:
        state = qm.x(state)
    return state

@qm.qkernel
def undo(
    state: qm.Q[qm.Bits[1]],
    enabled: qm.Param[qm.Bool],
) -> qm.Q[qm.Bits[1]]:
    return qm.adjoint(conditional, state, enabled)
"""
    program = qm.build_source(source, module_name="native_structured").specialize(
        "undo"
    )

    payload = bytes(native.lower_adjoint_calls_for_target(program.to_bytes()))
    package = qcore_wire.QCorePackageProto.FromString(payload)
    undo = next(
        function
        for function in package.functions
        if function.symbol == "native_structured.conditional.__qamomile_adjoint"
    )
    [conditional] = undo.body.operations
    attributes = {
        attribute.name: attribute.value for attribute in conditional.attributes
    }

    assert conditional.opcode == "qcore.if"
    assert set(attributes) == {
        "capture_names",
        "condition_index",
        "output_names",
        "stage",
    }
    assert list(conditional.operands[:2]) == [undo.parameters[1].value.id] * 2
    assert conditional.results[1].id == undo.body.terminator.operands[0]
    assert [
        [operation.opcode for operation in region.blocks[0].operations]
        for region in conditional.regions
    ] == [["qcore.adjoint", "qcore.adjoint"], ["qcore.adjoint"]]
    assert [
        [
            next(
                attribute.value.string_value
                for attribute in operation.attributes
                if attribute.name == "original"
            )
            for operation in region.blocks[0].operations
        ]
        for region in conditional.regions
    ] == [["quantum.t", "quantum.s"], ["quantum.x"]]
    assert all(
        region.blocks[0].terminator.operands[0] == region.blocks[0].arguments[0].id
        for region in conditional.regions
    )
    assert "qcore.adjoint_call" not in str(package)


def test_native_adjoint_lowering_materializes_a_structured_for_loop() -> None:
    """Rust reverses iteration and explicitly inverts each compact loop body."""

    source = """
import qamomile as qm

@qm.qkernel
def layered(
    state: qm.Q[qm.Bits[1]],
    rounds: qm.Const[qm.Int],
) -> qm.Q[qm.Bits[1]]:
    for _ in range(rounds):
        state = qm.t(qm.s(state))
    for _ in range(rounds):
        state = qm.h(qm.x(state))
    return state

@qm.qkernel
def undo(
    state: qm.Q[qm.Bits[1]],
    rounds: qm.Const[qm.Int],
) -> qm.Q[qm.Bits[1]]:
    return qm.adjoint(layered, state, rounds)
"""
    program = qm.build_source(source, module_name="native_structured_loop").specialize(
        "undo", **{"rounds": 2}
    )

    payload = bytes(native.lower_adjoint_calls_for_target(program.to_bytes()))
    package = qcore_wire.QCorePackageProto.FromString(payload)
    undo = next(
        function
        for function in package.functions
        if function.symbol == "native_structured_loop.layered.__qamomile_adjoint"
    )
    assert [operation.opcode for operation in undo.body.operations] == [
        "qcore.constant",
        "qcore.constant",
        "qcore.constant",
        "qcore.constant",
        "qcore.for",
        "qcore.for",
    ]
    loops = undo.body.operations[-2:]
    assert [
        [operation.opcode for operation in loop.regions[0].blocks[0].operations]
        for loop in loops
    ] == [["qcore.adjoint", "qcore.adjoint"], ["qcore.adjoint", "qcore.adjoint"]]
    assert [
        [
            next(
                attribute.value.string_value
                for attribute in operation.attributes
                if attribute.name == "original"
            )
            for operation in loop.regions[0].blocks[0].operations
        ]
        for loop in loops
    ] == [["quantum.h", "quantum.x"], ["quantum.t", "quantum.s"]]
    for loop in loops:
        attributes = {attribute.name: attribute.value for attribute in loop.attributes}
        assert set(attributes) == {
            "capture_names",
            "induction_name",
            "reverse",
            "stage",
        }
        assert attributes["reverse"].bool_value is True
        assert list(loop.operands[1::2]) == [undo.parameters[1].value.id] * 2
        assert not loop.regions[1].blocks[0].operations
    assert loops[-1].results[1].id == undo.body.terminator.operands[0]
    assert "qcore.adjoint_call" not in str(package)


def test_native_adjoint_lowering_recurses_through_nested_for_carries() -> None:
    """Nested loops prove forwarded classical captures and invert compactly."""

    source = """
import qamomile as qm

@qm.qkernel
def nested(
    state: qm.Q[qm.Bits[1]],
    rounds: qm.Const[qm.Int],
) -> qm.Q[qm.Bits[1]]:
    for _ in range(rounds):
        state = qm.h(state)
        for _ in range(rounds):
            state = qm.x(state)
    return state

@qm.qkernel
def undo(
    state: qm.Q[qm.Bits[1]],
    rounds: qm.Const[qm.Int],
) -> qm.Q[qm.Bits[1]]:
    return qm.adjoint(nested, state, rounds)
"""
    program = qm.build_source(source, module_name="native_nested_for").specialize(
        "undo", **{"rounds": 2}
    )

    payload = bytes(native.lower_adjoint_calls_for_target(program.to_bytes()))
    package = qcore_wire.QCorePackageProto.FromString(payload)
    undo = next(
        function
        for function in package.functions
        if function.symbol == "native_nested_for.nested.__qamomile_adjoint"
    )
    outer = undo.body.operations[-1]
    inner_operations = outer.regions[0].blocks[0].operations
    inner = next(
        operation for operation in inner_operations if operation.opcode == "qcore.for"
    )

    assert outer.opcode == "qcore.for"
    assert all(
        next(
            attribute.value.bool_value
            for attribute in loop.attributes
            if attribute.name == "reverse"
        )
        for loop in (outer, inner)
    )
    assert all(
        attribute.name != "body_mode"
        for loop in (outer, inner)
        for attribute in loop.attributes
    )
    assert any(
        operation.opcode == "qcore.adjoint"
        for operation in outer.regions[0].blocks[0].operations
    )
    assert any(
        operation.opcode == "qcore.adjoint"
        for operation in inner.regions[0].blocks[0].operations
    )
    assert native.inspect_target_adjoint(payload).verified is True


def test_native_adjoint_lowering_materializes_existing_cleanup_regions() -> None:
    """Target artifacts contain no target-interpreted structured adjoint mode."""

    source = """
import qamomile as qm

@qm.qkernel
def cleanup(enabled: qm.Param[qm.Bool]) -> qm.Unit:
    state = qm.qubits(1)
    if enabled:
        state = qm.s(state)
    else:
        state = qm.t(state)
    return None
"""
    program = qm.build_source(
        source, module_name="native_materialized_cleanup"
    ).specialize("cleanup")
    assert "body_mode='adjoint'" in program.inspect()

    payload = bytes(native.lower_adjoint_calls_for_target(program.to_bytes()))
    package = qcore_wire.QCorePackageProto.FromString(payload)
    conditionals = [
        operation
        for function in package.functions
        for operation in function.body.operations
        if operation.opcode == "qcore.if"
    ]

    assert len(conditionals) == 2
    assert all(
        all(attribute.name != "body_mode" for attribute in operation.attributes)
        for operation in conditionals
    )
    assert [
        next(
            attribute.value.string_value
            for attribute in branch.blocks[0].operations[0].attributes
            if attribute.name == "original"
        )
        for branch in conditionals[1].regions
    ] == ["quantum.s", "quantum.t"]
    assert native.inspect_target_adjoint(payload).verified is True


def test_native_adjoint_lowering_materializes_coherent_cleanup_helpers() -> None:
    """Rust turns coherent inverse proofs into ordinary helper-backed calls."""

    source = """
import qamomile as qm

@qm.qkernel
def unchanged(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return state

@qm.qkernel
def flipped(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(state)

@qm.qkernel
def cleanup() -> qm.Unit:
    index = qm.qubits(1)
    target = qm.qubits(1)
    index, target = qm.multiplex(index, (unchanged, flipped), target)
    scratch = qm.concat(index, target)
    return None
"""
    program = qm.build_source(source, module_name="native_coherent_target").specialize(
        "cleanup"
    )
    assert "original='qcore.select_call'" in program.inspect()

    payload = bytes(native.lower_adjoint_calls_for_target(program.to_bytes()))
    package = qcore_wire.QCorePackageProto.FromString(payload)
    helper_suffix = ".__qamomile_adjoint"
    helpers = {
        function.symbol
        for function in package.functions
        if helper_suffix in function.symbol
    }
    entry = next(
        function
        for function in package.functions
        if function.symbol == "native_coherent_target.cleanup"
    )
    selections = [
        operation
        for operation in entry.body.operations
        if operation.opcode == "qcore.select_call"
    ]
    inverse_callees = next(
        attribute.value.tuple_value.values
        for attribute in selections[-1].attributes
        if attribute.name == "callees"
    )

    assert helpers == {
        "native_coherent_target.flipped.__qamomile_adjoint",
        "native_coherent_target.unchanged.__qamomile_adjoint",
    }
    assert {value.string_value for value in inverse_callees} == helpers
    assert all(
        operation.opcode != "qcore.adjoint"
        or next(
            attribute.value.string_value
            for attribute in operation.attributes
            if attribute.name == "original"
        )
        not in {"qcore.control_call", "qcore.select_call"}
        for operation in entry.body.operations
    )
    assert "qcore.adjoint_call" not in str(package)
    assert native.inspect_target_adjoint(payload).verified is True


def test_native_adjoint_lowering_rejects_a_mutated_for_carry() -> None:
    """Iteration-dependent classical carries fail before specialization."""

    source = """
import qamomile as qm

@qm.qkernel
def loop_angle(
    state: qm.Q[qm.Bits[1]],
    rounds: qm.Const[qm.Int],
    theta: qm.Param[qm.Real],
) -> qm.Q[qm.Bits[1]]:
    angle = theta
    for _ in range(rounds):
        angle = angle / 2.0
        state = qm.rz(state, angle)
    return qm.rz(state, angle)

@qm.qkernel
def undo(
    state: qm.Q[qm.Bits[1]],
    rounds: qm.Const[qm.Int],
    theta: qm.Param[qm.Real],
) -> qm.Q[qm.Bits[1]]:
    return qm.adjoint(loop_angle, state, rounds, theta)
"""
    program = qm.build_source(source, module_name="native_loop_carry").specialize(
        "undo", **{"rounds": 2}
    )

    with pytest.raises(ValueError, match="Q7601.*loop classical carry"):
        native.lower_adjoint_calls_for_target(program.to_bytes())


def test_native_adjoint_lowering_rejects_a_mutated_conditional_carry() -> None:
    """Branch-dependent classical values fail before target specialization."""

    source = """
import qamomile as qm

@qm.qkernel
def conditional_angle(
    state: qm.Q[qm.Bits[1]],
    enabled: qm.Param[qm.Bool],
    theta: qm.Param[qm.Real],
) -> qm.Q[qm.Bits[1]]:
    angle = theta
    if enabled:
        angle = theta / 2.0
        state = qm.rz(state, angle)
    return qm.rz(state, angle)

@qm.qkernel
def undo(
    state: qm.Q[qm.Bits[1]],
    enabled: qm.Param[qm.Bool],
    theta: qm.Param[qm.Real],
) -> qm.Q[qm.Bits[1]]:
    return qm.adjoint(conditional_angle, state, enabled, theta)
"""
    program = qm.build_source(
        source, module_name="native_conditional_carry"
    ).specialize("undo")

    with pytest.raises(ValueError, match="Q7601.*conditional classical carry"):
        native.lower_adjoint_calls_for_target(program.to_bytes())


def test_native_adjoint_lowering_recursively_inverts_nested_conditionals() -> None:
    """Nested structured inversion retains both predicates and explicit daggers."""

    source = """
import qamomile as qm

@qm.qkernel
def nested(
    state: qm.Q[qm.Bits[1]],
    outer: qm.Param[qm.Bool],
    inner: qm.Param[qm.Bool],
) -> qm.Q[qm.Bits[1]]:
    if outer:
        if inner:
            state = qm.x(state)
    return state

@qm.qkernel
def undo(
    state: qm.Q[qm.Bits[1]],
    outer: qm.Param[qm.Bool],
    inner: qm.Param[qm.Bool],
) -> qm.Q[qm.Bits[1]]:
    return qm.adjoint(nested, state, outer, inner)
"""
    program = qm.build_source(source, module_name="native_nested_if").specialize("undo")

    payload = bytes(native.lower_adjoint_calls_for_target(program.to_bytes()))
    package = qcore_wire.QCorePackageProto.FromString(payload)
    undo = next(
        function
        for function in package.functions
        if function.symbol == "native_nested_if.nested.__qamomile_adjoint"
    )
    [outer] = undo.body.operations
    [inner] = outer.regions[0].blocks[0].operations
    [inverse_x] = inner.regions[0].blocks[0].operations

    assert outer.opcode == "qcore.if"
    assert inner.opcode == "qcore.if"
    assert all(
        attribute.name != "body_mode"
        for operation in (outer, inner)
        for attribute in operation.attributes
    )
    assert inverse_x.opcode == "qcore.adjoint"
    assert (
        next(
            attribute.value.string_value
            for attribute in inverse_x.attributes
            if attribute.name == "original"
        )
        == "quantum.x"
    )
    assert native.inspect_target_adjoint(payload).verified is True


def test_native_adjoint_lowering_replays_closed_classical_dependencies() -> None:
    """Runtime gate parameters are rebuilt before the reversed quantum diagram."""

    source = """
import qamomile as qm

@qm.qkernel
def rotate(
    state: qm.Q[qm.Bits[1]],
    theta: qm.Param[qm.Real],
) -> qm.Q[qm.Bits[1]]:
    return qm.rz(state, theta / 2.0)

@qm.qkernel
def undo(
    state: qm.Q[qm.Bits[1]],
    theta: qm.Param[qm.Real],
) -> qm.Q[qm.Bits[1]]:
    return qm.adjoint(rotate, state, theta)
"""
    program = qm.build_source(source, module_name="native_angle").specialize("undo")
    payload = bytes(native.lower_adjoint_calls_for_target(program.to_bytes()))
    package = qcore_wire.QCorePackageProto()
    package.ParseFromString(payload)
    undo = next(
        function
        for function in package.functions
        if function.symbol == "native_angle.rotate.__qamomile_adjoint"
    )

    assert [operation.opcode for operation in undo.body.operations] == [
        "qcore.constant",
        "classical.truediv",
        "qcore.adjoint",
    ]
    original = next(
        attribute.value.string_value
        for attribute in undo.body.operations[-1].attributes
        if attribute.name == "original"
    )
    assert original == "quantum.rz"


def test_native_adjoint_lowering_restores_a_multi_owner_diagram() -> None:
    """Tuple and unpack reverse while preserving projected ownership regions."""

    program = qm.build_source(
        MULTI_INPUT_CLEANUP_SOURCE,
        module_name="native_multi_adjoint",
    ).specialize("multi_input_cleanup")
    payload = bytes(native.lower_adjoint_calls_for_target(program.to_bytes()))
    package = qcore_wire.QCorePackageProto()
    package.ParseFromString(payload)
    entry = next(
        function
        for function in package.functions
        if function.symbol == "native_multi_adjoint.combine.__qamomile_adjoint"
    )
    opcodes = [operation.opcode for operation in entry.body.operations]
    reference_opcodes = [
        operation.opcode
        for operation in next(
            f
            for f in lower_adjoint_calls(program).functions
            if f.symbol == "native_multi_adjoint.combine.__qamomile_adjoint"
        ).body.operations
    ]
    repack_index = next(
        index
        for index, operation in enumerate(entry.body.operations)
        if operation.opcode == "qcore.tuple"
        and operation.attributes
        and operation.attributes[0].name == "proof"
    )
    inverse_unpack = entry.body.operations[repack_index - 1]
    repack = entry.body.operations[repack_index]

    assert "qcore.adjoint_call" not in opcodes
    assert opcodes == reference_opcodes
    assert [
        operation.opcode
        for operation in entry.body.operations[repack_index - 4 : repack_index + 1]
    ] == [
        "qcore.reshape",
        "qcore.adjoint",
        "qcore.reshape",
        "qcore.unpack",
        "qcore.tuple",
    ]
    assert [result.region.projections[-1] for result in inverse_unpack.results] == [
        "0",
        "1",
    ]
    assert list(repack.operands) == [result.id for result in inverse_unpack.results]


def test_native_adjoint_lowering_reverses_product_branches() -> None:
    """A source-level factor, branch transform, and tensor reverse in Rust."""

    source = """
import qamomile as qm

@qm.qkernel
def branched(state: qm.Q[qm.Bits[2]]) -> qm.Q[qm.Bits[2]]:
    left, right = qm.split_at(state, 1)
    return qm.concat(qm.x(left), right)

@qm.qkernel
def undo(state: qm.Q[qm.Bits[2]]) -> qm.Q[qm.Bits[2]]:
    return qm.adjoint(branched, state)
"""
    program = qm.build_source(source, module_name="native_branched").specialize("undo")
    payload = bytes(native.lower_adjoint_calls_for_target(program.to_bytes()))
    package = qcore_wire.QCorePackageProto()
    package.ParseFromString(payload)
    undo = next(
        function
        for function in package.functions
        if function.symbol == "native_branched.branched.__qamomile_adjoint"
    )
    opcodes = [operation.opcode for operation in undo.body.operations]
    reference = [
        operation.opcode
        for operation in next(
            f
            for f in lower_adjoint_calls(program).functions
            if f.symbol == "native_branched.branched.__qamomile_adjoint"
        ).body.operations
    ]

    assert opcodes == reference
    assert opcodes == [
        "qcore.reshape",
        "qcore.unpack",
        "qcore.adjoint",
        "qcore.tuple",
        "qcore.reshape",
    ]


def test_native_adjoint_lowering_recursively_inverts_ordinary_calls() -> None:
    """A wrapper inverse calls a shared typed inverse instead of expanding it."""

    source = """
import qamomile as qm

@qm.qkernel
def step(
    state: qm.Q[qm.Bits[1]],
    theta: qm.Param[qm.Real],
) -> qm.Q[qm.Bits[1]]:
    return qm.rz(qm.h(state), theta)

@qm.qkernel
def wrapper(
    state: qm.Q[qm.Bits[1]],
    theta: qm.Param[qm.Real],
) -> qm.Q[qm.Bits[1]]:
    return step(state, theta)

@qm.qkernel
def undo(
    state: qm.Q[qm.Bits[1]],
    theta: qm.Param[qm.Real],
) -> qm.Q[qm.Bits[1]]:
    return qm.adjoint(wrapper, state, theta)
"""
    program = qm.build_source(source, module_name="native_nested").specialize("undo")
    payload = bytes(native.lower_adjoint_calls_for_target(program.to_bytes()))
    package = qcore_wire.QCorePackageProto()
    package.ParseFromString(payload)
    undo = next(
        function
        for function in package.functions
        if function.symbol == "native_nested.step.__qamomile_adjoint"
    )
    originals = [
        next(
            attribute.value.string_value
            for attribute in operation.attributes
            if attribute.name == "original"
        )
        for operation in undo.body.operations
    ]

    assert originals == ["quantum.rz", "quantum.h"]
    wrapper = next(
        f
        for f in package.functions
        if f.symbol == "native_nested.wrapper.__qamomile_adjoint"
    )
    [call] = wrapper.body.operations
    assert call.opcode == "qcore.call"
    assert (
        next(a.value.string_value for a in call.attributes if a.name == "callee")
        == undo.symbol
    )


def test_native_adjoint_lowering_cancels_a_nested_callable_adjoint() -> None:
    """Taking a callable adjoint twice restores one typed ordinary call."""

    source = """
import qamomile as qm

@qm.qkernel
def step(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(state)

@qm.qkernel
def inverse(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.adjoint(step, state)

@qm.qkernel
def restore(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.adjoint(inverse, state)
"""
    program = qm.build_source(source, module_name="native_double").specialize("restore")
    payload = bytes(native.lower_adjoint_calls_for_target(program.to_bytes()))
    package = qcore_wire.QCorePackageProto()
    package.ParseFromString(payload)
    restore = next(
        function
        for function in package.functions
        if function.symbol == "native_double.inverse.__qamomile_adjoint"
    )

    assert [operation.opcode for operation in restore.body.operations] == ["qcore.call"]
    callee = next(
        attribute.value.string_value
        for attribute in restore.body.operations[0].attributes
        if attribute.name == "callee"
    )
    assert callee == "native_double.step"


def test_native_adjoint_lowering_retains_deep_call_graph_without_expansion() -> None:
    """A deep call graph creates bounded shared definitions without Rust recursion."""

    functions = [
        """
@qm.qkernel
def f0(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(state)
"""
    ]
    functions.extend(
        f"""
@qm.qkernel
def f{index}(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return f{index - 1}(state)
"""
        for index in range(1, 257)
    )
    functions.append(
        """
@qm.qkernel
def undo(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.adjoint(f256, state)
"""
    )
    source = "import qamomile as qm\n" + "\n".join(functions)
    program = qm.build_source(source, module_name="native_deep").specialize("undo")

    payload = bytes(native.lower_adjoint_calls_for_target(program.to_bytes()))
    package = qcore_wire.QCorePackageProto.FromString(payload)
    helpers = [f for f in package.functions if ".__qamomile_adjoint" in f.symbol]
    assert len(helpers) == 257
    assert all(len(f.body.operations) == 1 for f in helpers)
    assert sum(f.body.operations[0].opcode == "qcore.call" for f in helpers) == 256
    assert native.inspect_target_adjoint(payload).verified


def test_native_bridge_owns_portable_observable_algebra() -> None:
    """The bridge normalizes terms and multiplies Pauli phases in Rust."""

    from qamomile.observable import (
        _decode_observable_payload,
        _observable_payload,
    )

    x = qm.X(0)
    y = qm.Y(0)
    raw = _observable_payload(
        ((x.terms[0][0], 2 + 0j), (x.terms[0][0], -1 - 0j)),
        -0j,
        0,
    )
    normalized, hermitian = native.normalize_observable(raw)
    terms, constant, width = _decode_observable_payload(bytes(normalized))
    affine, affine_hermitian = native.affine_observable(
        bytes(normalized),
        (0.0, 1.0),
        (2.0, -1.0),
    )
    combined, combined_hermitian = native.add_scaled_observable(
        x._native_literal,
        y._native_literal,
        (-2.0, 0.0),
    )
    remapped, remapped_hermitian = native.remap_observable(
        x._native_literal,
        ((0, 3),),
    )
    product, product_hermitian = native.multiply_observables(
        _observable_payload(x.terms, x.constant, x.num_qubits),
        _observable_payload(y.terms, y.constant, y.num_qubits),
    )

    assert terms == x.terms
    assert constant.real.hex() == (0.0).hex()
    assert constant.imag.hex() == (0.0).hex()
    assert width == 1
    assert hermitian is True
    assert _decode_observable_payload(bytes(affine)) == (
        ((x.terms[0][0], 1j),),
        2 - 1j,
        1,
    )
    assert affine_hermitian is False
    assert _decode_observable_payload(bytes(combined)) == (
        ((x.terms[0][0], 1 + 0j), (y.terms[0][0], -2 + 0j)),
        0j,
        1,
    )
    assert combined_hermitian is True
    assert _decode_observable_payload(bytes(remapped)) == (
        ((qm.X(3).terms[0][0], 1 + 0j),),
        0j,
        4,
    )
    assert remapped_hermitian is True
    assert native.observable_is_pairwise_commuting(bytes(normalized)) is True
    assert native.pauli_words_commute(((0, "X"),), ((0, "Z"),)) is False
    assert native.pauli_words_commute(((0, "X"),), ((1, "Z"),)) is True
    assert native.plan_pauli_realization(
        _observable_payload(x.terms, x.constant, x.num_qubits),
        1,
    )[1] == [("h", [0], 1.0), ("rz", [0], 2.0), ("h", [0], 1.0)]
    with pytest.raises(ValueError, match="Q8038.*exceeds state width"):
        native.plan_pauli_realization(
            _observable_payload(x.terms, x.constant, x.num_qubits),
            0,
        )
    assert _decode_observable_payload(bytes(product)) == (
        ((qm.Z(0).terms[0][0], 1j),),
        0j,
        1,
    )
    assert product_hermitian is False

    one_norm = native.observable_lcu_one_norm(
        _observable_payload(
            ((x.terms[0][0], 3 + 4j),),
            -2 + 0j,
            1,
        )
    )
    assert one_norm == 7.0


def test_native_bridge_owns_dense_hermitian_arithmetic() -> None:
    """Dense matrix arithmetic preserves Rust-owned failures and coordinates."""

    x = [(0.0, 0.0), (1.0, 0.0), (1.0, 0.0), (0.0, 0.0)]
    identity = [(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0, 0.0)]

    assert native.add_scaled_hermitian_matrices((2, 2), x, (2, 2), identity, -2.0) == [
        (-2.0, 0.0),
        (1.0, 0.0),
        (1.0, 0.0),
        (-2.0, 0.0),
    ]
    assert native.scale_hermitian_matrix((2, 2), x, 2.0, True) == [
        (0.0, 0.0),
        (0.5, 0.0),
        (0.5, 0.0),
        (0.0, 0.0),
    ]
    with pytest.raises(ValueError, match="Q7417.*Shape mismatch"):
        native.add_scaled_hermitian_matrices((2, 2), x, (1, 1), identity[:1], 1.0)
    with pytest.raises(ZeroDivisionError, match="Q7416"):
        native.scale_hermitian_matrix((2, 2), x, 0.0, True)


@qm.qkernel
def _native_runtime_qsvt_sample(
    phases: qm.Param[qm.Vector[qm.Real, 3]],
) -> qm.Shot[qm.Bits[2]]:
    """Measure a reusable degree-two QSVT circuit family."""

    return qm.measure(_NATIVE_RUNTIME_QSVT(qm.qubits(2), phases))


@qm.qkernel
def _native_runtime_qaoa_sample(
    hamiltonian: qm.Const[qm.Observable],
    gammas: qm.Param[qm.Vector[qm.Real, 1]],
    betas: qm.Param[qm.Vector[qm.Real, 1]],
) -> qm.Shot[qm.Bits[2]]:
    """Measure one reusable QAOA layer over a bound commuting Hamiltonian."""

    return qm.measure(qaoa_state(2, 1, hamiltonian, gammas, betas))


@qm.qkernel
def _native_qft_roundtrip_sample() -> qm.Shot[qm.Bits[3]]:
    """Exercise a dependent-width modular adjoint through ordinary source."""

    state = qft(qm.prepare((1, 0, 1)), 3)
    return qm.measure(inverse_qft(state, 3))


@qm.qkernel
def _native_static_amplitude_sample() -> qm.Shot[qm.Bits[1]]:
    """Exercise immutable aggregate coordinates in state preparation."""

    return qm.measure(_NATIVE_STATIC_AMPLITUDE(qm.qubits(1)))


@qm.qkernel
def _native_sparse_lookup_sample() -> qm.Shot[qm.Bits[6]]:
    """Exercise partial sibling regrouping in coherent sparse lookup."""

    address, column, value = _NATIVE_SPARSE_LOOKUP(
        qm.prepare((1, 0)),
        qm.qubits(2),
        qm.qubits(2),
    )
    return qm.measure(qm.concat(address, column, value))


@qm.qkernel
def _native_runtime_fqaoa_sample(
    hamiltonian: qm.Const[qm.Observable],
    givens_angles: qm.Param[qm.Vector[qm.Real, 1]],
    hopping: qm.Param[qm.Real],
    gammas: qm.Param[qm.Vector[qm.Real, 1]],
    betas: qm.Param[qm.Vector[qm.Real, 1]],
) -> qm.Shot[qm.Bits[2]]:
    """Measure a reusable particle-preserving FQAOA circuit family."""

    state = fqaoa_state(
        2,
        1,
        1,
        [(0, 1)],
        givens_angles,
        1,
        hamiltonian,
        hopping,
        gammas,
        betas,
    )
    return qm.measure(state)


def _program_bytes() -> bytes:
    """Build one canonical Python-authored Program."""

    source = """
import qamomile as qm

@qm.qkernel
def answer() -> qm.Int:
    return 42
"""
    return (
        qm.build_source(source, module_name="native_bridge")
        .specialize("answer")
        .to_bytes()
    )


def _dependent_program_bytes() -> bytes:
    """Build one closed program containing an ordinary dependent call."""

    source = """
import qamomile as qm

@qm.qkernel
def allocate(width: qm.Const[qm.Int]) -> qm.Q[qm.Bits[width]]:
    return qm.qubits(width)

@qm.qkernel
def main() -> qm.Q[qm.Bits[3]]:
    return allocate(3)
"""
    return (
        qm.build_source(source, module_name="native_dependent")
        .specialize("main")
        .to_bytes()
    )


def _bound_dependent_program_bytes() -> bytes:
    """Build a dependent call instantiated by the entrypoint Const ABI."""

    source = """
import qamomile as qm

@qm.qkernel
def allocate(width: qm.Const[qm.Int]) -> qm.Q[qm.Bits[width]]:
    return qm.qubits(width)

@qm.qkernel
def main(width: qm.Const[qm.Int]) -> qm.Q[qm.Bits[width]]:
    return allocate(width)
"""
    return (
        qm.build_source(source, module_name="native_bound_dependent")
        .specialize("main", **{"width": 3})
        .to_bytes()
    )


def _gate_cleanup_program_bytes() -> bytes:
    """Build one cleanup chain containing exact primitive inverse steps."""

    source = """
import qamomile as qm

@qm.qkernel
def cleanup() -> qm.Unit:
    state = qm.qubits(1)
    state = qm.h(state)
    state = qm.t(state)
    return None
"""
    return (
        qm.build_source(source, module_name="native_gate_cleanup")
        .specialize("cleanup")
        .to_bytes()
    )


def _native_scalar_data(circuit: object, index: int) -> object:
    """Decode one bridge scalar node into the Python circuit JSON shape."""

    scalar = circuit.scalars[index]
    if scalar.kind == "bool":
        return scalar.bool_value
    if scalar.kind == "int":
        return scalar.int_value
    if scalar.kind == "real":
        return scalar.real_value
    if scalar.kind == "measurement":
        return {"measurement": scalar.measurement_index}
    if scalar.kind == "loop_value":
        return {"loop_value": scalar.loop_index, "type": scalar.scalar_type}
    if scalar.kind == "call_value":
        return {"call_value": scalar.call_index, "type": scalar.scalar_type}
    if scalar.kind == "parameter":
        return {
            "parameter": scalar.name,
            "path": [
                component.index if component.kind == "index" else component.field
                for component in scalar.path
            ],
        }
    if scalar.kind == "apply":
        return {
            "expr": scalar.opcode,
            "operands": [
                _native_scalar_data(circuit, operand) for operand in scalar.operands
            ],
            "attributes": {
                attribute.name: _native_scalar_attribute_data(attribute)
                for attribute in scalar.attributes
            },
        }
    raise AssertionError(f"unknown native scalar kind {scalar.kind!r}")


def _native_scalar_attribute_data(attribute: object) -> object:
    """Decode one typed bridge scalar attribute."""

    if attribute.kind == "int":
        return attribute.int_value
    if attribute.kind == "string":
        return attribute.string_value
    if attribute.kind == "strings":
        return list(attribute.string_values)
    raise AssertionError(f"unknown native scalar attribute kind {attribute.kind!r}")


def _native_instruction_data(circuit: object) -> list[tuple[object, ...]]:
    """Normalize typed bridge instructions for Python differential checks."""

    def parameter_data(parameter: object) -> object:
        """Normalize one typed circuit parameter."""

        if parameter.kind == "measurement":
            return parameter.index
        if parameter.kind == "scalar":
            return _native_scalar_data(circuit, parameter.index)
        if parameter.kind == "call_result":
            return {
                "call_result": parameter.index,
                "type": parameter.scalar_type,
            }
        raise AssertionError(f"unknown native parameter kind {parameter.kind!r}")

    return [
        (
            instruction.opcode,
            instruction.qubits,
            [parameter_data(parameter) for parameter in instruction.parameters],
        )
        for instruction in circuit.instructions
    ]


def test_native_bridge_accepts_python_program() -> None:
    """Rust and Python must agree on the persistent program boundary."""

    payload = _program_bytes()
    header = native.inspect_program(payload)

    assert header.schema_major == 5
    assert header.schema_minor == 0
    assert header.module_name == "native_bridge"
    assert header.entrypoint == "native_bridge.answer"
    assert header.function_count == 1
    assert header.type_count >= 1
    assert header.operation_count >= 1
    assert len(header.package_digest) == hashlib.sha256().digest_size * 2
    assert header.verified is True
    assert header.discharged_cleanup_count == 0
    assert header.entry_kind == "classical"
    assert header.entry_effects == []
    assert header.entry_capabilities == []
    assert header.entry_requirements == []
    assert header.entry_control_parameters == []
    assert len(header.contracts) == 1
    assert header.contracts[0].symbol == "native_bridge.answer"
    assert header.contracts[0].kind == header.entry_kind


def test_direct_program_construction_crosses_non_executable_native_typestate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transient Program rewrites are verified by Rust before use."""

    calls: list[bytes] = []
    inspect_specialization = native.inspect_specialization

    def recording_inspection(payload: bytes) -> object:
        """Record and delegate one specialization-request inspection."""

        calls.append(payload)
        return inspect_specialization(payload)

    monkeypatch.setattr(native, "inspect_specialization", recording_inspection)
    program = qm.Program.from_bytes(_program_bytes())
    rebuilt = replace(program)

    assert rebuilt == program
    assert len(calls) == 1
    header = inspect_specialization(calls[0])
    assert header.verified is True
    assert header.entrypoint == program.entrypoint
    assert header.pending_cleanup_obligation_count == 0
    assert header.contracts[0].kind == "classical"


def test_native_lowering_planner_selects_each_transient_route() -> None:
    """Rust owns direct, family, materialized, and unsupported selection."""

    program = qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def sample(enabled: qm.Param[qm.Bool]) -> qm.Shot[qm.Bits[1]]:
    state = qm.qubits(1)
    if enabled:
        state = qm.x(state)
    return qm.measure(state)
""",
        module_name="native_lowering_plan",
    ).specialize("sample")
    required = list(native.inspect_program(program.to_bytes()).entry_requirements)
    supported = [item for item in required if item != "host_dispatch"]

    direct = native.plan_target_lowering(program.to_bytes(), required, [], True)
    family = native.plan_target_lowering(
        program.to_bytes(), supported, ["host_dispatch"], True
    )
    materialized = native.plan_target_lowering(
        program.to_bytes(), supported, ["host_dispatch"], False
    )
    unsupported = native.plan_target_lowering(program.to_bytes(), supported, [], True)

    assert direct.route == "direct"
    assert family.route == "circuit_family"
    assert family.materialized == ["host_dispatch"]
    assert materialized.route == "launch_materialized"
    assert unsupported.route == "unsupported"
    assert unsupported.unsupported == ["host_dispatch"]
    with pytest.raises(ValueError, match="Q6303.*rotation"):
        native.plan_target_lowering(program.to_bytes(), required, ["rotation"], False)


def test_native_bridge_derives_artifact_abis_without_python() -> None:
    """Rust owns launch-slot filtering and nested sampled-result width."""

    import qamomile.compilation as compilation

    module = qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def sample(
    choose_one: qm.Const[qm.Bool],
    enabled: qm.Param[qm.Bool] = True,
) -> tuple[qm.Bits[1], qm.Bit]:
    first = qm.measure(qm.prepare((1,)))
    second = qm.measure(qm.prepare((1,)))[0]
    return first, second
""",
        module_name="native_artifact_abi",
    )
    program = module.specialize("sample", **{"choose_one": True})
    parameter_payload, result_payload = native.derive_artifact_abi(program.to_bytes())
    parameter_abi = qcore_wire.ParameterABIProto.FromString(parameter_payload)
    result_abi = qcore_wire.ResultABIProto.FromString(result_payload)

    assert [slot.name for slot in parameter_abi.slots] == ["enabled"]
    assert decode_type(parameter_abi.slots[0].type) == qm.Param[qm.Bool]
    assert decode_literal(parameter_abi.slots[0].default_value) is True
    assert result_abi.source_order == [0, 1]
    assert result_abi.native_width == 2
    assert decode_type(result_abi.type) == program.entry.result_type

    assert not hasattr(compilation, "_sample_leaf_types")
    assert compilation.ParameterABI.from_program(program).slots[0].name == "enabled"
    assert compilation.ResultABI.from_program(program).width == 2


def test_installed_native_artifact_abi_boundary_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale native distribution cannot reactivate Python ABI authority."""

    import qamomile.compilation as compilation

    program = qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def answer(value: qm.Param[qm.Int]) -> qm.Int:
    return value
""",
        module_name="stale_native_artifact_abi",
    ).specialize("answer")
    monkeypatch.delattr(native, "derive_artifact_abi")

    with pytest.raises(qm.ArtifactError, match="Q7013"):
        compilation.ParameterABI.from_program(program)


def test_native_result_codec_reconstructs_typed_values_without_python() -> None:
    """Every backend shares Rust's nested word and fixed-point result codec."""

    import qamomile.compilation as compilation

    abi = compilation.ResultABI(
        tuple[qm.Bits[2], qm.UFixed[0, 3]],
        (0, 1, 2, 3, 4),
        5,
    )
    assert not hasattr(compilation, "_decode_sample_value")
    assert not hasattr(compilation, "_encode_sample_value")
    assert not hasattr(compilation, "value_conforms")

    bits = (1, 0, 0, 0, 1)
    assert abi.decode_value(bits) == ((1, 0), 0.125)
    assert abi.encode_value(((1, 0), 0.125)) == bits

    boolean_abi = compilation.ResultABI(qm.Bool, (0,), 1)
    with pytest.raises(qm.ExecutionError, match="Q8143"):
        boolean_abi.encode_value(1)


def test_native_result_codec_projects_sdk_text_order() -> None:
    """The direct bridge owns SDK bit order, validation, and ABI projection."""

    import qamomile.compilation as compilation

    abi = compilation.ResultABI(qm.Bits[2], (2, 0), 3)

    assert native.decode_native_result_bitstring(abi._to_bytes(), "110", True) == [
        0,
        1,
    ]
    assert native.decode_native_result_bitstring(abi._to_bytes(), "1 10", False) == [
        1,
        0,
    ]
    with pytest.raises(ValueError, match="Q8103"):
        native.decode_native_result_bitstring(abi._to_bytes(), "1", False)
    with pytest.raises(ValueError, match="Q8104"):
        native.decode_native_result_bitstring(abi._to_bytes(), "1x0", True)


def test_native_bridge_normalizes_logical_sample_counts() -> None:
    """Rust derives canonical rows, exact shots, and deterministic mode once."""

    large = 1 << 100
    summary = native.normalize_sample_counts(
        [([1, 0], str(large)), ([0, 1], str(large)), ([0, 0], "0")],
        "2",
    )

    assert summary.entries == [
        ([0, 0], "0"),
        ([0, 1], str(large)),
        ([1, 0], str(large)),
    ]
    assert summary.shots == str(1 << 101)
    assert summary.most_likely == [1, 0]
    assert summary.probability(bytes((0, 1))) == 0.5
    assert summary.probability(bytes((1, 1))) == 0.0
    assert summary.expectation_z([0]) == 0.0

    import qamomile.compilation as compilation

    abi = compilation.ResultABI(qm.Bits[2], (0, 1), 2)
    typed = summary.decode_values(abi._to_bytes())
    assert [count for _, count in typed] == ["0", str(large), str(large)]
    assert all(isinstance(payload, bytes) for payload, _ in typed)
    with pytest.raises(ValueError, match="Q8107"):
        native.normalize_sample_counts([([0], "-1")], "1")
    with pytest.raises(ValueError, match="Q8113"):
        native.normalize_sample_counts([([0], "1")], "2")


def test_installed_native_result_codec_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale extension cannot silently restore Python result semantics."""

    import qamomile.compilation as compilation

    abi = compilation.ResultABI(qm.Bits[1], (0,), 1)
    monkeypatch.delattr(native, "decode_result_value")

    with pytest.raises(qm.ExecutionError, match="Q8144"):
        abi.decode_value((1,))

    monkeypatch.delattr(native, "encode_result_value")
    with pytest.raises(qm.ExecutionError, match="Q8144"):
        abi.encode_value((1,))


def test_native_bridge_accepts_an_unselected_dependent_module() -> None:
    """Rust verifies an open module without inventing Const bindings."""

    source = """
import qamomile as qm

@qm.qkernel
def allocate(width: qm.Const[qm.Int]) -> qm.Q[qm.Bits[width]]:
    return qm.qubits(width)
"""
    module = qm.build_source(source, module_name="native_module")
    payload = module.to_bytes()
    header = native.inspect_module(payload)

    assert header.schema_major == 5
    assert header.schema_minor == 0
    assert header.module_name == "native_module"
    assert header.function_count == 1
    assert header.operation_count >= 1
    assert len(header.package_digest) == hashlib.sha256().digest_size * 2
    assert header.verified is True
    assert header.pending_cleanup_obligation_count == 0
    assert len(header.contracts) == 1
    contract = header.contracts[0]
    assert contract.symbol == "native_module.allocate"
    assert contract.kind == "isometry"
    assert contract.effects == ["quantum", "allocation"]
    assert contract.capabilities == []
    assert contract.requirements == ["quantum"]
    with pytest.raises(ValueError, match="expected .*qamomile.program"):
        native.inspect_program(payload)
    with pytest.raises(ValueError, match="expected .*qamomile.module"):
        native.inspect_module(module.specialize("allocate", **{"width": 2}).to_bytes())


def test_source_build_crosses_the_native_typed_hir_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every parsed source module is checked as canonical typed HIR in Rust."""

    calls: list[bytes] = []
    lower_hir = native.lower_hir

    def recording_inspection(payload: bytes) -> object:
        """Record and delegate one native HIR lowering request."""

        calls.append(payload)
        return lower_hir(payload)

    monkeypatch.setattr(native, "lower_hir", recording_inspection)
    qm.build_source(
        """
import qamomile as qm

VALUE: qm.Int = 41

@qm.qkernel
def answer() -> qm.Int:
    value = VALUE + 1
    return value
""",
        module_name="native_hir_boundary",
    )

    assert len(calls) == 1
    package = hir_wire.HIRPackageProto.FromString(calls[0])
    header = native.inspect_hir(calls[0])
    assert package.artifact_kind == "qamomile.hir"
    assert package.module_name == "native_hir_boundary"
    assert len(package.package_digest) == hashlib.sha256().digest_size
    assert header.module_name == "native_hir_boundary"
    assert header.function_count == 1
    assert header.constant_count == 1
    assert header.statement_count == 2
    assert header.expression_count >= 4


def test_source_build_propagates_a_native_typed_hir_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Rust/Python HIR disagreement fails before ownership-SSA lowering."""

    def reject_hir(_payload: bytes) -> object:
        """Simulate one native HIR grammar rejection."""

        raise ValueError("Q7009: simulated typed HIR rejection")

    monkeypatch.setattr(native, "lower_hir", reject_hir)
    with pytest.raises(qm.VerificationError, match="Q7009"):
        qm.build_source(
            """
import qamomile as qm

@qm.qkernel
def answer() -> qm.Int:
    return 42
""",
            module_name="native_hir_rejection",
        )


def test_native_typed_hir_rejects_digest_tampering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ephemeral compiler request has one integrity-checked encoding."""

    calls: list[bytes] = []
    inspect_hir = native.inspect_hir
    lower_hir = native.lower_hir

    def recording_inspection(payload: bytes) -> object:
        """Retain the canonical request emitted by the source frontend."""

        calls.append(payload)
        return lower_hir(payload)

    monkeypatch.setattr(native, "lower_hir", recording_inspection)
    qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def answer() -> qm.Int:
    return 42
""",
        module_name="native_hir_tamper",
    )
    package = hir_wire.HIRPackageProto.FromString(calls[0])
    package.module_name = "forged"

    with pytest.raises(ValueError, match="Q7007"):
        inspect_hir(package.SerializeToString(deterministic=True))


def test_native_typed_hir_rejects_a_resealed_forged_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Canonical bytes cannot assert a type outside Rust's closed universe."""

    calls: list[bytes] = []
    inspect_hir = native.inspect_hir
    lower_hir = native.lower_hir

    def recording_inspection(payload: bytes) -> object:
        """Retain the canonical request emitted by the source frontend."""

        calls.append(payload)
        return lower_hir(payload)

    monkeypatch.setattr(native, "lower_hir", recording_inspection)
    qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def answer() -> qm.Int:
    return 42
""",
        module_name="native_hir_type_tamper",
    )
    package = hir_wire.HIRPackageProto.FromString(calls[0])
    package.functions[0].body.statements[0].expression.type.name = "Forged"
    package.package_digest = b""
    canonical = package.SerializeToString(deterministic=True)
    package.package_digest = hashlib.sha256(canonical).digest()

    with pytest.raises(ValueError, match="Q7010"):
        inspect_hir(package.SerializeToString(deterministic=True))


def test_native_hir_rederives_names_types_and_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Known but forged elaboration facts fail Rust QCore verification."""

    calls: list[bytes] = []
    lower_hir = native.lower_hir

    def recording_lowering(payload: bytes) -> bytes:
        """Retain and lower the canonical frontend request."""

        calls.append(payload)
        return bytes(lower_hir(payload))

    def reseal(package: hir_wire.HIRPackageProto) -> bytes:
        """Restore one modified request's canonical integrity digest."""

        package.package_digest = b""
        canonical = package.SerializeToString(deterministic=True)
        package.package_digest = hashlib.sha256(canonical).digest()
        return package.SerializeToString(deterministic=True)

    monkeypatch.setattr(native, "lower_hir", recording_lowering)
    qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def answer(value: qm.Param[qm.Int]) -> qm.Int:
    result = value + 1
    return result
""",
        module_name="native_hir_semantic_authority",
    )
    original = calls[0]

    forged_name = hir_wire.HIRPackageProto.FromString(original)
    forged_name.functions[0].body.statements[0].expression.operands[0].attributes[
        0
    ].value.string_value = "missing"
    with pytest.raises(ValueError, match="Q7102.*unresolved HIR name"):
        lower_hir(reseal(forged_name))

    forged_type = hir_wire.HIRPackageProto.FromString(original)
    forged_type.functions[0].body.statements[0].expression.type.arguments[
        0
    ].name = "Real"
    with pytest.raises(ValueError, match="Q7102.*malformed classical.add"):
        lower_hir(reseal(forged_type))

    forged_stage = hir_wire.HIRPackageProto.FromString(original)
    forged_stage.functions[0].body.statements[0].expression.type.name = "Const"
    with pytest.raises(ValueError, match="Q7102.*malformed classical.add"):
        lower_hir(reseal(forged_stage))


def test_native_hir_lowers_straight_line_classical_source_without_python() -> None:
    """The migrated classical subset does not invoke Python ownership lowering."""

    module = qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def answer() -> qm.Int:
    value = 40 + 2
    return value
""",
        module_name="native_hir_classical_lowering",
    )

    assert "classical.add" in module.inspect()
    assert qm.evaluate(module.specialize("answer")) == 42


def test_native_hir_unsupported_result_is_authoritative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An installed native compiler cannot redirect valid HIR to Python."""

    def reject_native_hir(_payload: bytes) -> bytes:
        """Report one explicit native compiler coverage failure."""

        raise ValueError("Q7101: native HIR lowering does not support test")

    monkeypatch.setattr(native, "lower_hir", reject_native_hir)
    with pytest.raises(qm.CapabilityError, match="Q7101"):
        qm.build_source(
            """
import qamomile as qm

@qm.qkernel
def answer() -> qm.Int:
    return 42
""",
            module_name="native_hir_authority",
        )


def test_native_hir_lowers_structured_quantum_if_without_python() -> None:
    """A normally yielding quantum branch crosses only the Rust HIR lowerer."""

    program = qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def sample() -> qm.Bits[1]:
    state = qm.qubits(1)
    if True:
        state = qm.x(state)
    return qm.measure(state)
""",
        module_name="native_hir_quantum_if",
    ).specialize("sample")

    assert qm.evaluate(program) == (1,)


def test_native_hir_lowers_dynamic_if_stages_and_ownership() -> None:
    """Param control stages classical results and threads one quantum owner."""

    module = qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def choose(enabled: qm.Param[qm.Bool]) -> qm.Int:
    value = 0
    if enabled:
        value = 1
    else:
        value = 2
    return value

@qm.qkernel
def sample(enabled: qm.Param[qm.Bool]) -> qm.Bits[1]:
    state = qm.qubits(1)
    if enabled:
        state = qm.x(state)
    return qm.measure(state)

@qm.qkernel
def nested(
    first: qm.Param[qm.Bool],
    second: qm.Param[qm.Bool],
) -> qm.Int:
    value = 0
    if first:
        if second:
            value = 1
        else:
            value = 2
    else:
        value = 3
    return value
""",
        module_name="native_hir_dynamic_if",
    )

    choose = module.specialize("choose")
    nested = module.specialize("nested")
    sample = module.specialize("sample")
    assert (
        qm.run(choose, on=qm.Reference(), parameters={"enabled": False}).result.value
        == 2
    )
    assert (
        qm.run(choose, on=qm.Reference(), parameters={"enabled": True}).result.value
        == 1
    )
    assert (
        qm.run(
            nested,
            on=qm.Reference(),
            parameters={"first": True, "second": False},
        ).result.value
        == 2
    )
    assert (
        qm.run(
            nested,
            on=qm.Reference(),
            parameters={"first": False, "second": True},
        ).result.value
        == 3
    )
    assert qm.run(
        sample,
        on=qm.Reference(),
        parameters={"enabled": False},
        shots=4,
    ).result.counts == {(0,): 4}
    assert qm.run(
        sample,
        on=qm.Reference(),
        parameters={"enabled": True},
        shots=4,
    ).result.counts == {(1,): 4}


def test_native_hir_lowers_branch_local_return_without_python() -> None:
    """A dynamic branch return crosses only the Rust HIR lowerer."""

    module = qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def choose(enabled: qm.Param[qm.Bool]) -> qm.Int:
    if enabled:
        return 1
    return 2
""",
        module_name="native_hir_branch_return",
    )
    program = module.specialize("choose")

    assert qm.evaluate(program, enabled=True) == 1
    assert qm.evaluate(program, enabled=False) == 2


def test_native_hir_lowers_single_owner_quantum_source_without_python() -> None:
    """Straight-line primitive ownership and cleanup are lowered in Rust."""

    module = qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def sample() -> qm.Bits[1]:
    state = qm.x(qm.qubits(1))
    return qm.measure(state)

@qm.qkernel
def cleanup() -> qm.Unit:
    state = qm.h(qm.qubits(1))
    return None
""",
        module_name="native_hir_quantum_lowering",
    )

    assert qm.evaluate(module.specialize("sample")) == (1,)
    cleanup = module.specialize("cleanup")
    assert "qcore.forget" not in cleanup.inspect()
    assert qm.evaluate(cleanup) is None


def test_native_hir_lowers_ordinary_calls_without_python() -> None:
    """Classical, fresh, threaded, and multi-owner calls lower in Rust."""

    module = qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def add_one(value: qm.Int) -> qm.Int:
    return value + 1

@qm.qkernel
def allocate() -> qm.Q[qm.Bits[1]]:
    return qm.qubits(1)

@qm.qkernel
def answer() -> qm.Int:
    return add_one(41)

@qm.qkernel
def flip(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(state)

@qm.qkernel
def pair(
    left: qm.Q[qm.Bits[1]],
    right: qm.Q[qm.Bits[1]],
) -> tuple[qm.Q[qm.Bits[1]], qm.Q[qm.Bits[1]]]:
    return (left, right)

@qm.qkernel
def sample() -> qm.Bits[1]:
    return qm.measure(flip(allocate()))

@qm.qkernel
def pair_sample() -> qm.Bits[2]:
    left = qm.prepare((1,))
    right = qm.qubits(1)
    left, right = pair(left, right)
    return qm.measure(qm.concat(left, right))
""",
        module_name="native_hir_ordinary_calls",
    )

    assert module.inspect().count("qcore.call") == 4
    assert qm.evaluate(module.specialize("answer")) == 42
    assert qm.evaluate(module.specialize("sample")) == (1,)
    assert qm.evaluate(module.specialize("pair_sample")) == (1, 0)


def test_native_hir_lowers_single_owner_adjoint_calls_without_python() -> None:
    """An exact inverse call reuses one owner and the native adjoint ABI."""

    module = qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def rotate(state: qm.Q[qm.Bits[1]], theta: qm.Real) -> qm.Q[qm.Bits[1]]:
    return qm.rx(state, theta)

@qm.qkernel
def sample() -> qm.Bits[1]:
    state = qm.qubits(1)
    state = rotate(state, 0.75)
    state = qm.adjoint(rotate, state, 0.75)
    return qm.measure(state)
""",
        module_name="native_hir_adjoint_call",
    )

    assert "qcore.adjoint_call" in module.inspect()
    assert qm.evaluate(module.specialize("sample")) == (0,)


def test_native_hir_lowers_coherent_control_and_select_without_python() -> None:
    """Control and select pack disjoint owners under a fresh verified root."""

    module = qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def flip(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(state)

@qm.qkernel
def phase(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.z(state)

@qm.qkernel
def controlled() -> qm.Bits[1]:
    controls = qm.prepare((1,))
    target = qm.qubits(1)
    controls, target = qm.control(flip, controls, target)
    qm.discard(controls)
    return qm.measure(target)

@qm.qkernel
def selected() -> qm.Bits[1]:
    selector = qm.prepare((1,))
    target = qm.qubits(1)
    selector, target = qm.multiplex(selector, (phase, flip), target)
    qm.discard(selector)
    return qm.measure(target)
""",
        module_name="native_hir_coherent_calls",
    )

    inspection = module.inspect()
    assert "qcore.control_call" in inspection
    assert "qcore.select_call" in inspection
    assert inspection.count("qcore.unpack") == 2
    assert qm.evaluate(module.specialize("controlled")) == (1,)
    assert qm.evaluate(module.specialize("selected")) == (1,)


def test_native_hir_lowers_owned_aggregate_structure_without_python() -> None:
    """Tuple, unpack, tensor, and factor preserve one verified owner tree."""

    module = qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def sample() -> qm.Bits[2]:
    left = qm.prepare((1,))
    right = qm.qubits(1)
    state = qm.concat(left, right)
    left, right = qm.split_at(state, 1)
    return qm.measure(qm.concat(left, right))
""",
        module_name="native_hir_owned_aggregate_structure",
    )

    inspection = module.inspect()
    assert inspection.count("qcore.tuple") == 3
    assert inspection.count("qcore.reshape") == 3
    assert "qcore.unpack" in inspection
    assert qm.evaluate(module.specialize("sample")) == (1, 0)


def test_native_hir_lowers_lazy_conditional_expressions_without_python() -> None:
    """Short-circuit, staged, and quantum expression branches lower in Rust."""

    module = qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def short_and() -> qm.Bool:
    return False and (1 // 0 == 0)

@qm.qkernel
def short_or() -> qm.Bool:
    return True or (1 // 0 == 0)

@qm.qkernel
def choose(enabled: qm.Param[qm.Bool]) -> qm.Int:
    return 7 if enabled else 9

@qm.qkernel
def sample(enabled: qm.Const[qm.Bool]) -> qm.Bits[1]:
    state = qm.qubits(1)
    state = qm.x(state) if enabled else qm.z(state)
    return qm.measure(state)
""",
        module_name="native_hir_lazy_if",
    )

    assert qm.evaluate(module.specialize("short_and")) is False
    assert qm.evaluate(module.specialize("short_or")) is True
    choose = module.specialize("choose")
    assert (
        qm.run(choose, on=qm.Reference(), parameters={"enabled": True}).result.value
        == 7
    )
    assert (
        qm.run(choose, on=qm.Reference(), parameters={"enabled": False}).result.value
        == 9
    )
    assert qm.evaluate(module.specialize("sample", **{"enabled": True})) == (1,)
    assert qm.evaluate(module.specialize("sample", **{"enabled": False})) == (0,)


def test_native_hir_lowers_normally_yielding_for_without_python() -> None:
    """Range and fixed aggregate loops lower with stages and owners in Rust."""

    module = qm.build_source(
        """
from typing import NamedTuple

import qamomile as qm


class Pair(NamedTuple):
    left: qm.Int
    right: qm.Int


@qm.qkernel
def total(stop: qm.Param[qm.Int]) -> qm.Int:
    result = 0
    for index in range(stop):
        result += index
    return result

@qm.qkernel
def array_total() -> qm.Int:
    result = 0
    for value in [1, 2, 3]:
        result += value
    return result

@qm.qkernel
def product_total() -> qm.Int:
    result = 0
    for value in (1, 2, 3):
        result += value
    return result

@qm.qkernel
def record_total(pair: qm.Param[Pair]) -> qm.Int:
    result = 0
    for value in pair:
        result += value
    return result

@qm.qkernel
def last_bit(bits: qm.Param[qm.Bits[3]]) -> qm.Param[qm.Bit]:
    result = bits[0]
    for bit in bits:
        result = bit
    return result

@qm.qkernel
def symbolic_total(
    size: qm.Const[qm.Int],
    values: qm.Const[qm.Vector[qm.Int, size]],
) -> qm.Int:
    result = 0
    for value in values:
        result += value
    return result

@qm.qkernel
def matrix_sample(
    angles: qm.Param[qm.Matrix[qm.Real, 2, 2]],
) -> qm.Bits[1]:
    state = qm.qubits(1)
    for row in angles:
        for angle in row:
            state = qm.rx(state, angle)
    return qm.measure(state)

@qm.qkernel
def sample() -> qm.Bits[1]:
    state = qm.qubits(1)
    for _ in range(2):
        state = qm.x(state)
    else:
        state = qm.x(state)
    return qm.measure(state)
""",
        module_name="native_hir_range_for",
    )

    total = module.specialize("total")
    assert qm.run(total, on=qm.Reference(), parameters={"stop": 4}).result.value == 6
    assert qm.evaluate(module.specialize("array_total")) == 6
    assert qm.evaluate(module.specialize("product_total")) == 6
    assert (
        qm.run(
            module.specialize("record_total"),
            on=qm.Reference(),
            parameters={"pair": {"left": 4, "right": 5}},
        ).result.value
        == 9
    )
    record_payload = native.evaluate_reference_value_at(
        module.specialize("record_total").to_bytes(),
        {"pair": {"right": 5, "left": 4}},
    )
    record_literal = qcore_wire.LiteralProto()
    record_literal.ParseFromString(record_payload)
    assert decode_literal(record_literal) == 9
    assert (
        qm.run(
            module.specialize("last_bit"),
            on=qm.Reference(),
            parameters={"bits": (0, 1, 0)},
        ).result.value
        == 0
    )
    assert (
        qm.evaluate(module.specialize("symbolic_total", **{"values": (1, 2, 3)})) == 6
    )
    assert qm.run(
        module.specialize("matrix_sample"),
        on=qm.Reference(),
        parameters={"angles": ((3.141592653589793, 0.0), (0.0, 0.0))},
        shots=4,
        seed=2,
    ).result.counts == {(1,): 4}
    assert qm.evaluate(module.specialize("sample")) == (1,)


def test_native_module_retains_cleanup_obligations_until_specialization() -> None:
    """Open modules keep Forget typed while executable programs discharge it."""

    source = """
import qamomile as qm

@qm.qkernel
def cleanup() -> qm.Unit:
    state = qm.qubits(1)
    state = qm.h(state)
    return None
"""
    module = qm.build_source(source, module_name="native_module_cleanup")
    module_header = native.inspect_module(module.to_bytes())
    program_header = native.inspect_program(module.specialize("cleanup").to_bytes())

    assert module_header.pending_cleanup_obligation_count == 1
    assert program_header.discharged_cleanup_count == 1


def test_source_build_automatically_crosses_the_native_module_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successful source compilation is accepted by Rust before it returns."""

    calls: list[bytes] = []
    inspect_module = native.inspect_module

    def recording_inspection(payload: bytes) -> object:
        """Record and delegate one native module inspection."""

        calls.append(payload)
        return inspect_module(payload)

    monkeypatch.setattr(native, "inspect_module", recording_inspection)
    module = qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def answer() -> qm.Int:
    return 42
""",
        module_name="native_build_boundary",
    )

    assert calls == [module.to_bytes()]
    calls.clear()
    restored = qm.Module.from_bytes(module.to_bytes())
    assert restored == module
    assert calls == [module.to_bytes()]


def test_source_module_has_no_python_contract_inference() -> None:
    """Rust is the sole callable-contract authority for source modules."""

    import qamomile.ir as ir

    assert not hasattr(ir, "infer_function_contracts")
    module = qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def answer() -> qm.Int:
    return 42
""",
        module_name="native_contract_authority",
    )

    assert module.function("answer").kind == "classical"
    assert module.function("answer").effects == frozenset()


def test_native_program_decode_has_no_python_contract_inference() -> None:
    """Rust is the sole callable-contract authority for decoded programs."""

    import qamomile.ir as ir

    module = qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def answer() -> qm.Int:
    return 42
""",
        module_name="native_program_contract_authority",
    )

    assert not hasattr(ir, "infer_function_contracts")
    program = module.specialize("answer")

    assert program.entry.kind == "classical"
    assert qm.Program.from_bytes(program.to_bytes()).entry.effects == frozenset()


def test_source_build_propagates_a_native_module_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Rust/Python semantic disagreement fails closed at source build."""

    def reject_module(_payload: bytes) -> object:
        """Simulate one native semantic rejection."""

        raise ValueError("Q4999: simulated native rejection")

    monkeypatch.setattr(native, "inspect_module", reject_module)
    with pytest.raises(qm.VerificationError, match="Q4999"):
        qm.build_source(
            """
import qamomile as qm

@qm.qkernel
def answer() -> qm.Int:
    return 42
""",
            module_name="native_rejection",
        )


def test_module_load_wraps_a_native_rejection_as_serialization_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Untrusted module bytes cannot bypass the Rust semantic boundary."""

    payload = qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def answer() -> qm.Int:
    return 42
""",
        module_name="native_load_rejection",
    ).to_bytes()

    def reject_module(_payload: bytes) -> object:
        """Simulate one native semantic rejection during deserialization."""

        raise ValueError("Q4999: simulated native rejection")

    monkeypatch.setattr(native, "inspect_module", reject_module)
    with pytest.raises(qm.SerializationError, match="Q4999"):
        qm.Module.from_bytes(payload)


def test_program_specialization_automatically_crosses_the_native_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Const type substitution and branch erasure are delegated to Rust."""

    calls: list[bytes] = []
    prepare_program = native.prepare_program

    def recording_specialization(payload: bytes) -> bytes:
        """Record and delegate one native specialization request."""

        calls.append(payload)
        return bytes(prepare_program(payload))

    monkeypatch.setattr(native, "prepare_program", recording_specialization)
    program = qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def choose(width: qm.Const[qm.Int], enabled: qm.Const[qm.Bool]) -> qm.Q[qm.Bits[width]]:
    state = qm.qubits(width)
    if enabled:
        state = qm.h(state, target=0)
    return state
""",
        module_name="native_specialization_boundary",
    ).specialize("choose", width=3, enabled=True)

    assert len(calls) == 1
    assert "qcore.if" not in program.inspect()
    assert "Q[Bits[3]]" in program.inspect()
    assert native.inspect_program(program.to_bytes()).verified is True


def test_native_selection_owns_defaults_extent_inference_and_linking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public source API sends one partial complete-module request to Rust."""

    from qamomile.proto import qcore_pb2 as wire

    calls: list[bytes] = []
    prepare_program = native.prepare_program

    def recording_selection(payload: bytes) -> bytes:
        """Record and delegate one native source selection request."""

        calls.append(payload)
        return bytes(prepare_program(payload))

    monkeypatch.setattr(native, "prepare_program", recording_selection)
    module = qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def helper(value: qm.Int) -> qm.Int:
    return value + 1

@qm.qkernel
def unused() -> qm.Int:
    return 100

@qm.qkernel
def selected(
    n: qm.Const[qm.Int],
    values: qm.Const[qm.Vector[qm.Int, n]],
    offset: qm.Const[qm.Int] = 4,
) -> qm.Int:
    return helper(values[0] + n + offset)
""",
        module_name="native_selection",
    )
    program = module.specialize("selected", **{"values": (2, 5, 8)})

    assert len(calls) == 1
    request = wire.QCorePackageProto.FromString(calls[0])
    assert request.artifact_kind == "qamomile.specialization"
    assert request.entrypoint == "selected"
    assert [function.symbol for function in request.functions] == [
        "native_selection.helper",
        "native_selection.selected",
        "native_selection.unused",
    ]
    assert [binding.name for binding in request.const_bindings] == ["values"]
    assert [function.symbol for function in program.functions] == [
        "native_selection.helper",
        "native_selection.selected",
    ]
    assert dict(program.const_bindings)["n"] == 3
    assert dict(program.const_bindings)["offset"] == 4
    assert qm.evaluate(program) == 10


@pytest.mark.parametrize(
    ("bindings", "diagnostic"),
    [
        ({}, "Q3002: missing Const bindings: n, values"),
        ({"values": (1, 2), "extra": 1}, "Q3003: unknown Const bindings: extra"),
        ({"values": (1, 2), "n": 3}, "Q3006: Const dimension 'n' expects 2"),
        ({"values": (1, "bad")}, "Q3006: Const binding 'values'"),
    ],
)
def test_native_selection_preserves_public_binding_diagnostics(
    bindings: dict[str, object],
    diagnostic: str,
) -> None:
    """Malformed partial bindings fail in Rust without Python selection."""
    module = qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def selected(
    n: qm.Const[qm.Int],
    values: qm.Const[qm.Vector[qm.Int, n]],
) -> qm.Int:
    return n
""",
        module_name="native_selection_diagnostics",
    )

    with pytest.raises(qm.SpecializationError, match=diagnostic):
        module.specialize("selected", **bindings)


def test_native_selection_validates_unreachable_functions_before_linking() -> None:
    """Reachability pruning cannot hide invalid code in the complete module."""

    from qamomile.proto import qcore_pb2 as wire
    from qamomile.serialization import encode_selection_request

    module = qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def selected() -> qm.Int:
    return 1

@qm.qkernel
def unreachable() -> qm.Int:
    return 2
""",
        module_name="native_selection_validation",
    )
    package = wire.QCorePackageProto.FromString(
        encode_selection_request(module, "selected", ())
    )
    unreachable = next(
        function
        for function in package.functions
        if function.symbol == "native_selection_validation.unreachable"
    )
    unreachable.body.operations[0].opcode = "classical.unknown"
    package.package_digest = b""
    canonical = package.SerializeToString(deterministic=True)
    package.package_digest = hashlib.sha256(canonical).digest()

    with pytest.raises(ValueError, match="Q1032: unknown QCore opcode"):
        native.prepare_program(package.SerializeToString(deterministic=True))


def test_program_specialization_propagates_a_native_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Rust specialization failure cannot silently use Python semantics."""

    def reject_specialization(_payload: bytes) -> bytes:
        """Simulate one native partial-evaluation rejection."""

        raise ValueError("Q3999: simulated native specialization rejection")

    monkeypatch.setattr(native, "prepare_program", reject_specialization)
    module = qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def answer(value: qm.Const[qm.Int]) -> qm.Int:
    return value + 1
""",
        module_name="native_specialization_rejection",
    )
    with pytest.raises(qm.SpecializationError, match="Q3999"):
        module.specialize("answer", **{"value": 41})


def test_straight_line_cleanup_is_synthesized_natively(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Primitive and modular-call inverse chains do not invoke Python cleanup."""

    calls: list[bytes] = []
    prepare_program = native.prepare_program

    def recording_preparation(payload: bytes) -> bytes:
        """Record and delegate one complete native preparation request."""

        calls.append(payload)
        return bytes(prepare_program(payload))

    monkeypatch.setattr(native, "prepare_program", recording_preparation)
    program = qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def step(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.h(state)

@qm.qkernel
def cleanup() -> qm.Unit:
    scratch = qm.qubits(1)
    scratch = step(scratch)
    inverse = qm.qubits(1)
    inverse = qm.adjoint(step, inverse)
    return None
""",
        module_name="native_cleanup_synthesis",
    ).specialize("cleanup")

    assert len(calls) == 1
    header = native.inspect_program(program.to_bytes())
    assert header.verified is True
    assert header.discharged_cleanup_count == 2
    assert "qcore.forget" not in program.inspect()
    assert "qcore.adjoint_call" in program.inspect()
    assert "qcore.call" in program.inspect()


def test_native_cleanup_rejects_without_python_synthesis_authority() -> None:
    """An unsupported cleanup producer remains a native capability failure."""

    with pytest.raises(qm.CapabilityError, match="Q6201"):
        qm.build_source(
            """
import qamomile as qm

@qm.qkernel
def cleanup() -> qm.Unit:
    state = qm.reset(qm.qubits(1))
    return None
""",
            module_name="native_cleanup_authority",
        ).specialize("cleanup")


def test_aggregate_specialization_stays_inside_native_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nested indexing and length fold without invoking Python compatibility."""

    calls: list[str] = []
    prepare_program = native.prepare_program

    def record_prepare(payload: bytes) -> bytes:
        """Record and delegate one complete native preparation."""

        calls.append("prepare")
        return bytes(prepare_program(payload))

    monkeypatch.setattr(native, "prepare_program", record_prepare)
    program = qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def pick(values: qm.Const[qm.Matrix[qm.Int, 2, 2]]) -> qm.Int:
    return values[-1][0] + len(values)
""",
        module_name="native_aggregate_specialization",
    ).specialize("pick", **{"values": ((4, 5), (7, 9))})

    assert calls == ["prepare"]
    assert qm.evaluate(program) == 9


def test_installed_native_preparation_rejection_is_authoritative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An installed native Q6202 fails without a Python specialization path."""

    module = qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def value() -> qm.Int:
    return 1
""",
        module_name="native_authoritative_preparation",
    )

    def reject_native_preparation(_payload: bytes) -> object:
        """Return one representative unfinished native rejection."""

        raise NotImplementedError("Q6202: unsupported native specialization")

    monkeypatch.setattr(native, "prepare_program", reject_native_preparation)

    with pytest.raises(qm.CapabilityError, match="Q6202"):
        module.specialize("value")


def test_invalid_exact_aggregate_index_fails_closed_in_native() -> None:
    """A selected out-of-bounds index retains the Q3008 public diagnostic."""

    module = qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def pick(
    values: qm.Const[qm.Vector[qm.Int, 2]],
    index: qm.Const[qm.Int],
) -> qm.Int:
    return values[index]
""",
        module_name="native_invalid_aggregate_index",
    )

    with pytest.raises(qm.SpecializationError, match="Q3008:.*index out of bounds"):
        module.specialize("pick", **{"values": (4, 5), "index": 2})


def test_nested_const_record_projection_uses_typed_declaration_order_natively(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Canonical mapping order cannot change NamedTuple field semantics."""

    calls: list[bytes] = []
    prepare_program = native.prepare_program

    def record_prepare(payload: bytes) -> bytes:
        """Record and delegate one complete native record specialization."""

        calls.append(payload)
        return bytes(prepare_program(payload))

    monkeypatch.setattr(native, "prepare_program", record_prepare)
    program = qm.build_source(
        """
from typing import NamedTuple
import qamomile as qm

class Coordinates(NamedTuple):
    zeta: qm.Int
    alpha: qm.Int

@qm.qkernel
def difference(values: qm.Const[qm.Vector[Coordinates, 2]]) -> qm.Int:
    return values[-1].zeta - values[0].alpha + len(values[-1])
""",
        module_name="native_const_record",
    ).specialize(
        "difference",
        **{
            "values": (
                {"alpha": 2, "zeta": 9},
                {"alpha": 4, "zeta": 11},
            )
        },
    )

    assert len(calls) == 1
    assert qm.evaluate(program) == 11


@pytest.mark.parametrize(
    ("annotation", "expression", "binding", "expected"),
    [
        ("qm.Int", "value * value + 1", 2**80, 2**160 + 1),
        ("qm.Complex", "value * (2 - 1j) + 3j", 1 + 2j, 4 + 6j),
        ("qm.Real", "value // 2.0 + value % 2.0", -7.5, -3.5),
    ],
)
def test_extended_numeric_tower_specializes_natively(
    monkeypatch: pytest.MonkeyPatch,
    annotation: str,
    expression: str,
    binding: object,
    expected: object,
) -> None:
    """BigInt, mixed Complex, and Real divmod share one native evaluator."""

    calls: list[bytes] = []
    prepare_program = native.prepare_program

    def record_prepare(payload: bytes) -> bytes:
        """Record and delegate one native numeric specialization."""

        calls.append(payload)
        return bytes(prepare_program(payload))

    monkeypatch.setattr(native, "prepare_program", record_prepare)
    program = qm.build_source(
        f"""
import qamomile as qm

@qm.qkernel
def calculate(value: qm.Const[{annotation}]) -> {annotation}:
    return {expression}
""",
        module_name="native_extended_numeric_tower",
    ).specialize("calculate", **{"value": binding})

    assert len(calls) == 1
    assert qm.evaluate(program) == expected


def test_complex_power_stays_in_the_declared_numeric_domain_natively() -> None:
    """An explicit Complex operand admits powers that leave the Real domain."""

    program = qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def square_root(value: qm.Const[qm.Complex]) -> qm.Complex:
    return value ** (0.5 + 0j)
""",
        module_name="native_complex_power",
    ).specialize("square_root", **{"value": -1 + 0j})

    assert qm.evaluate(program) == pytest.approx(1j)


@pytest.mark.parametrize(
    ("annotation", "return_annotation", "expression", "binding", "reason"),
    [
        ("qm.Int", "qm.Int", "1 << value", -1, "negative shift count"),
        (
            "qm.Int",
            "qm.Int",
            "1 << value",
            8 * 1024 * 1024,
            "left shift exceeds the QCore artifact budget",
        ),
        ("qm.Int", "qm.Int", "1 // value", 0, "division by zero"),
        ("qm.Int", "qm.Int", "2 ** value", -1, "negative integer exponent"),
        (
            "qm.Int",
            "qm.Int",
            "2 ** value",
            8 * 1024 * 1024,
            "integer power exceeds the QCore artifact budget",
        ),
        (
            "qm.Int",
            "qm.Real",
            "value + 1.0",
            1 << 4096,
            "integer is too large to convert to binary64",
        ),
        (
            "qm.Real",
            "qm.Real",
            "(-1.0) ** value",
            0.5,
            "Real power would produce a Complex result",
        ),
    ],
)
def test_extended_numeric_domain_errors_fail_closed_natively(
    annotation: str,
    return_annotation: str,
    expression: str,
    binding: object,
    reason: str,
) -> None:
    """Selected numeric exceptions retain Q3008 without compatibility."""

    module = qm.build_source(
        f"""
import qamomile as qm

@qm.qkernel
def calculate(value: qm.Const[{annotation}]) -> {return_annotation}:
    return {expression}
""",
        module_name="native_numeric_domain_error",
    )

    with pytest.raises(qm.SpecializationError, match=f"Q3008:.*{reason}"):
        module.specialize("calculate", **{"value": binding})


def test_structural_cleanup_is_synthesized_natively(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tensor-diagram inversion does not invoke Python cleanup compatibility."""

    calls: list[str] = []
    prepare_program = native.prepare_program

    def record_prepare(payload: bytes) -> bytes:
        """Record and delegate native structural cleanup synthesis."""

        calls.append("native")
        return bytes(prepare_program(payload))

    monkeypatch.setattr(native, "prepare_program", record_prepare)
    program = qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def rotate(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.h(state)

@qm.qkernel
def reverse(state: qm.Q[qm.Bits[2]]) -> qm.Q[qm.Bits[2]]:
    return qm.x(state, target=1)

@qm.qkernel
def cleanup() -> qm.Unit:
    state = qm.qubits(3)
    left, right = qm.split_at(state, 1)
    left = rotate(left)
    right = qm.adjoint(reverse, right)
    state = qm.concat(left, right)
    return None
""",
        module_name="native_structural_cleanup",
    ).specialize("cleanup")

    assert calls == ["native"]
    assert "qcore.forget" not in program.inspect()
    assert "qcore.unpack" in program.inspect()
    assert "qcore.tuple" in program.inspect()
    assert "qcore.adjoint_call" in program.inspect()
    header = native.inspect_program(program.to_bytes())
    assert header.verified is True
    assert header.discharged_cleanup_count == 1


def test_native_bridge_exposes_a_lexical_resource_and_lowering_inventory() -> None:
    """Rust owns one deterministic site census without unrolling loop work."""

    source = """
import qamomile as qm

@qm.qkernel
def step(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.h(state)

@qm.qkernel
def sample() -> qm.Bits[1]:
    state = qm.qubits(1)
    for _ in range(3):
        state = step(state)
    return qm.measure(state)
"""
    program = qm.build_source(source, module_name="native_structure").specialize(
        "sample"
    )
    payload = program.to_bytes()

    header = native.inspect_program(payload)
    structure = native.inspect_structure(payload)
    counts = native.inspect_exact_logical_counts(payload)

    assert structure.entrypoint == "native_structure.sample"
    assert structure.function_count == 2
    assert structure.operation_count == header.operation_count
    assert structure.block_count >= 3
    assert structure.region_count == 2
    assert structure.call_site_count == 1
    assert structure.opcodes["qcore.for"] == 1
    assert structure.opcodes["qcore.call"] == 1
    assert structure.opcodes["quantum.h"] == 1
    assert counts.work == 5
    assert counts.opcodes == {
        "quantum.alloc": 1,
        "quantum.h": 3,
        "quantum.measure": 1,
    }
    assert counts.peak_qubits == 1
    assert qm.resources.logical(program).metric(
        qm.resources.logical_operation_count("quantum.h")
    ).numeric_enclosure() == (3, 3)
    assert structure.function_opcodes["native_structure.step"] == {"quantum.h": 1}
    assert sum(structure.function_operations.values()) == structure.operation_count


def test_native_static_circuit_matches_python_for_an_inlined_bell_program() -> None:
    """Rust and Python must emit the same exact logical circuit slice."""

    source = """
import qamomile as qm

@qm.qkernel
def entangle(state: qm.Q[qm.Bits[2]]) -> qm.Q[qm.Bits[2]]:
    state = qm.h(state, target=0)
    return qm.cx(state, control=0, target=1)

@qm.qkernel
def bell() -> qm.Bits[2]:
    state = qm.qubits(2)
    state = entangle(state)
    return qm.measure(state)
"""
    program = qm.build_source(source, module_name="native_static_circuit").specialize(
        "bell"
    )

    native_circuit = native.inspect_static_circuit(program.to_bytes())
    python_circuit = lower_to_circuit(program)

    native_instructions = _native_instruction_data(native_circuit)
    python_instructions = [
        (instruction.opcode, list(instruction.qubits), list(instruction.parameters))
        for instruction in python_circuit.body.instructions
    ]
    assert native_circuit.width == python_circuit.width == 2
    assert native_circuit.measurement_count == 2
    assert (
        native_instructions
        == python_instructions
        == [
            ("h", [0], []),
            ("cx", [0, 1], []),
            ("measure", [0], [0]),
            ("measure", [1], [1]),
        ]
    )


def test_native_static_circuit_fails_closed_on_launch_parameters() -> None:
    """A symbolic launch parameter is not silently baked into one circuit."""

    source = """
import qamomile as qm

@qm.qkernel
def sample(target: qm.Param[qm.Int]) -> qm.Bits[2]:
    state = qm.qubits(2)
    state = qm.h(state, target=target)
    return qm.measure(state)
"""
    program = qm.build_source(
        source,
        module_name="native_dynamic_circuit",
    ).specialize("sample")

    with pytest.raises(ValueError, match="Q4301.*non-static selector.*target"):
        native.inspect_static_circuit(program.to_bytes())


def test_native_static_circuit_materializes_launch_parameter_selectors() -> None:
    """Topology dependence exposes exactly the selector needed at launch."""

    source = """
import qamomile as qm

@qm.qkernel
def sample(target: qm.Param[qm.Int], theta: qm.Param[qm.Real]) -> qm.Bits[2]:
    state = qm.qubits(2)
    state = qm.rx(state, theta, target=target)
    return qm.measure(state)
"""
    program = qm.build_source(
        source,
        module_name="native_dynamic_selector",
    ).specialize("sample")

    header = native.inspect_program(program.to_bytes())
    circuit = native.inspect_static_circuit_at(
        program.to_bytes(),
        {"target": 1},
    )

    assert header.entry_control_parameters == [0]
    assert [
        scalar.name for scalar in circuit.scalars if scalar.kind == "parameter"
    ] == ["theta"]
    assert _native_instruction_data(circuit)[0][:2] == ("rx", [1])


def test_native_static_circuit_materializes_dynamic_aggregate_paths() -> None:
    """Only an aggregate selector becomes control; leaf values stay runtime inputs."""

    source = """
import qamomile as qm

@qm.qkernel
def sample(
    values: qm.Param[qm.Matrix[qm.Real, 2, 2]],
    row: qm.Param[qm.Int],
    theta: qm.Param[qm.Real],
) -> qm.Bits[1]:
    state = qm.qubits(1)
    state = qm.rx(state, values[row, 0] + theta)
    return qm.measure(state)
"""
    program = qm.build_source(
        source,
        module_name="native_dynamic_aggregate_path",
    ).specialize("sample")

    header = native.inspect_program(program.to_bytes())
    circuit = native.inspect_static_circuit_at(program.to_bytes(), {"row": 1})

    assert "dynamic_classical_index" in header.entry_requirements
    assert header.entry_control_parameters == [1]
    parameters = [
        _native_scalar_data(circuit, index)
        for index, scalar in enumerate(circuit.scalars)
        if scalar.kind == "parameter"
    ]
    assert parameters == [
        {"parameter": "values", "path": [1, 0]},
        {"parameter": "theta", "path": []},
    ]


def test_native_static_circuit_uses_checked_scalar_helper_results() -> None:
    """Circuit selectors and resources share native scalar helper semantics."""

    source = """
import qamomile as qm

@qm.qkernel
def select_target(value: qm.Const[qm.Int]) -> qm.Int:
    return value - 2

@qm.qkernel
def sample() -> qm.Bits[2]:
    state = qm.qubits(2)
    state = qm.h(state, target=select_target(3))
    return qm.measure(state)
"""
    program = qm.build_source(source, module_name="native_scalar_circuit").specialize(
        "sample"
    )

    circuit = native.inspect_static_circuit(program.to_bytes())

    assert _native_instruction_data(circuit) == [
        ("h", [1], []),
        ("measure", [0], [0]),
        ("measure", [1], [1]),
    ]


def test_native_static_circuit_materializes_launch_parameter_for_bounds() -> None:
    """Exact scalar and aggregate launch bounds share one native entrypoint."""

    source = """
import qamomile as qm

@qm.qkernel
def sample(bounds: qm.Param[qm.Vector[qm.Int, 2]]) -> qm.Bits[1]:
    state = qm.qubits(1)
    for _ in range(bounds[1]):
        state = qm.x(state)
    return qm.measure(state)
"""
    program = qm.build_source(
        source,
        module_name="native_dynamic_for_circuit",
    ).specialize("sample")

    with pytest.raises(ValueError, match="Q4306.*qcore.for bounds"):
        native.inspect_static_circuit_family(program.to_bytes())

    circuit = native.inspect_static_circuit_at(
        program.to_bytes(),
        {"bounds": (100, 3)},
    )

    assert [instruction.opcode for instruction in circuit.instructions] == [
        "x",
        "x",
        "x",
        "measure",
    ]
    assert circuit.parameter_names == []


def test_native_validates_the_complete_launch_parameter_abi() -> None:
    """Structured targets share Rust name, type, and aggregate-shape checks."""

    source = """
import qamomile as qm

@qm.qkernel
def sample(
    bounds: qm.Param[qm.Vector[qm.Int, 2]],
    theta: qm.Param[qm.Real],
) -> qm.Bits[1]:
    state = qm.qubits(1)
    for _ in range(bounds[1]):
        state = qm.rx(state, theta)
    return qm.measure(state)
"""
    program = qm.build_source(
        source,
        module_name="native_complete_launch_abi",
    ).specialize("sample")

    assert (
        native.validate_launch_parameters(
            program.to_bytes(),
            {"bounds": (100, 3), "theta": 0.25},
        )
        is None
    )
    with pytest.raises(ValueError, match="parameter ABI"):
        native.validate_launch_parameters(
            program.to_bytes(),
            {"bounds": (100, 3)},
        )
    with pytest.raises(ValueError, match="aggregate ABI"):
        native.validate_launch_parameters(
            program.to_bytes(),
            {"bounds": (3,), "theta": 0.25},
        )


def test_native_circuit_family_lowers_boolean_host_control() -> None:
    """Rust derives exact static variants while retaining ordinary Params."""

    source = """
import qamomile as qm

@qm.qkernel
def sample(enabled: qm.Param[qm.Bool], theta: qm.Param[qm.Real]) -> qm.Bits[1]:
    state = qm.qubits(1)
    if enabled:
        state = qm.x(state)
    state = qm.rz(state, theta)
    return qm.measure(state)
"""
    program = qm.build_source(
        source,
        module_name="native_boolean_dispatch_family",
    ).specialize("sample")

    family = native.inspect_static_circuit_family(program.to_bytes())

    assert [
        (parameter.parameter_index, parameter.name)
        for parameter in family.dispatch_parameters
    ] == [(0, "enabled")]
    assert [
        [(decision.predicate_index, decision.value) for decision in variant.decisions]
        for variant in family.variants
    ] == [[(0, False)], [(0, True)]]
    assert [predicate.parameter_names for predicate in family.predicates] == [
        ["enabled"]
    ]
    assert [
        [instruction.opcode for instruction in variant.circuit.instructions]
        for variant in family.variants
    ] == [["rz", "measure"], ["x", "rz", "measure"]]
    for variant in family.variants:
        assert variant.circuit.parameter_names == ["theta"]


def test_native_circuit_family_retains_a_result_only_dispatch_parameter() -> None:
    """A selected host parameter remains available to result postprocessing."""

    source = """
import qamomile as qm

@qm.qkernel
def sample(enabled: qm.Param[qm.Bool]) -> qm.Shot[qm.Bool]:
    state = qm.qubits(1)
    if enabled:
        state = qm.x(state)
    bit, state = qm.measure_z(state, 0)
    qm.discard(state)
    return bit == enabled
"""
    program = qm.build_source(
        source,
        module_name="native_result_only_dispatch_parameter",
    ).specialize("sample")

    family = native.inspect_static_circuit_family(program.to_bytes())

    assert len(family.variants) == 2
    for variant in family.variants:
        assert variant.circuit.parameter_names == ["enabled"]


def test_native_circuit_family_lowers_an_integer_predicate() -> None:
    """Rust exports one closed scalar DAG instead of enumerating integer values."""

    source = """
import qamomile as qm

@qm.qkernel
def sample(limit: qm.Param[qm.Int]) -> qm.Bits[1]:
    state = qm.qubits(1)
    if limit > 0:
        state = qm.x(state)
    return qm.measure(state)
"""
    program = qm.build_source(
        source,
        module_name="native_integer_dispatch_predicate",
    ).specialize("sample")

    family = native.inspect_static_circuit_family(program.to_bytes())

    assert [
        (parameter.parameter_index, parameter.name)
        for parameter in family.dispatch_parameters
    ] == [(0, "limit")]
    assert len(family.predicates) == 1
    predicate = family.predicates[0]
    assert predicate.scalars[predicate.root].kind == "apply"
    assert [scalar.kind for scalar in predicate.scalars] == [
        "parameter",
        "int",
        "apply",
    ]
    assert [
        tuple(decision.value for decision in variant.decisions)
        for variant in family.variants
    ] == [(False,), (True,)]


def test_native_circuit_family_uses_transitive_source_order() -> None:
    """Dispatch keys follow entry ABI order through ordinary call control."""

    source = """
import qamomile as qm

@qm.qkernel
def toggle(state: qm.Q[qm.Bits[1]], enabled: qm.Param[qm.Bool]) -> qm.Q[qm.Bits[1]]:
    if enabled:
        state = qm.x(state)
    return state

@qm.qkernel
def sample(
    first: qm.Param[qm.Bool],
    theta: qm.Param[qm.Real],
    second: qm.Param[qm.Bool],
) -> qm.Bits[1]:
    state = qm.qubits(1)
    state = toggle(state, first)
    state = qm.rz(state, theta)
    state = toggle(state, second)
    return qm.measure(state)
"""
    program = qm.build_source(
        source,
        module_name="native_transitive_dispatch_family",
    ).specialize("sample")

    family = native.inspect_static_circuit_family(program.to_bytes())

    assert [parameter.name for parameter in family.dispatch_parameters] == [
        "first",
        "second",
    ]
    assert [
        tuple(decision.value for decision in variant.decisions)
        for variant in family.variants
    ] == [(False, False), (False, True), (True, False), (True, True)]


def test_native_circuit_family_enforces_variant_budget() -> None:
    """Finite host dispatch fails before exponential projection work escapes."""

    parameters = ", ".join(f"flag_{index}: qm.Param[qm.Bool]" for index in range(9))
    branches = "\n".join(
        f"    if flag_{index}:\n        state = qm.x(state)" for index in range(9)
    )
    source = f"""
import qamomile as qm

@qm.qkernel
def sample({parameters}) -> qm.Bits[1]:
    state = qm.qubits(1)
{branches}
    return qm.measure(state)
"""
    program = qm.build_source(
        source,
        module_name="native_dispatch_family_budget",
    ).specialize("sample")

    with pytest.raises(ValueError, match="more than 256 host-dispatch variants"):
        native.inspect_static_circuit_family(program.to_bytes())


def test_native_static_circuit_retains_hash_consed_symbolic_rotations() -> None:
    """Rotation angles retain one typed DAG and match Python circuit semantics."""

    source = """
import qamomile as qm

@qm.qkernel
def sample(theta: qm.Param[qm.Real]) -> qm.Bits[2]:
    state = qm.qubits(2)
    state = qm.p(state, 0.25, target=1)
    state = qm.rx(state, theta, target=0)
    state = qm.cp(state, -theta, control=0, target=1)
    state = qm.rzz(state, theta * 2.0, left=0, right=1)
    state = qm.global_phase(state, theta)
    return qm.measure(state)
"""
    program = qm.build_source(source, module_name="native_rotation_circuit").specialize(
        "sample"
    )

    native_circuit = native.inspect_static_circuit(program.to_bytes())
    python_circuit = lower_to_circuit(program)
    python_body = json.loads(python_circuit.to_bytes())["body"]

    assert _native_instruction_data(native_circuit) == [
        (instruction["op"], instruction["qubits"], instruction["parameters"])
        for instruction in python_body
    ]
    parameters = [
        scalar for scalar in native_circuit.scalars if scalar.kind == "parameter"
    ]
    assert len(parameters) == 1
    assert parameters[0].scalar_type == "Real"
    assert all(
        scalar.scalar_type == "Real"
        for scalar in native_circuit.scalars
        if scalar.kind == "apply"
    )


def test_native_static_circuit_retains_aggregate_parameter_paths() -> None:
    """Fixed array and record projections retain canonical typed leaf paths."""

    source = """
from typing import NamedTuple
import qamomile as qm

class Angles(NamedTuple):
    first: qm.Real
    second: qm.Real

@qm.qkernel
def sample(
    matrix: qm.Param[qm.Matrix[qm.Real, 2, 2]],
    record: qm.Param[Angles],
    pair: qm.Param[tuple[qm.Real, qm.Real]],
) -> qm.Bits[2]:
    state = qm.qubits(2)
    state = qm.rx(state, matrix[1, 0], target=0)
    state = qm.ry(state, matrix[-2, -1] + record.second, target=1)
    state = qm.rz(state, pair[0], target=0)
    return qm.measure(state)
"""
    program = qm.build_source(source, module_name="native_parameter_paths").specialize(
        "sample"
    )

    native_circuit = native.inspect_static_circuit(program.to_bytes())
    python_circuit = lower_to_circuit(program)
    python_body = json.loads(python_circuit.to_bytes())["body"]

    assert _native_instruction_data(native_circuit) == [
        (instruction["op"], instruction["qubits"], instruction["parameters"])
        for instruction in python_body
    ]
    parameters = [
        _native_scalar_data(native_circuit, index)
        for index, scalar in enumerate(native_circuit.scalars)
        if scalar.kind == "parameter"
    ]
    assert parameters == [
        {"parameter": "matrix", "path": [1, 0]},
        {"parameter": "matrix", "path": [0, 1]},
        {"parameter": "record", "path": ["second"]},
        {"parameter": "pair", "path": [0]},
    ]


def test_public_circuit_lowering_adopts_the_native_symbolic_projection() -> None:
    """The public Python lifecycle must use a successful native projection."""

    import qamomile.circuit as circuit_module

    source = """
import qamomile as qm

@qm.qkernel
def sample(theta: qm.Param[qm.Real]) -> qm.Bits[1]:
    state = qm.qubits(1)
    state = qm.ry(state, theta)
    return qm.measure(state)
"""
    program = qm.build_source(source, module_name="native_public_circuit").specialize(
        "sample"
    )

    circuit = circuit_module.lower_to_circuit(program)

    assert not hasattr(circuit_module, "_CircuitLowerer")
    assert circuit.width == 1
    assert circuit.result_abi.source_order == (0,)
    assert circuit.result_abi.native_width == 1
    assert circuit.parameter_abi.slots[0].name == "theta"
    assert json.loads(circuit.to_bytes())["body"][0] == {
        "condition": None,
        "controls": [],
        "op": "ry",
        "parameters": [{"parameter": "theta", "path": []}],
        "qubits": [0],
        "regions": [],
    }


def test_public_circuit_lowering_treats_native_rejection_as_authoritative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An installed native compiler rejection cannot enter a Python fallback."""

    import qamomile.circuit as circuit_module

    program = qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def sample() -> qm.Bits[1]:
    return qm.measure(qm.qubits(1))
""",
        module_name="native_authoritative_circuit",
    ).specialize("sample")

    def reject_native_projection(
        _payload: bytes,
        _realization_capabilities: list[str],
        _lowering: tuple[str, ...],
    ) -> object:
        """Return one representative native semantic rejection."""

        raise ValueError("Q4301: unsupported verified circuit")

    monkeypatch.setattr(native, "inspect_static_circuit", reject_native_projection)

    assert not hasattr(circuit_module, "_CircuitLowerer")
    with pytest.raises(qm.CapabilityError, match="Q4301"):
        circuit_module.lower_to_circuit(program)


def test_public_circuit_lowering_adopts_native_vector_parameters() -> None:
    """Public lowering uses native fixed-vector paths."""

    import qamomile.circuit as circuit_module

    source = """
import qamomile as qm

@qm.qkernel
def sample(angles: qm.Param[qm.Vector[qm.Real, 2]]) -> qm.Bits[1]:
    state = qm.qubits(1)
    state = qm.ry(state, angles[1])
    return qm.measure(state)
"""
    program = qm.build_source(
        source, module_name="native_public_vector_circuit"
    ).specialize("sample")

    circuit = circuit_module.lower_to_circuit(program)

    assert not hasattr(circuit_module, "_CircuitLowerer")
    assert json.loads(circuit.to_bytes())["body"][0]["parameters"] == [
        {"parameter": "angles", "path": [1]}
    ]


@pytest.mark.parametrize("enabled", [False, True])
def test_native_static_circuit_matches_specialized_control_flow(enabled: bool) -> None:
    """Static branches and fixed loops must match Python source-order emission."""

    source = """
import qamomile as qm

@qm.qkernel
def sample(enabled: qm.Const[qm.Bool]) -> qm.Bits[2]:
    state = qm.qubits(2)
    if enabled:
        state = qm.h(state, target=1)
    for target in range(2):
        state = qm.x(state, target=target)
    return qm.measure(state)
"""
    program = qm.build_source(source, module_name="native_control_circuit").specialize(
        "sample",
        **{"enabled": enabled},
    )

    native_circuit = native.inspect_static_circuit(program.to_bytes())
    python_circuit = lower_to_circuit(program)

    assert _native_instruction_data(native_circuit) == [
        (instruction.opcode, list(instruction.qubits), list(instruction.parameters))
        for instruction in python_circuit.body.instructions
    ]


def test_native_static_circuit_bounds_loop_expansion() -> None:
    """A fixed range cannot force unbounded native compile-time work."""

    source = """
import qamomile as qm

@qm.qkernel
def sample() -> qm.Bits[1]:
    state = qm.qubits(1)
    for _ in range(1_000_001):
        state = state
    return qm.measure(state)
"""
    program = qm.build_source(source, module_name="native_bounded_circuit").specialize(
        "sample"
    )

    with pytest.raises(ValueError, match="Q4301.*1000001 iterations"):
        native.inspect_static_circuit(program.to_bytes())


def test_native_static_circuit_matches_basis_and_measurement_channels() -> None:
    """Preparation, projection, partial measurement, and reset retain bit order."""

    source = """
import qamomile as qm

@qm.qkernel
def sample() -> qm.Bits[2]:
    state = qm.prepare((1, 0))
    projected, state = qm.measure_y(state, target=1)
    selected, state = qm.measure_z(state, 0)
    observed, state = qm.measure_reset(state)
    return qm.measure(state)
"""
    program = qm.build_source(source, module_name="native_measure_circuit").specialize(
        "sample"
    )

    native_circuit = native.inspect_static_circuit(program.to_bytes())
    counts = native.inspect_exact_logical_counts(program.to_bytes())
    python_circuit = lower_to_circuit(program)

    assert native_circuit.width == python_circuit.width == 2
    assert native_circuit.measurement_count == 6
    assert counts.measurements == 6
    assert counts.resets == 2
    assert counts.allocated_qubits == 2
    assert counts.circuit_qubits == 2
    assert _native_instruction_data(native_circuit) == [
        (instruction.opcode, list(instruction.qubits), list(instruction.parameters))
        for instruction in python_circuit.body.instructions
    ]


def test_native_static_circuit_samples_verified_programs_in_rust() -> None:
    """The native bridge materializes parameters and returns source-order counts."""

    source = """
import qamomile as qm

@qm.qkernel
def rotate(theta: qm.Param[qm.Real]) -> qm.Shot[qm.Bits[2]]:
    state = qm.qubits(2)
    state = qm.rx(state, theta, target=1)
    return qm.measure(state)

@qm.qkernel
def controlled(
    enabled: qm.Param[qm.Bool],
    theta: qm.Param[qm.Real],
) -> qm.Shot[qm.Bits[2]]:
    state = qm.qubits(2)
    if enabled:
        state = qm.x(state, target=0)
    state = qm.rx(state, theta, target=1)
    return qm.measure(state)

@qm.qkernel
def bell() -> qm.Shot[qm.Bits[2]]:
    state = qm.h(qm.qubits(2), target=0)
    state = qm.cx(state, control=0, target=1)
    return qm.measure(state)

@qm.qkernel
def prepare_one(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.h(state)

@qm.qkernel
def undo() -> qm.Shot[qm.Bits[1]]:
    state = qm.qubits(1)
    state = prepare_one(state)
    state = qm.adjoint(prepare_one, state)
    return qm.measure(state)

@qm.qkernel
def sealed_loop(rounds: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[1]]:
    state = qm.qubits(1)
    for _ in range(rounds):
        state = qm.x(state)
    return qm.measure(state)
"""
    module = qm.build_source(source, module_name="native_static_sampling")
    rotate = module.specialize("rotate")
    controlled = module.specialize("controlled")
    bell = module.specialize("bell")
    undo = module.specialize("undo")
    sealed_loop = module.specialize("sealed_loop", **{"rounds": 1})

    rotated = native.sample_static_circuit_at(
        rotate.to_bytes(),
        {"theta": 3.141592653589793},
        8,
        7,
    )
    assert {tuple(bits): count for bits, count in rotated} == {(0, 1): 8}

    selected = native.sample_static_circuit_at(
        controlled.to_bytes(),
        {"enabled": True, "theta": 3.141592653589793},
        8,
        7,
    )
    assert {tuple(bits): count for bits, count in selected} == {(1, 1): 8}

    first = native.sample_static_circuit_at(bell.to_bytes(), {}, 256, 19)
    second = native.sample_static_circuit_at(bell.to_bytes(), {}, 256, 19)
    counts = {tuple(bits): count for bits, count in first}
    assert first == second
    assert sum(counts.values()) == 256
    assert set(counts) <= {(0, 0), (1, 1)}

    inverted = native.sample_static_circuit_at(undo.to_bytes(), {}, 8, 5)
    assert {tuple(bits): count for bits, count in inverted} == {(0,): 8}

    sealed = native.sample_static_circuit_at(sealed_loop.to_bytes(), {}, 8, 5)
    assert {tuple(bits): count for bits, count in sealed} == {(1,): 8}

    target = native.lower_adjoint_calls_for_target(undo.to_bytes())
    with pytest.raises(ValueError, match="qamomile.program"):
        native.sample_static_circuit_at(
            target,
            {},
            1,
            5,
        )

    with pytest.raises(ValueError, match="Q4401.*shot limit"):
        native.sample_static_circuit_at(bell.to_bytes(), {}, 0, 0)
    with pytest.raises(ValueError, match="Q4301.*parameter ABI"):
        native.sample_static_circuit_at(rotate.to_bytes(), {}, 1, 0)


def test_native_reference_evaluates_structured_exact_values_in_rust() -> None:
    """Calls and structured control share the compiler's exact value algebra."""

    source = """
from typing import NamedTuple
import qamomile as qm

class Pair(NamedTuple):
    left: qm.Int
    right: qm.Int

@qm.qkernel
def add(left: qm.Int, right: qm.Int) -> qm.Int:
    return left + right

@qm.qkernel
def compute(
    limit: qm.Param[qm.Int],
    start: qm.Param[qm.Int],
) -> qm.Int:
    total = add(start, 1)
    for index in range(limit):
        if index % 2 == 0:
            total += index
        else:
            total -= 1
    for _ in range(6):
        if total >= 6:
            break
        total += 1
    return total

@qm.qkernel
def difference(pair: qm.Param[Pair]) -> qm.Int:
    return pair.left - pair.right
"""
    module = qm.build_source(source, module_name="native_reference_value")
    program = module.specialize("compute")

    payload = native.evaluate_reference_value_at(
        program.to_bytes(),
        {"limit": 4, "start": 0},
    )
    literal = qcore_wire.LiteralProto()
    literal.ParseFromString(payload)

    assert decode_literal(literal) == 6

    record_payload = native.evaluate_reference_value_at(
        module.specialize("difference").to_bytes(),
        {"pair": {"right": 2, "left": 7}},
    )
    literal.ParseFromString(record_payload)

    assert decode_literal(literal) == 5


def test_native_reference_selects_sampled_and_value_execution_from_abi() -> None:
    """One native boundary chooses its runtime from the derived result ABI."""

    module = qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def sample() -> qm.Shot[qm.Bits[1]]:
    return qm.measure(qm.x(qm.qubits(1)))

@qm.qkernel
def summarize(value: qm.Param[qm.Int]) -> qm.Int:
    qm.discard(qm.h(qm.qubits(1)))
    return value + 1
""",
        module_name="native_unified_reference_execution",
    )

    sampled = native.execute_reference_at(
        module.specialize("sample").to_bytes(),
        {},
        4,
        3,
    )
    assert sampled.kind == "samples"
    assert sampled.counts == [(b"\x01", 4)]
    assert sampled.value is None

    valued = native.execute_reference_at(
        module.specialize("summarize").to_bytes(),
        {"value": 6},
        4,
        3,
    )
    assert valued.kind == "value"
    assert valued.counts == []
    literal = qcore_wire.LiteralProto()
    literal.ParseFromString(valued.value)
    assert decode_literal(literal) == 7


def test_native_reference_value_errors_and_quantum_effects_are_disjoint() -> None:
    """Undefined arithmetic fails closed while quantum Unit executes natively."""

    source = """
import qamomile as qm

@qm.qkernel
def invalid(value: qm.Param[qm.Int]) -> qm.Real:
    return value / 0

@qm.qkernel
def quantum_value() -> qm.Unit:
    qm.discard(qm.qubits(1))
    return None
"""
    module = qm.build_source(source, module_name="native_reference_value_edges")

    with pytest.raises(ValueError, match="Q4502:.*division by zero"):
        native.evaluate_reference_value_at(
            module.specialize("invalid").to_bytes(),
            {"value": 1},
        )
    payload = native.evaluate_reference_value_at(
        module.specialize("quantum_value").to_bytes(),
        {},
    )
    literal = qcore_wire.LiteralProto()
    literal.ParseFromString(payload)
    assert decode_literal(literal) is None


def test_native_static_circuit_preserves_coherent_reshape_layout() -> None:
    """Factor and tensor reshape ownership without changing logical wire order."""

    source = """
import qamomile as qm

@qm.qkernel
def sample() -> qm.Bits[3]:
    state = qm.prepare((1, 0, 1))
    left, tail = qm.split_at(state, 1)
    middle, right = qm.split_at(tail, 1)
    middle = qm.h(middle)
    return qm.measure(qm.concat(left, middle, right))
"""
    program = qm.build_source(source, module_name="native_reshape_circuit").specialize(
        "sample"
    )

    native_circuit = native.inspect_static_circuit(program.to_bytes())
    python_circuit = lower_to_circuit(program)

    assert native_circuit.width == python_circuit.width == 3
    assert _native_instruction_data(native_circuit) == [
        (instruction.opcode, list(instruction.qubits), list(instruction.parameters))
        for instruction in python_circuit.body.instructions
    ]


def test_native_static_circuit_retains_indexed_coherent_controls() -> None:
    """Controlled calls retain selected wires and zero-basis conjugation."""

    source = """
import qamomile as qm

@qm.qkernel
def flip(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(state)

@qm.qkernel
def sample() -> qm.Bits[3]:
    controls = qm.prepare((1, 0))
    target = qm.qubits(1)
    controls, target = qm.control(
        flip,
        controls,
        target,
        at=1,
        control_state=0,
    )
    return qm.measure(qm.concat(controls, target))
"""
    program = qm.build_source(
        source, module_name="native_controlled_circuit"
    ).specialize("sample")

    native_projection = native.inspect_static_circuit(program.to_bytes())
    native_circuit = lower_to_circuit(program)

    assert native_circuit.width == 3
    controlled = [
        instruction
        for instruction in native_projection.instructions
        if instruction.controls
    ]
    assert len(controlled) == 1
    assert controlled[0].opcode == "x"
    assert controlled[0].qubits == [2]
    assert controlled[0].controls == [1]


def test_public_native_circuit_retains_runtime_qsvt_parameters() -> None:
    """Runtime QSVT uses one native circuit family with aggregate parameters."""

    program = _native_runtime_qsvt_sample.specialize()
    native_circuit = lower_to_circuit(program)

    body = json.loads(native_circuit.to_bytes())["body"]
    assert any(instruction["controls"] for instruction in body)
    payload = native_circuit.to_bytes()
    assert b'"parameter":"phases","path":[0]' in payload
    assert b'"parameter":"phases","path":[1]' in payload
    assert b'"parameter":"phases","path":[2]' in payload


def test_public_native_circuit_lowers_modular_adjoint() -> None:
    """Native adjoint replay reverses gates and symbolic angles exactly."""

    source = """
import qamomile as qm

@qm.qkernel
def transform(
    state: qm.Q[qm.Bits[1]],
    theta: qm.Real,
) -> qm.Q[qm.Bits[1]]:
    state = qm.h(state)
    state = qm.rx(state, theta)
    return qm.t(state)

@qm.qkernel
def sample(theta: qm.Param[qm.Real]) -> qm.Bits[1]:
    state = qm.qubits(1)
    state = qm.adjoint(transform, state, theta)
    return qm.measure(state)
"""
    program = qm.build_source(source, module_name="native_adjoint_circuit").specialize(
        "sample"
    )
    native_circuit = lower_to_circuit(program)

    body = json.loads(native_circuit.to_bytes())["body"]
    assert [instruction["op"] for instruction in body] == [
        "tdg",
        "rx",
        "h",
        "measure",
    ]
    assert body[1]["parameters"] == [
        {
            "attributes": {},
            "expr": "classical.neg",
            "operands": [{"parameter": "theta", "path": []}],
        }
    ]


def test_public_native_circuit_lowers_dependent_qft_adjoint() -> None:
    """Native QFT retains precise loop control and symbolic wire ownership."""

    program = _native_qft_roundtrip_sample.specialize()
    native_projection = native.inspect_static_circuit(program.to_bytes())
    native_circuit = lower_to_circuit(program)

    assert native_projection.width == 3
    opcodes = [
        instruction["op"]
        for instruction in json.loads(native_circuit.to_bytes())["body"]
    ]
    assert opcodes.count("h") == 6
    assert opcodes.count("cp") == 6
    assert opcodes.count("swap") == 2


def test_public_native_circuit_lowers_static_amplitude() -> None:
    """Native aggregate projection retains exact amplitude coordinates."""

    program = _native_static_amplitude_sample.specialize()
    native_projection = native.inspect_static_circuit(program.to_bytes())
    native_circuit = lower_to_circuit(program)

    assert native_projection.width == 1
    assert [
        instruction["op"]
        for instruction in json.loads(native_circuit.to_bytes())["body"]
    ] == ["ry", "rz", "global_phase", "measure"]


def test_public_native_circuit_lowers_sparse_lookup() -> None:
    """Partial sibling regrouping retains disjoint native ownership."""

    program = _native_sparse_lookup_sample.specialize()
    native_projection = native.inspect_static_circuit(program.to_bytes())
    native_circuit = lower_to_circuit(program)

    assert native_projection.width == 6
    assert native_circuit.width == 6
    assert qm.evaluate(program) == (1, 0, 0, 1, 1, 1)


def test_public_native_circuit_lowers_coherent_selection() -> None:
    """Native total selection composes every controlled callable exactly."""

    source = """
import qamomile as qm

@qm.qkernel
def unchanged(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return state

@qm.qkernel
def flipped(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(state)

@qm.qkernel
def sample() -> qm.Bits[2]:
    index = qm.prepare((1,))
    target = qm.qubits(1)
    index, target = qm.multiplex(index, (unchanged, flipped), target)
    return qm.measure(qm.concat(index, target))
"""
    program = qm.build_source(source, module_name="native_select_circuit").specialize(
        "sample"
    )
    native_circuit = lower_to_circuit(program)

    controlled = [
        instruction
        for instruction in json.loads(native_circuit.to_bytes())["body"]
        if instruction["controls"]
    ]
    assert [
        (instruction["op"], instruction["controls"]) for instruction in controlled
    ] == [("x", [0])]


def test_public_native_circuit_lowers_runtime_qaoa() -> None:
    """Runtime QAOA lowers its bound Pauli sum through the Rust projection."""

    hamiltonian = (
        0.25
        + 0.5 * (qm.X(0) * qm.X(1))
        - 0.75 * (qm.Y(0) * qm.Y(1))
        + 1.25 * (qm.Z(0) * qm.Z(1))
    )
    program = _native_runtime_qaoa_sample.specialize(**{"hamiltonian": hamiltonian})
    native_circuit = lower_to_circuit(program)

    payload = native_circuit.to_bytes()
    assert b'"parameter":"gammas","path":[0]' in payload
    assert b'"parameter":"betas","path":[0]' in payload
    body = json.loads(payload)["body"]
    assert {instruction["op"] for instruction in body} >= {
        "cx",
        "global_phase",
        "h",
        "rx",
        "rz",
        "s",
        "sdg",
    }


def test_public_native_circuit_lowers_runtime_fqaoa() -> None:
    """FQAOA retains all aggregate and scalar launch inputs in one circuit."""

    hamiltonian = 0.5 * (qm.Z(0) * qm.Z(1))
    program = _native_runtime_fqaoa_sample.specialize(**{"hamiltonian": hamiltonian})
    native_circuit = lower_to_circuit(program)

    payload = native_circuit.to_bytes()
    for name in ("givens_angles", "hopping", "gammas", "betas"):
        assert f'"parameter":"{name}"'.encode() in payload


def test_public_native_circuit_lowers_measurement_control() -> None:
    """Native structured if retains device conditions and nested blocks."""

    source = """
import qamomile as qm

@qm.qkernel
def sample() -> qm.Bits[2]:
    state = qm.prepare((1, 0))
    observed, state = qm.measure_z(state, 0)
    if not observed:
        state = qm.x(state, target=1)
    else:
        state = qm.h(state, target=1)
    return qm.measure(state)
"""
    program = qm.build_source(
        source, module_name="native_measurement_control"
    ).specialize("sample")
    native_circuit = lower_to_circuit(program)

    structured = [
        instruction
        for instruction in json.loads(native_circuit.to_bytes())["body"]
        if instruction["op"] == "if"
    ]
    assert len(structured) == 1
    assert structured[0]["condition"] == {
        "attributes": {},
        "expr": "classical.not",
        "operands": [{"measurement": 0}],
    }
    assert [[item["op"] for item in region] for region in structured[0]["regions"]] == [
        ["x"],
        ["h"],
    ]


def test_public_native_circuit_lowers_measurement_comparison() -> None:
    """Native scalar metadata preserves an ordinary Python comparison."""

    source = """
import qamomile as qm

@qm.qkernel
def sample() -> qm.Bits[1]:
    state = qm.qubits(1)
    observed, state = qm.measure_z(state, 0)
    if observed == False:
        state = qm.x(state)
    return qm.measure(state)
"""
    program = qm.build_source(
        source, module_name="native_measurement_comparison"
    ).specialize("sample")
    native_projection = native.inspect_static_circuit(program.to_bytes())
    native_circuit = lower_to_circuit(program)

    assert native_circuit.width == 1
    apply_indices = [
        index
        for index, scalar in enumerate(native_projection.scalars)
        if scalar.kind == "apply"
    ]
    assert len(apply_indices) == 1
    assert _native_scalar_data(
        native_projection,
        apply_indices[0],
    ) == {
        "expr": "classical.compare_chain",
        "operands": [{"measurement": 0}, False],
        "attributes": {"operators": ["eq"]},
    }


def test_public_native_circuit_lowers_unit_cleanup() -> None:
    """Canonical Unit results retain modular temporary cleanup natively."""

    source = """
import qamomile as qm

@qm.qkernel
def step(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.h(state)

@qm.qkernel
def sample() -> qm.Unit:
    scratch = qm.qubits(1)
    scratch = step(scratch)
    return None
"""
    program = qm.build_source(source, module_name="native_unit_cleanup").specialize(
        "sample"
    )
    native_projection = native.inspect_static_circuit(program.to_bytes())
    native_circuit = lower_to_circuit(program)

    assert native_projection.width == 1
    assert list(native_projection.result_measurements) == []
    assert [
        instruction["op"]
        for instruction in json.loads(native_circuit.to_bytes())["body"]
    ] == ["h", "h"]


def test_native_static_circuit_replays_direct_basis_cleanup() -> None:
    """Unused prepared scratch state is unprepared through verified cleanup."""

    source = """
import qamomile as qm

@qm.qkernel
def sample() -> qm.Bits[1]:
    scratch = qm.prepare((1, 0))
    output = qm.prepare((0,))
    return qm.measure(output)
"""
    program = qm.build_source(
        source, module_name="native_unprepare_circuit"
    ).specialize("sample")

    native_circuit = native.inspect_static_circuit(program.to_bytes())
    counts = native.inspect_exact_logical_counts(program.to_bytes())
    python_circuit = lower_to_circuit(program)

    assert counts.cleanup_work == 1
    assert _native_instruction_data(native_circuit) == [
        (instruction.opcode, list(instruction.qubits), list(instruction.parameters))
        for instruction in python_circuit.body.instructions
    ]


def test_native_exact_counts_fail_closed_on_runtime_control() -> None:
    """A Param branch is not collapsed into a fabricated native equality."""

    source = """
import qamomile as qm

@qm.qkernel
def sample(enabled: qm.Param[qm.Bool]) -> qm.Bits[1]:
    state = qm.qubits(1)
    if enabled:
        state = qm.h(state)
    return qm.measure(state)
"""
    payload = (
        qm.build_source(source, module_name="native_dynamic_resources")
        .specialize("sample")
        .to_bytes()
    )

    with pytest.raises(ValueError, match="Q4201.*dynamic qcore.if"):
        native.inspect_exact_logical_counts(payload)


@pytest.mark.parametrize(
    ("enabled", "expected"),
    [(False, 0), (True, 1)],
)
def test_native_exact_counts_select_only_the_static_branch(
    enabled: bool,
    expected: int,
) -> None:
    """A bound Const condition contributes only its selected dynamic work."""

    source = """
import qamomile as qm

@qm.qkernel
def sample(enabled: qm.Const[qm.Bool]) -> qm.Bits[1]:
    state = qm.qubits(1)
    if enabled:
        state = qm.h(state)
    return qm.measure(state)
"""
    payload = (
        qm.build_source(source, module_name="native_static_resources")
        .specialize("sample", **{"enabled": enabled})
        .to_bytes()
    )

    counts = native.inspect_exact_logical_counts(payload)

    assert counts.work == expected + 2
    assert counts.opcodes.get("quantum.h", 0) == expected


def test_native_exact_counts_interpret_changing_loop_carries() -> None:
    """A bounded loop carries exact values from one iteration to the next."""

    source = """
import qamomile as qm

@qm.qkernel
def sample() -> qm.Bits[1]:
    state = qm.qubits(1)
    enabled = True
    for _ in range(3):
        if enabled:
            state = qm.h(state)
        enabled = False
    return qm.measure(state)
"""
    payload = (
        qm.build_source(source, module_name="native_loop_carry_resources")
        .specialize("sample")
        .to_bytes()
    )

    counts = native.inspect_exact_logical_counts(payload)

    assert counts.work == 3
    assert counts.opcodes == {
        "quantum.alloc": 1,
        "quantum.h": 1,
        "quantum.measure": 1,
    }


def test_native_exact_counts_interpret_dependent_qft_loops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nested bounds derived from an induction value remain exact natively."""

    program = _native_qft_roundtrip_sample.specialize()
    counts = native.inspect_exact_logical_counts(program.to_bytes())
    decoded_study = qm.resources.logical(program)

    assert counts.work == 16
    assert counts.opcodes == {
        "qcore.adjoint": 7,
        "quantum.cp": 3,
        "quantum.h": 3,
        "quantum.measure": 1,
        "quantum.prepare": 1,
        "quantum.swap": 1,
    }
    assert decoded_study.metric(qm.resources.LOGICAL_WORK).numeric_enclosure() == (
        counts.work,
        counts.work,
    )


def test_native_exact_counts_retain_partial_qsvt_aggregate_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Static plan flags survive beside unknown runtime phase components."""

    program = _native_runtime_qsvt_sample.specialize()
    counts = native.inspect_exact_logical_counts(program.to_bytes())
    decoded_study = qm.resources.logical(program)

    assert counts.work == 10
    assert counts.opcodes == {
        "qcore.adjoint": 1,
        "quantum.alloc": 1,
        "quantum.global_phase": 6,
        "quantum.measure": 1,
        "quantum.x": 1,
    }
    assert decoded_study.metric(qm.resources.LOGICAL_WORK).numeric_enclosure() == (
        counts.work,
        counts.work,
    )


def test_native_exact_counts_preserve_proved_loop_invariants() -> None:
    """An unchanged Const carry may safely select the same branch every trip."""

    source = """
import qamomile as qm

@qm.qkernel
def sample(enabled: qm.Const[qm.Bool]) -> qm.Bits[1]:
    state = qm.qubits(1)
    for _ in range(3):
        if enabled:
            state = qm.h(state)
    return qm.measure(state)
"""
    payload = (
        qm.build_source(source, module_name="native_loop_invariant_resources")
        .specialize("sample", **{"enabled": True})
        .to_bytes()
    )

    counts = native.inspect_exact_logical_counts(payload)

    assert counts.work == 5
    assert counts.opcodes["quantum.h"] == 3


def test_native_exact_counts_scale_proved_large_loops() -> None:
    """An invariant million-trip body is summarized without expansion."""

    source = """
import qamomile as qm

@qm.qkernel
def sample() -> qm.Bits[1]:
    state = qm.qubits(1)
    for _ in range(1_000_000):
        state = qm.h(state)
    return qm.measure(state)
"""
    payload = (
        qm.build_source(source, module_name="native_large_loop_resources")
        .specialize("sample")
        .to_bytes()
    )

    counts = native.inspect_exact_logical_counts(payload)

    assert counts.work == 1_000_002
    assert counts.opcodes["quantum.h"] == 1_000_000


def test_native_exact_counts_skip_an_unreachable_empty_loop_body() -> None:
    """Dynamic control in a zero-trip body contributes no unsupported path."""

    source = """
import qamomile as qm

@qm.qkernel
def sample(enabled: qm.Param[qm.Bool]) -> qm.Bits[1]:
    state = qm.qubits(1)
    for _ in range(0):
        if enabled:
            state = qm.h(state)
    return qm.measure(state)
"""
    payload = (
        qm.build_source(source, module_name="native_empty_loop_resources")
        .specialize("sample")
        .to_bytes()
    )

    counts = native.inspect_exact_logical_counts(payload)

    assert counts.work == 2
    assert counts.opcodes == {"quantum.alloc": 1, "quantum.measure": 1}


def test_native_exact_counts_propagate_classical_helper_returns() -> None:
    """Ordinary classical helpers may select branches and fixed loop bounds."""

    source = """
import qamomile as qm

@qm.qkernel
def enabled(value: qm.Int) -> qm.Bool:
    return value * 2 == 6

@qm.qkernel
def trips(base: qm.Int) -> qm.Int:
    return base * 2 - 1

@qm.qkernel
def sample() -> qm.Bits[1]:
    state = qm.qubits(1)
    if enabled(3):
        for _ in range(trips(2)):
            state = qm.h(state)
    return qm.measure(state)
"""
    payload = (
        qm.build_source(source, module_name="native_helper_resources")
        .specialize("sample")
        .to_bytes()
    )

    counts = native.inspect_exact_logical_counts(payload)

    assert counts.work == 5
    assert counts.opcodes["quantum.h"] == 3


def test_native_exact_counts_compose_body_backed_adjoint_calls() -> None:
    """A verified modular adjoint contributes one inverse per callee quantum step."""

    source = """
import qamomile as qm

@qm.qkernel
def prepare_one(
    state: qm.Q[qm.Bits[1]],
    rounds: qm.Const[qm.Int],
) -> qm.Q[qm.Bits[1]]:
    for _ in range(rounds):
        state = qm.h(state)
    state = qm.t(state)
    return state

@qm.qkernel
def sample() -> qm.Bits[1]:
    state = qm.qubits(1)
    state = prepare_one(state, 1)
    state = qm.adjoint(prepare_one, state, 1)
    return qm.measure(state)
"""
    program = qm.build_source(
        source,
        module_name="native_adjoint_resources",
    ).specialize("sample")

    counts = native.inspect_exact_logical_counts(program.to_bytes())

    assert counts.work == 6
    assert counts.opcodes == {
        "qcore.adjoint": 2,
        "quantum.alloc": 1,
        "quantum.h": 1,
        "quantum.measure": 1,
        "quantum.t": 1,
    }
    assert counts.families == {
        "clifford_gates": 2,
        "gates": 4,
        "non_clifford_gates": 2,
        "single_qubit_gates": 4,
        "t_gates": 2,
    }
    assert qm.resources.logical(program).metric(
        qm.resources.logical_operation_count("qcore.adjoint")
    ).numeric_enclosure() == (2, 2)


def test_native_exact_counts_compose_control_and_select_bodies() -> None:
    """Control uses one callee body while total selection composes every case."""

    source = """
import qamomile as qm

@qm.qkernel
def unchanged(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return state

@qm.qkernel
def flipped(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(state)

@qm.qkernel
def sample() -> tuple[qm.Shot[qm.Bool], qm.Shot[qm.Bits[1]]]:
    index = qm.prepare(True)
    target = qm.qubits(1)
    index, target = qm.multiplex(index, (unchanged, flipped), target)
    index, target = qm.control(flipped, index, target, control_state=0)
    return qm.measure(index), qm.measure(target)
"""
    payload = (
        qm.build_source(source, module_name="native_coherent_resources")
        .specialize("sample")
        .to_bytes()
    )

    counts = native.inspect_exact_logical_counts(payload)

    assert counts.work == 6
    assert counts.opcodes == {
        "quantum.alloc": 1,
        "quantum.measure": 2,
        "quantum.prepare": 1,
        "quantum.x": 2,
    }
    assert counts.families is None


def test_public_logical_analysis_uses_one_native_symbolic_algebra() -> None:
    """Closed and open public studies share the native symbolic analyzer."""

    source = """
import qamomile as qm

@qm.qkernel
def sample() -> qm.Bits[1]:
    state = qm.qubits(1)
    for _ in range(3):
        state = qm.h(state)
    return qm.measure(state)
"""
    program = qm.build_source(
        source,
        module_name="native_public_resources",
    ).specialize("sample")

    counts = native.inspect_exact_logical_counts(program.to_bytes())
    decoded_study = qm.resources.logical(program)
    study = qm.resources.logical(program)

    assert "native-exact-logical-counts" not in study.derivations
    assert "native-symbolic-logical-counts" in study.derivations
    assert counts.families == {
        "clifford_gates": 3,
        "gates": 3,
        "single_qubit_gates": 3,
    }
    assert counts.measurements == 1
    assert counts.resets == 0
    assert study.metric(qm.resources.LOGICAL_WORK).numeric_enclosure() == (5, 5)
    assert study.metric(
        qm.resources.logical_operation_count("quantum.h")
    ).numeric_enclosure() == (3, 3)
    assert decoded_study.metric(qm.resources.LOGICAL_WORK).numeric_enclosure() == (
        counts.work,
        counts.work,
    )
    assert decoded_study.metric(
        qm.resources.logical_operation_count("quantum.h")
    ).numeric_enclosure() == (
        counts.opcodes["quantum.h"],
        counts.opcodes["quantum.h"],
    )
    assert study.metric(qm.resources.LOGICAL_CLIFFORD_GATES).numeric_enclosure() == (
        decoded_study.metric(qm.resources.LOGICAL_CLIFFORD_GATES).numeric_enclosure()
    )
    assert study.metric(qm.resources.LOGICAL_MEASUREMENTS).numeric_enclosure() == (
        decoded_study.metric(qm.resources.LOGICAL_MEASUREMENTS).numeric_enclosure()
    )


def test_native_symbolic_facts_are_lexical_and_source_local() -> None:
    """Repeated calls retain one canonical fact per lexical QCore operation."""

    from qamomile.resources.analysis import (
        _decode_native_resource_expressions,
        _decode_native_symbolic_count,
    )

    source = """
import qamomile as qm

@qm.qkernel
def step(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.h(state)

@qm.qkernel
def sample() -> qm.Bits[1]:
    state = qm.qubits(1)
    for _ in range(3):
        state = step(state)
    return qm.measure(state)
"""
    program = qm.build_source(
        source,
        module_name="native_source_local_resource_facts",
    ).specialize("sample")
    counts = native.inspect_symbolic_logical_counts(program.to_bytes())
    expressions = _decode_native_resource_expressions(counts.expressions)
    keys = [(fact.owner, fact.source, fact.opcode) for fact in counts.facts]
    study = qm.resources.logical(program)
    restored = qm.resources.ResourceStudy.from_bytes(study.to_bytes())
    public_keys = [
        (fact.owner, fact.source, fact.operation) for fact in restored._bundle.facts
    ]

    assert len(keys) == len(set(keys))
    assert public_keys == keys
    assert "native-source-local-facts" in restored.derivations
    h_facts = [fact for fact in counts.facts if fact.opcode == "quantum.h"]
    assert len(h_facts) == 1
    h_work = _decode_native_symbolic_count(h_facts[0].work, expressions)
    h_gates = _decode_native_symbolic_count(
        next(item.count for item in h_facts[0].families if item.family == "gates"),
        expressions,
    )
    assert h_work.lower.numeric() == h_work.upper.numeric() == 1
    assert h_gates.lower.numeric() == h_gates.upper.numeric() == 1


def test_native_exact_events_instantiate_a_callee_quantum_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ordinary call substitutes its concrete quantum argument width."""

    source = """
import qamomile as qm

@qm.qkernel
def instrument(
    state: qm.Q[qm.Bits[n]],
) -> tuple[qm.Bits[n], qm.Q[qm.Bits[n]]]:
    return qm.measure_reset(state)

@qm.qkernel
def sample() -> qm.Bits[3]:
    observed, state = instrument(qm.qubits(3))
    return qm.measure(state)
"""
    program = qm.build_source(
        source,
        module_name="native_dependent_exact_events",
    ).specialize("sample")

    counts = native.inspect_exact_logical_counts(program.to_bytes())
    study = qm.resources.logical(program)
    decoded_study = qm.resources.logical(program)

    assert counts.measurements == 6
    assert counts.resets == 3
    assert study.metric(qm.resources.LOGICAL_MEASUREMENTS).numeric_enclosure() == (
        6,
        6,
    )
    assert study.metric(qm.resources.LOGICAL_RESETS).numeric_enclosure() == (3, 3)
    assert (
        study.metric(qm.resources.LOGICAL_MEASUREMENTS).numeric_enclosure()
        == decoded_study.metric(qm.resources.LOGICAL_MEASUREMENTS).numeric_enclosure()
    )


@pytest.mark.parametrize(
    ("source", "bindings", "expected"),
    [
        (
            """
import qamomile as qm

@qm.qkernel
def sample(state: qm.Q[qm.Bits[3]]) -> qm.Bits[3]:
    return qm.measure(state)
""",
            {},
            3,
        ),
        (
            """
import qamomile as qm

@qm.qkernel
def sample(
    left: qm.Q[qm.Bits[2]],
    right: qm.Q[qm.Bits[3]],
) -> tuple[qm.Bits[2], qm.Bits[3]]:
    return (qm.measure(left), qm.measure(right))
""",
            {},
            5,
        ),
        (
            """
import qamomile as qm

@qm.qkernel
def sample(
    state: qm.Q[qm.Bits[width]],
    width: qm.Const[qm.Int],
) -> qm.Bits[width]:
    return qm.measure(state)
""",
            {"width": 4},
            4,
        ),
    ],
)
def test_native_exact_input_qubits_follow_the_entrypoint_type(
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    bindings: dict[str, int],
    expected: int,
) -> None:
    """Native input width sums fixed, product, and bound-dependent roots."""

    program = qm.build_source(
        source,
        module_name=f"native_input_width_{expected}",
    ).specialize("sample", **bindings)

    counts = native.inspect_exact_logical_counts(program.to_bytes())
    study = qm.resources.logical(program)
    decoded_study = qm.resources.logical(program)

    assert counts.input_qubits == expected
    assert study.metric(qm.resources.LOGICAL_INPUT_QUBITS).numeric_enclosure() == (
        expected,
        expected,
    )
    assert (
        study.metric(qm.resources.LOGICAL_INPUT_QUBITS).numeric_enclosure()
        == decoded_study.metric(qm.resources.LOGICAL_INPUT_QUBITS).numeric_enclosure()
    )


def test_native_exact_allocation_counts_each_structural_site_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A specialized call graph reserves each surviving lexical site once."""

    source = """
import qamomile as qm

@qm.qkernel
def reserve(width: qm.Const[qm.Int]) -> qm.Unit:
    scratch = qm.qubits(width)
    qm.discard(scratch)
    return None

@qm.qkernel
def sample(
    state: qm.Q[qm.Bits[2]],
    selected: qm.Const[qm.Bool],
) -> qm.Bits[2]:
    reserve(1)
    if selected:
        reserve(2)
    else:
        reserve(3)
    for _ in range(4):
        reserve(2)
    return qm.measure(state)
"""
    program = qm.build_source(
        source,
        module_name="native_exact_structural_allocation",
    ).specialize("sample", **{"selected": True})

    counts = native.inspect_exact_logical_counts(program.to_bytes())
    study = qm.resources.logical(program)
    decoded_study = qm.resources.logical(program)

    assert counts.input_qubits == 2
    assert counts.allocated_qubits == 5
    assert counts.circuit_qubits == 7
    for metric, expected in {
        qm.resources.LOGICAL_ALLOCATED_QUBITS: 5,
        qm.resources.LOGICAL_CIRCUIT_QUBITS: 7,
    }.items():
        assert study.metric(metric).numeric_enclosure() == (expected, expected)
        assert (
            study.metric(metric).numeric_enclosure()
            == decoded_study.metric(metric).numeric_enclosure()
        )


def test_public_logical_analysis_adopts_native_symbolic_dynamic_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dynamic control is enclosed by the native symbolic resource engine."""

    source = """
import qamomile as qm

@qm.qkernel
def sample(enabled: qm.Param[qm.Bool]) -> qm.Bits[1]:
    state = qm.qubits(1)
    if enabled:
        state = qm.h(state)
    return qm.measure(state)
"""
    program = qm.build_source(
        source,
        module_name="native_public_fallback",
    ).specialize("sample")

    study = qm.resources.logical(program)
    decoded_study = qm.resources.logical(program)

    assert "native-exact-logical-counts" not in study.derivations
    assert "native-symbolic-logical-counts" in study.derivations
    assert study.metric(qm.resources.LOGICAL_WORK).numeric_enclosure() == (2, 3)
    assert study.metric(
        qm.resources.logical_operation_count("quantum.h")
    ).numeric_enclosure() == (0, 1)
    assert study.metric(qm.resources.LOGICAL_WORK).numeric_enclosure() == (
        decoded_study.metric(qm.resources.LOGICAL_WORK).numeric_enclosure()
    )
    assert (
        study.metric(
            qm.resources.logical_operation_count("quantum.h")
        ).numeric_enclosure()
        == decoded_study.metric(
            qm.resources.logical_operation_count("quantum.h")
        ).numeric_enclosure()
    )
    assert study.metric(qm.resources.LOGICAL_CLIFFORD_GATES).numeric_enclosure() == (
        0,
        1,
    )
    assert study.metric(qm.resources.LOGICAL_CLIFFORD_GATES).numeric_enclosure() == (
        decoded_study.metric(qm.resources.LOGICAL_CLIFFORD_GATES).numeric_enclosure()
    )


def test_native_symbolic_resources_evaluate_exact_real_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exact binary64 facts select branches without entering count expressions."""

    from qamomile.serialization import encode_analysis_request

    source = """
import qamomile as qm

@qm.qkernel
def sample() -> qm.Bits[1]:
    state = qm.qubits(1)
    angle = -(0.25 + 0.25)
    if angle < 0.0:
        state = qm.h(state)
    else:
        state = qm.t(state)
    return qm.measure(state)
"""
    module = qm.build_source(source, module_name="native_symbolic_real_control")
    family = module["sample"]
    counts = native.inspect_symbolic_program_family_counts(
        encode_analysis_request(module, "sample")
    )
    study = qm.resources.logical(family)
    decoded_study = qm.resources.logical(family)

    opcodes = {item.opcode: item.count for item in counts.opcodes}
    assert counts.work.exact is True
    assert opcodes["quantum.h"].exact is True
    assert opcodes["quantum.t"].exact is True
    assert counts.expressions[opcodes["quantum.t"].lower].literal == "0"
    assert "native-symbolic-logical-counts" in study.derivations
    expected = {
        qm.resources.LOGICAL_WORK: (3, 3),
        qm.resources.logical_operation_count("quantum.h"): (1, 1),
        qm.resources.logical_operation_count("quantum.t"): (0, 0),
    }
    for metric, native_enclosure in expected.items():
        assert study.metric(metric).numeric_enclosure() == native_enclosure
        python_lower, python_upper = decoded_study.metric(metric).numeric_enclosure()
        assert python_lower <= native_enclosure[0]
        assert python_upper is not None and python_upper >= native_enclosure[1]


def test_native_symbolic_resources_compose_external_rules() -> None:
    """A canonical rule request composes an opaque call without Python callbacks."""

    from qamomile.proto import resource_analysis_pb2 as resource_wire
    from qamomile.resources._rule_wire import encode_resource_analysis_request
    from qamomile.resources.analysis import (
        _decode_native_resource_expressions,
        _decode_native_symbolic_count,
    )

    source = """
import qamomile as qm

@qm.qkernel(kind="unitary")
def oracle(state: qm.Q[qm.Bits[1]], rounds: qm.Int) -> qm.Q[qm.Bits[1]]:
    ...

@qm.qkernel
def algorithm(rounds: qm.Const[qm.Int]) -> qm.Bits[1]:
    state = qm.qubits(1)
    for _ in range(rounds):
        state = oracle(state, rounds)
    return qm.measure(state)
"""
    module = qm.build_source(source, module_name="native_external_rule")
    oracle = module.function("oracle")
    rounds = qm.resources.rule_parameter(
        oracle,
        "rounds",
        domain=qm.resources.Domain.NATURAL,
        constraints=("rounds >= 0",),
    )
    rule = qm.resources.ResourceRule(
        identity="example.native-oracle.gates.v1",
        callable=oracle,
        metric=qm.resources.LOGICAL_GATES,
        relation=qm.resources.Relation.EQUALITY,
        expression=qm.resources.piecewise(
            (
                (
                    qm.resources.comparison(
                        "greater",
                        rounds,
                        qm.resources.literal(0),
                    ),
                    qm.resources.multiply(qm.resources.literal(2), rounds),
                ),
            ),
            otherwise=qm.resources.literal(0),
        ),
        assumptions=("oracle decomposition is exact",),
        validity_domain=("rounds >= 0",),
        provenance="example.native-oracle.contract.v1",
    )
    span_rule = qm.resources.ResourceRule(
        identity="example.native-oracle.span.v1",
        callable=oracle,
        metric=qm.resources.LOGICAL_SPAN,
        relation=qm.resources.Relation.EQUALITY,
        expression=qm.resources.add(rounds, qm.resources.literal(1)),
        provenance="example.native-oracle.span-contract.v1",
    )
    gate_depth_rule = qm.resources.ResourceRule(
        identity="example.native-oracle.gate-depth.v1",
        callable=oracle,
        metric=qm.resources.LOGICAL_GATE_DEPTH,
        relation=qm.resources.Relation.EQUALITY,
        expression=rounds,
        provenance="example.native-oracle.gate-depth-contract.v1",
    )
    for bound_rounds, expected_gates, expected_span, expected_depth, expected_exact in (
        (3, 18, (4, 14), (3, 9), False),
        (0, 0, (2, 2), (0, 0), True),
    ):
        program = module.specialize("algorithm", **{"rounds": bound_rounds})
        request_payload = encode_resource_analysis_request(
            program.to_bytes(), (rule, span_rule, gate_depth_rule)
        )
        counts = native.inspect_symbolic_resource_analysis_request(request_payload)
        expressions = _decode_native_resource_expressions(counts.expressions)
        work = _decode_native_symbolic_count(counts.work, expressions)
        span = _decode_native_symbolic_count(counts.dependency_span, expressions)
        gate_depth = _decode_native_symbolic_count(
            next(item.count for item in counts.depths if item.family == "gates"),
            expressions,
        )
        native_gates = next(
            (item.count for item in counts.families if item.family == "gates"),
            None,
        )
        gates = (
            None
            if native_gates is None
            else _decode_native_symbolic_count(native_gates, expressions)
        )

        projected_gate_depth = next(
            item.count for item in counts.depths if item.family == "gates"
        )
        if bound_rounds:
            assert rule.identity in counts.work.rule_identities
            assert span_rule.identity in counts.dependency_span.rule_identities
            assert native_gates is not None
            assert native_gates.rule_identities == [rule.identity]
            assert projected_gate_depth.rule_identities == [gate_depth_rule.identity]
        else:
            assert counts.work.rule_identities == []
            assert counts.dependency_span.rule_identities == []
            assert native_gates is None or native_gates.rule_identities == []
            assert projected_gate_depth.rule_identities == []

        expected_work = expected_gates + 2
        assert (work.lower.numeric(), work.upper.numeric()) == (
            expected_work,
            expected_work,
        )
        if gates is None:
            assert expected_gates == 0
        else:
            assert (gates.lower.numeric(), gates.upper.numeric()) == (
                expected_gates,
                expected_gates,
            )
        assert (span.lower.numeric(), span.upper.numeric()) == expected_span
        assert (
            gate_depth.lower.numeric(),
            gate_depth.upper.numeric(),
        ) == expected_depth
        assert span.exact is expected_exact
        assert gate_depth.exact is expected_exact
        study = qm.resources.logical(
            program,
            rules=(rule, span_rule, gate_depth_rule),
            strict=True,
        )
        assert "native-symbolic-logical-counts" in study.derivations
        assert study.metric(qm.resources.LOGICAL_WORK).numeric_enclosure() == (
            expected_work,
            expected_work,
        )
        assert study.metric(qm.resources.LOGICAL_GATES).numeric_enclosure() == (
            expected_gates,
            expected_gates,
        )
        assert (
            study.metric(qm.resources.LOGICAL_SPAN).numeric_enclosure() == expected_span
        )
        assert (
            study.metric(qm.resources.LOGICAL_GATE_DEPTH).numeric_enclosure()
            == expected_depth
        )
        if bound_rounds:
            gate_value = study.metric(qm.resources.LOGICAL_GATES)
            assert any(
                evidence.reference == "example.native-oracle.contract.v1"
                for evidence in gate_value.evidence
            )
            assert gate_value.assumptions == (
                "oracle decomposition is exact",
                "validity: rounds >= 0",
            )
            assert any(
                evidence.reference == "example.native-oracle.span-contract.v1"
                for evidence in study.metric(qm.resources.LOGICAL_SPAN).evidence
            )
            assert any(
                evidence.reference == "example.native-oracle.gate-depth-contract.v1"
                for evidence in study.metric(qm.resources.LOGICAL_GATE_DEPTH).evidence
            )
        else:
            gate_value = study.metric(qm.resources.LOGICAL_GATES)
            assert all(
                evidence.reference != "example.native-oracle.contract.v1"
                for evidence in gate_value.evidence
            )
            assert gate_value.assumptions == ()

    malformed = resource_wire.ResourceAnalysisRequestProto.FromString(request_payload)
    symbol = next(
        node.expression.symbol
        for node in malformed.rules[0].expression.nodes
        if node.WhichOneof("node") == "expression" and node.expression.kind == "symbol"
    )
    symbol.name = "not_a_parameter"
    malformed.request_digest = b""
    malformed.request_digest = hashlib.sha256(
        malformed.SerializeToString(deterministic=True)
    ).digest()
    with pytest.raises(ValueError, match="unbound symbol"):
        native.inspect_symbolic_resource_analysis_request(
            malformed.SerializeToString(deterministic=True)
        )

    malformed = resource_wire.ResourceAnalysisRequestProto.FromString(request_payload)
    malformed.rules[0].scheduling_footprint.parameters.append("rounds")
    malformed.request_digest = b""
    malformed.request_digest = hashlib.sha256(
        malformed.SerializeToString(deterministic=True)
    ).digest()
    with pytest.raises(ValueError, match="non-quantum parameter"):
        native.inspect_symbolic_resource_analysis_request(
            malformed.SerializeToString(deterministic=True)
        )


def test_native_symbolic_count_rejects_undeclared_rule_usage() -> None:
    """The Python bridge cannot mint evidence for a forged native rule identity."""

    from qamomile.resources.analysis import _decode_native_symbolic_count

    forged = SimpleNamespace(
        lower=0,
        upper=0,
        exact=True,
        unknown=[],
        rule_identities=["example.forged-rule.v1"],
        relation="equality",
        dependency_span_relation="equality",
    )

    with pytest.raises(TypeError, match="undeclared resource rule"):
        _decode_native_symbolic_count(
            forged,
            (qm.resources.literal(0),),
            {},
        )


def test_native_symbolic_facts_reject_duplicate_lexical_identity() -> None:
    """A forged native result cannot duplicate one public explanation fact."""

    from qamomile.resources.analysis import (
        _Cost,
        _decode_native_analysis_facts,
    )

    fact = SimpleNamespace(
        owner="module.sample",
        source="qmsrc://example/module.py:1:1",
        opcode="quantum.h",
        work=None,
        opcodes=[],
        extra_metrics=[],
        families=[],
        measurements=None,
        resets=None,
        cleanup_work=None,
    )

    with pytest.raises(TypeError, match="duplicate lexical identity"):
        _decode_native_analysis_facts(
            [fact, fact],
            lambda _value: _Cost.zero(),
            (),
        )


def test_native_symbolic_resource_rules_retain_predicate_dags() -> None:
    """An open rule family preserves ordered symbolic Boolean control."""

    from qamomile.resources._rule_wire import encode_resource_analysis_request
    from qamomile.resources.analysis import (
        _decode_native_resource_expressions,
        _decode_native_symbolic_count,
    )
    from qamomile.serialization import encode_analysis_request

    source = """
import qamomile as qm

@qm.qkernel(kind="unitary")
def oracle(state: qm.Q[qm.Bits[1]], rounds: qm.Int) -> qm.Q[qm.Bits[1]]:
    ...

@qm.qkernel
def algorithm(rounds: qm.Const[qm.Int]) -> qm.Bits[1]:
    return qm.measure(oracle(qm.qubits(1), rounds))
"""
    module = qm.build_source(source, module_name="native_symbolic_rule_predicate")
    oracle = module.function("oracle")
    rounds = qm.resources.rule_parameter(
        oracle,
        "rounds",
        domain=qm.resources.Domain.NATURAL,
        constraints=("rounds >= 0",),
    )
    odd_positive = qm.resources.conjunction(
        qm.resources.comparison("greater", rounds, qm.resources.literal(0)),
        qm.resources.negate(qm.resources.divisible(rounds, qm.resources.literal(2))),
    )
    binder = qm.resources.Symbol(
        "example.native-symbolic-predicate",
        "iteration",
        qm.resources.Domain.NATURAL,
        origin="bounded_rule",
    )
    rule = qm.resources.ResourceRule(
        identity="example.native-symbolic-predicate.gates.v1",
        callable=oracle,
        metric=qm.resources.LOGICAL_GATES,
        relation=qm.resources.Relation.EQUALITY,
        expression=qm.resources.piecewise(
            (
                (
                    odd_positive,
                    qm.resources.bounded_product(
                        binder,
                        qm.resources.literal(1),
                        rounds,
                        qm.resources.literal(2),
                    ),
                ),
            ),
            otherwise=rounds,
        ),
        provenance="example.native-symbolic-predicate.contract.v1",
    )
    recurrence_metric = qm.resources.logical_operation_count("oracle.recurrence")
    recurrence_rule = qm.resources.ResourceRule(
        identity="example.native-symbolic-predicate.recurrence.v1",
        callable=oracle,
        metric=recurrence_metric,
        relation=qm.resources.Relation.EQUALITY,
        expression=qm.resources.recurrence(
            "example.oracle.recurrence",
            (
                qm.resources.absolute(rounds),
                qm.resources.floor(rounds),
                qm.resources.ceiling(rounds),
                qm.resources.floor_divide(rounds, qm.resources.literal(2)),
                qm.resources.ceiling_divide(rounds, qm.resources.literal(2)),
                qm.resources.literal(Fraction(1, 3)),
                qm.resources.literal(0.25),
                qm.resources.exponential(rounds),
                qm.resources.logarithm(
                    qm.resources.add(rounds, qm.resources.literal(1)),
                    base=qm.resources.literal(2),
                ),
                qm.resources.log2(qm.resources.add(rounds, qm.resources.literal(1))),
                qm.resources.square_root(rounds),
                qm.resources.minimum(
                    rounds,
                    qm.resources.literal(Fraction(1, 3)),
                ),
                qm.resources.maximum(
                    rounds,
                    qm.resources.literal(Fraction(2, 3)),
                ),
                qm.resources.bounded_sum(
                    binder,
                    qm.resources.literal(1),
                    rounds,
                    qm.resources.multiply(
                        qm.resources.literal(Fraction(1, 2)),
                        qm.resources.scoped_symbol(binder),
                    ),
                ),
                qm.resources.bounded_product(
                    binder,
                    qm.resources.literal(1),
                    rounds,
                    qm.resources.literal(Fraction(1, 2)),
                ),
            ),
            domain=qm.resources.Domain.NATURAL,
            unit=qm.resources.ONE,
        ),
        provenance="example.native-symbolic-predicate.recurrence-contract.v1",
    )
    analytic_metric = qm.resources.MetricDefinition(
        "example.oracle.analytic_cost",
        qm.resources.Coordinate.LOGICAL,
        qm.resources.Domain.REAL,
        qm.resources.ONE,
    )
    analytic_rule = qm.resources.ResourceRule(
        identity="example.native-symbolic-predicate.analytic.v1",
        callable=oracle,
        metric=analytic_metric,
        relation=qm.resources.Relation.EQUALITY,
        expression=qm.resources.log2(qm.resources.add(rounds, qm.resources.literal(1))),
        provenance="example.native-symbolic-predicate.analytic-contract.v1",
    )
    payload = encode_resource_analysis_request(
        encode_analysis_request(module, "algorithm"),
        (rule, recurrence_rule, analytic_rule),
    )

    counts = native.inspect_symbolic_resource_analysis_request(payload)
    expressions = _decode_native_resource_expressions(counts.expressions)
    gates = _decode_native_symbolic_count(
        next(item.count for item in counts.families if item.family == "gates"),
        expressions,
    )
    recurrence_count = _decode_native_symbolic_count(
        next(
            item.count for item in counts.opcodes if item.opcode == "oracle.recurrence"
        ),
        expressions,
    )
    analytic_count = _decode_native_symbolic_count(
        next(
            item.count
            for item in counts.extra_metrics
            if item.identifier == analytic_metric.identifier
        ),
        expressions,
    )
    study = qm.resources.logical(
        module["algorithm"],
        rules=(rule, recurrence_rule, analytic_rule),
        strict=True,
    )

    assert gates.exact is True
    assert gates.lower.kind == "piecewise"
    assert gates.lower.evaluate({"rounds": 3}).numeric() == 8
    assert gates.lower.evaluate({"rounds": 4}).numeric() == 4
    assert recurrence_count.exact is True
    assert recurrence_count.lower.kind == "recurrence"
    assert dict(recurrence_count.lower.attributes) == {
        "name": "example.oracle.recurrence",
        "version": "1",
    }
    assert analytic_count.exact is True
    assert analytic_count.lower.kind == "log2"
    assert "native-symbolic-logical-counts" in study.derivations
    assert (
        study.metric(qm.resources.LOGICAL_GATES).relation
        is qm.resources.Relation.EQUALITY
    )
    assert study.metric(qm.resources.LOGICAL_GATES).evaluate(
        {"rounds": 3}
    ).numeric_enclosure() == (8, 8)
    assert study.metric(qm.resources.LOGICAL_GATES).evaluate(
        {"rounds": 4}
    ).numeric_enclosure() == (4, 4)
    assert study.metric(recurrence_metric).relation is qm.resources.Relation.EQUALITY
    assert study.metric(recurrence_metric).expression.kind == "recurrence"
    assert tuple(
        argument.kind
        for argument in study.metric(recurrence_metric).expression.arguments
    ) == (
        "absolute",
        "floor",
        "ceiling",
        "floor_divide",
        "ceiling_divide",
        "literal",
        "literal",
        "exp",
        "log",
        "log2",
        "square_root",
        "minimum",
        "maximum",
        "bounded_sum",
        "bounded_product",
    )
    recurrence_arguments = study.metric(recurrence_metric).expression.arguments
    assert recurrence_arguments[5].numeric() == Fraction(1, 3)
    assert recurrence_arguments[6].numeric() == 0.25
    assert tuple(argument.domain for argument in recurrence_arguments[11:]) == (
        qm.resources.Domain.RATIONAL,
        qm.resources.Domain.RATIONAL,
        qm.resources.Domain.RATIONAL,
        qm.resources.Domain.RATIONAL,
    )
    assert study.metric(analytic_metric).relation is qm.resources.Relation.EQUALITY
    assert study.metric(analytic_metric).expression.kind == "log2"


def test_native_external_footprint_preserves_disjoint_parallelism(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An opaque exact span occupies only its declared quantum parameters."""

    source = """
import qamomile as qm

@qm.qkernel(kind="unitary")
def oracle(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    ...

@qm.qkernel
def algorithm() -> tuple[qm.Q[qm.Bits[1]], qm.Q[qm.Bits[1]]]:
    left = qm.qubits(1)
    right = qm.qubits(1)
    left = oracle(left)
    right = oracle(right)
    left = qm.h(left)
    return left, right
"""
    module = qm.build_source(source, module_name="native_external_footprint")
    oracle = module.function("oracle")
    gates = qm.resources.ResourceRule(
        identity="example.native-footprint.gates.v1",
        callable=oracle,
        metric=qm.resources.LOGICAL_GATES,
        relation=qm.resources.Relation.EQUALITY,
        expression=qm.resources.literal(4),
        provenance="example.native-footprint.gates-contract.v1",
    )
    span = qm.resources.ResourceRule(
        identity="example.native-footprint.span.v1",
        callable=oracle,
        metric=qm.resources.LOGICAL_SPAN,
        relation=qm.resources.Relation.EQUALITY,
        expression=qm.resources.literal(4),
        provenance="example.native-footprint.span-contract.v1",
        scheduling_footprint=qm.resources.SchedulingFootprint(("state",)),
    )

    localized = qm.resources.logical(module["algorithm"], rules=(gates, span))
    conservative = qm.resources.logical(
        module["algorithm"],
        rules=(gates, replace(span, scheduling_footprint=None)),
    )

    decoded_localized = qm.resources.logical(module["algorithm"], rules=(gates, span))

    assert localized.metric(qm.resources.LOGICAL_SPAN).numeric_enclosure() == (6, 6)
    assert decoded_localized.metric(qm.resources.LOGICAL_SPAN).numeric_enclosure() == (
        6,
        6,
    )
    assert localized.metric(qm.resources.LOGICAL_GATE_DEPTH).numeric_enclosure() == (
        1,
        9,
    )
    assert conservative.metric(qm.resources.LOGICAL_SPAN).numeric_enclosure() == (
        4,
        11,
    )


def test_native_symbolic_external_span_refines_timing_duration() -> None:
    """Rule-domain refinement applies to scalar scheduling durations."""

    source = """
import qamomile as qm

@qm.qkernel(kind="unitary")
def oracle(state: qm.Q[qm.Bits[1]], rounds: qm.Int) -> qm.Q[qm.Bits[1]]:
    ...

@qm.qkernel
def algorithm(
    rounds: qm.Param[qm.Int],
) -> tuple[qm.Q[qm.Bits[1]], qm.Q[qm.Bits[1]]]:
    left = qm.qubits(1)
    right = qm.qubits(1)
    left = oracle(left, rounds)
    right = qm.h(right)
    return left, right
"""
    module = qm.build_source(source, module_name="native_symbolic_external_span")
    oracle = module.function("oracle")
    rounds = qm.resources.rule_parameter(
        oracle,
        "rounds",
        domain=qm.resources.Domain.NATURAL,
        constraints=("rounds >= 0",),
    )
    rules = (
        qm.resources.ResourceRule(
            identity="example.native-symbolic-span.gates.v1",
            callable=oracle,
            metric=qm.resources.LOGICAL_GATES,
            relation=qm.resources.Relation.EQUALITY,
            expression=rounds,
            assumptions=("rounds >= 0",),
            provenance="example.native-symbolic-span.gates-contract.v1",
        ),
        qm.resources.ResourceRule(
            identity="example.native-symbolic-span.span.v1",
            callable=oracle,
            metric=qm.resources.LOGICAL_SPAN,
            relation=qm.resources.Relation.EQUALITY,
            expression=rounds,
            assumptions=("rounds >= 0",),
            provenance="example.native-symbolic-span.span-contract.v1",
            scheduling_footprint=qm.resources.SchedulingFootprint(("state",)),
        ),
    )

    import qamomile.resources.analysis as resource_analysis

    study = qm.resources.logical(module["algorithm"], rules=rules, strict=True)

    assert not hasattr(resource_analysis, "_LogicalAnalyzer")
    span = study.metric(qm.resources.LOGICAL_SPAN).evaluate({"rounds": 3})
    assert span.relation is qm.resources.Relation.EQUALITY
    assert span.numeric_enclosure() == (4, 4)


def test_native_external_adjoint_footprint_preserves_disjoint_parallelism(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct and adjoint opaque calls share an exact declared lane contract."""

    source = """
import qamomile as qm

@qm.qkernel(kind="unitary")
def oracle(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    ...

@qm.qkernel
def algorithm() -> tuple[qm.Q[qm.Bits[1]], qm.Q[qm.Bits[1]]]:
    left = qm.qubits(1)
    right = qm.qubits(1)
    left = oracle(left)
    right = qm.adjoint(oracle, right)
    left = qm.h(left)
    return left, right
"""
    module = qm.build_source(source, module_name="native_external_adjoint_footprint")
    oracle = module.function("oracle")
    modes = frozenset(
        {
            qm.resources.TransformMode.DIRECT,
            qm.resources.TransformMode.ADJOINT,
        }
    )
    gates = qm.resources.ResourceRule(
        identity="example.native-adjoint-footprint.gates.v1",
        callable=oracle,
        metric=qm.resources.LOGICAL_GATES,
        relation=qm.resources.Relation.EQUALITY,
        expression=qm.resources.literal(4),
        modes=modes,
        provenance="example.native-adjoint-footprint.gates-contract.v1",
    )
    span = qm.resources.ResourceRule(
        identity="example.native-adjoint-footprint.span.v1",
        callable=oracle,
        metric=qm.resources.LOGICAL_SPAN,
        relation=qm.resources.Relation.EQUALITY,
        expression=qm.resources.literal(4),
        modes=modes,
        provenance="example.native-adjoint-footprint.span-contract.v1",
        scheduling_footprint=qm.resources.SchedulingFootprint(("state",)),
    )

    localized = qm.resources.logical(module["algorithm"], rules=(gates, span))
    conservative = qm.resources.logical(
        module["algorithm"],
        rules=(gates, replace(span, scheduling_footprint=None)),
    )

    decoded_localized = qm.resources.logical(module["algorithm"], rules=(gates, span))

    assert localized.metric(qm.resources.LOGICAL_SPAN).numeric_enclosure() == (6, 6)
    assert decoded_localized.metric(qm.resources.LOGICAL_SPAN).numeric_enclosure() == (
        6,
        6,
    )
    assert conservative.metric(qm.resources.LOGICAL_SPAN).numeric_enclosure() == (
        4,
        11,
    )


def test_native_external_control_footprint_tracks_selected_control_lanes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A controlled opaque call occupies its target and selected control lane."""

    source = """
import qamomile as qm

@qm.qkernel(kind="unitary")
def oracle(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    ...

@qm.qkernel
def algorithm() -> tuple[qm.Q[qm.Bits[2]], qm.Q[qm.Bits[1]], qm.Q[qm.Bits[1]]]:
    controls = qm.qubits(2)
    left = qm.qubits(1)
    right = qm.qubits(1)
    controls, left = qm.control(oracle, controls, left, at=0)
    controls, right = qm.control(oracle, controls, right, at=1)
    left = qm.h(left)
    return controls, left, right
"""
    module = qm.build_source(source, module_name="native_external_control_footprint")
    oracle = module.function("oracle")
    mode = frozenset({qm.resources.TransformMode.CONTROLLED})
    gates = qm.resources.ResourceRule(
        identity="example.native-control-footprint.gates.v1",
        callable=oracle,
        metric=qm.resources.LOGICAL_GATES,
        relation=qm.resources.Relation.EQUALITY,
        expression=qm.resources.literal(4),
        modes=mode,
        provenance="example.native-control-footprint.gates-contract.v1",
    )
    span = qm.resources.ResourceRule(
        identity="example.native-control-footprint.span.v1",
        callable=oracle,
        metric=qm.resources.LOGICAL_SPAN,
        relation=qm.resources.Relation.EQUALITY,
        expression=qm.resources.literal(4),
        modes=mode,
        provenance="example.native-control-footprint.span-contract.v1",
        scheduling_footprint=qm.resources.SchedulingFootprint(("state",)),
    )

    localized = qm.resources.logical(module["algorithm"], rules=(gates, span))
    conservative = qm.resources.logical(
        module["algorithm"],
        rules=(gates, replace(span, scheduling_footprint=None)),
    )

    decoded_localized = qm.resources.logical(module["algorithm"], rules=(gates, span))

    assert localized.metric(qm.resources.LOGICAL_SPAN).numeric_enclosure() == (6, 6)
    assert decoded_localized.metric(qm.resources.LOGICAL_SPAN).numeric_enclosure() == (
        6,
        6,
    )
    conservative_span = conservative.metric(qm.resources.LOGICAL_SPAN)
    assert conservative_span.relation is qm.resources.Relation.ENCLOSURE
    assert conservative_span.numeric_enclosure() == (4, 12)


def test_native_body_adjoint_composes_nested_external_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Common adjoint lowering exposes nested external inverse contracts."""

    source = """
import qamomile as qm

@qm.qkernel(kind="unitary")
def oracle(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    ...

@qm.qkernel(kind="unitary")
def wrapper(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    state = qm.x(state)
    return oracle(state)

@qm.qkernel
def algorithm() -> qm.Q[qm.Bits[1]]:
    state = qm.qubits(1)
    state = qm.adjoint(wrapper, state)
    state = qm.h(state)
    return state
"""
    module = qm.build_source(source, module_name="native_body_adjoint_rule")
    oracle = module.function("oracle")
    mode = frozenset({qm.resources.TransformMode.ADJOINT})
    rules = (
        qm.resources.ResourceRule(
            identity="example.native-body-adjoint.gates.v1",
            callable=oracle,
            metric=qm.resources.LOGICAL_GATES,
            relation=qm.resources.Relation.EQUALITY,
            expression=qm.resources.literal(4),
            modes=mode,
            provenance="example.native-body-adjoint.gates-contract.v1",
        ),
        qm.resources.ResourceRule(
            identity="example.native-body-adjoint.span.v1",
            callable=oracle,
            metric=qm.resources.LOGICAL_SPAN,
            relation=qm.resources.Relation.EQUALITY,
            expression=qm.resources.literal(4),
            modes=mode,
            provenance="example.native-body-adjoint.span-contract.v1",
            scheduling_footprint=qm.resources.SchedulingFootprint(("state",)),
        ),
    )

    for analyzed in (
        module["algorithm"],
        module.specialize("algorithm"),
    ):
        semantic = qm.resources.logical(analyzed)
        localized = qm.resources.logical(analyzed, rules=rules, strict=True)
        assert localized.semantic_digest == semantic.semantic_digest
        assert "native-symbolic-logical-counts" in localized.derivations
        assert "native-transform-mode-composition" in localized.derivations
        assert localized.metric(qm.resources.LOGICAL_GATES).numeric_enclosure() == (
            6,
            6,
        )
        assert localized.metric(qm.resources.LOGICAL_SPAN).numeric_enclosure() == (
            7,
            7,
        )
        assert localized.metric(
            qm.resources.LOGICAL_CLEANUP_WORK
        ).numeric_enclosure() == (0, 0)


def test_native_nested_adjoint_restores_direct_external_mode() -> None:
    """Two body-backed adjoints compose to the direct external contract."""

    source = """
import qamomile as qm

@qm.qkernel(kind="unitary")
def oracle(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    ...

@qm.qkernel(kind="unitary")
def inverse(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.adjoint(oracle, state)

@qm.qkernel
def algorithm() -> qm.Q[qm.Bits[1]]:
    return qm.adjoint(inverse, qm.qubits(1))
"""
    module = qm.build_source(source, module_name="native_nested_adjoint_rule")
    oracle = module.function("oracle")
    rules = tuple(
        qm.resources.ResourceRule(
            identity=f"example.native-nested-adjoint.{mode.value}.v1",
            callable=oracle,
            metric=qm.resources.LOGICAL_GATES,
            relation=qm.resources.Relation.EQUALITY,
            expression=qm.resources.literal(gates),
            modes=frozenset({mode}),
            provenance=f"example.native-nested-adjoint.{mode.value}.contract.v1",
        )
        for mode, gates in (
            (qm.resources.TransformMode.DIRECT, 3),
            (qm.resources.TransformMode.ADJOINT, 11),
        )
    )

    study = qm.resources.logical(module["algorithm"], rules=rules, strict=True)

    assert "native-symbolic-logical-counts" in study.derivations
    assert study.metric(qm.resources.LOGICAL_GATES).numeric_enclosure() == (3, 3)


def test_native_controlled_adjoint_selects_combined_external_mode() -> None:
    """Controlled inverse wrappers select one explicit combined rule contract."""

    source = """
import qamomile as qm

@qm.qkernel(kind="unitary")
def oracle(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    ...

@qm.qkernel(kind="unitary")
def inverse(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.adjoint(oracle, state)

@qm.qkernel
def algorithm() -> tuple[qm.Q[qm.Bits[1]], qm.Q[qm.Bits[1]]]:
    control = qm.qubits(1)
    target = qm.qubits(1)
    control, target = qm.control(inverse, control, target)
    target = qm.h(target)
    return control, target
"""
    module = qm.build_source(source, module_name="native_controlled_adjoint_rule")
    oracle = module.function("oracle")
    mode = frozenset({qm.resources.TransformMode.CONTROLLED_ADJOINT})
    rules = (
        qm.resources.ResourceRule(
            identity="example.native-controlled-adjoint.gates.v1",
            callable=oracle,
            metric=qm.resources.LOGICAL_GATES,
            relation=qm.resources.Relation.EQUALITY,
            expression=qm.resources.literal(4),
            modes=mode,
            provenance="example.native-controlled-adjoint.gates-contract.v1",
        ),
        qm.resources.ResourceRule(
            identity="example.native-controlled-adjoint.span.v1",
            callable=oracle,
            metric=qm.resources.LOGICAL_SPAN,
            relation=qm.resources.Relation.EQUALITY,
            expression=qm.resources.literal(4),
            modes=mode,
            provenance="example.native-controlled-adjoint.span-contract.v1",
            scheduling_footprint=qm.resources.SchedulingFootprint(("state",)),
        ),
    )

    study = qm.resources.logical(module["algorithm"], rules=rules)

    assert "native-symbolic-logical-counts" in study.derivations
    assert study.metric(qm.resources.LOGICAL_GATES).numeric_enclosure() == (5, None)
    assert study.metric(qm.resources.LOGICAL_SPAN).numeric_enclosure() == (1, None)
    assert any(
        evidence.reference == "example.native-controlled-adjoint.gates-contract.v1"
        for evidence in study.metric(qm.resources.LOGICAL_GATES).evidence
    )
    with pytest.raises(ValueError, match="body-backed controlled transform"):
        qm.resources.logical(module["algorithm"], rules=rules, strict=True)


def test_native_transparent_control_composes_nested_external_rules() -> None:
    """Transparent controlled wrappers expose the external mode contract."""

    source = """
import qamomile as qm

@qm.qkernel(kind="unitary")
def oracle(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    ...

@qm.qkernel(kind="unitary")
def wrapper(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return oracle(state)

@qm.qkernel(kind="unitary")
def outer(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return wrapper(state)

@qm.qkernel
def algorithm() -> tuple[qm.Q[qm.Bits[1]], qm.Q[qm.Bits[1]]]:
    control = qm.qubits(1)
    target = qm.qubits(1)
    control, target = qm.control(outer, control, target)
    target = qm.h(target)
    return control, target
"""
    module = qm.build_source(source, module_name="native_transparent_control_rule")
    oracle = module.function("oracle")
    mode = frozenset({qm.resources.TransformMode.CONTROLLED})
    rules = (
        qm.resources.ResourceRule(
            identity="example.native-transparent-control.gates.v1",
            callable=oracle,
            metric=qm.resources.LOGICAL_GATES,
            relation=qm.resources.Relation.EQUALITY,
            expression=qm.resources.literal(4),
            modes=mode,
            provenance="example.native-transparent-control.gates-contract.v1",
        ),
        qm.resources.ResourceRule(
            identity="example.native-transparent-control.span.v1",
            callable=oracle,
            metric=qm.resources.LOGICAL_SPAN,
            relation=qm.resources.Relation.EQUALITY,
            expression=qm.resources.literal(4),
            modes=mode,
            provenance="example.native-transparent-control.span-contract.v1",
            scheduling_footprint=qm.resources.SchedulingFootprint(("state",)),
        ),
    )

    for analyzed in (
        module["algorithm"],
        module.specialize("algorithm"),
    ):
        semantic = qm.resources.logical(analyzed)
        localized = qm.resources.logical(analyzed, rules=rules, strict=True)
        assert localized.semantic_digest == semantic.semantic_digest
        assert "native-transform-mode-composition" in localized.derivations
        assert localized.metric(qm.resources.LOGICAL_GATES).numeric_enclosure() == (
            5,
            5,
        )
        assert localized.metric(qm.resources.LOGICAL_SPAN).numeric_enclosure() == (
            6,
            6,
        )
        assert localized.metric(
            qm.resources.LOGICAL_CLEANUP_WORK
        ).numeric_enclosure() == (0, 0)


def test_native_serial_control_composes_nested_external_rules() -> None:
    """Serial controlled wrappers expose each external mode contract."""

    source = """
import qamomile as qm

@qm.qkernel(kind="unitary")
def first(state: qm.Q[qm.Bits[1]], rounds: qm.Int) -> qm.Q[qm.Bits[1]]:
    ...

@qm.qkernel(kind="unitary")
def second(state: qm.Q[qm.Bits[1]], rounds: qm.Int) -> qm.Q[qm.Bits[1]]:
    ...

@qm.qkernel(kind="unitary")
def wrapper(state: qm.Q[qm.Bits[1]], rounds: qm.Int) -> qm.Q[qm.Bits[1]]:
    state = first(state, rounds)
    return second(state, rounds)

@qm.qkernel
def algorithm(
    rounds: qm.Param[qm.Int],
) -> tuple[qm.Q[qm.Bits[1]], qm.Q[qm.Bits[1]]]:
    control = qm.qubits(1)
    target = qm.qubits(1)
    control, target = qm.control(wrapper, control, target, rounds)
    target = qm.h(target)
    return control, target
"""
    module = qm.build_source(source, module_name="native_serial_control_rule")
    mode = frozenset({qm.resources.TransformMode.CONTROLLED})
    rules = tuple(
        qm.resources.ResourceRule(
            identity=f"example.native-serial-control.{name}.{metric}.v1",
            callable=module.function(name),
            metric=definition,
            relation=qm.resources.Relation.EQUALITY,
            expression=qm.resources.literal(2),
            modes=mode,
            provenance=f"example.native-serial-control.{name}.contract.v1",
            scheduling_footprint=(
                qm.resources.SchedulingFootprint(("state",))
                if metric == "span"
                else None
            ),
        )
        for name in ("first", "second")
        for metric, definition in (
            ("gates", qm.resources.LOGICAL_GATES),
            ("span", qm.resources.LOGICAL_SPAN),
        )
    )

    for analyzed in (
        module["algorithm"],
        module.specialize("algorithm"),
    ):
        semantic = qm.resources.logical(analyzed)
        localized = qm.resources.logical(analyzed, rules=rules, strict=True)
        assert localized.semantic_digest == semantic.semantic_digest
        assert "native-transform-mode-composition" in localized.derivations
        assert localized.metric(qm.resources.LOGICAL_GATES).numeric_enclosure() == (
            5,
            5,
        )
        assert localized.metric(qm.resources.LOGICAL_SPAN).numeric_enclosure() == (
            6,
            6,
        )
        assert localized.metric(
            qm.resources.LOGICAL_CLEANUP_WORK
        ).numeric_enclosure() == (0, 0)


def test_native_external_select_footprints_transfer_packed_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every opaque select case transfers selector and target timing to outputs."""

    source = """
import qamomile as qm

@qm.qkernel(kind="unitary")
def first(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    ...

@qm.qkernel(kind="unitary")
def second(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    ...

@qm.qkernel
def algorithm() -> tuple[qm.Q[qm.Bits[1]], qm.Q[qm.Bits[1]]]:
    index = qm.qubits(1)
    target = qm.qubits(1)
    index, target = qm.multiplex(index, (first, second), target)
    target = qm.h(target)
    return index, target
"""
    module = qm.build_source(source, module_name="native_external_select_footprint")
    mode = frozenset({qm.resources.TransformMode.CONTROLLED})
    rules = tuple(
        qm.resources.ResourceRule(
            identity=f"example.native-select-footprint.{name}.{metric}.v1",
            callable=module.function(name),
            metric=definition,
            relation=qm.resources.Relation.EQUALITY,
            expression=qm.resources.literal(2),
            modes=mode,
            provenance=f"example.native-select-footprint.{name}.contract.v1",
            scheduling_footprint=(
                qm.resources.SchedulingFootprint(("state",))
                if metric == "span"
                else None
            ),
        )
        for name in ("first", "second")
        for metric, definition in (
            ("gates", qm.resources.LOGICAL_GATES),
            ("span", qm.resources.LOGICAL_SPAN),
        )
    )

    localized = qm.resources.logical(module["algorithm"], rules=rules)

    decoded_localized = qm.resources.logical(module["algorithm"], rules=rules)

    assert localized.metric(qm.resources.LOGICAL_SPAN).numeric_enclosure() == (6, 6)
    assert decoded_localized.metric(qm.resources.LOGICAL_SPAN).numeric_enclosure() == (
        6,
        6,
    )


def test_native_component_control_preserves_unselected_lane_frontiers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A packed control result retains dependencies on every untouched lane."""

    source = """
import qamomile as qm

@qm.qkernel(kind="unitary")
def oracle(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    ...

@qm.qkernel
def algorithm() -> tuple[qm.Q[qm.Bits[2]], qm.Q[qm.Bits[1]]]:
    controls = qm.qubits(2)
    target = qm.qubits(1)
    controls = qm.x(controls, target=1)
    controls = qm.h(controls, target=1)
    controls = qm.z(controls, target=1)
    controls, target = qm.control(oracle, controls, target, at=0)
    controls = qm.x(controls, target=1)
    controls = qm.h(controls, target=1)
    controls = qm.z(controls, target=1)
    return controls, target
"""
    module = qm.build_source(source, module_name="native_control_lane_alias")
    oracle = module.function("oracle")
    mode = frozenset({qm.resources.TransformMode.CONTROLLED})
    rules = (
        qm.resources.ResourceRule(
            identity="example.native-control-lane-alias.gates.v1",
            callable=oracle,
            metric=qm.resources.LOGICAL_GATES,
            relation=qm.resources.Relation.EQUALITY,
            expression=qm.resources.literal(1),
            modes=mode,
            provenance="example.native-control-lane-alias.gates-contract.v1",
        ),
        qm.resources.ResourceRule(
            identity="example.native-control-lane-alias.span.v1",
            callable=oracle,
            metric=qm.resources.LOGICAL_SPAN,
            relation=qm.resources.Relation.EQUALITY,
            expression=qm.resources.literal(1),
            modes=mode,
            provenance="example.native-control-lane-alias.span-contract.v1",
            scheduling_footprint=qm.resources.SchedulingFootprint(("state",)),
        ),
    )

    localized = qm.resources.logical(module["algorithm"], rules=rules)

    decoded_localized = qm.resources.logical(module["algorithm"], rules=rules)

    assert localized.metric(qm.resources.LOGICAL_SPAN).numeric_enclosure() == (7, 7)
    assert decoded_localized.metric(qm.resources.LOGICAL_SPAN).numeric_enclosure() == (
        7,
        7,
    )


def test_native_rejects_conflicting_external_footprints() -> None:
    """Rust rejects two lane contracts for the same callable transform."""

    from qamomile.resources._rule_wire import encode_resource_analysis_request
    from qamomile.serialization import encode_analysis_request

    source = """
import qamomile as qm

@qm.qkernel(kind="unitary")
def oracle(
    left: qm.Q[qm.Bits[1]],
    right: qm.Q[qm.Bits[1]],
) -> tuple[qm.Q[qm.Bits[1]], qm.Q[qm.Bits[1]]]:
    ...

@qm.qkernel
def algorithm() -> tuple[qm.Q[qm.Bits[1]], qm.Q[qm.Bits[1]]]:
    left = qm.qubits(1)
    right = qm.qubits(1)
    return oracle(left, right)
"""
    module = qm.build_source(source, module_name="native_conflicting_footprints")
    oracle = module.function("oracle")
    rules = (
        qm.resources.ResourceRule(
            identity="example.conflicting-footprint.gates.v1",
            callable=oracle,
            metric=qm.resources.LOGICAL_GATES,
            relation=qm.resources.Relation.EQUALITY,
            expression=qm.resources.literal(1),
            provenance="example.conflicting-footprint.gates-contract.v1",
            scheduling_footprint=qm.resources.SchedulingFootprint(("left",)),
        ),
        qm.resources.ResourceRule(
            identity="example.conflicting-footprint.span.v1",
            callable=oracle,
            metric=qm.resources.LOGICAL_SPAN,
            relation=qm.resources.Relation.EQUALITY,
            expression=qm.resources.literal(1),
            provenance="example.conflicting-footprint.span-contract.v1",
            scheduling_footprint=qm.resources.SchedulingFootprint(("right",)),
        ),
    )
    with pytest.raises(ValueError, match="conflicting scheduling footprints"):
        encode_resource_analysis_request(
            encode_analysis_request(module, "algorithm"),
            rules,
        )


def test_native_symbolic_extra_metrics_compose_across_repetition() -> None:
    """A declared nonintegral metric is repeated by the native summary algebra."""

    from qamomile.resources._rule_wire import encode_resource_analysis_request
    from qamomile.resources.analysis import (
        _decode_native_resource_expressions,
        _decode_native_symbolic_count,
    )
    from qamomile.serialization import encode_analysis_request

    source = """
import qamomile as qm

@qm.qkernel(kind="unitary")
def oracle(state: qm.Q[qm.Bits[1]], rounds: qm.Int) -> qm.Q[qm.Bits[1]]:
    ...

@qm.qkernel
def algorithm(
    rounds: qm.Const[qm.Int],
    repetitions: qm.Const[qm.Int],
) -> qm.Bits[1]:
    state = qm.qubits(1)
    for _ in range(repetitions):
        state = oracle(state, rounds)
    return qm.measure(state)
"""
    module = qm.build_source(source, module_name="native_extra_metric_repeat")
    oracle = module.function("oracle")
    rounds = qm.resources.rule_parameter(
        oracle,
        "rounds",
        domain=qm.resources.Domain.NATURAL,
        constraints=("rounds >= 0",),
    )
    gates_rule = qm.resources.ResourceRule(
        identity="example.native-extra-metric.gates.v1",
        callable=oracle,
        metric=qm.resources.LOGICAL_GATES,
        relation=qm.resources.Relation.EQUALITY,
        expression=qm.resources.literal(1),
        provenance="example.native-extra-metric.gates-contract.v1",
    )
    metric = qm.resources.MetricDefinition(
        "example.oracle.analytic_cost",
        qm.resources.Coordinate.LOGICAL,
        qm.resources.Domain.REAL,
        qm.resources.ONE,
    )
    metric_rule = qm.resources.ResourceRule(
        identity="example.native-extra-metric.analytic.v1",
        callable=oracle,
        metric=metric,
        relation=qm.resources.Relation.EQUALITY,
        expression=qm.resources.log2(qm.resources.add(rounds, qm.resources.literal(1))),
        provenance="example.native-extra-metric.analytic-contract.v1",
    )
    rules = (gates_rule, metric_rule)
    payload = encode_resource_analysis_request(
        encode_analysis_request(module, "algorithm"),
        rules,
    )

    counts = native.inspect_symbolic_resource_analysis_request(payload)
    expressions = _decode_native_resource_expressions(counts.expressions)
    native_metric = _decode_native_symbolic_count(
        next(
            item.count
            for item in counts.extra_metrics
            if item.identifier == metric.identifier
        ),
        expressions,
    )
    study = qm.resources.logical(module["algorithm"], rules=rules, strict=True)

    assert native_metric.exact is True
    assert (
        native_metric.lower.evaluate({"rounds": 3, "repetitions": 2}).numeric() == 4.0
    )
    assert study.metric(metric).evaluate(
        {"rounds": 3, "repetitions": 2}
    ).numeric_enclosure() == (4.0, 4.0)


def test_native_symbolic_input_qubits_retain_a_bound_dependent_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dynamic control does not discard a proved dependent input width."""

    source = """
import qamomile as qm

@qm.qkernel
def sample(
    state: qm.Q[qm.Bits[width]],
    width: qm.Const[qm.Int],
    enabled: qm.Param[qm.Bool],
) -> qm.Bits[width]:
    if enabled:
        state = qm.h(state, target=0)
    return qm.measure(state)
"""
    program = qm.build_source(
        source,
        module_name="native_symbolic_input_width",
    ).specialize("sample", **{"width": 4})

    counts = native.inspect_symbolic_logical_counts(program.to_bytes())
    study = qm.resources.logical(program)
    decoded_study = qm.resources.logical(program)

    assert counts.input_qubits is not None
    assert counts.input_qubits.exact is True
    assert study.metric(qm.resources.LOGICAL_INPUT_QUBITS).numeric_enclosure() == (4, 4)
    assert (
        study.metric(qm.resources.LOGICAL_INPUT_QUBITS).numeric_enclosure()
        == decoded_study.metric(qm.resources.LOGICAL_INPUT_QUBITS).numeric_enclosure()
    )


@pytest.mark.parametrize(("enabled", "rounds"), [(False, -1), (True, 3)])
def test_native_symbolic_allocation_is_structural_not_dynamic(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
    rounds: int,
) -> None:
    """Dynamic branch choice and trip count do not scale reserved width."""

    source = """
import qamomile as qm

@qm.qkernel
def reserve(width: qm.Const[qm.Int]) -> qm.Unit:
    scratch = qm.qubits(width)
    qm.discard(scratch)
    return None

@qm.qkernel
def sample(
    state: qm.Q[qm.Bits[2]],
    enabled: qm.Param[qm.Bool],
    rounds: qm.Param[qm.Int],
) -> qm.Bits[2]:
    reserve(1)
    if enabled:
        reserve(2)
    else:
        reserve(3)
    for _ in range(rounds):
        reserve(2)
    return qm.measure(state)
"""
    program = qm.build_source(
        source,
        module_name="native_symbolic_structural_allocation",
    ).specialize("sample")

    counts = native.inspect_symbolic_logical_counts(program.to_bytes())
    study = qm.resources.logical(program).evaluate(enabled=enabled, rounds=rounds)
    decoded_study = qm.resources.logical(program).evaluate(
        enabled=enabled,
        rounds=rounds,
    )

    assert counts.allocated_qubits is not None
    assert counts.allocated_qubits.exact is True
    assert counts.circuit_qubits is not None
    assert counts.circuit_qubits.exact is True
    for metric, expected in {
        qm.resources.LOGICAL_ALLOCATED_QUBITS: 8,
        qm.resources.LOGICAL_CIRCUIT_QUBITS: 10,
    }.items():
        assert study.metric(metric).numeric_enclosure() == (expected, expected)
        assert (
            study.metric(metric).numeric_enclosure()
            == decoded_study.metric(metric).numeric_enclosure()
        )


@pytest.mark.parametrize(("enabled", "rounds"), [(False, -1), (True, 3)])
def test_native_symbolic_peak_tracks_dynamic_regions_and_dependent_widths(
    enabled: bool,
    rounds: int,
) -> None:
    """Runtime regions retain one exact ownership-root high-water expression."""

    source = """
import qamomile as qm

@qm.qkernel
def reserve(width: qm.Const[qm.Int]) -> qm.Unit:
    scratch = qm.qubits(width)
    qm.discard(scratch)
    return None

@qm.qkernel
def sample(
    state: qm.Q[qm.Bits[width]],
    width: qm.Const[qm.Int],
    enabled: qm.Param[qm.Bool],
    rounds: qm.Param[qm.Int],
) -> qm.Bits[width]:
    if enabled:
        reserve(width)
    else:
        reserve(2)
    for _ in range(rounds):
        reserve(3)
    return qm.measure(state)
"""
    program = qm.build_source(
        source,
        module_name="native_symbolic_peak_regions",
    ).specialize("sample", **{"width": 4})

    counts = native.inspect_symbolic_logical_counts(program.to_bytes())
    study = qm.resources.logical(program).evaluate(enabled=enabled, rounds=rounds)
    assert counts.peak_qubits is not None
    assert counts.peak_qubits.exact is True
    assert study.metric(qm.resources.LOGICAL_PEAK_QUBITS).numeric_enclosure() == (
        8,
        8,
    )


def test_native_symbolic_peak_matches_dynamic_branch_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A runtime branch high-water mark agrees with the Python ownership walk."""

    source = """
import qamomile as qm

@qm.qkernel
def reserve_two() -> qm.Unit:
    scratch = qm.qubits(2)
    qm.discard(scratch)
    return None

@qm.qkernel
def reserve_three() -> qm.Unit:
    scratch = qm.qubits(3)
    qm.discard(scratch)
    return None

@qm.qkernel
def sample(state: qm.Q[qm.Bits[1]], enabled: qm.Param[qm.Bool]) -> qm.Bits[1]:
    if enabled:
        reserve_two()
    else:
        reserve_three()
    return qm.measure(state)
"""
    program = qm.build_source(
        source,
        module_name="native_symbolic_peak_branch",
    ).specialize("sample")

    counts = native.inspect_symbolic_logical_counts(program.to_bytes())
    study = qm.resources.logical(program)
    decoded_study = qm.resources.logical(program)

    assert counts.peak_qubits is not None
    assert counts.peak_qubits.exact is True
    assert study.metric(qm.resources.LOGICAL_PEAK_QUBITS).numeric_enclosure() == (
        4,
        4,
    )
    assert (
        study.metric(qm.resources.LOGICAL_PEAK_QUBITS).numeric_enclosure()
        == decoded_study.metric(qm.resources.LOGICAL_PEAK_QUBITS).numeric_enclosure()
    )


def test_native_symbolic_counts_expose_a_closed_dynamic_branch_dag() -> None:
    """The bridge returns typed backward references for branch enclosures."""

    source = """
import qamomile as qm

@qm.qkernel
def sample(enabled: qm.Param[qm.Bool]) -> qm.Bits[1]:
    state = qm.qubits(1)
    if enabled:
        state = qm.h(state)
    return qm.measure(state)
"""
    program = qm.build_source(
        source,
        module_name="native_symbolic_branch",
    ).specialize("sample")

    counts = native.inspect_symbolic_logical_counts(program.to_bytes())
    kinds = {node.kind for node in counts.expressions}
    opcode_counts = {item.opcode: item.count for item in counts.opcodes}
    family_counts = {item.family: item.count for item in counts.families}

    assert counts.work.exact is False
    assert counts.work.upper is not None
    assert counts.measurements is not None and counts.measurements.exact is True
    assert counts.resets is not None and counts.resets.exact is True
    assert {"minimum", "maximum"}.issubset(kinds)
    assert opcode_counts["quantum.h"].exact is False
    assert family_counts["clifford_gates"].exact is False
    assert all(
        all(argument < index for argument in node.arguments)
        and (node.binder is None or node.binder < index)
        for index, node in enumerate(counts.expressions)
    )


@pytest.mark.parametrize(
    ("rounds", "expected_h"),
    [(-3, 0), (0, 0), (1, 1), (5, 5)],
)
def test_native_symbolic_loop_counts_evaluate_through_public_resources(
    monkeypatch: pytest.MonkeyPatch,
    rounds: int,
    expected_h: int,
) -> None:
    """A parameterized range remains reusable and evaluates without lowering."""

    source = """
import qamomile as qm

@qm.qkernel
def sample(rounds: qm.Param[qm.Int]) -> qm.Bits[1]:
    state = qm.qubits(1)
    for _ in range(rounds):
        state = qm.h(state)
    return qm.measure(state)
"""
    program = qm.build_source(
        source,
        module_name="native_symbolic_loop",
    ).specialize("sample")

    study = qm.resources.logical(program)
    evaluated = study.evaluate(rounds=rounds)
    decoded_evaluated = qm.resources.logical(program).evaluate(rounds=rounds)

    assert "native-symbolic-logical-counts" in study.derivations
    assert evaluated.metric(qm.resources.LOGICAL_WORK).numeric_enclosure() == (
        expected_h + 2,
        expected_h + 2,
    )
    assert evaluated.metric(
        qm.resources.logical_operation_count("quantum.h")
    ).numeric_enclosure() == (expected_h, expected_h)
    assert evaluated.metric(
        qm.resources.LOGICAL_CLIFFORD_GATES
    ).numeric_enclosure() == (
        expected_h,
        expected_h,
    )
    assert evaluated.metric(qm.resources.LOGICAL_WORK).numeric_enclosure() == (
        decoded_evaluated.metric(qm.resources.LOGICAL_WORK).numeric_enclosure()
    )


def test_native_symbolic_nested_loop_uses_a_bounded_sum() -> None:
    """Induction-dependent loop work is retained as a finite symbolic sum."""

    source = """
import qamomile as qm

@qm.qkernel
def sample(rounds: qm.Param[qm.Int]) -> qm.Bits[1]:
    state = qm.qubits(1)
    for outer in range(rounds):
        for _ in range(outer):
            state = qm.h(state)
    return qm.measure(state)
"""
    program = qm.build_source(
        source,
        module_name="native_symbolic_nested_loop",
    ).specialize("sample")

    counts = native.inspect_symbolic_logical_counts(program.to_bytes())
    study = qm.resources.logical(program).evaluate(rounds=4)

    assert any(node.kind == "bounded_sum" for node in counts.expressions)
    assert counts.allocated_qubits is not None
    assert counts.allocated_qubits.exact is True
    assert counts.circuit_qubits is not None
    assert counts.circuit_qubits.exact is True
    assert study.metric(qm.resources.LOGICAL_WORK).numeric_enclosure() == (8, 8)
    assert study.metric(qm.resources.LOGICAL_ALLOCATED_QUBITS).numeric_enclosure() == (
        1,
        1,
    )
    assert study.metric(qm.resources.LOGICAL_CIRCUIT_QUBITS).numeric_enclosure() == (
        1,
        1,
    )
    assert study.metric(
        qm.resources.logical_operation_count("quantum.h")
    ).numeric_enclosure() == (6, 6)
    assert study.metric(qm.resources.LOGICAL_GATES).numeric_enclosure() == (6, 6)


@pytest.mark.parametrize(("rounds", "trips"), [(-1, 0), (0, 0), (2, 2)])
def test_native_symbolic_events_scale_per_qubit_inside_a_loop(
    monkeypatch: pytest.MonkeyPatch,
    rounds: int,
    trips: int,
) -> None:
    """Symbolic instrument events scale by both trip count and logical width."""

    source = """
import qamomile as qm

@qm.qkernel
def sample(rounds: qm.Param[qm.Int]) -> qm.Bits[3]:
    state = qm.qubits(3)
    for _ in range(rounds):
        observed, state = qm.measure_reset(state)
    return qm.measure(state)
"""
    program = qm.build_source(
        source,
        module_name="native_symbolic_events",
    ).specialize("sample")

    native_study = qm.resources.logical(program).evaluate(rounds=rounds)
    decoded_study = qm.resources.logical(program).evaluate(rounds=rounds)

    expected_measurements = 3 * (trips + 1)
    expected_resets = 3 * trips
    assert native_study.metric(
        qm.resources.LOGICAL_MEASUREMENTS
    ).numeric_enclosure() == (expected_measurements, expected_measurements)
    assert native_study.metric(qm.resources.LOGICAL_RESETS).numeric_enclosure() == (
        expected_resets,
        expected_resets,
    )
    assert (
        native_study.metric(qm.resources.LOGICAL_MEASUREMENTS).numeric_enclosure()
        == decoded_study.metric(qm.resources.LOGICAL_MEASUREMENTS).numeric_enclosure()
    )


@pytest.mark.parametrize(("rounds", "trips"), [(-1, 0), (0, 0), (2, 2)])
def test_native_symbolic_events_instantiate_a_callee_quantum_width(
    monkeypatch: pytest.MonkeyPatch,
    rounds: int,
    trips: int,
) -> None:
    """A generic instrument call reuses its caller's width expression."""

    source = """
import qamomile as qm

@qm.qkernel
def instrument(
    state: qm.Q[qm.Bits[n]],
) -> tuple[qm.Bits[n], qm.Q[qm.Bits[n]]]:
    return qm.measure_reset(state)

@qm.qkernel
def relay(
    state: qm.Q[qm.Bits[m]],
) -> tuple[qm.Bits[m], qm.Q[qm.Bits[m]]]:
    return instrument(state)

@qm.qkernel
def sample(rounds: qm.Param[qm.Int]) -> qm.Bits[3]:
    state = qm.qubits(3)
    for _ in range(rounds):
        observed, state = relay(state)
    return qm.measure(state)
"""
    program = qm.build_source(
        source,
        module_name="native_dependent_symbolic_events",
    ).specialize("sample")

    counts = native.inspect_symbolic_logical_counts(program.to_bytes())
    native_study = qm.resources.logical(program).evaluate(rounds=rounds)
    decoded_study = qm.resources.logical(program).evaluate(rounds=rounds)

    expected_measurements = 3 * (trips + 1)
    expected_resets = 3 * trips
    assert counts.measurements is not None
    assert counts.resets is not None
    assert native_study.metric(
        qm.resources.LOGICAL_MEASUREMENTS
    ).numeric_enclosure() == (expected_measurements, expected_measurements)
    assert native_study.metric(qm.resources.LOGICAL_RESETS).numeric_enclosure() == (
        expected_resets,
        expected_resets,
    )
    assert (
        native_study.metric(qm.resources.LOGICAL_MEASUREMENTS).numeric_enclosure()
        == decoded_study.metric(qm.resources.LOGICAL_MEASUREMENTS).numeric_enclosure()
    )


def test_native_cleanup_excludes_an_explicit_source_adjoint() -> None:
    """Only compiler-proved restoration, not user algorithmic adjoint, is cleanup."""

    source = """
import qamomile as qm

@qm.qkernel
def step(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.h(state)

@qm.qkernel
def sample(state: qm.Q[qm.Bits[1]]) -> qm.Bits[1]:
    state = qm.adjoint(step, state)
    return qm.measure(state)
"""
    program = qm.build_source(
        source,
        module_name="native_explicit_adjoint_cleanup",
    ).specialize("sample")

    counts = native.inspect_exact_logical_counts(program.to_bytes())
    study = qm.resources.logical(program)

    assert counts.cleanup_work == 0
    assert study.metric(qm.resources.LOGICAL_CLEANUP_WORK).numeric_enclosure() == (
        0,
        0,
    )


def test_native_exact_peak_tracks_consuming_calls_and_callee_temporaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Owned inputs end at calls while nested scratch width raises the peak."""

    source = """
import qamomile as qm

@qm.qkernel
def consume(state: qm.Q[qm.Bits[1]]) -> qm.Unit:
    qm.discard(state)
    return None

@qm.qkernel
def borrow(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    scratch = qm.qubits(2)
    qm.discard(scratch)
    return state

@qm.qkernel
def sample() -> qm.Bits[1]:
    first = qm.qubits(1)
    consume(first)
    second = qm.qubits(1)
    second = borrow(second)
    return qm.measure(second)
"""
    program = qm.build_source(source, module_name="native_exact_peak_calls").specialize(
        "sample"
    )

    counts = native.inspect_exact_logical_counts(program.to_bytes())
    study = qm.resources.logical(program)
    decoded_study = qm.resources.logical(program)

    assert counts.peak_qubits == 3
    assert study.metric(qm.resources.LOGICAL_PEAK_QUBITS).numeric_enclosure() == (
        3,
        3,
    )
    assert (
        study.metric(qm.resources.LOGICAL_PEAK_QUBITS).numeric_enclosure()
        == decoded_study.metric(qm.resources.LOGICAL_PEAK_QUBITS).numeric_enclosure()
    )


@pytest.mark.parametrize("enabled", [True, False])
def test_native_exact_peak_takes_structured_branch_and_loop_high_water_mark(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
) -> None:
    """Branch alternatives and repeated bodies preserve simultaneous liveness."""

    source = """
import qamomile as qm

@qm.qkernel
def narrow() -> qm.Unit:
    scratch = qm.qubits(2)
    qm.discard(scratch)
    return None

@qm.qkernel
def wide() -> qm.Unit:
    scratch = qm.qubits(3)
    qm.discard(scratch)
    return None

@qm.qkernel
def sample(
    state: qm.Q[qm.Bits[1]],
    enabled: qm.Const[qm.Bool],
) -> qm.Bits[1]:
    if enabled:
        narrow()
    else:
        wide()
    for _ in range(2):
        scratch = qm.qubits(2)
        qm.discard(scratch)
    return qm.measure(state)
"""
    program = qm.build_source(
        source,
        module_name="native_exact_peak_regions",
    ).specialize("sample", **{"enabled": enabled})

    counts = native.inspect_exact_logical_counts(program.to_bytes())
    study = qm.resources.logical(program)
    decoded_study = qm.resources.logical(program)

    expected = decoded_study.metric(
        qm.resources.LOGICAL_PEAK_QUBITS
    ).numeric_enclosure()
    assert counts.peak_qubits == expected[0] == expected[1]
    assert (
        study.metric(qm.resources.LOGICAL_PEAK_QUBITS).numeric_enclosure() == expected
    )


@pytest.mark.parametrize(
    ("entrypoint", "expected"),
    [
        ("parallel", {"span": 3, "gates": 1, "measurements": 1}),
        ("disjoint", {"span": 3, "gates": 1, "measurements": 1}),
        ("synchronized", {"span": 5, "gates": 3, "t_gates": 2}),
        ("called", {"span": 3, "gates": 1, "measurements": 1}),
        ("reversed_call", {"span": 3, "gates": 1, "measurements": 1}),
        ("reshaped", {"span": 3, "gates": 1, "measurements": 1}),
        ("factor_parallel", {"span": 3, "gates": 1, "measurements": 1}),
        (
            "spectrum",
            {
                "span": 5,
                "gates": 3,
                "rotation_gates": 2,
                "toffoli_gates": 1,
                "non_clifford_gates": 3,
            },
        ),
        (
            "instruments",
            {"span": 3, "measurements": 1, "resets": 1},
        ),
        ("indexed", {"span": 3, "gates": 1, "measurements": 1}),
        ("shared", {"span": 7, "gates": 5, "measurements": 1}),
    ],
)
def test_native_exact_depths_preserve_lane_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
    expected: dict[str, int],
) -> None:
    """Exact span and depth families follow roots, lanes, calls, and loops."""

    import qamomile.resources.analysis as resource_analysis

    source = """
import qamomile as qm

@qm.qkernel
def parallel() -> tuple[qm.Bits[2], qm.Bits[2]]:
    left = qm.qubits(2)
    left = qm.h(left, target=0)
    right = qm.qubits(2)
    right = qm.h(right, target=1)
    return (qm.measure(left), qm.measure(right))

@qm.qkernel
def disjoint() -> qm.Bits[3]:
    state = qm.qubits(3)
    state = qm.h(state, target=0)
    state = qm.h(state, target=2)
    return qm.measure(state)

@qm.qkernel
def synchronized() -> qm.Bits[2]:
    state = qm.qubits(2)
    state = qm.t(state, target=0)
    state = qm.cx(state, control=0, target=1)
    state = qm.t(state, target=1)
    return qm.measure(state)

@qm.qkernel
def layer(state: qm.Q[qm.Bits[2]]) -> qm.Q[qm.Bits[2]]:
    state = qm.h(state, target=0)
    return qm.h(state, target=1)

@qm.qkernel
def called() -> qm.Bits[2]:
    return qm.measure(layer(qm.qubits(2)))

@qm.qkernel
def reversed_call() -> qm.Bits[2]:
    return qm.measure(qm.adjoint(layer, qm.qubits(2)))

@qm.qkernel
def reshaped() -> qm.Bits[2]:
    state = qm.qubits(2)
    left, right = qm.split_at(state, 1)
    left = qm.h(left)
    return qm.measure(qm.concat(left, right))

@qm.qkernel
def factor_parallel() -> qm.Bits[2]:
    state = qm.qubits(2)
    left, right = qm.split_at(state, 1)
    left = qm.h(left)
    right = qm.h(right)
    return qm.measure(qm.concat(left, right))

@qm.qkernel
def spectrum() -> qm.Bits[3]:
    state = qm.qubits(3)
    state = qm.rx(state, 0.25, target=0)
    state = qm.ccx(state, control1=0, control2=1, target=2)
    state = qm.rz(state, 0.5, target=2)
    return qm.measure(state)

@qm.qkernel
def instruments() -> qm.Bits[2]:
    state = qm.qubits(2)
    state = qm.reset(state)
    return qm.measure(state)

@qm.qkernel
def indexed() -> qm.Bits[5]:
    state = qm.qubits(5)
    for index in range(5):
        state = qm.h(state, target=index)
    return qm.measure(state)

@qm.qkernel
def shared() -> qm.Bits[5]:
    state = qm.qubits(5)
    for _ in range(5):
        state = qm.h(state, target=0)
    return qm.measure(state)
"""
    program = qm.build_source(
        source,
        module_name="native_exact_depths",
    ).specialize(entrypoint)

    counts = native.inspect_exact_logical_counts(program.to_bytes())
    study = qm.resources.logical(program)
    decoded_study = qm.resources.logical(program)

    assert counts.depths is not None
    for family, value in expected.items():
        assert counts.depths[family] == value
    for metric in resource_analysis._NATIVE_DEPTH_METRICS.values():
        assert (
            study.metric(metric).numeric_enclosure()
            == decoded_study.metric(metric).numeric_enclosure()
        )


@pytest.mark.parametrize(
    ("enabled", "span", "gate_depth"),
    [(False, 2, 0), (True, 3, 1)],
)
def test_native_exact_depth_selects_a_const_branch(
    enabled: bool,
    span: int,
    gate_depth: int,
) -> None:
    """A specialized condition schedules only its executable timing path."""

    source = """
import qamomile as qm

@qm.qkernel
def sample(enabled: qm.Const[qm.Bool]) -> qm.Bits[1]:
    state = qm.qubits(1)
    if enabled:
        state = qm.h(state)
    return qm.measure(state)
"""
    program = qm.build_source(
        source,
        module_name="native_exact_depth_branch",
    ).specialize("sample", **{"enabled": enabled})

    counts = native.inspect_exact_logical_counts(program.to_bytes())
    study = qm.resources.logical(program)

    assert counts.depths is not None
    assert counts.depths["span"] == span
    assert counts.depths["gates"] == gate_depth
    assert study.metric(qm.resources.LOGICAL_SPAN).numeric_enclosure() == (span, span)
    assert study.metric(qm.resources.LOGICAL_GATE_DEPTH).numeric_enclosure() == (
        gate_depth,
        gate_depth,
    )


@pytest.mark.parametrize("entrypoint", ["sample", "reversed_call"])
def test_native_symbolic_depth_proves_path_independent_control(
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
) -> None:
    """Equal dynamic branches retain one exact dependency schedule."""

    import qamomile.resources.analysis as resource_analysis

    source = """
import qamomile as qm

@qm.qkernel
def layer(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    state = qm.h(state)
    return qm.t(state)

@qm.qkernel
def sample(enabled: qm.Param[qm.Bool]) -> qm.Bits[1]:
    state = layer(qm.qubits(1))
    if enabled:
        state = state
    else:
        state = state
    return qm.measure(state)

@qm.qkernel
def reversed_call(enabled: qm.Param[qm.Bool]) -> qm.Bits[1]:
    state = qm.adjoint(layer, qm.qubits(1))
    if enabled:
        state = state
    else:
        state = state
    return qm.measure(state)
"""
    program = qm.build_source(
        source,
        module_name="native_symbolic_depths",
    ).specialize(entrypoint)

    counts = native.inspect_symbolic_logical_counts(program.to_bytes())
    study = qm.resources.logical(program)
    decoded_study = qm.resources.logical(program)

    assert counts.depths is not None
    assert {item.family for item in counts.depths} == set(
        resource_analysis._NATIVE_DEPTH_METRICS
    )
    for metric in resource_analysis._NATIVE_DEPTH_METRICS.values():
        assert (
            study.metric(metric).numeric_enclosure()
            == decoded_study.metric(metric).numeric_enclosure()
        )


def test_native_symbolic_depth_encloses_path_dependent_control() -> None:
    """Different dynamic schedules retain safe scalar depth enclosures."""

    from qamomile.resources.analysis import (
        _decode_native_resource_expressions,
        _decode_native_symbolic_count,
    )

    source = """
import qamomile as qm

@qm.qkernel
def sample(enabled: qm.Param[qm.Bool]) -> qm.Bits[1]:
    state = qm.qubits(1)
    if enabled:
        state = qm.h(state)
    else:
        state = qm.t(state)
    return qm.measure(state)
"""
    program = qm.build_source(
        source,
        module_name="native_symbolic_opaque_depths",
    ).specialize("sample")

    counts = native.inspect_symbolic_logical_counts(program.to_bytes())

    assert counts.depths is not None
    expressions = _decode_native_resource_expressions(counts.expressions)
    depths = {
        item.family: _decode_native_symbolic_count(item.count, expressions)
        for item in counts.depths
    }
    assert depths["gates"].exact is True
    assert (
        depths["gates"].lower.numeric(),
        depths["gates"].upper.numeric(),
    ) == (1, 1)
    for family in ("clifford_gates", "t_gates"):
        assert depths[family].exact is False
        assert (
            depths[family].lower.numeric(),
            depths[family].upper.numeric(),
        ) == (0, 1)
    assert depths["span"].exact is False
    assert (
        depths["span"].lower.numeric(),
        depths["span"].upper.numeric(),
    ) == (1, 3)


def test_native_open_family_depth_proves_a_dependent_lane_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unbound Const dimension stays symbolic across the native boundary."""

    import qamomile.resources.analysis as resource_analysis
    from qamomile.proto import qcore_pb2 as wire
    from qamomile.serialization import encode_analysis_request

    source = """
import qamomile as qm

@qm.qkernel
def layer(state: qm.Q[qm.Bits[n]], target: qm.Int) -> qm.Q[qm.Bits[n]]:
    state = qm.h(state, target=target)
    return qm.t(state, target=target)

@qm.qkernel
def mapped(n: qm.Const[qm.Int]) -> qm.Bits[n]:
    state = qm.qubits(n)
    for index in range(n):
        state = layer(state, index)
    return qm.measure(state)

@qm.qkernel
def unused() -> qm.Int:
    return 100
"""
    module = qm.build_source(source, module_name="native_open_family_depths")

    def reject_python_resolution(*_args: object, **_kwargs: object) -> object:
        """Fail if request encoding resolves the selector in Python."""

        raise AssertionError("Python entrypoint resolution was used")

    with monkeypatch.context() as context:
        context.setattr(type(module), "function", reject_python_resolution)
        payload = encode_analysis_request(module, "mapped")

    request = wire.QCorePackageProto.FromString(payload)
    assert request.entrypoint == "mapped"
    assert [function.symbol for function in request.functions] == [
        "native_open_family_depths.layer",
        "native_open_family_depths.mapped",
        "native_open_family_depths.unused",
    ]

    counts = native.inspect_symbolic_program_family_counts(payload)
    study = qm.resources.logical(module["mapped"]).evaluate(n=100_000)
    decoded_study = qm.resources.logical(module["mapped"]).evaluate(n=100_000)

    assert counts.depths is not None
    assert study.metric(qm.resources.LOGICAL_GATE_DEPTH).numeric_enclosure() == (2, 2)
    for metric in resource_analysis._NATIVE_DEPTH_METRICS.values():
        assert (
            study.metric(metric).numeric_enclosure()
            == decoded_study.metric(metric).numeric_enclosure()
        )
    with pytest.raises(ValueError, match="expected .*qamomile.program"):
        native.inspect_program(payload)
    closed = module.specialize("mapped", **{"n": 2})
    with pytest.raises(ValueError, match="expected .*qamomile.analysis"):
        native.inspect_symbolic_program_family_counts(closed.to_bytes())


def test_native_open_family_rejects_an_unresolved_raw_selector() -> None:
    """The analysis selector is resolved only after the full module is checked."""

    from qamomile.serialization import encode_analysis_request

    module = qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def selected() -> qm.Int:
    return 1
""",
        module_name="native_open_family_selector",
    )
    payload = encode_analysis_request(module, "missing")

    with pytest.raises(
        ValueError, match="Q3001: expected one function named 'missing'"
    ):
        native.inspect_symbolic_program_family_counts(payload)


@pytest.mark.parametrize(("rounds", "trips"), [(-1, 0), (0, 0), (2, 2)])
def test_native_symbolic_cleanup_scales_generated_restoration(
    monkeypatch: pytest.MonkeyPatch,
    rounds: int,
    trips: int,
) -> None:
    """Generated inverse and release work scale with a dynamic cleanup loop."""

    source = """
import qamomile as qm

@qm.qkernel
def sample(
    state: qm.Q[qm.Bits[1]],
    rounds: qm.Param[qm.Int],
) -> qm.Bits[1]:
    for _ in range(rounds):
        scratch = qm.qubits(1)
        scratch = qm.h(scratch)
    return qm.measure(state)
"""
    program = qm.build_source(
        source,
        module_name="native_symbolic_cleanup_loop",
    ).specialize("sample")

    counts = native.inspect_symbolic_logical_counts(program.to_bytes())
    study = qm.resources.logical(program).evaluate(rounds=rounds)
    decoded_study = qm.resources.logical(program).evaluate(rounds=rounds)

    expected = 2 * trips
    assert counts.cleanup_work.exact is True
    assert study.metric(qm.resources.LOGICAL_CLEANUP_WORK).numeric_enclosure() == (
        expected,
        expected,
    )
    assert (
        study.metric(qm.resources.LOGICAL_CLEANUP_WORK).numeric_enclosure()
        == decoded_study.metric(qm.resources.LOGICAL_CLEANUP_WORK).numeric_enclosure()
    )


@pytest.mark.parametrize("enabled", [True, False])
def test_native_symbolic_cleanup_encloses_dynamic_branches(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
) -> None:
    """Mutually exclusive generated cleanups retain a finite branch enclosure."""

    source = """
import qamomile as qm

@qm.qkernel
def cleanup_two() -> qm.Unit:
    scratch = qm.qubits(1)
    scratch = qm.h(scratch)
    return None

@qm.qkernel
def cleanup_three() -> qm.Unit:
    scratch = qm.qubits(1)
    scratch = qm.h(scratch)
    scratch = qm.t(scratch)
    return None

@qm.qkernel
def sample(state: qm.Q[qm.Bits[1]], enabled: qm.Param[qm.Bool]) -> qm.Bits[1]:
    if enabled:
        cleanup_two()
    else:
        cleanup_three()
    return qm.measure(state)
"""
    program = qm.build_source(
        source,
        module_name="native_symbolic_cleanup_branch",
    ).specialize("sample")

    counts = native.inspect_symbolic_logical_counts(program.to_bytes())
    study = qm.resources.logical(program).evaluate(enabled=enabled)
    decoded_study = qm.resources.logical(program).evaluate(enabled=enabled)

    assert counts.cleanup_work.exact is False
    assert counts.cleanup_work.upper is not None
    assert study.metric(qm.resources.LOGICAL_CLEANUP_WORK).numeric_enclosure() == (
        2,
        3,
    )
    assert (
        study.metric(qm.resources.LOGICAL_CLEANUP_WORK).numeric_enclosure()
        == decoded_study.metric(qm.resources.LOGICAL_CLEANUP_WORK).numeric_enclosure()
    )


@pytest.mark.parametrize(
    ("source", "entrypoint", "cleanup_count"),
    [
        (MODULAR_CLEANUP_SOURCE, "modular_cleanup", 1),
        (MULTI_INPUT_CLEANUP_SOURCE, "multi_input_cleanup", 3),
    ],
)
def test_native_bridge_replays_generated_cleanup_proofs(
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    entrypoint: str,
    cleanup_count: int,
) -> None:
    """Rust independently closes call, tensor, unpack, and release proof terms."""

    program = qm.build_source(
        source,
        module_name=f"native_{entrypoint}",
    ).specialize(entrypoint)
    payload = program.to_bytes()
    header = native.inspect_program(payload)
    counts = native.inspect_exact_logical_counts(payload)
    study = qm.resources.logical(program)
    decoded_study = qm.resources.logical(program)

    expected_cleanup = decoded_study.metric(
        qm.resources.LOGICAL_CLEANUP_WORK
    ).numeric_enclosure()
    assert expected_cleanup[0] > 0
    assert expected_cleanup[0] == expected_cleanup[1] == counts.cleanup_work
    assert (
        study.metric(qm.resources.LOGICAL_CLEANUP_WORK).numeric_enclosure()
        == expected_cleanup
    )
    assert header.verified is True
    assert header.discharged_cleanup_count == cleanup_count
    assert "quantum" in header.entry_effects
    assert "release" in header.entry_effects


def test_native_bridge_rejects_a_mismatched_inverse_step() -> None:
    """Changing a locally valid inverse opcode cannot forge cleanup evidence."""

    from qamomile.proto import qcore_pb2 as wire

    package = wire.QCorePackageProto.FromString(_gate_cleanup_program_bytes())
    inverse = next(
        operation
        for operation in package.functions[0].body.operations
        if operation.opcode == "qcore.adjoint"
    )
    next(
        attribute for attribute in inverse.attributes if attribute.name == "original"
    ).value.string_value = "quantum.x"
    package.package_digest = b""
    canonical = package.SerializeToString(deterministic=True)
    package.package_digest = hashlib.sha256(canonical).digest()

    with pytest.raises(ValueError, match="Q1113: inverse operation"):
        native.inspect_program(package.SerializeToString(deterministic=True))


def test_native_bridge_rejects_a_permuted_structural_inverse() -> None:
    """A type-valid tuple permutation cannot forge structural cleanup evidence."""

    from qamomile.proto import qcore_pb2 as wire

    program = qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def cleanup() -> qm.Unit:
    state = qm.qubits(2)
    left, right = qm.split_at(state, 1)
    left = qm.h(left)
    right = qm.x(right)
    state = qm.concat(left, right)
    return None
""",
        module_name="native_structural_cleanup_tamper",
    ).specialize("cleanup")
    package = wire.QCorePackageProto.FromString(program.to_bytes())
    inverse_unpack = next(
        operation
        for operation in package.functions[0].body.operations
        if operation.opcode == "qcore.unpack"
        and any(
            attribute.name == "proof"
            and attribute.value.string_value == "modular-adjoint"
            for attribute in operation.attributes
        )
    )
    left_region = inverse_unpack.results[0].region.SerializeToString()
    inverse_unpack.results[0].region.CopyFrom(inverse_unpack.results[1].region)
    inverse_unpack.results[1].region.ParseFromString(left_region)
    unpack_regions = {result.id: result.region for result in inverse_unpack.results}
    for operation in package.functions[0].body.operations:
        if (
            operation.opcode == "qcore.adjoint"
            and operation.operands[0] in unpack_regions
        ):
            operation.results[0].region.CopyFrom(unpack_regions[operation.operands[0]])
    inverse_tuple = next(
        operation
        for operation in package.functions[0].body.operations
        if operation.opcode == "qcore.tuple"
        and any(
            attribute.name == "proof"
            and attribute.value.string_value == "modular-adjoint"
            for attribute in operation.attributes
        )
    )
    inverse_tuple.operands.reverse()
    package.package_digest = b""
    canonical = package.SerializeToString(deterministic=True)
    package.package_digest = hashlib.sha256(canonical).digest()

    with pytest.raises(ValueError, match="Q1113: inverse operation"):
        native.inspect_program(package.SerializeToString(deterministic=True))


def test_native_bridge_synthesizes_coherent_cleanup_without_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rust closes control and select morphisms without compatibility helpers."""

    import qamomile.circuit as circuit_lowering

    calls: list[str] = []
    prepare_program = native.prepare_program

    def record_prepare(payload: bytes) -> bytes:
        """Record and delegate native coherent cleanup synthesis."""

        calls.append("native")
        return bytes(prepare_program(payload))

    monkeypatch.setattr(native, "prepare_program", record_prepare)
    program = qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def unchanged(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return state

@qm.qkernel
def flipped(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(state)

@qm.qkernel
def cleanup() -> qm.Unit:
    index = qm.qubits(1)
    target = qm.qubits(1)
    index, target = qm.multiplex(index, (unchanged, flipped), target)
    index, target = qm.control(flipped, index, target, control_state=0)
    system = qm.concat(index, target)
    return None
""",
        module_name="native_coherent_cleanup",
    ).specialize("cleanup")

    originals = {
        operation.attribute("original")
        for operation in program.entry.body.operations
        if operation.opcode == "qcore.adjoint"
    }
    native_circuit = circuit_lowering.lower_to_circuit(program)

    assert calls == ["native"]
    assert originals == {"qcore.control_call", "qcore.select_call"}
    assert ".__qamomile_adjoint" not in program.inspect()
    assert "qcore.forget" not in program.inspect()
    assert qm.evaluate(program) is None
    assert native_circuit.width == 2
    assert native.inspect_program(program.to_bytes()).discharged_cleanup_count == 3


def test_native_bridge_rejects_a_type_valid_coherent_inverse_tamper() -> None:
    """Changing a same-typed select table cannot forge cleanup evidence."""

    from qamomile.proto import qcore_pb2 as wire

    program = qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def unchanged(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return state

@qm.qkernel
def flipped(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(state)

@qm.qkernel
def cleanup() -> qm.Unit:
    index = qm.qubits(1)
    target = qm.qubits(1)
    index, target = qm.multiplex(index, (unchanged, flipped), target)
    system = qm.concat(index, target)
    return None
""",
        module_name="native_coherent_cleanup_tamper",
    ).specialize("cleanup")
    package = wire.QCorePackageProto.FromString(program.to_bytes())
    inverse = next(
        operation
        for operation in package.functions[0].body.operations
        if operation.opcode == "qcore.adjoint"
        and next(
            attribute
            for attribute in operation.attributes
            if attribute.name == "original"
        ).value.string_value
        == "qcore.select_call"
    )
    original_attributes = next(
        attribute
        for attribute in inverse.attributes
        if attribute.name == "original_attributes"
    ).value.mapping_value
    callees = next(
        entry for entry in original_attributes.entries if entry.key == "callees"
    ).value.tuple_value.values
    callees.reverse()
    package.package_digest = b""
    canonical = package.SerializeToString(deterministic=True)
    package.package_digest = hashlib.sha256(canonical).digest()

    with pytest.raises(ValueError, match="Q1113: inverse operation"):
        native.inspect_program(package.SerializeToString(deterministic=True))


def test_native_bridge_synthesizes_loop_cleanup_without_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rust closes a finite structured loop without unrolling or compatibility."""

    import qamomile.circuit as circuit_lowering

    calls: list[str] = []
    prepare_program = native.prepare_program

    def record_prepare(payload: bytes) -> bytes:
        """Record and delegate native structured-loop cleanup synthesis."""

        calls.append("native")
        return bytes(prepare_program(payload))

    monkeypatch.setattr(native, "prepare_program", record_prepare)
    program = qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def round_step(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    state = qm.s(state)
    return qm.t(state)

@qm.qkernel
def cleanup(rounds: qm.Const[qm.Int]) -> qm.Unit:
    state = qm.qubits(1)
    for _ in range(rounds):
        state = round_step(state)
    return None
""",
        module_name="native_loop_cleanup",
    ).specialize("cleanup", **{"rounds": 3})

    inverse = next(
        operation
        for operation in program.entry.body.operations
        if operation.opcode == "qcore.for"
        and operation.attribute("body_mode") == "adjoint"
    )
    native_circuit = circuit_lowering.lower_to_circuit(program)
    decoded_study = qm.resources.logical(program)

    counts = native.inspect_exact_logical_counts(program.to_bytes())
    cleanup = decoded_study.metric(qm.resources.LOGICAL_CLEANUP_WORK)

    assert calls == ["native"]
    assert inverse.attribute("reverse") is True
    assert inverse.attribute("proof") == "modular-adjoint"
    assert tuple(
        operation.opcode for operation in inverse.regions[0].entry.operations
    ) == ("qcore.call",)
    assert "qcore.forget" not in program.inspect()
    assert qm.evaluate(program) is None
    assert [
        instruction["op"]
        for instruction in json.loads(native_circuit.to_bytes())["body"]
    ] == ["s", "t", "s", "t", "s", "t", "tdg", "sdg", "tdg", "sdg", "tdg", "sdg"]
    assert native.inspect_program(program.to_bytes()).discharged_cleanup_count == 1
    assert counts.work == 14
    assert counts.opcodes == {
        "qcore.adjoint": 6,
        "quantum.alloc": 1,
        "quantum.release": 1,
        "quantum.s": 3,
        "quantum.t": 3,
    }
    assert counts.cleanup_work == 7
    assert cleanup.numeric_enclosure() == (7, 7)


def test_native_bridge_rejects_a_type_valid_loop_body_tamper() -> None:
    """Changing an inverse-loop gate cannot forge the retained region proof."""

    from qamomile.proto import qcore_pb2 as wire

    program = qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def cleanup() -> qm.Unit:
    state = qm.qubits(1)
    for _ in range(3):
        state = qm.h(state)
    return None
""",
        module_name="native_loop_cleanup_tamper",
    ).specialize("cleanup")
    package = wire.QCorePackageProto.FromString(program.to_bytes())
    inverse = next(
        operation
        for operation in package.functions[0].body.operations
        if operation.opcode == "qcore.for"
        and any(
            attribute.name == "body_mode" and attribute.value.string_value == "adjoint"
            for attribute in operation.attributes
        )
    )
    inverse.regions[0].blocks[0].operations[0].opcode = "quantum.x"
    package.package_digest = b""
    canonical = package.SerializeToString(deterministic=True)
    package.package_digest = hashlib.sha256(canonical).digest()

    with pytest.raises(ValueError, match="Q1113: inverse operation"):
        native.inspect_program(package.SerializeToString(deterministic=True))


def test_native_bridge_synthesizes_conditional_cleanup_without_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rust closes both arms of a Param conditional without compatibility."""

    calls: list[str] = []
    prepare_program = native.prepare_program

    def record_prepare(payload: bytes) -> bytes:
        """Record and delegate native conditional cleanup synthesis."""

        calls.append("native")
        return bytes(prepare_program(payload))

    monkeypatch.setattr(native, "prepare_program", record_prepare)
    program = qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def cleanup(enabled: qm.Param[qm.Bool]) -> qm.Unit:
    state = qm.qubits(1)
    if enabled:
        state = qm.s(state)
    else:
        state = qm.t(state)
    return None
""",
        module_name="native_conditional_cleanup",
    ).specialize("cleanup")

    inverse = next(
        operation
        for operation in program.entry.body.operations
        if operation.opcode == "qcore.if"
        and operation.attribute("body_mode") == "adjoint"
    )
    native_study = qm.resources.logical(program)
    decoded_study = qm.resources.logical(program)

    assert calls == ["native"]
    assert inverse.attribute("proof") == "modular-adjoint"
    assert tuple(branch.entry.operations[0].opcode for branch in inverse.regions) == (
        "quantum.s",
        "quantum.t",
    )
    assert "qcore.forget" not in program.inspect()
    assert qm.evaluate(program, enabled=False) is None
    assert qm.evaluate(program, enabled=True) is None
    assert native.inspect_program(program.to_bytes()).discharged_cleanup_count == 1
    assert native_study.metric(
        qm.resources.LOGICAL_CLEANUP_WORK
    ).numeric_enclosure() == (2, 2)
    assert decoded_study.metric(
        qm.resources.LOGICAL_CLEANUP_WORK
    ).numeric_enclosure() == (2, 2)


def test_native_static_circuit_inverts_measurement_conditional_cleanup() -> None:
    """The Rust circuit lowerer retains a device if with branch daggers."""

    import qamomile.circuit as circuit_lowering

    program = qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def cleanup() -> qm.Unit:
    selector = qm.qubits(1)
    selector = qm.h(selector)
    flag, selector = qm.measure_z(selector, 0)
    bits = qm.measure(selector)
    scratch = qm.qubits(1)
    if flag:
        scratch = qm.s(scratch)
    else:
        scratch = qm.t(scratch)
    return None
""",
        module_name="native_measurement_conditional_cleanup",
    ).specialize("cleanup")

    native_circuit = circuit_lowering.lower_to_circuit(program)
    body = json.loads(native_circuit.to_bytes())["body"]

    assert qm.evaluate(program, seed=2) is None
    assert [instruction["op"] for instruction in body[-2:]] == ["if", "if"]
    assert [instruction["op"] for instruction in body[-1]["regions"][0]] == ["sdg"]
    assert [instruction["op"] for instruction in body[-1]["regions"][1]] == ["tdg"]


def test_native_bridge_rejects_a_type_valid_conditional_body_tamper() -> None:
    """Changing one inverse branch cannot forge a conditional region proof."""

    from qamomile.proto import qcore_pb2 as wire

    program = qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def cleanup(enabled: qm.Param[qm.Bool]) -> qm.Unit:
    state = qm.qubits(1)
    if enabled:
        state = qm.s(state)
    else:
        state = qm.t(state)
    return None
""",
        module_name="native_conditional_cleanup_tamper",
    ).specialize("cleanup")
    package = wire.QCorePackageProto.FromString(program.to_bytes())
    inverse = next(
        operation
        for operation in package.functions[0].body.operations
        if operation.opcode == "qcore.if"
        and any(
            attribute.name == "body_mode" and attribute.value.string_value == "adjoint"
            for attribute in operation.attributes
        )
    )
    inverse.regions[0].blocks[0].operations[0].opcode = "quantum.z"
    package.package_digest = b""
    canonical = package.SerializeToString(deterministic=True)
    package.package_digest = hashlib.sha256(canonical).digest()

    with pytest.raises(ValueError, match="Q1113: inverse operation"):
        native.inspect_program(package.SerializeToString(deterministic=True))


def test_native_bridge_accepts_an_operand_derived_dependent_call() -> None:
    """Rust independently instantiates a Python-authored dependent call ABI."""

    header = native.inspect_program(_dependent_program_bytes())

    assert header.entrypoint == "native_dependent.main"
    assert header.function_count == 2


def test_native_bridge_accepts_an_entry_bound_dependent_call() -> None:
    """Rust uses the concrete entry Const binding for dependent call results."""

    header = native.inspect_program(_bound_dependent_program_bytes())

    assert header.entrypoint == "native_bound_dependent.main"
    assert header.function_count == 2


def test_native_bridge_rejects_tampered_program() -> None:
    """Rust must fail closed when canonical Python bytes are corrupted."""

    payload = bytearray(_program_bytes())
    payload[-1] ^= 1

    with pytest.raises(ValueError, match="Q40"):
        native.inspect_program(bytes(payload))


def test_native_bridge_rejects_resealed_unknown_semantic_type() -> None:
    """Rust must independently enforce the closed Qamomile type universe."""

    from qamomile.proto import qcore_pb2 as wire

    package = wire.QCorePackageProto.FromString(_program_bytes())
    package.functions[0].result_type.name = "Atom"
    package.package_digest = b""
    canonical = package.SerializeToString(deterministic=True)
    package.package_digest = hashlib.sha256(canonical).digest()

    with pytest.raises(ValueError, match="Q4002: invalid semantic type"):
        native.inspect_program(package.SerializeToString(deterministic=True))


def test_native_bridge_rejects_resealed_unknown_opcode() -> None:
    """Rust must reject operations outside the closed QCore vocabulary."""

    from qamomile.proto import qcore_pb2 as wire

    package = wire.QCorePackageProto.FromString(_program_bytes())
    package.functions[0].body.operations[0].opcode = "quantum.clone"
    package.package_digest = b""
    canonical = package.SerializeToString(deterministic=True)
    package.package_digest = hashlib.sha256(canonical).digest()

    with pytest.raises(ValueError, match="Q1032: unknown QCore opcode"):
        native.inspect_program(package.SerializeToString(deterministic=True))


def test_native_bridge_rejects_resealed_forged_classical_result() -> None:
    """Rust must rederive classical result types instead of trusting producers."""

    from qamomile.proto import qcore_pb2 as wire

    package = wire.QCorePackageProto.FromString(_program_bytes())
    body = package.functions[0].body
    template = body.operations[0]
    left = wire.OperationProto()
    left.CopyFrom(template)
    left.results[0].id = template.results[0].id + 1
    right = wire.OperationProto()
    right.CopyFrom(template)
    right.results[0].id = template.results[0].id + 2
    forged = wire.OperationProto()
    forged.CopyFrom(template)
    forged.opcode = "classical.add"
    forged.operands[:] = (left.results[0].id, right.results[0].id)
    forged.results[0].id = template.results[0].id + 3
    forged.results[0].type.name = "Bool"
    del forged.attributes[:]
    del body.operations[:]
    body.operations.extend((left, right, forged))
    body.terminator.operands[:] = (forged.results[0].id,)
    package.package_digest = b""
    canonical = package.SerializeToString(deterministic=True)
    package.package_digest = hashlib.sha256(canonical).digest()

    with pytest.raises(ValueError, match="Q1034: malformed classical.add"):
        native.inspect_program(package.SerializeToString(deterministic=True))


def test_native_bridge_rejects_resealed_unresolved_static_call() -> None:
    """Rust must close static call symbols before later whole-program checks."""

    from qamomile.proto import qcore_pb2 as wire

    package = wire.QCorePackageProto.FromString(_program_bytes())
    operation = package.functions[0].body.operations[0]
    operation.opcode = "qcore.call"
    del operation.attributes[:]
    attribute = operation.attributes.add(name="callee")
    attribute.value.string_value = "native_bridge.missing"
    package.package_digest = b""
    canonical = package.SerializeToString(deterministic=True)
    package.package_digest = hashlib.sha256(canonical).digest()

    with pytest.raises(ValueError, match="Q1026: unresolved static callable"):
        native.inspect_program(package.SerializeToString(deterministic=True))


def test_native_bridge_recursive_call_still_requires_a_valid_abi() -> None:
    """Admitting recursive edges cannot bypass parameter ABI verification."""

    from qamomile.proto import qcore_pb2 as wire

    package = wire.QCorePackageProto.FromString(_dependent_program_bytes())
    main = next(
        function for function in package.functions if function.symbol.endswith(".main")
    )
    call = next(
        operation
        for operation in main.body.operations
        if operation.opcode == "qcore.call"
    )
    next(
        attribute for attribute in call.attributes if attribute.name == "callee"
    ).value.string_value = main.symbol
    package.package_digest = b""
    canonical = package.SerializeToString(deterministic=True)
    package.package_digest = hashlib.sha256(canonical).digest()

    with pytest.raises(ValueError, match="Q1027:.*invalid parameter ABI"):
        native.inspect_program(package.SerializeToString(deterministic=True))


def test_native_bridge_rejects_result_only_dependent_binding() -> None:
    """A forged result shape cannot supply a binder absent from call operands."""

    from qamomile.proto import qcore_pb2 as wire

    package = wire.QCorePackageProto.FromString(_dependent_program_bytes())
    main = next(
        function for function in package.functions if function.symbol.endswith(".main")
    )
    literal = next(
        operation
        for operation in main.body.operations
        if operation.opcode == "qcore.constant"
    )
    next(
        attribute for attribute in literal.attributes if attribute.name == "value"
    ).value.int_value = -1
    package.package_digest = b""
    canonical = package.SerializeToString(deterministic=True)
    package.package_digest = hashlib.sha256(canonical).digest()

    with pytest.raises(ValueError, match="Q1028: call .* invalid result ABI"):
        native.inspect_program(package.SerializeToString(deterministic=True))


def test_native_bridge_rejects_non_positive_static_allocation_width() -> None:
    """A resealed literal cannot satisfy the native allocation equation."""

    from qamomile.proto import qcore_pb2 as wire

    source = """
import qamomile as qm

@qm.qkernel
def main() -> qm.Q[qm.Bits[1]]:
    return qm.qubits(1)
"""
    payload = (
        qm.build_source(source, module_name="native_domain")
        .specialize("main")
        .to_bytes()
    )
    package = wire.QCorePackageProto.FromString(payload)
    literal = next(
        operation
        for operation in package.functions[0].body.operations
        if operation.opcode == "qcore.constant"
    )
    next(
        attribute for attribute in literal.attributes if attribute.name == "value"
    ).value.int_value = 0
    package.package_digest = b""
    canonical = package.SerializeToString(deterministic=True)
    package.package_digest = hashlib.sha256(canonical).digest()

    with pytest.raises(
        ValueError, match="Q1035: quantum.alloc width must be a positive"
    ):
        native.inspect_program(package.SerializeToString(deterministic=True))


def test_native_bridge_rejects_resealed_hidden_value_region() -> None:
    """Rust must enforce the canonical has_region/presence pair on every value."""

    from qamomile.proto import qcore_pb2 as wire

    package = wire.QCorePackageProto.FromString(_program_bytes())
    result = package.functions[0].body.operations[0].results[0]
    result.has_region = False
    result.region.root = 17
    package.package_digest = b""
    canonical = package.SerializeToString(deterministic=True)
    package.package_digest = hashlib.sha256(canonical).digest()

    with pytest.raises(ValueError, match="Q4014"):
        native.inspect_program(package.SerializeToString(deterministic=True))


def test_native_bridge_rejects_resealed_unknown_asserted_kind() -> None:
    """Rust must independently close the persisted callable-kind vocabulary."""

    from qamomile.proto import qcore_pb2 as wire

    package = wire.QCorePackageProto.FromString(_program_bytes())
    package.functions[0].asserted_kind = "reversible"
    package.package_digest = b""
    canonical = package.SerializeToString(deterministic=True)
    package.package_digest = hashlib.sha256(canonical).digest()

    with pytest.raises(ValueError, match="Q4014"):
        native.inspect_program(package.SerializeToString(deterministic=True))


def test_native_bridge_evaluates_standalone_resource_expression_graph() -> None:
    """The independent distribution exposes exact resource-expression values."""

    from qamomile.resources._rule_wire import encode_resource_expression_graph

    expression = qm.resources.Expr(
        "add",
        qm.resources.Domain.RATIONAL,
        qm.resources.ONE,
        (
            qm.resources.literal(Fraction(1, 3)),
            qm.resources.literal(Fraction(1, 6)),
        ),
    )
    result = native.evaluate_resource_expression(
        encode_resource_expression_graph(expression)
    )

    assert result.kind == "rational"
    assert result.rational_numerator == "1"
    assert result.rational_denominator == "2"


def test_native_bridge_substitutes_qualified_resource_symbols() -> None:
    """The independent bridge projects Rust's capture-avoiding DAG transform."""

    from qamomile.resources._rule_wire import encode_resource_expression_graph

    expression = qm.resources.symbol("n", scope="native-bridge")
    replacement = qm.resources.literal(13)
    result = native.substitute_resource_expression(
        encode_resource_expression_graph(expression),
        [("native-bridge::n", encode_resource_expression_graph(replacement), False)],
    )
    evaluated = native.evaluate_resource_expression(result)

    assert evaluated.kind == "integer"
    assert evaluated.integer == "13"


def test_native_bridge_normalizes_literal_resource_subexpressions() -> None:
    """The independent bridge exposes the shared constructor normalizer."""

    from qamomile.resources._rule_wire import encode_resource_expression_graph

    expression = qm.resources.Expr(
        "absolute",
        qm.resources.Domain.INTEGER,
        qm.resources.ONE,
        (qm.resources.literal(-17),),
    )
    result = native.normalize_resource_expression(
        encode_resource_expression_graph(expression)
    )
    evaluated = native.evaluate_resource_expression(result)

    assert evaluated.kind == "integer"
    assert evaluated.integer == "17"


def test_native_bridge_constructs_typed_resource_expressions() -> None:
    """The bridge returns the Rust-derived algebraic domain and nominal unit."""

    from qamomile.proto import resource_analysis_pb2 as wire
    from qamomile.resources._rule_wire import (
        decode_resource_expression_graph,
        encode_resource_expression_graph,
    )

    quantity = qm.resources.symbol(
        "n",
        scope="native-bridge",
        domain=qm.resources.Domain.INTEGER,
        unit=qm.resources.SEMANTIC_OPS,
    )
    scale = qm.resources.literal(2)
    result = native.construct_resource_expression(
        "multiply",
        [
            encode_resource_expression_graph(quantity),
            encode_resource_expression_graph(scale),
        ],
        [],
        [],
        None,
        None,
    )
    expression = decode_resource_expression_graph(result)

    assert isinstance(expression, qm.resources.Expr)
    assert expression.domain is qm.resources.Domain.INTEGER
    assert expression.unit == qm.resources.SEMANTIC_OPS

    number = wire.ResourceNumberProto(integer="3")
    literal = decode_resource_expression_graph(
        native.construct_resource_literal(
            number.SerializeToString(deterministic=True),
            "1",
            None,
        )
    )
    assert isinstance(literal, qm.resources.Expr)
    assert literal.domain is qm.resources.Domain.NATURAL
    predicate = decode_resource_expression_graph(
        native.construct_resource_predicate(
            "greater",
            [
                encode_resource_expression_graph(literal),
                encode_resource_expression_graph(scale),
            ],
            [],
        )
    )
    assert isinstance(predicate, qm.resources.Predicate)
    assert predicate.truth() is True

    with pytest.raises(ValueError, match="different units"):
        native.construct_resource_expression(
            "add",
            [
                encode_resource_expression_graph(quantity),
                encode_resource_expression_graph(
                    qm.resources.literal(1, qm.resources.SECONDS)
                ),
            ],
            [],
            [],
            None,
            None,
        )


def test_native_bridge_projects_surface_code_resource_expressions() -> None:
    """The physical model returns four canonical expression graphs from Rust."""

    from qamomile.resources._rule_wire import encode_resource_expression_graph

    logical = encode_resource_expression_graph(
        qm.resources.literal(2, qm.resources.LOGICAL_QUBITS)
    )
    work = encode_resource_expression_graph(
        qm.resources.literal(0, qm.resources.SEMANTIC_OPS)
    )
    projection = native.project_surface_code_resources(
        logical,
        work,
        0.05,
        0.01,
        0.001,
        0.000_001,
    )

    assert isinstance(projection, tuple)
    assert len(projection) == 4
    evaluated = [native.evaluate_resource_expression(payload) for payload in projection]
    assert evaluated[0].integer == "3"
    assert evaluated[1].integer == "72"
    assert evaluated[2].real_bits == 0
    assert evaluated[3].real_bits == 0
    with pytest.raises(ValueError, match="Q4240"):
        native.project_surface_code_resources(
            logical,
            work,
            float("inf"),
            0.01,
            0.001,
            0.000_001,
        )


def test_native_bridge_validates_and_canonicalizes_target_profiles() -> None:
    """The bridge returns canonical bytes and a digest, or a Q703x rejection."""
    from qamomile.proto import target_profile_pb2 as profile_wire

    message = profile_wire.TargetProfileProto(schema_version=1, format="f")
    message.topology.num_qubits = 2
    message.topology.edges.add(control=0, target=1, symmetric=True)
    message.topology.one_qubit_operations.extend(["x", "h"])
    canonical, digest = native.validate_target_profile(
        message.SerializeToString(deterministic=True)
    )
    decoded = profile_wire.TargetProfileProto.FromString(canonical)

    assert list(decoded.topology.one_qubit_operations) == ["h", "x"]
    assert len(digest) == 32
    assert native.validate_target_profile(canonical) == (canonical, digest)
    assert native.MAX_TARGET_PROFILE_QUBITS == 65_536

    message.topology.edges.add(control=1, target=0)
    with pytest.raises(ValueError, match="Q7032"):
        native.validate_target_profile(message.SerializeToString(deterministic=True))
    with pytest.raises(ValueError, match="Q7030"):
        native.validate_target_profile(b"\xff")
