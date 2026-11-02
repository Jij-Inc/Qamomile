"""Operation substitution and complete Shor resource-estimation tests."""

from __future__ import annotations

import math

import pytest

import qamomile as qm
from qamomile import resources as qr
from qamomile.circuit import lower_to_circuit
from qamomile.stdlib import (
    ekera_hastad_factoring_state,
    modular_multi_exponentiation,
    product_tree_multi_exponentiation,
    shor_order_finding,
    shor_order_finding_state,
)

from ._programs import first_class_mod_pow_invalid_sample, first_class_mod_pow_sample

_PARAMETRIC_MULTI_EXPONENTIATION_SOURCE = """
import qamomile as qm

@qm.qkernel
def multi(
    d: qm.Const[qm.Int],
    m: qm.Const[qm.Int],
    n: qm.Const[qm.Int],
    b: qm.Const[qm.Int],
    bases: qm.Const[qm.Vector[qm.Int, d]],
    modulus: qm.Const[qm.Int],
    coordinates: qm.Q[qm.Bits[d * m]],
    target: qm.Q[qm.UInt[n]],
) -> tuple[qm.Q[qm.Bits[d * m]], qm.Q[qm.UInt[n]]]:
    return qm.mod_multi_pow(
        coordinates,
        target,
        bases=bases,
        base_width=b,
        modulus=modulus,
        coordinate_width=m,
    )
"""


_CONCRETE_MULTI_EXPONENTIATION_SOURCE = """
import qamomile as qm

@qm.qkernel
def sample(
    bases: qm.Const[qm.Vector[qm.Int, 1]],
) -> qm.Shot[qm.Bits[3]]:
    coordinates = qm.x(qm.qubits(1), target=0)
    target = qm.as_uint(qm.x(qm.qubits(2), target=0))
    coordinates, target = qm.mod_multi_pow(
        coordinates,
        target,
        bases=bases,
        base_width=2,
        modulus=3,
        coordinate_width=1,
    )
    return qm.measure(qm.concat(coordinates, qm.as_bits(target)))
"""


@qm.qkernel
def _one_bit_add_realization(
    state: qm.Q[qm.Bits[2]],
) -> qm.Q[qm.Bits[2]]:
    """Implement one-bit retained addition as one CNOT."""

    return qm.cx(state, control=0, target=1)


@qm.qkernel
def _one_bit_add_source(
    state: qm.Q[qm.Bits[2]],
) -> qm.Q[qm.Bits[2]]:
    """Retain one semantic addition application."""

    return qm.add_into(state)


@qm.qkernel
def _one_bit_add_closed() -> qm.Q[qm.Bits[2]]:
    """Allocate a closed state and invoke the semantic addition."""

    return _one_bit_add_source(qm.qubits(2))


@qm.qkernel
def _wrong_add_realization(
    state: qm.Q[qm.Bits[4]],
) -> qm.Q[qm.Bits[4]]:
    """Provide an intentionally incompatible realization ABI."""

    return state


@qm.qkernel
def _recursive_add_realization(
    state: qm.Q[qm.Bits[2]],
) -> qm.Q[qm.Bits[2]]:
    """Retain the operation being realized to form a rejected cycle."""

    return qm.add_into(state)


@qm.qkernel
def _small_mod_mul_sample() -> qm.Shot[qm.Bits[2]]:
    """Multiply one small basis value through the selected realization."""

    value = qm.as_uint(qm.x(qm.qubits(2), target=0))
    value = qm.mod_mul(value, factor=2, modulus=3)
    return qm.measure(qm.as_bits(value))


@qm.qkernel
def _small_mod_mul_add_sample() -> qm.Shot[qm.Bits[6]]:
    """Multiply-add three small basis values through the portable realization."""

    accumulator = qm.as_uint(qm.qubits(2))
    left = qm.as_uint(qm.x(qm.qubits(2), target=0))
    right = qm.as_uint(qm.x(qm.qubits(2), target=0))
    accumulator, left, right = qm.mod_mul_add(
        accumulator,
        left,
        right,
        modulus=3,
    )
    return qm.measure(
        qm.concat(
            qm.as_bits(accumulator), qm.concat(qm.as_bits(left), qm.as_bits(right))
        )
    )


@qm.qkernel
def _parallel_x_source() -> qm.Q[qm.Bits[2]]:
    """Apply two gates whose fixed dependency graph has one layer."""

    state = qm.qubits(2)
    state = qm.x(state, target=0)
    return qm.x(state, target=1)


@qm.qkernel
def _shared_control_source() -> qm.Q[qm.Bits[3]]:
    """Apply two gates serialized by one shared coherent control."""

    state = qm.qubits(3)
    state = qm.cx(state, control=0, target=1)
    return qm.cx(state, control=0, target=2)


@qm.qkernel
def _body_backed_baseline(
    state: qm.Q[qm.Bits[2]],
) -> qm.Q[qm.Bits[2]]:
    """Provide a two-gate body for whole-callable replacement tests."""

    state = qm.x(state, target=0)
    return qm.x(state, target=1)


@qm.qkernel
def _body_backed_alternative(
    state: qm.Q[qm.Bits[2]],
) -> qm.Q[qm.Bits[2]]:
    """Provide an ABI-compatible one-gate alternative body."""

    return qm.cx(state, control=0, target=1)


@qm.qkernel
def _body_backed_caller(
    state: qm.Q[qm.Bits[2]],
) -> qm.Q[qm.Bits[2]]:
    """Call the replaceable body through an ordinary verified edge."""

    return _body_backed_baseline(state)


@qm.qkernel
def _body_backed_wrong_abi(
    state: qm.Q[qm.Bits[3]],
) -> qm.Q[qm.Bits[3]]:
    """Provide an intentionally incompatible callable ABI."""

    return state


def test_custom_qkernel_realization_is_shared_by_resources_and_circuits() -> None:
    """Link one QKernel transiently without changing semantic QCore identity."""

    options = qm.RealizationOptions({qm.add_into: _one_bit_add_realization})
    source = _one_bit_add_source.module()
    original = source.to_bytes()
    study = qr.project(
        qr.logical(_one_bit_add_source),
        qr.GateCostModel(realizations=options),
    )
    assert study.metric(qr.portable_metric("gates")).numeric_enclosure() == (1, 1)
    program = _one_bit_add_closed.specialize()
    circuit = lower_to_circuit(program, realizations=options)
    assert circuit.operation_count == 1
    assert circuit.semantic_id == program.semantic_digest
    assert source.to_bytes() == original


def test_body_backed_callable_replacement_preserves_identity_and_callers() -> None:
    """Replace an entrypoint or callee body without mutating source QCore."""

    options = qm.RealizationOptions({_body_backed_baseline: _body_backed_alternative})
    source = _body_backed_baseline.module()
    original = source.to_bytes()
    for kernel in (_body_backed_baseline, _body_backed_caller):
        study = qr.project(qr.logical(kernel), qr.GateCostModel(realizations=options))
        assert study.metric(qr.portable_metric("gates")).numeric_enclosure() == (
            1,
            1,
        )
    assert source.to_bytes() == original


def test_body_backed_callable_replacement_requires_the_exact_abi() -> None:
    """Reject a replacement that would alter the target signature."""

    options = qm.RealizationOptions({_body_backed_baseline: _body_backed_wrong_abi})
    with pytest.raises(ValueError, match="Q8068"):
        qr.project(
            qr.logical(_body_backed_baseline),
            qr.GateCostModel(realizations=options),
        )


def test_portable_depth_is_the_checked_fixed_graph_makespan() -> None:
    """Distinguish parallel gate count from a shared-control critical path."""

    parallel = qr.project(qr.logical(_parallel_x_source), qr.GateCostModel())
    shared = qr.project(qr.logical(_shared_control_source), qr.GateCostModel())
    assert parallel.metric(qr.portable_metric("gates")).numeric_enclosure() == (2, 2)
    assert parallel.metric(qr.portable_metric("gate_depth")).numeric_enclosure() == (
        1,
        1,
    )
    assert shared.metric(qr.portable_metric("gate_depth")).numeric_enclosure() == (
        2,
        2,
    )
    parallel.require(qr.portable_metric("gate_depth"), coverage="exact")
    shared.require(qr.portable_metric("gate_depth"), coverage="exact")
    assert qr.ResourceStudy.from_bytes(parallel.to_bytes()) == parallel


def test_static_schoolbook_mod_mul_has_exact_gate_count_and_makespan() -> None:
    """Schedule the selected arithmetic circuit rather than its serial bound."""

    options = qm.RealizationOptions({qm.mod_mul: "mod_mul.schoolbook_ripple"})
    study = qr.project(
        qr.logical(_small_mod_mul_sample),
        qr.GateCostModel(realizations=options),
    )
    gates = study.metric(qr.portable_metric("gates"))
    depth = study.metric(qr.portable_metric("gate_depth"))
    assert gates.numeric_enclosure() == (1681, 1681)
    assert depth.numeric_enclosure() == (1653, 1653)
    assert depth.numeric_enclosure()[1] < gates.numeric_enclosure()[1]
    study.require(qr.portable_metric("gates"), coverage="exact")
    study.require(qr.portable_metric("gate_depth"), coverage="exact")
    assert qr.ResourceStudy.from_bytes(study.to_bytes()) == study


def test_symbolic_schoolbook_mod_mul_add_encloses_the_realized_gate_graph() -> None:
    """Keep the multiply-add work and dependency bounds above its fixed graph."""

    options = qm.RealizationOptions({qm.mod_mul_add: "mod_mul_add.schoolbook_ripple"})
    study = qr.project(
        qr.logical(_small_mod_mul_add_sample),
        qr.GateCostModel(realizations=options),
    )
    gates = study.metric(qr.portable_metric("gates"))
    depth = study.metric(qr.portable_metric("gate_depth"))
    module = _small_mod_mul_add_sample.module()
    program = module.specialize(module.functions[0].symbol)
    circuit = lower_to_circuit(program)
    realized_gates = circuit.operation_count - circuit.measurement_count
    gate_lower, gate_upper = gates.numeric_enclosure()
    depth_upper = depth.numeric_enclosure()[1]
    assert gate_upper is not None and gate_lower <= realized_gates <= gate_upper
    assert depth_upper is not None and realized_gates <= depth_upper < gate_upper


def test_custom_realization_rechecks_abi_and_cycles() -> None:
    """Reject incompatible and recursively self-expanding definitions."""

    for implementation in (_wrong_add_realization, _recursive_add_realization):
        options = qm.RealizationOptions({qm.add_into: implementation})
        with pytest.raises((ValueError, qm.VerificationError)):
            qr.project(
                qr.logical(_one_bit_add_source),
                qr.GateCostModel(realizations=options),
            )


def test_square_and_multiply_closes_large_symbolic_shor_costs() -> None:
    """Produce finite gate, dependency-depth, and clean-workspace expressions."""

    options = qm.RealizationOptions({qm.mod_pow: "mod_pow.square_and_multiply"})
    study = qr.project(
        qr.logical(shor_order_finding_state),
        qr.GateCostModel(realizations=options),
    )
    point = {"n": 2048, "p": 4096, "precision": 4096}
    gates = study.metric(qr.portable_metric("gates")).evaluate(point)
    toffoli = study.metric(qr.portable_metric("toffoli_gates")).evaluate(point)
    depth = study.metric(qr.portable_metric("gate_depth")).evaluate(point)
    scratch = study.metric(qr.portable_metric("clean_ancilla_qubits")).evaluate(point)
    assert gates.numeric_enclosure() == (41_953_280, 7_426_086_139_904)
    assert toffoli.numeric_enclosure() == (
        0,
        7_425_976_328_192,
    )
    assert depth.numeric_enclosure() == (None, 7_426_086_139_904)
    assert scratch.numeric_enclosure() == (0, 4_105)
    assert not gates.unknown_contributors
    assert not toffoli.unknown_contributors
    assert not depth.unknown_contributors
    assert not scratch.unknown_contributors


@pytest.mark.parametrize("width", [2, 3, 4])
def test_symbolic_mod_pow_child_selection_preserves_workspace_enclosures(
    width: int,
) -> None:
    """Price the selected child without assuming unknown bases are nonidentities."""

    module = qm.build_source(f"""
import qamomile as qm
@qm.qkernel
def power(
    base: qm.Const[qm.Int],
    target: qm.Q[qm.UInt[{width}]],
    exponent: qm.Q[qm.UInt[2]],
) -> tuple[qm.Q[qm.UInt[{width}]], qm.Q[qm.UInt[2]]]:
    return qm.mod_pow(target, exponent, base=base, modulus=3)
""")
    logical = qr.logical(module)
    for schoolbook in (False, True):
        options = qm.RealizationOptions(
            {qm.mod_mul: "mod_mul.schoolbook_ripple"} if schoolbook else {}
        )
        study = qr.project(logical, qr.GateCostModel(realizations=options))
        gates = study.metric(qr.portable_metric("gates"))
        scratch = study.metric(qr.portable_metric("clean_ancilla_qubits"))
        gate_lower, gate_upper = gates.numeric_enclosure()
        assert gate_lower == 0 and gate_upper is not None and gate_upper > 0
        expected_scratch = (
            2 * width + 9 if schoolbook or width > 3 else max(width - 2, 0)
        )
        assert scratch.numeric_enclosure() == (0, expected_scratch)
        assert not gates.unknown_contributors
        assert not scratch.unknown_contributors
        assert qr.ResourceStudy.from_bytes(study.to_bytes()) == study


def test_symbolic_independent_multi_pow_uses_small_child_workspace() -> None:
    """Keep exponent inputs outside the small permutation width and workspace."""

    module = qm.build_source("""
import qamomile as qm
@qm.qkernel
def multi(
    bases: qm.Const[qm.Vector[qm.Int, 1]],
    coordinates: qm.Q[qm.Bits[2]],
    target: qm.Q[qm.UInt[2]],
) -> tuple[qm.Q[qm.Bits[2]], qm.Q[qm.UInt[2]]]:
    return qm.mod_multi_pow(
        coordinates, target, bases=bases, base_width=2,
        modulus=3, coordinate_width=2,
    )
""")
    study = qr.project(
        qr.logical(module), qr.GateCostModel(realizations=qm.RealizationOptions())
    )
    scratch = study.metric(qr.portable_metric("clean_ancilla_qubits"))
    assert scratch.numeric_enclosure() == (0, 0)
    assert not scratch.unknown_contributors
    gates = study.metric(qr.portable_metric("gates"))
    assert gates.numeric_enclosure() == (0, 108)


def test_complete_shor_sampler_uses_only_public_dimension_symbols() -> None:
    """Close the sampled algorithm after substituting width and precision."""

    study = qr.project(qr.logical(shor_order_finding), qr.GateCostModel())
    symbolic_gates = study.metric(qr.portable_metric("gates"))
    symbolic_toffoli = study.metric(qr.portable_metric("toffoli_gates"))
    assert "power(width, 2)" in str(symbolic_gates)
    assert "width * (width + 1)" not in str(symbolic_gates)
    assert "precision" in str(symbolic_toffoli)
    assert "power(width, 2)" in str(symbolic_toffoli)
    point = {"width": 2048, "precision": 4096}
    gates = symbolic_gates.evaluate(point)
    depth = study.metric(qr.portable_metric("gate_depth")).evaluate(point)
    scratch = study.metric(qr.portable_metric("clean_ancilla_qubits")).evaluate(point)
    assert gates.numeric_enclosure() == (41_953_281, 7_426_082_555_905)
    assert depth.numeric_enclosure() == (None, 7_426_082_555_905)
    assert scratch.numeric_enclosure() == (0, 4_105)
    assert not gates.unknown_contributors
    assert not depth.unknown_contributors
    assert not scratch.unknown_contributors


def test_windowed_shor_sampler_has_a_finite_rsa_size_projection() -> None:
    """Evaluate the exact windowed model without constructing the large circuit."""

    options = qm.RealizationOptions({qm.mod_pow: "mod_pow.windowed_lookup"})
    study = qr.project(
        qr.logical(shor_order_finding),
        qr.GateCostModel(realizations=options),
    )
    point = {"width": 2048, "precision": 4096}
    assert study.metric(qr.portable_metric("gates")).evaluate(
        point
    ).numeric_enclosure() == (41_953_281, 4_682_786_351_901)
    assert study.metric(qr.portable_metric("toffoli_gates")).evaluate(
        point
    ).numeric_enclosure() == (0, 4_649_218_767_168)
    assert study.metric(qr.portable_metric("gate_depth")).evaluate(
        point
    ).numeric_enclosure() == (None, 4_682_786_351_901)
    assert study.metric(qr.portable_metric("clean_ancilla_qubits")).evaluate(
        point
    ).numeric_enclosure() == (0, 6_154)


def test_shor_source_breakdown_retains_one_semantic_mod_pow() -> None:
    """Inspect source operations without requiring a portable gate basis."""

    breakdown = qr.ResourceBreakdown.from_study(
        qr.logical(shor_order_finding), expand=None
    ).evaluate(width=2048, precision=4096)
    mod_pow = [
        row for row in breakdown.rows if "quantum.mod_pow" in row.metric.identifier
    ]
    assert len(mod_pow) == 1
    assert mod_pow[0].numeric_enclosure() == (1, 1)


def test_body_backed_controlled_arithmetic_has_a_finite_portable_profile() -> None:
    """Price clean-MCX transforms used by a source-backed arithmetic body."""

    state = ekera_hastad_factoring_state(
        generator=2,
        modulus=5,
        parameter_width=1,
    )
    study = qr.project(qr.logical(state), qr.GateCostModel())
    for name in ("gates", "gate_depth", "clean_ancilla_qubits"):
        value = study.metric(qr.portable_metric(name))
        assert value.numeric_enclosure()[1] is not None
        assert not value.unknown_contributors


def test_product_tree_is_selectable_as_a_complete_multi_exponentiation_body() -> None:
    """Select the product tree through the generic callable mechanism."""

    configuration = {
        "bases": (2, 2, 2, 2),
        "modulus": (1 << 16) - 1,
        "coordinate_width": 4,
    }
    baseline = modular_multi_exponentiation(**configuration)
    alternative = product_tree_multi_exponentiation(**configuration)
    original = baseline.module().to_bytes()
    selected = qr.project(
        qr.logical(baseline),
        qr.GateCostModel(realizations=qm.RealizationOptions({baseline: alternative})),
    )
    direct = qr.project(qr.logical(alternative), qr.GateCostModel())
    for name in ("gates", "gate_depth", "clean_ancilla_qubits"):
        selected_value = selected.metric(qr.portable_metric(name))
        direct_value = direct.metric(qr.portable_metric(name))
        assert selected_value.numeric_enclosure() == direct_value.numeric_enclosure()
        assert not selected_value.unknown_contributors
    assert baseline.module().to_bytes() == original


def test_parametric_multi_exponentiation_keeps_d_m_and_n_independent() -> None:
    """Retain independent dimensions through the selected realization graph."""

    module = qm.build_source(
        _PARAMETRIC_MULTI_EXPONENTIATION_SOURCE,
        module_name="parametric_multi_exponentiation",
    )
    options = qm.RealizationOptions({qm.mod_multi_pow: "mod_multi_pow.product_tree"})
    study = qr.project(qr.logical(module), qr.GateCostModel(realizations=options))
    gates = study.metric(qr.portable_metric("gates"))
    depth = study.metric(qr.portable_metric("gate_depth"))
    workspace = study.metric(qr.portable_metric("clean_ancilla_qubits"))

    for symbol in ("d", "m", "n"):
        assert symbol in str(gates)
        assert symbol in str(depth)
    point = {
        "d": 4,
        "m": 4,
        "n": 16,
        "b": 2,
        "bases": (2, 2, 2, 2),
        "modulus": (1 << 16) - 1,
    }
    assert gates.evaluate(point).numeric_enclosure() == (4_319_424, 5_067_104)
    assert depth.evaluate(point).numeric_enclosure() == (None, 5_043_649)
    assert workspace.evaluate(point).numeric_enclosure() == (0, 196)
    assert (
        depth.evaluate(point).numeric_enclosure()[1]
        < gates.evaluate(point).numeric_enclosure()[1]
    )


def test_independent_multi_exponentiation_materializes_one_checked_circuit() -> None:
    """Compile the semantic primitive through its executable baseline."""

    module = qm.build_source(
        _CONCRETE_MULTI_EXPONENTIATION_SOURCE,
        module_name="concrete_multi_exponentiation",
    )
    program = module.specialize("sample", bases=[2])
    circuit = lower_to_circuit(program)
    assert circuit.realizations == ("mod_multi_pow.independent", "arithmetic.portable")
    assert circuit.operation_count == 14
    assert circuit.width == 3
    assert circuit.measurement_count == 3


@pytest.mark.resource_slow
def test_product_tree_changes_schoolbook_scaling_from_cubic_to_five_halves() -> None:
    """Lock gate/makespan scaling and the product-tree workspace tradeoff."""

    rows = []
    for width in (16, 25, 36, 49):
        dimension = math.isqrt(width)
        configuration = {
            "bases": (2,) * dimension,
            "modulus": (1 << width) - 1,
            "coordinate_width": dimension,
        }
        baseline_logical = qr.logical(modular_multi_exponentiation(**configuration))
        product_logical = qr.logical(product_tree_multi_exponentiation(**configuration))
        baseline = qr.project(baseline_logical, qr.GateCostModel())
        product = qr.project(product_logical, qr.GateCostModel())

        def upper(study: qr.ResourceStudy, metric: str) -> int:
            """Return one finite portable upper bound.

            Args:
                study (qr.ResourceStudy): Fully projected resource study.
                metric (str): Portable metric suffix.

            Returns:
                int: Finite upper endpoint.
            """

            value = study.metric(qr.portable_metric(metric))
            assert not value.unknown_contributors
            endpoint = value.numeric_enclosure()[1]
            assert endpoint is not None
            return endpoint

        rows.append(
            (
                width,
                upper(baseline, "gates"),
                upper(baseline, "gate_depth"),
                baseline_logical.metric(qr.LOGICAL_PEAK_QUBITS).numeric_enclosure()[1]
                + upper(baseline, "clean_ancilla_qubits"),
                upper(product, "gates"),
                upper(product, "gate_depth"),
                product_logical.metric(qr.LOGICAL_PEAK_QUBITS).numeric_enclosure()[1]
                + upper(product, "clean_ancilla_qubits"),
            )
        )

    assert rows == [
        (16, 1_902_625, 1_902_625, 73, 5_067_010, 5_057_794, 252),
        (25, 7_074_376, 7_074_376, 109, 15_335_987, 15_317_987, 435),
        (36, 20_827_225, 20_827_225, 153, 37_440_374, 37_409_270, 636),
        (49, 52_068_136, 52_068_136, 205, 80_097_383, 80_047_991, 893),
    ]
    for metric_column in (1, 2):
        exponents = [
            math.log(right[metric_column] / left[metric_column])
            / math.log(right[0] / left[0])
            for left, right in zip(rows, rows[1:])
        ]
        assert all(2.85 < exponent < 3.1 for exponent in exponents)
    for metric_column in (4, 5):
        exponents = [
            math.log(right[metric_column] / left[metric_column])
            / math.log(right[0] / left[0])
            for left, right in zip(rows, rows[1:])
        ]
        assert all(2.25 < exponent < 2.7 for exponent in exponents)
    assert all(row[3] <= 5 * row[0] for row in rows)
    assert all(row[6] <= 4 * row[0] ** 1.5 for row in rows)


def test_schoolbook_mod_mul_is_total_on_small_basis() -> None:
    """Match the total modular permutation, including out-of-domain states."""

    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Statevector

    from qamomile.qiskit import Qiskit, to_circuit

    options = qm.RealizationOptions({qm.mod_mul: "mod_mul.schoolbook_ripple"})
    for value in range(4):
        bits = (bool(value & 1), bool(value & 2))
        module = qm.build_source(
            f"""
import qamomile as qm
@qm.qkernel
def multiply() -> qm.Q[qm.UInt[2]]:
    state = qm.as_uint(qm.prepare({bits!r}))
    return qm.mod_mul(state, factor=2, modulus=3)
"""
        )
        circuit = to_circuit(
            module.specialize("multiply"),
            target=Qiskit(realizations=options),
        )
        state = Statevector.from_instruction(circuit)
        observed = max(range(len(state.data)), key=lambda index: abs(state.data[index]))
        expected = 2 * value % 3 if value < 3 else value
        assert observed == expected
        projection = lower_to_circuit(
            module.specialize("multiply"), realizations=options
        )
        assert (
            projection.realization_stages.count("realization:mod_mul.schoolbook_ripple")
            == 1
        )


@pytest.mark.parametrize(
    "choice", ["mod_pow.square_and_multiply", "mod_pow.windowed_lookup"]
)
@pytest.mark.parametrize(
    ("program", "expected"),
    [
        (first_class_mod_pow_sample, (0, 0, 1, 0, 1)),
        (first_class_mod_pow_invalid_sample, (0, 1, 1, 0, 1)),
    ],
)
def test_mod_pow_realizations_are_exact_on_standard_residues(
    choice: str,
    program: qm.QKernel,
    expected: tuple[int, ...],
) -> None:
    """Execute both implementations and preserve the total-domain contract."""

    pytest.importorskip("qiskit_aer")
    from qamomile.qiskit import Aer

    options = qm.RealizationOptions({qm.mod_pow: choice})
    result = qm.run(program, on=Aer(realizations=options), shots=4, seed=0).result
    assert result.counts == {expected: 4}
    circuit = lower_to_circuit(program.specialize(), realizations=options)
    assert circuit.realizations == (
        (choice, "arithmetic.portable")
        if choice == "mod_pow.square_and_multiply"
        else (choice,)
    )


def test_mod_pow_realizations_close_symbolic_gate_costs() -> None:
    """Give both exact strategies finite gates, depth, and workspace."""

    for choice in ("mod_pow.square_and_multiply", "mod_pow.windowed_lookup"):
        options = qm.RealizationOptions({qm.mod_pow: choice})
        study = qr.project(
            qr.logical(shor_order_finding_state),
            qr.GateCostModel(realizations=options),
        )
        for name in ("gates", "toffoli_gates", "gate_depth", "clean_ancilla_qubits"):
            value = study.metric(qr.portable_metric(name)).evaluate(
                {"n": 64, "p": 128, "precision": 128}
            )
            assert value.numeric_enclosure()[1] is not None
            assert not value.unknown_contributors


def test_coset_mod_pow_reports_generic_accuracy_and_finite_resources() -> None:
    """Expose a generic output guarantee, not representation-specific metrics."""

    assert (
        qr.CIRCUIT_OUTPUT_TRACE_DISTANCE_BOUND.composition is qr.Composition.ERROR_BOUND
    )
    options = qm.RealizationOptions(
        {qm.mod_pow: qm.realize("mod_pow.coset_windowed", padding_bits=40)}
    )
    study = qr.project(
        qr.logical(shor_order_finding_state),
        qr.GateCostModel(realizations=options),
    )
    point = {"n": 64, "p": 128, "precision": 128}
    error = study.metric(qr.CIRCUIT_OUTPUT_TRACE_DISTANCE_BOUND).evaluate(point)
    lower, upper = error.numeric_enclosure()
    assert lower == 0
    assert upper is not None and 0 < upper < 0.001
    assert qr.ResourceStudy.from_bytes(study.to_bytes()) == study
    for name in ("gates", "toffoli_gates", "gate_depth", "clean_ancilla_qubits"):
        value = study.metric(qr.portable_metric(name)).evaluate(point)
        assert value.numeric_enclosure()[1] is not None
        assert not value.unknown_contributors


def test_exact_mod_pow_reports_zero_output_trace_distance() -> None:
    """Use the same public accuracy coordinate for exact implementations."""

    options = qm.RealizationOptions({qm.mod_pow: "mod_pow.windowed_lookup"})
    study = qr.project(
        qr.logical(first_class_mod_pow_sample),
        qr.GateCostModel(realizations=options),
    )
    assert study.metric(qr.CIRCUIT_OUTPUT_TRACE_DISTANCE_BOUND).numeric_enclosure() == (
        0,
        0,
    )


def test_coset_mod_pow_materializes_the_configured_channel() -> None:
    """Use the same configured realization in resource and circuit coordinates."""

    options = qm.RealizationOptions(
        {qm.mod_pow: qm.realize("mod_pow.coset_windowed", padding_bits=4)}
    )
    circuit = lower_to_circuit(
        first_class_mod_pow_sample.specialize(), realizations=options
    )
    assert circuit.realizations == ("mod_pow.coset_windowed[padding_bits=4]",)
    assert circuit.width == 36
    opcodes = tuple(instruction.opcode for instruction in circuit.body.instructions)
    assert "h" in opcodes
    assert "reset" in opcodes
    concrete_gates = sum(opcode not in {"measure", "reset"} for opcode in opcodes)
    study = qr.project(
        qr.logical(first_class_mod_pow_sample),
        qr.GateCostModel(realizations=options),
    )
    lower, upper = study.metric(qr.portable_metric("gates")).numeric_enclosure()
    assert lower is not None and upper is not None
    assert lower <= concrete_gates <= upper


@pytest.mark.parametrize(
    "choice", ["mod_pow.square_and_multiply", "mod_pow.windowed_lookup"]
)
def test_mod_pow_adjoint_and_persistence(choice: str) -> None:
    """Round-trip a coherent exponentiation through its exact adjoint."""

    pytest.importorskip("qiskit_aer")
    from qamomile.qiskit import Aer

    module = qm.build_source("""
import qamomile as qm
@qm.qkernel
def power(state: qm.Q[qm.Bits[5]]) -> qm.Q[qm.Bits[5]]:
    target, exponent = qm.split_at(state, 3)
    value, power = qm.mod_pow(
        qm.as_uint(target), qm.as_uint(exponent), base=2, modulus=5
    )
    return qm.concat(qm.as_bits(value), qm.as_bits(power))
@qm.qkernel
def sample() -> qm.Bits[5]:
    state = qm.prepare((1, 0, 0, 0, 1))
    state = power(state)
    state = qm.adjoint(power, state)
    return qm.measure(state)
""")
    program = module.specialize("sample")
    assert qm.Program.from_bytes(program.to_bytes()).to_bytes() == program.to_bytes()
    options = qm.RealizationOptions({qm.mod_pow: choice})
    result = qm.run(program, on=Aer(realizations=options), shots=4, seed=0).result
    assert result.counts == {(1, 0, 0, 0, 1): 4}


@pytest.mark.parametrize(("base", "modulus"), [(2, 0), (2, 6), (2, 9)])
def test_mod_pow_invalid_parameters_fail_closed(base: int, modulus: int) -> None:
    """Reject nonunits and moduli outside the target domain without panicking."""

    program = qm.build_source(f"""
import qamomile as qm
@qm.qkernel
def invalid() -> qm.Q[qm.Bits[5]]:
    target = qm.as_uint(qm.qubits(3))
    exponent = qm.as_uint(qm.qubits(2))
    target, exponent = qm.mod_pow(
        target, exponent, base={base}, modulus={modulus}
    )
    return qm.concat(qm.as_bits(target), qm.as_bits(exponent))
""").specialize("invalid")
    with pytest.raises(qm.CapabilityError, match="Q8050"):
        lower_to_circuit(program)
