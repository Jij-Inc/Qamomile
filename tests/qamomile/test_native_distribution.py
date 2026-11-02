"""Installation-contract tests for the Qamomile native compiler."""

from __future__ import annotations

import importlib
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import pytest


def test_qamomile_distribution_bundles_native_compiler() -> None:
    """The public wheel owns its native compiler without another distribution."""

    requirements = metadata.requires("qamomile")

    assert requirements is not None
    assert not any(value.startswith("qamomile-native") for value in requirements)
    native = importlib.import_module("qamomile._qamomile_native")
    assert Path(native.__file__).with_name("_qamomile_native.pyi").is_file()
    assert native.VerifiedProgramHandle.__module__ == "qamomile._qamomile_native"


def test_native_distribution_exposes_required_compiler_boundaries() -> None:
    """A normal environment must contain every authoritative public boundary."""

    native = importlib.import_module("qamomile._qamomile_native")
    required = {
        "FqaoaPlan",
        "EkeraHastadPlan",
        "LcuCoefficientPlan",
        "MAX_ARITHMETIC_WIDTH",
        "MAX_ARTIFACT_BYTES",
        "MAX_CERTIFIED_QSVT_DEGREE",
        "MAX_COMMUTING_MEASUREMENT_GATES",
        "MAX_GROVER_QUBITS",
        "MAX_GENERAL_BOOLEAN_PERMUTATION_BITS",
        "MAX_GENERAL_BOOLEAN_PERMUTATION_INSTRUCTIONS",
        "MAX_GENERAL_BOOLEAN_PERMUTATION_WORK",
        "MAX_LCU_CASE_COUNT",
        "MAX_LCU_INPUT_TERMS",
        "MAX_MODULAR_MULTIPLICATION_WIDTH",
        "MAX_ORDER_FINDING_PRECISION",
        "MAX_QCORE_BYTES",
        "MAX_QCORE_MESSAGE_NODES",
        "MAX_QCORE_NESTING",
        "MAX_QCORE_STRING_BYTES",
        "MAX_QSVT_PHASES",
        "MAX_RECIPROCAL_COEFFICIENT_WORK",
        "MAX_RECIPROCAL_QSVT_DEGREE",
        "MAX_QSVT_SYNTHESIS_ITERATIONS",
        "MAX_QSVT_SYNTHESIS_WORK",
        "PauliCorrelationPlan",
        "QracPlan",
        "QsvtPolynomialCertificate",
        "ReciprocalQsvtCertificate",
        "SparsePolynomial",
        "add_scaled_hermitian_matrices",
        "add_scaled_observable",
        "affine_observable",
        "ModularMultiplicationPlan",
        "bind_invocation",
        "bind_scalar_parameters",
        "convert_qsp_wx_phases",
        "decode_result_value",
        "decompose_hermitian_matrix",
        "decompose_pauli_matrix",
        "decompose_periodic_shift_matrix",
        "dense_matrix_entry_budget",
        "derive_artifact_abi",
        "encode_result_value",
        "statistics_mean",
        "statistics_combine_independent",
        "evaluate_classical",
        "execute_reference_at",
        "validate_result_value",
        "grover_iteration_count",
        "inspect_circuit_family_selector",
        "inspect_artifact",
        "inspect_module",
        "inspect_program",
        "inspect_qcore_envelope",
        "inspect_resource_expression_symbols",
        "inspect_specialization",
        "inspect_target_adjoint",
        "inspect_static_circuit",
        "inspect_static_circuit_at",
        "inspect_static_circuit_family",
        "inspect_standard_optimization_operation_visits",
        "inspect_symbolic_logical_counts",
        "inspect_symbolic_program_family_counts",
        "inspect_symbolic_resource_analysis_request",
        "lower_hir",
        "lower_adjoint_calls_for_target",
        "lower_optimized_adjoint_calls_for_target",
        "minimum_pce_qubits",
        "module_semantic_digest",
        "modular_addition_residue",
        "multiply_observables",
        "mottonen_amplitude_coordinates",
        "normalize_aggregate_index",
        "normalize_aggregate_slice",
        "normalize_observable",
        "normalize_periodic_shift_terms",
        "observable_is_pairwise_commuting",
        "optimize_standard_program",
        "pauli_words_commute",
        "plan_target",
        "plan_target_lowering",
        "plan_boolean_lifts",
        "plan_pauli_realization",
        "plan_lcu_coefficients",
        "plan_ekera_hastad",
        "plan_modular_multiplication",
        "plan_qwc_pauli_expectation",
        "plan_commuting_pauli_expectation",
        "postprocess_circuit_counts",
        "prepare_pauli_measurement",
        "prepare_qwc_pauli_measurement",
        "prepare_commuting_pauli_measurement",
        "prepare_program",
        "project_pauli_subspace",
        "remap_observable",
        "project_scalar_parameters",
        "program_semantic_digest",
        "rank_resource_hotspots",
        "resolve_module_entrypoint",
        "resolve_resource_symbol_bindings",
        "retag_resource_expression_unit",
        "scale_hermitian_matrix",
        "select_circuit_family_variant",
        "seal_artifact",
        "seal_hir_package",
        "seal_qcore_package",
        "seal_resource_analysis_request",
        "selected_module_semantic_digest",
        "suzuki_product_formula_schedule",
        "synthesize_reciprocal_qsvt",
        "synthesize_qsvt_polynomial",
        "uniformly_controlled_rotation_coordinates",
        "validate_launch_parameters",
        "validate_hermitian_matrix",
        "verify_reciprocal_qsvt",
        "verify_qsvt_polynomial",
        "wrapping_addition_offsets",
    }

    assert required.issubset(dir(native))
    assert "specialize_program" not in dir(native)


def test_native_aggregate_index_and_slice_normalization_preserve_semantics() -> None:
    """The installed compiler owns exact fixed aggregate subscript projection."""

    native = importlib.import_module("qamomile._qamomile_native")

    assert native.normalize_aggregate_index(3, "0") == 0
    assert native.normalize_aggregate_index(3, "-1") == 2
    assert native.normalize_aggregate_index(3, "-3") == 0
    assert native.normalize_aggregate_index(3, "3") is None
    assert native.normalize_aggregate_index(3, "-4") is None
    assert native.normalize_aggregate_index(3, str(1 << 1024)) is None
    assert native.normalize_aggregate_index(3, str(-(1 << 1024))) is None
    assert native.normalize_aggregate_slice(5, None, None, None) == [0, 1, 2, 3, 4]
    assert native.normalize_aggregate_slice(5, None, None, "-2") == [4, 2, 0]
    with pytest.raises(ValueError, match="slice step cannot be zero"):
        native.normalize_aggregate_slice(5, None, None, "0")
    with pytest.raises(ValueError, match="exceeds 1024 selected items"):
        native.normalize_aggregate_slice(1025, None, None, None)


def test_classical_evaluation_crosses_the_required_native_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concrete classical values cannot select a shadow Python evaluator."""

    operations = importlib.import_module("qamomile.operations")
    native_evaluate = operations.native.evaluate_classical
    calls = 0

    def record_native_evaluation(
        opcode: str,
        operands: tuple[bytes, ...],
        attributes: tuple[tuple[str, bytes], ...],
    ) -> object:
        """Record one compiler-owned classical operation evaluation."""

        nonlocal calls
        calls += 1
        return native_evaluate(opcode, operands, attributes)

    monkeypatch.setattr(
        operations.native,
        "evaluate_classical",
        record_native_evaluation,
    )

    assert operations.evaluate_classical("classical.floordiv", (-7, 3)) == -3
    assert calls == 1

    monkeypatch.setattr(operations.native, "evaluate_classical", None)
    with pytest.raises(TypeError, match="Q4500"):
        operations.evaluate_classical("classical.add", (1, 2))


def test_qamomile_import_fails_clearly_without_the_native_compiler() -> None:
    """A broken installation reports one stable dependency diagnostic."""

    script = """
import builtins

ordinary_import = builtins.__import__

def block_native(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "qamomile._qamomile_native" or "_qamomile_native" in (fromlist or ()):
        raise ImportError("simulated missing bundled compiler")
    return ordinary_import(name, globals, locals, fromlist, level)

builtins.__import__ = block_native
try:
    import qamomile
except ImportError as error:
    print(error)
else:
    raise AssertionError("qamomile imported without its required compiler")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == (
        "Q0001: Qamomile requires its bundled native compiler; reinstall Qamomile"
    )


def test_bundled_compiler_ignores_a_legacy_top_level_module() -> None:
    """An older separately installed extension cannot replace the bundled compiler."""
    script = """
import sys
from types import ModuleType
legacy = ModuleType("_qamomile_native")
sys.modules["_qamomile_native"] = legacy
from qamomile._native import compiler
assert compiler is not legacy
assert compiler.__name__ == "qamomile._qamomile_native"
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
