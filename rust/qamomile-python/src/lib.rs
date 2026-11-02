//! Python bindings for Qamomile; each module owns its projections and registration.

// Register integer limits once for both the runtime module and generated stubs.
macro_rules! add_integer_constant {
    ($module:ident, $constant:ident) => {{
        #[cfg(feature = "stub-gen")]
        pyo3_stub_gen::module_variable!("qamomile._qamomile_native", stringify!($constant), u64);
        $module.add(stringify!($constant), $constant)
    }};
}

mod abi_api;
mod algorithms_api;
mod circuit_api;
mod compiler_api;
mod conversion;
mod observables_api;
mod optimization_api;
mod persistence_api;
mod program_owner;
mod qsvt_api;
mod realization_api;
mod reference_api;
mod resources_api;
mod results_api;
mod shot_execution_api;
mod signal_api;
mod statistics_api;
mod target_profile_api;
mod types_api;
mod visualization_api;

use pyo3::prelude::*;
use qamomile_compiler::{
    MAX_ARITHMETIC_WIDTH, MAX_ARTIFACT_BYTES, MAX_CERTIFIED_QSVT_DEGREE,
    MAX_COMMUTING_MEASUREMENT_GATES, MAX_EXACT_GAS_BITS, MAX_EXECUTION_CONTEXT_ENTRIES,
    MAX_EXECUTION_SNAPSHOT_BYTES, MAX_EXECUTION_SNAPSHOT_DEPTH, MAX_EXECUTION_SNAPSHOT_NODES,
    MAX_EXECUTION_SNAPSHOT_REFERENCES, MAX_EXECUTION_STRING_BYTES,
    MAX_GENERAL_BOOLEAN_PERMUTATION_BITS, MAX_GENERAL_BOOLEAN_PERMUTATION_INSTRUCTIONS,
    MAX_GENERAL_BOOLEAN_PERMUTATION_WORK, MAX_GROVER_QUBITS, MAX_LCU_CASE_COUNT,
    MAX_LCU_INPUT_TERMS, MAX_MODULAR_MULTIPLICATION_WIDTH, MAX_ORDER_FINDING_PRECISION,
    MAX_QCORE_BYTES, MAX_QCORE_MESSAGE_NODES, MAX_QCORE_NESTING, MAX_QCORE_STRING_BYTES,
    MAX_QSVT_PHASES, MAX_QSVT_SYNTHESIS_ITERATIONS, MAX_QSVT_SYNTHESIS_WORK,
    MAX_RECIPROCAL_COEFFICIENT_WORK, MAX_RECIPROCAL_QSVT_DEGREE,
    MAX_SCALAR_PARAMETER_PATH_COMPONENTS, MAX_SCALAR_PARAMETERS, MAX_SIGNAL_BRANCHES,
    MAX_SIGNAL_DEPTH, MAX_SIGNAL_NODES, MAX_SIGNAL_REGION_WIDTH, MAX_STATISTICS_COUNT_BITS,
    MAX_STATISTICS_PATH, MAX_STATISTICS_TERMS, MAX_TARGET_PROFILE_BYTES, MAX_TARGET_PROFILE_EDGES,
    MAX_TARGET_PROFILE_QUBITS,
};

#[cfg(feature = "stub-gen")]
/// Gather the package-local native ABI without a separate Python project.
///
/// # Errors
///
/// Returns an error when the registered binding metadata cannot form a stub.
pub fn stub_info() -> pyo3_stub_gen::Result<pyo3_stub_gen::StubInfo> {
    pyo3_stub_gen::StubInfo::from_project_root(
        "qamomile._qamomile_native".to_owned(),
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."),
        true,
        pyo3_stub_gen::StubGenConfig::default(),
    )
}

/// Register the native compiler module.
#[pymodule]
#[pyo3(module = "qamomile")]
fn _qamomile_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    add_integer_constant!(module, MAX_EXACT_GAS_BITS)?;
    module.add_class::<program_owner::VerifiedProgramHandle>()?;
    module.add_class::<program_owner::VerifiedModuleHandle>()?;
    abi_api::register(module)?;
    optimization_api::register(module)?;
    types_api::register(module)?;
    visualization_api::register(module)?;
    compiler_api::register(module)?;
    resources_api::register(module)?;
    circuit_api::register(module)?;
    shot_execution_api::register(module)?;
    persistence_api::register(module)?;
    reference_api::register(module)?;
    observables_api::register(module)?;
    algorithms_api::register(module)?;
    qsvt_api::register(module)?;
    results_api::register(module)?;
    realization_api::register(module)?;
    signal_api::register(module)?;
    statistics_api::register(module)?;
    target_profile_api::register(module)?;
    add_integer_constant!(module, MAX_ARTIFACT_BYTES)?;
    add_integer_constant!(module, MAX_EXECUTION_SNAPSHOT_BYTES)?;
    add_integer_constant!(module, MAX_EXECUTION_SNAPSHOT_NODES)?;
    add_integer_constant!(module, MAX_EXECUTION_SNAPSHOT_DEPTH)?;
    add_integer_constant!(module, MAX_EXECUTION_SNAPSHOT_REFERENCES)?;
    add_integer_constant!(module, MAX_EXECUTION_CONTEXT_ENTRIES)?;
    add_integer_constant!(module, MAX_EXECUTION_STRING_BYTES)?;
    add_integer_constant!(module, MAX_QCORE_BYTES)?;
    add_integer_constant!(module, MAX_QCORE_MESSAGE_NODES)?;
    add_integer_constant!(module, MAX_QCORE_NESTING)?;
    add_integer_constant!(module, MAX_QCORE_STRING_BYTES)?;
    add_integer_constant!(module, MAX_SCALAR_PARAMETERS)?;
    add_integer_constant!(module, MAX_SCALAR_PARAMETER_PATH_COMPONENTS)?;
    add_integer_constant!(module, MAX_SIGNAL_BRANCHES)?;
    add_integer_constant!(module, MAX_SIGNAL_NODES)?;
    add_integer_constant!(module, MAX_SIGNAL_DEPTH)?;
    add_integer_constant!(module, MAX_SIGNAL_REGION_WIDTH)?;
    add_integer_constant!(module, MAX_QSVT_PHASES)?;
    add_integer_constant!(module, MAX_CERTIFIED_QSVT_DEGREE)?;
    add_integer_constant!(module, MAX_RECIPROCAL_QSVT_DEGREE)?;
    add_integer_constant!(module, MAX_RECIPROCAL_COEFFICIENT_WORK)?;
    add_integer_constant!(module, MAX_QSVT_SYNTHESIS_ITERATIONS)?;
    add_integer_constant!(module, MAX_QSVT_SYNTHESIS_WORK)?;
    add_integer_constant!(module, MAX_COMMUTING_MEASUREMENT_GATES)?;
    add_integer_constant!(module, MAX_GENERAL_BOOLEAN_PERMUTATION_BITS)?;
    add_integer_constant!(module, MAX_GENERAL_BOOLEAN_PERMUTATION_WORK)?;
    add_integer_constant!(module, MAX_GENERAL_BOOLEAN_PERMUTATION_INSTRUCTIONS)?;
    add_integer_constant!(module, MAX_STATISTICS_TERMS)?;
    add_integer_constant!(module, MAX_STATISTICS_PATH)?;
    add_integer_constant!(module, MAX_STATISTICS_COUNT_BITS)?;
    add_integer_constant!(module, MAX_ARITHMETIC_WIDTH)?;
    add_integer_constant!(module, MAX_GROVER_QUBITS)?;
    add_integer_constant!(module, MAX_LCU_INPUT_TERMS)?;
    add_integer_constant!(module, MAX_LCU_CASE_COUNT)?;
    add_integer_constant!(module, MAX_MODULAR_MULTIPLICATION_WIDTH)?;
    add_integer_constant!(module, MAX_ORDER_FINDING_PRECISION)?;
    add_integer_constant!(module, MAX_TARGET_PROFILE_BYTES)?;
    add_integer_constant!(module, MAX_TARGET_PROFILE_EDGES)?;
    add_integer_constant!(module, MAX_TARGET_PROFILE_QUBITS)?;
    Ok(())
}
