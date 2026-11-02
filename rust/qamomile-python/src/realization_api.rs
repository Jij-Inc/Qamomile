//! Realization projections and entry points for the native Python module.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use qamomile_compiler::{
    RealizationCapabilities, VerifiedProgram, plan_pauli_evolution_with_capabilities,
};

pub(super) fn lowering_config(
    labels: Option<&[String]>,
    overrides: Option<&[String]>,
) -> PyResult<qamomile_compiler::LoweringConfig> {
    qamomile_compiler::LoweringConfig::from_labels(
        labels.unwrap_or_default(),
        overrides.unwrap_or_default(),
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// List compiler-owned implementation definitions for discovery and docs.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn list_realizations() -> Vec<(String, String, String, Vec<String>)> {
    qamomile_compiler::Realization::ALL
        .into_iter()
        .map(|choice| {
            let info = choice.info();
            (
                info.name.into(),
                info.family.as_str().into(),
                info.description.into(),
                info.required_capabilities
                    .labels()
                    .map(str::to_owned)
                    .collect(),
            )
        })
        .collect()
}

/// Configure and canonicalize one parameterized realization selection.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[expect(
    clippy::needless_pass_by_value,
    reason = "PyO3 extracts an owned Python sequence"
)]
fn configure_realization(name: &str, parameters: Vec<(String, String)>) -> PyResult<String> {
    let kind = qamomile_compiler::Realization::ALL
        .into_iter()
        .find(|candidate| candidate.as_str() == name)
        .ok_or_else(|| PyValueError::new_err(format!("Q8043: unknown realization {name:?}")))?;
    qamomile_compiler::ConfiguredRealization::configure(kind, &parameters)
        .map(qamomile_compiler::ConfiguredRealization::canonical_label)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Normalize closed named choices independently of a particular target.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[pyo3(signature = (overrides, realization_capabilities=None))]
#[allow(clippy::needless_pass_by_value)]
fn validate_lowering_options(
    overrides: Vec<String>,
    realization_capabilities: Option<Vec<String>>,
) -> PyResult<Vec<String>> {
    let labels = qamomile_compiler::Realization::ALL
        .into_iter()
        .flat_map(|choice| choice.info().required_capabilities.labels())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let config = lowering_config(
        Some(realization_capabilities.as_deref().unwrap_or(&labels)),
        Some(&overrides),
    )?;
    Ok(config.overrides())
}

/// Link user-authored operation or callable realizations into a transient image.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[gen_stub(override_return_type(type_repr = "bytes"))]
#[allow(clippy::needless_pass_by_value)]
fn apply_operation_realizations(
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
    boundary: &str,
    target_kinds: Vec<String>,
    #[gen_stub(override_type(type_repr = "list[bytes]"))] target_modules: Vec<Vec<u8>>,
    target_symbols: Vec<String>,
    #[gen_stub(override_type(type_repr = "list[bytes]"))] implementation_modules: Vec<Vec<u8>>,
    implementation_symbols: Vec<String>,
) -> PyResult<Vec<u8>> {
    let row_count = target_kinds.len();
    if target_modules.len() != row_count
        || target_symbols.len() != row_count
        || implementation_modules.len() != row_count
        || implementation_symbols.len() != row_count
    {
        return Err(PyValueError::new_err(
            "Q8062: operation realization rows have inconsistent lengths",
        ));
    }
    let definitions = target_kinds
        .iter()
        .zip(&target_modules)
        .zip(&target_symbols)
        .zip(&implementation_modules)
        .zip(&implementation_symbols)
        .map(
            |(
                (((kind, target_module), target_symbol), implementation_module),
                implementation_symbol,
            )| {
                let definition = match kind.as_str() {
                    "operation" => {
                        qamomile_compiler::OperationRealizationDefinition::operation_from_bytes(
                            target_symbol,
                            implementation_module,
                            implementation_symbol,
                        )
                    }
                    "callable" => {
                        qamomile_compiler::OperationRealizationDefinition::callable_from_bytes(
                            target_module,
                            target_symbol,
                            implementation_module,
                            implementation_symbol,
                        )
                    }
                    _ => {
                        return Err(PyValueError::new_err(format!(
                            "Q8060: unknown realization target kind {kind:?}"
                        )));
                    }
                };
                definition.map_err(|error| PyValueError::new_err(error.to_string()))
            },
        )
        .collect::<PyResult<Vec<_>>>()?;
    match boundary {
        "module" => {
            let module = qamomile_compiler::VerifiedModule::from_bytes(payload)
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
            qamomile_compiler::realize_module(&module, definitions)
                .map(|realized| realized.canonical_bytes().to_vec())
                .map_err(|error| PyValueError::new_err(error.to_string()))
        }
        "program" => {
            let program = VerifiedProgram::from_bytes(payload)
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
            qamomile_compiler::realize_program(&program, definitions)
                .map(|realized| realized.canonical_bytes().to_vec())
                .map_err(|error| PyValueError::new_err(error.to_string()))
        }
        _ => Err(PyValueError::new_err(format!(
            "Q8060: unknown operation realization boundary {boundary:?}"
        ))),
    }
}

/// Produce a checked exact arithmetic schedule at the final target boundary.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[pyo3(signature = (state_width, realization_capabilities=None, lowering=None))]
#[allow(clippy::type_complexity, clippy::needless_pass_by_value)]
fn plan_addition(
    state_width: u64,
    realization_capabilities: Option<Vec<String>>,
    lowering: Option<Vec<String>>,
) -> PyResult<(String, Vec<(String, Vec<u64>, Vec<u64>, Option<f64>, bool)>)> {
    let config = lowering_config(realization_capabilities.as_deref(), lowering.as_deref())?;
    let plan = qamomile_compiler::plan_addition(state_width, config)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let realization = plan.realization().as_str().to_owned();
    Ok((
        realization,
        plan.into_steps()
            .into_iter()
            .map(|step| {
                (
                    step.gate().as_str().to_owned(),
                    step.indices().to_vec(),
                    step.controls().to_vec(),
                    step.angle(),
                    step.inherits_controls(),
                )
            })
            .collect(),
    ))
}

/// Select or materialize exact modular exponentiation at a target boundary.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[pyo3(signature = (
    target_width,
    exponent_width,
    base,
    modulus,
    inverse,
    control_width,
    realization_capabilities=None,
    lowering=None
))]
#[allow(clippy::type_complexity, clippy::needless_pass_by_value)]
#[expect(
    clippy::too_many_arguments,
    reason = "The Python bridge projects the compiler plan arguments and target capabilities"
)]
fn plan_mod_pow(
    target_width: u64,
    exponent_width: u64,
    base: i64,
    modulus: i64,
    inverse: bool,
    control_width: u64,
    realization_capabilities: Option<Vec<String>>,
    lowering: Option<Vec<String>>,
) -> PyResult<(
    String,
    String,
    u64,
    Vec<(String, Vec<u64>, Vec<u64>)>,
    Vec<String>,
)> {
    let config = lowering_config(realization_capabilities.as_deref(), lowering.as_deref())?;
    let identity = config
        .select(qamomile_compiler::RealizationFamily::ModPow)
        .canonical_label();
    let plan = qamomile_compiler::plan_mod_pow(
        target_width,
        exponent_width,
        base,
        modulus,
        inverse,
        control_width,
        config,
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    match plan {
        qamomile_compiler::ModularRealizationPlan::PreserveNative(_) => {
            Ok(("native".to_owned(), identity, 0, Vec::new(), Vec::new()))
        }
        qamomile_compiler::ModularRealizationPlan::Gates(plan) => {
            Ok(project_modular_plan(plan, identity))
        }
    }
}

#[allow(clippy::type_complexity)]
fn project_modular_plan(
    plan: qamomile_compiler::ModularGatePlan,
    identity: String,
) -> (
    String,
    String,
    u64,
    Vec<(String, Vec<u64>, Vec<u64>)>,
    Vec<String>,
) {
    let children = plan
        .child_realizations()
        .iter()
        .map(|child| child.as_str().to_owned())
        .collect();
    let scratch_width = plan.scratch_width();
    let steps = plan
        .into_steps()
        .into_iter()
        .map(|step| {
            (
                step.gate().as_str().to_owned(),
                step.wires().to_vec(),
                step.controls().to_vec(),
            )
        })
        .collect();
    ("gates".to_owned(), identity, scratch_width, steps, children)
}

/// Select a shared schoolbook schedule for the direct HUGR multiplication consumer.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[pyo3(signature = (width, factor, modulus, inverse, control_width, lowering=None))]
#[allow(clippy::type_complexity, clippy::needless_pass_by_value)]
fn plan_mod_mul(
    width: u64,
    factor: i64,
    modulus: i64,
    inverse: bool,
    control_width: u64,
    lowering: Option<Vec<String>>,
) -> PyResult<
    Option<(
        String,
        String,
        u64,
        Vec<(String, Vec<u64>, Vec<u64>)>,
        Vec<String>,
    )>,
> {
    let config = lowering_config(None, lowering.as_deref())?;
    let choice = config.select(qamomile_compiler::RealizationFamily::ModMul);
    if choice.kind() != qamomile_compiler::Realization::ModMulSchoolbookRipple {
        return Ok(None);
    }
    let plan = qamomile_compiler::plan_schoolbook_mod_mul(
        width,
        factor,
        modulus,
        inverse,
        control_width,
        RealizationCapabilities::EMPTY,
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(Some(project_modular_plan(plan, choice.canonical_label())))
}

/// Produce a bounded exact arithmetic permutation schedule for HUGR emission.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[allow(clippy::type_complexity, clippy::needless_pass_by_value)]
fn plan_arithmetic_permutation(
    opcode: String,
    state_width: u64,
    constants: Vec<i64>,
    adjoint: bool,
) -> PyResult<(String, Vec<(String, Vec<u64>, Vec<u64>, Option<f64>, bool)>)> {
    let operation = qamomile_compiler::ArithmeticOperation::try_from(opcode.as_str())
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let plan =
        qamomile_compiler::plan_arithmetic_permutation(operation, state_width, &constants, adjoint)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let realization = plan.realization().as_str().to_owned();
    Ok((
        realization,
        plan.into_steps()
            .into_iter()
            .map(|step| {
                (
                    step.gate().as_str().to_owned(),
                    step.indices().to_vec(),
                    step.controls().to_vec(),
                    step.angle(),
                    step.inherits_controls(),
                )
            })
            .collect(),
    ))
}

/// Select and generate one exact Pauli schedule for a non-native consumer.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[pyo3(signature = (payload, state_width, lowering=None))]
#[allow(clippy::type_complexity, clippy::needless_pass_by_value)]
fn plan_pauli_realization(
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
    state_width: u64,
    lowering: Option<Vec<String>>,
) -> PyResult<(String, Vec<(String, Vec<u64>, f64)>)> {
    let config = lowering_config(None, lowering.as_deref())?;
    let choice = config.select(qamomile_compiler::RealizationFamily::PauliEvolution);
    let steps = plan_pauli_evolution_with_capabilities(
        payload,
        state_width,
        RealizationCapabilities::EMPTY,
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok((
        choice.canonical_label(),
        steps
            .into_iter()
            .map(|step| {
                (
                    step.gate().as_str().to_owned(),
                    step.indices().to_vec(),
                    step.scale(),
                )
            })
            .collect(),
    ))
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(plan_pauli_realization, module)?)?;
    module.add_function(wrap_pyfunction!(plan_addition, module)?)?;
    module.add_function(wrap_pyfunction!(plan_mod_pow, module)?)?;
    module.add_function(wrap_pyfunction!(plan_mod_mul, module)?)?;
    module.add_function(wrap_pyfunction!(plan_arithmetic_permutation, module)?)?;
    module.add_function(wrap_pyfunction!(validate_lowering_options, module)?)?;
    module.add_function(wrap_pyfunction!(configure_realization, module)?)?;
    module.add_function(wrap_pyfunction!(apply_operation_realizations, module)?)?;
    module.add_function(wrap_pyfunction!(list_realizations, module)?)?;
    Ok(())
}
