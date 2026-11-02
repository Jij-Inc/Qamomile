//! Abi projections and entry points for the native Python module.

use crate::program_owner::program_owner;
use num_bigint::BigInt;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyBytes, PyComplex, PyDict, PyFloat, PyInt, PyList, PyTuple};
use qamomile_compiler::{
    CircuitLaunchBinding, CircuitLaunchValue,
    ScalarParameterPathComponent as NativeScalarParameterPathComponent,
    bind_invocation as bind_native_invocation,
    bind_scalar_parameters as bind_native_scalar_parameters,
    project_scalar_parameters as project_native_scalar_parameters,
};

pub(super) type FourPythonBytes<'py> = (
    Bound<'py, PyBytes>,
    Bound<'py, PyBytes>,
    Bound<'py, PyBytes>,
    Bound<'py, PyBytes>,
);

/// Derive canonical persistent artifact ABIs from one verified program.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn derive_artifact_abi<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "bytes | VerifiedProgramHandle"))] payload: &Bound<
        '_,
        PyAny,
    >,
) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
    let owner = program_owner(payload)?;
    let program = &owner.verified;
    let abi = program
        .artifact_abi()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok((
        PyBytes::new(py, abi.parameter_abi()),
        PyBytes::new(py, abi.result_abi()),
    ))
}

/// Validate every exact launch value against the native entrypoint ABI.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
pub(super) fn validate_launch_parameters(
    #[gen_stub(override_type(type_repr = "bytes | VerifiedProgramHandle"))] payload: &Bound<
        '_,
        PyAny,
    >,
    bindings: &Bound<'_, PyDict>,
) -> PyResult<()> {
    let owner = program_owner(payload)?;
    let program = &owner.verified;
    let launch_bindings = circuit_launch_bindings(bindings)?;
    program
        .validate_launch_parameters(&launch_bindings)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Bind portable launch values through the native artifact parameter ABI.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn bind_invocation<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "bytes"))] parameter_abi: &[u8],
    #[gen_stub(override_type(type_repr = "bytes"))] supplied_values: &[u8],
) -> PyResult<Bound<'py, PyBytes>> {
    bind_native_invocation(parameter_abi, supplied_values)
        .map(|payload| PyBytes::new(py, &payload))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

type ScalarParameterPathRow = (Option<u64>, Option<String>);
type ScalarParameterRow = (String, Vec<ScalarParameterPathRow>, String);
type ScalarParameterBindingRow<'py> = (
    String,
    Vec<ScalarParameterPathRow>,
    String,
    Bound<'py, PyBytes>,
);

fn scalar_parameter_path_rows(
    path: &[NativeScalarParameterPathComponent],
) -> Vec<ScalarParameterPathRow> {
    path.iter()
        .map(|component| match component {
            NativeScalarParameterPathComponent::Index(index) => (Some(*index), None),
            NativeScalarParameterPathComponent::Field(field) => (None, Some(field.clone())),
        })
        .collect()
}

/// Project a portable parameter ABI into ordered scalar SDK inputs.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn project_scalar_parameters(
    #[gen_stub(override_type(type_repr = "bytes"))] parameter_abi: &[u8],
) -> PyResult<Vec<ScalarParameterRow>> {
    project_native_scalar_parameters(parameter_abi)
        .map(|leaves| {
            leaves
                .into_iter()
                .map(|leaf| {
                    let path = scalar_parameter_path_rows(leaf.path());
                    (
                        leaf.name().to_owned(),
                        path,
                        leaf.scalar_type().as_str().to_owned(),
                    )
                })
                .collect()
        })
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Bind portable invocation values into ordered scalar SDK inputs.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn bind_scalar_parameters<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "bytes"))] parameter_abi: &[u8],
    #[gen_stub(override_type(type_repr = "bytes"))] supplied_values: &[u8],
) -> PyResult<Vec<ScalarParameterBindingRow<'py>>> {
    bind_native_scalar_parameters(parameter_abi, supplied_values)
        .map(|bindings| {
            bindings
                .into_iter()
                .map(|binding| {
                    let parameter = binding.parameter();
                    (
                        parameter.name().to_owned(),
                        scalar_parameter_path_rows(parameter.path()),
                        parameter.scalar_type().as_str().to_owned(),
                        PyBytes::new(py, binding.literal()),
                    )
                })
                .collect()
        })
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

pub(super) fn circuit_launch_bindings(
    bindings: &Bound<'_, PyDict>,
) -> PyResult<Vec<CircuitLaunchBinding>> {
    bindings
        .iter()
        .map(|(name, value)| {
            Ok(CircuitLaunchBinding::new(
                name.extract::<String>()?,
                circuit_launch_value(&value)?,
            ))
        })
        .collect()
}

fn circuit_launch_value(value: &Bound<'_, PyAny>) -> PyResult<CircuitLaunchValue> {
    if value.is_instance_of::<PyBool>() {
        return value.extract().map(CircuitLaunchValue::Bool);
    }
    if value.is_instance_of::<PyInt>() {
        if let Ok(value) = value.extract::<i64>() {
            return Ok(CircuitLaunchValue::Int(value));
        }
        let decimal = value.str()?;
        let parsed = decimal
            .to_str()?
            .parse::<BigInt>()
            .map_err(|_| PyValueError::new_err("Q4301: invalid integer launch value"))?;
        return Ok(CircuitLaunchValue::BigInt(
            parsed.to_signed_bytes_be().into_boxed_slice(),
        ));
    }
    if value.is_instance_of::<PyFloat>() {
        return value
            .extract::<f64>()
            .map(|value| CircuitLaunchValue::RealBits(value.to_bits()));
    }
    if value.is_instance_of::<PyComplex>() {
        let real = value.getattr("real")?.extract::<f64>()?;
        let imaginary = value.getattr("imag")?.extract::<f64>()?;
        return Ok(CircuitLaunchValue::ComplexBits(
            real.to_bits(),
            imaginary.to_bits(),
        ));
    }
    if let Ok(values) = value.cast::<PyTuple>() {
        return values
            .iter()
            .map(|value| circuit_launch_value(&value))
            .collect::<PyResult<Vec<_>>>()
            .map(|values| CircuitLaunchValue::Aggregate(values.into_boxed_slice()));
    }
    if let Ok(values) = value.cast::<PyList>() {
        return values
            .iter()
            .map(|value| circuit_launch_value(&value))
            .collect::<PyResult<Vec<_>>>()
            .map(|values| CircuitLaunchValue::Aggregate(values.into_boxed_slice()));
    }
    if let Ok(fields) = value.cast::<PyDict>() {
        return fields
            .iter()
            .map(|(name, value)| Ok((name.extract::<String>()?, circuit_launch_value(&value)?)))
            .collect::<PyResult<Vec<_>>>()
            .map(|fields| CircuitLaunchValue::Record(fields.into_boxed_slice()));
    }
    Err(PyValueError::new_err(
        "Q4301: unsupported launch materialization value",
    ))
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(validate_launch_parameters, module)?)?;
    module.add_function(wrap_pyfunction!(bind_invocation, module)?)?;
    module.add_function(wrap_pyfunction!(project_scalar_parameters, module)?)?;
    module.add_function(wrap_pyfunction!(bind_scalar_parameters, module)?)?;
    module.add_function(wrap_pyfunction!(derive_artifact_abi, module)?)?;
    Ok(())
}
