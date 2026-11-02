//! Circuit projections and entry points for the native Python module.

use super::abi_api::circuit_launch_bindings;
use super::realization_api::lowering_config;
use crate::program_owner::program_owner;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict};
use qamomile_compiler::{
    CircuitDispatchDecision as NativeCircuitDispatchDecision,
    CircuitDispatchParameter as NativeCircuitDispatchParameter,
    CircuitDispatchPredicate as NativeCircuitDispatchPredicate,
    CircuitParameter as NativeCircuitParameter, CircuitParameterPath,
    CircuitScalarAttribute as NativeCircuitScalarAttribute, CircuitScalarAttributeValue,
    CircuitScalarId, CircuitScalarNode, CircuitScalarType, StaticCircuit as NativeStaticCircuit,
    StaticCircuitResultValue as NativeStaticCircuitResultValue, VerifiedProgram,
    encode_circuit_family_selection as encode_native_circuit_family_selection,
    project_result_postprocessing as project_native_result_postprocessing,
};

/// One target-neutral instruction from the native static circuit projection.
#[derive(Clone)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native", skip_from_py_object)]
struct CircuitInstruction {
    #[pyo3(get)]
    opcode: String,
    #[pyo3(get)]
    qubits: Vec<u64>,
    #[pyo3(get)]
    parameters: Vec<CircuitParameter>,
    #[pyo3(get)]
    controls: Vec<u64>,
    #[pyo3(get)]
    condition: Option<u64>,
    #[pyo3(get)]
    regions: Vec<CircuitBlock>,
}

/// One nested target-neutral instruction block.
#[derive(Clone)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native", skip_from_py_object)]
struct CircuitBlock {
    #[pyo3(get)]
    instructions: Vec<CircuitInstruction>,
    #[pyo3(get)]
    has_return: bool,
    #[pyo3(get)]
    return_value: Option<CircuitResultValue>,
}

/// Live owner of a checked local definition; no Python semantic callback.
#[derive(Clone)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native", skip_from_py_object)]
struct CircuitDefinition {
    owner: qamomile_compiler::CircuitDefinition,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl CircuitDefinition {
    #[getter]
    fn identity(&self) -> usize {
        self.owner.identity()
    }
    #[getter]
    fn width(&self) -> u64 {
        self.owner.width()
    }
    #[getter]
    fn scalars(&self) -> Vec<CircuitScalar> {
        self.owner.scalars().iter().map(circuit_scalar).collect()
    }
    fn materialize(&self) -> PyResult<Vec<CircuitInstruction>> {
        Ok(self
            .owner
            .instructions()
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            .iter()
            .map(circuit_instruction)
            .collect())
    }
}

/// One tagged reference from an instruction into circuit-owned data.
#[derive(Clone)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native", skip_from_py_object)]
struct CircuitParameter {
    #[pyo3(get)]
    kind: String,
    #[pyo3(get)]
    index: u64,
    #[pyo3(get)]
    scalar_type: Option<String>,
    payload: Option<Vec<u8>>,
    #[pyo3(get)]
    boolean_outputs: Option<Vec<Vec<Vec<u64>>>>,
    #[pyo3(get)]
    definition: Option<CircuitDefinition>,
}

/// One typed structural component in a launch parameter path.
#[derive(Clone)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native", skip_from_py_object)]
struct CircuitPathComponent {
    #[pyo3(get)]
    kind: String,
    #[pyo3(get)]
    index: Option<u64>,
    #[pyo3(get)]
    field: Option<String>,
}

/// One typed attribute on a native scalar DAG operation.
#[derive(Clone)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native", skip_from_py_object)]
struct CircuitScalarAttribute {
    #[pyo3(get)]
    name: String,
    #[pyo3(get)]
    kind: String,
    #[pyo3(get)]
    int_value: Option<i64>,
    #[pyo3(get)]
    string_value: Option<String>,
    #[pyo3(get)]
    string_values: Vec<String>,
}

/// One node from the native circuit's hash-consed scalar DAG.
#[derive(Clone)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native", skip_from_py_object)]
struct CircuitScalar {
    #[pyo3(get)]
    kind: String,
    #[pyo3(get)]
    scalar_type: String,
    #[pyo3(get)]
    bool_value: Option<bool>,
    #[pyo3(get)]
    int_value: Option<i64>,
    #[pyo3(get)]
    real_value: Option<f64>,
    #[pyo3(get)]
    measurement_index: Option<u64>,
    #[pyo3(get)]
    call_index: Option<u64>,
    #[pyo3(get)]
    name: Option<String>,
    #[pyo3(get)]
    path: Vec<CircuitPathComponent>,
    #[pyo3(get)]
    opcode: Option<String>,
    #[pyo3(get)]
    attributes: Vec<CircuitScalarAttribute>,
    #[pyo3(get)]
    operands: Vec<u32>,
}

/// One recursive result expression retained by a native static circuit.
#[derive(Clone)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native", skip_from_py_object)]
struct CircuitResultValue {
    #[pyo3(get)]
    kind: String,
    #[pyo3(get)]
    scalar: Option<u32>,
    literal: Option<Vec<u8>>,
    #[pyo3(get)]
    elements: Vec<CircuitResultValue>,
    #[pyo3(get)]
    field_names: Vec<String>,
}

/// A closed target-neutral circuit from the native static projection.
#[derive(Clone)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native", skip_from_py_object)]
pub(crate) struct CircuitProjection {
    #[pyo3(get)]
    optimization_stages: Vec<String>,
    #[pyo3(get)]
    realizations: Vec<String>,
    #[pyo3(get)]
    width: u64,
    #[pyo3(get)]
    measurement_count: u64,
    #[pyo3(get)]
    operation_count: usize,
    #[pyo3(get)]
    instructions: Vec<CircuitInstruction>,
    #[pyo3(get)]
    scalars: Vec<CircuitScalar>,
    #[pyo3(get)]
    parameter_names: Vec<String>,
    #[pyo3(get)]
    result_measurements: Vec<u64>,
    #[pyo3(get)]
    output_wires: Option<Vec<u64>>,
    #[pyo3(get)]
    result_value: Option<CircuitResultValue>,
    #[pyo3(get)]
    result_route: String,
    result_postprocessing: Option<Vec<u8>>,
}

/// One launch parameter defining a finite native dispatch family.
#[derive(Clone)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native", skip_from_py_object)]
pub(super) struct CircuitDispatchParameter {
    #[pyo3(get)]
    parameter_index: usize,
    #[pyo3(get)]
    name: String,
}

/// One verified launch-time Boolean expression.
#[derive(Clone)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native", skip_from_py_object)]
struct CircuitDispatchPredicate {
    #[pyo3(get)]
    root: u32,
    #[pyo3(get)]
    scalars: Vec<CircuitScalar>,
    #[pyo3(get)]
    parameter_names: Vec<String>,
}

/// One truth assignment along a static variant's decision-tree path.
#[derive(Clone)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native", skip_from_py_object)]
pub(super) struct CircuitDispatchDecision {
    #[pyo3(get)]
    predicate_index: usize,
    #[pyo3(get)]
    value: bool,
}

/// One static circuit selected by an exact launch dispatch key.
#[derive(Clone)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native", skip_from_py_object)]
struct CircuitVariantProjection {
    #[pyo3(get)]
    decisions: Vec<CircuitDispatchDecision>,
    #[pyo3(get)]
    circuit: CircuitProjection,
}

/// Finite target-neutral native lowering for launch-time control.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native")]
struct CircuitFamilyProjection {
    #[pyo3(get)]
    dispatch_parameters: Vec<CircuitDispatchParameter>,
    #[pyo3(get)]
    predicates: Vec<CircuitDispatchPredicate>,
    #[pyo3(get)]
    variants: Vec<CircuitVariantProjection>,
    selection_plan: Vec<u8>,
}

// Explicit byte getters keep PyO3 runtime conversion and generated typing aligned.
// The generator currently treats Vec<u8> fields as lists of integers.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl CircuitParameter {
    #[getter]
    fn payload<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyBytes>> {
        self.payload.as_ref().map(|value| PyBytes::new(py, value))
    }
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl CircuitResultValue {
    #[getter]
    fn literal<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyBytes>> {
        self.literal.as_ref().map(|value| PyBytes::new(py, value))
    }
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl CircuitProjection {
    #[getter]
    fn result_postprocessing<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyBytes>> {
        self.result_postprocessing
            .as_ref()
            .map(|value| PyBytes::new(py, value))
    }
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl CircuitFamilyProjection {
    #[getter]
    fn selection_plan<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.selection_plan)
    }
}

/// Project an independently verified static program into logical instructions.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[pyo3(signature = (payload, realization_capabilities=None, lowering=None))]
#[allow(clippy::needless_pass_by_value)] // PyO3 extracts the optional labels by value.
fn inspect_static_circuit(
    #[gen_stub(override_type(type_repr = "bytes | VerifiedProgramHandle"))] payload: &Bound<
        '_,
        PyAny,
    >,
    realization_capabilities: Option<Vec<String>>,
    lowering: Option<Vec<String>>,
) -> PyResult<CircuitProjection> {
    let owner = program_owner(payload)?;
    let program = &owner.verified;
    let capabilities = lowering_config(realization_capabilities.as_deref(), lowering.as_deref())?;
    let sampled_width = program
        .artifact_abi()
        .map_err(|error| PyValueError::new_err(error.to_string()))?
        .sampled_width();
    let circuit = owner
        .prepared()?
        .static_circuit_with_lowering(capabilities)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    circuit_projection(&circuit, sampled_width)
}

/// Return dynamic `QCore` operation visits made by static-circuit lowering.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn inspect_static_circuit_operation_visits(
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
) -> PyResult<u128> {
    let program = VerifiedProgram::from_bytes(payload)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    program
        .static_circuit_with_work()
        .map(|(_, work)| work.operation_visits())
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Materialize one launch-controlled invocation into a static circuit.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[pyo3(signature = (payload, bindings, realization_capabilities=None, lowering=None))]
#[allow(clippy::needless_pass_by_value)] // PyO3 extracts the optional labels by value.
fn inspect_static_circuit_at(
    #[gen_stub(override_type(type_repr = "bytes | VerifiedProgramHandle"))] payload: &Bound<
        '_,
        PyAny,
    >,
    bindings: &Bound<'_, PyDict>,
    realization_capabilities: Option<Vec<String>>,
    lowering: Option<Vec<String>>,
) -> PyResult<CircuitProjection> {
    let owner = program_owner(payload)?;
    let program = &owner.verified;
    let capabilities = lowering_config(realization_capabilities.as_deref(), lowering.as_deref())?;
    let sampled_width = program
        .artifact_abi()
        .map_err(|error| PyValueError::new_err(error.to_string()))?
        .sampled_width();
    let launch_bindings = circuit_launch_bindings(bindings)?;
    let circuit = owner
        .prepared()?
        .static_circuit_at_with_lowering(&launch_bindings, capabilities)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    circuit_projection(&circuit, sampled_width)
}

/// Lower finite launch-time scalar predicates into static circuit variants.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[pyo3(signature = (payload, realization_capabilities=None, lowering=None))]
#[allow(clippy::needless_pass_by_value)] // PyO3 extracts the optional labels by value.
fn inspect_static_circuit_family(
    #[gen_stub(override_type(type_repr = "bytes | VerifiedProgramHandle"))] payload: &Bound<
        '_,
        PyAny,
    >,
    realization_capabilities: Option<Vec<String>>,
    lowering: Option<Vec<String>>,
) -> PyResult<CircuitFamilyProjection> {
    let owner = program_owner(payload)?;
    let program = &owner.verified;
    let capabilities = lowering_config(realization_capabilities.as_deref(), lowering.as_deref())?;
    let sampled_width = program
        .artifact_abi()
        .map_err(|error| PyValueError::new_err(error.to_string()))?
        .sampled_width();
    let family = owner
        .prepared()?
        .static_circuit_family_with_lowering(capabilities)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let selection_plan = encode_native_circuit_family_selection(&family)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(CircuitFamilyProjection {
        dispatch_parameters: family
            .dispatch_parameters()
            .iter()
            .map(circuit_dispatch_parameter)
            .collect(),
        predicates: family
            .predicates()
            .iter()
            .map(circuit_dispatch_predicate)
            .collect(),
        variants: family
            .variants()
            .iter()
            .map(|variant| {
                Ok(CircuitVariantProjection {
                    decisions: variant
                        .decisions()
                        .iter()
                        .map(circuit_dispatch_decision)
                        .collect(),
                    circuit: circuit_projection(variant.circuit(), sampled_width)?,
                })
            })
            .collect::<PyResult<Vec<_>>>()?,
        selection_plan,
    })
}

pub(crate) fn circuit_projection(
    circuit: &NativeStaticCircuit,
    sampled_width: u32,
) -> PyResult<CircuitProjection> {
    let result = project_native_result_postprocessing(circuit, sampled_width)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let result_route = result.route().as_str().to_owned();
    let result_postprocessing = result.into_postprocessing();
    Ok(CircuitProjection {
        optimization_stages: circuit.optimization_stages().to_vec(),
        realizations: circuit
            .realizations()
            .iter()
            .map(|choice| choice.canonical_label())
            .collect(),
        width: circuit.width(),
        measurement_count: circuit.measurement_count(),
        operation_count: circuit.operation_count(),
        instructions: circuit
            .instructions()
            .iter()
            .map(circuit_instruction)
            .collect(),
        scalars: circuit.scalars().iter().map(circuit_scalar).collect(),
        parameter_names: circuit
            .retained_parameter_names()
            .into_iter()
            .map(str::to_owned)
            .collect(),
        result_measurements: if result_route == "postprocessing" {
            Vec::new()
        } else {
            circuit.result_measurements().to_vec()
        },
        output_wires: circuit.output_wires().map(<[u64]>::to_vec),
        result_value: circuit.result_value().map(circuit_result_value),
        result_route,
        result_postprocessing,
    })
}

fn circuit_result_value(value: NativeStaticCircuitResultValue) -> CircuitResultValue {
    match value {
        NativeStaticCircuitResultValue::Unit => CircuitResultValue {
            kind: "unit".to_owned(),
            scalar: None,
            literal: None,
            elements: Vec::new(),
            field_names: Vec::new(),
        },
        NativeStaticCircuitResultValue::Literal(literal) => CircuitResultValue {
            kind: "literal".to_owned(),
            scalar: None,
            literal: Some(literal.into_vec()),
            elements: Vec::new(),
            field_names: Vec::new(),
        },
        NativeStaticCircuitResultValue::Scalar(identifier) => CircuitResultValue {
            kind: "scalar".to_owned(),
            scalar: Some(identifier.index()),
            literal: None,
            elements: Vec::new(),
            field_names: Vec::new(),
        },
        NativeStaticCircuitResultValue::Aggregate(elements) => CircuitResultValue {
            kind: "aggregate".to_owned(),
            scalar: None,
            literal: None,
            elements: elements
                .into_vec()
                .into_iter()
                .map(circuit_result_value)
                .collect(),
            field_names: Vec::new(),
        },
        NativeStaticCircuitResultValue::Record(fields) => {
            let (field_names, elements) = fields
                .into_vec()
                .into_iter()
                .map(|(name, value)| (name, circuit_result_value(value)))
                .unzip();
            CircuitResultValue {
                kind: "record".to_owned(),
                scalar: None,
                literal: None,
                elements,
                field_names,
            }
        }
    }
}

pub(super) fn circuit_dispatch_parameter(
    parameter: &NativeCircuitDispatchParameter,
) -> CircuitDispatchParameter {
    CircuitDispatchParameter {
        parameter_index: parameter.parameter_index(),
        name: parameter.name().to_owned(),
    }
}

fn circuit_dispatch_predicate(
    predicate: &NativeCircuitDispatchPredicate,
) -> CircuitDispatchPredicate {
    CircuitDispatchPredicate {
        root: predicate.root().index(),
        scalars: predicate.scalars().iter().map(circuit_scalar).collect(),
        parameter_names: predicate
            .parameter_names()
            .into_iter()
            .map(str::to_owned)
            .collect(),
    }
}

pub(super) fn circuit_dispatch_decision(
    decision: &NativeCircuitDispatchDecision,
) -> CircuitDispatchDecision {
    CircuitDispatchDecision {
        predicate_index: decision.predicate_index(),
        value: decision.value(),
    }
}

fn circuit_instruction(instruction: &qamomile_compiler::StaticInstruction) -> CircuitInstruction {
    CircuitInstruction {
        opcode: instruction.opcode().to_owned(),
        qubits: instruction.qubits().to_vec(),
        parameters: instruction
            .parameters()
            .iter()
            .map(|parameter| match parameter {
                NativeCircuitParameter::MeasurementIndex(index) => CircuitParameter {
                    definition: None,
                    kind: "measurement".to_owned(),
                    index: *index,
                    scalar_type: None,
                    payload: None,
                    boolean_outputs: None,
                },
                NativeCircuitParameter::Scalar(identifier) => CircuitParameter {
                    definition: None,
                    kind: "scalar".to_owned(),
                    index: u64::from(identifier.index()),
                    scalar_type: None,
                    payload: None,
                    boolean_outputs: None,
                },
                NativeCircuitParameter::CallResult {
                    target,
                    scalar_type,
                } => CircuitParameter {
                    definition: None,
                    kind: "call_result".to_owned(),
                    index: *target,
                    scalar_type: Some(circuit_scalar_type_name(*scalar_type).to_owned()),
                    payload: None,
                    boolean_outputs: None,
                },
                NativeCircuitParameter::BooleanPlan(plan) => CircuitParameter {
                    definition: None,
                    kind: "boolean_plan".to_owned(),
                    index: plan.input_width(),
                    scalar_type: None,
                    payload: None,
                    boolean_outputs: Some(
                        plan.outputs()
                            .iter()
                            .map(|output| {
                                output
                                    .monomials()
                                    .iter()
                                    .map(|monomial| monomial.variables().to_vec())
                                    .collect()
                            })
                            .collect(),
                    ),
                },
                NativeCircuitParameter::Definition(owner) => CircuitParameter {
                    kind: "definition".to_owned(),
                    index: 0,
                    scalar_type: None,
                    payload: None,
                    boolean_outputs: None,
                    definition: Some(CircuitDefinition {
                        owner: owner.clone(),
                    }),
                },
                NativeCircuitParameter::Hamiltonian(payload) => CircuitParameter {
                    definition: None,
                    kind: "hamiltonian".to_owned(),
                    index: 0,
                    scalar_type: None,
                    payload: Some(payload.to_vec()),
                    boolean_outputs: None,
                },
            })
            .collect(),
        controls: instruction.controls().to_vec(),
        condition: instruction
            .condition()
            .map(|identifier| u64::from(identifier.index())),
        regions: instruction.regions().iter().map(circuit_block).collect(),
    }
}

fn circuit_block(block: &qamomile_compiler::StaticBlock) -> CircuitBlock {
    let return_value = block.return_value().map(circuit_result_value);
    CircuitBlock {
        instructions: block
            .instructions()
            .iter()
            .map(circuit_instruction)
            .collect(),
        has_return: block.has_runtime_return(),
        return_value,
    }
}

fn circuit_scalar(node: &CircuitScalarNode) -> CircuitScalar {
    let scalar_type = circuit_scalar_type_name(node.scalar_type()).to_owned();
    if let CircuitScalarNode::Apply {
        opcode,
        attributes,
        operands,
        ..
    } = node
    {
        return circuit_apply_scalar(scalar_type, opcode, attributes, operands);
    }
    let kind = match node {
        CircuitScalarNode::Bool(_) => "bool",
        CircuitScalarNode::Int(_) => "int",
        CircuitScalarNode::RealBits(_) => "real",
        CircuitScalarNode::Measurement(_) => "measurement",
        CircuitScalarNode::LoopIndex(_) => "loop_index",
        CircuitScalarNode::CallValue { .. } => "call_value",
        CircuitScalarNode::Parameter { .. } => "parameter",
        CircuitScalarNode::Apply { .. } => unreachable!("handled scalar application"),
    };
    let mut scalar = CircuitScalar {
        kind: kind.to_owned(),
        scalar_type,
        bool_value: None,
        int_value: None,
        real_value: None,
        measurement_index: None,
        call_index: None,
        name: None,
        path: Vec::new(),
        opcode: None,
        attributes: Vec::new(),
        operands: Vec::new(),
    };
    match node {
        CircuitScalarNode::Bool(value) => scalar.bool_value = Some(*value),
        CircuitScalarNode::Int(value) => scalar.int_value = Some(*value),
        CircuitScalarNode::RealBits(value) => scalar.real_value = Some(f64::from_bits(*value)),
        CircuitScalarNode::Measurement(index) => scalar.measurement_index = Some(*index),
        CircuitScalarNode::LoopIndex(index) => scalar.int_value = Some(i64::from(*index)),
        CircuitScalarNode::CallValue { index, .. } => scalar.call_index = Some(*index),
        CircuitScalarNode::Parameter { name, path, .. } => {
            scalar.name = Some(name.clone());
            scalar.path = path
                .iter()
                .map(|component| match component {
                    CircuitParameterPath::Index(index) => CircuitPathComponent {
                        kind: "index".to_owned(),
                        index: Some(*index),
                        field: None,
                    },
                    CircuitParameterPath::Field(field) => CircuitPathComponent {
                        kind: "field".to_owned(),
                        index: None,
                        field: Some(field.clone()),
                    },
                })
                .collect();
        }
        CircuitScalarNode::Apply { .. } => unreachable!("handled scalar application"),
    }
    scalar
}

const fn circuit_scalar_type_name(scalar_type: CircuitScalarType) -> &'static str {
    match scalar_type {
        CircuitScalarType::Bool => "Bool",
        CircuitScalarType::Int => "Int",
        CircuitScalarType::Real => "Real",
    }
}

fn circuit_apply_scalar(
    scalar_type: String,
    opcode: &str,
    attributes: &[NativeCircuitScalarAttribute],
    operands: &[CircuitScalarId],
) -> CircuitScalar {
    CircuitScalar {
        kind: "apply".to_owned(),
        scalar_type,
        bool_value: None,
        int_value: None,
        real_value: None,
        measurement_index: None,
        call_index: None,
        name: None,
        path: Vec::new(),
        opcode: Some(opcode.to_owned()),
        attributes: attributes.iter().map(circuit_scalar_attribute).collect(),
        operands: operands.iter().map(|operand| operand.index()).collect(),
    }
}

fn circuit_scalar_attribute(attribute: &NativeCircuitScalarAttribute) -> CircuitScalarAttribute {
    let (kind, int_value, string_value, string_values) = match attribute.value() {
        CircuitScalarAttributeValue::Int(value) => {
            ("int".to_owned(), Some(*value), None, Vec::new())
        }
        CircuitScalarAttributeValue::String(value) => {
            ("string".to_owned(), None, Some(value.clone()), Vec::new())
        }
        CircuitScalarAttributeValue::Strings(values) => {
            ("strings".to_owned(), None, None, values.to_vec())
        }
    };
    CircuitScalarAttribute {
        name: attribute.name().to_owned(),
        kind,
        int_value,
        string_value,
        string_values,
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<CircuitInstruction>()?;
    module.add_class::<CircuitBlock>()?;
    module.add_class::<CircuitParameter>()?;
    module.add_class::<CircuitDefinition>()?;
    module.add_class::<CircuitPathComponent>()?;
    module.add_class::<CircuitScalarAttribute>()?;
    module.add_class::<CircuitScalar>()?;
    module.add_class::<CircuitResultValue>()?;
    module.add_class::<CircuitProjection>()?;
    module.add_class::<CircuitDispatchParameter>()?;
    module.add_class::<CircuitDispatchPredicate>()?;
    module.add_class::<CircuitDispatchDecision>()?;
    module.add_class::<CircuitVariantProjection>()?;
    module.add_class::<CircuitFamilyProjection>()?;
    module.add_function(wrap_pyfunction!(inspect_static_circuit, module)?)?;
    module.add_function(wrap_pyfunction!(
        inspect_static_circuit_operation_visits,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(inspect_static_circuit_at, module)?)?;
    module.add_function(wrap_pyfunction!(inspect_static_circuit_family, module)?)?;
    Ok(())
}
