//! Deterministic statevector execution for target-neutral static circuits.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use num_bigint::BigInt;
use num_complex::Complex64;
use num_traits::{FromPrimitive, ToPrimitive, Zero};
use prost::Message;
use thiserror::Error;

use crate::circuit::StaticCircuitValue;
use crate::numeric_values::{self, NumericEvaluationError};
use crate::static_values::{self, StaticScalar, StaticValue};
use crate::wire;
use crate::{
    CircuitLaunchBinding, CircuitLaunchValue, CircuitParameter, CircuitParameterPath,
    CircuitScalarAttributeValue, CircuitScalarId, CircuitScalarNode, CircuitScalarType,
    StaticCircuit, StaticInstruction,
};

const MAX_SIMULATION_QUBITS: u64 = 20;
const MAX_SIMULATION_SHOTS: u64 = 10_000_000;
const MAX_SIMULATION_WORK: u128 = 100_000_000;
const MIN_PROBABILITY: f64 = 1e-15;
const MAX_PROBABILITY_DRIFT: f64 = 1e-9;

/// Canonically ordered logical bit counts produced by native sampling.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StaticSampleCounts {
    entries: Box<[(Box<[u8]>, u64)]>,
}

impl StaticSampleCounts {
    pub(crate) fn from_entries(entries: BTreeMap<Box<[u8]>, u64>) -> Self {
        Self {
            entries: entries.into_iter().collect::<Vec<_>>().into_boxed_slice(),
        }
    }

    /// Return lexicographically ordered logical bitstrings and counts.
    #[must_use]
    pub fn entries(&self) -> &[(Box<[u8]>, u64)] {
        &self.entries
    }
}

/// Failure to execute a target-neutral static circuit exactly.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum StaticSimulationError {
    /// The requested execution would exceed a fixed memory or work budget.
    #[error("Q4401: native static simulation exceeds {0}")]
    Budget(String),
    /// The circuit contains an instruction outside the native simulator subset.
    #[error("Q4402: native static simulation does not support {0}")]
    Unsupported(String),
    /// A circuit-owned reference or runtime value is unavailable.
    #[error("Q4403: native static simulation is missing {0}")]
    Missing(String),
    /// A circuit instruction violates its verified logical shape.
    #[error("Q4404: malformed native static circuit: {0}")]
    Malformed(String),
    /// A selected exact operation is undefined for its runtime operands.
    #[error("Q4405: invalid native static circuit evaluation: {0}")]
    Invalid(String),
}

/// Sample one compiler-owned static circuit without an SDK or Python callback.
///
/// # Errors
///
/// Returns [`StaticSimulationError`] when a statevector, shot, or work budget
/// is exceeded, a launch scalar cannot be resolved, or an unsupported
/// instruction reaches this exact execution subset.
pub fn sample_static_circuit(
    circuit: &StaticCircuit,
    bindings: &[CircuitLaunchBinding],
    shots: u64,
    seed: u64,
) -> Result<StaticSampleCounts, StaticSimulationError> {
    let retained_scalar_count = if circuit.result_measurements().is_empty() {
        circuit
            .runtime_result_value()
            .map_or(0, count_result_scalars)
    } else {
        0
    };
    let (dimension, measurement_count, call_value_count) =
        validate_simulation(circuit, bindings, shots, retained_scalar_count)?;
    let mut random = SplitMix64::new(seed);
    let mut counts = BTreeMap::<Box<[u8]>, u64>::new();
    for _ in 0..shots {
        let mut state = initialized_shot(
            circuit,
            &mut random,
            dimension,
            measurement_count,
            call_value_count,
            shots,
        )?;
        let returned =
            match state.execute_instructions(circuit.instructions(), circuit, bindings)? {
                RuntimeControl::Next => None,
                RuntimeControl::Return(value) => Some(value),
            };
        let result = sample_result(
            circuit,
            bindings,
            &state.measurements,
            &state.call_values,
            returned.as_ref(),
        )?;
        *counts.entry(result).or_default() += 1;
    }
    Ok(StaticSampleCounts::from_entries(counts))
}

fn sample_result(
    circuit: &StaticCircuit,
    bindings: &[CircuitLaunchBinding],
    measurements: &[Option<bool>],
    call_values: &[Option<Scalar>],
    returned: Option<&StaticValue>,
) -> Result<Box<[u8]>, StaticSimulationError> {
    if let Some(value) = returned {
        let mut bits = Vec::new();
        flatten_sample_value(value, &mut bits)?;
        return Ok(bits.into_boxed_slice());
    }
    if !circuit.result_measurements().is_empty() || circuit.runtime_result_value().is_none() {
        return circuit
            .result_measurements()
            .iter()
            .map(|index| {
                usize::try_from(*index)
                    .ok()
                    .and_then(|index| measurements.get(index))
                    .copied()
                    .flatten()
                    .map(u8::from)
                    .ok_or_else(|| {
                        StaticSimulationError::Missing(format!("result measurement {index}"))
                    })
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Vec::into_boxed_slice);
    }
    let value = evaluate_result(
        circuit
            .runtime_result_value()
            .expect("checked retained sample result"),
        circuit,
        bindings,
        measurements,
        call_values,
    )?;
    let mut bits = Vec::new();
    flatten_sample_value(&value, &mut bits)?;
    Ok(bits.into_boxed_slice())
}

fn flatten_sample_value(
    value: &StaticValue,
    output: &mut Vec<u8>,
) -> Result<(), StaticSimulationError> {
    match value {
        StaticValue::Unit => {}
        StaticValue::Scalar(StaticScalar::Bool(value)) => output.push(u8::from(*value)),
        StaticValue::Aggregate(values) => {
            for value in values.iter() {
                flatten_sample_value(
                    value.as_ref().ok_or_else(|| {
                        StaticSimulationError::Malformed(
                            "incomplete retained sample result".to_owned(),
                        )
                    })?,
                    output,
                )?;
            }
        }
        StaticValue::Record(fields) => {
            for (_, value) in fields.iter() {
                flatten_sample_value(
                    value.as_ref().ok_or_else(|| {
                        StaticSimulationError::Malformed(
                            "incomplete retained sample result".to_owned(),
                        )
                    })?,
                    output,
                )?;
            }
        }
        StaticValue::Scalar(StaticScalar::RealBits(raw)) => {
            output.extend(
                crate::result_codec::encode_sampled_real(f64::from_bits(*raw))
                    .map_err(|error| StaticSimulationError::Malformed(error.to_string()))?,
            );
        }
        StaticValue::Scalar(StaticScalar::Int(_))
        | StaticValue::BigInt(_)
        | StaticValue::ComplexBits(_, _)
        | StaticValue::String(_)
        | StaticValue::Bytes(_)
        | StaticValue::Mapping(_) => {
            return Err(StaticSimulationError::Unsupported(
                "unsupported retained sample result".to_owned(),
            ));
        }
    }
    Ok(())
}

/// Execute one compiler-owned static circuit and return its retained value.
///
/// # Errors
///
/// Returns [`StaticSimulationError`] when the circuit has no non-sampled
/// result, exceeds a fixed statevector or work budget, or cannot evaluate its
/// retained scalar-expression tree.
pub(crate) fn evaluate_static_circuit(
    circuit: &StaticCircuit,
    bindings: &[CircuitLaunchBinding],
    seed: u64,
) -> Result<Vec<u8>, StaticSimulationError> {
    let result = circuit.runtime_result_value().ok_or_else(|| {
        StaticSimulationError::Unsupported("a retained non-sampled result".to_owned())
    })?;
    let result_scalar_count = count_result_scalars(result);
    let (dimension, measurement_count, call_value_count) =
        validate_simulation(circuit, bindings, 1, result_scalar_count)?;
    let mut random = SplitMix64::new(seed);
    let mut state = initialized_shot(
        circuit,
        &mut random,
        dimension,
        measurement_count,
        call_value_count,
        1,
    )?;
    let value = match state.execute_instructions(circuit.instructions(), circuit, bindings)? {
        RuntimeControl::Next => evaluate_result(
            result,
            circuit,
            bindings,
            &state.measurements,
            &state.call_values,
        )?,
        RuntimeControl::Return(value) => value,
    };
    let literal = crate::static_values::value_to_literal(&value).ok_or_else(|| {
        StaticSimulationError::Malformed("incomplete retained result value".to_owned())
    })?;
    Ok(literal.encode_to_vec())
}

/// Evaluate one checked result-plan row through the simulator's scalar semantics.
pub(crate) fn evaluate_retained_result(
    circuit: &StaticCircuit,
    bindings: &[CircuitLaunchBinding],
    measurements: &[u8],
) -> Result<wire::LiteralProto, StaticSimulationError> {
    let measurement_values = measurements
        .iter()
        .map(|value| match value {
            0 => Ok(Some(false)),
            1 => Ok(Some(true)),
            _ => Err(StaticSimulationError::Malformed(
                "non-bit result measurement".to_owned(),
            )),
        })
        .collect::<Result<Vec<_>, _>>()?;
    if measurement_values.len()
        != usize::try_from(circuit.measurement_count())
            .map_err(|_| StaticSimulationError::Budget("result measurement width".to_owned()))?
    {
        return Err(StaticSimulationError::Malformed(
            "result measurement width".to_owned(),
        ));
    }
    let result = evaluate_result(
        circuit
            .runtime_result_value()
            .ok_or_else(|| StaticSimulationError::Missing("retained result".to_owned()))?,
        circuit,
        bindings,
        &measurement_values,
        &[],
    )?;
    static_values::value_to_literal(&result).ok_or_else(|| {
        StaticSimulationError::Malformed("incomplete retained result value".to_owned())
    })
}

pub(crate) struct CircuitScalarEvaluator<'a> {
    circuit: &'a StaticCircuit,
    bindings: &'a [CircuitLaunchBinding],
    measurements: Vec<Option<bool>>,
    cache: Vec<Option<Scalar>>,
}

impl<'a> CircuitScalarEvaluator<'a> {
    pub(crate) fn new(
        circuit: &'a StaticCircuit,
        bindings: &'a [CircuitLaunchBinding],
        prefix: &[u8],
    ) -> Result<Self, StaticSimulationError> {
        let mut measurements = vec![
            None;
            usize::try_from(circuit.measurement_count()).map_err(|_| {
                StaticSimulationError::Malformed("measurement width".into())
            })?
        ];
        for (slot, value) in measurements.iter_mut().zip(prefix) {
            *slot = Some(*value != 0);
        }
        Ok(Self {
            circuit,
            bindings,
            measurements,
            cache: vec![None; circuit.scalars().len()],
        })
    }

    pub(crate) fn evaluate(
        &mut self,
        id: CircuitScalarId,
    ) -> Result<wire::LiteralProto, StaticSimulationError> {
        let value = scalar_with_cache(
            id,
            self.circuit.scalars(),
            self.bindings,
            &self.measurements,
            &[],
            &mut self.cache,
        )?;
        static_values::value_to_literal(&scalar_value(&value))
            .ok_or_else(|| StaticSimulationError::Malformed("scalar result".into()))
    }
}

/// Evaluate one compiler-verified launch predicate through shared scalar semantics.
pub(crate) fn evaluate_dispatch_predicate(
    predicate: &crate::CircuitDispatchPredicate,
    bindings: &[CircuitLaunchBinding],
) -> Result<bool, StaticSimulationError> {
    let mut cache = vec![None; predicate.scalars().len()];
    match scalar_with_cache(
        predicate.root(),
        predicate.scalars(),
        bindings,
        &[],
        &[],
        &mut cache,
    )? {
        Scalar::Bool(value) => Ok(value),
        Scalar::Int(_) | Scalar::Real(_) => Err(StaticSimulationError::Malformed(
            "dispatch predicate root type".to_owned(),
        )),
    }
}

fn validate_simulation(
    circuit: &StaticCircuit,
    bindings: &[CircuitLaunchBinding],
    shots: u64,
    additional_scalar_evaluations: u128,
) -> Result<(usize, usize, usize), StaticSimulationError> {
    let mut binding_names = BTreeSet::new();
    if let Some(binding) = bindings
        .iter()
        .find(|binding| !binding_names.insert(binding.name()))
    {
        return Err(StaticSimulationError::Malformed(format!(
            "duplicate launch parameter {:?}",
            binding.name()
        )));
    }
    if shots == 0 || shots > MAX_SIMULATION_SHOTS {
        return Err(StaticSimulationError::Budget(format!(
            "the shot limit of {MAX_SIMULATION_SHOTS}"
        )));
    }
    if circuit.width() > MAX_SIMULATION_QUBITS {
        return Err(StaticSimulationError::Budget(format!(
            "the {MAX_SIMULATION_QUBITS}-qubit statevector limit"
        )));
    }
    let dimension = 1usize
        .checked_shl(
            u32::try_from(circuit.width())
                .map_err(|_| StaticSimulationError::Budget("statevector width".to_owned()))?,
        )
        .ok_or_else(|| StaticSimulationError::Budget("statevector width".to_owned()))?;
    let measurement_count = usize::try_from(circuit.measurement_count())
        .map_err(|_| StaticSimulationError::Budget("measurement count".to_owned()))?;
    let call_value_count = call_value_count(circuit.instructions())?;
    let call_value_count = usize::try_from(call_value_count)
        .map_err(|_| StaticSimulationError::Budget("call value count".to_owned()))?;
    let (operation_count, scalar_evaluations) = simulation_operation_count(circuit)?;
    let scalar_evaluations = scalar_evaluations
        .checked_add(additional_scalar_evaluations)
        .ok_or_else(|| StaticSimulationError::Budget("the simulation work budget".to_owned()))?;
    let state_work = u128::try_from(dimension)
        .ok()
        .and_then(|dimension| dimension.checked_mul(operation_count.max(1)));
    let scalar_work = u128::try_from(circuit.scalars().len())
        .ok()
        .and_then(|scalar_count| scalar_count.checked_mul(scalar_evaluations));
    let work = state_work
        .and_then(|state_work| {
            scalar_work.and_then(|scalar_work| state_work.checked_add(scalar_work))
        })
        .and_then(|work| work.checked_mul(u128::from(shots)))
        .ok_or_else(|| StaticSimulationError::Budget("the simulation work budget".to_owned()))?;
    if work > MAX_SIMULATION_WORK {
        return Err(StaticSimulationError::Budget(format!(
            "the {MAX_SIMULATION_WORK}-step work limit"
        )));
    }
    Ok((dimension, measurement_count, call_value_count))
}

fn call_value_count(instructions: &[StaticInstruction]) -> Result<u64, StaticSimulationError> {
    let mut count = 0_u64;
    let mut pending = vec![instructions];
    while let Some(instructions) = pending.pop() {
        for instruction in instructions {
            for parameter in instruction.parameters() {
                if let CircuitParameter::CallResult { target, .. } = parameter {
                    count = count.max(target.checked_add(1).ok_or_else(|| {
                        StaticSimulationError::Budget("call value count".to_owned())
                    })?);
                }
            }
            pending.extend(
                instruction
                    .regions()
                    .iter()
                    .map(crate::StaticBlock::instructions),
            );
        }
    }
    Ok(count)
}

fn initialized_shot<'a>(
    circuit: &StaticCircuit,
    random: &'a mut SplitMix64,
    dimension: usize,
    measurement_count: usize,
    call_value_count: usize,
    shots: u64,
) -> Result<ShotState<'a>, StaticSimulationError> {
    let remaining_operation_work = MAX_SIMULATION_WORK
        .checked_div(u128::from(shots))
        .and_then(|work| work.checked_div(dimension as u128))
        .ok_or_else(|| StaticSimulationError::Budget("the simulation work budget".to_owned()))?;
    let mut state = ShotState {
        amplitudes: vec![Complex64::new(0.0, 0.0); dimension],
        measurements: vec![None; measurement_count],
        call_values: vec![None; call_value_count],
        remaining_operation_work,
        width: usize::try_from(circuit.width())
            .map_err(|_| StaticSimulationError::Budget("statevector width".to_owned()))?,
        random,
    };
    state.amplitudes[0] = Complex64::new(1.0, 0.0);
    Ok(state)
}

fn count_result_scalars(result: &StaticCircuitValue) -> u128 {
    match result {
        StaticCircuitValue::Scalar(_) => 1,
        StaticCircuitValue::Aggregate(values) => values.iter().map(count_result_scalars).sum(),
        StaticCircuitValue::Record(fields) => fields
            .iter()
            .map(|(_, value)| count_result_scalars(value))
            .sum(),
        StaticCircuitValue::Unit
        | StaticCircuitValue::Portable(_)
        | StaticCircuitValue::Wires(_) => 0,
    }
}

fn evaluate_result(
    result: &StaticCircuitValue,
    circuit: &StaticCircuit,
    bindings: &[CircuitLaunchBinding],
    measurements: &[Option<bool>],
    call_values: &[Option<Scalar>],
) -> Result<StaticValue, StaticSimulationError> {
    let mut cache = vec![None; circuit.scalars().len()];
    evaluate_result_with_cache(
        result,
        circuit,
        bindings,
        measurements,
        call_values,
        &mut cache,
    )
}

fn evaluate_result_with_cache(
    result: &StaticCircuitValue,
    circuit: &StaticCircuit,
    bindings: &[CircuitLaunchBinding],
    measurements: &[Option<bool>],
    call_values: &[Option<Scalar>],
    cache: &mut [Option<Scalar>],
) -> Result<StaticValue, StaticSimulationError> {
    match result {
        StaticCircuitValue::Unit | StaticCircuitValue::Wires(_) => Ok(StaticValue::Unit),
        StaticCircuitValue::Portable(value) => Ok(value.clone()),
        StaticCircuitValue::Scalar(identifier) => {
            let value = scalar_with_cache(
                *identifier,
                circuit.scalars(),
                bindings,
                measurements,
                call_values,
                cache,
            )?;
            Ok(scalar_value(&value))
        }
        StaticCircuitValue::Aggregate(values) => Ok(StaticValue::Aggregate(
            values
                .iter()
                .map(|value| {
                    evaluate_result_with_cache(
                        value,
                        circuit,
                        bindings,
                        measurements,
                        call_values,
                        cache,
                    )
                    .map(Some)
                })
                .collect::<Result<Vec<_>, _>>()?
                .into(),
        )),
        StaticCircuitValue::Record(fields) => Ok(StaticValue::Record(
            fields
                .iter()
                .map(|(name, value)| {
                    evaluate_result_with_cache(
                        value,
                        circuit,
                        bindings,
                        measurements,
                        call_values,
                        cache,
                    )
                    .map(|value| (Arc::from(name.as_str()), Some(value)))
                })
                .collect::<Result<Vec<_>, _>>()?
                .into(),
        )),
    }
}

fn simulation_operation_count(
    circuit: &StaticCircuit,
) -> Result<(u128, u128), StaticSimulationError> {
    let mut count = u128::from(circuit.measurement_count());
    let mut scalar_evaluations = 0u128;
    let mut pending = vec![circuit.instructions()];
    while let Some(instructions) = pending.pop() {
        count = count
            .checked_add(u128::try_from(instructions.len()).map_err(|_| {
                StaticSimulationError::Budget("the simulation work budget".to_owned())
            })?)
            .ok_or_else(|| {
                StaticSimulationError::Budget("the simulation work budget".to_owned())
            })?;
        for instruction in instructions {
            let scalar_uses = instruction
                .parameters()
                .iter()
                .map(|parameter| match parameter {
                    CircuitParameter::MeasurementIndex(_)
                    | CircuitParameter::CallResult { .. }
                    | CircuitParameter::Hamiltonian(_)
                    | CircuitParameter::BooleanPlan(_)
                    | CircuitParameter::Definition(_) => 0,
                    CircuitParameter::Scalar(_) => 1,
                })
                .sum::<usize>()
                + usize::from(instruction.condition().is_some());
            scalar_evaluations = scalar_evaluations
                .checked_add(u128::try_from(scalar_uses).map_err(|_| {
                    StaticSimulationError::Budget("the simulation work budget".to_owned())
                })?)
                .ok_or_else(|| {
                    StaticSimulationError::Budget("the simulation work budget".to_owned())
                })?;
            for region in instruction.regions() {
                if let Some(value) = region.runtime_return_value() {
                    scalar_evaluations = scalar_evaluations
                        .checked_add(count_result_scalars(value))
                        .ok_or_else(|| {
                            StaticSimulationError::Budget("the simulation work budget".to_owned())
                        })?;
                }
                pending.push(region.instructions());
            }
        }
    }
    Ok((count, scalar_evaluations))
}

struct ShotState<'a> {
    amplitudes: Vec<Complex64>,
    measurements: Vec<Option<bool>>,
    call_values: Vec<Option<Scalar>>,
    remaining_operation_work: u128,
    width: usize,
    random: &'a mut SplitMix64,
}

enum RuntimeControl {
    Next,
    Return(StaticValue),
}

impl ShotState<'_> {
    fn execute_instructions(
        &mut self,
        instructions: &[StaticInstruction],
        circuit: &StaticCircuit,
        bindings: &[CircuitLaunchBinding],
    ) -> Result<RuntimeControl, StaticSimulationError> {
        for instruction in instructions {
            let control = self.execute_instruction(instruction, circuit, bindings)?;
            if !matches!(control, RuntimeControl::Next) {
                return Ok(control);
            }
        }
        Ok(RuntimeControl::Next)
    }

    fn execute_block(
        &mut self,
        block: &crate::StaticBlock,
        circuit: &StaticCircuit,
        bindings: &[CircuitLaunchBinding],
    ) -> Result<RuntimeControl, StaticSimulationError> {
        let control = self.execute_instructions(block.instructions(), circuit, bindings)?;
        if !matches!(control, RuntimeControl::Next) {
            return Ok(control);
        }
        let Some(result) = block.runtime_return_value() else {
            return Ok(RuntimeControl::Next);
        };
        evaluate_result(
            result,
            circuit,
            bindings,
            &self.measurements,
            &self.call_values,
        )
        .map(RuntimeControl::Return)
    }

    fn execute_instruction(
        &mut self,
        instruction: &StaticInstruction,
        circuit: &StaticCircuit,
        bindings: &[CircuitLaunchBinding],
    ) -> Result<RuntimeControl, StaticSimulationError> {
        self.remaining_operation_work =
            self.remaining_operation_work
                .checked_sub(1)
                .ok_or_else(|| {
                    StaticSimulationError::Budget(format!(
                        "the {MAX_SIMULATION_WORK}-step work limit"
                    ))
                })?;
        let qubits = instruction
            .qubits()
            .iter()
            .map(|wire| self.wire(*wire))
            .collect::<Result<Vec<_>, _>>()?;
        let controls = instruction
            .controls()
            .iter()
            .map(|wire| self.wire(*wire))
            .collect::<Result<Vec<_>, _>>()?;
        match instruction.opcode() {
            "measure" => {
                let [CircuitParameter::MeasurementIndex(index)] = instruction.parameters() else {
                    return Err(StaticSimulationError::Malformed(
                        "measurement parameter ABI".to_owned(),
                    ));
                };
                let position = usize::try_from(*index).map_err(|_| {
                    StaticSimulationError::Malformed("measurement index".to_owned())
                })?;
                let outcome = self.measure(qubit(&qubits, 0)?)?;
                let target = self.measurements.get_mut(position).ok_or_else(|| {
                    StaticSimulationError::Malformed("measurement index".to_owned())
                })?;
                *target = Some(outcome);
                Ok(RuntimeControl::Next)
            }
            "reset" => {
                let wire = qubit(&qubits, 0)?;
                if self.measure(wire)? {
                    self.single(wire, &[], pauli_x());
                }
                Ok(RuntimeControl::Next)
            }
            "if" => {
                if !qubits.is_empty() || !controls.is_empty() || instruction.regions().len() != 2 {
                    return Err(StaticSimulationError::Malformed(
                        "structured if ABI".to_owned(),
                    ));
                }
                let condition = instruction.condition().ok_or_else(|| {
                    StaticSimulationError::Malformed("structured if condition".to_owned())
                })?;
                let selected = usize::from(
                    !scalar(
                        condition,
                        circuit,
                        bindings,
                        &self.measurements,
                        &self.call_values,
                    )?
                    .truth()?,
                );
                let block = instruction.regions().get(selected).ok_or_else(|| {
                    StaticSimulationError::Malformed("structured if branch".to_owned())
                })?;
                self.execute_block(block, circuit, bindings)
            }
            "call" => self.execute_call(instruction, circuit, bindings),
            _ => {
                self.execute_unitary(instruction, circuit, bindings, &qubits, &controls)?;
                Ok(RuntimeControl::Next)
            }
        }
    }

    fn execute_call(
        &mut self,
        instruction: &StaticInstruction,
        circuit: &StaticCircuit,
        bindings: &[CircuitLaunchBinding],
    ) -> Result<RuntimeControl, StaticSimulationError> {
        if !instruction.qubits().is_empty()
            || !instruction.controls().is_empty()
            || instruction.condition().is_some()
            || instruction.regions().len() != 1
            || instruction
                .parameters()
                .iter()
                .any(|parameter| !matches!(parameter, CircuitParameter::CallResult { .. }))
        {
            return Err(StaticSimulationError::Malformed(
                "structured call ABI".to_owned(),
            ));
        }
        let block = instruction
            .regions()
            .first()
            .ok_or_else(|| StaticSimulationError::Malformed("call body".to_owned()))?;
        let RuntimeControl::Return(value) = self.execute_block(block, circuit, bindings)? else {
            return Err(StaticSimulationError::Malformed(
                "structured call completed without a return value".to_owned(),
            ));
        };
        let mut values = Vec::new();
        collect_runtime_scalar_values(&value, &mut values)?;
        if values.len() != instruction.parameters().len() {
            return Err(StaticSimulationError::Malformed(
                "structured call result width".to_owned(),
            ));
        }
        for (parameter, value) in instruction.parameters().iter().zip(values) {
            let CircuitParameter::CallResult {
                target,
                scalar_type,
            } = parameter
            else {
                unreachable!("validated call result parameter")
            };
            let value = runtime_scalar_from_value(*scalar_type, value.clone())?;
            let target = usize::try_from(*target)
                .ok()
                .and_then(|target| self.call_values.get_mut(target))
                .ok_or_else(|| {
                    StaticSimulationError::Malformed("structured call result target".to_owned())
                })?;
            *target = Some(value);
        }
        Ok(RuntimeControl::Next)
    }

    fn execute_unitary(
        &mut self,
        instruction: &StaticInstruction,
        circuit: &StaticCircuit,
        bindings: &[CircuitLaunchBinding],
        qubits: &[usize],
        controls: &[usize],
    ) -> Result<(), StaticSimulationError> {
        match instruction.opcode() {
            "h" => self.single(qubit(qubits, 0)?, controls, hadamard()),
            "x" => self.single(qubit(qubits, 0)?, controls, pauli_x()),
            "y" => self.single(qubit(qubits, 0)?, controls, pauli_y()),
            "z" => self.phase_when(controls, &[qubit(qubits, 0)?], -1.0, 0.0),
            "s" => self.phase_when(controls, &[qubit(qubits, 0)?], 0.0, 1.0),
            "sdg" => self.phase_when(controls, &[qubit(qubits, 0)?], 0.0, -1.0),
            "t" => self.phase_angle(controls, &[qubit(qubits, 0)?], std::f64::consts::FRAC_PI_4),
            "tdg" => self.phase_angle(controls, &[qubit(qubits, 0)?], -std::f64::consts::FRAC_PI_4),
            "p" => self.phase_angle(
                controls,
                &[qubit(qubits, 0)?],
                self.instruction_angle(instruction, circuit, bindings)?,
            ),
            "rx" => self.single(
                qubit(qubits, 0)?,
                controls,
                rotation_x(self.instruction_angle(instruction, circuit, bindings)?),
            ),
            "ry" => self.single(
                qubit(qubits, 0)?,
                controls,
                rotation_y(self.instruction_angle(instruction, circuit, bindings)?),
            ),
            "rz" => self.single(
                qubit(qubits, 0)?,
                controls,
                rotation_z(self.instruction_angle(instruction, circuit, bindings)?),
            ),
            "cx" => self.single(
                qubit(qubits, 1)?,
                &with_control(controls, qubit(qubits, 0)?)?,
                pauli_x(),
            ),
            "cz" => self.phase_when(controls, &two_qubits(qubits)?, -1.0, 0.0),
            "ccx" => self.single(
                qubit(qubits, 2)?,
                &with_controls(controls, &qubits[..2])?,
                pauli_x(),
            ),
            "swap" => self.swap(two_qubits(qubits)?, controls),
            "cp" => self.phase_angle(
                controls,
                &two_qubits(qubits)?,
                self.instruction_angle(instruction, circuit, bindings)?,
            ),
            "rzz" => self.rzz(
                two_qubits(qubits)?,
                controls,
                self.instruction_angle(instruction, circuit, bindings)?,
            ),
            "global_phase" => self.global_phase(
                controls,
                self.instruction_angle(instruction, circuit, bindings)?,
            ),
            opcode => {
                return Err(StaticSimulationError::Unsupported(format!(
                    "instruction {opcode:?}"
                )));
            }
        }
        Ok(())
    }

    fn instruction_angle(
        &self,
        instruction: &StaticInstruction,
        circuit: &StaticCircuit,
        bindings: &[CircuitLaunchBinding],
    ) -> Result<f64, StaticSimulationError> {
        angle(
            instruction,
            circuit,
            bindings,
            &self.measurements,
            &self.call_values,
        )
    }

    fn wire(&self, wire: u64) -> Result<usize, StaticSimulationError> {
        let wire = usize::try_from(wire)
            .map_err(|_| StaticSimulationError::Malformed("wire index".to_owned()))?;
        if wire >= self.width {
            return Err(StaticSimulationError::Malformed("wire index".to_owned()));
        }
        Ok(wire)
    }

    fn single(&mut self, target: usize, controls: &[usize], matrix: [[Complex64; 2]; 2]) {
        let mask = 1usize << target;
        for base in 0..self.amplitudes.len() {
            if base & mask != 0 || !matches_controls(base, controls) {
                continue;
            }
            let paired = base | mask;
            let zero = self.amplitudes[base];
            let one = self.amplitudes[paired];
            self.amplitudes[base] = matrix[0][0] * zero + matrix[0][1] * one;
            self.amplitudes[paired] = matrix[1][0] * zero + matrix[1][1] * one;
        }
    }

    fn phase_when(&mut self, controls: &[usize], required: &[usize], real: f64, imaginary: f64) {
        let phase = Complex64::new(real, imaginary);
        for (basis, amplitude) in self.amplitudes.iter_mut().enumerate() {
            if matches_controls(basis, controls) && matches_controls(basis, required) {
                *amplitude *= phase;
            }
        }
    }

    fn phase_angle(&mut self, controls: &[usize], required: &[usize], angle: f64) {
        self.phase_when(controls, required, angle.cos(), angle.sin());
    }

    fn global_phase(&mut self, controls: &[usize], angle: f64) {
        let phase = Complex64::from_polar(1.0, angle);
        for (basis, amplitude) in self.amplitudes.iter_mut().enumerate() {
            if matches_controls(basis, controls) {
                *amplitude *= phase;
            }
        }
    }

    fn swap(&mut self, qubits: [usize; 2], controls: &[usize]) {
        let [left, right] = qubits;
        let left_mask = 1usize << left;
        let right_mask = 1usize << right;
        for basis in 0..self.amplitudes.len() {
            if basis & left_mask == 0
                && basis & right_mask != 0
                && matches_controls(basis, controls)
            {
                self.amplitudes.swap(basis, basis ^ left_mask ^ right_mask);
            }
        }
    }

    fn rzz(&mut self, qubits: [usize; 2], controls: &[usize], angle: f64) {
        let [left, right] = qubits;
        for (basis, amplitude) in self.amplitudes.iter_mut().enumerate() {
            if matches_controls(basis, controls) {
                let parity = ((basis >> left) ^ (basis >> right)) & 1;
                *amplitude *= Complex64::from_polar(
                    1.0,
                    if parity == 0 {
                        -angle / 2.0
                    } else {
                        angle / 2.0
                    },
                );
            }
        }
    }

    fn measure(&mut self, wire: usize) -> Result<bool, StaticSimulationError> {
        let (norm, probability_one) = self.amplitudes.iter().enumerate().fold(
            (0.0, 0.0),
            |(norm, probability_one), (basis, amplitude)| {
                let probability = amplitude.norm_sqr();
                (
                    norm + probability,
                    probability_one
                        + if (basis >> wire) & 1 == 1 {
                            probability
                        } else {
                            0.0
                        },
                )
            },
        );
        if !norm.is_finite() || (norm - 1.0).abs() > MAX_PROBABILITY_DRIFT {
            return Err(StaticSimulationError::Malformed(format!(
                "statevector norm {norm:?} before measurement"
            )));
        }
        let normalized_one = (probability_one / norm).clamp(0.0, 1.0);
        let mut outcome = self.random.next_f64() < normalized_one;
        let mut probability = if outcome {
            probability_one
        } else {
            norm - probability_one
        };
        if probability <= MIN_PROBABILITY {
            outcome = !outcome;
            probability = if outcome {
                probability_one
            } else {
                norm - probability_one
            };
        }
        if !probability.is_finite() || probability <= 0.0 {
            return Err(StaticSimulationError::Malformed(
                "measurement probability".to_owned(),
            ));
        }
        let scale = probability.sqrt().recip();
        for (basis, amplitude) in self.amplitudes.iter_mut().enumerate() {
            if ((basis >> wire) & 1 == 1) == outcome {
                *amplitude *= scale;
            } else {
                *amplitude = Complex64::new(0.0, 0.0);
            }
        }
        Ok(outcome)
    }
}

#[derive(Clone)]
enum Scalar {
    Bool(bool),
    Int(Arc<BigInt>),
    Real(f64),
}

impl Scalar {
    const fn scalar_type(&self) -> CircuitScalarType {
        match self {
            Self::Bool(_) => CircuitScalarType::Bool,
            Self::Int(_) => CircuitScalarType::Int,
            Self::Real(_) => CircuitScalarType::Real,
        }
    }

    fn truth(&self) -> Result<bool, StaticSimulationError> {
        match self {
            Self::Bool(value) => Ok(*value),
            _ => Err(StaticSimulationError::Malformed(
                "non-Boolean condition".to_owned(),
            )),
        }
    }

    fn real(&self) -> Result<f64, StaticSimulationError> {
        match self {
            Self::Int(value) => value.to_f64().ok_or_else(|| {
                StaticSimulationError::Unsupported(
                    "integer angle cannot be represented as binary64".to_owned(),
                )
            }),
            Self::Real(value) if value.is_finite() => Ok(*value),
            _ => Err(StaticSimulationError::Unsupported(
                "non-real circuit angle".to_owned(),
            )),
        }
    }
}

fn angle(
    instruction: &StaticInstruction,
    circuit: &StaticCircuit,
    bindings: &[CircuitLaunchBinding],
    measurements: &[Option<bool>],
    call_values: &[Option<Scalar>],
) -> Result<f64, StaticSimulationError> {
    let [CircuitParameter::Scalar(identifier)] = instruction.parameters() else {
        return Err(StaticSimulationError::Malformed(format!(
            "angle ABI for {:?}",
            instruction.opcode()
        )));
    };
    scalar(*identifier, circuit, bindings, measurements, call_values).and_then(|value| value.real())
}

fn scalar(
    identifier: CircuitScalarId,
    circuit: &StaticCircuit,
    bindings: &[CircuitLaunchBinding],
    measurements: &[Option<bool>],
    call_values: &[Option<Scalar>],
) -> Result<Scalar, StaticSimulationError> {
    let mut cache: Vec<Option<Scalar>> = vec![None; circuit.scalars().len()];
    scalar_with_cache(
        identifier,
        circuit.scalars(),
        bindings,
        measurements,
        call_values,
        &mut cache,
    )
}

fn scalar_with_cache(
    identifier: CircuitScalarId,
    scalars: &[CircuitScalarNode],
    bindings: &[CircuitLaunchBinding],
    measurements: &[Option<bool>],
    call_values: &[Option<Scalar>],
    cache: &mut [Option<Scalar>],
) -> Result<Scalar, StaticSimulationError> {
    let root = usize::try_from(identifier.index())
        .map_err(|_| StaticSimulationError::Malformed("scalar index".to_owned()))?;
    if root >= scalars.len() {
        return Err(StaticSimulationError::Malformed("scalar index".to_owned()));
    }
    let mut pending = vec![(root, false)];
    while let Some((index, operands_ready)) = pending.pop() {
        if cache[index].is_some() {
            continue;
        }
        let node = scalars
            .get(index)
            .ok_or_else(|| StaticSimulationError::Malformed("scalar index".to_owned()))?;
        if advance_lazy_select(index, node, cache, &mut pending, operands_ready)? {
            continue;
        }
        if let CircuitScalarNode::Apply { operands, .. } = node
            && !operands_ready
        {
            pending.push((index, true));
            for operand in operands.iter().rev() {
                let operand = usize::try_from(operand.index())
                    .map_err(|_| StaticSimulationError::Malformed("scalar index".to_owned()))?;
                if operand >= index {
                    return Err(StaticSimulationError::Malformed(
                        "non-topological scalar operand".to_owned(),
                    ));
                }
                if cache[operand].is_none() {
                    pending.push((operand, false));
                }
            }
            continue;
        }
        let value = match node {
            CircuitScalarNode::LoopIndex(_) => {
                return Err(StaticSimulationError::Malformed(
                    "scoped loop index is not a runtime scalar".to_owned(),
                ));
            }
            CircuitScalarNode::Bool(value) => Scalar::Bool(*value),
            CircuitScalarNode::Int(value) => Scalar::Int(Arc::new(BigInt::from(*value))),
            CircuitScalarNode::RealBits(value) => Scalar::Real(f64::from_bits(*value)),
            CircuitScalarNode::Measurement(index) => Scalar::Bool(
                usize::try_from(*index)
                    .ok()
                    .and_then(|index| measurements.get(index))
                    .copied()
                    .flatten()
                    .ok_or_else(|| {
                        StaticSimulationError::Missing(format!("measurement {index}"))
                    })?,
            ),
            CircuitScalarNode::CallValue { index, scalar_type } => {
                runtime_slot_value(*index, *scalar_type, call_values, "call value")?
            }
            CircuitScalarNode::Parameter {
                name,
                path,
                scalar_type,
            } => parameter_scalar(name, path, *scalar_type, bindings)?,
            CircuitScalarNode::Apply {
                opcode,
                result_type,
                attributes,
                operands,
            } => {
                let args = operands
                    .iter()
                    .map(|operand| {
                        let operand = usize::try_from(operand.index()).map_err(|_| {
                            StaticSimulationError::Malformed("scalar index".to_owned())
                        })?;
                        cache.get(operand).cloned().flatten().ok_or_else(|| {
                            StaticSimulationError::Malformed("unresolved scalar operand".to_owned())
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                apply_scalar(opcode, *result_type, attributes, &args)?
            }
        };
        cache[index] = Some(value);
    }
    cache[root]
        .clone()
        .ok_or_else(|| StaticSimulationError::Malformed("unresolved scalar root".to_owned()))
}

fn runtime_slot_value(
    index: u64,
    scalar_type: CircuitScalarType,
    values: &[Option<Scalar>],
    label: &str,
) -> Result<Scalar, StaticSimulationError> {
    let value = usize::try_from(index)
        .ok()
        .and_then(|index| values.get(index))
        .cloned()
        .flatten()
        .ok_or_else(|| StaticSimulationError::Missing(format!("{label} {index}")))?;
    if value.scalar_type() != scalar_type {
        return Err(StaticSimulationError::Malformed(format!("{label} type")));
    }
    Ok(value)
}

fn advance_lazy_select(
    index: usize,
    node: &CircuitScalarNode,
    cache: &mut [Option<Scalar>],
    pending: &mut Vec<(usize, bool)>,
    operands_ready: bool,
) -> Result<bool, StaticSimulationError> {
    let CircuitScalarNode::Apply {
        opcode, operands, ..
    } = node
    else {
        return Ok(false);
    };
    if opcode != "classical.select" {
        return Ok(false);
    }
    let [condition, when_true, when_false] = operands.as_ref() else {
        return Err(StaticSimulationError::Malformed(
            "lazy select scalar ABI".to_owned(),
        ));
    };
    let condition = scalar_operand_index(*condition, index)?;
    if !operands_ready {
        pending.push((index, true));
        if cache[condition].is_none() {
            pending.push((condition, false));
        }
        return Ok(true);
    }
    let selected = if cache[condition]
        .as_ref()
        .ok_or_else(|| StaticSimulationError::Malformed("unresolved select condition".to_owned()))?
        .truth()?
    {
        *when_true
    } else {
        *when_false
    };
    let selected = scalar_operand_index(selected, index)?;
    if cache[selected].is_none() {
        pending.push((index, true));
        pending.push((selected, false));
        return Ok(true);
    }
    let value = cache[selected].clone();
    cache[index] = value;
    Ok(true)
}

fn scalar_operand_index(
    identifier: CircuitScalarId,
    parent: usize,
) -> Result<usize, StaticSimulationError> {
    let operand = usize::try_from(identifier.index())
        .map_err(|_| StaticSimulationError::Malformed("scalar index".to_owned()))?;
    if operand >= parent {
        return Err(StaticSimulationError::Malformed(
            "non-topological scalar operand".to_owned(),
        ));
    }
    Ok(operand)
}

fn parameter_scalar(
    name: &str,
    path: &[CircuitParameterPath],
    scalar_type: CircuitScalarType,
    bindings: &[CircuitLaunchBinding],
) -> Result<Scalar, StaticSimulationError> {
    let mut value = bindings
        .iter()
        .find(|binding| binding.name() == name)
        .map(CircuitLaunchBinding::value)
        .ok_or_else(|| StaticSimulationError::Missing(format!("launch parameter {name:?}")))?;
    for component in path {
        value = match (component, value) {
            (CircuitParameterPath::Index(index), CircuitLaunchValue::Aggregate(values)) => {
                usize::try_from(*index)
                    .ok()
                    .and_then(|index| values.get(index))
            }
            (CircuitParameterPath::Field(field), CircuitLaunchValue::Record(values)) => values
                .iter()
                .find(|(name, _)| name == field)
                .map(|(_, value)| value),
            _ => None,
        }
        .ok_or_else(|| {
            StaticSimulationError::Malformed(format!("launch parameter path for {name:?}"))
        })?;
    }
    match (scalar_type, value) {
        (CircuitScalarType::Bool, CircuitLaunchValue::Bool(value)) => Ok(Scalar::Bool(*value)),
        (CircuitScalarType::Int, CircuitLaunchValue::Int(value)) => {
            Ok(Scalar::Int(Arc::new(BigInt::from(*value))))
        }
        (CircuitScalarType::Int, CircuitLaunchValue::BigInt(value)) => {
            Ok(Scalar::Int(Arc::new(BigInt::from_signed_bytes_be(value))))
        }
        (CircuitScalarType::Real, CircuitLaunchValue::RealBits(value)) => {
            Ok(Scalar::Real(f64::from_bits(*value)))
        }
        _ => Err(StaticSimulationError::Malformed(format!(
            "launch parameter type for {name:?}"
        ))),
    }
}

fn apply_scalar(
    opcode: &str,
    result_type: CircuitScalarType,
    attributes: &[crate::CircuitScalarAttribute],
    args: &[Scalar],
) -> Result<Scalar, StaticSimulationError> {
    let op = opcode.strip_prefix("classical.").unwrap_or(opcode);
    match op {
        "neg" | "pos" | "ceil" | "log2" | "sqrt" if args.len() == 1 => {
            numeric_operation(opcode, result_type, args)
        }
        "add" | "sub" | "mul" | "div" | "truediv" | "floordiv" | "mod" | "pow" | "lshift"
        | "rshift"
            if args.len() == 2 =>
        {
            numeric_operation(
                if op == "div" {
                    "classical.truediv"
                } else {
                    opcode
                },
                result_type,
                args,
            )
        }
        "and_bits" | "or_bits" | "xor" if args.len() == 2 => {
            bit_operation(opcode, result_type, args)
        }
        "invert" if args.len() == 1 => match &args[0] {
            Scalar::Bool(value) if result_type == CircuitScalarType::Bool => {
                Ok(Scalar::Bool(!value))
            }
            _ => numeric_operation(opcode, result_type, args),
        },
        "not" if args.len() == 1 => Ok(Scalar::Bool(!args[0].truth()?)),
        "and" | "or" if !args.is_empty() => Ok(Scalar::Bool(if op == "and" {
            args.iter()
                .map(Scalar::truth)
                .try_fold(true, |result, value| value.map(|value| result && value))?
        } else {
            args.iter()
                .map(Scalar::truth)
                .try_fold(false, |result, value| value.map(|value| result || value))?
        })),
        "select" if args.len() == 3 => Ok(if args[0].truth()? {
            args[1].clone()
        } else {
            args[2].clone()
        }),
        "compare_chain" if args.len() >= 2 => {
            let operators = attributes
                .iter()
                .find(|attribute| attribute.name() == "operators")
                .and_then(|attribute| match attribute.value() {
                    CircuitScalarAttributeValue::Strings(values) => Some(values.as_ref()),
                    _ => None,
                })
                .ok_or_else(|| {
                    StaticSimulationError::Malformed("comparison operators".to_owned())
                })?;
            if operators.len() + 1 != args.len() {
                return Err(StaticSimulationError::Malformed(
                    "comparison arity".to_owned(),
                ));
            }
            let mut result = true;
            for ((left, right), operator) in args
                .windows(2)
                .map(|pair| (&pair[0], &pair[1]))
                .zip(operators)
            {
                result &= compare_scalars(operator, left, right)?;
            }
            Ok(Scalar::Bool(result))
        }
        "cast" if args.len() == 1 => scalar_cast(attributes, result_type, args[0].clone()),
        _ => Err(StaticSimulationError::Unsupported(format!(
            "scalar operation {opcode:?}"
        ))),
    }
}

fn bit_operation(
    opcode: &str,
    result_type: CircuitScalarType,
    args: &[Scalar],
) -> Result<Scalar, StaticSimulationError> {
    if let [Scalar::Bool(left), Scalar::Bool(right)] = args {
        let value = match opcode.strip_prefix("classical.").unwrap_or(opcode) {
            "and_bits" => *left && *right,
            "or_bits" => *left || *right,
            "xor" => *left != *right,
            _ => unreachable!("bit operation is selected by the caller"),
        };
        return if result_type == CircuitScalarType::Bool {
            Ok(Scalar::Bool(value))
        } else {
            Err(StaticSimulationError::Malformed(
                "Boolean bit-operation result type".to_owned(),
            ))
        };
    }
    numeric_operation(opcode, result_type, args)
}

fn scalar_cast(
    attributes: &[crate::CircuitScalarAttribute],
    result_type: CircuitScalarType,
    value: Scalar,
) -> Result<Scalar, StaticSimulationError> {
    let target = attributes
        .iter()
        .find(|attribute| attribute.name() == "target")
        .and_then(|attribute| match attribute.value() {
            CircuitScalarAttributeValue::String(value) => Some(value.as_str()),
            _ => None,
        })
        .ok_or_else(|| StaticSimulationError::Malformed("scalar cast target".to_owned()))?;
    let converted = match (target, value) {
        ("Bool", Scalar::Bool(value)) => Scalar::Bool(value),
        ("Bool", Scalar::Int(value)) => Scalar::Bool(!value.is_zero()),
        ("Bool", Scalar::Real(value)) => Scalar::Bool(value != 0.0),
        ("Int", Scalar::Bool(value)) => Scalar::Int(Arc::new(BigInt::from(value))),
        ("Int", Scalar::Int(value)) => Scalar::Int(value),
        ("Int", Scalar::Real(value)) if value.is_finite() => Scalar::Int(Arc::new(
            BigInt::from_f64(value.trunc()).ok_or_else(|| {
                StaticSimulationError::Unsupported(
                    "Real cast cannot be represented as an integer".to_owned(),
                )
            })?,
        )),
        ("Real", Scalar::Int(value)) => Scalar::Real(value.to_f64().ok_or_else(|| {
            StaticSimulationError::Unsupported(
                "Int cast cannot be represented as binary64".to_owned(),
            )
        })?),
        ("Real", Scalar::Real(value)) => Scalar::Real(value),
        _ => {
            return Err(StaticSimulationError::Unsupported(format!(
                "scalar cast to {target:?}"
            )));
        }
    };
    if converted.scalar_type() == result_type {
        Ok(converted)
    } else {
        Err(StaticSimulationError::Malformed(
            "scalar cast result type".to_owned(),
        ))
    }
}

fn numeric_operation(
    opcode: &str,
    result_type: CircuitScalarType,
    args: &[Scalar],
) -> Result<Scalar, StaticSimulationError> {
    let values = args.iter().map(scalar_value).collect::<Vec<_>>();
    let operands = values.iter().collect::<Vec<_>>();
    let value = numeric_values::evaluate_plain(opcode, &operands)
        .map_err(|error| match error {
            NumericEvaluationError::Undefined(reason) => {
                StaticSimulationError::Invalid(reason.to_owned())
            }
            NumericEvaluationError::Limit(reason) => {
                StaticSimulationError::Budget(reason.to_owned())
            }
        })?
        .ok_or_else(|| {
            StaticSimulationError::Unsupported(format!("scalar operation {opcode:?}"))
        })?;
    scalar_from_value(result_type, value)
}

fn scalar_value(value: &Scalar) -> StaticValue {
    match value {
        Scalar::Bool(value) => StaticValue::Scalar(StaticScalar::Bool(*value)),
        Scalar::Int(value) => numeric_values::integer_static(value),
        Scalar::Real(value) => StaticValue::Scalar(StaticScalar::RealBits(value.to_bits())),
    }
}

fn collect_runtime_scalar_values<'a>(
    value: &'a StaticValue,
    output: &mut Vec<&'a StaticValue>,
) -> Result<(), StaticSimulationError> {
    match value {
        StaticValue::Scalar(_) | StaticValue::BigInt(_) => output.push(value),
        StaticValue::Aggregate(values) => {
            for value in values.iter() {
                collect_runtime_scalar_values(
                    value.as_ref().ok_or_else(|| {
                        StaticSimulationError::Malformed(
                            "incomplete structured call result".to_owned(),
                        )
                    })?,
                    output,
                )?;
            }
        }
        StaticValue::Record(fields) => {
            for (_, value) in fields.iter() {
                collect_runtime_scalar_values(
                    value.as_ref().ok_or_else(|| {
                        StaticSimulationError::Malformed(
                            "incomplete structured call result".to_owned(),
                        )
                    })?,
                    output,
                )?;
            }
        }
        StaticValue::Unit
        | StaticValue::ComplexBits(_, _)
        | StaticValue::String(_)
        | StaticValue::Bytes(_)
        | StaticValue::Mapping(_) => {}
    }
    Ok(())
}

fn runtime_scalar_from_value(
    scalar_type: CircuitScalarType,
    value: StaticValue,
) -> Result<Scalar, StaticSimulationError> {
    if let (CircuitScalarType::Bool, StaticValue::Scalar(StaticScalar::Bool(value))) =
        (scalar_type, &value)
    {
        return Ok(Scalar::Bool(*value));
    }
    scalar_from_value(scalar_type, value)
}

fn scalar_from_value(
    result_type: CircuitScalarType,
    value: StaticValue,
) -> Result<Scalar, StaticSimulationError> {
    match (result_type, value) {
        (CircuitScalarType::Int, StaticValue::Scalar(StaticScalar::Int(value))) => {
            Ok(Scalar::Int(Arc::new(BigInt::from(value))))
        }
        (CircuitScalarType::Int, StaticValue::BigInt(value)) => {
            Ok(Scalar::Int(Arc::new(BigInt::from_signed_bytes_be(&value))))
        }
        (CircuitScalarType::Real, StaticValue::Scalar(StaticScalar::RealBits(value))) => {
            Ok(Scalar::Real(f64::from_bits(value)))
        }
        _ => Err(StaticSimulationError::Unsupported(
            "scalar result exceeds the static circuit domain".to_owned(),
        )),
    }
}

fn compare_scalars(
    operator: &str,
    left: &Scalar,
    right: &Scalar,
) -> Result<bool, StaticSimulationError> {
    if let (Scalar::Bool(left), Scalar::Bool(right)) = (left, right) {
        return match operator {
            "eq" => Ok(left == right),
            "ne" => Ok(left != right),
            _ => Err(StaticSimulationError::Unsupported(format!(
                "Boolean comparison {operator:?}"
            ))),
        };
    }
    let left = scalar_value(left);
    let right = scalar_value(right);
    let equality = || numeric_values::equal(&left, &right);
    let order = || numeric_values::order(&left, &right);
    match operator {
        "eq" => equality(),
        "ne" => equality().map(|equal| !equal),
        "lt" => order().map(std::cmp::Ordering::is_lt),
        "le" => order().map(std::cmp::Ordering::is_le),
        "gt" => order().map(std::cmp::Ordering::is_gt),
        "ge" => order().map(std::cmp::Ordering::is_ge),
        _ => None,
    }
    .ok_or_else(|| StaticSimulationError::Unsupported(format!("comparison {operator:?}")))
}

fn qubit(qubits: &[usize], index: usize) -> Result<usize, StaticSimulationError> {
    qubits
        .get(index)
        .copied()
        .ok_or_else(|| StaticSimulationError::Malformed("instruction qubit ABI".to_owned()))
}
fn two_qubits(qubits: &[usize]) -> Result<[usize; 2], StaticSimulationError> {
    Ok([qubit(qubits, 0)?, qubit(qubits, 1)?])
}
fn with_control(controls: &[usize], control: usize) -> Result<Vec<usize>, StaticSimulationError> {
    with_controls(controls, &[control])
}
fn with_controls(controls: &[usize], added: &[usize]) -> Result<Vec<usize>, StaticSimulationError> {
    let mut result = controls.to_vec();
    result.extend_from_slice(added);
    result.sort_unstable();
    if result.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(StaticSimulationError::Malformed(
            "aliased controls".to_owned(),
        ));
    }
    Ok(result)
}
fn matches_controls(basis: usize, controls: &[usize]) -> bool {
    controls.iter().all(|wire| (basis >> wire) & 1 == 1)
}

fn hadamard() -> [[Complex64; 2]; 2] {
    let scale = std::f64::consts::FRAC_1_SQRT_2;
    [
        [Complex64::new(scale, 0.0), Complex64::new(scale, 0.0)],
        [Complex64::new(scale, 0.0), Complex64::new(-scale, 0.0)],
    ]
}
fn pauli_x() -> [[Complex64; 2]; 2] {
    [
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
    ]
}
fn pauli_y() -> [[Complex64; 2]; 2] {
    [
        [Complex64::new(0.0, 0.0), Complex64::new(0.0, -1.0)],
        [Complex64::new(0.0, 1.0), Complex64::new(0.0, 0.0)],
    ]
}
fn rotation_x(angle: f64) -> [[Complex64; 2]; 2] {
    let (sine, cosine) = (angle / 2.0).sin_cos();
    [
        [Complex64::new(cosine, 0.0), Complex64::new(0.0, -sine)],
        [Complex64::new(0.0, -sine), Complex64::new(cosine, 0.0)],
    ]
}
fn rotation_y(angle: f64) -> [[Complex64; 2]; 2] {
    let (sine, cosine) = (angle / 2.0).sin_cos();
    [
        [Complex64::new(cosine, 0.0), Complex64::new(-sine, 0.0)],
        [Complex64::new(sine, 0.0), Complex64::new(cosine, 0.0)],
    ]
}
fn rotation_z(angle: f64) -> [[Complex64; 2]; 2] {
    [
        [
            Complex64::from_polar(1.0, -angle / 2.0),
            Complex64::new(0.0, 0.0),
        ],
        [
            Complex64::new(0.0, 0.0),
            Complex64::from_polar(1.0, angle / 2.0),
        ],
    ]
}

struct SplitMix64 {
    state: u64,
}
impl SplitMix64 {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }
    fn next_f64(&mut self) -> f64 {
        const TWO_POW_53: f64 = 9_007_199_254_740_992.0;
        (self.next_u64() >> 11)
            .to_f64()
            .expect("a 53-bit integer is exactly representable as binary64")
            / TWO_POW_53
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StaticBlock;

    fn instruction(
        opcode: &str,
        qubits: &[u64],
        parameters: Vec<CircuitParameter>,
    ) -> StaticInstruction {
        StaticInstruction::test_new(
            opcode,
            qubits.to_vec(),
            parameters,
            Vec::new(),
            None,
            Vec::new(),
        )
    }

    fn angle_instruction(opcode: &str, qubits: &[u64], controls: &[u64]) -> StaticInstruction {
        StaticInstruction::test_new(
            opcode,
            qubits.to_vec(),
            vec![CircuitParameter::Scalar(CircuitScalarId::test_new(0))],
            controls.to_vec(),
            None,
            vec![],
        )
    }

    fn assert_basis_state(
        width: u64,
        mut instructions: Vec<StaticInstruction>,
        scalars: Vec<CircuitScalarNode>,
        expected: &[u8],
    ) {
        for wire in 0..width {
            instructions.push(instruction(
                "measure",
                &[wire],
                vec![CircuitParameter::MeasurementIndex(wire)],
            ));
        }
        let circuit =
            StaticCircuit::test_new(width, width, instructions, scalars, (0..width).collect());
        let counts = sample_static_circuit(&circuit, &[], 4, 11).expect("basis-state samples");
        assert_eq!(counts.entries(), &[(Box::from(expected), 4)]);
    }

    #[test]
    fn samples_bell_correlations_in_source_measurement_order() {
        let circuit = StaticCircuit::test_new(
            2,
            2,
            vec![
                instruction("h", &[0], vec![]),
                instruction("cx", &[0, 1], vec![]),
                instruction("measure", &[1], vec![CircuitParameter::MeasurementIndex(0)]),
                instruction("measure", &[0], vec![CircuitParameter::MeasurementIndex(1)]),
            ],
            vec![],
            vec![0, 1],
        );
        let counts = sample_static_circuit(&circuit, &[], 256, 7).expect("Bell circuit samples");
        assert_eq!(
            counts.entries().iter().map(|(_, count)| count).sum::<u64>(),
            256
        );
        assert!(
            counts
                .entries()
                .iter()
                .all(|(bits, _)| bits.as_ref() == [0, 0] || bits.as_ref() == [1, 1])
        );
    }

    #[test]
    fn samples_measurement_derived_scalar_results() {
        use crate::StaticCircuitResultValue;

        let circuit = StaticCircuit::test_new_with_result(
            1,
            1,
            vec![
                instruction("x", &[0], vec![]),
                instruction("measure", &[0], vec![CircuitParameter::MeasurementIndex(0)]),
            ],
            vec![
                CircuitScalarNode::Measurement(0),
                CircuitScalarNode::Apply {
                    opcode: "classical.not".to_owned(),
                    result_type: CircuitScalarType::Bool,
                    attributes: Box::new([]),
                    operands: Box::new([CircuitScalarId::test_new(0)]),
                },
            ],
            StaticCircuitValue::Aggregate(
                vec![
                    StaticCircuitValue::Scalar(CircuitScalarId::test_new(1)),
                    StaticCircuitValue::Scalar(CircuitScalarId::test_new(0)),
                ]
                .into_boxed_slice(),
            ),
        );

        assert_eq!(
            circuit.result_value(),
            Some(StaticCircuitResultValue::Aggregate(
                vec![
                    StaticCircuitResultValue::Scalar(CircuitScalarId::test_new(1)),
                    StaticCircuitResultValue::Scalar(CircuitScalarId::test_new(0)),
                ]
                .into_boxed_slice(),
            ))
        );

        let counts = sample_static_circuit(&circuit, &[], 4, 17).expect("derived samples");

        assert_eq!(counts.entries(), &[(Box::from([0, 1]), 4)]);
    }

    #[test]
    fn runtime_return_block_overrides_the_entry_continuation_result() {
        let circuit = StaticCircuit::test_new_with_result(
            1,
            1,
            vec![
                instruction("x", &[0], vec![]),
                instruction("measure", &[0], vec![CircuitParameter::MeasurementIndex(0)]),
                StaticInstruction::test_new(
                    "if",
                    vec![],
                    vec![],
                    vec![],
                    Some(CircuitScalarId::test_new(0)),
                    vec![
                        StaticBlock::test_returning(
                            vec![],
                            StaticCircuitValue::Scalar(CircuitScalarId::test_new(1)),
                        ),
                        StaticBlock::test_new(vec![]),
                    ],
                ),
            ],
            vec![
                CircuitScalarNode::Measurement(0),
                CircuitScalarNode::Bool(false),
            ],
            StaticCircuitValue::Scalar(CircuitScalarId::test_new(0)),
        );

        let counts = sample_static_circuit(&circuit, &[], 4, 3).expect("runtime return samples");

        assert_eq!(counts.entries(), &[(Box::from([0]), 4)]);
    }

    #[test]
    fn owned_call_return_resumes_without_a_scalar_result_slot() {
        let circuit = StaticCircuit::test_new(
            1,
            1,
            vec![
                StaticInstruction::test_new(
                    "call",
                    vec![],
                    vec![],
                    vec![],
                    None,
                    vec![StaticBlock::test_returning(
                        vec![],
                        StaticCircuitValue::Wires(vec![0].into_boxed_slice()),
                    )],
                ),
                instruction("x", &[0], vec![]),
                instruction("measure", &[0], vec![CircuitParameter::MeasurementIndex(0)]),
            ],
            vec![],
            vec![0],
        );

        let counts = sample_static_circuit(&circuit, &[], 4, 3).expect("owned return samples");

        assert_eq!(counts.entries(), &[(Box::from([1]), 4)]);
    }

    #[test]
    fn evaluates_quantum_effects_before_retained_record_results() {
        let circuit = StaticCircuit::test_new_with_result(
            1,
            1,
            vec![
                instruction("x", &[0], vec![]),
                instruction("measure", &[0], vec![CircuitParameter::MeasurementIndex(0)]),
            ],
            vec![
                CircuitScalarNode::Measurement(0),
                CircuitScalarNode::Int(5),
                CircuitScalarNode::Int(2),
                CircuitScalarNode::Apply {
                    opcode: "classical.add".to_owned(),
                    result_type: CircuitScalarType::Int,
                    attributes: Box::new([]),
                    operands: Box::new([
                        CircuitScalarId::test_new(1),
                        CircuitScalarId::test_new(2),
                    ]),
                },
            ],
            StaticCircuitValue::Record(
                vec![
                    (
                        "total".to_owned(),
                        StaticCircuitValue::Scalar(CircuitScalarId::test_new(3)),
                    ),
                    (
                        "accepted".to_owned(),
                        StaticCircuitValue::Scalar(CircuitScalarId::test_new(0)),
                    ),
                ]
                .into_boxed_slice(),
            ),
        );

        let encoded = evaluate_static_circuit(&circuit, &[], 7).expect("retained result");
        let literal = crate::wire::LiteralProto::decode(encoded.as_slice()).expect("literal");
        let crate::wire::literal_proto::Value::MappingValue(mapping) =
            literal.value.expect("result value")
        else {
            panic!("retained record must use the portable mapping encoding");
        };
        assert_eq!(
            mapping
                .entries
                .iter()
                .map(|entry| entry.key.as_str())
                .collect::<Vec<_>>(),
            ["accepted", "total"]
        );
        assert_eq!(
            mapping.entries[0]
                .value
                .as_ref()
                .and_then(|value| value.value.as_ref()),
            Some(&crate::wire::literal_proto::Value::BoolValue(true))
        );
        assert_eq!(
            mapping.entries[1]
                .value
                .as_ref()
                .and_then(|value| value.value.as_ref()),
            Some(&crate::wire::literal_proto::Value::IntValue(7))
        );
    }

    #[test]
    fn retains_exact_portable_values_after_quantum_effects() {
        let large_integer =
            StaticValue::BigInt((BigInt::from(1_u8) << 4096_u32).to_signed_bytes_be().into());
        let portable_values = vec![
            large_integer,
            StaticValue::ComplexBits(11.0_f64.to_bits(), 2.0_f64.to_bits()),
            StaticValue::String(Arc::from("native")),
            StaticValue::Bytes(Arc::from(b"q2".as_slice())),
        ];
        let expected = StaticValue::Aggregate(
            portable_values
                .iter()
                .cloned()
                .map(Some)
                .collect::<Vec<_>>()
                .into(),
        );
        let circuit = StaticCircuit::test_new_with_result(
            1,
            0,
            vec![instruction("x", &[0], vec![])],
            vec![],
            StaticCircuitValue::Aggregate(
                portable_values
                    .into_iter()
                    .map(StaticCircuitValue::Portable)
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            ),
        );

        let encoded = evaluate_static_circuit(&circuit, &[], 13).expect("portable result");
        let literal = crate::wire::LiteralProto::decode(encoded.as_slice()).expect("literal");

        assert_eq!(
            crate::static_values::value_literal(&literal),
            Some(expected)
        );
    }

    #[test]
    fn evaluates_runtime_angles_and_coherent_controls() {
        let scalar = CircuitScalarNode::Parameter {
            name: "theta".to_owned(),
            path: Box::new([]),
            scalar_type: CircuitScalarType::Real,
        };
        let controlled_rotation = StaticInstruction::test_new(
            "rx",
            vec![1],
            vec![CircuitParameter::Scalar(CircuitScalarId::test_new(0))],
            vec![0],
            None,
            vec![],
        );
        let circuit = StaticCircuit::test_new(
            2,
            2,
            vec![
                controlled_rotation.clone(),
                instruction("measure", &[0], vec![CircuitParameter::MeasurementIndex(0)]),
                instruction("measure", &[1], vec![CircuitParameter::MeasurementIndex(1)]),
            ],
            vec![scalar],
            vec![0, 1],
        );
        let bindings = [CircuitLaunchBinding::new(
            "theta".to_owned(),
            CircuitLaunchValue::RealBits(std::f64::consts::PI.to_bits()),
        )];
        let off =
            sample_static_circuit(&circuit, &bindings, 8, 1).expect("controlled rotation samples");
        assert_eq!(off.entries(), &[(Box::from([0, 0]), 8)]);

        let enabled = StaticCircuit::test_new(
            2,
            2,
            vec![
                instruction("x", &[0], vec![]),
                controlled_rotation,
                instruction("measure", &[0], vec![CircuitParameter::MeasurementIndex(0)]),
                instruction("measure", &[1], vec![CircuitParameter::MeasurementIndex(1)]),
            ],
            vec![CircuitScalarNode::Parameter {
                name: "theta".to_owned(),
                path: Box::new([]),
                scalar_type: CircuitScalarType::Real,
            }],
            vec![0, 1],
        );
        let on = sample_static_circuit(&enabled, &bindings, 8, 1)
            .expect("enabled controlled rotation samples");
        assert_eq!(on.entries(), &[(Box::from([1, 1]), 8)]);
    }

    #[test]
    fn preserves_phase_interference_and_exact_mixed_numeric_comparisons() {
        let circuit = StaticCircuit::test_new(
            1,
            1,
            vec![
                instruction("h", &[0], vec![]),
                instruction("z", &[0], vec![]),
                instruction("h", &[0], vec![]),
                instruction("measure", &[0], vec![CircuitParameter::MeasurementIndex(0)]),
            ],
            vec![],
            vec![0],
        );
        let counts = sample_static_circuit(&circuit, &[], 8, 3).expect("phase samples");
        assert_eq!(counts.entries(), &[(Box::from([1]), 8)]);

        let exact_integer = Scalar::Int(Arc::new(BigInt::from(9_007_199_254_740_993_i64)));
        let rounded_real = Scalar::Real(9_007_199_254_740_992.0);
        assert!(!compare_scalars("eq", &exact_integer, &rounded_real).expect("numeric equality"));
        assert!(compare_scalars("gt", &exact_integer, &rounded_real).expect("numeric ordering"));
    }

    #[test]
    fn executes_the_closed_unitary_gate_set() {
        let pi = || vec![CircuitScalarNode::RealBits(std::f64::consts::PI.to_bits())];
        assert_basis_state(
            1,
            vec![
                instruction("h", &[0], vec![]),
                instruction("h", &[0], vec![]),
                instruction("x", &[0], vec![]),
                instruction("x", &[0], vec![]),
                instruction("y", &[0], vec![]),
                instruction("y", &[0], vec![]),
                instruction("s", &[0], vec![]),
                instruction("sdg", &[0], vec![]),
                instruction("t", &[0], vec![]),
                instruction("tdg", &[0], vec![]),
            ],
            vec![],
            &[0],
        );
        for opcode in ["p", "rz"] {
            assert_basis_state(
                1,
                vec![
                    instruction("h", &[0], vec![]),
                    angle_instruction(opcode, &[0], &[]),
                    instruction("h", &[0], vec![]),
                ],
                pi(),
                &[1],
            );
        }
        for opcode in ["rx", "ry"] {
            assert_basis_state(1, vec![angle_instruction(opcode, &[0], &[])], pi(), &[1]);
        }
        for opcode in ["cz", "cp"] {
            let phase = if opcode == "cp" {
                angle_instruction(opcode, &[0, 1], &[])
            } else {
                instruction(opcode, &[0, 1], vec![])
            };
            assert_basis_state(
                2,
                vec![
                    instruction("x", &[0], vec![]),
                    instruction("h", &[1], vec![]),
                    phase,
                    instruction("h", &[1], vec![]),
                ],
                if opcode == "cp" { pi() } else { vec![] },
                &[1, 1],
            );
        }
        assert_basis_state(
            3,
            vec![
                instruction("x", &[0], vec![]),
                instruction("x", &[1], vec![]),
                instruction("ccx", &[0, 1, 2], vec![]),
            ],
            vec![],
            &[1, 1, 1],
        );
        assert_basis_state(
            2,
            vec![
                instruction("x", &[0], vec![]),
                instruction("swap", &[0, 1], vec![]),
            ],
            vec![],
            &[0, 1],
        );
        assert_basis_state(
            2,
            vec![
                instruction("h", &[0], vec![]),
                angle_instruction("rzz", &[0, 1], &[]),
                instruction("h", &[0], vec![]),
            ],
            pi(),
            &[1, 0],
        );
        assert_basis_state(
            1,
            vec![
                instruction("h", &[0], vec![]),
                angle_instruction("global_phase", &[], &[0]),
                instruction("h", &[0], vec![]),
            ],
            pi(),
            &[1],
        );
    }

    #[test]
    fn evaluates_deep_scalar_dags_without_recursion() {
        let mut scalars = vec![CircuitScalarNode::RealBits(0.0f64.to_bits())];
        for index in 0..20_000 {
            scalars.push(CircuitScalarNode::Apply {
                opcode: "classical.pos".to_owned(),
                result_type: CircuitScalarType::Real,
                attributes: Box::new([]),
                operands: Box::new([CircuitScalarId::test_new(
                    u32::try_from(index).expect("test scalar index fits u32"),
                )]),
            });
        }
        let root = CircuitScalarId::test_new(
            u32::try_from(scalars.len() - 1).expect("test scalar index fits u32"),
        );
        let circuit = StaticCircuit::test_new(
            1,
            1,
            vec![
                StaticInstruction::test_new(
                    "p",
                    vec![0],
                    vec![CircuitParameter::Scalar(root)],
                    vec![],
                    None,
                    vec![],
                ),
                instruction("measure", &[0], vec![CircuitParameter::MeasurementIndex(0)]),
            ],
            scalars,
            vec![0],
        );
        let counts = sample_static_circuit(&circuit, &[], 1, 0).expect("deep scalar samples");
        assert_eq!(counts.entries(), &[(Box::from([0]), 1)]);
    }

    #[test]
    fn measurement_normalizes_only_bounded_statevector_drift() {
        let mut random = SplitMix64::new(0);
        let mut bounded = ShotState {
            amplitudes: vec![
                Complex64::new(0.0, 0.0),
                Complex64::new((1.0 + MAX_PROBABILITY_DRIFT / 2.0).sqrt(), 0.0),
            ],
            measurements: vec![],
            call_values: vec![],
            remaining_operation_work: MAX_SIMULATION_WORK,
            width: 1,
            random: &mut random,
        };
        assert!(
            bounded
                .measure(0)
                .expect("bounded norm drift is normalized")
        );
        let normalized = bounded
            .amplitudes
            .iter()
            .map(Complex64::norm_sqr)
            .sum::<f64>();
        assert!((normalized - 1.0).abs() < MIN_PROBABILITY);

        let mut random = SplitMix64::new(0);
        let mut malformed = ShotState {
            amplitudes: vec![
                Complex64::new(0.0, 0.0),
                Complex64::new((1.0 + MAX_PROBABILITY_DRIFT * 2.0).sqrt(), 0.0),
            ],
            measurements: vec![],
            call_values: vec![],
            remaining_operation_work: MAX_SIMULATION_WORK,
            width: 1,
            random: &mut random,
        };
        assert!(matches!(
            malformed.measure(0),
            Err(StaticSimulationError::Malformed(message))
                if message.contains("statevector norm")
        ));
    }

    #[test]
    fn executes_measurement_conditioned_regions_and_reset() {
        let condition = CircuitScalarNode::Measurement(0);
        let branch = StaticBlock::test_new(vec![instruction("x", &[1], vec![])]);
        let circuit = StaticCircuit::test_new(
            2,
            2,
            vec![
                instruction("x", &[0], vec![]),
                instruction("measure", &[0], vec![CircuitParameter::MeasurementIndex(0)]),
                StaticInstruction::test_new(
                    "if",
                    vec![],
                    vec![],
                    vec![],
                    Some(CircuitScalarId::test_new(0)),
                    vec![branch, StaticBlock::test_new(vec![])],
                ),
                instruction("reset", &[0], vec![]),
                instruction("measure", &[1], vec![CircuitParameter::MeasurementIndex(1)]),
            ],
            vec![condition],
            vec![0, 1],
        );
        let counts = sample_static_circuit(&circuit, &[], 4, 2).expect("feedback samples");
        assert_eq!(counts.entries(), &[(Box::from([1, 1]), 4)]);
    }

    #[test]
    fn lazily_selects_branch_local_measurement_results() {
        let circuit = StaticCircuit::test_new_with_result(
            3,
            3,
            vec![
                instruction("h", &[0], vec![]),
                instruction("measure", &[0], vec![CircuitParameter::MeasurementIndex(0)]),
                StaticInstruction::test_new(
                    "if",
                    vec![],
                    vec![],
                    vec![],
                    Some(CircuitScalarId::test_new(0)),
                    vec![
                        StaticBlock::test_new(vec![
                            instruction("x", &[1], vec![]),
                            instruction(
                                "measure",
                                &[1],
                                vec![CircuitParameter::MeasurementIndex(1)],
                            ),
                        ]),
                        StaticBlock::test_new(vec![instruction(
                            "measure",
                            &[2],
                            vec![CircuitParameter::MeasurementIndex(2)],
                        )]),
                    ],
                ),
            ],
            vec![
                CircuitScalarNode::Measurement(0),
                CircuitScalarNode::Measurement(1),
                CircuitScalarNode::Measurement(2),
                CircuitScalarNode::Apply {
                    opcode: "classical.select".to_owned(),
                    result_type: CircuitScalarType::Bool,
                    attributes: Box::new([]),
                    operands: Box::new([
                        CircuitScalarId::test_new(0),
                        CircuitScalarId::test_new(1),
                        CircuitScalarId::test_new(2),
                    ]),
                },
            ],
            StaticCircuitValue::Aggregate(
                vec![
                    StaticCircuitValue::Scalar(CircuitScalarId::test_new(0)),
                    StaticCircuitValue::Scalar(CircuitScalarId::test_new(3)),
                ]
                .into_boxed_slice(),
            ),
        );

        let counts = sample_static_circuit(&circuit, &[], 128, 7).expect("lazy branch samples");

        assert!(
            counts
                .entries()
                .iter()
                .all(|(bits, _)| { matches!(bits.as_ref(), [0, 0] | [1, 1]) })
        );
    }

    #[test]
    fn rejects_unbounded_statevectors_and_unknown_instructions() {
        let duplicate_bindings = [
            CircuitLaunchBinding::new("theta".to_owned(), CircuitLaunchValue::Int(1)),
            CircuitLaunchBinding::new("theta".to_owned(), CircuitLaunchValue::Int(2)),
        ];
        let empty = StaticCircuit::test_new(0, 0, vec![], vec![], vec![]);
        assert!(matches!(
            sample_static_circuit(&empty, &duplicate_bindings, 1, 0),
            Err(StaticSimulationError::Malformed(_))
        ));
        let too_wide = StaticCircuit::test_new(21, 0, vec![], vec![], vec![]);
        assert!(matches!(
            sample_static_circuit(&too_wide, &[], 1, 0),
            Err(StaticSimulationError::Budget(_))
        ));
        let too_much_work = StaticCircuit::test_new(
            20,
            1,
            vec![instruction(
                "measure",
                &[0],
                vec![CircuitParameter::MeasurementIndex(0)],
            )],
            vec![],
            vec![0],
        );
        assert!(matches!(
            sample_static_circuit(&too_much_work, &[], 64, 0),
            Err(StaticSimulationError::Budget(_))
        ));
        let unknown = StaticCircuit::test_new(
            1,
            0,
            vec![instruction("mystery", &[0], vec![])],
            vec![],
            vec![],
        );
        assert!(matches!(
            sample_static_circuit(&unknown, &[], 1, 0),
            Err(StaticSimulationError::Unsupported(_))
        ));
    }
}
