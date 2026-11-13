//! Symbolic widths, allocation counts, and peak liveness.

use crate::operations::defers_gate_families;
use crate::resource_rules::{Domain as ResourceDomain, TransformMode};
use crate::resources::analysis::{
    function_value_regions, function_value_roots, loop_carry_invariants, while_carry_invariants,
};
use crate::types::{CoherentLeafWidth, Dimension, PolynomialDimension, SemanticType, TypeId};
use crate::wire;
use std::collections::HashMap;

use super::{
    Arc, ExprId, ExpressionArena, ResourceExpressionNode, ScalarFact, SymbolicAnalyzer,
    SymbolicCount, SymbolicDimensionBindings, SymbolicFunctionContext, SymbolicResourceError,
    SymbolicTimingFootprint, SymbolicTimingRoot, SymbolicTimingSelectors, SymbolicTimingStep,
    ValueFact, attribute, bind_exact_results, bind_region, bind_results, bind_unpack, bind_values,
    first_operand, int_fact, is_coherent_external, merge_optional_symbolic, scalar_type,
    span_identity, string_attribute, symbolic_depth_families, symbolic_dimension_bindings,
    symbolic_timing_selectors, symbolic_unit_durations,
};

impl SymbolicAnalyzer<'_> {
    pub(super) fn input_width(
        &mut self,
        function: &wire::FunctionProto,
        dimensions: &SymbolicDimensionBindings,
    ) -> Result<Option<SymbolicCount>, SymbolicResourceError> {
        let Some(types) = self.operations.value_types(&function.symbol) else {
            return Ok(None);
        };
        let mut widths = Vec::new();
        for value in function
            .parameters
            .iter()
            .filter_map(|parameter| parameter.value.as_ref().filter(|value| value.has_region))
        {
            let Some(type_id) = types.get(&value.id).copied() else {
                return Ok(None);
            };
            let Some(width) = self.coherent_width(type_id, dimensions)? else {
                return Ok(None);
            };
            widths.push(width);
        }
        Ok(Some(SymbolicCount::exact_value(self.arena.add(widths)?)))
    }

    pub(super) fn peak_function(
        &mut self,
        function: &wire::FunctionProto,
        arguments: &[Option<ValueFact>],
        dimension_arguments: &HashMap<Dimension, ExprId>,
    ) -> Result<Option<SymbolicCount>, SymbolicResourceError> {
        if arguments.len() != function.parameters.len() {
            return Ok(None);
        }
        let dimensions = symbolic_dimension_bindings(function, arguments, dimension_arguments);
        if function.external {
            if let Some(rule) = self.external_rule_count(
                function,
                arguments,
                &dimensions,
                TransformMode::Direct,
                "qamomile.logical.peak_live_qubits",
            )? {
                return Ok(Some(rule));
            }
            let Some(input) = self.input_width(function, &dimensions)? else {
                return Ok(None);
            };
            if is_coherent_external(function) {
                return Ok(Some(input));
            }
            return Ok(Some(SymbolicCount::unbounded(
                &input,
                &format!(
                    "external callable {} has no peak-width rule",
                    function.symbol
                ),
            )));
        }
        let mut environment = function
            .parameters
            .iter()
            .zip(arguments)
            .map(|(parameter, value)| {
                parameter
                    .value
                    .as_ref()
                    .map(|definition| (definition.id, value.clone()))
                    .ok_or_else(|| SymbolicResourceError::Missing("parameter value".to_owned()))
            })
            .collect::<Result<HashMap<_, _>, _>>()?;
        let Some(types) = self.operations.value_types(&function.symbol) else {
            return Ok(None);
        };
        let mut active = HashMap::new();
        for parameter in &function.parameters {
            let Some(value) = parameter.value.as_ref().filter(|value| value.has_region) else {
                continue;
            };
            let (Some(region), Some(type_id)) = (value.region.as_ref(), types.get(&value.id))
            else {
                return Ok(None);
            };
            let Some(width) = self.coherent_width(*type_id, &dimensions)? else {
                return Ok(None);
            };
            active.insert(region.root, width);
        }
        let body = function.body.as_ref().ok_or_else(|| {
            SymbolicResourceError::Missing(format!("body for {:?}", function.symbol))
        })?;
        let roots = function_value_roots(function);
        let peak = self.peak_block(
            body,
            &mut environment,
            &mut active,
            &roots,
            SymbolicFunctionContext {
                owner: &function.symbol,
                dimensions: &dimensions,
                mode: TransformMode::Direct,
                controls: Some(0),
            },
        )?;
        Ok(peak
            .filter(|value| {
                !self
                    .arena
                    .depends_on_symbol_origin(*value, "loop_iteration")
            })
            .map(SymbolicCount::exact_value))
    }

    pub(super) fn peak_block(
        &mut self,
        block: &wire::BlockProto,
        environment: &mut HashMap<u64, Option<ValueFact>>,
        active: &mut HashMap<u64, ExprId>,
        roots: &HashMap<u64, u64>,
        context: SymbolicFunctionContext<'_>,
    ) -> Result<Option<ExprId>, SymbolicResourceError> {
        self.charge_block(block);
        let mut peak = self.active_width(active)?;
        for operation in &block.operations {
            if operation.opcode == "qcore.call" {
                let Some(call_peak) = self.peak_call(operation, environment, active, context)?
                else {
                    return Ok(None);
                };
                peak = self.arena.extremum(true, [peak, call_peak])?;
            }
            let Some(region_peak) =
                self.peak_regions(operation, environment, active, roots, context)?
            else {
                return Ok(None);
            };
            peak = self.arena.extremum(true, [peak, region_peak])?;
            if !self.advance_peak_liveness(operation, active, roots, context)? {
                return Ok(None);
            }
            let current = self.active_width(active)?;
            peak = self.arena.extremum(true, [peak, current])?;
            self.bind_peak_results(operation, environment, context)?;
        }
        Ok(Some(peak))
    }

    pub(super) fn peak_call(
        &mut self,
        operation: &wire::OperationProto,
        environment: &HashMap<u64, Option<ValueFact>>,
        active: &HashMap<u64, ExprId>,
        context: SymbolicFunctionContext<'_>,
    ) -> Result<Option<ExprId>, SymbolicResourceError> {
        let callee_name = string_attribute(operation, "callee")
            .ok_or_else(|| SymbolicResourceError::Missing("callee".to_owned()))?;
        let callee =
            self.functions.get(callee_name).copied().ok_or_else(|| {
                SymbolicResourceError::Missing(format!("callable {callee_name:?}"))
            })?;
        let arguments = operation
            .operands
            .iter()
            .map(|operand| environment.get(operand).cloned().flatten())
            .collect::<Vec<_>>();
        let dimension_arguments = self.call_dimensions(operation, callee, context)?;
        let Some(callee_peak) = self.peak_function(callee, &arguments, &dimension_arguments)?
        else {
            return Ok(None);
        };
        let dimensions = symbolic_dimension_bindings(callee, &arguments, &dimension_arguments);
        let Some(input_width) = self.input_width(callee, &dimensions)? else {
            return Ok(None);
        };
        let Some(extra) = self.peak_extra(callee_peak.lower, input_width.lower)? else {
            return Ok(None);
        };
        let current = self.active_width(active)?;
        self.arena.add([current, extra]).map(Some)
    }

    pub(super) fn peak_extra(
        &mut self,
        peak: ExprId,
        input: ExprId,
    ) -> Result<Option<ExprId>, SymbolicResourceError> {
        if let (Some(peak), Some(input)) = (
            self.arena.integer_value(peak),
            self.arena.integer_value(input),
        ) {
            return self
                .arena
                .literal_u128(peak.saturating_sub(input).max(0).try_into().map_err(|_| {
                    SymbolicResourceError::Unsupported(
                        "negative peak-liveness enclosure".to_owned(),
                    )
                })?)
                .map(Some);
        }
        if peak == input {
            return self.arena.literal_u128(0).map(Some);
        }
        if self.arena.is_literal(input, "0") {
            return Ok(Some(peak));
        }
        let zero = self.arena.literal_u128(0)?;
        let negative_one = self.arena.literal_i64(-1)?;
        let negative_input = self.arena.multiply([negative_one, input])?;
        let difference = self.arena.add([peak, negative_input])?;
        self.arena
            .intern(ResourceExpressionNode::Maximum {
                domain: ResourceDomain::Natural,
                arguments: Box::new([zero, difference]),
            })
            .map(Some)
    }

    pub(super) fn peak_regions(
        &mut self,
        operation: &wire::OperationProto,
        environment: &HashMap<u64, Option<ValueFact>>,
        active: &HashMap<u64, ExprId>,
        roots: &HashMap<u64, u64>,
        context: SymbolicFunctionContext<'_>,
    ) -> Result<Option<ExprId>, SymbolicResourceError> {
        if operation.regions.is_empty() {
            return self.arena.literal_u128(0).map(Some);
        }
        if operation.opcode == "qcore.while"
            && self
                .allocated_while(operation, environment, context)?
                .is_none_or(|count| {
                    count
                        .upper
                        .is_none_or(|upper| !self.arena.is_literal(upper, "0"))
                })
        {
            return Ok(None);
        }
        let mut peaks = Vec::with_capacity(operation.regions.len());
        for (index, region) in operation.regions.iter().enumerate() {
            let block = region.blocks.first().ok_or_else(|| {
                SymbolicResourceError::Missing(format!("{} region", operation.opcode))
            })?;
            let Some(mut local) =
                self.peak_region_environment(operation, block, index, environment, context)?
            else {
                return Ok(None);
            };
            let mut nested_active = active.clone();
            let Some(nested_peak) =
                self.peak_block(block, &mut local, &mut nested_active, roots, context)?
            else {
                return Ok(None);
            };
            peaks.push(nested_peak);
        }
        self.arena.extremum(true, peaks).map(Some)
    }

    pub(super) fn peak_region_environment(
        &mut self,
        operation: &wire::OperationProto,
        block: &wire::BlockProto,
        index: usize,
        environment: &HashMap<u64, Option<ValueFact>>,
        context: SymbolicFunctionContext<'_>,
    ) -> Result<Option<HashMap<u64, Option<ValueFact>>>, SymbolicResourceError> {
        match operation.opcode.as_str() {
            "qcore.if" => bind_region(
                block,
                operation.operands.get(1..).unwrap_or_default(),
                environment,
                false,
                None,
            )
            .map(Some),
            "qcore.for" if index == 0 => {
                self.peak_loop_body_environment(operation, block, environment, context)
            }
            "qcore.for" => bind_region(
                block,
                operation.operands.get(3..).unwrap_or_default(),
                environment,
                false,
                None,
            )
            .map(Some),
            "qcore.while" => {
                let invariants = while_carry_invariants(operation);
                let values = super::invariant_values(&operation.operands, environment, &invariants);
                bind_values(block, values).map(Some)
            }
            _ => Ok(None),
        }
    }

    pub(super) fn peak_loop_body_environment(
        &mut self,
        operation: &wire::OperationProto,
        block: &wire::BlockProto,
        environment: &HashMap<u64, Option<ValueFact>>,
        context: SymbolicFunctionContext<'_>,
    ) -> Result<Option<HashMap<u64, Option<ValueFact>>>, SymbolicResourceError> {
        let [lower_id, _, stride_id, carries @ ..] = operation.operands.as_slice() else {
            return Ok(None);
        };
        let Some(lower) = int_fact(environment.get(lower_id).and_then(Option::as_ref)) else {
            return Ok(None);
        };
        let Some(stride) = int_fact(environment.get(stride_id).and_then(Option::as_ref)) else {
            return Ok(None);
        };
        let lower = lower.expression;
        let stride = stride.expression;
        let invariants = loop_carry_invariants(block, carries.len());
        let binder = self.arena.symbol(
            &format!("{}:{}", context.owner, span_identity(operation)),
            string_attribute(operation, "induction_name").unwrap_or("iteration"),
            ResourceDomain::Natural,
            "loop_iteration",
            &["0 <= iteration"],
        )?;
        let scaled = self.arena.multiply([stride, binder])?;
        let induction = self.arena.add([lower, scaled])?;
        bind_region(
            block,
            carries,
            environment,
            true,
            Some((induction, &invariants)),
        )
        .map(Some)
    }

    pub(super) fn advance_peak_liveness(
        &mut self,
        operation: &wire::OperationProto,
        active: &mut HashMap<u64, ExprId>,
        roots: &HashMap<u64, u64>,
        context: SymbolicFunctionContext<'_>,
    ) -> Result<bool, SymbolicResourceError> {
        let result_roots = operation
            .results
            .iter()
            .filter_map(|result| result.region.as_ref().map(|region| region.root))
            .collect::<Vec<_>>();
        let Some(types) = self.operations.value_types(context.owner) else {
            return Ok(false);
        };
        for result in &operation.results {
            let Some(region) = result.region.as_ref() else {
                continue;
            };
            if active.contains_key(&region.root) {
                continue;
            }
            let Some(type_id) = types.get(&result.id).copied() else {
                return Ok(false);
            };
            let Some(width) = self.coherent_width(type_id, context.dimensions)? else {
                return Ok(false);
            };
            active.insert(region.root, width);
        }
        for operand in &operation.operands {
            let Some(root) = roots.get(operand) else {
                continue;
            };
            if !result_roots.contains(root) {
                active.remove(root);
            }
        }
        Ok(true)
    }

    pub(super) fn active_width(
        &mut self,
        active: &HashMap<u64, ExprId>,
    ) -> Result<ExprId, SymbolicResourceError> {
        self.arena.add(active.values().copied())
    }

    pub(super) fn bind_peak_results(
        &mut self,
        operation: &wire::OperationProto,
        environment: &mut HashMap<u64, Option<ValueFact>>,
        context: SymbolicFunctionContext<'_>,
    ) -> Result<(), SymbolicResourceError> {
        match operation.opcode.as_str() {
            "qcore.constant" => {
                let value = attribute(operation, "value")
                    .map(|literal| self.literal_fact(literal))
                    .transpose()?
                    .flatten();
                bind_results(operation, environment, value.as_ref());
            }
            "qcore.stage_lift" | "qcore.reshape" => {
                let value = first_operand(operation, environment);
                bind_results(operation, environment, value.as_ref());
            }
            "tuple" | "array" | "qcore.tuple" => {
                let value = ValueFact::Aggregate(
                    operation
                        .operands
                        .iter()
                        .map(|operand| environment.get(operand).cloned().flatten())
                        .collect::<Vec<_>>()
                        .into_boxed_slice(),
                );
                bind_results(operation, environment, Some(&value));
            }
            "qcore.unpack" => bind_unpack(operation, environment)?,
            "qcore.call" => {
                let summary = self.analyze_call(operation, environment, context)?;
                bind_exact_results(operation, environment, &summary.values)?;
            }
            "qcore.if" => {
                let summary = self.analyze_if(operation, environment, context)?;
                bind_exact_results(operation, environment, &summary.values)?;
            }
            "qcore.for" | "qcore.while" => {
                let summary = if operation.opcode == "qcore.while" {
                    self.analyze_while(operation, environment, context)?
                } else {
                    self.analyze_for(operation, environment, context)?
                };
                bind_exact_results(operation, environment, &summary.values)?;
            }
            opcode if opcode.starts_with("classical.") => {
                let value = self.evaluate_classical(operation, environment, context)?;
                bind_results(operation, environment, value.as_ref());
            }
            _ => bind_results(operation, environment, None),
        }
        Ok(())
    }

    pub(super) fn allocated_function(
        &mut self,
        function: &wire::FunctionProto,
        arguments: &[Option<ValueFact>],
        dimension_arguments: &HashMap<Dimension, ExprId>,
    ) -> Result<Option<SymbolicCount>, SymbolicResourceError> {
        if arguments.len() != function.parameters.len() {
            return Ok(None);
        }
        let dimensions = symbolic_dimension_bindings(function, arguments, dimension_arguments);
        if function.external {
            if let Some(rule) = self.external_rule_count(
                function,
                arguments,
                &dimensions,
                TransformMode::Direct,
                "qamomile.logical.allocated_qubits",
            )? {
                return Ok(Some(rule));
            }
            if is_coherent_external(function) {
                return Ok(Some(SymbolicCount::exact_value(
                    self.arena.literal_u128(0)?,
                )));
            }
            return SymbolicCount::unresolved(
                format!(
                    "external callable {} has no allocated-width rule",
                    function.symbol
                ),
                &mut self.arena,
            )
            .map(Some);
        }
        let mut environment = function
            .parameters
            .iter()
            .zip(arguments)
            .map(|(parameter, value)| {
                parameter
                    .value
                    .as_ref()
                    .map(|definition| (definition.id, value.clone()))
                    .ok_or_else(|| SymbolicResourceError::Missing("parameter value".to_owned()))
            })
            .collect::<Result<HashMap<_, _>, _>>()?;
        let body = function.body.as_ref().ok_or_else(|| {
            SymbolicResourceError::Missing(format!("body for {:?}", function.symbol))
        })?;
        self.allocated_block(
            body,
            &mut environment,
            SymbolicFunctionContext {
                owner: &function.symbol,
                dimensions: &dimensions,
                mode: TransformMode::Direct,
                controls: Some(0),
            },
        )
    }

    pub(super) fn allocated_block(
        &mut self,
        block: &wire::BlockProto,
        environment: &mut HashMap<u64, Option<ValueFact>>,
        context: SymbolicFunctionContext<'_>,
    ) -> Result<Option<SymbolicCount>, SymbolicResourceError> {
        self.charge_block(block);
        let zero = SymbolicCount::exact_value(self.arena.literal_u128(0)?);
        let mut allocated = Some(zero);
        for operation in &block.operations {
            match operation.opcode.as_str() {
                "qcore.constant" => {
                    let value = attribute(operation, "value")
                        .map(|literal| self.literal_fact(literal))
                        .transpose()?
                        .flatten();
                    bind_results(operation, environment, value.as_ref());
                }
                "qcore.stage_lift" | "qcore.reshape" => {
                    let value = first_operand(operation, environment);
                    bind_results(operation, environment, value.as_ref());
                }
                "tuple" | "array" | "qcore.tuple" => {
                    let value = ValueFact::Aggregate(
                        operation
                            .operands
                            .iter()
                            .map(|operand| environment.get(operand).cloned().flatten())
                            .collect::<Vec<_>>()
                            .into_boxed_slice(),
                    );
                    bind_results(operation, environment, Some(&value));
                }
                "qcore.unpack" => bind_unpack(operation, environment)?,
                "qcore.call" => {
                    let contribution = self.allocated_call(operation, environment, context)?;
                    merge_optional_symbolic(
                        &mut allocated,
                        contribution.as_ref(),
                        &mut self.arena,
                    )?;
                }
                "qcore.if" => {
                    let contribution = self.allocated_if(operation, environment, context)?;
                    merge_optional_symbolic(
                        &mut allocated,
                        contribution.as_ref(),
                        &mut self.arena,
                    )?;
                }
                "qcore.for" => {
                    let contribution = self.allocated_for(operation, environment, context)?;
                    merge_optional_symbolic(
                        &mut allocated,
                        contribution.as_ref(),
                        &mut self.arena,
                    )?;
                }
                "qcore.while" => {
                    let contribution = self.allocated_while(operation, environment, context)?;
                    merge_optional_symbolic(
                        &mut allocated,
                        contribution.as_ref(),
                        &mut self.arena,
                    )?;
                }
                "quantum.alloc" | "quantum.prepare" => {
                    let contribution = self
                        .allocation_result_width(operation, context)?
                        .map(SymbolicCount::exact_value);
                    merge_optional_symbolic(
                        &mut allocated,
                        contribution.as_ref(),
                        &mut self.arena,
                    )?;
                    bind_results(operation, environment, None);
                }
                opcode if opcode.starts_with("classical.") => {
                    let value = self.evaluate_classical(operation, environment, context)?;
                    bind_results(operation, environment, value.as_ref());
                }
                _ => bind_results(operation, environment, None),
            }
        }
        Ok(allocated)
    }

    pub(super) fn allocated_call(
        &mut self,
        operation: &wire::OperationProto,
        environment: &mut HashMap<u64, Option<ValueFact>>,
        context: SymbolicFunctionContext<'_>,
    ) -> Result<Option<SymbolicCount>, SymbolicResourceError> {
        let callee_name = string_attribute(operation, "callee")
            .ok_or_else(|| SymbolicResourceError::Missing("callee".to_owned()))?;
        let callee =
            self.functions.get(callee_name).copied().ok_or_else(|| {
                SymbolicResourceError::Missing(format!("callable {callee_name:?}"))
            })?;
        let arguments = operation
            .operands
            .iter()
            .map(|operand| environment.get(operand).cloned().flatten())
            .collect::<Vec<_>>();
        let dimensions = self.call_dimensions(operation, callee, context)?;
        let allocated = self
            .allocated_function(callee, &arguments, &dimensions)
            .unwrap_or_default();
        if let Ok(summary) = self.analyze_function(callee, &arguments, &dimensions) {
            bind_exact_results(operation, environment, &summary.values)?;
        } else {
            bind_results(operation, environment, None);
        }
        Ok(allocated)
    }

    pub(super) fn allocated_if(
        &mut self,
        operation: &wire::OperationProto,
        environment: &mut HashMap<u64, Option<ValueFact>>,
        context: SymbolicFunctionContext<'_>,
    ) -> Result<Option<SymbolicCount>, SymbolicResourceError> {
        let captures = operation.operands.get(1..).unwrap_or_default();
        let zero = SymbolicCount::exact_value(self.arena.literal_u128(0)?);
        let mut allocated = Some(zero);
        for region in &operation.regions {
            let branch = region
                .blocks
                .first()
                .ok_or_else(|| SymbolicResourceError::Missing("qcore.if branch".to_owned()))?;
            let mut local = bind_region(branch, captures, environment, false, None)?;
            let contribution = self.allocated_block(branch, &mut local, context)?;
            merge_optional_symbolic(&mut allocated, contribution.as_ref(), &mut self.arena)?;
        }
        if let Ok(summary) = self.analyze_if(operation, environment, context) {
            bind_exact_results(operation, environment, &summary.values)?;
        } else {
            bind_results(operation, environment, None);
        }
        Ok(allocated)
    }

    pub(super) fn allocated_while(
        &mut self,
        operation: &wire::OperationProto,
        environment: &HashMap<u64, Option<ValueFact>>,
        context: SymbolicFunctionContext<'_>,
    ) -> Result<Option<SymbolicCount>, SymbolicResourceError> {
        let count = self.while_trip_count(operation, context)?;
        let binder = self.arena.symbol(
            &format!("{}:{}", context.owner, span_identity(operation)),
            "iteration",
            ResourceDomain::Natural,
            "loop_iteration",
            &["0 <= iteration"],
        )?;
        let invariants = while_carry_invariants(operation);
        let values = super::invariant_values(&operation.operands, environment, &invariants);
        let mut allocated = SymbolicCount::exact_value(self.arena.literal_u128(0)?);
        for (index, region) in operation.regions.iter().enumerate() {
            let block = &region.blocks[0];
            let mut local = bind_values(block, values.clone())?;
            let Some(cost) = self.allocated_block(block, &mut local, context)? else {
                return Ok(None);
            };
            let cost = if index < 2 {
                cost.repeat(count, binder, &mut self.arena)?
            } else {
                cost
            };
            allocated = allocated.then(&cost, &mut self.arena)?;
        }
        if allocated
            .upper
            .is_some_and(|upper| self.arena.is_literal(upper, "0"))
        {
            return Ok(Some(allocated));
        }
        // Normal-exit allocation and the final condition are path dependent.
        Ok(Some(SymbolicCount::unresolved_from(
            &allocated,
            "while allocation requires exit-sensitive liveness analysis".to_owned(),
            &mut self.arena,
        )?))
    }

    pub(super) fn allocated_for(
        &mut self,
        operation: &wire::OperationProto,
        environment: &mut HashMap<u64, Option<ValueFact>>,
        context: SymbolicFunctionContext<'_>,
    ) -> Result<Option<SymbolicCount>, SymbolicResourceError> {
        let body = operation
            .regions
            .first()
            .and_then(|region| region.blocks.first())
            .ok_or_else(|| SymbolicResourceError::Missing("qcore.for body".to_owned()))?;
        let carries = operation.operands.get(3..).unwrap_or_default();
        let invariants = loop_carry_invariants(body, carries.len());
        let arguments = std::iter::once(None)
            .chain(carries.iter().zip(&invariants).map(|(operand, invariant)| {
                invariant
                    .then(|| environment.get(operand).cloned().flatten())
                    .flatten()
            }))
            .collect();
        let mut local = bind_values(body, arguments)?;
        let mut allocated = self.allocated_block(body, &mut local, context)?;
        if let Some(otherwise) = operation
            .regions
            .get(1)
            .and_then(|region| region.blocks.first())
        {
            let values = carries
                .iter()
                .map(|operand| environment.get(operand).cloned().flatten())
                .collect();
            let mut local = bind_values(otherwise, values)?;
            let contribution = self.allocated_block(otherwise, &mut local, context)?;
            merge_optional_symbolic(&mut allocated, contribution.as_ref(), &mut self.arena)?;
        }
        if let Ok(summary) = self.analyze_for(operation, environment, context) {
            bind_exact_results(operation, environment, &summary.values)?;
        } else {
            bind_results(operation, environment, None);
        }
        Ok(allocated)
    }

    pub(super) fn allocation_result_width(
        &mut self,
        operation: &wire::OperationProto,
        context: SymbolicFunctionContext<'_>,
    ) -> Result<Option<ExprId>, SymbolicResourceError> {
        let Some(types) = self.operations.value_types(context.owner) else {
            return Ok(None);
        };
        for result in &operation.results {
            let Some(type_id) = types.get(&result.id).copied() else {
                continue;
            };
            let Some(width) = self.coherent_width(type_id, context.dimensions)? else {
                continue;
            };
            return Ok(
                (!self.arena.depends_on_symbol_origin(width, "type_dimension")).then_some(width),
            );
        }
        Ok(None)
    }

    pub(super) fn event_width(
        &mut self,
        owner: &str,
        operation: &wire::OperationProto,
        dimensions: &SymbolicDimensionBindings,
    ) -> Result<Option<ExprId>, SymbolicResourceError> {
        let types = self.operations.value_types(owner);
        let type_id = if operation.opcode == "quantum.prepare" {
            operation
                .results
                .first()
                .and_then(|result| types?.get(&result.id))
                .copied()
        } else {
            operation
                .operands
                .first()
                .and_then(|operand| types?.get(operand))
                .copied()
        };
        let Some(type_id) = type_id else {
            return Ok(None);
        };
        self.coherent_width(type_id, dimensions)
    }

    pub(super) fn mod_pow_widths(
        &mut self,
        owner: &str,
        operation: &wire::OperationProto,
        dimensions: &SymbolicDimensionBindings,
    ) -> Result<Option<(ExprId, ExprId)>, SymbolicResourceError> {
        let Some(type_id) = operation
            .operands
            .first()
            .and_then(|operand| self.operations.value_types(owner)?.get(operand).copied())
        else {
            return Ok(None);
        };
        let mut widths = Vec::new();
        let complete = self
            .types
            .visit_coherent_value_leaves(type_id, &mut |leaf| {
                let width = match leaf {
                    CoherentLeafWidth::Bit => self.arena.literal_u128(1)?,
                    CoherentLeafWidth::Word(dimension) => {
                        Self::mod_pow_dimension_width(&mut self.arena, dimension, dimensions)?
                    }
                    CoherentLeafWidth::Fixed {
                        integer_bits,
                        fractional_bits,
                    } => {
                        let integer = Self::mod_pow_dimension_width(
                            &mut self.arena,
                            integer_bits,
                            dimensions,
                        )?;
                        let fractional = Self::mod_pow_dimension_width(
                            &mut self.arena,
                            fractional_bits,
                            dimensions,
                        )?;
                        self.arena.add([integer, fractional])?
                    }
                };
                widths.push(width);
                Ok(true)
            })?;
        match widths.as_slice() {
            [target, exponent] if complete => Ok(Some((*target, *exponent))),
            _ => Ok(None),
        }
    }

    pub(super) fn mod_multi_pow_dimensions(
        &mut self,
        owner: &str,
        operation: &wire::OperationProto,
        environment: &HashMap<u64, Option<ValueFact>>,
        dimensions: &SymbolicDimensionBindings,
    ) -> Result<Option<(ExprId, ExprId, ExprId, ExprId)>, SymbolicResourceError> {
        let Some(types) = self.operations.value_types(owner) else {
            return Ok(None);
        };
        let Some(state_type) = operation
            .operands
            .first()
            .and_then(|operand| types.get(operand).copied())
        else {
            return Ok(None);
        };
        let mut widths = Vec::new();
        let complete = self
            .types
            .visit_coherent_value_leaves(state_type, &mut |leaf| {
                let width = match leaf {
                    CoherentLeafWidth::Bit => self.arena.literal_u128(1)?,
                    CoherentLeafWidth::Word(dimension) => {
                        Self::dimension_width(&mut self.arena, dimension, dimensions)?
                    }
                    CoherentLeafWidth::Fixed {
                        integer_bits,
                        fractional_bits,
                    } => {
                        let integer =
                            Self::dimension_width(&mut self.arena, integer_bits, dimensions)?;
                        let fractional =
                            Self::dimension_width(&mut self.arena, fractional_bits, dimensions)?;
                        self.arena.add([integer, fractional])?
                    }
                };
                widths.push(width);
                Ok(true)
            })?;
        let [coordinate_total, target_width] = widths.as_slice() else {
            return Ok(None);
        };
        if !complete {
            return Ok(None);
        }
        let Some(bases_type) = operation
            .operands
            .get(1)
            .and_then(|operand| types.get(operand).copied())
        else {
            return Ok(None);
        };
        let bases_type = self.types.strip_stage(bases_type);
        let Some(SemanticType::Array { shape, .. }) = self.types.get(bases_type) else {
            return Ok(None);
        };
        let [coordinate_count] = shape.as_slice() else {
            return Ok(None);
        };
        let coordinate_count =
            Self::dimension_width(&mut self.arena, coordinate_count, dimensions)?;
        let Some(base_width) = operation
            .operands
            .get(2)
            .and_then(|operand| environment.get(operand))
            .and_then(Option::as_ref)
            .and_then(|value| int_fact(Some(value)))
            .map(|fact| fact.expression)
        else {
            return Ok(None);
        };
        let base_width = self.natural_operation_dimension(base_width)?;
        let Some(coordinate_width) = operation
            .operands
            .get(4)
            .and_then(|operand| environment.get(operand))
            .and_then(Option::as_ref)
            .and_then(|value| int_fact(Some(value)))
            .map(|fact| fact.expression)
        else {
            return Ok(None);
        };
        let coordinate_width = self.natural_operation_dimension(coordinate_width)?;
        let expected_total = self.arena.multiply([coordinate_count, coordinate_width])?;
        if expected_total != *coordinate_total {
            return Ok(None);
        }
        Ok(Some((
            coordinate_count,
            coordinate_width,
            *target_width,
            base_width,
        )))
    }

    pub(super) fn natural_operation_dimension(
        &mut self,
        expression: ExprId,
    ) -> Result<ExprId, SymbolicResourceError> {
        match self.arena.node(expression) {
            Some(node) if node.domain() == ResourceDomain::Natural => Ok(expression),
            Some(ResourceExpressionNode::Symbol { scope, name, .. }) => {
                let (scope, name) = (scope.clone(), name.clone());
                let constraint = format!("{name} >= 1");
                self.arena.symbol(
                    &scope,
                    &name,
                    ResourceDomain::Natural,
                    "operation_dimension",
                    &[constraint.as_str()],
                )
            }
            _ => self.arena.absolute(expression, ResourceDomain::Natural),
        }
    }

    pub(super) fn mod_pow_dimension_width(
        arena: &mut ExpressionArena,
        dimension: &Dimension,
        bindings: &SymbolicDimensionBindings,
    ) -> Result<ExprId, SymbolicResourceError> {
        let Some(expression) = bindings.values.get(dimension).copied() else {
            return Self::dimension_width(arena, dimension, bindings);
        };
        if arena
            .node(expression)
            .is_some_and(|node| node.domain() == ResourceDomain::Natural)
        {
            Ok(expression)
        } else {
            arena.contracted(
                "qamomile.positive_dimension",
                ResourceDomain::Natural,
                [expression],
            )
        }
    }

    pub(super) fn coherent_width(
        &mut self,
        type_id: TypeId,
        dimensions: &SymbolicDimensionBindings,
    ) -> Result<Option<ExprId>, SymbolicResourceError> {
        let mut widths = Vec::new();
        let complete = self
            .types
            .visit_coherent_value_leaves(type_id, &mut |leaf| {
                let width = match leaf {
                    CoherentLeafWidth::Bit => self.arena.literal_u128(1)?,
                    CoherentLeafWidth::Word(dimension) => {
                        Self::dimension_width(&mut self.arena, dimension, dimensions)?
                    }
                    CoherentLeafWidth::Fixed {
                        integer_bits,
                        fractional_bits,
                    } => {
                        let integer =
                            Self::dimension_width(&mut self.arena, integer_bits, dimensions)?;
                        let fractional =
                            Self::dimension_width(&mut self.arena, fractional_bits, dimensions)?;
                        self.arena.add([integer, fractional])?
                    }
                };
                widths.push(width);
                Ok(true)
            })?;
        if !complete {
            return Ok(None);
        }
        self.arena.add(widths).map(Some)
    }

    pub(super) fn symbolic_quantum_timing(
        &mut self,
        operation: &wire::OperationProto,
        environment: &HashMap<u64, Option<ValueFact>>,
        effective: &str,
        context: SymbolicFunctionContext<'_>,
    ) -> Result<Option<SymbolicTimingStep>, SymbolicResourceError> {
        if defers_gate_families(effective) {
            return Ok(None);
        }
        let function = self
            .functions
            .get(context.owner)
            .copied()
            .ok_or_else(|| SymbolicResourceError::Missing("timing owner".to_owned()))?;
        let regions = function_value_regions(function);
        let value_ids = if matches!(effective, "quantum.alloc" | "quantum.prepare") {
            operation
                .results
                .iter()
                .filter(|value| value.has_region)
                .map(|value| value.id)
                .collect::<Vec<_>>()
        } else {
            operation
                .operands
                .iter()
                .filter(|operand| regions.contains_key(operand))
                .copied()
                .collect()
        };
        if value_ids.is_empty() {
            return Ok(None);
        }
        let selectors =
            match symbolic_timing_selectors(operation, effective, environment, &mut self.arena)? {
                SymbolicTimingSelectors::Whole => None,
                SymbolicTimingSelectors::Selected(selectors) => Some(selectors),
                SymbolicTimingSelectors::Unsupported => return Ok(None),
            };
        let types = self
            .operations
            .value_types(context.owner)
            .ok_or_else(|| SymbolicResourceError::Missing("timing value types".to_owned()))?;
        let mut footprints = Vec::with_capacity(value_ids.len());
        for value_id in value_ids {
            let region = regions
                .get(&value_id)
                .ok_or_else(|| SymbolicResourceError::Missing("timing region".to_owned()))?;
            let type_id = types
                .get(&value_id)
                .copied()
                .ok_or_else(|| SymbolicResourceError::Missing("timing type".to_owned()))?;
            footprints.push(SymbolicTimingFootprint {
                root: self.canonical_timing_root(SymbolicTimingRoot {
                    scope: context.owner.to_owned(),
                    root: region.root,
                    projections: region.projections.clone(),
                    instance: Vec::new(),
                }),
                selectors: selectors.clone(),
                width: self.coherent_width(type_id, context.dimensions)?,
            });
        }
        Ok(Some(SymbolicTimingStep {
            footprints: footprints.into_boxed_slice(),
            durations: symbolic_unit_durations(
                &symbolic_depth_families(effective),
                &mut self.arena,
            )?,
        }))
    }

    pub(super) fn dimension_width(
        arena: &mut ExpressionArena,
        dimension: &Dimension,
        bindings: &SymbolicDimensionBindings,
    ) -> Result<ExprId, SymbolicResourceError> {
        match dimension {
            Dimension::Fixed(value) => arena.literal_u128(u128::from(*value)),
            Dimension::Symbol { .. } if bindings.values.contains_key(dimension) => {
                let expression = bindings.values[dimension];
                match arena.node(expression).cloned() {
                    Some(node) if node.domain() == ResourceDomain::Natural => Ok(expression),
                    _ => arena.contracted(
                        "qamomile.positive_dimension",
                        ResourceDomain::Natural,
                        [expression],
                    ),
                }
            }
            Dimension::Symbol { scope, name } => {
                if let Ok(polynomial) = PolynomialDimension::parse(name) {
                    let natural = polynomial.constant >= 0
                        && polynomial
                            .terms
                            .values()
                            .all(|coefficient| *coefficient > 0);
                    let terms = polynomial_width_terms(arena, scope, &polynomial, bindings)?;
                    let sum = arena.add(terms)?;
                    if natural {
                        return Ok(sum);
                    }
                    // Signed polynomials such as `n-k` (the suffix of a symbolic
                    // split) are positive for every admitted binding because
                    // specialization rejects nonpositive widths. The contracted
                    // positive-dimension node states that fact and evaluates to
                    // the polynomial itself.
                    return arena.contracted(
                        "qamomile.positive_dimension",
                        ResourceDomain::Natural,
                        [sum],
                    );
                }
                let constraint = format!("{name} >= 1");
                arena.symbol(
                    scope,
                    name,
                    ResourceDomain::Natural,
                    "type_dimension",
                    &[constraint.as_str()],
                )
            }
        }
    }

    pub(super) fn parameter_fact(
        &mut self,
        function: &wire::FunctionProto,
        parameter: &wire::ParameterProto,
    ) -> Result<Option<ValueFact>, SymbolicResourceError> {
        let value = parameter
            .value
            .as_ref()
            .ok_or_else(|| SymbolicResourceError::Missing("parameter value".to_owned()))?;
        let Some((name, stage)) = scalar_type(value.r#type.as_ref()) else {
            return Ok(None);
        };
        match name {
            "Int" => {
                let dimension = Dimension::Symbol {
                    scope: Arc::from(function.symbol.as_str()),
                    name: Arc::from(parameter.name.as_str()),
                };
                let owns_dimension =
                    self.operations
                        .value_types(&function.symbol)
                        .is_some_and(|types| {
                            self.types
                                .symbolic_dimensions_in(types.values().copied())
                                .contains(&dimension)
                        });
                let constraint = format!("{} >= 1", parameter.name);
                let constraints = owns_dimension
                    .then_some(constraint.as_str())
                    .into_iter()
                    .collect::<Vec<_>>();
                Ok(Some(ValueFact::Scalar(ScalarFact::Int {
                    expression: self.arena.symbol(
                        &function.symbol,
                        &parameter.name,
                        if owns_dimension {
                            ResourceDomain::Natural
                        } else {
                            ResourceDomain::Integer
                        },
                        stage.unwrap_or("parameter").to_ascii_lowercase().as_str(),
                        &constraints,
                    )?,
                    exact: None,
                })))
            }
            "Bool" => Ok(Some(ValueFact::Scalar(ScalarFact::Bool(None)))),
            "Real" => Ok(Some(ValueFact::Scalar(ScalarFact::Real {
                expression: Some(self.arena.symbol(
                    &function.symbol,
                    &parameter.name,
                    ResourceDomain::Real,
                    stage.unwrap_or("parameter").to_ascii_lowercase().as_str(),
                    &[],
                )?),
                exact: None,
            }))),
            _ => Ok(None),
        }
    }

    pub(super) fn literal_fact(
        &mut self,
        literal: &wire::LiteralProto,
    ) -> Result<Option<ValueFact>, SymbolicResourceError> {
        use wire::literal_proto::Value;

        match literal.value.as_ref() {
            Some(Value::IntValue(value)) => Ok(Some(ValueFact::Scalar(ScalarFact::Int {
                expression: self.arena.literal_i64(*value)?,
                exact: Some(*value),
            }))),
            Some(Value::BigintValue(value)) => Ok(Some(ValueFact::Scalar(ScalarFact::Int {
                expression: self.arena.literal_bigint(value)?,
                exact: None,
            }))),
            Some(Value::BoolValue(value)) => {
                Ok(Some(ValueFact::Scalar(ScalarFact::Bool(Some(*value)))))
            }
            Some(Value::RealValue(value)) => Ok(Some(ValueFact::Scalar(ScalarFact::Real {
                expression: None,
                exact: Some(value.to_bits()),
            }))),
            Some(Value::TupleValue(sequence)) => Ok(Some(ValueFact::Aggregate(
                sequence
                    .values
                    .iter()
                    .map(|value| self.literal_fact(value))
                    .collect::<Result<Vec<_>, _>>()?
                    .into_boxed_slice(),
            ))),
            _ => Ok(None),
        }
    }
}

/// Build one resource term per polynomial monomial, binding known symbols.
fn polynomial_width_terms(
    arena: &mut ExpressionArena,
    scope: &Arc<str>,
    polynomial: &PolynomialDimension,
    bindings: &SymbolicDimensionBindings,
) -> Result<Vec<ExprId>, SymbolicResourceError> {
    let mut terms = Vec::with_capacity(polynomial.terms.len() + 1);
    if polynomial.constant != 0 {
        terms.push(
            arena.literal_bigint(
                &num_bigint::BigInt::from(polynomial.constant).to_signed_bytes_be(),
            )?,
        );
    }
    for (monomial, coefficient) in &polynomial.terms {
        let mut factors = Vec::with_capacity(monomial.len() + 1);
        if *coefficient != 1 {
            factors
                .push(arena.literal_bigint(
                    &num_bigint::BigInt::from(*coefficient).to_signed_bytes_be(),
                )?);
        }
        for symbol_name in monomial {
            if let Some((numerator, divisor)) = crate::types::parse_floor_quotient(symbol_name) {
                let terms = polynomial_width_terms(arena, scope, &numerator, bindings)?;
                let numerator = arena.add(terms)?;
                let divisor = arena.literal_u128(divisor.unsigned_abs().into())?;
                // A nonnegative width divided by a positive literal is natural.
                factors.push(arena.contracted(
                    "qamomile.floor_divide",
                    ResourceDomain::Natural,
                    [numerator, divisor],
                )?);
                continue;
            }
            let symbol = Dimension::Symbol {
                scope: Arc::clone(scope),
                name: Arc::from(symbol_name.as_str()),
            };
            if let Some(bound) = bindings.values.get(&symbol).copied() {
                factors.push(bound);
            } else {
                let constraint = format!("{symbol_name} >= 1");
                factors.push(arena.symbol(
                    scope,
                    symbol_name,
                    ResourceDomain::Natural,
                    "type_dimension",
                    &[constraint.as_str()],
                )?);
            }
        }
        terms.push(arena.multiply(factors)?);
    }
    Ok(terms)
}
