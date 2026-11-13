//! Structured `QCore` traversal and interprocedural symbolic analysis.

use crate::resource_rules::{Domain as ResourceDomain, TransformMode};
use crate::resources::analysis::{
    function_value_regions, is_cleanup_operation, loop_carry_invariants, while_carry_invariants,
};
use crate::static_values::range_len as static_range_len;
use crate::types::{Dimension, compare_dimensions};
use crate::wire;
use std::collections::{BTreeMap, HashMap, HashSet};

use super::timing::{collect_timing_roots, schedule_symbolic_depth_bounds};
use super::{
    BlockExit, BlockSummary, Counts, ExprId, MAX_SYMBOLIC_LOOP_INTERPRETATIONS, ScalarFact,
    SymbolicAnalysisFact, SymbolicAnalysisKey, SymbolicAnalyzer, SymbolicCount,
    SymbolicDimensionBindings, SymbolicFunctionContext, SymbolicResourceError,
    SymbolicTimingFootprint, SymbolicTimingNode, SymbolicTimingRoot, ValueFact,
    append_symbolic_timing, apply_control_footprints, attribute, bind_exact_results, bind_region,
    bind_results, bind_unpack, bind_values, bool_attribute, bool_fact, coherent_callees,
    first_operand, instantiate_symbolic_timing, int_fact, invariant_values,
    is_structural_coherent_wrapper, join_values, mapping_string_attribute,
    mapping_string_tuple_attribute, reverse_symbolic_timing, span_identity, string_attribute,
    symbolic_adjoint_roots, symbolic_block_exit, symbolic_call_roots, symbolic_coherent_roots,
    symbolic_dimension_bindings, symbolic_quantum_multiplicity, symbolic_timing_root,
};

impl SymbolicAnalyzer<'_> {
    pub(super) fn analyze_function(
        &mut self,
        function: &wire::FunctionProto,
        arguments: &[Option<ValueFact>],
        dimension_arguments: &HashMap<Dimension, ExprId>,
    ) -> Result<BlockSummary, SymbolicResourceError> {
        self.analyze_function_mode(
            function,
            arguments,
            dimension_arguments,
            TransformMode::Direct,
            Some(0),
        )
    }

    pub(super) fn analyze_function_mode(
        &mut self,
        function: &wire::FunctionProto,
        arguments: &[Option<ValueFact>],
        dimension_arguments: &HashMap<Dimension, ExprId>,
        mode: TransformMode,
        controls: Option<u32>,
    ) -> Result<BlockSummary, SymbolicResourceError> {
        if self.recurrences.contains(&function.symbol) {
            return self.recursive_summary(function, arguments, mode, controls);
        }
        let mut dimensions = dimension_arguments
            .iter()
            .map(|(dimension, value)| (dimension.clone(), *value))
            .collect::<Vec<_>>();
        dimensions.sort_unstable_by(|(left, _), (right, _)| compare_dimensions(left, right));
        let key = SymbolicAnalysisKey {
            symbol: function.symbol.clone(),
            arguments: arguments.into(),
            dimensions: dimensions.into(),
            mode,
            controls,
        };
        if !self.while_functions.contains(&function.symbol)
            && let Some(summary) = self.analysis_cache.get(&key)
        {
            return Ok(summary.clone());
        }
        if self.observation.is_some() && self.analysis_cache.len() >= 4096 {
            return Err(SymbolicResourceError::Unsupported(
                "Q4261: resource observation exceeds the context budget".to_owned(),
            ));
        }
        let summary = self.analyze_function_mode_uncached(
            function,
            arguments,
            dimension_arguments,
            mode,
            controls,
        )?;
        if !self.while_functions.contains(&function.symbol) {
            self.analysis_cache.insert(key, summary.clone());
        }
        Ok(summary)
    }

    pub(super) fn analyze_function_mode_uncached(
        &mut self,
        function: &wire::FunctionProto,
        arguments: &[Option<ValueFact>],
        dimension_arguments: &HashMap<Dimension, ExprId>,
        mode: TransformMode,
        controls: Option<u32>,
    ) -> Result<BlockSummary, SymbolicResourceError> {
        if function.external {
            let dimensions = symbolic_dimension_bindings(function, arguments, dimension_arguments);
            let counts = self.external_counts(function, arguments, &dimensions, mode)?;
            let timing = self.external_timing(function, &dimensions, mode, &counts)?;
            return Ok(BlockSummary {
                counts,
                exit: BlockExit::RETURN,
                values: Vec::new(),
                timing,
            });
        }
        if arguments.len() != function.parameters.len() {
            return Err(SymbolicResourceError::Missing(format!(
                "argument ABI for {:?}",
                function.symbol
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
        let dimensions = symbolic_dimension_bindings(function, arguments, dimension_arguments);
        let body = function.body.as_ref().ok_or_else(|| {
            SymbolicResourceError::Missing(format!("body for {:?}", function.symbol))
        })?;
        let context = SymbolicFunctionContext {
            owner: &function.symbol,
            dimensions: &dimensions,
            mode,
            controls,
        };
        let summary = self.analyze_block(body, &mut environment, context)?;
        if summary.exit != BlockExit::RETURN {
            return Err(SymbolicResourceError::Missing(format!(
                "return terminator in {:?}",
                function.symbol
            )));
        }
        Ok(summary)
    }

    pub(super) fn record_operation_fact(
        &mut self,
        operation: &wire::OperationProto,
        owner: &str,
        counts: &Counts,
    ) {
        let source = span_identity(operation);
        let key = (owner.to_owned(), source.clone(), operation.opcode.clone());
        if !self.fact_keys.insert(key) {
            return;
        }
        self.facts.push(SymbolicAnalysisFact {
            owner: owner.to_owned(),
            source,
            opcode: operation.opcode.clone(),
            work: counts.work.clone(),
            opcodes: counts.opcodes.clone(),
            extra_metrics: counts.extra_metrics.clone(),
            families: counts.families.clone(),
            measurements: counts.measurements.clone(),
            resets: counts.resets.clone(),
            cleanup_work: counts.cleanup_work.clone(),
        });
    }

    pub(super) fn optional_counts(
        &mut self,
        counts: &Counts,
        optional: bool,
    ) -> Result<Counts, SymbolicResourceError> {
        if !optional {
            return Ok(counts.clone());
        }
        Counts::branch(
            &[Counts::zero(&mut self.arena)?, counts.clone()],
            &mut self.arena,
        )
    }

    #[allow(clippy::too_many_lines)]
    pub(super) fn analyze_block(
        &mut self,
        block: &wire::BlockProto,
        environment: &mut HashMap<u64, Option<ValueFact>>,
        context: SymbolicFunctionContext<'_>,
    ) -> Result<BlockSummary, SymbolicResourceError> {
        self.charge_block(block);
        let mut counts = Counts::zero(&mut self.arena)?;
        let mut timing = Some(Vec::new());
        let mut optional_continuation = false;
        let mut continuation_predicate = None;
        let mut pending_exits = BlockExit(0);
        for operation in &block.operations {
            let mut recorded = false;
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
                    let effective = self.guard_continuation(
                        &summary.counts,
                        optional_continuation,
                        continuation_predicate,
                    )?;
                    self.record_operation_fact(operation, context.owner, &effective);
                    recorded = true;
                    counts.then(&effective, &self.error_bound_metrics, &mut self.arena)?;
                    if !optional_continuation {
                        append_symbolic_timing(&mut timing, summary.timing.clone());
                    }
                    bind_exact_results(operation, environment, &summary.values)?;
                }
                "qcore.if" => {
                    let summary = self.analyze_if(operation, environment, context)?;
                    let effective = self.guard_continuation(
                        &summary.counts,
                        optional_continuation,
                        continuation_predicate,
                    )?;
                    self.record_operation_fact(operation, context.owner, &effective);
                    recorded = true;
                    counts.then(&effective, &self.error_bound_metrics, &mut self.arena)?;
                    if !optional_continuation {
                        append_symbolic_timing(&mut timing, summary.timing.clone());
                    }
                    if summary.exit.contains(BlockExit::YIELD) {
                        bind_exact_results(operation, environment, &summary.values)?;
                        pending_exits = pending_exits.union(summary.exit.without(BlockExit::YIELD));
                        if summary.exit != BlockExit::YIELD {
                            continuation_predicate = self.fallthrough_predicate(
                                operation,
                                environment,
                                continuation_predicate,
                            )?;
                            optional_continuation = true;
                            timing = None;
                        }
                    } else {
                        return Ok(BlockSummary {
                            counts,
                            exit: pending_exits.union(summary.exit),
                            values: if optional_continuation {
                                vec![None; summary.values.len()]
                            } else {
                                summary.values
                            },
                            timing,
                        });
                    }
                }
                "qcore.for" | "qcore.while" => {
                    let summary = if operation.opcode == "qcore.while" {
                        self.analyze_while(operation, environment, context)?
                    } else {
                        self.analyze_for(operation, environment, context)?
                    };
                    let effective = self.guard_continuation(
                        &summary.counts,
                        optional_continuation,
                        continuation_predicate,
                    )?;
                    self.record_operation_fact(operation, context.owner, &effective);
                    recorded = true;
                    counts.then(&effective, &self.error_bound_metrics, &mut self.arena)?;
                    if !optional_continuation {
                        append_symbolic_timing(&mut timing, summary.timing.clone());
                    }
                    if !summary.exit.contains(BlockExit::YIELD) {
                        return Ok(BlockSummary {
                            counts,
                            exit: pending_exits.union(summary.exit),
                            values: summary.values,
                            timing,
                        });
                    }
                    bind_exact_results(operation, environment, &summary.values)?;
                    pending_exits = pending_exits.union(summary.exit.without(BlockExit::YIELD));
                    if summary.exit != BlockExit::YIELD {
                        optional_continuation = true;
                        timing = None;
                    }
                }
                "qcore.adjoint_call" => {
                    let (nested, nested_timing) =
                        self.analyze_adjoint(operation, environment, context)?;
                    let effective = self.guard_continuation(
                        &nested,
                        optional_continuation,
                        continuation_predicate,
                    )?;
                    self.record_operation_fact(operation, context.owner, &effective);
                    recorded = true;
                    counts.then(&effective, &self.error_bound_metrics, &mut self.arena)?;
                    if !optional_continuation {
                        append_symbolic_timing(&mut timing, nested_timing);
                    }
                    bind_results(operation, environment, None);
                }
                "qcore.control_call" | "qcore.select_call" => {
                    let (nested, nested_timing) =
                        self.analyze_coherent(operation, environment, context)?;
                    let effective = self.guard_continuation(
                        &nested,
                        optional_continuation,
                        continuation_predicate,
                    )?;
                    self.record_operation_fact(operation, context.owner, &effective);
                    recorded = true;
                    counts.then(&effective, &self.error_bound_metrics, &mut self.arena)?;
                    if !optional_continuation {
                        append_symbolic_timing(&mut timing, nested_timing);
                    }
                    bind_results(operation, environment, None);
                }
                opcode @ ("qcore.phase_by" | "qcore.xor_by" | "qcore.permute_by") => {
                    let event_width =
                        self.event_width(context.owner, operation, context.dimensions)?;
                    let operation_count = self.arena.literal_u128(1)?;
                    let mut quantum = Counts::quantum(
                        opcode,
                        opcode,
                        event_width,
                        operation_count,
                        is_cleanup_operation(operation),
                        &mut self.arena,
                    )?;
                    self.observe_primitive(
                        &mut quantum,
                        opcode,
                        context,
                        operation_count,
                        event_width,
                        None,
                        None,
                    )?;
                    let effective = self.guard_continuation(
                        &quantum,
                        optional_continuation,
                        continuation_predicate,
                    )?;
                    self.record_operation_fact(operation, context.owner, &effective);
                    recorded = true;
                    counts.then(&effective, &self.error_bound_metrics, &mut self.arena)?;
                    let step =
                        self.symbolic_quantum_timing(operation, environment, opcode, context)?;
                    if !optional_continuation {
                        append_symbolic_timing(
                            &mut timing,
                            step.map(|step| vec![SymbolicTimingNode::Step(step)]),
                        );
                    }
                    bind_results(operation, environment, None);
                }
                "qcore.adjoint" => {
                    let original = string_attribute(operation, "original").ok_or_else(|| {
                        SymbolicResourceError::Missing("adjoint original opcode".to_owned())
                    })?;
                    if matches!(original, "qcore.control_call" | "qcore.select_call") {
                        let mut nested = self.analyze_coherent_adjoint(operation, environment)?;
                        let one = self.arena.literal_u128(1)?;
                        self.observe_primitive(
                            &mut nested,
                            original,
                            SymbolicFunctionContext {
                                mode: context.mode.adjoint(),
                                ..context
                            },
                            one,
                            None,
                            None,
                            None,
                        )?;
                        let effective = self.guard_continuation(
                            &nested,
                            optional_continuation,
                            continuation_predicate,
                        )?;
                        self.record_operation_fact(operation, context.owner, &effective);
                        recorded = true;
                        counts.then(&effective, &self.error_bound_metrics, &mut self.arena)?;
                        timing = None;
                    } else {
                        let event_width =
                            self.event_width(context.owner, operation, context.dimensions)?;
                        let operation_count = symbolic_quantum_multiplicity(
                            operation,
                            original,
                            event_width,
                            &mut self.arena,
                        )?;
                        let mut quantum = Counts::quantum(
                            "qcore.adjoint",
                            original,
                            event_width,
                            operation_count,
                            is_cleanup_operation(operation),
                            &mut self.arena,
                        )?;
                        let mod_pow_widths =
                            if matches!(original, "quantum.mod_pow" | "quantum.mod_pow_inverse") {
                                self.mod_pow_widths(context.owner, operation, context.dimensions)?
                            } else {
                                None
                            };
                        let mod_multi_pow_dimensions = if matches!(
                            original,
                            "quantum.mod_multi_pow" | "quantum.mod_multi_pow_inverse"
                        ) {
                            self.mod_multi_pow_dimensions(
                                context.owner,
                                operation,
                                environment,
                                context.dimensions,
                            )?
                        } else {
                            None
                        };
                        self.observe_primitive(
                            &mut quantum,
                            original,
                            SymbolicFunctionContext {
                                mode: context.mode.adjoint(),
                                ..context
                            },
                            operation_count,
                            event_width,
                            mod_pow_widths,
                            mod_multi_pow_dimensions,
                        )?;
                        let effective = self.guard_continuation(
                            &quantum,
                            optional_continuation,
                            continuation_predicate,
                        )?;
                        self.record_operation_fact(operation, context.owner, &effective);
                        recorded = true;
                        counts.then(&effective, &self.error_bound_metrics, &mut self.arena)?;
                        let step = self.symbolic_quantum_timing(
                            operation,
                            environment,
                            original,
                            context,
                        )?;
                        if !optional_continuation {
                            append_symbolic_timing(
                                &mut timing,
                                step.map(|step| vec![SymbolicTimingNode::Step(step)]),
                            );
                        }
                    }
                    bind_results(operation, environment, None);
                }
                opcode if opcode.starts_with("quantum.") => {
                    let event_width =
                        self.event_width(context.owner, operation, context.dimensions)?;
                    let operation_count = symbolic_quantum_multiplicity(
                        operation,
                        opcode,
                        event_width,
                        &mut self.arena,
                    )?;
                    let mut quantum = Counts::quantum(
                        opcode,
                        opcode,
                        event_width,
                        operation_count,
                        is_cleanup_operation(operation),
                        &mut self.arena,
                    )?;
                    let mod_pow_widths =
                        if matches!(opcode, "quantum.mod_pow" | "quantum.mod_pow_inverse") {
                            self.mod_pow_widths(context.owner, operation, context.dimensions)?
                        } else {
                            None
                        };
                    let mod_multi_pow_dimensions = if matches!(
                        opcode,
                        "quantum.mod_multi_pow" | "quantum.mod_multi_pow_inverse"
                    ) {
                        self.mod_multi_pow_dimensions(
                            context.owner,
                            operation,
                            environment,
                            context.dimensions,
                        )?
                    } else {
                        None
                    };
                    self.observe_primitive(
                        &mut quantum,
                        opcode,
                        context,
                        operation_count,
                        event_width,
                        mod_pow_widths,
                        mod_multi_pow_dimensions,
                    )?;
                    let effective = self.guard_continuation(
                        &quantum,
                        optional_continuation,
                        continuation_predicate,
                    )?;
                    self.record_operation_fact(operation, context.owner, &effective);
                    recorded = true;
                    counts.then(&effective, &self.error_bound_metrics, &mut self.arena)?;
                    let step =
                        self.symbolic_quantum_timing(operation, environment, opcode, context)?;
                    if !optional_continuation {
                        append_symbolic_timing(
                            &mut timing,
                            step.map(|step| vec![SymbolicTimingNode::Step(step)]),
                        );
                    }
                    bind_results(operation, environment, None);
                }
                opcode if opcode.starts_with("classical.") => {
                    let value = self.evaluate_classical(operation, environment, context)?;
                    bind_results(operation, environment, value.as_ref());
                }
                _ => bind_results(operation, environment, None),
            }
            if !recorded {
                let zero = Counts::zero(&mut self.arena)?;
                self.record_operation_fact(operation, context.owner, &zero);
            }
        }
        let terminator = block
            .terminator
            .as_ref()
            .ok_or_else(|| SymbolicResourceError::Missing("block terminator".to_owned()))?;
        let values = if optional_continuation {
            vec![None; terminator.operands.len()]
        } else {
            terminator
                .operands
                .iter()
                .map(|operand| environment.get(operand).cloned().flatten())
                .collect()
        };
        Ok(BlockSummary {
            counts,
            exit: pending_exits.union(symbolic_block_exit(terminator)?),
            values,
            timing,
        })
    }

    pub(super) fn analyze_call(
        &mut self,
        operation: &wire::OperationProto,
        environment: &HashMap<u64, Option<ValueFact>>,
        context: SymbolicFunctionContext<'_>,
    ) -> Result<BlockSummary, SymbolicResourceError> {
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
        let mut summary = self.analyze_function_mode(
            callee,
            &arguments,
            &dimension_arguments,
            context.mode,
            context.controls,
        )?;
        self.observe_call(&mut summary.counts, callee, context.mode, context.controls)?;
        if callee.external || self.recurrences.contains(&callee.symbol) {
            summary.values = vec![None; operation.results.len()];
        }
        let owner_function = self
            .functions
            .get(context.owner)
            .copied()
            .ok_or_else(|| SymbolicResourceError::Missing("caller".to_owned()))?;
        let mut roots = symbolic_call_roots(callee, operation, owner_function)?;
        for root in roots.values_mut() {
            *root = self.canonical_timing_root(root.clone());
        }
        summary.timing = summary.timing.map(|timing| {
            instantiate_symbolic_timing(
                timing,
                &callee.symbol,
                &roots,
                &format!("{}:{}", context.owner, span_identity(operation)),
            )
        });
        Ok(summary)
    }

    pub(super) fn call_dimensions(
        &mut self,
        operation: &wire::OperationProto,
        callee: &wire::FunctionProto,
        context: SymbolicFunctionContext<'_>,
    ) -> Result<HashMap<Dimension, ExprId>, SymbolicResourceError> {
        let declared_types = self.operations.value_types(&callee.symbol).ok_or_else(|| {
            SymbolicResourceError::Missing(format!("type roots for {:?}", callee.symbol))
        })?;
        let expected = callee
            .parameters
            .iter()
            .map(|parameter| {
                parameter
                    .value
                    .as_ref()
                    .and_then(|value| declared_types.get(&value.id))
                    .copied()
                    .ok_or_else(|| {
                        SymbolicResourceError::Missing(format!(
                            "parameter type in {:?}",
                            callee.symbol
                        ))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let local_types = self.operations.value_types(context.owner).ok_or_else(|| {
            SymbolicResourceError::Missing(format!("type roots for {:?}", context.owner))
        })?;
        let actual = operation
            .operands
            .iter()
            .map(|operand| {
                local_types.get(operand).copied().ok_or_else(|| {
                    SymbolicResourceError::Missing(format!("operand type in {:?}", context.owner))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let inferred = crate::abi::infer_call_dimensions(self.types, &expected, &actual)
            .ok_or_else(|| SymbolicResourceError::Missing("verified call dimensions".to_owned()))?;
        inferred
            .into_iter()
            .filter_map(|(formal, actual)| {
                self.resolve_dimension(&actual, context.dimensions)
                    .transpose()
                    .map(|resolved| resolved.map(|value| (formal, value)))
            })
            .collect()
    }

    pub(super) fn coherent_call_dimensions(
        &mut self,
        operation: &wire::OperationProto,
        callee: &wire::FunctionProto,
        context: SymbolicFunctionContext<'_>,
        offset: usize,
    ) -> Result<HashMap<Dimension, ExprId>, SymbolicResourceError> {
        let declared_types = self.operations.value_types(&callee.symbol).ok_or_else(|| {
            SymbolicResourceError::Missing(format!("type roots for {:?}", callee.symbol))
        })?;
        let expected = callee
            .parameters
            .iter()
            .map(|parameter| {
                parameter
                    .value
                    .as_ref()
                    .and_then(|value| declared_types.get(&value.id))
                    .copied()
                    .ok_or_else(|| {
                        SymbolicResourceError::Missing(format!(
                            "parameter type in {:?}",
                            callee.symbol
                        ))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let local_types = self.operations.value_types(context.owner).ok_or_else(|| {
            SymbolicResourceError::Missing(format!("type roots for {:?}", context.owner))
        })?;
        let actual = operation
            .operands
            .get(offset..)
            .ok_or_else(|| SymbolicResourceError::Missing("coherent call ABI".to_owned()))?
            .iter()
            .map(|operand| {
                local_types.get(operand).copied().ok_or_else(|| {
                    SymbolicResourceError::Missing(format!(
                        "coherent operand type in {:?}",
                        context.owner
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let inferred = crate::abi::infer_call_dimensions(self.types, &expected, &actual)
            .ok_or_else(|| {
                SymbolicResourceError::Missing("verified coherent call dimensions".to_owned())
            })?;
        inferred
            .into_iter()
            .filter_map(|(formal, actual)| {
                self.resolve_dimension(&actual, context.dimensions)
                    .transpose()
                    .map(|resolved| resolved.map(|value| (formal, value)))
            })
            .collect()
    }

    pub(super) fn resolve_dimension(
        &mut self,
        dimension: &Dimension,
        bindings: &SymbolicDimensionBindings,
    ) -> Result<Option<ExprId>, SymbolicResourceError> {
        match dimension {
            Dimension::Fixed(value) => self.arena.literal_u128(u128::from(*value)).map(Some),
            Dimension::Symbol { .. } if bindings.values.contains_key(dimension) => {
                Self::dimension_width(&mut self.arena, dimension, bindings).map(Some)
            }
            // A polynomial width such as `-half+w` resolves when every symbol
            // it mentions is bound in the caller.
            Dimension::Symbol { scope, name }
                if !name.starts_with("slice(")
                    && crate::types::PolynomialDimension::parse(name).is_ok_and(|polynomial| {
                        polynomial.symbols().iter().all(|symbol| {
                            bindings.values.contains_key(&Dimension::Symbol {
                                scope: std::sync::Arc::clone(scope),
                                name: std::sync::Arc::from(symbol.as_str()),
                            })
                        })
                    }) =>
            {
                Self::dimension_width(&mut self.arena, dimension, bindings).map(Some)
            }
            Dimension::Symbol { .. } => Ok(None),
        }
    }

    pub(super) fn analyze_if(
        &mut self,
        operation: &wire::OperationProto,
        environment: &HashMap<u64, Option<ValueFact>>,
        context: SymbolicFunctionContext<'_>,
    ) -> Result<BlockSummary, SymbolicResourceError> {
        let condition = operation
            .operands
            .first()
            .and_then(|operand| environment.get(operand))
            .and_then(Option::as_ref)
            .and_then(bool_fact);
        let captures = operation.operands.get(1..).unwrap_or_default();
        let mut branches = Vec::new();
        for region in &operation.regions {
            let block = region
                .blocks
                .first()
                .ok_or_else(|| SymbolicResourceError::Missing("qcore.if branch".to_owned()))?;
            let mut local = bind_region(block, captures, environment, false, None)?;
            let mut summary = self.analyze_block(block, &mut local, context)?;
            if string_attribute(operation, "body_mode") == Some("adjoint") {
                summary.counts = summary.counts.cleanup_adjoint(&mut self.arena)?;
                summary.values = captures
                    .iter()
                    .map(|operand| environment.get(operand).cloned().flatten())
                    .collect();
                summary.timing = summary.timing.map(reverse_symbolic_timing);
            }
            branches.push(summary);
        }
        if branches.len() != 2 {
            return Err(SymbolicResourceError::Missing(
                "two qcore.if branches".to_owned(),
            ));
        }
        if let Some(selected) = condition {
            return Ok(branches.remove(usize::from(!selected)));
        }
        if branches
            .iter()
            .any(|branch| branch.exit != BlockExit::YIELD)
        {
            let mut summary = self.analyze_non_yielding_if(&branches)?;
            if self.recurrences.encountered
                && branches[0].exit == branches[1].exit
                && let Some(ValueFact::Scalar(ScalarFact::Predicate(predicate))) = operation
                    .operands
                    .first()
                    .and_then(|operand| environment.get(operand))
                    .and_then(Option::as_ref)
            {
                summary.counts = Counts::conditional(
                    &branches[0].counts,
                    &branches[1].counts,
                    *predicate,
                    &mut self.arena,
                )?;
            }
            return Ok(summary);
        }
        let mut counts = Counts::branch(
            &branches
                .iter()
                .map(|branch| branch.counts.clone())
                .collect::<Vec<_>>(),
            &mut self.arena,
        )?;
        // A symbolic branch predicate (for example a Const comparison) selects
        // one branch for every admitted binding, so additive costs are exact
        // piecewise functions rather than the [min, max] join of both arms.
        if let Some(ValueFact::Scalar(ScalarFact::Predicate(predicate))) = operation
            .operands
            .first()
            .and_then(|operand| environment.get(operand))
            .and_then(Option::as_ref)
        {
            counts = Counts::conditional(
                &branches[0].counts,
                &branches[1].counts,
                *predicate,
                &mut self.arena,
            )?;
        }
        let values = join_values(branches.iter().map(|branch| branch.values.as_slice()))?;
        let timing = if branches[0].timing == branches[1].timing {
            branches[0].timing.clone()
        } else {
            let predicate = match operation
                .operands
                .first()
                .and_then(|operand| environment.get(operand))
                .and_then(Option::as_ref)
            {
                Some(ValueFact::Scalar(ScalarFact::Predicate(predicate))) => Some(*predicate),
                _ => None,
            };
            self.branch_barrier(&branches, predicate)
        };
        Ok(BlockSummary {
            counts,
            exit: BlockExit::YIELD,
            values,
            timing,
        })
    }

    /// Summarize branches with different schedules as one occupying region.
    ///
    /// The region occupies every owner either arm touches. With a symbolic
    /// branch predicate (for example a Const comparison) its duration is the
    /// selected arm's makespan as a piecewise expression; otherwise either arm
    /// may run and it lasts the longer arm's makespan.
    fn branch_barrier(
        &mut self,
        branches: &[BlockSummary],
        predicate: Option<ExprId>,
    ) -> Option<Vec<SymbolicTimingNode>> {
        let mut roots = Vec::new();
        let mut durations: Option<BTreeMap<String, ExprId>> = None;
        for branch in branches {
            let nodes = branch.timing.as_deref()?;
            for root in collect_timing_roots(nodes) {
                if !roots.contains(&root) {
                    roots.push(root);
                }
            }
            let (maxima, _) = schedule_symbolic_depth_bounds(nodes, &mut self.arena).ok()?;
            durations = Some(match durations {
                None => maxima,
                Some(previous) => {
                    let mut joined = BTreeMap::new();
                    for (family, left) in previous {
                        if let Some(right) = maxima.get(&family) {
                            let duration = match predicate {
                                Some(predicate) => self
                                    .arena
                                    .piecewise(
                                        &[predicate],
                                        &[left, *right],
                                        ResourceDomain::Natural,
                                    )
                                    .ok()?,
                                None => self.arena.extremum(true, [left, *right]).ok()?,
                            };
                            joined.insert(family, duration);
                        }
                    }
                    joined
                }
            });
        }
        Some(vec![SymbolicTimingNode::Barrier {
            roots: roots.into_boxed_slice(),
            durations: durations?,
        }])
    }

    pub(super) fn analyze_non_yielding_if(
        &mut self,
        branches: &[BlockSummary],
    ) -> Result<BlockSummary, SymbolicResourceError> {
        let exits = branches
            .iter()
            .map(|branch| branch.exit)
            .collect::<Vec<_>>();
        let counts = Counts::branch(
            &branches
                .iter()
                .map(|branch| branch.counts.clone())
                .collect::<Vec<_>>(),
            &mut self.arena,
        )?;
        let exit = exits.iter().copied().fold(BlockExit(0), BlockExit::union);
        if exits.iter().all(|candidate| *candidate == exit)
            || exit.without(BlockExit::BREAK.union(BlockExit::CONTINUE)) == BlockExit(0)
        {
            let values = join_values(branches.iter().map(|branch| branch.values.as_slice()))?;
            let timing = (branches[0].timing == branches[1].timing)
                .then(|| branches[0].timing.clone())
                .flatten();
            return Ok(BlockSummary {
                counts,
                exit,
                values,
                timing,
            });
        }
        if exit.contains(BlockExit::YIELD) {
            let values = branches
                .iter()
                .find(|branch| branch.exit.contains(BlockExit::YIELD))
                .map(|branch| branch.values.clone())
                .ok_or_else(|| {
                    SymbolicResourceError::Missing("fallthrough qcore.if branch values".to_owned())
                })?;
            return Ok(BlockSummary {
                counts,
                exit,
                values,
                timing: None,
            });
        }
        let exits = exits.iter().map(|exit| exit.as_str()).collect::<Vec<_>>();
        Err(SymbolicResourceError::Unsupported(format!(
            "path-dependent qcore.if branch terminators {exits:?}"
        )))
    }

    #[allow(clippy::too_many_lines)]
    /// Preserve coverage when changing carries prevent an inner range formula.
    fn unresolved_while_body(
        &mut self,
        body: &wire::BlockProto,
        context: SymbolicFunctionContext<'_>,
    ) -> Result<BlockSummary, SymbolicResourceError> {
        self.unresolved_region(
            body,
            "while body cost depends on changing loop carries",
            context,
        )
    }

    /// Summarize a region whose cost cannot be derived as named unknowns.
    fn unresolved_region(
        &mut self,
        body: &wire::BlockProto,
        reason: &str,
        context: SymbolicFunctionContext<'_>,
    ) -> Result<BlockSummary, SymbolicResourceError> {
        let count = SymbolicCount::unresolved(reason.to_owned(), &mut self.arena)?;
        let function = self
            .functions
            .get(context.owner)
            .copied()
            .ok_or_else(|| SymbolicResourceError::Missing("while owner".to_owned()))?;
        let opcodes = super::statically_reachable_quantum_opcodes(function, &self.functions)
            .into_iter()
            .map(|opcode| (opcode, count.clone()))
            .collect();
        let mut extra_metrics = self
            .rules
            .values()
            .map(|rule| (rule.metric.identifier.clone(), count.clone()))
            .collect::<BTreeMap<_, _>>();
        for function in self.functions.values().filter(|function| function.external) {
            for coordinate in ["call_count", "query_count"] {
                extra_metrics.insert(
                    format!("qamomile.algorithmic.{coordinate}.{}", function.symbol),
                    count.clone(),
                );
            }
        }
        let exit = body
            .terminator
            .as_ref()
            .ok_or_else(|| SymbolicResourceError::Missing("while body terminator".to_owned()))?;
        Ok(BlockSummary {
            counts: Counts {
                work: count.clone(),
                span: count.clone(),
                cleanup_work: count.clone(),
                depths: Some(
                    super::external_depth_metrics()
                        .iter()
                        .map(|(family, _)| ((*family).to_owned(), count.clone()))
                        .collect(),
                ),
                opcodes,
                extra_metrics,
                families: Some(
                    super::external_family_metrics()
                        .iter()
                        .map(|(family, _)| ((*family).to_owned(), count.clone()))
                        .collect(),
                ),
                measurements: Some(count.clone()),
                resets: Some(count),
            },
            exit: symbolic_block_exit(exit)?,
            values: vec![None; exit.operands.len()],
            timing: None,
        })
    }

    pub(super) fn while_trip_count(
        &mut self,
        operation: &wire::OperationProto,
        context: SymbolicFunctionContext<'_>,
    ) -> Result<ExprId, SymbolicResourceError> {
        let source = format!("{}:{}", context.owner, span_identity(operation));
        if self.while_contexts >= 4096 {
            return Err(SymbolicResourceError::Unsupported(
                "while resource context budget".to_owned(),
            ));
        }
        let scope = format!("{source}:context:{}", self.while_contexts);
        self.while_contexts += 1;
        let count = self.arena.symbol(
            &scope,
            "iterations",
            ResourceDomain::Natural,
            "while_iterations",
            &["actual body executions per invocation"],
        )?;

        Ok(count)
    }

    pub(super) fn analyze_while(
        &mut self,
        operation: &wire::OperationProto,
        environment: &HashMap<u64, Option<ValueFact>>,
        context: SymbolicFunctionContext<'_>,
    ) -> Result<BlockSummary, SymbolicResourceError> {
        let invariants = while_carry_invariants(operation);
        let values = invariant_values(&operation.operands, environment, &invariants);
        let condition = &operation.regions[0].blocks[0];
        let body = &operation.regions[1].blocks[0];
        let otherwise = &operation.regions[2].blocks[0];
        let count = self.while_trip_count(operation, context)?;
        let mut local = bind_values(condition, values.clone())?;
        let test = self.analyze_block(condition, &mut local, context)?;
        if test.values.first().is_some_and(|value| {
            matches!(
                value,
                Some(ValueFact::Scalar(ScalarFact::Bool(Some(false))))
            )
        }) {
            let mut local = bind_values(otherwise, values)?;
            let mut exit = self.analyze_block(otherwise, &mut local, context)?;
            let mut counts = test.counts;
            counts.then(&exit.counts, &self.error_bound_metrics, &mut self.arena)?;
            exit.counts = counts;
            return Ok(exit);
        }
        let binder = self.arena.symbol(
            &format!("{}:{}", context.owner, span_identity(operation)),
            "iteration",
            ResourceDomain::Natural,
            "loop_iteration",
            &["0 <= iteration"],
        )?;
        let mut local = bind_values(body, values.clone())?;
        let summary = match self.analyze_block(body, &mut local, context) {
            Ok(summary) => summary,
            Err(SymbolicResourceError::Missing(reason)) if reason == "qcore.for bounds" => {
                self.unresolved_while_body(body, context)?
            }
            Err(error) => return Err(error),
        };
        let mut counts =
            summary
                .counts
                .repeat(count, binder, &self.error_bound_metrics, &mut self.arena)?;
        let repeated_test =
            test.counts
                .repeat(count, binder, &self.error_bound_metrics, &mut self.arena)?;
        counts.then(&repeated_test, &self.error_bound_metrics, &mut self.arena)?;
        let normal_exit =
            !summary.exit.contains(BlockExit::BREAK) && !summary.exit.contains(BlockExit::RETURN);
        let final_test = self.optional_counts(&test.counts, !normal_exit)?;
        counts.then(&final_test, &self.error_bound_metrics, &mut self.arena)?;
        let mut local = bind_values(otherwise, values.clone())?;
        let exit = self.analyze_block(otherwise, &mut local, context)?;
        let exit_counts = self.optional_counts(&exit.counts, !normal_exit)?;
        counts.then(&exit_counts, &self.error_bound_metrics, &mut self.arena)?;
        let all_paths_return = operation.results.is_empty() && exit.exit == BlockExit::RETURN;
        let loop_exit = BlockExit::YIELD
            .union(
                summary.exit.without(
                    BlockExit::YIELD
                        .union(BlockExit::CONTINUE)
                        .union(BlockExit::BREAK),
                ),
            )
            .union(exit.exit.without(BlockExit::YIELD));
        Ok(BlockSummary {
            counts,
            exit: if all_paths_return {
                BlockExit::RETURN
            } else {
                loop_exit
            },
            values: if all_paths_return {
                vec![None]
            } else if exit.exit == BlockExit::YIELD && normal_exit {
                exit.values
            } else {
                vec![None; operation.results.len()]
            },
            timing: None,
        })
    }

    #[expect(
        clippy::too_many_lines,
        reason = "Keep bounded-loop counts and structured exits together"
    )]
    pub(super) fn analyze_for(
        &mut self,
        operation: &wire::OperationProto,
        environment: &HashMap<u64, Option<ValueFact>>,
        context: SymbolicFunctionContext<'_>,
    ) -> Result<BlockSummary, SymbolicResourceError> {
        let [lower_id, upper_id, stride_id, ..] = operation.operands.as_slice() else {
            return Err(SymbolicResourceError::Missing(
                "qcore.for bounds".to_owned(),
            ));
        };
        let Some(bounds) = [*lower_id, *upper_id, *stride_id]
            .map(|operand| int_fact(environment.get(&operand).and_then(Option::as_ref)))
            .into_iter()
            .collect::<Option<Vec<_>>>()
        else {
            // A bound computed by an unanalyzed carry (for example a Const
            // `while`) leaves this loop's cost unknown without hiding the
            // costs of the rest of the program.
            let body = operation
                .regions
                .first()
                .and_then(|region| region.blocks.first())
                .ok_or_else(|| SymbolicResourceError::Missing("qcore.for body".to_owned()))?;
            let mut summary = self.unresolved_region(
                body,
                "qcore.for trip count depends on a value the analyzer could not derive",
                context,
            )?;
            summary.exit = BlockExit::YIELD;
            summary.values = vec![None; operation.results.len()];
            return Ok(summary);
        };
        let [lower, upper, stride] = bounds.as_slice() else {
            return Err(SymbolicResourceError::Missing(
                "qcore.for bounds".to_owned(),
            ));
        };
        let exact_bounds = [lower.exact, upper.exact, stride.exact];
        let exact_trip = exact_bounds
            .iter()
            .copied()
            .collect::<Option<Vec<_>>>()
            .and_then(|values| static_range_len(values[0], values[1], values[2]));
        if exact_trip.is_some_and(|count| count <= MAX_SYMBOLIC_LOOP_INTERPRETATIONS) {
            return self.interpret_for(
                operation,
                environment,
                context,
                exact_bounds[0].expect("checked exact bound"),
                exact_bounds[2].expect("checked exact bound"),
                exact_trip.expect("checked exact trip count"),
            );
        }
        let trip_count = if let Some(value) = exact_trip {
            self.arena.literal_u128(value)?
        } else {
            self.arena.contracted(
                "qamomile.range_trip_count",
                ResourceDomain::Natural,
                [lower.expression, upper.expression, stride.expression],
            )?
        };
        let body = operation
            .regions
            .first()
            .and_then(|region| region.blocks.first())
            .ok_or_else(|| SymbolicResourceError::Missing("qcore.for body".to_owned()))?;
        let carry_operands = operation.operands.get(3..).unwrap_or_default();
        let invariants = loop_carry_invariants(body, carry_operands.len());
        let binder = self.arena.symbol(
            &format!("{}:{}", context.owner, span_identity(operation)),
            string_attribute(operation, "induction_name").unwrap_or("iteration"),
            ResourceDomain::Natural,
            "loop_iteration",
            &["0 <= iteration"],
        )?;
        let scaled = self.arena.multiply([stride.expression, binder])?;
        let induction = self.arena.add([lower.expression, scaled])?;
        let mut local = bind_region(
            body,
            carry_operands,
            environment,
            true,
            Some((induction, &invariants)),
        )?;
        let mut summary = self.analyze_block(body, &mut local, context)?;
        let loop_exits = BlockExit::YIELD
            .union(BlockExit::CONTINUE)
            .union(BlockExit::BREAK);
        if summary.exit.without(loop_exits) != BlockExit(0) {
            return Err(SymbolicResourceError::Unsupported(format!(
                "qcore.for body terminator {:?}",
                summary.exit.as_str()
            )));
        }
        let adjoint_body = string_attribute(operation, "body_mode") == Some("adjoint");
        let may_break = summary.exit.contains(BlockExit::BREAK);
        let path_dependent =
            summary.exit != BlockExit::YIELD && summary.exit != BlockExit::CONTINUE;
        if adjoint_body {
            summary.counts = summary.counts.cleanup_adjoint(&mut self.arena)?;
            summary.timing = summary.timing.map(reverse_symbolic_timing);
        }
        let timing = summary.timing.filter(|_| !path_dependent).map(|body| {
            vec![SymbolicTimingNode::Repeat {
                body: body.into_boxed_slice(),
                binder,
                trip_count,
            }]
        });
        let body_counts = self.optional_counts(&summary.counts, may_break)?;
        let mut counts = body_counts.repeat(
            trip_count,
            binder,
            &self.error_bound_metrics,
            &mut self.arena,
        )?;
        let mut carries = invariant_values(carry_operands, environment, &invariants);
        let mut timing = timing;
        if !adjoint_body
            && let Some(otherwise) = operation
                .regions
                .get(1)
                .and_then(|region| region.blocks.first())
        {
            let mut local = bind_values(otherwise, carries)?;
            let summary = self.analyze_block(otherwise, &mut local, context)?;
            if summary.exit != BlockExit::YIELD {
                return Err(SymbolicResourceError::Unsupported(format!(
                    "qcore.for else terminator {:?}",
                    summary.exit.as_str()
                )));
            }
            let otherwise_counts = self.optional_counts(&summary.counts, may_break)?;
            counts.then(
                &otherwise_counts,
                &self.error_bound_metrics,
                &mut self.arena,
            )?;
            if may_break {
                timing = None;
                carries = vec![None; carry_operands.len()];
            } else {
                append_symbolic_timing(&mut timing, summary.timing.clone());
                carries = summary.values;
            }
        }
        Ok(BlockSummary {
            counts,
            exit: BlockExit::YIELD,
            values: carries,
            timing,
        })
    }

    #[expect(
        clippy::too_many_lines,
        reason = "Keep bounded loop interpretation and structured exits together"
    )]
    pub(super) fn interpret_for(
        &mut self,
        operation: &wire::OperationProto,
        environment: &HashMap<u64, Option<ValueFact>>,
        context: SymbolicFunctionContext<'_>,
        lower: i64,
        stride: i64,
        trip_count: u128,
    ) -> Result<BlockSummary, SymbolicResourceError> {
        let body = operation
            .regions
            .first()
            .and_then(|region| region.blocks.first())
            .ok_or_else(|| SymbolicResourceError::Missing("qcore.for body".to_owned()))?;
        let mut carries = operation
            .operands
            .get(3..)
            .unwrap_or_default()
            .iter()
            .map(|operand| environment.get(operand).cloned().flatten())
            .collect::<Vec<_>>();
        let mut counts = Counts::zero(&mut self.arena)?;
        let mut timing = Some(Vec::new());
        let reverse = bool_attribute(operation, "reverse").unwrap_or(false);
        let mut broke = false;
        let mut may_break = false;
        for position in 0..trip_count {
            let logical = if reverse {
                trip_count - position - 1
            } else {
                position
            };
            let offset = i128::try_from(logical)
                .ok()
                .and_then(|value| value.checked_mul(i128::from(stride)))
                .ok_or(SymbolicResourceError::Overflow)?;
            let induction = i128::from(lower)
                .checked_add(offset)
                .and_then(|value| i64::try_from(value).ok())
                .ok_or(SymbolicResourceError::Overflow)?;
            let induction = ValueFact::Scalar(ScalarFact::Int {
                expression: self.arena.literal_i64(induction)?,
                exact: Some(induction),
            });
            let arguments = std::iter::once(Some(induction))
                .chain(carries.iter().cloned())
                .collect();
            let mut local = bind_values(body, arguments)?;
            let mut summary = self.analyze_block(body, &mut local, context)?;
            if string_attribute(operation, "body_mode") == Some("adjoint") {
                summary.counts = summary.counts.cleanup_adjoint(&mut self.arena)?;
                summary.timing = summary.timing.map(reverse_symbolic_timing);
                summary.values.clone_from(&carries);
            }
            let effective = self.optional_counts(&summary.counts, may_break)?;
            counts.then(&effective, &self.error_bound_metrics, &mut self.arena)?;
            if !may_break {
                append_symbolic_timing(&mut timing, summary.timing.clone());
            }
            match summary.exit {
                BlockExit::YIELD | BlockExit::CONTINUE => {
                    carries = if may_break {
                        vec![None; carries.len()]
                    } else {
                        summary.values
                    };
                }
                BlockExit::BREAK => {
                    carries = if may_break {
                        vec![None; carries.len()]
                    } else {
                        summary.values
                    };
                    broke = true;
                    break;
                }
                BlockExit::RETURN => {
                    if may_break {
                        return Err(SymbolicResourceError::Unsupported(
                            "path-dependent qcore.for return".to_owned(),
                        ));
                    }
                    return Ok(BlockSummary {
                        counts,
                        timing,
                        ..summary
                    });
                }
                exit if exit.without(
                    BlockExit::YIELD
                        .union(BlockExit::BREAK)
                        .union(BlockExit::CONTINUE),
                ) == BlockExit(0) =>
                {
                    may_break |= exit.contains(BlockExit::BREAK);
                    carries.fill(None);
                    timing = None;
                }
                exit => {
                    return Err(SymbolicResourceError::Unsupported(format!(
                        "qcore.for body terminator {:?}",
                        exit.as_str()
                    )));
                }
            }
        }
        if !broke
            && string_attribute(operation, "body_mode") != Some("adjoint")
            && let Some(otherwise) = operation
                .regions
                .get(1)
                .and_then(|region| region.blocks.first())
        {
            let carry_count = carries.len();
            let mut local = bind_values(otherwise, carries)?;
            let summary = self.analyze_block(otherwise, &mut local, context)?;
            let effective = self.optional_counts(&summary.counts, may_break)?;
            counts.then(&effective, &self.error_bound_metrics, &mut self.arena)?;
            if !may_break {
                append_symbolic_timing(&mut timing, summary.timing.clone());
            }
            if summary.exit == BlockExit::RETURN {
                // Return values and loop carries have different ABIs. Preserve
                // the existing exclusion until mixed return paths model both.
                if may_break {
                    return Err(SymbolicResourceError::Unsupported(
                        "path-dependent qcore.for else return".to_owned(),
                    ));
                }
                return Ok(BlockSummary {
                    counts,
                    timing,
                    ..summary
                });
            }
            if summary.exit != BlockExit::YIELD {
                return Err(SymbolicResourceError::Unsupported(format!(
                    "qcore.for else terminator {:?}",
                    summary.exit.as_str()
                )));
            }
            carries = if may_break {
                vec![None; carry_count]
            } else {
                summary.values
            };
        }
        Ok(BlockSummary {
            counts,
            exit: BlockExit::YIELD,
            values: carries,
            timing,
        })
    }

    pub(super) fn analyze_adjoint(
        &mut self,
        operation: &wire::OperationProto,
        environment: &HashMap<u64, Option<ValueFact>>,
        context: SymbolicFunctionContext<'_>,
    ) -> Result<(Counts, Option<Vec<SymbolicTimingNode>>), SymbolicResourceError> {
        let callee_name = string_attribute(operation, "callee")
            .ok_or_else(|| SymbolicResourceError::Missing("adjoint callee".to_owned()))?;
        let callee =
            self.functions.get(callee_name).copied().ok_or_else(|| {
                SymbolicResourceError::Missing(format!("callable {callee_name:?}"))
            })?;
        let mut classical = operation.operands.iter().skip(1);
        let arguments = callee
            .parameters
            .iter()
            .map(|parameter| {
                let value = parameter.value.as_ref().ok_or_else(|| {
                    SymbolicResourceError::Missing("adjoint parameter".to_owned())
                })?;
                if value.has_region {
                    Ok(None)
                } else {
                    let operand = classical.next().ok_or_else(|| {
                        SymbolicResourceError::Missing("adjoint classical argument".to_owned())
                    })?;
                    Ok(environment.get(operand).cloned().flatten())
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let owner_function = self
            .functions
            .get(context.owner)
            .copied()
            .ok_or_else(|| SymbolicResourceError::Missing("adjoint owner".to_owned()))?;
        let roots = symbolic_adjoint_roots(callee, operation, owner_function)?.map(|mut roots| {
            for root in roots.values_mut() {
                *root = self.canonical_timing_root(root.clone());
            }
            roots
        });
        let nested_mode = context.mode.adjoint();
        if callee.external {
            let dimensions = symbolic_dimension_bindings(callee, &arguments, &HashMap::new());
            let mut counts = self.external_counts(callee, &arguments, &dimensions, nested_mode)?;
            self.observe_call(&mut counts, callee, nested_mode, context.controls)?;
            let timing = if let Some(roots) = roots.as_ref() {
                self.external_timing(callee, &dimensions, nested_mode, &counts)?
                    .map(|timing| {
                        reverse_symbolic_timing(instantiate_symbolic_timing(
                            timing,
                            &callee.symbol,
                            roots,
                            &format!("{}:{}", context.owner, span_identity(operation)),
                        ))
                    })
            } else {
                None
            };
            return Ok((counts, timing));
        }
        let direct = self.analyze_function_mode(
            callee,
            &arguments,
            &HashMap::new(),
            nested_mode,
            context.controls,
        )?;
        let timing = roots.as_ref().and_then(|roots| {
            direct.timing.map(|timing| {
                reverse_symbolic_timing(instantiate_symbolic_timing(
                    timing,
                    &callee.symbol,
                    roots,
                    &format!("{}:{}", context.owner, span_identity(operation)),
                ))
            })
        });
        let mut direct = direct.counts;
        self.observe_call(&mut direct, callee, nested_mode, context.controls)?;
        let zero = SymbolicCount::exact_value(self.arena.literal_u128(0)?);
        Ok((
            Counts {
                work: direct.work.clone(),
                span: direct.span.clone(),
                depths: direct.depths.clone(),
                opcodes: BTreeMap::from([("qcore.adjoint".to_owned(), direct.work.clone())]),
                extra_metrics: direct.extra_metrics,
                families: direct.families,
                measurements: Some(zero.clone()),
                resets: Some(zero.clone()),
                cleanup_work: if string_attribute(operation, "proof") == Some("modular-adjoint") {
                    direct.work.clone()
                } else {
                    zero
                },
            },
            timing,
        ))
    }

    pub(super) fn analyze_coherent(
        &mut self,
        operation: &wire::OperationProto,
        environment: &HashMap<u64, Option<ValueFact>>,
        context: SymbolicFunctionContext<'_>,
    ) -> Result<(Counts, Option<Vec<SymbolicTimingNode>>), SymbolicResourceError> {
        let names = coherent_callees(operation)?;
        let first = names
            .first()
            .and_then(|name| self.functions.get(name).copied())
            .ok_or_else(|| SymbolicResourceError::Missing("coherent callable".to_owned()))?;
        let offset = operation
            .operands
            .len()
            .checked_sub(first.parameters.len())
            .ok_or_else(|| SymbolicResourceError::Missing("coherent ABI".to_owned()))?;
        let arguments = operation.operands[offset..]
            .iter()
            .map(|operand| environment.get(operand).cloned().flatten())
            .collect::<Vec<_>>();
        let mut counts = Counts::zero(&mut self.arena)?;
        let mut timing = Some(Vec::new());
        let mut body_backed = Vec::new();
        let nested_mode = context.mode.controlled();
        let added_controls = self.observation_control_width(operation, offset, context)?;
        let nested_controls = context
            .controls
            .zip(added_controls)
            .and_then(|(outer, inner)| outer.checked_add(inner));
        let control_footprints =
            self.symbolic_control_footprints(first, operation, environment, context, offset)?;
        let dimension_arguments =
            self.coherent_call_dimensions(operation, first, context, offset)?;
        let owner = self
            .functions
            .get(context.owner)
            .copied()
            .ok_or_else(|| SymbolicResourceError::Missing("coherent owner".to_owned()))?;
        for (case_index, name) in names.into_iter().enumerate() {
            let callee = self
                .functions
                .get(name)
                .copied()
                .ok_or_else(|| SymbolicResourceError::Missing(format!("callable {name:?}")))?;
            let (mut nested, mut nested_timing) = if callee.external {
                let dimensions =
                    symbolic_dimension_bindings(callee, &arguments, &dimension_arguments);
                let nested = self.external_counts(callee, &arguments, &dimensions, nested_mode)?;
                let nested_timing =
                    self.external_timing(callee, &dimensions, nested_mode, &nested)?;
                (nested, nested_timing)
            } else {
                let transparent =
                    is_structural_coherent_wrapper(callee, &self.functions, &mut HashSet::new());
                let summary = self.analyze_function_mode(
                    callee,
                    &arguments,
                    &dimension_arguments,
                    nested_mode,
                    nested_controls,
                )?;
                let nested_timing = if transparent {
                    summary.timing
                } else {
                    body_backed.push(callee.symbol.clone());
                    None
                };
                (summary.counts, nested_timing)
            };
            if let Some(local) = nested_timing.take() {
                let mut roots = symbolic_coherent_roots(callee, operation, owner, offset)?;
                for root in roots.values_mut() {
                    *root = self.canonical_timing_root(root.clone());
                }
                nested_timing = Some(instantiate_symbolic_timing(
                    local,
                    &callee.symbol,
                    &roots,
                    &format!("{}:{}", context.owner, span_identity(operation)),
                ));
            }
            apply_control_footprints(&mut nested_timing, control_footprints.as_deref());
            self.observe_call(&mut nested, callee, nested_mode, nested_controls)?;
            self.price_control_predicate(
                &mut nested,
                operation,
                case_index,
                added_controls,
                context.controls,
            )?;
            counts.then(&nested, &self.error_bound_metrics, &mut self.arena)?;
            append_symbolic_timing(&mut timing, nested_timing);
        }
        let one = SymbolicCount::exact_value(self.arena.literal_u128(1)?);
        counts.opcodes.insert(operation.opcode.clone(), one);
        if !body_backed.is_empty() {
            let reason = format!(
                "body-backed controlled transform of {} has no exact gate-family or dependency-span profile",
                body_backed.join(", ")
            );
            counts.invalidate_body_backed_coherent(&reason, &mut self.arena)?;
            timing = None;
        }
        Ok((counts, timing))
    }

    pub(super) fn symbolic_control_footprints(
        &mut self,
        callee: &wire::FunctionProto,
        operation: &wire::OperationProto,
        environment: &HashMap<u64, Option<ValueFact>>,
        context: SymbolicFunctionContext<'_>,
        operand_offset: usize,
    ) -> Result<Option<Vec<SymbolicTimingFootprint>>, SymbolicResourceError> {
        let owner = self
            .functions
            .get(context.owner)
            .copied()
            .ok_or_else(|| SymbolicResourceError::Missing("coherent owner".to_owned()))?;
        let control = operation
            .operands
            .first()
            .ok_or_else(|| SymbolicResourceError::Missing("coherent control".to_owned()))?;
        let region = function_value_regions(owner)
            .get(control)
            .cloned()
            .ok_or_else(|| SymbolicResourceError::Missing("coherent control region".to_owned()))?;
        let type_id = self
            .operations
            .value_types(context.owner)
            .and_then(|types| types.get(control))
            .copied()
            .ok_or_else(|| SymbolicResourceError::Missing("coherent control type".to_owned()))?;
        let quantum_parameters = callee
            .parameters
            .iter()
            .enumerate()
            .filter(|(_, parameter)| {
                parameter
                    .value
                    .as_ref()
                    .is_some_and(|value| value.has_region)
            })
            .collect::<Vec<_>>();
        let [(target_index, _)] = quantum_parameters.as_slice() else {
            return Ok(None);
        };
        let target = operation
            .operands
            .get(operand_offset + target_index)
            .ok_or_else(|| SymbolicResourceError::Missing("coherent target".to_owned()))?;
        let result_region = operation
            .results
            .first()
            .and_then(|result| result.region.as_ref())
            .ok_or_else(|| SymbolicResourceError::Missing("coherent result region".to_owned()))?;
        let selectors = if operation.opcode == "qcore.control_call" && operand_offset == 2 {
            let component = operation
                .operands
                .get(1)
                .and_then(|operand| environment.get(operand))
                .and_then(Option::as_ref)
                .and_then(|value| int_fact(Some(value)))
                .map(|fact| vec![fact.expression].into_boxed_slice());
            let Some(component) = component else {
                return Ok(None);
            };
            Some(component)
        } else {
            None
        };
        let control_width = self.coherent_width(type_id, context.dimensions)?;
        let mut output_control = symbolic_timing_root(owner, result_region);
        output_control.projections.push("0".to_owned());
        let mut output_target = symbolic_timing_root(owner, result_region);
        output_target.projections.push("1".to_owned());
        let input_control = self.canonical_timing_root(symbolic_timing_root(owner, &region));
        let target_region = function_value_regions(owner)
            .get(target)
            .cloned()
            .ok_or_else(|| SymbolicResourceError::Missing("coherent target region".to_owned()))?;
        let input_target = self.canonical_timing_root(symbolic_timing_root(owner, &target_region));
        let aliases = [
            (output_control, input_control.clone()),
            (output_target, input_target),
        ];
        if aliases.iter().any(|(source, destination)| {
            self.timing_aliases
                .get(source)
                .is_some_and(|existing| existing != destination)
        }) {
            return Ok(None);
        }
        for (source, destination) in aliases {
            self.timing_aliases.insert(source, destination);
        }
        Ok(Some(vec![SymbolicTimingFootprint {
            root: input_control,
            selectors,
            width: control_width,
        }]))
    }

    pub(super) fn canonical_timing_root(&self, mut root: SymbolicTimingRoot) -> SymbolicTimingRoot {
        let mut visited = HashSet::new();
        while visited.insert(root.clone()) {
            let Some((source, target)) = self
                .timing_aliases
                .iter()
                .filter(|(source, _)| {
                    source.scope == root.scope
                        && source.root == root.root
                        && root.projections.starts_with(&source.projections)
                })
                .max_by_key(|(source, _)| source.projections.len())
            else {
                break;
            };
            let mut projections = target.projections.clone();
            projections.extend_from_slice(&root.projections[source.projections.len()..]);
            let mut instance = target.instance.clone();
            instance.extend_from_slice(&root.instance);
            root = SymbolicTimingRoot {
                scope: target.scope.clone(),
                root: target.root,
                projections,
                instance,
            };
        }
        root
    }

    pub(super) fn analyze_coherent_adjoint(
        &mut self,
        operation: &wire::OperationProto,
        environment: &HashMap<u64, Option<ValueFact>>,
    ) -> Result<Counts, SymbolicResourceError> {
        let original = string_attribute(operation, "original")
            .ok_or_else(|| SymbolicResourceError::Missing("coherent adjoint opcode".to_owned()))?;
        let names = if original == "qcore.control_call" {
            vec![
                mapping_string_attribute(operation, "original_attributes", "callee").ok_or_else(
                    || SymbolicResourceError::Missing("control adjoint callee".to_owned()),
                )?,
            ]
        } else {
            mapping_string_tuple_attribute(operation, "original_attributes", "callees").ok_or_else(
                || SymbolicResourceError::Missing("select adjoint callees".to_owned()),
            )?
        };
        let first = names
            .first()
            .and_then(|name| self.functions.get(name).copied())
            .ok_or_else(|| {
                SymbolicResourceError::Missing("coherent adjoint callable".to_owned())
            })?;
        let supplied = operation.operands.get(1..).unwrap_or_default();
        let component_count = if original == "qcore.control_call" {
            supplied
                .len()
                .checked_add(2)
                .and_then(|count| count.checked_sub(first.parameters.len()))
                .and_then(|offset| offset.checked_sub(1))
                .filter(|count| *count <= 1)
                .ok_or_else(|| SymbolicResourceError::Missing("control adjoint ABI".to_owned()))?
        } else {
            0
        };
        let mut classical = supplied
            .get(component_count..)
            .ok_or_else(|| SymbolicResourceError::Missing("coherent adjoint arguments".to_owned()))?
            .iter();
        let arguments = first
            .parameters
            .iter()
            .map(|parameter| {
                let value = parameter.value.as_ref().ok_or_else(|| {
                    SymbolicResourceError::Missing("coherent adjoint parameter".to_owned())
                })?;
                if value.has_region {
                    Ok(None)
                } else {
                    let operand = classical.next().ok_or_else(|| {
                        SymbolicResourceError::Missing("coherent adjoint argument".to_owned())
                    })?;
                    Ok(environment.get(operand).cloned().flatten())
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        if classical.next().is_some() {
            return Err(SymbolicResourceError::Missing(
                "coherent adjoint argument ABI".to_owned(),
            ));
        }
        let mut direct = Counts::zero(&mut self.arena)?;
        for name in names {
            let callee = self
                .functions
                .get(name)
                .copied()
                .ok_or_else(|| SymbolicResourceError::Missing(format!("callable {name:?}")))?;
            direct.then(
                &self
                    .analyze_function(callee, &arguments, &HashMap::new())?
                    .counts,
                &self.error_bound_metrics,
                &mut self.arena,
            )?;
        }
        let zero = SymbolicCount::exact_value(self.arena.literal_u128(0)?);
        let work = direct.work.clone();
        Ok(Counts {
            work: work.clone(),
            span: direct.span,
            depths: None,
            opcodes: BTreeMap::from([("qcore.adjoint".to_owned(), work.clone())]),
            extra_metrics: direct.extra_metrics,
            families: None,
            measurements: Some(zero.clone()),
            resets: Some(zero.clone()),
            cleanup_work: if string_attribute(operation, "proof") == Some("modular-adjoint") {
                work
            } else {
                zero
            },
        })
    }
}
