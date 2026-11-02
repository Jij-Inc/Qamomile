//! Portable gate pricing for symbolic observations.

use crate::lowering::approximation::{
    ApproximationCertificate, OUTPUT_TRACE_DISTANCE_BOUND_METRIC,
};
use crate::lowering::realization::{Realization, RealizationFamily};
use crate::operations::logical_gate_families;
use crate::resource_rules::{Domain as ResourceDomain, TransformMode};
use crate::resources::analysis::{is_measurement, is_reset};
use crate::wire;

use super::{
    Counts, ExprId, FRONTIER_PREFIX, PORTABLE_PREFIX, ParametricDependencyScheduler,
    ParametricGateProfile, SymbolicAnalyzer, SymbolicCount, SymbolicFunctionContext,
    SymbolicResourceError, attribute, external_family_metrics, natural_literal,
    small_permutation_step_bound, transform_mode_name,
};

impl SymbolicAnalyzer<'_> {
    pub(super) fn finish_portable_counts(
        &mut self,
        counts: &mut Counts,
    ) -> Result<(), SymbolicResourceError> {
        if self.observation.is_none_or(|policy| policy.basis.is_none()) {
            return Ok(());
        }
        let zero = SymbolicCount::exact_value(self.arena.literal_u128(0)?);
        for (family, _) in external_family_metrics() {
            counts
                .extra_metrics
                .entry(format!("{PORTABLE_PREFIX}{family}"))
                .or_insert_with(|| zero.clone());
        }
        let gates = counts.extra_metrics[&format!("{PORTABLE_PREFIX}gates")].clone();
        // Serial execution bounds depth, including shared-control dependencies.
        let depth = SymbolicCount {
            lower: zero.lower,
            exact: gates.upper == Some(zero.lower),
            ..gates.clone()
        };
        counts
            .extra_metrics
            .entry(format!("{PORTABLE_PREFIX}gate_depth"))
            .or_insert(depth);
        let required_scratch = counts
            .extra_metrics
            .get(&format!("{PORTABLE_PREFIX}clean_ancilla_qubits"))
            .cloned()
            .unwrap_or_else(|| zero.clone());
        let scratch = if gates.unknown.is_empty() {
            required_scratch
        } else {
            SymbolicCount::unbounded(
                &required_scratch,
                "unrealized operations have no workspace contract",
            )
        };
        counts
            .extra_metrics
            .insert(format!("{PORTABLE_PREFIX}clean_ancilla_qubits"), scratch);
        Ok(())
    }

    pub(super) fn observation_control_width(
        &mut self,
        operation: &wire::OperationProto,
        offset: usize,
        context: SymbolicFunctionContext<'_>,
    ) -> Result<Option<u32>, SymbolicResourceError> {
        if self.observation.is_none() {
            return Ok(Some(0));
        }
        if operation.opcode == "qcore.control_call" && offset == 2 {
            return Ok(Some(1));
        }
        let type_id = operation.operands.first().and_then(|operand| {
            self.operations
                .value_types(context.owner)?
                .get(operand)
                .copied()
        });
        match type_id {
            Some(type_id) => Ok(self
                .coherent_width(type_id, context.dimensions)?
                .and_then(|width| self.arena.integer_value(width))
                .and_then(|width| u32::try_from(width).ok())),
            None => Ok(None),
        }
    }

    pub(super) fn price_control_predicate(
        &mut self,
        counts: &mut Counts,
        operation: &wire::OperationProto,
        case_index: usize,
        width: Option<u32>,
        outer_controls: Option<u32>,
    ) -> Result<(), SymbolicResourceError> {
        use crate::control_state::ControlState;
        if self.observation.is_none_or(|policy| policy.basis.is_none()) {
            return Ok(());
        }
        let state = if operation.opcode == "qcore.select_call" {
            Some(ControlState::exact_u64(
                u64::try_from(case_index).map_err(|_| SymbolicResourceError::Overflow)?,
            ))
        } else {
            attribute(operation, "control_state").and_then(ControlState::decode)
        };
        match (state, width) {
            (Some(ControlState::AllActive), _) => Ok(()),
            (Some(ControlState::Exact(value)), Some(width)) => {
                let inactive = u64::from(width)
                    .checked_sub(value.count_ones())
                    .ok_or(SymbolicResourceError::Overflow)?;
                if inactive == 0 {
                    return Ok(());
                }
                let multiplicity = self.arena.literal_u128(u128::from(inactive) * 2)?;
                self.price_primitive(counts, "quantum.x", outer_controls, multiplicity)
            }
            _ => self.portable_unknown(
                counts,
                "symbolic control predicate normalization requires a width",
            ),
        }
    }

    pub(super) fn observation_name(
        kind: &str,
        name: &str,
        mode: TransformMode,
        controls: Option<u32>,
    ) -> String {
        let controls = controls.map_or_else(|| "symbolic".to_owned(), |count| count.to_string());
        format!(
            "{kind}.{}.c{controls}.{name}",
            transform_mode_name(mode).replace('-', "_")
        )
    }

    pub(super) fn portable_unknown(
        &mut self,
        counts: &mut Counts,
        reason: &str,
    ) -> Result<(), SymbolicResourceError> {
        let zero = SymbolicCount::exact_value(self.arena.literal_u128(0)?);
        for (family, _) in external_family_metrics() {
            let key = format!("{PORTABLE_PREFIX}{family}");
            let known = counts.extra_metrics.get(&key).unwrap_or(&zero);
            counts
                .extra_metrics
                .insert(key, SymbolicCount::unbounded(known, reason));
        }
        Ok(())
    }

    pub(super) fn price_primitive(
        &mut self,
        counts: &mut Counts,
        opcode: &str,
        controls: Option<u32>,
        multiplicity: ExprId,
    ) -> Result<(), SymbolicResourceError> {
        let Some(basis) = self.observation.and_then(|policy| policy.basis) else {
            return Ok(());
        };
        // Allocation and terminal operations have separate source metrics.
        if controls == Some(0)
            && (is_measurement(opcode)
                || is_reset(opcode)
                || matches!(
                    opcode,
                    "quantum.alloc" | "quantum.discard" | "quantum.release"
                ))
        {
            return Ok(());
        }
        let recipe = controls.and_then(|controls| {
            crate::resources::portable::portable_gate_recipe(opcode, controls, basis)
        });
        let Some(recipe) = recipe else {
            if let Some(controls) = controls
                && matches!(
                    opcode,
                    "quantum.x" | "quantum.cx" | "quantum.ccx" | "quantum.swap"
                )
            {
                return self.price_controlled_classical_primitive(
                    counts,
                    opcode,
                    controls,
                    multiplicity,
                );
            }
            return self.portable_unknown(
                counts,
                &format!("{opcode} with {controls:?} controls: no portable realization"),
            );
        };
        let zero = SymbolicCount::exact_value(self.arena.literal_u128(0)?);
        let unit = SymbolicCount::exact_value(multiplicity);
        let mut gate_terms = Vec::with_capacity(recipe.len());
        for step in recipe {
            gate_terms.push(multiplicity);
            for family in logical_gate_families(step.opcode()) {
                let key = format!("{PORTABLE_PREFIX}{family}");
                let value = counts
                    .extra_metrics
                    .get(&key)
                    .unwrap_or(&zero)
                    .then(&unit, &mut self.arena)?;
                counts.extra_metrics.insert(key, value);
            }
        }
        let serial_depth = self.arena.add(gate_terms)?;
        counts.extra_metrics.insert(
            format!("{PORTABLE_PREFIX}gate_depth"),
            SymbolicCount::bounded_by(&SymbolicCount::exact_value(serial_depth), &mut self.arena)?,
        );
        Ok(())
    }

    pub(super) fn price_controlled_classical_primitive(
        &mut self,
        counts: &mut Counts,
        opcode: &str,
        controls: u32,
        multiplicity: ExprId,
    ) -> Result<(), SymbolicResourceError> {
        let (gates, widest_control) = match opcode {
            "quantum.x" => (
                self.price_mcx_exact(counts, controls, multiplicity)?,
                controls,
            ),
            "quantum.cx" => {
                let total = controls
                    .checked_add(1)
                    .ok_or(SymbolicResourceError::Overflow)?;
                (self.price_mcx_exact(counts, total, multiplicity)?, total)
            }
            "quantum.ccx" => {
                let total = controls
                    .checked_add(2)
                    .ok_or(SymbolicResourceError::Overflow)?;
                (self.price_mcx_exact(counts, total, multiplicity)?, total)
            }
            "quantum.swap" => {
                let controlled_x_arity = controls
                    .checked_add(1)
                    .ok_or(SymbolicResourceError::Overflow)?;
                let toffoli_arity = controls
                    .checked_add(2)
                    .ok_or(SymbolicResourceError::Overflow)?;
                let two = self.arena.literal_u128(2)?;
                let outer = self.arena.multiply([multiplicity, two])?;
                let cx = self.price_mcx_exact(counts, controlled_x_arity, outer)?;
                let ccx = self.price_mcx_exact(counts, toffoli_arity, multiplicity)?;
                (self.arena.add([cx, ccx])?, toffoli_arity)
            }
            _ => {
                return self.portable_unknown(
                    counts,
                    &format!("{opcode}: no controlled classical realization"),
                );
            }
        };
        counts.extra_metrics.insert(
            format!("{PORTABLE_PREFIX}gate_depth"),
            SymbolicCount::bounded_by(&SymbolicCount::exact_value(gates), &mut self.arena)?,
        );
        let scratch = self
            .arena
            .literal_u128(u128::from(widest_control.saturating_sub(2)))?;
        counts.extra_metrics.insert(
            format!("{PORTABLE_PREFIX}clean_ancilla_qubits"),
            SymbolicCount::exact_value(scratch),
        );
        Ok(())
    }

    pub(super) fn add_portable_upper(
        &mut self,
        counts: &mut Counts,
        family: &str,
        expression: ExprId,
    ) -> Result<(), SymbolicResourceError> {
        let exact = SymbolicCount::exact_value(expression);
        let contribution = SymbolicCount::bounded_by(&exact, &mut self.arena)?;
        let key = format!("{PORTABLE_PREFIX}{family}");
        let merged = counts.extra_metrics.get(&key).map_or_else(
            || Ok(contribution.clone()),
            |current| current.then(&contribution, &mut self.arena),
        )?;
        counts.extra_metrics.insert(key, merged);
        Ok(())
    }

    pub(super) fn add_portable_exact(
        &mut self,
        counts: &mut Counts,
        family: &str,
        expression: ExprId,
    ) -> Result<(), SymbolicResourceError> {
        let contribution = SymbolicCount::exact_value(expression);
        let key = format!("{PORTABLE_PREFIX}{family}");
        let merged = counts.extra_metrics.get(&key).map_or_else(
            || Ok(contribution.clone()),
            |current| current.then(&contribution, &mut self.arena),
        )?;
        counts.extra_metrics.insert(key, merged);
        Ok(())
    }

    pub(super) fn price_mcx_upper(
        &mut self,
        counts: &mut Counts,
        controls: u32,
        multiplicity: ExprId,
    ) -> Result<ExprId, SymbolicResourceError> {
        let basis = self
            .observation
            .and_then(|observation| observation.basis)
            .ok_or_else(|| SymbolicResourceError::Missing("portable basis".to_owned()))?;
        let (opcode, copies) = match controls {
            0 => ("quantum.x", 1),
            1 => ("quantum.cx", 1),
            2 => ("quantum.ccx", 1),
            value => (
                "quantum.ccx",
                value
                    .checked_mul(2)
                    .and_then(|value| value.checked_sub(3))
                    .ok_or(SymbolicResourceError::Overflow)?,
            ),
        };
        let copies = self.arena.literal_u128(u128::from(copies))?;
        let amount = self.arena.multiply([multiplicity, copies])?;
        let recipe = crate::resources::portable::portable_gate_recipe(opcode, 0, basis)
            .ok_or_else(|| SymbolicResourceError::Missing("portable MCX recipe".to_owned()))?;
        let mut gates = Vec::with_capacity(recipe.len());
        for step in recipe {
            gates.push(amount);
            for family in logical_gate_families(step.opcode()) {
                self.add_portable_upper(counts, family, amount)?;
            }
        }
        self.arena.add(gates)
    }

    pub(super) fn price_mcx_exact(
        &mut self,
        counts: &mut Counts,
        controls: u32,
        multiplicity: ExprId,
    ) -> Result<ExprId, SymbolicResourceError> {
        let basis = self
            .observation
            .and_then(|observation| observation.basis)
            .ok_or_else(|| SymbolicResourceError::Missing("portable basis".to_owned()))?;
        let (opcode, copies) = match controls {
            0 => ("quantum.x", 1),
            1 => ("quantum.cx", 1),
            2 => ("quantum.ccx", 1),
            value => (
                "quantum.ccx",
                value
                    .checked_mul(2)
                    .and_then(|value| value.checked_sub(3))
                    .ok_or(SymbolicResourceError::Overflow)?,
            ),
        };
        let copies = self.arena.literal_u128(u128::from(copies))?;
        let amount = self.arena.multiply([multiplicity, copies])?;
        let recipe = crate::resources::portable::portable_gate_recipe(opcode, 0, basis)
            .ok_or_else(|| SymbolicResourceError::Missing("portable MCX recipe".to_owned()))?;
        let mut gates = Vec::with_capacity(recipe.len());
        for step in recipe {
            gates.push(amount);
            for family in logical_gate_families(step.opcode()) {
                self.add_portable_exact(counts, family, amount)?;
            }
        }
        self.arena.add(gates)
    }

    pub(super) fn price_ripple_constant_adders(
        &mut self,
        counts: &mut Counts,
        controls: u32,
        width: ExprId,
        repetitions: ExprId,
    ) -> Result<ParametricGateProfile, SymbolicResourceError> {
        let twice = self.arena.literal_u128(2)?;
        let four = self.arena.literal_u128(4)?;
        let base = self.arena.multiply([width, repetitions])?;
        let prepared = self.arena.multiply([base, twice])?;
        let cnot = self.arena.multiply([base, four])?;
        let toffoli = self.arena.multiply([base, twice])?;
        let cnot_controls = controls
            .checked_add(1)
            .ok_or(SymbolicResourceError::Overflow)?;
        let toffoli_controls = controls
            .checked_add(2)
            .ok_or(SymbolicResourceError::Overflow)?;
        let prepared_gates = self.price_mcx_upper(counts, controls, prepared)?;
        let cnot_gates = self.price_mcx_exact(counts, cnot_controls, cnot)?;
        let toffoli_gates = self.price_mcx_exact(counts, toffoli_controls, toffoli)?;
        let gates = self
            .arena
            .add([prepared_gates, cnot_gates, toffoli_gates])?;

        // With no surrounding coherent control, preparation of the constant
        // word occupies disjoint lanes and takes one layer in each direction.
        // A surrounding control is a shared scheduler port and therefore
        // serializes those flips. The Cuccaro carry path is serial in either
        // case. This is an upper profile of the actual emitted dependency DAG,
        // not an assumption that all gate work is sequential.
        let preparation_depth = if controls == 0 {
            self.arena.multiply([twice, repetitions])?
        } else {
            prepared_gates
        };
        let makespan = self
            .arena
            .add([preparation_depth, cnot_gates, toffoli_gates])?;
        Ok(ParametricGateProfile { gates, makespan })
    }

    pub(super) fn price_ripple_register_adders(
        &mut self,
        counts: &mut Counts,
        controls: u32,
        width: ExprId,
        repetitions: ExprId,
    ) -> Result<ParametricGateProfile, SymbolicResourceError> {
        let twice = self.arena.literal_u128(2)?;
        let four = self.arena.literal_u128(4)?;
        let base = self.arena.multiply([width, repetitions])?;
        let cnot = self.arena.multiply([base, four])?;
        let toffoli = self.arena.multiply([base, twice])?;
        let cnot_controls = controls
            .checked_add(1)
            .ok_or(SymbolicResourceError::Overflow)?;
        let toffoli_controls = controls
            .checked_add(2)
            .ok_or(SymbolicResourceError::Overflow)?;
        let cnot_gates = self.price_mcx_exact(counts, cnot_controls, cnot)?;
        let toffoli_gates = self.price_mcx_exact(counts, toffoli_controls, toffoli)?;
        let gates = self.arena.add([cnot_gates, toffoli_gates])?;
        // Every majority/unmajority block lies on the retained carry path.
        Ok(ParametricGateProfile {
            gates,
            makespan: gates,
        })
    }

    pub(super) fn record_approximation(
        &mut self,
        counts: &mut Counts,
        certificate: ApproximationCertificate<ExprId>,
    ) -> Result<(), SymbolicResourceError> {
        let value = match certificate {
            ApproximationCertificate::Exact => {
                SymbolicCount::exact_value(self.arena.literal_u128(0)?)
            }
            ApproximationCertificate::OutputTraceDistanceBound(upper) => {
                SymbolicCount::bounded_by(&SymbolicCount::exact_value(upper), &mut self.arena)?
            }
        };
        counts
            .extra_metrics
            .insert(OUTPUT_TRACE_DISTANCE_BOUND_METRIC.to_owned(), value);
        Ok(())
    }

    pub(super) fn bound_factor_dependent_profile(
        &mut self,
        counts: &mut Counts,
    ) -> Result<(), SymbolicResourceError> {
        // Unknown classical factors may be identities. Their suppression removes
        // the entire schedule and workspace, so these are only upper bounds.
        for (name, value) in &mut counts.extra_metrics {
            if name.starts_with(PORTABLE_PREFIX) {
                *value = SymbolicCount::dependency_bounded_by(value, &mut self.arena)?;
            }
        }
        Ok(())
    }

    pub(super) fn price_mod_mul_child(
        &mut self,
        counts: &mut Counts,
        controls: Option<u32>,
        repetitions: ExprId,
        width: ExprId,
    ) -> Result<ParametricGateProfile, SymbolicResourceError> {
        let config = self
            .observation
            .expect("portable observation checked")
            .realizations;
        let known_width = natural_literal(&self.arena, width);
        if let Some(known_width) = known_width {
            if config.select_mod_mul_child(known_width).kind() == Realization::ArithmeticPortable {
                return self.price_small_mod_mul(counts, controls, repetitions, known_width);
            }
            return self.price_schoolbook_mod_mul(counts, controls, repetitions, Some(width));
        }
        let schoolbook =
            self.price_schoolbook_mod_mul(counts, controls, repetitions, Some(width))?;
        if config.select_mod_mul_child(1).kind() != Realization::ArithmeticPortable {
            return Ok(schoolbook);
        }
        // Unresolved widths may choose either route. Summing nonnegative work
        // bounds encloses both without enumerating either circuit or assuming
        // that the schoolbook polynomial dominates small permutation synthesis.
        let scratch_key = format!("{PORTABLE_PREFIX}clean_ancilla_qubits");
        let schoolbook_scratch = counts.extra_metrics.get(&scratch_key).cloned();
        let small = self.price_small_mod_mul(counts, controls, repetitions, 3)?;
        if let Some(schoolbook_scratch) = schoolbook_scratch {
            let small_scratch = counts.extra_metrics[&scratch_key].clone();
            counts.extra_metrics.insert(
                scratch_key,
                SymbolicCount::branch(&[schoolbook_scratch, small_scratch], &mut self.arena)?,
            );
        }
        let gates = self.arena.add([schoolbook.gates, small.gates])?;
        let makespan = self.arena.add([schoolbook.makespan, small.makespan])?;
        counts.extra_metrics.insert(
            format!("{PORTABLE_PREFIX}gate_depth"),
            SymbolicCount::bounded_by(&SymbolicCount::exact_value(makespan), &mut self.arena)?,
        );
        Ok(ParametricGateProfile { gates, makespan })
    }

    pub(super) fn price_small_mod_mul(
        &mut self,
        counts: &mut Counts,
        controls: Option<u32>,
        repetitions: ExprId,
        width: u64,
    ) -> Result<ParametricGateProfile, SymbolicResourceError> {
        let Some(controls) = controls else {
            self.portable_unknown(
                counts,
                "small modular multiplication requires a known control width",
            )?;
            let zero = self.arena.literal_u128(0)?;
            return Ok(ParametricGateProfile {
                gates: zero,
                makespan: zero,
            });
        };
        let steps = small_permutation_step_bound(width)?;
        let steps = self.arena.literal_u128(steps)?;
        let repetitions = self.arena.multiply([steps, repetitions])?;
        let width = u32::try_from(width).map_err(|_| SymbolicResourceError::Overflow)?;
        let mut terms = Vec::new();
        // Bound every possible arity separately: pricing only the widest MCX
        // would not enclose each gate family in every declared portable basis.
        for intrinsic in 0..width {
            let total = controls
                .checked_add(intrinsic)
                .ok_or(SymbolicResourceError::Overflow)?;
            terms.push(self.price_mcx_upper(counts, total, repetitions)?);
        }
        let gates = self.arena.add(terms)?;
        counts.extra_metrics.insert(
            format!("{PORTABLE_PREFIX}gate_depth"),
            SymbolicCount::bounded_by(&SymbolicCount::exact_value(gates), &mut self.arena)?,
        );
        let widest = controls
            .checked_add(width - 1)
            .ok_or(SymbolicResourceError::Overflow)?;
        let scratch = self
            .arena
            .literal_u128(u128::from(widest.saturating_sub(2)))?;
        counts.extra_metrics.insert(
            format!("{PORTABLE_PREFIX}clean_ancilla_qubits"),
            SymbolicCount::dependency_bounded_by(
                &SymbolicCount::exact_value(scratch),
                &mut self.arena,
            )?,
        );
        Ok(ParametricGateProfile {
            gates,
            makespan: gates,
        })
    }

    pub(super) fn price_schoolbook_mod_mul(
        &mut self,
        counts: &mut Counts,
        controls: Option<u32>,
        multiplicity: ExprId,
        width: Option<ExprId>,
    ) -> Result<ParametricGateProfile, SymbolicResourceError> {
        let (Some(controls), Some(width)) = (controls, width) else {
            self.portable_unknown(
                counts,
                "schoolbook modular multiplication requires a known control and target width",
            )?;
            let zero = self.arena.literal_u128(0)?;
            return Ok(ParametricGateProfile {
                gates: zero,
                makespan: zero,
            });
        };
        let one = self.arena.literal_u128(1)?;
        let two = self.arena.literal_u128(2)?;
        let four = self.arena.literal_u128(4)?;
        let eight = self.arena.literal_u128(8)?;
        let extended = self.arena.add([width, one])?;
        let repeated_width = self.arena.multiply([width, multiplicity])?;
        let controls_one = controls
            .checked_add(1)
            .ok_or(SymbolicResourceError::Overflow)?;
        let controls_two = controls
            .checked_add(2)
            .ok_or(SymbolicResourceError::Overflow)?;
        let controls_three = controls
            .checked_add(3)
            .ok_or(SymbolicResourceError::Overflow)?;

        // Two n-step accumulator passes. Each modular addition contains four
        // ordinary and one flag-controlled constant ripple adders.
        let ordinary_adders = self.arena.multiply([repeated_width, eight])?;
        let conditional_adders = self.arena.multiply([repeated_width, two])?;
        let ordinary =
            self.price_ripple_constant_adders(counts, controls_two, extended, ordinary_adders)?;
        let conditional = self.price_ripple_constant_adders(
            counts,
            controls_three,
            extended,
            conditional_adders,
        )?;
        let mut gate_terms = vec![ordinary.gates, conditional.gates];
        let mut dependency_durations = vec![ordinary.makespan, conditional.makespan];

        // Reduction flags inside each modular addition.
        let flag_flip_work = self.arena.multiply([repeated_width, four])?;
        let flag_flip = self.price_mcx_exact(counts, controls_two, flag_flip_work)?;
        gate_terms.push(flag_flip);
        dependency_durations.push(flag_flip);
        let controlled_flag_work = self.arena.multiply([repeated_width, four])?;
        let controlled_flag = self.price_mcx_exact(counts, controls_three, controlled_flag_work)?;
        gate_terms.push(controlled_flag);
        dependency_durations.push(controlled_flag);

        // Compute and uncompute the total-domain x < modulus guard.
        let guard_adders = self.arena.multiply([multiplicity, four])?;
        let guard = self.price_ripple_constant_adders(counts, controls, extended, guard_adders)?;
        gate_terms.push(guard.gates);
        dependency_durations.push(guard.makespan);
        let guard_cx = self.arena.multiply([multiplicity, two])?;
        let guard_flip = self.price_mcx_exact(counts, controls_one, guard_cx)?;
        gate_terms.push(guard_flip);
        dependency_durations.push(guard_flip);

        // Controlled swap between the input and accumulator registers.
        let swap_cx = self.arena.multiply([repeated_width, two])?;
        let swap_outer = self.price_mcx_exact(counts, controls_one, swap_cx)?;
        let swap_inner = self.price_mcx_exact(counts, controls_two, repeated_width)?;
        gate_terms.extend([swap_outer, swap_inner]);
        dependency_durations.extend([swap_outer, swap_inner]);

        let gates = self.arena.add(gate_terms)?;
        let mut scheduler = ParametricDependencyScheduler::new(&mut self.arena)?;
        for duration in dependency_durations {
            // State, accumulator, and the reused clean workspace are explicit
            // shared ports. Independent layers inside each ripple profile have
            // already been joined by that profile's local dependency rule.
            scheduler.step(&[0, 1, 2], duration, &mut self.arena)?;
        }
        let makespan = scheduler.makespan();
        let depth_key = format!("{PORTABLE_PREFIX}gate_depth");
        counts.extra_metrics.insert(
            depth_key,
            SymbolicCount::bounded_by(&SymbolicCount::exact_value(makespan), &mut self.arena)?,
        );

        // accumulator n + guard/reduction/overflow 3 + constant word n+1 +
        // carry 1 + the longest clean MCX chain (controls+3).
        let fixed = self
            .arena
            .literal_u128(u128::from(controls).saturating_add(8))?;
        let twice_width = self.arena.multiply([width, two])?;
        let scratch = self.arena.add([twice_width, fixed])?;
        counts.extra_metrics.insert(
            format!("{PORTABLE_PREFIX}clean_ancilla_qubits"),
            SymbolicCount::exact_value(scratch),
        );
        self.bound_factor_dependent_profile(counts)?;
        Ok(ParametricGateProfile { gates, makespan })
    }

    pub(super) fn price_schoolbook_mod_mul_add(
        &mut self,
        counts: &mut Counts,
        controls: Option<u32>,
        multiplicity: ExprId,
        total_width: Option<ExprId>,
    ) -> Result<ParametricGateProfile, SymbolicResourceError> {
        let (Some(controls), Some(total_width)) = (controls, total_width) else {
            self.portable_unknown(
                counts,
                "schoolbook modular multiply-add requires known control and register widths",
            )?;
            let zero = self.arena.literal_u128(0)?;
            return Ok(ParametricGateProfile {
                gates: zero,
                makespan: zero,
            });
        };
        let one = self.arena.literal_u128(1)?;
        let two = self.arena.literal_u128(2)?;
        let three = self.arena.literal_u128(3)?;
        let four = self.arena.literal_u128(4)?;
        let six = self.arena.literal_u128(6)?;
        let twelve = self.arena.literal_u128(12)?;
        let width = self.arena.floor_divide(total_width, three)?;
        let width = self.arena.absolute(width, ResourceDomain::Natural)?;
        let extended = self.arena.add([width, one])?;
        let twice_width = self.arena.multiply([width, two])?;

        // Compute and uncompute the ordinary 2n-bit product. Each of the n
        // multiplier bits drives one retained register addition whose width is
        // at most 2n. Three valid-domain flags and the multiplier bit are local
        // controls.
        let product_adders = self.arena.multiply([width, two, multiplicity])?;
        let local_controls = controls
            .checked_add(4)
            .ok_or(SymbolicResourceError::Overflow)?;
        let product =
            self.price_ripple_register_adders(counts, local_controls, twice_width, product_adders)?;
        let mut gate_terms = vec![product.gates];
        let mut dependency_durations = vec![product.makespan];

        // Every one of the 2n product bits contributes one precomputed residue.
        // A clean modular constant addition has four ordinary ripple additions,
        // one reduction-controlled ripple addition, and four flag updates. The
        // full-width upper bound is independent of the modulus bit pattern.
        let residue_updates = self.arena.multiply([twice_width, multiplicity])?;
        let ordinary_residue_adders = self.arena.multiply([residue_updates, four])?;
        let ordinary_residues = self.price_ripple_constant_adders(
            counts,
            local_controls,
            extended,
            ordinary_residue_adders,
        )?;
        gate_terms.push(ordinary_residues.gates);
        dependency_durations.push(ordinary_residues.makespan);
        let reduction_controls = local_controls
            .checked_add(1)
            .ok_or(SymbolicResourceError::Overflow)?;
        let conditional_residues = self.price_ripple_constant_adders(
            counts,
            reduction_controls,
            extended,
            residue_updates,
        )?;
        gate_terms.push(conditional_residues.gates);
        dependency_durations.push(conditional_residues.makespan);
        let reduction_updates = self.arena.multiply([residue_updates, two])?;
        let ordinary_reductions =
            self.price_mcx_upper(counts, local_controls, reduction_updates)?;
        let controlled_reductions =
            self.price_mcx_upper(counts, reduction_controls, reduction_updates)?;
        gate_terms.extend([ordinary_reductions, controlled_reductions]);
        dependency_durations.extend([ordinary_reductions, controlled_reductions]);

        // Three total-domain guards are computed and cleared. Each comparison
        // consists of a forward and inverse constant ripple addition.
        let guard_adders = self.arena.multiply([multiplicity, twelve])?;
        let guards = self.price_ripple_constant_adders(counts, controls, extended, guard_adders)?;
        gate_terms.push(guards.gates);
        dependency_durations.push(guards.makespan);
        let guard_updates = self.arena.multiply([multiplicity, six])?;
        let guard_updates = self.price_mcx_upper(
            counts,
            controls
                .checked_add(1)
                .ok_or(SymbolicResourceError::Overflow)?,
            guard_updates,
        )?;
        gate_terms.push(guard_updates);
        dependency_durations.push(guard_updates);

        let gates = self.arena.add(gate_terms)?;
        let mut scheduler = ParametricDependencyScheduler::new(&mut self.arena)?;
        for duration in dependency_durations {
            // Product, accumulator, validity flags, and clean arithmetic
            // workspace are reused by the emitted implementation.
            scheduler.step(&[0, 1, 2, 3], duration, &mut self.arena)?;
        }
        let makespan = scheduler.makespan();
        counts.extra_metrics.insert(
            format!("{PORTABLE_PREFIX}gate_depth"),
            SymbolicCount::bounded_by(&SymbolicCount::exact_value(makespan), &mut self.arena)?,
        );
        let fixed = self
            .arena
            .literal_u128(u128::from(controls).saturating_add(12))?;
        let four_width = self.arena.multiply([width, four])?;
        let scratch = self.arena.add([four_width, fixed])?;
        counts.extra_metrics.insert(
            format!("{PORTABLE_PREFIX}clean_ancilla_qubits"),
            SymbolicCount::dependency_bounded_by(
                &SymbolicCount::exact_value(scratch),
                &mut self.arena,
            )?,
        );
        Ok(ParametricGateProfile { gates, makespan })
    }

    pub(super) fn price_schoolbook_mul_add(
        &mut self,
        counts: &mut Counts,
        controls: Option<u32>,
        multiplicity: ExprId,
        total_width: Option<ExprId>,
    ) -> Result<ParametricGateProfile, SymbolicResourceError> {
        let (Some(controls), Some(total_width)) = (controls, total_width) else {
            self.portable_unknown(
                counts,
                "schoolbook multiply-add requires known control and factor widths",
            )?;
            let zero = self.arena.literal_u128(0)?;
            return Ok(ParametricGateProfile {
                gates: zero,
                makespan: zero,
            });
        };
        let one = self.arena.literal_u128(1)?;
        let four = self.arena.literal_u128(4)?;
        let factor_width = self.arena.floor_divide(total_width, four)?;
        let factor_width = self.arena.absolute(factor_width, ResourceDomain::Natural)?;
        let square = self.arena.multiply([factor_width, factor_width])?;
        let successor = self.arena.add([factor_width, one])?;
        let steps = self.arena.multiply([square, successor, multiplicity])?;

        // The portable shift-add plan emits n^2(n+1) controlled flips. Its
        // longest selector contains both factor bits plus at most n carry bits.
        // Symbolic widths cannot inhabit the fixed u32 control-count field, so
        // price the clean-MCX chain algebraically: at most 2n+1 Toffolis per
        // controlled flip, including any fixed surrounding controls.
        let two = self.arena.literal_u128(2)?;
        let control_fixed = self.arena.literal_u128(u128::from(controls) + 1)?;
        let variable_chain = self.arena.multiply([factor_width, two])?;
        let chain = self.arena.add([variable_chain, control_fixed])?;
        let gates = self.arena.multiply([steps, chain])?;
        self.add_portable_upper(counts, "gates", gates)?;
        self.add_portable_upper(counts, "toffoli_gates", gates)?;
        counts.extra_metrics.insert(
            format!("{PORTABLE_PREFIX}gate_depth"),
            SymbolicCount::bounded_by(&SymbolicCount::exact_value(gates), &mut self.arena)?,
        );
        let fixed_scratch = self.arena.literal_u128(u128::from(controls))?;
        let scratch = self.arena.add([factor_width, fixed_scratch])?;
        counts.extra_metrics.insert(
            format!("{PORTABLE_PREFIX}clean_ancilla_qubits"),
            SymbolicCount::exact_value(scratch),
        );
        Ok(ParametricGateProfile {
            gates,
            // The portable shift-add realization reuses the product register
            // and clean MCX chain for every controlled flip, so this serial
            // critical path is induced by its dependencies rather than used as
            // a default for its callers. Independent tree nodes are scheduled
            // separately by the enclosing realization graph.
            makespan: gates,
        })
    }

    pub(super) fn price_computational_basis_preparation(
        &mut self,
        counts: &mut Counts,
        controls: Option<u32>,
        width: Option<ExprId>,
    ) -> Result<(), SymbolicResourceError> {
        let (Some(controls), Some(width)) = (controls, width) else {
            return self.portable_unknown(
                counts,
                "computational-basis preparation requires known control and value widths",
            );
        };
        let gates = self.price_mcx_upper(counts, controls, width)?;
        counts.extra_metrics.insert(
            format!("{PORTABLE_PREFIX}gate_depth"),
            SymbolicCount::bounded_by(&SymbolicCount::exact_value(gates), &mut self.arena)?,
        );
        let scratch = self
            .arena
            .literal_u128(u128::from(controls.saturating_sub(2)))?;
        counts.extra_metrics.insert(
            format!("{PORTABLE_PREFIX}clean_ancilla_qubits"),
            SymbolicCount::exact_value(scratch),
        );
        Ok(())
    }

    pub(super) fn price_windowed_mod_pow(
        &mut self,
        counts: &mut Counts,
        controls: Option<u32>,
        target_width: ExprId,
        exponent_width: ExprId,
    ) -> Result<(), SymbolicResourceError> {
        let Some(controls) = controls else {
            return self.portable_unknown(
                counts,
                "windowed modular exponentiation requires a known control width",
            );
        };
        let one = self.arena.literal_u128(1)?;
        let two = self.arena.literal_u128(2)?;
        let three = self.arena.literal_u128(3)?;
        let four = self.arena.literal_u128(4)?;
        let twelve = self.arena.literal_u128(12)?;
        let sixty_four = self.arena.literal_u128(64)?;
        let exponent_windows = self.arena.ceiling_divide(exponent_width, three)?;
        let exponent_windows = self
            .arena
            .absolute(exponent_windows, ResourceDomain::Natural)?;
        let source_windows = self.arena.ceiling_divide(target_width, three)?;
        let source_windows = self
            .arena
            .absolute(source_windows, ResourceDomain::Natural)?;
        let passes = self
            .arena
            .multiply([exponent_windows, source_windows, two])?;
        let extended = self.arena.add([target_width, one])?;

        // Each source/exponent window pair loads and unloads one at-most-six-bit
        // table address. The all-ones data row is a sound bit-pattern upper bound.
        let lookup_visits = self.arena.multiply([passes, two, sixty_four])?;
        let address_flips = self.arena.multiply([lookup_visits, twelve])?;
        let lookup_bits = self.arena.multiply([lookup_visits, extended])?;
        let mut gate_terms = vec![
            self.price_mcx_upper(counts, 0, address_flips)?,
            self.price_mcx_upper(counts, 6, lookup_bits)?,
        ];

        // One exact modular register addition per pass: four ordinary ripple
        // additions, one reduction-controlled addition, and four flag updates.
        let ordinary = self.arena.multiply([passes, four])?;
        let local = controls
            .checked_add(1)
            .ok_or(SymbolicResourceError::Overflow)?;
        let conditional = local
            .checked_add(1)
            .ok_or(SymbolicResourceError::Overflow)?;
        gate_terms.push(
            self.price_ripple_constant_adders(counts, local, extended, ordinary)?
                .gates,
        );
        gate_terms.push(
            self.price_ripple_constant_adders(counts, conditional, extended, passes)?
                .gates,
        );
        let flag_work = self.arena.multiply([passes, four])?;
        gate_terms.push(self.price_mcx_upper(counts, conditional, flag_work)?);

        // One total-domain guard and one controlled swap per exponent window.
        let guard_adders = self.arena.multiply([exponent_windows, four])?;
        gate_terms.push(
            self.price_ripple_constant_adders(counts, controls, extended, guard_adders)?
                .gates,
        );
        let guard_cx = self.arena.multiply([exponent_windows, two])?;
        gate_terms.push(self.price_mcx_upper(counts, controls + 1, guard_cx)?);
        let swap_cx = self.arena.multiply([exponent_windows, target_width, two])?;
        let swaps = self.arena.multiply([exponent_windows, target_width])?;
        gate_terms.push(self.price_mcx_upper(counts, controls + 1, swap_cx)?);
        gate_terms.push(self.price_mcx_upper(counts, controls + 2, swaps)?);

        let gates = self.arena.add(gate_terms)?;
        counts.extra_metrics.insert(
            format!("{PORTABLE_PREFIX}gate_depth"),
            SymbolicCount::bounded_by(&SymbolicCount::exact_value(gates), &mut self.arena)?,
        );
        let chain = u128::from(controls.saturating_add(2)).max(4);
        let fixed = self.arena.literal_u128(6 + chain)?;
        let triple_width = self.arena.multiply([target_width, three])?;
        let scratch = self.arena.add([triple_width, fixed])?;
        counts.extra_metrics.insert(
            format!("{PORTABLE_PREFIX}clean_ancilla_qubits"),
            SymbolicCount::dependency_bounded_by(
                &SymbolicCount::exact_value(scratch),
                &mut self.arena,
            )?,
        );
        self.bound_factor_dependent_profile(counts)
    }

    pub(super) fn price_coset_windowed_mod_pow(
        &mut self,
        counts: &mut Counts,
        controls: Option<u32>,
        target_width: ExprId,
        exponent_width: ExprId,
        padding_bits: u16,
    ) -> Result<(), SymbolicResourceError> {
        let Some(controls) = controls else {
            return self.portable_unknown(
                counts,
                "coset windowed modular exponentiation requires a known control width",
            );
        };
        let one = self.arena.literal_u128(1)?;
        let two = self.arena.literal_u128(2)?;
        let three = self.arena.literal_u128(3)?;
        let four = self.arena.literal_u128(4)?;
        let twelve = self.arena.literal_u128(12)?;
        let sixty_four = self.arena.literal_u128(64)?;
        let padding = self.arena.literal_u128(u128::from(padding_bits))?;
        let exponent_windows = self.arena.ceiling_divide(exponent_width, three)?;
        let exponent_windows = self
            .arena
            .absolute(exponent_windows, ResourceDomain::Natural)?;
        let source_windows = self.arena.ceiling_divide(target_width, three)?;
        let source_windows = self
            .arena
            .absolute(source_windows, ResourceDomain::Natural)?;
        let passes = self
            .arena
            .multiply([exponent_windows, source_windows, two])?;
        let encoded_width = self.arena.add([target_width, padding])?;

        // The same bounded lookup schedule drives both exact and encoded
        // implementations. The output word is widened, but its address stays
        // at most six bits.
        let lookup_visits = self.arena.multiply([passes, two, sixty_four])?;
        let address_flips = self.arena.multiply([lookup_visits, twelve])?;
        let lookup_bits = self.arena.multiply([lookup_visits, encoded_width])?;
        let mut gate_terms = vec![
            self.price_mcx_upper(counts, 0, address_flips)?,
            self.price_mcx_upper(counts, 6, lookup_bits)?,
        ];

        // One ordinary encoded register addition replaces each exact modular
        // reduction. The valid-domain flag contributes one local control.
        let local = controls
            .checked_add(1)
            .ok_or(SymbolicResourceError::Overflow)?;
        gate_terms.push(
            self.price_ripple_register_adders(counts, local, encoded_width, passes)?
                .gates,
        );

        // Each exponent window encodes and decodes the accumulator twice.
        // Padding-controlled constant additions have decreasing widths; using
        // the full encoded width is the deliberate finite upper enclosure.
        let encoding_adders = self.arena.multiply([exponent_windows, padding, four])?;
        let selector_controls = controls
            .checked_add(1)
            .ok_or(SymbolicResourceError::Overflow)?;
        gate_terms.push(
            self.price_ripple_constant_adders(
                counts,
                selector_controls,
                encoded_width,
                encoding_adders,
            )?
            .gates,
        );
        let selector_h = self.arena.multiply([exponent_windows, padding, four])?;
        gate_terms.push(self.price_mcx_exact(counts, 0, selector_h)?);

        // Preserve the total-domain identity and controlled swap contract.
        let extended_target = self.arena.add([target_width, one])?;
        let guard_adders = self.arena.multiply([exponent_windows, four])?;
        gate_terms.push(
            self.price_ripple_constant_adders(counts, controls, extended_target, guard_adders)?
                .gates,
        );
        let guard_cx = self.arena.multiply([exponent_windows, two])?;
        gate_terms.push(self.price_mcx_upper(counts, controls + 1, guard_cx)?);
        let swap_cx = self.arena.multiply([exponent_windows, target_width, two])?;
        let swaps = self.arena.multiply([exponent_windows, target_width])?;
        gate_terms.push(self.price_mcx_upper(counts, controls + 1, swap_cx)?);
        gate_terms.push(self.price_mcx_upper(counts, controls + 2, swaps)?);

        let gates = self.arena.add(gate_terms)?;
        counts.extra_metrics.insert(
            format!("{PORTABLE_PREFIX}gate_depth"),
            SymbolicCount::bounded_by(&SymbolicCount::exact_value(gates), &mut self.arena)?,
        );

        // accumulator (n+m), selector m, lookup (n+m), constant (n+m),
        // valid/overflow/carry, and the longest clean MCX chain.
        let chain = u128::from(controls.saturating_add(2)).max(4);
        let fixed = self.arena.literal_u128(3 + chain)?;
        let triple_width = self.arena.multiply([target_width, three])?;
        let four_padding = self.arena.multiply([padding, four])?;
        let scratch = self.arena.add([triple_width, four_padding, fixed])?;
        counts.extra_metrics.insert(
            format!("{PORTABLE_PREFIX}clean_ancilla_qubits"),
            SymbolicCount::dependency_bounded_by(
                &SymbolicCount::exact_value(scratch),
                &mut self.arena,
            )?,
        );

        self.bound_factor_dependent_profile(counts)?;

        // K translations each have deviation at most 2^-m. The trace-distance
        // consequence is min(1, 2 sqrt(min(1, K 2^-m))). Representation-
        // specific K and m remain private to this producer.
        let negative_padding = self.arena.literal_i64(-i64::from(padding_bits))?;
        let two_to_negative_padding = self.arena.power(two, negative_padding)?;
        let deviation = self.arena.multiply([passes, two_to_negative_padding])?;
        let deviation = self.arena.extremum(false, [one, deviation])?;
        let trace_distance = self.arena.square_root(deviation)?;
        let trace_distance = self.arena.multiply([two, trace_distance])?;
        let trace_distance = self.arena.extremum(false, [one, trace_distance])?;
        self.record_approximation(
            counts,
            ApproximationCertificate::output_trace_distance_bound(trace_distance),
        )
    }

    #[expect(
        clippy::too_many_lines,
        reason = "Keep the product-tree resource derivation in dependency order"
    )]
    pub(super) fn price_mod_multi_pow(
        &mut self,
        counts: &mut Counts,
        controls: Option<u32>,
        coordinate_count: ExprId,
        coordinate_width: ExprId,
        target_width: ExprId,
        base_width: ExprId,
    ) -> Result<(), SymbolicResourceError> {
        let Some(controls) = controls else {
            return self.portable_unknown(
                counts,
                "modular multi-exponentiation requires a known coherent-control width",
            );
        };
        let realization = self
            .observation
            .expect("portable observation checked")
            .realizations
            .select(RealizationFamily::ModMultiPow)
            .kind();
        let one = self.arena.literal_u128(1)?;
        let two = self.arena.literal_u128(2)?;
        let four = self.arena.literal_u128(4)?;
        let six = self.arena.literal_u128(6)?;
        let three = self.arena.literal_u128(3)?;
        match realization {
            Realization::ModMultiPowIndependent => {
                let repetitions = self.arena.multiply([coordinate_count, coordinate_width])?;
                let controls = controls
                    .checked_add(1)
                    .ok_or(SymbolicResourceError::Overflow)?;
                let profile =
                    self.price_mod_mul_child(counts, Some(controls), repetitions, target_width)?;
                let mut scheduler = ParametricDependencyScheduler::new(&mut self.arena)?;
                scheduler.step(&[0, 1], profile.makespan, &mut self.arena)?;
                counts.extra_metrics.insert(
                    format!("{PORTABLE_PREFIX}gate_depth"),
                    SymbolicCount::bounded_by(
                        &SymbolicCount::exact_value(scheduler.makespan()),
                        &mut self.arena,
                    )?,
                );
            }
            Realization::ModMultiPowProductTree => {
                let logarithm = self.arena.log2(coordinate_count)?;
                let levels = self.arena.ceiling(logarithm)?;
                let levels = self.arena.absolute(levels, ResourceDomain::Natural)?;
                let padded = self.arena.power(two, levels)?;
                let level = self.arena.symbol(
                    "qamomile.realization.mod_multi_pow.product_tree",
                    "level",
                    ResourceDomain::Natural,
                    "realization_binder",
                    &["0 <= level"],
                )?;
                let level_scale = self.arena.power(two, level)?;
                let factor_width = self.arena.multiply([base_width, level_scale])?;
                let factor_square = self.arena.multiply([factor_width, factor_width])?;
                let factor_successor = self.arena.add([factor_width, one])?;
                let variable_chain = self.arena.multiply([factor_width, two])?;
                let control_fixed = self.arena.literal_u128(u128::from(controls) + 1)?;
                let chain = self.arena.add([variable_chain, control_fixed])?;
                let per_tree_node =
                    self.arena
                        .multiply([factor_square, factor_successor, chain])?;
                let denominator = self.arena.multiply([two, level_scale])?;
                let nodes_at_level = self.arena.floor_divide(padded, denominator)?;
                let nodes_at_level = self
                    .arena
                    .absolute(nodes_at_level, ResourceDomain::Natural)?;
                let level_work = self.arena.multiply([nodes_at_level, per_tree_node])?;
                let zero = self.arena.literal_u128(0)?;
                let tree_work = self.arena.bounded_sum(level, zero, levels, level_work)?;
                let tree_repetitions = self.arena.multiply([four, coordinate_width])?;
                let tree_gates = self.arena.multiply([tree_repetitions, tree_work])?;
                self.add_portable_upper(counts, "gates", tree_gates)?;
                self.add_portable_upper(counts, "toffoli_gates", tree_gates)?;

                // Account for leaf preparation/selection, coordinate rotation,
                // and coherent word copies.  The selected constants can only
                // reduce these finite upper bounds, so they remain valid while
                // the concrete bases are intentionally absent from the
                // parametric resource expression.
                let selected_leaf_bits = self.arena.multiply([coordinate_count, base_width])?;
                let prepared_leaf_bits = padded;
                let leaf_round = self.arena.add([prepared_leaf_bits, selected_leaf_bits])?;
                let leaf_rounds = self.arena.multiply([four, coordinate_width])?;
                let leaf_overhead = self.arena.multiply([leaf_rounds, leaf_round])?;
                let rotation_rounds =
                    self.arena
                        .multiply([six, coordinate_width, coordinate_count])?;
                let rotation_overhead = self.arena.multiply([rotation_rounds, coordinate_width])?;
                let copy_rounds = self.arena.multiply([four, coordinate_width])?;
                let copy_overhead = self.arena.multiply([copy_rounds, target_width])?;
                let overhead = self.arena.add([
                    leaf_overhead,
                    rotation_overhead,
                    copy_overhead,
                    target_width,
                ])?;
                self.add_portable_upper(counts, "gates", overhead)?;
                self.add_portable_upper(counts, "clifford_gates", overhead)?;

                let modular_repetitions = self.arena.multiply([six, coordinate_width])?;
                let total_modular_width = self.arena.multiply([three, target_width])?;
                self.price_schoolbook_mod_mul_add(
                    counts,
                    Some(controls),
                    modular_repetitions,
                    Some(total_modular_width),
                )?;
                let mut local_counts = Counts::zero(&mut self.arena)?;
                let modular_profile = self.price_schoolbook_mod_mul_add(
                    &mut local_counts,
                    Some(controls),
                    one,
                    Some(total_modular_width),
                )?;
                let tree_depth = self.arena.bounded_sum(level, zero, levels, per_tree_node)?;
                let leaf_depth = self.arena.add([base_width, two])?;
                let tree_phase_depth = self.arena.add([leaf_depth, tree_depth])?;
                let rotation_depth = coordinate_width;

                // Schedule one round from its actual port dependencies.  The
                // selected-base tree overlaps the square; after the combine,
                // tree cleanup overlaps inverse-square cleanup.
                let mut round = ParametricDependencyScheduler::new(&mut self.arena)?;
                round.step(&[0, 1], one, &mut self.arena)?;
                round.step(&[0, 2], modular_profile.makespan, &mut self.arena)?;
                round.step(&[1, 3], tree_phase_depth, &mut self.arena)?;
                round.step(&[2, 3, 4], modular_profile.makespan, &mut self.arena)?;
                round.step(&[1, 3], tree_phase_depth, &mut self.arena)?;
                round.step(&[0, 2], modular_profile.makespan, &mut self.arena)?;
                round.step(&[0, 1], one, &mut self.arena)?;
                round.step(&[1], rotation_depth, &mut self.arena)?;

                let mut complete = ParametricDependencyScheduler::new(&mut self.arena)?;
                complete.repeat_step(
                    &[0, 1],
                    round.makespan(),
                    coordinate_width,
                    &mut self.arena,
                )?;
                complete.step(&[0, 2], one, &mut self.arena)?;
                complete.repeat_step(
                    &[0, 1],
                    round.makespan(),
                    coordinate_width,
                    &mut self.arena,
                )?;
                counts.extra_metrics.insert(
                    format!("{PORTABLE_PREFIX}gate_depth"),
                    SymbolicCount::bounded_by(
                        &SymbolicCount::exact_value(complete.makespan()),
                        &mut self.arena,
                    )?,
                );

                let value_count = self.arena.add([coordinate_width, one])?;
                let values = self.arena.multiply([value_count, target_width])?;
                let tree_level_count = self.arena.add([levels, one])?;
                let tree_storage = self
                    .arena
                    .multiply([padded, base_width, tree_level_count])?;
                let structural = self.arena.add([values, tree_storage, target_width])?;
                let arithmetic_fixed = self
                    .arena
                    .literal_u128(u128::from(controls).saturating_add(12))?;
                let arithmetic_width = self.arena.multiply([target_width, four])?;
                let arithmetic = self.arena.add([arithmetic_width, arithmetic_fixed])?;
                let scratch = self.arena.add([structural, arithmetic])?;
                counts.extra_metrics.insert(
                    format!("{PORTABLE_PREFIX}clean_ancilla_qubits"),
                    SymbolicCount::dependency_bounded_by(
                        &SymbolicCount::exact_value(scratch),
                        &mut self.arena,
                    )?,
                );
            }
            _ => unreachable!("modular multi-exponentiation family is closed"),
        }
        Ok(())
    }

    #[expect(
        clippy::too_many_lines,
        clippy::too_many_arguments,
        reason = "Dispatch uses the independently derived dimensions for each primitive family"
    )]
    pub(super) fn observe_primitive(
        &mut self,
        counts: &mut Counts,
        opcode: &str,
        context: SymbolicFunctionContext<'_>,
        multiplicity: ExprId,
        width: Option<ExprId>,
        mod_pow_widths: Option<(ExprId, ExprId)>,
        mod_multi_pow_dimensions: Option<(ExprId, ExprId, ExprId, ExprId)>,
    ) -> Result<(), SymbolicResourceError> {
        if self.observation.is_none() {
            return Ok(());
        }
        let name = Self::observation_name("operation", opcode, context.mode, context.controls);
        counts.extra_metrics.insert(
            format!("{FRONTIER_PREFIX}{name}"),
            SymbolicCount::exact_value(multiplicity),
        );
        if self
            .observation
            .is_none_or(|observation| observation.basis.is_none())
        {
            return Ok(());
        }
        if matches!(opcode, "quantum.prepare" | "quantum.unprepare") {
            return self.price_computational_basis_preparation(counts, context.controls, width);
        }
        if opcode == "quantum.mod_mul"
            && self.observation.is_some_and(|observation| {
                observation
                    .realizations
                    .select(RealizationFamily::ModMul)
                    .kind()
                    == Realization::ModMulSchoolbookRipple
            })
        {
            self.price_schoolbook_mod_mul(counts, context.controls, multiplicity, width)?;
            return Ok(());
        }
        if matches!(opcode, "quantum.mod_mul_add" | "quantum.mod_mul_sub") {
            self.price_schoolbook_mod_mul_add(counts, context.controls, multiplicity, width)?;
            return Ok(());
        }
        if matches!(opcode, "quantum.mul_add" | "quantum.mul_sub") {
            self.price_schoolbook_mul_add(counts, context.controls, multiplicity, width)?;
            return Ok(());
        }
        if matches!(opcode, "quantum.mod_pow" | "quantum.mod_pow_inverse") {
            let Some((target_width, exponent_width)) = mod_pow_widths else {
                return self.portable_unknown(
                    counts,
                    "modular exponentiation requires known target and exponent widths",
                );
            };
            let realization = self
                .observation
                .expect("observation checked")
                .realizations
                .select(RealizationFamily::ModPow);
            return match realization.kind() {
                Realization::ModPowNative => {
                    self.record_approximation(counts, ApproximationCertificate::Exact)?;
                    self.portable_unknown(
                        counts,
                        "native modular exponentiation requires a target cost model",
                    )
                }
                Realization::ModPowSquareAndMultiply => {
                    self.record_approximation(counts, ApproximationCertificate::Exact)?;
                    let controls = context.controls.and_then(|value| value.checked_add(1));
                    self.price_mod_mul_child(counts, controls, exponent_width, target_width)?;
                    Ok(())
                }
                Realization::ModPowWindowedLookup => {
                    self.record_approximation(counts, ApproximationCertificate::Exact)?;
                    self.price_windowed_mod_pow(
                        counts,
                        context.controls,
                        target_width,
                        exponent_width,
                    )
                }
                Realization::ModPowCosetWindowed => self.price_coset_windowed_mod_pow(
                    counts,
                    context.controls,
                    target_width,
                    exponent_width,
                    realization
                        .coset_padding_bits()
                        .expect("coset realization is checked")
                        .get(),
                ),
                _ => unreachable!("mod_pow family is closed"),
            };
        }
        if matches!(
            opcode,
            "quantum.mod_multi_pow" | "quantum.mod_multi_pow_inverse"
        ) {
            let Some((coordinate_count, coordinate_width, target_width, base_width)) =
                mod_multi_pow_dimensions
            else {
                return self.portable_unknown(
                    counts,
                    "modular multi-exponentiation requires symbolic d, m, n, and base width",
                );
            };
            return self.price_mod_multi_pow(
                counts,
                context.controls,
                coordinate_count,
                coordinate_width,
                target_width,
                base_width,
            );
        }
        self.price_primitive(counts, opcode, context.controls, multiplicity)
    }

    pub(super) fn observe_call(
        &mut self,
        counts: &mut Counts,
        callee: &wire::FunctionProto,
        mode: TransformMode,
        controls: Option<u32>,
    ) -> Result<(), SymbolicResourceError> {
        let Some(policy) = self.observation else {
            return Ok(());
        };
        let name = Self::observation_name("call", &callee.symbol, mode, controls);
        let one = SymbolicCount::exact_value(self.arena.literal_u128(1)?);
        counts.extra_metrics.insert(
            format!("qamomile.algorithmic.call_count.{}", callee.symbol),
            one.clone(),
        );
        let collapsed = callee.external
            || policy
                .expanded
                .as_ref()
                .is_some_and(|expanded| !expanded.contains(&callee.symbol));
        if collapsed {
            counts.extra_metrics.retain(|key, _| {
                !key.starts_with(FRONTIER_PREFIX) && !key.starts_with(PORTABLE_PREFIX)
            });
            counts
                .extra_metrics
                .insert(format!("{FRONTIER_PREFIX}{name}"), one);
            if policy.basis.is_some() {
                self.portable_unknown(
                    counts,
                    &format!("{}: callable body is not realized", callee.symbol),
                )?;
            }
        }
        Ok(())
    }
}
