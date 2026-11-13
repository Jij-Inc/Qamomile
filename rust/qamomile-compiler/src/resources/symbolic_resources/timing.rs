//! Dependency scheduling over symbolic quantum footprints.

use crate::operations::{logical_gate_families, primitive_gate_spec};
use crate::resource_rules::Domain as ResourceDomain;
use crate::resources::analysis::{function_value_regions, is_measurement, is_reset};
use crate::wire;
use std::collections::{BTreeMap, HashMap, HashSet};

use super::{
    ExprId, ExpressionArena, ResourceExpressionNode, SymbolicLaneTimeline, SymbolicResourceError,
    SymbolicTimingFootprint, SymbolicTimingNode, SymbolicTimingRoot, SymbolicTimingSelectors,
    SymbolicTimingStep, ValueFact, int_fact, mapping_string_tuple_attribute,
    string_tuple_attribute,
};

pub(super) fn append_symbolic_timing(
    destination: &mut Option<Vec<SymbolicTimingNode>>,
    source: Option<Vec<SymbolicTimingNode>>,
) {
    match (destination.as_mut(), source) {
        (Some(destination), Some(source)) => destination.extend(source),
        _ => *destination = None,
    }
}

pub(super) fn add_symbolic_timing_footprints(
    nodes: &mut [SymbolicTimingNode],
    added: &[SymbolicTimingFootprint],
) {
    for node in nodes {
        match node {
            SymbolicTimingNode::Step(step) => {
                let mut footprints = step.footprints.to_vec();
                footprints.extend_from_slice(added);
                step.footprints = footprints.into_boxed_slice();
            }
            SymbolicTimingNode::Repeat { body, .. } => {
                add_symbolic_timing_footprints(body, added);
            }
            SymbolicTimingNode::Barrier { roots, .. } => {
                let mut extended = roots.to_vec();
                for footprint in added {
                    if !extended.contains(&footprint.root) {
                        extended.push(footprint.root.clone());
                    }
                }
                *roots = extended.into_boxed_slice();
            }
        }
    }
}

pub(super) fn symbolic_call_roots(
    callee: &wire::FunctionProto,
    operation: &wire::OperationProto,
    owner: &wire::FunctionProto,
) -> Result<HashMap<u64, SymbolicTimingRoot>, SymbolicResourceError> {
    let owner_regions = function_value_regions(owner);
    let formal_regions = function_value_regions(callee);
    let mut roots = HashMap::new();
    for (parameter, operand) in callee.parameters.iter().zip(&operation.operands) {
        if let (Some(formal), Some(actual)) = (
            parameter
                .value
                .as_ref()
                .and_then(|value| value.region.as_ref()),
            owner_regions.get(operand),
        ) && !insert_symbolic_timing_root(
            &mut roots,
            formal.root,
            symbolic_timing_root(owner, actual),
        ) {
            return Err(SymbolicResourceError::Unsupported(
                "conflicting timing root mapping".to_owned(),
            ));
        }
    }
    if let Some(terminator) = callee
        .body
        .as_ref()
        .and_then(|body| body.terminator.as_ref())
    {
        for (returned, result) in terminator.operands.iter().zip(&operation.results) {
            if let (Some(formal), Some(actual)) =
                (formal_regions.get(returned), owner_regions.get(&result.id))
                && !insert_symbolic_timing_root(
                    &mut roots,
                    formal.root,
                    symbolic_timing_root(owner, actual),
                )
            {
                return Err(SymbolicResourceError::Unsupported(
                    "conflicting timing result mapping".to_owned(),
                ));
            }
        }
    }
    Ok(roots)
}

pub(super) fn symbolic_coherent_roots(
    callee: &wire::FunctionProto,
    operation: &wire::OperationProto,
    owner: &wire::FunctionProto,
    operand_offset: usize,
) -> Result<HashMap<u64, SymbolicTimingRoot>, SymbolicResourceError> {
    let owner_regions = function_value_regions(owner);
    let mut roots = HashMap::new();
    for (parameter, operand) in callee
        .parameters
        .iter()
        .zip(operation.operands.iter().skip(operand_offset))
    {
        if let (Some(formal), Some(actual)) = (
            parameter
                .value
                .as_ref()
                .and_then(|value| value.region.as_ref()),
            owner_regions.get(operand),
        ) && !insert_symbolic_timing_root(
            &mut roots,
            formal.root,
            symbolic_timing_root(owner, actual),
        ) {
            return Err(SymbolicResourceError::Unsupported(
                "conflicting coherent timing root mapping".to_owned(),
            ));
        }
    }
    Ok(roots)
}

pub(super) fn symbolic_adjoint_roots(
    callee: &wire::FunctionProto,
    operation: &wire::OperationProto,
    owner: &wire::FunctionProto,
) -> Result<Option<HashMap<u64, SymbolicTimingRoot>>, SymbolicResourceError> {
    let quantum = callee
        .parameters
        .iter()
        .filter_map(|parameter| {
            parameter
                .value
                .as_ref()
                .and_then(|value| value.region.as_ref())
        })
        .collect::<Vec<_>>();
    let [formal] = quantum.as_slice() else {
        return Ok(None);
    };
    let owner_regions = function_value_regions(owner);
    let actual = operation
        .operands
        .first()
        .and_then(|operand| owner_regions.get(operand))
        .ok_or_else(|| SymbolicResourceError::Missing("adjoint timing state".to_owned()))?;
    let mut roots = HashMap::from([(formal.root, symbolic_timing_root(owner, actual))]);
    let formal_regions = function_value_regions(callee);
    if let Some(terminator) = callee
        .body
        .as_ref()
        .and_then(|body| body.terminator.as_ref())
    {
        for (returned, result) in terminator.operands.iter().zip(&operation.results) {
            if let (Some(formal), Some(actual)) =
                (formal_regions.get(returned), owner_regions.get(&result.id))
                && !insert_symbolic_timing_root(
                    &mut roots,
                    formal.root,
                    symbolic_timing_root(owner, actual),
                )
            {
                return Ok(None);
            }
        }
    }
    Ok(Some(roots))
}

pub(super) fn symbolic_timing_root(
    owner: &wire::FunctionProto,
    region: &wire::RegionRefProto,
) -> SymbolicTimingRoot {
    SymbolicTimingRoot {
        scope: owner.symbol.clone(),
        root: region.root,
        projections: region.projections.clone(),
        instance: Vec::new(),
    }
}

pub(super) fn insert_symbolic_timing_root(
    roots: &mut HashMap<u64, SymbolicTimingRoot>,
    formal: u64,
    actual: SymbolicTimingRoot,
) -> bool {
    if let Some(existing) = roots.get(&formal) {
        existing == &actual
    } else {
        roots.insert(formal, actual);
        true
    }
}

pub(super) fn reverse_symbolic_timing(nodes: Vec<SymbolicTimingNode>) -> Vec<SymbolicTimingNode> {
    nodes
        .into_iter()
        .rev()
        .map(|node| match node {
            SymbolicTimingNode::Step(step) => SymbolicTimingNode::Step(step),
            SymbolicTimingNode::Repeat {
                body,
                binder,
                trip_count,
            } => SymbolicTimingNode::Repeat {
                body: reverse_symbolic_timing(body.into_vec()).into_boxed_slice(),
                binder,
                trip_count,
            },
            barrier @ SymbolicTimingNode::Barrier { .. } => barrier,
        })
        .collect()
}

pub(super) fn instantiate_symbolic_timing(
    nodes: Vec<SymbolicTimingNode>,
    callee_scope: &str,
    roots: &HashMap<u64, SymbolicTimingRoot>,
    instance: &str,
) -> Vec<SymbolicTimingNode> {
    pub(super) fn map_root(
        root: &SymbolicTimingRoot,
        callee_scope: &str,
        roots: &HashMap<u64, SymbolicTimingRoot>,
        instance: &str,
    ) -> SymbolicTimingRoot {
        if root.scope == callee_scope
            && let Some(actual) = roots.get(&root.root)
        {
            let mut mapped = actual.clone();
            mapped.projections.extend(root.projections.iter().cloned());
            return mapped;
        }
        let mut mapped = root.clone();
        mapped.instance.insert(0, instance.to_owned());
        mapped
    }

    nodes
        .into_iter()
        .map(|node| match node {
            SymbolicTimingNode::Step(step) => SymbolicTimingNode::Step(SymbolicTimingStep {
                footprints: step
                    .footprints
                    .iter()
                    .map(|footprint| SymbolicTimingFootprint {
                        root: map_root(&footprint.root, callee_scope, roots, instance),
                        selectors: footprint.selectors.clone(),
                        width: footprint.width,
                    })
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
                durations: step.durations,
            }),
            SymbolicTimingNode::Repeat {
                body,
                binder,
                trip_count,
            } => SymbolicTimingNode::Repeat {
                body: instantiate_symbolic_timing(body.into_vec(), callee_scope, roots, instance)
                    .into_boxed_slice(),
                binder,
                trip_count,
            },
            SymbolicTimingNode::Barrier {
                roots: barrier_roots,
                durations,
            } => SymbolicTimingNode::Barrier {
                roots: barrier_roots
                    .iter()
                    .map(|root| map_root(root, callee_scope, roots, instance))
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
                durations,
            },
        })
        .collect()
}

pub(super) fn symbolic_timing_selectors(
    operation: &wire::OperationProto,
    effective: &str,
    environment: &HashMap<u64, Option<ValueFact>>,
    arena: &mut ExpressionArena,
) -> Result<SymbolicTimingSelectors, SymbolicResourceError> {
    let expression = |operand: &u64| {
        environment
            .get(operand)
            .and_then(Option::as_ref)
            .and_then(|value| int_fact(Some(value)))
            .map(|fact| fact.expression)
    };
    if matches!(
        effective,
        "quantum.measure_x" | "quantum.measure_y" | "quantum.measure_z"
    ) {
        let selector = operation
            .operands
            .get(1)
            .and_then(expression)
            .map_or_else(|| arena.literal_u128(0), Ok)?;
        return Ok(SymbolicTimingSelectors::Selected(
            vec![selector].into_boxed_slice(),
        ));
    }
    let Some(gate) = primitive_gate_spec(effective) else {
        return Ok(SymbolicTimingSelectors::Whole);
    };
    let names = gate.selectors();
    let keywords = if operation.opcode == "qcore.adjoint" {
        mapping_string_tuple_attribute(operation, "original_attributes", "keywords")
    } else {
        string_tuple_attribute(operation, "keywords")
    };
    let Some(keywords) = keywords else {
        return Ok(SymbolicTimingSelectors::Unsupported);
    };
    let arguments = operation.operands.get(1..).unwrap_or_default();
    if keywords.len() > arguments.len() {
        return Ok(SymbolicTimingSelectors::Unsupported);
    }
    let positional_count = arguments.len() - keywords.len();
    if gate.broadcasts_when_unselected()
        && positional_count == usize::from(gate.has_angle())
        && !keywords.iter().any(|name| gate.selectors().contains(name))
    {
        return Ok(SymbolicTimingSelectors::Whole);
    }
    let positional = &arguments[..positional_count];
    let keyword_values = keywords
        .iter()
        .zip(&arguments[positional_count..])
        .collect::<HashMap<_, _>>();
    let mut selectors = Vec::with_capacity(names.len());
    for (index, name) in names.iter().enumerate() {
        let value = keyword_values.get(name).copied().copied().or_else(|| {
            positional
                .get(index + usize::from(gate.has_angle()))
                .copied()
        });
        let selector = if let Some(expression) = value.and_then(|value| expression(&value)) {
            expression
        } else {
            let default = gate
                .selector_default(index)
                .and_then(|value| u128::try_from(value).ok())
                .ok_or(SymbolicResourceError::Overflow)?;
            arena.literal_u128(default)?
        };
        selectors.push(selector);
    }
    Ok(SymbolicTimingSelectors::Selected(
        selectors.into_boxed_slice(),
    ))
}

pub(super) fn symbolic_quantum_multiplicity(
    operation: &wire::OperationProto,
    effective: &str,
    event_width: Option<ExprId>,
    arena: &mut ExpressionArena,
) -> Result<ExprId, SymbolicResourceError> {
    let Some(gate) = primitive_gate_spec(effective) else {
        return arena.literal_u128(1);
    };
    if !gate.broadcasts_when_unselected() {
        return arena.literal_u128(1);
    }
    let keywords = if operation.opcode == "qcore.adjoint" {
        mapping_string_tuple_attribute(operation, "original_attributes", "keywords")
    } else {
        string_tuple_attribute(operation, "keywords")
    }
    .ok_or_else(|| SymbolicResourceError::Missing("gate keywords".to_owned()))?;
    let arguments = operation.operands.get(1..).unwrap_or_default();
    if keywords.len() > arguments.len() {
        return Err(SymbolicResourceError::Missing(
            "gate selector ABI".to_owned(),
        ));
    }
    let positional_count = arguments.len() - keywords.len();
    let selector_is_omitted = positional_count == usize::from(gate.has_angle())
        && !keywords.iter().any(|name| gate.selectors().contains(name));
    if selector_is_omitted {
        event_width.ok_or_else(|| {
            SymbolicResourceError::Missing(format!(
                "broadcast width for primitive gate {effective:?}"
            ))
        })
    } else {
        arena.literal_u128(1)
    }
}

pub(super) fn symbolic_depth_families(opcode: &str) -> Vec<&'static str> {
    let mut families = logical_gate_families(opcode)
        .iter()
        .copied()
        .filter(|family| {
            matches!(
                *family,
                "gates"
                    | "clifford_gates"
                    | "rotation_gates"
                    | "t_gates"
                    | "toffoli_gates"
                    | "non_clifford_gates"
            )
        })
        .collect::<Vec<_>>();
    if is_measurement(opcode) {
        families.push("measurements");
    }
    if is_reset(opcode) {
        families.push("resets");
    }
    families.sort_unstable();
    families.dedup();
    families
}

pub(super) fn symbolic_unit_durations(
    active: &[&str],
    arena: &mut ExpressionArena,
) -> Result<BTreeMap<String, ExprId>, SymbolicResourceError> {
    let zero = arena.literal_u128(0)?;
    let one = arena.literal_u128(1)?;
    Ok(SYMBOLIC_DEPTH_FAMILIES
        .iter()
        .map(|family| {
            (
                (*family).to_owned(),
                if *family == "span" || active.contains(family) {
                    one
                } else {
                    zero
                },
            )
        })
        .collect())
}

pub(super) const SYMBOLIC_DEPTH_FAMILIES: &[&str] = &[
    "span",
    "gates",
    "clifford_gates",
    "rotation_gates",
    "t_gates",
    "toffoli_gates",
    "non_clifford_gates",
    "measurements",
    "resets",
];

/// Schedule the exact dependency depths of one timing graph.
///
/// Fails when any part of the graph needed a conservative rule; callers that
/// accept upper bounds use [`schedule_symbolic_depth_bounds`].
#[cfg(test)]
pub(super) fn schedule_symbolic_depths(
    nodes: &[SymbolicTimingNode],
    arena: &mut ExpressionArena,
) -> Result<BTreeMap<String, super::SymbolicCount>, SymbolicResourceError> {
    let (maxima, exact) = schedule_symbolic_depth_bounds(nodes, arena)?;
    if !exact {
        return Err(SymbolicResourceError::Unsupported(
            "symbolic schedule needs a conservative rule".to_owned(),
        ));
    }
    Ok(maxima
        .into_iter()
        .map(|(family, value)| (family, super::SymbolicCount::exact_value(value)))
        .collect())
}

/// Schedule dependency depths, returning per-family makespans and exactness.
///
/// Exact rules (independent lanes, loop-index lanes, adjacent chains) keep the
/// result exact. Where they do not apply the scheduler stays sound instead of
/// failing: an unprovable lane selection occupies its whole owner, a loop whose
/// iterations are proved disjoint runs in parallel, and any other loop is the
/// sequential composition of its locally scheduled body. Each such step makes
/// the result an upper bound.
pub(super) fn schedule_symbolic_depth_bounds(
    nodes: &[SymbolicTimingNode],
    arena: &mut ExpressionArena,
) -> Result<(BTreeMap<String, ExprId>, bool), SymbolicResourceError> {
    let zero = arena.literal_u128(0)?;
    let mut timelines = SYMBOLIC_DEPTH_FAMILIES
        .iter()
        .map(|family| (*family, HashMap::new()))
        .collect::<HashMap<&str, HashMap<SymbolicTimingRoot, SymbolicLaneTimeline>>>();
    let mut maxima = SYMBOLIC_DEPTH_FAMILIES
        .iter()
        .map(|family| ((*family).to_owned(), zero))
        .collect::<BTreeMap<_, _>>();
    let mut exact = true;
    schedule_symbolic_nodes(nodes, &mut timelines, &mut maxima, &mut exact, arena)?;
    Ok((maxima, exact))
}

pub(super) fn schedule_symbolic_nodes(
    nodes: &[SymbolicTimingNode],
    timelines: &mut HashMap<&str, HashMap<SymbolicTimingRoot, SymbolicLaneTimeline>>,
    maxima: &mut BTreeMap<String, ExprId>,
    exact: &mut bool,
    arena: &mut ExpressionArena,
) -> Result<(), SymbolicResourceError> {
    for node in nodes {
        match node {
            SymbolicTimingNode::Step(step) => {
                schedule_symbolic_step(step, timelines, maxima, exact, arena)?;
            }
            SymbolicTimingNode::Repeat {
                body,
                binder,
                trip_count,
            } => schedule_symbolic_repeat(
                body,
                *binder,
                *trip_count,
                timelines,
                maxima,
                exact,
                arena,
            )?,
            SymbolicTimingNode::Barrier { roots, durations } => {
                *exact = false;
                let step = SymbolicTimingStep {
                    footprints: roots
                        .iter()
                        .map(|root| SymbolicTimingFootprint {
                            root: root.clone(),
                            selectors: None,
                            width: None,
                        })
                        .collect::<Vec<_>>()
                        .into_boxed_slice(),
                    durations: durations.clone(),
                };
                schedule_symbolic_step(&step, timelines, maxima, exact, arena)?;
            }
        }
    }
    Ok(())
}

/// Return each footprint's lane selection, or `None` when it must occupy the
/// whole owner because its selectors cannot be proved distinct and in range.
fn tolerant_timing_selections(
    footprints: &[SymbolicTimingFootprint],
    arena: &ExpressionArena,
    exact: &mut bool,
) -> Vec<Option<Vec<ExprId>>> {
    footprints
        .iter()
        .map(|footprint| {
            let selected = footprint.selectors.as_deref()?.to_vec();
            let duplicate = selected.iter().collect::<HashSet<_>>().len() != selected.len();
            let out_of_range = footprint
                .width
                .and_then(|width| natural_literal(arena, width))
                .is_some_and(|width| {
                    selected.iter().any(|selector| {
                        natural_literal(arena, *selector).is_some_and(|selector| selector >= width)
                    })
                });
            let dynamic_multi_lane = selected.len() > 1
                && selected
                    .iter()
                    .any(|selector| natural_literal(arena, *selector).is_none());
            if duplicate || out_of_range || dynamic_multi_lane {
                *exact = false;
                None
            } else {
                Some(selected)
            }
        })
        .collect()
}

#[allow(clippy::too_many_lines)]
pub(super) fn schedule_symbolic_step(
    step: &SymbolicTimingStep,
    timelines: &mut HashMap<&str, HashMap<SymbolicTimingRoot, SymbolicLaneTimeline>>,
    maxima: &mut BTreeMap<String, ExprId>,
    exact: &mut bool,
    arena: &mut ExpressionArena,
) -> Result<(), SymbolicResourceError> {
    let selections = tolerant_timing_selections(&step.footprints, arena, exact);
    let zero = arena.literal_u128(0)?;
    for family in SYMBOLIC_DEPTH_FAMILIES {
        if !timelines.contains_key(family) {
            continue;
        }
        let Some(duration) = step.durations.get(*family).copied() else {
            timelines.remove(family);
            maxima.remove(*family);
            continue;
        };
        let roots = timelines
            .get_mut(family)
            .ok_or_else(|| SymbolicResourceError::Missing("depth family".to_owned()))?;
        // Demote selections that could alias a lane already in flight.
        let mut effective = selections.clone();
        for (footprint, selected) in step.footprints.iter().zip(effective.iter_mut()) {
            for (root, frontier) in roots.iter() {
                if !symbolic_timing_roots_overlap(root, &footprint.root) {
                    continue;
                }
                let ambiguous = symbolic_timing_projection_is_ambiguous(root, footprint, frontier);
                let aliases = root == &footprint.root
                    && selected.as_deref().is_some_and(|selected| {
                        symbolic_timing_selector_aliases(selected, &frontier.lanes, arena)
                            || (frontier.chain.is_some()
                                && selected
                                    .iter()
                                    .any(|selector| natural_literal(arena, *selector).is_none()))
                    });
                if selected.is_some() && (ambiguous || aliases) {
                    *selected = None;
                    *exact = false;
                }
            }
        }
        let mut dependencies = Vec::new();
        for (footprint, selected) in step.footprints.iter().zip(&effective) {
            for (root, frontier) in roots.iter() {
                if !symbolic_timing_roots_overlap(root, &footprint.root) {
                    continue;
                }
                dependencies.push(frontier.barrier);
                if root == &footprint.root
                    && let Some(selected) = selected
                {
                    for selector in selected {
                        let completion = if let Some(value) = frontier.lanes.get(selector) {
                            *value
                        } else if let Some(chain) = &frontier.chain {
                            chain.lane(*selector, arena)?
                        } else {
                            frontier.barrier
                        };
                        dependencies.push(completion);
                    }
                } else if selected.is_none() || root != &footprint.root {
                    dependencies.extend(frontier.lanes.values().copied());
                    if let Some(chain) = &frontier.chain {
                        dependencies.push(chain.finish(arena)?);
                    }
                }
            }
            roots
                .entry(footprint.root.clone())
                .or_insert_with(|| SymbolicLaneTimeline {
                    barrier: zero,
                    lanes: HashMap::new(),
                    chain: None,
                });
        }
        let start = arena.extremum(true, dependencies)?;
        let finish = arena.add([start, duration])?;
        for (footprint, selected) in step.footprints.iter().zip(&effective) {
            let frontier = roots
                .get_mut(&footprint.root)
                .ok_or_else(|| SymbolicResourceError::Missing("timing frontier".to_owned()))?;
            if let Some(selected) = selected {
                for selector in selected {
                    frontier.lanes.insert(*selector, finish);
                }
            } else {
                frontier.barrier = finish;
                frontier.lanes.clear();
                frontier.chain = None;
            }
        }
        let current = maxima
            .get(*family)
            .copied()
            .ok_or_else(|| SymbolicResourceError::Missing("depth maximum".to_owned()))?;
        maxima.insert(
            (*family).to_owned(),
            arena.extremum(true, [current, finish])?,
        );
    }
    Ok(())
}

pub(super) fn schedule_symbolic_repeat(
    body: &[SymbolicTimingNode],
    binder: ExprId,
    trip_count: ExprId,
    timelines: &mut HashMap<&str, HashMap<SymbolicTimingRoot, SymbolicLaneTimeline>>,
    maxima: &mut BTreeMap<String, ExprId>,
    exact: &mut bool,
    arena: &mut ExpressionArena,
) -> Result<(), SymbolicResourceError> {
    if super::timing_chain::try_schedule_adjacent_chain(
        body, binder, trip_count, timelines, maxima, arena,
    )? {
        return Ok(());
    }
    if try_schedule_lane_repeat(body, binder, trip_count, timelines, maxima, exact, arena)? {
        return Ok(());
    }
    if try_schedule_disjoint_repeat(body, binder, trip_count, timelines, maxima, exact, arena)? {
        return Ok(());
    }
    schedule_sequential_repeat(body, binder, trip_count, timelines, maxima, exact, arena)
}

/// Exact rule: every step occupies only lane `binder` of owners whose width
/// the trip count covers, so iterations are independent and the repeat
/// costs one body.
#[allow(clippy::too_many_lines)]
fn try_schedule_lane_repeat(
    body: &[SymbolicTimingNode],
    binder: ExprId,
    trip_count: ExprId,
    timelines: &mut HashMap<&str, HashMap<SymbolicTimingRoot, SymbolicLaneTimeline>>,
    maxima: &mut BTreeMap<String, ExprId>,
    exact: &mut bool,
    arena: &mut ExpressionArena,
) -> Result<bool, SymbolicResourceError> {
    let mut touched = HashMap::<SymbolicTimingRoot, ExprId>::new();
    let mut local_steps = Vec::new();
    let zero = arena.literal_u128(0)?;
    for node in body {
        let SymbolicTimingNode::Step(step) = node else {
            return Ok(false);
        };
        let mut footprints = Vec::new();
        for footprint in &step.footprints {
            if footprint.selectors.as_deref() != Some(&[binder]) {
                return Ok(false);
            }
            let Some(width) = footprint.width else {
                return Ok(false);
            };
            if !symbolic_trip_covers_width(arena, trip_count, width) {
                return Ok(false);
            }
            if touched
                .insert(footprint.root.clone(), width)
                .is_some_and(|old| old != width)
            {
                return Ok(false);
            }
            footprints.push(SymbolicTimingFootprint {
                root: footprint.root.clone(),
                selectors: Some(vec![zero].into_boxed_slice()),
                width: footprint.width,
            });
        }
        local_steps.push(SymbolicTimingNode::Step(SymbolicTimingStep {
            footprints: footprints.into_boxed_slice(),
            durations: step.durations.clone(),
        }));
    }
    if touched.is_empty() {
        return Ok(true);
    }
    let (local_maxima, local_exact) = schedule_symbolic_depth_bounds(&local_steps, arena)?;
    *exact &= local_exact;
    let one = arena.literal_u128(1)?;
    let active_iterations = arena.extremum(false, [one, trip_count])?;
    let mut durations = BTreeMap::new();
    for (family, local) in local_maxima {
        durations.insert(family, arena.multiply([active_iterations, local])?);
    }
    install_parallel_region(
        &touched.keys().cloned().collect::<Vec<_>>(),
        &durations,
        timelines,
        maxima,
        exact,
        arena,
    )?;
    Ok(true)
}

/// Occupy `roots` for `durations` after everything they already wait for.
///
/// Prior per-lane completions are joined into the start time; that is exact
/// when the roots carry no lane detail and conservative otherwise.
fn install_parallel_region(
    roots_touched: &[SymbolicTimingRoot],
    durations: &BTreeMap<String, ExprId>,
    timelines: &mut HashMap<&str, HashMap<SymbolicTimingRoot, SymbolicLaneTimeline>>,
    maxima: &mut BTreeMap<String, ExprId>,
    exact: &mut bool,
    arena: &mut ExpressionArena,
) -> Result<(), SymbolicResourceError> {
    let zero = arena.literal_u128(0)?;
    for family in SYMBOLIC_DEPTH_FAMILIES {
        let Some(duration) = durations.get(*family).copied() else {
            timelines.remove(family);
            maxima.remove(*family);
            continue;
        };
        let Some(roots) = timelines.get_mut(family) else {
            continue;
        };
        let mut dependencies = Vec::new();
        for touched_root in roots_touched {
            for (root, frontier) in roots.iter() {
                if symbolic_timing_roots_overlap(root, touched_root) {
                    if !frontier.lanes.is_empty() || frontier.chain.is_some() {
                        *exact = false;
                        dependencies.extend(frontier.lanes.values().copied());
                        if let Some(chain) = &frontier.chain {
                            dependencies.push(chain.finish(arena)?);
                        }
                    }
                    dependencies.push(frontier.barrier);
                }
            }
            roots
                .entry(touched_root.clone())
                .or_insert_with(|| SymbolicLaneTimeline {
                    barrier: zero,
                    lanes: HashMap::new(),
                    chain: None,
                });
        }
        let start = arena.extremum(true, dependencies)?;
        let finish = arena.add([start, duration])?;
        for root in roots_touched {
            let frontier = roots
                .get_mut(root)
                .ok_or_else(|| SymbolicResourceError::Missing("repeat frontier".to_owned()))?;
            frontier.barrier = finish;
            frontier.lanes.clear();
            frontier.chain = None;
        }
        let current = *maxima
            .get(*family)
            .ok_or_else(|| SymbolicResourceError::Missing("depth maximum".to_owned()))?;
        maxima.insert(
            (*family).to_owned(),
            arena.extremum(true, [current, finish])?,
        );
    }
    Ok(())
}

/// Upper-bound rule: iterations whose lanes are pairwise disjoint run in
/// parallel.
///
/// Every selector must be `binder + c` with `c` independent of the binder.
/// Two lanes `i + c_a` and `j + c_b` of distinct iterations `i != j` can only
/// coincide when `|c_a - c_b| = |i - j| < trip_count`, so offsets on one owner
/// that differ by at least the trip count prove the iterations disjoint (a
/// shared offset is the same lane only within one iteration). The repeat then
/// costs one locally scheduled body. Owners it does not fully cover are
/// treated as busy for that whole time, so the result is an upper bound.
#[allow(clippy::too_many_lines)]
fn try_schedule_disjoint_repeat(
    body: &[SymbolicTimingNode],
    binder: ExprId,
    trip_count: ExprId,
    timelines: &mut HashMap<&str, HashMap<SymbolicTimingRoot, SymbolicLaneTimeline>>,
    maxima: &mut BTreeMap<String, ExprId>,
    exact: &mut bool,
    arena: &mut ExpressionArena,
) -> Result<bool, SymbolicResourceError> {
    let extent = trip_extent(arena, trip_count);
    let Some(extent) = arena.polynomial(extent) else {
        return Ok(false);
    };
    let mut offsets: HashMap<SymbolicTimingRoot, Vec<super::SymbolicPolynomial>> = HashMap::new();
    let mut lane_steps = Vec::new();
    for node in body {
        let SymbolicTimingNode::Step(step) = node else {
            return Ok(false);
        };
        let mut footprints = Vec::new();
        for footprint in &step.footprints {
            let Some(selectors) = footprint.selectors.as_deref() else {
                return Ok(false);
            };
            let known = offsets.entry(footprint.root.clone()).or_default();
            let mut lanes = Vec::new();
            for selector in selectors {
                let Some(offset) = selector_offset(arena, *selector, binder) else {
                    return Ok(false);
                };
                let index = if let Some(index) = known.iter().position(|item| item == &offset) {
                    index
                } else {
                    known.push(offset);
                    known.len() - 1
                };
                lanes.push(arena.literal_u128(index as u128)?);
            }
            footprints.push(SymbolicTimingFootprint {
                root: footprint.root.clone(),
                selectors: Some(lanes.into_boxed_slice()),
                width: None,
            });
        }
        lane_steps.push(SymbolicTimingNode::Step(SymbolicTimingStep {
            footprints: footprints.into_boxed_slice(),
            durations: step.durations.clone(),
        }));
    }
    if offsets.is_empty() {
        return Ok(false);
    }
    let roots_touched = offsets.keys().cloned().collect::<Vec<_>>();
    for (index, left) in roots_touched.iter().enumerate() {
        if roots_touched[index + 1..]
            .iter()
            .any(|right| symbolic_timing_roots_overlap(left, right))
        {
            return Ok(false);
        }
    }
    for owner_offsets in offsets.values() {
        for (index, left) in owner_offsets.iter().enumerate() {
            for right in &owner_offsets[index + 1..] {
                if !offsets_separated(left, right, &extent) {
                    return Ok(false);
                }
            }
        }
    }
    let (local_maxima, _) = schedule_symbolic_depth_bounds(&lane_steps, arena)?;
    *exact = false;
    let one = arena.literal_u128(1)?;
    let active_iterations = arena.extremum(false, [one, trip_count])?;
    let mut durations = BTreeMap::new();
    for (family, local) in local_maxima {
        durations.insert(family, arena.multiply([active_iterations, local])?);
    }
    install_parallel_region(&roots_touched, &durations, timelines, maxima, exact, arena)?;
    Ok(true)
}

/// Upper-bound rule: run the iterations one after another.
///
/// The body is scheduled on its own (keeping its internal parallelism), and
/// the repeat occupies every owner the body touches for the sum of the body
/// makespans over all iterations.
fn schedule_sequential_repeat(
    body: &[SymbolicTimingNode],
    binder: ExprId,
    trip_count: ExprId,
    timelines: &mut HashMap<&str, HashMap<SymbolicTimingRoot, SymbolicLaneTimeline>>,
    maxima: &mut BTreeMap<String, ExprId>,
    exact: &mut bool,
    arena: &mut ExpressionArena,
) -> Result<(), SymbolicResourceError> {
    let roots_touched = collect_timing_roots(body);
    if roots_touched.is_empty() {
        return Ok(());
    }
    let (local_maxima, _) = schedule_symbolic_depth_bounds(body, arena)?;
    *exact = false;
    let zero = arena.literal_u128(0)?;
    let negative_one = arena.literal_i64(-1)?;
    let last = arena.add([trip_count, negative_one])?;
    let mut durations = BTreeMap::new();
    for (family, local) in local_maxima {
        let total = if arena.depends_on(local, binder) {
            arena.bounded_sum(binder, zero, last, local)?
        } else {
            arena.multiply([trip_count, local])?
        };
        durations.insert(family, total);
    }
    install_parallel_region(&roots_touched, &durations, timelines, maxima, exact, arena)
}

/// Every owner touched anywhere in a timing graph, in first-touch order.
pub(super) fn collect_timing_roots(nodes: &[SymbolicTimingNode]) -> Vec<SymbolicTimingRoot> {
    fn visit(nodes: &[SymbolicTimingNode], output: &mut Vec<SymbolicTimingRoot>) {
        for node in nodes {
            match node {
                SymbolicTimingNode::Step(step) => {
                    for footprint in &step.footprints {
                        if !output.contains(&footprint.root) {
                            output.push(footprint.root.clone());
                        }
                    }
                }
                SymbolicTimingNode::Repeat { body, .. } => visit(body, output),
                SymbolicTimingNode::Barrier { roots, .. } => {
                    for root in roots {
                        if !output.contains(root) {
                            output.push(root.clone());
                        }
                    }
                }
            }
        }
    }
    let mut output = Vec::new();
    visit(nodes, &mut output);
    output
}

/// The number of values a repeat binder takes, as a plain expression.
fn trip_extent(arena: &ExpressionArena, trip_count: ExprId) -> ExprId {
    match arena.node(trip_count) {
        Some(ResourceExpressionNode::Contracted {
            name, arguments, ..
        }) if name == "qamomile.range_trip_count"
            && arguments.len() == 3
            && arena.is_literal(arguments[0], "0")
            && arena.is_literal(arguments[2], "1") =>
        {
            arguments[1]
        }
        _ => trip_count,
    }
}

/// Write a lane selector as `binder + offset`, returning the offset.
fn selector_offset(
    arena: &ExpressionArena,
    selector: ExprId,
    binder: ExprId,
) -> Option<super::SymbolicPolynomial> {
    let mut polynomial = arena.polynomial(selector)?;
    let mut powers = BTreeMap::new();
    powers.insert(binder, 1);
    let linear = super::SymbolicMonomial(powers);
    if polynomial.terms.remove(&linear)? != num_bigint::BigInt::from(1) {
        return None;
    }
    let independent = polynomial.terms.keys().all(|monomial| {
        monomial
            .0
            .keys()
            .all(|atom| *atom != binder && !arena.depends_on(*atom, binder))
    });
    independent.then_some(polynomial)
}

/// Prove `|left - right| >= extent` for symbolic polynomial offsets.
fn offsets_separated(
    left: &super::SymbolicPolynomial,
    right: &super::SymbolicPolynomial,
    extent: &super::SymbolicPolynomial,
) -> bool {
    let negate = |value: &super::SymbolicPolynomial| {
        value.multiplied(&super::SymbolicPolynomial::constant(
            num_bigint::BigInt::from(-1),
        ))
    };
    let nonnegative_constant = |value: &super::SymbolicPolynomial| {
        value.terms.iter().all(|(monomial, coefficient)| {
            monomial.0.is_empty() && coefficient >= &num_bigint::BigInt::from(0)
        })
    };
    let difference = |a: &super::SymbolicPolynomial, b: &super::SymbolicPolynomial| {
        let mut result = a.clone();
        result.add_assign(negate(b)?)?;
        result.add_assign(negate(extent)?)?;
        Some(result)
    };
    difference(left, right).is_some_and(|value| nonnegative_constant(&value))
        || difference(right, left).is_some_and(|value| nonnegative_constant(&value))
}

pub(super) fn symbolic_trip_covers_width(
    arena: &ExpressionArena,
    trip: ExprId,
    width: ExprId,
) -> bool {
    if trip == width || same_symbol_identity(arena, trip, width) {
        return true;
    }
    matches!(
        arena.node(trip),
        Some(ResourceExpressionNode::Contracted { name, arguments, .. })
            if name == "qamomile.range_trip_count"
                && arguments.len() == 3
                && arena.is_literal(arguments[0], "0")
                && (arguments[1] == width
                    || same_symbol_identity(arena, arguments[1], width))
                && arena.is_literal(arguments[2], "1")
    )
}

pub(super) fn same_symbol_identity(arena: &ExpressionArena, left: ExprId, right: ExprId) -> bool {
    matches!(
        (arena.node(left), arena.node(right)),
        (
            Some(ResourceExpressionNode::Symbol { scope: left_scope, name: left_name, .. }),
            Some(ResourceExpressionNode::Symbol { scope: right_scope, name: right_name, .. }),
        ) if left_scope == right_scope && left_name == right_name
    )
}

// A permutation on 2^n states needs at most 2^n-1 transpositions.
// A Gray-code transposition uses at most (2n-1)^2 controlled-X steps.
// Only the deliberately bounded automatic child route may use this estimate.
pub(super) fn small_permutation_step_bound(width: u64) -> Result<u128, SymbolicResourceError> {
    if !(1..=3).contains(&width) {
        return Err(SymbolicResourceError::Unsupported(
            "small modular multiplication requires one through three target bits".to_owned(),
        ));
    }
    Ok(((1_u128 << width) - 1) * u128::from(2 * width - 1).pow(2))
}

pub(super) fn natural_literal(arena: &ExpressionArena, expression: ExprId) -> Option<u64> {
    match arena.node(expression)? {
        ResourceExpressionNode::Literal {
            value,
            domain: ResourceDomain::Natural,
        } => value.parse().ok(),
        _ => None,
    }
}

pub(super) fn symbolic_timing_selector_aliases(
    selected: &[ExprId],
    lanes: &HashMap<ExprId, ExprId>,
    arena: &ExpressionArena,
) -> bool {
    selected.iter().any(|selector| {
        lanes.keys().any(|known| {
            selector != known
                && !matches!(
                    (natural_literal(arena, *selector), natural_literal(arena, *known)),
                    (Some(left), Some(right)) if left != right
                )
        })
    })
}

pub(super) fn symbolic_timing_roots_overlap(
    left: &SymbolicTimingRoot,
    right: &SymbolicTimingRoot,
) -> bool {
    left.scope == right.scope
        && left.root == right.root
        && left.instance == right.instance
        && (left.projections.starts_with(&right.projections)
            || right.projections.starts_with(&left.projections))
}

pub(super) fn symbolic_timing_projection_is_ambiguous(
    existing: &SymbolicTimingRoot,
    footprint: &SymbolicTimingFootprint,
    frontier: &SymbolicLaneTimeline,
) -> bool {
    if existing == &footprint.root {
        return false;
    }
    let existing_is_ancestor = footprint
        .root
        .projections
        .starts_with(&existing.projections);
    ((!frontier.lanes.is_empty() || frontier.chain.is_some()) && existing_is_ancestor)
        || (footprint.selectors.is_some() && !existing_is_ancestor)
}
