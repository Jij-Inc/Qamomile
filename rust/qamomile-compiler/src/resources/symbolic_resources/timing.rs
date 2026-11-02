//! Dependency scheduling over symbolic quantum footprints.

use crate::operations::{logical_gate_families, primitive_gate_spec};
use crate::resource_rules::Domain as ResourceDomain;
use crate::resources::analysis::{function_value_regions, is_measurement, is_reset};
use crate::wire;
use std::collections::{BTreeMap, HashMap, HashSet};

use super::{
    ExprId, ExpressionArena, ResourceExpressionNode, SymbolicCount, SymbolicLaneTimeline,
    SymbolicResourceError, SymbolicTimingFootprint, SymbolicTimingNode, SymbolicTimingRoot,
    SymbolicTimingSelectors, SymbolicTimingStep, ValueFact, int_fact,
    mapping_string_tuple_attribute, string_tuple_attribute,
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

pub(super) fn schedule_symbolic_depths(
    nodes: &[SymbolicTimingNode],
    arena: &mut ExpressionArena,
) -> Result<BTreeMap<String, SymbolicCount>, SymbolicResourceError> {
    let zero = arena.literal_u128(0)?;
    let mut timelines = SYMBOLIC_DEPTH_FAMILIES
        .iter()
        .map(|family| (*family, HashMap::new()))
        .collect::<HashMap<&str, HashMap<SymbolicTimingRoot, SymbolicLaneTimeline>>>();
    let mut maxima = SYMBOLIC_DEPTH_FAMILIES
        .iter()
        .map(|family| ((*family).to_owned(), zero))
        .collect::<BTreeMap<_, _>>();
    schedule_symbolic_nodes(nodes, &mut timelines, &mut maxima, arena)?;
    Ok(maxima
        .into_iter()
        .map(|(family, value)| (family, SymbolicCount::exact_value(value)))
        .collect())
}

pub(super) fn schedule_symbolic_nodes(
    nodes: &[SymbolicTimingNode],
    timelines: &mut HashMap<&str, HashMap<SymbolicTimingRoot, SymbolicLaneTimeline>>,
    maxima: &mut BTreeMap<String, ExprId>,
    arena: &mut ExpressionArena,
) -> Result<(), SymbolicResourceError> {
    for node in nodes {
        match node {
            SymbolicTimingNode::Step(step) => {
                schedule_symbolic_step(step, timelines, maxima, arena)?;
            }
            SymbolicTimingNode::Repeat {
                body,
                binder,
                trip_count,
            } => schedule_symbolic_repeat(body, *binder, *trip_count, timelines, maxima, arena)?,
        }
    }
    Ok(())
}

pub(super) fn schedule_symbolic_step(
    step: &SymbolicTimingStep,
    timelines: &mut HashMap<&str, HashMap<SymbolicTimingRoot, SymbolicLaneTimeline>>,
    maxima: &mut BTreeMap<String, ExprId>,
    arena: &mut ExpressionArena,
) -> Result<(), SymbolicResourceError> {
    let selections = validate_symbolic_timing_selections(&step.footprints, arena)?;
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
        let mut dependencies = Vec::new();
        for (footprint, selected) in step.footprints.iter().zip(&selections) {
            for (root, frontier) in roots.iter() {
                if !symbolic_timing_roots_overlap(root, &footprint.root) {
                    continue;
                }
                if symbolic_timing_projection_is_ambiguous(root, footprint, frontier) {
                    return Err(SymbolicResourceError::Unsupported(
                        "projected timing selector alias".to_owned(),
                    ));
                }
                dependencies.push(frontier.barrier);
                if root == &footprint.root
                    && let Some(selected) = selected
                {
                    if symbolic_timing_selector_aliases(selected, &frontier.lanes, arena) {
                        return Err(SymbolicResourceError::Unsupported(
                            "dynamic timing selector may alias another lane".to_owned(),
                        ));
                    }
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
                }
                if selected.is_none() {
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
        for (footprint, selected) in step.footprints.iter().zip(&selections) {
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

pub(super) fn validate_symbolic_timing_selections(
    footprints: &[SymbolicTimingFootprint],
    arena: &ExpressionArena,
) -> Result<Vec<Option<Vec<ExprId>>>, SymbolicResourceError> {
    let selections = footprints
        .iter()
        .map(|footprint| footprint.selectors.as_deref().map(<[ExprId]>::to_vec))
        .collect::<Vec<_>>();
    for (footprint, selected) in footprints.iter().zip(&selections) {
        if let Some(selected) = selected
            && (selected.iter().collect::<HashSet<_>>().len() != selected.len()
                || footprint
                    .width
                    .and_then(|width| natural_literal(arena, width))
                    .is_some_and(|width| {
                        selected.iter().any(|selector| {
                            natural_literal(arena, *selector)
                                .is_some_and(|selector| selector >= width)
                        })
                    }))
        {
            return Err(SymbolicResourceError::Unsupported(
                "invalid timing selector".to_owned(),
            ));
        }
        if selected.as_ref().is_some_and(|selected| {
            selected.len() > 1
                && selected
                    .iter()
                    .any(|selector| natural_literal(arena, *selector).is_none())
        }) {
            return Err(SymbolicResourceError::Unsupported(
                "dynamic multi-lane timing selector".to_owned(),
            ));
        }
    }
    Ok(selections)
}

#[allow(clippy::too_many_lines)]
pub(super) fn schedule_symbolic_repeat(
    body: &[SymbolicTimingNode],
    binder: ExprId,
    trip_count: ExprId,
    timelines: &mut HashMap<&str, HashMap<SymbolicTimingRoot, SymbolicLaneTimeline>>,
    maxima: &mut BTreeMap<String, ExprId>,
    arena: &mut ExpressionArena,
) -> Result<(), SymbolicResourceError> {
    if super::timing_chain::try_schedule_adjacent_chain(
        body, binder, trip_count, timelines, maxima, arena,
    )? {
        return Ok(());
    }
    let mut touched = HashMap::<SymbolicTimingRoot, ExprId>::new();
    let mut local_steps = Vec::new();
    let zero = arena.literal_u128(0)?;
    for node in body {
        let SymbolicTimingNode::Step(step) = node else {
            return Err(SymbolicResourceError::Unsupported(
                "nested symbolic timing repeat".to_owned(),
            ));
        };
        let mut footprints = Vec::new();
        for footprint in &step.footprints {
            if footprint.selectors.as_deref() != Some(&[binder]) {
                return Err(SymbolicResourceError::Unsupported(
                    "non-disjoint symbolic timing repeat".to_owned(),
                ));
            }
            let width = footprint.width.ok_or_else(|| {
                SymbolicResourceError::Unsupported("unknown timing width".to_owned())
            })?;
            if !symbolic_trip_covers_width(arena, trip_count, width) {
                return Err(SymbolicResourceError::Unsupported(
                    "partial symbolic timing coverage".to_owned(),
                ));
            }
            if touched
                .insert(footprint.root.clone(), width)
                .is_some_and(|old| old != width)
            {
                return Err(SymbolicResourceError::Unsupported(
                    "inconsistent symbolic timing width".to_owned(),
                ));
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
        return Ok(());
    }
    let mut local_timelines = SYMBOLIC_DEPTH_FAMILIES
        .iter()
        .map(|family| (*family, HashMap::new()))
        .collect::<HashMap<&str, HashMap<SymbolicTimingRoot, SymbolicLaneTimeline>>>();
    let mut local_maxima = SYMBOLIC_DEPTH_FAMILIES
        .iter()
        .map(|family| ((*family).to_owned(), zero))
        .collect::<BTreeMap<_, _>>();
    schedule_symbolic_nodes(&local_steps, &mut local_timelines, &mut local_maxima, arena)?;
    let one = arena.literal_u128(1)?;
    let active_iterations = arena.extremum(false, [one, trip_count])?;
    for family in SYMBOLIC_DEPTH_FAMILIES {
        let Some(local) = local_maxima.get(*family).copied() else {
            timelines.remove(family);
            maxima.remove(*family);
            continue;
        };
        if !timelines.contains_key(family) {
            continue;
        }
        let roots = timelines
            .get_mut(family)
            .ok_or_else(|| SymbolicResourceError::Missing("depth family".to_owned()))?;
        let mut dependencies = Vec::new();
        for touched_root in touched.keys() {
            for (root, frontier) in roots.iter() {
                if symbolic_timing_roots_overlap(root, touched_root) {
                    if !frontier.lanes.is_empty() || frontier.chain.is_some() {
                        return Err(SymbolicResourceError::Unsupported(
                            "symbolic repeat follows selected lanes".to_owned(),
                        ));
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
        let contribution = arena.multiply([active_iterations, local])?;
        let finish = arena.add([start, contribution])?;
        for root in touched.keys() {
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
