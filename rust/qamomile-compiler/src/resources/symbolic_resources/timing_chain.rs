//! Checked adjacent-lane dependence chains without per-qubit expansion.

use std::collections::{BTreeMap, HashMap};

use super::timing::{
    SYMBOLIC_DEPTH_FAMILIES, natural_literal, same_symbol_identity, symbolic_timing_roots_overlap,
};
use super::{
    ExprId, ExpressionArena, ResourceExpressionNode, SymbolicLaneTimeline, SymbolicResourceError,
    SymbolicTimingNode, SymbolicTimingRoot,
};

/// Compact completion times after the verified lane pairs `(i, i + 1)`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct AdjacentChainFrontier {
    start: ExprId,
    duration: ExprId,
    trip_count: ExprId,
}

impl AdjacentChainFrontier {
    pub(super) fn finish(
        &self,
        arena: &mut ExpressionArena,
    ) -> Result<ExprId, SymbolicResourceError> {
        let work = arena.multiply([self.trip_count, self.duration])?;
        arena.add([self.start, work])
    }

    pub(super) fn lane(
        &self,
        selector: ExprId,
        arena: &mut ExpressionArena,
    ) -> Result<ExprId, SymbolicResourceError> {
        let index = natural_literal(arena, selector).ok_or_else(|| {
            SymbolicResourceError::Unsupported(
                "symbolic selector after adjacent timing chain".to_owned(),
            )
        })?;
        let last_touch = arena.literal_u128(u128::from(index) + 1)?;
        let last_touch = arena.extremum(false, [last_touch, self.trip_count])?;
        let work = arena.multiply([last_touch, self.duration])?;
        arena.add([self.start, work])
    }
}

/// Prove a complete forward chain before installing its compact frontier.
pub(super) fn try_schedule_adjacent_chain(
    body: &[SymbolicTimingNode],
    binder: ExprId,
    trip_count: ExprId,
    timelines: &mut HashMap<&str, HashMap<SymbolicTimingRoot, SymbolicLaneTimeline>>,
    maxima: &mut BTreeMap<String, ExprId>,
    arena: &mut ExpressionArena,
) -> Result<bool, SymbolicResourceError> {
    let [SymbolicTimingNode::Step(step)] = body else {
        return Ok(false);
    };
    let [footprint] = step.footprints.as_ref() else {
        return Ok(false);
    };
    let Some(width) = footprint.width else {
        return Ok(false);
    };
    let Some([left, right]) = footprint.selectors.as_deref() else {
        return Ok(false);
    };
    let offsets = [
        constant_offset(arena, *left, binder),
        constant_offset(arena, *right, binder),
    ];
    if !matches!(offsets, [Some(0), Some(1)] | [Some(1), Some(0)])
        || !covers_adjacent_pairs(arena, trip_count, width)
        || step
            .durations
            .values()
            .any(|value| arena.depends_on(*value, binder))
    {
        return Ok(false);
    }
    // A lane-zero prefix (or a whole-owner barrier) seeds the entire chain.
    // Other selected prefixes require a richer max-plus recurrence; fail
    // conservatively rather than making an unproved exact claim.
    for roots in timelines.values() {
        for (root, frontier) in roots {
            if symbolic_timing_roots_overlap(root, &footprint.root)
                && (root != &footprint.root
                    || frontier.chain.is_some()
                    || frontier
                        .lanes
                        .keys()
                        .any(|lane| natural_literal(arena, *lane) != Some(0)))
            {
                return Ok(false);
            }
        }
    }
    let zero = arena.literal_u128(0)?;
    for family in SYMBOLIC_DEPTH_FAMILIES {
        let Some(duration) = step.durations.get(*family).copied() else {
            timelines.remove(family);
            maxima.remove(*family);
            continue;
        };
        let Some(roots) = timelines.get_mut(family) else {
            continue;
        };
        let frontier =
            roots
                .entry(footprint.root.clone())
                .or_insert_with(|| SymbolicLaneTimeline {
                    barrier: zero,
                    lanes: HashMap::new(),
                    chain: None,
                });
        let mut incoming = vec![frontier.barrier];
        incoming.extend(frontier.lanes.values().copied());
        let start = arena.extremum(true, incoming)?;
        let chain = AdjacentChainFrontier {
            start,
            duration,
            trip_count,
        };
        let finish = chain.finish(arena)?;
        frontier.lanes.clear();
        frontier.chain = Some(chain);
        let current = *maxima
            .get(*family)
            .ok_or_else(|| SymbolicResourceError::Missing("chain depth maximum".to_owned()))?;
        maxima.insert(
            (*family).to_owned(),
            arena.extremum(true, [current, finish])?,
        );
    }
    Ok(true)
}

fn covers_adjacent_pairs(arena: &ExpressionArena, trip: ExprId, width: ExprId) -> bool {
    if let Some(width) = natural_literal(arena, width)
        && let Some(trip) = natural_literal(arena, trip)
    {
        return width.checked_sub(1) == Some(trip);
    }
    matches!(arena.node(trip), Some(ResourceExpressionNode::Contracted { name, arguments, .. })
        if name == "qamomile.range_trip_count" && arguments.len() == 3
            && arena.is_literal(arguments[0], "0") && arena.is_literal(arguments[2], "1")
            && constant_offset(arena, arguments[1], width) == Some(-1))
}

fn constant_offset(arena: &ExpressionArena, expression: ExprId, base: ExprId) -> Option<i128> {
    let is_base = |value| value == base || same_symbol_identity(arena, value, base);
    if is_base(expression) {
        return Some(0);
    }
    let ResourceExpressionNode::Add { arguments, .. } = arena.node(expression)? else {
        return None;
    };
    let mut found = false;
    let mut offset = 0_i128;
    for argument in arguments {
        if is_base(*argument) {
            if found {
                return None;
            }
            found = true;
        } else {
            let ResourceExpressionNode::Literal { value, .. } = arena.node(*argument)? else {
                return None;
            };
            offset = offset.checked_add(value.parse::<i128>().ok()?)?;
        }
    }
    found.then_some(offset)
}
