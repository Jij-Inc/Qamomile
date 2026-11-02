//! Checked minimum makespan for one fixed, target-neutral gate graph.
//!
//! Realization selection and gate decomposition happen before this module. The
//! scheduler therefore does not search rewrites: it computes the critical-path
//! optimum of the fixed dependency graph induced by source order on every
//! participating logical wire. Every emitted gate has unit duration and the
//! machine is all-to-all connected.

use std::collections::{HashMap, HashSet};

use thiserror::Error;

use crate::lowering::circuit::{StaticCircuit, StaticInstruction};

use super::portable::portable_gate_recipe;
use super::symbolic_resources::GateBasis;

const MAX_TIMING_EVENTS: usize = 100_000;
const MAX_TIMING_PARTICIPANTS: usize = 1_000_000;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct EventId(usize);

#[derive(Clone, Debug, Eq, PartialEq)]
struct TimingEvent {
    participants: Box<[u64]>,
    duration: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ScheduledEvent {
    start: u128,
    finish: u128,
    critical_predecessor: Option<EventId>,
}

/// Independently checked certificate for one fixed realized gate graph.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct FixedGateSchedule {
    gate_count: u128,
    makespan: u128,
    critical_chain: Box<[EventId]>,
}

impl FixedGateSchedule {
    /// Return the number of unit-duration portable gates in the graph.
    pub(crate) const fn gate_count(&self) -> u128 {
        self.gate_count
    }

    /// Return the proved minimum makespan for this fixed graph.
    pub(crate) const fn makespan(&self) -> u128 {
        self.makespan
    }
}

/// Failure to construct or certify one fixed timing graph.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub(crate) enum TimingError {
    /// The circuit still contains control that requires a pathwise summary.
    #[error("structured or dynamically conditioned circuit instruction")]
    DynamicInstruction,
    /// An instruction has no exact recipe in the selected portable basis.
    #[error("portable timing has no recipe for {0:?}")]
    MissingRecipe(String),
    /// A checked instruction or recipe contains an invalid local lane.
    #[error("portable timing has an invalid lane selection")]
    InvalidLane,
    /// Materializing or scheduling the graph would exceed its fixed budget.
    #[error("portable timing graph exceeds its work budget")]
    Budget,
    /// Checked integer arithmetic cannot represent the schedule.
    #[error("portable timing schedule overflow")]
    Overflow,
    /// The independently replayed certificate disagrees with the schedule.
    #[error("portable timing certificate is inconsistent")]
    InvalidCertificate,
}

/// Compute and certify the minimum gate makespan of one fixed static circuit.
pub(crate) fn schedule_portable_circuit(
    circuit: &StaticCircuit,
    basis: GateBasis,
) -> Result<FixedGateSchedule, TimingError> {
    let events = portable_events(circuit.instructions(), basis)?;
    schedule_fixed_graph(&events)
}

fn portable_events(
    instructions: &[StaticInstruction],
    basis: GateBasis,
) -> Result<Vec<TimingEvent>, TimingError> {
    let mut events = Vec::new();
    let mut participant_count = 0_usize;
    for instruction in instructions {
        if instruction.condition().is_some() || !instruction.regions().is_empty() {
            return Err(TimingError::DynamicInstruction);
        }
        let mut instruction_lanes = instruction.controls().to_vec();
        instruction_lanes.extend_from_slice(instruction.qubits());
        validate_participants(&instruction_lanes)?;
        if matches!(instruction.opcode(), "measure" | "reset") {
            push_event(
                &mut events,
                &mut participant_count,
                TimingEvent {
                    participants: instruction_lanes.into_boxed_slice(),
                    duration: 0,
                },
            )?;
            continue;
        }
        let opcode = format!("quantum.{}", instruction.opcode());
        let controls =
            u32::try_from(instruction.controls().len()).map_err(|_| TimingError::Budget)?;
        let recipe = portable_gate_recipe(&opcode, controls, basis)
            .ok_or_else(|| TimingError::MissingRecipe(opcode.clone()))?;
        for step in recipe {
            let participants = step
                .lanes()
                .iter()
                .map(|lane| {
                    instruction_lanes
                        .get(usize::from(*lane))
                        .copied()
                        .ok_or(TimingError::InvalidLane)
                })
                .collect::<Result<Vec<_>, _>>()?;
            validate_participants(&participants)?;
            push_event(
                &mut events,
                &mut participant_count,
                TimingEvent {
                    participants: participants.into_boxed_slice(),
                    duration: 1,
                },
            )?;
        }
    }
    Ok(events)
}

fn push_event(
    events: &mut Vec<TimingEvent>,
    participant_count: &mut usize,
    event: TimingEvent,
) -> Result<(), TimingError> {
    if events.len() >= MAX_TIMING_EVENTS {
        return Err(TimingError::Budget);
    }
    *participant_count = participant_count
        .checked_add(event.participants.len())
        .filter(|count| *count <= MAX_TIMING_PARTICIPANTS)
        .ok_or(TimingError::Budget)?;
    events.push(event);
    Ok(())
}

fn validate_participants(participants: &[u64]) -> Result<(), TimingError> {
    if participants.iter().copied().collect::<HashSet<_>>().len() != participants.len() {
        return Err(TimingError::InvalidLane);
    }
    Ok(())
}

fn schedule_fixed_graph(events: &[TimingEvent]) -> Result<FixedGateSchedule, TimingError> {
    let mut last_on_lane = HashMap::<u64, EventId>::new();
    let mut scheduled = Vec::<ScheduledEvent>::with_capacity(events.len());
    let mut terminal = None;
    let mut makespan = 0_u128;
    let mut gate_count = 0_u128;
    for (index, event) in events.iter().enumerate() {
        validate_participants(&event.participants)?;
        let predecessor = event
            .participants
            .iter()
            .filter_map(|lane| last_on_lane.get(lane).copied())
            .max_by_key(|event| (scheduled[event.0].finish, event.0));
        let start = predecessor.map_or(0, |event| scheduled[event.0].finish);
        let finish = start
            .checked_add(u128::from(event.duration))
            .ok_or(TimingError::Overflow)?;
        let identifier = EventId(index);
        for lane in &event.participants {
            last_on_lane.insert(*lane, identifier);
        }
        scheduled.push(ScheduledEvent {
            start,
            finish,
            critical_predecessor: predecessor,
        });
        gate_count = gate_count
            .checked_add(u128::from(event.duration))
            .ok_or(TimingError::Overflow)?;
        if finish > makespan || (finish == makespan && terminal.is_none()) {
            makespan = finish;
            terminal = Some(identifier);
        }
    }
    let critical_chain = critical_chain(terminal, &scheduled)?;
    check_schedule(events, &scheduled, gate_count, makespan, &critical_chain)?;
    Ok(FixedGateSchedule {
        gate_count,
        makespan,
        critical_chain: critical_chain.into_boxed_slice(),
    })
}

fn critical_chain(
    mut current: Option<EventId>,
    scheduled: &[ScheduledEvent],
) -> Result<Vec<EventId>, TimingError> {
    let mut reversed = Vec::new();
    while let Some(identifier) = current {
        let event = scheduled
            .get(identifier.0)
            .ok_or(TimingError::InvalidCertificate)?;
        reversed.push(identifier);
        current = event.critical_predecessor;
    }
    reversed.reverse();
    Ok(reversed)
}

fn check_schedule(
    events: &[TimingEvent],
    scheduled: &[ScheduledEvent],
    gate_count: u128,
    makespan: u128,
    critical_chain: &[EventId],
) -> Result<(), TimingError> {
    if events.len() != scheduled.len()
        || events.iter().try_fold(0_u128, |count, event| {
            count.checked_add(u128::from(event.duration))
        }) != Some(gate_count)
        || scheduled
            .iter()
            .map(|event| event.finish)
            .max()
            .unwrap_or(0)
            != makespan
    {
        return Err(TimingError::InvalidCertificate);
    }
    let mut last_on_lane = HashMap::<u64, EventId>::new();
    for (index, (event, placement)) in events.iter().zip(scheduled).enumerate() {
        let expected_start = event
            .participants
            .iter()
            .filter_map(|lane| last_on_lane.get(lane).copied())
            .map(|identifier| scheduled[identifier.0].finish)
            .max()
            .unwrap_or(0);
        if placement.start != expected_start
            || placement.finish
                != placement
                    .start
                    .checked_add(u128::from(event.duration))
                    .ok_or(TimingError::Overflow)?
        {
            return Err(TimingError::InvalidCertificate);
        }
        for lane in &event.participants {
            last_on_lane.insert(*lane, EventId(index));
        }
    }
    let chain_duration = critical_chain.iter().try_fold(0_u128, |total, identifier| {
        events
            .get(identifier.0)
            .and_then(|event| total.checked_add(u128::from(event.duration)))
    });
    if chain_duration != Some(makespan) {
        return Err(TimingError::InvalidCertificate);
    }
    for pair in critical_chain.windows(2) {
        let [left, right] = pair else {
            unreachable!("windows of two have two elements")
        };
        if scheduled[left.0].finish != scheduled[right.0].start
            || !events[left.0]
                .participants
                .iter()
                .any(|lane| events[right.0].participants.contains(lane))
        {
            return Err(TimingError::InvalidCertificate);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lowering::circuit::{StaticCircuit, StaticInstruction};

    fn gate(participants: &[u64]) -> TimingEvent {
        TimingEvent {
            participants: participants.into(),
            duration: 1,
        }
    }

    #[test]
    fn independent_gates_share_one_layer() {
        let schedule = schedule_fixed_graph(&[gate(&[0]), gate(&[1]), gate(&[2])]).unwrap();
        assert_eq!(schedule.gate_count(), 3);
        assert_eq!(schedule.makespan(), 1);
    }

    #[test]
    fn shared_control_serializes_otherwise_disjoint_targets() {
        let schedule = schedule_fixed_graph(&[gate(&[0, 1]), gate(&[0, 2])]).unwrap();
        assert_eq!(schedule.gate_count(), 2);
        assert_eq!(schedule.makespan(), 2);
    }

    #[test]
    fn fork_join_uses_the_critical_path_instead_of_gate_count() {
        let schedule =
            schedule_fixed_graph(&[gate(&[0, 1]), gate(&[2]), gate(&[3]), gate(&[1, 2])]).unwrap();
        assert_eq!(schedule.gate_count(), 4);
        assert_eq!(schedule.makespan(), 2);
        assert_eq!(schedule.critical_chain.len(), 2);
    }

    #[test]
    fn duplicate_participants_fail_closed() {
        assert_eq!(
            schedule_fixed_graph(&[gate(&[0, 0])]),
            Err(TimingError::InvalidLane)
        );
    }

    #[test]
    fn circuit_recipes_are_scheduled_after_controlled_decomposition() {
        let circuit = StaticCircuit::test_new(
            4,
            0,
            vec![
                StaticInstruction::test_new("x", vec![0], vec![], vec![], None, vec![]),
                StaticInstruction::test_new("x", vec![1], vec![], vec![], None, vec![]),
                StaticInstruction::test_new("h", vec![3], vec![], vec![2], None, vec![]),
            ],
            vec![],
            vec![],
        );
        let schedule =
            schedule_portable_circuit(&circuit, GateBasis::CliffordToffoliRotations).unwrap();
        assert_eq!(schedule.gate_count(), 5);
        assert_eq!(schedule.makespan(), 3);
    }
}
