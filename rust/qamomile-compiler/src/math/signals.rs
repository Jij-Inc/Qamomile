//! Bounded fixed-basis embeddings used by ordinary signal-composition functions.

use num_bigint::BigUint;
use thiserror::Error;

/// Maximum ambient width accepted before constructing a region layout.
pub const MAX_SIGNAL_REGION_WIDTH: usize = 4096;

/// Failure to construct a fixed-basis signal region.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum SignalRegionError {
    /// An embedding requires a positive, bounded ambient width.
    #[error("Q7730: signal region width must be between 1 and {MAX_SIGNAL_REGION_WIDTH}")]
    Width,
    /// Each source-ordered lane must be fixed to zero/one or left free.
    #[error("Q7731: signal region pattern must contain only '0', '1', and '*'")]
    Pattern,
}

/// Checked phase action of a fixed-basis region.
#[derive(Clone, Debug, PartialEq, Eq)]
enum RegionPhase {
    Full,
    Selected {
        target: usize,
        target_is_zero: bool,
        controls: Vec<usize>,
        control_state: BigUint,
    },
}

/// An immutable embedding with source-ordered fixed and free coordinates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SignalRegion {
    pattern: String,
    payload_indices: Vec<usize>,
    phase: RegionPhase,
    routing_swaps: Vec<RegionSwap>,
}

/// One exact coordinate exchange in a region's bounded routing witness.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegionSwap {
    left: usize,
    right: usize,
}

impl RegionSwap {
    /// Return the first ambient lane.
    #[must_use]
    pub const fn left(&self) -> usize {
        self.left
    }

    /// Return the second ambient lane.
    #[must_use]
    pub const fn right(&self) -> usize {
        self.right
    }
}

impl SignalRegion {
    /// Validate an ASCII pattern and derive its phase controls in linear space.
    ///
    /// Lane zero is the first character. The last constrained lane is the
    /// phase target; earlier constrained lanes form its source-ordered control
    /// word. Free lanes remain payload coordinates. No basis states are enumerated.
    ///
    /// # Errors
    /// Returns a typed width or pattern diagnostic before constructing a layout.
    pub fn new(pattern: &str) -> Result<Self, SignalRegionError> {
        if pattern.is_empty() || pattern.len() > MAX_SIGNAL_REGION_WIDTH {
            return Err(SignalRegionError::Width);
        }
        if !pattern.bytes().all(|bit| matches!(bit, b'0' | b'1' | b'*')) {
            return Err(SignalRegionError::Pattern);
        }
        let payload_indices = pattern
            .bytes()
            .enumerate()
            .filter_map(|(index, bit)| (bit == b'*').then_some(index))
            .collect();
        let phase = match pattern.bytes().rposition(|bit| bit != b'*') {
            None => RegionPhase::Full,
            Some(target) => {
                let mut controls = Vec::new();
                let mut control_state = BigUint::default();
                for (index, bit) in pattern.bytes().take(target).enumerate() {
                    if bit == b'*' {
                        continue;
                    }
                    if bit == b'1' {
                        control_state.set_bit(controls.len() as u64, true);
                    }
                    controls.push(index);
                }
                RegionPhase::Selected {
                    target,
                    target_is_zero: pattern.as_bytes()[target] == b'0',
                    controls,
                    control_state,
                }
            }
        };
        let mut routing_swaps = Vec::new();
        let mut current: Vec<_> = (0..pattern.len()).collect();
        let mut positions = current.clone();
        for (destination, original) in pattern
            .bytes()
            .enumerate()
            .filter_map(|(index, bit)| (bit != b'*').then_some(index))
            .enumerate()
        {
            let source = positions[original];
            if source != destination {
                routing_swaps.push(RegionSwap {
                    left: destination,
                    right: source,
                });
                positions[current[destination]] = source;
                positions[original] = destination;
                current.swap(destination, source);
            }
        }
        Ok(Self {
            pattern: pattern.to_owned(),
            payload_indices,
            phase,
            routing_swaps,
        })
    }

    /// Return the canonical source-ordered embedding pattern.
    #[must_use]
    pub fn pattern(&self) -> &str {
        &self.pattern
    }

    /// Return the complete coherent width.
    #[must_use]
    pub fn width(&self) -> usize {
        self.pattern.len()
    }

    /// Return the ordered free coordinates, possibly empty for a rank-one region.
    #[must_use]
    pub fn payload_indices(&self) -> &[usize] {
        &self.payload_indices
    }

    /// Return the selected phase target, or none for the full-space phase.
    #[must_use]
    pub const fn target(&self) -> Option<usize> {
        match &self.phase {
            RegionPhase::Full => None,
            RegionPhase::Selected { target, .. } => Some(*target),
        }
    }

    /// Return whether the target needs conjugation by X around its phase gate.
    #[must_use]
    pub const fn target_is_zero(&self) -> bool {
        matches!(
            &self.phase,
            RegionPhase::Selected {
                target_is_zero: true,
                ..
            }
        )
    }

    /// Return control indices in the register remaining after target removal.
    #[must_use]
    pub fn controls(&self) -> &[usize] {
        match &self.phase {
            RegionPhase::Full => &[],
            RegionPhase::Selected { controls, .. } => controls,
        }
    }

    /// Return the size of the constrained prefix after routing.
    #[must_use]
    pub fn phase_width(&self) -> usize {
        match &self.phase {
            RegionPhase::Full => 0,
            RegionPhase::Selected { controls, .. } => controls.len() + 1,
        }
    }

    /// Return coordinate exchanges that pack fixed lanes in source order.
    #[must_use]
    pub fn routing_swaps(&self) -> &[RegionSwap] {
        &self.routing_swaps
    }

    /// Return the exact control word, or zero when no controls are needed.
    #[must_use]
    pub fn control_state(&self) -> BigUint {
        match &self.phase {
            RegionPhase::Full => BigUint::default(),
            RegionPhase::Selected { control_state, .. } => control_state.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asymmetric_layout_preserves_source_order() {
        let region = SignalRegion::new("1*00*").unwrap();
        assert_eq!(region.width(), 5);
        assert_eq!(region.target(), Some(3));
        assert!(region.target_is_zero());
        assert_eq!(region.controls(), &[0, 2]);
        assert_eq!(region.control_state(), BigUint::from(1_u8));
        assert_eq!(region.payload_indices(), &[1, 4]);
        assert_eq!(region.pattern(), "1*00*");
        let mut order: Vec<_> = (0..region.width()).collect();
        for swap in region.routing_swaps() {
            order.swap(swap.left(), swap.right());
        }
        assert_eq!(&order[..region.phase_width()], &[0, 2, 3]);
    }

    #[test]
    fn full_and_rank_one_regions_have_distinct_phase_actions() {
        let full = SignalRegion::new("***").unwrap();
        assert_eq!(full.target(), None);
        assert_eq!(full.payload_indices(), &[0, 1, 2]);
        assert!(full.controls().is_empty());
        assert!(!full.target_is_zero());
        let point = SignalRegion::new("01").unwrap();
        assert_eq!(point.target(), Some(1));
        assert!(!point.target_is_zero());
        assert!(point.payload_indices().is_empty());
        assert_eq!(point.control_state(), BigUint::default());
    }

    #[test]
    fn budgets_precede_layout_allocation() {
        assert_eq!(SignalRegion::new(""), Err(SignalRegionError::Width));
        assert_eq!(SignalRegion::new("01x"), Err(SignalRegionError::Pattern));
        assert_eq!(SignalRegion::new("é"), Err(SignalRegionError::Pattern));
        assert_eq!(
            SignalRegion::new(&"*".repeat(MAX_SIGNAL_REGION_WIDTH + 1)),
            Err(SignalRegionError::Width)
        );
        let largest = SignalRegion::new(&"1".repeat(MAX_SIGNAL_REGION_WIDTH)).unwrap();
        assert_eq!(largest.control_state().bits(), 4095);
        assert_eq!(largest.controls().len(), 4095);
    }
}
