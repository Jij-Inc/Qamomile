//! Canonical coherent-control activation states.
//!
//! An absent source-level `control_state` is retained as the unit literal so
//! dependent control widths do not need an eagerly constructed all-ones mask.
//! Explicit states remain arbitrary-precision nonnegative integers and are
//! checked against the control width as soon as that width is fixed.

use std::fmt;

use num_bigint::{BigInt, BigUint};

use crate::wire;

/// One canonical activation pattern for a coherent control operation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ControlState {
    /// Every coherent control component must be one.
    AllActive,
    /// Match one explicit little-endian computational-basis state.
    Exact(BigUint),
}

impl ControlState {
    /// Decode the closed wire representation of one activation pattern.
    pub(crate) fn decode(literal: &wire::LiteralProto) -> Option<Self> {
        use wire::literal_proto::Value;

        match literal.value.as_ref()? {
            Value::UnitValue(_) => Some(Self::AllActive),
            Value::IntValue(value) if *value >= 0 => {
                Some(Self::Exact(BigUint::from(u64::try_from(*value).ok()?)))
            }
            Value::BigintValue(encoded) => BigInt::from_signed_bytes_be(encoded)
                .to_biguint()
                .map(Self::Exact),
            _ => None,
        }
    }

    /// Construct an exact state already known to fit in a machine word.
    pub(crate) fn exact_u64(value: u64) -> Self {
        Self::Exact(BigUint::from(value))
    }

    /// Return whether this pattern is valid for one fixed positive width.
    pub(crate) fn fits(&self, width: u64) -> bool {
        match self {
            Self::AllActive => width > 0,
            Self::Exact(value) => width > 0 && value.bits() <= width,
        }
    }

    /// Return whether one source-ordered component must be active.
    pub(crate) fn component_is_active(&self, index: usize) -> bool {
        match self {
            Self::AllActive => true,
            Self::Exact(value) => u64::try_from(index).is_ok_and(|index| value.bit(index)),
        }
    }
}

impl fmt::Display for ControlState {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AllActive => formatter.write_str("all-active"),
            Self::Exact(value) => value.fmt(formatter),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn literal(value: wire::literal_proto::Value) -> wire::LiteralProto {
        wire::LiteralProto { value: Some(value) }
    }

    #[test]
    fn unit_retains_an_unbounded_all_active_pattern() {
        let state = ControlState::decode(&literal(wire::literal_proto::Value::UnitValue(true)))
            .expect("unit is the canonical all-active pattern");

        assert_eq!(state, ControlState::AllActive);
        assert!(state.fits(1));
        assert!(state.fits(65));
        assert!(state.component_is_active(64));
    }

    #[test]
    fn explicit_states_are_nonnegative_arbitrary_precision_masks() {
        let state = ControlState::decode(&literal(wire::literal_proto::Value::BigintValue(
            BigInt::from(1_u128 << 80).to_signed_bytes_be(),
        )))
        .expect("positive bigint is a valid exact pattern");

        assert!(!state.fits(80));
        assert!(state.fits(81));
        assert!(state.component_is_active(80));
        assert!(!state.component_is_active(79));
        assert!(ControlState::decode(&literal(wire::literal_proto::Value::IntValue(-1))).is_none());
    }
}
