//! Bounded plans for coherent immutable lookup-table construction.

use num_bigint::BigUint;
use thiserror::Error;

/// Largest number of coherent addresses accepted by lookup planning.
pub const MAX_LOOKUP_ROWS: usize = 1 << 20;
/// Largest target register accepted by lookup planning.
pub const MAX_LOOKUP_WIDTH: usize = 4096;
/// Largest explicit X-gate schedule accepted by lookup planning.
pub const MAX_LOOKUP_FLIPS: usize = 10_000_000;

/// Failure to derive one bounded lookup plan.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum LookupError {
    /// The address domain is not a nontrivial power of two.
    #[error("Q7650: lookup table length must be a power of two of at least two")]
    InvalidTableLength,
    /// The address domain exceeds the explicit row budget.
    #[error("Q7651: lookup table exceeds the {MAX_LOOKUP_ROWS}-row budget")]
    RowBudget,
    /// An output register has no components.
    #[error("Q7652: lookup value width must be positive")]
    EmptyWidth,
    /// An output register exceeds the explicit width budget.
    #[error("Q7653: lookup value width exceeds the {MAX_LOOKUP_WIDTH}-qubit budget")]
    WidthBudget,
    /// A lookup value is outside its output register.
    #[error("Q7654: a lookup value does not fit in value_width")]
    ValueOutsideWidth,
    /// Sparse input matrices do not have one common rectangular shape.
    #[error("Q7655: column_indices and encoded_values must have the same shape")]
    SparseShapeMismatch,
    /// A sparse address extent is not a power of two.
    #[error("Q7656: sparse row and slot extents must be powers of two")]
    SparseExtent,
    /// The sparse address domain is trivial.
    #[error("Q7657: sparse lookup needs at least two row/slot addresses")]
    SparseTrivial,
    /// A sparse column index is outside its output register.
    #[error("Q7658: a column index does not fit in column_width")]
    ColumnOutsideWidth,
    /// The explicit gate schedule exceeds its work budget.
    #[error("Q7659: lookup schedule exceeds the {MAX_LOOKUP_FLIPS}-flip budget")]
    FlipBudget,
}

/// One canonical XOR lookup plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct XorLookupPlan {
    address_width: usize,
    value_width: usize,
    flip_targets: Vec<Vec<usize>>,
}

impl XorLookupPlan {
    /// Return the coherent address-register width.
    #[must_use]
    pub const fn address_width(&self) -> usize {
        self.address_width
    }

    /// Return the lookup output-register width.
    #[must_use]
    pub const fn value_width(&self) -> usize {
        self.value_width
    }

    /// Return one ascending X-target row per address.
    #[must_use]
    pub fn flip_targets(&self) -> &[Vec<usize>] {
        &self.flip_targets
    }
}

/// One canonical sparse row/slot lookup plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SparseLookupPlan {
    address_width: usize,
    column_width: usize,
    value_width: usize,
    flip_targets: Vec<Vec<usize>>,
}

impl SparseLookupPlan {
    /// Return the packed row/slot address-register width.
    #[must_use]
    pub const fn address_width(&self) -> usize {
        self.address_width
    }

    /// Return the column output-register width.
    #[must_use]
    pub const fn column_width(&self) -> usize {
        self.column_width
    }

    /// Return the encoded-value output-register width.
    #[must_use]
    pub const fn value_width(&self) -> usize {
        self.value_width
    }

    /// Return packed column/value X targets in slot-major, row-minor order.
    #[must_use]
    pub fn flip_targets(&self) -> &[Vec<usize>] {
        &self.flip_targets
    }
}

/// Derive a bounded total XOR lookup plan.
///
/// # Errors
///
/// Returns [`LookupError`] when the address count, selected width, one value,
/// or the explicit flip schedule is outside its supported domain.
pub fn plan_xor_lookup(
    values: &[BigUint],
    value_width: Option<usize>,
) -> Result<XorLookupPlan, LookupError> {
    validate_address_count(values.len())?;
    let width = selected_width(values, value_width)?;
    let flip_targets = plan_flip_targets(values, width)?;
    Ok(XorLookupPlan {
        address_width: values.len().ilog2() as usize,
        value_width: width,
        flip_targets,
    })
}

/// Derive a bounded sparse row/slot lookup plan.
///
/// # Errors
///
/// Returns [`LookupError`] when the matrix shapes, address extents, output
/// widths, values, or explicit flip schedule are outside their domains.
pub fn plan_sparse_lookup(
    columns: &[Vec<BigUint>],
    values: &[Vec<BigUint>],
    column_width: usize,
    value_width: Option<usize>,
) -> Result<SparseLookupPlan, LookupError> {
    let column_shape = matrix_shape(columns);
    let value_shape = matrix_shape(values);
    let Some((row_count, slot_count)) = column_shape.filter(|shape| Some(*shape) == value_shape)
    else {
        return Err(LookupError::SparseShapeMismatch);
    };
    if !row_count.is_power_of_two() || !slot_count.is_power_of_two() {
        return Err(LookupError::SparseExtent);
    }
    let address_count = row_count
        .checked_mul(slot_count)
        .ok_or(LookupError::RowBudget)?;
    if address_count < 2 {
        return Err(LookupError::SparseTrivial);
    }
    if address_count > MAX_LOOKUP_ROWS {
        return Err(LookupError::RowBudget);
    }
    validate_width(column_width)?;
    if columns
        .iter()
        .flatten()
        .any(|value| value.bits() > column_width as u64)
    {
        return Err(LookupError::ColumnOutsideWidth);
    }
    let selected_value_width = selected_width_for_bits(
        values
            .iter()
            .flatten()
            .map(BigUint::bits)
            .max()
            .unwrap_or_default(),
        value_width,
    )?;
    let packed_width = column_width
        .checked_add(selected_value_width)
        .ok_or(LookupError::WidthBudget)?;
    validate_width(packed_width)?;
    let mut packed = Vec::with_capacity(address_count);
    for slot in 0..slot_count {
        for row in 0..row_count {
            packed.push(&columns[row][slot] | (&values[row][slot] << column_width));
        }
    }
    let flip_targets = plan_flip_targets(&packed, packed_width)?;
    Ok(SparseLookupPlan {
        address_width: address_count.ilog2() as usize,
        column_width,
        value_width: selected_value_width,
        flip_targets,
    })
}

fn validate_address_count(count: usize) -> Result<(), LookupError> {
    if count > MAX_LOOKUP_ROWS {
        return Err(LookupError::RowBudget);
    }
    if count < 2 || !count.is_power_of_two() {
        return Err(LookupError::InvalidTableLength);
    }
    Ok(())
}

fn validate_width(width: usize) -> Result<(), LookupError> {
    if width == 0 {
        return Err(LookupError::EmptyWidth);
    }
    if width > MAX_LOOKUP_WIDTH {
        return Err(LookupError::WidthBudget);
    }
    Ok(())
}

fn selected_width(values: &[BigUint], explicit: Option<usize>) -> Result<usize, LookupError> {
    let required = values.iter().map(BigUint::bits).max().unwrap_or_default();
    selected_width_for_bits(required, explicit)
}

fn selected_width_for_bits(required: u64, explicit: Option<usize>) -> Result<usize, LookupError> {
    let width = match explicit {
        Some(width) => width,
        None => usize::try_from(required.max(1)).map_err(|_| LookupError::WidthBudget)?,
    };
    validate_width(width)?;
    if required > width as u64 {
        return Err(LookupError::ValueOutsideWidth);
    }
    Ok(width)
}

fn plan_flip_targets(values: &[BigUint], width: usize) -> Result<Vec<Vec<usize>>, LookupError> {
    let mut work = 0_usize;
    let mut rows = Vec::with_capacity(values.len());
    for value in values {
        let mut targets = Vec::new();
        for (word_index, mut word) in value.iter_u64_digits().enumerate() {
            while word != 0 {
                let bit = word.trailing_zeros() as usize;
                targets.push(word_index * u64::BITS as usize + bit);
                word &= word - 1;
            }
        }
        work = work
            .checked_add(targets.len())
            .ok_or(LookupError::FlipBudget)?;
        if work > MAX_LOOKUP_FLIPS || targets.last().is_some_and(|target| *target >= width) {
            return Err(LookupError::FlipBudget);
        }
        rows.push(targets);
    }
    Ok(rows)
}

fn matrix_shape(matrix: &[Vec<BigUint>]) -> Option<(usize, usize)> {
    let columns = matrix.first()?.len();
    (columns > 0 && matrix.iter().all(|row| row.len() == columns))
        .then_some((matrix.len(), columns))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn value(value: u32) -> BigUint {
        BigUint::from(value)
    }

    #[test]
    fn xor_plan_derives_width_address_and_ascending_flip_targets() {
        let plan =
            plan_xor_lookup(&[value(0), value(3), value(1), value(2)], None).expect("lookup plan");

        assert_eq!(plan.address_width(), 2);
        assert_eq!(plan.value_width(), 2);
        assert_eq!(plan.flip_targets(), &[vec![], vec![0, 1], vec![0], vec![1]]);
    }

    #[test]
    fn sparse_plan_uses_slot_major_row_minor_packing() {
        let columns = vec![vec![value(0), value(1)], vec![value(1), value(0)]];
        let values = vec![vec![value(1), value(2)], vec![value(3), value(0)]];
        let plan = plan_sparse_lookup(&columns, &values, 1, None).expect("sparse plan");

        assert_eq!(plan.address_width(), 2);
        assert_eq!(plan.column_width(), 1);
        assert_eq!(plan.value_width(), 2);
        assert_eq!(
            plan.flip_targets(),
            &[vec![1], vec![0, 1, 2], vec![0, 2], vec![]]
        );
    }

    #[test]
    fn malformed_and_unbounded_lookup_plans_fail_closed() {
        assert_eq!(
            plan_xor_lookup(&[value(0), value(1), value(2)], None),
            Err(LookupError::InvalidTableLength)
        );
        assert_eq!(
            plan_xor_lookup(&[value(0), value(4)], Some(2)),
            Err(LookupError::ValueOutsideWidth)
        );
        assert_eq!(
            plan_sparse_lookup(&[vec![value(0)]], &[vec![value(0), value(1)]], 1, None),
            Err(LookupError::SparseShapeMismatch)
        );
        assert_eq!(
            plan_xor_lookup(&[value(0), value(1)], Some(MAX_LOOKUP_WIDTH + 1)),
            Err(LookupError::WidthBudget)
        );
    }
}
