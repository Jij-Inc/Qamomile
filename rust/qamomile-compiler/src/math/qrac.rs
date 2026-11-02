//! Bounded quantum random-access encoding plans over canonical polynomials.

use std::collections::{BTreeMap, BTreeSet};

use thiserror::Error;

use crate::binary64::exact_sum;
use crate::{
    CanonicalPolynomial, NormalizedObservable, ObservableError, Pauli, PolynomialDomain,
    PolynomialError, convert_polynomial, normalize_real_observable,
};

/// Maximum raw Pauli terms generated before native normalization.
pub const MAX_QRAC_GENERATED_TERMS: usize = 10_000_000;
/// Maximum rounded rows accepted by one decode call.
pub const MAX_QRAC_DECODE_ROWS: usize = 1_000_000;
/// Maximum values inspected by one decode call.
pub const MAX_QRAC_DECODE_CELLS: usize = 10_000_000;

/// Supported quantum random-access encoding geometry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QracVariant {
    /// Two variables encoded on one qubit.
    Qrac21,
    /// Three variables encoded on one qubit.
    Qrac31,
    /// Three variables encoded on two qubits.
    Qrac32,
    /// Sequential two-variable groups with the local QRAO product rule.
    SpaceEfficient,
}

#[derive(Clone, Debug, PartialEq)]
struct QracTerm {
    factors: Vec<(usize, Pauli)>,
    coefficient: f64,
}

#[derive(Clone, Debug, PartialEq)]
struct QracObservable {
    width: usize,
    constant: f64,
    terms: Vec<QracTerm>,
}

/// One aggregated decoded assignment.
#[derive(Clone, Debug, PartialEq)]
pub struct QracDecodedSample {
    values: Vec<i8>,
    energy: f64,
    occurrences: usize,
}

impl QracDecodedSample {
    /// Return values in canonical source-variable order.
    #[must_use]
    pub fn values(&self) -> &[i8] {
        &self.values
    }

    /// Return source-polynomial energy.
    #[must_use]
    pub const fn energy(&self) -> f64 {
        self.energy
    }

    /// Return the number of equal decoded input rows.
    #[must_use]
    pub const fn occurrences(&self) -> usize {
        self.occurrences
    }
}

/// One checked QRAC grouping, observable, relaxation, and decoding plan.
#[derive(Clone, Debug, PartialEq)]
pub struct QracPlan {
    source: CanonicalPolynomial,
    width: usize,
    groups: Vec<Vec<usize>>,
    observables: Vec<NormalizedObservable>,
    hamiltonian: NormalizedObservable,
}

impl QracPlan {
    /// Derive all QRAC facts from one canonical quadratic model.
    ///
    /// # Errors
    ///
    /// Returns [`QracError`] for nonquadratic models, excessive generated work,
    /// or non-finite derived coefficients.
    pub fn new(source: &CanonicalPolynomial, variant: QracVariant) -> Result<Self, QracError> {
        if source.terms().iter().any(|term| term.variables().len() > 2) {
            return Err(QracError::NonQuadratic);
        }
        let spin = convert_polynomial(source, PolynomialDomain::Spin)?;
        let groups = build_groups(&spin, variant);
        let (raw_observables, scales, width) = build_observables(&groups, variant)?;
        let raw_hamiltonian =
            encode_hamiltonian(&spin, &groups, &raw_observables, &scales, width, variant)?;
        let observables = raw_observables
            .iter()
            .map(QracObservable::normalized)
            .collect::<Result<Vec<_>, _>>()?;
        let hamiltonian = raw_hamiltonian.normalized()?;
        Ok(Self {
            source: source.clone(),
            width,
            groups,
            observables,
            hamiltonian,
        })
    }

    /// Return the physical register width.
    #[must_use]
    pub const fn width(&self) -> usize {
        self.width
    }

    /// Return variable-position groups in deterministic color order.
    #[must_use]
    pub fn groups(&self) -> &[Vec<usize>] {
        &self.groups
    }

    /// Return one decode observable per source variable.
    #[must_use]
    pub fn observables(&self) -> &[NormalizedObservable] {
        &self.observables
    }

    /// Return the relaxed encoded Hamiltonian.
    #[must_use]
    pub const fn hamiltonian(&self) -> &NormalizedObservable {
        &self.hamiltonian
    }

    /// Sign-round one bounded expectation row and decode it.
    ///
    /// # Errors
    ///
    /// Returns [`QracError`] when the row shape, values, or resulting source
    /// energy violates the plan contract.
    pub fn sign_decode(&self, values: &[f64]) -> Result<Vec<QracDecodedSample>, QracError> {
        if values.len() != self.source.variables().len()
            || values
                .iter()
                .any(|value| !value.is_finite() || !(-1.0..=1.0).contains(value))
        {
            return Err(QracError::InvalidExpectations);
        }
        let spins = values
            .iter()
            .map(|value| if *value >= 0.0 { 1_i8 } else { -1_i8 })
            .collect::<Vec<_>>();
        self.decode(&[spins])
    }

    /// Decode and aggregate rounded spin rows in the source domain.
    ///
    /// # Errors
    ///
    /// Returns [`QracError`] for an empty, malformed, or excessive collection,
    /// or an unrepresentable source energy.
    pub fn decode(&self, rows: &[Vec<i8>]) -> Result<Vec<QracDecodedSample>, QracError> {
        if rows.is_empty() {
            return Err(QracError::InvalidSpins);
        }
        if rows.len() > MAX_QRAC_DECODE_ROWS {
            return Err(QracError::DecodeBudget);
        }
        let cells = rows
            .len()
            .checked_mul(self.source.variables().len())
            .ok_or(QracError::DecodeBudget)?;
        if cells > MAX_QRAC_DECODE_CELLS {
            return Err(QracError::DecodeBudget);
        }
        let mut counts = BTreeMap::<Vec<i8>, usize>::new();
        for row in rows {
            if row.len() != self.source.variables().len()
                || row.iter().any(|value| !matches!(value, -1 | 1))
            {
                return Err(QracError::InvalidSpins);
            }
            let decoded = match self.source.domain() {
                PolynomialDomain::Binary => row.iter().map(|spin| (1 - spin) / 2).collect(),
                PolynomialDomain::Spin => row.clone(),
            };
            let count = counts.entry(decoded).or_default();
            *count = count.checked_add(1).ok_or(QracError::DecodeBudget)?;
        }
        counts
            .into_iter()
            .map(|(values, occurrences)| {
                let energy = self.source.energy(&values)?;
                Ok(QracDecodedSample {
                    values,
                    energy,
                    occurrences,
                })
            })
            .collect()
    }
}

/// Failure to derive or use a QRAC plan.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum QracError {
    /// QRAC accepts only quadratic source objectives.
    #[error("Q7581: QRAC encoding requires a quadratic model")]
    NonQuadratic,
    /// Grouping or observable materialization exceeds a bound.
    #[error("Q7582: QRAC planning exceeds its structural budget")]
    PlanBudget,
    /// Expectation coordinates are malformed or outside the physical range.
    #[error("Q7583: QRAC expectations must be finite values in [-1, 1]")]
    InvalidExpectations,
    /// Rounded spin rows are empty, malformed, or outside the spin domain.
    #[error("Q7584: QRAC spin rows must be nonempty fixed-width values in {{-1, 1}}")]
    InvalidSpins,
    /// Rounded-row decoding exceeds its explicit work budget.
    #[error("Q7585: QRAC decoding exceeds its work budget")]
    DecodeBudget,
    /// A derived observable coefficient is outside finite binary64.
    #[error("Q7586: QRAC observable coefficient exceeds finite binary64")]
    CoefficientRange,
    /// The shared polynomial normal form rejected conversion or evaluation.
    #[error(transparent)]
    Polynomial(#[from] PolynomialError),
    /// The common Pauli normal form rejected one derived observable.
    #[error(transparent)]
    Observable(#[from] ObservableError),
}

type Word = Vec<(usize, Pauli)>;

fn build_groups(spin: &CanonicalPolynomial, variant: QracVariant) -> Vec<Vec<usize>> {
    let count = spin.variables().len();
    if variant == QracVariant::SpaceEfficient {
        return (0..count)
            .collect::<Vec<_>>()
            .chunks(2)
            .map(<[usize]>::to_vec)
            .collect();
    }
    let capacity = if variant == QracVariant::Qrac21 { 2 } else { 3 };
    let positions = spin
        .variables()
        .iter()
        .enumerate()
        .map(|(position, variable)| (variable, position))
        .collect::<BTreeMap<_, _>>();
    let mut adjacency = vec![Vec::new(); count];
    for term in spin.terms() {
        if let [left, right] = term.variables() {
            let left = positions[left];
            let right = positions[right];
            adjacency[left].push(right);
            adjacency[right].push(left);
        }
    }
    let mut colors = vec![None; count];
    let mut groups = Vec::<Vec<usize>>::new();
    let mut available = BTreeSet::<usize>::new();
    for variable in 0..count {
        let forbidden = adjacency[variable]
            .iter()
            .filter_map(|neighbor| colors[*neighbor])
            .collect::<BTreeSet<_>>();
        let color = available
            .iter()
            .copied()
            .find(|candidate| !forbidden.contains(candidate));
        let color = if let Some(color) = color {
            color
        } else {
            let color = groups.len();
            groups.push(Vec::new());
            available.insert(color);
            color
        };
        groups[color].push(variable);
        colors[variable] = Some(color);
        if groups[color].len() == capacity {
            available.remove(&color);
        }
    }
    groups
}

fn build_observables(
    groups: &[Vec<usize>],
    variant: QracVariant,
) -> Result<(Vec<QracObservable>, Vec<f64>, usize), QracError> {
    let count = groups.iter().map(Vec::len).sum();
    let mut observables = vec![QracObservable::empty(); count];
    let mut scales = vec![0.0; count];
    let width = match variant {
        QracVariant::Qrac32 => groups.iter().try_fold(0_usize, |width, group| {
            width
                .checked_add(if group.len() == 1 { 1 } else { 2 })
                .ok_or(QracError::PlanBudget)
        })?,
        _ => groups.len(),
    };
    let mut start = 0;
    for (group_index, group) in groups.iter().enumerate() {
        for (offset, variable) in group.iter().copied().enumerate() {
            let (observable, scale) = match variant {
                QracVariant::SpaceEfficient => (
                    QracObservable::single(
                        if offset == 0 { Pauli::X } else { Pauli::Y },
                        group_index,
                        width,
                    ),
                    3.0_f64.sqrt(),
                ),
                QracVariant::Qrac21 => (
                    QracObservable::single([Pauli::Z, Pauli::X][offset], group_index, width),
                    group_size(group)?.sqrt(),
                ),
                QracVariant::Qrac31 => (
                    QracObservable::single(
                        [Pauli::Z, Pauli::X, Pauli::Y][offset],
                        group_index,
                        width,
                    ),
                    group_size(group)?.sqrt(),
                ),
                QracVariant::Qrac32 if group.len() == 1 => {
                    (QracObservable::single(Pauli::Z, start, width), 1.0)
                }
                QracVariant::Qrac32 => (
                    prime_observable([Pauli::Z, Pauli::X, Pauli::Y][offset], start, width)?,
                    (2.0 * group_size(group)?).sqrt(),
                ),
            };
            observables[variable] = observable;
            scales[variable] = scale;
        }
        if variant == QracVariant::Qrac32 {
            start += if group.len() == 1 { 1 } else { 2 };
        }
    }
    Ok((observables, scales, width))
}

fn group_size(group: &[usize]) -> Result<f64, QracError> {
    u32::try_from(group.len())
        .map(f64::from)
        .map_err(|_| QracError::PlanBudget)
}

fn prime_observable(pauli: Pauli, start: usize, width: usize) -> Result<QracObservable, QracError> {
    let normalization = 1.0 / 6.0_f64.sqrt();
    let terms = match pauli {
        Pauli::X => vec![
            (vec![(start, Pauli::X), (start + 1, Pauli::X)], 0.5),
            (vec![(start, Pauli::X), (start + 1, Pauli::Z)], 0.5),
            (vec![(start, Pauli::Z)], 1.0),
        ],
        Pauli::Y => vec![
            (vec![(start + 1, Pauli::X)], 0.5),
            (vec![(start + 1, Pauli::Z)], 1.0),
            (vec![(start, Pauli::Y), (start + 1, Pauli::Y)], 0.5),
        ],
        Pauli::Z => vec![
            (vec![(start, Pauli::Z), (start + 1, Pauli::Z)], 1.0),
            (vec![(start, Pauli::X)], -0.5),
            (vec![(start, Pauli::Z), (start + 1, Pauli::X)], -0.5),
        ],
    };
    QracObservable::normalize(
        width,
        0.0,
        terms
            .into_iter()
            .map(|(word, coefficient)| (word, coefficient * normalization)),
    )
}

fn encode_hamiltonian(
    spin: &CanonicalPolynomial,
    groups: &[Vec<usize>],
    observables: &[QracObservable],
    scales: &[f64],
    width: usize,
    variant: QracVariant,
) -> Result<QracObservable, QracError> {
    let positions = spin
        .variables()
        .iter()
        .enumerate()
        .map(|(position, variable)| (variable, position))
        .collect::<BTreeMap<_, _>>();
    let mut group_of = vec![0_usize; spin.variables().len()];
    for (group_index, group) in groups.iter().enumerate() {
        for variable in group {
            group_of[*variable] = group_index;
        }
    }
    let mut terms = Vec::<(Word, f64)>::new();
    for term in spin.terms() {
        let coefficient = term.coefficient();
        match term.variables() {
            [variable] => {
                let index = positions[variable];
                extend_scaled(&mut terms, &observables[index], coefficient * scales[index])?;
            }
            [left, right] => {
                let left = positions[left];
                let right = positions[right];
                if variant == QracVariant::SpaceEfficient && group_of[left] == group_of[right] {
                    push_term(
                        &mut terms,
                        vec![(group_of[left], Pauli::Z)],
                        coefficient * 3.0_f64.sqrt(),
                    )?;
                } else {
                    extend_product(
                        &mut terms,
                        &observables[left],
                        &observables[right],
                        coefficient * scales[left] * scales[right],
                    )?;
                }
            }
            _ => return Err(QracError::NonQuadratic),
        }
    }
    QracObservable::normalize(width, spin.constant(), terms)
}

fn extend_scaled(
    target: &mut Vec<(Word, f64)>,
    observable: &QracObservable,
    scale: f64,
) -> Result<(), QracError> {
    for term in &observable.terms {
        push_term(target, term.factors.clone(), scale * term.coefficient)?;
    }
    Ok(())
}

fn extend_product(
    target: &mut Vec<(Word, f64)>,
    left: &QracObservable,
    right: &QracObservable,
    scale: f64,
) -> Result<(), QracError> {
    for left_term in &left.terms {
        for right_term in &right.terms {
            let mut word = left_term.factors.clone();
            word.extend_from_slice(&right_term.factors);
            word.sort_unstable_by_key(|(index, _)| *index);
            if word.windows(2).any(|pair| pair[0].0 == pair[1].0) {
                return Err(QracError::PlanBudget);
            }
            push_term(
                target,
                word,
                scale * left_term.coefficient * right_term.coefficient,
            )?;
        }
    }
    Ok(())
}

fn push_term(target: &mut Vec<(Word, f64)>, word: Word, coefficient: f64) -> Result<(), QracError> {
    if !coefficient.is_finite() {
        return Err(QracError::CoefficientRange);
    }
    if coefficient != 0.0 {
        if target.len() == MAX_QRAC_GENERATED_TERMS {
            return Err(QracError::PlanBudget);
        }
        target.push((word, coefficient));
    }
    Ok(())
}

impl QracObservable {
    fn normalized(&self) -> Result<NormalizedObservable, ObservableError> {
        let terms = self
            .terms
            .iter()
            .map(|term| (term.factors.clone(), term.coefficient))
            .collect::<Vec<_>>();
        normalize_real_observable(self.width, self.constant, &terms)
    }

    fn empty() -> Self {
        Self {
            width: 0,
            constant: 0.0,
            terms: Vec::new(),
        }
    }

    fn single(pauli: Pauli, index: usize, width: usize) -> Self {
        Self {
            width,
            constant: 0.0,
            terms: vec![QracTerm {
                factors: vec![(index, pauli)],
                coefficient: 1.0,
            }],
        }
    }

    fn normalize(
        width: usize,
        constant: f64,
        terms: impl IntoIterator<Item = (Word, f64)>,
    ) -> Result<Self, QracError> {
        if !constant.is_finite() {
            return Err(QracError::CoefficientRange);
        }
        let mut grouped = BTreeMap::<Word, Vec<f64>>::new();
        for (word, coefficient) in terms {
            if !coefficient.is_finite()
                || word.is_empty()
                || word.windows(2).any(|pair| pair[0].0 >= pair[1].0)
                || word.iter().any(|(index, _)| *index >= width)
            {
                return Err(QracError::CoefficientRange);
            }
            grouped.entry(word).or_default().push(coefficient);
        }
        let mut normalized = Vec::with_capacity(grouped.len());
        for (word, coefficients) in grouped {
            let coefficient = exact_sum(coefficients).ok_or(QracError::CoefficientRange)?;
            if coefficient != 0.0 {
                normalized.push(QracTerm {
                    factors: word,
                    coefficient,
                });
            }
        }
        Ok(Self {
            width,
            constant: if constant == 0.0 { 0.0 } else { constant },
            terms: normalized,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(
        domain: PolynomialDomain,
        terms: &[Vec<num_bigint::BigUint>],
        coefficients: &[f64],
        constant: f64,
    ) -> CanonicalPolynomial {
        crate::normalize_polynomial(domain, terms, coefficients, constant, None).expect("model")
    }

    #[test]
    fn coloring_is_independent_capacity_bounded_and_deterministic() {
        let source = model(
            PolynomialDomain::Spin,
            &[
                vec![0_u8.into()],
                vec![1_u8.into()],
                vec![2_u8.into()],
                vec![3_u8.into()],
                vec![0_u8.into(), 1_u8.into()],
                vec![1_u8.into(), 2_u8.into()],
                vec![2_u8.into(), 3_u8.into()],
            ],
            &[1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0],
            0.0,
        );
        let plan = QracPlan::new(&source, QracVariant::Qrac21).expect("plan");

        assert_eq!(plan.groups(), &[vec![0, 2], vec![1, 3]]);
        assert_eq!(plan.width(), 2);
        assert_eq!(plan.observables().len(), 4);
    }

    #[test]
    fn variants_derive_expected_local_observables() {
        let source = model(
            PolynomialDomain::Spin,
            &[vec![0_u8.into()], vec![1_u8.into()], vec![2_u8.into()]],
            &[1.0, 1.0, 1.0],
            0.0,
        );
        let qrac32 = QracPlan::new(&source, QracVariant::Qrac32).expect("3,2 plan");
        let compact = QracPlan::new(&source, QracVariant::SpaceEfficient).expect("compact plan");

        assert_eq!(qrac32.width(), 2);
        assert!(
            qrac32
                .observables()
                .iter()
                .all(NormalizedObservable::is_hermitian)
        );
        assert_eq!(compact.groups(), &[vec![0, 1], vec![2]]);
        assert!(compact.hamiltonian().is_hermitian());
    }

    #[test]
    fn decoding_aggregates_rows_in_the_source_domain() {
        let source = model(
            PolynomialDomain::Binary,
            &[vec![10_u8.into()], vec![10_u8.into(), 20_u8.into()]],
            &[1.0, -2.0],
            0.0,
        );
        let plan = QracPlan::new(&source, QracVariant::Qrac31).expect("plan");
        let decoded = plan
            .decode(&[vec![1, -1], vec![1, -1], vec![-1, 1]])
            .expect("decode");

        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].values(), &[0, 1]);
        assert_eq!(decoded[0].occurrences(), 2);
        assert!(decoded[0].energy().abs() < f64::EPSILON);
        assert_eq!(decoded[1].values(), &[1, 0]);
        assert!((decoded[1].energy() - 1.0).abs() < f64::EPSILON);
        assert_eq!(
            plan.sign_decode(&[0.2, -0.3]).expect("sign")[0].values(),
            &[0, 1]
        );
    }

    #[test]
    fn malformed_domains_and_unrepresentable_scaling_fail_closed() {
        let cubic = model(
            PolynomialDomain::Binary,
            &[vec![0_u8.into(), 1_u8.into(), 2_u8.into()]],
            &[1.0],
            0.0,
        );
        let extreme = model(
            PolynomialDomain::Spin,
            &[vec![0_u8.into()], vec![1_u8.into()]],
            &[f64::MAX, 1.0],
            0.0,
        );

        assert_eq!(
            QracPlan::new(&cubic, QracVariant::Qrac31),
            Err(QracError::NonQuadratic)
        );
        assert_eq!(
            QracPlan::new(&extreme, QracVariant::Qrac31),
            Err(QracError::CoefficientRange)
        );
    }
}
