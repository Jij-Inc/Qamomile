//! Canonical arbitrary-precision-variable BINARY and SPIN polynomials.

mod integer;
pub use integer::{ExactBinaryEncoding, IntegerEncodingError, MAX_EXACT_GAS_BITS};

use std::collections::{BTreeMap, BTreeSet};

use num_bigint::{BigInt, BigUint};
use num_traits::Zero;
use thiserror::Error;

use crate::binary64::{
    SUBNORMAL_EXPONENT, exact_sum, float_to_subnormal_units, scaled_integer_to_f64,
};
use crate::observable::{NormalizedObservable, ObservableError, Pauli, normalize_real_observable};
use crate::sample_counts::{
    MAX_SAMPLE_COUNT_BITS, MAX_SAMPLE_COUNT_CELLS, MAX_SAMPLE_COUNT_ROWS, SampleCountSummary,
    biguint_ratio,
};

/// Maximum raw or canonical terms accepted at the host polynomial boundary.
pub const MAX_POLYNOMIAL_TERMS: usize = 1_000_000;
/// Maximum variable occurrences accepted across raw polynomial terms.
pub const MAX_POLYNOMIAL_COMPONENTS: usize = 10_000_000;
/// Maximum declared variables retained by one canonical polynomial.
pub const MAX_POLYNOMIAL_VARIABLES: usize = 1_000_000;
/// Maximum source degree expanded by an exact domain substitution.
pub const MAX_POLYNOMIAL_TRANSFORM_DEGREE: usize = 20;
/// Maximum aggregate arbitrary-precision value bits across decoded samples.
pub const MAX_POLYNOMIAL_SAMPLE_VALUE_BITS: u64 = 10_000_000;

/// Value algebra of one polynomial objective.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PolynomialDomain {
    /// Boolean values encoded as zero and one.
    Binary,
    /// Ising spins encoded as minus one and plus one.
    Spin,
}

/// One nonzero canonical product coefficient.
#[derive(Clone, Debug, PartialEq)]
pub struct PolynomialTerm {
    variables: Vec<BigUint>,
    coefficient: f64,
}

/// One decoded assignment in the source polynomial's variable order.
#[derive(Clone, Debug, PartialEq)]
pub struct DecodedPolynomialSample {
    variables: Vec<BigUint>,
    values: Vec<BigInt>,
    energy: f64,
    occurrences: BigUint,
}

impl DecodedPolynomialSample {
    /// Construct one transient decoded assignment for distribution validation.
    #[must_use]
    pub fn new(
        variables: Vec<BigUint>,
        values: Vec<BigInt>,
        energy: f64,
        occurrences: BigUint,
    ) -> Self {
        Self {
            variables,
            values,
            energy,
            occurrences,
        }
    }

    /// Return strictly increasing original variable IDs.
    #[must_use]
    pub fn variables(&self) -> &[BigUint] {
        &self.variables
    }

    /// Return assignment values aligned with [`Self::variables`].
    #[must_use]
    pub fn values(&self) -> &[BigInt] {
        &self.values
    }

    /// Return the finite objective value attached to this assignment.
    #[must_use]
    pub const fn energy(&self) -> f64 {
        self.energy
    }

    /// Return the positive arbitrary-precision observation count.
    #[must_use]
    pub const fn occurrences(&self) -> &BigUint {
        &self.occurrences
    }
}

/// One validated immutable distribution over a polynomial value domain.
#[derive(Clone, Debug, PartialEq)]
pub struct DecodedPolynomialSamples {
    domain: Option<PolynomialDomain>,
    samples: Vec<DecodedPolynomialSample>,
    total_occurrences: BigUint,
    mean_energy: f64,
    lowest_index: usize,
}

impl DecodedPolynomialSamples {
    /// Return the optional BINARY/SPIN value-domain contract.
    #[must_use]
    pub const fn domain(&self) -> Option<PolynomialDomain> {
        self.domain
    }

    /// Return decoded rows in their supplied deterministic order.
    #[must_use]
    pub fn samples(&self) -> &[DecodedPolynomialSample] {
        &self.samples
    }

    /// Return the exact total number of represented observations.
    #[must_use]
    pub const fn total_occurrences(&self) -> &BigUint {
        &self.total_occurrences
    }

    /// Return the occurrence-weighted finite binary64 objective mean.
    #[must_use]
    pub const fn mean_energy(&self) -> f64 {
        self.mean_energy
    }

    /// Return the minimum-energy row index with assignment tie-breaking.
    #[must_use]
    pub const fn lowest_index(&self) -> usize {
        self.lowest_index
    }
}

impl PolynomialTerm {
    /// Return the strictly increasing arbitrary-precision variable IDs.
    #[must_use]
    pub fn variables(&self) -> &[BigUint] {
        &self.variables
    }

    /// Return the finite nonzero coefficient.
    #[must_use]
    pub const fn coefficient(&self) -> f64 {
        self.coefficient
    }
}

/// One immutable canonical sparse polynomial.
#[derive(Clone, Debug, PartialEq)]
pub struct CanonicalPolynomial {
    domain: PolynomialDomain,
    terms: Vec<PolynomialTerm>,
    constant: f64,
    variables: Vec<BigUint>,
}

impl CanonicalPolynomial {
    /// Return the value algebra.
    #[must_use]
    pub const fn domain(&self) -> PolynomialDomain {
        self.domain
    }

    /// Return lexicographically ordered nonzero product terms.
    #[must_use]
    pub fn terms(&self) -> &[PolynomialTerm] {
        &self.terms
    }

    /// Return the finite constant term.
    #[must_use]
    pub const fn constant(&self) -> f64 {
        self.constant
    }

    /// Return the complete strictly increasing variable universe.
    #[must_use]
    pub fn variables(&self) -> &[BigUint] {
        &self.variables
    }

    /// Return a model divided by its largest absolute nonconstant coefficient.
    ///
    /// A constant-only model has no normalization scale and is returned
    /// unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::CoefficientRange`] when a scaled coefficient
    /// or constant cannot be represented as finite binary64.
    pub fn normalized_by_abs_max(&self) -> Result<Self, PolynomialError> {
        let Some(factor) = self
            .terms
            .iter()
            .map(|term| term.coefficient.abs())
            .reduce(f64::max)
        else {
            return Ok(self.clone());
        };
        self.normalized_by_factor(factor)
    }

    /// Return a model divided by the sum of each degree's coefficient RMS.
    ///
    /// More precisely, the scale is the square root of the sum, across
    /// represented degrees, of each degree's mean squared coefficient. A
    /// constant-only model has no normalization scale and is returned
    /// unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::CoefficientRange`] when the normalization
    /// factor or a scaled coefficient is outside finite binary64.
    pub fn normalized_by_rms(&self) -> Result<Self, PolynomialError> {
        if self.terms.is_empty() {
            return Ok(self.clone());
        }
        let mut by_degree = BTreeMap::<usize, (f64, usize)>::new();
        for term in &self.terms {
            let entry = by_degree.entry(term.variables.len()).or_default();
            entry.0 = entry.0.hypot(term.coefficient);
            entry.1 += 1;
        }
        let mut factor = 0.0_f64;
        for (norm, count) in by_degree.values() {
            let count = u32::try_from(*count).map_err(|_| PolynomialError::CoefficientRange)?;
            factor = factor.hypot(*norm / f64::from(count).sqrt());
        }
        self.normalized_by_factor(factor)
    }

    fn normalized_by_factor(&self, factor: f64) -> Result<Self, PolynomialError> {
        if !factor.is_finite() || factor <= 0.0 {
            return Err(PolynomialError::CoefficientRange);
        }
        let coefficients = self
            .terms
            .iter()
            .map(|term| term.coefficient / factor)
            .collect::<Vec<_>>();
        let constant = self.constant / factor;
        if !constant.is_finite() || coefficients.iter().any(|value| !value.is_finite()) {
            return Err(PolynomialError::CoefficientRange);
        }
        let terms = self
            .terms
            .iter()
            .map(|term| term.variables.clone())
            .collect::<Vec<_>>();
        normalize_polynomial(
            self.domain,
            &terms,
            &coefficients,
            constant,
            Some(&self.variables),
        )
    }

    /// Evaluate one dense assignment with exact binary64 accumulation.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError`] when the assignment has the wrong width or
    /// domain, or the exactly accumulated result is outside finite binary64.
    pub fn energy(&self, values: &[i8]) -> Result<f64, PolynomialError> {
        if values.len() != self.variables.len()
            || values.iter().any(|value| !valid_value(self.domain, *value))
        {
            return Err(PolynomialError::InvalidAssignment);
        }
        let positions = self
            .variables
            .iter()
            .enumerate()
            .map(|(position, variable)| (variable, position))
            .collect::<BTreeMap<_, _>>();
        self.energy_with_positions(values, &positions)
    }

    /// Normalize one dense or original-ID assignment into canonical variable order.
    ///
    /// When `variables` is absent, `values` already follow [`Self::variables`].
    /// Otherwise variable IDs and values are paired in caller order, and the IDs
    /// must name the canonical variable universe exactly once.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::InvalidAssignment`] when the layout, width, or
    /// value domain disagrees with this polynomial.
    pub fn normalize_assignment(
        &self,
        variables: Option<&[BigUint]>,
        values: &[BigInt],
    ) -> Result<Vec<i8>, PolynomialError> {
        if values.len() != self.variables.len() {
            return Err(PolynomialError::InvalidAssignment);
        }
        let ordered_values = if let Some(variables) = variables {
            if variables.len() != values.len() {
                return Err(PolynomialError::InvalidAssignment);
            }
            let mut by_variable = BTreeMap::new();
            for (variable, value) in variables.iter().zip(values) {
                if by_variable.insert(variable, value).is_some() {
                    return Err(PolynomialError::InvalidAssignment);
                }
            }
            self.variables
                .iter()
                .map(|variable| {
                    by_variable
                        .get(variable)
                        .copied()
                        .ok_or(PolynomialError::InvalidAssignment)
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            values.iter().collect()
        };
        ordered_values
            .into_iter()
            .map(|value| assignment_value(self.domain, value))
            .collect()
    }

    /// Validate and evaluate one dense or original-ID assignment.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError`] when assignment normalization or exact
    /// binary64 evaluation fails.
    pub fn energy_assignment(
        &self,
        variables: Option<&[BigUint]>,
        values: &[BigInt],
    ) -> Result<f64, PolynomialError> {
        let values = self.normalize_assignment(variables, values)?;
        self.energy(&values)
    }

    /// Project this objective into one dense commuting Pauli-Z Hamiltonian.
    ///
    /// Original variable IDs map to consecutive logical qubits in canonical
    /// ascending order. BINARY objectives first enter the exact bounded SPIN
    /// substitution used by [`convert_polynomial`].
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialHamiltonianError`] when domain conversion or the
    /// canonical observable projection fails.
    pub fn hamiltonian(&self) -> Result<NormalizedObservable, PolynomialHamiltonianError> {
        let converted = (self.domain == PolynomialDomain::Binary)
            .then(|| convert_polynomial(self, PolynomialDomain::Spin))
            .transpose()?;
        let spin = converted.as_ref().unwrap_or(self);
        let terms = spin
            .terms
            .iter()
            .map(|term| {
                let word = term
                    .variables
                    .iter()
                    .map(|variable| {
                        spin.variables
                            .binary_search(variable)
                            .map(|position| (position, Pauli::Z))
                            .map_err(|_| PolynomialError::MissingVariable)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok((word, term.coefficient))
            })
            .collect::<Result<Vec<_>, PolynomialError>>()?;
        normalize_real_observable(spin.variables.len(), spin.constant, &terms)
            .map_err(PolynomialHamiltonianError::Observable)
    }

    fn energy_with_positions(
        &self,
        values: &[i8],
        positions: &BTreeMap<&BigUint, usize>,
    ) -> Result<f64, PolynomialError> {
        let contributions = self.terms.iter().map(|term| {
            term.coefficient
                * term
                    .variables
                    .iter()
                    .map(|variable| f64::from(values[positions[variable]]))
                    .product::<f64>()
        });
        exact_sum(std::iter::once(self.constant).chain(contributions))
            .map(canonical_zero)
            .ok_or(PolynomialError::CoefficientRange)
    }

    /// Decode one canonical logical histogram in a single native pass.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialSampleError`] when the histogram width disagrees
    /// with the model, energy evaluation fails, or the decoded distribution
    /// violates its domain or deterministic complexity budget.
    pub fn decode_samples(
        &self,
        counts: &SampleCountSummary,
    ) -> Result<DecodedPolynomialSamples, PolynomialSampleError> {
        let positions = self
            .variables
            .iter()
            .enumerate()
            .map(|(position, variable)| (variable, position))
            .collect::<BTreeMap<_, _>>();
        let mut samples = Vec::with_capacity(counts.entries().len());
        for (bits, occurrences) in counts.entries() {
            if bits.len() != self.variables.len() {
                return Err(PolynomialSampleError::Invalid("sample width"));
            }
            if occurrences.is_zero() {
                continue;
            }
            let dense_values = bits
                .iter()
                .map(|bit| {
                    let bit = i8::try_from(*bit)
                        .map_err(|_| PolynomialSampleError::Invalid("logical bit value"))?;
                    Ok(match self.domain {
                        PolynomialDomain::Binary => bit,
                        PolynomialDomain::Spin => 1 - 2 * bit,
                    })
                })
                .collect::<Result<Vec<_>, PolynomialSampleError>>()?;
            let energy = self
                .energy_with_positions(&dense_values, &positions)
                .map_err(PolynomialSampleError::Polynomial)?;
            samples.push(DecodedPolynomialSample::new(
                self.variables.clone(),
                dense_values.into_iter().map(BigInt::from).collect(),
                energy,
                occurrences.clone(),
            ));
        }
        validate_decoded_polynomial_samples(Some(self.domain), samples)
    }

    /// Evaluate finite real coordinates in dense variable order.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError`] for wrong width, non-finite coordinates,
    /// or an intermediate product or sum outside finite binary64.
    pub fn energy_reals(&self, values: &[f64]) -> Result<f64, PolynomialError> {
        if values.len() != self.variables.len() || values.iter().any(|value| !value.is_finite()) {
            return Err(PolynomialError::InvalidAssignment);
        }
        let positions = self
            .variables
            .iter()
            .enumerate()
            .map(|(position, variable)| (variable, position))
            .collect::<BTreeMap<_, _>>();
        let mut contributions = Vec::with_capacity(self.terms.len() + 1);
        contributions.push(self.constant);
        for term in &self.terms {
            let contribution = term.coefficient
                * term
                    .variables
                    .iter()
                    .map(|variable| values[positions[variable]])
                    .product::<f64>();
            if !contribution.is_finite() {
                return Err(PolynomialError::CoefficientRange);
            }
            contributions.push(contribution);
        }
        exact_sum(contributions)
            .map(canonical_zero)
            .ok_or(PolynomialError::CoefficientRange)
    }
}

/// Failure to normalize, transform, or evaluate a polynomial.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum PolynomialError {
    /// Raw coefficient rows disagree or exceed their structural budget.
    #[error("Q7561: malformed or excessive polynomial coefficient rows")]
    InvalidTerms,
    /// A coefficient or the constant is not finite.
    #[error("Q7562: polynomial coefficients must be finite")]
    NonFiniteCoefficient,
    /// The declared variable universe contains duplicates or exceeds its cap.
    #[error("Q7563: polynomial variables must not contain duplicate IDs")]
    InvalidVariables,
    /// A nonzero term names a variable absent from the declared universe.
    #[error("Q7564: coefficient variable IDs are absent from the variable universe")]
    MissingVariable,
    /// Exact coefficient accumulation cannot produce a finite binary64 value.
    #[error("Q7565: polynomial coefficient sum exceeds the finite binary64 range")]
    CoefficientRange,
    /// Domain substitution would expand beyond its deterministic budget.
    #[error("Q7566: polynomial domain conversion exceeds its expansion budget")]
    TransformBudget,
    /// A dense assignment has the wrong width or value domain.
    #[error("Q7567: polynomial assignment disagrees with its variable domain")]
    InvalidAssignment,
}

/// Failure to project a canonical polynomial into a Pauli Hamiltonian.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum PolynomialHamiltonianError {
    /// Exact BINARY-to-SPIN conversion failed.
    #[error(transparent)]
    Polynomial(#[from] PolynomialError),
    /// The dense Pauli-Z projection violated the observable contract.
    #[error(transparent)]
    Observable(#[from] ObservableError),
}

/// Failure to validate or summarize decoded optimization samples.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum PolynomialSampleError {
    /// A row, layout, value domain, or distribution invariant is invalid.
    #[error("Q7569: invalid decoded polynomial samples: {0}")]
    Invalid(&'static str),
    /// A deterministic decoded-distribution budget is exceeded.
    #[error("Q7569: decoded polynomial samples exceed the {0} budget")]
    Budget(&'static str),
    /// Occurrence-weighted statistics cannot remain finite binary64.
    #[error("Q7569: decoded polynomial sample statistics exceed finite binary64")]
    NumericalRange,
    /// Source-polynomial evaluation failed for a decoded assignment.
    #[error(transparent)]
    Polynomial(#[from] PolynomialError),
}

/// Validate and summarize one decoded optimization-sample distribution.
///
/// `domain=None` validates a standalone row without constraining its integer
/// values. A concrete domain additionally enforces BINARY or SPIN membership.
///
/// # Errors
///
/// Returns [`PolynomialSampleError`] when row shape, ordering, uniqueness,
/// value domain, energy, occurrence counts, or bounded work is invalid.
pub fn validate_decoded_polynomial_samples(
    domain: Option<PolynomialDomain>,
    samples: Vec<DecodedPolynomialSample>,
) -> Result<DecodedPolynomialSamples, PolynomialSampleError> {
    if samples.is_empty() {
        return Err(PolynomialSampleError::Invalid("empty distribution"));
    }
    if samples.len() > MAX_SAMPLE_COUNT_ROWS {
        return Err(PolynomialSampleError::Budget("row-count"));
    }
    let width = samples[0].variables.len();
    if width
        .checked_mul(samples.len())
        .is_none_or(|cells| cells > MAX_SAMPLE_COUNT_CELLS)
    {
        return Err(PolynomialSampleError::Budget("cell-count"));
    }
    let layout = &samples[0].variables;
    let mut assignments = BTreeSet::new();
    let mut total_occurrences = BigUint::ZERO;
    let mut value_bits = 0_u64;
    for sample in &samples {
        if sample.variables.len() != sample.values.len()
            || sample.variables.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(PolynomialSampleError::Invalid(
                "assignment layout must be strictly ordered",
            ));
        }
        if &sample.variables != layout {
            return Err(PolynomialSampleError::Invalid(
                "assignments must share one variable layout",
            ));
        }
        if let Some(domain) = domain
            && sample
                .values
                .iter()
                .any(|value| !valid_decoded_value(domain, value))
        {
            return Err(PolynomialSampleError::Invalid(
                "assignment value disagrees with its domain",
            ));
        }
        if !sample.energy.is_finite() {
            return Err(PolynomialSampleError::Invalid("energy must be finite"));
        }
        if sample.occurrences.is_zero() {
            return Err(PolynomialSampleError::Invalid(
                "occurrences must be positive",
            ));
        }
        if sample.occurrences.bits() > MAX_SAMPLE_COUNT_BITS {
            return Err(PolynomialSampleError::Budget("numeric-size"));
        }
        if !assignments.insert(sample.values.clone()) {
            return Err(PolynomialSampleError::Invalid("assignments must be unique"));
        }
        value_bits = sample
            .values
            .iter()
            .try_fold(value_bits, |total, value| total.checked_add(value.bits()))
            .ok_or(PolynomialSampleError::Budget("value-size"))?;
        if value_bits > MAX_POLYNOMIAL_SAMPLE_VALUE_BITS {
            return Err(PolynomialSampleError::Budget("value-size"));
        }
        total_occurrences += &sample.occurrences;
        if total_occurrences.bits() > MAX_SAMPLE_COUNT_BITS {
            return Err(PolynomialSampleError::Budget("numeric-size"));
        }
    }
    let contributions = samples.iter().map(|sample| {
        biguint_ratio(&sample.occurrences, &total_occurrences)
            .map(|weight| sample.energy * weight)
            .filter(|value| value.is_finite())
            .ok_or(PolynomialSampleError::NumericalRange)
    });
    let mean_energy = exact_sum(contributions.collect::<Result<Vec<_>, _>>()?)
        .ok_or(PolynomialSampleError::NumericalRange)?;
    let lowest_index = samples
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| {
            left.energy
                .partial_cmp(&right.energy)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.variables.cmp(&right.variables))
                .then_with(|| left.values.cmp(&right.values))
        })
        .map(|(index, _)| index)
        .ok_or(PolynomialSampleError::Invalid("empty distribution"))?;
    Ok(DecodedPolynomialSamples {
        domain,
        samples,
        total_occurrences,
        mean_energy: canonical_zero(mean_energy),
        lowest_index,
    })
}

/// Normalize arbitrary raw product rows into one canonical polynomial.
///
/// BINARY products use idempotence and SPIN products use parity. Like terms
/// and constants are accumulated exactly in binary64 subnormal units.
///
/// # Errors
///
/// Returns [`PolynomialError`] for malformed rows, invalid variable universes,
/// non-finite coefficients, unrepresentable sums, or exceeded size budgets.
pub fn normalize_polynomial(
    domain: PolynomialDomain,
    raw_terms: &[Vec<BigUint>],
    coefficients: &[f64],
    constant: f64,
    explicit_variables: Option<&[BigUint]>,
) -> Result<CanonicalPolynomial, PolynomialError> {
    if raw_terms.len() != coefficients.len() || raw_terms.len() > MAX_POLYNOMIAL_TERMS {
        return Err(PolynomialError::InvalidTerms);
    }
    if !constant.is_finite() || coefficients.iter().any(|value| !value.is_finite()) {
        return Err(PolynomialError::NonFiniteCoefficient);
    }
    let component_count = raw_terms
        .iter()
        .try_fold(0_usize, |total, row| total.checked_add(row.len()));
    if component_count.is_none_or(|count| count > MAX_POLYNOMIAL_COMPONENTS) {
        return Err(PolynomialError::InvalidTerms);
    }

    let mut grouped = BTreeMap::<Vec<BigUint>, Vec<f64>>::new();
    grouped.entry(Vec::new()).or_default().push(constant);
    for (raw_variables, coefficient) in raw_terms.iter().zip(coefficients) {
        let variables = reduce_variables(domain, raw_variables);
        grouped.entry(variables).or_default().push(*coefficient);
    }

    let constant_values = grouped
        .remove(&Vec::new())
        .ok_or(PolynomialError::InvalidTerms)?;
    let constant = exact_sum(constant_values)
        .map(canonical_zero)
        .ok_or(PolynomialError::CoefficientRange)?;
    let mut terms = Vec::with_capacity(grouped.len());
    let mut inferred = BTreeSet::new();
    for (variables, values) in grouped {
        let coefficient = exact_sum(values)
            .map(canonical_zero)
            .ok_or(PolynomialError::CoefficientRange)?;
        if coefficient != 0.0 {
            inferred.extend(variables.iter().cloned());
            terms.push(PolynomialTerm {
                variables,
                coefficient,
            });
        }
    }
    let variables = variable_universe(explicit_variables, &inferred)?;
    Ok(CanonicalPolynomial {
        domain,
        terms,
        constant,
        variables,
    })
}

/// Convert one canonical polynomial into the other value algebra.
///
/// # Errors
///
/// Returns [`PolynomialError`] when bounded subset expansion or finite
/// coefficient accumulation cannot complete.
pub fn convert_polynomial(
    polynomial: &CanonicalPolynomial,
    target: PolynomialDomain,
) -> Result<CanonicalPolynomial, PolynomialError> {
    if polynomial.domain == target {
        return Ok(polynomial.clone());
    }
    let mut expansion_count = 0_usize;
    let mut component_count = 0_usize;
    let mut max_degree = 0_usize;
    for term in &polynomial.terms {
        let degree = term.variables.len();
        if degree > MAX_POLYNOMIAL_TRANSFORM_DEGREE {
            return Err(PolynomialError::TransformBudget);
        }
        max_degree = max_degree.max(degree);
        expansion_count = expansion_count
            .checked_add(1_usize << degree)
            .ok_or(PolynomialError::TransformBudget)?;
        let expanded_components = degree
            .checked_mul(1_usize << degree.saturating_sub(1))
            .ok_or(PolynomialError::TransformBudget)?;
        component_count = component_count
            .checked_add(expanded_components)
            .ok_or(PolynomialError::TransformBudget)?;
        if expansion_count > MAX_POLYNOMIAL_TERMS || component_count > MAX_POLYNOMIAL_COMPONENTS {
            return Err(PolynomialError::TransformBudget);
        }
    }

    let denominator_extra = if polynomial.domain == PolynomialDomain::Binary {
        max_degree
    } else {
        0
    };
    let denominator_exponent = SUBNORMAL_EXPONENT
        .checked_add(denominator_extra)
        .ok_or(PolynomialError::TransformBudget)?;
    let mut grouped = BTreeMap::<Vec<BigUint>, BigInt>::new();
    grouped.insert(
        Vec::new(),
        float_to_subnormal_units(polynomial.constant) << denominator_extra,
    );
    for term in &polynomial.terms {
        let degree = term.variables.len();
        let units = float_to_subnormal_units(term.coefficient);
        for mask in 0..(1_usize << degree) {
            let variables = term
                .variables
                .iter()
                .enumerate()
                .filter(|(position, _)| mask & (1_usize << position) != 0)
                .map(|(_, variable)| variable.clone())
                .collect::<Vec<_>>();
            let selected =
                usize::try_from(mask.count_ones()).map_err(|_| PolynomialError::TransformBudget)?;
            let shift = match polynomial.domain {
                PolynomialDomain::Binary => max_degree - degree,
                PolynomialDomain::Spin => selected,
            };
            let contribution = &units << shift;
            let total = grouped.entry(variables).or_default();
            if selected % 2 == 0 {
                *total += contribution;
            } else {
                *total -= contribution;
            }
        }
    }

    let constant_units = grouped
        .remove(&Vec::new())
        .ok_or(PolynomialError::InvalidTerms)?;
    let constant = scaled_integer_to_f64(&constant_units, denominator_exponent)
        .map(canonical_zero)
        .ok_or(PolynomialError::CoefficientRange)?;
    let mut terms = Vec::with_capacity(grouped.len());
    for (variables, units) in grouped {
        let coefficient = scaled_integer_to_f64(&units, denominator_exponent)
            .map(canonical_zero)
            .ok_or(PolynomialError::CoefficientRange)?;
        if coefficient != 0.0 {
            terms.push(PolynomialTerm {
                variables,
                coefficient,
            });
        }
    }
    Ok(CanonicalPolynomial {
        domain: target,
        terms,
        constant,
        variables: polynomial.variables.clone(),
    })
}

fn reduce_variables(domain: PolynomialDomain, raw: &[BigUint]) -> Vec<BigUint> {
    let mut counts = BTreeMap::<BigUint, usize>::new();
    for variable in raw {
        *counts.entry(variable.clone()).or_default() += 1;
    }
    counts
        .into_iter()
        .filter(|(_, count)| domain == PolynomialDomain::Binary || count % 2 == 1)
        .map(|(variable, _)| variable)
        .collect()
}

fn variable_universe(
    explicit: Option<&[BigUint]>,
    inferred: &BTreeSet<BigUint>,
) -> Result<Vec<BigUint>, PolynomialError> {
    let Some(explicit) = explicit else {
        if inferred.len() > MAX_POLYNOMIAL_VARIABLES {
            return Err(PolynomialError::InvalidVariables);
        }
        return Ok(inferred.iter().cloned().collect());
    };
    if explicit.len() > MAX_POLYNOMIAL_VARIABLES {
        return Err(PolynomialError::InvalidVariables);
    }
    let variables = explicit.iter().cloned().collect::<BTreeSet<_>>();
    if variables.len() != explicit.len() {
        return Err(PolynomialError::InvalidVariables);
    }
    if !inferred.is_subset(&variables) {
        return Err(PolynomialError::MissingVariable);
    }
    Ok(variables.into_iter().collect())
}

fn valid_value(domain: PolynomialDomain, value: i8) -> bool {
    match domain {
        PolynomialDomain::Binary => matches!(value, 0 | 1),
        PolynomialDomain::Spin => matches!(value, -1 | 1),
    }
}

fn valid_decoded_value(domain: PolynomialDomain, value: &BigInt) -> bool {
    match domain {
        PolynomialDomain::Binary => value == &BigInt::ZERO || value == &BigInt::from(1_u8),
        PolynomialDomain::Spin => value == &BigInt::from(-1_i8) || value == &BigInt::from(1_u8),
    }
}

fn assignment_value(domain: PolynomialDomain, value: &BigInt) -> Result<i8, PolynomialError> {
    match domain {
        PolynomialDomain::Binary if value == &BigInt::ZERO => Ok(0),
        PolynomialDomain::Spin if value == &BigInt::from(-1_i8) => Ok(-1),
        PolynomialDomain::Binary | PolynomialDomain::Spin if value == &BigInt::from(1_u8) => Ok(1),
        PolynomialDomain::Binary | PolynomialDomain::Spin => {
            Err(PolynomialError::InvalidAssignment)
        }
    }
}

fn canonical_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample_counts::normalize_sample_counts;

    fn count_row(bits: &[u8], count: BigInt) -> (Box<[u8]>, BigInt) {
        (bits.into(), count)
    }

    #[test]
    fn normalizes_binary_idempotence_spin_parity_and_exact_aliases() {
        let residue = f64::from_bits(1);
        let binary = normalize_polynomial(
            PolynomialDomain::Binary,
            &[
                vec![9_u8.into(), 9_u8.into(), 2_u8.into()],
                vec![2_u8.into(), 9_u8.into()],
            ],
            &[f64::MAX, -f64::MAX],
            residue,
            Some(&[12_u8.into(), 2_u8.into(), 9_u8.into()]),
        )
        .expect("binary polynomial");
        let spin = normalize_polynomial(
            PolynomialDomain::Spin,
            &[
                vec![9_u8.into(), 9_u8.into(), 2_u8.into()],
                vec![2_u8.into()],
            ],
            &[3.0, -1.0],
            0.0,
            None,
        )
        .expect("spin polynomial");

        assert!(binary.terms().is_empty());
        assert_eq!(binary.constant().to_bits(), residue.to_bits());
        assert_eq!(
            binary.variables(),
            &[2_u8.into(), 9_u8.into(), 12_u8.into()]
        );
        assert_eq!(spin.terms()[0].variables(), &[2_u8.into()]);
        assert_eq!(spin.terms()[0].coefficient().to_bits(), 2.0_f64.to_bits());
    }

    #[test]
    fn coefficient_normalizations_preserve_domain_variables_and_energy_scale() {
        let polynomial = normalize_polynomial(
            PolynomialDomain::Binary,
            &[
                vec![1_u8.into()],
                vec![3_u8.into()],
                vec![1_u8.into(), 3_u8.into()],
            ],
            &[1.0, 2.0, 3.0],
            5.0,
            Some(&[8_u8.into(), 3_u8.into(), 1_u8.into()]),
        )
        .expect("binary polynomial");

        let abs_max = polynomial
            .normalized_by_abs_max()
            .expect("absolute-maximum normalization");
        let rms = polynomial.normalized_by_rms().expect("RMS normalization");
        let rms_factor = 11.5_f64.sqrt();

        assert_eq!(abs_max.domain(), PolynomialDomain::Binary);
        assert_eq!(abs_max.variables(), polynomial.variables());
        assert_eq!(abs_max.constant().to_bits(), (5.0_f64 / 3.0).to_bits());
        assert_eq!(
            abs_max.terms()[1].coefficient().to_bits(),
            1.0_f64.to_bits()
        );
        assert_eq!(rms.constant().to_bits(), (5.0_f64 / rms_factor).to_bits());
        assert_eq!(
            rms.terms()[1].coefficient().to_bits(),
            (3.0_f64 / rms_factor).to_bits()
        );
        assert_eq!(
            abs_max
                .energy(&[1, 0, 0])
                .expect("normalized energy")
                .to_bits(),
            (polynomial.energy(&[1, 0, 0]).expect("source energy") / 3.0).to_bits()
        );
    }

    #[test]
    fn coefficient_normalization_handles_constants_and_fails_on_overflow() {
        let constant =
            normalize_polynomial(PolynomialDomain::Spin, &[], &[], 2.0, Some(&[7_u8.into()]))
                .expect("constant polynomial");
        assert_eq!(
            constant.normalized_by_abs_max().expect("constant abs max"),
            constant
        );
        assert_eq!(
            constant.normalized_by_rms().expect("constant RMS"),
            constant
        );

        let extreme = normalize_polynomial(
            PolynomialDomain::Binary,
            &[vec![0_u8.into()]],
            &[f64::from_bits(1)],
            f64::MAX,
            None,
        )
        .expect("extreme polynomial");
        assert_eq!(
            extreme.normalized_by_abs_max(),
            Err(PolynomialError::CoefficientRange)
        );
        assert_eq!(
            extreme.normalized_by_rms(),
            Err(PolynomialError::CoefficientRange)
        );
    }

    #[test]
    fn normalizes_dense_and_original_id_assignments_once() {
        let polynomial = normalize_polynomial(
            PolynomialDomain::Binary,
            &[vec![2_u8.into()], vec![2_u8.into(), 9_u8.into()]],
            &[1.0, 3.0],
            0.5,
            Some(&[9_u8.into(), 2_u8.into(), 12_u8.into()]),
        )
        .expect("binary polynomial");
        let mapping_ids = vec![12_u8.into(), 9_u8.into(), 2_u8.into()];
        let mapping_values = vec![0_u8.into(), 1_u8.into(), 1_u8.into()];

        assert_eq!(
            polynomial
                .normalize_assignment(Some(&mapping_ids), &mapping_values)
                .expect("original-ID assignment"),
            vec![1, 1, 0]
        );
        assert_eq!(
            polynomial
                .energy_assignment(Some(&mapping_ids), &mapping_values)
                .expect("original-ID energy")
                .to_bits(),
            4.5_f64.to_bits()
        );
        assert_eq!(
            polynomial
                .energy_assignment(None, &[1_u8.into(), 1_u8.into(), 0_u8.into()],)
                .expect("dense energy")
                .to_bits(),
            4.5_f64.to_bits()
        );
        let hamiltonian = polynomial.hamiltonian().expect("Pauli Hamiltonian");
        let expected_hamiltonian = normalize_real_observable(
            3,
            1.75,
            &[
                (vec![(0, Pauli::Z)], -1.25),
                (vec![(0, Pauli::Z), (1, Pauli::Z)], 0.75),
                (vec![(1, Pauli::Z)], -0.75),
            ],
        )
        .expect("expected Pauli Hamiltonian");
        assert_eq!(hamiltonian, expected_hamiltonian);
        for (ids, values) in [
            (
                Some(vec![2_u8.into(), 9_u8.into()]),
                vec![1_u8.into(), 1_u8.into()],
            ),
            (
                Some(vec![2_u8.into(), 9_u8.into(), 13_u8.into()]),
                vec![1_u8.into(), 1_u8.into(), 0_u8.into()],
            ),
            (
                Some(vec![2_u8.into(), 2_u8.into(), 12_u8.into()]),
                vec![1_u8.into(), 1_u8.into(), 0_u8.into()],
            ),
            (None, vec![1_u8.into(), 2_u8.into(), 0_u8.into()]),
        ] {
            assert_eq!(
                polynomial.normalize_assignment(ids.as_deref(), &values),
                Err(PolynomialError::InvalidAssignment)
            );
        }
    }

    #[test]
    fn decodes_one_histogram_with_native_domain_energy_and_statistics() {
        let polynomial = normalize_polynomial(
            PolynomialDomain::Spin,
            &[vec![2_u8.into()]],
            &[1.0],
            0.0,
            Some(&[2_u8.into()]),
        )
        .expect("spin polynomial");
        let counts = normalize_sample_counts(
            &[
                count_row(&[1], BigInt::from(1_u8)),
                count_row(&[0], BigInt::from(3_u8)),
                count_row(&[0], BigInt::ZERO),
            ],
            Some(1),
        );
        assert_eq!(
            counts,
            Err(crate::sample_counts::SampleCountError::DuplicateRow)
        );
        let counts = normalize_sample_counts(
            &[
                count_row(&[1], BigInt::from(1_u8)),
                count_row(&[0], BigInt::from(3_u8)),
            ],
            Some(1),
        )
        .expect("counts");

        let decoded = polynomial.decode_samples(&counts).expect("decoded samples");

        assert_eq!(decoded.domain(), Some(PolynomialDomain::Spin));
        assert_eq!(decoded.samples().len(), 2);
        assert_eq!(decoded.samples()[0].variables(), &[2_u8.into()]);
        assert_eq!(decoded.samples()[0].values(), &[BigInt::from(1_u8)]);
        assert_eq!(decoded.samples()[0].energy().to_bits(), 1.0_f64.to_bits());
        assert_eq!(decoded.samples()[1].values(), &[BigInt::from(-1_i8)]);
        assert_eq!(decoded.total_occurrences(), &BigUint::from(4_u8));
        assert_eq!(decoded.mean_energy().to_bits(), 0.5_f64.to_bits());
        assert_eq!(decoded.lowest_index(), 1);
    }

    #[test]
    fn decoded_distribution_rejects_shape_domain_and_duplicate_rows() {
        let row = |variables: &[u8], values: &[i8]| {
            DecodedPolynomialSample::new(
                variables.iter().copied().map(BigUint::from).collect(),
                values.iter().copied().map(BigInt::from).collect(),
                0.0,
                BigUint::from(1_u8),
            )
        };
        assert!(matches!(
            validate_decoded_polynomial_samples(
                Some(PolynomialDomain::Binary),
                vec![row(&[1, 0], &[0, 1])],
            ),
            Err(PolynomialSampleError::Invalid(message))
                if message.contains("strictly ordered")
        ));
        assert!(matches!(
            validate_decoded_polynomial_samples(
                Some(PolynomialDomain::Binary),
                vec![row(&[0], &[-1])],
            ),
            Err(PolynomialSampleError::Invalid(message))
                if message.contains("domain")
        ));
        assert!(matches!(
            validate_decoded_polynomial_samples(
                Some(PolynomialDomain::Spin),
                vec![row(&[0], &[1]), row(&[0], &[1])],
            ),
            Err(PolynomialSampleError::Invalid(message))
                if message.contains("unique")
        ));
    }

    #[test]
    fn decoded_mean_handles_arbitrary_precision_occurrences_without_float_conversion() {
        let huge = BigUint::from(1_u8) << 100_000_usize;
        let decoded = validate_decoded_polynomial_samples(
            Some(PolynomialDomain::Binary),
            vec![
                DecodedPolynomialSample::new(
                    vec![0_u8.into()],
                    vec![BigInt::ZERO],
                    f64::MAX,
                    huge.clone(),
                ),
                DecodedPolynomialSample::new(
                    vec![0_u8.into()],
                    vec![BigInt::from(1_u8)],
                    -f64::MAX,
                    huge,
                ),
            ],
        )
        .expect("decoded samples");

        assert_eq!(decoded.mean_energy().to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn domain_round_trip_preserves_all_small_model_energies() {
        let binary = normalize_polynomial(
            PolynomialDomain::Binary,
            &[
                vec![1_u8.into()],
                vec![1_u8.into(), 4_u8.into(), 7_u8.into()],
            ],
            &[-0.25, 1.5],
            0.125,
            None,
        )
        .expect("binary polynomial");
        let spin = convert_polynomial(&binary, PolynomialDomain::Spin).expect("to spin");
        let restored = convert_polynomial(&spin, PolynomialDomain::Binary).expect("to binary");

        for bits in 0_u8..8 {
            let binary_values = (0..3)
                .map(|position| i8::from(bits & (1 << position) != 0))
                .collect::<Vec<_>>();
            let spin_values = binary_values
                .iter()
                .map(|value| 1 - 2 * value)
                .collect::<Vec<_>>();
            assert!(
                (binary.energy(&binary_values).expect("binary energy")
                    - spin.energy(&spin_values).expect("spin energy"))
                .abs()
                    < 1e-12
            );
            assert!(
                (binary.energy(&binary_values).expect("binary energy")
                    - restored.energy(&binary_values).expect("restored energy"))
                .abs()
                    < 1e-12
            );
        }
    }

    #[test]
    fn continuous_energy_accumulates_term_contributions_exactly() {
        let residue = f64::from_bits(1);
        let polynomial = normalize_polynomial(
            PolynomialDomain::Spin,
            &[vec![0_u8.into()], vec![1_u8.into()], vec![2_u8.into()]],
            &[f64::MAX, -f64::MAX, residue],
            0.0,
            None,
        )
        .expect("polynomial");

        assert_eq!(
            polynomial
                .energy_reals(&[1.0, 1.0, 1.0])
                .expect("continuous energy")
                .to_bits(),
            residue.to_bits()
        );
        assert_eq!(
            polynomial.energy_reals(&[1.0, f64::NAN, 1.0]),
            Err(PolynomialError::InvalidAssignment)
        );
    }

    #[test]
    fn invalid_universe_assignment_and_expansion_fail_closed() {
        assert_eq!(
            normalize_polynomial(
                PolynomialDomain::Binary,
                &[vec![1_u8.into()]],
                &[1.0],
                0.0,
                Some(&[0_u8.into()]),
            ),
            Err(PolynomialError::MissingVariable)
        );
        let polynomial = normalize_polynomial(
            PolynomialDomain::Binary,
            &[vec![0_u8.into()]],
            &[1.0],
            0.0,
            None,
        )
        .expect("polynomial");
        assert_eq!(
            polynomial.energy(&[2]),
            Err(PolynomialError::InvalidAssignment)
        );
        let excessive = normalize_polynomial(
            PolynomialDomain::Binary,
            &[(0..=MAX_POLYNOMIAL_TRANSFORM_DEGREE)
                .map(BigUint::from)
                .collect()],
            &[1.0],
            0.0,
            None,
        )
        .expect("high-order polynomial");
        assert_eq!(
            convert_polynomial(&excessive, PolynomialDomain::Spin),
            Err(PolynomialError::TransformBudget)
        );
    }
}
