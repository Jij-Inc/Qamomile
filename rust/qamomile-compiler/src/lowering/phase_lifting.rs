//! Exact coherent lifting for pure Boolean `QCore` functions.
//!
//! A function is normalized coordinate-wise into algebraic normal form over the
//! Boolean ring. Each retained monomial denotes one exact controlled phase or
//! target-X contribution. This avoids truth-table enumeration and temporary
//! quantum storage while preserving constant terms under coherent control.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use num_bigint::BigInt;
use thiserror::Error;

use crate::operations::LocallyCheckedOperations;
use crate::static_values::normalize_aggregate_index;
use crate::types::{Dimension, ScalarType, SemanticType, TypeArena, WordKind};
use crate::verification::QCoreVerificationWork;
use crate::wire;

/// Largest coherent Boolean input accepted by the native lifting proof.
pub const MAX_PHASE_LIFT_INPUT_BITS: u64 = 4_096;
/// Largest coherent Boolean output accepted by the native lifting proof.
pub const MAX_BOOLEAN_LIFT_OUTPUT_BITS: u64 = 4_096;
/// Largest canonical ANF retained by one lifted predicate.
pub const MAX_PHASE_LIFT_TERMS: usize = 100_000;
/// Largest retained source-variable occurrence count in one Boolean lift.
pub const MAX_BOOLEAN_LIFT_VARIABLE_OCCURRENCES: usize = 1_000_000;
/// Largest aggregate ANF retained across one decoded program.
const MAX_PHASE_LIFT_PLAN_TERMS: usize = 1_000_000;
/// Largest distributive-product work performed while normalizing one predicate.
pub const MAX_PHASE_LIFT_PRODUCT_WORK: usize = 1_000_000;
/// Largest nested pure-call or structured-conditional depth.
pub const MAX_PHASE_LIFT_DEPTH: usize = 256;
/// Largest input accepted by the bounded exhaustive permutation proof.
pub const MAX_GENERAL_BOOLEAN_PERMUTATION_BITS: u64 = 10;
/// Largest truth-table proof work accepted for one general permutation.
pub const MAX_GENERAL_BOOLEAN_PERMUTATION_WORK: usize = 50_000_000;
/// Largest primitive circuit expansion accepted for one general permutation.
pub const MAX_GENERAL_BOOLEAN_PERMUTATION_INSTRUCTIONS: usize = 100_000;

/// One basis-state transposition in a proved general Boolean permutation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BooleanBasisTransposition {
    left: u64,
    right: u64,
}

impl BooleanBasisTransposition {
    /// Return the first distinct basis-state index.
    #[must_use]
    pub const fn left(&self) -> u64 {
        self.left
    }

    /// Return the second distinct basis-state index.
    #[must_use]
    pub const fn right(&self) -> u64 {
        self.right
    }
}

/// One exact Boolean monomial represented by ascending source bit positions.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BooleanMonomial {
    variables: Box<[u64]>,
}

impl BooleanMonomial {
    /// Return the source-order input bits in this conjunction.
    #[must_use]
    pub fn variables(&self) -> &[u64] {
        &self.variables
    }
}

/// One source-ordered output coordinate in an exact Boolean lift.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BooleanLiftOutput {
    monomials: Box<[BooleanMonomial]>,
}

impl BooleanLiftOutput {
    /// Return canonical ANF monomials in degree-lexicographic order.
    #[must_use]
    pub fn monomials(&self) -> &[BooleanMonomial] {
        &self.monomials
    }
}

/// Canonical exact ANF realization of one Boolean function.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BooleanLiftPlan {
    input_width: u64,
    outputs: Box<[BooleanLiftOutput]>,
    permutation_order: Option<Box<[u64]>>,
    basis_transpositions: Option<Box<[BooleanBasisTransposition]>>,
}

/// Independently verified phase plans retained by one decoded program.
#[derive(Clone, Debug, Default)]
pub(crate) struct VerifiedBooleanLiftPlans {
    plans: BTreeMap<String, BooleanLiftPlan>,
    term_count: usize,
    transposition_count: usize,
}

impl VerifiedBooleanLiftPlans {
    /// Return the verified plan for one referenced predicate.
    #[must_use]
    pub(crate) fn get(&self, predicate: &str) -> Option<&BooleanLiftPlan> {
        self.plans.get(predicate)
    }

    fn get_mut(&mut self, predicate: &str) -> Option<&mut BooleanLiftPlan> {
        self.plans.get_mut(predicate)
    }

    /// Iterate over verified plans in canonical predicate-symbol order.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (&str, &BooleanLiftPlan)> {
        self.plans
            .iter()
            .map(|(predicate, plan)| (predicate.as_str(), plan))
    }

    fn insert(&mut self, predicate: String, plan: BooleanLiftPlan) -> Result<(), BooleanLiftError> {
        self.insert_with_limit(predicate, plan, MAX_PHASE_LIFT_PLAN_TERMS)
    }

    fn insert_with_limit(
        &mut self,
        predicate: String,
        plan: BooleanLiftPlan,
        limit: usize,
    ) -> Result<(), BooleanLiftError> {
        let plan_term_count = plan
            .outputs
            .iter()
            .try_fold(0_usize, |count, output| {
                count.checked_add(output.monomials.len())
            })
            .ok_or(BooleanLiftError::ProgramTooManyTerms)?;
        let term_count = self
            .term_count
            .checked_add(plan_term_count)
            .filter(|count| *count <= limit)
            .ok_or(BooleanLiftError::ProgramTooManyTerms)?;
        self.plans.insert(predicate, plan);
        self.term_count = term_count;
        Ok(())
    }
}

impl BooleanLiftPlan {
    /// Return the fixed source-order predicate width.
    #[must_use]
    pub const fn input_width(&self) -> u64 {
        self.input_width
    }

    /// Return canonical monomials for the sole phase-compatible output.
    ///
    /// Phase emission uses a verified single-output predicate. General
    /// Boolean-lift consumers use [`Self::outputs`] for all coordinates.
    #[must_use]
    pub fn monomials(&self) -> &[BooleanMonomial] {
        self.outputs
            .first()
            .map_or(&[], BooleanLiftOutput::monomials)
    }

    /// Return every source-ordered Boolean output coordinate.
    #[must_use]
    pub fn outputs(&self) -> &[BooleanLiftOutput] {
        &self.outputs
    }

    /// Return the fixed number of Boolean output coordinates.
    #[must_use]
    pub fn output_width(&self) -> u64 {
        u64::try_from(self.outputs.len()).unwrap_or(u64::MAX)
    }

    /// Return a proved target-before-dependency in-place permutation schedule.
    #[must_use]
    pub fn permutation_order(&self) -> Option<&[u64]> {
        self.permutation_order.as_deref()
    }

    /// Return a replayed chronological basis-transposition realization.
    #[must_use]
    pub fn basis_transpositions(&self) -> Option<&[BooleanBasisTransposition]> {
        self.basis_transpositions.as_deref()
    }
}

/// Failure to prove and finitely normalize one coherent Boolean lift.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum BooleanLiftError {
    /// The referenced callable is not one supported fixed-width Boolean function.
    #[error(
        "Q7680: Boolean function {0:?} must be body-backed Bits[N] -> Bool or tuple[Bool, ...]"
    )]
    InvalidPredicate(String),
    /// The predicate uses a classical construct outside the proved Boolean graph.
    #[error("Q7681: Boolean predicate uses unsupported construct {0:?}")]
    Unsupported(String),
    /// The predicate input exceeds the explicit coherent-width budget.
    #[error("Q7682: Boolean predicate input exceeds the {MAX_PHASE_LIFT_INPUT_BITS}-bit budget")]
    InputTooWide,
    /// Canonical Boolean-ring normalization would retain too many terms.
    #[error("Q7683: Boolean predicate exceeds the {MAX_PHASE_LIFT_TERMS}-term budget")]
    TooManyTerms,
    /// Distributive Boolean-ring normalization would perform excessive work.
    #[error(
        "Q7684: Boolean predicate exceeds the {MAX_PHASE_LIFT_PRODUCT_WORK}-product-work budget"
    )]
    ProductWorkExceeded,
    /// Nested helpers or structured predicates exceed the proof depth.
    #[error("Q7685: Boolean predicate exceeds the {MAX_PHASE_LIFT_DEPTH}-level depth budget")]
    DepthExceeded,
    /// Verified compiler facts required by the independent proof are absent.
    #[error("Q7686: Boolean predicate proof is missing {0}")]
    Missing(String),
    /// The fixed Boolean output exceeds the explicit coherent-width budget.
    #[error(
        "Q7688: Boolean predicate output exceeds the {MAX_BOOLEAN_LIFT_OUTPUT_BITS}-bit budget"
    )]
    OutputTooWide,
    /// Retained monomials contain too many source-variable occurrences.
    #[error(
        "Q7689: Boolean predicate exceeds the {MAX_BOOLEAN_LIFT_VARIABLE_OCCURRENCES}-variable-occurrence budget"
    )]
    ProofWorkExceeded,
    /// The selected function is not bijective in the proved finite domain.
    #[error("Q7690: Boolean function {0:?} is not a proved Boolean permutation")]
    InvalidPermutation(String),
    /// A non-triangular permutation exceeds the exhaustive width budget.
    #[error(
        "Q7691: general Boolean permutation exceeds the {MAX_GENERAL_BOOLEAN_PERMUTATION_BITS}-bit proof budget"
    )]
    GeneralPermutationTooWide,
    /// Exhaustive evaluation would exceed the explicit proof-work budget.
    #[error(
        "Q7692: general Boolean permutation exceeds the {MAX_GENERAL_BOOLEAN_PERMUTATION_WORK}-unit proof-work budget"
    )]
    GeneralPermutationWorkExceeded,
    /// The late primitive realization would exceed its output budget.
    #[error(
        "Q7693: general Boolean permutation exceeds the {MAX_GENERAL_BOOLEAN_PERMUTATION_INSTRUCTIONS}-instruction realization budget"
    )]
    GeneralPermutationOutputExceeded,
    /// An internally produced transposition proof failed independent replay.
    #[error("Q7694: invalid general Boolean permutation certificate")]
    InvalidPermutationCertificate,
    /// Retaining all independently proved predicates would exceed program memory bounds.
    #[error(
        "Q7687: program exceeds the {MAX_PHASE_LIFT_PLAN_TERMS}-term retained Boolean-plan budget"
    )]
    ProgramTooManyTerms,
}

type Monomial = Vec<u64>;

#[derive(Clone, Debug, Eq, PartialEq)]
struct BooleanAnf {
    terms: BTreeSet<Monomial>,
}

impl BooleanAnf {
    fn constant(value: bool) -> Self {
        Self {
            terms: if value {
                BTreeSet::from([Vec::new()])
            } else {
                BTreeSet::new()
            },
        }
    }

    fn variable(index: u64) -> Self {
        Self {
            terms: BTreeSet::from([vec![index]]),
        }
    }

    fn xor(mut self, other: &Self) -> Result<Self, BooleanLiftError> {
        for term in &other.terms {
            if !self.terms.remove(term) {
                self.terms.insert(term.clone());
                if self.terms.len() > MAX_PHASE_LIFT_TERMS {
                    return Err(BooleanLiftError::TooManyTerms);
                }
            }
        }
        Ok(self)
    }

    fn not(self) -> Result<Self, BooleanLiftError> {
        self.xor(&Self::constant(true))
    }

    fn and(&self, other: &Self, product_work: &mut usize) -> Result<Self, BooleanLiftError> {
        let additional = self
            .terms
            .len()
            .checked_mul(other.terms.len())
            .ok_or(BooleanLiftError::ProductWorkExceeded)?;
        *product_work = product_work
            .checked_add(additional)
            .filter(|work| *work <= MAX_PHASE_LIFT_PRODUCT_WORK)
            .ok_or(BooleanLiftError::ProductWorkExceeded)?;
        let mut result = Self::constant(false);
        for left in &self.terms {
            for right in &other.terms {
                let mut product = Vec::with_capacity(left.len() + right.len());
                product.extend(left.iter().copied());
                product.extend(right.iter().copied());
                product.sort_unstable();
                product.dedup();
                if !result.terms.remove(&product) {
                    result.terms.insert(product);
                    if result.terms.len() > MAX_PHASE_LIFT_TERMS {
                        return Err(BooleanLiftError::TooManyTerms);
                    }
                }
            }
        }
        Ok(result)
    }

    fn or(self, other: &Self, product_work: &mut usize) -> Result<Self, BooleanLiftError> {
        let product = self.and(other, product_work)?;
        self.xor(other)?.xor(&product)
    }

    fn select(
        condition: &Self,
        when_true: &Self,
        when_false: &Self,
        product_work: &mut usize,
    ) -> Result<Self, BooleanLiftError> {
        let difference = when_true.clone().xor(when_false)?;
        when_false
            .clone()
            .xor(&condition.and(&difference, product_work)?)
    }
}

#[derive(Clone, Debug)]
enum PredicateValue {
    Bits { width: usize },
    Bool(BooleanAnf),
    Int(BigInt),
    Product(Box<[PredicateValue]>),
}

struct Planner<'a> {
    functions: &'a HashMap<&'a str, &'a wire::FunctionProto>,
    proof_work: usize,
    product_work: usize,
}

fn plan_with_functions<'a>(
    functions: &'a HashMap<&'a str, &'a wire::FunctionProto>,
    predicate: &str,
    arena: &TypeArena,
    operations: &LocallyCheckedOperations,
    work: &mut QCoreVerificationWork,
) -> Result<BooleanLiftPlan, BooleanLiftError> {
    work.charge_phase_lifting(1);
    let function = functions
        .get(predicate)
        .copied()
        .ok_or_else(|| BooleanLiftError::InvalidPredicate(predicate.to_owned()))?;
    let (width, output_width) = predicate_shape(function, arena, operations)?;
    if width > MAX_PHASE_LIFT_INPUT_BITS {
        return Err(BooleanLiftError::InputTooWide);
    }
    if output_width > MAX_BOOLEAN_LIFT_OUTPUT_BITS {
        return Err(BooleanLiftError::OutputTooWide);
    }
    let width_usize = usize::try_from(width).map_err(|_| BooleanLiftError::InputTooWide)?;
    let mut planner = Planner {
        functions,
        proof_work: 0,
        product_work: 0,
    };
    let value = planner.evaluate_function(
        function,
        vec![PredicateValue::Bits { width: width_usize }],
        0,
    )?;
    let results = match value {
        PredicateValue::Bool(result) => vec![result],
        PredicateValue::Product(values) => values
            .into_vec()
            .into_iter()
            .map(|value| match value {
                PredicateValue::Bool(result) => Ok(result),
                _ => Err(BooleanLiftError::InvalidPredicate(predicate.to_owned())),
            })
            .collect::<Result<Vec<_>, _>>()?,
        _ => return Err(BooleanLiftError::InvalidPredicate(predicate.to_owned())),
    };
    if u64::try_from(results.len()).ok() != Some(output_width) {
        return Err(BooleanLiftError::InvalidPredicate(predicate.to_owned()));
    }
    let mut retained_terms = 0_usize;
    let mut variable_occurrences = 0_usize;
    let outputs = results
        .into_iter()
        .map(|result| {
            let mut terms = result.terms.into_iter().collect::<Vec<_>>();
            retained_terms = retained_terms
                .checked_add(terms.len())
                .filter(|count| *count <= MAX_PHASE_LIFT_TERMS)
                .ok_or(BooleanLiftError::TooManyTerms)?;
            variable_occurrences = extend_variable_occurrence_count(variable_occurrences, &terms)?;
            planner.charge(
                terms
                    .iter()
                    .fold(terms.len(), |count, term| count.saturating_add(term.len())),
            );
            terms.sort_by(|left, right| left.len().cmp(&right.len()).then_with(|| left.cmp(right)));
            Ok(BooleanLiftOutput {
                monomials: terms
                    .into_iter()
                    .map(|variables| BooleanMonomial {
                        variables: variables.into_boxed_slice(),
                    })
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            })
        })
        .collect::<Result<Vec<_>, BooleanLiftError>>()?;
    let permutation_order = prove_acyclic_triangular_permutation(width, &outputs);
    let plan = BooleanLiftPlan {
        input_width: width,
        outputs: outputs.into_boxed_slice(),
        permutation_order,
        basis_transpositions: None,
    };
    work.charge_phase_lifting(planner.proof_work.saturating_add(planner.product_work));
    Ok(plan)
}

fn prove_general_boolean_permutation(
    predicate: &str,
    input_width: u64,
    outputs: &[BooleanLiftOutput],
    work: &mut QCoreVerificationWork,
) -> Result<Box<[BooleanBasisTransposition]>, BooleanLiftError> {
    if input_width > MAX_GENERAL_BOOLEAN_PERMUTATION_BITS {
        return Err(BooleanLiftError::GeneralPermutationTooWide);
    }
    let width = usize::try_from(input_width)
        .ok()
        .filter(|width| *width > 0 && outputs.len() == *width)
        .ok_or(BooleanLiftError::InvalidPermutationCertificate)?;
    let state_count = 1_usize
        .checked_shl(
            u32::try_from(input_width).map_err(|_| BooleanLiftError::GeneralPermutationTooWide)?,
        )
        .ok_or(BooleanLiftError::GeneralPermutationTooWide)?;
    let evaluation_work = general_permutation_evaluation_work(
        outputs,
        state_count,
        MAX_GENERAL_BOOLEAN_PERMUTATION_WORK,
    )?;
    work.charge_phase_lifting(evaluation_work);
    let mut mapping = Vec::with_capacity(state_count);
    let mut seen = vec![false; state_count];
    for state in 0..state_count {
        let mut result = 0_usize;
        for (target, output) in outputs.iter().enumerate() {
            let active = evaluate_boolean_output(output, state, width)?;
            if active {
                result |= 1_usize << target;
            }
        }
        if seen.get(result).copied() != Some(false) {
            return Err(BooleanLiftError::InvalidPermutation(predicate.to_owned()));
        }
        seen[result] = true;
        mapping.push(result);
    }
    let mut visited = vec![false; state_count];
    let mut transpositions = Vec::new();
    let mut instruction_count = 0_usize;
    for start in 0..state_count {
        if visited[start] {
            continue;
        }
        let mut current = start;
        let mut cycle = Vec::new();
        while !visited[current] {
            visited[current] = true;
            cycle.push(current);
            current = mapping[current];
        }
        if current != start {
            return Err(BooleanLiftError::InvalidPermutationCertificate);
        }
        for right in cycle.into_iter().skip(1) {
            let left = u64::try_from(start)
                .map_err(|_| BooleanLiftError::InvalidPermutationCertificate)?;
            let right = u64::try_from(right)
                .map_err(|_| BooleanLiftError::InvalidPermutationCertificate)?;
            instruction_count = extend_general_instruction_count(
                instruction_count,
                left,
                right,
                width,
                MAX_GENERAL_BOOLEAN_PERMUTATION_INSTRUCTIONS,
            )?;
            transpositions.push(BooleanBasisTransposition { left, right });
        }
    }
    verify_basis_transpositions(&mapping, &transpositions)?;
    work.charge_phase_lifting(state_count.saturating_add(transpositions.len()));
    Ok(transpositions.into_boxed_slice())
}

fn evaluate_boolean_output(
    output: &BooleanLiftOutput,
    state: usize,
    width: usize,
) -> Result<bool, BooleanLiftError> {
    let mut result = false;
    for monomial in output.monomials() {
        let mut active = true;
        for variable in monomial.variables() {
            let variable = usize::try_from(*variable)
                .ok()
                .filter(|variable| *variable < width)
                .ok_or(BooleanLiftError::InvalidPermutationCertificate)?;
            active &= state & (1_usize << variable) != 0;
        }
        result ^= active;
    }
    Ok(result)
}

fn general_permutation_evaluation_work(
    outputs: &[BooleanLiftOutput],
    state_count: usize,
    limit: usize,
) -> Result<usize, BooleanLiftError> {
    outputs
        .iter()
        .try_fold(outputs.len(), |work, output| {
            output.monomials().iter().try_fold(work, |work, monomial| {
                work.checked_add(monomial.variables().len().saturating_add(1))
            })
        })
        .and_then(|work| work.checked_mul(state_count))
        .filter(|work| *work <= limit)
        .ok_or(BooleanLiftError::GeneralPermutationWorkExceeded)
}

fn extend_general_instruction_count(
    current: usize,
    left: u64,
    right: u64,
    width: usize,
    limit: usize,
) -> Result<usize, BooleanLiftError> {
    current
        .checked_add(transposition_primitive_count(left, right, width)?)
        .filter(|count| *count <= limit)
        .ok_or(BooleanLiftError::GeneralPermutationOutputExceeded)
}

fn transposition_primitive_count(
    left: u64,
    right: u64,
    width: usize,
) -> Result<usize, BooleanLiftError> {
    let differing = left ^ right;
    if differing == 0 {
        return Err(BooleanLiftError::InvalidPermutationCertificate);
    }
    let mut current = left;
    let mut gate_costs = Vec::new();
    for target in 0..width {
        if differing & (1_u64 << target) == 0 {
            continue;
        }
        let control_mask = !(1_u64 << target) & ((1_u64 << width) - 1);
        let zero_controls = usize::try_from(((!current) & control_mask).count_ones())
            .map_err(|_| BooleanLiftError::InvalidPermutationCertificate)?;
        gate_costs.push(1_usize.saturating_add(2_usize.saturating_mul(zero_controls)));
        current ^= 1_u64 << target;
    }
    if current != right {
        return Err(BooleanLiftError::InvalidPermutationCertificate);
    }
    let forward = gate_costs
        .iter()
        .try_fold(0_usize, |count, cost| count.checked_add(*cost));
    let reverse = gate_costs
        .iter()
        .take(gate_costs.len().saturating_sub(1))
        .try_fold(0_usize, |count, cost| count.checked_add(*cost));
    forward
        .and_then(|forward| reverse.and_then(|reverse| forward.checked_add(reverse)))
        .ok_or(BooleanLiftError::GeneralPermutationOutputExceeded)
}

fn verify_basis_transpositions(
    expected: &[usize],
    transpositions: &[BooleanBasisTransposition],
) -> Result<(), BooleanLiftError> {
    let mut realized = (0..expected.len()).collect::<Vec<_>>();
    let mut preimage = realized.clone();
    for transposition in transpositions {
        let left = usize::try_from(transposition.left)
            .ok()
            .filter(|index| *index < expected.len())
            .ok_or(BooleanLiftError::InvalidPermutationCertificate)?;
        let right = usize::try_from(transposition.right)
            .ok()
            .filter(|index| *index < expected.len() && *index != left)
            .ok_or(BooleanLiftError::InvalidPermutationCertificate)?;
        let left_source = preimage[left];
        let right_source = preimage[right];
        realized.swap(left_source, right_source);
        preimage.swap(left, right);
    }
    (realized == expected)
        .then_some(())
        .ok_or(BooleanLiftError::InvalidPermutationCertificate)
}

fn extend_variable_occurrence_count(
    current: usize,
    terms: &[Monomial],
) -> Result<usize, BooleanLiftError> {
    terms
        .iter()
        .try_fold(current, |count, term| count.checked_add(term.len()))
        .filter(|count| *count <= MAX_BOOLEAN_LIFT_VARIABLE_OCCURRENCES)
        .ok_or(BooleanLiftError::ProofWorkExceeded)
}

fn prove_acyclic_triangular_permutation(
    input_width: u64,
    outputs: &[BooleanLiftOutput],
) -> Option<Box<[u64]>> {
    let width = usize::try_from(input_width).ok()?;
    if outputs.len() != width || width == 0 {
        return None;
    }
    let mut adjacency = vec![BTreeSet::<usize>::new(); width];
    let mut indegrees = vec![0_usize; width];
    for (target, output) in outputs.iter().enumerate() {
        let mut has_self = false;
        for monomial in output.monomials() {
            let variables = monomial.variables();
            if variables == [u64::try_from(target).ok()?] {
                has_self = true;
                continue;
            }
            if variables.contains(&u64::try_from(target).ok()?) {
                return None;
            }
            for variable in variables {
                let dependency = usize::try_from(*variable).ok()?;
                if dependency >= width {
                    return None;
                }
                if adjacency[target].insert(dependency) {
                    indegrees[dependency] = indegrees[dependency].checked_add(1)?;
                }
            }
        }
        if !has_self {
            return None;
        }
    }
    let mut ready = indegrees
        .iter()
        .enumerate()
        .filter_map(|(index, indegree)| (*indegree == 0).then_some(index))
        .collect::<BTreeSet<_>>();
    let mut order = Vec::with_capacity(width);
    while let Some(target) = ready.pop_first() {
        order.push(u64::try_from(target).ok()?);
        for dependency in &adjacency[target] {
            indegrees[*dependency] = indegrees[*dependency].checked_sub(1)?;
            if indegrees[*dependency] == 0 {
                ready.insert(*dependency);
            }
        }
    }
    (order.len() == width).then(|| order.into_boxed_slice())
}

/// Replay and retain every referenced Boolean predicate before contract derivation.
pub(crate) fn verify_all_with_work(
    package: &wire::QCorePackageProto,
    arena: &TypeArena,
    operations: &LocallyCheckedOperations,
    work: &mut QCoreVerificationWork,
) -> Result<VerifiedBooleanLiftPlans, BooleanLiftError> {
    work.charge_phase_lifting(package.functions.len());
    let functions = package
        .functions
        .iter()
        .map(|function| (function.symbol.as_str(), function))
        .collect::<HashMap<_, _>>();
    let mut plans = VerifiedBooleanLiftPlans::default();
    let mut blocks = package
        .functions
        .iter()
        .filter_map(|function| function.body.as_ref())
        .collect::<Vec<_>>();
    while let Some(block) = blocks.pop() {
        work.charge_phase_lifting(1);
        for operation in &block.operations {
            work.charge_phase_lifting(1);
            if matches!(
                operation.opcode.as_str(),
                "qcore.phase_by" | "qcore.xor_by" | "qcore.permute_by"
            ) {
                work.charge_phase_lifting(1_usize.saturating_add(operation.attributes.len()));
                let predicate = string_attribute(operation, "predicate")
                    .ok_or_else(|| BooleanLiftError::Missing("Boolean predicate".to_owned()))?;
                if plans.get(predicate).is_none() {
                    let plan = plan_with_functions(&functions, predicate, arena, operations, work)?;
                    plans.insert(predicate.to_owned(), plan)?;
                }
                if operation.opcode == "qcore.permute_by"
                    && plans.get(predicate).is_some_and(|plan| {
                        plan.permutation_order().is_none() && plan.basis_transpositions().is_none()
                    })
                {
                    let transpositions = {
                        let plan = plans.get(predicate).ok_or_else(|| {
                            BooleanLiftError::Missing("Boolean permutation plan".to_owned())
                        })?;
                        prove_general_boolean_permutation(
                            predicate,
                            plan.input_width,
                            &plan.outputs,
                            work,
                        )?
                    };
                    plans.transposition_count = plans
                        .transposition_count
                        .checked_add(transpositions.len())
                        .filter(|count| {
                            count.saturating_add(plans.term_count) <= MAX_PHASE_LIFT_PLAN_TERMS
                        })
                        .ok_or(BooleanLiftError::ProgramTooManyTerms)?;
                    plans
                        .get_mut(predicate)
                        .ok_or_else(|| {
                            BooleanLiftError::Missing("Boolean permutation plan".to_owned())
                        })?
                        .basis_transpositions = Some(transpositions);
                }
            }
            for region in &operation.regions {
                work.charge_phase_lifting(1);
                blocks.extend(&region.blocks);
            }
        }
    }
    Ok(plans)
}

fn predicate_shape(
    function: &wire::FunctionProto,
    arena: &TypeArena,
    operations: &LocallyCheckedOperations,
) -> Result<(u64, u64), BooleanLiftError> {
    if function.external || function.parameters.len() != 1 || function.body.is_none() {
        return Err(BooleanLiftError::InvalidPredicate(function.symbol.clone()));
    }
    let types = operations
        .value_types(&function.symbol)
        .ok_or_else(|| BooleanLiftError::Missing("predicate type roots".to_owned()))?;
    let input = function.parameters[0]
        .value
        .as_ref()
        .and_then(|value| types.get(&value.id))
        .copied()
        .ok_or_else(|| BooleanLiftError::Missing("predicate input type".to_owned()))?;
    let result = operations
        .result_type(&function.symbol)
        .ok_or_else(|| BooleanLiftError::Missing("predicate result type".to_owned()))?;
    let width = match arena.get(input) {
        Some(SemanticType::Word {
            kind: WordKind::Bits,
            width: Dimension::Fixed(width),
        }) => *width,
        _ => return Err(BooleanLiftError::InvalidPredicate(function.symbol.clone())),
    };
    let output_width = boolean_output_width(arena, result)
        .ok_or_else(|| BooleanLiftError::InvalidPredicate(function.symbol.clone()))?;
    Ok((width, output_width))
}

/// Return the fixed coordinate count of one supported Boolean function result.
pub(crate) fn boolean_output_width(arena: &TypeArena, result: crate::TypeId) -> Option<u64> {
    match arena.get(result)? {
        SemanticType::Scalar(ScalarType::Bool) => Some(1),
        SemanticType::Product(elements)
            if !elements.is_empty()
                && elements.iter().all(|element| {
                    arena.get(*element) == Some(&SemanticType::Scalar(ScalarType::Bool))
                }) =>
        {
            u64::try_from(elements.len()).ok()
        }
        _ => None,
    }
}

impl Planner<'_> {
    fn charge(&mut self, count: usize) {
        self.proof_work = self.proof_work.saturating_add(count);
    }

    fn evaluate_function(
        &mut self,
        function: &wire::FunctionProto,
        arguments: Vec<PredicateValue>,
        depth: usize,
    ) -> Result<PredicateValue, BooleanLiftError> {
        Self::check_depth(depth)?;
        self.charge(1_usize.saturating_add(function.parameters.len()));
        if function.external || function.parameters.len() != arguments.len() {
            return Err(BooleanLiftError::Unsupported(format!(
                "call to {}",
                function.symbol
            )));
        }
        let mut environment = function
            .parameters
            .iter()
            .zip(arguments)
            .map(|(parameter, value)| {
                parameter
                    .value
                    .as_ref()
                    .map(|definition| (definition.id, value))
                    .ok_or_else(|| BooleanLiftError::Missing("predicate parameter".to_owned()))
            })
            .collect::<Result<HashMap<_, _>, _>>()?;
        let body = function
            .body
            .as_ref()
            .ok_or_else(|| BooleanLiftError::Unsupported("external predicate".to_owned()))?;
        self.evaluate_block(body, &mut environment, depth)
    }

    fn evaluate_block(
        &mut self,
        block: &wire::BlockProto,
        environment: &mut HashMap<u64, PredicateValue>,
        depth: usize,
    ) -> Result<PredicateValue, BooleanLiftError> {
        Self::check_depth(depth)?;
        self.charge(
            1_usize
                .saturating_add(block.operations.len())
                .saturating_add(
                    block
                        .terminator
                        .as_ref()
                        .map_or(0, |terminator| terminator.operands.len()),
                ),
        );
        for operation in &block.operations {
            let value = self.evaluate_operation(operation, environment, depth)?;
            let [result] = operation.results.as_slice() else {
                return Err(BooleanLiftError::Unsupported(operation.opcode.clone()));
            };
            environment.insert(result.id, value);
        }
        let terminator = block
            .terminator
            .as_ref()
            .ok_or_else(|| BooleanLiftError::Missing("predicate terminator".to_owned()))?;
        if !matches!(terminator.kind.as_str(), "return" | "yield") {
            return Err(BooleanLiftError::Unsupported(format!(
                "{} terminator",
                terminator.kind
            )));
        }
        let [result] = terminator.operands.as_slice() else {
            return Err(BooleanLiftError::Unsupported(
                "predicate result arity".to_owned(),
            ));
        };
        environment
            .get(result)
            .cloned()
            .ok_or_else(|| BooleanLiftError::Missing("predicate result".to_owned()))
    }

    fn evaluate_operation(
        &mut self,
        operation: &wire::OperationProto,
        environment: &HashMap<u64, PredicateValue>,
        depth: usize,
    ) -> Result<PredicateValue, BooleanLiftError> {
        self.charge(
            1_usize
                .saturating_add(operation.operands.len())
                .saturating_add(operation.results.len())
                .saturating_add(operation.attributes.len())
                .saturating_add(operation.regions.len()),
        );
        match operation.opcode.as_str() {
            "qcore.constant" => Self::constant(operation),
            "qcore.stage_lift" => Self::operand(operation, environment, 0),
            "qcore.tuple" | "tuple" => Self::tuple(operation, environment),
            "classical.index" => Self::index(operation, environment),
            "classical.cast" => match Self::operand(operation, environment, 0)? {
                PredicateValue::Bool(value) => Ok(PredicateValue::Bool(value)),
                PredicateValue::Bits { .. }
                | PredicateValue::Int(_)
                | PredicateValue::Product(_) => {
                    Err(BooleanLiftError::Unsupported("non-Boolean cast".to_owned()))
                }
            },
            "classical.not" => Ok(PredicateValue::Bool(
                Self::boolean_operand(operation, environment, 0)?.not()?,
            )),
            "classical.xor" => {
                self.binary_boolean(operation, environment, |left, right, _| left.xor(right))
            }
            "classical.and_bits" | "classical.and" => {
                self.fold_boolean(operation, environment, false)
            }
            "classical.or_bits" | "classical.or" => self.fold_boolean(operation, environment, true),
            "classical.compare_chain" => self.compare_chain(operation, environment),
            "classical.select" => self.select(operation, environment),
            "qcore.call" => self.call(operation, environment, depth + 1),
            "qcore.if" => self.conditional(operation, environment, depth + 1),
            opcode => Err(BooleanLiftError::Unsupported(opcode.to_owned())),
        }
    }

    fn constant(operation: &wire::OperationProto) -> Result<PredicateValue, BooleanLiftError> {
        use wire::literal_proto::Value;
        let literal = attribute(operation, "value")
            .and_then(|literal| literal.value.as_ref())
            .ok_or_else(|| BooleanLiftError::Missing("predicate constant".to_owned()))?;
        match literal {
            Value::BoolValue(value) => Ok(PredicateValue::Bool(BooleanAnf::constant(*value))),
            Value::IntValue(value) => Ok(PredicateValue::Int(BigInt::from(*value))),
            Value::BigintValue(value) => {
                Ok(PredicateValue::Int(BigInt::from_signed_bytes_be(value)))
            }
            _ => Err(BooleanLiftError::Unsupported(
                "non-Boolean predicate constant".to_owned(),
            )),
        }
    }

    fn index(
        operation: &wire::OperationProto,
        environment: &HashMap<u64, PredicateValue>,
    ) -> Result<PredicateValue, BooleanLiftError> {
        let [source, index] = operation.operands.as_slice() else {
            return Err(BooleanLiftError::Unsupported("dynamic index".to_owned()));
        };
        let Some(PredicateValue::Bits { width }) = environment.get(source) else {
            return Err(BooleanLiftError::Unsupported(
                "index of a derived value".to_owned(),
            ));
        };
        let Some(PredicateValue::Int(index)) = environment.get(index) else {
            return Err(BooleanLiftError::Unsupported("dynamic index".to_owned()));
        };
        let position = normalize_aggregate_index(*width, index)
            .ok_or_else(|| BooleanLiftError::Unsupported("out-of-range index".to_owned()))?;
        Ok(PredicateValue::Bool(BooleanAnf::variable(
            u64::try_from(position).map_err(|_| BooleanLiftError::InputTooWide)?,
        )))
    }

    fn tuple(
        operation: &wire::OperationProto,
        environment: &HashMap<u64, PredicateValue>,
    ) -> Result<PredicateValue, BooleanLiftError> {
        if operation.operands.is_empty() {
            return Err(BooleanLiftError::Unsupported(
                "empty Boolean product".to_owned(),
            ));
        }
        let values = operation
            .operands
            .iter()
            .map(|operand| {
                environment
                    .get(operand)
                    .cloned()
                    .ok_or_else(|| BooleanLiftError::Missing("product operand".to_owned()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if values
            .iter()
            .any(|value| !matches!(value, PredicateValue::Bool(_)))
        {
            return Err(BooleanLiftError::Unsupported(
                "non-Boolean product".to_owned(),
            ));
        }
        Ok(PredicateValue::Product(values.into_boxed_slice()))
    }

    fn operand(
        operation: &wire::OperationProto,
        environment: &HashMap<u64, PredicateValue>,
        index: usize,
    ) -> Result<PredicateValue, BooleanLiftError> {
        operation
            .operands
            .get(index)
            .and_then(|operand| environment.get(operand))
            .cloned()
            .ok_or_else(|| BooleanLiftError::Missing("predicate operand".to_owned()))
    }

    fn boolean_operand(
        operation: &wire::OperationProto,
        environment: &HashMap<u64, PredicateValue>,
        index: usize,
    ) -> Result<BooleanAnf, BooleanLiftError> {
        match Self::operand(operation, environment, index)? {
            PredicateValue::Bool(value) => Ok(value),
            PredicateValue::Bits { .. } | PredicateValue::Int(_) | PredicateValue::Product(_) => {
                Err(BooleanLiftError::Unsupported(
                    "non-Boolean operand".to_owned(),
                ))
            }
        }
    }

    fn binary_boolean(
        &mut self,
        operation: &wire::OperationProto,
        environment: &HashMap<u64, PredicateValue>,
        combine: impl FnOnce(
            BooleanAnf,
            &BooleanAnf,
            &mut usize,
        ) -> Result<BooleanAnf, BooleanLiftError>,
    ) -> Result<PredicateValue, BooleanLiftError> {
        let left = Self::boolean_operand(operation, environment, 0)?;
        let right = Self::boolean_operand(operation, environment, 1)?;
        combine(left, &right, &mut self.product_work).map(PredicateValue::Bool)
    }

    fn fold_boolean(
        &mut self,
        operation: &wire::OperationProto,
        environment: &HashMap<u64, PredicateValue>,
        disjunction: bool,
    ) -> Result<PredicateValue, BooleanLiftError> {
        let mut operands = operation.operands.iter();
        let first = operands
            .next()
            .and_then(|operand| environment.get(operand))
            .and_then(|value| match value {
                PredicateValue::Bool(value) => Some(value.clone()),
                PredicateValue::Bits { .. }
                | PredicateValue::Int(_)
                | PredicateValue::Product(_) => None,
            })
            .ok_or_else(|| BooleanLiftError::Unsupported(operation.opcode.clone()))?;
        let result = operands.try_fold(first, |current, operand| {
            let PredicateValue::Bool(next) = environment
                .get(operand)
                .ok_or_else(|| BooleanLiftError::Missing("predicate operand".to_owned()))?
            else {
                return Err(BooleanLiftError::Unsupported(operation.opcode.clone()));
            };
            if disjunction {
                current.or(next, &mut self.product_work)
            } else {
                current.and(next, &mut self.product_work)
            }
        })?;
        Ok(PredicateValue::Bool(result))
    }

    fn compare_chain(
        &mut self,
        operation: &wire::OperationProto,
        environment: &HashMap<u64, PredicateValue>,
    ) -> Result<PredicateValue, BooleanLiftError> {
        let operators = string_tuple_attribute(operation, "operators")
            .ok_or_else(|| BooleanLiftError::Unsupported("comparison".to_owned()))?;
        if operators.len() + 1 != operation.operands.len() {
            return Err(BooleanLiftError::Unsupported("comparison".to_owned()));
        }
        let mut result = BooleanAnf::constant(true);
        for (operator, pair) in operators.iter().zip(operation.operands.windows(2)) {
            let left = environment
                .get(&pair[0])
                .and_then(|value| match value {
                    PredicateValue::Bool(value) => Some(value.clone()),
                    PredicateValue::Bits { .. }
                    | PredicateValue::Int(_)
                    | PredicateValue::Product(_) => None,
                })
                .ok_or_else(|| BooleanLiftError::Unsupported("comparison".to_owned()))?;
            let right = environment
                .get(&pair[1])
                .and_then(|value| match value {
                    PredicateValue::Bool(value) => Some(value.clone()),
                    PredicateValue::Bits { .. }
                    | PredicateValue::Int(_)
                    | PredicateValue::Product(_) => None,
                })
                .ok_or_else(|| BooleanLiftError::Unsupported("comparison".to_owned()))?;
            let comparison = match *operator {
                "eq" => left.xor(&right)?.not()?,
                "ne" => left.xor(&right)?,
                _ => return Err(BooleanLiftError::Unsupported("comparison".to_owned())),
            };
            result = result.and(&comparison, &mut self.product_work)?;
        }
        Ok(PredicateValue::Bool(result))
    }

    fn select(
        &mut self,
        operation: &wire::OperationProto,
        environment: &HashMap<u64, PredicateValue>,
    ) -> Result<PredicateValue, BooleanLiftError> {
        let condition = Self::boolean_operand(operation, environment, 0)?;
        let when_true = Self::boolean_operand(operation, environment, 1)?;
        let when_false = Self::boolean_operand(operation, environment, 2)?;
        BooleanAnf::select(&condition, &when_true, &when_false, &mut self.product_work)
            .map(PredicateValue::Bool)
    }

    fn call(
        &mut self,
        operation: &wire::OperationProto,
        environment: &HashMap<u64, PredicateValue>,
        depth: usize,
    ) -> Result<PredicateValue, BooleanLiftError> {
        let callee_name = string_attribute(operation, "callee")
            .ok_or_else(|| BooleanLiftError::Missing("predicate callee".to_owned()))?;
        let callee = self
            .functions
            .get(callee_name)
            .copied()
            .ok_or_else(|| BooleanLiftError::Missing("predicate callee".to_owned()))?;
        self.charge(operation.operands.len());
        let arguments = operation
            .operands
            .iter()
            .map(|operand| {
                environment
                    .get(operand)
                    .cloned()
                    .ok_or_else(|| BooleanLiftError::Missing("call argument".to_owned()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.evaluate_function(callee, arguments, depth)
    }

    fn conditional(
        &mut self,
        operation: &wire::OperationProto,
        environment: &HashMap<u64, PredicateValue>,
        depth: usize,
    ) -> Result<PredicateValue, BooleanLiftError> {
        Self::check_depth(depth)?;
        let condition = Self::boolean_operand(operation, environment, 0)?;
        let captures = operation.operands.get(1..).unwrap_or_default();
        self.charge(operation.regions.len().saturating_add(captures.len()));
        let mut branches = Vec::with_capacity(2);
        for region in &operation.regions {
            let [block] = region.blocks.as_slice() else {
                return Err(BooleanLiftError::Unsupported(
                    "conditional region".to_owned(),
                ));
            };
            if block.arguments.len() != captures.len() {
                return Err(BooleanLiftError::Unsupported("conditional ABI".to_owned()));
            }
            let mut local = block
                .arguments
                .iter()
                .zip(captures)
                .map(|(argument, capture)| {
                    environment
                        .get(capture)
                        .cloned()
                        .map(|value| (argument.id, value))
                        .ok_or_else(|| BooleanLiftError::Missing("conditional capture".to_owned()))
                })
                .collect::<Result<HashMap<_, _>, _>>()?;
            branches.push(self.evaluate_block(block, &mut local, depth)?);
        }
        let [when_true, when_false] = branches.as_slice() else {
            return Err(BooleanLiftError::Unsupported(
                "conditional result".to_owned(),
            ));
        };
        Self::select_value(&condition, when_true, when_false, &mut self.product_work)
    }

    fn select_value(
        condition: &BooleanAnf,
        when_true: &PredicateValue,
        when_false: &PredicateValue,
        product_work: &mut usize,
    ) -> Result<PredicateValue, BooleanLiftError> {
        match (when_true, when_false) {
            (PredicateValue::Bool(when_true), PredicateValue::Bool(when_false)) => {
                BooleanAnf::select(condition, when_true, when_false, product_work)
                    .map(PredicateValue::Bool)
            }
            (PredicateValue::Product(when_true), PredicateValue::Product(when_false))
                if when_true.len() == when_false.len() && !when_true.is_empty() =>
            {
                when_true
                    .iter()
                    .zip(when_false)
                    .map(|(when_true, when_false)| {
                        Self::select_value(condition, when_true, when_false, product_work)
                    })
                    .collect::<Result<Vec<_>, _>>()
                    .map(|values| PredicateValue::Product(values.into_boxed_slice()))
            }
            _ => Err(BooleanLiftError::Unsupported(
                "conditional result".to_owned(),
            )),
        }
    }

    fn check_depth(depth: usize) -> Result<(), BooleanLiftError> {
        if depth > MAX_PHASE_LIFT_DEPTH {
            Err(BooleanLiftError::DepthExceeded)
        } else {
            Ok(())
        }
    }
}

fn attribute<'a>(
    operation: &'a wire::OperationProto,
    name: &str,
) -> Option<&'a wire::LiteralProto> {
    operation
        .attributes
        .iter()
        .find(|attribute| attribute.name == name)
        .and_then(|attribute| attribute.value.as_ref())
}

fn string_attribute<'a>(operation: &'a wire::OperationProto, name: &str) -> Option<&'a str> {
    use wire::literal_proto::Value;
    match attribute(operation, name)?.value.as_ref()? {
        Value::StringValue(value) => Some(value),
        _ => None,
    }
}

fn string_tuple_attribute<'a>(
    operation: &'a wire::OperationProto,
    name: &str,
) -> Option<Vec<&'a str>> {
    use wire::literal_proto::Value;
    let Value::TupleValue(sequence) = attribute(operation, name)?.value.as_ref()? else {
        return None;
    };
    sequence
        .values
        .iter()
        .map(|value| match value.value.as_ref()? {
            Value::StringValue(value) => Some(value.as_str()),
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds_aggregate_retained_phase_plan_terms_before_insertion() {
        let one_term = || BooleanLiftPlan {
            input_width: 1,
            outputs: vec![BooleanLiftOutput {
                monomials: vec![BooleanMonomial {
                    variables: vec![0].into_boxed_slice(),
                }]
                .into_boxed_slice(),
            }]
            .into_boxed_slice(),
            permutation_order: Some(vec![0].into_boxed_slice()),
            basis_transpositions: None,
        };
        let mut plans = VerifiedBooleanLiftPlans::default();

        plans
            .insert_with_limit("first".to_owned(), one_term(), 1)
            .expect("the exact aggregate term limit is accepted");
        assert_eq!(
            plans.insert_with_limit("second".to_owned(), one_term(), 1),
            Err(BooleanLiftError::ProgramTooManyTerms)
        );
        assert_eq!(plans.term_count, 1);
        assert!(plans.get("second").is_none());
    }

    #[test]
    fn bounds_retained_variable_occurrences_before_plan_allocation() {
        let one_variable = vec![vec![0]];

        assert_eq!(
            extend_variable_occurrence_count(
                MAX_BOOLEAN_LIFT_VARIABLE_OCCURRENCES - 1,
                &one_variable,
            ),
            Ok(MAX_BOOLEAN_LIFT_VARIABLE_OCCURRENCES)
        );
        assert_eq!(
            extend_variable_occurrence_count(MAX_BOOLEAN_LIFT_VARIABLE_OCCURRENCES, &one_variable,),
            Err(BooleanLiftError::ProofWorkExceeded)
        );
    }

    #[test]
    fn proves_only_acyclic_triangular_permutation_schedules() {
        let output = |terms: &[&[u64]]| BooleanLiftOutput {
            monomials: terms
                .iter()
                .map(|variables| BooleanMonomial {
                    variables: variables.to_vec().into_boxed_slice(),
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        };
        let triangular = [output(&[&[0], &[1]]), output(&[&[1]])];
        let cyclic = [output(&[&[1]]), output(&[&[0]])];
        let self_nonlinear = [output(&[&[0], &[0, 1]]), output(&[&[1]])];

        assert_eq!(
            prove_acyclic_triangular_permutation(2, &triangular).as_deref(),
            Some([0, 1].as_slice())
        );
        assert_eq!(prove_acyclic_triangular_permutation(2, &cyclic), None);
        assert_eq!(
            prove_acyclic_triangular_permutation(2, &self_nonlinear),
            None
        );
    }

    #[test]
    fn proves_and_replays_canonical_general_permutation_cycles() {
        let output = |terms: &[&[u64]]| BooleanLiftOutput {
            monomials: terms
                .iter()
                .map(|variables| BooleanMonomial {
                    variables: variables.to_vec().into_boxed_slice(),
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        };
        let cyclic = [output(&[&[], &[1]]), output(&[&[0]])];
        let mut work = QCoreVerificationWork::default();
        let transpositions = prove_general_boolean_permutation("cyclic", 2, &cyclic, &mut work)
            .expect("four-cycle permutation proof");

        assert_eq!(
            transpositions.as_ref(),
            [
                BooleanBasisTransposition { left: 0, right: 1 },
                BooleanBasisTransposition { left: 0, right: 3 },
                BooleanBasisTransposition { left: 0, right: 2 },
            ]
        );
        assert_eq!(transposition_primitive_count(0, 3, 2), Ok(7));
        assert_eq!(general_permutation_evaluation_work(&cyclic, 4, 28), Ok(28));
        assert_eq!(
            general_permutation_evaluation_work(&cyclic, 4, 27),
            Err(BooleanLiftError::GeneralPermutationWorkExceeded)
        );
        assert_eq!(extend_general_instruction_count(0, 0, 3, 2, 7), Ok(7));
        assert_eq!(
            extend_general_instruction_count(0, 0, 3, 2, 6),
            Err(BooleanLiftError::GeneralPermutationOutputExceeded)
        );
    }

    #[test]
    fn general_permutation_proof_rejects_collisions_and_width_overflow() {
        let output = |terms: &[&[u64]]| BooleanLiftOutput {
            monomials: terms
                .iter()
                .map(|variables| BooleanMonomial {
                    variables: variables.to_vec().into_boxed_slice(),
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        };
        let collision = [output(&[&[0]]), output(&[&[0]])];
        let mut work = QCoreVerificationWork::default();
        assert_eq!(
            prove_general_boolean_permutation("collision", 2, &collision, &mut work),
            Err(BooleanLiftError::InvalidPermutation("collision".to_owned()))
        );
        let identity = (0..=MAX_GENERAL_BOOLEAN_PERMUTATION_BITS)
            .map(|index| output(&[&[index]]))
            .collect::<Vec<_>>();
        assert_eq!(
            prove_general_boolean_permutation(
                "identity",
                MAX_GENERAL_BOOLEAN_PERMUTATION_BITS + 1,
                &identity,
                &mut work,
            ),
            Err(BooleanLiftError::GeneralPermutationTooWide)
        );
    }
}
