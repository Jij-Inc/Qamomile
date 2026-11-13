//! Closed semantic type universe shared by all native compiler passes.
//! Dependent extents retain bounded polynomial or Const-slice intent until
//! native specialization resolves their positive, nonempty concrete shape.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use prost::Message;

use crate::{verification::QCoreVerificationWork, wire};

/// A static or symbolic dependent shape dimension.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum Dimension {
    /// A positive, statically known extent.
    Fixed(u64),
    /// A nonempty function-scoped dependent dimension expression.
    Symbol {
        /// Function-local namespace that owns the binder.
        scope: Arc<str>,
        /// Source spelling within that namespace.
        name: Arc<str>,
    },
}

pub(crate) fn compare_dimensions(left: &Dimension, right: &Dimension) -> std::cmp::Ordering {
    match (left, right) {
        (Dimension::Fixed(left), Dimension::Fixed(right)) => left.cmp(right),
        (Dimension::Fixed(_), Dimension::Symbol { .. }) => std::cmp::Ordering::Less,
        (Dimension::Symbol { .. }, Dimension::Fixed(_)) => std::cmp::Ordering::Greater,
        (
            Dimension::Symbol {
                scope: left_scope,
                name: left_name,
            },
            Dimension::Symbol {
                scope: right_scope,
                name: right_name,
            },
        ) => (left_scope, left_name).cmp(&(right_scope, right_name)),
    }
}

const MAX_AFFINE_TERMS: usize = 64;
const MAX_AFFINE_MAGNITUDE: i64 = 1_i64 << 31;
const MAX_DIMENSION_FACTORS: usize = 64;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct AffineDimension {
    pub(crate) constant: i64,
    pub(crate) terms: BTreeMap<String, i64>,
}

impl AffineDimension {
    pub(crate) fn parse(source: &str) -> Result<Self, String> {
        if source.is_empty() || source.chars().any(char::is_whitespace) {
            return Err("a symbolic shape dimension must be nonempty and compact".to_owned());
        }
        let mut expression = Self {
            constant: 0,
            terms: BTreeMap::new(),
        };
        let mut start = 0;
        let mut sign = 1_i64;
        for (index, character) in source.char_indices() {
            if index == 0 && character == '-' {
                sign = -1;
                start = 1;
                continue;
            }
            if index > start && matches!(character, '+' | '-') {
                expression.add_term(&source[start..index], sign)?;
                sign = if character == '-' { -1 } else { 1 };
                start = index + character.len_utf8();
            }
        }
        if start >= source.len() {
            return Err("affine dimension ends with an operator".to_owned());
        }
        expression.add_term(&source[start..], sign)?;
        if expression.terms.is_empty() {
            return Err("a symbolic dimension must contain a symbol".to_owned());
        }
        if expression.terms.len() > MAX_AFFINE_TERMS
            || expression.constant.unsigned_abs() > MAX_AFFINE_MAGNITUDE.unsigned_abs()
            || expression
                .terms
                .values()
                .any(|coefficient| coefficient.unsigned_abs() > MAX_AFFINE_MAGNITUDE.unsigned_abs())
        {
            return Err("affine dimension exceeds its complexity budget".to_owned());
        }
        if expression.render() != source {
            return Err("affine dimension is not canonically encoded".to_owned());
        }
        Ok(expression)
    }

    fn add_term(&mut self, source: &str, sign: i64) -> Result<(), String> {
        if source.is_empty() {
            return Err("affine dimension contains an empty term".to_owned());
        }
        if source.bytes().all(|value| value.is_ascii_digit()) {
            let value = source
                .parse::<i64>()
                .map_err(|_| "affine constant exceeds i64".to_owned())?;
            let signed = value
                .checked_mul(sign)
                .ok_or_else(|| "affine constant exceeds i64".to_owned())?;
            self.constant = self
                .constant
                .checked_add(signed)
                .ok_or_else(|| "affine constant exceeds i64".to_owned())?;
            return Ok(());
        }
        let (coefficient, name) = if let Some((coefficient, name)) = source.split_once('*') {
            if name.contains('*') || coefficient.is_empty() {
                return Err("affine term must be coefficient*symbol".to_owned());
            }
            let coefficient = coefficient
                .parse::<i64>()
                .map_err(|_| "affine coefficient must be an integer".to_owned())?;
            (coefficient, name)
        } else {
            (1, source)
        };
        if coefficient == 0 || !valid_dimension_name(name) {
            return Err("affine term has an invalid coefficient or symbol".to_owned());
        }
        let signed = coefficient
            .checked_mul(sign)
            .ok_or_else(|| "affine coefficient exceeds i64".to_owned())?;
        let entry = self.terms.entry(name.to_owned()).or_default();
        *entry = entry
            .checked_add(signed)
            .ok_or_else(|| "affine coefficient exceeds i64".to_owned())?;
        if *entry == 0 {
            self.terms.remove(name);
        }
        Ok(())
    }

    pub(crate) fn render(&self) -> String {
        let mut result = String::new();
        for (name, coefficient) in &self.terms {
            push_affine_piece(&mut result, *coefficient, name, false);
        }
        if self.constant != 0 {
            push_affine_piece(
                &mut result,
                self.constant,
                &self.constant.unsigned_abs().to_string(),
                true,
            );
        }
        result
    }
}

/// Canonical bounded polynomial used by dependent shape dimensions.
///
/// The public wire representation remains one compact string.  This parser is
/// the single authority that decides whether that string denotes a supported
/// dimension expression; callers never interpret the spelling themselves.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PolynomialDimension {
    pub(crate) constant: i64,
    pub(crate) terms: BTreeMap<Vec<String>, i64>,
}

impl PolynomialDimension {
    pub(crate) fn parse(source: &str) -> Result<Self, String> {
        if source.is_empty() || source.chars().any(char::is_whitespace) {
            return Err("a symbolic shape dimension must be nonempty and compact".to_owned());
        }
        let mut expression = Self {
            constant: 0,
            terms: BTreeMap::new(),
        };
        let mut start = 0;
        let mut sign = 1_i64;
        let mut depth = 0_usize;
        for (index, character) in source.char_indices() {
            match character {
                '(' => depth += 1,
                ')' => {
                    depth = depth
                        .checked_sub(1)
                        .ok_or_else(|| "unbalanced dimension parentheses".to_owned())?;
                }
                _ => {}
            }
            if depth > 0 || character == ')' {
                continue;
            }
            if index == 0 && character == '-' {
                sign = -1;
                start = 1;
                continue;
            }
            if index > start && matches!(character, '+' | '-') {
                expression.add_term(&source[start..index], sign)?;
                sign = if character == '-' { -1 } else { 1 };
                start = index + character.len_utf8();
            }
        }
        if depth != 0 {
            return Err("unbalanced dimension parentheses".to_owned());
        }
        if start >= source.len() {
            return Err("dimension expression ends with an operator".to_owned());
        }
        expression.add_term(&source[start..], sign)?;
        if expression.terms.is_empty() {
            return Err("a symbolic dimension must contain a symbol".to_owned());
        }
        let factors = expression.terms.keys().map(Vec::len).sum::<usize>();
        if expression.terms.len() > MAX_AFFINE_TERMS
            || factors > MAX_DIMENSION_FACTORS
            || expression.constant.unsigned_abs() > MAX_AFFINE_MAGNITUDE.unsigned_abs()
            || expression
                .terms
                .values()
                .any(|coefficient| coefficient.unsigned_abs() > MAX_AFFINE_MAGNITUDE.unsigned_abs())
        {
            return Err("dimension expression exceeds its complexity budget".to_owned());
        }
        if expression.render() != source {
            return Err("dimension expression is not canonically encoded".to_owned());
        }
        Ok(expression)
    }

    fn add_term(&mut self, source: &str, sign: i64) -> Result<(), String> {
        if source.is_empty() {
            return Err("dimension expression contains an empty term".to_owned());
        }
        if source.bytes().all(|value| value.is_ascii_digit()) {
            let value = source
                .parse::<i64>()
                .map_err(|_| "dimension constant exceeds i64".to_owned())?;
            self.constant = self
                .constant
                .checked_add(
                    value
                        .checked_mul(sign)
                        .ok_or_else(|| "dimension constant exceeds i64".to_owned())?,
                )
                .ok_or_else(|| "dimension constant exceeds i64".to_owned())?;
            return Ok(());
        }
        let mut pieces = split_top_level_factors(source);
        let coefficient = pieces
            .first()
            .and_then(|piece| piece.parse::<i64>().ok())
            .map_or(1, |value| {
                pieces.remove(0);
                value
            });
        if coefficient == 0
            || pieces.is_empty()
            || pieces.iter().any(|piece| !valid_dimension_factor(piece))
            || !pieces.windows(2).all(|pair| pair[0] <= pair[1])
        {
            return Err("dimension term must be coefficient times ordered symbols".to_owned());
        }
        let monomial = pieces.into_iter().map(str::to_owned).collect::<Vec<_>>();
        let signed = coefficient
            .checked_mul(sign)
            .ok_or_else(|| "dimension coefficient exceeds i64".to_owned())?;
        let entry = self.terms.entry(monomial.clone()).or_default();
        *entry = entry
            .checked_add(signed)
            .ok_or_else(|| "dimension coefficient exceeds i64".to_owned())?;
        if *entry == 0 {
            self.terms.remove(&monomial);
        }
        Ok(())
    }

    pub(crate) fn render(&self) -> String {
        let mut result = String::new();
        for (symbols, coefficient) in &self.terms {
            let monomial = symbols.join("*");
            push_affine_piece(&mut result, *coefficient, &monomial, false);
        }
        if self.constant != 0 {
            push_affine_piece(
                &mut result,
                self.constant,
                &self.constant.unsigned_abs().to_string(),
                true,
            );
        }
        result
    }

    /// Read a fixed or canonical symbolic dimension as a polynomial.
    ///
    /// Slice shapes and noncanonical spellings have no polynomial reading.
    pub(crate) fn of_dimension(dimension: &Dimension) -> Option<Self> {
        match dimension {
            Dimension::Fixed(value) => Some(Self {
                constant: i64::try_from(*value).ok()?,
                terms: BTreeMap::new(),
            }),
            Dimension::Symbol { name, .. } if name.starts_with("slice(") => None,
            Dimension::Symbol { name, .. } => Self::parse(name).ok(),
        }
    }

    /// Return `self + scale * other` with checked coefficients.
    pub(crate) fn add_scaled(&self, other: &Self, scale: i64) -> Option<Self> {
        let mut result = self.clone();
        result.constant = result
            .constant
            .checked_add(other.constant.checked_mul(scale)?)?;
        for (monomial, coefficient) in &other.terms {
            let entry = result.terms.entry(monomial.clone()).or_default();
            *entry = entry.checked_add(coefficient.checked_mul(scale)?)?;
            if *entry == 0 {
                result.terms.remove(monomial);
            }
        }
        Some(result)
    }

    /// Every dependent binder the polynomial mentions, including those inside
    /// floor-quotient factors, in sorted order.
    pub(crate) fn symbols(&self) -> Vec<String> {
        let mut symbols = std::collections::BTreeSet::new();
        for monomial in self.terms.keys() {
            for factor in monomial {
                if let Some((numerator, _)) = parse_floor_quotient(factor) {
                    symbols.extend(numerator.symbols());
                } else {
                    symbols.insert(factor.clone());
                }
            }
        }
        symbols.into_iter().collect()
    }

    /// Evaluate with signed intermediate values, resolving binders by name.
    fn evaluate_with(&self, resolve: &mut dyn FnMut(&str) -> Option<i128>) -> Option<i128> {
        let mut total = i128::from(self.constant);
        for (monomial, coefficient) in &self.terms {
            let mut product = 1_i128;
            for factor in monomial {
                let value = if let Some((numerator, divisor)) = parse_floor_quotient(factor) {
                    numerator
                        .evaluate_with(resolve)?
                        .div_euclid(i128::from(divisor))
                } else {
                    resolve(factor)?
                };
                product = product.checked_mul(value)?;
            }
            total = total.checked_add(i128::from(*coefficient).checked_mul(product)?)?;
        }
        Some(total)
    }

    pub(crate) fn evaluate(&self, mut resolve: impl FnMut(&str) -> Option<u64>) -> Option<u64> {
        let total = self.evaluate_with(&mut |name| resolve(name).map(i128::from))?;
        u64::try_from(total).ok().filter(|value| *value > 0)
    }
    pub(crate) fn evaluate_signed(&self, resolve: &impl Fn(&str) -> Option<i64>) -> Option<i128> {
        self.evaluate_with(&mut |name| resolve(name).map(i128::from))
    }
}

/// Split one monomial at top-level `*`, keeping parenthesized factors whole.
fn split_top_level_factors(source: &str) -> Vec<&str> {
    let mut pieces = Vec::new();
    let mut depth = 0_usize;
    let mut start = 0;
    for (index, character) in source.char_indices() {
        match character {
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            '*' if depth == 0 => {
                pieces.push(&source[start..index]);
                start = index + 1;
            }
            _ => {}
        }
    }
    pieces.push(&source[start..]);
    pieces
}

/// Parse a canonical floor-quotient factor `(name//d)` or `((poly)//d)`.
///
/// The numerator is a dimension polynomial and `d >= 2`. Inside a monomial the
/// factor is an opaque atom, so `n - (n//2) + (n//2)` cancels to `n`; only
/// evaluation computes the quotient.
pub(crate) fn parse_floor_quotient(factor: &str) -> Option<(PolynomialDimension, i64)> {
    let inner = factor.strip_prefix('(')?.strip_suffix(')')?;
    let (numerator_text, divisor_text) = inner.rsplit_once("//")?;
    if divisor_text.is_empty() || !divisor_text.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let divisor = divisor_text
        .parse::<i64>()
        .ok()
        .filter(|value| *value >= 2)?;
    let numerator = if let Some(nested) = numerator_text
        .strip_prefix('(')
        .and_then(|text| text.strip_suffix(')'))
    {
        PolynomialDimension::parse(nested).ok()?
    } else if valid_dimension_name(numerator_text) {
        PolynomialDimension::parse(numerator_text).ok()?
    } else {
        return None;
    };
    (render_floor_quotient(&numerator, divisor) == factor).then_some((numerator, divisor))
}

/// Canonical spelling of the floor-quotient factor `numerator // divisor`.
pub(crate) fn render_floor_quotient(numerator: &PolynomialDimension, divisor: i64) -> String {
    let rendered = numerator.render();
    if valid_dimension_name(&rendered) {
        format!("({rendered}//{divisor})")
    } else {
        format!("(({rendered})//{divisor})")
    }
}

fn valid_dimension_factor(factor: &str) -> bool {
    valid_dimension_name(factor) || parse_floor_quotient(factor).is_some()
}

fn valid_dimension_name(name: &str) -> bool {
    let mut characters = name.chars();
    characters
        .next()
        .is_some_and(|character| character == '_' || character.is_alphabetic())
        && characters.all(|character| character == '_' || character.is_alphanumeric())
}

fn push_affine_piece(result: &mut String, coefficient: i64, value: &str, literal: bool) {
    if result.is_empty() {
        if coefficient < 0 {
            result.push('-');
        }
    } else {
        result.push(if coefficient < 0 { '-' } else { '+' });
    }
    if !literal && coefficient.unsigned_abs() != 1 {
        result.push_str(&coefficient.unsigned_abs().to_string());
        result.push('*');
    }
    result.push_str(value);
}

pub(crate) fn dimension_symbols(dimension: &Dimension) -> Vec<Dimension> {
    match dimension {
        Dimension::Fixed(_) => Vec::new(),
        Dimension::Symbol { scope, name } if name.starts_with("slice(") => {
            crate::core::slice_shape::SliceShape::parse(name)
                .map(|shape| {
                    shape
                        .symbols()
                        .into_iter()
                        .map(|name| Dimension::Symbol {
                            scope: Arc::clone(scope),
                            name: Arc::from(name),
                        })
                        .collect()
                })
                .unwrap_or_default()
        }
        Dimension::Symbol { scope, name } => PolynomialDimension::parse(name)
            .map(|expression| {
                expression
                    .symbols()
                    .into_iter()
                    .map(|name| Dimension::Symbol {
                        scope: Arc::clone(scope),
                        name: Arc::from(name),
                    })
                    .collect()
            })
            .unwrap_or_default(),
    }
}

fn substitute_dimension(
    dimension: Dimension,
    bindings: &HashMap<Dimension, Dimension>,
) -> Dimension {
    let Dimension::Symbol { scope, name } = &dimension else {
        return dimension;
    };
    if let Some(replacement) = bindings.get(&dimension) {
        return replacement.clone();
    }
    if name.starts_with("slice(") {
        return crate::core::slice_shape::SliceShape::parse(name)
            .map(|shape| shape.substitute(scope, bindings))
            .unwrap_or(dimension);
    }
    let Ok(expression) = PolynomialDimension::parse(name) else {
        return dimension;
    };
    expression
        .evaluate(|name| {
            let symbol = Dimension::Symbol {
                scope: Arc::clone(scope),
                name: Arc::from(name),
            };
            match bindings.get(&symbol) {
                Some(Dimension::Fixed(bound)) => Some(*bound),
                _ => None,
            }
        })
        .map(Dimension::Fixed)
        .unwrap_or(dimension)
}

pub(crate) fn instantiate_dimension(
    dimension: &Dimension,
    bindings: &HashMap<Dimension, Dimension>,
) -> Dimension {
    substitute_dimension(dimension.clone(), bindings)
}

/// One scalar in the closed Qamomile semantic universe.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ScalarType {
    /// The unique zero-width value.
    Unit,
    /// One logical binary digit.
    Bit,
    /// A classical truth value.
    Bool,
    /// An arbitrary-precision signed integer.
    Int,
    /// A real scalar.
    Real,
    /// A complex scalar.
    Complex,
    /// A real value constrained to the unit interval.
    Probability,
    /// A portable observable value.
    Observable,
}

impl ScalarType {
    /// Return the canonical source and wire spelling of this scalar type.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unit => "Unit",
            Self::Bit => "Bit",
            Self::Bool => "Bool",
            Self::Int => "Int",
            Self::Real => "Real",
            Self::Complex => "Complex",
            Self::Probability => "Probability",
            Self::Observable => "Observable",
        }
    }
}

/// Interpretation attached to a fixed-width bit word.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum WordKind {
    /// An uninterpreted source-ordered bit string.
    Bits,
    /// An unsigned source-ordered integer.
    UInt,
}

/// Availability stage for classical values.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Stage {
    /// Available while specializing a program family.
    Const,
    /// Available when launching a compiled artifact.
    Param,
    /// Available only during one execution shot.
    Shot,
}

impl Stage {
    /// Return the canonical source and wire spelling of this stage.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Const => "Const",
            Self::Param => "Param",
            Self::Shot => "Shot",
        }
    }

    /// Parse one canonical source or wire spelling.
    #[must_use]
    pub(crate) fn from_name(value: &str) -> Option<Self> {
        match value {
            "Const" => Some(Self::Const),
            "Param" => Some(Self::Param),
            "Shot" => Some(Self::Shot),
            _ => None,
        }
    }

    /// Join two optional availability stages in the source lattice.
    #[must_use]
    pub(crate) fn join(left: Option<Self>, right: Option<Self>) -> Option<Self> {
        left.max(right)
    }
}

/// Canonical semantic type reconstructed from persistent `QCore`.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum SemanticType {
    /// A closed scalar type.
    Scalar(ScalarType),
    /// A statically shaped homogeneous classical collection.
    Array {
        /// Interned canonical non-array leaf type.
        element: TypeId,
        /// One or more dimensions in row-major order.
        shape: Vec<Dimension>,
    },
    /// A fixed-width classical bit word.
    Word {
        /// Logical interpretation of the bits.
        kind: WordKind,
        /// Static or dependent width.
        width: Dimension,
    },
    /// An unsigned binary fixed-point scalar.
    Fixed {
        /// Digits to the left of the radix point.
        integer_bits: Dimension,
        /// Digits to the right of the radix point.
        fractional_bits: Dimension,
    },
    /// A positional product.
    Product(Box<[TypeId]>),
    /// A named structural product.
    Record {
        /// Stable record identity.
        name: String,
        /// Unique fields in semantic order.
        fields: Box<[(String, TypeId)]>,
    },
    /// The coherent modality over a finite bit layout.
    Quantum(TypeId),
    /// Availability annotation over unstaged classical data.
    Staged {
        /// Earliest lifecycle point at which the value exists.
        stage: Stage,
        /// Unstaged classical payload.
        inner: TypeId,
    },
    /// An immutable closed function value.
    Function {
        /// Complete source parameter labels, or an empty positional-only row.
        parameter_names: Box<[String]>,
        /// Ordered logical input types.
        inputs: Box<[TypeId]>,
        /// Logical output type.
        output: TypeId,
    },
}

/// One ordered logical-width leaf inside an admissible coherent type.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CoherentLeafWidth<'a> {
    /// One coherent binary digit.
    Bit,
    /// One fixed or dependent word width.
    Word(&'a Dimension),
    /// One statically partitioned unsigned fixed-point width.
    Fixed {
        /// Digits to the left of the radix point.
        integer_bits: &'a Dimension,
        /// Digits to the right of the radix point.
        fractional_bits: &'a Dimension,
    },
}

/// Stable index of one interned semantic type.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TypeId(u32);

/// Per-program hash-consing arena for semantic types.
#[derive(Clone, Debug, Default)]
pub struct TypeArena {
    entries: Vec<Arc<SemanticType>>,
    indices: HashMap<Arc<SemanticType>, TypeId>,
    strings: HashSet<Arc<str>>,
    quantum_containment: Vec<bool>,
    aggregate_summaries: Vec<Option<AggregateSummary>>,
    sequence_elements: HashMap<TypeId, TypeId>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ElementFact {
    base: TypeId,
    stage: Option<Stage>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AggregateSummary {
    len: usize,
    uniform: Option<ElementFact>,
}

impl TypeArena {
    /// Return the number of structurally unique types in this program.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return whether no semantic type has been interned.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Resolve one stable type identifier.
    #[must_use]
    pub fn get(&self, id: TypeId) -> Option<&SemanticType> {
        self.entries.get(id.0 as usize).map(AsRef::as_ref)
    }

    /// Return whether an interned type contains owned coherent data.
    #[must_use]
    pub fn is_quantum(&self, id: TypeId) -> bool {
        self.quantum_containment
            .get(id.0 as usize)
            .copied()
            .unwrap_or(false)
    }

    /// Remove one outer availability stage from an interned type.
    #[must_use]
    pub(crate) fn strip_stage(&self, id: TypeId) -> TypeId {
        match self.get(id) {
            Some(SemanticType::Staged { inner, .. }) => *inner,
            _ => id,
        }
    }

    /// Return the outer availability stage of an interned type.
    #[must_use]
    pub(crate) fn stage_of(&self, id: TypeId) -> Option<Stage> {
        match self.get(id) {
            Some(SemanticType::Staged { stage, .. }) => Some(*stage),
            _ => None,
        }
    }

    /// Return the latest availability occurring anywhere in one type tree.
    ///
    /// Aggregate parameters may retain staged fields even when the aggregate
    /// itself has no outer stage. Interprocedural control and artifact-ABI
    /// requirements must therefore inspect the complete semantic type rather
    /// than only its outer wrapper.
    #[must_use]
    pub(crate) fn latest_availability(&self, id: TypeId) -> Option<Stage> {
        match self.get(id) {
            Some(SemanticType::Staged { stage, inner }) => {
                Stage::join(Some(*stage), self.latest_availability(*inner))
            }
            Some(SemanticType::Array { element, .. } | SemanticType::Quantum(element)) => {
                self.latest_availability(*element)
            }
            Some(SemanticType::Product(elements)) => {
                elements.iter().fold(None, |stage, element| {
                    Stage::join(stage, self.latest_availability(*element))
                })
            }
            Some(SemanticType::Record { fields, .. }) => {
                fields.iter().fold(None, |stage, (_, field_type)| {
                    Stage::join(stage, self.latest_availability(*field_type))
                })
            }
            Some(
                SemanticType::Scalar(_)
                | SemanticType::Word { .. }
                | SemanticType::Fixed { .. }
                | SemanticType::Function { .. },
            )
            | None => None,
        }
    }

    /// Return the fixed width of one logical bit-like leaf.
    ///
    /// One outer availability stage and one coherent modality are transparent.
    /// Products remain structural and therefore do not collapse to a width.
    ///
    /// # Errors
    ///
    /// Returns an error when a fixed-point width overflows `u64`.
    pub(crate) fn logical_width(&self, id: TypeId) -> Result<Option<u64>, String> {
        let mut base = self.strip_stage(id);
        if let Some(SemanticType::Quantum(inner)) = self.get(base) {
            base = *inner;
        }
        match self.get(base) {
            Some(SemanticType::Scalar(ScalarType::Bit | ScalarType::Bool)) => Ok(Some(1)),
            Some(SemanticType::Word {
                width: Dimension::Fixed(width),
                ..
            }) => Ok(Some(*width)),
            Some(SemanticType::Fixed {
                integer_bits,
                fractional_bits,
            }) => match (integer_bits, fractional_bits) {
                (Dimension::Fixed(integer), Dimension::Fixed(fractional)) => integer
                    .checked_add(*fractional)
                    .map(Some)
                    .ok_or_else(|| "fixed-point logical width exceeds u64".to_owned()),
                _ => Ok(None),
            },
            _ => Ok(None),
        }
    }

    /// Return the ordered fixed widths of coherent leaves in one type.
    ///
    /// # Errors
    ///
    /// Returns an error when a fixed-point leaf width overflows `u64`.
    pub(crate) fn coherent_leaf_widths(&self, root: TypeId) -> Result<Option<Box<[u64]>>, String> {
        let mut widths = Vec::new();
        self.visit_coherent_value_widths(root, &mut fixed_dimension, &mut |width| {
            widths.push(width);
            Ok(())
        })
        .map(|complete| complete.then(|| widths.into_boxed_slice()))
    }

    /// Return the total fixed width of one coherent value or owner product.
    ///
    /// # Errors
    ///
    /// Returns an error when the sum of otherwise fixed leaf widths overflows
    /// `u64`.
    pub(crate) fn coherent_width(&self, root: TypeId) -> Result<Option<u64>, String> {
        self.coherent_width_with(root, fixed_dimension)
    }

    /// Resolve and total one coherent width with a caller-provided dimension map.
    ///
    /// # Errors
    ///
    /// Returns an error when the complete resolved width overflows `u64`.
    pub(crate) fn coherent_width_with(
        &self,
        root: TypeId,
        mut resolve: impl FnMut(&Dimension) -> Option<u64>,
    ) -> Result<Option<u64>, String> {
        let mut total = 0_u64;
        self.visit_coherent_value_widths(root, &mut resolve, &mut |width| {
            total = total
                .checked_add(width)
                .ok_or_else(|| "coherent width exceeds u64".to_owned())?;
            Ok(())
        })
        .map(|complete| complete.then_some(total))
    }

    /// Return the total fixed width of one admissible coherent payload.
    ///
    /// # Errors
    ///
    /// Returns an error when a fixed-point leaf or the complete payload width
    /// overflows `u64`.
    pub(crate) fn coherent_payload_width(&self, root: TypeId) -> Result<Option<u64>, String> {
        let mut total = 0_u64;
        self.visit_coherent_payload_widths(root, &mut fixed_dimension, &mut |width| {
            total = total
                .checked_add(width)
                .ok_or_else(|| "coherent width exceeds u64".to_owned())?;
            Ok(())
        })
        .map(|complete| complete.then_some(total))
    }

    /// Return the fixed width of one pure sampled result tree.
    ///
    /// `Shot` is transparent, while other availability stages make the tree a
    /// value result. Products and records are sampled only when every child has
    /// a fixed sampled representation.
    ///
    /// # Errors
    ///
    /// Returns an error when the complete sampled width overflows `u64`.
    pub(crate) fn sampled_width(&self, root: TypeId) -> Result<Option<u64>, String> {
        self.sampled_width_at(root, false)
    }

    fn sampled_width_at(&self, root: TypeId, shot: bool) -> Result<Option<u64>, String> {
        if let Some(SemanticType::Staged { stage, inner }) = self.get(root) {
            return if *stage == Stage::Shot {
                self.sampled_width_at(*inner, true)
            } else {
                Ok(None)
            };
        }
        if let Some(width) = self.logical_width(root)? {
            return Ok(Some(width));
        }
        match self.get(root) {
            Some(SemanticType::Product(elements)) => {
                self.sum_sampled_widths(elements.iter().copied(), shot)
            }
            Some(SemanticType::Record { fields, .. }) => {
                self.sum_sampled_widths(fields.iter().map(|(_, field_type)| *field_type), shot)
            }
            Some(SemanticType::Scalar(ScalarType::Real)) if shot => Ok(Some(64)),
            _ => Ok(None),
        }
    }

    /// Return whether one logical type may inhabit the coherent modality.
    #[must_use]
    pub(crate) fn is_coherent_payload(&self, root: TypeId) -> bool {
        let mut pending = vec![root];
        let mut visited = HashSet::new();
        while let Some(type_id) = pending.pop() {
            if !visited.insert(type_id) {
                continue;
            }
            match self.get(type_id) {
                Some(
                    SemanticType::Scalar(ScalarType::Bit | ScalarType::Bool)
                    | SemanticType::Word { .. }
                    | SemanticType::Fixed { .. },
                ) => {}
                Some(SemanticType::Product(elements)) if !elements.is_empty() => {
                    pending.extend(elements.iter().copied());
                }
                Some(
                    SemanticType::Scalar(_)
                    | SemanticType::Array { .. }
                    | SemanticType::Product(_)
                    | SemanticType::Record { .. }
                    | SemanticType::Quantum(_)
                    | SemanticType::Staged { .. }
                    | SemanticType::Function { .. },
                )
                | None => return false,
            }
        }
        true
    }

    /// Return whether one classical type may receive an availability stage.
    #[must_use]
    pub(crate) fn is_stageable(&self, root: TypeId) -> bool {
        !self.is_quantum(root) && !matches!(self.get(root), Some(SemanticType::Staged { .. }))
    }

    fn visit_coherent_value_widths(
        &self,
        root: TypeId,
        resolve: &mut impl FnMut(&Dimension) -> Option<u64>,
        visit: &mut impl FnMut(u64) -> Result<(), String>,
    ) -> Result<bool, String> {
        self.visit_coherent_value_leaves(root, &mut |leaf| {
            let Some(width) = resolve_coherent_leaf_width(leaf, resolve)? else {
                return Ok(false);
            };
            visit(width)?;
            Ok(true)
        })
    }

    /// Visit ordered logical-width leaves in one coherent value or owner product.
    /// Named records of owners follow the same declared-order traversal as products.
    ///
    /// The callback returns `false` to stop at a leaf whose width cannot be
    /// represented in the caller's target algebra.
    pub(crate) fn visit_coherent_value_leaves<E>(
        &self,
        root: TypeId,
        visit: &mut impl FnMut(CoherentLeafWidth<'_>) -> Result<bool, E>,
    ) -> Result<bool, E> {
        let base = self.strip_stage(root);
        match self.get(base) {
            Some(SemanticType::Quantum(inner)) => self.visit_coherent_payload_leaves(*inner, visit),
            Some(SemanticType::Product(elements)) if !elements.is_empty() => {
                for element in elements {
                    if !self.visit_coherent_value_leaves(*element, visit)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            Some(SemanticType::Record { fields, .. }) if !fields.is_empty() => {
                for (_, field) in fields {
                    if !self.visit_coherent_value_leaves(*field, visit)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    fn visit_coherent_payload_widths(
        &self,
        root: TypeId,
        resolve: &mut impl FnMut(&Dimension) -> Option<u64>,
        visit: &mut impl FnMut(u64) -> Result<(), String>,
    ) -> Result<bool, String> {
        self.visit_coherent_payload_leaves(root, &mut |leaf| {
            let Some(width) = resolve_coherent_leaf_width(leaf, resolve)? else {
                return Ok(false);
            };
            visit(width)?;
            Ok(true)
        })
    }

    fn visit_coherent_payload_leaves<E>(
        &self,
        root: TypeId,
        visit: &mut impl FnMut(CoherentLeafWidth<'_>) -> Result<bool, E>,
    ) -> Result<bool, E> {
        match self.get(root) {
            Some(SemanticType::Scalar(ScalarType::Bit | ScalarType::Bool)) => {
                visit(CoherentLeafWidth::Bit)
            }
            Some(SemanticType::Word { width, .. }) => visit(CoherentLeafWidth::Word(width)),
            Some(SemanticType::Fixed {
                integer_bits,
                fractional_bits,
            }) => visit(CoherentLeafWidth::Fixed {
                integer_bits,
                fractional_bits,
            }),
            Some(SemanticType::Product(elements)) if !elements.is_empty() => {
                for element in elements {
                    if !self.visit_coherent_payload_leaves(*element, visit)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    fn sum_sampled_widths(
        &self,
        mut children: impl Iterator<Item = TypeId>,
        shot: bool,
    ) -> Result<Option<u64>, String> {
        children.try_fold(Some(0_u64), |total, child| {
            let Some(total) = total else {
                return Ok(None);
            };
            let Some(width) = self.sampled_width_at(child, shot)? else {
                return Ok(None);
            };
            total
                .checked_add(width)
                .map(Some)
                .ok_or_else(|| "sampled result width exceeds u64".to_owned())
        })
    }

    /// Return whether one availability stage occurs anywhere in a type tree.
    #[must_use]
    pub(crate) fn contains_stage(&self, id: TypeId, expected: Stage) -> bool {
        match self.get(id) {
            Some(SemanticType::Staged { stage, inner }) => {
                *stage == expected || self.contains_stage(*inner, expected)
            }
            Some(SemanticType::Array { element, .. } | SemanticType::Quantum(element)) => {
                self.contains_stage(*element, expected)
            }
            Some(SemanticType::Product(elements)) => elements
                .iter()
                .any(|element| self.contains_stage(*element, expected)),
            Some(SemanticType::Record { fields, .. }) => fields
                .iter()
                .any(|(_, field_type)| self.contains_stage(*field_type, expected)),
            Some(
                SemanticType::Scalar(_)
                | SemanticType::Word { .. }
                | SemanticType::Fixed { .. }
                | SemanticType::Function { .. },
            )
            | None => false,
        }
    }

    /// Return whether a function-value node occurs anywhere in one type DAG.
    #[must_use]
    pub(crate) fn contains_function(&self, root: TypeId) -> bool {
        let mut pending = vec![root];
        let mut visited = HashSet::new();
        while let Some(type_id) = pending.pop() {
            if !visited.insert(type_id) {
                continue;
            }
            match self.get(type_id) {
                Some(SemanticType::Function { .. }) => return true,
                Some(SemanticType::Array { element, .. } | SemanticType::Quantum(element)) => {
                    pending.push(*element);
                }
                Some(SemanticType::Product(elements)) => {
                    pending.extend(elements.iter().copied());
                }
                Some(SemanticType::Record { fields, .. }) => {
                    pending.extend(fields.iter().map(|(_, field_type)| *field_type));
                }
                Some(SemanticType::Staged { inner, .. }) => pending.push(*inner),
                Some(
                    SemanticType::Scalar(_)
                    | SemanticType::Word { .. }
                    | SemanticType::Fixed { .. },
                )
                | None => {}
            }
        }
        false
    }

    /// Return every scoped dependent dimension occurring in one type DAG.
    #[must_use]
    pub(crate) fn symbolic_dimensions(&self, root: TypeId) -> HashSet<Dimension> {
        self.symbolic_dimensions_in([root])
    }

    /// Return every scoped dependent dimension occurring below several roots.
    ///
    /// Shared type-DAG nodes are visited once across the complete root row.
    #[must_use]
    pub(crate) fn symbolic_dimensions_in(
        &self,
        roots: impl IntoIterator<Item = TypeId>,
    ) -> HashSet<Dimension> {
        self.dimensions_in(roots)
            .iter()
            .flat_map(dimension_symbols)
            .collect()
    }

    pub(crate) fn dimensions_in(
        &self,
        roots: impl IntoIterator<Item = TypeId>,
    ) -> HashSet<Dimension> {
        let mut dimensions = HashSet::new();
        let mut pending = roots.into_iter().collect::<Vec<_>>();
        let mut visited = HashSet::new();
        while let Some(type_id) = pending.pop() {
            if !visited.insert(type_id) {
                continue;
            }
            match self.get(type_id) {
                Some(SemanticType::Array { element, shape }) => {
                    dimensions.extend(shape.iter().cloned());
                    pending.push(*element);
                }
                Some(SemanticType::Word { width, .. }) => {
                    dimensions.insert(width.clone());
                }
                Some(SemanticType::Fixed {
                    integer_bits,
                    fractional_bits,
                }) => {
                    dimensions.extend([integer_bits, fractional_bits].into_iter().cloned());
                }
                Some(SemanticType::Product(elements)) => {
                    pending.extend(elements.iter().copied());
                }
                Some(SemanticType::Record { fields, .. }) => {
                    pending.extend(fields.iter().map(|(_, field_type)| *field_type));
                }
                Some(SemanticType::Quantum(inner) | SemanticType::Staged { inner, .. }) => {
                    pending.push(*inner);
                }
                Some(SemanticType::Function { inputs, output, .. }) => {
                    pending.extend(inputs.iter().copied());
                    pending.push(*output);
                }
                Some(SemanticType::Scalar(_)) | None => {}
            }
        }
        dimensions
    }

    /// Apply a scoped dependent-dimension substitution to one type DAG.
    ///
    /// # Errors
    ///
    /// Returns an error if a referenced type is missing or the arena cannot
    /// represent a derived node.
    pub(crate) fn substitute_dimensions(
        &mut self,
        root: TypeId,
        bindings: &HashMap<Dimension, Dimension>,
    ) -> Result<TypeId, String> {
        self.substitute_dimensions_cached(root, bindings, &mut HashMap::new())
    }

    /// Apply one substitution while reusing a caller-owned type-id cache.
    ///
    /// # Errors
    ///
    /// Returns an error under the same conditions as [`Self::substitute_dimensions`].
    pub(crate) fn substitute_dimensions_cached(
        &mut self,
        root: TypeId,
        bindings: &HashMap<Dimension, Dimension>,
        memo: &mut HashMap<TypeId, TypeId>,
    ) -> Result<TypeId, String> {
        if let Some(substituted) = memo.get(&root) {
            return Ok(*substituted);
        }
        let type_spec = self
            .get(root)
            .cloned()
            .ok_or_else(|| "missing semantic type during dimension substitution".to_owned())?;
        let substituted = match type_spec {
            SemanticType::Scalar(_) => root,
            SemanticType::Fixed {
                integer_bits,
                fractional_bits,
            } => self.intern_derived(SemanticType::Fixed {
                integer_bits: substitute_dimension(integer_bits, bindings),
                fractional_bits: substitute_dimension(fractional_bits, bindings),
            })?,
            SemanticType::Array { element, shape } => {
                let element = self.substitute_dimensions_cached(element, bindings, memo)?;
                self.intern_derived(SemanticType::Array {
                    element,
                    shape: shape
                        .into_iter()
                        .map(|dimension| substitute_dimension(dimension, bindings))
                        .collect(),
                })?
            }
            SemanticType::Word { kind, width } => self.intern_derived(SemanticType::Word {
                kind,
                width: substitute_dimension(width, bindings),
            })?,
            SemanticType::Product(elements) => {
                let elements = elements
                    .iter()
                    .map(|element| self.substitute_dimensions_cached(*element, bindings, memo))
                    .collect::<Result<Vec<_>, _>>()?;
                self.intern_derived(SemanticType::Product(elements.into_boxed_slice()))?
            }
            SemanticType::Record { name, fields } => {
                let fields = fields
                    .iter()
                    .map(|(name, field_type)| {
                        Ok((
                            name.clone(),
                            self.substitute_dimensions_cached(*field_type, bindings, memo)?,
                        ))
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                self.intern_derived(SemanticType::Record {
                    name,
                    fields: fields.into_boxed_slice(),
                })?
            }
            SemanticType::Quantum(inner) => {
                let inner = self.substitute_dimensions_cached(inner, bindings, memo)?;
                self.intern_derived(SemanticType::Quantum(inner))?
            }
            SemanticType::Staged { stage, inner } => {
                let inner = self.substitute_dimensions_cached(inner, bindings, memo)?;
                self.intern_derived(SemanticType::Staged { stage, inner })?
            }
            SemanticType::Function {
                parameter_names,
                inputs,
                output,
            } => {
                let inputs = inputs
                    .iter()
                    .map(|input| self.substitute_dimensions_cached(*input, bindings, memo))
                    .collect::<Result<Vec<_>, _>>()?;
                let output = self.substitute_dimensions_cached(output, bindings, memo)?;
                self.intern_derived(SemanticType::Function {
                    parameter_names,
                    inputs: inputs.into_boxed_slice(),
                    output,
                })?
            }
        };
        memo.insert(root, substituted);
        Ok(substituted)
    }

    /// Test rigid equality for two peers in one local classical equation.
    ///
    /// Only the operands' outer availability stages are ignored. Dependent
    /// dimensions and stages nested inside aggregates remain part of the
    /// structural identity. Flexible dimension substitution belongs to the
    /// separate call-site ABI verifier and must not leak into local equations.
    #[must_use]
    pub(crate) fn local_peers(&self, left: TypeId, right: TypeId) -> bool {
        self.strip_stage(left) == self.strip_stage(right)
    }

    /// Intern a type derived by a native semantic pass.
    ///
    /// # Errors
    ///
    /// Returns an error if the per-program arena cannot represent another
    /// stable identifier.
    pub(crate) fn intern_derived(&mut self, type_spec: SemanticType) -> Result<TypeId, String> {
        self.intern(type_spec)
    }

    /// Resolve one already-validated wire type in its dependent binder scope.
    ///
    /// # Errors
    ///
    /// Returns an error if the wire value lies outside the closed semantic
    /// universe. A successful program-wide type pass has already interned the
    /// returned node, so normal callers do not grow the arena here.
    pub(crate) fn resolve_wire_type(
        &mut self,
        source: &wire::TypeProto,
        scope: &Arc<str>,
    ) -> Result<TypeId, String> {
        TypeDecoder {
            scope: Arc::clone(scope),
            arena: self,
            work: None,
        }
        .decode(source)
    }

    /// Decode one canonical standalone type under a chosen binder scope.
    ///
    /// # Errors
    ///
    /// Returns an error before recursive decoding when the payload is
    /// malformed or exceeds the shared structural budgets, and after decoding
    /// when it is not the canonical encoding of its semantic type.
    pub(crate) fn decode_canonical_wire_type(
        &mut self,
        payload: &[u8],
        scope: &Arc<str>,
    ) -> Result<TypeId, String> {
        crate::budget::preflight_type(payload)
            .map_err(|error| format!("type payload: {error:?}"))?;
        let encoded =
            wire::TypeProto::decode(payload).map_err(|error| format!("type payload: {error}"))?;
        let type_id = self.resolve_wire_type(&encoded, scope)?;
        if self.encode_wire_type(type_id)?.encode_to_vec() != payload {
            return Err("type payload is not canonical".to_owned());
        }
        Ok(type_id)
    }

    /// Encode one interned semantic type into its canonical wire form.
    ///
    /// # Errors
    ///
    /// Returns an error if `id` or one of its referenced child identifiers is
    /// absent from this arena.
    pub(crate) fn encode_wire_type(&self, id: TypeId) -> Result<wire::TypeProto, String> {
        use wire::type_proto::Kind;

        let empty = |kind: Kind, name: &str| wire::TypeProto {
            kind: kind.into(),
            name: name.to_owned(),
            ..wire::TypeProto::default()
        };
        match self
            .get(id)
            .ok_or_else(|| "unknown semantic type identifier".to_owned())?
        {
            SemanticType::Scalar(scalar) => Ok(empty(Kind::Scalar, scalar.as_str())),
            SemanticType::Array { element, shape } => Ok(wire::TypeProto {
                kind: Kind::Array.into(),
                name: "Array".to_owned(),
                arguments: vec![self.encode_wire_type(*element)?],
                dimensions: shape.iter().map(encode_dimension).collect(),
                field_names: Vec::new(),
            }),
            SemanticType::Word { kind, width } => Ok(wire::TypeProto {
                kind: Kind::Array.into(),
                name: match kind {
                    WordKind::Bits => "Bits",
                    WordKind::UInt => "UInt",
                }
                .to_owned(),
                arguments: vec![empty(Kind::Scalar, "Bit")],
                dimensions: vec![encode_dimension(width)],
                field_names: Vec::new(),
            }),
            SemanticType::Fixed {
                integer_bits,
                fractional_bits,
            } => Ok(wire::TypeProto {
                kind: Kind::Fixed.into(),
                name: "UFixed".to_owned(),
                arguments: Vec::new(),
                dimensions: vec![
                    encode_dimension(integer_bits),
                    encode_dimension(fractional_bits),
                ],
                field_names: Vec::new(),
            }),
            SemanticType::Product(elements) => Ok(wire::TypeProto {
                kind: Kind::Product.into(),
                name: String::new(),
                arguments: elements
                    .iter()
                    .map(|element| self.encode_wire_type(*element))
                    .collect::<Result<Vec<_>, _>>()?,
                dimensions: Vec::new(),
                field_names: Vec::new(),
            }),
            SemanticType::Record { name, fields } => Ok(wire::TypeProto {
                kind: Kind::Record.into(),
                name: name.clone(),
                arguments: fields
                    .iter()
                    .map(|(_, field_type)| self.encode_wire_type(*field_type))
                    .collect::<Result<Vec<_>, _>>()?,
                dimensions: Vec::new(),
                field_names: fields.iter().map(|(name, _)| name.clone()).collect(),
            }),
            SemanticType::Quantum(inner) => Ok(wire::TypeProto {
                kind: Kind::Quantum.into(),
                name: String::new(),
                arguments: vec![self.encode_wire_type(*inner)?],
                dimensions: Vec::new(),
                field_names: Vec::new(),
            }),
            SemanticType::Staged { stage, inner } => Ok(wire::TypeProto {
                kind: Kind::Staged.into(),
                name: stage.as_str().to_owned(),
                arguments: vec![self.encode_wire_type(*inner)?],
                dimensions: Vec::new(),
                field_names: Vec::new(),
            }),
            SemanticType::Function {
                parameter_names,
                inputs,
                output,
            } => Ok(wire::TypeProto {
                kind: Kind::Function.into(),
                name: String::new(),
                arguments: inputs
                    .iter()
                    .chain(std::iter::once(output))
                    .map(|item| self.encode_wire_type(*item))
                    .collect::<Result<Vec<_>, _>>()?,
                dimensions: Vec::new(),
                field_names: parameter_names.to_vec(),
            }),
        }
    }

    /// Return the shared identity for one dependent-dimension scope.
    pub(crate) fn intern_scope(&mut self, scope: &str) -> Arc<str> {
        self.intern_string(scope)
    }

    /// Return the number of fields in a positional or named aggregate.
    #[must_use]
    pub(crate) fn aggregate_len(&self, aggregate: TypeId) -> Option<usize> {
        self.aggregate_summaries
            .get(aggregate.0 as usize)
            .copied()
            .flatten()
            .map(|summary| summary.len)
    }

    /// Return the homogeneous logical base of an aggregate, when one exists.
    #[must_use]
    pub(crate) fn homogeneous_aggregate_base(&self, aggregate: TypeId) -> Option<TypeId> {
        self.aggregate_summaries
            .get(aggregate.0 as usize)
            .copied()
            .flatten()
            .and_then(|summary| summary.uniform)
            .map(|fact| fact.base)
    }

    /// Resolve one projected aggregate field without materializing its full row.
    ///
    /// # Errors
    ///
    /// Returns an error if the derived stage wrapper cannot be interned.
    pub(crate) fn aggregate_element_at(
        &mut self,
        aggregate: TypeId,
        index: usize,
    ) -> Result<Option<TypeId>, String> {
        let outer_stage = self.stage_of(aggregate);
        let base = self.strip_stage(aggregate);
        let element = match self.get(base) {
            Some(SemanticType::Product(elements)) => elements.get(index).copied(),
            Some(SemanticType::Record { fields, .. }) => {
                fields.get(index).map(|(_, field_type)| *field_type)
            }
            _ => None,
        };
        let Some(element) = element else {
            return Ok(None);
        };
        let fact = self.element_fact(element, outer_stage);
        self.materialize_element(fact).map(Some)
    }

    /// Resolve and cache one rank-reducing sequence element type.
    ///
    /// # Errors
    ///
    /// Returns an error if a derived array or stage node cannot be interned.
    pub(crate) fn sequence_element(&mut self, sequence: TypeId) -> Result<Option<TypeId>, String> {
        if let Some(element) = self.sequence_elements.get(&sequence) {
            return Ok(Some(*element));
        }
        let stage = self.stage_of(sequence);
        let base = self.strip_stage(sequence);
        let derived = match self.get(base) {
            Some(SemanticType::Word { .. }) => Some((None, None)),
            Some(SemanticType::Array { element, shape }) if shape.len() == 1 => {
                Some((Some(*element), None))
            }
            Some(SemanticType::Array { element, shape }) => {
                Some((Some(*element), Some(shape[1..].to_vec())))
            }
            _ => None,
        };
        let Some((element, remaining_shape)) = derived else {
            return Ok(None);
        };
        let child = if let Some(shape) = remaining_shape {
            self.intern(SemanticType::Array {
                element: element.expect("array elements are present"),
                shape,
            })?
        } else if let Some(element) = element {
            element
        } else {
            self.intern(SemanticType::Scalar(ScalarType::Bit))?
        };
        let child = self.materialize_element(ElementFact {
            base: self.strip_stage(child),
            stage,
        })?;
        self.sequence_elements.insert(sequence, child);
        Ok(Some(child))
    }

    fn intern(&mut self, type_spec: SemanticType) -> Result<TypeId, String> {
        if let Some(id) = self.indices.get(&type_spec) {
            return Ok(*id);
        }
        let index = u32::try_from(self.entries.len())
            .map_err(|_| "QCore type arena exceeds its supported capacity".to_owned())?;
        let id = TypeId(index);
        let contains_quantum = self.type_contains_quantum(&type_spec);
        let aggregate_summary = self.summarize_aggregate(&type_spec);
        let type_spec = Arc::new(type_spec);
        self.entries.push(Arc::clone(&type_spec));
        self.indices.insert(type_spec, id);
        self.quantum_containment.push(contains_quantum);
        self.aggregate_summaries.push(aggregate_summary);
        Ok(id)
    }

    fn intern_string(&mut self, value: &str) -> Arc<str> {
        if let Some(shared) = self.strings.get(value) {
            return Arc::clone(shared);
        }
        let shared = Arc::<str>::from(value);
        self.strings.insert(Arc::clone(&shared));
        shared
    }

    fn type_contains_quantum(&self, type_spec: &SemanticType) -> bool {
        match type_spec {
            SemanticType::Quantum(_) => true,
            SemanticType::Array { element, .. } => self.is_quantum(*element),
            SemanticType::Product(elements) => {
                elements.iter().any(|element| self.is_quantum(*element))
            }
            SemanticType::Record { fields, .. } => fields
                .iter()
                .any(|(_, field_type)| self.is_quantum(*field_type)),
            SemanticType::Staged { inner, .. } => self.is_quantum(*inner),
            SemanticType::Scalar(_)
            | SemanticType::Word { .. }
            | SemanticType::Fixed { .. }
            | SemanticType::Function { .. } => false,
        }
    }

    fn summarize_aggregate(&self, type_spec: &SemanticType) -> Option<AggregateSummary> {
        match type_spec {
            SemanticType::Product(elements) => {
                Some(self.summarize_elements(elements.iter().copied(), None))
            }
            SemanticType::Record { fields, .. } => {
                Some(self.summarize_elements(fields.iter().map(|(_, value)| *value), None))
            }
            SemanticType::Staged { stage, inner } => match self.get(*inner) {
                Some(SemanticType::Product(elements)) => {
                    Some(self.summarize_elements(elements.iter().copied(), Some(*stage)))
                }
                Some(SemanticType::Record { fields, .. }) => Some(
                    self.summarize_elements(fields.iter().map(|(_, value)| *value), Some(*stage)),
                ),
                _ => None,
            },
            _ => None,
        }
    }

    fn summarize_elements(
        &self,
        elements: impl IntoIterator<Item = TypeId>,
        outer_stage: Option<Stage>,
    ) -> AggregateSummary {
        let mut len = 0_usize;
        let mut uniform = None;
        let mut heterogeneous = false;
        for element in elements {
            len += 1;
            let fact = self.element_fact(element, outer_stage);
            if let Some(first) = uniform {
                heterogeneous |= first != fact;
            } else {
                uniform = Some(fact);
            }
        }
        AggregateSummary {
            len,
            uniform: if heterogeneous { None } else { uniform },
        }
    }

    fn element_fact(&self, element: TypeId, outer_stage: Option<Stage>) -> ElementFact {
        ElementFact {
            base: self.strip_stage(element),
            stage: Stage::join(outer_stage, self.stage_of(element)),
        }
    }

    fn materialize_element(&mut self, fact: ElementFact) -> Result<TypeId, String> {
        match fact.stage {
            Some(stage) => self.intern(SemanticType::Staged {
                stage,
                inner: fact.base,
            }),
            None => Ok(fact.base),
        }
    }
}

#[cfg(test)]
pub(crate) fn validate_program_types(
    package: &wire::QCorePackageProto,
) -> Result<TypeArena, String> {
    let mut work = QCoreVerificationWork::default();
    validate_program_types_with_work(package, &mut work)
}

pub(crate) fn validate_program_types_with_work(
    package: &wire::QCorePackageProto,
    work: &mut QCoreVerificationWork,
) -> Result<TypeArena, String> {
    let mut arena = TypeArena::default();
    let constant_scope = arena.intern_scope(&format!("{}::<constants>", package.module_name));
    for constant in &package.constants {
        validate_required_type(
            constant.r#type.as_ref(),
            || format!("constant {:?}", constant.name),
            &constant_scope,
            &mut arena,
            work,
        )?;
    }
    for function in &package.functions {
        let function_scope = arena.intern_scope(&function.symbol);
        validate_required_type(
            function.result_type.as_ref(),
            || format!("result of {:?}", function.symbol),
            &function_scope,
            &mut arena,
            work,
        )?;
        for parameter in &function.parameters {
            let value = parameter.value.as_ref().ok_or_else(|| {
                format!(
                    "parameter {:?} of {:?} has no value definition",
                    parameter.name, function.symbol
                )
            })?;
            validate_value(value, &function_scope, &mut arena, work)?;
        }
        if let Some(body) = &function.body {
            validate_block(body, &function_scope, &mut arena, work)?;
        }
    }
    Ok(arena)
}

fn validate_block(
    block: &wire::BlockProto,
    function: &Arc<str>,
    arena: &mut TypeArena,
    work: &mut QCoreVerificationWork,
) -> Result<(), String> {
    for argument in &block.arguments {
        validate_value(argument, function, arena, work)?;
    }
    for operation in &block.operations {
        for result in &operation.results {
            validate_value(result, function, arena, work)?;
        }
        for region in &operation.regions {
            for nested in &region.blocks {
                validate_block(nested, function, arena, work)?;
            }
        }
    }
    Ok(())
}

fn validate_value(
    value: &wire::ValueProto,
    function: &Arc<str>,
    arena: &mut TypeArena,
    work: &mut QCoreVerificationWork,
) -> Result<(), String> {
    validate_required_type(
        value.r#type.as_ref(),
        || format!("value %{} in {function:?}", value.id),
        function,
        arena,
        work,
    )
}

fn validate_required_type<F>(
    source: Option<&wire::TypeProto>,
    context: F,
    scope: &Arc<str>,
    arena: &mut TypeArena,
    work: &mut QCoreVerificationWork,
) -> Result<(), String>
where
    F: FnOnce() -> String,
{
    let Some(source) = source else {
        return Err(format!("{} has no semantic type", context()));
    };
    let context = context();
    let type_id = TypeDecoder {
        scope: Arc::clone(scope),
        arena,
        work: Some(work),
    }
    .decode(source)
    .map_err(|error| format!("{context}: {error}"))?;
    if arena.contains_function(type_id) {
        return Err(format!(
            "{context}: function values must be eliminated before persistent QCore"
        ));
    }
    Ok(())
}

struct TypeDecoder<'arena, 'work> {
    scope: Arc<str>,
    arena: &'arena mut TypeArena,
    work: Option<&'work mut QCoreVerificationWork>,
}

impl TypeDecoder<'_, '_> {
    fn decode(&mut self, source: &wire::TypeProto) -> Result<TypeId, String> {
        use wire::type_proto::Kind;

        if let Some(work) = self.work.as_deref_mut() {
            work.charge_types(
                1_usize
                    .saturating_add(source.dimensions.len())
                    .saturating_add(source.field_names.len()),
            );
        }

        let kind = Kind::try_from(source.kind)
            .map_err(|_| format!("unknown semantic type kind {}", source.kind))?;
        match kind {
            Kind::Scalar => self.decode_scalar(source),
            Kind::Array => self.decode_array(source),
            Kind::Fixed => self.decode_fixed(source),
            Kind::Product => self.decode_product(source),
            Kind::Record => self.decode_record(source),
            Kind::Quantum => self.decode_quantum(source),
            Kind::Staged => self.decode_staged(source),
            Kind::Function => self.decode_function(source),
            Kind::Unspecified => Err("unknown semantic type kind 0".to_owned()),
        }
    }

    fn decode_scalar(&mut self, source: &wire::TypeProto) -> Result<TypeId, String> {
        require_empty(&source.arguments, "scalar arguments")?;
        require_empty(&source.dimensions, "scalar dimensions")?;
        require_empty(&source.field_names, "scalar field names")?;
        let scalar = match source.name.as_str() {
            "Unit" => ScalarType::Unit,
            "Bit" => ScalarType::Bit,
            "Bool" => ScalarType::Bool,
            "Int" => ScalarType::Int,
            "Real" => ScalarType::Real,
            "Complex" => ScalarType::Complex,
            "Probability" => ScalarType::Probability,
            "Observable" => ScalarType::Observable,
            name => return Err(format!("unknown scalar type {name:?}")),
        };
        self.arena.intern(SemanticType::Scalar(scalar))
    }

    fn decode_array(&mut self, source: &wire::TypeProto) -> Result<TypeId, String> {
        require_empty(&source.field_names, "array field names")?;
        if source.arguments.len() != 1 {
            return Err(format!(
                "array expects one element type, got {}",
                source.arguments.len()
            ));
        }
        if source.dimensions.is_empty() {
            return Err("an array must have at least one dimension".to_owned());
        }
        let element = self.decode(&source.arguments[0])?;
        let shape = source
            .dimensions
            .iter()
            .map(|dimension| self.decode_dimension(dimension))
            .collect::<Result<Vec<_>, _>>()?;
        match source.name.as_str() {
            "Array" => {
                if self.arena.is_quantum(element) {
                    return Err("classical arrays cannot contain owned quantum data".to_owned());
                }
                if matches!(self.arena.get(element), Some(SemanticType::Array { .. })) {
                    return Err("nested arrays are not in canonical flattened form".to_owned());
                }
                self.arena.intern(SemanticType::Array { element, shape })
            }
            "Bits" | "UInt"
                if self.arena.get(element) == Some(&SemanticType::Scalar(ScalarType::Bit))
                    && shape.len() == 1 =>
            {
                let width = shape
                    .into_iter()
                    .next()
                    .expect("a canonical word has exactly one dimension");
                self.arena.intern(SemanticType::Word {
                    kind: if source.name == "Bits" {
                        WordKind::Bits
                    } else {
                        WordKind::UInt
                    },
                    width,
                })
            }
            name => Err(format!("unknown array encoding {name:?}")),
        }
    }

    fn decode_fixed(&mut self, source: &wire::TypeProto) -> Result<TypeId, String> {
        require_empty(&source.arguments, "UFixed arguments")?;
        require_empty(&source.field_names, "UFixed field names")?;
        if source.name != "UFixed" || source.dimensions.len() != 2 {
            return Err("malformed UFixed encoding".to_owned());
        }
        let widths = source
            .dimensions
            .iter()
            .map(|dimension| self.decode_fixed_dimension(dimension))
            .collect::<Result<Vec<_>, _>>()?;
        if widths[0] == Dimension::Fixed(0) && widths[1] == Dimension::Fixed(0) {
            return Err("UFixed must contain at least one bit".to_owned());
        }
        self.arena.intern(SemanticType::Fixed {
            integer_bits: widths[0].clone(),
            fractional_bits: widths[1].clone(),
        })
    }

    fn decode_product(&mut self, source: &wire::TypeProto) -> Result<TypeId, String> {
        require_empty_name(source, "product")?;
        require_empty(&source.dimensions, "product dimensions")?;
        require_empty(&source.field_names, "product field names")?;
        let elements = source
            .arguments
            .iter()
            .map(|argument| self.decode(argument))
            .collect::<Result<Vec<_>, _>>()?;
        self.arena
            .intern(SemanticType::Product(elements.into_boxed_slice()))
    }

    fn decode_record(&mut self, source: &wire::TypeProto) -> Result<TypeId, String> {
        require_empty(&source.dimensions, "record dimensions")?;
        if source.name.is_empty() || source.arguments.is_empty() {
            return Err("record types require nonempty names and fields".to_owned());
        }
        if source.field_names.len() != source.arguments.len() {
            return Err("record field/type arity mismatch".to_owned());
        }
        if source.field_names.iter().any(String::is_empty)
            || source.field_names.iter().collect::<HashSet<_>>().len() != source.field_names.len()
        {
            return Err("record fields must have unique nonempty names".to_owned());
        }
        let mut fields = Vec::with_capacity(source.arguments.len());
        for (name, field_type) in source.field_names.iter().zip(&source.arguments) {
            fields.push((name.clone(), self.decode(field_type)?));
        }
        self.arena.intern(SemanticType::Record {
            name: source.name.clone(),
            fields: fields.into_boxed_slice(),
        })
    }

    fn decode_quantum(&mut self, source: &wire::TypeProto) -> Result<TypeId, String> {
        require_empty_name(source, "quantum")?;
        require_empty(&source.dimensions, "quantum dimensions")?;
        require_empty(&source.field_names, "quantum field names")?;
        if source.arguments.len() != 1 {
            return Err(format!(
                "quantum expects one payload type, got {}",
                source.arguments.len()
            ));
        }
        let inner = self.decode(&source.arguments[0])?;
        if !self.arena.is_coherent_payload(inner) {
            return Err(
                "Q[T] requires a bit, word, fixed point, or nonempty product of those types"
                    .to_owned(),
            );
        }
        self.arena.intern(SemanticType::Quantum(inner))
    }

    fn decode_staged(&mut self, source: &wire::TypeProto) -> Result<TypeId, String> {
        require_empty(&source.dimensions, "staged dimensions")?;
        require_empty(&source.field_names, "staged field names")?;
        if source.arguments.len() != 1 {
            return Err(format!(
                "staged type expects one payload type, got {}",
                source.arguments.len()
            ));
        }
        let stage = Stage::from_name(&source.name)
            .ok_or_else(|| format!("unknown availability stage {:?}", source.name))?;
        let inner = self.decode(&source.arguments[0])?;
        if !self.arena.is_stageable(inner) {
            return Err("availability stages contain only unstaged classical data".to_owned());
        }
        self.arena.intern(SemanticType::Staged { stage, inner })
    }

    fn decode_function(&mut self, source: &wire::TypeProto) -> Result<TypeId, String> {
        require_empty_name(source, "function")?;
        require_empty(&source.dimensions, "function dimensions")?;
        if source.arguments.is_empty() {
            return Err(format!(
                "function type expects an output type, got {}",
                source.arguments.len()
            ));
        }
        let inputs = source.arguments[..source.arguments.len() - 1]
            .iter()
            .map(|argument| self.decode(argument))
            .collect::<Result<Vec<_>, _>>()?;
        let output = self.decode(&source.arguments[source.arguments.len() - 1])?;
        if !source.field_names.is_empty()
            && (source.field_names.len() != inputs.len()
                || source.field_names.iter().any(String::is_empty)
                || source.field_names.iter().collect::<HashSet<_>>().len()
                    != source.field_names.len())
        {
            return Err(
                "function parameter names must be a complete unique nonempty row".to_owned(),
            );
        }
        self.arena.intern(SemanticType::Function {
            parameter_names: source.field_names.clone().into_boxed_slice(),
            inputs: inputs.into_boxed_slice(),
            output,
        })
    }

    fn decode_dimension(&mut self, source: &wire::DimensionProto) -> Result<Dimension, String> {
        self.decode_dimension_with_zero(source, false)
    }

    fn decode_fixed_dimension(
        &mut self,
        source: &wire::DimensionProto,
    ) -> Result<Dimension, String> {
        self.decode_dimension_with_zero(source, true)
    }

    fn decode_dimension_with_zero(
        &mut self,
        source: &wire::DimensionProto,
        allow_zero: bool,
    ) -> Result<Dimension, String> {
        match &source.value {
            Some(wire::dimension_proto::Value::Fixed(value)) if *value > 0 || allow_zero => {
                Ok(Dimension::Fixed(*value))
            }
            Some(wire::dimension_proto::Value::Fixed(_)) => {
                Err("a shape dimension must be positive".to_owned())
            }
            Some(wire::dimension_proto::Value::Symbol(value)) if !value.is_empty() => {
                if value.starts_with("slice(") {
                    let shape = crate::core::slice_shape::SliceShape::parse(value)?;
                    if shape.symbols().is_empty() {
                        return Err("resolved slice shapes must use fixed dimensions".to_owned());
                    }
                } else {
                    PolynomialDimension::parse(value)?;
                }
                Ok(Dimension::Symbol {
                    scope: Arc::clone(&self.scope),
                    name: self.arena.intern_string(value),
                })
            }
            Some(wire::dimension_proto::Value::Symbol(_)) => {
                Err("a symbolic shape dimension must be nonempty".to_owned())
            }
            None => Err("array dimension has no value".to_owned()),
        }
    }
}

fn encode_dimension(dimension: &Dimension) -> wire::DimensionProto {
    wire::DimensionProto {
        value: Some(match dimension {
            Dimension::Fixed(value) => wire::dimension_proto::Value::Fixed(*value),
            Dimension::Symbol { name, .. } => {
                wire::dimension_proto::Value::Symbol(name.to_string())
            }
        }),
    }
}

const fn fixed_dimension(dimension: &Dimension) -> Option<u64> {
    match dimension {
        Dimension::Fixed(width) => Some(*width),
        Dimension::Symbol { .. } => None,
    }
}

fn resolve_coherent_leaf_width(
    leaf: CoherentLeafWidth<'_>,
    resolve: &mut impl FnMut(&Dimension) -> Option<u64>,
) -> Result<Option<u64>, String> {
    match leaf {
        CoherentLeafWidth::Bit => Ok(Some(1)),
        CoherentLeafWidth::Word(dimension) => Ok(resolve(dimension)),
        CoherentLeafWidth::Fixed {
            integer_bits,
            fractional_bits,
        } => match (resolve(integer_bits), resolve(fractional_bits)) {
            (Some(integer), Some(fractional)) => integer
                .checked_add(fractional)
                .map(Some)
                .ok_or_else(|| "fixed-point coherent width exceeds u64".to_owned()),
            _ => Ok(None),
        },
    }
}

fn require_empty<T>(values: &[T], description: &str) -> Result<(), String> {
    if values.is_empty() {
        Ok(())
    } else {
        Err(format!("{description} must be empty"))
    }
}

fn require_empty_name(source: &wire::TypeProto, description: &str) -> Result<(), String> {
    if source.name.is_empty() {
        Ok(())
    } else {
        Err(format!("{description} type name must be empty"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affine_dimensions_are_canonical_bounded_and_symbolic() {
        let expression = AffineDimension::parse("m+3*n-2").expect("canonical affine");

        assert_eq!(expression.constant, -2);
        assert_eq!(expression.terms.get("m"), Some(&1));
        assert_eq!(expression.terms.get("n"), Some(&3));
        assert_eq!(expression.render(), "m+3*n-2");
        assert!(AffineDimension::parse("3*n+m-2").is_err());
        assert!(AffineDimension::parse("n*m").is_err());
        assert!(AffineDimension::parse("n + 1").is_err());
        assert!(AffineDimension::parse("n-9223372036854775807-1").is_err());
        assert!(AffineDimension::parse("n-9223372036854775807*n-n").is_err());
    }

    #[test]
    fn polynomial_dimensions_preserve_independent_factors() {
        let expression = PolynomialDimension::parse("d*m+2*n-1").expect("canonical polynomial");

        assert_eq!(expression.constant, -1);
        assert_eq!(
            expression.terms.get(&vec!["d".into(), "m".into()]),
            Some(&1)
        );
        assert_eq!(expression.terms.get(&vec!["n".into()]), Some(&2));
        assert_eq!(expression.render(), "d*m+2*n-1");
        assert_eq!(
            expression.evaluate(|name| match name {
                "d" => Some(3),
                "m" => Some(4),
                "n" => Some(5),
                _ => None,
            }),
            Some(21)
        );
        assert!(PolynomialDimension::parse("m*d").is_err());
        assert!(PolynomialDimension::parse("d * m").is_err());
    }

    #[test]
    fn floor_quotient_factors_are_atoms_that_evaluate() {
        let expression = PolynomialDimension::parse("-(n//2)+n").expect("canonical floor quotient");
        assert_eq!(expression.render(), "-(n//2)+n");
        assert_eq!(expression.symbols(), vec!["n".to_owned()]);
        let resolve = |name: &str| (name == "n").then_some(7);
        assert_eq!(expression.evaluate(resolve), Some(4));

        let nested = PolynomialDimension::parse("2*((n+1)//3)").expect("nested numerator");
        assert_eq!(nested.evaluate(|name| (name == "n").then_some(8)), Some(6));

        let half = PolynomialDimension::parse("(n//2)").expect("atom");
        let suffix = PolynomialDimension::parse("-(n//2)+n").expect("suffix");
        let whole = half.add_scaled(&suffix, 1).expect("sum");
        assert_eq!(whole.render(), "n");

        for invalid in [
            "n//2", "(n//1)", "((n)//2)", "(n//2", "(2*n//2)", "(n+1//2)",
        ] {
            assert!(PolynomialDimension::parse(invalid).is_err(), "{invalid}");
        }
    }

    fn scalar(name: &str) -> wire::TypeProto {
        wire::TypeProto {
            kind: wire::type_proto::Kind::Scalar.into(),
            name: name.to_owned(),
            ..Default::default()
        }
    }

    fn staged(name: &str, inner: wire::TypeProto) -> wire::TypeProto {
        wire::TypeProto {
            kind: wire::type_proto::Kind::Staged.into(),
            name: name.to_owned(),
            arguments: vec![inner],
            ..Default::default()
        }
    }

    fn decode_in_scope(
        arena: &mut TypeArena,
        encoded: &wire::TypeProto,
        scope: &str,
    ) -> Result<TypeId, String> {
        let scope = arena.intern_scope(scope);
        TypeDecoder {
            scope,
            arena,
            work: None,
        }
        .decode(encoded)
    }

    #[test]
    fn decodes_a_dependent_word() {
        let encoded = wire::TypeProto {
            kind: wire::type_proto::Kind::Array.into(),
            name: "Bits".to_owned(),
            arguments: vec![scalar("Bit")],
            dimensions: vec![wire::DimensionProto {
                value: Some(wire::dimension_proto::Value::Symbol("width".to_owned())),
            }],
            ..Default::default()
        };

        let mut arena = TypeArena::default();
        let id =
            decode_in_scope(&mut arena, &encoded, "native_test.main").expect("valid word type");

        assert_eq!(
            arena.get(id),
            Some(&SemanticType::Word {
                kind: WordKind::Bits,
                width: Dimension::Symbol {
                    scope: Arc::from("native_test.main"),
                    name: Arc::from("width"),
                },
            })
        );
        assert_eq!(arena.encode_wire_type(id), Ok(encoded));
    }

    #[test]
    fn rejects_zero_array_dimensions() {
        let encoded = wire::TypeProto {
            kind: wire::type_proto::Kind::Array.into(),
            name: "Array".to_owned(),
            arguments: vec![scalar("Real")],
            dimensions: vec![wire::DimensionProto {
                value: Some(wire::dimension_proto::Value::Fixed(0)),
            }],
            ..Default::default()
        };

        let mut arena = TypeArena::default();
        assert!(decode_in_scope(&mut arena, &encoded, "native_test.main").is_err());
    }

    #[test]
    fn rejects_nested_staging() {
        let encoded = staged("Param", staged("Const", scalar("Int")));

        let mut arena = TypeArena::default();
        assert!(decode_in_scope(&mut arena, &encoded, "native_test.main").is_err());
    }

    #[test]
    fn function_types_are_non_linear_and_preserve_signature_dimensions() {
        let encoded = wire::TypeProto {
            kind: wire::type_proto::Kind::Function.into(),
            arguments: vec![
                wire::TypeProto {
                    kind: wire::type_proto::Kind::Array.into(),
                    name: "Bits".to_owned(),
                    arguments: vec![scalar("Bit")],
                    dimensions: vec![wire::DimensionProto {
                        value: Some(wire::dimension_proto::Value::Symbol("n".to_owned())),
                    }],
                    ..Default::default()
                },
                scalar("Int"),
                scalar("Bool"),
            ],
            field_names: vec!["bits".to_owned(), "count".to_owned()],
            ..Default::default()
        };
        let mut arena = TypeArena::default();
        let function =
            decode_in_scope(&mut arena, &encoded, "native_test.main").expect("valid function type");

        assert!(!arena.is_quantum(function));
        assert!(arena.contains_function(function));
        assert!(!arena.symbolic_dimensions(function).is_empty());
        assert_eq!(arena.encode_wire_type(function), Ok(encoded.clone()));

        let nullary_encoded = wire::TypeProto {
            kind: wire::type_proto::Kind::Function.into(),
            arguments: vec![scalar("Bool")],
            ..Default::default()
        };
        let nullary = decode_in_scope(&mut arena, &nullary_encoded, "native_test.main")
            .expect("valid nullary function type");
        assert!(matches!(
            arena.get(nullary),
            Some(SemanticType::Function { inputs, .. }) if inputs.is_empty()
        ));
        assert_eq!(arena.encode_wire_type(nullary), Ok(nullary_encoded));

        let malformed_names = wire::TypeProto {
            kind: wire::type_proto::Kind::Function.into(),
            arguments: vec![scalar("Int"), scalar("Bool")],
            field_names: vec!["left".to_owned(), "extra".to_owned()],
            ..Default::default()
        };
        assert!(decode_in_scope(&mut arena, &malformed_names, "native_test.main").is_err());

        let mut persistent_arena = TypeArena::default();
        let persistent_scope = persistent_arena.intern_scope("native_test.main");
        let mut work = QCoreVerificationWork::default();
        assert!(
            validate_required_type(
                Some(&encoded),
                || "function parameter".to_owned(),
                &persistent_scope,
                &mut persistent_arena,
                &mut work,
            )
            .is_err_and(|error| error.contains("function values must be eliminated"))
        );
    }

    #[test]
    fn rejects_noncoherent_quantum_payloads() {
        let encoded = wire::TypeProto {
            kind: wire::type_proto::Kind::Quantum.into(),
            arguments: vec![scalar("Real")],
            ..Default::default()
        };

        let mut arena = TypeArena::default();
        assert!(decode_in_scope(&mut arena, &encoded, "native_test.main").is_err());
    }

    #[test]
    fn derives_fixed_coherent_totals_from_one_leaf_projection() {
        let mut arena = TypeArena::default();
        let bit = arena
            .intern(SemanticType::Scalar(ScalarType::Bit))
            .expect("bit fits the arena");
        let word = arena
            .intern(SemanticType::Word {
                kind: WordKind::Bits,
                width: Dimension::Fixed(2),
            })
            .expect("word fits the arena");
        let fixed = arena
            .intern(SemanticType::Fixed {
                integer_bits: Dimension::Fixed(1),
                fractional_bits: Dimension::Fixed(2),
            })
            .expect("fixed point fits the arena");
        let payload = arena
            .intern(SemanticType::Product(Box::new([word, fixed])))
            .expect("payload fits the arena");
        let state = arena
            .intern(SemanticType::Quantum(payload))
            .expect("state fits the arena");
        let qubit = arena
            .intern(SemanticType::Quantum(bit))
            .expect("qubit fits the arena");
        let owners = arena
            .intern(SemanticType::Product(Box::new([qubit, state])))
            .expect("owner product fits the arena");
        let record = arena
            .intern(SemanticType::Record {
                name: "Pair".to_owned(),
                fields: Box::new([("first".to_owned(), qubit), ("second".to_owned(), state)]),
            })
            .expect("owner record fits the arena");
        let nested_record = arena
            .intern(SemanticType::Record {
                name: "Nested".to_owned(),
                fields: Box::new([("pair".to_owned(), record), ("last".to_owned(), qubit)]),
            })
            .expect("nested owner record fits the arena");
        let symbolic_word = arena
            .intern(SemanticType::Word {
                kind: WordKind::Bits,
                width: Dimension::Symbol {
                    scope: Arc::from("test"),
                    name: Arc::from("n"),
                },
            })
            .expect("symbolic word fits the arena");
        let symbolic_state = arena
            .intern(SemanticType::Quantum(symbolic_word))
            .expect("symbolic state fits the arena");

        assert_eq!(arena.coherent_payload_width(payload), Ok(Some(5)));
        assert_eq!(arena.coherent_width(state), Ok(Some(5)));
        assert_eq!(arena.coherent_width(owners), Ok(Some(6)));
        assert_eq!(arena.coherent_width(record), Ok(Some(6)));
        assert_eq!(arena.coherent_width(nested_record), Ok(Some(7)));
        assert_eq!(arena.coherent_payload_width(record), Ok(None));
        assert!(!arena.is_coherent_payload(record));
        assert_eq!(arena.coherent_width(payload), Ok(None));
        assert_eq!(arena.coherent_width(symbolic_state), Ok(None));
        assert_eq!(
            arena.coherent_width_with(symbolic_state, |_| Some(7)),
            Ok(Some(7))
        );
    }

    #[test]
    fn derives_sampled_widths_and_rejects_mixed_availability() {
        let mut arena = TypeArena::default();
        let bit = arena
            .intern(SemanticType::Scalar(ScalarType::Bit))
            .expect("bit fits the arena");
        let word = arena
            .intern(SemanticType::Word {
                kind: WordKind::Bits,
                width: Dimension::Fixed(3),
            })
            .expect("word fits the arena");
        let sampled_bit = arena
            .intern(SemanticType::Staged {
                stage: Stage::Shot,
                inner: bit,
            })
            .expect("sampled bit fits the arena");
        let sampled_word = arena
            .intern(SemanticType::Staged {
                stage: Stage::Shot,
                inner: word,
            })
            .expect("sampled word fits the arena");
        let sampled_product = arena
            .intern(SemanticType::Product(Box::new([sampled_bit, sampled_word])))
            .expect("sampled product fits the arena");
        let parameter_bit = arena
            .intern(SemanticType::Staged {
                stage: Stage::Param,
                inner: bit,
            })
            .expect("parameter bit fits the arena");
        let mixed = arena
            .intern(SemanticType::Product(Box::new([
                sampled_bit,
                parameter_bit,
            ])))
            .expect("mixed product fits the arena");

        assert_eq!(arena.sampled_width(sampled_product), Ok(Some(4)));
        assert_eq!(arena.sampled_width(mixed), Ok(None));
        assert!(arena.contains_stage(mixed, Stage::Shot));
    }

    #[test]
    fn scopes_dependent_dimensions_by_function() {
        let encoded = wire::TypeProto {
            kind: wire::type_proto::Kind::Array.into(),
            name: "UInt".to_owned(),
            arguments: vec![scalar("Bit")],
            dimensions: vec![wire::DimensionProto {
                value: Some(wire::dimension_proto::Value::Symbol("n".to_owned())),
            }],
            ..Default::default()
        };
        let mut arena = TypeArena::default();
        let left =
            decode_in_scope(&mut arena, &encoded, "native_test.left").expect("valid left type");
        let right =
            decode_in_scope(&mut arena, &encoded, "native_test.right").expect("valid right type");

        assert_ne!(left, right);
    }

    #[test]
    fn hash_conses_structurally_equal_types() {
        let encoded = scalar("Real");
        let mut arena = TypeArena::default();
        let first =
            decode_in_scope(&mut arena, &encoded, "native_test.main").expect("valid first type");
        let second =
            decode_in_scope(&mut arena, &encoded, "native_test.main").expect("valid second type");

        assert_eq!(first, second);
        assert_eq!(arena.len(), 1);
    }

    #[test]
    fn shares_large_scope_and_type_storage_across_many_dimensions() {
        let long_scope = "s".repeat(1024 * 1024 - 64);
        let encoded = wire::DimensionProto {
            value: Some(wire::dimension_proto::Value::Symbol("n".to_owned())),
        };
        let mut arena = TypeArena::default();
        let scope = arena.intern_scope(&long_scope);
        let dimensions = {
            let mut decoder = TypeDecoder {
                scope,
                arena: &mut arena,
                work: None,
            };
            (0..2048)
                .map(|_| decoder.decode_dimension(&encoded))
                .collect::<Result<Vec<_>, _>>()
                .expect("valid repeated symbolic dimensions")
        };
        let Dimension::Symbol {
            scope: first_scope,
            name: first_name,
        } = &dimensions[0]
        else {
            panic!("test dimension must be symbolic");
        };
        assert_eq!(first_scope.as_ref(), long_scope.as_str());
        assert_eq!(first_name.as_ref(), "n");
        for dimension in &dimensions {
            let Dimension::Symbol { scope, name } = dimension else {
                panic!("test dimension must be symbolic");
            };
            assert!(Arc::ptr_eq(first_scope, scope));
            assert!(Arc::ptr_eq(first_name, name));
        }

        let element = arena
            .intern(SemanticType::Scalar(ScalarType::Real))
            .expect("test scalar fits the arena");
        let array = arena
            .intern(SemanticType::Array {
                element,
                shape: vec![dimensions[0].clone()],
            })
            .expect("test array fits the arena");
        let stored = &arena.entries[array.0 as usize];
        let (indexed, _) = arena
            .indices
            .get_key_value(stored.as_ref())
            .expect("stored node has a hash-cons index");
        assert!(Arc::ptr_eq(stored, indexed));
    }

    #[test]
    fn caches_quantum_and_aggregate_facts_per_type_id() {
        let mut arena = TypeArena::default();
        let integer = arena
            .intern(SemanticType::Scalar(ScalarType::Int))
            .expect("test scalar fits the arena");
        let bit = arena
            .intern(SemanticType::Scalar(ScalarType::Bit))
            .expect("test scalar fits the arena");
        let quantum = arena
            .intern(SemanticType::Quantum(bit))
            .expect("test quantum type fits the arena");
        let mixed = arena
            .intern(SemanticType::Product(Box::new([integer, quantum])))
            .expect("test product fits the arena");
        let homogeneous = arena
            .intern(SemanticType::Product(
                vec![integer; 4096].into_boxed_slice(),
            ))
            .expect("large homogeneous product fits the arena");
        let record = arena
            .intern(SemanticType::Record {
                name: "Pair".to_owned(),
                fields: Box::new([("left".to_owned(), integer), ("right".to_owned(), integer)]),
            })
            .expect("test record fits the arena");

        assert!(arena.is_quantum(mixed));
        assert!(!arena.is_quantum(homogeneous));
        assert_eq!(
            arena.aggregate_summaries[homogeneous.0 as usize],
            Some(AggregateSummary {
                len: 4096,
                uniform: Some(ElementFact {
                    base: integer,
                    stage: None,
                }),
            })
        );
        assert_eq!(arena.homogeneous_aggregate_base(record), Some(integer));
        let type_count = arena.len();
        for _ in 0..2048 {
            assert_eq!(
                arena
                    .aggregate_element_at(homogeneous, 4095)
                    .expect("cached projection remains valid"),
                Some(integer)
            );
            assert_eq!(arena.homogeneous_aggregate_base(record), Some(integer));
            assert!(arena.is_quantum(mixed));
        }
        assert_eq!(arena.len(), type_count);
    }

    #[test]
    fn caches_rank_reducing_sequence_derivations() {
        let mut arena = TypeArena::default();
        let real = arena
            .intern(SemanticType::Scalar(ScalarType::Real))
            .expect("test scalar fits the arena");
        let array = arena
            .intern(SemanticType::Array {
                element: real,
                shape: vec![Dimension::Fixed(2); 4096],
            })
            .expect("large array fits the arena");

        let element = arena
            .sequence_element(array)
            .expect("rank reduction succeeds")
            .expect("array has an element type");
        let type_count = arena.len();
        assert_eq!(arena.sequence_elements.get(&array), Some(&element));
        for _ in 0..2048 {
            assert_eq!(
                arena
                    .sequence_element(array)
                    .expect("cached lookup succeeds"),
                Some(element)
            );
        }
        assert_eq!(arena.len(), type_count);
        assert_eq!(arena.sequence_elements.len(), 1);
    }

    #[test]
    fn finds_availability_nested_inside_aggregate_parameters() {
        let mut arena = TypeArena::default();
        let integer = arena
            .intern(SemanticType::Scalar(ScalarType::Int))
            .expect("test scalar fits the arena");
        let boolean = arena
            .intern(SemanticType::Scalar(ScalarType::Bool))
            .expect("test scalar fits the arena");
        let parameter = arena
            .intern(SemanticType::Staged {
                stage: Stage::Param,
                inner: integer,
            })
            .expect("test parameter fits the arena");
        let shot = arena
            .intern(SemanticType::Staged {
                stage: Stage::Shot,
                inner: boolean,
            })
            .expect("test shot value fits the arena");
        let aggregate = arena
            .intern(SemanticType::Record {
                name: "NestedAvailability".to_owned(),
                fields: Box::new([
                    ("parameter".to_owned(), parameter),
                    ("shot".to_owned(), shot),
                ]),
            })
            .expect("test record fits the arena");

        assert_eq!(arena.stage_of(aggregate), None);
        assert_eq!(arena.latest_availability(aggregate), Some(Stage::Shot));
        assert!(arena.contains_stage(aggregate, Stage::Param));
        assert!(arena.contains_stage(aggregate, Stage::Shot));
        assert!(!arena.contains_stage(aggregate, Stage::Const));
    }

    #[test]
    fn availability_stage_forms_one_ordered_join_lattice() {
        assert!(Stage::Const < Stage::Param);
        assert!(Stage::Param < Stage::Shot);
        assert_eq!(Stage::join(None, Some(Stage::Const)), Some(Stage::Const));
        assert_eq!(
            Stage::join(Some(Stage::Shot), Some(Stage::Param)),
            Some(Stage::Shot)
        );
        assert_eq!(Stage::from_name("Param"), Some(Stage::Param));
        assert_eq!(Stage::Shot.as_str(), "Shot");
        assert_eq!(Stage::from_name("Later"), None);
    }
}
