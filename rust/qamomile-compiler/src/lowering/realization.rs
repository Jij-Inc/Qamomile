//! Closed, transient selection of exact implementations of semantic operations.
//!
//! `QCore` never stores these choices. A target supplies capabilities, and an
//! optional named override selects a checked implementation. Selection is
//! bounded and deterministic; it does not search arbitrary rewrite graphs.

use thiserror::Error;

/// Composite instructions accepted by the next emission boundary.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct RealizationCapabilities(u16);

impl RealizationCapabilities {
    const LABELS: [(&'static str, Self); 12] = [
        ("rzz", Self::RZZ),
        ("named_call", Self::NAMED_CALL),
        ("native_for", Self::NATIVE_FOR),
        ("pauli_evolution", Self::PAULI_EVOLUTION),
        ("add_into", Self::ADD_INTO),
        ("mod_pow", Self::MOD_POW),
        ("mcx", Self::MCX),
        ("controlled_swap", Self::CONTROLLED_SWAP),
        ("coherent_regions", Self::COHERENT_REGIONS),
        ("pauli_measurement", Self::PAULI_MEASUREMENT),
        ("indexed_for", Self::INDEXED_FOR),
        ("boolean_store", Self::BOOLEAN_STORE),
    ];

    /// Project capability names for discovery without a second vocabulary.
    pub fn labels(self) -> impl Iterator<Item = &'static str> {
        Self::LABELS
            .into_iter()
            .filter_map(move |(label, capability)| self.contains(capability).then_some(label))
    }

    /// No optional composite instruction is accepted.
    pub const EMPTY: Self = Self(0);
    /// Exact two-qubit ZZ rotation.
    pub const RZZ: Self = Self(1);
    /// Verified named callable body.
    pub const NAMED_CALL: Self = Self(2);
    /// Index-independent structured loop.
    pub const NATIVE_FOR: Self = Self(4);
    /// Exact commuting Pauli evolution, including coherent control and inverse.
    pub const PAULI_EVOLUTION: Self = Self(8);
    /// Equal-width little-endian addition modulo two to the register width.
    pub const ADD_INTO: Self = Self(16);
    /// Exact modular exponentiation, including control, inverse, and invalid-domain behavior.
    pub const MOD_POW: Self = Self(32);
    /// Exact positive-control X with any bounded number of controls.
    pub const MCX: Self = Self(64);

    /// Exact SWAP with any bounded positive control list.
    pub const CONTROLLED_SWAP: Self = Self(128);

    /// Verified coherent bodies with outer control and inverse modifiers.
    pub const COHERENT_REGIONS: Self = Self(256);
    /// Non-destructive single-axis Pauli measurement with its result bit.
    pub const PAULI_MEASUREMENT: Self = Self(512);
    /// Structured loops with a scoped scalar index and invariant wire layout.
    pub const INDEXED_FOR: Self = Self(1024);
    /// Verified top-level Boolean assignment to private classical storage.
    pub const BOOLEAN_STORE: Self = Self(2048);

    /// Return whether every capability in `required` is available.
    #[must_use]
    pub const fn contains(self, required: Self) -> bool {
        self.0 & required.0 == required.0
    }

    /// Return whether named calls may remain abstract.
    #[must_use]
    pub const fn supports_named_call(self) -> bool {
        self.contains(Self::NAMED_CALL)
    }

    /// Return whether eligible loops may remain structured.
    #[must_use]
    pub const fn supports_native_for(self) -> bool {
        self.contains(Self::NATIVE_FOR)
    }

    /// Return whether a two-body ZZ term may remain one rotation.
    #[must_use]
    pub const fn supports_rzz(self) -> bool {
        self.contains(Self::RZZ)
    }

    /// Decode the closed capability vocabulary before lowering any operation.
    ///
    /// # Errors
    /// Returns a stable error for an unknown or repeated label.
    pub fn from_labels(labels: &[String]) -> Result<Self, RealizationError> {
        let mut result = Self::EMPTY;
        for label in labels {
            let capability = Self::LABELS
                .iter()
                .find_map(|(name, capability)| (*name == label).then_some(*capability))
                .ok_or_else(|| RealizationError::UnknownCapability(label.clone()))?;
            if result.contains(capability) {
                return Err(RealizationError::DuplicateCapability(label.clone()));
            }
            result.0 |= capability.0;
        }
        Ok(result)
    }
}

/// Semantic families with independently useful implementation alternatives.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RealizationFamily {
    /// Exact commuting Hamiltonian evolution.
    PauliEvolution,
    /// Retained addend and wrapping in-place target register.
    AddInto,
    /// Total multiplication by a classical unit modulo a classical modulus.
    ModMul,
    /// Retained-register modular multiply-accumulate.
    ModMulAdd,
    /// Coherent exponentiation of a classical modular unit.
    ModPow,
    /// Coherent product of independently exponentiated modular units.
    ModMultiPow,
    /// Exact reversible unsigned arithmetic with a bounded permutation fallback.
    Arithmetic,
}

impl RealizationFamily {
    /// Return the semantic operation implemented by this family.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PauliEvolution => "quantum.pauli_evolution",
            Self::AddInto => "quantum.add_into",
            Self::ModMul => "quantum.mod_mul",
            Self::ModMulAdd => "quantum.mod_mul_add",
            Self::ModPow => "quantum.mod_pow",
            Self::ModMultiPow => "quantum.mod_multi_pow",
            Self::Arithmetic => "quantum.arithmetic",
        }
    }
}

/// One compiler-owned definition shared by validation, discovery, and docs.
#[derive(Clone, Copy, Debug)]
pub struct RealizationInfo {
    /// User-selectable implementation name, not a recipe version.
    pub name: &'static str,
    /// Semantic operation family.
    pub family: RealizationFamily,
    /// Implementation strategy and relevant tradeoff.
    pub description: &'static str,
    /// Composite instructions required from the next consumer.
    pub required_capabilities: RealizationCapabilities,
}

/// Largest accepted coset padding carried by one configured realization.
pub const MAX_COSET_PADDING_BITS: u16 = 4096;

/// Checked positive number of padding bits used by an encoded realization.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CosetPaddingBits(u16);

impl CosetPaddingBits {
    /// Construct one bounded positive padding value.
    ///
    /// # Errors
    /// Returns [`RealizationError::InvalidParameters`] outside the supported range.
    pub fn new(value: u16) -> Result<Self, RealizationError> {
        if value == 0 || value > MAX_COSET_PADDING_BITS {
            return Err(RealizationError::InvalidParameters(
                "coset padding_bits must be between 1 and 4096".to_owned(),
            ));
        }
        Ok(Self(value))
    }

    /// Return the checked padding count.
    #[must_use]
    pub const fn get(self) -> u16 {
        self.0
    }
}

/// Named implementations, deliberately independent of SDK names and releases.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Realization {
    /// Preserve exact commuting evolution at the next boundary.
    PauliNative,
    /// Use RZZ for two-body ZZ terms and parity networks otherwise.
    PauliRzz,
    /// Use basis changes and a CNOT parity network for every term.
    PauliParity,
    /// Preserve wrapping register addition at the next boundary.
    AddNative,
    /// Exact Fourier addition without truncating any controlled rotation.
    AddQft,
    /// Ancilla-free linear ripple-carry addition with relative-phase Toffolis.
    AddRipple,
    /// Schoolbook accumulation with clean ripple-carry arithmetic.
    ModMulSchoolbookRipple,
    /// Schoolbook retained-register product and modular accumulation.
    ModMulAddSchoolbookRipple,
    /// Preserve exact modular exponentiation at the next boundary.
    ModPowNative,
    /// One controlled schoolbook multiplication per exponent bit.
    ModPowSquareAndMultiply,
    /// Window the exponent and source register around exact table lookups.
    ModPowWindowedLookup,
    /// Use windowed lookup arithmetic over an approximate coset encoding.
    ModPowCosetWindowed,
    /// Apply one modular exponentiation independently for every coordinate.
    ModMultiPowIndependent,
    /// Share repeated squaring through one balanced selected-base product tree.
    ModMultiPowProductTree,
    /// Synthesize exact bounded arithmetic with reversible gate networks.
    ArithmeticPortable,
}

impl Realization {
    /// Closed list used by option validation and user-facing discovery.
    pub const ALL: [Self; 15] = [
        Self::PauliNative,
        Self::PauliRzz,
        Self::PauliParity,
        Self::AddNative,
        Self::AddQft,
        Self::AddRipple,
        Self::ModMulSchoolbookRipple,
        Self::ModMulAddSchoolbookRipple,
        Self::ModPowNative,
        Self::ModPowSquareAndMultiply,
        Self::ModPowWindowedLookup,
        Self::ModPowCosetWindowed,
        Self::ModMultiPowIndependent,
        Self::ModMultiPowProductTree,
        Self::ArithmeticPortable,
    ];

    /// Return the single authoritative definition of this implementation.
    #[must_use]
    #[expect(
        clippy::too_many_lines,
        reason = "Keep the closed realization metadata table exhaustive in one match"
    )]
    pub const fn info(self) -> RealizationInfo {
        use RealizationCapabilities as Caps;
        use RealizationFamily::{
            AddInto, Arithmetic, ModMul, ModMulAdd, ModMultiPow, ModPow, PauliEvolution,
        };
        let (name, family, description, required_capabilities) = match self {
            Self::PauliNative => (
                "pauli.native",
                PauliEvolution,
                "Delegate exact commuting evolution, including its constant-term phase, to the target.",
                Caps::PAULI_EVOLUTION,
            ),
            Self::PauliRzz => (
                "pauli.rzz",
                PauliEvolution,
                "Use target RZZ rotations for two-body ZZ terms and parity networks for other terms.",
                Caps::RZZ,
            ),
            Self::PauliParity => (
                "pauli.parity",
                PauliEvolution,
                "Use exact basis changes and CNOT parity networks; no composite target capability is required.",
                Caps::EMPTY,
            ),
            Self::AddNative => (
                "add.native",
                AddInto,
                "Delegate exact wrapping register addition to the target; synthesis depends on the target SDK.",
                Caps::ADD_INTO,
            ),
            Self::AddQft => (
                "add.qft",
                AddInto,
                "Use ancilla-free exact Fourier addition: O(n^2) steps, with no rotation truncation.",
                Caps::EMPTY,
            ),
            Self::AddRipple => (
                "add.ripple",
                AddInto,
                "Use ancilla-free ripple-carry addition with relative-phase Toffolis: O(n) steps and 16n-13 CNOTs.",
                Caps::EMPTY,
            ),
            Self::ModMulSchoolbookRipple => (
                "mod_mul.schoolbook_ripple",
                ModMul,
                "Use guarded schoolbook accumulation, clean ripple-carry modular addition, swap, and inverse uncomputation.",
                Caps::EMPTY,
            ),
            Self::ModMulAddSchoolbookRipple => (
                "mod_mul_add.schoolbook_ripple",
                ModMulAdd,
                "Compute a retained-register product, accumulate its modular residues with clean ripple carry, and uncompute the product.",
                Caps::EMPTY,
            ),
            Self::ModPowNative => (
                "mod_pow.native",
                ModPow,
                "Delegate exact modular exponentiation, including coherent control, inverse, invalid-domain identity, and clean scratch, to the target.",
                Caps::MOD_POW,
            ),
            Self::ModPowSquareAndMultiply => (
                "mod_pow.square_and_multiply",
                ModPow,
                "Apply exact controlled modular multiplications, using bounded small permutations or schoolbook ripple.",
                Caps::EMPTY,
            ),
            Self::ModPowWindowedLookup => (
                "mod_pow.windowed_lookup",
                ModPow,
                "Use exact bounded exponent windows and precomputed modular-factor lookup tables.",
                Caps::EMPTY,
            ),
            Self::ModPowCosetWindowed => (
                "mod_pow.coset_windowed",
                ModPow,
                "Use bounded exponent windows over a coset encoding with a certified output trace-distance bound; padding_bits is required.",
                Caps::EMPTY,
            ),
            Self::ModMultiPowIndependent => (
                "mod_multi_pow.independent",
                ModMultiPow,
                "Apply one selected modular-power realization per coordinate against the shared target.",
                Caps::EMPTY,
            ),
            Self::ModMultiPowProductTree => (
                "mod_multi_pow.product_tree",
                ModMultiPow,
                "Use repeated squaring and balanced selected-base products before modular accumulation.",
                Caps::EMPTY,
            ),
            Self::ArithmeticPortable => (
                "arithmetic.portable",
                Arithmetic,
                "Use exact bounded reversible gates: controlled increments and shift-add for ordinary arithmetic, permutation cycles for modular operations and comparison.",
                Caps::EMPTY,
            ),
        };
        RealizationInfo {
            name,
            family,
            description,
            required_capabilities,
        }
    }

    /// Return the user-selectable implementation name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        self.info().name
    }

    /// Return the denotation that this realization implements.
    #[must_use]
    pub const fn family(self) -> RealizationFamily {
        self.info().family
    }

    const fn required(self) -> RealizationCapabilities {
        self.info().required_capabilities
    }
}

/// Closed parameters of one selected implementation.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
enum RealizationParameters {
    Exact,
    Coset { padding_bits: CosetPaddingBits },
}

/// One fully checked realization selection used by every compiler consumer.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ConfiguredRealization {
    kind: Realization,
    parameters: RealizationParameters,
}

impl ConfiguredRealization {
    /// Construct an exact parameter-free selection.
    #[must_use]
    pub const fn exact(kind: Realization) -> Self {
        Self {
            kind,
            parameters: RealizationParameters::Exact,
        }
    }

    /// Configure one catalog choice from a canonical named parameter row.
    ///
    /// Parameter values cross language boundaries as text and are parsed by
    /// the implementation that owns their meaning. This keeps the boundary
    /// stable when another realization introduces a different parameter set.
    ///
    /// # Errors
    /// Rejects unordered, duplicate, missing, unknown, or invalid parameters.
    pub fn configure(
        kind: Realization,
        parameters: &[(String, String)],
    ) -> Result<Self, RealizationError> {
        if parameters
            .windows(2)
            .any(|pair| pair[0].0.as_str() >= pair[1].0.as_str())
        {
            return Err(RealizationError::InvalidParameters(
                "realization parameters must be unique and name-ordered".to_owned(),
            ));
        }
        match (kind, parameters) {
            (Realization::ModPowCosetWindowed, [(name, value)]) if name == "padding_bits" => {
                let padding_bits = value.parse::<u16>().map_err(|_| {
                    RealizationError::InvalidParameters(
                        "coset padding_bits must be between 1 and 4096".to_owned(),
                    )
                })?;
                Self::coset(kind, padding_bits)
            }
            (Realization::ModPowCosetWindowed, _) => Err(RealizationError::InvalidParameters(
                "mod_pow.coset_windowed requires exactly one padding_bits parameter".to_owned(),
            )),
            (_, []) => Ok(Self::exact(kind)),
            _ => Err(RealizationError::InvalidParameters(format!(
                "realization {:?} does not accept parameters",
                kind.as_str()
            ))),
        }
    }

    /// Construct the coset realization with checked padding.
    ///
    /// # Errors
    /// Rejects another realization kind or an invalid padding count.
    pub fn coset(kind: Realization, padding_bits: u16) -> Result<Self, RealizationError> {
        if kind != Realization::ModPowCosetWindowed {
            return Err(RealizationError::InvalidParameters(format!(
                "realization {:?} does not accept padding_bits",
                kind.as_str()
            )));
        }
        Ok(Self {
            kind,
            parameters: RealizationParameters::Coset {
                padding_bits: CosetPaddingBits::new(padding_bits)?,
            },
        })
    }

    /// Decode one canonical boundary label into a typed selection.
    ///
    /// # Errors
    /// Rejects unknown names, missing parameters, and noncanonical parameter rows.
    pub fn parse(label: &str) -> Result<Self, RealizationError> {
        const PREFIX: &str = "mod_pow.coset_windowed[padding_bits=";
        if let Some(value) = label
            .strip_prefix(PREFIX)
            .and_then(|value| value.strip_suffix(']'))
        {
            let padding_bits = value
                .parse::<u16>()
                .map_err(|_| RealizationError::InvalidParameters(label.to_owned()))?;
            return Self::coset(Realization::ModPowCosetWindowed, padding_bits);
        }
        let kind = Realization::ALL
            .into_iter()
            .find(|candidate| candidate.as_str() == label)
            .ok_or_else(|| RealizationError::UnknownRealization(label.to_owned()))?;
        if kind == Realization::ModPowCosetWindowed {
            return Err(RealizationError::InvalidParameters(
                "mod_pow.coset_windowed requires padding_bits".to_owned(),
            ));
        }
        Ok(Self::exact(kind))
    }

    /// Return the catalog identity independently of configuration.
    #[must_use]
    pub const fn kind(self) -> Realization {
        self.kind
    }

    /// Return coset padding only for the encoded realization.
    #[must_use]
    pub const fn coset_padding_bits(self) -> Option<CosetPaddingBits> {
        match self.parameters {
            RealizationParameters::Exact => None,
            RealizationParameters::Coset { padding_bits } => Some(padding_bits),
        }
    }

    /// Return the canonical boundary label retained by fingerprints.
    #[must_use]
    pub fn canonical_label(self) -> String {
        match self.parameters {
            RealizationParameters::Exact => self.kind.as_str().to_owned(),
            RealizationParameters::Coset { padding_bits } => format!(
                "{}[padding_bits={}]",
                self.kind.as_str(),
                padding_bits.get()
            ),
        }
    }
}

/// Immutable target input; not a persistent semantic plan or registry.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct LoweringConfig {
    /// Instructions accepted at this emission boundary.
    pub capabilities: RealizationCapabilities,
    pauli: Option<ConfiguredRealization>,
    addition: Option<ConfiguredRealization>,
    arithmetic: Option<ConfiguredRealization>,
    mod_mul: Option<ConfiguredRealization>,
    mod_mul_add: Option<ConfiguredRealization>,
    mod_pow: Option<ConfiguredRealization>,
    mod_multi_pow: Option<ConfiguredRealization>,
}

impl LoweringConfig {
    /// Portable, deterministic defaults with no optional target instructions.
    pub const EMPTY: Self = Self {
        capabilities: RealizationCapabilities::EMPTY,
        pauli: None,
        addition: None,
        arithmetic: None,
        mod_mul: None,
        mod_mul_add: None,
        mod_pow: None,
        mod_multi_pow: None,
    };

    /// Whether no explicit addition realization was requested.
    pub(crate) const fn addition_is_default(self) -> bool {
        self.addition.is_none()
    }

    /// Decode at most one override per semantic family.
    ///
    /// # Errors
    /// Rejects unknown identities, duplicate families and unsupported requests.
    pub fn from_labels(
        capabilities: &[String],
        overrides: &[String],
    ) -> Result<Self, RealizationError> {
        let mut config = Self {
            capabilities: RealizationCapabilities::from_labels(capabilities)?,
            ..Self::EMPTY
        };
        if overrides.len() > 7 {
            return Err(RealizationError::OverrideBudget);
        }
        for label in overrides {
            let realization = ConfiguredRealization::parse(label)?;
            let slot = match realization.kind().family() {
                RealizationFamily::PauliEvolution => &mut config.pauli,
                RealizationFamily::AddInto => &mut config.addition,
                RealizationFamily::ModMul => &mut config.mod_mul,
                RealizationFamily::ModMulAdd => &mut config.mod_mul_add,
                RealizationFamily::ModPow => &mut config.mod_pow,
                RealizationFamily::ModMultiPow => &mut config.mod_multi_pow,
                RealizationFamily::Arithmetic => &mut config.arithmetic,
            };
            if slot.replace(realization).is_some() {
                return Err(RealizationError::ConflictingOverride(label.clone()));
            }
            if !config.capabilities.contains(realization.kind().required()) {
                return Err(RealizationError::Unavailable(label.clone()));
            }
        }
        Ok(config)
    }

    /// Return canonical override identities for fingerprints, never auto choices.
    #[must_use]
    pub fn overrides(self) -> Vec<String> {
        let mut result = [
            self.pauli,
            self.addition,
            self.mod_mul,
            self.mod_mul_add,
            self.mod_pow,
            self.mod_multi_pow,
            self.arithmetic,
        ]
        .into_iter()
        .flatten()
        .map(ConfiguredRealization::canonical_label)
        .collect::<Vec<_>>();
        result.sort_unstable();
        result
    }

    /// Select a generated modular multiplication without expanding its parent.
    ///
    /// Explicit multiplication choices apply to children as well. Automatic
    /// permutation synthesis is restricted to three target bits, independently
    /// of the exponent and surrounding controls. Standalone defaults are unchanged.
    #[must_use]
    pub const fn select_mod_mul_child(self, target_width: u64) -> ConfiguredRealization {
        match self.mod_mul {
            Some(choice) => choice,
            None if target_width <= 3 => {
                ConfiguredRealization::exact(Realization::ArithmeticPortable)
            }
            None => ConfiguredRealization::exact(Realization::ModMulSchoolbookRipple),
        }
    }

    /// Select an exact implementation without mutating the semantic program.
    #[must_use]
    pub const fn select(self, family: RealizationFamily) -> ConfiguredRealization {
        match family {
            RealizationFamily::PauliEvolution => match self.pauli {
                Some(choice) => choice,
                None if self
                    .capabilities
                    .contains(RealizationCapabilities::PAULI_EVOLUTION) =>
                {
                    ConfiguredRealization::exact(Realization::PauliNative)
                }
                None if self.capabilities.supports_rzz() => {
                    ConfiguredRealization::exact(Realization::PauliRzz)
                }
                None => ConfiguredRealization::exact(Realization::PauliParity),
            },
            RealizationFamily::AddInto => match self.addition {
                Some(choice) => choice,
                None if self
                    .capabilities
                    .contains(RealizationCapabilities::ADD_INTO) =>
                {
                    ConfiguredRealization::exact(Realization::AddNative)
                }
                // Width-dependent automatic choice; see `plan_addition`.
                None => ConfiguredRealization::exact(Realization::AddRipple),
            },
            RealizationFamily::ModMul => match self.mod_mul {
                Some(choice) => choice,
                None => ConfiguredRealization::exact(Realization::ArithmeticPortable),
            },
            RealizationFamily::ModMulAdd => match self.mod_mul_add {
                Some(choice) => choice,
                None => ConfiguredRealization::exact(Realization::ModMulAddSchoolbookRipple),
            },
            RealizationFamily::ModPow => match self.mod_pow {
                Some(choice) => choice,
                None if self.capabilities.contains(RealizationCapabilities::MOD_POW) => {
                    ConfiguredRealization::exact(Realization::ModPowNative)
                }
                None => ConfiguredRealization::exact(Realization::ModPowSquareAndMultiply),
            },
            RealizationFamily::ModMultiPow => match self.mod_multi_pow {
                Some(choice) => choice,
                None => ConfiguredRealization::exact(Realization::ModMultiPowIndependent),
            },
            RealizationFamily::Arithmetic => {
                ConfiguredRealization::exact(Realization::ArithmeticPortable)
            }
        }
    }
}

/// Stable errors shared by all realization consumers.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum RealizationError {
    /// An emission capability is unknown.
    #[error("Q8041: unknown realization capability {0:?}")]
    UnknownCapability(String),
    /// An emission capability is repeated.
    #[error("Q8042: duplicate realization capability {0:?}")]
    DuplicateCapability(String),
    /// A named implementation is unknown.
    #[error("Q8043: unknown realization {0:?}")]
    UnknownRealization(String),
    /// More than one implementation was requested for a semantic family.
    #[error("Q8044: conflicting realization override {0:?}")]
    ConflictingOverride(String),
    /// The requested implementation cannot be preserved by this target.
    #[error("Q8045: realization {0:?} is not supported by this target")]
    Unavailable(String),
    /// A request exceeds the closed family count.
    #[error("Q8046: at most one realization override per supported family is allowed")]
    OverrideBudget,
    /// Arithmetic width is inconsistent with the semantic contract.
    #[error("Q8047: add_into requires a positive even coherent width")]
    AdditionWidth,
    /// Generating a schedule would exceed bounded time or memory.
    #[error("Q8048: arithmetic realization exceeds its instruction or selector budget")]
    ScheduleBudget,
    /// Arithmetic parameters do not define an exact total permutation.
    #[error("Q8050: arithmetic parameters do not define a reversible permutation")]
    ArithmeticParameters,
    /// The generic arithmetic realization only supports a bounded width.
    #[error("Q8051: arithmetic width exceeds the exact portable realization limit")]
    ArithmeticWidth,
    /// A parameterized realization is missing or contains an invalid value.
    #[error("Q8052: invalid realization parameters: {0}")]
    InvalidParameters(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_preserve_only_advertised_composites() {
        assert_eq!(
            LoweringConfig::EMPTY
                .select(RealizationFamily::AddInto)
                .kind(),
            Realization::AddRipple
        );
        assert_eq!(
            LoweringConfig::EMPTY
                .select(RealizationFamily::PauliEvolution)
                .kind(),
            Realization::PauliParity
        );
        let config = LoweringConfig::from_labels(&["add_into".into(), "rzz".into()], &[]).unwrap();
        assert_eq!(
            config.select(RealizationFamily::AddInto).kind(),
            Realization::AddNative
        );
        assert_eq!(
            config.select(RealizationFamily::PauliEvolution).kind(),
            Realization::PauliRzz
        );
    }

    #[test]
    fn overrides_are_canonical_and_fail_closed() {
        let config =
            LoweringConfig::from_labels(&[], &["pauli.parity".into(), "add.qft".into()]).unwrap();
        assert_eq!(config.overrides(), ["add.qft", "pauli.parity"]);
        assert_eq!(
            config.select(RealizationFamily::AddInto).kind(),
            Realization::AddQft
        );
        assert!(matches!(
            LoweringConfig::from_labels(&[], &["add.native".into()]),
            Err(RealizationError::Unavailable(_))
        ));
        assert!(matches!(
            LoweringConfig::from_labels(&[], &["add.qft".into(), "add.ripple".into()]),
            Err(RealizationError::ConflictingOverride(_))
        ));
    }

    #[test]
    fn discovery_definitions_are_unique_and_drive_validation() {
        let mut names = std::collections::BTreeSet::new();
        for choice in Realization::ALL {
            let info = choice.info();
            assert!(names.insert(info.name));
            assert!(!info.description.is_empty());
            let labels = info
                .required_capabilities
                .labels()
                .map(str::to_owned)
                .collect::<Vec<_>>();
            let selection = if choice == Realization::ModPowCosetWindowed {
                "mod_pow.coset_windowed[padding_bits=16]".to_owned()
            } else {
                info.name.to_owned()
            };
            let config =
                LoweringConfig::from_labels(&labels, std::slice::from_ref(&selection)).unwrap();
            assert_eq!(config.select(info.family).kind(), choice);
            assert_eq!(
                RealizationCapabilities::from_labels(&labels).unwrap(),
                info.required_capabilities
            );
            for missing in &labels {
                let incomplete = labels
                    .iter()
                    .filter(|label| *label != missing)
                    .cloned()
                    .collect::<Vec<_>>();
                assert!(matches!(
                    LoweringConfig::from_labels(&incomplete, std::slice::from_ref(&selection)),
                    Err(RealizationError::Unavailable(_))
                ));
            }
        }
    }

    #[test]
    fn configured_parameters_are_generic_at_the_boundary_and_typed_by_the_owner() {
        let configured = ConfiguredRealization::configure(
            Realization::ModPowCosetWindowed,
            &[("padding_bits".to_owned(), "40".to_owned())],
        )
        .expect("coset parameters");
        assert_eq!(
            configured.canonical_label(),
            "mod_pow.coset_windowed[padding_bits=40]"
        );
        assert!(matches!(
            ConfiguredRealization::configure(
                Realization::ModPowCosetWindowed,
                &[("precision".to_owned(), "40".to_owned())],
            ),
            Err(RealizationError::InvalidParameters(_))
        ));
        assert_eq!(
            ConfiguredRealization::configure(Realization::AddQft, &[])
                .expect("parameter-free selection")
                .canonical_label(),
            "add.qft"
        );
    }
}
