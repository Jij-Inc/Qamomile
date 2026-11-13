"""Immutable compiler pass composition over the required native QCore boundary.

This module composes transformations; Rust owns graph edits, verification, ABI
association, and portable preparation receipts. It is unrelated to mathematical
optimization models in ``qamomile.optimization``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, runtime_checkable

from ._native import compiler as native
from ._target_profile import TargetProfile
from .errors import ArtifactError
from .ir import Module, Program

OPTIMIZATION_METADATA_KEY = "qamomile.optimization.v1"


def _require_optimization_api() -> Any:
    """Require the complete selected-preparation native protocol.

    Returns:
        Any: Compatible native verified-program constructor.

    Raises:
        ArtifactError: If the installed native distribution lacks this boundary.
    """
    constructor = getattr(native, "VerifiedProgramHandle", None)
    members = (
        "validate_pipeline",
        "with_optimization",
        "validate_result",
        "restore_optimization",
        "optimization_receipt",
        "optimization_stages",
        "compilation_source_digest",
        "simplification_configuration",
        "simplify",
        "rewrite",
        "eliminate_common_expressions",
        "preserve_optimization",
    )
    if not isinstance(constructor, type) or any(
        not callable(getattr(constructor, name, None)) for name in members
    ):
        raise ArtifactError(
            "Q7802: installed native compiler lacks optimization pipeline support"
        )
    return constructor


@dataclass(frozen=True, slots=True)
class OptimizationContext:
    """Portable constraints available to an optimization pass.

    Args:
        target_fingerprint (str): Immutable target policy identity.
        capabilities (frozenset[str]): Supported target requirement labels.
        seed (int): Unsigned deterministic optimization seed.
        max_work (int): Maximum native rewrite operation visits per invocation.
        profile (TargetProfile | None): Native-validated device facts the target
            supplied, or None when it has none. The profile digest is already
            part of ``target_fingerprint`` whenever a profile exists.
    """

    target_fingerprint: str
    capabilities: frozenset[str]
    seed: int = 0
    max_work: int = 1_000_000
    profile: TargetProfile | None = None


@dataclass(frozen=True, slots=True)
class OptimizationProgram:
    """Retain a verified native program between extension passes.

    Use ``from_bytes`` for independently distributed extension results. Reuse
    ``simplify`` and ``rewrite`` for local edits without encoding QCore or SSA.

    Args:
        _handle (Any): Native verified owner, supplied by the pipeline.
    """

    _handle: Any = field(repr=False)

    def __post_init__(self) -> None:
        """Require the native verified typestate at facade construction.

        Raises:
            TypeError: If the supplied owner is not a native verified program.
            ArtifactError: If native optimization support is incompatible.
        """
        if not isinstance(self._handle, _require_optimization_api()):
            raise TypeError("OptimizationProgram requires a native verified program")

    @classmethod
    def from_bytes(cls, payload: bytes) -> OptimizationProgram:
        """Verify an external pass result through Rust.

        Args:
            payload (bytes): Canonical QCore image returned by an extension.

        Returns:
            OptimizationProgram: Fully checked native program.

        Raises:
            ArtifactError: If native graph or cleanup verification fails.
        """
        try:
            return cls(_require_optimization_api()(payload))
        except (TypeError, ValueError) as error:
            raise ArtifactError(
                f"Q7802: invalid optimization result: {error}"
            ) from error

    def to_bytes(self) -> bytes:
        """Export canonical QCore for an external native module or service.

        Returns:
            bytes: Verified canonical program without live callbacks.
        """
        return self._handle.to_bytes()

    def eliminate_common_expressions(
        self, *, context: OptimizationContext
    ) -> OptimizationProgram:
        """Share dominated classical computations without evaluating parameters.

        Args:
            context (OptimizationContext): Native traversal work limit.

        Returns:
            OptimizationProgram: Reverified candidate with source rounding intact.

        Raises:
            ArtifactError: If rewriting exceeds limits or verification fails.
        """
        try:
            return OptimizationProgram(
                self._handle.eliminate_common_expressions(context.max_work)
            )
        except ValueError as error:
            raise ArtifactError(str(error)) from error

    def simplify(
        self,
        *,
        context: OptimizationContext | None = None,
        options: StandardSimplification | None = None,
    ) -> OptimizationProgram:
        """Reuse the native standard simplifier without target lowering.

        Args:
            context (OptimizationContext | None): Native operation-visit budget;
                None uses one million visits.
            options (StandardSimplification | None): Reusable built-in rewrite
                settings; None enables every standard simplification.

        Returns:
            OptimizationProgram: Completely reverified simplified program.

        Raises:
            ArtifactError: If bounded simplification or verification fails.
        """
        try:
            return OptimizationProgram(
                self._handle.simplify(
                    1_000_000 if context is None else context.max_work,
                    None if options is None else options._flags(),
                )
            )
        except ValueError as error:
            raise ArtifactError(str(error)) from error

    def rewrite(
        self,
        pattern: Sequence[str],
        replacement: Sequence[str],
        *,
        context: OptimizationContext,
    ) -> OptimizationProgram:
        """Rewrite unary gate sequences with native traversal and SSA checks.

        Rules use canonical opcodes such as ``quantum.h``. They contract a
        nonempty sequence of at most sixteen angle-free unary gates into a
        nonempty sequence on the same selectors. The author asserts equivalence,
        including global phase on the defined execution domain. This initial driver
        matches consecutive operations; it does not move gates across constants.

        Args:
            pattern (Sequence[str]): Consecutive source gate opcodes.
            replacement (Sequence[str]): Non-expanding replacement gate opcodes.
            context (OptimizationContext): Target policy and native work limit.

        Returns:
            OptimizationProgram: New verified program; this input is unchanged.

        Raises:
            ArtifactError: If the rule, budget, ownership, or result is invalid.
        """
        try:
            return OptimizationProgram(
                self._handle.rewrite(list(pattern), list(replacement), context.max_work)
            )
        except (TypeError, ValueError) as error:
            raise ArtifactError(str(error)) from error


@runtime_checkable
class OptimizationPass(Protocol):
    """Implement one QCore transformation without implementing the compiler.

    Attributes:
        name (str): Unique namespaced pass identifier.
        version (str): Stable implementation version.
        configuration (bytes): Deterministic opaque configuration identity.
    """

    @property
    def name(self) -> str:
        """Return the stable namespaced identity.

        Returns:
            str: Public pass identifier.
        """
        ...

    @property
    def version(self) -> str:
        """Return the immutable implementation version.

        Returns:
            str: Version used in portable provenance.
        """
        ...

    @property
    def configuration(self) -> bytes:
        """Return the deterministic configuration identity.

        Returns:
            bytes: Portable opaque options without live callbacks.
        """
        ...

    def run(
        self, program: OptimizationProgram, context: OptimizationContext
    ) -> OptimizationProgram:
        """Transform one checked program and preserve its observable semantics.

        Args:
            program (OptimizationProgram): Verified input retained by Rust.
            context (OptimizationContext): Portable target and work constraints.

        Returns:
            OptimizationProgram: Verified candidate with the same public ABI.
        """
        ...


@dataclass(frozen=True, slots=True)
class StandardSimplification:
    """Configure the OSS simplifier as the public ``qcore.simplify`` pass.

    Args:
        identities (bool): Eliminate identity operations.
        inverses (bool): Cancel inverse gates and callables on their defined
            execution domain, without requiring a totality or termination proof.
            Known source-domain failures are diagnosed before rewriting.
        structure (bool): Simplify tuple, unpack, and reshape round trips.
        vacuous (bool): Remove immediately released quantum roots.
        dead_constants (bool): Remove dead total constants and availability lifts.
    """

    identities: bool = True
    inverses: bool = True
    structure: bool = True
    vacuous: bool = True
    dead_constants: bool = True
    name: str = field(default="qcore.simplify", init=False)
    version: str = field(default="1", init=False)

    def __post_init__(self) -> None:
        """Validate the fixed boolean option projection before composing passes.

        Raises:
            TypeError: If an option is not a bool.
        """
        if any(type(flag) is not bool for flag in self._flags()):
            raise TypeError("simplification options must be bools")

    def _flags(self) -> list[bool]:
        """Project the fixed native-owned option order.

        Returns:
            list[bool]: Native rewrite switches in their fixed ABI order.
        """
        return [
            self.identities,
            self.inverses,
            self.structure,
            self.vacuous,
            self.dead_constants,
        ]

    @property
    def configuration(self) -> bytes:
        """Return the native canonical option identity.

        Returns:
            bytes: Deterministic built-in simplification configuration.
        """
        return native.VerifiedProgramHandle.simplification_configuration(self._flags())

    def run(
        self, program: OptimizationProgram, context: OptimizationContext
    ) -> OptimizationProgram:
        """Apply standard exact QCore simplification.

        Args:
            program (OptimizationProgram): Verified source for simplification.
            context (OptimizationContext): Portable policy for this invocation.

        Returns:
            OptimizationProgram: Simplified, fully reverified result.

        Raises:
            ArtifactError: If native simplification fails.
        """
        return program.simplify(context=context, options=self)


@dataclass(frozen=True, slots=True)
class OptimizationFamily:
    """Retain a native symbolic module between passes without serialization.

    Args:
        _handle (Any): Verified native module supplied by the driver.
        entrypoint (str): Selected callable in that module.
    """

    _handle: Any = field(repr=False)
    entrypoint: str
    _module: Module | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Check the native typestate and selected entry before a pass runs.

        Raises:
            ArtifactError: If the native boundary is absent or incompatible.
            TypeError: If the supplied module is not natively verified.
        """
        constructor = getattr(native, "VerifiedModuleHandle", None)
        if not isinstance(constructor, type):
            raise ArtifactError("Q7802: native symbolic optimization is unavailable")
        if not isinstance(self._handle, constructor):
            raise TypeError("OptimizationFamily requires a native verified module")
        object.__setattr__(
            self, "entrypoint", self._handle.resolve_entrypoint(self.entrypoint)
        )

    @classmethod
    def from_bytes(cls, payload: bytes, entrypoint: str) -> OptimizationFamily:
        """Validate an independent provider's symbolic result.

        Args:
            payload (bytes): Canonical native module image.
            entrypoint (str): Selected callable.

        Returns:
            OptimizationFamily: Checked symbolic input.

        Raises:
            ArtifactError: If native verification fails.
        """
        constructor = getattr(native, "VerifiedModuleHandle", None)
        if not isinstance(constructor, type):
            raise ArtifactError("Q7802: native symbolic optimization is unavailable")
        try:
            return cls(constructor(payload), entrypoint)
        except ValueError as error:
            raise ArtifactError(str(error)) from error

    def to_bytes(self) -> bytes:
        """Export canonical module bytes for an independent provider.

        Returns:
            bytes: Portable verified symbolic module.
        """
        return self._handle.to_bytes()

    @property
    def module(self) -> Module:
        """Project a Python module lazily for inspection or specialization.

        Returns:
            Module: Immutable verified source facade.
        """
        if self._module is None:
            object.__setattr__(self, "_module", Module.from_bytes(self.to_bytes()))
        assert self._module is not None
        return self._module

    def eliminate_common_expressions(
        self, *, context: OptimizationContext
    ) -> OptimizationFamily:
        """Reuse native CSE without specializing or enumerating symbolic loops.

        Args:
            context (OptimizationContext): Native traversal work limit.

        Returns:
            OptimizationFamily: Retained verified symbolic result.

        Raises:
            ArtifactError: If bounded native rewriting fails.
        """
        try:
            return OptimizationFamily(
                self._handle.eliminate_common_expressions(context.max_work),
                self.entrypoint,
            )
        except ValueError as error:
            raise ArtifactError(str(error)) from error


@runtime_checkable
class FamilyOptimizationPass(Protocol):
    """Transform a symbolic family while preserving every admitted binding.

    Attributes:
        name (str): Namespaced pass identity.
        version (str): Implementation version.
        configuration (bytes): Portable deterministic configuration.
    """

    @property
    def name(self) -> str:
        """Return the stable pass identifier."""
        ...

    @property
    def version(self) -> str:
        """Return the implementation version."""
        ...

    @property
    def configuration(self) -> bytes:
        """Return the portable configuration identity."""
        ...

    def run(
        self, program: OptimizationFamily, context: OptimizationContext
    ) -> OptimizationFamily:
        """Transform an unbound family without changing its public interface.

        Args:
            program (OptimizationFamily): Verified symbolic source.
            context (OptimizationContext): Target and work policy.

        Returns:
            OptimizationFamily: Verified candidate; equivalence is the author's obligation.
        """
        ...


@dataclass(frozen=True, slots=True)
class CommonExpressions:
    """Remove dominated duplicate classical expressions in specialized QCore."""

    name: str = field(default="qcore.common-expressions", init=False)
    version: str = field(default="1", init=False)
    configuration: bytes = field(default=b"", init=False)

    def run(
        self, program: OptimizationProgram, context: OptimizationContext
    ) -> OptimizationProgram:
        """Apply shared native CSE to a closed program.

        Args:
            program (OptimizationProgram): Verified input.
            context (OptimizationContext): Native work policy.

        Returns:
            OptimizationProgram: Verified result.
        """
        return program.eliminate_common_expressions(context=context)


@dataclass(frozen=True, slots=True)
class FamilyCommonExpressions:
    """Remove duplicate classical expressions while Const values remain symbolic."""

    name: str = field(default="family.common-expressions", init=False)
    version: str = field(default="1", init=False)
    configuration: bytes = field(default=b"", init=False)

    def run(
        self, program: OptimizationFamily, context: OptimizationContext
    ) -> OptimizationFamily:
        """Apply the shared native CSE driver before specialization.

        Args:
            program (OptimizationFamily): Unbound verified module and entry.
            context (OptimizationContext): Native work policy.

        Returns:
            OptimizationFamily: Verified symbolic result.
        """
        return program.eliminate_common_expressions(context=context)


@dataclass(frozen=True, slots=True)
class CircuitSimplification:
    """Apply phase-exact local gate rules after checked circuit projection.

    Each gate moves back past earlier gates that provably commute with it
    (diagonal or X-algebra action on every shared wire; controls are diagonal),
    then inverse pairs cancel, exact phase gates fuse, rotation angles merge,
    and zero rotations vanish. Top-level SWAPs become wire relabeling. Rules
    retain measurements, classically conditioned gates, structured regions,
    ordered numeric checks, coherent controls, and segment boundaries. The
    configuration is portable to deferred execution.
    """

    name: str = field(default="circuit.exact-local", init=False)
    version: str = field(default="1", init=False)
    configuration: bytes = field(default=b"", init=False)


@dataclass(frozen=True, slots=True, init=False)
class OptimizationPipeline:
    """Compose immutable passes across symbolic and specialized boundaries.

    Args:
        passes (Sequence[OptimizationPass]): Pass implementations in execution order.
        seed (int): Unsigned 64-bit deterministic optimization seed.
        max_work (int): Native rewrite visits per call, from one to one million.
        family_passes (Sequence[FamilyOptimizationPass]): Symbolic passes before
            Const binding; requires a QKernel or ProgramFamily input.
        circuit (CircuitSimplification | None): Portable late circuit policy.

    Raises:
        TypeError: If pass implementations or configuration types are invalid.
        ValueError: If identities, work limits, or pipeline composition are invalid.
    """

    passes: tuple[OptimizationPass, ...]
    seed: int
    max_work: int
    _identities: tuple[tuple[str, str, bytes], ...] = field(repr=False)
    family_passes: tuple[FamilyOptimizationPass, ...]
    circuit: CircuitSimplification | None
    _family_identities: tuple[tuple[str, str, bytes], ...] = field(repr=False)

    def __init__(
        self,
        passes: Sequence[OptimizationPass] = (),
        *,
        seed: int = 0,
        max_work: int = 1_000_000,
        family_passes: Sequence[FamilyOptimizationPass] = (),
        circuit: CircuitSimplification | None = None,
    ) -> None:
        """Snapshot bounded pass configuration before any extension invocation.

        Args:
            passes (Sequence[OptimizationPass]): Explicit ordered implementations.
            seed (int): Deterministic unsigned 64-bit optimization seed.
            max_work (int): Maximum visits per native rewrite invocation.
            family_passes (Sequence[FamilyOptimizationPass]): Symbolic stage passes.
            circuit (CircuitSimplification | None): Deferred circuit policy.

        Raises:
            TypeError: If a pass or policy has an invalid type.
            ValueError: If identities or limits violate the native contract.
        """
        constructor = _require_optimization_api()
        if type(seed) is not int or not 0 <= seed < 2**64:
            raise ValueError("optimization seed must be an unsigned 64-bit integer")
        if type(max_work) is not int or not 1 <= max_work <= 1_000_000:
            raise ValueError("max_work must be between one and one million")
        if len(passes) + len(family_passes) + int(circuit is not None) > 64:
            raise ValueError("pipeline exceeds 64 passes")
        if circuit is not None and type(circuit) is not CircuitSimplification:
            raise TypeError("circuit must be CircuitSimplification or None")
        frozen = tuple(passes)
        identities = tuple(_identity(item) for item in frozen)
        families = tuple(family_passes)
        family_identities = tuple(_identity(item) for item in families)
        object.__setattr__(self, "passes", frozen)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "max_work", max_work)
        object.__setattr__(self, "_identities", identities)
        object.__setattr__(self, "family_passes", families)
        object.__setattr__(self, "_family_identities", family_identities)
        object.__setattr__(self, "circuit", circuit)
        constructor.validate_pipeline(list(self._all_identities))

    @property
    def _all_identities(self) -> tuple[tuple[str, str, bytes], ...]:
        """Return every pass identity in driver order for native validation.

        Returns:
            tuple[tuple[str, str, bytes], ...]: Family, specialized, then circuit identities.
        """
        circuit = self.circuit
        late = (
            ()
            if circuit is None
            else ((circuit.name, circuit.version, circuit.configuration),)
        )
        return (*self._family_identities, *self._identities, *late)

    def _with(
        self,
        *,
        passes: Sequence[OptimizationPass] | None = None,
        family_passes: Sequence[FamilyOptimizationPass] | None = None,
    ) -> OptimizationPipeline:
        """Rebuild this immutable composition with one stage replaced.

        Args:
            passes (Sequence[OptimizationPass] | None): New specialized passes, or
                None to retain the current ones.
            family_passes (Sequence[FamilyOptimizationPass] | None): New symbolic
                passes, or None to retain the current ones.

        Returns:
            OptimizationPipeline: New checked pipeline sharing every other setting.
        """
        return OptimizationPipeline(
            self.passes if passes is None else passes,
            seed=self.seed,
            max_work=self.max_work,
            family_passes=(
                self.family_passes if family_passes is None else family_passes
            ),
            circuit=self.circuit,
        )

    @classmethod
    def standard(cls, *, seed: int = 0) -> OptimizationPipeline:
        """Create the standard pipeline, reusable by local and external extensions.

        Args:
            seed (int): Unsigned deterministic optimization seed.

        Returns:
            OptimizationPipeline: Pipeline containing the public standard pass.
        """
        return cls((StandardSimplification(),), seed=seed)

    def append(self, optimization_pass: OptimizationPass) -> OptimizationPipeline:
        """Append one pass while retaining this immutable base.

        Args:
            optimization_pass (OptimizationPass): Additional QCore transformation.

        Returns:
            OptimizationPipeline: New checked pipeline.

        Raises:
            TypeError: If the pass has an invalid protocol shape.
            ValueError: If identity or pipeline bounds fail validation.
        """
        return self._with(passes=(*self.passes, optimization_pass))

    @classmethod
    def multilevel(
        cls,
        *,
        family: bool = True,
        circuit: CircuitSimplification | None = CircuitSimplification(),
        seed: int = 0,
        max_work: int = 1_000_000,
    ) -> OptimizationPipeline:
        """Compose working symbolic, hybrid, and circuit stages.

        Args:
            family (bool): Run symbolic CSE; requires source or a ProgramFamily.
                Set False when compiling an already specialized Program.
            circuit (CircuitSimplification | None): Late exact circuit rules;
                requires a circuit target or Reference.
            seed (int): Deterministic unsigned optimization seed.
            max_work (int): Native traversal bound per invocation.

        Returns:
            OptimizationPipeline: Immutable OSS pipeline with replaceable defaults.
        """
        if type(family) is not bool:
            raise TypeError("family must be a bool")
        return cls(
            (
                CommonExpressions(),
                StandardSimplification(),
            ),
            family_passes=(FamilyCommonExpressions(),) if family else (),
            circuit=circuit,
            seed=seed,
            max_work=max_work,
        )

    def append_family(
        self, optimization_pass: FamilyOptimizationPass
    ) -> OptimizationPipeline:
        """Extend the symbolic stage while retaining the other stage defaults.

        Args:
            optimization_pass (FamilyOptimizationPass): Symbolic family transformation.

        Returns:
            OptimizationPipeline: New immutable stage composition.
        """
        return self._with(family_passes=(*self.family_passes, optimization_pass))

    def replace_family(
        self, name: str, optimization_pass: FamilyOptimizationPass
    ) -> OptimizationPipeline:
        """Replace one symbolic pass while retaining all other stage defaults.

        Args:
            name (str): Existing family pass identity.
            optimization_pass (FamilyOptimizationPass): Replacement transformation.

        Returns:
            OptimizationPipeline: New checked immutable composition.

        Raises:
            ValueError: If the slot is absent or the composition is invalid.
            TypeError: If the replacement has an invalid protocol shape.
        """
        names = tuple(identity[0] for identity in self._family_identities)
        if name not in names:
            raise ValueError(f"unknown family optimization pass: {name}")
        index = names.index(name)
        return self._with(
            family_passes=(
                *self.family_passes[:index],
                optimization_pass,
                *self.family_passes[index + 1 :],
            )
        )

    def optimize_family(
        self, family: object, *, context: OptimizationContext | None = None
    ) -> OptimizationFamily:
        """Run only the symbolic stage, without requiring any Const bindings.

        Args:
            family (object): Source QKernel or ProgramFamily.
            context (OptimizationContext | None): Target constraints; None is neutral.

        Returns:
            OptimizationFamily: Verified symbolic candidate.

        Raises:
            TypeError: If the input lacks an unspecialized source family.
            ArtifactError: If a pass changes identity, interface, or fails.
        """
        from .source import ProgramFamily, QKernel

        if isinstance(family, QKernel):
            family = family.family()
        if not isinstance(family, ProgramFamily):
            raise TypeError("family optimization requires a QKernel or ProgramFamily")
        context = replace(
            context or OptimizationContext("", frozenset()),
            seed=self.seed,
            max_work=self.max_work,
        )
        source = OptimizationFamily.from_bytes(
            family.module.to_bytes(), family.entrypoint
        )
        current = source
        for optimization_pass, identity in zip(
            self.family_passes, self._family_identities, strict=True
        ):
            if _identity(optimization_pass) != identity:
                raise ArtifactError("Q7802: family pass configuration changed")
            result = optimization_pass.run(current, context)
            if (
                not isinstance(result, OptimizationFamily)
                or result.entrypoint != current.entrypoint
            ):
                raise ArtifactError("Q7802: family pass changed the selected entry")
            if _identity(optimization_pass) != identity:
                raise ArtifactError("Q7802: family pass configuration changed")
            try:
                source._handle.validate_result(result._handle)
            except ValueError as error:
                raise ArtifactError(str(error)) from error
            current = result
        return current

    def replace(
        self, name: str, optimization_pass: OptimizationPass
    ) -> OptimizationPipeline:
        """Replace one public pass slot, preserving all other defaults.

        Args:
            name (str): Existing public pass identity.
            optimization_pass (OptimizationPass): Replacement implementation.

        Returns:
            OptimizationPipeline: New checked ordered pipeline.

        Raises:
            ValueError: If the slot is absent or composition is invalid.
            TypeError: If the replacement has an invalid protocol shape.
        """
        index = self._index(name)
        return self._with(
            passes=(*self.passes[:index], optimization_pass, *self.passes[index + 1 :])
        )

    def insert_before(
        self, name: str, optimization_pass: OptimizationPass
    ) -> OptimizationPipeline:
        """Insert a pass before one stable public optimization boundary.

        Args:
            name (str): Existing pass identity.
            optimization_pass (OptimizationPass): Pass to insert.

        Returns:
            OptimizationPipeline: New checked pipeline.

        Raises:
            ValueError: If the slot or composition is invalid.
            TypeError: If the inserted pass is malformed.
        """
        index = self._index(name)
        return self._with(
            passes=(*self.passes[:index], optimization_pass, *self.passes[index:])
        )

    def insert_after(
        self, name: str, optimization_pass: OptimizationPass
    ) -> OptimizationPipeline:
        """Insert a pass after one stable public optimization boundary.

        Args:
            name (str): Existing public pass identity.
            optimization_pass (OptimizationPass): Additional QCore transformation.

        Returns:
            OptimizationPipeline: New checked immutable pipeline.

        Raises:
            ValueError: If the slot or composition is invalid.
            TypeError: If the inserted pass is malformed.
        """
        index = self._index(name) + 1
        return self._with(
            passes=(*self.passes[:index], optimization_pass, *self.passes[index:])
        )

    def _index(self, name: str) -> int:
        """Resolve a stable slot without exposing implementation positions.

        Args:
            name (str): Existing public pass identifier.

        Returns:
            int: Position in the checked immutable sequence.

        Raises:
            ValueError: If the requested pass is absent.
        """
        for index, identity in enumerate(self._identities):
            if identity[0] == name:
                return index
        raise ValueError(f"unknown optimization pass {name!r}")

    def _prepare(
        self,
        program: Program,
        context: OptimizationContext,
        *,
        candidate: Program | None = None,
    ) -> Program:
        """Execute explicit passes and associate their checked result in Rust.

        Args:
            program (Program): Original specialized source authority.
            context (OptimizationContext): Immutable portable target constraints.
            candidate (Program | None): Specialized output of the symbolic stage.

        Returns:
            Program: Selected candidate retaining native source association.

        Raises:
            ArtifactError: If a pass fails, changes identity or ABI, or returns invalid data.
        """
        context = replace(context, seed=self.seed, max_work=self.max_work)
        source = program._native_program()
        if self.family_passes and candidate is None:
            raise ArtifactError(
                "Q7802: symbolic stage requires an unspecialized source family"
            )
        current = OptimizationProgram(
            source if candidate is None else candidate._native_program()
        )
        try:
            source.validate_result(current._handle)
        except (TypeError, ValueError) as error:
            raise ArtifactError(str(error)) from error
        for optimization_pass, identity in zip(
            self.passes, self._identities, strict=True
        ):
            if _identity(optimization_pass) != identity:
                raise ArtifactError("Q7802: optimization pass configuration changed")
            result = optimization_pass.run(current, context)
            if not isinstance(result, OptimizationProgram):
                raise ArtifactError("Q7802: pass must return OptimizationProgram")
            if _identity(optimization_pass) != identity:
                raise ArtifactError("Q7802: optimization pass configuration changed")
            try:
                source.validate_result(result._handle)
            except (TypeError, ValueError) as error:
                raise ArtifactError(str(error)) from error
            current = result
        try:
            owner = source.with_optimization(
                current._handle,
                list(self._all_identities),
                self.seed,
                self.max_work,
                context.target_fingerprint,
            )
        except (TypeError, ValueError) as error:
            raise ArtifactError(str(error)) from error
        return _program_with_owner(owner)


def _identity(
    optimization_pass: OptimizationPass | FamilyOptimizationPass,
) -> tuple[str, str, bytes]:
    """Snapshot the open extension protocol's portable identity fields.

    Args:
        optimization_pass (OptimizationPass): Installed implementation object.

    Returns:
        tuple[str, str, bytes]: Namespaced ID, version, and deterministic configuration.

    Raises:
        TypeError: If a protocol member has the wrong shape.
    """
    if not isinstance(optimization_pass, OptimizationPass) or not callable(
        optimization_pass.run
    ):
        raise TypeError("optimization passes must implement OptimizationPass")
    identity = (
        optimization_pass.name,
        optimization_pass.version,
        optimization_pass.configuration,
    )
    if (
        type(identity[0]) is not str
        or type(identity[1]) is not str
        or type(identity[2]) is not bytes
    ):
        raise TypeError(
            "pass name/version must be strings and configuration must be bytes"
        )
    if identity[0].startswith(
        ("qcore.", "family.", "circuit.", "realization.")
    ) and type(optimization_pass) not in (
        StandardSimplification,
        CommonExpressions,
        FamilyCommonExpressions,
    ):
        raise ValueError(
            "qcore pass identities are reserved for built-in implementations"
        )
    return identity


def _preserve_preparation(program: Program, transformed: Program) -> Program:
    """Keep explicit preparation selected across native realization substitution.

    Args:
        program (Program): Input carrying the driver's selected pipeline.
        transformed (Program): Natively verified realization substitution.

    Returns:
        Program: Substituted result without another implicit standard pass.
    """
    if program._native_program().optimization_stages() is None:
        return transformed
    return _program_with_owner(
        program._native_program().preserve_optimization(transformed._native_program())
    )


def _program_with_owner(owner: Any) -> Program:
    """Adopt the Python view of a fully verified native preparation.

    Args:
        owner (Any): Native checked owner retaining source and preparation.

    Returns:
        Program: Immutable facade of its selected QCore image.
    """
    program = Program.from_bytes(owner.to_bytes())
    object.__setattr__(program, "_owner", owner)
    return program


def _target_optimization_stages(program: Program) -> tuple[str, ...]:
    """Report lazy default preparation only when the driver has not selected one.

    Args:
        program (Program): Program passed to native target projection.

    Returns:
        tuple[str, ...]: Default stage, or empty when the driver owns the receipt.
    """
    return (
        ("rust-qcore-standard-optimization",)
        if program._native_program().optimization_stages() is None
        else ()
    )


def _restore_optimization(
    source: Program, candidate: Program, receipt: bytes
) -> Program:
    """Recover a selected result without invoking any extension implementation.

    Args:
        source (Program): Original authority retained by the resource study.
        candidate (Program): Prepared QCore stored in an execution recipe.
        receipt (bytes): Canonical Rust-owned optimization association.

    Returns:
        Program: Selected native preparation with matching recipe bytes.

    Raises:
        ArtifactError: If association, graph, ABI, or recipe identity fails.
    """
    try:
        owner = source._native_program().restore_optimization(receipt)
        if owner.to_bytes() != candidate.to_bytes():
            raise ValueError("optimization receipt disagrees with execution recipe")
    except (TypeError, ValueError) as error:
        raise ArtifactError(f"Q7802: {error}") from error
    return _program_with_owner(owner)
