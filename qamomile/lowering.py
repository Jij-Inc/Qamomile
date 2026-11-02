"""Immutable choices for late, compiler-owned operation realization.

These options are target configuration. They never alter QCore identity or
logical resource counts. Rust validates implementation names and selects a checked
implementation; Python retains only the canonical projection.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field

from ._native import compiler as native
from .errors import CapabilityError as _CapabilityError

__all__ = ["RealizationOptions", "RealizationInfo", "list_realizations", "realize"]


@dataclass(frozen=True, slots=True)
class RealizationInfo:
    """Read-only compiler projection of one available implementation.

    Args:
        name (str): Selectable implementation name, without a version suffix.
        family (str): Semantic QCore operation implemented by this choice.
        description (str): Strategy and relevant tradeoffs from the Rust owner.
        required_capabilities (tuple[str, ...]): Instructions the next consumer
            must support. An empty tuple denotes a portable implementation.
    """

    name: str
    family: str
    description: str
    required_capabilities: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ConfiguredRealization:
    """Retain one native-validated parameterized realization request.

    Args:
        name (str): Catalog realization identity without parameters.
        label (str): Canonical compiler-owned configured identity.
    """

    name: str
    label: str


@dataclass(frozen=True, slots=True)
class _CustomRealization:
    """Retain one transient verified callable substitution.

    Args:
        target_kind (str): Closed native target kind, ``operation`` or
            ``callable``.
        target_module (bytes): Canonical target module for callable targets,
            or empty bytes for semantic operations.
        target_symbol (str): Canonical opcode or callable symbol.
        implementation_module (bytes): Canonical replacement module.
        implementation_symbol (str): Resolved replacement symbol.
    """

    target_kind: str
    target_module: bytes
    target_symbol: str
    implementation_module: bytes
    implementation_symbol: str


def realize(name: str, **parameters: int) -> object:
    """Configure one parameterized compiler-owned realization.

    The returned immutable token is intended as a value in
    :class:`RealizationOptions`. Parameter meaning and canonicalization remain
    owned by the native compiler.

    Args:
        name (str): Catalog realization name.
        **parameters (int): Named integer parameters owned by that realization.
            The first parameterized choice accepts ``padding_bits`` from 1
            through 4096.

    Returns:
        object: Native-validated configured realization token.

    Raises:
        TypeError: If the name or a parameter has an invalid Python type.
        ValueError: If the name or parameter row is unsupported.
        CapabilityError: If native configuration support is unavailable.
    """

    if not isinstance(name, str):
        raise TypeError("realization name must be a string")
    for parameter, value in parameters.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{parameter} must be an integer")
    configure = getattr(native, "configure_realization", None)
    if not callable(configure):
        raise _CapabilityError("Q8049: native realization configuration is unavailable")
    encoded = [
        (parameter, str(parameters[parameter])) for parameter in sorted(parameters)
    ]
    label = configure(name, encoded)
    if not isinstance(label, str) or not label:
        raise TypeError("native realization configuration projection violates the ABI")
    return _ConfiguredRealization(name, label)


def list_realizations() -> tuple[RealizationInfo, ...]:
    """Discover implementations supplied by the installed native compiler.

    The list is not a guarantee of support by a particular backend. Explicit
    overrides are checked against its capabilities when the target is used.
    Names select algorithms, not frozen gate sequences: retain package/SDK
    versions and emitted artifacts when an experiment needs reproducibility.

    Returns:
        tuple[RealizationInfo, ...]: Immutable definitions in compiler order.

    Raises:
        CapabilityError: If the native discovery API is unavailable.
        TypeError: If the native projection violates the ABI.
    """

    discover = getattr(native, "list_realizations", None)
    if not callable(discover):
        raise _CapabilityError("Q8049: native lowering discovery is unavailable")
    rows = discover()
    if not isinstance(rows, list):
        raise TypeError("native lowering discovery projection violates the ABI")
    result = []
    names: set[str] = set()
    for row in rows:
        if (
            not isinstance(row, tuple)
            or len(row) != 4
            or any(not isinstance(value, str) or not value for value in row[:3])
            or not isinstance(row[3], list)
            or any(not isinstance(label, str) or not label for label in row[3])
            or len(set(row[3])) != len(row[3])
            or row[0] in names
        ):
            raise TypeError("native lowering discovery projection violates the ABI")
        names.add(row[0])
        result.append(RealizationInfo(*row[:3], tuple(row[3])))
    return tuple(result)


@dataclass(frozen=True, slots=True, init=False)
class RealizationOptions:
    """Choose implementations for semantic operations without changing QCore.

    A value is a compiler-owned implementation name, a token returned by
    :func:`realize`, or an ordinary source-backed :class:`qamomile.QKernel`.
    Custom kernels are linked into a
    transient, fully reverified QCore image before resource projection or
    target emission; the original module and its semantic digest are retained.
    Qamomile checks its ABI and callable effects, but the mapping itself is the
    user's assertion that the custom kernel has the operation's semantics.

    Args:
        choices (Mapping[object, object] | None): Semantic intrinsic or
            body-backed QKernel to a built-in name, configured token, or
            replacement QKernel. Supported intrinsic keys are
            ``qm.add_into``, ``qm.mod_mul``, ``qm.mod_mul_add``,
            ``qm.mod_multi_pow``, ``qm.mod_pow``, ``qm.pauli_evolve``,
            and the explicit ``"quantum.arithmetic"`` family. Callable keys
            accept only QKernel replacements.

    Raises:
        TypeError: If a key or implementation has an invalid type.
        ValueError: If an implementation belongs to another operation.
        CapabilityError: If native validation is unavailable.
    """

    builtins: tuple[str, ...]
    _custom: tuple[_CustomRealization, ...] = field(repr=False)

    def __init__(self, choices: Mapping[object, object] | None = None) -> None:
        """Normalize a small immutable operation-to-implementation mapping.

        Args:
            choices (Mapping[object, object] | None): Requested realizations.

        Raises:
            TypeError: If the mapping, keys, or values are malformed.
            ValueError: If a built-in belongs to a different operation.
            CapabilityError: If the native catalog or validator is unavailable.
        """

        if choices is None:
            choices = {}
        if not isinstance(choices, Mapping):
            raise TypeError("realizations must be an operation mapping")
        if len(choices) > 32:
            raise ValueError("realizations may contain at most 32 operations")
        catalog = {item.name: item for item in list_realizations()}
        builtins: list[str] = []
        custom: list[_CustomRealization] = []
        seen: set[str] = set()
        for operation, implementation in choices.items():
            from .source import QKernel

            target_kernel = operation if isinstance(operation, QKernel) else None
            target_name = (
                f"callable:{target_kernel.symbol}"
                if target_kernel is not None
                else _operation_name(operation)
            )
            if target_name in seen:
                raise ValueError(f"duplicate realization for {target_name}")
            seen.add(target_name)
            if isinstance(implementation, (str, _ConfiguredRealization)):
                if target_kernel is not None:
                    raise TypeError(
                        "a callable realization target requires a QKernel implementation"
                    )
                name = (
                    implementation.name
                    if isinstance(implementation, _ConfiguredRealization)
                    else implementation
                )
                label = (
                    implementation.label
                    if isinstance(implementation, _ConfiguredRealization)
                    else implementation
                )
                info = catalog.get(name)
                if info is None:
                    raise ValueError(f"Q8043: unknown realization {name!r}")
                if info.family != target_name:
                    raise ValueError(
                        f"realization {name!r} implements {info.family}, not {target_name}"
                    )
                builtins.append(label)
                continue
            if not isinstance(implementation, QKernel):
                raise TypeError(
                    "a realization must be a built-in name, configured token, or QKernel"
                )
            if target_kernel is None:
                if target_name not in {
                    "quantum.add_into",
                    "quantum.mod_mul",
                    "quantum.mod_mul_add",
                }:
                    raise ValueError(
                        f"custom realization is not supported for {target_name}"
                    )
                custom.append(
                    _CustomRealization(
                        "operation",
                        b"",
                        target_name,
                        implementation.module().to_bytes(),
                        implementation.symbol,
                    )
                )
            else:
                custom.append(
                    _CustomRealization(
                        "callable",
                        target_kernel.module().to_bytes(),
                        target_kernel.symbol,
                        implementation.module().to_bytes(),
                        implementation.symbol,
                    )
                )
        validate = getattr(native, "validate_lowering_options", None)
        if not callable(validate):
            raise _CapabilityError(
                "Q8049: native lowering option validation is unavailable"
            )
        normalized = validate(tuple(builtins))
        if (
            not isinstance(normalized, list)
            or any(not isinstance(label, str) for label in normalized)
            or sorted(normalized) != sorted(builtins)
        ):
            raise TypeError("native lowering option projection violates the ABI")
        object.__setattr__(self, "builtins", tuple(normalized))
        object.__setattr__(
            self,
            "_custom",
            tuple(sorted(custom, key=lambda row: (row.target_kind, row.target_symbol))),
        )

    @property
    def overrides(self) -> tuple[str, ...]:
        """Return compiler-owned choices for the existing emission boundary.

        Returns:
            tuple[str, ...]: Canonical built-in implementation names.
        """

        return self.builtins

    @property
    def fingerprint(self) -> str:
        """Return a deterministic identity for built-in and custom choices.

        Returns:
            str: SHA-256 hexadecimal digest of the canonical option contents.
        """

        digest = hashlib.sha256()
        for label in self.builtins:
            digest.update(label.encode())
            digest.update(b"\0")
        for row in self._custom:
            digest.update(row.target_kind.encode())
            digest.update(b"\0")
            digest.update(row.target_symbol.encode())
            digest.update(b"\0")
            digest.update(row.target_module)
            digest.update(b"\0")
            digest.update(row.implementation_symbol.encode())
            digest.update(b"\0")
            digest.update(row.implementation_module)
        return digest.hexdigest()

    def _apply(self, payload: bytes, *, boundary: str) -> bytes:
        """Apply custom definitions to a transient verified QCore image.

        Args:
            payload (bytes): Canonical Module or Program bytes.
            boundary (str): Either ``module`` or ``program``.

        Returns:
            bytes: Reverified transformed bytes, or the original bytes.

        Raises:
            CapabilityError: If the native realization pass is unavailable.
            ValueError: If linking, ABI checking, or verification fails.
        """

        if not self._custom:
            return payload
        apply = getattr(native, "apply_operation_realizations", None)
        if not callable(apply):
            raise _CapabilityError("Q8049: native operation realization is unavailable")
        result = apply(
            payload,
            boundary,
            [row.target_kind for row in self._custom],
            [row.target_module for row in self._custom],
            [row.target_symbol for row in self._custom],
            [row.implementation_module for row in self._custom],
            [row.implementation_symbol for row in self._custom],
        )
        if not isinstance(result, bytes):
            raise TypeError("native operation realization projection violates the ABI")
        return result

    def require_capabilities(self, labels: tuple[str, ...]) -> None:
        """Validate every override against the selected emission boundary.

        Args:
            labels (tuple[str, ...]): Closed native capability identities.

        Raises:
            CapabilityError: If a requested realization is unavailable.
        """

        validate = getattr(native, "validate_lowering_options", None)
        if not callable(validate):
            raise _CapabilityError(
                "Q8049: native lowering option validation is unavailable"
            )
        try:
            validate(self.builtins, labels)
        except ValueError as error:
            raise _CapabilityError(str(error)) from error


def _operation_name(operation: object) -> str:
    """Resolve one public semantic intrinsic to its canonical opcode.

    Args:
        operation (object): Intrinsic function or explicit supported name.

    Returns:
        str: Canonical semantic opcode.

    Raises:
        TypeError: If the key is not a supported intrinsic identity.
    """

    name = (
        operation
        if isinstance(operation, str)
        else getattr(operation, "__name__", None)
    )
    aliases = {
        "add_into": "quantum.add_into",
        "quantum.add_into": "quantum.add_into",
        "mod_mul": "quantum.mod_mul",
        "quantum.mod_mul": "quantum.mod_mul",
        "mod_mul_add": "quantum.mod_mul_add",
        "quantum.mod_mul_add": "quantum.mod_mul_add",
        "mod_multi_pow": "quantum.mod_multi_pow",
        "quantum.mod_multi_pow": "quantum.mod_multi_pow",
        "mod_pow": "quantum.mod_pow",
        "quantum.mod_pow": "quantum.mod_pow",
        "pauli_evolve": "quantum.pauli_evolution",
        "quantum.pauli_evolution": "quantum.pauli_evolution",
        "quantum.arithmetic": "quantum.arithmetic",
    }
    if not isinstance(name, str) or name not in aliases:
        raise TypeError("realization keys must be supported semantic intrinsics")
    return aliases[name]
