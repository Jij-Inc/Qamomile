"""Validation of explicit structured-region interfaces."""

from __future__ import annotations

from collections.abc import Sequence

from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation import Operation, Region
from qamomile.circuit.ir.operation.callable import InvokeOperation
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    HasNestedOps,
    IfOperation,
    WhileOperation,
    genuine_input_values,
    validate_region_args,
)
from qamomile.circuit.ir.operation.select import SelectOperation
from qamomile.circuit.ir.value import (
    ArrayValue,
    DictValue,
    TupleValue,
    Value,
    ValueBase,
)
from qamomile.circuit.transpiler.errors import ValidationError
from qamomile.circuit.transpiler.passes import Pass


class RegionValidationPass(Pass[Block, Block]):
    """Verify dominance and signatures for every explicit semantic region."""

    def __init__(self) -> None:
        """Initialize an empty reachable-block visitation set."""
        self._visited_blocks: set[int] = set()

    @property
    def name(self) -> str:
        """Return the compiler-visible pass name.

        Returns:
            str: Stable pass name used in diagnostics.
        """
        return "region_validation"

    def run(self, input: Block) -> Block:
        """Validate a block graph and return it unchanged.

        Args:
            input (Block): Semantic entrypoint whose regions should be
                verified.

        Returns:
            Block: The validated input block.

        Raises:
            ValidationError: If a region has an undeclared capture, violates
                dominance, or has an inconsistent block/yield signature.
        """
        self._visited_blocks = set()
        self._validate_block(input)
        return input

    def _validate_block(self, block: Block) -> None:
        """Validate one block and every operation-owned block it reaches.

        Args:
            block (Block): Block to validate.
        """
        identity = id(block)
        if identity in self._visited_blocks:
            return
        self._visited_blocks.add(identity)

        available: dict[str, ValueBase] = {}
        for value in block.input_values:
            self._register_value_graph(value, available)
        for value in block.parameters.values():
            self._register_value_graph(value, available)
        for slot in block.static_bindings:
            for field in slot.fields:
                self._register_value_graph(field.value, available)
        self._validate_operations(
            block.operations,
            available,
            "block",
            check_reads=False,
        )

    def _validate_operations(
        self,
        operations: Sequence[Operation],
        available: dict[str, ValueBase],
        location: str,
        *,
        check_reads: bool = True,
    ) -> dict[str, ValueBase]:
        """Validate operations sequentially under one dominance scope.

        Args:
            operations (Sequence[Operation]): Operations in program order.
            available (dict[str, ValueBase]): Values dominating the sequence.
            location (str): Human-readable location used in diagnostics.
            check_reads (bool): Whether to enforce sequential read dominance.
                Top-level blocks may contain values materialized from static
                binding slots rather than ordinary producer operations.
                Defaults to ``True`` for regions.

        Returns:
            dict[str, ValueBase]: Final dominance scope after the sequence.

        Raises:
            ValidationError: If an operation reads a value before its
                definition or outside a declared region interface.
        """
        scope = dict(available)
        for operation_index, operation in enumerate(operations):
            operation_location = (
                f"{location} operation {operation_index} ({type(operation).__name__})"
            )
            if check_reads:
                for value in self._outer_input_values(operation):
                    logically_available = value.type.is_quantum() and any(
                        candidate.logical_id == value.logical_id
                        and candidate.type == value.type
                        for candidate in scope.values()
                    )
                    if not self._is_available(value, scope) and not logically_available:
                        related = [
                            candidate
                            for candidate in scope.values()
                            if candidate.name == value.name
                            or candidate.logical_id == value.logical_id
                        ]
                        related_text = ", ".join(
                            f"{candidate.name!r}/{candidate.uuid!r}/"
                            f"{candidate.logical_id!r}"
                            for candidate in related[:4]
                        )
                        raise ValidationError(
                            f"{operation_location} reads value {value.name!r}/"
                            f"{value.uuid!r}/{value.logical_id!r} before it is "
                            "defined."
                            + (
                                f" Related in-scope values: {related_text}."
                                if related
                                else ""
                            )
                        )
            if isinstance(operation, HasNestedOps):
                self._validate_nested_operation(operation, scope, operation_location)
            self._validate_owned_blocks(operation)
            for result in operation.results:
                self._register_value_graph(result, scope)
            # Legacy loop rebind records can expose a traced post-body value
            # before it has been upgraded to an explicit RegionArg result.
            # Treat that value as a compatibility definition at the loop
            # boundary; it remains subject to the dedicated path-sensitive
            # rebind validation later in the pipeline.
            for rebind in getattr(operation, "loop_carried_rebinds", ()):
                self._register_value_graph(rebind.after, scope)
        return scope

    @staticmethod
    def _outer_input_values(operation: Operation) -> list[ValueBase]:
        """Return values read from the operation's enclosing scope.

        Generic ``genuine_input_values`` intentionally includes region yields
        so whole-operation liveness can see back edges. Dominance validation at
        the enclosing sequence must instead validate those values inside their
        own region.

        Args:
            operation (Operation): Operation to inspect.

        Returns:
            list[ValueBase]: Enclosing operands, captures, and loop
                initializers.
        """
        if not isinstance(operation, HasNestedOps):
            return genuine_input_values(operation)
        operands = (
            operation.operands[:1]
            if isinstance(operation, WhileOperation)
            else operation.operands
        )
        values: list[ValueBase] = [
            value for value in operands if isinstance(value, ValueBase)
        ]
        for region in operation.nested_regions():
            values.extend(region.captures)
        if isinstance(operation, (ForOperation, ForItemsOperation, WhileOperation)):
            values.extend(region_arg.init for region_arg in operation.region_args)
        return values

    def _validate_nested_operation(
        self,
        operation: HasNestedOps,
        outer_available: dict[str, ValueBase],
        location: str,
    ) -> None:
        """Validate every region owned by one structured operation.

        Args:
            operation (HasNestedOps): Structured operation to validate.
            outer_available (dict[str, ValueBase]): Enclosing values that
                dominate the operation.
            location (str): Human-readable operation location.

        Raises:
            ValidationError: If a region boundary or yield is inconsistent.
        """
        regions = operation.nested_regions()
        self._validate_operation_signature(operation, regions, location)
        for region_index, region in enumerate(regions):
            self._validate_region(
                region,
                outer_available,
                f"{location} region {region_index}",
            )

    def _validate_region(
        self,
        region: Region,
        outer_available: dict[str, ValueBase],
        location: str,
    ) -> None:
        """Validate one explicit region boundary and body.

        Args:
            region (Region): Region interface and operation sequence.
            outer_available (dict[str, ValueBase]): Enclosing dominance scope.
            location (str): Human-readable region location.

        Raises:
            ValidationError: If captures, block arguments, body reads, or
                yields violate the explicit interface.
        """
        block_arg_ids = [value.uuid for value in region.block_args]
        capture_ids = [value.uuid for value in region.captures]
        if len(set(block_arg_ids)) != len(block_arg_ids):
            raise ValidationError(f"{location} has duplicate block arguments.")
        if len(set(capture_ids)) != len(capture_ids):
            raise ValidationError(f"{location} has duplicate captures.")
        overlap = set(block_arg_ids) & set(capture_ids)
        if overlap:
            raise ValidationError(
                f"{location} declares values as both block arguments and "
                f"captures: {sorted(overlap)}."
            )
        for capture in region.captures:
            if not self._is_available(capture, outer_available):
                raise ValidationError(
                    f"{location} capture {capture.uuid!r} does not dominate the region."
                )

        scope: dict[str, ValueBase] = {}
        for value in (*region.block_args, *region.captures):
            self._register_value_graph(value, scope)
        scope = self._validate_operations(region.operations, scope, location)

        for yielded in region.yields:
            logically_available = any(
                candidate.logical_id == yielded.logical_id
                and candidate.type == yielded.type
                for candidate in scope.values()
            )
            if not self._is_available(yielded, scope) and not logically_available:
                related = [
                    value
                    for value in scope.values()
                    if value.name == yielded.name
                    or value.logical_id == yielded.logical_id
                ]
                related_text = ", ".join(
                    f"{value.name!r}/{value.uuid!r}/{value.logical_id!r}"
                    for value in related[:4]
                )
                raise ValidationError(
                    f"{location} yields undeclared value {yielded.name!r}/"
                    f"{yielded.uuid!r}/{yielded.logical_id!r}."
                    + (f" Related in-scope values: {related_text}." if related else "")
                )

    @staticmethod
    def _validate_operation_signature(
        operation: HasNestedOps,
        regions: tuple[Region, ...],
        location: str,
    ) -> None:
        """Validate concrete operation arity and type invariants.

        Args:
            operation (HasNestedOps): Operation owning ``regions``.
            regions (tuple[Region, ...]): Canonical region interfaces.
            location (str): Human-readable operation location.

        Raises:
            ValidationError: If a concrete operation's region signature is
                malformed.
        """
        if isinstance(operation, (ForOperation, ForItemsOperation, WhileOperation)):
            try:
                validate_region_args(operation)
            except ValueError as error:
                raise ValidationError(f"{location}: {error}") from error
            if len(regions) != 1:
                raise ValidationError(f"{location} must own exactly one region.")
            carried_yields = regions[0].yields[: len(operation.region_args)]
            for region_arg, yielded in zip(
                operation.region_args,
                carried_yields,
                strict=True,
            ):
                if yielded.type != region_arg.block_arg.type:
                    raise ValidationError(
                        f"{location} yield for {region_arg.var_name!r} has "
                        "a type different from its block argument."
                    )
        elif isinstance(operation, IfOperation):
            if len(regions) != 2:
                raise ValidationError(f"{location} must own two branch regions.")
            for branch_name, region in zip(("true", "false"), regions, strict=True):
                if len(region.yields) != len(operation.results):
                    raise ValidationError(
                        f"{location} {branch_name} branch yields "
                        f"{len(region.yields)} values for "
                        f"{len(operation.results)} results."
                    )
                for yielded, result in zip(
                    region.yields, operation.results, strict=True
                ):
                    if yielded.type != result.type:
                        raise ValidationError(
                            f"{location} {branch_name} yield type does not "
                            "match its result type."
                        )

    def _validate_owned_blocks(self, operation: Operation) -> None:
        """Validate blocks owned by callables and SELECT operations.

        Args:
            operation (Operation): Operation whose owned blocks are visited.
        """
        if isinstance(operation, InvokeOperation) and operation.definition is not None:
            definition = operation.definition
            if definition.body is not None:
                self._validate_block(definition.body)
            for implementation in definition.implementations:
                if implementation.body is not None:
                    self._validate_block(implementation.body)
        if isinstance(operation, SelectOperation):
            for case_block in operation.case_blocks:
                self._validate_block(case_block)

    @classmethod
    def _register_value_graph(
        cls,
        value: ValueBase,
        destination: dict[str, ValueBase],
    ) -> None:
        """Register one definition and its owned aggregate components.

        Array shape formals are owned by an array definition. Slice ancestry,
        element indices, and ``parent_array`` are dependencies; registering
        those would weaken region dominance by treating an outer array as
        local whenever an operation produces one element version.

        Args:
            value (ValueBase): Defined value-like root to traverse.
            destination (dict[str, ValueBase]): Registry updated in place.
        """
        if value.uuid in destination:
            return
        destination[value.uuid] = value
        if isinstance(value, TupleValue):
            for element in value.elements:
                cls._register_value_graph(element, destination)
        elif isinstance(value, DictValue):
            for key, entry_value in value.entries:
                cls._register_value_graph(key, destination)
                cls._register_value_graph(entry_value, destination)
        elif isinstance(value, ArrayValue):
            # Shape formals are owned metadata of an array definition (for
            # example a symbolic ForItems vector key). Slice ancestry and
            # indices remain ordinary dependencies and are not registered.
            for dimension in value.shape:
                cls._register_value_graph(dimension, destination)

    @staticmethod
    def _is_available(
        value: ValueBase,
        available: dict[str, ValueBase],
    ) -> bool:
        """Return whether a value is constant or present in a scope.

        Args:
            value (ValueBase): Candidate read.
            available (dict[str, ValueBase]): UUID-indexed dominance scope.

        Returns:
            bool: Whether the value may be read in the scope.
        """
        if value.uuid in available or value.is_constant():
            return True
        root = value if isinstance(value, ArrayValue) else None
        if isinstance(value, Value) and value.parent_array is not None:
            root = value.parent_array
        while root is not None and root.slice_of is not None:
            root = root.slice_of
        if root is None:
            return False
        if root.uuid in available:
            return True
        return any(
            candidate.logical_id == root.logical_id and candidate.type == root.type
            for candidate in available.values()
        )
