from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SourceSpanProto(_message.Message):
    __slots__ = ("uri", "start_line", "start_column", "end_line", "end_column")
    URI_FIELD_NUMBER: _ClassVar[int]
    START_LINE_FIELD_NUMBER: _ClassVar[int]
    START_COLUMN_FIELD_NUMBER: _ClassVar[int]
    END_LINE_FIELD_NUMBER: _ClassVar[int]
    END_COLUMN_FIELD_NUMBER: _ClassVar[int]
    uri: str
    start_line: int
    start_column: int
    end_line: int
    end_column: int
    def __init__(self, uri: _Optional[str] = ..., start_line: _Optional[int] = ..., start_column: _Optional[int] = ..., end_line: _Optional[int] = ..., end_column: _Optional[int] = ...) -> None: ...

class DimensionProto(_message.Message):
    __slots__ = ("fixed", "symbol")
    FIXED_FIELD_NUMBER: _ClassVar[int]
    SYMBOL_FIELD_NUMBER: _ClassVar[int]
    fixed: int
    symbol: str
    def __init__(self, fixed: _Optional[int] = ..., symbol: _Optional[str] = ...) -> None: ...

class TypeProto(_message.Message):
    __slots__ = ("kind", "name", "arguments", "dimensions", "field_names")
    class Kind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        KIND_UNSPECIFIED: _ClassVar[TypeProto.Kind]
        SCALAR: _ClassVar[TypeProto.Kind]
        ARRAY: _ClassVar[TypeProto.Kind]
        PRODUCT: _ClassVar[TypeProto.Kind]
        RECORD: _ClassVar[TypeProto.Kind]
        QUANTUM: _ClassVar[TypeProto.Kind]
        STAGED: _ClassVar[TypeProto.Kind]
        FIXED: _ClassVar[TypeProto.Kind]
        FUNCTION: _ClassVar[TypeProto.Kind]
    KIND_UNSPECIFIED: TypeProto.Kind
    SCALAR: TypeProto.Kind
    ARRAY: TypeProto.Kind
    PRODUCT: TypeProto.Kind
    RECORD: TypeProto.Kind
    QUANTUM: TypeProto.Kind
    STAGED: TypeProto.Kind
    FIXED: TypeProto.Kind
    FUNCTION: TypeProto.Kind
    KIND_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    ARGUMENTS_FIELD_NUMBER: _ClassVar[int]
    DIMENSIONS_FIELD_NUMBER: _ClassVar[int]
    FIELD_NAMES_FIELD_NUMBER: _ClassVar[int]
    kind: TypeProto.Kind
    name: str
    arguments: _containers.RepeatedCompositeFieldContainer[TypeProto]
    dimensions: _containers.RepeatedCompositeFieldContainer[DimensionProto]
    field_names: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, kind: _Optional[_Union[TypeProto.Kind, str]] = ..., name: _Optional[str] = ..., arguments: _Optional[_Iterable[_Union[TypeProto, _Mapping]]] = ..., dimensions: _Optional[_Iterable[_Union[DimensionProto, _Mapping]]] = ..., field_names: _Optional[_Iterable[str]] = ...) -> None: ...

class LiteralProto(_message.Message):
    __slots__ = ("unit_value", "bool_value", "int_value", "real_value", "string_value", "bytes_value", "tuple_value", "mapping_value", "complex_value", "bigint_value")
    class Complex(_message.Message):
        __slots__ = ("real", "imaginary")
        REAL_FIELD_NUMBER: _ClassVar[int]
        IMAGINARY_FIELD_NUMBER: _ClassVar[int]
        real: float
        imaginary: float
        def __init__(self, real: _Optional[float] = ..., imaginary: _Optional[float] = ...) -> None: ...
    class Sequence(_message.Message):
        __slots__ = ("values",)
        VALUES_FIELD_NUMBER: _ClassVar[int]
        values: _containers.RepeatedCompositeFieldContainer[LiteralProto]
        def __init__(self, values: _Optional[_Iterable[_Union[LiteralProto, _Mapping]]] = ...) -> None: ...
    class Mapping(_message.Message):
        __slots__ = ("entries",)
        ENTRIES_FIELD_NUMBER: _ClassVar[int]
        entries: _containers.RepeatedCompositeFieldContainer[LiteralProto.Entry]
        def __init__(self, entries: _Optional[_Iterable[_Union[LiteralProto.Entry, _Mapping]]] = ...) -> None: ...
    class Entry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: LiteralProto
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[LiteralProto, _Mapping]] = ...) -> None: ...
    UNIT_VALUE_FIELD_NUMBER: _ClassVar[int]
    BOOL_VALUE_FIELD_NUMBER: _ClassVar[int]
    INT_VALUE_FIELD_NUMBER: _ClassVar[int]
    REAL_VALUE_FIELD_NUMBER: _ClassVar[int]
    STRING_VALUE_FIELD_NUMBER: _ClassVar[int]
    BYTES_VALUE_FIELD_NUMBER: _ClassVar[int]
    TUPLE_VALUE_FIELD_NUMBER: _ClassVar[int]
    MAPPING_VALUE_FIELD_NUMBER: _ClassVar[int]
    COMPLEX_VALUE_FIELD_NUMBER: _ClassVar[int]
    BIGINT_VALUE_FIELD_NUMBER: _ClassVar[int]
    unit_value: bool
    bool_value: bool
    int_value: int
    real_value: float
    string_value: str
    bytes_value: bytes
    tuple_value: LiteralProto.Sequence
    mapping_value: LiteralProto.Mapping
    complex_value: LiteralProto.Complex
    bigint_value: bytes
    def __init__(self, unit_value: bool = ..., bool_value: bool = ..., int_value: _Optional[int] = ..., real_value: _Optional[float] = ..., string_value: _Optional[str] = ..., bytes_value: _Optional[bytes] = ..., tuple_value: _Optional[_Union[LiteralProto.Sequence, _Mapping]] = ..., mapping_value: _Optional[_Union[LiteralProto.Mapping, _Mapping]] = ..., complex_value: _Optional[_Union[LiteralProto.Complex, _Mapping]] = ..., bigint_value: _Optional[bytes] = ...) -> None: ...

class AttributeProto(_message.Message):
    __slots__ = ("name", "value")
    NAME_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    name: str
    value: LiteralProto
    def __init__(self, name: _Optional[str] = ..., value: _Optional[_Union[LiteralProto, _Mapping]] = ...) -> None: ...

class RegionRefProto(_message.Message):
    __slots__ = ("root", "projections")
    ROOT_FIELD_NUMBER: _ClassVar[int]
    PROJECTIONS_FIELD_NUMBER: _ClassVar[int]
    root: int
    projections: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, root: _Optional[int] = ..., projections: _Optional[_Iterable[str]] = ...) -> None: ...

class ValueProto(_message.Message):
    __slots__ = ("id", "type", "hint", "region", "has_region")
    ID_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    HINT_FIELD_NUMBER: _ClassVar[int]
    REGION_FIELD_NUMBER: _ClassVar[int]
    HAS_REGION_FIELD_NUMBER: _ClassVar[int]
    id: int
    type: TypeProto
    hint: str
    region: RegionRefProto
    has_region: bool
    def __init__(self, id: _Optional[int] = ..., type: _Optional[_Union[TypeProto, _Mapping]] = ..., hint: _Optional[str] = ..., region: _Optional[_Union[RegionRefProto, _Mapping]] = ..., has_region: bool = ...) -> None: ...

class TerminatorProto(_message.Message):
    __slots__ = ("kind", "operands", "attributes", "span")
    KIND_FIELD_NUMBER: _ClassVar[int]
    OPERANDS_FIELD_NUMBER: _ClassVar[int]
    ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    SPAN_FIELD_NUMBER: _ClassVar[int]
    kind: str
    operands: _containers.RepeatedScalarFieldContainer[int]
    attributes: _containers.RepeatedCompositeFieldContainer[AttributeProto]
    span: SourceSpanProto
    def __init__(self, kind: _Optional[str] = ..., operands: _Optional[_Iterable[int]] = ..., attributes: _Optional[_Iterable[_Union[AttributeProto, _Mapping]]] = ..., span: _Optional[_Union[SourceSpanProto, _Mapping]] = ...) -> None: ...

class BlockProto(_message.Message):
    __slots__ = ("arguments", "operations", "terminator")
    ARGUMENTS_FIELD_NUMBER: _ClassVar[int]
    OPERATIONS_FIELD_NUMBER: _ClassVar[int]
    TERMINATOR_FIELD_NUMBER: _ClassVar[int]
    arguments: _containers.RepeatedCompositeFieldContainer[ValueProto]
    operations: _containers.RepeatedCompositeFieldContainer[OperationProto]
    terminator: TerminatorProto
    def __init__(self, arguments: _Optional[_Iterable[_Union[ValueProto, _Mapping]]] = ..., operations: _Optional[_Iterable[_Union[OperationProto, _Mapping]]] = ..., terminator: _Optional[_Union[TerminatorProto, _Mapping]] = ...) -> None: ...

class RegionProto(_message.Message):
    __slots__ = ("blocks",)
    BLOCKS_FIELD_NUMBER: _ClassVar[int]
    blocks: _containers.RepeatedCompositeFieldContainer[BlockProto]
    def __init__(self, blocks: _Optional[_Iterable[_Union[BlockProto, _Mapping]]] = ...) -> None: ...

class OperationProto(_message.Message):
    __slots__ = ("opcode", "operands", "results", "attributes", "regions", "span", "intent")
    OPCODE_FIELD_NUMBER: _ClassVar[int]
    OPERANDS_FIELD_NUMBER: _ClassVar[int]
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    REGIONS_FIELD_NUMBER: _ClassVar[int]
    SPAN_FIELD_NUMBER: _ClassVar[int]
    INTENT_FIELD_NUMBER: _ClassVar[int]
    opcode: str
    operands: _containers.RepeatedScalarFieldContainer[int]
    results: _containers.RepeatedCompositeFieldContainer[ValueProto]
    attributes: _containers.RepeatedCompositeFieldContainer[AttributeProto]
    regions: _containers.RepeatedCompositeFieldContainer[RegionProto]
    span: SourceSpanProto
    intent: _containers.RepeatedCompositeFieldContainer[AttributeProto]
    def __init__(self, opcode: _Optional[str] = ..., operands: _Optional[_Iterable[int]] = ..., results: _Optional[_Iterable[_Union[ValueProto, _Mapping]]] = ..., attributes: _Optional[_Iterable[_Union[AttributeProto, _Mapping]]] = ..., regions: _Optional[_Iterable[_Union[RegionProto, _Mapping]]] = ..., span: _Optional[_Union[SourceSpanProto, _Mapping]] = ..., intent: _Optional[_Iterable[_Union[AttributeProto, _Mapping]]] = ...) -> None: ...

class ParameterProto(_message.Message):
    __slots__ = ("name", "value", "default_value", "has_default")
    NAME_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    DEFAULT_VALUE_FIELD_NUMBER: _ClassVar[int]
    HAS_DEFAULT_FIELD_NUMBER: _ClassVar[int]
    name: str
    value: ValueProto
    default_value: LiteralProto
    has_default: bool
    def __init__(self, name: _Optional[str] = ..., value: _Optional[_Union[ValueProto, _Mapping]] = ..., default_value: _Optional[_Union[LiteralProto, _Mapping]] = ..., has_default: bool = ...) -> None: ...

class FunctionProto(_message.Message):
    __slots__ = ("symbol", "parameters", "result_type", "body", "span", "asserted_kind", "external", "intent")
    SYMBOL_FIELD_NUMBER: _ClassVar[int]
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    RESULT_TYPE_FIELD_NUMBER: _ClassVar[int]
    BODY_FIELD_NUMBER: _ClassVar[int]
    SPAN_FIELD_NUMBER: _ClassVar[int]
    ASSERTED_KIND_FIELD_NUMBER: _ClassVar[int]
    EXTERNAL_FIELD_NUMBER: _ClassVar[int]
    INTENT_FIELD_NUMBER: _ClassVar[int]
    symbol: str
    parameters: _containers.RepeatedCompositeFieldContainer[ParameterProto]
    result_type: TypeProto
    body: BlockProto
    span: SourceSpanProto
    asserted_kind: str
    external: bool
    intent: _containers.RepeatedCompositeFieldContainer[AttributeProto]
    def __init__(self, symbol: _Optional[str] = ..., parameters: _Optional[_Iterable[_Union[ParameterProto, _Mapping]]] = ..., result_type: _Optional[_Union[TypeProto, _Mapping]] = ..., body: _Optional[_Union[BlockProto, _Mapping]] = ..., span: _Optional[_Union[SourceSpanProto, _Mapping]] = ..., asserted_kind: _Optional[str] = ..., external: bool = ..., intent: _Optional[_Iterable[_Union[AttributeProto, _Mapping]]] = ...) -> None: ...

class ConstantProto(_message.Message):
    __slots__ = ("name", "type", "value")
    NAME_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    name: str
    type: TypeProto
    value: LiteralProto
    def __init__(self, name: _Optional[str] = ..., type: _Optional[_Union[TypeProto, _Mapping]] = ..., value: _Optional[_Union[LiteralProto, _Mapping]] = ...) -> None: ...

class BindingProto(_message.Message):
    __slots__ = ("name", "value")
    NAME_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    name: str
    value: LiteralProto
    def __init__(self, name: _Optional[str] = ..., value: _Optional[_Union[LiteralProto, _Mapping]] = ...) -> None: ...

class QCorePackageProto(_message.Message):
    __slots__ = ("schema_major", "schema_minor", "artifact_kind", "module_name", "functions", "constants", "entrypoint", "const_bindings", "package_digest")
    SCHEMA_MAJOR_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_MINOR_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_KIND_FIELD_NUMBER: _ClassVar[int]
    MODULE_NAME_FIELD_NUMBER: _ClassVar[int]
    FUNCTIONS_FIELD_NUMBER: _ClassVar[int]
    CONSTANTS_FIELD_NUMBER: _ClassVar[int]
    ENTRYPOINT_FIELD_NUMBER: _ClassVar[int]
    CONST_BINDINGS_FIELD_NUMBER: _ClassVar[int]
    PACKAGE_DIGEST_FIELD_NUMBER: _ClassVar[int]
    schema_major: int
    schema_minor: int
    artifact_kind: str
    module_name: str
    functions: _containers.RepeatedCompositeFieldContainer[FunctionProto]
    constants: _containers.RepeatedCompositeFieldContainer[ConstantProto]
    entrypoint: str
    const_bindings: _containers.RepeatedCompositeFieldContainer[BindingProto]
    package_digest: bytes
    def __init__(self, schema_major: _Optional[int] = ..., schema_minor: _Optional[int] = ..., artifact_kind: _Optional[str] = ..., module_name: _Optional[str] = ..., functions: _Optional[_Iterable[_Union[FunctionProto, _Mapping]]] = ..., constants: _Optional[_Iterable[_Union[ConstantProto, _Mapping]]] = ..., entrypoint: _Optional[str] = ..., const_bindings: _Optional[_Iterable[_Union[BindingProto, _Mapping]]] = ..., package_digest: _Optional[bytes] = ...) -> None: ...

class ParameterSlotProto(_message.Message):
    __slots__ = ("name", "type", "default_value", "has_default")
    NAME_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    DEFAULT_VALUE_FIELD_NUMBER: _ClassVar[int]
    HAS_DEFAULT_FIELD_NUMBER: _ClassVar[int]
    name: str
    type: TypeProto
    default_value: LiteralProto
    has_default: bool
    def __init__(self, name: _Optional[str] = ..., type: _Optional[_Union[TypeProto, _Mapping]] = ..., default_value: _Optional[_Union[LiteralProto, _Mapping]] = ..., has_default: bool = ...) -> None: ...

class ParameterABIProto(_message.Message):
    __slots__ = ("slots",)
    SLOTS_FIELD_NUMBER: _ClassVar[int]
    slots: _containers.RepeatedCompositeFieldContainer[ParameterSlotProto]
    def __init__(self, slots: _Optional[_Iterable[_Union[ParameterSlotProto, _Mapping]]] = ...) -> None: ...

class ResultABIProto(_message.Message):
    __slots__ = ("type", "source_order", "native_width")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_ORDER_FIELD_NUMBER: _ClassVar[int]
    NATIVE_WIDTH_FIELD_NUMBER: _ClassVar[int]
    type: TypeProto
    source_order: _containers.RepeatedScalarFieldContainer[int]
    native_width: int
    def __init__(self, type: _Optional[_Union[TypeProto, _Mapping]] = ..., source_order: _Optional[_Iterable[int]] = ..., native_width: _Optional[int] = ...) -> None: ...

class CompilationReportProto(_message.Message):
    __slots__ = ("stages", "assumptions", "warnings", "resource_study")
    STAGES_FIELD_NUMBER: _ClassVar[int]
    ASSUMPTIONS_FIELD_NUMBER: _ClassVar[int]
    WARNINGS_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_STUDY_FIELD_NUMBER: _ClassVar[int]
    stages: _containers.RepeatedScalarFieldContainer[str]
    assumptions: _containers.RepeatedScalarFieldContainer[str]
    warnings: _containers.RepeatedScalarFieldContainer[str]
    resource_study: bytes
    def __init__(self, stages: _Optional[_Iterable[str]] = ..., assumptions: _Optional[_Iterable[str]] = ..., warnings: _Optional[_Iterable[str]] = ..., resource_study: _Optional[bytes] = ...) -> None: ...

class ArtifactProto(_message.Message):
    __slots__ = ("schema_major", "schema_minor", "format", "payload", "target_fingerprint", "parameter_abi", "result_abi", "report", "metadata", "artifact_digest", "execution_requirements")
    SCHEMA_MAJOR_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_MINOR_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    TARGET_FINGERPRINT_FIELD_NUMBER: _ClassVar[int]
    PARAMETER_ABI_FIELD_NUMBER: _ClassVar[int]
    RESULT_ABI_FIELD_NUMBER: _ClassVar[int]
    REPORT_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_REQUIREMENTS_FIELD_NUMBER: _ClassVar[int]
    schema_major: int
    schema_minor: int
    format: str
    payload: bytes
    target_fingerprint: str
    parameter_abi: ParameterABIProto
    result_abi: ResultABIProto
    report: CompilationReportProto
    metadata: _containers.RepeatedCompositeFieldContainer[AttributeProto]
    artifact_digest: bytes
    execution_requirements: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, schema_major: _Optional[int] = ..., schema_minor: _Optional[int] = ..., format: _Optional[str] = ..., payload: _Optional[bytes] = ..., target_fingerprint: _Optional[str] = ..., parameter_abi: _Optional[_Union[ParameterABIProto, _Mapping]] = ..., result_abi: _Optional[_Union[ResultABIProto, _Mapping]] = ..., report: _Optional[_Union[CompilationReportProto, _Mapping]] = ..., metadata: _Optional[_Iterable[_Union[AttributeProto, _Mapping]]] = ..., artifact_digest: _Optional[bytes] = ..., execution_requirements: _Optional[_Iterable[str]] = ...) -> None: ...

class PolynomialTermProto(_message.Message):
    __slots__ = ("variables", "coefficient")
    VARIABLES_FIELD_NUMBER: _ClassVar[int]
    COEFFICIENT_FIELD_NUMBER: _ClassVar[int]
    variables: _containers.RepeatedScalarFieldContainer[str]
    coefficient: float
    def __init__(self, variables: _Optional[_Iterable[str]] = ..., coefficient: _Optional[float] = ...) -> None: ...

class OptimizationProblemProto(_message.Message):
    __slots__ = ("schema_major", "schema_minor", "artifact_kind", "domain", "terms", "constant", "variables", "problem_digest")
    class Domain(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        DOMAIN_UNSPECIFIED: _ClassVar[OptimizationProblemProto.Domain]
        BINARY: _ClassVar[OptimizationProblemProto.Domain]
        SPIN: _ClassVar[OptimizationProblemProto.Domain]
    DOMAIN_UNSPECIFIED: OptimizationProblemProto.Domain
    BINARY: OptimizationProblemProto.Domain
    SPIN: OptimizationProblemProto.Domain
    SCHEMA_MAJOR_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_MINOR_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_KIND_FIELD_NUMBER: _ClassVar[int]
    DOMAIN_FIELD_NUMBER: _ClassVar[int]
    TERMS_FIELD_NUMBER: _ClassVar[int]
    CONSTANT_FIELD_NUMBER: _ClassVar[int]
    VARIABLES_FIELD_NUMBER: _ClassVar[int]
    PROBLEM_DIGEST_FIELD_NUMBER: _ClassVar[int]
    schema_major: int
    schema_minor: int
    artifact_kind: str
    domain: OptimizationProblemProto.Domain
    terms: _containers.RepeatedCompositeFieldContainer[PolynomialTermProto]
    constant: float
    variables: _containers.RepeatedScalarFieldContainer[str]
    problem_digest: bytes
    def __init__(self, schema_major: _Optional[int] = ..., schema_minor: _Optional[int] = ..., artifact_kind: _Optional[str] = ..., domain: _Optional[_Union[OptimizationProblemProto.Domain, str]] = ..., terms: _Optional[_Iterable[_Union[PolynomialTermProto, _Mapping]]] = ..., constant: _Optional[float] = ..., variables: _Optional[_Iterable[str]] = ..., problem_digest: _Optional[bytes] = ...) -> None: ...
