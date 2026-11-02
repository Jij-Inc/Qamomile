from qamomile.proto import qcore_pb2 as _qcore_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class HIRExpressionProto(_message.Message):
    __slots__ = ("opcode", "type", "operands", "attributes", "span", "regions")
    OPCODE_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    OPERANDS_FIELD_NUMBER: _ClassVar[int]
    ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    SPAN_FIELD_NUMBER: _ClassVar[int]
    REGIONS_FIELD_NUMBER: _ClassVar[int]
    opcode: str
    type: _qcore_pb2.TypeProto
    operands: _containers.RepeatedCompositeFieldContainer[HIRExpressionProto]
    attributes: _containers.RepeatedCompositeFieldContainer[_qcore_pb2.AttributeProto]
    span: _qcore_pb2.SourceSpanProto
    regions: _containers.RepeatedCompositeFieldContainer[HIRExpressionProto]
    def __init__(self, opcode: _Optional[str] = ..., type: _Optional[_Union[_qcore_pb2.TypeProto, _Mapping]] = ..., operands: _Optional[_Iterable[_Union[HIRExpressionProto, _Mapping]]] = ..., attributes: _Optional[_Iterable[_Union[_qcore_pb2.AttributeProto, _Mapping]]] = ..., span: _Optional[_Union[_qcore_pb2.SourceSpanProto, _Mapping]] = ..., regions: _Optional[_Iterable[_Union[HIRExpressionProto, _Mapping]]] = ...) -> None: ...

class HIRStatementProto(_message.Message):
    __slots__ = ("opcode", "expression", "targets", "attributes", "regions", "span", "result_types")
    OPCODE_FIELD_NUMBER: _ClassVar[int]
    EXPRESSION_FIELD_NUMBER: _ClassVar[int]
    TARGETS_FIELD_NUMBER: _ClassVar[int]
    ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    REGIONS_FIELD_NUMBER: _ClassVar[int]
    SPAN_FIELD_NUMBER: _ClassVar[int]
    RESULT_TYPES_FIELD_NUMBER: _ClassVar[int]
    opcode: str
    expression: HIRExpressionProto
    targets: _containers.RepeatedScalarFieldContainer[str]
    attributes: _containers.RepeatedCompositeFieldContainer[_qcore_pb2.AttributeProto]
    regions: _containers.RepeatedCompositeFieldContainer[HIRBlockProto]
    span: _qcore_pb2.SourceSpanProto
    result_types: _containers.RepeatedCompositeFieldContainer[_qcore_pb2.TypeProto]
    def __init__(self, opcode: _Optional[str] = ..., expression: _Optional[_Union[HIRExpressionProto, _Mapping]] = ..., targets: _Optional[_Iterable[str]] = ..., attributes: _Optional[_Iterable[_Union[_qcore_pb2.AttributeProto, _Mapping]]] = ..., regions: _Optional[_Iterable[_Union[HIRBlockProto, _Mapping]]] = ..., span: _Optional[_Union[_qcore_pb2.SourceSpanProto, _Mapping]] = ..., result_types: _Optional[_Iterable[_Union[_qcore_pb2.TypeProto, _Mapping]]] = ...) -> None: ...

class HIRBlockProto(_message.Message):
    __slots__ = ("statements",)
    STATEMENTS_FIELD_NUMBER: _ClassVar[int]
    statements: _containers.RepeatedCompositeFieldContainer[HIRStatementProto]
    def __init__(self, statements: _Optional[_Iterable[_Union[HIRStatementProto, _Mapping]]] = ...) -> None: ...

class HIRParameterProto(_message.Message):
    __slots__ = ("name", "type", "default_value", "has_default")
    NAME_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    DEFAULT_VALUE_FIELD_NUMBER: _ClassVar[int]
    HAS_DEFAULT_FIELD_NUMBER: _ClassVar[int]
    name: str
    type: _qcore_pb2.TypeProto
    default_value: _qcore_pb2.LiteralProto
    has_default: bool
    def __init__(self, name: _Optional[str] = ..., type: _Optional[_Union[_qcore_pb2.TypeProto, _Mapping]] = ..., default_value: _Optional[_Union[_qcore_pb2.LiteralProto, _Mapping]] = ..., has_default: bool = ...) -> None: ...

class HIRFunctionProto(_message.Message):
    __slots__ = ("symbol", "parameters", "result_type", "body", "declared_kind", "span", "external", "intent")
    SYMBOL_FIELD_NUMBER: _ClassVar[int]
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    RESULT_TYPE_FIELD_NUMBER: _ClassVar[int]
    BODY_FIELD_NUMBER: _ClassVar[int]
    DECLARED_KIND_FIELD_NUMBER: _ClassVar[int]
    SPAN_FIELD_NUMBER: _ClassVar[int]
    EXTERNAL_FIELD_NUMBER: _ClassVar[int]
    INTENT_FIELD_NUMBER: _ClassVar[int]
    symbol: str
    parameters: _containers.RepeatedCompositeFieldContainer[HIRParameterProto]
    result_type: _qcore_pb2.TypeProto
    body: HIRBlockProto
    declared_kind: str
    span: _qcore_pb2.SourceSpanProto
    external: bool
    intent: _containers.RepeatedCompositeFieldContainer[_qcore_pb2.AttributeProto]
    def __init__(self, symbol: _Optional[str] = ..., parameters: _Optional[_Iterable[_Union[HIRParameterProto, _Mapping]]] = ..., result_type: _Optional[_Union[_qcore_pb2.TypeProto, _Mapping]] = ..., body: _Optional[_Union[HIRBlockProto, _Mapping]] = ..., declared_kind: _Optional[str] = ..., span: _Optional[_Union[_qcore_pb2.SourceSpanProto, _Mapping]] = ..., external: bool = ..., intent: _Optional[_Iterable[_Union[_qcore_pb2.AttributeProto, _Mapping]]] = ...) -> None: ...

class HIRConstantProto(_message.Message):
    __slots__ = ("name", "type", "value")
    NAME_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    name: str
    type: _qcore_pb2.TypeProto
    value: _qcore_pb2.LiteralProto
    def __init__(self, name: _Optional[str] = ..., type: _Optional[_Union[_qcore_pb2.TypeProto, _Mapping]] = ..., value: _Optional[_Union[_qcore_pb2.LiteralProto, _Mapping]] = ...) -> None: ...

class HIRPackageProto(_message.Message):
    __slots__ = ("schema_major", "schema_minor", "artifact_kind", "module_name", "functions", "constants", "package_digest")
    SCHEMA_MAJOR_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_MINOR_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_KIND_FIELD_NUMBER: _ClassVar[int]
    MODULE_NAME_FIELD_NUMBER: _ClassVar[int]
    FUNCTIONS_FIELD_NUMBER: _ClassVar[int]
    CONSTANTS_FIELD_NUMBER: _ClassVar[int]
    PACKAGE_DIGEST_FIELD_NUMBER: _ClassVar[int]
    schema_major: int
    schema_minor: int
    artifact_kind: str
    module_name: str
    functions: _containers.RepeatedCompositeFieldContainer[HIRFunctionProto]
    constants: _containers.RepeatedCompositeFieldContainer[HIRConstantProto]
    package_digest: bytes
    def __init__(self, schema_major: _Optional[int] = ..., schema_minor: _Optional[int] = ..., artifact_kind: _Optional[str] = ..., module_name: _Optional[str] = ..., functions: _Optional[_Iterable[_Union[HIRFunctionProto, _Mapping]]] = ..., constants: _Optional[_Iterable[_Union[HIRConstantProto, _Mapping]]] = ..., package_digest: _Optional[bytes] = ...) -> None: ...
