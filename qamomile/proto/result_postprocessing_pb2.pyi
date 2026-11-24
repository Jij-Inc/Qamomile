from qamomile.proto import qcore_pb2 as _qcore_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class CircuitResultScalarTypeProto(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CIRCUIT_RESULT_SCALAR_TYPE_UNSPECIFIED: _ClassVar[CircuitResultScalarTypeProto]
    CIRCUIT_RESULT_BOOL: _ClassVar[CircuitResultScalarTypeProto]
    CIRCUIT_RESULT_INT: _ClassVar[CircuitResultScalarTypeProto]
    CIRCUIT_RESULT_REAL: _ClassVar[CircuitResultScalarTypeProto]
CIRCUIT_RESULT_SCALAR_TYPE_UNSPECIFIED: CircuitResultScalarTypeProto
CIRCUIT_RESULT_BOOL: CircuitResultScalarTypeProto
CIRCUIT_RESULT_INT: CircuitResultScalarTypeProto
CIRCUIT_RESULT_REAL: CircuitResultScalarTypeProto

class CircuitResultPathProto(_message.Message):
    __slots__ = ("index", "field")
    INDEX_FIELD_NUMBER: _ClassVar[int]
    FIELD_FIELD_NUMBER: _ClassVar[int]
    index: int
    field: str
    def __init__(self, index: _Optional[int] = ..., field: _Optional[str] = ...) -> None: ...

class CircuitResultParameterProto(_message.Message):
    __slots__ = ("name", "path", "type")
    NAME_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    name: str
    path: _containers.RepeatedCompositeFieldContainer[CircuitResultPathProto]
    type: CircuitResultScalarTypeProto
    def __init__(self, name: _Optional[str] = ..., path: _Optional[_Iterable[_Union[CircuitResultPathProto, _Mapping]]] = ..., type: _Optional[_Union[CircuitResultScalarTypeProto, str]] = ...) -> None: ...

class CircuitResultApplyProto(_message.Message):
    __slots__ = ("opcode", "type", "attributes", "operands")
    OPCODE_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    OPERANDS_FIELD_NUMBER: _ClassVar[int]
    opcode: str
    type: CircuitResultScalarTypeProto
    attributes: _containers.RepeatedCompositeFieldContainer[_qcore_pb2.AttributeProto]
    operands: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, opcode: _Optional[str] = ..., type: _Optional[_Union[CircuitResultScalarTypeProto, str]] = ..., attributes: _Optional[_Iterable[_Union[_qcore_pb2.AttributeProto, _Mapping]]] = ..., operands: _Optional[_Iterable[int]] = ...) -> None: ...

class CircuitResultIndexedProto(_message.Message):
    __slots__ = ("index", "elements", "type")
    INDEX_FIELD_NUMBER: _ClassVar[int]
    ELEMENTS_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    index: int
    elements: _containers.RepeatedScalarFieldContainer[int]
    type: CircuitResultScalarTypeProto
    def __init__(self, index: _Optional[int] = ..., elements: _Optional[_Iterable[int]] = ..., type: _Optional[_Union[CircuitResultScalarTypeProto, str]] = ...) -> None: ...

class CircuitResultScalarProto(_message.Message):
    __slots__ = ("bool_value", "int_value", "real_bits", "measurement", "parameter", "apply", "current_shot", "indexed")
    BOOL_VALUE_FIELD_NUMBER: _ClassVar[int]
    INT_VALUE_FIELD_NUMBER: _ClassVar[int]
    REAL_BITS_FIELD_NUMBER: _ClassVar[int]
    MEASUREMENT_FIELD_NUMBER: _ClassVar[int]
    PARAMETER_FIELD_NUMBER: _ClassVar[int]
    APPLY_FIELD_NUMBER: _ClassVar[int]
    CURRENT_SHOT_FIELD_NUMBER: _ClassVar[int]
    INDEXED_FIELD_NUMBER: _ClassVar[int]
    bool_value: bool
    int_value: int
    real_bits: int
    measurement: int
    parameter: CircuitResultParameterProto
    apply: CircuitResultApplyProto
    current_shot: bool
    indexed: CircuitResultIndexedProto
    def __init__(self, bool_value: bool = ..., int_value: _Optional[int] = ..., real_bits: _Optional[int] = ..., measurement: _Optional[int] = ..., parameter: _Optional[_Union[CircuitResultParameterProto, _Mapping]] = ..., apply: _Optional[_Union[CircuitResultApplyProto, _Mapping]] = ..., current_shot: bool = ..., indexed: _Optional[_Union[CircuitResultIndexedProto, _Mapping]] = ...) -> None: ...

class CircuitResultSequenceProto(_message.Message):
    __slots__ = ("values",)
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedCompositeFieldContainer[CircuitResultValueProto]
    def __init__(self, values: _Optional[_Iterable[_Union[CircuitResultValueProto, _Mapping]]] = ...) -> None: ...

class CircuitResultFieldProto(_message.Message):
    __slots__ = ("name", "value")
    NAME_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    name: str
    value: CircuitResultValueProto
    def __init__(self, name: _Optional[str] = ..., value: _Optional[_Union[CircuitResultValueProto, _Mapping]] = ...) -> None: ...

class CircuitResultRecordProto(_message.Message):
    __slots__ = ("fields",)
    FIELDS_FIELD_NUMBER: _ClassVar[int]
    fields: _containers.RepeatedCompositeFieldContainer[CircuitResultFieldProto]
    def __init__(self, fields: _Optional[_Iterable[_Union[CircuitResultFieldProto, _Mapping]]] = ...) -> None: ...

class CircuitResultValueProto(_message.Message):
    __slots__ = ("unit", "literal", "scalar", "aggregate", "record")
    UNIT_FIELD_NUMBER: _ClassVar[int]
    LITERAL_FIELD_NUMBER: _ClassVar[int]
    SCALAR_FIELD_NUMBER: _ClassVar[int]
    AGGREGATE_FIELD_NUMBER: _ClassVar[int]
    RECORD_FIELD_NUMBER: _ClassVar[int]
    unit: bool
    literal: _qcore_pb2.LiteralProto
    scalar: int
    aggregate: CircuitResultSequenceProto
    record: CircuitResultRecordProto
    def __init__(self, unit: bool = ..., literal: _Optional[_Union[_qcore_pb2.LiteralProto, _Mapping]] = ..., scalar: _Optional[int] = ..., aggregate: _Optional[_Union[CircuitResultSequenceProto, _Mapping]] = ..., record: _Optional[_Union[CircuitResultRecordProto, _Mapping]] = ...) -> None: ...

class CircuitResultPostprocessingProto(_message.Message):
    __slots__ = ("schema_major", "schema_minor", "measurement_count", "scalars", "result")
    SCHEMA_MAJOR_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_MINOR_FIELD_NUMBER: _ClassVar[int]
    MEASUREMENT_COUNT_FIELD_NUMBER: _ClassVar[int]
    SCALARS_FIELD_NUMBER: _ClassVar[int]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    schema_major: int
    schema_minor: int
    measurement_count: int
    scalars: _containers.RepeatedCompositeFieldContainer[CircuitResultScalarProto]
    result: CircuitResultValueProto
    def __init__(self, schema_major: _Optional[int] = ..., schema_minor: _Optional[int] = ..., measurement_count: _Optional[int] = ..., scalars: _Optional[_Iterable[_Union[CircuitResultScalarProto, _Mapping]]] = ..., result: _Optional[_Union[CircuitResultValueProto, _Mapping]] = ...) -> None: ...

class ShotBindingSlotProto(_message.Message):
    __slots__ = ("arena", "root")
    ARENA_FIELD_NUMBER: _ClassVar[int]
    ROOT_FIELD_NUMBER: _ClassVar[int]
    arena: int
    root: int
    def __init__(self, arena: _Optional[int] = ..., root: _Optional[int] = ...) -> None: ...

class ShotCircuitTemplateProto(_message.Message):
    __slots__ = ("schema_major", "arenas", "slots", "result_plan", "parameter_abi", "result_abi", "target_fingerprint", "source_digest")
    SCHEMA_MAJOR_FIELD_NUMBER: _ClassVar[int]
    ARENAS_FIELD_NUMBER: _ClassVar[int]
    SLOTS_FIELD_NUMBER: _ClassVar[int]
    RESULT_PLAN_FIELD_NUMBER: _ClassVar[int]
    PARAMETER_ABI_FIELD_NUMBER: _ClassVar[int]
    RESULT_ABI_FIELD_NUMBER: _ClassVar[int]
    TARGET_FINGERPRINT_FIELD_NUMBER: _ClassVar[int]
    SOURCE_DIGEST_FIELD_NUMBER: _ClassVar[int]
    schema_major: int
    arenas: _containers.RepeatedCompositeFieldContainer[CircuitResultPostprocessingProto]
    slots: _containers.RepeatedCompositeFieldContainer[ShotBindingSlotProto]
    result_plan: bytes
    parameter_abi: bytes
    result_abi: bytes
    target_fingerprint: str
    source_digest: str
    def __init__(self, schema_major: _Optional[int] = ..., arenas: _Optional[_Iterable[_Union[CircuitResultPostprocessingProto, _Mapping]]] = ..., slots: _Optional[_Iterable[_Union[ShotBindingSlotProto, _Mapping]]] = ..., result_plan: _Optional[bytes] = ..., parameter_abi: _Optional[bytes] = ..., result_abi: _Optional[bytes] = ..., target_fingerprint: _Optional[str] = ..., source_digest: _Optional[str] = ...) -> None: ...
