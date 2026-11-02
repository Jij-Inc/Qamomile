from qamomile.proto import qcore_pb2 as _qcore_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ExecutionContextEntryProto(_message.Message):
    __slots__ = ("key", "value")
    KEY_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    key: str
    value: str
    def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

class ExecutionReferenceProto(_message.Message):
    __slots__ = ("provider", "job_ids", "target", "has_target", "group_id", "has_group_id", "context")
    PROVIDER_FIELD_NUMBER: _ClassVar[int]
    JOB_IDS_FIELD_NUMBER: _ClassVar[int]
    TARGET_FIELD_NUMBER: _ClassVar[int]
    HAS_TARGET_FIELD_NUMBER: _ClassVar[int]
    GROUP_ID_FIELD_NUMBER: _ClassVar[int]
    HAS_GROUP_ID_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    provider: str
    job_ids: _containers.RepeatedScalarFieldContainer[str]
    target: str
    has_target: bool
    group_id: str
    has_group_id: bool
    context: _containers.RepeatedCompositeFieldContainer[ExecutionContextEntryProto]
    def __init__(self, provider: _Optional[str] = ..., job_ids: _Optional[_Iterable[str]] = ..., target: _Optional[str] = ..., has_target: bool = ..., group_id: _Optional[str] = ..., has_group_id: bool = ..., context: _Optional[_Iterable[_Union[ExecutionContextEntryProto, _Mapping]]] = ...) -> None: ...

class ExecutionSnapshotNodeProto(_message.Message):
    __slots__ = ("kind", "reference", "local_value", "children")
    class Kind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        KIND_UNSPECIFIED: _ClassVar[ExecutionSnapshotNodeProto.Kind]
        REMOTE: _ClassVar[ExecutionSnapshotNodeProto.Kind]
        LOCAL: _ClassVar[ExecutionSnapshotNodeProto.Kind]
        COMPOSITE: _ClassVar[ExecutionSnapshotNodeProto.Kind]
    KIND_UNSPECIFIED: ExecutionSnapshotNodeProto.Kind
    REMOTE: ExecutionSnapshotNodeProto.Kind
    LOCAL: ExecutionSnapshotNodeProto.Kind
    COMPOSITE: ExecutionSnapshotNodeProto.Kind
    KIND_FIELD_NUMBER: _ClassVar[int]
    REFERENCE_FIELD_NUMBER: _ClassVar[int]
    LOCAL_VALUE_FIELD_NUMBER: _ClassVar[int]
    CHILDREN_FIELD_NUMBER: _ClassVar[int]
    kind: ExecutionSnapshotNodeProto.Kind
    reference: ExecutionReferenceProto
    local_value: _qcore_pb2.LiteralProto
    children: _containers.RepeatedCompositeFieldContainer[ExecutionSnapshotNodeProto]
    def __init__(self, kind: _Optional[_Union[ExecutionSnapshotNodeProto.Kind, str]] = ..., reference: _Optional[_Union[ExecutionReferenceProto, _Mapping]] = ..., local_value: _Optional[_Union[_qcore_pb2.LiteralProto, _Mapping]] = ..., children: _Optional[_Iterable[_Union[ExecutionSnapshotNodeProto, _Mapping]]] = ...) -> None: ...

class ExecutionSnapshotProto(_message.Message):
    __slots__ = ("schema_major", "schema_minor", "artifact_kind", "root", "snapshot_digest")
    SCHEMA_MAJOR_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_MINOR_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_KIND_FIELD_NUMBER: _ClassVar[int]
    ROOT_FIELD_NUMBER: _ClassVar[int]
    SNAPSHOT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    schema_major: int
    schema_minor: int
    artifact_kind: str
    root: ExecutionSnapshotNodeProto
    snapshot_digest: bytes
    def __init__(self, schema_major: _Optional[int] = ..., schema_minor: _Optional[int] = ..., artifact_kind: _Optional[str] = ..., root: _Optional[_Union[ExecutionSnapshotNodeProto, _Mapping]] = ..., snapshot_digest: _Optional[bytes] = ...) -> None: ...
