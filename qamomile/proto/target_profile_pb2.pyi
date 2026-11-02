from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TargetProfileProto(_message.Message):
    __slots__ = ("schema_version", "format", "topology", "calibration", "logical", "backend_name", "calibration_time", "provenance")
    SCHEMA_VERSION_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    TOPOLOGY_FIELD_NUMBER: _ClassVar[int]
    CALIBRATION_FIELD_NUMBER: _ClassVar[int]
    LOGICAL_FIELD_NUMBER: _ClassVar[int]
    BACKEND_NAME_FIELD_NUMBER: _ClassVar[int]
    CALIBRATION_TIME_FIELD_NUMBER: _ClassVar[int]
    PROVENANCE_FIELD_NUMBER: _ClassVar[int]
    schema_version: int
    format: str
    topology: TopologyProto
    calibration: CalibrationProto
    logical: LogicalProto
    backend_name: str
    calibration_time: str
    provenance: str
    def __init__(self, schema_version: _Optional[int] = ..., format: _Optional[str] = ..., topology: _Optional[_Union[TopologyProto, _Mapping]] = ..., calibration: _Optional[_Union[CalibrationProto, _Mapping]] = ..., logical: _Optional[_Union[LogicalProto, _Mapping]] = ..., backend_name: _Optional[str] = ..., calibration_time: _Optional[str] = ..., provenance: _Optional[str] = ...) -> None: ...

class CouplingEdgeProto(_message.Message):
    __slots__ = ("control", "target", "symmetric")
    CONTROL_FIELD_NUMBER: _ClassVar[int]
    TARGET_FIELD_NUMBER: _ClassVar[int]
    SYMMETRIC_FIELD_NUMBER: _ClassVar[int]
    control: int
    target: int
    symmetric: bool
    def __init__(self, control: _Optional[int] = ..., target: _Optional[int] = ..., symmetric: bool = ...) -> None: ...

class TopologyProto(_message.Message):
    __slots__ = ("num_qubits", "edges", "one_qubit_operations", "two_qubit_operations")
    NUM_QUBITS_FIELD_NUMBER: _ClassVar[int]
    EDGES_FIELD_NUMBER: _ClassVar[int]
    ONE_QUBIT_OPERATIONS_FIELD_NUMBER: _ClassVar[int]
    TWO_QUBIT_OPERATIONS_FIELD_NUMBER: _ClassVar[int]
    num_qubits: int
    edges: _containers.RepeatedCompositeFieldContainer[CouplingEdgeProto]
    one_qubit_operations: _containers.RepeatedScalarFieldContainer[str]
    two_qubit_operations: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, num_qubits: _Optional[int] = ..., edges: _Optional[_Iterable[_Union[CouplingEdgeProto, _Mapping]]] = ..., one_qubit_operations: _Optional[_Iterable[str]] = ..., two_qubit_operations: _Optional[_Iterable[str]] = ...) -> None: ...

class QubitPropertyProto(_message.Message):
    __slots__ = ("qubit", "name", "value", "unit")
    QUBIT_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    UNIT_FIELD_NUMBER: _ClassVar[int]
    qubit: int
    name: str
    value: float
    unit: str
    def __init__(self, qubit: _Optional[int] = ..., name: _Optional[str] = ..., value: _Optional[float] = ..., unit: _Optional[str] = ...) -> None: ...

class EdgePropertyProto(_message.Message):
    __slots__ = ("control", "target", "name", "value", "unit")
    CONTROL_FIELD_NUMBER: _ClassVar[int]
    TARGET_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    UNIT_FIELD_NUMBER: _ClassVar[int]
    control: int
    target: int
    name: str
    value: float
    unit: str
    def __init__(self, control: _Optional[int] = ..., target: _Optional[int] = ..., name: _Optional[str] = ..., value: _Optional[float] = ..., unit: _Optional[str] = ...) -> None: ...

class CalibrationProto(_message.Message):
    __slots__ = ("qubit_properties", "edge_properties", "provenance")
    QUBIT_PROPERTIES_FIELD_NUMBER: _ClassVar[int]
    EDGE_PROPERTIES_FIELD_NUMBER: _ClassVar[int]
    PROVENANCE_FIELD_NUMBER: _ClassVar[int]
    qubit_properties: _containers.RepeatedCompositeFieldContainer[QubitPropertyProto]
    edge_properties: _containers.RepeatedCompositeFieldContainer[EdgePropertyProto]
    provenance: str
    def __init__(self, qubit_properties: _Optional[_Iterable[_Union[QubitPropertyProto, _Mapping]]] = ..., edge_properties: _Optional[_Iterable[_Union[EdgePropertyProto, _Mapping]]] = ..., provenance: _Optional[str] = ...) -> None: ...

class NamedConstantProto(_message.Message):
    __slots__ = ("name", "value")
    NAME_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    name: str
    value: float
    def __init__(self, name: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...

class LogicalProto(_message.Message):
    __slots__ = ("code_family", "code_distance", "cycle_time", "cycle_time_unit", "magic_state_cost", "magic_state_cost_unit", "rotation_synthesis_model", "rotation_synthesis_constants")
    CODE_FAMILY_FIELD_NUMBER: _ClassVar[int]
    CODE_DISTANCE_FIELD_NUMBER: _ClassVar[int]
    CYCLE_TIME_FIELD_NUMBER: _ClassVar[int]
    CYCLE_TIME_UNIT_FIELD_NUMBER: _ClassVar[int]
    MAGIC_STATE_COST_FIELD_NUMBER: _ClassVar[int]
    MAGIC_STATE_COST_UNIT_FIELD_NUMBER: _ClassVar[int]
    ROTATION_SYNTHESIS_MODEL_FIELD_NUMBER: _ClassVar[int]
    ROTATION_SYNTHESIS_CONSTANTS_FIELD_NUMBER: _ClassVar[int]
    code_family: str
    code_distance: int
    cycle_time: float
    cycle_time_unit: str
    magic_state_cost: float
    magic_state_cost_unit: str
    rotation_synthesis_model: str
    rotation_synthesis_constants: _containers.RepeatedCompositeFieldContainer[NamedConstantProto]
    def __init__(self, code_family: _Optional[str] = ..., code_distance: _Optional[int] = ..., cycle_time: _Optional[float] = ..., cycle_time_unit: _Optional[str] = ..., magic_state_cost: _Optional[float] = ..., magic_state_cost_unit: _Optional[str] = ..., rotation_synthesis_model: _Optional[str] = ..., rotation_synthesis_constants: _Optional[_Iterable[_Union[NamedConstantProto, _Mapping]]] = ...) -> None: ...
