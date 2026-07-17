from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ParameterKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    PARAMETER_KIND_UNSPECIFIED: _ClassVar[ParameterKind]
    POSITIONAL_ONLY: _ClassVar[ParameterKind]
    POSITIONAL_OR_KEYWORD: _ClassVar[ParameterKind]
    VAR_POSITIONAL: _ClassVar[ParameterKind]
    KEYWORD_ONLY: _ClassVar[ParameterKind]
    VAR_KEYWORD: _ClassVar[ParameterKind]

class FrontendAnnotationKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    FRONTEND_ANNOTATION_KIND_UNSPECIFIED: _ClassVar[FrontendAnnotationKind]
    QAMOMILE_UINT: _ClassVar[FrontendAnnotationKind]
    PYTHON_INT: _ClassVar[FrontendAnnotationKind]
    QAMOMILE_FLOAT: _ClassVar[FrontendAnnotationKind]
    PYTHON_FLOAT: _ClassVar[FrontendAnnotationKind]
    QAMOMILE_BIT: _ClassVar[FrontendAnnotationKind]
    PYTHON_BOOL: _ClassVar[FrontendAnnotationKind]
    QAMOMILE_QUBIT: _ClassVar[FrontendAnnotationKind]
    QAMOMILE_QFIXED: _ClassVar[FrontendAnnotationKind]
    QAMOMILE_OBSERVABLE: _ClassVar[FrontendAnnotationKind]
    QAMOMILE_VECTOR: _ClassVar[FrontendAnnotationKind]
    QAMOMILE_MATRIX: _ClassVar[FrontendAnnotationKind]
    QAMOMILE_TENSOR: _ClassVar[FrontendAnnotationKind]
    QAMOMILE_TUPLE: _ClassVar[FrontendAnnotationKind]
    QAMOMILE_DICT: _ClassVar[FrontendAnnotationKind]
    PYTHON_TUPLE: _ClassVar[FrontendAnnotationKind]

class ValueKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    VALUE_KIND_UNSPECIFIED: _ClassVar[ValueKind]
    VALUE: _ClassVar[ValueKind]
    ARRAY_VALUE: _ClassVar[ValueKind]
    TUPLE_VALUE: _ClassVar[ValueKind]
    DICT_VALUE: _ClassVar[ValueKind]

class ValueTypeKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    VALUE_TYPE_KIND_UNSPECIFIED: _ClassVar[ValueTypeKind]
    UINT_TYPE: _ClassVar[ValueTypeKind]
    FLOAT_TYPE: _ClassVar[ValueTypeKind]
    BIT_TYPE: _ClassVar[ValueTypeKind]
    QUBIT_TYPE: _ClassVar[ValueTypeKind]
    BLOCK_TYPE: _ClassVar[ValueTypeKind]
    OBSERVABLE_TYPE: _ClassVar[ValueTypeKind]
    TUPLE_TYPE: _ClassVar[ValueTypeKind]
    DICT_TYPE: _ClassVar[ValueTypeKind]
    QFIXED_TYPE: _ClassVar[ValueTypeKind]
    QUINT_TYPE: _ClassVar[ValueTypeKind]

class OperationType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    OPERATION_TYPE_UNSPECIFIED: _ClassVar[OperationType]
    GATE_OPERATION: _ClassVar[OperationType]
    MEASURE_OPERATION: _ClassVar[OperationType]
    PROJECT_OPERATION: _ClassVar[OperationType]
    RESET_OPERATION: _ClassVar[OperationType]
    MEASURE_VECTOR_OPERATION: _ClassVar[OperationType]
    MEASURE_QFIXED_OPERATION: _ClassVar[OperationType]
    DECODE_QFIXED_OPERATION: _ClassVar[OperationType]
    STORE_ARRAY_ELEMENT_OPERATION: _ClassVar[OperationType]
    DICT_GET_ITEM_OPERATION: _ClassVar[OperationType]
    CAST_OPERATION: _ClassVar[OperationType]
    QINIT_OPERATION: _ClassVar[OperationType]
    CINIT_OPERATION: _ClassVar[OperationType]
    SLICE_ARRAY_OPERATION: _ClassVar[OperationType]
    RELEASE_SLICE_VIEW_OPERATION: _ClassVar[OperationType]
    RETURN_OPERATION: _ClassVar[OperationType]
    EXPVAL_OPERATION: _ClassVar[OperationType]
    PAULI_EVOLVE_OPERATION: _ClassVar[OperationType]
    BIN_OPERATION: _ClassVar[OperationType]
    COMP_OPERATION: _ClassVar[OperationType]
    COND_OPERATION: _ClassVar[OperationType]
    NOT_OPERATION: _ClassVar[OperationType]
    RUNTIME_CLASSICAL_OPERATION: _ClassVar[OperationType]
    FOR_OPERATION: _ClassVar[OperationType]
    FOR_ITEMS_OPERATION: _ClassVar[OperationType]
    WHILE_OPERATION: _ClassVar[OperationType]
    IF_OPERATION: _ClassVar[OperationType]
    CONCRETE_CONTROLLED_OPERATION: _ClassVar[OperationType]
    SYMBOLIC_CONTROLLED_OPERATION: _ClassVar[OperationType]
    INVOKE_OPERATION: _ClassVar[OperationType]
    INVERSE_BLOCK_OPERATION: _ClassVar[OperationType]
    GLOBAL_PHASE_OPERATION: _ClassVar[OperationType]
    SELECT_OPERATION: _ClassVar[OperationType]
    RETURN_QUANTUM_ARRAY_ELEMENT_OPERATION: _ClassVar[OperationType]
PARAMETER_KIND_UNSPECIFIED: ParameterKind
POSITIONAL_ONLY: ParameterKind
POSITIONAL_OR_KEYWORD: ParameterKind
VAR_POSITIONAL: ParameterKind
KEYWORD_ONLY: ParameterKind
VAR_KEYWORD: ParameterKind
FRONTEND_ANNOTATION_KIND_UNSPECIFIED: FrontendAnnotationKind
QAMOMILE_UINT: FrontendAnnotationKind
PYTHON_INT: FrontendAnnotationKind
QAMOMILE_FLOAT: FrontendAnnotationKind
PYTHON_FLOAT: FrontendAnnotationKind
QAMOMILE_BIT: FrontendAnnotationKind
PYTHON_BOOL: FrontendAnnotationKind
QAMOMILE_QUBIT: FrontendAnnotationKind
QAMOMILE_QFIXED: FrontendAnnotationKind
QAMOMILE_OBSERVABLE: FrontendAnnotationKind
QAMOMILE_VECTOR: FrontendAnnotationKind
QAMOMILE_MATRIX: FrontendAnnotationKind
QAMOMILE_TENSOR: FrontendAnnotationKind
QAMOMILE_TUPLE: FrontendAnnotationKind
QAMOMILE_DICT: FrontendAnnotationKind
PYTHON_TUPLE: FrontendAnnotationKind
VALUE_KIND_UNSPECIFIED: ValueKind
VALUE: ValueKind
ARRAY_VALUE: ValueKind
TUPLE_VALUE: ValueKind
DICT_VALUE: ValueKind
VALUE_TYPE_KIND_UNSPECIFIED: ValueTypeKind
UINT_TYPE: ValueTypeKind
FLOAT_TYPE: ValueTypeKind
BIT_TYPE: ValueTypeKind
QUBIT_TYPE: ValueTypeKind
BLOCK_TYPE: ValueTypeKind
OBSERVABLE_TYPE: ValueTypeKind
TUPLE_TYPE: ValueTypeKind
DICT_TYPE: ValueTypeKind
QFIXED_TYPE: ValueTypeKind
QUINT_TYPE: ValueTypeKind
OPERATION_TYPE_UNSPECIFIED: OperationType
GATE_OPERATION: OperationType
MEASURE_OPERATION: OperationType
PROJECT_OPERATION: OperationType
RESET_OPERATION: OperationType
MEASURE_VECTOR_OPERATION: OperationType
MEASURE_QFIXED_OPERATION: OperationType
DECODE_QFIXED_OPERATION: OperationType
STORE_ARRAY_ELEMENT_OPERATION: OperationType
DICT_GET_ITEM_OPERATION: OperationType
CAST_OPERATION: OperationType
QINIT_OPERATION: OperationType
CINIT_OPERATION: OperationType
SLICE_ARRAY_OPERATION: OperationType
RELEASE_SLICE_VIEW_OPERATION: OperationType
RETURN_OPERATION: OperationType
EXPVAL_OPERATION: OperationType
PAULI_EVOLVE_OPERATION: OperationType
BIN_OPERATION: OperationType
COMP_OPERATION: OperationType
COND_OPERATION: OperationType
NOT_OPERATION: OperationType
RUNTIME_CLASSICAL_OPERATION: OperationType
FOR_OPERATION: OperationType
FOR_ITEMS_OPERATION: OperationType
WHILE_OPERATION: OperationType
IF_OPERATION: OperationType
CONCRETE_CONTROLLED_OPERATION: OperationType
SYMBOLIC_CONTROLLED_OPERATION: OperationType
INVOKE_OPERATION: OperationType
INVERSE_BLOCK_OPERATION: OperationType
GLOBAL_PHASE_OPERATION: OperationType
SELECT_OPERATION: OperationType
RETURN_QUANTUM_ARRAY_ELEMENT_OPERATION: OperationType

class QKernel(_message.Message):
    __slots__ = ("qamomile_version", "name", "parameters", "results", "body", "value_table", "callable_table", "callable_definition", "return_annotation")
    QAMOMILE_VERSION_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    BODY_FIELD_NUMBER: _ClassVar[int]
    VALUE_TABLE_FIELD_NUMBER: _ClassVar[int]
    CALLABLE_TABLE_FIELD_NUMBER: _ClassVar[int]
    CALLABLE_DEFINITION_FIELD_NUMBER: _ClassVar[int]
    RETURN_ANNOTATION_FIELD_NUMBER: _ClassVar[int]
    qamomile_version: str
    name: str
    parameters: _containers.RepeatedCompositeFieldContainer[KernelParameter]
    results: _containers.RepeatedCompositeFieldContainer[KernelType]
    body: Block
    value_table: _containers.RepeatedCompositeFieldContainer[ValueNode]
    callable_table: _containers.RepeatedCompositeFieldContainer[CallableEntry]
    callable_definition: CallableDefinition
    return_annotation: FrontendAnnotation
    def __init__(self, qamomile_version: _Optional[str] = ..., name: _Optional[str] = ..., parameters: _Optional[_Iterable[_Union[KernelParameter, _Mapping]]] = ..., results: _Optional[_Iterable[_Union[KernelType, _Mapping]]] = ..., body: _Optional[_Union[Block, _Mapping]] = ..., value_table: _Optional[_Iterable[_Union[ValueNode, _Mapping]]] = ..., callable_table: _Optional[_Iterable[_Union[CallableEntry, _Mapping]]] = ..., callable_definition: _Optional[_Union[CallableDefinition, _Mapping]] = ..., return_annotation: _Optional[_Union[FrontendAnnotation, _Mapping]] = ...) -> None: ...

class PauliLCUBlockEncodingArtifact(_message.Message):
    __slots__ = ("artifact_kind", "qamomile_version", "num_qubits", "terms")
    ARTIFACT_KIND_FIELD_NUMBER: _ClassVar[int]
    QAMOMILE_VERSION_FIELD_NUMBER: _ClassVar[int]
    NUM_QUBITS_FIELD_NUMBER: _ClassVar[int]
    TERMS_FIELD_NUMBER: _ClassVar[int]
    artifact_kind: str
    qamomile_version: str
    num_qubits: BigInteger
    terms: _containers.RepeatedCompositeFieldContainer[PauliLCUBlockEncodingTerm]
    def __init__(self, artifact_kind: _Optional[str] = ..., qamomile_version: _Optional[str] = ..., num_qubits: _Optional[_Union[BigInteger, _Mapping]] = ..., terms: _Optional[_Iterable[_Union[PauliLCUBlockEncodingTerm, _Mapping]]] = ...) -> None: ...

class PauliLCUBlockEncodingTerm(_message.Message):
    __slots__ = ("coefficient", "operators")
    COEFFICIENT_FIELD_NUMBER: _ClassVar[int]
    OPERATORS_FIELD_NUMBER: _ClassVar[int]
    coefficient: Complex64
    operators: _containers.RepeatedCompositeFieldContainer[PauliLCUBlockEncodingOperator]
    def __init__(self, coefficient: _Optional[_Union[Complex64, _Mapping]] = ..., operators: _Optional[_Iterable[_Union[PauliLCUBlockEncodingOperator, _Mapping]]] = ...) -> None: ...

class PauliLCUBlockEncodingOperator(_message.Message):
    __slots__ = ("pauli", "index")
    PAULI_FIELD_NUMBER: _ClassVar[int]
    INDEX_FIELD_NUMBER: _ClassVar[int]
    pauli: str
    index: BigInteger
    def __init__(self, pauli: _Optional[str] = ..., index: _Optional[_Union[BigInteger, _Mapping]] = ...) -> None: ...

class KernelParameter(_message.Message):
    __slots__ = ("name", "type", "kind", "has_default", "default", "differentiable")
    NAME_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    HAS_DEFAULT_FIELD_NUMBER: _ClassVar[int]
    DEFAULT_FIELD_NUMBER: _ClassVar[int]
    DIFFERENTIABLE_FIELD_NUMBER: _ClassVar[int]
    name: str
    type: KernelType
    kind: ParameterKind
    has_default: bool
    default: Payload
    differentiable: bool
    def __init__(self, name: _Optional[str] = ..., type: _Optional[_Union[KernelType, _Mapping]] = ..., kind: _Optional[_Union[ParameterKind, str]] = ..., has_default: bool = ..., default: _Optional[_Union[Payload, _Mapping]] = ..., differentiable: bool = ...) -> None: ...

class KernelType(_message.Message):
    __slots__ = ("value_type", "ndim", "annotation")
    VALUE_TYPE_FIELD_NUMBER: _ClassVar[int]
    NDIM_FIELD_NUMBER: _ClassVar[int]
    ANNOTATION_FIELD_NUMBER: _ClassVar[int]
    value_type: ValueType
    ndim: int
    annotation: FrontendAnnotation
    def __init__(self, value_type: _Optional[_Union[ValueType, _Mapping]] = ..., ndim: _Optional[int] = ..., annotation: _Optional[_Union[FrontendAnnotation, _Mapping]] = ...) -> None: ...

class FrontendAnnotation(_message.Message):
    __slots__ = ("kind", "arguments")
    KIND_FIELD_NUMBER: _ClassVar[int]
    ARGUMENTS_FIELD_NUMBER: _ClassVar[int]
    kind: FrontendAnnotationKind
    arguments: _containers.RepeatedCompositeFieldContainer[FrontendAnnotation]
    def __init__(self, kind: _Optional[_Union[FrontendAnnotationKind, str]] = ..., arguments: _Optional[_Iterable[_Union[FrontendAnnotation, _Mapping]]] = ...) -> None: ...

class Block(_message.Message):
    __slots__ = ("kind", "name", "label_args", "input_value_refs", "output_value_refs", "output_names", "parameters", "operations")
    KIND_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    LABEL_ARGS_FIELD_NUMBER: _ClassVar[int]
    INPUT_VALUE_REFS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_VALUE_REFS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_NAMES_FIELD_NUMBER: _ClassVar[int]
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    OPERATIONS_FIELD_NUMBER: _ClassVar[int]
    kind: str
    name: str
    label_args: _containers.RepeatedCompositeFieldContainer[Payload]
    input_value_refs: _containers.RepeatedScalarFieldContainer[str]
    output_value_refs: _containers.RepeatedScalarFieldContainer[str]
    output_names: _containers.RepeatedScalarFieldContainer[str]
    parameters: _containers.RepeatedCompositeFieldContainer[NamedReference]
    operations: _containers.RepeatedCompositeFieldContainer[Operation]
    def __init__(self, kind: _Optional[str] = ..., name: _Optional[str] = ..., label_args: _Optional[_Iterable[_Union[Payload, _Mapping]]] = ..., input_value_refs: _Optional[_Iterable[str]] = ..., output_value_refs: _Optional[_Iterable[str]] = ..., output_names: _Optional[_Iterable[str]] = ..., parameters: _Optional[_Iterable[_Union[NamedReference, _Mapping]]] = ..., operations: _Optional[_Iterable[_Union[Operation, _Mapping]]] = ...) -> None: ...

class NamedReference(_message.Message):
    __slots__ = ("name", "value_ref")
    NAME_FIELD_NUMBER: _ClassVar[int]
    VALUE_REF_FIELD_NUMBER: _ClassVar[int]
    name: str
    value_ref: str
    def __init__(self, name: _Optional[str] = ..., value_ref: _Optional[str] = ...) -> None: ...

class ValueNode(_message.Message):
    __slots__ = ("value_kind", "uuid", "logical_id", "name", "version", "value_type", "metadata", "parent_array_ref", "element_index_refs", "shape_refs", "slice_of_ref", "slice_start_ref", "slice_step_ref", "element_refs", "entry_refs")
    VALUE_KIND_FIELD_NUMBER: _ClassVar[int]
    UUID_FIELD_NUMBER: _ClassVar[int]
    LOGICAL_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    VALUE_TYPE_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    PARENT_ARRAY_REF_FIELD_NUMBER: _ClassVar[int]
    ELEMENT_INDEX_REFS_FIELD_NUMBER: _ClassVar[int]
    SHAPE_REFS_FIELD_NUMBER: _ClassVar[int]
    SLICE_OF_REF_FIELD_NUMBER: _ClassVar[int]
    SLICE_START_REF_FIELD_NUMBER: _ClassVar[int]
    SLICE_STEP_REF_FIELD_NUMBER: _ClassVar[int]
    ELEMENT_REFS_FIELD_NUMBER: _ClassVar[int]
    ENTRY_REFS_FIELD_NUMBER: _ClassVar[int]
    value_kind: ValueKind
    uuid: str
    logical_id: str
    name: str
    version: int
    value_type: ValueType
    metadata: ValueMetadata
    parent_array_ref: str
    element_index_refs: _containers.RepeatedScalarFieldContainer[str]
    shape_refs: _containers.RepeatedScalarFieldContainer[str]
    slice_of_ref: str
    slice_start_ref: str
    slice_step_ref: str
    element_refs: _containers.RepeatedScalarFieldContainer[str]
    entry_refs: _containers.RepeatedCompositeFieldContainer[ReferencePair]
    def __init__(self, value_kind: _Optional[_Union[ValueKind, str]] = ..., uuid: _Optional[str] = ..., logical_id: _Optional[str] = ..., name: _Optional[str] = ..., version: _Optional[int] = ..., value_type: _Optional[_Union[ValueType, _Mapping]] = ..., metadata: _Optional[_Union[ValueMetadata, _Mapping]] = ..., parent_array_ref: _Optional[str] = ..., element_index_refs: _Optional[_Iterable[str]] = ..., shape_refs: _Optional[_Iterable[str]] = ..., slice_of_ref: _Optional[str] = ..., slice_start_ref: _Optional[str] = ..., slice_step_ref: _Optional[str] = ..., element_refs: _Optional[_Iterable[str]] = ..., entry_refs: _Optional[_Iterable[_Union[ReferencePair, _Mapping]]] = ...) -> None: ...

class ReferencePair(_message.Message):
    __slots__ = ("key_ref", "value_ref")
    KEY_REF_FIELD_NUMBER: _ClassVar[int]
    VALUE_REF_FIELD_NUMBER: _ClassVar[int]
    key_ref: str
    value_ref: str
    def __init__(self, key_ref: _Optional[str] = ..., value_ref: _Optional[str] = ...) -> None: ...

class ValueMetadata(_message.Message):
    __slots__ = ("scalar", "cast", "qfixed", "array_runtime", "dict_runtime")
    SCALAR_FIELD_NUMBER: _ClassVar[int]
    CAST_FIELD_NUMBER: _ClassVar[int]
    QFIXED_FIELD_NUMBER: _ClassVar[int]
    ARRAY_RUNTIME_FIELD_NUMBER: _ClassVar[int]
    DICT_RUNTIME_FIELD_NUMBER: _ClassVar[int]
    scalar: ScalarMetadata
    cast: CastMetadata
    qfixed: QFixedMetadata
    array_runtime: ArrayRuntimeMetadata
    dict_runtime: DictRuntimeMetadata
    def __init__(self, scalar: _Optional[_Union[ScalarMetadata, _Mapping]] = ..., cast: _Optional[_Union[CastMetadata, _Mapping]] = ..., qfixed: _Optional[_Union[QFixedMetadata, _Mapping]] = ..., array_runtime: _Optional[_Union[ArrayRuntimeMetadata, _Mapping]] = ..., dict_runtime: _Optional[_Union[DictRuntimeMetadata, _Mapping]] = ...) -> None: ...

class ScalarMetadata(_message.Message):
    __slots__ = ("const_value", "parameter_name")
    CONST_VALUE_FIELD_NUMBER: _ClassVar[int]
    PARAMETER_NAME_FIELD_NUMBER: _ClassVar[int]
    const_value: Payload
    parameter_name: str
    def __init__(self, const_value: _Optional[_Union[Payload, _Mapping]] = ..., parameter_name: _Optional[str] = ...) -> None: ...

class CastMetadata(_message.Message):
    __slots__ = ("source_uuid", "qubit_uuids", "source_logical_id", "qubit_logical_ids")
    SOURCE_UUID_FIELD_NUMBER: _ClassVar[int]
    QUBIT_UUIDS_FIELD_NUMBER: _ClassVar[int]
    SOURCE_LOGICAL_ID_FIELD_NUMBER: _ClassVar[int]
    QUBIT_LOGICAL_IDS_FIELD_NUMBER: _ClassVar[int]
    source_uuid: str
    qubit_uuids: _containers.RepeatedScalarFieldContainer[str]
    source_logical_id: str
    qubit_logical_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, source_uuid: _Optional[str] = ..., qubit_uuids: _Optional[_Iterable[str]] = ..., source_logical_id: _Optional[str] = ..., qubit_logical_ids: _Optional[_Iterable[str]] = ...) -> None: ...

class QFixedMetadata(_message.Message):
    __slots__ = ("qubit_uuids", "num_bits", "int_bits")
    QUBIT_UUIDS_FIELD_NUMBER: _ClassVar[int]
    NUM_BITS_FIELD_NUMBER: _ClassVar[int]
    INT_BITS_FIELD_NUMBER: _ClassVar[int]
    qubit_uuids: _containers.RepeatedScalarFieldContainer[str]
    num_bits: int
    int_bits: int
    def __init__(self, qubit_uuids: _Optional[_Iterable[str]] = ..., num_bits: _Optional[int] = ..., int_bits: _Optional[int] = ...) -> None: ...

class ArrayRuntimeMetadata(_message.Message):
    __slots__ = ("const_array", "element_uuids", "element_logical_ids", "element_parent_uuids", "element_parent_indices")
    CONST_ARRAY_FIELD_NUMBER: _ClassVar[int]
    ELEMENT_UUIDS_FIELD_NUMBER: _ClassVar[int]
    ELEMENT_LOGICAL_IDS_FIELD_NUMBER: _ClassVar[int]
    ELEMENT_PARENT_UUIDS_FIELD_NUMBER: _ClassVar[int]
    ELEMENT_PARENT_INDICES_FIELD_NUMBER: _ClassVar[int]
    const_array: Payload
    element_uuids: _containers.RepeatedScalarFieldContainer[str]
    element_logical_ids: _containers.RepeatedScalarFieldContainer[str]
    element_parent_uuids: _containers.RepeatedCompositeFieldContainer[OptionalString]
    element_parent_indices: _containers.RepeatedCompositeFieldContainer[OptionalInteger]
    def __init__(self, const_array: _Optional[_Union[Payload, _Mapping]] = ..., element_uuids: _Optional[_Iterable[str]] = ..., element_logical_ids: _Optional[_Iterable[str]] = ..., element_parent_uuids: _Optional[_Iterable[_Union[OptionalString, _Mapping]]] = ..., element_parent_indices: _Optional[_Iterable[_Union[OptionalInteger, _Mapping]]] = ...) -> None: ...

class DictRuntimeMetadata(_message.Message):
    __slots__ = ("bound_data",)
    BOUND_DATA_FIELD_NUMBER: _ClassVar[int]
    bound_data: _containers.RepeatedCompositeFieldContainer[PayloadEntry]
    def __init__(self, bound_data: _Optional[_Iterable[_Union[PayloadEntry, _Mapping]]] = ...) -> None: ...

class OptionalString(_message.Message):
    __slots__ = ("value",)
    VALUE_FIELD_NUMBER: _ClassVar[int]
    value: str
    def __init__(self, value: _Optional[str] = ...) -> None: ...

class OptionalInteger(_message.Message):
    __slots__ = ("value",)
    VALUE_FIELD_NUMBER: _ClassVar[int]
    value: int
    def __init__(self, value: _Optional[int] = ...) -> None: ...

class ValueType(_message.Message):
    __slots__ = ("kind", "element_types", "key_type", "value_type", "integer_bits", "fractional_bits", "width")
    KIND_FIELD_NUMBER: _ClassVar[int]
    ELEMENT_TYPES_FIELD_NUMBER: _ClassVar[int]
    KEY_TYPE_FIELD_NUMBER: _ClassVar[int]
    VALUE_TYPE_FIELD_NUMBER: _ClassVar[int]
    INTEGER_BITS_FIELD_NUMBER: _ClassVar[int]
    FRACTIONAL_BITS_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    kind: ValueTypeKind
    element_types: _containers.RepeatedCompositeFieldContainer[ValueType]
    key_type: ValueType
    value_type: ValueType
    integer_bits: RegisterWidth
    fractional_bits: RegisterWidth
    width: RegisterWidth
    def __init__(self, kind: _Optional[_Union[ValueTypeKind, str]] = ..., element_types: _Optional[_Iterable[_Union[ValueType, _Mapping]]] = ..., key_type: _Optional[_Union[ValueType, _Mapping]] = ..., value_type: _Optional[_Union[ValueType, _Mapping]] = ..., integer_bits: _Optional[_Union[RegisterWidth, _Mapping]] = ..., fractional_bits: _Optional[_Union[RegisterWidth, _Mapping]] = ..., width: _Optional[_Union[RegisterWidth, _Mapping]] = ...) -> None: ...

class RegisterWidth(_message.Message):
    __slots__ = ("concrete", "value_ref")
    CONCRETE_FIELD_NUMBER: _ClassVar[int]
    VALUE_REF_FIELD_NUMBER: _ClassVar[int]
    concrete: int
    value_ref: str
    def __init__(self, concrete: _Optional[int] = ..., value_ref: _Optional[str] = ...) -> None: ...

class Operation(_message.Message):
    __slots__ = ("operation_type", "operand_refs", "result_refs", "gate_type", "axis", "num_bits", "int_bits", "key_arity", "source_type", "target_type", "qubit_mapping", "expression_kind", "loop_var", "loop_var_value_ref", "key_vars", "value_var", "key_is_vector", "key_var_value_refs", "has_key_var_value_refs", "value_var_value_ref", "max_iterations", "loop_carried_rebinds", "region_args", "body", "true_body", "false_body", "true_yield_refs", "false_yield_refs", "branch_rebinds", "num_controls", "num_controls_ref", "power", "control_index_refs", "has_control_index_refs", "num_control_args", "unitary_block", "callable_ref", "callable_attrs", "control_value", "target", "transform", "attrs", "definition_ref", "num_control_qubits", "num_target_qubits", "custom_name", "source_block", "implementation_block", "num_index_qubits", "case_blocks", "num_index_qubits_ref", "num_index_args")
    OPERATION_TYPE_FIELD_NUMBER: _ClassVar[int]
    OPERAND_REFS_FIELD_NUMBER: _ClassVar[int]
    RESULT_REFS_FIELD_NUMBER: _ClassVar[int]
    GATE_TYPE_FIELD_NUMBER: _ClassVar[int]
    AXIS_FIELD_NUMBER: _ClassVar[int]
    NUM_BITS_FIELD_NUMBER: _ClassVar[int]
    INT_BITS_FIELD_NUMBER: _ClassVar[int]
    KEY_ARITY_FIELD_NUMBER: _ClassVar[int]
    SOURCE_TYPE_FIELD_NUMBER: _ClassVar[int]
    TARGET_TYPE_FIELD_NUMBER: _ClassVar[int]
    QUBIT_MAPPING_FIELD_NUMBER: _ClassVar[int]
    EXPRESSION_KIND_FIELD_NUMBER: _ClassVar[int]
    LOOP_VAR_FIELD_NUMBER: _ClassVar[int]
    LOOP_VAR_VALUE_REF_FIELD_NUMBER: _ClassVar[int]
    KEY_VARS_FIELD_NUMBER: _ClassVar[int]
    VALUE_VAR_FIELD_NUMBER: _ClassVar[int]
    KEY_IS_VECTOR_FIELD_NUMBER: _ClassVar[int]
    KEY_VAR_VALUE_REFS_FIELD_NUMBER: _ClassVar[int]
    HAS_KEY_VAR_VALUE_REFS_FIELD_NUMBER: _ClassVar[int]
    VALUE_VAR_VALUE_REF_FIELD_NUMBER: _ClassVar[int]
    MAX_ITERATIONS_FIELD_NUMBER: _ClassVar[int]
    LOOP_CARRIED_REBINDS_FIELD_NUMBER: _ClassVar[int]
    REGION_ARGS_FIELD_NUMBER: _ClassVar[int]
    BODY_FIELD_NUMBER: _ClassVar[int]
    TRUE_BODY_FIELD_NUMBER: _ClassVar[int]
    FALSE_BODY_FIELD_NUMBER: _ClassVar[int]
    TRUE_YIELD_REFS_FIELD_NUMBER: _ClassVar[int]
    FALSE_YIELD_REFS_FIELD_NUMBER: _ClassVar[int]
    BRANCH_REBINDS_FIELD_NUMBER: _ClassVar[int]
    NUM_CONTROLS_FIELD_NUMBER: _ClassVar[int]
    NUM_CONTROLS_REF_FIELD_NUMBER: _ClassVar[int]
    POWER_FIELD_NUMBER: _ClassVar[int]
    CONTROL_INDEX_REFS_FIELD_NUMBER: _ClassVar[int]
    HAS_CONTROL_INDEX_REFS_FIELD_NUMBER: _ClassVar[int]
    NUM_CONTROL_ARGS_FIELD_NUMBER: _ClassVar[int]
    UNITARY_BLOCK_FIELD_NUMBER: _ClassVar[int]
    CALLABLE_REF_FIELD_NUMBER: _ClassVar[int]
    CALLABLE_ATTRS_FIELD_NUMBER: _ClassVar[int]
    CONTROL_VALUE_FIELD_NUMBER: _ClassVar[int]
    TARGET_FIELD_NUMBER: _ClassVar[int]
    TRANSFORM_FIELD_NUMBER: _ClassVar[int]
    ATTRS_FIELD_NUMBER: _ClassVar[int]
    DEFINITION_REF_FIELD_NUMBER: _ClassVar[int]
    NUM_CONTROL_QUBITS_FIELD_NUMBER: _ClassVar[int]
    NUM_TARGET_QUBITS_FIELD_NUMBER: _ClassVar[int]
    CUSTOM_NAME_FIELD_NUMBER: _ClassVar[int]
    SOURCE_BLOCK_FIELD_NUMBER: _ClassVar[int]
    IMPLEMENTATION_BLOCK_FIELD_NUMBER: _ClassVar[int]
    NUM_INDEX_QUBITS_FIELD_NUMBER: _ClassVar[int]
    CASE_BLOCKS_FIELD_NUMBER: _ClassVar[int]
    NUM_INDEX_QUBITS_REF_FIELD_NUMBER: _ClassVar[int]
    NUM_INDEX_ARGS_FIELD_NUMBER: _ClassVar[int]
    operation_type: OperationType
    operand_refs: _containers.RepeatedScalarFieldContainer[str]
    result_refs: _containers.RepeatedScalarFieldContainer[str]
    gate_type: str
    axis: str
    num_bits: int
    int_bits: int
    key_arity: int
    source_type: ValueType
    target_type: ValueType
    qubit_mapping: _containers.RepeatedScalarFieldContainer[str]
    expression_kind: str
    loop_var: str
    loop_var_value_ref: str
    key_vars: _containers.RepeatedScalarFieldContainer[str]
    value_var: str
    key_is_vector: bool
    key_var_value_refs: _containers.RepeatedScalarFieldContainer[str]
    has_key_var_value_refs: bool
    value_var_value_ref: str
    max_iterations: int
    loop_carried_rebinds: _containers.RepeatedCompositeFieldContainer[LoopCarriedRebind]
    region_args: _containers.RepeatedCompositeFieldContainer[RegionArg]
    body: _containers.RepeatedCompositeFieldContainer[Operation]
    true_body: _containers.RepeatedCompositeFieldContainer[Operation]
    false_body: _containers.RepeatedCompositeFieldContainer[Operation]
    true_yield_refs: _containers.RepeatedScalarFieldContainer[str]
    false_yield_refs: _containers.RepeatedScalarFieldContainer[str]
    branch_rebinds: _containers.RepeatedCompositeFieldContainer[BranchRebind]
    num_controls: int
    num_controls_ref: str
    power: IntegerOrReference
    control_index_refs: _containers.RepeatedScalarFieldContainer[str]
    has_control_index_refs: bool
    num_control_args: int
    unitary_block: Block
    callable_ref: CallableRef
    callable_attrs: Payload
    control_value: BigInteger
    target: CallableRef
    transform: str
    attrs: Payload
    definition_ref: str
    num_control_qubits: int
    num_target_qubits: int
    custom_name: str
    source_block: Block
    implementation_block: Block
    num_index_qubits: int
    case_blocks: _containers.RepeatedCompositeFieldContainer[Block]
    num_index_qubits_ref: str
    num_index_args: int
    def __init__(self, operation_type: _Optional[_Union[OperationType, str]] = ..., operand_refs: _Optional[_Iterable[str]] = ..., result_refs: _Optional[_Iterable[str]] = ..., gate_type: _Optional[str] = ..., axis: _Optional[str] = ..., num_bits: _Optional[int] = ..., int_bits: _Optional[int] = ..., key_arity: _Optional[int] = ..., source_type: _Optional[_Union[ValueType, _Mapping]] = ..., target_type: _Optional[_Union[ValueType, _Mapping]] = ..., qubit_mapping: _Optional[_Iterable[str]] = ..., expression_kind: _Optional[str] = ..., loop_var: _Optional[str] = ..., loop_var_value_ref: _Optional[str] = ..., key_vars: _Optional[_Iterable[str]] = ..., value_var: _Optional[str] = ..., key_is_vector: bool = ..., key_var_value_refs: _Optional[_Iterable[str]] = ..., has_key_var_value_refs: bool = ..., value_var_value_ref: _Optional[str] = ..., max_iterations: _Optional[int] = ..., loop_carried_rebinds: _Optional[_Iterable[_Union[LoopCarriedRebind, _Mapping]]] = ..., region_args: _Optional[_Iterable[_Union[RegionArg, _Mapping]]] = ..., body: _Optional[_Iterable[_Union[Operation, _Mapping]]] = ..., true_body: _Optional[_Iterable[_Union[Operation, _Mapping]]] = ..., false_body: _Optional[_Iterable[_Union[Operation, _Mapping]]] = ..., true_yield_refs: _Optional[_Iterable[str]] = ..., false_yield_refs: _Optional[_Iterable[str]] = ..., branch_rebinds: _Optional[_Iterable[_Union[BranchRebind, _Mapping]]] = ..., num_controls: _Optional[int] = ..., num_controls_ref: _Optional[str] = ..., power: _Optional[_Union[IntegerOrReference, _Mapping]] = ..., control_index_refs: _Optional[_Iterable[str]] = ..., has_control_index_refs: bool = ..., num_control_args: _Optional[int] = ..., unitary_block: _Optional[_Union[Block, _Mapping]] = ..., callable_ref: _Optional[_Union[CallableRef, _Mapping]] = ..., callable_attrs: _Optional[_Union[Payload, _Mapping]] = ..., control_value: _Optional[_Union[BigInteger, _Mapping]] = ..., target: _Optional[_Union[CallableRef, _Mapping]] = ..., transform: _Optional[str] = ..., attrs: _Optional[_Union[Payload, _Mapping]] = ..., definition_ref: _Optional[str] = ..., num_control_qubits: _Optional[int] = ..., num_target_qubits: _Optional[int] = ..., custom_name: _Optional[str] = ..., source_block: _Optional[_Union[Block, _Mapping]] = ..., implementation_block: _Optional[_Union[Block, _Mapping]] = ..., num_index_qubits: _Optional[int] = ..., case_blocks: _Optional[_Iterable[_Union[Block, _Mapping]]] = ..., num_index_qubits_ref: _Optional[str] = ..., num_index_args: _Optional[int] = ...) -> None: ...

class IntegerOrReference(_message.Message):
    __slots__ = ("integer", "value_ref")
    INTEGER_FIELD_NUMBER: _ClassVar[int]
    VALUE_REF_FIELD_NUMBER: _ClassVar[int]
    integer: BigInteger
    value_ref: str
    def __init__(self, integer: _Optional[_Union[BigInteger, _Mapping]] = ..., value_ref: _Optional[str] = ...) -> None: ...

class LoopCarriedRebind(_message.Message):
    __slots__ = ("var_name", "before_ref", "after_ref", "before_synthesized")
    VAR_NAME_FIELD_NUMBER: _ClassVar[int]
    BEFORE_REF_FIELD_NUMBER: _ClassVar[int]
    AFTER_REF_FIELD_NUMBER: _ClassVar[int]
    BEFORE_SYNTHESIZED_FIELD_NUMBER: _ClassVar[int]
    var_name: str
    before_ref: str
    after_ref: str
    before_synthesized: bool
    def __init__(self, var_name: _Optional[str] = ..., before_ref: _Optional[str] = ..., after_ref: _Optional[str] = ..., before_synthesized: bool = ...) -> None: ...

class RegionArg(_message.Message):
    __slots__ = ("var_name", "init_ref", "block_arg_ref", "yielded_ref", "result_ref")
    VAR_NAME_FIELD_NUMBER: _ClassVar[int]
    INIT_REF_FIELD_NUMBER: _ClassVar[int]
    BLOCK_ARG_REF_FIELD_NUMBER: _ClassVar[int]
    YIELDED_REF_FIELD_NUMBER: _ClassVar[int]
    RESULT_REF_FIELD_NUMBER: _ClassVar[int]
    var_name: str
    init_ref: str
    block_arg_ref: str
    yielded_ref: str
    result_ref: str
    def __init__(self, var_name: _Optional[str] = ..., init_ref: _Optional[str] = ..., block_arg_ref: _Optional[str] = ..., yielded_ref: _Optional[str] = ..., result_ref: _Optional[str] = ...) -> None: ...

class BranchRebind(_message.Message):
    __slots__ = ("var_name", "before_ref", "rebound_in_true", "rebound_in_false")
    VAR_NAME_FIELD_NUMBER: _ClassVar[int]
    BEFORE_REF_FIELD_NUMBER: _ClassVar[int]
    REBOUND_IN_TRUE_FIELD_NUMBER: _ClassVar[int]
    REBOUND_IN_FALSE_FIELD_NUMBER: _ClassVar[int]
    var_name: str
    before_ref: str
    rebound_in_true: bool
    rebound_in_false: bool
    def __init__(self, var_name: _Optional[str] = ..., before_ref: _Optional[str] = ..., rebound_in_true: bool = ..., rebound_in_false: bool = ...) -> None: ...

class CallableEntry(_message.Message):
    __slots__ = ("id", "definition")
    ID_FIELD_NUMBER: _ClassVar[int]
    DEFINITION_FIELD_NUMBER: _ClassVar[int]
    id: str
    definition: CallableDefinition
    def __init__(self, id: _Optional[str] = ..., definition: _Optional[_Union[CallableDefinition, _Mapping]] = ...) -> None: ...

class CallableDefinition(_message.Message):
    __slots__ = ("ref", "signature", "body", "body_ref", "implementations", "default_policy", "attrs")
    REF_FIELD_NUMBER: _ClassVar[int]
    SIGNATURE_FIELD_NUMBER: _ClassVar[int]
    BODY_FIELD_NUMBER: _ClassVar[int]
    BODY_REF_FIELD_NUMBER: _ClassVar[int]
    IMPLEMENTATIONS_FIELD_NUMBER: _ClassVar[int]
    DEFAULT_POLICY_FIELD_NUMBER: _ClassVar[int]
    ATTRS_FIELD_NUMBER: _ClassVar[int]
    ref: CallableRef
    signature: Signature
    body: Block
    body_ref: CallableBodyRef
    implementations: _containers.RepeatedCompositeFieldContainer[CallableImplementation]
    default_policy: str
    attrs: Payload
    def __init__(self, ref: _Optional[_Union[CallableRef, _Mapping]] = ..., signature: _Optional[_Union[Signature, _Mapping]] = ..., body: _Optional[_Union[Block, _Mapping]] = ..., body_ref: _Optional[_Union[CallableBodyRef, _Mapping]] = ..., implementations: _Optional[_Iterable[_Union[CallableImplementation, _Mapping]]] = ..., default_policy: _Optional[str] = ..., attrs: _Optional[_Union[Payload, _Mapping]] = ...) -> None: ...

class CallableRef(_message.Message):
    __slots__ = ("namespace", "name", "version")
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    namespace: str
    name: str
    version: str
    def __init__(self, namespace: _Optional[str] = ..., name: _Optional[str] = ..., version: _Optional[str] = ...) -> None: ...

class CallableBodyRef(_message.Message):
    __slots__ = ("ref", "kind", "attrs")
    REF_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    ATTRS_FIELD_NUMBER: _ClassVar[int]
    ref: CallableRef
    kind: str
    attrs: Payload
    def __init__(self, ref: _Optional[_Union[CallableRef, _Mapping]] = ..., kind: _Optional[str] = ..., attrs: _Optional[_Union[Payload, _Mapping]] = ...) -> None: ...

class CallableImplementation(_message.Message):
    __slots__ = ("transform", "backend", "strategy", "body", "body_ref", "attrs")
    TRANSFORM_FIELD_NUMBER: _ClassVar[int]
    BACKEND_FIELD_NUMBER: _ClassVar[int]
    STRATEGY_FIELD_NUMBER: _ClassVar[int]
    BODY_FIELD_NUMBER: _ClassVar[int]
    BODY_REF_FIELD_NUMBER: _ClassVar[int]
    ATTRS_FIELD_NUMBER: _ClassVar[int]
    transform: str
    backend: str
    strategy: str
    body: Block
    body_ref: CallableBodyRef
    attrs: Payload
    def __init__(self, transform: _Optional[str] = ..., backend: _Optional[str] = ..., strategy: _Optional[str] = ..., body: _Optional[_Union[Block, _Mapping]] = ..., body_ref: _Optional[_Union[CallableBodyRef, _Mapping]] = ..., attrs: _Optional[_Union[Payload, _Mapping]] = ...) -> None: ...

class Signature(_message.Message):
    __slots__ = ("operands", "results")
    OPERANDS_FIELD_NUMBER: _ClassVar[int]
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    operands: _containers.RepeatedCompositeFieldContainer[ParamHint]
    results: _containers.RepeatedCompositeFieldContainer[ParamHint]
    def __init__(self, operands: _Optional[_Iterable[_Union[ParamHint, _Mapping]]] = ..., results: _Optional[_Iterable[_Union[ParamHint, _Mapping]]] = ...) -> None: ...

class ParamHint(_message.Message):
    __slots__ = ("present", "name", "type")
    PRESENT_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    present: bool
    name: str
    type: ValueType
    def __init__(self, present: bool = ..., name: _Optional[str] = ..., type: _Optional[_Union[ValueType, _Mapping]] = ...) -> None: ...

class Payload(_message.Message):
    __slots__ = ("null_value", "bool_value", "integer_value", "float_value", "string_value", "bytes_value", "list_value", "tuple_value", "set_value", "frozenset_value", "map_value", "complex_value", "numpy_array", "numpy_scalar", "hamiltonian")
    NULL_VALUE_FIELD_NUMBER: _ClassVar[int]
    BOOL_VALUE_FIELD_NUMBER: _ClassVar[int]
    INTEGER_VALUE_FIELD_NUMBER: _ClassVar[int]
    FLOAT_VALUE_FIELD_NUMBER: _ClassVar[int]
    STRING_VALUE_FIELD_NUMBER: _ClassVar[int]
    BYTES_VALUE_FIELD_NUMBER: _ClassVar[int]
    LIST_VALUE_FIELD_NUMBER: _ClassVar[int]
    TUPLE_VALUE_FIELD_NUMBER: _ClassVar[int]
    SET_VALUE_FIELD_NUMBER: _ClassVar[int]
    FROZENSET_VALUE_FIELD_NUMBER: _ClassVar[int]
    MAP_VALUE_FIELD_NUMBER: _ClassVar[int]
    COMPLEX_VALUE_FIELD_NUMBER: _ClassVar[int]
    NUMPY_ARRAY_FIELD_NUMBER: _ClassVar[int]
    NUMPY_SCALAR_FIELD_NUMBER: _ClassVar[int]
    HAMILTONIAN_FIELD_NUMBER: _ClassVar[int]
    null_value: NullValue
    bool_value: bool
    integer_value: BigInteger
    float_value: Float64
    string_value: str
    bytes_value: bytes
    list_value: PayloadList
    tuple_value: PayloadList
    set_value: PayloadList
    frozenset_value: PayloadList
    map_value: PayloadMap
    complex_value: Complex64
    numpy_array: NumpyValue
    numpy_scalar: NumpyValue
    hamiltonian: Hamiltonian
    def __init__(self, null_value: _Optional[_Union[NullValue, _Mapping]] = ..., bool_value: bool = ..., integer_value: _Optional[_Union[BigInteger, _Mapping]] = ..., float_value: _Optional[_Union[Float64, _Mapping]] = ..., string_value: _Optional[str] = ..., bytes_value: _Optional[bytes] = ..., list_value: _Optional[_Union[PayloadList, _Mapping]] = ..., tuple_value: _Optional[_Union[PayloadList, _Mapping]] = ..., set_value: _Optional[_Union[PayloadList, _Mapping]] = ..., frozenset_value: _Optional[_Union[PayloadList, _Mapping]] = ..., map_value: _Optional[_Union[PayloadMap, _Mapping]] = ..., complex_value: _Optional[_Union[Complex64, _Mapping]] = ..., numpy_array: _Optional[_Union[NumpyValue, _Mapping]] = ..., numpy_scalar: _Optional[_Union[NumpyValue, _Mapping]] = ..., hamiltonian: _Optional[_Union[Hamiltonian, _Mapping]] = ...) -> None: ...

class NullValue(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class BigInteger(_message.Message):
    __slots__ = ("negative", "magnitude")
    NEGATIVE_FIELD_NUMBER: _ClassVar[int]
    MAGNITUDE_FIELD_NUMBER: _ClassVar[int]
    negative: bool
    magnitude: bytes
    def __init__(self, negative: bool = ..., magnitude: _Optional[bytes] = ...) -> None: ...

class Float64(_message.Message):
    __slots__ = ("bits",)
    BITS_FIELD_NUMBER: _ClassVar[int]
    bits: int
    def __init__(self, bits: _Optional[int] = ...) -> None: ...

class Complex64(_message.Message):
    __slots__ = ("real_bits", "imag_bits")
    REAL_BITS_FIELD_NUMBER: _ClassVar[int]
    IMAG_BITS_FIELD_NUMBER: _ClassVar[int]
    real_bits: int
    imag_bits: int
    def __init__(self, real_bits: _Optional[int] = ..., imag_bits: _Optional[int] = ...) -> None: ...

class PayloadList(_message.Message):
    __slots__ = ("items",)
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    items: _containers.RepeatedCompositeFieldContainer[Payload]
    def __init__(self, items: _Optional[_Iterable[_Union[Payload, _Mapping]]] = ...) -> None: ...

class PayloadMap(_message.Message):
    __slots__ = ("entries",)
    ENTRIES_FIELD_NUMBER: _ClassVar[int]
    entries: _containers.RepeatedCompositeFieldContainer[PayloadEntry]
    def __init__(self, entries: _Optional[_Iterable[_Union[PayloadEntry, _Mapping]]] = ...) -> None: ...

class PayloadEntry(_message.Message):
    __slots__ = ("key", "value")
    KEY_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    key: Payload
    value: Payload
    def __init__(self, key: _Optional[_Union[Payload, _Mapping]] = ..., value: _Optional[_Union[Payload, _Mapping]] = ...) -> None: ...

class NumpyValue(_message.Message):
    __slots__ = ("dtype", "shape", "data")
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    dtype: str
    shape: _containers.RepeatedScalarFieldContainer[int]
    data: bytes
    def __init__(self, dtype: _Optional[str] = ..., shape: _Optional[_Iterable[int]] = ..., data: _Optional[bytes] = ...) -> None: ...

class Hamiltonian(_message.Message):
    __slots__ = ("terms", "constant", "num_qubits")
    TERMS_FIELD_NUMBER: _ClassVar[int]
    CONSTANT_FIELD_NUMBER: _ClassVar[int]
    NUM_QUBITS_FIELD_NUMBER: _ClassVar[int]
    terms: _containers.RepeatedCompositeFieldContainer[HamiltonianTerm]
    constant: Number
    num_qubits: int
    def __init__(self, terms: _Optional[_Iterable[_Union[HamiltonianTerm, _Mapping]]] = ..., constant: _Optional[_Union[Number, _Mapping]] = ..., num_qubits: _Optional[int] = ...) -> None: ...

class HamiltonianTerm(_message.Message):
    __slots__ = ("operators", "coefficient")
    OPERATORS_FIELD_NUMBER: _ClassVar[int]
    COEFFICIENT_FIELD_NUMBER: _ClassVar[int]
    operators: _containers.RepeatedCompositeFieldContainer[PauliOperator]
    coefficient: Number
    def __init__(self, operators: _Optional[_Iterable[_Union[PauliOperator, _Mapping]]] = ..., coefficient: _Optional[_Union[Number, _Mapping]] = ...) -> None: ...

class PauliOperator(_message.Message):
    __slots__ = ("pauli", "index")
    PAULI_FIELD_NUMBER: _ClassVar[int]
    INDEX_FIELD_NUMBER: _ClassVar[int]
    pauli: str
    index: int
    def __init__(self, pauli: _Optional[str] = ..., index: _Optional[int] = ...) -> None: ...

class Number(_message.Message):
    __slots__ = ("integer", "floating", "complex")
    INTEGER_FIELD_NUMBER: _ClassVar[int]
    FLOATING_FIELD_NUMBER: _ClassVar[int]
    COMPLEX_FIELD_NUMBER: _ClassVar[int]
    integer: BigInteger
    floating: Float64
    complex: Complex64
    def __init__(self, integer: _Optional[_Union[BigInteger, _Mapping]] = ..., floating: _Optional[_Union[Float64, _Mapping]] = ..., complex: _Optional[_Union[Complex64, _Mapping]] = ...) -> None: ...
