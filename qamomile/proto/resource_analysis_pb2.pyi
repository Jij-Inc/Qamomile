from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ResourceDomainProto(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    RESOURCE_DOMAIN_UNSPECIFIED: _ClassVar[ResourceDomainProto]
    RESOURCE_NATURAL: _ClassVar[ResourceDomainProto]
    RESOURCE_INTEGER: _ClassVar[ResourceDomainProto]
    RESOURCE_RATIONAL: _ClassVar[ResourceDomainProto]
    RESOURCE_REAL: _ClassVar[ResourceDomainProto]
    RESOURCE_PROBABILITY: _ClassVar[ResourceDomainProto]
    RESOURCE_BOOLEAN: _ClassVar[ResourceDomainProto]

class ResourceCoordinateProto(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    RESOURCE_COORDINATE_UNSPECIFIED: _ClassVar[ResourceCoordinateProto]
    RESOURCE_ALGORITHMIC: _ClassVar[ResourceCoordinateProto]
    RESOURCE_LOGICAL: _ClassVar[ResourceCoordinateProto]
    RESOURCE_CIRCUIT: _ClassVar[ResourceCoordinateProto]
    RESOURCE_FAULT_TOLERANT: _ClassVar[ResourceCoordinateProto]
    RESOURCE_PHYSICAL: _ClassVar[ResourceCoordinateProto]
    RESOURCE_EMITTED: _ClassVar[ResourceCoordinateProto]
    RESOURCE_OBSERVED_EXECUTION: _ClassVar[ResourceCoordinateProto]

class ResourceCompositionProto(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    RESOURCE_COMPOSITION_UNSPECIFIED: _ClassVar[ResourceCompositionProto]
    RESOURCE_ADDITIVE_WORK: _ClassVar[ResourceCompositionProto]
    RESOURCE_DEPENDENCY_SPAN: _ClassVar[ResourceCompositionProto]
    RESOURCE_PEAK_LIVENESS: _ClassVar[ResourceCompositionProto]
    RESOURCE_REPLACEMENT: _ClassVar[ResourceCompositionProto]
    RESOURCE_ERROR_BOUND: _ClassVar[ResourceCompositionProto]

class ResourceRelationProto(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    RESOURCE_RELATION_UNSPECIFIED: _ClassVar[ResourceRelationProto]
    RESOURCE_EQUALITY: _ClassVar[ResourceRelationProto]
    RESOURCE_LOWER_BOUND: _ClassVar[ResourceRelationProto]
    RESOURCE_UPPER_BOUND: _ClassVar[ResourceRelationProto]
    RESOURCE_ENCLOSURE: _ClassVar[ResourceRelationProto]
    RESOURCE_PARTIAL_ENCLOSURE: _ClassVar[ResourceRelationProto]
    RESOURCE_UNKNOWN: _ClassVar[ResourceRelationProto]

class ResourceTransformModeProto(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    RESOURCE_TRANSFORM_MODE_UNSPECIFIED: _ClassVar[ResourceTransformModeProto]
    RESOURCE_DIRECT: _ClassVar[ResourceTransformModeProto]
    RESOURCE_CONTROLLED: _ClassVar[ResourceTransformModeProto]
    RESOURCE_ADJOINT: _ClassVar[ResourceTransformModeProto]
    RESOURCE_CONTROLLED_ADJOINT: _ClassVar[ResourceTransformModeProto]

class ResourceEvidenceAuthorityProto(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    RESOURCE_EVIDENCE_AUTHORITY_UNSPECIFIED: _ClassVar[ResourceEvidenceAuthorityProto]
    RESOURCE_STRUCTURALLY_DERIVED: _ClassVar[ResourceEvidenceAuthorityProto]
    RESOURCE_TRUSTED_AXIOM: _ClassVar[ResourceEvidenceAuthorityProto]
    RESOURCE_HEURISTIC_MODEL: _ClassVar[ResourceEvidenceAuthorityProto]
RESOURCE_DOMAIN_UNSPECIFIED: ResourceDomainProto
RESOURCE_NATURAL: ResourceDomainProto
RESOURCE_INTEGER: ResourceDomainProto
RESOURCE_RATIONAL: ResourceDomainProto
RESOURCE_REAL: ResourceDomainProto
RESOURCE_PROBABILITY: ResourceDomainProto
RESOURCE_BOOLEAN: ResourceDomainProto
RESOURCE_COORDINATE_UNSPECIFIED: ResourceCoordinateProto
RESOURCE_ALGORITHMIC: ResourceCoordinateProto
RESOURCE_LOGICAL: ResourceCoordinateProto
RESOURCE_CIRCUIT: ResourceCoordinateProto
RESOURCE_FAULT_TOLERANT: ResourceCoordinateProto
RESOURCE_PHYSICAL: ResourceCoordinateProto
RESOURCE_EMITTED: ResourceCoordinateProto
RESOURCE_OBSERVED_EXECUTION: ResourceCoordinateProto
RESOURCE_COMPOSITION_UNSPECIFIED: ResourceCompositionProto
RESOURCE_ADDITIVE_WORK: ResourceCompositionProto
RESOURCE_DEPENDENCY_SPAN: ResourceCompositionProto
RESOURCE_PEAK_LIVENESS: ResourceCompositionProto
RESOURCE_REPLACEMENT: ResourceCompositionProto
RESOURCE_ERROR_BOUND: ResourceCompositionProto
RESOURCE_RELATION_UNSPECIFIED: ResourceRelationProto
RESOURCE_EQUALITY: ResourceRelationProto
RESOURCE_LOWER_BOUND: ResourceRelationProto
RESOURCE_UPPER_BOUND: ResourceRelationProto
RESOURCE_ENCLOSURE: ResourceRelationProto
RESOURCE_PARTIAL_ENCLOSURE: ResourceRelationProto
RESOURCE_UNKNOWN: ResourceRelationProto
RESOURCE_TRANSFORM_MODE_UNSPECIFIED: ResourceTransformModeProto
RESOURCE_DIRECT: ResourceTransformModeProto
RESOURCE_CONTROLLED: ResourceTransformModeProto
RESOURCE_ADJOINT: ResourceTransformModeProto
RESOURCE_CONTROLLED_ADJOINT: ResourceTransformModeProto
RESOURCE_EVIDENCE_AUTHORITY_UNSPECIFIED: ResourceEvidenceAuthorityProto
RESOURCE_STRUCTURALLY_DERIVED: ResourceEvidenceAuthorityProto
RESOURCE_TRUSTED_AXIOM: ResourceEvidenceAuthorityProto
RESOURCE_HEURISTIC_MODEL: ResourceEvidenceAuthorityProto

class ResourceRationalProto(_message.Message):
    __slots__ = ("numerator", "denominator")
    NUMERATOR_FIELD_NUMBER: _ClassVar[int]
    DENOMINATOR_FIELD_NUMBER: _ClassVar[int]
    numerator: str
    denominator: str
    def __init__(self, numerator: _Optional[str] = ..., denominator: _Optional[str] = ...) -> None: ...

class ResourceNumberProto(_message.Message):
    __slots__ = ("integer", "rational", "real_bits")
    INTEGER_FIELD_NUMBER: _ClassVar[int]
    RATIONAL_FIELD_NUMBER: _ClassVar[int]
    REAL_BITS_FIELD_NUMBER: _ClassVar[int]
    integer: str
    rational: ResourceRationalProto
    real_bits: int
    def __init__(self, integer: _Optional[str] = ..., rational: _Optional[_Union[ResourceRationalProto, _Mapping]] = ..., real_bits: _Optional[int] = ...) -> None: ...

class ResourceSymbolProto(_message.Message):
    __slots__ = ("scope", "name", "domain", "unit", "origin", "constraints")
    SCOPE_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    DOMAIN_FIELD_NUMBER: _ClassVar[int]
    UNIT_FIELD_NUMBER: _ClassVar[int]
    ORIGIN_FIELD_NUMBER: _ClassVar[int]
    CONSTRAINTS_FIELD_NUMBER: _ClassVar[int]
    scope: str
    name: str
    domain: ResourceDomainProto
    unit: str
    origin: str
    constraints: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, scope: _Optional[str] = ..., name: _Optional[str] = ..., domain: _Optional[_Union[ResourceDomainProto, str]] = ..., unit: _Optional[str] = ..., origin: _Optional[str] = ..., constraints: _Optional[_Iterable[str]] = ...) -> None: ...

class ResourceAttributeProto(_message.Message):
    __slots__ = ("name", "value")
    NAME_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    name: str
    value: str
    def __init__(self, name: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

class ResourceExpressionNodeProto(_message.Message):
    __slots__ = ("kind", "domain", "unit", "arguments", "literal", "symbol", "attributes", "predicates")
    KIND_FIELD_NUMBER: _ClassVar[int]
    DOMAIN_FIELD_NUMBER: _ClassVar[int]
    UNIT_FIELD_NUMBER: _ClassVar[int]
    ARGUMENTS_FIELD_NUMBER: _ClassVar[int]
    LITERAL_FIELD_NUMBER: _ClassVar[int]
    SYMBOL_FIELD_NUMBER: _ClassVar[int]
    ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    PREDICATES_FIELD_NUMBER: _ClassVar[int]
    kind: str
    domain: ResourceDomainProto
    unit: str
    arguments: _containers.RepeatedScalarFieldContainer[int]
    literal: ResourceNumberProto
    symbol: ResourceSymbolProto
    attributes: _containers.RepeatedCompositeFieldContainer[ResourceAttributeProto]
    predicates: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, kind: _Optional[str] = ..., domain: _Optional[_Union[ResourceDomainProto, str]] = ..., unit: _Optional[str] = ..., arguments: _Optional[_Iterable[int]] = ..., literal: _Optional[_Union[ResourceNumberProto, _Mapping]] = ..., symbol: _Optional[_Union[ResourceSymbolProto, _Mapping]] = ..., attributes: _Optional[_Iterable[_Union[ResourceAttributeProto, _Mapping]]] = ..., predicates: _Optional[_Iterable[int]] = ...) -> None: ...

class ResourcePredicateNodeProto(_message.Message):
    __slots__ = ("kind", "arguments", "predicates")
    KIND_FIELD_NUMBER: _ClassVar[int]
    ARGUMENTS_FIELD_NUMBER: _ClassVar[int]
    PREDICATES_FIELD_NUMBER: _ClassVar[int]
    kind: str
    arguments: _containers.RepeatedScalarFieldContainer[int]
    predicates: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, kind: _Optional[str] = ..., arguments: _Optional[_Iterable[int]] = ..., predicates: _Optional[_Iterable[int]] = ...) -> None: ...

class ResourceNodeProto(_message.Message):
    __slots__ = ("expression", "predicate")
    EXPRESSION_FIELD_NUMBER: _ClassVar[int]
    PREDICATE_FIELD_NUMBER: _ClassVar[int]
    expression: ResourceExpressionNodeProto
    predicate: ResourcePredicateNodeProto
    def __init__(self, expression: _Optional[_Union[ResourceExpressionNodeProto, _Mapping]] = ..., predicate: _Optional[_Union[ResourcePredicateNodeProto, _Mapping]] = ...) -> None: ...

class ResourceExpressionGraphProto(_message.Message):
    __slots__ = ("nodes", "root")
    NODES_FIELD_NUMBER: _ClassVar[int]
    ROOT_FIELD_NUMBER: _ClassVar[int]
    nodes: _containers.RepeatedCompositeFieldContainer[ResourceNodeProto]
    root: int
    def __init__(self, nodes: _Optional[_Iterable[_Union[ResourceNodeProto, _Mapping]]] = ..., root: _Optional[int] = ...) -> None: ...

class ResourceMetricProto(_message.Message):
    __slots__ = ("identifier", "coordinate", "domain", "unit", "composition", "version")
    IDENTIFIER_FIELD_NUMBER: _ClassVar[int]
    COORDINATE_FIELD_NUMBER: _ClassVar[int]
    DOMAIN_FIELD_NUMBER: _ClassVar[int]
    UNIT_FIELD_NUMBER: _ClassVar[int]
    COMPOSITION_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    identifier: str
    coordinate: ResourceCoordinateProto
    domain: ResourceDomainProto
    unit: str
    composition: ResourceCompositionProto
    version: str
    def __init__(self, identifier: _Optional[str] = ..., coordinate: _Optional[_Union[ResourceCoordinateProto, str]] = ..., domain: _Optional[_Union[ResourceDomainProto, str]] = ..., unit: _Optional[str] = ..., composition: _Optional[_Union[ResourceCompositionProto, str]] = ..., version: _Optional[str] = ...) -> None: ...

class ResourceRuleProto(_message.Message):
    __slots__ = ("identity", "callable", "metric", "relation", "expression", "modes", "assumptions", "validity_domain", "authority", "provenance", "scheduling_footprint")
    IDENTITY_FIELD_NUMBER: _ClassVar[int]
    CALLABLE_FIELD_NUMBER: _ClassVar[int]
    METRIC_FIELD_NUMBER: _ClassVar[int]
    RELATION_FIELD_NUMBER: _ClassVar[int]
    EXPRESSION_FIELD_NUMBER: _ClassVar[int]
    MODES_FIELD_NUMBER: _ClassVar[int]
    ASSUMPTIONS_FIELD_NUMBER: _ClassVar[int]
    VALIDITY_DOMAIN_FIELD_NUMBER: _ClassVar[int]
    AUTHORITY_FIELD_NUMBER: _ClassVar[int]
    PROVENANCE_FIELD_NUMBER: _ClassVar[int]
    SCHEDULING_FOOTPRINT_FIELD_NUMBER: _ClassVar[int]
    identity: str
    callable: str
    metric: ResourceMetricProto
    relation: ResourceRelationProto
    expression: ResourceExpressionGraphProto
    modes: _containers.RepeatedScalarFieldContainer[ResourceTransformModeProto]
    assumptions: _containers.RepeatedScalarFieldContainer[str]
    validity_domain: _containers.RepeatedScalarFieldContainer[str]
    authority: ResourceEvidenceAuthorityProto
    provenance: str
    scheduling_footprint: ResourceSchedulingFootprintProto
    def __init__(self, identity: _Optional[str] = ..., callable: _Optional[str] = ..., metric: _Optional[_Union[ResourceMetricProto, _Mapping]] = ..., relation: _Optional[_Union[ResourceRelationProto, str]] = ..., expression: _Optional[_Union[ResourceExpressionGraphProto, _Mapping]] = ..., modes: _Optional[_Iterable[_Union[ResourceTransformModeProto, str]]] = ..., assumptions: _Optional[_Iterable[str]] = ..., validity_domain: _Optional[_Iterable[str]] = ..., authority: _Optional[_Union[ResourceEvidenceAuthorityProto, str]] = ..., provenance: _Optional[str] = ..., scheduling_footprint: _Optional[_Union[ResourceSchedulingFootprintProto, _Mapping]] = ...) -> None: ...

class ResourceSchedulingFootprintProto(_message.Message):
    __slots__ = ("parameters",)
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    parameters: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, parameters: _Optional[_Iterable[str]] = ...) -> None: ...

class ResourceAnalysisRequestProto(_message.Message):
    __slots__ = ("schema_major", "schema_minor", "program_family", "rules", "request_digest")
    SCHEMA_MAJOR_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_MINOR_FIELD_NUMBER: _ClassVar[int]
    PROGRAM_FAMILY_FIELD_NUMBER: _ClassVar[int]
    RULES_FIELD_NUMBER: _ClassVar[int]
    REQUEST_DIGEST_FIELD_NUMBER: _ClassVar[int]
    schema_major: int
    schema_minor: int
    program_family: bytes
    rules: _containers.RepeatedCompositeFieldContainer[ResourceRuleProto]
    request_digest: bytes
    def __init__(self, schema_major: _Optional[int] = ..., schema_minor: _Optional[int] = ..., program_family: _Optional[bytes] = ..., rules: _Optional[_Iterable[_Union[ResourceRuleProto, _Mapping]]] = ..., request_digest: _Optional[bytes] = ...) -> None: ...

class ResourceEvidenceProto(_message.Message):
    __slots__ = ("authority", "reference", "details")
    AUTHORITY_FIELD_NUMBER: _ClassVar[int]
    REFERENCE_FIELD_NUMBER: _ClassVar[int]
    DETAILS_FIELD_NUMBER: _ClassVar[int]
    authority: ResourceEvidenceAuthorityProto
    reference: str
    details: _containers.RepeatedCompositeFieldContainer[ResourceAttributeProto]
    def __init__(self, authority: _Optional[_Union[ResourceEvidenceAuthorityProto, str]] = ..., reference: _Optional[str] = ..., details: _Optional[_Iterable[_Union[ResourceAttributeProto, _Mapping]]] = ...) -> None: ...

class ResourceValueProto(_message.Message):
    __slots__ = ("metric", "relation", "expression", "lower", "upper", "unknown_contributors", "evidence", "assumptions")
    METRIC_FIELD_NUMBER: _ClassVar[int]
    RELATION_FIELD_NUMBER: _ClassVar[int]
    EXPRESSION_FIELD_NUMBER: _ClassVar[int]
    LOWER_FIELD_NUMBER: _ClassVar[int]
    UPPER_FIELD_NUMBER: _ClassVar[int]
    UNKNOWN_CONTRIBUTORS_FIELD_NUMBER: _ClassVar[int]
    EVIDENCE_FIELD_NUMBER: _ClassVar[int]
    ASSUMPTIONS_FIELD_NUMBER: _ClassVar[int]
    metric: ResourceMetricProto
    relation: ResourceRelationProto
    expression: ResourceExpressionGraphProto
    lower: ResourceExpressionGraphProto
    upper: ResourceExpressionGraphProto
    unknown_contributors: _containers.RepeatedScalarFieldContainer[str]
    evidence: _containers.RepeatedCompositeFieldContainer[ResourceEvidenceProto]
    assumptions: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, metric: _Optional[_Union[ResourceMetricProto, _Mapping]] = ..., relation: _Optional[_Union[ResourceRelationProto, str]] = ..., expression: _Optional[_Union[ResourceExpressionGraphProto, _Mapping]] = ..., lower: _Optional[_Union[ResourceExpressionGraphProto, _Mapping]] = ..., upper: _Optional[_Union[ResourceExpressionGraphProto, _Mapping]] = ..., unknown_contributors: _Optional[_Iterable[str]] = ..., evidence: _Optional[_Iterable[_Union[ResourceEvidenceProto, _Mapping]]] = ..., assumptions: _Optional[_Iterable[str]] = ...) -> None: ...

class ResourceMetricValueProto(_message.Message):
    __slots__ = ("identifier", "value")
    IDENTIFIER_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    identifier: str
    value: ResourceValueProto
    def __init__(self, identifier: _Optional[str] = ..., value: _Optional[_Union[ResourceValueProto, _Mapping]] = ...) -> None: ...

class ResourceContributionProto(_message.Message):
    __slots__ = ("identifier", "relation", "expression", "lower", "upper", "unknown_contributors")
    IDENTIFIER_FIELD_NUMBER: _ClassVar[int]
    RELATION_FIELD_NUMBER: _ClassVar[int]
    EXPRESSION_FIELD_NUMBER: _ClassVar[int]
    LOWER_FIELD_NUMBER: _ClassVar[int]
    UPPER_FIELD_NUMBER: _ClassVar[int]
    UNKNOWN_CONTRIBUTORS_FIELD_NUMBER: _ClassVar[int]
    identifier: str
    relation: ResourceRelationProto
    expression: ResourceExpressionGraphProto
    lower: ResourceExpressionGraphProto
    upper: ResourceExpressionGraphProto
    unknown_contributors: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, identifier: _Optional[str] = ..., relation: _Optional[_Union[ResourceRelationProto, str]] = ..., expression: _Optional[_Union[ResourceExpressionGraphProto, _Mapping]] = ..., lower: _Optional[_Union[ResourceExpressionGraphProto, _Mapping]] = ..., upper: _Optional[_Union[ResourceExpressionGraphProto, _Mapping]] = ..., unknown_contributors: _Optional[_Iterable[str]] = ...) -> None: ...

class ResourceOperationFactProto(_message.Message):
    __slots__ = ("opcode", "contributions")
    OPCODE_FIELD_NUMBER: _ClassVar[int]
    CONTRIBUTIONS_FIELD_NUMBER: _ClassVar[int]
    opcode: str
    contributions: _containers.RepeatedCompositeFieldContainer[ResourceContributionProto]
    def __init__(self, opcode: _Optional[str] = ..., contributions: _Optional[_Iterable[_Union[ResourceContributionProto, _Mapping]]] = ...) -> None: ...

class ResourceAnalysisFactProto(_message.Message):
    __slots__ = ("kind", "owner", "source", "operation")
    KIND_FIELD_NUMBER: _ClassVar[int]
    OWNER_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    OPERATION_FIELD_NUMBER: _ClassVar[int]
    kind: str
    owner: str
    source: str
    operation: ResourceOperationFactProto
    def __init__(self, kind: _Optional[str] = ..., owner: _Optional[str] = ..., source: _Optional[str] = ..., operation: _Optional[_Union[ResourceOperationFactProto, _Mapping]] = ...) -> None: ...

class ResourceAnalysisBundleProto(_message.Message):
    __slots__ = ("entrypoint", "analysis_version", "qcore_payload", "facts")
    ENTRYPOINT_FIELD_NUMBER: _ClassVar[int]
    ANALYSIS_VERSION_FIELD_NUMBER: _ClassVar[int]
    QCORE_PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    FACTS_FIELD_NUMBER: _ClassVar[int]
    entrypoint: str
    analysis_version: str
    qcore_payload: bytes
    facts: _containers.RepeatedCompositeFieldContainer[ResourceAnalysisFactProto]
    def __init__(self, entrypoint: _Optional[str] = ..., analysis_version: _Optional[str] = ..., qcore_payload: _Optional[bytes] = ..., facts: _Optional[_Iterable[_Union[ResourceAnalysisFactProto, _Mapping]]] = ...) -> None: ...

class ResourceStudyProto(_message.Message):
    __slots__ = ("schema_major", "schema_minor", "bundle", "metrics", "derivations", "study_digest")
    SCHEMA_MAJOR_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_MINOR_FIELD_NUMBER: _ClassVar[int]
    BUNDLE_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    DERIVATIONS_FIELD_NUMBER: _ClassVar[int]
    STUDY_DIGEST_FIELD_NUMBER: _ClassVar[int]
    schema_major: int
    schema_minor: int
    bundle: ResourceAnalysisBundleProto
    metrics: _containers.RepeatedCompositeFieldContainer[ResourceMetricValueProto]
    derivations: _containers.RepeatedScalarFieldContainer[str]
    study_digest: bytes
    def __init__(self, schema_major: _Optional[int] = ..., schema_minor: _Optional[int] = ..., bundle: _Optional[_Union[ResourceAnalysisBundleProto, _Mapping]] = ..., metrics: _Optional[_Iterable[_Union[ResourceMetricValueProto, _Mapping]]] = ..., derivations: _Optional[_Iterable[str]] = ..., study_digest: _Optional[bytes] = ...) -> None: ...
