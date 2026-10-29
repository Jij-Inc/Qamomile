---
status: accepted
---

# Validate destination-specific Nexus settings before submission

The H2 and Helios paths share one execution options class but have different submission and result contracts. Repeated H2 metadata failures showed that SDK-shaped mocks alone did not establish compatibility with the provider's complete response. Preserve exact result contracts and validate settings at the owning boundary rather than weakening decoding after execution.

Helios submission requires an explicit maximum cost per program. Require a finite nonnegative cap before authentication, project resolution, or upload, as covered by the [configuration-contract tests](../../tests/hugr/test_nexus_configuration_contract.py). Do not invent a default spending limit. Accept a deprecated native backend configuration cap as a fallback and forward it into the SDK's per-program field. Reject conflicting public and native caps because their server precedence is unverified. Keep equal values and explicit zero valid. H2 retains optional cost caps. Restoring an existing job does not require a new cap because it does not submit work.

Resolve a private immutable destination containing target, program format, and `shot_origin`, including native configuration precedence. Do not expose shot origin as a user option. Both H2 and Helios use internal origin 1. The Helios value is an adopted assumption and remains unverified on hardware. H2 normalizes all shot reads before QIR conversion, including programs without arrays and programs whose original conversion succeeds. Only the limited table proof remains a fallback for converter-rejected abort calls. The fallback starts from the original program to avoid subtracting the origin twice. Helios normalizes only its upload copy and keeps its HUGR entry and RAW result contract. H2-specific proofs, guards, and private markers are not added to Helios. Save the applied `shot_origin` and required `shot_success_tag` with H2 guarded jobs; Helios references need no extra origin field because result retrieval does not reinterpret the submitted program. Restoration does not rederive the origin, reconvert, or resubmit either target's job.

The Helios subtraction assumes native values start at 1. If the device supplies 0, unsigned subtraction wraps to the maximum UInt64 value; this minimal policy change does not detect an incorrect assumption. Confirming the actual hardware convention may require revising the internal destination policy. Local tests emulate native numbers 1 through 4 without provider submission.

An H2 syntax checker returns synthetic zero readouts. Reject guarded shot-index execution on an SC target before authentication rather than misreporting zero success markers as an actual shot-range failure. Ordinary unguarded syntax checking keeps its existing path.

Result decoding belongs to the HUGR execution handle. Cache output-schema failures there and report local FAILED status for synchronous and asynchronous retrieval. Keep the original exception as the cause, preserve provider status through `raw_status()`, and keep remote references restorable. Do not change the shared mapped handle or cache transient retrieval failures and wait timeouts as decoding failures.

Verification uses real installed SDK serialization and RAW chunk assembly with mocked HTTP boundaries, actual QIR conversion where applicable, public typed jobs, JSON snapshot restoration, and the complete local HUGR suite. The audit does not submit provider jobs or establish Helios hardware numbering, H2 numbering across execution chunks, or the applicability of circuit-only options to QIR.
