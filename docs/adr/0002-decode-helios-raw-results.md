---
status: accepted
---

# Decode Helios results with explicit namespace provenance

Nexus DEFAULT version 3 strips the protocol prefix from user output names. A user field named `METRICS:INT:RXY_COUNT` can therefore look exactly like RAW execution statistics. Inferring record type from a DEFAULT name can erase a legitimate result.

Request `ResultVersions.RAW` version 4 for Helios HUGR results, including restored jobs. Interpret the original namespace before stripping exactly one `USER` prefix. Preserve DEFAULT decoding as a separate explicit mode with literal output names. Do not fall back to DEFAULT when RAW retrieval fails.

RAW user records support BOOL, INT, UINT, and FLOAT. METRICS records support BOOL, INT, and FLOAT with a nonempty remaining name, which can contain colons. Repeated metric records are not repeated public outputs. Accept scalar numeric metric payloads without inferring their Python type from the label. The [result-contract regressions](../../tests/hugr/test_nexus.py) include a floating-point duration under an INT label.

Reject EXIT and encoded exception context as incompatible with Qamomile's complete-result contract, even when the provider reports COMPLETED. A provider-successful early exit does not establish that every requested shot reached the program's normal return. Keep user-output cardinality, normalized-name collision, shot-count, and typed output checks.

Do not apply H2 register conventions to Helios. H2 keeps the DEFAULT bit-register path, the four observed bookkeeping names `JOB_total_shots`, `JOB_shotnum`, `JOB_chunk_total_shots`, and `JOB_chunk_shotnum`, and the required private success-marker checks. No broad JOB namespace filter is introduced. Unknown names are reported together. Read-only retrieval of an existing H2 job established all six registers at width 64 and recovered two false and two true public outputs with all four success markers true.

Distinguish provider completion from usable public results. A HUGR output-schema conversion failure becomes a cached `ExecutionError` and a local FAILED status for both synchronous and asynchronous retrieval. `raw_status()` retains the provider state. A saved reference still restores a fresh decoder without uploading or executing again.

The [result-contract tests](../../tests/hugr/test_nexus.py) cover the DEFAULT/RAW distinction, scoped and unscoped metric names, numeric payloads, and early-exit rejection. The [session tests](../../tests/hugr/test_nexus_session.py) cover format selection when restoring existing jobs. These checks exercise the installed SDK without provider submission.

The user authorized the local format fix and primary-source investigation on 2026-10-28 and explicitly declined provider submission. Tests use the installed SDK, real compilation where relevant, and mocked service boundaries. Neither successful local decoding nor emulator source establishes the Helios hardware shot origin.
