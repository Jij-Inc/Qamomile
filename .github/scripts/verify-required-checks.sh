#!/usr/bin/env bash

# Verify that every quality-check job required by change classification
# completed successfully.
#
# Expected environment variables:
#   CHANGES_RESULT       result of the change-classification job
#   FULL_CI_REQUIRED     whether source-oriented jobs were required
#   DOCS_CHANGED         whether documentation paths changed
#   PYTHON_TEST_RESULT   result of the Python test job
#   RUST_RESULT          result of the Rust test job
#   DOCS_TEST_RESULT     result of the documentation test job
#   RUFF_RESULT          result of the Ruff job
#   ZUBAN_RESULT         result of the Zuban job

set -euo pipefail

: "${CHANGES_RESULT:?CHANGES_RESULT is required.}"
: "${FULL_CI_REQUIRED:?FULL_CI_REQUIRED is required.}"
: "${DOCS_CHANGED:?DOCS_CHANGED is required.}"
: "${PYTHON_TEST_RESULT:?PYTHON_TEST_RESULT is required.}"
: "${RUST_RESULT:?RUST_RESULT is required.}"
: "${DOCS_TEST_RESULT:?DOCS_TEST_RESULT is required.}"
: "${RUFF_RESULT:?RUFF_RESULT is required.}"
: "${ZUBAN_RESULT:?ZUBAN_RESULT is required.}"

validate_boolean() {
  local name="$1"
  local value="$2"

  case "${value}" in
    true|false) ;;
    *)
      echo "${name} has an invalid Boolean value: ${value}" >&2
      exit 1
      ;;
  esac
}

validate_boolean "FULL_CI_REQUIRED" "${FULL_CI_REQUIRED}"
validate_boolean "DOCS_CHANGED" "${DOCS_CHANGED}"

failure=false

check_job() {
  local job_name="$1"
  local result="$2"
  local required="$3"

  if [[ "${result}" == "success" ]]; then
    return
  fi
  if [[ "${result}" == "skipped" && "${required}" != "true" ]]; then
    return
  fi

  echo "${job_name} finished with '${result}' (required=${required})." >&2
  failure=true
}

if [[ "${CHANGES_RESULT}" != "success" ]]; then
  echo "Change classification finished with '${CHANGES_RESULT}'." >&2
  failure=true
fi

docs_required=false
if [[ "${FULL_CI_REQUIRED}" == "true" || "${DOCS_CHANGED}" == "true" ]]; then
  docs_required=true
fi

check_job "Python Test" "${PYTHON_TEST_RESULT}" "${FULL_CI_REQUIRED}"
check_job "Rust" "${RUST_RESULT}" "${FULL_CI_REQUIRED}"
check_job "Docs Test" "${DOCS_TEST_RESULT}" "${docs_required}"
check_job "Ruff" "${RUFF_RESULT}" "${FULL_CI_REQUIRED}"
check_job "Zuban" "${ZUBAN_RESULT}" "${FULL_CI_REQUIRED}"

if [[ "${failure}" == "true" ]]; then
  exit 1
fi
