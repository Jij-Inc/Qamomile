#!/usr/bin/env bash

# Verify that every quality-check job required by change classification
# completed successfully.
#
# Usage:
#   verify-required-checks.sh
#
# Arguments:
#   None.
#
# Environment:
#   CHANGES_RESULT:      Result of the change-classification job.
#   FULL_CI_REQUIRED:    Whether source-oriented jobs were required.
#   DOCS_CHANGED:        Whether documentation paths changed.
#   PYTHON_TEST_RESULT:  Result of the Python test job.
#   DOCS_TEST_RESULT:    Result of the documentation test job.
#   RUFF_RESULT:         Result of the Ruff job.
#   ZUBAN_RESULT:        Result of the Zuban job.
#
# Outputs:
#   Writes invalid inputs and unacceptable job results to stderr.
#
# Exit status:
#   0 if classification succeeded and every required job succeeded.
#   1 if an input is invalid or a required job did not succeed.

set -euo pipefail

: "${CHANGES_RESULT:?CHANGES_RESULT is required.}"
: "${FULL_CI_REQUIRED:?FULL_CI_REQUIRED is required.}"
: "${DOCS_CHANGED:?DOCS_CHANGED is required.}"
: "${PYTHON_TEST_RESULT:?PYTHON_TEST_RESULT is required.}"
: "${DOCS_TEST_RESULT:?DOCS_TEST_RESULT is required.}"
: "${RUFF_RESULT:?RUFF_RESULT is required.}"
: "${ZUBAN_RESULT:?ZUBAN_RESULT is required.}"

# Validate one named Boolean workflow input.
#
# Arguments:
#   $1: Environment-variable name used in error messages.
#   $2: Value to validate.
#
# Outputs:
#   Writes a validation error to stderr.
#
# Returns:
#   0 if the value is "true" or "false".
#   Exits the script with status 1 otherwise.
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

# Record whether one quality-check job has an acceptable result.
#
# Arguments:
#   $1: Human-readable job name.
#   $2: GitHub Actions job result.
#   $3: "true" if the job was required; "false" otherwise.
#
# Globals:
#   failure: Set to "true" when the result is unacceptable.
#
# Outputs:
#   Writes unacceptable job results to stderr.
#
# Returns:
#   0 after inspecting the result. Failure is accumulated in the global
#   failure flag and handled after all jobs are checked.
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
check_job "Docs Test" "${DOCS_TEST_RESULT}" "${docs_required}"
check_job "Ruff" "${RUFF_RESULT}" "${FULL_CI_REQUIRED}"
check_job "Zuban" "${ZUBAN_RESULT}" "${FULL_CI_REQUIRED}"

if [[ "${failure}" == "true" ]]; then
  exit 1
fi
