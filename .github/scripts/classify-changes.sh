#!/usr/bin/env bash
#
# Classify the files changed between two commits for CI job routing.
#
# Usage:
#   classify-changes.sh BASE_SHA HEAD_SHA [DIFF_MODE]
#
# Arguments:
#   BASE_SHA:
#     Commit used as the diff base. An empty or all-zero value requests the
#     fail-safe full CI classification.
#   HEAD_SHA:
#     Commit used as the diff head. An empty or all-zero value requests the
#     fail-safe full CI classification.
#   DIFF_MODE:
#     Optional diff strategy. Use "merge-base" for pull requests or "direct"
#     for pushes. Defaults to "merge-base".
#
# Environment:
#   GITHUB_OUTPUT:
#     Required path to the GitHub Actions step-output file.
#
# Outputs written to GITHUB_OUTPUT:
#   classification_complete
#                 true only after all classification outputs were written.
#   full_ci_required
#                 true when a changed path requires the complete CI suite,
#                 including source, tests, build configuration, CI, and
#                 otherwise unclassified paths.
#   docs_changed  true when a documentation path changed.
#
# Exit status:
#   0 if the change set was classified and all outputs were written.
#   2 if DIFF_MODE is unsupported.
#   A nonzero status if a Git operation or output write fails.

set -euo pipefail

base_sha="${1:-}"
head_sha="${2:-}"
diff_mode="${3:-merge-base}"
full_ci_required=false
docs_changed=false

# Return success if the given object ID contains only zeros.
#
# An all-zero value represents a missing commit rather than a Git object.
#
# Arguments:
#   $1: Git object ID to inspect.
#
# Returns:
#   0 if the value contains only zeros.
#   1 otherwise.
is_zero_sha() {
  local sha="$1"

  [[ "${sha}" =~ ^0+$ ]]
}

# Return success if the given path requires the full CI suite.
#
# Arguments:
#   $1: Repository-relative file path.
#
# Returns:
#   0 if the path requires the full CI suite.
#   1 otherwise.
is_full_ci_path() {
  local path="$1"

  case "${path}" in
    qamomile/* | tests/* | pyproject.toml | uv.lock | .python-version)
      return 0
      ;;
    .github/workflows/* | .github/scripts/* | .github/ci-skip-paths/*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

# Return success if the given path should trigger documentation checks.
#
# Arguments:
#   $1: Repository-relative file path.
#
# Returns:
#   0 if the path should trigger documentation checks.
#   1 otherwise.
is_docs_path() {
  local path="$1"

  case "${path}" in
    docs/*)
      return 0
      ;;
    *.ipynb)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

# Return success if the given path matches a configured CI-skip pattern.
#
# Arguments:
#   $1: Repository-relative file path.
#
# Globals:
#   ci_skip_paths_file: File containing one shell pattern per line.
#
# Returns:
#   0 if the path can skip all quality checks.
#   1 otherwise.
is_ci_skip_path() {
  local path="$1"
  local pattern

  [[ -n "${ci_skip_paths_file}" && -f "${ci_skip_paths_file}" ]] || return 1

  while IFS= read -r pattern || [[ -n "${pattern}" ]]; do
    # Trim \r in case that this script is executed on Windows system (CRLF).
    pattern="${pattern%$'\r'}"
    # Continue if pattern is empty or is not starting with #, which is a comment.
    [[ -n "${pattern}" && "${pattern}" != \#* ]] || continue
    if [[ "${path}" == ${pattern} ]]; then
      return 0
    fi
  done < "${ci_skip_paths_file}"
  return 1
}

if [[ -z "${base_sha}" || -z "${head_sha}" ]] || is_zero_sha "${base_sha}" || is_zero_sha "${head_sha}"; then
  # If either SHA is missing or all zeros, fail safe by requesting full and documentation CI.
  full_ci_required=true
  docs_changed=true
else
  # Set the differentiation range according to the differentiation mode.
  # Exit with an error code if the differentiation mode is invalid.
  case "${diff_mode}" in
    merge-base)
      diff_range="${base_sha}...${head_sha}"
      ;;
    direct)
      diff_range="${base_sha}..${head_sha}"
      ;;
    *)
      echo "Unsupported diff mode: ${diff_mode}" >&2
      exit 2
      ;;
  esac

  # Iniitalise the files with settings of removing them all automatically when 
  # the process is finished.
  changed_paths_file="$(mktemp)"
  ci_skip_paths_file="$(mktemp)"
  ci_skip_path_files="$(mktemp)"
  trap 'rm -f "${changed_paths_file}" "${ci_skip_paths_file}" "${ci_skip_path_files}"' EXIT

  # Initialise ci_skip_paths_file.
  #   ":": Do nothing.
  #   >: Create an empty file (even if that exists).
  : > "${ci_skip_paths_file}"
  # Read the files in .github/ci-skip-paths/.
  #   git ls-tree: Show every files in the specified commit.
  #   -r: recursively.
  #   --name-only: without SHAs and file types.
  #   -z: with the delimiter NUL instead of a breakline (\n).
  git ls-tree -r --name-only -z "${base_sha}" -- .github/ci-skip-paths/ \
    > "${ci_skip_path_files}"
  #   -d '': until NUL
  while IFS= read -r -d '' ci_skip_path_file; do
    case "${ci_skip_path_file}" in
      .github/ci-skip-paths/*.txt)
        git show "${base_sha}:${ci_skip_path_file}" >> "${ci_skip_paths_file}"
        printf '\n' >> "${ci_skip_paths_file}"
        ;;
    esac
  done < "${ci_skip_path_files}"

  # --no-renames: Treat rename (R) as delete an old path and create a new path.
  git diff --no-renames --name-only -z "${diff_range}" > "${changed_paths_file}"

  while IFS= read -r -d '' changed_file; do
    if is_full_ci_path "${changed_file}"; then
      full_ci_required=true
    elif is_docs_path "${changed_file}"; then
      docs_changed=true
    elif ! is_ci_skip_path "${changed_file}"; then
      # Set full_ci_required true if unknown file comes in.
      full_ci_required=true
    fi
  done < "${changed_paths_file}"
fi

{
  echo "full_ci_required=${full_ci_required}"
  echo "docs_changed=${docs_changed}"
  # Keep this marker last so a partial write can never look complete.
  echo "classification_complete=true"
} >> "${GITHUB_OUTPUT}"
