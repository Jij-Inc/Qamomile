#!/usr/bin/env bash

# Classify the files changed between two commits for CI job routing.
#
# Outputs written to GITHUB_OUTPUT:
#   docs_only     true when no code file changed and at least one docs/ file
#                 did; the test matrices and lint/type jobs are skipped and
#                 documentation tests run selectively. Agent guidance files
#                 may accompany the docs.
#   skip_all      true when every changed file is AI-agent guidance that no
#                 CI job consumes; every job is skipped.
#   changed_files newline-separated repository-relative changed paths,
#                 emitted only when docs_only is true (its sole consumer is
#                 the selective documentation run, and unconditional emission
#                 would risk the GITHUB_OUTPUT size limit on huge diffs).
#
# Missing SHAs (push, workflow_dispatch) keep both flags false so those
# events retain full coverage.

set -euo pipefail

base_sha="${1:-}"
head_sha="${2:-}"
docs_only=false
skip_all=false
changed_files=""

if [[ -n "${base_sha}" && -n "${head_sha}" ]]; then
  # --no-renames: rename detection would list only the destination path, so
  # a code file moved under docs/ or .claude/ would classify as docs/agent
  # and skip the very tests that would catch the now-missing module.
  changed_files="$(git diff --no-renames --name-only "${base_sha}...${head_sha}")"

  if [[ -n "${changed_files}" ]]; then
    has_docs=false
    has_agent=false
    has_code=false
    while IFS= read -r changed_file; do
      case "${changed_file}" in
        docs/*)
          has_docs=true
          ;;
        .claude/* | .codex | .codex/* | AGENTS.md | CLAUDE.md)
          has_agent=true
          ;;
        *)
          has_code=true
          break
          ;;
      esac
    done <<< "${changed_files}"

    if [[ "${has_code}" == false ]]; then
      if [[ "${has_docs}" == true ]]; then
        docs_only=true
      elif [[ "${has_agent}" == true ]]; then
        skip_all=true
      fi
    fi
  fi
fi

# A random heredoc delimiter so no repository path in the diff can terminate
# the block early and inject extra outputs such as docs_only=true.
delimiter="changed_files_$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')"

if [[ "${docs_only}" != true ]]; then
  changed_files=""
fi

{
  echo "docs_only=${docs_only}"
  echo "skip_all=${skip_all}"
  echo "changed_files<<${delimiter}"
  printf '%s\n' "${changed_files}"
  echo "${delimiter}"
} >> "${GITHUB_OUTPUT}"
