#!/usr/bin/env bash

set -euo pipefail

base_sha="${1:-}"
head_sha="${2:-}"
docs_only=false
changed_files=""

if [[ -n "${base_sha}" && -n "${head_sha}" ]]; then
  changed_files="$(git diff --name-only "${base_sha}...${head_sha}")"

  if [[ -n "${changed_files}" ]]; then
    docs_only=true
    while IFS= read -r changed_file; do
      if [[ "${changed_file}" != docs/* ]]; then
        docs_only=false
        break
      fi
    done <<< "${changed_files}"
  fi
fi

# A random heredoc delimiter so no repository path in the diff can terminate
# the block early and inject extra outputs such as docs_only=true.
delimiter="changed_files_$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')"

{
  echo "docs_only=${docs_only}"
  echo "changed_files<<${delimiter}"
  printf '%s\n' "${changed_files}"
  echo "${delimiter}"
} >> "${GITHUB_OUTPUT}"
