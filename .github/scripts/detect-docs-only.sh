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

{
  echo "docs_only=${docs_only}"
  echo "changed_files<<__CHANGED_FILES__"
  printf '%s\n' "${changed_files}"
  echo "__CHANGED_FILES__"
} >> "${GITHUB_OUTPUT}"
