#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -gt 1 ]]; then
  echo "Usage: $0 [distribution-directory]" >&2
  exit 2
fi

distribution_directory="${1:-dist}"
arguments=("${distribution_directory}")
if [[ -n "${EXPECTED_VERSION:-}" ]]; then
  arguments+=(--expected-version "${EXPECTED_VERSION}")
fi
uv run --isolated --no-project \
  python .github/scripts/verify-distributions.py "${arguments[@]}"

shopt -s nullglob
wheels=("${distribution_directory}"/qamomile-*.whl)
source_archives=("${distribution_directory}"/qamomile-*.tar.gz)
if [[ "${#wheels[@]}" -ne 1 || "${#source_archives[@]}" -ne 1 ]]; then
  echo "Expected one combined wheel and one source archive." >&2
  exit 1
fi
source_name="${source_archives[0]#"${distribution_directory}"/qamomile-}"
EXPECTED_VERSION="${EXPECTED_VERSION:-${source_name%.tar.gz}}" \
  uv run --isolated --no-project --with "${wheels[0]}" \
    python -I .github/scripts/smoke-installed-distribution.py
