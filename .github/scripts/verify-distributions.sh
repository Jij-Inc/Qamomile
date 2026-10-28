#!/usr/bin/env bash

# Verify distribution names, versions, installation, and importability.
#
# Usage:
#   verify-distributions.sh
#
# Arguments:
#   None.
#
# Environment:
#   EXPECTED_VERSION:
#     Optional package version that both distributions must contain.
#
# Inputs:
#   dist/qamomile-*.whl     exactly one wheel.
#   dist/qamomile-*.tar.gz  exactly one source distribution.
#
# Working directory:
#   Directory containing dist/.
#
# Outputs:
#   Writes installation diagnostics to the standard streams and validation
#   errors or failed assertions to stderr.
#
# Exit status:
#   0 if both artifacts have the same expected version and can be installed
#   and imported in isolated environments.
#   1 if artifact discovery or version validation fails.
#   A nonzero status if isolated installation or import verification fails.

set -euo pipefail

shopt -s nullglob
wheels=(dist/qamomile-*.whl)
sdists=(dist/qamomile-*.tar.gz)
if [[ "${#wheels[@]}" -ne 1 || "${#sdists[@]}" -ne 1 ]]; then
  echo "Expected exactly one qamomile wheel and source distribution." >&2
  exit 1
fi

wheel_name="${wheels[0]#dist/}"
sdist_name="${sdists[0]#dist/}"
if [[ ! "${sdist_name}" =~ ^qamomile-(.+)\.tar\.gz$ ]]; then
  echo "Could not extract the version from the source distribution name." >&2
  exit 1
fi
package_version="${BASH_REMATCH[1]}"

if [[ "${wheel_name}" != "qamomile-${package_version}-"*.whl ]]; then
  echo "Wheel and source distribution versions do not match." >&2
  exit 1
fi

if [[ -n "${EXPECTED_VERSION:-}" && "${package_version}" != "${EXPECTED_VERSION}" ]]; then
  echo "Distribution version mismatch: expected ${EXPECTED_VERSION}, got ${package_version}." >&2
  exit 1
fi

for artifact in "${wheels[0]}" "${sdists[0]}"; do
  EXPECTED_VERSION="${package_version}" \
    uv run --isolated --no-project --with "${artifact}" \
    python -I -c \
    'import importlib.metadata as m, os; import qamomile.circuit; assert m.version("qamomile") == os.environ["EXPECTED_VERSION"]'
done
