#!/usr/bin/env bash

# Validate that tag metadata identifies the same tag, commit, and package
# version, and that the commit is on the main branch's
# first-parent history.
#
# Usage:
#   validate-tag-and-export-version.sh
#
# Arguments:
#   None.
#
# Environment:
#   EVENT_REF:
#     Required full Git ref from the release event.
#   EVENT_SHA:
#     Required commit SHA from the release event.
#   RELEASE_TAG:
#     Required release tag in vX.Y.Z form with an optional alpha, beta, or rc
#     suffix and number.
#   GITHUB_ENV:
#     Required path to the GitHub Actions environment file.
#   RUNNER_TEMP:
#     Required private runner temporary directory.
#
# Required repository state:
#   refs/remotes/origin/main must be fetched by the trusted caller.
#   RELEASE_TAG and HEAD must resolve in the local repository.
#
# Outputs written to GITHUB_ENV:
#   EXPECTED_VERSION  normalized Python package version derived from the tag.
#
# Exit status:
#   0 if the event, tag, commit, history, and version agree.
#   1 if validation fails.
#   A nonzero status if a Git operation or output write fails.

set -euo pipefail

: "${EVENT_REF:?EVENT_REF is required.}"
: "${EVENT_SHA:?EVENT_SHA is required.}"
: "${RELEASE_TAG:?RELEASE_TAG is required.}"
: "${GITHUB_ENV:?GITHUB_ENV is required.}"
: "${RUNNER_TEMP:?RUNNER_TEMP is required.}"

# Check if the release tag matches the expected format and extract version components.
if [[ ! "${RELEASE_TAG}" =~ ^v([0-9]+)\.([0-9]+)\.([0-9]+)(-(alpha|beta|rc)([.-]?[0-9]+)?)?$ ]]; then
  echo "Release tags must have the form vX.Y.Z with an optional alpha, beta, or rc suffix." >&2
  exit 1
fi
# Construct the expected version string from the release tag components.
expected_version="${BASH_REMATCH[1]}.${BASH_REMATCH[2]}.${BASH_REMATCH[3]}"  # X.Y.Z
# Add the prerelease suffix without hyphen if present.
if [[ -n "${BASH_REMATCH[4]}" ]]; then
  prerelease_number="${BASH_REMATCH[6]#\.}"
  prerelease_number="${prerelease_number#-}"
  prerelease_number="${prerelease_number:-0}"
  case "${BASH_REMATCH[5]}" in
    alpha)
      expected_version+="a${prerelease_number}"
      ;;
    beta)
      expected_version+="b${prerelease_number}"
      ;;
    rc)
      expected_version+="rc${prerelease_number}"
      ;;
    *)
      echo "Unsupported prerelease identifier: ${BASH_REMATCH[5]}" >&2
      exit 1
      ;;
  esac
fi

# Check if the release event ref matches the release tag.
if [[ "${EVENT_REF}" != "refs/tags/${RELEASE_TAG}" ]]; then
  echo "The release event ref does not match the release tag." >&2
  exit 1
fi

# Check if the release event commit and the tag commit match the checked-out commit.
tag_commit="$(git rev-parse "${RELEASE_TAG}^{commit}")"
head_commit="$(git rev-parse HEAD)"
if [[ "${head_commit}" != "${EVENT_SHA}" || "${tag_commit}" != "${EVENT_SHA}" ]]; then
  echo "The release event, checked-out commit, and tag do not agree." >&2
  exit 1
fi

# Check if the release version's commit is in the main branch's first-parent history.
main_history="${RUNNER_TEMP}/main-first-parent-commits"
trap 'rm -f "${main_history}"' EXIT
git rev-list --first-parent refs/remotes/origin/main > "${main_history}"
if ! grep -Fqx -- "${head_commit}" "${main_history}"; then
  echo "The release tag is not on the main branch's first-parent history." >&2
  exit 1
fi

echo "EXPECTED_VERSION=${expected_version}" >> "${GITHUB_ENV}"
