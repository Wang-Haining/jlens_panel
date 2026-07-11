#!/bin/bash
# Provision ignored Tempest inputs after cloning the tracked GitHub branch.

set -euo pipefail

ROOT=${JLENS_PANEL_ROOT:-/home/g91p721/jlens_panel}
BRANCH=${JLENS_PANEL_BRANCH:-codex/dirty-run-poc}
UPSTREAM_COMMIT=581d398613e5602a5af361e1c34d3a92ea82ba8e

cd "${ROOT}"
git fetch origin "${BRANCH}"
git checkout "${BRANCH}"
git pull --ff-only origin "${BRANCH}"

mkdir -p data/fit_corpus logs vendor

SOURCE_CORPUS=/home/g91p721/jd/data/fit_corpus/c4_slice.json
TARGET_CORPUS=${ROOT}/data/fit_corpus/c4_slice.json
if [[ ! -f "${TARGET_CORPUS}" ]]; then
  cp "${SOURCE_CORPUS}" "${TARGET_CORPUS}"
elif ! cmp -s "${SOURCE_CORPUS}" "${TARGET_CORPUS}"; then
  echo "Refusing to overwrite a different fit corpus: ${TARGET_CORPUS}" >&2
  exit 1
fi

if [[ ! -d vendor/jacobian-lens/.git ]]; then
  git clone https://github.com/anthropics/jacobian-lens.git vendor/jacobian-lens
fi
git -C vendor/jacobian-lens checkout "${UPSTREAM_COMMIT}"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Tracked checkout is dirty after provisioning" >&2
  git status --short >&2
  exit 1
fi

echo "Tempest inputs provisioned at ${ROOT}"
