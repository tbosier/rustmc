#!/usr/bin/env bash
# Verify that version in Cargo.toml and pyproject.toml matches the git tag (e.g. v0.6.0 -> 0.6.0).
# Run from repo root. Usage: ./scripts/verify_version.sh
set -e
TAG="${1:-${GITHUB_REF#refs/tags/}}"
if [[ ! "$TAG" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "No version tag (e.g. v0.6.0). Skipping check."
  exit 0
fi
EXPECTED="${TAG#v}"
CORE=$(grep '^version = ' rust_core/Cargo.toml | sed 's/.*"\(.*\)"/\1/')
BINDINGS=$(grep '^version = ' python_bindings/Cargo.toml | sed 's/.*"\(.*\)"/\1/')
PY=$(grep '^version = ' pyproject.toml | head -1 | sed 's/.*= *"\(.*\)"/\1/')
if [[ "$CORE" != "$EXPECTED" || "$BINDINGS" != "$EXPECTED" || "$PY" != "$EXPECTED" ]]; then
  echo "Version mismatch: tag=$TAG (expected $EXPECTED)"
  echo "  rust_core:          $CORE"
  echo "  python_bindings:    $BINDINGS"
  echo "  pyproject.toml:     $PY"
  exit 1
fi
echo "Version OK: $EXPECTED"
