#!/usr/bin/env bash
# Verify synchronized manifests/dependency/lock and, when supplied, vX.Y.Z tag.
# Works from any directory; Python 3.9+ requires no third-party dependencies.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/verify_version.py" "$@"
