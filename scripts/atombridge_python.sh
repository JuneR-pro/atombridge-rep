#!/usr/bin/env bash
# Run AtomBridge with the existing OpenPI interpreter and project-local packages.
set -euo pipefail

atombridge_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$atombridge_root/scripts/activate_atombridge.sh"
exec "$ATOMBRIDGE_BASE_PYTHON" "$@"
