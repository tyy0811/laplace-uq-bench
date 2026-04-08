#!/usr/bin/env bash
# Build Theoretical_Framework.pdf from docs/theory.md
# Usage: bash docs/build_theory.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

pandoc "$SCRIPT_DIR/theory.md" \
    -o "$REPO_ROOT/Theoretical_Framework.pdf" \
    --pdf-engine=xelatex \
    --variable=geometry:margin=1in \
    --variable=fontsize=11pt

echo "Built Theoretical_Framework.pdf"
