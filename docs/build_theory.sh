#!/usr/bin/env bash
# Build Theoretical_Framework.pdf from docs/theory.md
# Usage: bash docs/build_theory.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

pandoc "$SCRIPT_DIR/theory.md" \
    -o "$REPO_ROOT/Theoretical_Framework.pdf" \
    --pdf-engine=xelatex \
    --variable=fontsize:12pt \
    --variable=geometry:"margin=1in" \
    -V mainfont="Times New Roman" \
    -V mathfont="STIX Two Math" \
    -V monofont="Courier New" \
    -V monofontoptions="Scale=MatchLowercase" \
    -H "$SCRIPT_DIR/header.tex"

echo "Built Theoretical_Framework.pdf"
