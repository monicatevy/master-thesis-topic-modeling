#!/bin/bash
# ───────── Runner: Babelfy annotation ─────────
# Usage:
#   ./run_babelfy.sh <input_json> <output_dir>
#
# Output:
#   - <output_dir>/<basename>.csv       ← annotations
#   - <output_dir>/_meta/<basename>_done.json  ← progress tracking
#
# Args:
#   $1 : Input JSON corpus ({doc_id: text})
#   $2 : Output directory for annotations
# ─────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"           # .../src/scripts
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"       # .../ (racine du projet)
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

KEY="yourkey"

LIMIT=40
LANG="EN"

INPUT_FILE="${1:?Usage: $0 <input_json> <output_dir>}"
OUTPUT_DIR="${2:?Usage: $0 <input_json> <output_dir>}"

base=$(basename "$INPUT_FILE" .json)
META_DIR="$OUTPUT_DIR/_meta"
mkdir -p "$OUTPUT_DIR" "$META_DIR"

python "$PROJECT_ROOT/src/crea/preprocessing/babelfy.py" \
       --input-type json \
       --input  "$INPUT_FILE" \
       --output "$OUTPUT_DIR/${base}.csv" \
       --done   "$META_DIR/${base}_done.json" \
       --key    "$KEY" \
       --limit  "$LIMIT" \
       --lang   "$LANG"