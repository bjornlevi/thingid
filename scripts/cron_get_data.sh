#!/usr/bin/env bash

set -euo pipefail

# Resolve repository root even when triggered from cron.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Optional virtualenv activation if present at .venv.
if [ -d "$REPO_ROOT/.venv/bin" ]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.venv/bin/activate"
fi

PYTHON=${PYTHON:-python3}
DB_PATH=${DB_PATH:-"$REPO_ROOT/data/althingi.db"}
SCHEMA=${SCHEMA:-"$REPO_ROOT/schema_map.json"}
MODELS_DIR=${MODELS_DIR:-"$REPO_ROOT/app"}
LOG_DIR=${LOG_DIR:-"$REPO_ROOT/data/logs"}

mkdir -p "$(dirname "$DB_PATH")" "$LOG_DIR"

timestamp="$(date -Iseconds)"
{
  echo "[$timestamp] Running get_data..."
  "$PYTHON" scripts/get_data.py --db "$DB_PATH" --schema "$SCHEMA" --models-dir "$MODELS_DIR"
  echo "[$(date -Iseconds)] get_data finished"
} >> "$LOG_DIR/get_data.log" 2>&1
