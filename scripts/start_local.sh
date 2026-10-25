#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${GENCO_WEB_PYTHON:-.venv/bin/python}"
PORT="${GENCO_PORT:-8017}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Web Python not found: $PYTHON_BIN" >&2
  echo "Create it with: python -m venv .venv && .venv/bin/python -m pip install -r requirements.txt" >&2
  exit 2
fi

"$PYTHON_BIN" -c 'import fastapi, uvicorn, yaml' || {
  echo "Install web dependencies with: $PYTHON_BIN -m pip install -r requirements.txt" >&2
  exit 2
}
if ! command -v npm >/dev/null 2>&1; then
  echo "Node.js/npm is required to build the local web interface." >&2
  exit 2
fi

npm run check
npm run build

args=(--host 127.0.0.1 --port "$PORT")
if [[ -f .env.local ]]; then
  args+=(--env-file .env.local)
fi

echo "Starting GENCO Studio at http://127.0.0.1:$PORT/"
exec "$PYTHON_BIN" -m uvicorn backend.main:app "${args[@]}"
