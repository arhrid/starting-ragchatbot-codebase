#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

echo "Running black..."
uv run black backend/ main.py "$@"
echo "Done."
