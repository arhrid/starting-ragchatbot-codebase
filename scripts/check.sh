#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

echo "Checking formatting with black..."
uv run black --check backend/ main.py "$@"
echo "All checks passed."
