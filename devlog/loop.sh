#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p devlog/site
LOG_FILE="devlog/site/_loop.log"

while true; do
  {
    printf '=== [%s] devlog refresh ===\n' "$(date -Iseconds)"
    uv run python devlog/collect.py || printf 'collect.py failed (exit %d)\n' "$?"
    uv run python devlog/build.py || printf 'build.py failed (exit %d)\n' "$?"
  } >>"$LOG_FILE" 2>&1
  sleep 900
done
