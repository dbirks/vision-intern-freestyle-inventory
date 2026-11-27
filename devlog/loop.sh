#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p devlog/site
LOG_FILE="devlog/site/_loop.log"

WATCH_PID=""
cleanup() {
  if [[ -n "$WATCH_PID" ]]; then
    kill "$WATCH_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

start_watch() {
  uv run python devlog/build.py --watch >>"$LOG_FILE" 2>&1 &
  WATCH_PID=$!
}

start_watch

while true; do
  {
    printf '=== [%s] devlog refresh ===\n' "$(date -Iseconds)"
    uv run python devlog/collect.py || printf 'collect.py failed (exit %d)\n' "$?"
    if [[ -z "$WATCH_PID" ]] || ! kill -0 "$WATCH_PID" 2>/dev/null; then
      printf 'build.py --watch not running; restarting\n'
      start_watch
    fi
    uv run python devlog/build.py || printf 'build.py failed (exit %d)\n' "$?"
  } >>"$LOG_FILE" 2>&1
  sleep 900
done
