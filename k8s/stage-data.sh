#!/usr/bin/env bash
# Put data/ (and baseline repo code) onto cluster storage so GPU jobs can read it.
#
#   k8s/stage-data.sh [--gpu 5090|6000|all] [--no-code] [--no-data] [--list]
#
# What it does per target GPU:
#   1. ensures the node-local data PVC + workspace pod (mounted readOnly: true by jobs),
#      and the separate writable outputs PVC + outputs workspace pod;
#   2. streams data/ into the data PVC as a tar (verified by size + sha256);
#   3. streams a baseline repo tarball + provenance metadata into the outputs PVC at
#      /work/repo/repo.tar (for --no-code runs).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

GPUS="all"; DO_CODE=1; DO_DATA=1; LIST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu) [ $# -ge 2 ] || die "--gpu requires an argument"; GPUS="$2"; shift 2 ;;
    --no-code) DO_CODE=0; shift ;;
    --no-data) DO_DATA=0; shift ;;
    --list) LIST=1; shift ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) die "unknown arg $1" ;;
  esac
done
case "$GPUS" in
  all) GPUS="5090 6000" ;;
  5090|6000) : ;;
  *) die "--gpu must be 5090, 6000, or all (got '$GPUS')" ;;
esac

if [ "$LIST" = 1 ]; then
  for g in $GPUS; do
    echo "== gpu $g  node $(gpu_node "$g")  data_pvc $(pvc_name "$g")  out_pvc $(out_pvc_name "$g")"
    ensure_workspace "$g"
    ws_sh "$g" 'du -sh /work/* 2>/dev/null; find /work/data -maxdepth 2 | head -20' 2>&1 || true
    if kctl get pod "$(out_ws_name "$g")" >/dev/null 2>&1; then
      ensure_out_workspace "$g"
      out_ws_sh "$g" 'du -sh /work/* 2>/dev/null || true' 2>&1 || true
    fi
  done
  exit 0
fi

tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

if [ "$DO_DATA" = 1 ]; then
  [ -d "$REPO_ROOT/data" ] || die "no data/ in $REPO_ROOT"
  log "packing data/ (jpgs: no gzip)"
  tar -C "$REPO_ROOT" -cf "$tmp/data.tar" data
  log "data.tar = $(du -h "$tmp/data.tar" | cut -f1)"
fi

if [ "$DO_CODE" = 1 ]; then
  log "packing baseline repo (tracked + new untracked files, minus data/ runs/ artifacts/ .venv/)"
  prov_json=$(git_provenance_json)
  repo_tarball "$tmp/repo.tar" >/dev/null
  repo_sha=$(sha256_file "$tmp/repo.tar")
  python3 -c '
import json, sys, time
prov = json.loads(sys.argv[1])
prov.update({"repo_tar_sha256": sys.argv[2], "staged_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
print(json.dumps(prov, indent=2, sort_keys=True))
' "$prov_json" "$repo_sha" > "$tmp/repo.tar.meta.json"
  log "repo.tar = $(du -h "$tmp/repo.tar" | cut -f1) ($(tar -tf "$tmp/repo.tar" | wc -l) entries, sha256=${repo_sha:0:12})"
fi

for g in $GPUS; do
  if [ "$DO_DATA" = 1 ]; then
    ensure_workspace "$g"
    push_file "$g" "$tmp/data.tar" /work/data.tar
    ws_sh "$g" 'chmod -R u+w /work/data 2>/dev/null || true; rm -rf /work/data.next && mkdir -p /work/data.next && tar -C /work/data.next -xf /work/data.tar && rm -rf /work/data.prev && ( [ ! -d /work/data ] || mv /work/data /work/data.prev ) && mv /work/data.next/data /work/data && rm -rf /work/data.next /work/data.prev /work/data.tar'
    log "gpu $g data dir:"
    ws_sh "$g" 'du -sh /work/data 2>/dev/null || true' || true
  fi
  if [ "$DO_CODE" = 1 ]; then
    ensure_out_workspace "$g"
    push_out_file "$g" "$tmp/repo.tar" /work/repo/repo.tar
    push_out_file "$g" "$tmp/repo.tar.meta.json" /work/repo/repo.tar.meta.json
    log "gpu $g outputs dir:"
    out_ws_sh "$g" 'du -sh /work/* 2>/dev/null || true' || true
  fi
done
log "staged to: $GPUS"
