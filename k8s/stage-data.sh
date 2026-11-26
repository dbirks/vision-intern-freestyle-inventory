#!/usr/bin/env bash
# Put data/ (and the repo code) onto cluster storage so GPU jobs can read it.
#
#   k8s/stage-data.sh [--gpu 5090|6000|all] [--no-code] [--no-data] [--list]
#
# What it does per target GPU:
#   1. ensures the node-local work PVC (k8s/pvc-template.yaml) + the busybox workspace pod
#      (k8s/workspace-template.yaml) that holds it open on that node;
#   2. streams data/ into it as a tar (over kubectl exec, no registry, no node shell);
#   3. streams a tarball of the tracked repo files into /work/repo/repo.tar.
# Jobs mount the PVC: /work/data (read-only), /work/repo/repo.tar, /out (writable).
#
# Images under data/ are private: they only ever go to our own cluster, never a registry,
# never GitHub. Tar streams are plain tar (jpgs do not compress).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

GPUS="all"; DO_CODE=1; DO_DATA=1; LIST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu) GPUS="$2"; shift 2 ;;
    --no-code) DO_CODE=0; shift ;;
    --no-data) DO_DATA=0; shift ;;
    --list) LIST=1; shift ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) die "unknown arg $1" ;;
  esac
done
[ "$GPUS" = all ] && GPUS="5090 6000"

if [ "$LIST" = 1 ]; then
  for g in $GPUS; do
    echo "== gpu $g  node $(gpu_node "$g")  pvc $(pvc_name "$g")"
    ws_sh "$g" 'du -sh /work/* 2>/dev/null; find /work/data -maxdepth 2 | head -20' 2>&1 || true
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
  log "packing repo (tracked + new untracked files, minus data/ runs/ artifacts/ .venv/)"
  repo_tarball "$tmp/repo.tar" >/dev/null
  log "repo.tar = $(du -h "$tmp/repo.tar" | cut -f1) ($(tar -tf "$tmp/repo.tar" | wc -l) entries)"
fi

for g in $GPUS; do
  ws=$(ws_name "$g")
  ensure_workspace "$g"
  if [ "$DO_DATA" = 1 ]; then
    push_file "$g" "$tmp/data.tar" /work/data.tar
    # chmod first: a previous job can leave /work/data read-only (0555), and the
    # workspace pod runs uid 0 with capabilities dropped -- as owner it can chmod, but
    # `rm -rf` would otherwise fail to unlink inside the non-writable directory.
    ws_sh "$g" 'chmod -R u+w /work/data 2>/dev/null; rm -rf /work/data && tar -C /work -xf /work/data.tar && rm -f /work/data.tar'
  fi
  if [ "$DO_CODE" = 1 ]; then
    push_file "$g" "$tmp/repo.tar" /work/repo/repo.tar
    ws_sh "$g" 'rm -f /work/repo.tar'
  fi
  log "gpu $g work dir now:"
  ws_sh "$g" 'du -sh /work/* 2>/dev/null || true' || true
done
log "staged to: $GPUS"
