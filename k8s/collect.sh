#!/usr/bin/env bash
# Copy a finished job's /out back to the host: k8s/collect.sh <name> [dest]
#
# dest defaults to runs/<name>/ (gitignored). Reads /work/artifacts/<name> out of the
# node-local work PVC through that node's workspace pod with kubectl cp (which is tar over
# exec under the hood; plain `kubectl exec -i -- cat` truncates on this cluster).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

NAME="${1:-}"; DEST="${2:-}"
[ -n "$NAME" ] || { sed -n '2,7p' "$0"; exit 2; }
DEST=${DEST:-"runs/$NAME"}

job=$(kctl get job "$NAME" -o jsonpath='{.metadata.name}' 2>/dev/null || true)
[ -n "$job" ] || die "no job/$NAME in namespace $VI_NS (context $VI_CONTEXT)"
gpu=$(kctl get job "$NAME" -o jsonpath='{.metadata.labels.vision-intern/gpu}')
node=$(kctl get job "$NAME" -o jsonpath='{.spec.template.spec.nodeSelector.kubernetes\.io/hostname}')
ok=$(kctl get job "$NAME" -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}')
bad=$(kctl get job "$NAME" -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}')
[ -n "$gpu" ] || die "job/$NAME has no vision-intern/gpu label; not ours"
log "job/$NAME gpu=$gpu node=$node complete=${ok:-False} failed=${bad:-False}"
[ "$ok" = "True" ] || log "warning: job has not reported Complete"

ws=$(ws_name "$gpu"); node2=$(gpu_node "$gpu")
[ "$node" = "$node2" ] || die "job ran on $node but our $gpu PVC lives on $node2 -- collect manually"
ensure_workspace "$gpu"

have=$(ws_sh "$gpu" "ls -A /work/artifacts/$NAME 2>/dev/null | wc -l" || echo 0)
[ "${have:-0}" -gt 0 ] || die "job/$NAME produced nothing under /work/artifacts/$NAME"

log "copying /work/artifacts/$NAME -> $DEST"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
# kubectl cp nests a copied directory under a subdir named after it; copy into a scratch
# dir and move the payload, so runs/<name>/ holds the artifacts directly.
kctl cp "$(ws_name "$gpu"):/work/artifacts/$NAME" "$tmp/payload" >/dev/null 2>&1 || true
src="$tmp/payload"; [ -d "$src/$NAME" ] && src="$tmp/payload/$NAME"
[ -d "$src" ] || die "download produced nothing"
[ "$(find "$src" -type f | wc -l)" -gt 0 ] || die "job/$NAME produced no files under /work/artifacts/$NAME"
mkdir -p "$DEST"; cp -a "$src"/. "$DEST"/
log "got:"
( cd "$DEST" && du -sh . && find . -type f | sed 's/^/  /' )
