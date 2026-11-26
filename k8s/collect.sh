#!/usr/bin/env bash
# Copy a finished (or failed) job's /out back to the host: k8s/collect.sh <name> [dest]
#
# dest defaults to runs/<name>/ (gitignored). Reads /work/artifacts/<run-id> out of the
# node-local outputs PVC through that node's outputs workspace pod with kubectl cp, verifies
# file checksums against the remote tree, records the pod's runtime image digest in env.txt,
# and publishes atomically into a fresh destination directory.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

NAME="${1:-}"; DEST="${2:-}"
[ -n "$NAME" ] || { sed -n '2,8p' "$0" >&2; exit 2; }
validate_dns_label "$NAME" "job name"
DEST=${DEST:-"runs/$NAME"}

if [ -e "$DEST" ] && [ -n "$(ls -A "$DEST" 2>/dev/null || true)" ]; then
  die "destination '$DEST' already exists and is non-empty (refusing to merge/overwrite)"
fi

job_json=$(kctl get job "$NAME" -o json 2>/dev/null || true)
[ -n "$job_json" ] || die "no job/$NAME in namespace $VI_NS (context $VI_CONTEXT)"

meta=$(printf '%s' "$job_json" | python3 -c '
import json, sys
app_want = sys.argv[1]
obj = json.loads(sys.stdin.read())
meta = obj.get("metadata", {})
labels = meta.get("labels", {}) or {}
anns = meta.get("annotations", {}) or {}
spec = obj.get("spec", {}).get("template", {}).get("spec", {})
node = (spec.get("nodeSelector") or {}).get("kubernetes.io/hostname", "")
conds = {c.get("type"): c.get("status") for c in (obj.get("status") or {}).get("conditions", [])}
job_name = meta.get("name")
if labels.get("app") != app_want:
    sys.stderr.write(f"error: job/{job_name} does not carry app={app_want}\n")
    sys.exit(1)
gpu = labels.get("vision-intern/gpu", "")
if gpu not in ("5090", "6000"):
    sys.stderr.write(f"error: job/{job_name} has invalid vision-intern/gpu={gpu!r}\n")
    sys.exit(1)
run_id = anns.get("vision-intern/run-id") or job_name
comp = conds.get("Complete", "False")
fail = conds.get("Failed", "False")
print(f"{gpu}|{node}|{run_id}|{comp}|{fail}")
' "$VI_LABEL_APP")
IFS='|' read -r gpu node run_id ok bad <<< "$meta"

log "job/$NAME run_id=$run_id gpu=$gpu node=$node complete=$ok failed=$bad"
[ "$ok" = "True" ] || log "warning: job has not reported Complete"

node2=$(gpu_node "$gpu")
[ "$node" = "$node2" ] || die "job ran on $node but our $gpu PVC lives on $node2 -- collect manually"
ensure_out_workspace "$gpu"
ws=$(out_ws_name "$gpu")
remote_dir="/work/artifacts/$run_id"

remote_manifest=$(out_ws_sh "$gpu" "if [ -d '$remote_dir' ]; then cd '$remote_dir' && find . -type f | sort | while IFS= read -r f; do sha256sum \"\$f\"; done; fi" || true)
[ -n "$remote_manifest" ] || die "job/$NAME produced no files under $remote_dir"

log "copying $ws:$remote_dir -> $DEST"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
kctl cp "$ws:$remote_dir" "$tmp/payload" >/dev/null
src="$tmp/payload"
[ -d "$src/$run_id" ] && src="$tmp/payload/$run_id"
[ -d "$src" ] || die "download produced no directory"

# Verify every downloaded file matches remote sha256sum.
(
  cd "$src"
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    want_sha="${line%% *}"
    rel="${line#*  }"
    [ -f "$rel" ] || die "missing downloaded artifact file: $rel"
    got_sha=$(sha256_file "$rel")
    [ "$got_sha" = "$want_sha" ] || die "checksum mismatch on $rel (want $want_sha, got $got_sha)"
  done <<< "$remote_manifest"
)

pod_image_id=$(kctl get pods -l "vision-intern/job=$NAME" -o jsonpath='{.items[0].status.containerStatuses[0].imageID}' 2>/dev/null || true)
if [ -n "$pod_image_id" ]; then
  if [ ! -f "$src/env.txt" ] || ! grep -qx "pod_image_id=.*" "$src/env.txt" 2>/dev/null; then
    printf 'pod_image_id=%s\n' "$pod_image_id" >> "$src/env.txt"
  fi
fi

# Atomic publish into destination.
parent_dir=$(dirname "$DEST")
mkdir -p "$parent_dir"
stage=$(mktemp -d "$parent_dir/.collect.tmp.XXXXXX")
cp -a "$src"/. "$stage"/
if [ -d "$DEST" ]; then
  # Existing empty dir (e.g. tempfile.TemporaryDirectory in launch.py)
  ( shopt -s dotglob nullglob; mv "$stage"/* "$DEST"/ )
  rmdir "$stage"
else
  mv "$stage" "$DEST"
fi

log "got:"
( cd "$DEST" && du -sh . && find . -type f | sort | sed 's/^/  /' )
