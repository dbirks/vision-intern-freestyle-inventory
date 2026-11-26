# shellcheck shell=bash
# Shared helpers for the k8s/*.sh scripts. Source this, do not execute it.
#
# Everything here is scoped to the home cluster (kubectl context admin@home) and to the
# `default` namespace; every object we create carries `app=vision-intern` so it is easy to
# find and easy to delete by label.

set -euo pipefail

VI_CONTEXT="${VI_CONTEXT:-admin@home}"
VI_NS="${VI_NS:-default}"
VI_LABEL_APP="vision-intern"
# CUDA 13.0 matches what the node driver (580.x) reports; runtime image ships torch + cuDNN
# with sm_120 (Blackwell) kernels, so a job needs no torch install for plain training loops.
VI_IMAGE="${VI_IMAGE:-pytorch/pytorch:2.11.0-cuda13.0-cudnn9-runtime}"
VI_WS_IMAGE="${VI_WS_IMAGE:-busybox:1.37}"
VI_PVC_SIZE="${VI_PVC_SIZE:-20Gi}"
VI_STORAGE_CLASS="${VI_STORAGE_CLASS:-local-path}"

export NAMESPACE="$VI_NS" APP_LABEL="$VI_LABEL_APP"

K8S_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$K8S_DIR/.." && pwd)"

die() { echo "error: $*" >&2; exit 1; }
log() { echo "[vision-intern] $*" >&2; }

kctl() { kubectl --context "$VI_CONTEXT" -n "$VI_NS" "$@"; }

# ---- gpu -> node / card type / pvc / workspace pod ---------------------------------------
gpu_node() {
  case "$1" in
    5090) echo "${VI_NODE_5090:-talos-8me-tcy}" ;;
    6000) echo "${VI_NODE_6000:-talos-210-73x}" ;;
    *) die "gpu must be 5090 or 6000 (got '$1')" ;;
  esac
}

# Substring of the HAMi-registered device name (case-insensitive). Never "RTX" -- it
# matches both cards here.
gpu_type() {
  case "$1" in
    5090) echo "RTX 5090" ;;
    6000) echo "PRO 6000" ;;
    *) die "gpu must be 5090 or 6000 (got '$1')" ;;
  esac
}

pvc_name() { echo "vision-intern-data-$1"; }   # one PVC per GPU node: local-path is node-local
ws_name() { echo "vision-intern-ws-$1"; }      # long-lived helper pod that holds the PVC open

# ---- templating ------------------------------------------------------------------------
# envsubst render with a hard failure on any placeholder that is unset in the environment.
# usage: render k8s/job-template.yaml [EXTRA=VALUE ...] > /tmp/out.yaml
render() {
  local tmpl="$1"; shift
  local kv v vars fmt
  local -a env_over=()
  for kv in "$@"; do env_over+=("$kv"); done
  tmpl="$REPO_ROOT/$tmpl"
  [ -f "$tmpl" ] || die "no template $tmpl"
  vars=$(grep -o '\${[A-Za-z_][A-Za-z0-9_]*}' "$tmpl" | tr -d '${}' | sort -u)
  fmt=""
  for v in $vars; do
    if ! printf '%s\n' "${env_over[@]:-}" | grep -qx "${v}=.*" && [ -z "${!v+x}" ]; then
      die "refusing to render $tmpl: \$$v is unset (pass $v=... or export it)"
    fi
    fmt="$fmt\${$v} "
  done
  if [ -n "$fmt" ]; then
    (
      for kv in "${env_over[@]:-}"; do if [ -n "$kv" ]; then export "$kv"; fi; done
      envsubst "$fmt" < "$tmpl"
    )
  else
    cat "$tmpl"
  fi
}

# ---- cluster state helpers -------------------------------------------------------------
# Real physical free VRAM on a node, read from the resident gpu-vram-exporter pod
# (it runs unaccounted with NVIDIA_VISIBLE_DEVICES=all, so it sees the whole card).
free_vram_mib() {
  local node="$1" pod out
  pod=$(kctl get pods -A -o wide --field-selector "spec.nodeName=$node" 2>/dev/null \
        | awk '/gpu-vram-exporter/{print $2; exit}' || true)
  [ -n "$pod" ] || { echo ""; return 0; }
  out=$(kctl exec "$pod" -- nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
        | head -1 | tr -d ' ' || true)
  echo "$out"
}

# HAMi accounting free (what the scheduler will actually let us reserve) for one card type.
hami_free_mib() {
  local node="$1" want="$2" reg
  reg=$(kctl get node "$node" -o jsonpath='{.metadata.annotations.hami\.io/node-nvidia-register}' 2>/dev/null || true)
  [ -n "$reg" ] || { echo ""; return 0; }
  printf '%s' "$reg" | python3 -c '
import json,sys
want=sys.argv[1].lower()
try: reg=json.loads(sys.stdin.read() or "[]")
except Exception: sys.exit(0)
used={}
import subprocess
out=subprocess.run(["kubectl","--context",sys.argv[2],"-n",sys.argv[3],"get","pods","-A","-o","json"],
                   capture_output=True,text=True).stdout
try: pods=json.loads(out)
except Exception: pods={"items":[]}
for p in pods["items"]:
    if p["status"].get("phase") not in ("Running","Pending"): continue
    ann=p["metadata"].get("annotations",{})
    alloc=ann.get("hami.io/vgpu-devices-allocated","")
    for part in alloc.split(";"):
        f=part.split(",")
        if len(f)>=3 and f[2].isdigit(): used[f[0]]=used.get(f[0],0)+int(f[2])
for d in reg:
    if want in (d.get("type","").lower()):
        print(max(0, int(d.get("devmem",0)) - used.get(d.get("id",""),0)))
' "$want" "$VI_CONTEXT" "$VI_NS" 2>/dev/null || echo ""
}

# ---- storage ---------------------------------------------------------------------------
ensure_pvc() { # gpu -> creates the PVC for that node if missing
  local gpu="$1" name="$2"
  if kctl get pvc "$name" >/dev/null 2>&1; then return 0; fi
  log "creating PVC $name ($VI_PVC_SIZE, $VI_STORAGE_CLASS) -- binds to $(gpu_node "$gpu") on first use"
  render k8s/pvc-template.yaml PVC_NAME="$name" STORAGE_CLASS="$VI_STORAGE_CLASS" PVC_SIZE="$VI_PVC_SIZE" \
    | kctl apply -f - >/dev/null
}

ensure_workspace() { # gpu -> ensures the busybox helper pod for that node is Running
  local gpu="$1" pvc ws node phase
  pvc=$(pvc_name "$gpu"); ws=$(ws_name "$gpu"); node=$(gpu_node "$gpu")
  ensure_pvc "$gpu" "$pvc"
  phase=$(kctl get pod "$ws" -o jsonpath='{.status.phase}' 2>/dev/null || true)
  if [ "$phase" = "Running" ]; then return 0; fi
  if [ -n "$phase" ]; then log "workspace pod $ws is $phase, recreating"; kctl delete pod "$ws" --wait=true >/dev/null; fi
  log "starting workspace pod $ws on $node (mounts $pvc at /work)"
  render k8s/workspace-template.yaml WS_NAME="$ws" NODE="$node" PVC_NAME="$pvc" WS_IMAGE="$VI_WS_IMAGE" GPU="$gpu" \
    | kctl apply -f - >/dev/null
  kctl wait --for=jsonpath='{.status.phase}'=Running "pod/$ws" --timeout=180s >/dev/null
}

# ---- file transfer ---------------------------------------------------------------------
# NOTE: raw `kubectl exec -i <pod> -- cat > f` silently truncates at ~192 KiB against this
# cluster with kubectl v1.37.1 (websocket flow control; KUBECTL_REMOTE_COMMAND_WEBSOCKETS=
# false does not help). `kubectl cp` (tar over exec) transfers MBs correctly, so use it.
ws_sh() { kctl exec "$(ws_name "$1")" -- sh -c "$2"; }   # workspace pod shell (short output)

push_file() { # gpu local_path remote_path
  local gpu="$1" src="$2" dst="$3" ws try got want
  ws=$(ws_name "$gpu"); want=$(wc -c < "$src")
  log "uploading $src ($(du -h "$src" | cut -f1)) -> $ws:$dst"
  for try in 1 2 3; do
    kctl cp "$src" "$ws:$dst" >/dev/null 2>&1 || true
    got=$(ws_sh "$gpu" "wc -c < $dst 2>/dev/null || echo 0" | tr -d ' ')
    [ "$got" = "$want" ] && return 0
    log "upload attempt $try landed $got of $want bytes, retrying"
    sleep 2
  done
  die "kubectl cp to $ws:$dst never landed ($got/$want bytes)"
}

repo_tarball() { # -> path of a tarball with everything git tracks (plus new untracked code)
  local out="$1"
  ( cd "$REPO_ROOT" && git ls-files -co --exclude-standard -z \
      | grep -zv '^\(data\|runs\|artifacts\|\.venv\|viz_output\|viz_judged\|viz_test_predictions\)/' \
      | tar -C "$REPO_ROOT" --null -cf "$out" -T - )
  echo "$out"
}
