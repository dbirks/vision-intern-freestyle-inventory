# shellcheck shell=bash
# Shared helpers for the k8s/*.sh scripts. Source this, do not execute it.
#
# Everything here is scoped to the home cluster (kubectl context admin@home) and to the
# `default` namespace; every object we create carries `app=vision-intern` and
# `vision-intern/owner=<VI_OWNER_PREFIX>...` so ownership is verified before reuse/deletion.

set -euo pipefail

VI_CONTEXT="${VI_CONTEXT:-admin@home}"
VI_NS="${VI_NS:-default}"
VI_LABEL_APP="vision-intern"
VI_OWNER_PREFIX="${VI_OWNER_PREFIX:-vision-intern}"
VI_OWNER="${VI_OWNER:-${VI_OWNER_PREFIX}/${USER:-local}}"
# Pinned tag + immutable digest verified on both talos-8me-tcy and talos-210-73x.
VI_IMAGE="${VI_IMAGE:-pytorch/pytorch:2.11.0-cuda13.0-cudnn9-runtime@sha256:bfbb4a2b4fdba0fefdb428ea737e626d61bb3daf74a16e1ff935bdb03aa7c3f0}"
VI_WS_IMAGE="${VI_WS_IMAGE:-busybox:1.37@sha256:9db7b59979c38555a39def84a31fb98b5296952f9e3afd4f6f11f05b07adfab0}"
VI_PVC_SIZE="${VI_PVC_SIZE:-20Gi}"
VI_OUT_PVC_SIZE="${VI_OUT_PVC_SIZE:-$VI_PVC_SIZE}"
VI_STORAGE_CLASS="${VI_STORAGE_CLASS:-local-path}"
VI_HF_SECRET="${VI_HF_SECRET:-hf-token}"

die() { echo "error: $*" >&2; exit 1; }
log() { echo "[vision-intern] $*" >&2; }

case "$VI_OWNER" in
  "$VI_OWNER_PREFIX"*) : ;;
  *) die "VI_OWNER '$VI_OWNER' must start with VI_OWNER_PREFIX '$VI_OWNER_PREFIX'" ;;
esac
[ -n "$VI_OWNER_PREFIX" ] || die "VI_OWNER_PREFIX must not be empty"

export NAMESPACE="$VI_NS" APP_LABEL="$VI_LABEL_APP" OWNER_PREFIX="$VI_OWNER_PREFIX" OWNER="$VI_OWNER"

K8S_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${VI_REPO_ROOT:-$(cd "$K8S_DIR/.." && pwd)}"

kctl() { kubectl --context "$VI_CONTEXT" -n "$VI_NS" "$@"; }

# ---- validation helpers ----------------------------------------------------------------
validate_dns_label() {
  local val="$1" what="${2:-name}"
  if [ "${#val}" -gt 57 ] || ! [[ "$val" =~ ^[a-z0-9]([a-z0-9-]*[a-z0-9])?$ ]]; then
    die "$what '$val' must be a lowercase DNS-1123 label <= 57 chars (a-z, 0-9, '-'; start/end alphanumeric)"
  fi
}

validate_pos_int() {
  local val="$1" flag="$2"
  [[ "$val" =~ ^[1-9][0-9]*$ ]] || die "$flag must be a positive integer (got '$val')"
}

validate_nonneg_int() {
  local val="$1" flag="$2"
  [[ "$val" =~ ^(0|[1-9][0-9]*)$ ]] || die "$flag must be a non-negative integer (got '$val')"
}

validate_cpu_qty() {
  local val="$1" flag="$2"
  if ! [[ "$val" =~ ^([1-9][0-9]*m|[0-9]+(\.[0-9]+)?)$ ]] || [ "$val" = "0" ] || [ "$val" = "0.0" ]; then
    die "$flag must be a valid positive Kubernetes CPU quantity (got '$val')"
  fi
}

validate_mem_qty() {
  local val="$1" flag="$2"
  [[ "$val" =~ ^[1-9][0-9]*(Ki|Mi|Gi|Ti|K|M|G|T)?$ ]] || die "$flag must be a valid Kubernetes memory quantity (got '$val')"
}

validate_image_ref() {
  local val="$1"
  [ -n "$val" ] || die "--image must not be empty"
  if [[ "$val" =~ [[:space:]\'\"\\] ]]; then
    die "--image contains invalid whitespace or quote characters"
  fi
}

validate_pip_specs() {
  local raw="$1" flag="$2" spec
  local bad_re='[;&|$`()<>'\''"\\*{}?!]'
  local ok_re='^[]A-Za-z0-9._+/=@:,!~[-]+$'
  local -a specs=()
  read -r -a specs <<< "$raw"
  for spec in "${specs[@]}"; do
    if [[ "$spec" == -* ]]; then
      die "$flag spec '$spec' must not start with '-'"
    fi
    if [[ "$spec" =~ $bad_re ]]; then
      die "$flag spec '$spec' contains disallowed shell metacharacters"
    fi
    if ! [[ "$spec" =~ $ok_re ]]; then
      die "$flag spec '$spec' is not a valid requirement specifier"
    fi
  done
}

sha256_file() {
  local path="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$path" | awk '{print $1}'
  else
    python3 -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$path"
  fi
}

# ---- gpu -> node / card type / pvcs / workspace pods -----------------------------------
gpu_node() {
  case "$1" in
    5090) echo "${VI_NODE_5090:-talos-8me-tcy}" ;;
    6000) echo "${VI_NODE_6000:-talos-210-73x}" ;;
    *) die "gpu must be 5090 or 6000 (got '$1')" ;;
  esac
}

gpu_type() {
  case "$1" in
    5090) echo "RTX 5090" ;;
    6000) echo "PRO 6000" ;;
    *) die "gpu must be 5090 or 6000 (got '$1')" ;;
  esac
}

pvc_name() { echo "vision-intern-data-$1"; }       # read-only data PVC per GPU node
out_pvc_name() { echo "vision-intern-out-$1"; }    # separate writable outputs/jobs PVC
ws_name() { echo "vision-intern-ws-$1"; }          # helper pod for data PVC
out_ws_name() { echo "vision-intern-out-ws-$1"; }  # helper pod for outputs PVC

# ---- templating ------------------------------------------------------------------------
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
free_vram_mib() {
  local node="$1" pod out
  pod=$(kctl get pods -A -o wide --field-selector "spec.nodeName=$node" 2>/dev/null \
        | awk '/gpu-vram-exporter/{print $2; exit}' || true)
  [ -n "$pod" ] || { echo ""; return 0; }
  out=$(kctl exec "$pod" -- nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
        | head -1 | tr -d ' ' || true)
  echo "$out"
}

hami_free_mib() {
  local node="$1" want="$2" reg
  reg=$(kctl get node "$node" -o jsonpath='{.metadata.annotations.hami\.io/node-nvidia-register}' 2>/dev/null || true)
  [ -n "$reg" ] || { echo ""; return 0; }
  printf '%s' "$reg" | python3 -c '
import json,sys,subprocess
want=sys.argv[1].lower()
try: reg=json.loads(sys.stdin.read() or "[]")
except Exception: sys.exit(0)
used={}
out=subprocess.run(["kubectl","--context",sys.argv[2],"-n",sys.argv[3],"get","pods","-A","-o","json"],
                   capture_output=True,text=True).stdout
try: pods=json.loads(out)
except Exception: pods={"items":[]}
for p in pods.get("items",[]):
    if p.get("status",{}).get("phase") not in ("Running","Pending"): continue
    ann=p.get("metadata",{}).get("annotations",{})
    alloc=ann.get("hami.io/vgpu-devices-allocated","")
    for part in alloc.split(";"):
        f=part.split(",")
        if len(f)>=3 and f[2].isdigit(): used[f[0]]=used.get(f[0],0)+int(f[2])
for d in reg:
    if want in (d.get("type","").lower()):
        print(max(0, int(d.get("devmem",0)) - used.get(d.get("id",""),0)))
' "$want" "$VI_CONTEXT" "$VI_NS" 2>/dev/null || echo ""
}

# ---- ownership verification & storage --------------------------------------------------
verify_owned_job_for_replace() { # name -> prints UID or dies if not owned
  local name="$1" job_json
  job_json=$(kctl get job "$name" -o json 2>/dev/null || true)
  [ -n "$job_json" ] || die "job/$name disappeared before inspection"
  printf '%s' "$job_json" | python3 -c '
import json, sys
app_want, owner_prefix, name = sys.argv[1], sys.argv[2], sys.argv[3]
obj = json.loads(sys.stdin.read())
meta = obj.get("metadata", {})
labels = meta.get("labels", {}) or {}
anns = meta.get("annotations", {}) or {}
app = labels.get("app", "")
owner = anns.get("vision-intern/owner", "")
uid = meta.get("uid", "")
if app != app_want:
    sys.stderr.write(f"error: refusing to delete job/{name}: label app={app!r} != {app_want!r}\n")
    sys.exit(1)
if not owner_prefix or not owner.startswith(owner_prefix):
    sys.stderr.write(f"error: refusing to delete job/{name}: annotation vision-intern/owner={owner!r} does not start with {owner_prefix!r}\n")
    sys.exit(1)
if not uid:
    sys.stderr.write(f"error: refusing to delete job/{name}: missing metadata.uid\n")
    sys.exit(1)
print(uid)
' "$VI_LABEL_APP" "$VI_OWNER_PREFIX" "$name"
}

verify_owned_pvc() { # gpu pvc_name
  local gpu="$1" name="$2" node pvc_json
  node=$(gpu_node "$gpu")
  pvc_json=$(kctl get pvc "$name" -o json 2>/dev/null || true)
  [ -n "$pvc_json" ] || die "pvc/$name not found"
  printf '%s' "$pvc_json" | python3 -c '
import json, sys
app_want, owner_prefix, expect_node, name = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
obj = json.loads(sys.stdin.read())
meta = obj.get("metadata", {})
labels = meta.get("labels", {}) or {}
anns = meta.get("annotations", {}) or {}
if labels.get("app") != app_want or labels.get("app.kubernetes.io/component") != "workdir":
    sys.stderr.write(f"error: pvc/{name} exists but lacks vision-intern ownership labels (labels={labels!r})\n")
    sys.exit(1)
owner = anns.get("vision-intern/owner")
if owner is not None and not owner.startswith(owner_prefix):
    sys.stderr.write(f"error: pvc/{name} annotation vision-intern/owner={owner!r} != {owner_prefix!r}*\n")
    sys.exit(1)
selected = anns.get("volume.kubernetes.io/selected-node", "")
if selected and selected != expect_node:
    sys.stderr.write(f"error: pvc/{name} is bound to node {selected!r}, expected {expect_node!r}\n")
    sys.exit(1)
' "$VI_LABEL_APP" "$VI_OWNER_PREFIX" "$node" "$name"
}

ensure_pvc() { # gpu pvc_name [size]
  local gpu="$1" name="$2" size="${3:-$VI_PVC_SIZE}" out
  if kctl get pvc "$name" >/dev/null 2>&1; then
    verify_owned_pvc "$gpu" "$name"
    return 0
  fi
  log "creating PVC $name ($size, $VI_STORAGE_CLASS) -- binds to $(gpu_node "$gpu") on first use"
  out=$(render k8s/pvc-template.yaml PVC_NAME="$name" STORAGE_CLASS="$VI_STORAGE_CLASS" PVC_SIZE="$size" \
    | kctl create -f - 2>&1) || {
    if kctl get pvc "$name" >/dev/null 2>&1; then
      verify_owned_pvc "$gpu" "$name"
      return 0
    fi
    die "failed to create pvc/$name: $out"
  }
}

verify_owned_ws() { # gpu ws_name pvc_name -> prints "<phase> <uid>"
  local gpu="$1" ws="$2" pvc="$3" node pod_json
  node=$(gpu_node "$gpu")
  pod_json=$(kctl get pod "$ws" -o json 2>/dev/null || true)
  [ -n "$pod_json" ] || return 1
  printf '%s' "$pod_json" | python3 -c '
import json, sys
app_want, owner_prefix, gpu, node, pvc, ws = sys.argv[1:7]
obj = json.loads(sys.stdin.read())
meta = obj.get("metadata", {})
labels = meta.get("labels", {}) or {}
anns = meta.get("annotations", {}) or {}
spec = obj.get("spec", {}) or {}
status = obj.get("status", {}) or {}
if (labels.get("app") != app_want or
    labels.get("app.kubernetes.io/component") != "workspace" or
    labels.get("vision-intern/gpu") != gpu):
    sys.stderr.write(f"error: pod/{ws} exists but is not an owned workspace pod for gpu={gpu} (labels={labels!r})\n")
    sys.exit(2)
owner = anns.get("vision-intern/owner")
if owner is not None and not owner.startswith(owner_prefix):
    sys.stderr.write(f"error: pod/{ws} annotation vision-intern/owner={owner!r} != {owner_prefix!r}*\n")
    sys.exit(2)
sel = (spec.get("nodeSelector") or {}).get("kubernetes.io/hostname", "")
if sel != node:
    sys.stderr.write(f"error: pod/{ws} nodeSelector={sel!r}, expected {node!r}\n")
    sys.exit(2)
node_name = spec.get("nodeName", "")
if node_name and node_name != node:
    sys.stderr.write(f"error: pod/{ws} scheduled on {node_name!r}, expected {node!r}\n")
    sys.exit(2)
claims = [v.get("persistentVolumeClaim", {}).get("claimName") for v in spec.get("volumes", []) if "persistentVolumeClaim" in v]
if pvc not in claims:
    sys.stderr.write(f"error: pod/{ws} mounts PVCs {claims!r}, expected {pvc!r}\n")
    sys.exit(2)
uid = meta.get("uid", "")
phase = status.get("phase", "")
if not uid:
    sys.stderr.write(f"error: pod/{ws} missing uid\n")
    sys.exit(2)
print(f"{phase} {uid}")
' "$VI_LABEL_APP" "$VI_OWNER_PREFIX" "$gpu" "$node" "$pvc" "$ws"
}

ensure_workspace_for() { # gpu ws_name pvc_name [pvc_size]
  local gpu="$1" ws="$2" pvc="$3" size="${4:-$VI_PVC_SIZE}" node info phase uid uid2 rc out
  node=$(gpu_node "$gpu")
  ensure_pvc "$gpu" "$pvc" "$size"
  if kctl get pod "$ws" >/dev/null 2>&1; then
    set +e
    info=$(verify_owned_ws "$gpu" "$ws" "$pvc")
    rc=$?
    set -e
    [ "$rc" = 0 ] || exit 1
    phase="${info%% *}"
    uid="${info#* }"
    if [ "$phase" = "Running" ]; then return 0; fi
    log "owned workspace pod $ws (uid=$uid) is $phase, recreating"
    uid2=$(verify_owned_ws "$gpu" "$ws" "$pvc" | awk '{print $2}')
    [ "$uid2" = "$uid" ] || die "pod/$ws UID changed before delete ($uid -> $uid2)"
    kctl delete pod "$ws" --wait=true >/dev/null
  fi
  log "starting workspace pod $ws on $node (mounts $pvc at /work)"
  out=$(render k8s/workspace-template.yaml WS_NAME="$ws" NODE="$node" PVC_NAME="$pvc" WS_IMAGE="$VI_WS_IMAGE" GPU="$gpu" \
    | kctl create -f - 2>&1) || {
    if kctl get pod "$ws" >/dev/null 2>&1; then
      verify_owned_ws "$gpu" "$ws" "$pvc" >/dev/null || exit 1
    else
      die "failed to create workspace pod $ws: $out"
    fi
  }
  kctl wait --for=jsonpath='{.status.phase}'=Running "pod/$ws" --timeout=180s >/dev/null
}

ensure_workspace() { # gpu -> ensures data PVC + data workspace pod
  ensure_workspace_for "$1" "$(ws_name "$1")" "$(pvc_name "$1")" "$VI_PVC_SIZE"
}

ensure_out_workspace() { # gpu -> ensures outputs PVC + outputs workspace pod
  ensure_workspace_for "$1" "$(out_ws_name "$1")" "$(out_pvc_name "$1")" "$VI_OUT_PVC_SIZE"
}

# ---- file transfer ---------------------------------------------------------------------
# NOTE: raw `kubectl exec -i <pod> -- cat > f` silently truncates at ~192 KiB against this
# cluster with kubectl v1.37.1. `kubectl cp` transfers MBs correctly; we always verify both
# size and sha256 on the pod before atomically renaming into place.
ws_sh() { kctl exec "$(ws_name "$1")" -- sh -c "$2"; }
out_ws_sh() { kctl exec "$(out_ws_name "$1")" -- sh -c "$2"; }

push_file_to_pod() { # pod local_path remote_path
  local pod="$1" src="$2" dst="$3" try got_bytes got_sha want_bytes want_sha tmp_dst dir_dst cp_err
  local retry_sleep="${VI_RETRY_SLEEP_SECS:-2}"
  want_bytes=$(wc -c < "$src" | tr -d ' ')
  want_sha=$(sha256_file "$src")
  dir_dst=$(dirname "$dst")
  tmp_dst="${dst}.tmp.$$"
  log "uploading $src ($(du -h "$src" | cut -f1), sha256=${want_sha:0:12}) -> $pod:$dst"
  kctl exec "$pod" -- mkdir -p "$dir_dst" >/dev/null
  for try in 1 2 3; do
    if cp_err=$(kctl cp "$src" "$pod:$tmp_dst" 2>&1 >/dev/null); then
      got_bytes=$(kctl exec "$pod" -- sh -c "wc -c < '$tmp_dst' 2>/dev/null || echo 0" | tr -d ' ')
      got_sha=$(kctl exec "$pod" -- sh -c "sha256sum '$tmp_dst' 2>/dev/null | awk '{print \$1}'" | tr -d ' \r\n')
      if [ "$got_bytes" = "$want_bytes" ] && [ "$got_sha" = "$want_sha" ]; then
        kctl exec "$pod" -- mv -f "$tmp_dst" "$dst" >/dev/null
        return 0
      fi
      log "upload attempt $try mismatch on $pod:$dst (bytes=${got_bytes:-0}/$want_bytes sha=${got_sha:-?}/$want_sha), retrying"
    else
      log "upload attempt $try kubectl cp failed for $pod:$dst ($cp_err), retrying"
    fi
    kctl exec "$pod" -- rm -f "$tmp_dst" >/dev/null 2>&1 || true
    if [ "$retry_sleep" != "0" ]; then sleep "$retry_sleep"; fi
  done
  die "kubectl cp to $pod:$dst failed verification after 3 attempts (want $want_bytes bytes, sha256=$want_sha)"
}

push_file() { push_file_to_pod "$(ws_name "$1")" "$2" "$3"; }
push_out_file() { push_file_to_pod "$(out_ws_name "$1")" "$2" "$3"; }

git_provenance_json() {
  (
    cd "$REPO_ROOT"
    python3 -c '
import hashlib, json, subprocess
from pathlib import Path

sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
pathspecs = ["--", ".", ":(exclude)data", ":(exclude)runs"]
staged = subprocess.check_output(["git", "diff", "--cached", "--binary", "HEAD", *pathspecs])
unstaged = subprocess.check_output(["git", "diff", "--binary", *pathspecs])
names = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "-z", *pathspecs]).split(b"\0")
digest = hashlib.sha256()
digest.update(len(staged).to_bytes(8, "big"))
digest.update(staged)
digest.update(unstaged)
dirty = bool(staged or unstaged)
for name in sorted(filter(None, names)):
    p = Path(name.decode("utf-8", "surrogateescape"))
    dirty = True
    digest.update(len(name).to_bytes(8, "big"))
    digest.update(name)
    if p.is_symlink():
        contents = str(p.readlink()).encode()
        digest.update(b"symlink")
    else:
        contents = p.read_bytes()
        digest.update(b"file")
    digest.update(len(contents).to_bytes(8, "big"))
    digest.update(contents)
print(json.dumps({"git_sha": sha, "git_dirty": dirty, "git_diff_hash": digest.hexdigest() if dirty else None}))
'
  )
}

repo_tarball() { # -> path of a tarball with everything git tracks (plus new untracked code)
  local out="$1"
  ( cd "$REPO_ROOT" && git ls-files -co --exclude-standard -z \
      | grep -zv '^\(data\|runs\|artifacts\|\.venv\|viz_output\|viz_judged\|viz_test_predictions\)/' \
      | tar -C "$REPO_ROOT" --null -cf "$out" -T - )
  echo "$out"
}
