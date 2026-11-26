#!/usr/bin/env bash
# Render k8s/job-template.yaml and submit it as a Job on the home cluster.
#
#   k8s/submit.sh <name> <gpu:5090|6000> [flags] -- <command...>
#   k8s/submit.sh smoke-5090 5090 --mode share --mem 192
#   k8s/submit.sh train-x 6000 --mem 8192 -- python -m freestyle.train --cfg c.yaml
#
# Flags (all optional):
#   --mem MB                VRAM for the job (default 4096; 0 allowed only with --mode none)
#   --mode M                hami | share | none | auto (default auto: hami if HAMi has room,
#                           else fail closed; share must be opted into explicitly)
#   --image IMG             base image (default pinned $VI_IMAGE)
#   --cpu N / --cpu-lim N   cpu request / limit                  (defaults 2 / 8)
#   --mem-host R / --mem-host-lim L   host RAM request / limit   (defaults 4Gi / 16Gi)
#   --shm SIZE              /dev/shm size                        (default 2Gi)
#   --timeout SECS          activeDeadlineSeconds                (default 3600)
#   --ttl SECS              ttlSecondsAfterFinished              (default 86400)
#   --pip "specs"           pip requirements (written to requirements.txt, never eval'd)
#   --pip-constraints "s"   pip constraints passed via -c constraints.txt
#   --hf-token              mount secret hf-token (key HF_TOKEN) as env HF_TOKEN
#   --no-code               use baseline /outputs/repo/repo.tar staged by stage-data.sh
#   --replace               delete an existing OWNED job with the same name first
#   --dry-run               print rendered manifest without mutating cluster
#   --dry-run=server        validate with kubectl create --dry-run=server and print manifest
#   --wait                  wait for Complete OR Failed (collects logs + /out on failure)
#   --wait-for S            wait timeout in seconds              (default 1200)
#   --collect-dir DIR       destination dir when --wait collects on failure
#   --expect-git-sha SHA    verify repo HEAD matches launcher expectation
#   --expect-git-dirty B    verify repo dirty state (true|false) matches expectation
#   --expect-git-diff-hash H verify dirty diff hash matches expectation
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

NAME="${1:-}"; GPU="${2:-}"
if [ -z "$NAME" ] || [ -z "$GPU" ]; then sed -n '2,32p' "$0" >&2; exit 2; fi
case "$GPU" in 5090|6000) : ;; *) die "gpu must be 5090 or 6000 (got '$GPU')" ;; esac
validate_dns_label "$NAME" "job name"
shift 2

MEM=4096; MEM_SET=0; MODE=auto; IMAGE="$VI_IMAGE"
VI_CTX_SLACK="${VI_CTX_SLACK:-512}"; CPU_REQ=2; CPU_LIM=8
MEM_REQ=4Gi; MEM_LIM=16Gi; SHM=2Gi; DEADLINE=3600; TTL=86400
PIP=""; PIP_CONSTRAINTS=""; HF_TOKEN=0
UPLOAD_CODE=1; DRY=0; DRY_SERVER=0; WAIT=0; WAIT_FOR=1200; REPLACE=0
COLLECT_DIR=""
EXPECT_GIT_SHA=""; EXPECT_GIT_DIRTY=""; EXPECT_GIT_DIFF_HASH=""
CMD=""

need_arg() { [ $# -ge 2 ] || die "$1 requires an argument"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --mem) need_arg "$@"; MEM="$2"; MEM_SET=1; shift 2 ;;
    --mode) need_arg "$@"; MODE="$2"; shift 2 ;;
    --image) need_arg "$@"; IMAGE="$2"; shift 2 ;;
    --cpu) need_arg "$@"; CPU_REQ="$2"; shift 2 ;;
    --cpu-lim) need_arg "$@"; CPU_LIM="$2"; shift 2 ;;
    --mem-host) need_arg "$@"; MEM_REQ="$2"; shift 2 ;;
    --mem-host-lim) need_arg "$@"; MEM_LIM="$2"; shift 2 ;;
    --shm) need_arg "$@"; SHM="$2"; shift 2 ;;
    --timeout) need_arg "$@"; DEADLINE="$2"; shift 2 ;;
    --ttl) need_arg "$@"; TTL="$2"; shift 2 ;;
    --pip) need_arg "$@"; PIP="$2"; shift 2 ;;
    --pip-constraints) need_arg "$@"; PIP_CONSTRAINTS="$2"; shift 2 ;;
    --hf-token) HF_TOKEN=1; shift ;;
    --no-code) UPLOAD_CODE=0; shift ;;
    --dry-run) DRY=1; shift ;;
    --dry-run=server) DRY=1; DRY_SERVER=1; shift ;;
    --wait) WAIT=1; shift ;;
    --wait-for) need_arg "$@"; WAIT_FOR="$2"; shift 2 ;;
    --collect-dir) need_arg "$@"; COLLECT_DIR="$2"; shift 2 ;;
    --replace) REPLACE=1; shift ;;
    --expect-git-sha) need_arg "$@"; EXPECT_GIT_SHA="$2"; shift 2 ;;
    --expect-git-dirty) need_arg "$@"; EXPECT_GIT_DIRTY="$2"; shift 2 ;;
    --expect-git-diff-hash) need_arg "$@"; EXPECT_GIT_DIFF_HASH="$2"; shift 2 ;;
    --) shift; [ $# -gt 0 ] || die "command after -- must not be empty"; CMD=$(printf '%q ' "$@"); break ;;
    *) die "unknown flag $1 (command args must follow --)" ;;
  esac
done

case "$MODE" in hami|share|none|auto) : ;; *) die "bad --mode $MODE (expected hami|share|none|auto)" ;; esac
if [ -z "$CMD" ]; then
  CMD="python k8s/smoke/smoke.py"
  if [ "$MEM_SET" = 0 ]; then MEM=192; fi
  log "no command given, running the smoke test: $CMD (--mem $MEM)"
fi

if [ "$MODE" = none ]; then
  validate_nonneg_int "$MEM" "--mem"
else
  validate_pos_int "$MEM" "--mem"
fi
validate_nonneg_int "$VI_CTX_SLACK" "VI_CTX_SLACK"
validate_pos_int "$DEADLINE" "--timeout"
validate_nonneg_int "$TTL" "--ttl"
validate_pos_int "$WAIT_FOR" "--wait-for"
validate_cpu_qty "$CPU_REQ" "--cpu"
validate_cpu_qty "$CPU_LIM" "--cpu-lim"
validate_mem_qty "$MEM_REQ" "--mem-host"
validate_mem_qty "$MEM_LIM" "--mem-host-lim"
validate_mem_qty "$SHM" "--shm"
validate_image_ref "$IMAGE"
[ -z "$PIP" ] || validate_pip_specs "$PIP" "--pip"
[ -z "$PIP_CONSTRAINTS" ] || validate_pip_specs "$PIP_CONSTRAINTS" "--pip-constraints"

NODE=$(gpu_node "$GPU"); TYPE=$(gpu_type "$GPU")
DATA_PVC=$(pvc_name "$GPU"); OUT_PVC=$(out_pvc_name "$GPU")

# ---- pick and verify GPU accounting mode -----------------------------------------------
HAMI_FREE=$(hami_free_mib "$NODE" "$TYPE")
PHYS_FREE=$(free_vram_mib "$NODE")
log "gpu $GPU ($TYPE on $NODE): HAMi reservation free=${HAMI_FREE:-?} MiB, physical free=${PHYS_FREE:-?} MiB"

if [ "$MODE" = auto ]; then
  if [[ "$HAMI_FREE" =~ ^[0-9]+$ ]] && [ "$HAMI_FREE" -ge "$MEM" ]; then
    MODE=hami
  elif [ "$DRY" = 1 ]; then
    MODE=hami
    log "dry-run: HAMi free=${HAMI_FREE:-?} MiB < ${MEM} MiB; previewing the hami manifest"
  else
    die "auto mode requires HAMi free >= ${MEM} MiB on $GPU (HAMi free=${HAMI_FREE:-?} MiB, physical free=${PHYS_FREE:-?} MiB). Free HAMi room or opt into self-capped sharing explicitly with --mode share."
  fi
fi
log "mode=$MODE"

if [ "$MODE" = none ]; then
  REQUESTS="{cpu: $CPU_REQ, memory: $MEM_REQ}"
  LIMITS="{cpu: $CPU_LIM, memory: $MEM_LIM}"
  POD_GPU_ANNOTATION='# no gpu-type annotation'
  SCHEDULER_NAME=""
  SHARE_ENV='- { name: NVIDIA_VISIBLE_DEVICES, value: "" }'
  PYTHONPATH_VALUE="/src"
  PYTORCH_ALLOC_CONF="expandable_segments:True"
  log "none mode: card hidden (NVIDIA_VISIBLE_DEVICES=empty)"
elif [ "$MODE" = hami ]; then
  if [ "$DRY" = 0 ] && [[ "$HAMI_FREE" =~ ^[0-9]+$ ]] && [ "$HAMI_FREE" -lt "$MEM" ]; then
    die "hami mode refuses: HAMi free is ${HAMI_FREE} MiB on $GPU, need ${MEM} MiB"
  fi
  REQUESTS="{cpu: $CPU_REQ, memory: $MEM_REQ, nvidia.com/gpu: 1, nvidia.com/gpumem: $MEM}"
  LIMITS="{cpu: $CPU_LIM, memory: $MEM_LIM, nvidia.com/gpu: 1, nvidia.com/gpumem: $MEM}"
  POD_GPU_ANNOTATION="nvidia.com/use-gputype: \"$TYPE\""
  SCHEDULER_NAME=hami-scheduler
  SHARE_ENV='# no extra gpu env'
  PYTHONPATH_VALUE="/src"
  PYTORCH_ALLOC_CONF="expandable_segments:True"
else
  # Share mode: fail closed unless physical free VRAM measurement is present AND >= MEM + 512 MiB.
  if ! [[ "$PHYS_FREE" =~ ^[0-9]+$ ]]; then
    die "share mode refuses: physical free VRAM measurement unavailable from gpu-vram-exporter on $NODE"
  fi
  need_phys=$((MEM + VI_CTX_SLACK))
  if [ "$PHYS_FREE" -lt "$need_phys" ]; then
    die "share mode refuses: ${PHYS_FREE} MiB physically free on $GPU, need ${MEM} MiB + ${VI_CTX_SLACK} MiB = ${need_phys} MiB"
  fi
  REQUESTS="{cpu: $CPU_REQ, memory: $MEM_REQ}"
  LIMITS="{cpu: $CPU_LIM, memory: $MEM_LIM}"
  POD_GPU_ANNOTATION="nvidia.com/use-gputype: \"$TYPE\""
  SCHEDULER_NAME=""
  SHARE_ENV='- { name: NVIDIA_VISIBLE_DEVICES, value: "all" }
            - { name: CUDA_DISABLE_CONTROL, value: "true" }'
  PYTHONPATH_VALUE="/_vi_site:/src"
  PYTORCH_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
  log "share mode: no HAMi reservation; sitecustomize enforces ${MEM} MiB cap (phys free ${PHYS_FREE} >= ${need_phys} MiB)"
fi

if [ "$HF_TOKEN" = 1 ]; then
  HF_TOKEN_ENV='- { name: HF_TOKEN, valueFrom: { secretKeyRef: { name: "'"$VI_HF_SECRET"'", key: HF_TOKEN } } }'
  if [ "$DRY" = 0 ]; then
    kctl get secret "$VI_HF_SECRET" >/dev/null 2>&1 || die "missing secret $VI_HF_SECRET in namespace $VI_NS (needed for HF_TOKEN)"
  fi
else
  HF_TOKEN_ENV='# no hf-token secret'
fi

RUN_ID="${NAME}-$(date -u +%Y%m%d%H%M%S)-$(python3 -c 'import secrets; print(secrets.token_hex(4))')"

# ---- inspect existing job before dry-run / mutation ------------------------------------
EXISTING_UID=""
if kctl get job "$NAME" >/dev/null 2>&1; then
  if [ "$REPLACE" = 1 ]; then
    EXISTING_UID=$(verify_owned_job_for_replace "$NAME")
    if [ "$DRY" = 1 ]; then
      log "dry-run: would delete owned job/$NAME (uid=$EXISTING_UID)"
    fi
  else
    die "job/$NAME already exists (use --replace to overwrite owned job, or a new name)"
  fi
fi

# ---- dry-run renders and exits before any staging or cluster mutation ------------------
if [ "$DRY" = 1 ]; then
  if [ "$UPLOAD_CODE" = 1 ]; then
    REPO_TAR_PATH="/outputs/repo/${RUN_ID}.tar"
  else
    REPO_TAR_PATH="/outputs/repo/repo.tar"
  fi
  REPO_TAR_SHA256="dry-run"
  manifest=$({
    export REQUESTS LIMITS POD_GPU_ANNOTATION SCHEDULER_NAME SHARE_ENV HF_TOKEN_ENV
    export PYTHONPATH_VALUE PYTORCH_ALLOC_CONF REPO_TAR_PATH REPO_TAR_SHA256
    export JOB_NAME="$NAME" RUN_ID GPU NODE DATA_PVC_NAME="$DATA_PVC" OUT_PVC_NAME="$OUT_PVC"
    export IMAGE SHM_SIZE="$SHM" ACTIVE_DEADLINE="$DEADLINE" TTL VI_GPU_MODE="$MODE" VI_GPU_MEM_MB="$MEM"
    render k8s/job-template.yaml
  })
  if [ "$DRY_SERVER" = 1 ]; then
    printf '%s\n' "$manifest" | kctl create --dry-run=server -f - >/dev/null
  fi
  printf '%s\n' "$manifest"
  exit 0
fi

# ---- stage code, provenance, requirements, and command into writable outputs PVC -------
ensure_workspace "$GPU"
ensure_out_workspace "$GPU"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

prov_json=$(git_provenance_json)
actual_sha=$(printf '%s' "$prov_json" | python3 -c 'import json,sys; print(json.load(sys.stdin)["git_sha"])')
actual_dirty=$(printf '%s' "$prov_json" | python3 -c 'import json,sys; print(str(json.load(sys.stdin)["git_dirty"]).lower())')
actual_diff=$(printf '%s' "$prov_json" | python3 -c 'import json,sys; print(json.load(sys.stdin)["git_diff_hash"] or "")')

if [ "$UPLOAD_CODE" = 1 ]; then
  if [ -n "$EXPECT_GIT_SHA" ] && [ "$actual_sha" != "$EXPECT_GIT_SHA" ]; then
    die "git HEAD changed before tar snapshot ($EXPECT_GIT_SHA -> $actual_sha)"
  fi
  if [ -n "$EXPECT_GIT_DIRTY" ] && [ "$actual_dirty" != "$EXPECT_GIT_DIRTY" ]; then
    die "git dirty state changed before tar snapshot ($EXPECT_GIT_DIRTY -> $actual_dirty)"
  fi
  if [ -n "$EXPECT_GIT_DIFF_HASH" ] && [ "$actual_diff" != "$EXPECT_GIT_DIFF_HASH" ]; then
    die "working tree changed before tar snapshot ($EXPECT_GIT_DIFF_HASH -> $actual_diff)"
  fi
  repo_tarball "$tmp/$RUN_ID.tar" >/dev/null
  REPO_TAR_SHA256=$(sha256_file "$tmp/$RUN_ID.tar")
  push_out_file "$GPU" "$tmp/$RUN_ID.tar" "/work/repo/$RUN_ID.tar"
  REPO_TAR_PATH="/outputs/repo/$RUN_ID.tar"
else
  REPO_TAR_PATH="/outputs/repo/repo.tar"
  REPO_TAR_SHA256=$(out_ws_sh "$GPU" "[ -s /work/repo/repo.tar ] && sha256sum /work/repo/repo.tar | awk '{print \$1}'" 2>/dev/null || true)
  [ -n "$REPO_TAR_SHA256" ] || die "--no-code selected, but /work/repo/repo.tar is missing or empty on $(out_pvc_name "$GPU"); run k8s/stage-data.sh"
  meta_raw=$(out_ws_sh "$GPU" "cat /work/repo/repo.tar.meta.json 2>/dev/null || true" || true)
  if [ -n "$meta_raw" ]; then
    prov_json=$(printf '%s' "$meta_raw" | python3 -c '
import json, sys
m = json.loads(sys.stdin.read())
print(json.dumps({"git_sha": m.get("git_sha"), "git_dirty": m.get("git_dirty"), "git_diff_hash": m.get("git_diff_hash")}))
')
  fi
fi

python3 -c '
import json, sys, time
prov = json.loads(sys.argv[1])
prov.update({
    "run_id": sys.argv[2],
    "job_name": sys.argv[3],
    "repo_tar_path": sys.argv[4],
    "repo_tar_sha256": sys.argv[5],
    "image": sys.argv[6],
    "gpu": sys.argv[7],
    "mode": sys.argv[8],
    "vram_mb": int(sys.argv[9]),
    "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
})
print(json.dumps(prov, indent=2, sort_keys=True))
' "$prov_json" "$RUN_ID" "$NAME" "$REPO_TAR_PATH" "$REPO_TAR_SHA256" "$IMAGE" "$GPU" "$MODE" "$MEM" > "$tmp/provenance.json"
push_out_file "$GPU" "$tmp/provenance.json" "/work/jobs/$RUN_ID.provenance.json"

if [ -n "$PIP" ]; then
  read -r -a _pip_arr <<< "$PIP"
  printf '%s\n' "${_pip_arr[@]}" > "$tmp/requirements.txt"
  push_out_file "$GPU" "$tmp/requirements.txt" "/work/jobs/$RUN_ID.requirements.txt"
fi
if [ -n "$PIP_CONSTRAINTS" ]; then
  read -r -a _cons_arr <<< "$PIP_CONSTRAINTS"
  printf '%s\n' "${_cons_arr[@]}" > "$tmp/constraints.txt"
  push_out_file "$GPU" "$tmp/constraints.txt" "/work/jobs/$RUN_ID.constraints.txt"
fi

{
  echo 'set -euo pipefail'
  printf '%s\n' "$CMD"
} > "$tmp/$RUN_ID.sh"
push_out_file "$GPU" "$tmp/$RUN_ID.sh" "/work/jobs/$RUN_ID.sh"
out_ws_sh "$GPU" "mkdir -p '/work/artifacts/$RUN_ID'"

if [ "$(ws_sh "$GPU" 'ls -A /work/data 2>/dev/null | wc -l' | tr -d ' ')" = "0" ]; then
  log "WARNING: no data/ staged on $GPU -- run k8s/stage-data.sh (a job that reads /src/data will fail)"
fi

manifest=$({
  export REQUESTS LIMITS POD_GPU_ANNOTATION SCHEDULER_NAME SHARE_ENV HF_TOKEN_ENV
  export PYTHONPATH_VALUE PYTORCH_ALLOC_CONF REPO_TAR_PATH REPO_TAR_SHA256
  export JOB_NAME="$NAME" RUN_ID GPU NODE DATA_PVC_NAME="$DATA_PVC" OUT_PVC_NAME="$OUT_PVC"
  export IMAGE SHM_SIZE="$SHM" ACTIVE_DEADLINE="$DEADLINE" TTL VI_GPU_MODE="$MODE" VI_GPU_MEM_MB="$MEM"
  render k8s/job-template.yaml
})

# ---- replace owned job (re-verified) and create exclusively ----------------------------
if [ -n "$EXISTING_UID" ]; then
  uid_now=$(verify_owned_job_for_replace "$NAME")
  [ "$uid_now" = "$EXISTING_UID" ] || die "job/$NAME UID changed before delete ($EXISTING_UID -> $uid_now)"
  log "deleting owned job/$NAME (uid=$EXISTING_UID)"
  kctl delete job "$NAME" --wait=true >/dev/null
fi

printf '%s\n' "$manifest" | kctl create -f - >/dev/null
log "submitted job/$NAME (run_id=$RUN_ID gpu=$GPU mode=$MODE mem=${MEM}MiB host=${MEM_REQ}/${MEM_LIM} deadline=${DEADLINE}s)"
log "watch:   kubectl --context $VI_CONTEXT -n $VI_NS get pod -l vision-intern/job=$NAME -w"
log "logs:    kubectl --context $VI_CONTEXT -n $VI_NS logs -f job/$NAME"
log "collect: k8s/collect.sh $NAME"

collect_failure_artifacts() {
  local name="$1" dest="$2" logs_out
  logs_out=$(kctl logs "job/$name" --all-pods=true 2>/dev/null || kctl logs "job/$name" 2>/dev/null || true)
  if [ -n "$logs_out" ]; then
    printf '%s\n' "$logs_out" | tail -n 200 >&2 || true
  fi
  if [ "${VI_WAIT_COLLECT:-1}" = "0" ]; then
    return 0
  fi
  if [ -z "$dest" ]; then
    dest="runs/$name"
    if [ -e "$dest" ] && [ -n "$(ls -A "$dest" 2>/dev/null || true)" ]; then
      dest="runs/${name}-${RUN_ID}"
    fi
  fi
  "$K8S_DIR/collect.sh" "$name" "$dest" >/dev/null 2>&1 || true
  mkdir -p "$dest"
  if [ -n "$logs_out" ]; then
    printf '%s\n' "$logs_out" > "$dest/job.log"
  fi
  log "collected failure logs/artifacts into $dest"
}

if [ "$WAIT" = 1 ]; then
  log "waiting up to ${WAIT_FOR}s for Complete or Failed"
  poll="${VI_WAIT_POLL_SECS:-5}"
  start=$(date +%s)
  api_errs=0
  while :; do
    if job_json=$(kctl get job "$NAME" -o json 2>&1); then
      api_errs=0
      status_info=$(printf '%s' "$job_json" | python3 -c '
import json, sys
obj = json.loads(sys.stdin.read())
conds = {c.get("type"): c for c in (obj.get("status") or {}).get("conditions", [])}
comp = conds.get("Complete", {}).get("status", "False") or "False"
fail = conds.get("Failed", {}).get("status", "False") or "False"
reason = (conds.get("Failed", {}).get("reason", "") or conds.get("Complete", {}).get("reason", "")).replace("|", "/")
msg = (conds.get("Failed", {}).get("message", "") or conds.get("Complete", {}).get("message", "")).replace("|", "/").replace("\n", " ")
print(f"{comp}|{fail}|{reason}|{msg}")
' 2>/dev/null || printf 'False|False||')
      IFS='|' read -r complete failed reason msg <<< "$status_info"
      if [ "$complete" = "True" ]; then
        log "job/$NAME Complete"
        kctl logs "job/$NAME" --all-pods=true --tail=100 2>/dev/null || kctl logs "job/$NAME" --tail=100 || true
        exit 0
      fi
      if [ "$failed" = "True" ]; then
        log "job/$NAME Failed (reason=${reason:-unknown} ${msg:-})"
        collect_failure_artifacts "$NAME" "$COLLECT_DIR"
        exit 1
      fi
    else
      api_errs=$((api_errs + 1))
      log "warning: kubectl get job/$NAME error ($api_errs/5): $job_json"
      if [ "$api_errs" -ge 5 ]; then
        log "aborting wait after $api_errs consecutive API errors"
        collect_failure_artifacts "$NAME" "$COLLECT_DIR"
        exit 2
      fi
    fi
    now=$(date +%s)
    if [ $((now - start)) -ge "$WAIT_FOR" ]; then
      log "wait timed out after ${WAIT_FOR}s for job/$NAME"
      collect_failure_artifacts "$NAME" "$COLLECT_DIR"
      exit 124
    fi
    if [ "$poll" -gt 0 ]; then sleep "$poll"; fi
  done
fi
