#!/usr/bin/env bash
# Render k8s/job-template.yaml and submit it as a Job on the home cluster.
#
#   k8s/submit.sh <name> <gpu:5090|6000> [flags] -- <command...>
#   k8s/submit.sh smoke-5090 5090                      # runs k8s/smoke/smoke.py
#   k8s/submit.sh train-x 6000 --mem 8192 -- python -m freestyle.train --cfg c.yaml
#
# Flags (all optional):
#   --mem MB        VRAM for the job. hami mode: the nvidia.com/gpumem reservation.
#                   share mode: the self-imposed cap.           (default 4096)
#   --mode M        hami | share | none | auto (default auto: hami if HAMi has the memory
#                   free, else share if the card physically has room, else refuse).
#                   none = no GPU at all: proves the stage/unpack/artifact path in seconds
#                   without touching a card (image pull still costs ~6 min the first time).
#   --image IMG     base image (default $VI_IMAGE, a public CUDA PyTorch runtime)
#   --cpu N / --cpu-lim N       cpu request / limit          (defaults 2 / 8)
#   --mem-host R / --mem-host-lim L   host RAM request / limit (defaults 4Gi / 16Gi)
#   --shm SIZE      /dev/shm size                        (default 2Gi)
#   --timeout SECS  activeDeadlineSeconds                (default 3600)
#   --ttl SECS      ttlSecondsAfterFinished              (default 86400)
#   --pip "specs"   pip install --break-system-packages args run before the command
#                   (e.g. --pip "supervision>=0.26"); ~10-40 s, deps are NOT cached
#   --no-code       do not re-upload the repo tarball (use whatever stage-data.sh staged)
#   --replace       delete an existing job with the same name first
#   --dry-run       print the rendered manifest, do not apply
#   --wait          kubectl wait for the job to complete (20m default, --wait-for to change)
#   --wait-for S    wait timeout in seconds              (default 1200)
#
# Environment: VI_CONTEXT (kubectl context, default admin@home), VI_NS (default default),
# plus every VI_* default in k8s/lib.sh.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

NAME="${1:-}"; GPU="${2:-}"
if [ -z "$NAME" ] || [ -z "$GPU" ]; then sed -n '2,27p' "$0" >&2; exit 2; fi
case "$GPU" in 5090|6000) : ;; *) die "gpu must be 5090 or 6000 (got '$GPU')" ;; esac
case "$NAME" in *[!a-zA-Z0-9.-]*|"") die "job name '$NAME' must be dns-label safe" ;; esac
shift 2

MEM=4096; MEM_SET=0; MODE=auto; IMAGE="$VI_IMAGE"
# Slack for our own CUDA context + cuBLAS handles (~0.5 GiB on Blackwell, measured).
VI_CTX_SLACK="${VI_CTX_SLACK:-512}"; CPU_REQ=2; CPU_LIM=8
MEM_REQ=4Gi; MEM_LIM=16Gi; SHM=2Gi; DEADLINE=3600; TTL=86400; PIP=""
UPLOAD_CODE=1; DRY=0; WAIT=0; WAIT_FOR=1200; REPLACE=0; CMD=""
while [ $# -gt 0 ]; do
  case "$1" in
    --mem) MEM="$2"; MEM_SET=1; shift 2 ;; --mode) MODE="$2"; shift 2 ;; --image) IMAGE="$2"; shift 2 ;;
    --cpu) CPU_REQ="$2"; shift 2 ;; --cpu-lim) CPU_LIM="$2"; shift 2 ;;
    --mem-host) MEM_REQ="$2"; shift 2 ;; --mem-host-lim) MEM_LIM="$2"; shift 2 ;;
    --shm) SHM="$2"; shift 2 ;; --timeout) DEADLINE="$2"; shift 2 ;; --ttl) TTL="$2"; shift 2 ;;
    --pip) PIP="$2"; shift 2 ;; --no-code) UPLOAD_CODE=0; shift ;;
    --dry-run) DRY=1; shift ;; --wait) WAIT=1; shift ;; --wait-for) WAIT_FOR="$2"; shift 2 ;;
    --replace) REPLACE=1; shift ;;
    --) shift; CMD=$(printf '%q ' "$@"); break ;;   # %q keeps the caller's quoting intact
    *) die "unknown flag $1 (command args must follow --)" ;;
  esac
done

NODE=$(gpu_node "$GPU"); TYPE=$(gpu_type "$GPU"); PVC=$(pvc_name "$GPU")

if [ -z "$CMD" ]; then
  CMD="python k8s/smoke/smoke.py"           # the default job is the smoke test
  if [ "$MEM_SET" = 0 ]; then MEM=192; fi   # a smoke test does not need much VRAM
  log "no command given, running the smoke test: $CMD (--mem $MEM)"
fi
case "$MODE" in hami|share|none|auto) : ;; *) die "bad --mode $MODE" ;; esac

# ---- pick the GPU accounting mode ------------------------------------------------------
HAMI_FREE=$(hami_free_mib "$NODE" "$TYPE"); PHYS_FREE=$(free_vram_mib "$NODE")
log "gpu $GPU ($TYPE on $NODE): HAMi reservation free=${HAMI_FREE:-?} MiB, physical free=${PHYS_FREE:-?} MiB"
if [ "$MODE" = auto ]; then
  if [ -n "$HAMI_FREE" ] && [ "$HAMI_FREE" -ge "$MEM" ]; then MODE=hami
  elif [ -n "$PHYS_FREE" ] && [ "$PHYS_FREE" -ge $((MEM + VI_CTX_SLACK)) ]; then MODE=share
  elif [ "$DRY" = 1 ]; then
    # --dry-run previews the command a job would get once the card is free: render the
    # polite HAMi shape instead of refusing on today's (booked) occupancy.
    MODE=hami; log "dry-run: no room today (HAMi ${HAMI_FREE:-?}/phys ${PHYS_FREE:-?} MiB); previewing the hami command a free card would use"
  else
    die "no card room for ${MEM} MiB on $GPU (HAMi free ${HAMI_FREE:-?} MiB, physical free ${PHYS_FREE:-?} MiB).
Free VRAM by scaling a resident engine to 0 (GitOps, apps/*.yaml), shrink its
      nvidia.com/gpumem reservation, or pass a smaller --mem."
  fi
fi
log "mode=$MODE"

if [ "$MODE" = none ]; then
  LIMITS="{cpu: $CPU_LIM, memory: $MEM_LIM}"
  POD_ANNOTATIONS="{}"
  SCHEDULER_NAME=""
  # runtimeClassName: nvidia + NVIDIA_DRIVER_CAPABILITIES stay set (the image needs the
  # runtime), so the card is otherwise VISIBLE here: I measured a `--mode none` pod whose
  # nvidia-smi reported the RTX 5090 and torch.cuda.is_available() was True. Without an
  # explicit device the zero-shot backends' `_resolve_device(None)` would pick 'cuda' and
  # silently allocate on the raw card next to the serving engines. NVIDIA_VISIBLE_DEVICES=""
  # makes 'none' mean no GPU at all (torch.cuda.is_available() -> False), which is exactly
  # what this mode promises; experiments that truly need the card use hami/share.
  SHARE_ENV='- { name: NVIDIA_VISIBLE_DEVICES, value: "" }'
  log "none mode: card hidden (NVIDIA_VISIBLE_DEVICES=empty) so no code path can touch it"
else
if [ "$MODE" = hami ]; then
  LIMITS="{cpu: $CPU_LIM, memory: $MEM_LIM, nvidia.com/gpu: 1, nvidia.com/gpumem: $MEM}"
  POD_ANNOTATIONS="{nvidia.com/use-gputype: \"$TYPE\"}"
  SCHEDULER_NAME=hami-scheduler
  SHARE_ENV=""
else
  if [ -n "$PHYS_FREE" ] && [ "$PHYS_FREE" -lt $((MEM + VI_CTX_SLACK)) ]; then
    die "share mode refuses: ${PHYS_FREE} MiB physically free on $GPU, need ${MEM} MiB + ${VI_CTX_SLACK} MiB for the CUDA context. Smaller --mem, or free the card (GitOps: scale a resident engine to 0)."
  fi
  LIMITS="{cpu: $CPU_LIM, memory: $MEM_LIM}"
  POD_ANNOTATIONS="{nvidia.com/use-gputype: \"$TYPE\"}"
  SCHEDULER_NAME=""                      # empty => default-scheduler (no HAMi booking)
  SHARE_ENV='- { name: NVIDIA_VISIBLE_DEVICES, value: "all" }
            - { name: CUDA_DISABLE_CONTROL, value: "true" }'
  log "share mode: no HAMi reservation, the job caps itself to $MEM MiB (VI_GPU_MEM_MB)"
fi
fi

# ---- render (--dry-run stops here, before anything is created) ------------------------
if [ "$MODE" = hami ]; then
  REQ="{cpu: $CPU_REQ, memory: $MEM_REQ, nvidia.com/gpu: 1, nvidia.com/gpumem: $MEM}"
else
  REQ="{cpu: $CPU_REQ, memory: $MEM_REQ}"
fi
if kctl get job "$NAME" >/dev/null 2>&1; then
  if [ "$REPLACE" = 1 ]; then log "replacing existing job/$NAME"; kctl delete job "$NAME" --wait=true; else
    die "job/$NAME already exists (use --replace to overwrite, or a new name; jobs are immutable)"
  fi
fi
manifest=$({
  export REQUESTS="$REQ" LIMITS POD_ANNOTATIONS SCHEDULER_NAME SHARE_ENV
  export JOB_NAME="$NAME" GPU NODE PVC_NAME="$PVC" IMAGE SHM_SIZE="$SHM"
  export ACTIVE_DEADLINE="$DEADLINE" TTL VI_GPU_MODE="$MODE" VI_GPU_MEM_MB="$MEM"
  render k8s/job-template.yaml
})
if [ "$DRY" = 1 ]; then printf '%s\n' "$manifest"; exit 0; fi

# ---- stage code + the command into the PVC --------------------------------------------
ensure_workspace "$GPU"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
if [ "$UPLOAD_CODE" = 1 ]; then
  repo_tarball "$tmp/$NAME.tar" >/dev/null
  push_file "$GPU" "$tmp/$NAME.tar" "/work/repo/$NAME.tar"
fi
# The command lives in a file so quoting survives the trip through envsubst + JSON + bash.
{
  echo 'set -euo pipefail'
  if [ -n "$PIP" ]; then
    echo "echo '== installing deps: $PIP'"
    # The runtime image's interpreter is a Debian externally-managed-environment one, so
    # plain pip refuses; the container is disposable, so install into it directly.
    echo "pip install --no-cache-dir --break-system-packages $PIP"
  fi
  printf '%s\n' "$CMD"
} > "$tmp/$NAME.sh"
push_file "$GPU" "$tmp/$NAME.sh" "/work/jobs/$NAME.sh"
ws_sh "$GPU" "mkdir -p /work/artifacts/$NAME"
if [ "$(ws_sh "$GPU" 'ls -A /work/data 2>/dev/null | wc -l' | tr -d ' ')" = "0" ]; then
  log "WARNING: no data/ staged on $GPU -- run k8s/stage-data.sh (a job that reads /src/data will fail)"
fi

printf '%s\n' "$manifest" > "$tmp/$NAME.yaml"
printf '%s\n' "$manifest" | kctl apply -f - >/dev/null
log "submitted job/$NAME (gpu=$GPU mode=$MODE mem=${MEM}MiB host=${MEM_REQ}/${MEM_LIM} deadline=${DEADLINE}s)"
log "watch:   kubectl --context $VI_CONTEXT -n $VI_NS get pod -l vision-intern/job=$NAME -w"
log "logs:    kubectl --context $VI_CONTEXT -n $VI_NS logs -f job/$NAME"
log "collect: k8s/collect.sh $NAME"

if [ "$WAIT" = 1 ]; then
  log "waiting up to ${WAIT_FOR}s for completion"
  kctl wait --for=condition=complete --timeout="${WAIT_FOR}s" "job/$NAME"
  kctl logs "job/$NAME" --tail=40 || true
fi
