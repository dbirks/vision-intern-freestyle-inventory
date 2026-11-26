# GPU jobs on the home cluster — how vision-intern runs experiments

Everything here is `kubectl` against context `admin@home`, namespace `default`, and every
object we create is labelled `app=vision-intern` (+ `vision-intern/job=<name>`,
`vision-intern/gpu=<5090|6000>`). No git remote, no registry push, no CI: the repo reaches
the cluster as a tarball over `kubectl cp`, and private images never leave the cluster.

```
k8s/lib.sh                 shared helpers (context, node/pvc names, render, transfer)
k8s/job-template.yaml      the parameterised Job (envsubst ${VAR} placeholders)
k8s/pvc-template.yaml      work PVC, one per GPU node (local-path is node-local)
k8s/workspace-template.yaml busybox helper pod that holds each PVC open = the cp landing pad
k8s/stage-data.sh          data/ + code -> cluster storage
k8s/submit.sh              render + apply a job
k8s/collect.sh             job's /out -> runs/<name>/
k8s/smoke/smoke.py         torch/CUDA probe + 4096x4096 matmul -> /out/smoke.json
```

## Quick start

```bash
k8s/stage-data.sh                              # data/ + repo -> both GPU nodes (~10 s)
k8s/submit.sh smoke-5090 5090 --wait           # the smoke test on the RTX 5090
k8s/submit.sh smoke-6000 6000 --wait           # the smoke test on the RTX PRO 6000
k8s/collect.sh smoke-5090                      # -> runs/smoke-5090/
k8s/collect.sh smoke-6000

# a real experiment
k8s/stage-data.sh --gpu 6000                   # re-stage after you touch data/
k8s/submit.sh yolo-runs-1 6000 --mem 8192 --timeout 21600 \
    --pip "supervision>=0.26 pycocotools" -- \
    python -m freestyle.train --cfg configs/experiments/yolo.yaml
k8s/collect.sh yolo-runs-1                     # -> runs/yolo-runs-1/
```

`submit.sh --dry-run` prints the rendered manifest and changes nothing. Flags: `--mem`,
`--mode hami|share|none|auto`, `--image`, `--cpu/-lim`, `--mem-host/-lim`, `--shm`,
`--timeout`, `--ttl`, `--pip`, `--no-code`, `--replace`, `--wait [--wait-for S]`. Flags must
come **before** `--`; everything after `--` is the command.

## The cluster as we found it (measured 2026-11-26)

| node | card | driver | VRAM | who lives there | HAMi free | physically free |
|---|---|---|---|---|---|---|
| `talos-210-73x` | NVIDIA RTX PRO 6000 Blackwell WS | 580.167.08 | 97887 MiB | `pennyroyal-flashnext` (SGLang, Qwen3.8-Flash-Next) books **97800 MiB** | **87 MiB** | 1317 MiB |
| `talos-8me-tcy` (label `home-k8s/egpu=rtx5090`) | NVIDIA GeForce RTX 5090 | 580.178.04 | 32607 MiB | `inkling-2-mini-basalt` books `gpumem-percentage: 100` = the **whole card** | **0 MiB** | ~909-975 MiB |
| `dave-droid` (taint `dedicated=pascal:NoSchedule`) | NVIDIA GeForce GTX 1070 | — | 8192 MiB | `qwen3-asr` books 3000 MiB (+ LFM3 voice) | ~5192 MiB | — |

`dave-droid` is **not a target**: Pascal `sm_61` has no kernels in the CUDA 13 wheels, and a
toleration for its taint would have to be added on purpose.

Both Blackwell cards are fully **booked** in HAMi by the resident LLM engines and ~97 %
**full** in hardware, so nothing meaningful can be co-resident next to them:
`nvidia.com/gpumem: 4096` will sit in `Pending` forever. Read `AGENTS.md` before "fixing"
that: `deviceMemoryScaling: 1` (no oversubscription, 46 GB host RAM), the whole-card
claim is what keeps the engines deterministic, and touching `prereqs/hami.yaml` can wedge
Flux repo-wide.

### The two GPU modes, and why `share` exists

* **`hami`** (what a training job should use): `nvidia.com/gpu: 1` + `nvidia.com/gpumem: <MiB>`,
  `schedulerName: hami-scheduler`, annotation `nvidia.com/use-gputype: "RTX 5090"` /
  `"PRO 6000"`, plus `nodeSelector: kubernetes.io/hostname`. This is the polite shape: HAMi
  refuses to co-schedule us over the engines' quota, and libvgpu hard-caps our process.
  Verified working on this cluster (job `hami-probe-1` bound `80 MiB` on the 6000, and
  `nvidia-smi` inside the pod reported an 80 MiB card).
  * Never ask for `nvidia.com/gpu` without `nvidia.com/gpumem` — that books the WHOLE card
    and starves the engines (`hami-gpu-accounting` skill).
  * Never set `nvidia.com/gpucores`: compute capping is unreliable on Blackwell here. `auto`
    mode therefore only ever sets memory.
* **`share`** (what the smoke test used, because it is the only thing that fits today):
  request **no** GPU resource, set `NVIDIA_VISIBLE_DEVICES=all` +
  `CUDA_DISABLE_CONTROL=true`, `schedulerName: ""`. This is the cluster's own
  `gpu-vram-exporter` / `gpu-power-limit` pattern — raw card, no HAMi booking, no isolation.
  Safety comes from two self-imposed limits instead of the scheduler:
  1. `submit.sh` reads the *real* free VRAM from the node's `gpu-vram-exporter` pod and
     refuses to start unless `free >= --mem + VI_CTX_SLACK` (512 MiB by default: our CUDA
     context alone measured 450-632 MiB of process memory before the first allocation);
  2. `smoke.py` then caps `torch.cuda.set_per_process_memory_fraction` to what the allocator
     can really use and walks down a ladder (fp16 → bf16 → fp32 → in-place fp16/bf16 →
     2048² in-place) instead of OOM-killing the serving pod next door. The 5090 accepted a
     192 MiB budget at `free = 909 MiB`; the 6000 needed 512 MiB at `free = 1317 MiB`
     (its CUDA context was 632 MiB).
* **`none`**: no GPU at all — use it to test staging/artifacts without touching a card.

### Freeing room for a real training job (owner decision — nothing here was applied)

Room for a 24 GiB training job appears the moment a resident engine leaves that much in
HAMi's books. Either park an engine through GitOps (`replicas: 0`, which takes a Collective
model down — the owner's call), or shrink its reservation. The latter is the small change;
suggested, **not applied** (it edits `apps/`, i.e. Flux owns it, and a rollout restarts the
engine):

```yaml
# apps/inkling-2-mini-basalt.yaml — 5090: 100 % of the card -> leave 6 GiB for jobs
           limits:
-            nvidia.com/gpumem-percentage: "100"
+            nvidia.com/gpumem: "26000"        # 32607 - 6607
# apps/pennyroyal-flashnext.yaml — PRO 6000: 97800 of 97887 -> leave ~8 GiB for jobs
-              nvidia.com/gpumem: 97800
+              nvidia.com/gpumem: 88000
```

A dedicated namespace + PVC manifest in `home-k8s` is *not* needed: `k8s/stage-data.sh`
creates the two PVCs and workspace pods live (labelled `app=vision-intern`) in `default`.
If the owner prefers them GitOps-managed, `k8s/pvc-template.yaml` +
`k8s/workspace-template.yaml` are already valid manifests.

## Storage: how data gets in and artifacts get out

`local-path` is the default StorageClass and provisions on the **node where the first
consumer pod lands** (RWO, node-bound) — so we use one PVC per GPU node:
`vision-intern-data-5090` (on `talos-8me-tcy`, NVMe) and `vision-intern-data-6000` (on
`talos-210-73x`). Each is held open by a `vision-intern-ws-<gpu>` busybox pod, which is the
landing pad for transfers; jobs mount the same PVC.

```
PVC layout                       job container
/work/data/**            <──     /src/data (symlink)      read: the staged copy of data/
/work/repo/repo.tar            staged by stage-data.sh (baseline code)
/work/repo/<job>.tar     <──     unpacked to /src         submit.sh uploads the tree per job
/work/jobs/<job>.sh      <──     the command, as a file   (no quoting survives 4 layers, so
/work/artifacts/<job>/** ──>     /out                     we ship it as a file); collect.sh
/work/cache/**                   HOME / pip / uv / HF cache (survives between jobs)
```

* `stage-data.sh` tars `data/` (14 MB → ~10 s to both nodes) and the tracked+untracked-but-
  not-ignored repo files (`git ls-files -co --exclude-standard`, minus `data/ runs/
  artifacts/ .venv/`, ~3.6 MB) and pushes them with `kubectl cp`.
* Private images go only to our own cluster, never a registry, never GitHub. The container
  image is the **public** upstream `pytorch/pytorch` one; nothing private is ever baked in.
* The Job mounts the claim read-write (it needs `/work/artifacts/<job>`), so `/work/data` is
  not write-protected — treat it as read-only, and re-run `stage-data.sh --no-code` if a job
  mangled it (seconds). Derived data (crops, undistorted tiles) belongs in `/out`.

### Gotchas found the hard way

* **`kubectl exec -i pod -- cat > f` silently truncates at ~192 KiB** against this cluster
  with kubectl v1.37.1 (`websocket: close 1006`, `connection reset by peer`;
  `KUBECTL_REMOTE_COMMAND_WEBSOCKETS=false` does not help). `kubectl cp` (15 MB, verified
  byte-identical) is the transfer that works, so `lib.sh:push_file`/`collect.sh` use it.
* **Mounting one PVC twice in the same pod, the second time with a `subPath`, writes into a
  directory the helper pod cannot see.** Verified with a busybox probe: `echo > /out/x`
  succeeded and `/work/artifacts/...` stayed empty. So the Job mounts the claim exactly once
  (`/work`, rw) and `ln -sfn /work/artifacts/<job> /out`.

## Monitoring, collecting, cleanup

```bash
kubectl --context admin@home -n default get job -l app=vision-intern
kubectl --context admin@home -n default get pods -l vision-intern/job=<name> -o wide
kubectl --context admin@home -n default logs -f job/<name>
kubectl --context admin@home -n default describe pod <pod>     # scheduling failures: Events
kubectl --context admin@home -n default get pod <pod> -o jsonpath='{.metadata.annotations}' \
  | tr ',' '\n' | grep -i hami                                 # what HAMi booked
# who is using the card right now (real, physical, per-process):
kubectl --context admin@home -n default exec <gpu-vram-exporter-on-that-node> -- \
  nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv
```

Jobs are immutable: change anything → `--replace` (deletes + recreates) or a new name.
`backoffLimit: 0` (no retry — a failed experiment should fail loudly),
`ttlSecondsAfterFinished: 86400` (pods and their logs survive a day for forensics),
`activeDeadlineSeconds` (`--timeout`) kills runaway jobs. Delete only your own objects, by
name. The workspace pods + PVCs are meant to stay; `kubectl --context admin@home -n default
delete job,pvc,pod -l app=vision-intern` wipes everything we made (PVCs are `Delete`
reclaim: the work dirs go with them).

## Things a training job needs to know

* **Image**: `pytorch/pytorch:2.11.0-cuda13.0-cudnn9-runtime` (torch 2.11.0+cu130, cuDNN
  9.19, sm_120 kernels present — verified on both cards). Driver reports CUDA 13.0; anything
  `cuda12.8`+ works too, and `2.9.1-cuda13.0-cudnn9-runtime` is the fallback tag.
  **First pull 6 m 6 s** (3.0 GB compressed) on `talos-8me-tcy`, 59 s on `talos-210-73x`. Warm a node by submitting
  `--mode none --mem 0 -- true` before a long job, or keep `imagePullPolicy: IfNotPresent`.
* **torch is already in the image**, python 3.12 at `/usr/local` (Debian
  externally-managed-environment, `dist-packages`). Nothing needs installing to *import* the
  repo: the container sets `PYTHONPATH=/src`, `cwd=/src` and symlinks `/src/data`, so
  `python -m freestyle.<x>` just works. `--pip` runs `pip install --break-system-packages` (the image's interpreter is a Debian
  externally-managed one and refuses otherwise); wheels land in the cached
  `/work/cache/pip`, so re-runs are quick — installing `six` cost ~2 s. PyPI egress works
  from pods. The repo's `train` extra would replace the image's cu130 torch with the default
  wheel — pass individual specs (`--pip "supervision pycocotools"`) instead of extras.
* **`/dev/shm` is 2 Gi** (`--shm`), an emptyDir on `medium: Memory`, so it counts against the
  container's **memory limit**: DataLoader workers → raise `--mem-host-lim` with `--shm`.
* **Host RAM is the binding constraint** on `talos-210-73x` (~47 GiB allocatable, ~14 GiB of
  *requests* spare next to pennyroyal). The 5090 node has ~24 GiB spare.
* Measured 4096² matmul, co-resident with the serving engine (no exclusive card — these are
  "what you get next to the LLM", not card peaks; the spread is the neighbour's load):
  **RTX 5090** 207.8-219.2 TFLOPS fp16, 108.2-222.3 bf16, peak 104 MiB, 24 s job;
  **RTX PRO 6000** 101.6-116.6 fp16, 105.2-112.0 bf16, 19.6-20.5 fp32, peak 200 MiB, 31 s
  job. Raw JSON in `runs/smoke-5090/smoke.json` / `runs/smoke-6000/smoke.json` (gitignored).
  The 6000 number is taken while SGLang sits at the 299 W/300 W cap at 94 % util.
* The job container's shell is a minimal userland: `tail -4` is rejected ("option used in
  invalid context"); write `tail -n 4`.
* Node clocks run ~14 s behind this workstation: `tar` prints "time stamp … is in the
  future" when unpacking a fresh repo tarball. Harmless; `touch`-ing before tar would fix it.
* `enableServiceLinks: false` is mandatory here (AGENTS.md): ~200 services in `default`
  would otherwise inject a wall of `*_PORT`/`*_SERVICE_HOST` env vars, and `*_PORT` collides
  with serving-stack variables (vLLM's `VLLM_PORT`). Keep it even though we run Jobs.
* The PodSecurity warnings for the `default` namespace (`runAsNonRoot`, capabilities) are
  advisory — nothing enforces them here, and the workspace pod needs uid 0 because
  local-path provisions root-owned directories.
* A GPU pod stuck `Pending` in `hami` mode: that is the expected answer today. Measured,
  `--mode hami --mem 4096` on the 6000 gives
  `FailedScheduling ... 0/3 nodes are available: 1 NodeUnfitPod, ... no new claims to
  deallocate` from `hami-scheduler` — the reservation, not the hardware, is what is full. If a pod is Pending with *"Insufficient memory"* but the card looks free,
  check `schedulerName` — pods admitted while `hami-webhook` was down keep the
  `default-scheduler`, which cannot satisfy `gpumem` (`hami-gpu-accounting` skill;
  `apps/gpu-pod-rescuer` CronJob heals deployments, not jobs — delete the pod to reschedule).
* If the node starts logging `NV_ERR_GPU_IN_FULLCHIP_RESET` and `nvidia.com/gpu: 0`, stop:
  that is the PRO 6000 GSP wedge; the only fix is the BMC power-cycle in
  `/home/david/dev/home-k8s/AGENTS.md` ("GPU crash recovery"). The workstation watchdog
  does it automatically. Do not retry-loop GPU jobs into a wedged card.
