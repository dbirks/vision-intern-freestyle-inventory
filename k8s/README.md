# GPU jobs on the home cluster — how vision-intern runs experiments

Everything here is `kubectl` against context `admin@home`, namespace `default`, and every
object we create is labelled `app=vision-intern` (+ `vision-intern/job=<name>`,
`vision-intern/gpu=<5090|6000>`) and annotated `vision-intern/owner=<VI_OWNER_PREFIX>...`
+ `vision-intern/run-id=<run_id>`. Ownership is verified before reusing PVCs/workspace
pods or replacing a Job. No git remote, no registry push, no CI: the repo reaches the
cluster as a SHA-256-verified tarball over `kubectl cp`, and private images never leave
the cluster.

```
k8s/lib.sh                 shared helpers (ownership checks, validation, render, transfer)
k8s/job-template.yaml      the parameterised Job (envsubst ${VAR} placeholders)
k8s/pvc-template.yaml      node-local PVC template (data PVC + separate outputs PVC)
k8s/workspace-template.yaml busybox helper pod template (holds each PVC open for cp)
k8s/stage-data.sh          data/ + baseline repo -> cluster storage
k8s/submit.sh              validate, stage immutable run, render + create a Job
k8s/collect.sh             verified atomic collection of job's /out -> runs/<name>/
k8s/smoke/smoke.py         torch/CUDA probe + 4096x4096 matmul -> /out/smoke.json
```

## Quick start

```bash
k8s/stage-data.sh                              # data/ + baseline repo -> both GPU nodes
k8s/submit.sh smoke-5090 5090 --mode share --mem 192 --wait
k8s/submit.sh smoke-6000 6000 --mode share --mem 512 --wait
k8s/collect.sh smoke-5090                      # -> runs/smoke-5090/
k8s/collect.sh smoke-6000

# a real experiment (via launcher or direct submit; defaults to hami and fails closed)
uv run python -m freestyle.experiments.launch configs/experiments/C1_rfdetr_small_agnostic_dinov2.yaml --dry-run
k8s/submit.sh yolo-runs-1 6000 --mem 8192 --timeout 21600 \
    --pip "supervision==0.26.1 pycocotools==2.0.11" -- \
    python -m freestyle.train --cfg configs/experiments/yolo.yaml
k8s/collect.sh yolo-runs-1                     # -> runs/yolo-runs-1/
```

`submit.sh --dry-run` renders and prints the Job manifest without mutating the cluster
(no `delete`, no `apply`, no staging). Use `--dry-run=server` to also validate against the
API server with `kubectl create --dry-run=server`.

Flags: `--mem`, `--mode hami|share|none|auto`, `--image`, `--cpu`/`--cpu-lim`,
`--mem-host`/`--mem-host-lim`, `--shm`, `--timeout`, `--ttl`, `--pip`,
`--pip-constraints`, `--hf-token`, `--no-code`, `--replace`, `--dry-run` /
`--dry-run=server`, `--wait [--wait-for S]`, `--collect-dir DIR`. Flags must come
**before** `--`; everything after `--` is the command.

## Validation, ownership, and safety guarantees

* **Names and flags**: `<name>` must be a lowercase DNS-1123 label `<= 57` chars
  (`^[a-z0-9]([a-z0-9-]*[a-z0-9])?$`, leaving room for pod `-xxxxx` suffix). `--mem`,
  `--timeout`, `--ttl`, `--wait-for`, `--cpu*`, `--mem-host*`, `--shm`, `--image`,
  `--pip`, and `--pip-constraints` are strictly validated. `--pip` and
  `--pip-constraints` are written to requirements/constraints files and passed via
  `pip install -r ... -c ...` (never interpolated into shell source).
* **Safe `--replace`**: `--replace` refuses to delete any Job unless it carries both
  label `app=vision-intern` and annotation `vision-intern/owner` starting with
  `VI_OWNER_PREFIX` (`vision-intern`), and verifies its `metadata.uid` before deletion.
  Jobs are created with `kubectl create` (exclusive) rather than `apply`.
* **Verified workspace & PVC reuse**: `lib.sh` checks `app=vision-intern`,
  `app.kubernetes.io/component=workdir|workspace`, `vision-intern/gpu`, node affinity/
  `nodeSelector`, and PVC claim binding before reusing (or deleting a non-Running owned)
  workspace pod or PVC.
* **Wait & failure collection**: `--wait` polls for `Complete` OR `Failed` (and detects
  API errors/timeouts). On failure or timeout, `submit.sh` and `launch.py` collect both
  pod logs (`job.log`) and `/out` artifacts.
* **Gated Hugging Face models (`HF_TOKEN`)**: Secret `hf-token` (key `HF_TOKEN` in
  `default` namespace) is mounted via `secretKeyRef` **only** when `--hf-token` is passed
  (triggered by `needs_hf_token: true` in experiment configs). The token value is never
  logged or printed. If the secret is missing at submission time, `submit.sh` fails fast.
* **Immutable snapshot, provenance, and environment**:
  * Every submission generates a unique `RUN_ID` (`vision-intern/run-id`) so staged
    tarballs, scripts, and `/out` artifacts never collide across `--replace` or TTL reuse.
  * `push_file_to_pod` verifies both byte length and SHA-256 on the pod before atomic
    rename; the Job verifies `${REPO_TAR_SHA256}` before unpacking `/src`.
  * `/out/provenance.json` records `run_id`, `job_name`, `repo_tar_path`,
    `repo_tar_sha256`, `git_sha`, `git_dirty`, `git_diff_hash`, `image`, `gpu`, `mode`,
    and `vram_mb`.
  * `VI_IMAGE` is pinned by tag + digest
    (`pytorch/pytorch:2.11.0-cuda13.0-cudnn9-runtime@sha256:bfbb4a2b4fdba0fefdb428ea737e626d61bb3daf74a16e1ff935bdb03aa7c3f0`).
  * `/out/env.txt` records the pinned image, Python version, `pip freeze`, and (on
    collection) the kubelet-reported `pod_image_id` digest.

## The cluster as we found it (measured 2026-11-26)

| node | card | driver | VRAM | who lives there | HAMi free | physically free |
|---|---|---|---|---|---|---|
| `talos-210-73x` | NVIDIA RTX PRO 6000 Blackwell WS | 580.167.08 | 97887 MiB | `pennyroyal-flashnext` (SGLang, Qwen3.8-Flash-Next) books **97800 MiB** | **87 MiB** | 1317 MiB |
| `talos-8me-tcy` (label `home-k8s/egpu=rtx5090`) | NVIDIA GeForce RTX 5090 | 580.178.04 | 32607 MiB | `inkling-2-mini-basalt` books `gpumem-percentage: 100` = the **whole card** | **0 MiB** | ~909-975 MiB |
| `dave-droid` (taint `dedicated=pascal:NoSchedule`) | NVIDIA GeForce GTX 1070 | — | 8192 MiB | `qwen3-asr` books 3000 MiB (+ LFM3 voice) | ~5192 MiB | — |

`dave-droid` is **not a target**: Pascal `sm_61` has no kernels in the CUDA 13 wheels, and a
toleration for its taint would have to be added on purpose.

### GPU modes (`hami`, `share`, `none`, `auto`)

* **`hami`** (default for real experiments via `launch.py`, and for `auto` when HAMi has
  room): `nvidia.com/gpu: 1` + `nvidia.com/gpumem: <MiB>`, `schedulerName: hami-scheduler`,
  annotation `nvidia.com/use-gputype`, plus `nodeSelector: kubernetes.io/hostname`.
  * `auto` mode **fails closed** when HAMi free `< --mem` — it never silently enters
    unisolated `share` mode.
  * Never ask for `nvidia.com/gpu` without `nvidia.com/gpumem` — that books the WHOLE card.
  * Never set `nvidia.com/gpucores`: compute capping is unreliable on Blackwell here.
* **`share`** (explicit opt-in `--mode share` only):
  requests **no** GPU resource, sets `NVIDIA_VISIBLE_DEVICES=all` +
  `CUDA_DISABLE_CONTROL=true`, `schedulerName: ""`.
  Safety checks and self-enforcement:
  1. `submit.sh` requires a valid physical free VRAM reading from `gpu-vram-exporter` and
     refuses unless `phys_free >= --mem + VI_CTX_SLACK` (default `512` MiB). Missing
     measurements fail closed.
  2. The Job injects `/_vi_site/sitecustomize.py` on `PYTHONPATH` and sets
     `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128`. On `torch`
     import, `sitecustomize.py` calls `torch.cuda.set_per_process_memory_fraction` to
     hard-cap PyTorch's caching allocator to `--mem` MiB (and clamps subsequent raises).
     Note: this caps PyTorch allocations, not arbitrary non-PyTorch CUDA binaries.
* **`none`**: no GPU at all (`NVIDIA_VISIBLE_DEVICES=""`) — proves staging/artifacts/CPU
  pipelines without touching a card.

## Storage: read-only data and separate writable outputs

`local-path` provisions node-bound RWO volumes, so each GPU node uses **two separate PVCs**:

1. **Data PVC** (`vision-intern-data-<gpu>`, held by `vision-intern-ws-<gpu>`):
   staged by `stage-data.sh`, and mounted in every Job at `/data-ro` with
   **`readOnly: true`**. `/src/data` and `/work/data` symlink to `/data-ro/data`, so any
   write to dataset inputs fails at the kernel level (`EROFS`).
2. **Outputs PVC** (`vision-intern-out-<gpu>`, held by `vision-intern-out-ws-<gpu>`):
   mounted read-write at `/outputs` in every Job. Per-run tarballs, scripts, pip caches,
   and `/out` (`/outputs/artifacts/<run_id>`) live here, isolated from `data/`.

* Data under `data/` is private: only trusted operators should submit Jobs/images, and
  collected artifacts under `runs/` (which may contain crops or prediction visualizations)
  remain gitignored and are never pushed to a public registry or remote. If stricter
  pod egress isolation is desired (beyond PyPI/HF downloads), add a Kubernetes
  `NetworkPolicy` in `default` allowing only DNS and approved endpoints.
* `collect.sh` refuses non-empty destinations, verifies remote-to-local SHA-256 checksums
  for every file, and publishes atomically.

## Monitoring, collecting, cleanup

```bash
kubectl --context admin@home -n default get job -l app=vision-intern
kubectl --context admin@home -n default get pods -l vision-intern/job=<name> -o wide
kubectl --context admin@home -n default logs -f job/<name>
kubectl --context admin@home -n default describe pod <pod>
k8s/stage-data.sh --list
```
