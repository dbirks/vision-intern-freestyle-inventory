# Kubernetes launch review

## Findings

### Critical — Dry run can delete an unrelated Job

**Location:** `k8s/submit.sh:112-123`  
**Scenario:** `submit.sh existing-name 5090 --replace --dry-run --mode none -- true` deletes the existing Job, including one belonging to another application.  
**Evidence:** The name lookup and unconditional `kctl delete job` run before the dry-run exit; no ownership labels or UID are checked. The README promises dry-run changes nothing (`k8s/README.md:36`). Even without dry-run, `--replace` can delete anyone's Job.  
**Fix:** Exit after rendering for dry-run before any mutations. For replacement, verify an ownership marker and expected identity, delete with a UID precondition, and refuse unrelated objects; avoid name-based apply races by using create and handling AlreadyExists.

### Critical — Auto share runs arbitrary training without an enforced GPU limit

**Location:** `k8s/submit.sh:71-74,93-102`; `k8s/job-template.yaml:94-107`; `freestyle/experiments/launch.py:19-25`  
**Scenario:** HAMi has insufficient free reservation but physical VRAM appears available, so an experiment automatically enters share mode alongside a serving workload.  
**Evidence:** Share mode requests no GPU resource and exposes all devices; `VI_GPU_MEM_MB` is only an environment variable. The smoke test implements an allocator cap, but the experiment launcher does not arrange one. Even an allocator cap would not isolate other CUDA allocations. A point-in-time free-memory check cannot reserve memory against other users or concurrent jobs; explicit share also proceeds when the measurement is missing (`submit.sh:94`).  
**Fix:** Default real experiments to HAMi and fail closed when reservation is unavailable. Make unisolated sharing explicit opt-in for known workloads, reject missing measurements, and use actual isolation/reservation for a reliable limit.

### High — Failed Jobs wait until timeout and lose automatic collection

**Location:** `k8s/submit.sh:156-159`; `freestyle/experiments/launch.py:85-94`  
**Scenario:** The command exits nonzero early, or the Job fails its active deadline.  
**Evidence:** `kubectl wait` watches only Complete, so Failed does not end the wait; with `set -e`, timeout exits before logs. The launcher uses the full experiment timeout for waiting and `check=True` prevents collection on failure. At the deadline, scheduling and termination delays can also make the client time out before the Job reports its terminal condition.  
**Fix:** Watch both terminal conditions with a bounded deadline and distinguish failure, timeout, and API errors. Fetch logs and attempt partial artifact collection on failure; allow time beyond the Job deadline for terminal reporting.

### High — Existing workspace and PVC are trusted and a foreign pod can be deleted

**Location:** `k8s/lib.sh:123-141`  
**Scenario:** A different owner has a PVC or pod with one of the fixed workspace names, or a stale workspace is on the wrong node.  
**Evidence:** Any existing PVC is accepted by name; a Running pod is accepted without inspecting labels, node, mount, or UID. A non-Running pod is deleted unconditionally. This can destroy another owner's pod or stage private files into an unintended volume.  
**Fix:** Verify ownership, expected node and claim binding/mount before use; refuse mismatches. Delete only verified owned pods with a UID precondition and handle create races safely.

### High — Gated Hugging Face models receive no token

**Location:** `k8s/job-template.yaml:94-107`  
**Scenario:** An experiment loads a gated model such as SAM3 with the existing `default/hf-token` secret.  
**Evidence:** The container environment has no `HF_TOKEN` secretKeyRef or secret mount; the backend requires `HF_TOKEN` (`freestyle/propose/backends.py:474-480`). A previously cached credential is neither guaranteed nor a safe credential delivery contract.  
**Fix:** Inject `HF_TOKEN` from the actual key of the existing `hf-token` Secret using `secretKeyRef`, without logging its value; document namespace and missing-secret behavior.

### High — Artifact and snapshot reuse can corrupt results

**Location:** `k8s/submit.sh:112-115,128-150`; `k8s/collect.sh:28-39`  
**Scenario:** A name is reused after TTL or with `--replace`, or two submissions race on the same name.  
**Evidence:** Uploaded script and tarball and artifact directory are keyed only by name; artifacts are merely `mkdir -p` and never cleared. Competing submitters can overwrite inputs before a pod reads them, and collection merges into an existing destination. `kubectl cp` errors are ignored and a partial copy containing one file is accepted as success.  
**Fix:** Use a unique immutable run identifier and atomic staging, create the Job exclusively, and never reuse its artifact directory. Check transfer exit status and integrity and collect into a fresh destination with an atomic publish.

### Medium — `--no-code` points at a tarball that is never staged

**Location:** `k8s/submit.sh:23,128-130`; `k8s/stage-data.sh:61-63`; `k8s/job-template.yaml:82-84`  
**Scenario:** Submit a fresh name with `--no-code` after staging the baseline.  
**Evidence:** Staging writes `/work/repo/repo.tar`, while the Job always extracts `/work/repo/$VI_JOB_NAME.tar`; disabling upload leaves that file absent or stale from a previous run.  
**Fix:** Select the baseline path explicitly when `--no-code` is used, or remove the flag; validate the selected snapshot and record its digest.

### Medium — Recorded Git state does not identify the actual executed snapshot

**Location:** `freestyle/experiments/launch.py:65-73,81-85`; `k8s/lib.sh:164-168`  
**Scenario:** Files change between state calculation and tar creation, or the direct submit script is used.  
**Evidence:** The launcher records HEAD and a working-tree diff hash before `submit.sh` separately creates a tar of tracked and untracked working files. Direct submissions record neither SHA nor archive digest. The dirty tree is not preserved as a reconstructible snapshot, and the archive is not content-verified on upload (`lib.sh:150-161` checks length only).  
**Fix:** Create one immutable snapshot, hash its exact bytes, record its digest and Git HEAD/dirty provenance in the artifacts, and verify the remote bytes. Preserve or publish the snapshot where reproducibility requires it.

### Medium — Dependency and image resolution are not reproducible

**Location:** `freestyle/experiments/launch.py:16-18`; `k8s/lib.sh:15-16`; `k8s/submit.sh:135-139`  
**Scenario:** Re-running the same configuration after a package or image changes yields different code or behavior.  
**Evidence:** `pycocotools` and `scipy` and transitive dependencies are unpinned; the README also suggests unpinned or range specs. The default image has a versioned tag but no immutable digest, and `IfNotPresent` may select different cached tag contents across nodes.  
**Fix:** Use an image pinned by digest and a fully resolved, version- and preferably hash-locked dependency set; record the actual image ID and installed versions with each run.

### Medium — Private data is writable and arbitrary job commands can export it

**Location:** `k8s/job-template.yaml:85-87,111-119`; `k8s/README.md:127-134`  
**Scenario:** A buggy or untrusted training command modifies staged data or sends it over network egress.  
**Evidence:** The whole PVC is mounted read-write and `/src/data` points into it; the README acknowledges this. The job also installs packages from PyPI and has no egress restrictions. The scripts themselves stage data to the cluster and collect artifacts to the host, but cannot guarantee that private data never leaves the cluster PVC.  
**Fix:** Mount a separate data volume read-only, limit who can supply commands and images, and enforce a suitable network egress policy; define whether artifact collection may contain private data.

### Medium — Names and command options are not safely validated

**Location:** `k8s/submit.sh:37,47-54,135-139`; `freestyle/experiments/launch.py:19`  
**Scenario:** A name contains uppercase, dots, starts or ends with punctuation, is too long, or normalization makes two names identical; a pip spec contains shell metacharacters.  
**Evidence:** The name check accepts values Kubernetes Job/label validation rejects and does not check length or normalized collisions. `--pip` is inserted directly into shell source and an echoed single-quoted string; shell syntax or metacharacters change execution. Ordinary command arguments after `--` are preserved using Bash `%q`.  
**Fix:** Validate DNS and label constraints including lengths after normalization, use unique names, and pass pip requirements as separately quoted arguments or a requirements file rather than shell source.

## STATUS

Changes requested. The current launcher can delete unrelated objects, run a GPU job without enforced isolation, and wait until timeout on failed Jobs.

## FILES

Reviewed `k8s/*.sh`, `k8s/*.yaml`, `k8s/README.md`, and `freestyle/experiments/launch.py`; consulted relevant experiment schema, entry and backend references. This review is the only file written.

## CHECKS

- `bash -n k8s/*.sh` passed.
- Static inspection of rendering, staging, scheduling, wait, collection, ownership, credentials, and provenance paths.
- No Jobs submitted or deleted; no cluster state or private data inspected. No runtime or dry-run validation was performed.

## NOTES

The versioned default PyTorch image tag and launcher Git metadata are useful starting points, but do not by themselves pin the executed image and tarball. Job and pod templates include app and GPU labels, node selectors, and a deadline/TTL; ownership must still be verified before reuse or deletion. Job TTL removes Kubernetes objects, not persistent PVC scripts, snapshots, caches, or artifacts, so an owned, retention-aware cleanup mechanism is still needed.
