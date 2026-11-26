"""Hermetic bash-level tests for k8s/*.sh scripts and launch.py.

Every test runs against an isolated fixture with:
- fake kubectl, tar, sha256sum, shasum, and git placed FIRST on PATH
- bogus VI_CONTEXT / VI_NS / VI_NODE_* so real cluster state is never reachable
- VI_REPO_ROOT pointing to a tiny temp repo (never reading real data/ or .git/)
- network proxies blocked + in-process socket blocked + zero retry/poll sleeps
"""
import json
import os
import socket
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from freestyle.experiments import launch, schema

REAL_K8S_DIR = Path(__file__).resolve().parents[3] / "k8s"
BOGUS_CONTEXT = "bogus-context-do-not-use"
BOGUS_NS = "bogus-ns"
FAKE_NODE_5090 = "fake-node-5090"
FAKE_NODE_6000 = "fake-node-6000"


def _write_exe(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _install_fake_tools(bin_dir: Path, state_file: Path, log_file: Path, work_tmp: Path) -> None:
    py = sys.executable

    # 1. Fake sha256sum / shasum (pure Python hashlib, deterministic)
    sha_code = f"""#!{py}
import hashlib, sys
args = [a for a in sys.argv[1:] if a not in ("-a", "256", "-b", "-t")]
if not args or args == ["-"]:
    data = sys.stdin.buffer.read()
    print(f"{{hashlib.sha256(data).hexdigest()}}  -")
else:
    for p in args:
        with open(p, "rb") as fh:
            h = hashlib.sha256(fh.read()).hexdigest()
        print(f"{{h}}  {{p}}")
"""
    _write_exe(bin_dir / "sha256sum", sha_code)
    _write_exe(bin_dir / "shasum", sha_code)

    # 2. Fake tar (deterministic Python tarfile supporting -C, --null, -T, -cf, -xf, -tf)
    _write_exe(
        bin_dir / "tar",
        f"""#!{py}
import os, sys, tarfile
from pathlib import Path

args = sys.argv[1:]
mode = None
archive = None
chdir = None
null_sep = False
files_from = None
positional = []
i = 0
while i < len(args):
    a = args[i]
    if a == "-C":
        chdir = args[i + 1]; i += 2
    elif a == "--null":
        null_sep = True; i += 1
    elif a == "-T":
        files_from = args[i + 1]; i += 2
    elif a in ("-cf", "-c"):
        mode = "c"
        if a == "-cf":
            archive = args[i + 1]; i += 2
        else:
            i += 1
    elif a in ("-xf", "-x"):
        mode = "x"
        if a == "-xf":
            archive = args[i + 1]; i += 2
        else:
            i += 1
    elif a in ("-tf", "-t"):
        mode = "t"
        if a == "-tf":
            archive = args[i + 1]; i += 2
        else:
            i += 1
    elif a == "-f":
        archive = args[i + 1]; i += 2
    elif a.startswith("-"):
        if "c" in a: mode = "c"
        elif "x" in a: mode = "x"
        elif "t" in a: mode = "t"
        if a.endswith("f"):
            archive = args[i + 1]; i += 2
        else:
            i += 1
    else:
        positional.append(a); i += 1

base = Path(chdir) if chdir else Path.cwd()

def _norm(ti: tarfile.TarInfo) -> tarfile.TarInfo:
    ti.uid = 0; ti.gid = 0; ti.uname = "root"; ti.gname = "root"; ti.mtime = 0
    return ti

if mode == "c":
    members = []
    if files_from == "-":
        raw = sys.stdin.buffer.read()
        members = [x.decode() for x in (raw.split(b"\\0") if null_sep else raw.splitlines()) if x]
    elif files_from:
        raw = Path(files_from).read_bytes()
        members = [x.decode() for x in (raw.split(b"\\0") if null_sep else raw.splitlines()) if x]
    else:
        members = positional
    with tarfile.open(archive, "w") as tf:
        for m in sorted(members):
            p = base / m
            if p.is_dir():
                for sub in sorted(p.rglob("*")):
                    tf.add(sub, arcname=str(sub.relative_to(base)), recursive=False, filter=_norm)
            else:
                tf.add(p, arcname=m, recursive=False, filter=_norm)
elif mode == "x":
    with tarfile.open(archive, "r") as tf:
        tf.extractall(base, filter="data")
elif mode == "t":
    with tarfile.open(archive, "r") as tf:
        for m in tf.getnames():
            print(m)
""",
    )

    # 3. Fake git (deterministic repo metadata + file list; never reads real .git)
    _write_exe(
        bin_dir / "git",
        f"""#!{py}
import sys
args = sys.argv[1:]
if args[:2] == ["rev-parse", "HEAD"]:
    print("0123456789abcdef0123456789abcdef01234567")
elif args and args[0] == "diff":
    sys.exit(0)
elif args and args[0] == "ls-files":
    if "--others" in args:
        sys.exit(0)
    sys.stdout.buffer.write(b"app.py\\0")
else:
    sys.exit(0)
""",
    )

    # 4. Fake kubectl (enforces bogus context & namespace, stores all state in tmp_path)
    _write_exe(
        bin_dir / "kubectl",
        f"""#!{py}
import json, shutil, subprocess, sys
from pathlib import Path

STATE = Path({str(state_file)!r})
LOG = Path({str(log_file)!r})
WORK_TMP = Path({str(work_tmp)!r})
EXPECT_CTX = {BOGUS_CONTEXT!r}
EXPECT_NS = {BOGUS_NS!r}

args = sys.argv[1:]
with LOG.open("a") as fh:
    fh.write(json.dumps(args) + "\\n")

ctx = None
ns = None
i = 0
while i < len(args):
    if args[i] == "--context":
        ctx = args[i + 1]; i += 2
    elif args[i] in ("-n", "--namespace"):
        ns = args[i + 1]; i += 2
    else:
        break

if ctx != EXPECT_CTX or ns != EXPECT_NS:
    sys.stderr.write(f"HERMETIC VIOLATION: unexpected context={{ctx!r}} ns={{ns!r}}\\n")
    sys.exit(99)

cmd = args[i:]
state = json.loads(STATE.read_text())

def save():
    STATE.write_text(json.dumps(state))

def get_jsonpath(obj, expr):
    if expr == "{{.metadata.annotations.hami\\\\.io/node-nvidia-register}}":
        return obj.get("metadata", {{}}).get("annotations", {{}}).get("hami.io/node-nvidia-register", "")
    if expr == "{{.status.phase}}":
        return obj.get("status", {{}}).get("phase", "")
    if expr == "{{.items[0].status.containerStatuses[0].imageID}}":
        items = obj.get("items", [])
        if not items:
            return ""
        cs = items[0].get("status", {{}}).get("containerStatuses", [])
        return cs[0].get("imageID", "") if cs else ""
    return ""

if not cmd:
    sys.exit(1)

verb = cmd[0]
if verb == "get":
    out_fmt = ""
    if "-o" in cmd:
        out_fmt = cmd[cmd.index("-o") + 1]
    kind = cmd[1]
    if kind == "pods":
        if "-A" in cmd:
            if "--field-selector" in cmd:
                fs = cmd[cmd.index("--field-selector") + 1]
                node = fs.split("spec.nodeName=")[-1]
                for p in state.get("cluster_pods", []):
                    if p.get("node") == node:
                        print(f"default {{p['name']}} 1/1 Running 0 1h 10.0.0.1 {{node}}")
                sys.exit(0)
            print(json.dumps({{"items": state.get("hami_pods", [])}}))
            sys.exit(0)
        if "-l" in cmd:
            lbl = cmd[cmd.index("-l") + 1]
            items = [p for p in state.get("job_pods", []) if p.get("label") == lbl]
            if out_fmt.startswith("jsonpath="):
                print(get_jsonpath({{"items": items}}, out_fmt[len("jsonpath="):]), end="")
            else:
                print(json.dumps({{"items": items}}))
            sys.exit(0)
    name = cmd[2] if len(cmd) > 2 and not cmd[2].startswith("-") else ""
    store = state.get(kind + "s", {{}})
    if name not in store:
        sys.stderr.write(f"Error from server (NotFound): {{kind}} {{name}} not found\\n")
        sys.exit(1)
    obj = store[name]
    if kind == "job" and state.get("poll_sequence"):
        idx = state.get("poll_idx", 0)
        seq = state["poll_sequence"]
        conds = seq[min(idx, len(seq) - 1)]
        state["poll_idx"] = idx + 1
        save()
        obj = json.loads(json.dumps(obj))
        obj["status"] = {{"conditions": conds}}
    if out_fmt == "json":
        print(json.dumps(obj))
    elif out_fmt.startswith("jsonpath="):
        print(get_jsonpath(obj, out_fmt[len("jsonpath="):]), end="")
    else:
        print(name)
    sys.exit(0)

if verb == "exec":
    sep = cmd.index("--")
    pod = cmd[1]
    ecmd = cmd[sep + 1:]
    if pod.startswith("gpu-vram-exporter"):
        val = state.get("phys_free", "")
        if val == "":
            sys.exit(1)
        print(val)
        sys.exit(0)
    root = Path(state["pod_fs"][pod])
    root.mkdir(parents=True, exist_ok=True)
    if ecmd[0] == "mkdir":
        for d in ecmd[2:]:
            (root / d.lstrip("/")).mkdir(parents=True, exist_ok=True)
        sys.exit(0)
    if ecmd[0] == "mv":
        src = root / ecmd[-2].lstrip("/")
        dst = root / ecmd[-1].lstrip("/")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        sys.exit(0)
    if ecmd[0] == "rm":
        for f in ecmd[2:]:
            p = root / f.lstrip("/")
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            elif p.exists():
                p.unlink()
        sys.exit(0)
    if ecmd[0] == "sh" and ecmd[1] == "-c":
        sh_cmd = ecmd[2].replace("/work", str(root / "work"))
        res = subprocess.run(["sh", "-c", sh_cmd], capture_output=True, text=True)
        sys.stdout.write(res.stdout)
        sys.stderr.write(res.stderr)
        sys.exit(res.returncode)
    sys.exit(0)

if verb == "cp":
    src, dst = cmd[1], cmd[2]
    if state.get("fail_cp"):
        sys.stderr.write("cp error\\n")
        sys.exit(1)
    if ":" in dst and ":" not in src:
        pod, rpath = dst.split(":", 1)
        target = Path(state["pod_fs"][pod]) / rpath.lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
    elif ":" in src and ":" not in dst:
        pod, rpath = src.split(":", 1)
        source = Path(state["pod_fs"][pod]) / rpath.lstrip("/")
        Path(dst).mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, Path(dst) / source.name)
        else:
            shutil.copy2(source, dst)
    sys.exit(0)

if verb in ("create", "apply"):
    import yaml as _yaml
    raw = sys.stdin.read()
    obj = _yaml.safe_load(raw)
    if "--dry-run=server" in cmd:
        sys.exit(0)
    kind = obj["kind"].lower()
    name = obj["metadata"]["name"]
    store = state.setdefault(kind + "s", {{}})
    if verb == "create" and name in store:
        sys.stderr.write(f"Error from server (AlreadyExists): {{kind}} {{name}} already exists\\n")
        sys.exit(1)
    obj.setdefault("metadata", {{}}).setdefault("uid", f"uid-{{name}}")
    if kind == "pod":
        obj["status"] = {{"phase": "Running"}}
        pod_dir = WORK_TMP / ("fs_" + name)
        pod_dir.mkdir(parents=True, exist_ok=True)
        state.setdefault("pod_fs", {{}}).setdefault(name, str(pod_dir))
    store[name] = obj
    save()
    sys.exit(0)

if verb == "delete":
    kind, name = cmd[1], cmd[2]
    store = state.get(kind + "s", {{}})
    if name not in store:
        sys.exit(1)
    del store[name]
    save()
    sys.exit(0)

if verb == "wait":
    sys.exit(0)

if verb == "logs":
    print(state.get("job_logs", "line1\\nline2"))
    sys.exit(0)

sys.exit(0)
""",
    )


def _default_state(tmp_path: Path, hami_free=0, phys_free="1500") -> dict:
    fs_ws = tmp_path / "fs_ws"
    fs_out = tmp_path / "fs_out"
    (fs_ws / "work" / "data").mkdir(parents=True, exist_ok=True)
    (fs_ws / "work" / "data" / "sample.txt").write_text("hi")
    (fs_out / "work" / "repo").mkdir(parents=True, exist_ok=True)
    (fs_out / "work" / "jobs").mkdir(parents=True, exist_ok=True)
    (fs_out / "work" / "artifacts").mkdir(parents=True, exist_ok=True)
    hami_used = 32607 - hami_free
    return {
        "phys_free": str(phys_free),
        "cluster_pods": [{"name": "gpu-vram-exporter-fake", "node": FAKE_NODE_5090}],
        "hami_pods": [
            {
                "status": {"phase": "Running"},
                "metadata": {"annotations": {"hami.io/vgpu-devices-allocated": f"GPU-0,RTX 5090,{hami_used},100"}},
            }
        ],
        "nodes": {
            FAKE_NODE_5090: {
                "metadata": {
                    "annotations": {
                        "hami.io/node-nvidia-register": json.dumps([{"id": "GPU-0", "type": "FAKE RTX 5090", "devmem": 32607}])
                    }
                }
            }
        },
        "secrets": {"hf-token": {"metadata": {"name": "hf-token"}}},
        "pvcs": {
            "vision-intern-data-5090": {
                "metadata": {
                    "name": "vision-intern-data-5090",
                    "labels": {"app": "vision-intern", "app.kubernetes.io/component": "workdir"},
                }
            },
            "vision-intern-out-5090": {
                "metadata": {
                    "name": "vision-intern-out-5090",
                    "labels": {"app": "vision-intern", "app.kubernetes.io/component": "workdir"},
                    "annotations": {"vision-intern/owner": "vision-intern/test"},
                }
            },
        },
        "pods": {
            "vision-intern-ws-5090": {
                "metadata": {
                    "name": "vision-intern-ws-5090",
                    "uid": "uid-ws-5090",
                    "labels": {
                        "app": "vision-intern",
                        "app.kubernetes.io/component": "workspace",
                        "vision-intern/gpu": "5090",
                    },
                },
                "spec": {
                    "nodeSelector": {"kubernetes.io/hostname": FAKE_NODE_5090},
                    "volumes": [{"persistentVolumeClaim": {"claimName": "vision-intern-data-5090"}}],
                },
                "status": {"phase": "Running"},
            },
            "vision-intern-out-ws-5090": {
                "metadata": {
                    "name": "vision-intern-out-ws-5090",
                    "uid": "uid-out-ws-5090",
                    "labels": {
                        "app": "vision-intern",
                        "app.kubernetes.io/component": "workspace",
                        "vision-intern/gpu": "5090",
                    },
                    "annotations": {"vision-intern/owner": "vision-intern/test"},
                },
                "spec": {
                    "nodeSelector": {"kubernetes.io/hostname": FAKE_NODE_5090},
                    "volumes": [{"persistentVolumeClaim": {"claimName": "vision-intern-out-5090"}}],
                },
                "status": {"phase": "Running"},
            },
        },
        "jobs": {},
        "pod_fs": {
            "vision-intern-ws-5090": str(fs_ws),
            "vision-intern-out-ws-5090": str(fs_out),
        },
    }


@pytest.fixture
def k8s_harness(tmp_path, monkeypatch):
    """Provide a fully hermetic fake-tool + isolated-repo harness for k8s/*.sh."""
    def _blocked_socket(*args, **kwargs):
        raise RuntimeError("network blocked in hermetic k8s test")

    monkeypatch.setattr(socket, "create_connection", _blocked_socket)
    monkeypatch.setattr(socket.socket, "connect", _blocked_socket)

    fake_repo = tmp_path / "repo"
    fake_repo.mkdir()
    (fake_repo / "k8s").symlink_to(REAL_K8S_DIR)
    (fake_repo / "app.py").write_text("print('hermetic')\n")
    (fake_repo / "data").mkdir()
    (fake_repo / "data" / "sample.txt").write_text("fake-data\n")

    def build(name="default", hami_free=0, phys_free="1500", mutator=None):
        root = tmp_path / name
        root.mkdir(parents=True, exist_ok=True)
        bin_dir = root / "bin"
        bin_dir.mkdir()
        state_file = root / "state.json"
        log_file = root / "kubectl.log"
        state = _default_state(root, hami_free=hami_free, phys_free=phys_free)
        if mutator:
            mutator(state, root)
        state_file.write_text(json.dumps(state))
        _install_fake_tools(bin_dir, state_file, log_file, root)

        env = os.environ.copy()
        env["PATH"] = f"{bin_dir}:{env['PATH']}"
        env["VI_CONTEXT"] = BOGUS_CONTEXT
        env["VI_NS"] = BOGUS_NS
        env["VI_NODE_5090"] = FAKE_NODE_5090
        env["VI_NODE_6000"] = FAKE_NODE_6000
        env["VI_REPO_ROOT"] = str(fake_repo)
        env["VI_WAIT_POLL_SECS"] = "0"
        env["VI_RETRY_SLEEP_SECS"] = "0"
        # Block network egress from child processes.
        env["HTTP_PROXY"] = "http://127.0.0.1:0"
        env["HTTPS_PROXY"] = "http://127.0.0.1:0"
        env["ALL_PROXY"] = "http://127.0.0.1:0"
        env["NO_PROXY"] = ""
        env["HF_HUB_OFFLINE"] = "1"
        env["PIP_NO_INDEX"] = "1"
        return {
            "root": root,
            "repo": fake_repo,
            "env": env,
            "state_file": state_file,
            "log_file": log_file,
            "read_state": lambda: json.loads(state_file.read_text()),
            "write_state": lambda s: state_file.write_text(json.dumps(s)),
            "calls": lambda: [json.loads(x) for x in log_file.read_text().splitlines() if x.strip()]
            if log_file.exists()
            else [],
            "run": lambda args: subprocess.run(
                args, cwd=fake_repo, env=env, capture_output=True, text=True
            ),
        }

    return build


def test_dry_run_never_mutates_and_replace_requires_ownership(k8s_harness):
    def add_jobs(state, _root):
        state["jobs"]["owned-job"] = {
            "metadata": {
                "name": "owned-job",
                "uid": "uid-owned",
                "labels": {"app": "vision-intern"},
                "annotations": {"vision-intern/owner": "vision-intern/tester"},
            }
        }
        state["jobs"]["foreign-job"] = {
            "metadata": {
                "name": "foreign-job",
                "uid": "uid-foreign",
                "labels": {"app": "other-app"},
                "annotations": {"vision-intern/owner": "other-owner"},
            }
        }

    h = k8s_harness(mutator=add_jobs)

    res = h["run"](["bash", "k8s/submit.sh", "owned-job", "5090", "--replace", "--dry-run", "--mode", "none", "--", "true"])
    assert res.returncode == 0, res.stderr
    assert not any("delete" in c or "apply" in c or "create" in c for c in h["calls"]())
    assert yaml.safe_load(res.stdout)["metadata"]["name"] == "owned-job"

    for extra_flag in (["--dry-run"], []):
        h["log_file"].unlink()
        res2 = h["run"](["bash", "k8s/submit.sh", "foreign-job", "5090", "--replace", *extra_flag, "--mode", "none", "--", "true"])
        assert res2.returncode != 0
        assert "refusing to delete" in res2.stderr
        assert not any("delete" in c for c in h["calls"]())

    h["log_file"].unlink()
    res3 = h["run"](["bash", "k8s/submit.sh", "owned-job", "5090", "--replace", "--mode", "none", "--", "true"])
    assert res3.returncode == 0, res3.stderr
    verbs = [c[4] for c in h["calls"]() if len(c) > 4]
    assert "delete" in verbs and "create" in verbs and "apply" not in verbs


def test_auto_fails_closed_and_share_mode_enforces_limit(k8s_harness):
    h = k8s_harness("room", hami_free=0, phys_free="1500")

    res = h["run"](["bash", "k8s/submit.sh", "train-job", "5090", "--mem", "512", "--", "python", "train.py"])
    assert res.returncode != 0
    assert "auto mode requires HAMi free" in res.stderr

    res_small = h["run"](["bash", "k8s/submit.sh", "train-job", "5090", "--mode", "share", "--mem", "1000", "--dry-run", "--", "python", "train.py"])
    assert res_small.returncode != 0
    assert "share mode refuses" in res_small.stderr

    h_miss = k8s_harness("missing", hami_free=0, phys_free="")
    res_miss = h_miss["run"](["bash", "k8s/submit.sh", "train-job", "5090", "--mode", "share", "--mem", "512", "--dry-run", "--", "python", "train.py"])
    assert res_miss.returncode != 0
    assert "unavailable" in res_miss.stderr

    res_ok = h["run"](["bash", "k8s/submit.sh", "train-job", "5090", "--mode", "share", "--mem", "512", "--dry-run", "--", "python", "train.py"])
    assert res_ok.returncode == 0, res_ok.stderr
    doc = yaml.safe_load(res_ok.stdout)
    c = doc["spec"]["template"]["spec"]["containers"][0]
    env_map = {e["name"]: e.get("value") for e in c["env"]}
    assert env_map["PYTHONPATH"] == "/_vi_site:/src"
    assert "expandable_segments:True" in env_map["PYTORCH_CUDA_ALLOC_CONF"]
    assert "set_per_process_memory_fraction" in c["args"][0]
    assert "sitecustomize.py" in c["args"][0]


def test_wait_returns_on_failed_and_collects_logs_and_out(k8s_harness):
    def set_fail(state, _root):
        state["poll_sequence"] = [
            [{"type": "Failed", "status": "True", "reason": "DeadlineExceeded", "message": "boom"}],
        ]
        state["job_logs"] = "train step 1\ncrash traceback\n"

    h = k8s_harness(mutator=set_fail)
    collect_dir = h["root"] / "collected"

    res = h["run"]([
        "bash", "k8s/submit.sh", "fail-job", "5090", "--mode", "none",
        "--wait", "--wait-for", "60", "--collect-dir", str(collect_dir), "--", "false",
    ])
    assert res.returncode == 1
    assert "Failed" in res.stderr
    assert (collect_dir / "job.log").read_text() == "train step 1\ncrash traceback\n"

    state2 = h["read_state"]()
    run_id = state2["jobs"]["fail-job"]["metadata"]["annotations"]["vision-intern/run-id"]
    art_dir = Path(state2["pod_fs"]["vision-intern-out-ws-5090"]) / "work" / "artifacts" / run_id
    art_dir.mkdir(parents=True, exist_ok=True)
    (art_dir / "partial.txt").write_text("partial-output")
    (art_dir / "env.txt").write_text("image=test\n")
    state2["job_pods"] = [
        {
            "label": "vision-intern/job=fail-job",
            "status": {"containerStatuses": [{"imageID": "docker.io/pytorch/pytorch@sha256:deadbeef"}]},
        }
    ]
    h["write_state"](state2)

    dest2 = h["root"] / "collected2"
    res_col = h["run"](["bash", "k8s/collect.sh", "fail-job", str(dest2)])
    assert res_col.returncode == 0, res_col.stderr
    assert (dest2 / "partial.txt").read_text() == "partial-output"
    assert "pod_image_id=docker.io/pytorch/pytorch@sha256:deadbeef" in (dest2 / "env.txt").read_text()

    res_col2 = h["run"](["bash", "k8s/collect.sh", "fail-job", str(dest2)])
    assert res_col2.returncode != 0
    assert "non-empty" in res_col2.stderr


def test_foreign_pvc_and_workspace_refused(k8s_harness):
    def bad_pod(state, _root):
        state["pods"]["vision-intern-ws-5090"]["metadata"]["labels"]["app"] = "someone-else"

    h1 = k8s_harness("bad_pod", mutator=bad_pod)
    res = h1["run"](["bash", "k8s/submit.sh", "job-a", "5090", "--mode", "none", "--", "true"])
    assert res.returncode != 0
    assert "not an owned workspace pod" in res.stderr
    assert not any("delete" in c for c in h1["calls"]())

    def bad_pvc(state, _root):
        state["pvcs"]["vision-intern-data-5090"]["metadata"]["labels"]["app"] = "foreign"

    h2 = k8s_harness("bad_pvc", mutator=bad_pvc)
    res2 = h2["run"](["bash", "k8s/submit.sh", "job-a", "5090", "--mode", "none", "--", "true"])
    assert res2.returncode != 0
    assert "lacks vision-intern ownership labels" in res2.stderr


def test_hf_token_secret_mounting_and_volumes_and_nocode(k8s_harness):
    h = k8s_harness()

    res_no = h["run"](["bash", "k8s/submit.sh", "job-no-hf", "5090", "--mode", "none", "--dry-run", "--", "true"])
    assert res_no.returncode == 0, res_no.stderr
    doc_no = yaml.safe_load(res_no.stdout)
    c_no = doc_no["spec"]["template"]["spec"]["containers"][0]
    assert all(e["name"] != "HF_TOKEN" for e in c_no["env"])

    mounts = {m["name"]: m for m in c_no["volumeMounts"]}
    vols = {v["name"]: v for v in doc_no["spec"]["template"]["spec"]["volumes"]}
    assert mounts["data"]["readOnly"] is True
    assert mounts["data"]["mountPath"] == "/data-ro"
    assert mounts["outputs"].get("readOnly") is not True
    assert vols["data"]["persistentVolumeClaim"]["claimName"] == "vision-intern-data-5090"
    assert vols["outputs"]["persistentVolumeClaim"]["claimName"] == "vision-intern-out-5090"
    assert "@sha256:" in c_no["image"]

    res_hf = h["run"](["bash", "k8s/submit.sh", "job-hf", "5090", "--mode", "none", "--hf-token", "--no-code", "--dry-run", "--", "true"])
    assert res_hf.returncode == 0, res_hf.stderr
    doc_hf = yaml.safe_load(res_hf.stdout)
    c_hf = doc_hf["spec"]["template"]["spec"]["containers"][0]
    hf_envs = [e for e in c_hf["env"] if e["name"] == "HF_TOKEN"]
    assert hf_envs == [{"name": "HF_TOKEN", "valueFrom": {"secretKeyRef": {"name": "hf-token", "key": "HF_TOKEN"}}}]
    assert "/outputs/repo/repo.tar" in c_hf["args"][0]


def test_name_and_flag_validation_and_provenance(k8s_harness):
    h = k8s_harness()

    for bad_name in ("BadName", "has.dot", "-leading", "trailing-", "a" * 58, "under_score"):
        res = h["run"](["bash", "k8s/submit.sh", bad_name, "5090", "--mode", "none", "--dry-run", "--", "true"])
        assert res.returncode != 0, f"expected {bad_name!r} to be rejected"

    for bad_pip in ("foo; id", "foo && ls", "--index-url http://bad"):
        res = h["run"](["bash", "k8s/submit.sh", "good-name", "5090", "--mode", "none", "--pip", bad_pip, "--dry-run", "--", "true"])
        assert res.returncode != 0, f"expected pip {bad_pip!r} to be rejected"

    res_ok = h["run"]([
        "bash", "k8s/submit.sh", "good-name", "5090", "--mode", "none",
        "--pip", "pycocotools scipy", "--pip-constraints", "pycocotools==2.0.11 scipy==1.18.0",
        "--", "python", "-c", "print(1)",
    ])
    assert res_ok.returncode == 0, res_ok.stderr
    state_after = h["read_state"]()
    run_id = state_after["jobs"]["good-name"]["metadata"]["annotations"]["vision-intern/run-id"]
    out_root = Path(state_after["pod_fs"]["vision-intern-out-ws-5090"]) / "work"
    prov = json.loads((out_root / "jobs" / f"{run_id}.provenance.json").read_text())
    assert prov["git_sha"] == "0123456789abcdef0123456789abcdef01234567"
    assert len(prov["repo_tar_sha256"]) == 64
    assert (out_root / "jobs" / f"{run_id}.requirements.txt").read_text().splitlines() == ["pycocotools", "scipy"]
    assert (out_root / "jobs" / f"{run_id}.constraints.txt").read_text().splitlines() == ["pycocotools==2.0.11", "scipy==1.18.0"]


def test_stage_data_hermetic(k8s_harness):
    h = k8s_harness()
    res = h["run"](["bash", "k8s/stage-data.sh", "--gpu", "5090"])
    assert res.returncode == 0, res.stderr
    ws_data = Path(h["read_state"]()["pod_fs"]["vision-intern-ws-5090"]) / "work" / "data" / "sample.txt"
    out_repo = Path(h["read_state"]()["pod_fs"]["vision-intern-out-ws-5090"]) / "work" / "repo"
    assert ws_data.read_text() == "fake-data\n"
    assert (out_repo / "repo.tar").exists()
    meta = json.loads((out_repo / "repo.tar.meta.json").read_text())
    assert len(meta["repo_tar_sha256"]) == 64


def test_launch_hf_token_hami_default_and_failure_collect(k8s_harness):
    _ = k8s_harness()
    c = schema.resolve({
        "id": "B2_test.run",
        "family": "B_zeroshot_plus_embed",
        "compute": {"target": "k8s:5090"},
        "localizer": {"type": "zeroshot", "backend": "sam3"},
        "needs_hf_token": True,
    })
    cmd = launch.submit_argv(c, "configs/x.yaml", "B2_test.run-12345678", dry_run=True)
    assert cmd[cmd.index("--mode") + 1] == "hami"
    assert "--hf-token" in cmd
    assert "--pip-constraints" in cmd
    assert "." not in cmd[1] and len(cmd[1]) <= 57
    assert launch.normalize_job_name("a.b_" + "x" * 60) != launch.normalize_job_name("a-b-" + "x" * 60)


def test_sitecustomize_enforces_share_memory_fraction(k8s_harness):
    h = k8s_harness("share", hami_free=0, phys_free="1500")
    res = h["run"](["bash", "k8s/submit.sh", "share-test", "5090", "--mode", "share", "--mem", "512", "--dry-run", "--", "python", "-c", "pass"])
    assert res.returncode == 0, res.stderr
    doc = yaml.safe_load(res.stdout)
    args_src = doc["spec"]["template"]["spec"]["containers"][0]["args"][0]
    marker = "cat <<'PYEOF' > /_vi_site/sitecustomize.py\n"
    start = args_src.index(marker) + len(marker)
    end = args_src.index("\nPYEOF\n", start)
    site_dir = h["root"] / "site"
    site_dir.mkdir()
    (site_dir / "sitecustomize.py").write_text(args_src[start:end])

    fake_torch_dir = h["root"] / "fake_torch"
    fake_torch_dir.mkdir()
    (fake_torch_dir / "torch.py").write_text(
        """class _Props:
    total_memory = 1024 * 1024 * 1024
class cuda:
    calls = []
    @staticmethod
    def is_available(): return True
    @staticmethod
    def device_count(): return 1
    @staticmethod
    def get_device_properties(idx): return _Props()
    @staticmethod
    def set_per_process_memory_fraction(frac, idx=0): cuda.calls.append((frac, idx))
"""
    )
    env = dict(h["env"])
    env["PYTHONPATH"] = f"{site_dir}:{fake_torch_dir}"
    env["VI_GPU_MODE"] = "share"
    env["VI_GPU_MEM_MB"] = "512"
    check = subprocess.run(
        [sys.executable, "-c", "import torch; torch.cuda.set_per_process_memory_fraction(0.99, 0); print(torch.cuda.calls)"],
        env=env, capture_output=True, text=True,
    )
    assert check.returncode == 0, check.stderr
    assert "[(0.5, 0), (0.5, 0)]" in check.stdout


def test_launch_collects_logs_and_out_on_submit_failure(k8s_harness, monkeypatch):
    h = k8s_harness()
    cfg = h["root"] / "exp.yaml"
    cfg.write_text(yaml.safe_dump({
        "id": "C1_fail_test",
        "family": "C_det_agnostic_plus_embed",
        "compute": {"target": "k8s:5090", "mode": "none", "vram_mb": 1024, "timeout": 60},
        "localizer": {"type": "zeroshot", "backend": "owlv2"},
    }))
    monkeypatch.chdir(h["root"])
    monkeypatch.setenv("VI_CONTEXT", BOGUS_CONTEXT)
    monkeypatch.setenv("VI_NS", BOGUS_NS)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[0] == "k8s/submit.sh":
            return subprocess.CompletedProcess(cmd, 1)
        if cmd[0] == "kubectl" and "logs" in cmd:
            assert cmd[cmd.index("--context") + 1] == BOGUS_CONTEXT
            return subprocess.CompletedProcess(cmd, 0, stdout="pod failure log\n", stderr="")
        if cmd[0] == "k8s/collect.sh":
            dest = Path(cmd[2])
            (dest / "partial.txt").write_text("saved")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(launch.subprocess, "run", fake_run)
    monkeypatch.setattr(launch.schema, "git_sha", lambda: "a" * 40)
    monkeypatch.setattr(launch.schema, "git_state", lambda: {"dirty": False, "diff_hash": None})

    with pytest.raises(subprocess.CalledProcessError):
        launch.main([str(cfg)])

    run_dirs = list((h["root"] / "runs").iterdir())
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "job.log").read_text() == "pod failure log\n"
    assert (run_dirs[0] / "partial.txt").read_text() == "saved"
    assert any(c[0] == "k8s/collect.sh" for c in calls)
