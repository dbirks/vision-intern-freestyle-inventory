"""Resolve and launch Freestyle experiments."""
import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from freestyle.experiments import schema

_PINNED_CONSTRAINTS = ('pycocotools==2.0.11', 'scipy==1.18.0')


def normalize_job_name(name):
    """Normalize an experiment/run name to a collision-free DNS-1123 label <= 57 chars."""
    slug = re.sub(r'[^a-z0-9-]+', '-', name.lower()).strip('-')
    if not slug:
        slug = 'job'
    if slug == name and len(slug) <= 57:
        return slug
    if len(slug) <= 57 and re.fullmatch(r'[a-z0-9-]+', name.lower().replace('_', '-')):
        # Keep simple lowercase/underscore replacements intact when already unambiguous.
        if '.' not in name and not name.startswith('-') and not name.endswith('-'):
            return slug
    suffix = hashlib.sha256(name.encode()).hexdigest()[:8]
    prefix = slug[:57 - 9].rstrip('-') or 'job'
    return f'{prefix}-{suffix}'


def submit_argv(c, config_path, name, dry_run=False, sha=None, state=None):
    gpu = c['compute']['target'].split(':', 1)[1]
    specs = ['transformers==5.24.1', 'pycocotools', 'scipy']
    if c['family'].startswith(('C_', 'D_', 'E_', 'F_')):
        specs.append('rfdetr==1.12.1')
    needs_hf_token = c.get('needs_hf_token', False)
    if not isinstance(needs_hf_token, bool):
        raise ValueError('needs_hf_token must be a boolean')
    mode = c['compute']['mode']
    if mode == 'auto':
        mode = 'hami'
    wait_for = c['compute']['timeout'] + 300
    argv = ['k8s/submit.sh', normalize_job_name(name), gpu,
            '--mem', str(c['compute']['vram_mb']), '--timeout', str(c['compute']['timeout']),
            '--pip', ' '.join(specs), '--wait', '--wait-for', str(wait_for),
            '--mode', mode,
            '--pip-constraints', ' '.join(_PINNED_CONSTRAINTS)]
    if needs_hf_token:
        argv.append('--hf-token')
    if sha:
        argv += ['--expect-git-sha', sha]
    if state is not None:
        argv += ['--expect-git-dirty', str(state['dirty']).lower()]
        if state['diff_hash']:
            argv += ['--expect-git-diff-hash', state['diff_hash']]
    if dry_run:
        argv.append('--dry-run')
    argv += ['--', 'python', '-m', 'freestyle.experiments.entry', '--config', str(config_path),
             '--data-root', '/work/data', '--out', '/out']
    if sha:
        argv += ['--git-sha', sha]
    if state is not None:
        argv += ['--git-dirty', str(state['dirty']).lower()]
        if state['diff_hash']:
            argv += ['--git-diff-hash', state['diff_hash']]
    return argv


def evaluate(c, run):
    data = Path('data')
    pred = run / 'test' / 'predictions.json'
    val_pred = run / 'val' / 'predictions.json'
    command = [sys.executable, '-m', 'freestyle.eval.evaluate', '--gt', str(data / c['data']['test']),
               '--pred', str(pred), '--select-thr-on', str(data / c['data']['val']), str(val_pred),
               '--split-map', str(data / c['data']['splits']), '--out', str(run / 'metrics.json'),
               '--seed', str(c['seed']), '--experiment', c['id']]
    subprocess.run(command, check=True)
    threshold = json.loads((run / 'metrics.json').read_text())['score_thr']
    subprocess.run([sys.executable, '-m', 'freestyle.eval.failures', '--gt', str(data / c['data']['test']),
                    '--pred', str(pred), '--images-root', str(data / c['data']['images_root']),
                    '--score-thr', str(threshold), '--out', str(run / 'failures')], check=True)
    # A chain-check run is not a reported experiment; it must not regenerate (or clobber a
    # concurrent worker's) shared RESULTS.md. metrics.json + failures/ above are the point.
    if not c.get('chain_check'):
        subprocess.run([sys.executable, '-m', 'freestyle.eval.report', '--runs-dir', 'runs',
                        '--out', 'docs/freestyle/RESULTS.md'], check=True)


def _collect_k8s_run(job_name, run):
    ctx = os.environ.get('VI_CONTEXT', 'admin@home')
    ns = os.environ.get('VI_NS', 'default')
    log_proc = subprocess.run(
        ['kubectl', '--context', ctx, '-n', ns, 'logs', f'job/{job_name}', '--all-pods=true'],
        capture_output=True, text=True, check=False,
    )
    if log_proc.stdout or log_proc.stderr:
        (run / 'job.log').write_text(log_proc.stdout + log_proc.stderr)
    with tempfile.TemporaryDirectory() as temp:
        collect_proc = subprocess.run(['k8s/collect.sh', job_name, temp], check=False)
        for item in Path(temp).iterdir():
            shutil.move(str(item), run / item.name)
        return collect_proc.returncode


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config', nargs='?')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--status', action='store_true')
    args = p.parse_args(argv)
    if args.status:
        for path in sorted(Path('runs').glob('*/resolved.yaml')):
            stages = sorted(x.stem for x in (path.parent / 'stages').glob('*.done'))
            print(path.parent.name, 'complete' if (path.parent / 'metrics.json').exists() else ','.join(stages) or 'pending')
        return
    if not args.config:
        p.error('config is required')
    c = schema.load(args.config)
    sha = schema.git_sha()
    state = schema.git_state()
    # A chain-check run is plumbing validation on CPU with a stand-in GT, not a real
    # experiment: keep it under runs/chain-check/ so `freestyle.eval.report`'s
    # `runs/*/metrics.json` glob (one level) never lists it in RESULTS.md.
    run_root = 'runs/chain-check' if c['chain_check'] else 'runs'
    run = schema.run_dir(c, sha, run_root, state=state)
    if c['compute']['target'] == 'local':
        cmd = [sys.executable, '-m', 'freestyle.experiments.entry', '--config', str(run / 'resolved.yaml'),
               '--data-root', 'data', '--out', str(run), '--git-sha', sha]
    else:
        cmd = submit_argv(c, args.config, run.name, args.dry_run, sha, state)
    if c['compute']['target'] == 'local':
        cmd += ['--git-dirty', str(state['dirty']).lower()]
        if state['diff_hash']:
            cmd += ['--git-diff-hash', state['diff_hash']]
    if args.dry_run:
        print(shlex.join(cmd))
        return
    schema.create_run_dir(c, sha, run_root, state=state)
    (run / 'resolved.yaml').write_text(yaml.safe_dump(c, sort_keys=False))
    (run / 'git_sha').write_text(sha + '\n')
    (run / 'git_state.json').write_text(json.dumps(state, indent=2) + '\n')
    if c['compute']['target'] == 'local':
        subprocess.run(cmd, check=True)
    else:
        env = {**os.environ, 'VI_WAIT_COLLECT': '0'}
        proc = subprocess.run(cmd, check=False, env=env)
        collect_rc = _collect_k8s_run(cmd[1], run)
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)
        if collect_rc != 0:
            raise subprocess.CalledProcessError(collect_rc, ['k8s/collect.sh', cmd[1]])
    evaluate(c, run)


if __name__ == '__main__':
    main()
