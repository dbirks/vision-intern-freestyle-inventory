"""Resolve and launch Freestyle experiments."""
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

from freestyle.experiments import schema


def submit_argv(c, config_path, name, dry_run=False, sha=None, state=None):
    gpu = c['compute']['target'].split(':', 1)[1]
    specs = ['transformers==5.24.1', 'pycocotools', 'scipy']
    if c['family'].startswith(('C_', 'D_', 'E_', 'F_')):
        specs.append('rfdetr==1.12.1')
    argv = ['k8s/submit.sh', name.lower().replace('_', '-'), gpu,
            '--mem', str(c['compute']['vram_mb']), '--timeout', str(c['compute']['timeout']),
            '--pip', ' '.join(specs), '--wait', '--wait-for', str(c['compute']['timeout'])]
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
    subprocess.run([sys.executable, '-m', 'freestyle.eval.report', '--runs-dir', 'runs',
                    '--out', 'docs/freestyle/RESULTS.md'], check=True)


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
    run = schema.run_dir(c, sha, state=state)
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
    schema.create_run_dir(c, sha, state=state)
    (run / 'resolved.yaml').write_text(yaml.safe_dump(c, sort_keys=False))
    (run / 'git_sha').write_text(sha + '\n')
    (run / 'git_state.json').write_text(json.dumps(state, indent=2) + '\n')
    subprocess.run(cmd, check=True)
    if c['compute']['target'] != 'local':
        # collect.sh nests the remote directory when the destination already contains files.
        import tempfile
        import shutil
        with tempfile.TemporaryDirectory() as temp:
            subprocess.run(['k8s/collect.sh', cmd[1], temp], check=True)
            for item in Path(temp).iterdir():
                shutil.move(str(item), run / item.name)
    evaluate(c, run)


if __name__ == '__main__':
    main()
