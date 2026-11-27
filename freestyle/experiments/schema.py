"""Validation and stable identities for experiment configurations."""
import copy
import hashlib
import json
import re
import subprocess
from pathlib import Path

import yaml

FAMILIES = {'A_vlm', 'B_zeroshot_plus_embed', 'C_det_agnostic_plus_embed',
            'D_det_multiclass', 'E_shelf', 'F_det_plus_vlm_id'}


def resolve(config):
    c = copy.deepcopy(config)
    if not isinstance(c, dict):
        raise ValueError('experiment must be a mapping')
    if not isinstance(c.get('id'), str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', c['id']):
        raise ValueError('id must be a safe nonempty name')
    if c.get('family') not in FAMILIES:
        raise ValueError('unknown family')
    for key in ('compute', 'data', 'localizer', 'identifier', 'count', 'eval'):
        c.setdefault(key, {})
        if not isinstance(c[key], dict):
            raise ValueError(f'{key} must be a mapping')
    compute = c['compute']
    compute.setdefault('target', 'local')
    compute.setdefault('vram_mb', 8192)
    compute.setdefault('timeout', 3600)
    # k8s submit mode: how the GPU (or its absence) is accounted for. 'none' runs the
    # whole chain CPU-only (no card touched) to de-risk a future GPU window.
    compute.setdefault('mode', 'auto')
    if compute['target'] not in ('local', 'k8s:5090', 'k8s:6000'):
        raise ValueError('invalid compute target')
    if compute['mode'] not in ('auto', 'hami', 'share', 'none'):
        raise ValueError('invalid compute mode')
    for key in ('vram_mb', 'timeout'):
        if not isinstance(compute[key], int) or isinstance(compute[key], bool) or compute[key] <= 0:
            raise ValueError(f'compute.{key} must be a positive integer')
    # chain_check: a plumbing smoke run (CPU, no human GT) that must never be reported
    # as a real experiment result.
    c.setdefault('chain_check', False)
    if not isinstance(c['chain_check'], bool):
        raise ValueError('chain_check must be a boolean')
    data = c['data']
    for key, value in {'train': 'gt/train.json', 'val': 'gt/val.json', 'test': 'gt/test.json',
                       'splits': 'splits.json', 'images_root': 'raw/freestyle-vision-data/images',
                       'label_source': 'gt'}.items():
        data.setdefault(key, value)
    if data['label_source'] not in ('gt', 'bootstrap', 'bootstrap+gt'):
        raise ValueError('invalid label source')
    if data['label_source'] != 'gt':
        if 'bootstrap' not in data:
            raise ValueError('data.bootstrap is required for bootstrap labels')
        data.setdefault('bootstrap_filter', {})
        filt = data['bootstrap_filter']
        if not isinstance(filt, dict):
            raise ValueError('data.bootstrap_filter must be a mapping')
        filt.setdefault('verdicts', ['correct'])
        if not isinstance(filt['verdicts'], list) or not filt['verdicts'] or not all(isinstance(x, str) for x in filt['verdicts']):
            raise ValueError('bootstrap verdicts must be a nonempty list')
        for key in ('min_judge_score', 'min_support', 'min_id_confidence'):
            filt.setdefault(key, 0.0)
            if isinstance(filt[key], bool) or not isinstance(filt[key], (int, float)) or not 0 <= filt[key] <= 1:
                raise ValueError(f'invalid bootstrap threshold {key}')
        if not isinstance(data['bootstrap'], str) or not data['bootstrap'].startswith('labels/'):
            raise ValueError('data.bootstrap must be under labels/')
    for key in ('train', 'val', 'test', 'splits', 'images_root', *(['bootstrap'] if 'bootstrap' in data else [])):
        if not isinstance(data[key], str) or not data[key] or Path(data[key]).is_absolute() or '..' in Path(data[key]).parts:
            raise ValueError(f'data.{key} must be a relative path within data root')
    embeddings = c.get('embeddings')
    if embeddings is not None:
        # Crop-embedding artifact for a later identifier fit; not a gallery and not GT.
        if not isinstance(embeddings, dict):
            raise ValueError('embeddings must be a mapping')
        if not isinstance(embeddings.get('model'), str) or not embeddings['model']:
            raise ValueError('embeddings.model is required')
        embeddings.setdefault('batch_size', 64)
        embeddings.setdefault('device', 'cuda')
        if not isinstance(embeddings.get('batch_size', 64), int) or embeddings.get('batch_size', 64) <= 0:
            raise ValueError('embeddings.batch_size must be a positive integer')
        if not isinstance(embeddings.get('device', 'cuda'), str):
            raise ValueError('embeddings.device must be a string')
    c['identifier'].setdefault('type', 'none')
    c['count'].setdefault('identity_thr', 0.5)
    if c['family'] == 'E_shelf':
        if c['count'].get('neighbour_vote', 'report') != 'report':
            raise ValueError('E_shelf requires neighbour_vote=report')
        c['count']['neighbour_vote'] = 'report'
    c['eval'].setdefault('score_thr', 'val')
    if c['eval']['score_thr'] != 'val':
        raise ValueError('score threshold must be selected on val')
    c.setdefault('seed', 42)
    if not isinstance(c['seed'], int) or isinstance(c['seed'], bool):
        raise ValueError('seed must be an integer')
    if not isinstance(c['localizer'].get('type'), str):
        raise ValueError('localizer.type is required')
    return c


def load(path):
    return resolve(yaml.safe_load(Path(path).read_text()))


def git_sha():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()


def git_state():
    """Hash tracked diffs and untracked non-ignored files without touching private data."""
    pathspecs = ['--', '.', ':(exclude)data', ':(exclude)runs']
    staged = subprocess.check_output(['git', 'diff', '--cached', '--binary', 'HEAD', *pathspecs])
    unstaged = subprocess.check_output(['git', 'diff', '--binary', *pathspecs])
    names = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard', '-z',
                                     '--', '.', ':(exclude)data', ':(exclude)runs']).split(b'\0')
    digest = hashlib.sha256()
    digest.update(len(staged).to_bytes(8, 'big'))
    digest.update(staged)
    digest.update(unstaged)
    dirty = bool(staged or unstaged)
    for name in sorted(filter(None, names)):
        path = Path(name.decode('utf-8', 'surrogateescape'))
        dirty = True
        digest.update(len(name).to_bytes(8, 'big'))
        digest.update(name)
        if path.is_symlink():
            contents = str(path.readlink()).encode()
            digest.update(b'symlink')
        else:
            contents = path.read_bytes()
            digest.update(b'file')
        digest.update(len(contents).to_bytes(8, 'big'))
        digest.update(contents)
    return {'dirty': dirty, 'diff_hash': digest.hexdigest() if dirty else None}


def content_hash(config, sha, state=None):
    state = state if state is not None else {'dirty': False, 'diff_hash': None}
    payload = json.dumps({'config': resolve(config), 'git_sha': sha, 'git_state': state},
                         sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def run_dir(config, sha, root='runs', state=None):
    return Path(root) / f"{config['id']}-{content_hash(config, sha, state)[:8]}"


def create_run_dir(config, sha, root='runs', state=None):
    path = run_dir(config, sha, root, state)
    path.mkdir(parents=True, exist_ok=False)
    return path
