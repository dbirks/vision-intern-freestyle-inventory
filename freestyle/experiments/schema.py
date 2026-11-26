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
    if compute['target'] not in ('local', 'k8s:5090', 'k8s:6000'):
        raise ValueError('invalid compute target')
    for key in ('vram_mb', 'timeout'):
        if not isinstance(compute[key], int) or isinstance(compute[key], bool) or compute[key] <= 0:
            raise ValueError(f'compute.{key} must be a positive integer')
    data = c['data']
    for key, value in {'train': 'gt/train.json', 'val': 'gt/val.json', 'test': 'gt/test.json',
                       'splits': 'splits.json', 'images_root': 'raw/freestyle-vision-data/images',
                       'label_source': 'gt'}.items():
        data.setdefault(key, value)
    if data['label_source'] not in ('gt', 'bootstrap', 'bootstrap+gt'):
        raise ValueError('invalid label source')
    for key in ('train', 'val', 'test', 'splits', 'images_root'):
        if not isinstance(data[key], str) or not data[key] or Path(data[key]).is_absolute() or '..' in Path(data[key]).parts:
            raise ValueError(f'data.{key} must be a relative path within data root')
    c['identifier'].setdefault('type', 'none')
    c['count'].setdefault('identity_thr', 0.5)
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


def content_hash(config, sha):
    payload = json.dumps({'config': resolve(config), 'git_sha': sha}, sort_keys=True,
                         separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def run_dir(config, sha, root='runs'):
    return Path(root) / f"{config['id']}-{content_hash(config, sha)[:8]}"


def create_run_dir(config, sha, root='runs'):
    path = run_dir(config, sha, root)
    path.mkdir(parents=True, exist_ok=False)
    return path
