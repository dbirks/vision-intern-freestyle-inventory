"""Execute resumable experiment stages in a pod or local process."""
import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import yaml

from freestyle.experiments.schema import load


def prepare_training(c, root, out, filename='training_input.json'):
    """Build a train-only COCO file; human annotations replace pseudo labels per image."""
    data = c['data']
    splits = json.loads((root / data['splits']).read_text())['images']
    gt = json.loads((root / data['train']).read_text())
    categories = gt['categories']
    ids = {category['name']: category['id'] for category in categories}
    source = data['label_source']
    sources = []
    if source in ('gt', 'bootstrap+gt'):
        sources.append((gt, False))
    if source in ('bootstrap', 'bootstrap+gt'):
        sources.append((json.loads((root / data['bootstrap']).read_text()), True))
    result = {'info': {}, 'licenses': [], 'categories': categories, 'images': [], 'annotations': []}
    seen = set()
    for coco, pseudo in sources:
        source_names = {cat['id']: cat['name'] for cat in coco['categories']}
        anns = {}
        for ann in coco['annotations']:
            anns.setdefault(ann['image_id'], []).append(ann)
        for image in coco['images']:
            name = image['file_name']
            if splits.get(name) != 'train' or name in seen:
                continue
            seen.add(name)
            image_id = len(result['images']) + 1
            result['images'].append({**image, 'id': image_id})
            for ann in anns.get(image['id'], []):
                if pseudo:
                    attrs = ann.get('attributes', {})
                    filters = data['bootstrap_filter']
                    if (attrs.get('judge_verdict') not in filters['verdicts'] or
                        attrs.get('judge_score', 0) < filters['min_judge_score'] or
                        attrs.get('support', 0) < filters['min_support'] or
                        attrs.get('id_confidence', 0) < filters['min_id_confidence']):
                        continue
                category = source_names.get(ann['category_id'])
                if category not in ids:
                    raise ValueError(f'unknown training category: {category}')
                result['annotations'].append({**ann, 'id': len(result['annotations']) + 1,
                                              'image_id': image_id, 'category_id': ids[category]})
    if not result['images']:
        raise ValueError('no training images in selected label source')
    path = out / filename
    path.write_text(json.dumps(result))
    return path


def split_manifest(c, root, out, split):
    """Return a COCO manifest holding only *split* images, deriving a subset if needed.

    Bootstrap labels live in one combined file (all splits, plus 'exclude'), so a job
    that reads ``data.val`` would otherwise propose on train/exclude images too. A
    per-split manifest is written to the writable run dir; the read-only data root is
    never touched. Single-split manifests (human GT files) are used as they are.
    """
    root, out = Path(root), Path(out)
    source = root / c['data'][split]
    coco = json.loads(source.read_text())
    splits = json.loads((root / c['data']['splits']).read_text())['images']
    if all(splits.get(i['file_name']) == split for i in coco['images']):
        return source
    images = [i for i in coco['images'] if splits.get(i['file_name']) == split]
    if not images:
        raise ValueError(f'no images in split {split} of {source}')
    keep = {i['id'] for i in images}
    subset = {**coco, 'images': images,
              'annotations': [a for a in coco['annotations'] if a['image_id'] in keep]}
    path = out / f'{split}_manifest.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(subset))
    return path


def find_checkpoint(training_dir, name='checkpoint_best_total.pth'):
    """Locate the trained checkpoint wherever the trainer wrote it (falls back to the
    conventional flat path, which the backend reports itself when it is missing)."""
    training_dir = Path(training_dir)
    for candidate in (training_dir / name, training_dir / 'checkpoints' / name):
        if candidate.is_file():
            return candidate
    hits = sorted(training_dir.rglob(name)) or sorted(training_dir.rglob('*.pth'))
    return hits[-1] if hits else training_dir / name


def stage_train(c, root, out):
    if c['family'] not in ('C_det_agnostic_plus_embed', 'D_det_multiclass', 'E_shelf',
                           'F_det_plus_vlm_id'):
        return
    from freestyle.train.export_tiles import export_tiles
    from freestyle.train.train_detector import train

    dataset = out / 'dataset'
    combined_path = prepare_training(c, root, out)
    train_coco = json.loads(combined_path.read_text())
    split_map = json.loads((root / c['data']['splits']).read_text())
    from freestyle.identify.cli import _resolve_groups_and_splits
    images = train_coco['images']
    groups, _ = _resolve_groups_and_splits([i['file_name'] for i in images],
                                           [str(i['id']) for i in images], split_map)
    unique_groups = sorted(set(groups))
    if len(unique_groups) < 2:
        raise ValueError('detector training requires at least two train groups for validation')
    # A deterministic group holdout avoids using experiment val or test in training.
    import random
    random.Random(c['seed']).shuffle(unique_groups)
    held_out = set(unique_groups[:max(1, len(unique_groups) // 5)])
    training_splits = {'images': {i['file_name']: 'val' if g in held_out else 'train'
                                  for i, g in zip(images, groups)}}
    training_split_path = out / 'training_splits.json'
    training_split_path.write_text(json.dumps(training_splits))
    # Tiling must agree between export and tiled inference, so both read the same knobs.
    export_tiles(str(combined_path), str(root / c['data']['images_root']),
                 str(dataset), str(training_split_path),
                 tile=c['localizer'].get('tile', 1024), overlap=c['localizer'].get('overlap', 0.25),
                 class_agnostic=c['family'] != 'D_det_multiclass')
    train({**c['localizer']['training'], 'dataset_dir': str(dataset), 'seed': c['seed']},
          out / 'training')
    prune_training_weights(out / 'training')


def prune_training_weights(training_dir):
    """Drop Lightning resumption states so the run stays collectable.

    Each ``*.ckpt`` is a full optimizer/LR-scheduler snapshot (0.5 GB per epoch
    checkpoint here) and only useful for resuming an interrupted fit; inference needs the
    ``.pth`` weights, which stay. Returns the bytes freed.
    """
    freed = 0
    for path in Path(training_dir).rglob('*.ckpt'):
        freed += path.stat().st_size
        path.unlink()
    return freed


def stage_gallery(c, root, out):
    ident = c['identifier']
    if ident['type'] != 'embedding_knn':
        return
    from freestyle.identify.cli import main
    gt_config = {**c, 'data': {**c['data'], 'label_source': 'gt'}}
    gallery_input = prepare_training(gt_config, root, out, 'gallery_input.json')
    main(['embed', '--split-map', str(root / c['data']['splits']), '--coco',
          str(gallery_input), '--images-root', str(root / c['data']['images_root']),
          '--out', str(out / 'gallery.npz'), '--model', ident['model']])


def stage_embeddings(c, root, out):
    """Embed every model-labelled crop of the train/val/test images to ``embeddings.npz``.

    There is no human GT yet, so this is deliberately *not* a KNN gallery: it is the crop
    set (with image_id / split / group / proposal annotation id per row) that a human will
    label so identifiers can be fitted later. Rows keep the model's proposed label only.
    """
    spec = c.get('embeddings')
    if not spec:
        return
    root, out = Path(root), Path(out)
    data = c['data']
    if data['label_source'] == 'gt':
        return  # once human GT crops exist, stage_gallery embeds them instead
    splits = json.loads((root / data['splits']).read_text())['images']
    coco = json.loads((root / data['bootstrap']).read_text())
    keep = {i['id'] for i in coco['images'] if splits.get(i['file_name']) in ('train', 'val', 'test')}
    subset = {**coco, 'images': [i for i in coco['images'] if i['id'] in keep],
              'annotations': [a for a in coco['annotations'] if a['image_id'] in keep]}
    source = out / 'crop_labels.json'
    source.write_text(json.dumps(subset))
    from freestyle.identify.cli import main as identify

    identify(['embed', '--split-map', str(root / data['splits']), '--split', 'all',
              '--coco', str(source), '--images-root', str(root / data['images_root']),
              '--out', str(out / 'embeddings.npz'), '--model', spec['model'],
              '--device', str(spec.get('device', 'cuda')),
              '--batch-size', str(spec.get('batch_size', 64))])


def stage_split(c, root, out, split):
    from freestyle.pipeline import main

    root, out = Path(root), Path(out)
    manifest = split_manifest(c, root, out, split)
    gt = json.loads(manifest.read_text())
    loc = dict(c['localizer'])
    loc.pop('training', None)
    if loc['type'] == 'zeroshot':
        from freestyle.propose.run import main as propose
        backend = loc.pop('backend')
        loc.pop('type')
        args = ['--backend', backend, '--images', str(manifest),
                '--images-root', str(root / c['data']['images_root']), '--splits',
                str(root / c['data']['splits']), '--out', str(out / f'proposals_{split}')]
        for k, v in loc.items():
            key = '--' + k.replace('_', '-')
            # List-valued knobs (e.g. prompts) map to argparse `nargs='*'` flags: pass
            # each element as its own token, otherwise str(v) would emit a Python repr
            # like "['cartridge box']" and become a single mangled prompt.
            if isinstance(v, (list, tuple)):
                args.extend([key, *(str(x) for x in v)])
            elif isinstance(v, bool):
                if v:
                    args.append(key)
            else:
                args.extend([key, str(v)])
        propose(args)
        loc = {'type': 'proposals', 'path': str(out / f'proposals_{split}' / 'predictions.json')}
    elif loc['type'] == 'tiled_detector':
        loc.setdefault('checkpoint', str(find_checkpoint(out / 'training')))
    ident = dict(c['identifier'])
    if ident['type'] == 'embedding_knn':
        ident['gallery'] = str(out / 'gallery.npz')
    categories = {x['name']: x['id'] for x in gt['categories']}
    if loc['type'] == 'proposals':
        # Zero-shot proposals are class-agnostic and carry a stand-in category id; without
        # this the id would be resolved against the catalog and every proposal would be
        # reported as whichever catalog class happens to share that id.
        categories['cartridge'] = next((x['id'] for x in gt['categories'] if x['name'] == 'cartridge'), 1)
    cfg = {'localizer': loc, 'identifier': ident, 'counting': c['count'],
           'manifest': str(root / c['data'][split]), 'images_root': str(root / c['data']['images_root']),
           'split_map': str(root / c['data']['splits']), 'categories': categories}
    path = out / f'pipeline_{split}.yaml'
    path.write_text(yaml.safe_dump(cfg))
    # Use the (possibly derived) split manifest, not every image in the global split map.
    main(['--config', str(path), '--images', *[i['file_name'] for i in gt['images']],
          '--out', str(out / split)])


def split_inference_stats(directory, proposals=None):
    """Per-split inference cost: pipeline timing records plus the proposal stage's own.

    A zero-shot run scores from a cached proposals file, so the pipeline's per-image time is
    only a file lookup; the GPU cost of the proposal backend lives in proposals/<split>.
    """
    seconds = []
    for path in sorted((Path(directory) / 'images').glob('*.json')):
        try:
            timings = json.loads(path.read_text()).get('timings_ms', {})
        except (OSError, ValueError):
            continue
        if timings:
            seconds.append(sum(timings.values()) / 1000)
    stats = {'images': len(seconds),
             'seconds_per_image': sum(seconds) / len(seconds) if seconds else None,
             'total_seconds': sum(seconds) if seconds else None}
    if proposals is not None and Path(proposals, 'meta.json').is_file():
        source = json.loads(Path(proposals, 'meta.json').read_text())
        stats['proposal_seconds_per_image'] = source.get('seconds_per_image')
        stats['proposal_peak_gpu_bytes'] = (source.get('peak_gpu_mem_gb') or 0) * 1024**3
    return stats


def run(c, root, out, stages=None, git_sha=None, git_state=None):
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    markers = out / 'stages'
    markers.mkdir(exist_ok=True)
    stages = stages or [('train', lambda: stage_train(c, root, out)),
                        ('gallery', lambda: stage_gallery(c, root, out)),
                        ('embeddings', lambda: stage_embeddings(c, root, out)),
                        ('val', lambda: stage_split(c, root, out, 'val')),
                        ('test', lambda: stage_split(c, root, out, 'test'))]
    meta_path = out / 'meta.json'
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {'timings': {}}
    meta.update(experiment=c['id'], config=c, git_sha=git_sha,
                git_dirty=git_state['dirty'] if git_state else None,
                git_diff_hash=git_state['diff_hash'] if git_state else None,
                versions={'python': platform.python_version()})
    try:
        import torch
        meta['versions']['torch'] = str(torch.__version__)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        torch = None
    try:
        for name, fn in stages:
            marker = markers / (name + '.done')
            if marker.exists():
                continue
            start = time.perf_counter()
            fn()
            meta['timings'][name] = time.perf_counter() - start
            if name in ('val', 'test'):
                meta.setdefault('inference', {})[name] = split_inference_stats(
                    out / name, out / f'proposals_{name}')
            meta_path.write_text(json.dumps(meta, indent=2) + '\n')
            marker.write_text('done\n')
    finally:
        meta['peak_gpu_bytes'] = torch.cuda.max_memory_allocated() if torch and torch.cuda.is_available() else 0
        try:
            meta['nvidia_smi'] = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
                '--format=csv,noheader,nounits'], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            meta['nvidia_smi'] = None
        meta_path.write_text(json.dumps(meta, indent=2) + '\n')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--git-sha')
    p.add_argument('--git-dirty', choices=('true', 'false'))
    p.add_argument('--git-diff-hash')
    args = p.parse_args(argv)
    state = ({'dirty': args.git_dirty == 'true', 'diff_hash': args.git_diff_hash}
             if args.git_dirty is not None else None)
    run(load(args.config), args.data_root, args.out, git_sha=args.git_sha, git_state=state)


if __name__ == '__main__':
    main()
