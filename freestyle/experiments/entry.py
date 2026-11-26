"""Execute resumable experiment stages in a pod or local process."""
import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import yaml

from freestyle.experiments.schema import load


def stage_train(c, root, out):
    if c['family'] not in ('C_det_agnostic_plus_embed', 'D_det_multiclass', 'F_det_plus_vlm_id'):
        return
    if c['data']['label_source'] != 'gt':
        raise NotImplementedError('bootstrap training requires an explicit prepared dataset')
    from freestyle.train.export_tiles import export_tiles
    from freestyle.train.train_detector import train

    loc = c['localizer']
    dataset = out / 'dataset'
    train_gt = json.loads((root / c['data']['train']).read_text())
    val_gt = json.loads((root / c['data']['val']).read_text())
    if train_gt['categories'] != val_gt['categories']:
        raise ValueError('train and val category mappings must match')
    combined = dict(train_gt)
    combined['images'] = list(train_gt['images'])
    combined['annotations'] = list(train_gt['annotations'])
    next_id = max((i['id'] for i in combined['images']), default=0) + 1
    for image in val_gt['images']:
        new_id = next_id
        next_id += 1
        combined['images'].append({**image, 'id': new_id})
        combined['annotations'].extend({**a, 'image_id': new_id} for a in val_gt['annotations']
                                       if a['image_id'] == image['id'])
    combined_path = out / 'training_input.json'
    combined_path.write_text(json.dumps(combined))
    export_tiles(str(combined_path), str(root / c['data']['images_root']),
                 str(dataset), str(root / c['data']['splits']),
                 class_agnostic=c['family'] != 'D_det_multiclass')
    train({**loc['training'], 'dataset_dir': str(dataset), 'seed': c['seed']}, out / 'training')


def stage_gallery(c, root, out):
    ident = c['identifier']
    if ident['type'] != 'embedding_knn':
        return
    from freestyle.identify.cli import main
    main(['embed', '--split-map', str(root / c['data']['splits']), '--coco',
          str(root / c['data']['train']), '--images-root', str(root / c['data']['images_root']),
          '--out', str(out / 'gallery.npz'), '--model', ident['model']])


def stage_split(c, root, out, split):
    from freestyle.pipeline import main

    loc = dict(c['localizer'])
    loc.pop('training', None)
    if loc['type'] == 'zeroshot':
        from freestyle.propose.run import main as propose
        backend = loc.pop('backend')
        loc.pop('type')
        args = ['--backend', backend, '--images', str(root / c['data'][split]),
                '--images-root', str(root / c['data']['images_root']), '--splits',
                str(root / c['data']['splits']), '--out', str(out / f'proposals_{split}')]
        for k, v in loc.items():
            args.extend(['--' + k.replace('_', '-'), str(v)])
        propose(args)
        loc = {'type': 'proposals', 'path': str(out / f'proposals_{split}' / 'predictions.json')}
    elif loc['type'] == 'tiled_detector':
        loc.setdefault('checkpoint', str(out / 'training' / 'checkpoint_best_total.pth'))
    ident = dict(c['identifier'])
    if ident['type'] == 'embedding_knn':
        ident['gallery'] = str(out / 'gallery.npz')
    gt = json.loads((root / c['data'][split]).read_text())
    categories = {x['name']: x['id'] for x in gt['categories']}
    cfg = {'localizer': loc, 'identifier': ident, 'counting': c['count'],
           'manifest': str(root / c['data'][split]), 'images_root': str(root / c['data']['images_root']),
           'split_map': str(root / c['data']['splits']), 'categories': categories}
    path = out / f'pipeline_{split}.yaml'
    path.write_text(yaml.safe_dump(cfg))
    # Use the split manifest itself, not every image listed in the global split map.
    main(['--config', str(path), '--images', *[i['file_name'] for i in gt['images']],
          '--out', str(out / split)])


def run(c, root, out, stages=None, git_sha=None):
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    markers = out / 'stages'
    markers.mkdir(exist_ok=True)
    stages = stages or [('train', lambda: stage_train(c, root, out)),
                        ('gallery', lambda: stage_gallery(c, root, out)),
                        ('val', lambda: stage_split(c, root, out, 'val')),
                        ('test', lambda: stage_split(c, root, out, 'test'))]
    meta_path = out / 'meta.json'
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {'timings': {}}
    meta.update(experiment=c['id'], config=c, git_sha=git_sha,
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
    args = p.parse_args(argv)
    run(load(args.config), args.data_root, args.out, git_sha=args.git_sha)


if __name__ == '__main__':
    main()
