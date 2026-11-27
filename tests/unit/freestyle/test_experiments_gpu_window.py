"""Regression tests for the bootstrap-label GPU window fixes (per-split manifests,
proposal class-agnostic semantics, checkpoint discovery, crop embeddings)."""
import json
from pathlib import Path

import pytest
import yaml

from freestyle.experiments import entry, launch, schema

CATEGORIES = [{'id': 1, 'name': 'cola'}, {'id': 2, 'name': 'unknown'}]
# One group per split, plus a second train group so detector validation has a holdout.
GROUPS = {'ga': 'train', 'gb': 'train', 'gc': 'val', 'gd': 'test', 'ge': 'exclude'}


def combined_bootstrap():
    """One label file covering every split, the way data/labels/bootstrap_v2.json is."""
    images, anns = [], []
    for idx, (group, _split) in enumerate(sorted(GROUPS.items()), 1):
        images.append({'id': idx, 'file_name': f'{group}/1.jpg', 'width': 4, 'height': 4})
        anns.append({'id': idx, 'image_id': idx, 'category_id': 2, 'bbox': [0, 0, 2, 2],
                     'attributes': {'judge_verdict': 'correct'}})
    return {'categories': CATEGORIES, 'images': images, 'annotations': anns}


def splits_document():
    return {'images': {f'{g}/1.jpg': s for g, s in GROUPS.items()}, 'groups': GROUPS}


def write_root(root: Path) -> Path:
    (root / 'labels').mkdir(parents=True)
    (root / 'gt').mkdir(parents=True)
    (root / 'raw').mkdir()
    combined = combined_bootstrap()
    (root / 'labels' / 'bootstrap.json').write_text(json.dumps(combined))
    (root / 'splits.json').write_text(json.dumps(splits_document()))
    val = {'categories': CATEGORIES, 'images': combined['images'][2:3],
           'annotations': combined['annotations'][2:3]}
    (root / 'gt' / 'val.json').write_text(json.dumps(val))
    return root


def bootstrap_config(**overrides):
    base = {'id': 'X1', 'family': 'C_det_agnostic_plus_embed',
            'compute': {'target': 'k8s:5090'},
            'data': {'train': 'labels/bootstrap.json', 'val': 'labels/bootstrap.json',
                     'test': 'labels/bootstrap.json', 'bootstrap': 'labels/bootstrap.json',
                     'label_source': 'bootstrap', 'images_root': 'raw'},
            'localizer': {'type': 'tiled_detector', 'training': {}},
            'identifier': {'type': 'none'}}
    return schema.resolve({**base, **overrides})


def test_split_manifest_derives_subset(tmp_path):
    root, out = write_root(tmp_path), tmp_path / 'out'
    path = entry.split_manifest(bootstrap_config(), root, out, 'val')
    coco = json.loads(path.read_text())
    assert [i['file_name'] for i in coco['images']] == ['gc/1.jpg']
    assert [a['image_id'] for a in coco['annotations']] == [3]
    assert path == out / 'val_manifest.json'


def test_split_manifest_passes_single_split_file_through(tmp_path):
    root, out = write_root(tmp_path), tmp_path / 'out'
    c = bootstrap_config(data={'val': 'gt/val.json'})
    assert entry.split_manifest(c, root, out, 'val') == root / 'gt' / 'val.json'
    assert not (out / 'val_manifest.json').exists()


def test_split_manifest_missing_split(tmp_path):
    root, out = write_root(tmp_path), tmp_path / 'out'
    (root / 'splits.json').write_text(json.dumps({'images': {'gc/1.jpg': 'val'},
                                                  'groups': {'gc': 'val'}}))
    with pytest.raises(ValueError):
        entry.split_manifest(bootstrap_config(), root, out, 'test')


def test_find_checkpoint_searches_nested_dirs(tmp_path):
    nested = tmp_path / 'training' / 'run-1' / 'checkpoints'
    nested.mkdir(parents=True)
    found = nested / 'checkpoint_best_total.pth'
    found.write_text('x')
    assert entry.find_checkpoint(tmp_path / 'training') == found
    # Nothing trained yet: the conventional flat path is returned so the backend reports it.
    assert entry.find_checkpoint(tmp_path / 'empty') == tmp_path / 'empty' / 'checkpoint_best_total.pth'


def test_tiled_checkpoint_uses_discovered_path(tmp_path, monkeypatch):
    root, out = write_root(tmp_path), tmp_path / 'out'
    best = out / 'training' / 'run-1' / 'checkpoints' / 'checkpoint_best_total.pth'
    best.parent.mkdir(parents=True)
    best.write_text('x')
    captured = {}

    def fake_main(argv):
        captured['cfg'] = yaml.safe_load(Path(argv[argv.index('--config') + 1]).read_text())

    monkeypatch.setattr('freestyle.pipeline.main', fake_main)
    entry.stage_split(bootstrap_config(), root, out, 'val')
    assert captured['cfg']['localizer']['checkpoint'] == str(best)


def test_proposals_are_not_relabelled_with_a_catalog_id(tmp_path, monkeypatch):
    root, out = write_root(tmp_path), tmp_path / 'out'
    out.mkdir()
    captured = {}

    def fake_propose(argv):
        pred = Path(argv[argv.index('--out') + 1]) / 'predictions.json'
        pred.parent.mkdir(parents=True, exist_ok=True)
        pred.write_text(json.dumps([{'image_id': 3, 'bbox': [0, 0, 2, 2], 'score': 0.9,
                                     'category_id': 1, 'id_score': 0.9}]))

    def fake_main(argv):
        captured['cfg'] = yaml.safe_load(Path(argv[argv.index('--config') + 1]).read_text())
        captured['images'] = argv[argv.index('--images') + 1:]

    monkeypatch.setattr('freestyle.propose.run.main', fake_propose)
    monkeypatch.setattr('freestyle.pipeline.main', fake_main)
    c = bootstrap_config(family='B_zeroshot_plus_embed',
                         localizer={'type': 'zeroshot', 'backend': 'owlv2'})
    entry.stage_split(c, root, out, 'val')
    # Proposals carry category id 1, which is a catalog id in the manifest; it must resolve
    # to the class-agnostic 'cartridge' label, never to 'cola'.
    assert captured['cfg']['categories']['cartridge'] == 1
    assert captured['cfg']['localizer']['type'] == 'proposals'
    assert captured['images'][0] == 'gc/1.jpg'


def test_export_and_inference_share_the_tiling_knobs(tmp_path, monkeypatch):
    root, out = write_root(tmp_path), tmp_path / 'out'
    out.mkdir()
    seen = {}

    def fake_export(coco, images_root, out_dir, split_map, tile=None, overlap=None, **kwargs):
        seen['tile'], seen['overlap'] = tile, overlap

    def fake_train(cfg, out_dir):
        seen['trained'] = True

    monkeypatch.setattr('freestyle.train.export_tiles.export_tiles', fake_export)
    monkeypatch.setattr('freestyle.train.train_detector.train', fake_train)
    c = bootstrap_config(localizer={'type': 'tiled_detector', 'tile': 768, 'overlap': 0.4,
                                    'training': {'model': 'RFDETRSmall'}})
    entry.stage_train(c, root, out)
    assert seen == {'tile': 768, 'overlap': 0.4, 'trained': True}


def test_embeddings_stage_embeds_all_labeled_splits(tmp_path, monkeypatch):
    root, out = write_root(tmp_path), tmp_path / 'out'
    out.mkdir()
    calls = []

    def fake_identify(argv):
        coco = json.loads(Path(argv[argv.index('--coco') + 1]).read_text())
        splits = splits_document()['images']
        calls.append(argv)
        calls.append(sorted({splits[i['file_name']] for i in coco['images']}))

    monkeypatch.setattr('freestyle.identify.cli.main', fake_identify)
    entry.stage_embeddings(bootstrap_config(embeddings={'model': 'stub'}), root, out)
    argv = calls[0]
    assert argv[0] == 'embed' and argv[argv.index('--split') + 1] == 'all'
    assert argv[argv.index('--model') + 1] == 'stub'
    assert argv[argv.index('--out') + 1] == str(out / 'embeddings.npz')
    assert calls[1] == ['test', 'train', 'val']  # 'exclude' crops are left out


def test_embeddings_stage_is_skipped_without_the_block(tmp_path, monkeypatch):
    root, out = write_root(tmp_path), tmp_path / 'out'
    out.mkdir()
    monkeypatch.setattr('freestyle.identify.cli.main', lambda argv: pytest.fail('must not run'))
    entry.stage_embeddings(bootstrap_config(), root, out)  # no embeddings block -> no-op
    gt_config = bootstrap_config(embeddings={'model': 'stub'})
    gt_config['data']['label_source'] = 'gt'
    entry.stage_embeddings(schema.resolve(gt_config), root, out)  # human GT crops are embedded by stage_gallery


def test_embeddings_schema_validation():
    resolved = schema.resolve(bootstrap_config(embeddings={'model': 'stub'}))
    assert resolved['embeddings'] == {'model': 'stub', 'batch_size': 64, 'device': 'cuda'}
    for bad in ({}, {'model': ''}, {'model': 'stub', 'batch_size': 0}, {'model': 1}, 'stub'):
        with pytest.raises(ValueError):
            schema.resolve(bootstrap_config(embeddings=bad))


def test_train_pip_specs_include_rfdetr_training_extras():
    # rfdetr alone cannot train: the [train] deps must reach the pod through --pip.
    cmd = launch.submit_argv(schema.resolve(bootstrap_config()), 'x.yaml', 'x1-abcd1234')
    specs = set(cmd[cmd.index('--pip') + 1].split())
    assert {'rfdetr==1.12.1', 'pytorch_lightning', 'vernier', 'torch-hungarian'} <= specs
    assert not any('[' in spec for spec in specs)  # submit.sh --pip rejects brackets


def test_evaluate_uses_per_split_gt(tmp_path, monkeypatch):
    import subprocess

    root = write_root(tmp_path)
    run = tmp_path / 'run'
    (run / 'val').mkdir(parents=True)
    (run / 'test').mkdir(parents=True)
    (run / 'val' / 'predictions.json').write_text('[]')
    (run / 'test' / 'predictions.json').write_text('[]')
    commands = []

    def fake_run(command, **kwargs):
        command = [str(p) for p in command]
        commands.append(command)
        if 'freestyle.eval.evaluate' in command and '--out' in command:
            Path(command[command.index('--out') + 1]).write_text(json.dumps({'score_thr': 0.3}))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(launch.subprocess, 'run', fake_run)
    (tmp_path / 'data').symlink_to(root, target_is_directory=True)
    monkeypatch.chdir(tmp_path)  # evaluate() resolves 'data' relative to the repo root
    launch.evaluate(bootstrap_config(), run)
    evaluate_cmd = next(cmd for cmd in commands if 'freestyle.eval.evaluate' in cmd)
    gt = Path(evaluate_cmd[evaluate_cmd.index('--gt') + 1])
    selection = Path(evaluate_cmd[evaluate_cmd.index('--select-thr-on') + 1])
    assert json.loads(gt.read_text())['images'][0]['file_name'] == 'gd/1.jpg'
    assert json.loads(selection.read_text())['images'][0]['file_name'] == 'gc/1.jpg'
    assert 'freestyle.eval.report' in ' '.join(commands[-1])


def _run_epoch(offset):
    from datetime import datetime, timedelta, timezone

    base = datetime(2026, 11, 27, tzinfo=timezone.utc) + timedelta(seconds=offset)
    return int(base.timestamp())


def test_job_wall_splits_setup_from_command():
    pod = json.dumps({'items': [{'status': {
        'startTime': '2026-11-27T00:00:00Z',
        'containerStatuses': [{'state': {'terminated': {
            'startedAt': '2026-11-27T00:00:20Z', 'finishedAt': '2026-11-27T00:03:00Z'}}}]}}]})
    log = ('== python: Python 3.12.3, torch import starts now '
           f'({_run_epoch(60)} epoch)\n== finished in 150 s\n')
    assert launch.job_wall(log, pod) == {'job_total': 150, 'command_start_epoch': _run_epoch(60),
                                         'image_and_init': 20, 'unpack_and_pip': 40, 'command': 110}
    # A job that never reported, or a pod already cleaned up by TTL, degrades to {} / partial.
    assert launch.job_wall('', '') == {}
    assert launch.job_wall(log, '') == {'job_total': 150, 'command_start_epoch': _run_epoch(60)}


def test_record_job_wall_updates_meta(tmp_path, monkeypatch):
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'job.log').write_text('== finished in 10 s\n')
    (run / 'meta.json').write_text(json.dumps({'timings': {'val': 1.0}}))
    monkeypatch.setattr(launch, 'job_wall', lambda log, pod: {'job_total': 10})
    monkeypatch.setattr(launch, '_kubectl_pod_json', lambda name: '')
    launch.record_job_wall(run, 'job-x')
    assert json.loads((run / 'meta.json').read_text())['job_wall'] == {'job_total': 10}
    launch.record_job_wall(tmp_path / 'missing', 'job-x')  # no log/meta -> silent no-op


def test_prune_training_weights_keeps_the_model(tmp_path):
    training = tmp_path / 'training'
    (training / 'run' ).mkdir(parents=True)
    keep = training / 'checkpoint_best_total.pth'
    keep.write_bytes(b'weights')
    state = training / 'run' / 'checkpoint_19.ckpt'
    state.write_bytes(b'x' * 1000)
    assert entry.prune_training_weights(training) == 1000
    assert keep.is_file() and not state.exists()
