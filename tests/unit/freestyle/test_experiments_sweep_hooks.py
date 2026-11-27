"""Unit tests for the GPU-sweep hooks in freestyle.experiments.entry.

Covers the copy-paste augmentation hook (S3), the shelf-sized proposal rejection (S6) and
the disk pruning that keeps a collected run small.
"""
import json
from pathlib import Path

import pytest

from freestyle.experiments import entry


def _config(augment=None):
    training = {'model': 'RFDETRSmall', 'resolution': 512, 'epochs': 1}
    if augment:
        training['augment'] = augment
    return {
        'id': 'X_test', 'family': 'C_det_agnostic_plus_embed', 'seed': 7,
        'data': {'splits': 'splits.json', 'images_root': 'images',
                 'label_source': 'bootstrap', 'bootstrap': 'labels.json'},
        'localizer': {'type': 'tiled_detector', 'tile': 1024, 'overlap': 0.25,
                      'training': training},
    }


def _write_inputs(root):
    (root / 'images' / 'shelf').mkdir(parents=True)
    (root / 'images' / 'shelf' / 'a.jpg').write_bytes(b'img')
    (root / 'images' / 'shelf' / 'b.jpg').write_bytes(b'img')
    (root / 'splits.json').write_text(json.dumps({'images': {
        'shelf/a.jpg': 'train', 'shelf/b.jpg': 'train', 'shelf/c.jpg': 'val'}}))


def _train_coco():
    return {'categories': [{'id': 1, 'name': 'cartridge'}],
            'images': [{'id': 1, 'file_name': 'shelf/a.jpg', 'width': 4, 'height': 4},
                       {'id': 2, 'file_name': 'shelf/b.jpg', 'width': 4, 'height': 4}],
            'annotations': [{'id': 1, 'image_id': 1, 'bbox': [0, 0, 2, 2], 'category_id': 1},
                            {'id': 2, 'image_id': 2, 'bbox': [1, 1, 2, 2], 'category_id': 1}]}


def test_merge_coco_renumbers_both_id_spaces():
    extra = {'categories': [{'id': 1, 'name': 'cartridge'}],
             'images': [{'id': 1, 'file_name': 'images/synth_1.png'}],
             'annotations': [{'id': 1, 'image_id': 1, 'bbox': [0, 0, 1, 1], 'category_id': 1}]}
    merged = entry.merge_coco(_train_coco(), extra)
    assert [i['id'] for i in merged['images']] == [1, 2, 3]
    assert [(a['id'], a['image_id']) for a in merged['annotations']] == [
        (1, 1), (2, 2), (3, 3)]
    bad = {**extra, 'categories': [{'id': 99, 'name': 'other'}]}
    with pytest.raises(ValueError, match='category'):
        entry.merge_coco(_train_coco(), bad)


def test_augment_training_set_uses_train_split_guard_and_inherits_splits(tmp_path, monkeypatch):
    root = tmp_path / 'data'
    _write_inputs(root)
    out = tmp_path / 'run'
    out.mkdir()
    coco = _train_coco()
    combined = out / 'training_input.json'
    combined.write_text(json.dumps(coco))
    seen = {}

    def fake_run_cli(**kwargs):
        # The CLI's own leakage guard: only the train split may ever be synthesised from.
        seen.update(kwargs)
        assert kwargs['split'] == 'train'
        aug = Path(kwargs['out_dir'])
        (aug / 'images').mkdir(parents=True, exist_ok=True)
        (aug / 'images' / 'a_synth_001_000.png').write_bytes(b'synth')
        (aug / 'contact_sheet.png').write_bytes(b'sheet')
        (aug / 'annotations.json').write_text(json.dumps({
            'categories': coco['categories'],
            'images': [{'id': 1, 'file_name': 'images/a_synth_001_000.png', 'width': 4,
                        'height': 4,
                        'attributes': {'source': 'synthetic:copy_paste',
                                       'parent': 'shelf/a.jpg'}}],
            'annotations': [{'id': 1, 'image_id': 1, 'bbox': [0, 0, 1, 1], 'category_id': 1}]}))
        (aug / 'splits.json').write_text(json.dumps({'images': {}}))

    monkeypatch.setattr('freestyle.augment.copy_paste.run_cli', fake_run_cli)
    splits = {'images': {'shelf/a.jpg': 'train', 'shelf/b.jpg': 'val'}}
    c = _config({'n_per_image': 3, 'slot_shuffle_prob': 0.5})
    coco_path, split_path, images_root = entry.augment_training_set(
        c, root, out, combined, coco, splits)

    assert seen['n_per_image'] == 3 and seen['slot_shuffle_prob'] == 0.5
    assert Path(images_root) == out / 'augment'
    merged = json.loads(coco_path.read_text())
    assert [i['file_name'] for i in merged['images']] == [
        'shelf/a.jpg', 'shelf/b.jpg', 'images/a_synth_001_000.png']
    # The synthetic image follows its parent's holdout split, so val stays out of training.
    assert json.loads(split_path.read_text())['images'] == {
        'shelf/a.jpg': 'train', 'shelf/b.jpg': 'val', 'images/a_synth_001_000.png': 'train'}
    assert (out / 'augment' / 'shelf' / 'a.jpg').is_symlink()
    assert (out / 'augment' / 'shelf' / 'a.jpg').readlink() == (
        root / 'images' / 'shelf' / 'a.jpg').resolve()


def test_augment_training_set_rejects_synthetic_without_train_parent(tmp_path, monkeypatch):
    root = tmp_path / 'data'
    _write_inputs(root)
    out = tmp_path / 'run'
    out.mkdir()
    coco = _train_coco()
    combined = out / 'training_input.json'
    combined.write_text(json.dumps(coco))

    def fake_run_cli(**kwargs):
        aug = Path(kwargs['out_dir'])
        aug.mkdir(parents=True, exist_ok=True)
        (aug / 'annotations.json').write_text(json.dumps({
            'categories': coco['categories'],
            'images': [{'id': 1, 'file_name': 'images/x.png',
                        'attributes': {'parent': 'shelf/val-only.jpg'}}],
            'annotations': []}))

    monkeypatch.setattr('freestyle.augment.copy_paste.run_cli', fake_run_cli)
    with pytest.raises(ValueError, match='train-split parent'):
        entry.augment_training_set(_config({}), root, out, combined, coco,
                                   {'images': {'shelf/a.jpg': 'train'}})


def test_filter_oversized_proposals_drops_shelf_boxes(tmp_path):
    preds = tmp_path / 'predictions.json'
    preds.write_text(json.dumps([
        {'image_id': 1, 'bbox': [0, 0, 100, 100], 'score': 0.4},   # 100% of a 100x100 frame
        {'image_id': 1, 'bbox': [0, 0, 20, 20], 'score': 0.3},     # 4%
        {'image_id': 2, 'bbox': [0, 0, 30, 30], 'score': 0.2},     # unknown size -> kept
    ]))
    kept, dropped = entry.filter_oversized_proposals(preds, {1: (100, 100)}, 0.08)
    assert dropped == 1
    assert [k['bbox'] for k in kept] == [[0, 0, 20, 20], [0, 0, 30, 30]]
    with pytest.raises(ValueError):
        entry.filter_oversized_proposals(preds, {}, 0.0)


def test_prune_keeps_only_best_weights_and_drops_derived_inputs(tmp_path):
    training = tmp_path / 'training'
    (training / 'run').mkdir(parents=True)
    best = training / 'checkpoint_best_total.pth'
    best.write_bytes(b'best')
    for name in ('last_ema.pth', 'checkpoint_best_ema.pth'):
        (training / name).write_bytes(b'x' * 50)
    (training / 'run' / 'checkpoint_1.ckpt').write_bytes(b'y' * 70)
    assert entry.prune_training_weights(training) == 170
    assert best.is_file()
    assert not (training / 'last_ema.pth').exists()

    (tmp_path / 'dataset' / 'train').mkdir(parents=True)
    tile = tmp_path / 'dataset' / 'train' / 't1.png'
    tile.write_bytes(b'z' * 10)
    (tmp_path / 'augment').mkdir()
    (tmp_path / 'augment' / 'contact_sheet.png').write_bytes(b's' * 3)
    (tmp_path / 'augment' / 'images').mkdir()
    (tmp_path / 'augment' / 'images' / 's.png').write_bytes(b'w' * 11)
    link = tmp_path / 'augment' / 'shelf.jpg'
    link.symlink_to(tmp_path / 'nowhere.jpg')
    assert entry.prune_derived_inputs(tmp_path) == 21
    assert tile.exists() is False and link.exists() is False
    assert (tmp_path / 'augment' / 'contact_sheet.png').is_file()


def test_class_ids_align_with_the_checkpoint_class_names():
    """A class-agnostic head predicts label id 1 while class_names is keyed 0 (window-1 S1/S2)."""
    from freestyle.infer.detector import align_class_index

    assert align_class_index({0: 'cartridge'}, 1) == 0        # dict keyed by head index
    assert align_class_index(['cartridge'], 1) == 0           # plain list of names
    assert align_class_index({1: 'cartridge'}, 1) == 1        # already keyed by label id
    assert align_class_index(['coke', 'fanta', 'sprite'], 2) == 2   # multiclass untouched
