"""Tests for the fail-closed human GT preparation."""

import pytest

from freestyle.experiments.score_all import build_human_gt, main


def test_missing_answers_fails_without_report(tmp_path):
    out = tmp_path / 'report.md'
    (tmp_path / 'queue.json').write_text('{}')
    with pytest.raises(SystemExit):
        main(['--answers', str(tmp_path / 'missing'), '--queue', str(tmp_path / 'queue.json'),
              '--runs', 'runs/*', '--out', str(out)])
    assert not out.exists()


def test_human_gt_filters_skips_and_model_labels(monkeypatch):
    coco = {'images': [
        {'id': 1, 'file_name': 'good', 'attributes': {'human_counts': {'total': 1}, 'verified': False}},
        {'id': 2, 'file_name': 'bad', 'attributes': {'human_counts': {'total': 1}, 'verified': False}},
    ], 'annotations': [
        {'image_id': 1, 'attributes': {'identity_source': 'human'}},
        {'image_id': 1, 'attributes': {'identity_source': 'model'}},
    ], 'info': {}}
    counts = {'images': {'good': {}, 'bad': {}}, 'mismatches': [], 'unverified': ['good', 'bad']}
    monkeypatch.setattr('freestyle.experiments.score_all.build_coco', lambda *args: (coco, counts))
    questions = [{'qid': f'{name}{i}', 'image_id': name} for name in ('good', 'bad') for i in range(5)]
    answers = {f'good{i}': {'result': 'correct'} for i in range(4)}
    answers.update({f'bad{i}': {'result': 'correct'} for i in range(3)})
    answers['bad3'] = {'result': 'skip'}
    result, result_counts, excluded = build_human_gt({'questions': questions}, answers, 'test')
    assert [image['file_name'] for image in result['images']] == ['good']
    assert len(result['annotations']) == 1
    assert list(result_counts['images']) == ['good']
    assert excluded[0]['image_id'] == 'bad'
    assert excluded[0]['answered'] == 3


def test_full_scoring_with_fake_runs(tmp_path, monkeypatch):
    import json

    import numpy as np

    def save(path, obj):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj))

    images = []
    questions = []
    records = []
    proposals = []
    audit = []
    groups = {}
    split_images = {}
    predictions = {'val': [], 'test': []}
    for idx, (split, scene, blur) in enumerate([
        ('val', 'cabinet_full', 'none'), ('val', 'cabinet_full', 'none'),
        ('test', 'cabinet_full', 'none'), ('test', 'cabinet_partial', 'mild'),
        ('test', 'cabinet_full', 'none'),
    ], 1):
        name = f'local/{idx}.jpg'
        split_images[name] = split
        groups[name] = split
        images.append({'image_id': name, 'coco_image_id': idx, 'split': split, 'width': 100,
                       'height': 100, 'model_top': 1, 'model_bottom': 0, 'model_total': 1})
        count = f'count:{name}'
        crop = f'crop:{name}:{idx}'
        questions += [{'qid': count, 'image_id': name, 'kind': 'count'},
                      {'qid': crop, 'image_id': name, 'kind': 'crop', 'ann_id': idx,
                       'bbox': [10, 10, 20, 30], 'guess': 'sprite'}]
        if idx != 5:
            records += [{'qid': count, 'answer': {'result': 'correct'}},
                        {'qid': crop, 'answer': {'result': 'correct'}}]
        audit.append({'image_id': name, 'scene_label': scene, 'motion_blur': blur})
        proposals.append({'id': idx, 'image_id': idx, 'bbox': [10, 10, 10, 20], 'category_id': 1,
                          'attributes': {'id_mode': 'two_stage', 'id_confidence': .9}})
        predictions[split].append({'image_id': idx, 'bbox': [10, 10, 10, 20], 'category_id': 1,
                                   'score': .9, 'proposed_class': 'sprite', 'id_score': .9})
    queue = {'version': 1, 'classes': [{'id': 1, 'name': 'sprite'}, {'id': 2, 'name': 'unknown'}],
             'images': images, 'questions': questions}
    queue_path = tmp_path / 'queue.json'
    answer_path = tmp_path / 'answers.jsonl'
    save(queue_path, queue)
    answer_path.write_text(''.join(json.dumps(r) + '\n' for r in records))
    save(tmp_path / 'splits.json', {'images': split_images, 'groups': groups})
    save(tmp_path / 'audit.json', {'images': audit})
    save(tmp_path / 'proposals.json', {'images': [{'id': i + 1, 'file_name': im['image_id']}
                                                   for i, im in enumerate(images)],
                                       'annotations': proposals, 'categories': queue['classes']})
    run = tmp_path / 'runs' / 'D_fake'
    save(run / 'meta.json', {'config': {'family': 'D_det_multiclass'}})
    for split in predictions:
        save(run / split / 'predictions.json', predictions[split])
    np.savez(run / 'embeddings.npz', embeddings=np.ones((5, 2)),
             file_names=np.array([im['image_id'] for im in images]),
             annotation_ids=np.array([str(i) for i in range(1, 6)]),
             bboxes=np.array([[10, 10, 10, 20]] * 5))
    out = tmp_path / 'RESULTS_GT.md'
    assert main(['--queue', str(queue_path), '--answers', str(answer_path), '--runs', str(run),
                 '--splits', str(tmp_path / 'splits.json'), '--audit', str(tmp_path / 'audit.json'),
                 '--proposals', str(tmp_path / 'proposals.json'), '--out', str(out), '--bootstrap', '20']) == 0
    data = json.loads((tmp_path / 'results_gt.json').read_text())
    assert len(data['rows']) == 10
    assert {r['identifier'] for r in data['rows']} == {'none', 'own', 'knn', 'proto', 'vlm'}
    assert data['excluded']['test'][0]['image_id'] == 'local/5.jpg'
    for row in data['rows']:
        assert row['threshold_source'] == 'val'
        assert row['subsets']['all']['bootstrap_ci95']['n_groups'] == 2
        assert row['subsets']['sharp-full']['n_images'] == 1
        assert row['subsets']['partial/blurry']['n_images'] == 1
        metrics = row['subsets']['all']['metrics']
        ci = row['subsets']['all']['bootstrap_ci95']['metrics']
        assert metrics['total_error'] == 0
        assert metrics['ap50'] == 1
        assert ci['total_error'] == [0, 0]
        assert ci['ap50'] == [1, 1]
        if row['identifier'] != 'none':
            assert metrics['class:sprite'] == 0
            assert metrics['identity_accuracy'] == 1
            assert metrics['exact_inventory'] == 1
        else:
            assert 'identity_accuracy' not in metrics
    assert 'model proposals' in out.read_text()
    monkeypatch.chdir(tmp_path)
    assert main(['--demo', '--queue', str(queue_path), '--runs', str(run),
                 '--splits', str(tmp_path / 'splits.json'), '--audit', str(tmp_path / 'audit.json'),
                 '--proposals', str(tmp_path / 'proposals.json'), '--bootstrap', '5']) == 0
    demo = tmp_path / 'runs' / 'score_all_demo'
    assert len((demo / 'quiz_answers.jsonl').read_text().splitlines()) == len(questions)
    assert json.loads((demo / 'results_gt.json').read_text())['demo'] is True
    assert 'DEMO' in (demo / 'RESULTS_GT.md').read_text()


def test_pair_rule_and_no_test_gallery():
    from freestyle.experiments.score_all import gallery, prepare_rows

    import numpy as np

    rows = [
        {'bbox': [10, 10, 10, 20], 'score': .9, 'class': 'coca-cola-part1',
         'identity_score': .9, 'family': 'coca cola', 'part': 1, 'family_confidence': .9,
         'part_confidence': .9},
        {'bbox': [25, 10, 10, 20], 'score': .9, 'class': 'unknown', 'identity_score': .9,
         'family': 'coca cola', 'part': None, 'family_confidence': .9, 'part_confidence': 0},
    ]
    assert prepare_rows(rows, {'height': 100}, 0, 0, True)[1]['class'] == 'coca-cola-part2'
    assert prepare_rows(rows, {'height': 100}, 0, 0, False)[1]['class'] == 'unknown'
    emb = {'file_names': np.array(['val', 'test']), 'annotation_ids': np.array(['1', '2'])}
    val_gt = {'categories': [{'id': 1, 'name': 'sprite'}],
              'images': [{'id': 1, 'file_name': 'val', 'group_id': 'g'}],
              'annotations': [{'image_id': 1, 'category_id': 1, 'attributes': {'proposal_ann_id': 1}}]}
    assert gallery(emb, val_gt) == ([0], ['sprite'])
    assert gallery(emb, val_gt, 'g') == ([], [])
