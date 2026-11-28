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


def test_val_images_need_only_the_human_count(monkeypatch):
    """The val/tune split is only used to pick thresholds, so counts alone are enough."""
    import copy

    coco = {'images': [
        {'id': 1, 'file_name': 'valimg', 'attributes': {'human_counts': {'total': 3},
                                                        'usable': True, 'verified': False}},
        {'id': 2, 'file_name': 'nocount', 'attributes': {'human_counts': None,
                                                         'usable': True, 'verified': False}},
    ], 'annotations': [
        {'image_id': 1, 'attributes': {'identity_source': 'human'}},
        {'image_id': 1, 'attributes': {'identity_source': 'model'}},
    ], 'info': {}}
    counts = {'images': {'valimg': {}, 'nocount': {}}, 'mismatches': [], 'unverified': []}
    monkeypatch.setattr('freestyle.experiments.score_all.build_coco',
                        lambda *args: (copy.deepcopy(coco), copy.deepcopy(counts)))
    questions = [{'qid': 'count:valimg', 'image_id': 'valimg', 'kind': 'count'},
                 *[{'qid': f'crop:valimg:{i}', 'image_id': 'valimg', 'kind': 'crop'} for i in range(7)],
                 {'qid': 'count:nocount', 'image_id': 'nocount', 'kind': 'count'}]
    answers = {'count:valimg': {'result': 'correct'}, 'crop:valimg:3': {'result': 'skip'}}
    queue = {'questions': questions}

    result, result_counts, excluded = build_human_gt(queue, answers, 'val')
    assert [image['file_name'] for image in result['images']] == ['valimg']
    assert list(result_counts['images']) == ['valimg']
    assert len(result['annotations']) == 1                    # model identities are never GT
    assert [(x['image_id'], x['reason']) for x in excluded] == [('nocount', 'missing_count')]

    # the same answers are not enough for the eval split, which keeps the 80% crop rule
    result, _counts, excluded = build_human_gt(queue, answers, 'test')
    assert result['images'] == []
    assert [(x['image_id'], x['reason'], x['answered']) for x in excluded] == [
        ('valimg', 'below_80_percent', 1), ('nocount', 'missing_count', 0)]


def test_count_only_val_scores_and_reports_val_crops(tmp_path):
    """Answered val counts must score end to end; val-crop galleries degrade, not fail."""
    import json

    import numpy as np

    def save(path, obj):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj))

    images, questions, records, proposals, audit = [], [], [], [], []
    predictions = {'val': [], 'test': []}
    split_images = {'local/train.jpg': 'train'}
    groups = {'local/train.jpg': 'train'}
    for idx, (split, name) in enumerate([('val', 'local/val1.jpg'), ('test', 'local/t1.jpg'),
                                         ('test', 'local/t2.jpg')], 1):
        images.append({'image_id': name, 'coco_image_id': idx, 'split': split, 'width': 100,
                       'height': 100, 'model_top': 1, 'model_bottom': 0, 'model_total': 1})
        split_images[name] = split
        groups[name] = split
        audit.append({'image_id': name, 'scene_label': 'cabinet_full', 'motion_blur': 'none'})
        proposals.append({'id': idx, 'image_id': idx, 'bbox': [10, 10, 10, 20], 'category_id': 1,
                          'attributes': {'id_mode': 'two_stage', 'id_confidence': .9}})
        predictions[split].append({'image_id': idx, 'bbox': [10, 10, 10, 20], 'category_id': 1,
                                   'score': .9, 'proposed_class': 'sprite', 'id_score': .9})
        purpose = 'tune' if split == 'val' else 'eval'
        questions.append({'qid': f'count:{name}', 'image_id': name, 'kind': 'count', 'purpose': purpose})
        records.append({'qid': f'count:{name}', 'answer': {'result': 'correct'}})
        # val keeps its crops unanswered (2 of them, so coverage is 1/3), test answers everything
        for crop in range(2 if split == 'val' else 1):
            qid = f'crop:{name}:{crop}'
            questions.append({'qid': qid, 'image_id': name, 'kind': 'crop', 'purpose': purpose,
                              'ann_id': idx * 10 + crop, 'bbox': [10, 10, 20, 30], 'guess': 'sprite'})
            if split == 'test':
                records.append({'qid': qid, 'answer': {'result': 'correct'}})
    proposals.append({'id': 4, 'image_id': 4, 'bbox': [10, 10, 10, 20], 'category_id': 1,
                      'attributes': {'source': 'vlm:bootstrap', 'id_confidence': .9}})
    queue = {'version': 1, 'classes': [{'id': 1, 'name': 'sprite'}, {'id': 2, 'name': 'unknown'}],
             'images': images, 'questions': questions}
    queue_path = tmp_path / 'queue.json'
    answer_path = tmp_path / 'answers.jsonl'
    save(queue_path, queue)
    answer_path.write_text(''.join(json.dumps(r) + '\n' for r in records))
    save(tmp_path / 'splits.json', {'images': split_images, 'groups': groups})
    save(tmp_path / 'audit.json', {'images': audit})
    save(tmp_path / 'proposals.json', {'images': [{'id': i + 1, 'file_name': im['image_id']}
                                                  for i, im in enumerate(images)]
                                                 + [{'id': 4, 'file_name': 'local/train.jpg'}],
                                       'annotations': proposals, 'categories': queue['classes']})
    run = tmp_path / 'runs' / 'C_fake'
    save(run / 'meta.json', {'config': {'family': 'C_det_agnostic'}})
    for split in predictions:
        save(run / split / 'predictions.json', predictions[split])
    np.savez(run / 'embeddings.npz', embeddings=np.ones((4, 2)),
             file_names=np.array([im['image_id'] for im in images] + ['local/train.jpg']),
             annotation_ids=np.array(['1', '2', '3', '4']), bboxes=np.array([[10, 10, 10, 20]] * 4))
    out = tmp_path / 'RESULTS_GT.md'
    assert main(['--queue', str(queue_path), '--answers', str(answer_path), '--runs', str(run),
                 '--splits', str(tmp_path / 'splits.json'), '--audit', str(tmp_path / 'audit.json'),
                 '--proposals', str(tmp_path / 'proposals.json'), '--out', str(out),
                 '--bootstrap', '20']) == 0
    data = json.loads((tmp_path / 'results_gt.json').read_text())
    assert data['excluded']['val'] == []                       # count-only val images are kept
    assert {r['identifier'] for r in data['rows']} == {'none', 'vlm', 'knn_trainvlm', 'knn_trainvlm+val',
                                                       'linear_trainvlm', 'linear_trainvlm+val'}
    assert [{'run': x['run'], 'identifier': x['identifier'], 'reason': x['reason']}
            for x in data['skipped_runs'] if 'insufficient' in x['reason']] == [
        {'run': 'C_fake', 'identifier': 'knn', 'reason': 'insufficient val crops'},
        {'run': 'C_fake', 'identifier': 'proto', 'reason': 'insufficient val crops'}]
    assert {x.get('identifier') for x in data['skipped_runs']} == {'knn', 'proto'}
    assert all(x['reason'] == 'insufficient val crops' for x in data['skipped_runs'])
    assert all(row['val_gallery_size'] == 0 for row in data['rows'])
    assert any('No answered val crops' in note for note in data['limitations'])
    for row in data['rows']:
        assert row['threshold_source'] == 'val'
        assert row['subsets']['all']['n_images'] == 2
        assert row['subsets']['all']['metrics']['total_error'] == 0
        assert row['subsets']['all']['bootstrap_ci95']['n_groups'] == 2
    assert 'unanswered validation crops just shrink' in out.read_text()


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
    split_images['local/train.jpg'] = 'train'
    groups['local/train.jpg'] = 'train'
    proposals.append({'id': 6, 'image_id': 6, 'bbox': [10, 10, 10, 20], 'category_id': 1,
                      'attributes': {'source': 'vlm:bootstrap', 'id_confidence': .9}})
    queue = {'version': 1, 'classes': [{'id': 1, 'name': 'sprite'}, {'id': 2, 'name': 'unknown'}],
             'images': images, 'questions': questions}
    queue_path = tmp_path / 'queue.json'
    answer_path = tmp_path / 'answers.jsonl'
    save(queue_path, queue)
    answer_path.write_text(''.join(json.dumps(r) + '\n' for r in records))
    save(tmp_path / 'splits.json', {'images': split_images, 'groups': groups})
    save(tmp_path / 'audit.json', {'images': audit})
    save(tmp_path / 'proposals.json', {'images': [{'id': i + 1, 'file_name': im['image_id']}
                                                  for i, im in enumerate(images)]
                                                 + [{'id': 6, 'file_name': 'local/train.jpg'}],
                                       'annotations': proposals, 'categories': queue['classes']})
    run = tmp_path / 'runs' / 'D_fake'
    save(run / 'meta.json', {'config': {'family': 'D_det_multiclass'}})
    for split in predictions:
        save(run / split / 'predictions.json', predictions[split])
    np.savez(run / 'embeddings.npz', embeddings=np.ones((6, 2)),
             file_names=np.array([im['image_id'] for im in images] + ['local/train.jpg']),
             annotation_ids=np.array([str(i) for i in range(1, 7)]),
             bboxes=np.array([[10, 10, 10, 20]] * 6))
    out = tmp_path / 'RESULTS_GT.md'
    assert main(['--queue', str(queue_path), '--answers', str(answer_path), '--runs', str(run),
                 '--splits', str(tmp_path / 'splits.json'), '--audit', str(tmp_path / 'audit.json'),
                 '--proposals', str(tmp_path / 'proposals.json'), '--out', str(out), '--bootstrap', '20']) == 0
    data = json.loads((tmp_path / 'results_gt.json').read_text())
    assert len(data['rows']) == 18
    assert {r['identifier'] for r in data['rows']} == {'none', 'own', 'knn', 'proto', 'vlm', 'knn_trainvlm',
                                                   'knn_trainvlm+val', 'linear_trainvlm',
                                                   'linear_trainvlm+val'}
    assert data['excluded']['test'][0]['image_id'] == 'local/5.jpg'
    for row in data['rows']:
        assert row['threshold_source'] == 'val'
        if 'trainvlm' in row['identifier']:
            assert row['train_vlm_confidence_thr'] in (0., .5, .75, .9)
            assert row['train_gallery_size'] == 1
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


def test_train_gallery_selection_and_validation_calibration():
    import numpy as np

    from freestyle.experiments.score_all import identify, select_thresholds, train_gallery

    names = ['train'] * 4 + ['val', 'test']
    emb = {'file_names': np.array(names), 'annotation_ids': np.array([str(i) for i in range(6)]),
           'embeddings': np.ones((6, 2)), 'bboxes': np.array([[0, 0, 10, 10]] * 6)}
    proposals = {'train': [{'id': i, 'class': 'wrong' if i < 3 else 'correct',
                            'attributes': {'source': 'vlm:bootstrap',
                                           'id_confidence': .4 if i < 3 else .9}}
                           for i in range(4)],
                 'test': [{'id': 5, 'class': 'wrong',
                           'attributes': {'source': 'vlm:bootstrap', 'id_confidence': 1.}}]}
    splits = {'images': {'train': 'train', 'val': 'val', 'test': 'test'}}
    assert train_gallery(emb, proposals, splits, 0)[0] == [0, 1, 2, 3]
    assert train_gallery(emb, proposals, splits, .5) == ([3], ['correct'])
    image = {'id': 1, 'file_name': 'val', 'group_id': 'vg', 'height': 100,
             'attributes': {'human_counts': {'total': 1}}}
    gt = {'images': [image], 'annotations': [], 'categories': []}
    counts = {'images': {'val': {'classes': {'correct': 1}}}}
    pred = [{'bbox': [0, 0, 10, 10], 'score': .9}]
    outcomes = {}
    for cutoff in (0., .5):
        items = train_gallery(emb, proposals, splits, cutoff)
        rows = identify(pred, image, 'knn_trainvlm', proposals, emb, gt,
                        validation=True, train_items=items)
        outcomes[cutoff] = select_thresholds([image], gt, counts, {1: rows}, False, False)
    assert outcomes[.5][:2] < outcomes[0.][:2]


def test_pair_derived_family_and_threshold():
    from freestyle.experiments.score_all import prepare_rows

    rows = [{'bbox': [x, 10, 10, 20], 'score': .9, 'class': 'coca-cola-part1',
             'identity_score': .6} for x in (10, 25)]
    assert [r['class'] for r in prepare_rows(rows, {'height': 100}, 0, .5, True)] == [
        'coca-cola-part1', 'coca-cola-part2']
    assert all(r['class'] == 'unknown' for r in prepare_rows(rows, {'height': 100}, 0, .75, True))


def test_negative_bypasses_completion_cutoff(monkeypatch):
    image = {'id': 1, 'file_name': 'negative', 'attributes': {
        'human_counts': {'total': 0}, 'usable': False, 'verified': True}}
    coco = {'images': [image], 'annotations': [], 'info': {}}
    counts = {'images': {'negative': {}}, 'mismatches': [], 'unverified': []}
    monkeypatch.setattr('freestyle.experiments.score_all.build_coco', lambda *args: (coco, counts))
    queue = {'questions': [{'qid': str(i), 'image_id': 'negative'} for i in range(6)]}
    assert build_human_gt(queue, {'0': {'result': 'not_cabinet'}}, 'test')[0]['images'] == [image]


def test_incomplete_class_metrics_are_excluded():
    from freestyle.experiments.score_all import aggregate, per_image

    image = {'attributes': {'human_counts': {'total': 2, 'top': 2, 'bottom': 0}}}
    counts = {'classes': {'sprite': 1}, 'verified': False, 'counts_mismatch': True}
    record = per_image(image, [], [], counts, {}, False)
    assert record['total_error'] == 2
    assert record['class_errors'] is None
    assert aggregate([record], [0], ['sprite'], False)['class:sprite'] is None


def test_generic_and_predicted_only_class():
    from freestyle.experiments.score_all import aggregate, per_image, prepare_rows

    rows = [{'bbox': [0, 0, 10, 10], 'score': .9, 'class': 'cartridge', 'identity_score': .9}]
    rows = prepare_rows(rows, {'height': 100}, 0, 0, False)
    image = {'attributes': {'human_counts': {'total': 1, 'top': 1, 'bottom': 0}}}
    record = per_image(image, [], rows, {'classes': {}, 'verified': True, 'counts_mismatch': False}, {}, False)
    assert aggregate([record], [0], list(record['class_errors']), False)['class:unknown'] == 1


def test_threshold_ties_choose_middle():
    from freestyle.experiments.score_all import select_thresholds

    image = {'id': 1, 'file_name': 'a', 'height': 100, 'attributes': {'human_counts': {'total': 0}}}
    counts = {'images': {'a': {'classes': {}, 'verified': True, 'counts_mismatch': False}}}
    assert select_thresholds([image], {}, counts, {1: []}, False, True)[2] == .45


def test_matching_prefers_detection_score():
    from freestyle.experiments.score_all import per_image

    image = {'attributes': {'human_counts': {'total': 1, 'top': 1, 'bottom': 0}}}
    rows = [{'bbox': [0, 0, 10, 10], 'score': .2, 'class': 'wrong', 'shelf_name': 'top'},
            {'bbox': [1, 0, 10, 10], 'score': .9, 'class': 'sprite', 'shelf_name': 'top'}]
    record = per_image(image, [{'bbox': [0, 0, 10, 10], 'category_id': 1}], rows,
                       {'classes': {'sprite': 1}, 'verified': True, 'counts_mismatch': False}, {1: 'sprite'}, False)
    assert record['id_correct'] == 1


def test_prediction_keys_and_invalid_geometry():
    from freestyle.experiments.score_all import grouped_predictions

    rows = [{'image_id': 'a.jpg', 'bbox': [0, 0, 1, 1]},
            {'image_id': 'a.jpg', 'bbox': [0, 0, 0, 1]}]
    assert grouped_predictions(rows)['a.jpg'] == rows[:1]


def test_audit_subsets_exclude_verified_negative():
    from freestyle.experiments.score_all import audit_subsets

    negative = {'file_name': 'negative', 'attributes': {'usable': False}}
    sharp, partial = audit_subsets([negative], {'negative': {'motion_blur': 'severe'}})
    assert sharp == partial == []


# --------------------------------------------------------------------------- vlm_flash


def _fake_run_fixture(tmp_path, flash_class=None):
    """Two val + two test frames, one answered crop each, one run, no embeddings.

    ``proposals.json`` carries the Inkling two_stage identities (``sprite``, matching the
    human answer); ``proposals_flash.json`` carries the Flash Next ``two_stage_shelf``
    identities for the same boxes (``flash_class``, default ``fanta-orange``).
    """
    import json

    def save(path, obj):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj))

    classes = [{'id': 1, 'name': 'sprite'}, {'id': 2, 'name': 'unknown'},
               {'id': 3, 'name': 'fanta-orange'}]
    flash_class = flash_class or 'fanta-orange'
    flash_id = {'sprite': 1, 'fanta-orange': 3}[flash_class]
    images, questions, records, audit, splits_map, groups = [], [], [], [], {}, {}
    predictions = {'val': [], 'test': []}
    v2, v3 = [], []
    for idx, split in enumerate(['val', 'val', 'test', 'test'], 1):
        name = f'local/{idx}.jpg'
        splits_map[name], groups[name] = split, split
        images.append({'image_id': name, 'coco_image_id': idx, 'split': split, 'width': 100,
                       'height': 100, 'model_top': 1, 'model_bottom': 0, 'model_total': 1})
        audit.append({'image_id': name, 'scene_label': 'cabinet_full', 'motion_blur': 'none'})
        count, crop = f'count:{name}', f'crop:{name}:{idx}'
        questions += [{'qid': count, 'image_id': name, 'kind': 'count'},
                      {'qid': crop, 'image_id': name, 'kind': 'crop', 'ann_id': idx,
                       'bbox': [10, 10, 20, 30], 'guess': 'sprite'}]
        records += [{'qid': count, 'answer': {'result': 'correct'}},
                    {'qid': crop, 'answer': {'result': 'correct'}}]
        v2.append({'id': idx, 'image_id': idx, 'bbox': [10, 10, 10, 20], 'category_id': 1,
                   'attributes': {'id_mode': 'two_stage', 'id_confidence': .9,
                                  'id_family': 'sprite', 'source': 'vlm:inkling-2-mini'}})
        v3.append({'id': idx, 'image_id': idx, 'bbox': [10, 10, 10, 20],
                   'category_id': flash_id,
                   'attributes': {'id_mode': 'two_stage_shelf', 'id_confidence': .8,
                                  'id_family': 'sprite', 'id_part': None,
                                  'source': 'vlm:qwen3-8-flash-next'}})
        predictions[split].append({'image_id': idx, 'bbox': [10, 10, 10, 20], 'category_id': 1,
                                   'score': .9, 'proposed_class': 'sprite', 'id_score': .9})
    # a flash box that matches no predicted box (IoU 0) keeps no identity
    v3.append({'id': 99, 'image_id': 3, 'bbox': [80, 80, 10, 10], 'category_id': 3,
               'attributes': {'id_mode': 'two_stage_shelf', 'id_confidence': .8}})
    coco_images = [{'id': i + 1, 'file_name': im['image_id']} for i, im in enumerate(images)]
    queue_path, answer_path = tmp_path / 'queue.json', tmp_path / 'answers.jsonl'
    save(queue_path, {'version': 1, 'classes': classes, 'images': images,
                      'questions': questions})
    answer_path.write_text(''.join(json.dumps(r) + '\n' for r in records))
    save(tmp_path / 'splits.json', {'images': splits_map, 'groups': groups})
    save(tmp_path / 'audit.json', {'images': audit})
    save(tmp_path / 'proposals.json', {'images': coco_images, 'annotations': v2,
                                       'categories': classes})
    save(tmp_path / 'proposals_flash.json', {'images': coco_images, 'annotations': v3,
                                             'categories': classes})
    run = tmp_path / 'runs' / 'C_fake'
    save(run / 'meta.json', {'config': {'family': 'C_det_agnostic'}})
    for split in predictions:
        save(run / split / 'predictions.json', predictions[split])
    return ['--queue', str(queue_path), '--answers', str(answer_path), '--runs', str(run),
            '--splits', str(tmp_path / 'splits.json'), '--audit', str(tmp_path / 'audit.json'),
            '--proposals', str(tmp_path / 'proposals.json'),
            '--proposals-flash', str(tmp_path / 'proposals_flash.json'), '--bootstrap', '20']


def test_vlm_flash_scores_beside_vlm(tmp_path):
    import json

    argv = _fake_run_fixture(tmp_path) + ['--out', str(tmp_path / 'RESULTS_GT.md')]
    assert main(argv) == 0
    data = json.loads((tmp_path / 'results_gt.json').read_text())
    rows = {(row['identifier'], row['pair_rule']): row for row in data['rows']}
    assert set(rows) == {('none', 'off'), ('none', 'on'), ('vlm', 'off'), ('vlm', 'on'),
                         ('vlm_flash', 'off'), ('vlm_flash', 'on')}

    vlm = rows[('vlm', 'off')]['subsets']['all']['metrics']
    flash = rows[('vlm_flash', 'off')]['subsets']['all']['metrics']
    assert vlm['identity_accuracy'] == 1 and vlm['exact_inventory'] == 1
    assert vlm['total_error'] == flash['total_error'] == 0        # same boxes, same counts
    # the flash reader calls every box fanta-orange, so it loses identity and inventory
    assert flash['identity_accuracy'] == 0
    assert flash['exact_inventory'] == 0
    assert flash['class:fanta-orange'] == 1 and flash['class:sprite'] == 1
    assert any('bootstrap' not in note and 'vlm_flash identities' in note
               for note in data['limitations'])
    for row in data['rows']:
        if row['identifier'] == 'vlm_flash':
            assert row['train_gallery_size'] == 0 and row['threshold_source'] == 'val'


def test_vlm_flash_agrees_when_the_readers_agree(tmp_path):
    import json

    argv = (_fake_run_fixture(tmp_path, flash_class='sprite')
            + ['--out', str(tmp_path / 'RESULTS_GT.md')])
    assert main(argv) == 0
    rows = {(row['identifier'], row['pair_rule']): row
            for row in json.loads((tmp_path / 'results_gt.json').read_text())['rows']}
    for key in (('vlm', 'off'), ('vlm_flash', 'off')):
        assert rows[key]['subsets']['all']['metrics']['identity_accuracy'] == 1
    # with the pair rule on, the family hint still resolves the second box's part
    assert rows[('vlm_flash', 'on')]['subsets']['all']['metrics']['identity_accuracy'] == 1


def test_vlm_flash_requires_its_proposal_file(tmp_path):
    argv = _fake_run_fixture(tmp_path) + ['--out', str(tmp_path / 'RESULTS_GT.md'),
                                          '--proposals-flash', str(tmp_path / 'nope.json')]
    with pytest.raises(SystemExit):
        main(argv)
    assert not (tmp_path / 'RESULTS_GT.md').exists()


def test_vlm_flash_matching_and_iou_gate():
    from freestyle.experiments.score_all import identify

    image = {'id': 1, 'file_name': 'a', 'group_id': 'g', 'height': 100,
             'attributes': {'human_counts': {'total': 1}}}
    pred = [{'bbox': [0, 0, 10, 10], 'score': .9}]
    shelf = {'a': [{'bbox': [0, 0, 10, 10], 'class': 'hi-c',
                    'attributes': {'id_mode': 'two_stage_shelf', 'id_confidence': .7,
                                   'id_family': 'hi c'}}]}
    rows = identify(pred, image, 'vlm_flash', {}, None, {}, flash_proposals=shelf)
    assert (rows[0]['class'], rows[0]['identity_score'], rows[0]['family']) == ('hi-c', .7, 'hi c')

    # 'vlm' stays strict: it does not borrow the shelf-mode replies of the other reader
    assert identify(pred, image, 'vlm', shelf, {}, None)[0]['class'] == 'unknown'

    # and vlm_flash needs a real box match: IoU 0 leaves the prediction unidentified
    far = {'a': [{'bbox': [60, 60, 10, 10], 'class': 'hi-c',
                  'attributes': {'id_mode': 'two_stage_shelf', 'id_confidence': .7}}]}
    assert identify(pred, image, 'vlm_flash', {}, None, {}, flash_proposals=far)[0]['class'] == 'unknown'


def _without(argv, flag):
    """Drop one ``--flag VALUE`` pair from an argv list (demo mode synthesizes answers)."""
    out = list(argv)
    index = out.index(flag)
    del out[index:index + 2]
    return out


def test_demo_works_with_and_without_the_flash_file(tmp_path, monkeypatch):
    import json

    argv = _fake_run_fixture(tmp_path) + ['--out', str(tmp_path / 'RESULTS_GT.md')]
    monkeypatch.chdir(tmp_path)
    demo = tmp_path / 'runs' / 'score_all_demo'

    # no --proposals-flash (how finalize_after_gt.sh behaves while the file is missing)
    no_flash = _without(argv, '--proposals-flash')
    assert main(['--demo', *_without(no_flash, '--answers')]) == 0
    assert {row['identifier'] for row in
            json.loads((demo / 'results_gt.json').read_text())['rows']} == {'none', 'vlm'}

    assert main(['--demo', *_without(argv, '--answers')]) == 0
    data = json.loads((demo / 'results_gt.json').read_text())
    assert data['demo'] is True
    assert {row['identifier'] for row in data['rows']} == {'none', 'vlm', 'vlm_flash'}
    assert len((demo / 'quiz_answers.jsonl').read_text().splitlines()) == 8
    assert 'DEMO' in (demo / 'RESULTS_GT.md').read_text()

