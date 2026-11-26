import pytest

from freestyle.count.pairs import infer_pairs


def boxes(families, **extra):
    return [dict(shelf_idx=0, column_idx=i, tier=0, family=family,
                 family_confidence=.9, part=None, part_confidence=0, **extra)
            for i, family in enumerate(families)]


def parts(result):
    return [d['part_inferred'] for d in result]


def test_even_report_apply_and_off():
    source = boxes(['coca cola'] * 4, **{'class': 'unknown'})
    assert parts(infer_pairs(source)) == [1, 2, 1, 2]
    assert [d['class'] for d in infer_pairs(source, 'apply')] == [
        'coca-cola-part1', 'coca-cola-part2', 'coca-cola-part1', 'coca-cola-part2']
    assert infer_pairs(source, 'off') == source
    assert 'part_inferred' not in source[0]


def test_anchor_shift_and_conflict():
    source = boxes(['diet coke'] * 4)
    source[0].update(part=2, part_confidence=.9)
    assert parts(infer_pairs(source)) == [2, 1, 2, 1]
    source[1].update(part=2, part_confidence=.9)
    result = infer_pairs(source, 'apply')
    assert parts(result) == [None] * 4
    assert all('conflicting_anchors' in d['flags'] for d in result)


def test_odd_run_only_anchored_complete_pairs():
    source = boxes(['dr pepper'] * 5)
    assert parts(infer_pairs(source)) == [None] * 5
    source[2].update(part=1, part_confidence=.9)
    result = infer_pairs(source)
    assert parts(result) == [None, None, 1, 2, None]
    assert all('odd_run' in d['flags'] for d in result)


def test_families_tiers_gaps_and_flip():
    source = boxes(['coca cola', 'coca cola', 'diet coke', 'diet coke'])
    source += [dict(source[0], tier=1), dict(source[1], tier=1)]
    result = infer_pairs(source, upside_down=True)
    assert parts(result) == [2, 1, 2, 1, 2, 1]
    assert len({d['run_id'] for d in result}) == 3
    source[1]['column_idx'] = 4
    assert infer_pairs(source)[0]['flags'] == ['odd_run']


def test_unknown_confidence_and_validation():
    source = boxes(['coca cola'] * 2)
    source[0]['family_confidence'] = .1
    result = infer_pairs(source)
    assert result[0]['run_id'] is None and result[1]['flags'] == ['odd_run']
    with pytest.raises(ValueError):
        infer_pairs(source, 'invalid')
