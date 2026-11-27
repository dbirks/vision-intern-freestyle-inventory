import pytest

from freestyle.count import aggregate_counts, assign_shelves, fuse_counts, suspicious_gaps, suspicious_overlaps


def test_curved_rows_stack_and_flags():
    boxes = []
    for y in (100, 220):
        for x in (0, 30, 60, 90, 120, 150):
            bottom = y + .003 * (x - 75) ** 2
            boxes.append({'box': [x, bottom - 20, x + 20, bottom], 'score': 1})
    boxes.append({'box': [60, boxes[2]['box'][1] - 20, 80, boxes[2]['box'][1]], 'score': 1})
    result = assign_shelves(boxes)
    assert len({d['shelf_idx'] for d in result}) == 2
    assert result[2]['shelf_idx'] == result[-1]['shelf_idx']
    assert result[2]['column_idx'] == result[-1]['column_idx']
    assert {result[2]['tier'], result[-1]['tier']} == {0, 1}
    assert suspicious_gaps([[0, 0, 10, 10], [30, 0, 40, 10]]) == [(0, 1)]
    assert suspicious_overlaps([[0, 0, 10, 10], [1, 1, 11, 11]]) == [(0, 1)]


def test_aggregate_and_fusion():
    detections = [dict(score=1, **{'class': c}, id_score=p, shelf_idx=0, column_idx=i)
                  for i, (c, p) in enumerate([('a', 1), ('unknown', .1), ('a', 1)])]
    assert aggregate_counts(detections)[1] == 1
    counts, unknown, flags = aggregate_counts(detections, neighbour_vote='report')
    assert counts == {'a': 2} and unknown == 1 and flags[0]['suggested_class'] == 'a'
    assert aggregate_counts(detections, neighbour_vote='apply')[:2] == ({'a': 3}, 0)
    frames = [{'a': 2}, {'a': 3}, {'a': 3}]
    assert fuse_counts(frames) == {'a': 3}
    assert fuse_counts(frames, method='median') == {'a': 3}
    assert fuse_counts(frames, weights=[5, 1, 1], method='weighted_mode') == {'a': 2}
    with pytest.raises(ValueError):
        fuse_counts(frames, weights=[-1, 1, 1])


def test_two_shelves_with_a_two_high_stack_stay_two_rows():
    """A stacked NNS column is a tier, not a third shelf row."""
    boxes = []
    for x in (0, 30, 60, 90, 120):                       # bottom shelf
        boxes.append({'box': [x, 300, x + 25, 380], 'score': 1})
    for x in (0, 30, 60, 90, 120):                       # top shelf
        boxes.append({'box': [x, 100, x + 25, 180], 'score': 1})
    boxes.append({'box': [30, 215, 55, 295], 'score': 1})  # second cartridge stacked up
    two = assign_shelves(boxes, max_rows=2)
    assert len({d['shelf_idx'] for d in two}) == 2
    stacked = [two[1], two[-1]]                          # same column, two tiers
    assert stacked[0]['shelf_idx'] == stacked[1]['shelf_idx']
    assert {stacked[0]['tier'], stacked[1]['tier']} == {0, 1}
    three = assign_shelves(boxes, max_rows=4)
    assert len({d['shelf_idx'] for d in three}) >= len({d['shelf_idx'] for d in two})
    with pytest.raises(ValueError):
        assign_shelves(boxes, max_rows=0)
