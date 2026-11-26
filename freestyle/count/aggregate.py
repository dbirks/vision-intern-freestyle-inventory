"""Count detections and combine counts from a stationary inventory burst."""

from collections import Counter

import numpy as np


def aggregate_counts(detections, thr=0.0, identity_thr=0.5, neighbour_vote='off'):
    """Return counts, unknown count and review flags; optionally apply neighbour votes.

    A vote requires known immediate neighbours on both sides in a row, or two
    other tiers in the same column. Only detections passing ``thr`` participate.
    """
    if neighbour_vote not in ('off', 'report', 'apply'):
        raise ValueError('neighbour_vote must be off, report, or apply')
    active = [(i, dict(d)) for i, d in enumerate(detections) if d['score'] >= thr]
    flags = []
    for index, det in active:
        if neighbour_vote == 'off' or det.get('id_score', 0) >= identity_thr:
            continue
        row = [d for _, d in active if d.get('shelf_idx') == det.get('shelf_idx')]
        col = det.get('column_idx')
        if col is None or det.get('shelf_idx') is None:
            continue
        left = [d for d in row if d.get('column_idx') == col - 1]
        right = [d for d in row if d.get('column_idx') == col + 1]
        peers = [d for d in row if d is not det and d.get('column_idx') == col]
        votes = left + right if left and right else peers if len(peers) >= 2 else []
        classes = {d.get('class', 'unknown') for d in votes
                   if d.get('id_score', 0) >= identity_thr}
        if votes and len(classes) == 1 and all(d.get('id_score', 0) >= identity_thr for d in votes):
            suggestion = classes.pop()
            if suggestion != 'unknown':
                flags.append({'type': 'neighbour_vote', 'detection': index, 'suggested_class': suggestion})
                if neighbour_vote == 'apply':
                    det['class'] = suggestion
    counts = Counter(d.get('class', 'unknown') for _, d in active)
    unknown = counts.pop('unknown', 0)
    return dict(counts), unknown, flags


def fuse_counts(list_of_per_frame_counts, weights=None, method='max'):
    """Fuse per-class counts without summing frames of the same door opening.

    Max and median do not add repeated observations across frames, but can still
    over-count duplicates within a frame. Max can preserve false positives;
    median under-counts when most frames show only a partial view. Weighted mode
    selects the most supported observed count, breaking ties toward the lower one.
    """
    if method not in ('max', 'median', 'weighted_mode'):
        raise ValueError('unknown fusion method')
    frames = list(list_of_per_frame_counts)
    w = np.ones(len(frames)) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (len(frames),) or not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError('weights must be finite, nonnegative and match frames')
    if frames and not w.any():
        raise ValueError('at least one weight must be positive')
    result = {}
    for cls in sorted(set().union(*(f.keys() for f in frames))):
        values = [f.get(cls, 0) for f in frames]
        if any(not isinstance(v, (int, np.integer)) or v < 0 for v in values):
            raise ValueError('counts must be nonnegative integers')
        if method == 'max':
            result[cls] = max(values)
        elif method == 'median':
            result[cls] = float(np.median(values))
        else:
            totals = {v: sum(weight for value, weight in zip(values, w) if value == v)
                      for v in set(values)}
            result[cls] = min(totals, key=lambda v: (-totals[v], v))
    return result
