"""Heuristic fisheye shelf grouping, using robust quadratic row fits."""

import itertools

import numpy as np


def _box(item):
    return np.asarray(item['box'] if isinstance(item, dict) else item, dtype=float)


def _fit(x, y, threshold):
    degree = min(2, len(x) - 1, len(np.unique(x)) - 1)
    if degree <= 0:
        return np.array([np.median(y)]), np.ones(len(x), dtype=bool)
    best = None
    samples = list(itertools.combinations(range(len(x)), degree + 1))
    if len(samples) > 300:
        rng = np.random.default_rng(0)
        samples = [samples[i] for i in rng.choice(len(samples), 300, replace=False)]
    for sample in samples:
        if len(np.unique(x[list(sample)])) <= degree:
            continue
        coef = np.polyfit(x[list(sample)], y[list(sample)], degree)
        residual = np.abs(y - np.polyval(coef, x))
        mask = residual <= threshold
        key = (int(mask.sum()), -float(np.minimum(residual, threshold).sum()))
        if best is None or key > best[0]:
            best = (key, mask, coef)
    if best is None:
        return np.array([np.median(y)]), np.ones(len(x), dtype=bool)
    mask = best[1]
    coef = np.polyfit(x[mask], y[mask], degree) if mask.sum() > degree else best[2]
    return coef, mask


def assign_shelves(detections, max_rows=4):
    """Return copies with zero-based shelf, column and top-to-bottom tier indices.

    Stack candidates are collapsed before fitting; BIC selects up to ``max_rows``
    quadratic bands. Small datasets use lower-degree fits to avoid singular models.
    """
    if not 1 <= max_rows <= 4:
        raise ValueError('max_rows must be between 1 and 4')
    result = [dict(d) for d in detections]
    if not result:
        return result
    boxes = np.array([_box(d) for d in result])
    if not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError('boxes must be finite and nondegenerate')
    heights = boxes[:, 3] - boxes[:, 1]
    parent = list(range(len(boxes)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in itertools.combinations(range(len(boxes)), 2):
        a, b = boxes[i], boxes[j]
        overlap = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        ratio = overlap / min(a[2] - a[0], b[2] - b[0])
        vertical_gap = max(a[1], b[1]) - min(a[3], b[3])
        if ratio > .5 and -.25 * min(heights[i], heights[j]) <= vertical_gap <= .35 * max(heights[i], heights[j]):
            parent[root(j)] = root(i)
    groups = {}
    for i in range(len(boxes)):
        groups.setdefault(root(i), []).append(i)
    stacks = list(groups.values())
    x = np.array([np.mean((boxes[g, 0] + boxes[g, 2]) / 2) for g in stacks])
    y = np.array([max(boxes[g, 3]) for g in stacks])
    scale = max(float(np.median(heights)) * .2, 1.)
    # Iterative robust curve extraction followed by reassignment; compare BIC.
    candidates = []
    for k in range(1, min(max_rows, len(stacks)) + 1):
        remaining = np.arange(len(stacks))
        models = []
        for _ in range(k - 1):
            coef, mask = _fit(x[remaining], y[remaining], scale)
            if not mask.any() or mask.all():
                break
            models.append(coef)
            remaining = remaining[~mask]
        if len(models) != k - 1 or not len(remaining):
            continue
        models.append(_fit(x[remaining], y[remaining], scale)[0])
        for _ in range(5):
            residual = np.array([abs(y - np.polyval(c, x)) for c in models])
            labels = residual.argmin(axis=0)
            if len(set(labels)) != k:
                break
            models = [_fit(x[labels == i], y[labels == i], scale)[0] for i in range(k)]
        if len(set(labels)) != k:
            continue
        errors = np.array([y[i] - np.polyval(models[labels[i]], x[i]) for i in range(len(x))])
        # Noise floor prevents exact fits from rewarding singleton rows.
        rss = np.sum(errors ** 2) + len(x) * scale ** 2
        params = sum(len(c) for c in models) + k - 1
        bic = len(x) * np.log(rss / len(x)) + params * np.log(max(len(x), 2))
        candidates.append((bic, labels, models))
    _, labels, models = min(candidates, key=lambda v: v[0])
    order = sorted(range(len(models)), key=lambda i: np.polyval(models[i], np.median(x)))
    for shelf, label in enumerate(order):
        members = sorted(np.flatnonzero(labels == label), key=lambda i: x[i])
        for column, idx in enumerate(members):
            for tier, box_idx in enumerate(sorted(stacks[idx], key=lambda i: boxes[i, 1])):
                result[box_idx].update(shelf_idx=shelf, column_idx=column, tier=tier)
    return result


def suspicious_gaps(row, factor=1.6):
    """Return pairs of indices separated by unusually wide edge-to-edge gaps."""
    boxes = [_box(d) for d in row]
    if len(boxes) < 2:
        return []
    width = np.median([b[2] - b[0] for b in boxes])
    order = sorted(range(len(boxes)), key=lambda i: boxes[i][0])
    return [(a, b) for a, b in zip(order, order[1:])
            if boxes[b][0] - boxes[a][2] > factor * width]


def suspicious_overlaps(row, iou_threshold=0.5):
    """Return index pairs with high IoU, potentially duplicate detections."""
    boxes = [_box(d) for d in row]
    pairs = []
    for i, j in itertools.combinations(range(len(boxes)), 2):
        a, b = boxes[i], boxes[j]
        inter = np.prod(np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])))
        union = np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - inter
        if union > 0 and inter / union >= iou_threshold:
            pairs.append((i, j))
    return pairs
