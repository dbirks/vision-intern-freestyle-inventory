"""Infer two-cartridge parts from shelf order.

``infer_pairs`` accepts the shelf/column/tier fields from ``assign_shelves``
(or their public ``shelf``/``column`` aliases). Family and part confidences
are independent probabilities; an absent confidence is treated as unknown.
Results are copies in input order. Odd runs only resolve complete pairs that
contain a confident part reading, since the missing box's position is unknown.
"""

import re


_FAMILIES = {
    'coca cola': 'coca-cola',
    'coke zero': 'coke-zero',
    'coca cola zero': 'coke-zero',
    'diet coke': 'diet-coke',
    'dr pepper': 'dr-pepper',
}


def _family(value):
    return _FAMILIES.get(re.sub(r'[^a-z0-9]+', ' ', str(value).lower()).strip())


def _confident(value, threshold):
    try:
        return threshold <= float(value) <= 1
    except (TypeError, ValueError):
        return False


def infer_pairs(detections, mode='report', upside_down=False, family_threshold=0.5,
                part_threshold=0.5):
    """Return annotated copies of one image's detections.

    Inputs use ``family``, ``part``, ``family_confidence`` and
    ``part_confidence``. ``id_family`` and ``id_part`` are also accepted.
    Confident families delimit runs; columns must be consecutive within a tier.
    In apply mode only boxes with an unknown part have their class changed.
    Flags and reasons are per-box strings/lists, and run IDs are image-local.
    """
    if mode not in ('off', 'report', 'apply'):
        raise ValueError('mode must be off, report, or apply')
    if not 0 <= family_threshold <= 1 or not 0 <= part_threshold <= 1:
        raise ValueError('confidence thresholds must be between 0 and 1')
    result = [dict(d) for d in detections]
    if mode == 'off':
        return result
    rows = {}
    for i, d in enumerate(result):
        d.update(part_inferred=None, rule_reason='ineligible', run_id=None, flags=[])
        shelf = d.get('shelf_idx', d.get('shelf'))
        column = d.get('column_idx', d.get('column'))
        if shelf is None or column is None:
            raise ValueError('each detection requires shelf and column assignments')
        rows.setdefault((shelf, d.get('tier', 0)), []).append((column, i))

    run_number = 0
    for row in sorted(rows):
        entries = sorted(rows[row], reverse=upside_down)
        runs = []
        current = []
        previous_column = None
        previous_family = None
        for column, i in entries:
            d = result[i]
            family = _family(d.get('family', d.get('id_family', '')))
            confidence = d.get('family_confidence', d.get('id_family_confidence'))
            if not _confident(confidence, family_threshold):
                family = None
            adjacent = previous_column is not None and abs(column - previous_column) == 1
            if current and (family != previous_family or not adjacent):
                runs.append(current)
                current = []
            if family:
                current.append((i, family))
            previous_column, previous_family = column, family
        if current:
            runs.append(current)
        for run in runs:
            run_number += 1
            anchors = []
            for pos, (i, _) in enumerate(run):
                d = result[i]
                part = d.get('part', d.get('id_part'))
                confidence = d.get('part_confidence', d.get('id_part_confidence'))
                if not isinstance(part, bool) and part in (1, 2) and _confident(confidence, part_threshold):
                    anchors.append((pos, part))
            phases = {(part - 1 - pos) % 2 for pos, part in anchors}
            conflict = len(phases) > 1
            odd = len(run) % 2 == 1
            phase = next(iter(phases), 0)
            anchored = {pos for pos, _ in anchors}
            for pos, (i, family) in enumerate(run):
                d = result[i]
                d['run_id'] = run_number
                d['flags'] = (['odd_run'] if odd else []) + (['conflicting_anchors'] if conflict else [])
                part = (pos + phase) % 2 + 1
                partner = pos + 1 if part == 1 else pos - 1
                complete = 0 <= partner < len(run)
                if conflict:
                    d['rule_reason'] = 'conflicting_anchors'
                elif odd and not (complete and (pos in anchored or partner in anchored)):
                    d['rule_reason'] = 'odd_run_unresolved'
                else:
                    d['part_inferred'] = part
                    d['rule_reason'] = 'anchored_pair' if odd else ('anchored_alternation' if anchors else 'alternation')
                    existing = d.get('part', d.get('id_part'))
                    if mode == 'apply' and existing not in (1, 2):
                        d['class'] = f'{family}-part{part}'
    return result
