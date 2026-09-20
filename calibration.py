"""Offline temperature fitting/evaluation, adapted from open-alternative-jev (Apache-2.0).

See NOTICE.md. This tool never modifies serving probabilities or server settings.
"""
import argparse
import json
import math
from pathlib import Path


# Search grid and zero-probability floor follow the pinned upstream calibration.py.
DEFAULT_GRID = tuple(x / 20 for x in range(4, 81))
PROBABILITY_FLOOR = 1e-30


def _temperature(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError('temperature must be finite and positive')
    return value


def _rows(probabilities, correct=None):
    rows = [list(row) for row in probabilities]
    if not rows:
        raise ValueError('probabilities must not be empty')
    for row in rows:
        if len(row) < 2 or any(isinstance(p, bool) or not isinstance(p, (int, float))
                               or not math.isfinite(p) or not 0 <= p <= 1 for p in row):
            raise ValueError('each distribution needs at least two finite probabilities')
        if not math.isclose(sum(row), 1., rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError('probabilities must sum to one')
    if correct is not None:
        if len(rows) != len(correct) or any(type(y) is not int or not 0 <= y < len(row)
                                          for row, y in zip(rows, correct)):
            raise ValueError('correct indices must match distributions')
    return rows


def _scale(row, temperature):
    logits = [math.log(max(p, PROBABILITY_FLOOR)) for p in row]
    peak = max(logits)
    values = [math.exp((p - peak) / temperature) for p in logits]
    total = sum(values)
    return [p / total for p in values]


def _nll(rows, correct):
    return -sum(math.log(max(row[y], PROBABILITY_FLOOR)) for row, y in zip(rows, correct)) / len(rows)


def metrics(probabilities, correct, bins=10):
    """Accuracy, negative log likelihood and equal-width expected calibration error."""
    rows = _rows(probabilities, correct)
    if type(bins) is not int or bins < 1:
        raise ValueError('bins must be a positive integer')
    buckets = [[] for _ in range(bins)]
    hits = []
    for row, y in zip(rows, correct):
        conf = max(row)
        hit = max(range(len(row)), key=row.__getitem__) == y
        hits.append(hit)
        buckets[min(int(conf * bins), bins - 1)].append((conf, hit))
    ece = sum(abs(sum(c - hit for c, hit in bucket)) for bucket in buckets) / len(rows)
    return {'accuracy': sum(hits) / len(rows), 'nll': _nll(rows, correct), 'ece': ece}


class TemperatureScaler:
    def __init__(self, temperature=1.):
        self.temperature = _temperature(temperature)

    def fit(self, probabilities, correct, grid=None):
        rows = _rows(probabilities, correct)
        candidates = list(DEFAULT_GRID if grid is None else grid)
        if not candidates:
            raise ValueError('temperature grid must not be empty')
        for t in candidates:
            _temperature(t)
        self.temperature = min(candidates, key=lambda t: _nll([_scale(row, t) for row in rows], correct))
        return self

    def transform(self, probabilities):
        return [_scale(row, _temperature(self.temperature)) for row in _rows(probabilities)]


def evaluate(fit_records, evaluation_records):
    """Fit only on training records; reject shared context groups across the split."""
    groups = []
    for records in (fit_records, evaluation_records):
        if not records or any(not isinstance(r.get('group'), str) or not r['group'] for r in records):
            raise ValueError('records require a non-empty context group')
        groups.append({r['group'] for r in records})
    if groups[0] & groups[1]:
        raise ValueError('fitting and evaluation context groups must be disjoint')
    fit_probs = [r['probabilities'] for r in fit_records]
    fit_truth = [r['correct'] for r in fit_records]
    eval_probs = [r['probabilities'] for r in evaluation_records]
    eval_truth = [r['correct'] for r in evaluation_records]
    before = metrics(eval_probs, eval_truth)
    scaler = TemperatureScaler().fit(fit_probs, fit_truth)
    after = metrics(scaler.transform(eval_probs), eval_truth)
    return {'temperature': scaler.temperature, 'fit_count': len(fit_records),
            'evaluation_count': len(evaluation_records), 'before': before, 'after': after,
            'applied_to_server': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit', required=True, type=Path, help='Training JSON array')
    parser.add_argument('--evaluate', required=True, type=Path, help='Held-out JSON array')
    args = parser.parse_args()
    try:
        report = evaluate(json.loads(args.fit.read_text(encoding='utf-8')),
                          json.loads(args.evaluate.read_text(encoding='utf-8')))
    except (ValueError, TypeError, KeyError, AttributeError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == '__main__':
    main()
