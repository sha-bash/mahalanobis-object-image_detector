"""Independently check saved counts, operating points and split membership."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiments/evidence/multimodal_v1'


def main():
    def read(name):
        return json.loads((BASE / (name + '.json')).read_text(encoding='utf-8'))

    crops = read('manifest')['crops']
    assert len(crops) == len({r['key'] for r in crops}) == 120
    identities = {r['identity'] for r in crops}
    assert len(identities) == 24
    for identity in identities:
        rows = [r for r in crops if r['identity'] == identity]
        assert len({r['source_frame'] for r in rows}) == 5
        assert sum(r['role'] == 'reference' for r in rows) == 3
    results = read('results')
    scores = read('scores')
    assert len(scores) == len(results['rows']) == 104
    for r, s in zip(results['rows'], scores):
        assert (r['method'], r['k'], r['test']) == (s['method'], s['k'], s['test'])
        assert len({r['train'], r['validation'], r['test']}) == 3
        pairs = s['test_pairs']
        assert len(pairs) == 72 and sum(p['same'] for p in pairs) == 12
        tp = sum(p['same'] and p['distance'] <= r['threshold'] for p in pairs)
        fp = sum(not p['same'] and p['distance'] <= r['threshold'] for p in pairs)
        assert (tp, fp, 12-tp, 60-fp) == (r['tp'], r['fp'], r['fn'], r['tn'])
        assert np.isclose(r['balanced_accuracy'], (tp/12 + 1-fp/60)/2)
        choices = []
        for tau in {p['distance'] for p in s['validation_pairs']}:
            accepted = [p for p in s['validation_pairs'] if p['distance'] <= tau]
            vtp = sum(p['same'] for p in accepted)
            vfp = len(accepted)-vtp
            if vtp/12 >= .7:
                choices.append((vfp/60, -vtp/12, tau))
        assert min(choices)[2] == r['threshold']
    for p in read('parameters'):
        indices = set(p['train_indices'])
        assert len(indices) == 30
        assert all(crops[i]['clip'] == p['train'] for i in indices)
        assert len(p['positive_pairs']) == 36
        assert all(a in indices and b in indices and crops[a]['identity'] == crops[b]['identity']
                   for a, b in p['positive_pairs'])
        assert np.linalg.eigvalsh(p['covariance']).min() > 0
    for m in results['macro']:
        rows = [r for r in results['rows'] if r['method'] == m['method'] and r['k'] == m['k']
                and ((r['test'] != 'car-3') == (m['scope'] == 'real'))]
        for metric in ['tpr', 'fpr', 'balanced_accuracy', 'top1']:
            assert np.isclose(m[metric], np.mean([r[metric] for r in rows]))
    sources = ['scripts/multimodal_data.py', 'scripts/evaluate_multimodal.py',
               'src/moid/multimodal_metrics.py', 'scripts/verify_multimodal_artifacts.py']
    result = dict(status='passed', configurations=104, crops=120, identities=24,
                  checks=['unique source frames', 'train-only covariance pairs',
                          'positive definite covariance', 'all test confusion matrices',
                          'validation-only threshold optimality', 'macro means'],
                  final_source_sha256={s: hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources})
    (BASE/'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
