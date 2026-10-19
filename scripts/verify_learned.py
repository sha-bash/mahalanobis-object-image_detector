"""Independent artifact checks from saved decisions, split membership and hashes."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/evidence/mlp_v2'


def read(name): return json.loads((BASE/name).read_text(encoding='utf-8'))


def verify_full_frames():
    """Recount localization directly from saved xyxy coordinates, without evaluator imports."""
    base = ROOT/'experiments/evidence/mlp_full_image'
    frames = {r['key']: r for r in json.loads((base/'manifest.json').read_text())['frames']}
    rows = json.loads((base/'predictions.json').read_text())

    def overlap(a, b):
        if a is None:
            return 0.
        areas = [(v[2]-v[0])*(v[3]-v[1]) for v in (a, b)]
        intersection = np.prod([max(0., min(a[d+2], b[d+2])-max(a[d], b[d])) for d in (0, 1)])
        return float(intersection/(sum(areas)-intersection))

    assert len(rows) == len(frames)*5 == 210
    assert len({(r['key'], r['method']) for r in rows}) == len(rows)
    for r in rows:
        frame = frames[r['key']]
        value = overlap(r['bbox'], frame['bbox'])
        tp = int(r['bbox'] is not None and value >= .5)
        fp = int(r['bbox'] is not None and not tp)
        assert abs(value-r['iou']) < 1e-12
        assert (r['tp'], r['fp'], r['fn']) == (tp, fp, 1-tp)
        assert r['wrong_instance'] == int(fp and any(overlap(r['bbox'], b) >= .5 for b in frame['distractors']))
    for result in json.loads((base/'summary.json').read_text()):
        group = [r for r in rows if r['split'] == result['split'] and r['method'] == result['method']]
        for key in ['tp', 'fp', 'fn', 'wrong_instance']:
            assert sum(r[key] for r in group) == result[key]
        assert len(group) == result['frames']
        assert abs(np.mean([r['iou'] for r in group])-result['mean_iou']) < 1e-12
        assert abs(result['tp']/len(group)-result['recall']) < 1e-12
    return len(rows)


def main():
    manifest=read('manifest.json')['crops']; pairs=read('test_pairs.json')
    pervideo=read('test_per_video.json'); summary=read('test_summary.json')
    frozen=read('frozen_thresholds.json')
    assert len(manifest)==560
    identities={split:{r['identity'] for r in manifest if r['split']==split} for split in ['train','validation','test']}
    assert not (identities['train']&identities['test'] or identities['train']&identities['validation'] or identities['validation']&identities['test'])
    for seed in [2027,2028,2029]:
        for modality in ['text','visual']:
            metadata=read(f'{modality}_mlp_s{seed}_training.json')
            assert all(manifest[i]['split']=='train' for i in metadata['train_indices'])
            path=ROOT/f'experiments/local/mlp_v2/{modality}_mlp_s{seed}.pt'
            assert hashlib.sha256(path.read_bytes()).hexdigest()==metadata['weights_sha256']
            assert len(metadata['epochs'])==20
    for r in pervideo:
        subset=[p for p in pairs if p['method']==r['method'] and p['k']==r['k'] and p['clip']==r['clip']]
        assert len(subset)==32
        threshold=frozen['thresholds'][str(r['k'])][r['method']]['threshold']
        assert threshold<1e29
        assert all(p['accepted']==(p['distance']<=threshold) for p in subset)
        counts={'tp':sum(p['accepted'] and p['same'] for p in subset),'fp':sum(p['accepted'] and not p['same'] for p in subset),'fn':sum(not p['accepted'] and p['same'] for p in subset),'tn':sum(not p['accepted'] and not p['same'] for p in subset)}
        assert all(r[k]==v for k,v in counts.items())
        assert counts['tp']+counts['fn']==8 and counts['fp']+counts['tn']==24
    for r in summary:
        group=[v for v in pervideo if v['method']==r['method'] and v['k']==r['k']]
        assert len(group)==14
        for key in ['tpr','fpr','precision','f1','top1']:
            assert abs(np.mean([g[key] for g in group])-r[key])<1e-12
    invalid=[r['key'] for r in read('captions.json')['records'] if not r.get('valid_description',True)]
    payload=dict(status='passed',manifest_rows=len(manifest),method_protocol_rows=len(summary),checked_decisions=len(pairs),checked_full_frame_predictions=verify_full_frames(),permanent_description_failures=invalid,train_ids=len(identities['train']),test_ids=len(identities['test']),tolerance=1e-12)
    (BASE/'verification.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(json.dumps(payload,indent=2))


if __name__=='__main__': main()
