"""Independent recount of held-out decisions and geometry, with identity leakage assertions."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/evidence/instance_v3'


def read(name): return json.loads((BASE/name).read_text(encoding='utf-8'))


def main():
    manifest=read('manifest.json')['crops']
    ids={s:{r['identity'] for r in manifest if r['split']==s} for s in ['train','validation','test']}
    assert all(len(v)==24 for v in ids.values())
    assert not (ids['train']&ids['validation'] or ids['train']&ids['test'] or ids['test']&ids['validation'])
    for mode in ['text','visual','visual_random']:
        training=read(mode+'_training.json')
        assert not set(training['train_ids']) & (ids['test']|ids['validation'])
        assert len(training['epochs'])==30
    predictions=read('test_predictions.json'); by_key={r['key']:r for r in manifest}
    frozen=read('frozen_thresholds.json')
    visual={(r['query'],r['candidate']):r for r in predictions if r['method']==frozen['selected_visual']}
    text={(r['query'],r['candidate']):r for r in predictions if r['method']=='text_hard_full'}
    for r in predictions:
        assert r['query'] in ids['test'] and by_key[r['candidate']]['split']=='test'
        assert r['same']==(r['query']==by_key[r['candidate']]['identity'])
        if r['method']!='conditional_text' and frozen['thresholds'][r['method']] is not None:
            assert r['accepted']==(r['distance']<=frozen['thresholds'][r['method']])
        if r['method']=='conditional_text':
            key=(r['query'],r['candidate']); v=visual[key]; t=text[key]; gate=frozen['gate']
            requested=v['accepted'] and gate['enabled'] and v['distance']>=gate['ambiguity_threshold']
            expected=v['accepted'] and (not requested or t['distance']>=1e29 or t['distance']<=gate['text_threshold'])
            assert r['accepted']==expected and r['text_requested']==requested
    for result in read('test_summary.json'):
        rows=[r for r in predictions if r['method']==result['method']]
        assert len(rows)==192
        tp=sum(r['same'] and r['accepted'] for r in rows); fp=sum(not r['same'] and r['accepted'] for r in rows)
        assert (tp,fp,96-tp,96-fp)==(result['tp'],result['fp'],result['fn'],result['tn'])
        categories=[r for r in read('test_per_category.json') if r['method']==result['method']]
        assert len(categories)==12
        for metric in ['tpr','fpr','precision','f1']:
            assert abs(np.mean([r[metric] for r in categories])-result[metric])<1e-12
    scene=ROOT/'experiments/evidence/instance_scenes_v3'
    m=json.loads((scene/'manifest.json').read_text()); frames={r['key']:r for r in m['frames']}
    assert not set(m['test_ids']) & (set(m['validation_ids'])|ids['train']|ids['validation']|ids['test'])
    def overlap(a,b):
        if a is None or b is None: return 0.
        inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
        area=lambda r:(r[2]-r[0])*(r[3]-r[1])
        return inter/(area(a)+area(b)-inter)
    full=json.loads((scene/'predictions.json').read_text())
    for r in full:
        frame=frames[r['key']]; value=overlap(r['bbox'],frame['bbox'])
        assert abs(value-r['iou'])<1e-12
        tp=int(frame['present'] and value>=.5 and r['bbox'] is not None)
        assert r['tp']==tp and r['fp']==int(r['bbox'] is not None and not tp)
        assert r['fn']==int(frame['present'] and not tp)
        assert r['tn']==int(not frame['present'] and r['bbox'] is None)
    payload=dict(status='passed',new_ids={k:len(v) for k,v in ids.items()},pair_decisions=len(predictions),scene_decisions=len(full),scene_query_ids=len(m['test_ids']))
    (BASE/'verification.json').write_text(json.dumps(payload,indent=2),encoding='utf-8'); print(payload)


if __name__=='__main__': main()
