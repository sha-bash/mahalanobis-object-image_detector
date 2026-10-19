"""Independent count/risk/coordinate checks against preserved v4 predictions."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/evidence/instance_v4'


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def main():
    rows=read(BASE/'manifest.json')['crops']; frozen=read(BASE/'frozen_thresholds.json')
    train=set(read(BASE/'training_provenance.json')['train_ids'])
    val={r['identity'] for r in rows if r['split']=='validation'}; test={r['identity'] for r in rows if r['split']=='test'}
    assert len(val)==len(test)==40 and not val&test and not (val|test)&train
    assert frozen['protocol_sha256']==hashlib.sha256((ROOT/'experiments/INSTANCE_V4_PROTOCOL.md').read_bytes()).hexdigest()
    pred=read(BASE/'test_predictions.json'); result=read(BASE/'test_summary.json')
    provenance=read(BASE/'test_provenance.json')
    for key,path in [('thresholds_sha256',BASE/'frozen_thresholds.json'),('predictions_sha256',BASE/'test_predictions.json'),
                     ('code_sha256',ROOT/'scripts/instance_experiment_v4.py')]:
        assert provenance[key]==hashlib.sha256(path.read_bytes()).hexdigest()
    for summary in result:
        group=[r for r in pred if r['method']==summary['method'] and r['delta']==summary['delta']]
        assert len(group)==320 and sum(r['same'] for r in group)==160
        y=np.array([r['same'] for r in group]); p=np.array([r['accepted'] for r in group])
        c=dict(tp=int(np.sum(y&p)),fp=int(np.sum(~y&p)),fn=int(np.sum(y&~p)),tn=int(np.sum(~y&~p)))
        assert all(c[k]==summary[k] for k in c)
        assert abs(summary['risk']-(5*c['fp']+c['fn'])/320)<1e-12
    for seed in [2027,2028,2029]:
        r=read(BASE/f'training_s{seed}.json');assert r['seed']==seed and len(r['epochs'])==30
        assert set(r['train_ids'])<=train and not set(r['train_ids'])&(val|test)
    video=ROOT/'experiments/evidence/davis_v4'
    manifest=read(video/'manifest.json')
    assert manifest['protocol_sha256']==hashlib.sha256((ROOT/'experiments/INSTANCE_V4_VIDEO_PROTOCOL.md').read_bytes()).hexdigest()
    predictions=read(video/'predictions.json'); frames={r['key']:r for r in manifest['frames']}
    def overlap(a,b):
        if a is None or b is None:return 0.
        intersect=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
        return intersect/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersect)
    for r in predictions:
        gt=frames[r['key']]; value=overlap(r['bbox'],gt['bbox'])
        assert abs(value-r['iou'])<1e-12
        tp=int(gt['present'] and r['bbox'] is not None and value>=.5)
        assert r['tp']==tp and r['fn']==int(gt['present'] and not tp)
        assert r['fp']==int(r['bbox'] is not None and not tp)
    for summary in read(video/'summary.json'):
        subset=[r for r in predictions if r['method']==summary['method'] and r[summary['subset']]]
        assert all(sum(r[k] for r in subset)==summary[k] for k in ['tp','fp','fn','tn'])
    payload=dict(status='passed',test_ids=len(test),validation_ids=len(val),train_ids=len(train),
                 pair_decisions=len(pred),video_decisions=len(predictions))
    gate=[r for r in pred if r['delta']==.02 and r['method']=='gate_text_s2027_full']
    strict={(r['a'],r['b']):r for r in pred if r['delta']==.02 and r['method']=='visual_strict'}
    payload['conditional_text_module']={
        'requested':sum(r['text_requested'] for r in gate),
        **{name:sum(r['text_requested'] and r['same']==truth and r['accepted']==accept for r in gate)
           for name,truth,accept in [('tp',True,True),('fp',False,True),('fn',True,False),('tn',False,False)]}}
    payload['pair_transitions_vs_strict']={
        name:sum(r['same']==truth and strict[(r['a'],r['b'])]['accepted']==before and r['accepted']==after for r in gate)
        for name,truth,before,after in [('removed_fp',False,True,False),('added_fp',False,False,True),
                                      ('removed_tp',True,True,False),('added_tp',True,False,True)]}
    (BASE/'verification.json').write_text(json.dumps(payload,indent=2),encoding='utf-8');print(payload)


if __name__=='__main__': main()
