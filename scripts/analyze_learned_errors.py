"""Describe failure modes without changing selection, parameters or thresholds."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/evidence/mlp_v2'


def read(name): return json.loads((BASE/name).read_text(encoding='utf-8'))


def main():
    rows=read('manifest.json')['crops']
    captions={r['key']:r for r in read('captions.json')['records']}
    refs={r['identity']:r for r in rows if r['position']==0}
    pairs=[p for p in read('test_pairs.json') if p['k']==1 and p['method']=='text_mlp_s2027_full']
    collisions=[]
    for p in pairs:
        a=captions[refs[p['query']]['key']]; b=captions[p['candidate']]
        if a.get('valid_description',True) and b.get('valid_description',True) and a['caption']==b['caption']:
            collisions.append(dict(query=p['query'],candidate=p['candidate'],same=p['same'],caption=a['caption']))
    error_examples=[]
    for label,items in [('false_accept',[p for p in pairs if p['accepted'] and not p['same']]),('false_reject',[p for p in pairs if not p['accepted'] and p['same']])]:
        ordered=sorted(items,key=lambda p:p['distance'],reverse=label=='false_reject')[:5]
        for p in ordered:
            error_examples.append(dict(kind=label,**p,reference_caption=captions[refs[p['query']]['key']]['caption'],candidate_caption=captions[p['candidate']]['caption']))
    full=json.loads((ROOT/'experiments/evidence/mlp_full_image/predictions.json').read_text())
    changes=[]
    for method in ['visual_cos','visual_mlp_s2027_cos']:
        raw={r['key']:r for r in full if r['split']=='test' and r['method']==method}
        gate={r['key']:r for r in full if r['split']=='test' and r['method']==method+'+text'}
        changes.append(dict(method=method,removed_fp=sum(raw[k]['fp']-gate[k]['fp'] for k in raw),lost_tp=sum(raw[k]['tp']-gate[k]['tp'] for k in raw),changed_frames=[k for k in raw if raw[k]['bbox']!=gate[k]['bbox']]))
    payload=dict(identical_caption_positive_pairs=sum(c['same'] for c in collisions),identical_caption_negative_pairs=sum(not c['same'] for c in collisions),collisions=collisions,examples=error_examples,example_rule='Five lowest-distance false accepts and five highest-distance false rejects; diagnostic only',full_image_gate_changes=changes)
    (BASE/'error_analysis.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in payload.items() if k not in ['collisions','examples']},ensure_ascii=False,indent=2))


if __name__=='__main__': main()
