"""Compare cosine and train-only regularized diagonal distance on labelled pairs.

Input JSON: pairs [{split, identities, videos, same, reference: [...], candidate: [...]}].
identity/video sets must not overlap splits. Outputs are candidate-level, not video recall.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def compare(pairs):
    owners={}
    for p in pairs:
        if p['split'] not in {'train','validation','test'}:raise ValueError('Unknown split')
        if not p['identities'] or not p['videos']:raise ValueError('Identity and video provenance required')
        if not isinstance(p['same'],bool):raise ValueError('Boolean identity label required')
        for kind in ['identities','videos']:
            for value in p[kind]:
                token=(kind,value)
                if token in owners and owners[token]!=p['split']:raise ValueError('Split leakage')
                owners[token]=p['split']
    a=np.array([p['reference'] for p in pairs],dtype=float)
    b=np.array([p['candidate'] for p in pairs],dtype=float)
    if a.ndim!=2 or a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Invalid features')
    if np.any(np.linalg.norm(a,axis=1)==0) or np.any(np.linalg.norm(b,axis=1)==0):raise ValueError('Zero feature')
    a/=np.linalg.norm(a,axis=1,keepdims=True);b/=np.linalg.norm(b,axis=1,keepdims=True)
    splits=np.array([p['split'] for p in pairs]);labels=np.array([p['same'] for p in pairs])
    train=(splits=='train')&labels
    if train.sum()<2:raise ValueError('Need at least two positive training pairs')
    # Symmetric +/- residual population: within-identity second moment, zero mean.
    variance=np.mean((a[train]-b[train])**2,axis=0)
    variance=0.5*variance+0.5*variance.mean()+0.001
    scores={'cosine_distance':np.clip(1-np.sum(a*b,axis=1),0,2),
            'regularized_diagonal':np.sqrt(np.sum((a-b)**2/variance,axis=1))}
    out=dict(scope='candidate pairs only; not end-to-end video performance',shrinkage=0.5,ridge=0.001,diagonal=variance.tolist(),metrics={})
    for split in ['validation','test']:
        m=splits==split
        if not labels[m].any() or labels[m].all():raise ValueError('Both labels required in validation and test')
    for name,score in scores.items():
        val=splits=='validation';test=splits=='test'
        def rates(mask,tau):
            accepted=score[mask]<=tau;y=labels[mask]
            tp=int(np.sum(accepted&y));fp=int(np.sum(accepted&~y))
            return dict(tp=tp,fp=fp,positive=int(y.sum()),negative=int((~y).sum()),recall=tp/int(y.sum()),fpr=fp/int((~y).sum()))
        candidates=[]
        for tau in np.unique(score[val]):
            r=rates(val,tau)
            if r['recall']>=0.7:candidates.append((r['fpr'],-r['recall'],float(tau)))
        tau=min(candidates)[2]
        out['metrics'][name]=dict(threshold=tau,validation=rates(val,tau),test=rates(test,tau))
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('pairs',type=Path);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();result=compare(json.loads(args.pairs.read_text())['pairs'])
    args.out.write_text(json.dumps(result,indent=2),encoding='utf-8')
