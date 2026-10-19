"""New held-out comparison with equal recall constraints and cost 5:1."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from instance_data_v4 import ROOT, LOCAL, OUT, PROTOCOL, save, sha

sys.path.insert(0,str(ROOT/'src'))
from moid.multimodal_metrics import distance_matrix
from moid.learned_metrics import covariance_for_identities, best_f1_threshold, threshold_grid
from instance_experiment_v3 import fit_hard, pairs, vectors, decision_rates, gate_decision


def read(name): return json.loads((OUT/name).read_text(encoding='utf-8'))


def train():
    rows=read('manifest.json')['crops']; feature=np.load(OUT/'features.npz')
    oldrows=[]; oldtext=[]; oldvalid=[]
    for group in ['mlp_v2','instance_v3']:
        base=ROOT/'experiments/evidence'/group
        oldrows.extend(dict(r,split='train',clip=group+'/'+r['clip']) for r in json.loads((base/'manifest.json').read_text())['crops'])
        f=np.load(base/'features.npz'); oldtext.append(f['text']); oldvalid.append(f['valid_text'])
    assert not {r['identity'] for r in oldrows}&{r['identity'] for r in rows}
    combined=oldrows+rows; offset=len(oldrows); x=np.vstack(oldtext+[feature['text']])
    valid=np.concatenate(oldvalid+[feature['valid_text']]); trainidx=[i for i in range(offset) if valid[i]]
    matrices={'visual_cos':distance_matrix(feature['visual']), 'text_raw_cos':distance_matrix(feature['text'])}
    cov=covariance_for_identities(x,combined,trainidx)
    matrices['text_raw_full']=distance_matrix(feature['text'],'mahalanobis',cov['full'])
    covariances={'text_raw_full':cov['full'].tolist()}
    for seed in [2027,2028,2029]:
        z,history=fit_hard(x,combined,valid,LOCAL/f'text_s{seed}.json',seed=seed)
        save(OUT/f'training_s{seed}.json',history)
        cov=covariance_for_identities(z,combined,trainidx)
        matrices[f'text_s{seed}_cos']=distance_matrix(z[offset:])
        matrices[f'text_s{seed}_full']=distance_matrix(z[offset:],'mahalanobis',cov['full'])
        covariances[f'text_s{seed}_full']=cov['full'].tolist()
    for name,matrix in matrices.items():
        if name.startswith('text'):
            matrix[~feature['valid_text'],:]=1e30; matrix[:,~feature['valid_text']]=1e30
    np.savez(LOCAL/'distances.npz',**matrices)
    save(OUT/'covariance.json',covariances)
    save(OUT/'training_provenance.json',dict(protocol_sha256=sha(PROTOCOL),manifest_sha256=sha(OUT/'manifest.json'),
        distance_sha256=sha(LOCAL/'distances.npz'),train_ids=sorted({r['identity'] for r in oldrows}),
        train_images=offset,seed_primary=2027,seeds=[2027,2028,2029],code_sha256=sha(Path(__file__))))
    print('Training and train-only covariances completed',flush=True)


def counts(accepted,y):
    return dict(tp=int((accepted&y).sum()),fp=int((accepted&~y).sum()),
                fn=int((~accepted&y).sum()),tn=int((~accepted&~y).sum()))


def select_policy(v,t,valid,y,g,tau0,minimum,text):
    choices=[]
    # Compute only the constrained macro recall in the inner loop; full reporting
    # metrics are needed once for the winning policy, not for every threshold.
    groups=np.unique(g); positive_weights=np.zeros(len(y))
    for group in groups:
        positive=y & (g==group)
        if not positive.any(): raise ValueError('Each calibration group needs positives')
        positive_weights[positive]=1/(len(groups)*positive.sum())
    if text:
        configs=[dict(enabled=False,visual_threshold=tau0,ambiguity_threshold=0.,text_threshold=0.)]
        for ambiguity in np.unique(np.quantile(v[y],[0,.5,.75,1])) if valid.any() else []:
            configs += [dict(enabled=True,visual_threshold=tau0,ambiguity_threshold=float(ambiguity),text_threshold=float(tau)) for tau in threshold_grid(t[valid])]
    else:
        configs=[dict(enabled=False,visual_threshold=float(tau),ambiguity_threshold=0.,text_threshold=0.) for tau in sorted(set([tau0]+[float(z) for z in np.unique(v) if z<=tau0]))]
    for config in configs:
        accepted,requested=gate_decision(v,t,valid,config)
        c=counts(accepted,y)
        if float(accepted@positive_weights)+1e-12<minimum: continue
        loss=5*c['fp']+c['fn']
        key=(loss,int(requested.sum()),c['fp'],-c['tp'],config['visual_threshold'],config['ambiguity_threshold'],config['text_threshold'])
        choices.append((key,dict(config=config,counts=c,loss=loss,requested_fraction=float(requested.mean()))))
    if not choices: raise ValueError('No feasible policy; do not relax the constraint')
    winner=min(choices,key=lambda item:item[0])[1]
    accepted,_=gate_decision(v,t,valid,winner['config'])
    winner['rates']=decision_rates(accepted,y,g)
    return winner


def calibrate():
    rows=read('manifest.json')['crops']; matrix=np.load(LOCAL/'distances.npz'); f=np.load(OUT/'features.npz')
    records=pairs(rows,'validation'); v,y,g=vectors(records,matrix['visual_cos'])
    tau0=float(best_f1_threshold(v,y,g)); r0=decision_rates(v<=tau0,y,g)['tpr']
    if r0<=.02: raise ValueError('Degenerate visual baseline')
    valid=np.array([f['valid_text'][r['a']] and f['valid_text'][r['b']] for r in records])
    result={}
    for delta in [.02,0.,.01,.05]:
        minimum=max(r0-delta,1e-12)
        group={'visual_baseline':dict(config=dict(enabled=False,visual_threshold=tau0,ambiguity_threshold=0.,text_threshold=0.),rates=decision_rates(v<=tau0,y,g))}
        group['visual_strict']=select_policy(v,np.zeros(len(v)),valid,y,g,tau0,minimum,False)
        for name in matrix.files:
            if not name.startswith('text'): continue
            t,_,_=vectors(records,matrix[name])
            group['gate_'+name]=select_policy(v,t,valid,y,g,tau0,minimum,True)
        result[str(delta)]=dict(delta=delta,recall_minimum=minimum,policies=group)
    save(OUT/'frozen_thresholds.json',dict(primary_delta=.02,fp_cost=5,validation_baseline_recall=r0,
        settings=result,manifest_sha256=sha(OUT/'manifest.json'),protocol_sha256=sha(PROTOCOL),
        distance_sha256=sha(LOCAL/'distances.npz')))
    print('Frozen validation anchor recall:',r0,'minimum:',r0-.02,flush=True)


def evaluate():
    rows=read('manifest.json')['crops']; frozen=read('frozen_thresholds.json')
    assert frozen['distance_sha256']==sha(LOCAL/'distances.npz')
    matrix=np.load(LOCAL/'distances.npz'); f=np.load(OUT/'features.npz'); records=pairs(rows,'test')
    v,y,g=vectors(records,matrix['visual_cos']); valid=np.array([f['valid_text'][r['a']] and f['valid_text'][r['b']] for r in records])
    predictions=[]; summary=[]; category_rows=[]
    for delta,setting in frozen['settings'].items():
        for name,policy in setting['policies'].items():
            t=vectors(records,matrix[name.removeprefix('gate_')])[0] if name.startswith('gate_') else np.zeros(len(v))
            accepted,requested=gate_decision(v,t,valid,policy['config']); c=counts(accepted,y)
            summary.append(dict(delta=float(delta),method=name,**c,**decision_rates(accepted,y,g),
                risk=(5*c['fp']+c['fn'])/len(y),text_requested_fraction=float(requested.mean())))
            for cat in sorted(set(g)):
                mask=g==cat; cc=counts(accepted[mask],y[mask]); rates=decision_rates(accepted[mask],y[mask],g[mask])
                category_rows.append(dict(delta=float(delta),method=name,category=cat,**cc,**rates,risk=(5*cc['fp']+cc['fn'])/mask.sum()))
            for r,acc,req in zip(records,accepted,requested):
                predictions.append(dict(delta=float(delta),method=name,**r,accepted=bool(acc),text_requested=bool(req)))
    cats=sorted(set(g)); rng=np.random.default_rng(2027); samples=rng.integers(0,len(cats),(10000,len(cats)))
    def catrows(name): return sorted([r for r in category_rows if r['delta']==.02 and r['method']==name],key=lambda r:r['category'])
    strict=catrows('visual_strict'); base=catrows('visual_baseline'); gate=catrows('gate_text_s2027_full')
    drisk=np.array([a['risk']-b['risk'] for a,b in zip(strict,gate)])
    drecall=np.array([b['tpr']-a['tpr'] for a,b in zip(base,gate)])
    dfpr=np.array([a['fpr']-b['fpr'] for a,b in zip(strict,gate)])
    ci=np.quantile(drisk[samples].mean(1),[.025,.975]); rci=np.quantile(drecall[samples].mean(1),[.025,.975])
    minrec=frozen['settings']['0.02']['recall_minimum']; recall=float(np.mean([r['tpr'] for r in gate]))
    stats=dict(unit='category',units=len(cats),risk_reduction_vs_strict=float(drisk.mean()),risk_ci95=ci.tolist(),
        recall_change_vs_original=float(drecall.mean()),recall_ci95=rci.tolist(),fpr_reduction_vs_strict=float(dfpr.mean()),
        frozen_min_recall=minrec,mean_test_recall=recall,
        confirmatory_success=bool(ci[0]>0 and rci[0]>=-.02 and recall>=minrec and dfpr.mean()>0))
    for name,payload in [('test_predictions.json',predictions),('test_summary.json',summary),('test_per_category.json',category_rows),('statistics.json',stats)]: save(OUT/name,payload)
    sensitivity=[dict(method=r['method'],fp_cost=c,prevalence=pi,risk=c*(1-pi)*r['fpr']+pi*(1-r['tpr']))
                 for r in summary if r['delta']==.02 for c in [1,2,5,10] for pi in [.1,.5,.9]]
    save(OUT/'cost_sensitivity.json',sensitivity)
    save(OUT/'test_provenance.json',dict(thresholds_sha256=sha(OUT/'frozen_thresholds.json'),predictions_sha256=sha(OUT/'test_predictions.json'),code_sha256=sha(Path(__file__))))
    print(json.dumps(dict(primary=[r for r in summary if r['delta']==.02],statistics=stats),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('stage',choices=['train','calibrate','evaluate'])
    globals()[p.parse_args().stage]()
