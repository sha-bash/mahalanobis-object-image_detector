"""Train-only batch-hard adaptation, validation-frozen conditional gate, held-out RGB-D evaluation."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from moid.multimodal_metrics import distance_matrix
from moid.learned_metrics import covariance_for_identities, best_f1_threshold, calibrate_recall, macro_rates, threshold_grid
from instance_data_v3 import LOCAL, OUT, PROTOCOL, save, sha


def data():
    rows=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))['crops']
    return rows,np.load(OUT/'features.npz')


def fit_hard(x, rows, valid, path, mining='hard', seed=2027):
    import torch
    from mcd.projection.projector import DeepMahalanobisProjector
    from mcd.projection.runtime import save_projector, project, load_projector
    torch.set_num_threads(4); torch.manual_seed(seed); torch.use_deterministic_algorithms(True)
    rng=np.random.default_rng(seed)
    triplet_rng=np.random.default_rng(seed+1)
    model=DeepMahalanobisProjector(input_dim=x.shape[1])
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    blocks=sorted({r['clip'] for r in rows if r['split']=='train'})
    history=[]; started=time.perf_counter()
    for epoch in range(30):
        model.train(); losses=[]
        for block in rng.permutation(blocks):
            idx=[i for i,r in enumerate(rows) if r['split']=='train' and r['clip']==block and valid[i]]
            ids=[rows[i]['identity'] for i in idx]
            if len(set(ids))<2:
                continue
            same=torch.tensor([[a==b for b in ids] for a in ids])
            positive=same & ~torch.eye(len(ids),dtype=torch.bool)
            eligible=positive.any(1)
            if not eligible.any():
                continue
            z=torch.nn.functional.normalize(model(torch.tensor(x[idx],dtype=torch.float32)),dim=1)
            dist=torch.cdist(z,z,p=2)
            if mining=='hard':
                hardest_pos=dist.masked_fill(~positive,-1).max(1).values
                hardest_neg=dist.masked_fill(same,float('inf')).min(1).values
            else:
                pos=[int(triplet_rng.choice(np.flatnonzero(positive[i].numpy()))) if positive[i].any() else i for i in range(len(ids))]
                neg=[int(triplet_rng.choice(np.flatnonzero(~same[i].numpy()))) for i in range(len(ids))]
                hardest_pos=dist[torch.arange(len(ids)),pos]
                hardest_neg=dist[torch.arange(len(ids)),neg]
            loss=torch.relu(hardest_pos-hardest_neg+.2)[eligible].mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        history.append(dict(epoch=epoch+1,loss=float(np.mean(losses))))
    model.eval(); save_projector(model,path)
    projected=project(model,x)
    np.testing.assert_allclose(project(load_projector(path),x),projected,atol=1e-7,rtol=1e-6)
    return projected,dict(epochs=history,sec=time.perf_counter()-started,seed=seed,mining=mining,
        train_ids=sorted({r['identity'] for i,r in enumerate(rows) if r['split']=='train' and valid[i]}),
        weights_sha256=sha(path.with_suffix('.pt')),architecture=model.architecture_dict())


def train():
    rows,features=data()
    old=ROOT/'experiments/evidence/mlp_v2'
    oldrows=json.loads((old/'manifest.json').read_text())['crops']
    oldfeatures=np.load(old/'features.npz')
    combined=[dict(r,split='train') for r in oldrows]+rows
    offset=len(oldrows)
    trainidx=[i for i,r in enumerate(combined) if r['split']=='train']
    matrices={'sift':1-features['sift'],'visual_cos':distance_matrix(features['visual']),'text_cos':distance_matrix(features['text'])}
    covariances={}; representations={}
    for modality in ['visual','text']:
        x=np.vstack([oldfeatures[modality],features[modality]])
        valid=np.r_[oldfeatures['valid_text'],features['valid_text']] if modality=='text' else np.ones(len(x),bool)
        z,history=fit_hard(x,combined,valid,LOCAL/f'{modality}_hard.json')
        save(OUT/f'{modality}_training.json',history)
        representations[modality+'_hard']=z[offset:]
        cov=covariance_for_identities(z,combined,[i for i in trainidx if valid[i]])
        covariances[modality+'_hard']={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in cov.items()}
        matrices[modality+'_hard_cos']=distance_matrix(z[offset:])
        for metric in ['full','diagonal']:
            matrices[modality+'_hard_'+metric]=distance_matrix(z[offset:],'mahalanobis',cov[metric])
        if modality=='text':
            cov=covariance_for_identities(x,combined,[i for i in trainidx if valid[i]])
            matrices['text_raw_full']=distance_matrix(x[offset:],'mahalanobis',cov['full'])
            covariances['text_raw']={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in cov.items()}
    x=np.vstack([oldfeatures['visual'],features['visual']])
    z,history=fit_hard(x,combined,np.ones(len(x),bool),LOCAL/'visual_random.json',mining='random')
    save(OUT/'visual_random_training.json',history); representations['visual_random']=z[offset:]
    cov=covariance_for_identities(z,combined,trainidx)
    covariances['visual_random']={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in cov.items()}
    matrices['visual_random_cos']=distance_matrix(z[offset:])
    matrices['visual_random_full']=distance_matrix(z[offset:],'mahalanobis',cov['full'])
    for name,matrix in matrices.items():
        if name.startswith('text'):
            matrix[~features['valid_text'],:]=1e30; matrix[:,~features['valid_text']]=1e30
    np.savez(LOCAL/'distances.npz',**matrices)
    np.savez(OUT/'projections.npz',**representations)
    save(OUT/'covariance.json',covariances)
    save(OUT/'training_provenance.json',dict(protocol_sha256=sha(PROTOCOL),manifest_sha256=sha(OUT/'manifest.json'),
        features_sha256=sha(OUT/'features.npz'),old_features_sha256=sha(old/'features.npz'),
        code_sha256=sha(Path(__file__)),training_ids=len({r['identity'] for r in combined if r['split']=='train'})))
    print('Trained two batch-hard original MLPs and matched visual random-triplet control',flush=True)


def pairs(rows,split):
    result=[]
    for cat in sorted({r['category'] for r in rows}):
        refs=[i for i,r in enumerate(rows) if r['split']==split and r['category']==cat and r['role']=='reference']
        candidates=[i for i,r in enumerate(rows) if r['split']==split and r['category']==cat and r['role']=='candidate']
        for a in refs:
            for b in candidates:
                result.append(dict(a=a,b=b,category=cat,query=rows[a]['identity'],candidate=rows[b]['key'],same=rows[a]['identity']==rows[b]['identity']))
    return result


def vectors(records,matrix):
    return np.array([matrix[r['a'],r['b']] for r in records]),np.array([r['same'] for r in records]),np.array([r['category'] for r in records])


def decision_rates(accepted,y,groups):
    return macro_rates(np.where(accepted,0.,1.),y,groups,.5)


def gate_decision(v,t,valid,config):
    base=v<=config['visual_threshold']
    requested=base & (v>=config['ambiguity_threshold']) & config['enabled']
    accepted=base & (~requested | ~valid | (t<=config['text_threshold']))
    return accepted,requested


def calibrate():
    rows,features=data(); matrix=np.load(LOCAL/'distances.npz'); records=pairs(rows,'validation')
    baselines=[]
    for name in ['sift','visual_cos']:
        v,y,g=vectors(records,matrix[name]); tau=best_f1_threshold(v,y,g)
        baselines.append(dict(method=name,threshold=tau,**macro_rates(v,y,g,tau)))
    budget=float(np.mean([r['tpr'] for r in baselines])); thresholds={}; validation={}
    for name in matrix.files:
        v,y,g=vectors(records,matrix[name])
        try:
            tau=calibrate_recall(v,y,g,budget)
            thresholds[name]=tau; validation[name]=macro_rates(v,y,g,tau)
        except ValueError:
            thresholds[name]=None; validation[name]={'infeasible':True}
    candidates=['visual_cos','visual_hard_cos','visual_hard_full']
    chosen=min(candidates,key=lambda name:(validation[name]['fpr'],-validation[name]['tpr'],name))
    v,y,g=vectors(records,matrix[chosen]); t,_,_=vectors(records,matrix['text_hard_full'])
    valid=np.array([features['valid_text'][r['a']] and features['valid_text'][r['b']] for r in records])
    default=dict(enabled=False,visual_method=chosen,visual_threshold=thresholds[chosen],ambiguity_threshold=0.,text_threshold=0.)
    choices=[]
    base=v<=thresholds[chosen]; r=decision_rates(base,y,g)
    choices.append(((r['fpr'],0,-r['tpr'],0.,0.),default,r,0.))
    for ambiguity in np.unique(np.quantile(v[y],[0,.5,.75,1])):
        for text_tau in threshold_grid(t[valid]):
            config=dict(enabled=True,visual_method=chosen,visual_threshold=thresholds[chosen],ambiguity_threshold=float(ambiguity),text_threshold=float(text_tau))
            accepted,requested=gate_decision(v,t,valid,config); r=decision_rates(accepted,y,g)
            if r['tpr']+1e-12>=budget:
                choices.append(((r['fpr'],int(requested.sum()),-r['tpr'],float(ambiguity),float(text_tau)),config,r,float(requested.mean())))
    _,config,rate,fraction=min(choices,key=lambda x:x[0])
    payload=dict(baselines=baselines,recall_budget=budget,thresholds=thresholds,validation=validation,
        selected_visual=chosen,gate=config,gate_validation=rate,gate_requested_fraction=fraction,
        manifest_sha256=sha(OUT/'manifest.json'),protocol_sha256=sha(PROTOCOL),distance_sha256=sha(LOCAL/'distances.npz'))
    save(OUT/'frozen_thresholds.json',payload)
    print(json.dumps(payload,indent=2),flush=True)


def evaluate():
    from scipy.stats import binomtest
    rows,features=data(); matrix=np.load(LOCAL/'distances.npz')
    frozen=json.loads((OUT/'frozen_thresholds.json').read_text())
    assert frozen['distance_sha256']==sha(LOCAL/'distances.npz')
    records=pairs(rows,'test'); predictions=[]; percategory=[]; summary=[]
    for name in list(matrix.files)+['conditional_text']:
        if name=='conditional_text':
            v,y,g=vectors(records,matrix[frozen['selected_visual']]); t,_,_=vectors(records,matrix['text_hard_full'])
            valid=np.array([features['valid_text'][r['a']] and features['valid_text'][r['b']] for r in records])
            accepted,requested=gate_decision(v,t,valid,frozen['gate'])
        else:
            v,y,g=vectors(records,matrix[name]); tau=frozen['thresholds'][name]
            accepted=v<=tau if tau is not None else np.zeros(len(v),bool); requested=np.zeros(len(v),bool)
        for row,dist,acc,req in zip(records,v,accepted,requested):
            predictions.append(dict(method=name,**row,distance=float(dist),accepted=bool(acc),text_requested=bool(req)))
        for cat in sorted(set(g)):
            mask=g==cat; yy=y[mask]; aa=accepted[mask]
            percategory.append(dict(method=name,category=cat,tp=int((aa&yy).sum()),fp=int((aa&~yy).sum()),
                fn=int((~aa&yy).sum()),tn=int((~aa&~yy).sum()),**decision_rates(aa,yy,g[mask])))
        summary.append(dict(method=name,**decision_rates(accepted,y,g),text_requested_fraction=float(requested.mean()),
            tp=int((accepted&y).sum()),fp=int((accepted&~y).sum()),fn=int((~accepted&y).sum()),tn=int((~accepted&~y).sum())))
    a=[r for r in percategory if r['method']==frozen['selected_visual']]
    b=[r for r in percategory if r['method']=='conditional_text']
    d=np.array([x['fpr']-z['fpr'] for x,z in zip(a,b)]); dr=np.array([z['tpr']-x['tpr'] for x,z in zip(a,b)])
    nonzero=np.abs(d)>1e-12; rng=np.random.default_rng(2027); sample=rng.integers(0,len(d),(10000,len(d)))
    statistics=dict(unit='category',units=len(d),primary_visual=frozen['selected_visual'],fpr_reduction=float(d.mean()),
        ci95=np.quantile(d[sample].mean(1),[.025,.975]).tolist(),recall_difference=float(dr.mean()),
        recall_ci95=np.quantile(dr[sample].mean(1),[.025,.975]).tolist(),
        sign_p=float(binomtest(int((d>1e-12).sum()),int(nonzero.sum()),.5,alternative='greater').pvalue) if nonzero.any() else 1.)
    for filename,payload in [('test_predictions.json',predictions),('test_per_category.json',percategory),('test_summary.json',summary),('statistics.json',statistics)]:
        save(OUT/filename,payload)
    save(OUT/'test_provenance.json',dict(thresholds_sha256=sha(OUT/'frozen_thresholds.json'),predictions_sha256=sha(OUT/'test_predictions.json'),code_sha256=sha(Path(__file__))))
    print(json.dumps(dict(summary=summary,statistics=statistics),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('stage',choices=['train','calibrate','evaluate'])
    globals()[p.parse_args().stage]()
