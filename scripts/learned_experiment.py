"""Train the real upstream MLP, freeze validation thresholds, then evaluate test."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from moid.learned_metrics import covariance_for_identities, best_f1_threshold, calibrate_recall, macro_rates, refusal_aware_ap
from moid.multimodal_metrics import distance_matrix, rates, score_scene

EVIDENCE=ROOT/'experiments/evidence/mlp_v2'
LOCAL=ROOT/'experiments/local/mlp_v2'


def save(path,data):
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs():
    rows=json.loads((EVIDENCE/'manifest.json').read_text(encoding='utf-8'))['crops']
    return rows,np.load(EVIDENCE/'features.npz')


def train_head(x,rows,seed,out,valid):
    import torch
    from mcd.projection.projector import DeepMahalanobisProjector
    from mcd.projection.runtime import save_projector,load_projector,project
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    rng=np.random.default_rng(seed)
    model=DeepMahalanobisProjector(input_dim=x.shape[1])
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    criterion=torch.nn.TripletMarginLoss(margin=1,p=2)
    clips=sorted({r['clip'] for r in rows if r['split']=='train'})
    epochs=[]
    start=time.perf_counter()
    for epoch in range(20):
        losses=[]
        model.train()
        for clip in rng.permutation(clips):
            indices=[i for i,r in enumerate(rows) if r['clip']==clip and valid[i]]
            a,p,n=[],[],[]
            for i in indices:
                positive=[j for j in indices if rows[j]['identity']==rows[i]['identity'] and j!=i]
                negative=[j for j in indices if rows[j]['identity']!=rows[i]['identity']]
                if not positive or not negative:
                    continue
                a.append(i); p.append(int(rng.choice(positive))); n.append(int(rng.choice(negative)))
            # One joint pass gives all branches identical BatchNorm statistics.
            if not a:
                continue
            z=model(torch.tensor(x[a+p+n],dtype=torch.float32))
            za,zp,zn=z.chunk(3)
            loss=criterion(za,zp,zn)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        epochs.append(dict(epoch=epoch+1,loss=float(np.mean(losses))))
    model.eval()
    save_projector(model,out)
    projected=project(model,x)
    np.testing.assert_allclose(project(load_projector(out),x),projected,atol=1e-7,rtol=1e-6)
    return projected,dict(seed=seed,epochs=epochs,wall_sec=time.perf_counter()-start,architecture=model.architecture_dict(),weights_sha256=sha(out.with_suffix('.pt')),train_indices=[i for i,r in enumerate(rows) if r['split']=='train' and valid[i]])


def train():
    rows,features=inputs()
    train_indices=[i for i,r in enumerate(rows) if r['split']=='train']
    valid_text=features['valid_text']
    matrices={'sift':1-features['sift'],'visual_cos':distance_matrix(features['visual']),'text_cos':distance_matrix(features['text'])}
    parameters={}
    projected={}
    timings={}
    start=time.perf_counter()
    for name,x in [('text_raw',features['text'])]+[(f'{modality}_mlp_s{seed}',None) for seed in [2027,2028,2029] for modality in ['visual','text']]:
        if x is None:
            modality=name.split('_')[0]
            seed=int(name.rsplit('s',1)[1])
            valid=valid_text if modality=='text' else np.ones(len(rows),bool)
            x,history=train_head(features[modality],rows,seed,LOCAL/f'{name}.json',valid)
            save(EVIDENCE/f'{name}_training.json',history)
            projected[name]=x
            matrices[name+'_cos']=distance_matrix(x)
        cov=covariance_for_identities(x,rows,[i for i in train_indices if not name.startswith('text') or valid_text[i]])
        t=time.perf_counter()
        for metric in ['diagonal','full']:
            matrices[name+'_'+metric]=distance_matrix(x,'mahalanobis',cov[metric])
        timings[name]=time.perf_counter()-t
        parameters[name]={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in cov.items()}
    negative=[(a,b) for a in train_indices for b in train_indices if a<b and rows[a]['clip']==rows[b]['clip'] and rows[a]['identity']!=rows[b]['identity']]
    for seed in [2027,2028,2029]:
        components=[f'{modality}_mlp_s{seed}_full' for modality in ['visual','text']]
        scales=[float(np.median([matrices[name][a,b] for a,b in negative if valid_text[a] and valid_text[b]])) for name in components]
        if min(scales)<=0:
            raise ValueError('Degenerate train fusion scale')
        matrices[f'fusion_s{seed}']=sum(matrices[name]/scale for name,scale in zip(components,scales))/2
        parameters[f'fusion_s{seed}']=dict(components=components,scales=scales)
    for name,matrix in matrices.items():
        if name.startswith(('text','fusion')):
            matrix[~valid_text,:]=1e30
            matrix[:,~valid_text]=1e30
    np.savez(LOCAL/'distances.npz',**matrices)
    np.savez(EVIDENCE/'projections.npz',**projected)
    save(EVIDENCE/'covariance.json',parameters)
    save(EVIDENCE/'training_provenance.json',dict(protocol_sha256=sha(ROOT/'experiments/MLP_PROTOCOL.md'),manifest_sha256=sha(EVIDENCE/'manifest.json'),features_sha256=sha(EVIDENCE/'features.npz'),distances_sha256=sha(LOCAL/'distances.npz'),wall_sec=time.perf_counter()-start,distance_matrix_sec=timings,commit=subprocess.check_output(['git','-c',f'safe.directory={ROOT.as_posix()}','rev-parse','HEAD'],text=True).strip(),source_sha256={str(p.relative_to(ROOT)):sha(p) for directory in ['src','scripts'] for p in (ROOT/directory).rglob('*.py')},versions={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}))
    print('Trained six real MLP projectors and covariance ablations',flush=True)


def pairs_for(matrix,rows,split,k):
    records=[]
    for clip in sorted({r['clip'] for r in rows if r['split']==split}):
        pairs,rank=score_scene(matrix,rows,clip,k)
        records.extend(dict(clip=clip,top1=rank,**p) for p in pairs)
    return records


def arrays(pairs):
    return np.array([p['distance'] for p in pairs]),np.array([p['same'] for p in pairs]),np.array([p['clip'] for p in pairs])


def calibrate():
    rows,_=inputs()
    matrices=np.load(LOCAL/'distances.npz')
    config=dict(protocol_sha256=sha(ROOT/'experiments/MLP_PROTOCOL.md'),distances_sha256=sha(LOCAL/'distances.npz'),budgets={},thresholds={})
    for k in [1,3]:
        baseline=[]
        for name in ['sift','visual_cos']:
            args=arrays(pairs_for(matrices[name],rows,'validation',k))
            tau=best_f1_threshold(*args)
            baseline.append(dict(method=name,threshold=tau,**macro_rates(*args,tau)))
        budget=float(np.mean([r['tpr'] for r in baseline]))
        config['budgets'][str(k)]=dict(recall=budget,baselines=baseline)
        config['thresholds'][str(k)]={}
        for name in matrices.files:
            args=arrays(pairs_for(matrices[name],rows,'validation',k))
            try:
                tau=calibrate_recall(*args,budget); feasible=True
            except ValueError as exc:
                if str(exc)!='Recall constraint infeasible':
                    raise
                finite=args[0][args[0]<1e29]
                tau=float(finite.max()) if len(finite) else -1.; feasible=False
            config['thresholds'][str(k)][name]=dict(threshold=tau,feasible=feasible,validation=macro_rates(*args,tau))
    save(EVIDENCE/'frozen_thresholds.json',config)
    print(json.dumps(config['budgets'],indent=2),flush=True)


def evaluate():
    from sklearn.metrics import roc_auc_score
    rows,_=inputs()
    matrices=np.load(LOCAL/'distances.npz')
    frozen=json.loads((EVIDENCE/'frozen_thresholds.json').read_text())
    if frozen['distances_sha256']!=sha(LOCAL/'distances.npz'):
        raise ValueError('Changed features after calibration')
    all_pairs=[]
    per_clip=[]
    summary=[]
    for k in [1,3]:
        for name in matrices.files:
            pairs=pairs_for(matrices[name],rows,'test',k)
            tau=frozen['thresholds'][str(k)][name]['threshold']
            all_pairs.extend(dict(method=name,k=k,accepted=p['distance']<=tau,**p) for p in pairs)
            for clip in sorted({p['clip'] for p in pairs}):
                subset=[p for p in pairs if p['clip']==clip]
                scores,labels,_=arrays(subset)
                r=rates(scores,labels,tau)
                r.update(f1=2*r['tp']/(2*r['tp']+r['fp']+r['fn']),precision=r['precision'] or 0)
                per_clip.append(dict(method=name,k=k,clip=clip,top1=subset[0]['top1'],roc_auc=roc_auc_score(labels,-scores),average_precision=refusal_aware_ap(labels,scores),**r))
            relevant=[r for r in per_clip if r['method']==name and r['k']==k]
            summary.append(dict(method=name,k=k,threshold=tau,reference_recall=frozen['budgets'][str(k)]['recall'],**{key:float(np.mean([r[key] for r in relevant])) for key in ['tpr','fpr','precision','f1','top1','roc_auc','average_precision']},**{key:int(sum(r[key] for r in relevant)) for key in ['tp','fp','fn','tn']}))
    a=[r for r in per_clip if r['method']=='visual_cos' and r['k']==1]
    b=[r for r in per_clip if r['method']=='fusion_s2027' and r['k']==1]
    assert [r['clip'] for r in a]==[r['clip'] for r in b]
    delta=np.array([x['fpr']-y['fpr'] for x,y in zip(a,b)])
    recall=np.array([y['tpr']-x['tpr'] for x,y in zip(a,b)])
    rng=np.random.default_rng(2027)
    draws=rng.integers(0,len(a),size=(10000,len(a)))
    from scipy.stats import binomtest
    nonzero=np.abs(delta)>1e-12
    statistics=dict(primary='visual_cos vs fusion_s2027, K=1',unit='video',videos=len(a),fpr_reduction=float(delta.mean()),fpr_reduction_ci95=np.quantile(delta[draws].mean(axis=1),[.025,.975]).tolist(),recall_difference=float(recall.mean()),recall_difference_ci95=np.quantile(recall[draws].mean(axis=1),[.025,.975]).tolist(),sign_test_p=float(binomtest(int((delta>1e-12).sum()),int(nonzero.sum()),.5,alternative='greater').pvalue) if nonzero.any() else 1.,tie_videos=int((~nonzero).sum()),category_fpr_reduction={category:float(np.mean([d for r,d in zip(a,delta) if r['clip'].rsplit('-',1)[0]==category])) for category in sorted({r['category'] for r in rows})},note='Sign test targets median direction; bootstrap targets mean. No proof of equivalence from nonsignificance.')
    save(EVIDENCE/'test_pairs.json',all_pairs)
    save(EVIDENCE/'test_per_video.json',per_clip)
    save(EVIDENCE/'test_summary.json',summary)
    save(EVIDENCE/'statistics.json',statistics)
    save(EVIDENCE/'test_provenance.json',dict(frozen_thresholds_sha256=sha(EVIDENCE/'frozen_thresholds.json'),protocol_sha256=sha(ROOT/'experiments/MLP_PROTOCOL.md'),scores_sha256=sha(EVIDENCE/'test_pairs.json')))
    import csv
    for filename,records in [('test_summary.csv',summary),('test_per_video.csv',per_clip)]:
        with (EVIDENCE/filename).open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    print(json.dumps(statistics,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['train','calibrate','evaluate'])
    globals()[parser.parse_args().stage]()
