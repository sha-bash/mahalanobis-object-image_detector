"""All preregistered folds, features, metric ablations and reference counts."""
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from moid.multimodal_metrics import normalize,fit_metric,distance_matrix,score_scene,choose_threshold,rates


def main():
    base=ROOT/'experiments/evidence/multimodal_v1'
    rows=json.loads((base/'manifest.json').read_text(encoding='utf-8'))['crops']
    arrays=np.load(base/'features.npz')
    sources={m:normalize(arrays[m]) for m in ['visual','text']}
    sources['fusion']=np.concatenate([sources['visual'],sources['text']],axis=1)/np.sqrt(2)
    folds=[('car-0','car-1','car-2'),('car-1','car-2','car-0'),('car-2','car-0','car-1'),('car-0','car-1','car-3')]
    results=[]
    parameters=[]
    saved_scores=[]
    start=time.perf_counter()
    for train,val,test in folds:
        assert len({train,val,test})==3
        indices=[i for i,r in enumerate(rows) if r['clip']==train]
        positive_pairs=[(i,j) for i in indices for j in indices if rows[i]['role']=='reference' and rows[j]['role']=='candidate' and rows[i]['identity']==rows[j]['identity']]
        matrices={'sift':1-arrays['sift']}
        for modality,x in sources.items():
            z,params=fit_metric(x,indices,positive_pairs)
            parameters.append(dict(train=train,validation=val,test=test,modality=modality,**params))
            matrices[modality+'_raw_cos']=distance_matrix(x)
            matrices[modality+'_pca_cos']=distance_matrix(z)
            matrices[modality+'_diag']=distance_matrix(z,'mahalanobis',np.diag(params['diagonal']))
            matrices[modality+'_full']=distance_matrix(z,'mahalanobis',np.array(params['covariance']))
        for name,matrix in matrices.items():
            for k in [1,3]:
                vp,vr=score_scene(matrix,rows,val,k)
                tp,tr=score_scene(matrix,rows,test,k)
                tau=choose_threshold([r['distance'] for r in vp],[r['same'] for r in vp])
                validation=rates([r['distance'] for r in vp],[r['same'] for r in vp],tau)
                test_rates=rates([r['distance'] for r in tp],[r['same'] for r in tp],tau)
                results.append(dict(train=train,validation=val,test=test,method=name,k=k,threshold=tau,
                    validation_tpr=validation['tpr'],validation_fpr=validation['fpr'],top1=tr,**test_rates))
                saved_scores.append(dict(train=train,validation=val,test=test,method=name,k=k,threshold=tau,
                                         validation_pairs=vp,test_pairs=tp))
    macro=[]
    for scope,tests in [('real',['car-0','car-1','car-2']),('rendered',['car-3'])]:
        for name in matrices:
            for k in [1,3]:
                rs=[r for r in results if r['test'] in tests and r['method']==name and r['k']==k]
                macro.append(dict(scope=scope,method=name,k=k,scenes=len(rs),
                    **{metric:float(np.mean([r[metric] for r in rs])) for metric in ['tpr','fpr','balanced_accuracy','top1']},
                    **{metric:sum(r[metric] for r in rs) for metric in ['tp','fp','fn','tn']},
                    recall_ge_070_scenes=sum(r['tpr']>=.7 for r in rs)))
    payload=dict(status='completed',wall_sec=time.perf_counter()-start,fold_count=4,
                 rows=results,macro=macro,ci_note='Three real clips; no confirmatory inference; repeated pairs dependent',
                 top1_ties='Fraction of true targets among candidates tied at minimum score (atol1e-12)',
                 comparisons='Mahalanobis vs pca_cos same features; fusion vs visual same metric')
    for name,value in [('results',payload),('parameters',parameters),('scores',saved_scores)]:
        (base/(name+'.json')).write_text(json.dumps(value,indent=2),encoding='utf-8')
    for name,rs in [('results',results),('macro',macro)]:
        with (base/(name+'.csv')).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rs[0]))
            writer.writeheader()
            writer.writerows(rs)
    print(json.dumps(macro,indent=2))


if __name__=='__main__':
    main()
