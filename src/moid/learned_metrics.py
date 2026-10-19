"""Train-only covariance and validation-only operating points for instance pairs."""
import numpy as np
from sklearn.covariance import LedoitWolf

from moid.multimodal_metrics import normalize, rates


def covariance_for_identities(features,rows,train_indices):
    train_indices=list(train_indices)
    if not train_indices or any(rows[i]['split']!='train' for i in train_indices):
        raise ValueError('Covariance requires training identities only')
    x=normalize(features)
    pairs=[(a,b) for a in train_indices for b in train_indices if a<b and rows[a]['identity']==rows[b]['identity']]
    if len(pairs)<2:
        raise ValueError('Insufficient within-identity pairs')
    d=np.array([x[a]-x[b] for a,b in pairs])
    residual=np.vstack([d,-d])
    fitted=LedoitWolf(assume_centered=True).fit(residual)
    variance=np.mean(residual**2,axis=0)
    return dict(full=fitted.covariance_+1e-4*np.eye(x.shape[1]),diagonal=np.diag(.5*variance+.5*variance.mean()+1e-4),shrinkage=float(fitted.shrinkage_),positive_pairs=pairs)


def macro_rates(scores,labels,clips,threshold):
    clips=np.asarray(clips)
    values=[rates(np.asarray(scores)[clips==c],np.asarray(labels)[clips==c],threshold) for c in np.unique(clips)]
    for v in values:
        v['f1']=2*v['tp']/(2*v['tp']+v['fp']+v['fn'])
        v['precision']=v['precision'] or 0.
    return {key:float(np.mean([v[key] for v in values])) for key in ['tpr','fpr','precision','f1']}


def threshold_grid(scores):
    scores=np.asarray(scores,float)
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError('Invalid validation distances')
    # 1e30 is a permanent refusal, not an optimizable numerical threshold.
    scores=scores[scores<1e29]
    if not len(scores):
        return np.array([-1.])
    return np.r_[np.nextafter(scores.min(),-np.inf),np.unique(scores)]


def best_f1_threshold(scores,labels,clips):
    records=[]
    for tau in threshold_grid(scores):
        r=macro_rates(scores,labels,clips,tau)
        records.append((-r['f1'],r['fpr'],float(tau)))
    return min(records)[2]


def calibrate_recall(scores,labels,clips,reference_recall):
    if not 0<=reference_recall<=1:
        raise ValueError('Invalid recall budget')
    choices=[]
    for tau in threshold_grid(scores):
        r=macro_rates(scores,labels,clips,tau)
        if r['tpr']+1e-12>=reference_recall:
            choices.append((r['fpr'],-r['tpr'],float(tau)))
    if not choices:
        raise ValueError('Recall constraint infeasible')
    return min(choices)[2]


def refusal_aware_ap(labels,distances):
    """Area under the step PR curve, without accepting permanent refusals."""
    from sklearn.metrics import precision_recall_curve
    precision,recall,thresholds=precision_recall_curve(labels,-np.asarray(distances))
    return float(-np.sum(np.diff(recall)*precision[:-1]*(thresholds>-1e29)))
