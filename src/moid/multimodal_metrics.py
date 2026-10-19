"""Train-only low-dimensional metrics and validation-only operating points."""
from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA


def normalize(x):
    x=np.asarray(x,dtype=float)
    lengths=np.linalg.norm(x,axis=1,keepdims=True)
    if not np.isfinite(x).all() or np.any(lengths<1e-12):
        raise ValueError('Nonfinite or zero feature')
    return x/lengths


def fit_metric(features,train_indices,positive_pairs):
    """No validation/test feature contributes to PCA or covariance."""
    train_indices=list(train_indices)
    permitted=set(train_indices)
    if any(a not in permitted or b not in permitted for a,b in positive_pairs):
        raise ValueError('Covariance split leakage')
    if len(positive_pairs)<2:
        raise ValueError('Insufficient training pairs')
    x=normalize(features)
    pca=PCA(n_components=min(8,len(train_indices)-1,x.shape[1]),svd_solver='full')
    pca.fit(x[train_indices])
    z=normalize(pca.transform(x))
    residual=np.array([z[a]-z[b] for a,b in positive_pairs])
    variance=np.mean(residual**2,axis=0)
    diagonal=.5*variance+.5*variance.mean()+1e-4
    estimator=LedoitWolf(assume_centered=True).fit(np.vstack([residual,-residual]))
    covariance=estimator.covariance_+1e-4*np.eye(z.shape[1])
    return z,dict(mean=pca.mean_.tolist(),components=pca.components_.tolist(),
                  explained_variance_ratio=pca.explained_variance_ratio_.tolist(),
                  diagonal=diagonal.tolist(),covariance=covariance.tolist(),
                  shrinkage=float(estimator.shrinkage_),train_indices=train_indices,
                  positive_pairs=positive_pairs)


def distance_matrix(features,metric='cosine',covariance=None):
    x=normalize(features)
    if metric=='cosine':
        return np.clip(1-x@x.T,0,2)
    if covariance is None:
        raise ValueError('Covariance required')
    inv=np.linalg.inv(covariance)
    # Quadratic pairwise distances, without a large N x N x D allocation.
    gram=x@inv@x.T
    sq=np.diag(gram)[:,None]+np.diag(gram)[None,:]-2*gram
    return np.sqrt(np.maximum(sq,0))


def rates(scores,labels,threshold):
    scores=np.asarray(scores)
    labels=np.asarray(labels,dtype=bool)
    if scores.shape!=labels.shape or not np.isfinite(scores).all():
        raise ValueError('Invalid scores')
    p,n=int(labels.sum()),int((~labels).sum())
    if not p or not n:
        raise ValueError('Both labels required')
    accepted=scores<=threshold
    tp,fp=int((accepted&labels).sum()),int((accepted&~labels).sum())
    return dict(tp=tp,fp=fp,fn=p-tp,tn=n-fp,positives=p,negatives=n,
                tpr=tp/p,fpr=fp/n,balanced_accuracy=(tp/p+1-fp/n)/2,
                precision=tp/(tp+fp) if tp+fp else None)


def choose_threshold(validation_scores,validation_labels,minimum_recall=.7):
    choices=[]
    for tau in np.unique(validation_scores):
        r=rates(validation_scores,validation_labels,tau)
        if r['tpr']>=minimum_recall:
            choices.append((r['fpr'],-r['tpr'],float(tau)))
    if not choices:
        raise ValueError('No feasible validation configuration')
    return min(choices)[2]


def score_scene(matrix,rows,clip,k):
    """Shared GT-crop candidate gallery; ties in top1 receive fractional credit."""
    ids=sorted({r['identity'] for r in rows if r['clip']==clip})
    candidates=[i for i,r in enumerate(rows) if r['clip']==clip and r['role']=='candidate']
    pairs=[]
    ranks=[]
    for identity in ids:
        refs=sorted((i for i,r in enumerate(rows) if r['identity']==identity and r['role']=='reference'),key=lambda i:rows[i]['position'])[:k]
        if len(refs)!=k:
            raise ValueError('Missing reference')
        scores=np.min(matrix[np.ix_(refs,candidates)],axis=0)
        truth=np.array([rows[i]['identity']==identity for i in candidates])
        best=np.isclose(scores,scores.min(),rtol=0,atol=1e-12)&(scores<1e29)
        ranks.append(float(truth[best].mean()) if best.any() else 0.)
        pairs.extend(dict(query=identity,candidate=rows[i]['key'],same=bool(y),distance=float(s))
                     for i,y,s in zip(candidates,truth,scores))
    return pairs,float(np.mean(ranks))
