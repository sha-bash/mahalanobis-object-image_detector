"""Paired bootstrap of cluster means. Caller supplies independent cluster IDs."""
import numpy as np


def paired_cluster_interval(records, repetitions=10000, seed=2027):
    """Rows: cluster_id, a, b (paired sequence metric). Difference is b-a.

    Within each cluster average sequences first; clusters have equal weight.
    Do not mix reference protocols or reuse repeated seeds as new sequences.
    """
    groups={}
    for row in records:
        values=np.asarray([row['a'],row['b']],dtype=float)
        if not np.isfinite(values).all():raise ValueError('Finite paired metrics required')
        groups.setdefault(row['cluster_id'],[]).append(values[1]-values[0])
    if not groups:raise ValueError('No paired observations')
    differences=np.array([np.mean(v) for v in groups.values()])
    n=len(differences)
    result=dict(clusters=n,mean_difference=float(differences.mean()),ci95=None,
                note='One cluster: uncertainty not estimable' if n==1 else 'Exploratory interval; few independent clusters' if n<10 else 'Paired cluster bootstrap')
    if n>1:
        if repetitions<100:raise ValueError('At least 100 bootstrap repetitions required')
        rng=np.random.default_rng(seed)
        means=np.array([rng.choice(differences,n,replace=True).mean() for _ in range(repetitions)])
        result['ci95']=np.quantile(means,[.025,.975]).tolist()
    return result
