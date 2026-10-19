"""Post-hoc cost sensitivity of frozen predictions; never tune the test set."""
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'experiments/evidence/instance_v3'
OUT = ROOT / 'experiments/evidence/instance_v3_cost'


def weighted_risk(fpr, fnr, prevalence, fp_cost):
    """Expected loss per pair in units of one missed true match."""
    if not 0 <= prevalence <= 1 or fp_cost < 0:
        raise ValueError('Prevalence must be in [0,1] and cost nonnegative')
    return fp_cost * (1 - prevalence) * fpr + prevalence * fnr


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    read = lambda name: json.loads((SOURCE / name).read_text(encoding='utf-8'))
    frozen = read('frozen_thresholds.json')
    rows = read('test_predictions.json')
    baseline = { (r['a'], r['b']): r for r in rows if r['method'] == frozen['selected_visual'] }
    filtered = { (r['a'], r['b']): r for r in rows if r['method'] == 'conditional_text' }
    assert baseline.keys() == filtered.keys()
    changes = []
    for key, a in baseline.items():
        b = filtered[key]
        assert a['same'] == b['same']
        assert not b['accepted'] or a['accepted'], 'A rejection-only filter cannot add acceptances'
        if a['accepted'] != b['accepted']:
            changes.append(dict(query=a['query'], candidate=a['candidate'], category=a['category'],
                                same=a['same'], effect='lost_tp' if a['same'] else 'removed_fp'))
    removed = sum(not r['same'] for r in changes)
    lost = sum(r['same'] for r in changes)
    summary = {r['method']: r for r in read('test_summary.json')}
    a = summary[frozen['selected_visual']]; b = summary['conditional_text']
    assert removed == a['fp'] - b['fp'] and lost == b['fn'] - a['fn']
    cats = read('test_per_category.json')
    by_method = {name: {r['category']: r for r in cats if r['method'] == name}
                 for name in [frozen['selected_visual'], 'conditional_text']}
    differences = []
    for cat, x in by_method[frozen['selected_visual']].items():
        y = by_method['conditional_text'][cat]
        differences.append(dict(category=cat, removed_fp=x['fp']-y['fp'], added_fn=y['fn']-x['fn']))
    diff = np.array([[r['removed_fp'], r['added_fn']] for r in differences])
    rng = np.random.default_rng(2027)
    # Resample complete categories jointly for both methods, not individual frames.
    draws = diff[rng.integers(0, len(diff), size=(10000, len(diff)))].sum(axis=1)
    costs = []
    for cost in [1, 2, 5, 10]:
        before = cost*a['fp'] + a['fn']; after = cost*b['fp'] + b['fn']
        costs.append(dict(fp_cost=cost, baseline_loss=before, filtered_loss=after,
                          loss_reduction=before-after, relative_reduction=(before-after)/before,
                          ci95_loss_reduction=np.quantile(cost*draws[:, 0]-draws[:, 1], [.025, .975]).tolist()))
    sensitivity = []
    for pi in [.1, .5, .9]:
        sensitivity.append(dict(prevalence=pi, break_even_cost=pi*lost/((1-pi)*removed)))
    payload = dict(analysis='post-hoc sensitivity, no training or threshold changes',
                   source_commit='d9e237c', input_sha256={name: hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()
                     for name in ['frozen_thresholds.json','test_predictions.json','test_summary.json','test_per_category.json']},
                   baseline=a, filtered=b, removed_fp=removed, lost_tp=lost,
                   relative_fp_reduction=removed/a['fp'], relative_tp_loss=lost/a['tp'],
                   rejection_precision=removed/len(changes), changed_pairs=changes,
                   affected_query_ids=len({r['query'] for r in changes}),
                   affected_categories=len({r['category'] for r in changes}),
                   per_category=differences, costs=costs, prevalence_sensitivity=sensitivity,
                   micro_precision_before=a['tp']/(a['tp']+a['fp']),
                   micro_precision_after=b['tp']/(b['tp']+b['fp']),
                   original_primary_test_p=read('statistics.json')['sign_p'],
                   limitations=['cost ratios illustrative, not elicited deployment costs',
                     'new loss analysis not the original hypothesis test',
                     'prevalence reweighting assumes unchanged conditional error rates',
                     'VLM latency excluded from loss; retain a recall constraint to exclude always-reject'])
    (OUT/'analysis.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    grid = np.linspace(.1, 10, 200)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    effect = removed*grid-lost
    ci = np.quantile(grid[:, None]*draws[:,0]-draws[:,1], [.025,.975], axis=1)
    axs[0].plot(grid, effect, label='Снижение взвешенных ошибок')
    axs[0].fill_between(grid, ci[0], ci[1], alpha=.2, label='95% интервал по категориям')
    axs[0].axhline(0,color='black',lw=1)
    axs[0].set(xlabel='Цена ложного принятия / цена пропуска',ylabel='Выигрыш на 192 парах',title='Наблюдаемый эффект при разных ценах')
    axs[0].legend(fontsize=8)
    for pi in [.1,.5,.9]:
        gain = grid*(1-pi)*(a['fpr']-b['fpr'])-pi*(a['tpr']-b['tpr'])
        axs[1].plot(grid, gain, label=f'Доля истинных совпадений {100*pi:.0f}%')
    axs[1].axhline(0,color='black',lw=1)
    axs[1].set(xlabel='Цена ложного принятия / цена пропуска',ylabel='Снижение риска на одну пару',title='Сценарии частоты цели')
    axs[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT/'cost_sensitivity.png',dpi=170); plt.close(fig)
    print(json.dumps({k:payload[k] for k in ['removed_fp','lost_tp','affected_categories','costs','prevalence_sensitivity']},indent=2))


if __name__ == '__main__':
    main()
