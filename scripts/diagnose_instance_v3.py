"""Decompose saved localization errors without rerunning or changing any model."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from moid.evaluation import iou

BASE = ROOT/'experiments/evidence/mlp_full_image'
OUT = ROOT/'experiments/evidence/instance_v3'


def read(name):
    return json.loads((BASE/name).read_text(encoding='utf-8'))


def decompose(overlaps, chosen, accepted, gate_accepted, threshold=.5):
    """Mutually exclusive localization outcomes; candidate indices are local."""
    available = any(v >= threshold for v in overlaps)
    selected_correct = chosen is not None and overlaps[chosen] >= threshold
    if not available:
        reason = 'proposal_missing'
    elif not selected_correct:
        reason = 'wrong_selection'
    elif not accepted:
        reason = 'visual_rejection'
    elif not gate_accepted:
        reason = 'text_rejection'
    else:
        reason = 'success'
    return dict(available=available, selected_correct=selected_correct, reason=reason)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    proposals = {r['key']: r for r in read('proposals.json')['records']}
    selections = {r['key']: r for r in read('selections.json')['records']}
    predictions = {(r['key'], r['method']): r for r in read('predictions.json')}
    results = []
    for frame in read('manifest.json')['frames']:
        boxes = proposals[frame['key']]['boxes']
        overlaps = [iou(b['bbox'], frame['bbox']) for b in boxes]
        for method, index in selections[frame['key']]['selected'].items():
            chosen = next((j for j, b in enumerate(boxes) if b['crop_index'] == index), None)
            raw = predictions[frame['key'], method]
            gate = predictions[frame['key'], method+'+text']
            detail = decompose(overlaps, chosen, raw['bbox'] is not None, gate['bbox'] is not None)
            results.append(dict(key=frame['key'], split=frame['split'], category=frame['clip'].rsplit('-', 1)[0],
                                method=method, max_iou=max(overlaps, default=0), proposals=len(boxes),
                                min_target_side=min(frame['bbox'][2]-frame['bbox'][0], frame['bbox'][3]-frame['bbox'][1]),
                                selected_iou=overlaps[chosen] if chosen is not None else 0,
                                removed_fp=raw['fp']-gate['fp'], lost_tp=raw['tp']-gate['tp'], **detail))
    summary = []
    for split in ['validation', 'test']:
        for method in ['visual_cos', 'visual_mlp_s2027_cos']:
            rows = [r for r in results if r['split'] == split and r['method'] == method]
            available = sum(r['available'] for r in rows)
            selected = sum(r['selected_correct'] for r in rows)
            summary.append(dict(split=split, method=method, frames=len(rows), proposal_available=available,
                                proposal_recall=available/len(rows), selected_correct=selected,
                                selection_success_given_available=selected/available if available else None,
                                reasons={key:sum(r['reason'] == key for r in rows) for key in
                                         ['proposal_missing','wrong_selection','visual_rejection','text_rejection','success']},
                                removed_fp=sum(r['removed_fp'] for r in rows), lost_tp=sum(r['lost_tp'] for r in rows)))
    categories = []
    for category in sorted({r['category'] for r in results}):
        rows = [r for r in results if r['split'] == 'test' and r['method'] == 'visual_cos' and r['category'] == category]
        categories.append(dict(category=category, frames=len(rows), available=sum(r['available'] for r in rows),
                               median_min_side=float(np.median([r['min_target_side'] for r in rows]))))
    payload = dict(source='Saved v2 predictions, now development data; no new independent test',
                   summary=summary, categories=categories, frames=results)
    (OUT/'diagnostics.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12})
    selected = [r for r in summary if r['split'] == 'test']
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bottom = np.zeros(2)
    labels = ['Нет правильного кандидата','Выбран другой кандидат','Отказ визуального метода','Отказ текста от цели','Правильное обнаружение']
    for key, label in zip(selected[0]['reasons'], labels):
        values = np.array([r['reasons'][key] for r in selected])
        bars = ax.barh(['DINOv2 + текст','Сиамский MLP + текст'], values, left=bottom, label=label)
        ax.bar_label(bars, labels=[str(v) if v else '' for v in values], label_type='center')
        bottom += values
    ax.set_xlabel('Число кадров из 28'); ax.set_xlim(0, 28)
    ax.legend(loc='upper center', bbox_to_anchor=(.5,-.18), ncol=2, fontsize=10)
    fig.tight_layout(); fig.savefig(OUT/'error_decomposition.png', dpi=160, bbox_inches='tight'); plt.close(fig)
    print(json.dumps(dict(summary=summary, categories=categories), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
