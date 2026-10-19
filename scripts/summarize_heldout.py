"""Summarize all preregistered video runs without selecting test thresholds."""
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from moid.evaluation import evaluate, iou

BASE = ROOT/'experiments/evidence/heldout_v1'
RUNS = ROOT/'experiments/runs/heldout_v1'
VARIANTS = ['geometry', 'isotropic', 'text_cosine', 'text_full']


def sign_test(deltas):
    positive = sum(d > 1e-12 for d in deltas)
    negative = sum(d < -1e-12 for d in deltas)
    n = positive+negative
    conditional = sum(math.comb(n, k) for k in range(positive, n+1))/2**n if n else 1.
    # Recall is discrete with possible mass at zero difference. For H0 median<=0,
    # P(delta>0)<=1/2; retain ties as non-improvements for a conservative test.
    total = len(deltas)
    p = sum(math.comb(total, k) for k in range(positive, total+1))/2**total if total else 1.
    return dict(positive=positive, negative=negative, ties=total-n, one_sided_p=p,
                conditional_nonzero_p=conditional)


def main():
    sequences = json.loads((BASE/'manifest.json').read_text())['sequences']
    execution = json.loads((BASE/'execution.json').read_text())
    rows, details, semantic = [], [], []
    max_float_delta = 0.
    for seq in sequences:
        anns = json.loads((ROOT/seq['annotations']).read_text())['frames']
        for variant in VARIANTS:
            out = RUNS/seq['id']/variant
            meta = json.loads((out/'provenance.json').read_text())
            assert meta['status'] in {'completed', 'operational_refusal'}, meta
            result = json.loads((out/'results.json').read_text())
            metrics = evaluate(result, anns, .5)
            saved_metrics = json.loads((out/'metrics.json').read_text())
            for key, value in metrics.items():
                if isinstance(value, float):
                    max_float_delta = max(max_float_delta, abs(value-saved_metrics[key]))
                    assert math.isclose(value, saved_metrics[key], rel_tol=0, abs_tol=1e-12), (seq['id'], variant, key)
                else:
                    assert value == saved_metrics[key], (seq['id'], variant, key)
            runtime = next(r for r in execution if r['sequence'] == seq['id'] and r['variant'] == variant)
            assert runtime['returncode'] == 0
            row = dict(sequence=seq['id'], variant=variant, status=meta['status'],
                       processed_frames=result['processed_frames'], **{k:v for k, v in metrics.items() if k not in {'frames', 'acquisition_delays_frames'}},
                       wall_sec=runtime['wall_sec'], peak_tree_rss_bytes=runtime['peak_tree_rss_bytes'])
            checks = result.get('semantic_checks', [])
            row['semantic_checks'] = len(checks)
            row['accepted_checks'] = sum(c['accepted'] for c in checks)
            descriptions = [c['candidate_description'] for c in checks if 'candidate_description' in c]
            if result.get('semantic_profile'):
                descriptions.append(result['semantic_profile']['reference_description'])
            row['caption_cache_hits'] = sum(d['cache_hit'] for d in descriptions)
            row['caption_cold_descriptions'] = sum(not d['cache_hit'] for d in descriptions)
            rows.append(row)
            predictions = {r['frame']:r for r in result['frames']}
            details.extend(dict(sequence=seq['id'], variant=variant, bbox_xyxy=predictions[f['frame']].get('bbox_xyxy'), **f) for f in metrics['frames'])
            ow, oh = result['output_size']
            for c in checks:
                a = anns[c['frame']]
                sw, sh = a['size']
                def scaled(box):
                    return [v*s for v, s in zip(box, [ow/sw, oh/sh, ow/sw, oh/sh])]
                overlap = iou(c['bbox_xyxy'], scaled(a['bbox_xyxy'])) if a['status'] == 'present' else None
                semantic.append(dict(sequence=seq['id'], variant=variant, target_iou=overlap,
                                     positive_candidate=None if overlap is None else overlap >= .5, **c))
            compact = dict(provenance=meta, output_size=result['output_size'], predictions=[{k:v for k, v in f.items() if k in {'frame', 'found', 'bbox_xyxy', 'source', 'semantic_check_id', 'semantic_source_frame', 'semantic_age_sec'}} for f in result['frames']],
                           semantic_profile=result.get('semantic_profile'), semantic_checks=checks)
            (BASE/f'{seq["id"]}_{variant}.json').write_text(json.dumps(compact, indent=2, ensure_ascii=False), encoding='utf-8')
    macro = []
    for variant in VARIANTS:
        rs = [r for r in rows if r['variant'] == variant]
        sums = {k:sum(r[k] for r in rs) for k in ['tp', 'fp', 'fn', 'present', 'ignored', 'wrong_instance', 'losses', 'recoveries', 'semantic_checks', 'accepted_checks', 'caption_cache_hits', 'caption_cold_descriptions']}
        macro.append(dict(variant=variant, scenes=len(rs), recall=float(np.mean([r['recall'] for r in rs])),
                          mean_iou_present=float(np.mean([r['mean_iou_present'] for r in rs])),
                          micro_recall=sums['tp']/sums['present'],
                          micro_precision=sums['tp']/(sums['tp']+sums['fp']) if sums['tp']+sums['fp'] else None,
                          total_wall_sec=sum(r['wall_sec'] for r in rs), **sums))
    differences = {}
    for name, a, b in [('primary', 'geometry', 'text_full'), ('cosine_vs_geometry', 'geometry', 'text_cosine'),
                       ('isotropic_vs_geometry', 'geometry', 'isotropic'), ('mahalanobis_vs_cosine', 'text_cosine', 'text_full')]:
        result = dict(before=a, after=b)
        changed = {}
        for s in sequences:
            first = json.loads((RUNS/s['id']/a/'results.json').read_text())['frames']
            second = json.loads((RUNS/s['id']/b/'results.json').read_text())['frames']
            assert len(first) == len(second)
            changed[s['id']] = sum((x['found'], x.get('bbox_xyxy')) != (y['found'], y.get('bbox_xyxy'))
                                   for x, y in zip(first, second))
        result['prediction_changed_frames'] = changed
        for metric in ['recall', 'mean_iou_present']:
            delta = [next(r[metric] for r in rows if r['sequence'] == s['id'] and r['variant'] == b)-
                     next(r[metric] for r in rows if r['sequence'] == s['id'] and r['variant'] == a) for s in sequences]
            rng = np.random.default_rng(2027)
            samples = np.array(delta)[rng.integers(0, len(delta), size=(10000, len(delta)))].mean(axis=1)
            result[metric] = dict(deltas=delta, mean=float(np.mean(delta)), median=float(np.median(delta)),
                                  descriptive_bootstrap_95=np.quantile(samples, [.025, .975]).tolist(),
                                  bootstrap_degenerate=bool(np.ptp(samples) == 0), **sign_test(delta))
        differences[name] = result
    payload = dict(rows=rows, macro=macro, comparisons=differences,
                   recomputation=dict(python=sys.version, maximum_float_difference=max_float_delta,
                                      tolerance=1e-12, integer_counts_and_per_frame_records='exact'),
                   primary_h0='Median recall(text_full)-recall(geometry) <= 0; IoU >= 0.5',
                   interval_warning='Descriptive clip bootstrap; eight clips and uncertain source independence; not equivalence proof')
    for name, data in [('summary', payload), ('frames', details), ('semantic_checks', semantic)]:
        (BASE/f'{name}.json').write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    for name, data in [('results', rows), ('macro', macro)]:
        with (BASE/f'{name}.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    print(json.dumps(dict(macro=macro, primary=differences['primary']), indent=2))


if __name__ == '__main__':
    main()
