"""Recompute article claims without trusting the previous validation.json."""
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from moid.evaluation import coco_target_annotations, evaluate


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    target = ROOT / 'experiments/evidence'
    target.mkdir(parents=True, exist_ok=True)
    gt = coco_target_annotations(json.loads((ROOT/'annotations/instances.json').read_text()))
    summary, payloads, inventory = [], {}, []
    for path in sorted((ROOT/'reports').glob('*/results.json')):
        r = json.loads(path.read_text(encoding='utf-8'))
        if not r.get('processed_frames'):
            continue
        name = path.parent.name
        subset = [a for a in gt if a['frame'] < r['processed_frames']]
        m = evaluate(r, subset)
        payloads[name] = r
        summary.append(dict(run=name, frames=r['processed_frames'], found=r['found_frames'],
                            elapsed_sec=r['elapsed_sec'], fps=r['processed_frames']/r['elapsed_sec'],
                            checks=len(r.get('semantic_checks', [])),
                            **{k:v for k,v in m.items() if k != 'frames'}))
        inventory.append(dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size, sha256=sha(path)))
    a = payloads['instance_20260911_174123_917']
    b = payloads['instance_20260911_170335_184']
    differing = [x['frame'] for x,y in zip(a['frames'],b['frames'])
                 if (x['found'],x.get('bbox_xyxy')) != (y['found'],y.get('bbox_xyxy'))]
    for relative in ['data/search/video_test.mp4', 'data/refs/frame_000000.jpg',
                     'annotations/instances.json', 'output/article/Shabashev_SBERT_Mahalanobis_article.docx']:
        p = ROOT/relative
        inventory.append(dict(path=relative, bytes=p.stat().st_size, sha256=sha(p)))
    gate_valid = all(not row['found'] or (
        a['semantic_checks'][row['semantic_check_id']]['accepted'] is True and
        0 <= a['semantic_checks'][row['semantic_check_id']]['distance'] <= a['semantic_checks'][row['semantic_check_id']]['threshold'] and
        row['semantic_source_frame'] == a['semantic_checks'][row['semantic_check_id']]['frame'] <= row['frame'])
        for row in a['frames'])
    checks = [dict(frame=c['frame'], distance=c['distance'], threshold=c['threshold'],
                   cosine_similarity=1-0.01*c['distance']**2/2, accepted=c['accepted'],
                   elapsed_sec=c['elapsed_sec']) for c in a['semantic_checks']]
    evidence = dict(status='recomputed from historical artifacts, not a new inference run',
                    annotated_frame_range=[min(x['frame'] for x in gt),max(x['frame'] for x in gt)],
                    runs=summary, gate_log_valid=gate_valid, semantic_checks=checks,
                    historical_geometry_vs_semantic_differing_frames=differing,
                    historical_timing_not_controlled=True, inventory=inventory)
    (target/'historical_audit.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    with (target/'historical_runs.csv').open('w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=summary[0].keys()); writer.writeheader(); writer.writerows(summary)
    # Compact reproducible numeric predictions; no images, video or descriptions.
    for name in ['instance_20260911_174123_917','instance_20260911_170335_184']:
        r = payloads[name]
        numeric={k:r[k] for k in ['processed_frames','output_size','bbox_coordinates','elapsed_sec']}
        numeric['frames']=[{k:v for k,v in x.items() if k in {'frame','found','bbox_xyxy'}} for x in r['frames']]
        (target/(name+'.json')).write_text(json.dumps(numeric, separators=(',',':')),encoding='utf-8')
    print(json.dumps(dict(runs=len(summary), differing_frames=len(differing),gate_log_valid=gate_valid)))


if __name__ == '__main__':
    main()
