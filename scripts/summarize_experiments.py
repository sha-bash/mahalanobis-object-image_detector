"""Regenerate compact numeric evidence, CSV tables, and publication figures."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from moid.evaluation import evaluate,coco_target_annotations


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,default=ROOT/'experiments/runs')
    p.add_argument('--out',type=Path,default=ROOT/'experiments/evidence')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    rows=[];series={};failures=[];journal=[]
    for prov in sorted(args.runs.rglob('provenance.json')):
        meta=json.loads(prov.read_text(encoding='utf-8'))
        name=str(prov.parent.relative_to(args.runs)).replace('\\','/')
        if meta['status']!='completed':
            failures.append(dict(run=name,status=meta['status'],error=meta.get('error')));continue
        result_path=prov.parent/('results.json' if meta['variant']=='template' else 'pipeline/results.json')
        result=json.loads(result_path.read_text(encoding='utf-8'))
        manifest=json.loads(Path(meta['config']['manifest']).read_text(encoding='utf-8'))
        seq=next(s for s in manifest['sequences'] if s['id']==meta['sequence'])
        annotations=json.loads((ROOT/seq['annotations']).read_text(encoding='utf-8'))
        gt=coco_target_annotations(annotations) if seq.get('annotation_format')=='legacy_single_target_coco' else annotations['frames']
        m=evaluate(result,gt)
        monitor_path=prov.parent.parent/(meta['variant']+'_monitor.json')
        monitor=json.loads(monitor_path.read_text()) if monitor_path.exists() else {}
        row=dict(run=name,variant=meta['variant'],sequence=meta['sequence'],reference_protocol=seq['reference_protocol'],split=seq['split'],
                 processed_frames=result['processed_frames'],found_frames=result['found_frames'],
                 labelled_present=m['present'],labelled_absent=m['absent'],tp=m['tp'],fn=m['fn'],fp=m['fp'],
                 mean_iou=m['mean_iou_present'],success_05=m['recall'],precision_labelled=m['precision'],
                 absence_fpr=m['absence_fpr'],semantic_checks=len(result.get('semantic_checks',[])),
                 pipeline_elapsed_sec=result['elapsed_sec'],pipeline_fps=result['processed_frames']/result['elapsed_sec'],
                 runner_wall_sec=monitor.get('wall_sec',meta['runner_elapsed_sec']),
                 peak_tree_rss_bytes=monitor.get('peak_tree_rss_sampled_bytes'),
                 cache_policy=meta['cache_policy'],fresh_captions=meta.get('fresh_captions',False),
                 export_video=meta['variant']!='template',ci95=None)
        rows.append(row);series[name]=m['frames']
        journal.append(dict(run=name,commit=meta['commit'],code_sha256=meta['code_sha256'],
                            input_sha256=meta['inputs'],protocol_sha256=meta['protocol_sha256'],
                            python=meta['python'],platform=meta['platform'],versions=meta['versions'],
                            seed=meta['seed'],model_manifests=meta.get('model_manifests'),
                            local_sbert_main_revision=meta.get('local_sbert_main_revision'),
                            cache_policy=meta['cache_policy'],fresh_captions=meta.get('fresh_captions',False),
                            memory=monitor))
        compact={k:result[k] for k in ['processed_frames','output_size','bbox_coordinates','elapsed_sec']}
        compact['frames']=[{k:v for k,v in f.items() if k in {'frame','found','bbox_xyxy','score'}} for f in result['frames']]
        (args.out/(name.replace('/','__')+'_predictions.json')).write_text(json.dumps(compact,separators=(',',':')),encoding='utf-8')
    payload=dict(scope='one debug video, two queries: van 391, 66 722 present labels; no labelled absence',
                 ci_note='Not estimable from one independent sequence; no frame bootstrap',runs=rows,failures=failures)
    (args.out/'comparisons.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    (args.out/'experiment_journal.json').write_text(json.dumps(journal,indent=2),encoding='utf-8')
    groups={'geometry':['matrix_cached/legacy_geometry','query66/geometry'],
            'semantic':['fresh_vlm/legacy_semantic','query66/semantic'],
            'foreground_geometry':['matrix_cached/foreground_geometry','query66/foreground_geometry'],
            'template':['matrix_cached/template','query66/template']}
    aggregate=[]
    for method,names in groups.items():
        selected=[r for r in rows if r['run'] in names]
        if len(selected)!=len(names):continue
        total=sum(r['labelled_present'] for r in selected)
        tp=sum(r['tp'] for r in selected);fp=sum(r['fp'] for r in selected)
        aggregate.append(dict(method=method,query_count=2,independent_scenes=1,labelled_present=total,
            tp=tp,fp=fp,fn=total-tp,pooled_recall=tp/total,precision=tp/(tp+fp) if tp+fp else None,
            macro_query_recall=sum(r['success_05'] for r in selected)/2,
            pooled_mean_iou=sum(r['mean_iou']*r['labelled_present'] for r in selected)/total,
            macro_query_mean_iou=sum(r['mean_iou'] for r in selected)/2,
            absence_fpr=None,ci95=None))
    (args.out/'aggregate.json').write_text(json.dumps(dict(note='Two dependent queries of one video; macro-query is not macro-independent-video; no CI',rows=aggregate),indent=2),encoding='utf-8')
    with (args.out/'comparisons.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for sequence in sorted({r['sequence'] for r in rows}):
        fig,ax=plt.subplots(figsize=(9,4.5))
        for row in rows:
            if row['sequence']!=sequence:continue
            name=row['run'];frames=series[name]
            ax.plot([x['frame'] for x in frames],[x['iou'] for x in frames],label=name,linewidth=1.2)
        ax.axhline(.5,color='black',linestyle='--',linewidth=.8,label='Success threshold 0.5')
        ax.set(xlabel='Video frame index (0-based)',ylabel='IoU (miss = 0)',ylim=(0,1.05),
               title=sequence+': labelled presence; source: saved predictions and human-validated boxes')
        ax.legend(fontsize=6,loc='lower left');fig.tight_layout()
        fig.savefig(args.out/(sequence+'_localization.png'),dpi=180);fig.savefig(args.out/(sequence+'_localization.svg'));plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,6))
    names=[r['run'] for r in rows];ax.barh(names,[r['success_05'] for r in rows],color='#35678a')
    ax.set(xlabel='Fraction of labelled present frames with IoU >= 0.5 (van: 391; 66: 722)',xlim=(0,1.05),
           title='One debug scene, two queries; absent-frame FPR is unmeasured')
    ax.tick_params(axis='y',labelsize=7);fig.tight_layout();fig.savefig(args.out/'success.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(7,2.8),sharey=True)
    panels=[('GAZEL (391 labelled frames)',{'G':'matrix_cached/legacy_geometry','S':'fresh_vlm/legacy_semantic','FG':'matrix_cached/foreground_geometry'}),
            ('66 (722 labelled frames)',{'G':'query66/geometry','S':'query66/semantic','FG':'query66/foreground_geometry'})]
    styles={'G':('#2467a0','-'),'S':('#333333','--'),'FG':('#c66a16','-')}
    for ax,(title,mapping) in zip(axes,panels):
        for label,name in mapping.items():
            if name not in series:continue
            frames=series[name];color,linestyle=styles[label]
            ax.plot([x['frame'] for x in frames],[x['iou'] for x in frames],label=label,color=color,linestyle=linestyle,linewidth=.9)
        ax.axhline(.5,color='#999999',linestyle=':',linewidth=.7)
        ax.set(title=title,xlabel='Frame index',ylim=(-.03,1.03));ax.tick_params(labelsize=8)
    axes[0].set_ylabel('IoU (miss = 0)');axes[1].legend(fontsize=8,loc='lower left')
    fig.tight_layout(rect=[0,.16,1,1]);fig.text(.5,.02,'G: geometry; S: geometry + text; FG: foreground geometry.\nOne video; same-scene references; human-validated tracker-assisted labels.',ha='center',fontsize=7)
    fig.savefig(args.out/'paper_comparison.png',dpi=240);fig.savefig(args.out/'paper_comparison.svg');plt.close(fig)
    historical=json.loads((args.out/'historical_audit.json').read_text())
    checks=historical['semantic_checks'];taus=[0,2,4,6,8,10,12,14.142135623730951]
    sensitivity=[dict(tau=t,cosine_min=1-.01*t*t/2,accepted=sum(c['distance']<=t for c in checks),candidates=len(checks)) for t in taus]
    (args.out/'historical_candidate_thresholds.json').write_text(json.dumps(dict(scope='three saved candidates only; no online rerun, no negative labels',rows=sensitivity),indent=2))
    probe_path=args.out/'semantic_candidate_probe.json'
    if probe_path.exists():
        probe=json.loads(probe_path.read_text())
        records=probe['records']
        grid=json.loads((ROOT/'experiments/protocol.json').read_text())['threshold_grids']['semantic_distance']
        sensitivity=[dict(tau=t,
            positive_accepted=sum(r['distance'] is not None and r['distance']<=t and r['same_identity'] for r in records),
            negative_accepted=sum(r['distance'] is not None and r['distance']<=t and not r['same_identity'] for r in records)) for t in grid]
        (args.out/'probe_thresholds.json').write_text(json.dumps(dict(scope=probe['scope'],positive_candidates=3,negative_candidates=3,
            note='Candidate replay, not online system; one positive and one negative ID in one scene; no calibration',rows=sensitivity),indent=2))
        fig,ax=plt.subplots(figsize=(6,3))
        ax.plot(grid,[r['positive_accepted']/3 for r in sensitivity],'o-',label='True target accepted (3 crops)')
        ax.plot(grid,[r['negative_accepted']/3 for r in sensitivity],'s--',label='Other instance accepted (3 crops)')
        ax.axvline(6,color='gray',linestyle=':',label='Fixed pipeline threshold 6')
        ax.set(xlabel='Mahalanobis threshold (distance units)',ylabel='Accepted candidate fraction',ylim=(-.03,1.03),title='Diagnostic GT-crop panel: one scene, two identities')
        ax.legend(fontsize=7);fig.tight_layout();fig.savefig(args.out/'candidate_thresholds.png',dpi=200);plt.close(fig)
    # Matplotlib emits trailing spaces in SVG path data; normalize for clean diffs.
    for svg_path in args.out.glob('*.svg'):
        svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    print(json.dumps(payload,indent=2))


if __name__=='__main__':main()
