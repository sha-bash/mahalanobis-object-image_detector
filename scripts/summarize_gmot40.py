"""Recompute the external pilot from saved predictions and benchmark annotations."""
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from moid.evaluation import evaluate, iou


def main():
    out=ROOT/'experiments/evidence/gmot40'
    manifest=json.loads((ROOT/'experiments/gmot40_manifest.json').read_text())
    rows=[]
    journal=[]
    all_checks=[]
    for seq in manifest['sequences']:
        annotations=json.loads((ROOT/seq['annotations']).read_text())['frames']
        for variant in ['geometry','semantic']:
            run=ROOT/'experiments/runs/gmot40'/seq['id']/variant
            if not (run/'provenance.json').exists():
                continue
            meta=json.loads((run/'provenance.json').read_text())
            monitor_path=run.parent/(variant+'_monitor.json')
            monitor=json.loads(monitor_path.read_text()) if monitor_path.exists() else None
            journal.append(dict(clip=seq['id'],variant=variant,provenance=meta,monitor=monitor))
            if meta['status'] != 'completed':
                rows.append(dict(clip=seq['id'],variant=variant,status=meta['status'],error=meta.get('error')))
                continue
            result=json.loads((run/'pipeline/results.json').read_text())
            metric=evaluate(result,annotations,.5)
            known_wrong=0
            ignored_outputs=0
            ow,oh=result['output_size']
            def overlaps(bbox,ann):
                w,h=ann['size']
                return [bbox[0]*w/ow,bbox[1]*h/oh,bbox[2]*w/ow,bbox[3]*h/oh]
            for pred,ann in zip(result['frames'],annotations):
                assert pred['frame']==ann['frame']
                if not pred['found']:
                    continue
                b=overlaps(pred['bbox_xyxy'],ann)
                true_target=ann['status']=='present' and iou(b,ann['bbox_xyxy'])>=.5
                known_wrong+=int(not true_target and any(iou(b,d)>=.5 for d in ann['distractor_boxes']))
                ignored_outputs+=int(ann['status']=='ignore')
            checks=result.get('semantic_checks',result.get('checks',[]))
            for check in checks:
                ann=annotations[check['frame']]
                b=overlaps(check['bbox_xyxy'],ann)
                target_iou=iou(b,ann['bbox_xyxy']) if ann['status']=='present' else None
                other_iou=max((iou(b,d) for d in ann['distractor_boxes']),default=0)
                label='correct_target' if target_iou is not None and target_iou>=.5 else 'known_other' if other_iou>=.5 else 'unresolved'
                all_checks.append(dict(clip=seq['id'],variant=variant,target_iou=target_iou,other_iou=other_iou,label=label,**check))
            row=dict(clip=seq['id'],variant=variant,status='completed',synthetic_scene=seq['id']=='car-3',
                     frames=result['processed_frames'],present=metric['present'],ignored=metric['ignored'],
                     tp=metric['tp'],fp=metric['fp'],fn=metric['fn'],recall=metric['recall'],precision=metric['precision'],
                     mean_iou=metric['mean_iou_present'],known_wrong_instance_frames=known_wrong,
                     outputs_on_unlabelled_target_frames=ignored_outputs,verified_absence_fpr=None,
                     losses=metric['losses'],recoveries=metric['recoveries'],checks=len(checks),
                     pipeline_sec=result['elapsed_sec'],runner_sec=meta['runner_elapsed_sec'],
                     peak_tree_rss_bytes=monitor['peak_tree_rss_sampled_bytes'] if monitor else None)
            rows.append(row)
            compact={k:result[k] for k in ['processed_frames','output_size','bbox_coordinates','elapsed_sec']}
            compact['frames']=[{k:v for k,v in r.items() if k in {'frame','found','bbox_xyxy','score'}} for r in result['frames']]
            (out/f"{seq['id']}_{variant}_predictions.json").write_text(json.dumps(compact,separators=(',',':')),encoding='utf-8')
    aggregate=[]
    for scope,clips in [('all_four',['car-0','car-1','car-2','car-3']),('real_only',['car-0','car-1','car-2'])]:
        for variant in ['geometry','semantic']:
            selected=[r for r in rows if r['clip'] in clips and r['variant']==variant]
            valid=[r for r in selected if r['status']=='completed']
            total=sum(r['present'] for r in valid)
            aggregate.append(dict(scope=scope,variant=variant,expected_clips=len(clips),completed=len(valid),
                failed_or_unfinished=len(clips)-len(valid),
                macro_recall=sum(r['recall'] for r in valid)/len(valid) if len(valid)==len(clips) else None,
                pooled_recall=sum(r['tp'] for r in valid)/total if len(valid)==len(clips) else None,
                available_case_macro_recall=sum(r['recall'] for r in valid)/len(valid) if valid else None,
                available_case_pooled_recall=sum(r['tp'] for r in valid)/total if total else None,
                available_case_warning='Conditional on initialization success; cannot hide failed clips',
                known_wrong_instance_frames=sum(r['known_wrong_instance_frames'] for r in valid),
                runner_sec=sum(r['runner_sec'] for r in valid),ci95=None))
    summary=dict(scope='External fixed-configuration pilot; one reference per clip, same-capture; frame 0 excluded',
                 rows=rows,aggregate=aggregate,notes=['car-3 appears to be rendered video-game footage; separate real-only summaries',
                 'Three real clips, one synthetic; no independent-capture query; no confirmatory CI',
                 'Unlabelled target frames ignored, never equated to proven absence'])
    for name,value in [('summary',summary),('journal',journal),('semantic_checks',all_checks)]:
        (out/(name+'.json')).write_text(json.dumps(value,indent=2),encoding='utf-8')
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (out/'comparisons.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    probe_path=ROOT/'experiments/runs/gmot40_probe/results.json'
    if probe_path.exists():
        probe=json.loads(probe_path.read_text())
        (out/'probe_raw.json').write_text(json.dumps(probe,indent=2),encoding='utf-8')
        panel_stats=[]
        for scope,clips in [('all_four',['car-0','car-1','car-2','car-3']),('real_only',['car-0','car-1','car-2'])]+[(c,[c]) for c in ['car-0','car-1','car-2','car-3']]:
            rs=[r for r in probe['records'] if r['clip'] in clips]
            for tau in [0,2,4,6,8,10,12,14.142135623730951]:
                pos=[r for r in rs if r['same_identity']]
                neg=[r for r in rs if not r['same_identity']]
                tp=sum(r['distance'] is not None and r['distance']<=tau for r in pos)
                fp=sum(r['distance'] is not None and r['distance']<=tau for r in neg)
                panel_stats.append(dict(scope=scope,tau=tau,positives=len(pos),negatives=len(neg),tp=tp,fp=fp,
                    positive_id_count=len({(r['clip'],r['candidate_identity']) for r in pos}),
                    negative_id_count=len({(r['clip'],r['candidate_identity']) for r in neg}),
                    tpr=tp/len(pos) if pos else None,fpr=fp/len(neg) if neg else None,
                    balanced_accuracy=(tp/len(pos)+1-fp/len(neg))/2 if pos and neg else None))
        (out/'probe_summary.json').write_text(json.dumps(dict(status=probe['status'],failures=probe['failures'],rows=panel_stats),indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if probe_path.exists() and probe['records']:
        fig,ax=plt.subplots(figsize=(7,3))
        for index,clip in enumerate(['car-0','car-1','car-2','car-3']):
            for same,offset,marker,color in [(True,-.12,'o','#2467a0'),(False,.12,'x','#c66a16')]:
                rs=[r for r in probe['records'] if r['clip']==clip and r['same_identity']==same and r['distance'] is not None]
                ax.scatter([r['distance'] for r in rs],[index+offset]*len(rs),marker=marker,color=color,
                           label=('Target' if same else 'Other instance') if index==0 else None)
        ax.axvline(6,color='gray',linestyle='--',label='Fixed threshold 6: accept left')
        ax.set_yticks(range(4),['car-0','car-1','car-2','car-3*'])
        ax.set(xlabel='Isotropic Mahalanobis distance',ylabel='Clip (* rendered scene)',
               title='Semantic module on annotated candidate crops')
        ax.legend(fontsize=7,loc='best')
        fig.tight_layout(rect=[0,.1,1,1])
        fig.text(.5,.015,'Source: frozen 16-crop panel; repeated frames are not independent; this bypasses localization.',ha='center',fontsize=7)
        fig.savefig(out/'candidate_distances.png',dpi=200)
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(8,3.4))
    for offset,variant,color in [(-.18,'geometry','#2467a0'),(.18,'semantic','#c66a16')]:
        valid=[r for r in rows if r['variant']==variant and r['status']=='completed']
        for index,clip in enumerate(['car-0','car-1','car-2','car-3']):
            r=next((r for r in valid if r['clip']==clip),None)
            for ax,key in zip(axes,['recall','known_wrong_instance_frames']):
                if r:
                    ax.bar(index+offset,r[key],width=.34,color=color,label=variant if index==0 else None)
                else:
                    ax.text(index+offset,0,'FAIL',rotation=90,fontsize=7,ha='center')
    for ax in axes:
        ax.set_xticks(range(4),['car-0','car-1','car-2','car-3*'])
        ax.set_xlabel('GMOT-40 clip (* rendered scene)')
    axes[0].set(ylabel='Recall on labelled target frames',ylim=(0,1.05),title='Localization, IoU >= 0.5')
    axes[1].set(ylabel='Wrong-instance frames (count)',title='Known wrong-instance acceptances')
    axes[1].set_ylim(0,max(1,max((r.get('known_wrong_instance_frames',0) for r in rows),default=0)*1.2))
    for index,clip in enumerate(['car-0','car-1','car-2','car-3']):
        complete=[r for r in rows if r['clip']==clip and r['status']=='completed']
        if len(complete)==2 and all(r['known_wrong_instance_frames']==0 for r in complete):
            axes[1].text(index,.02,'both 0',ha='center',fontsize=8)
    axes[0].legend(fontsize=8)
    fig.tight_layout(rect=[0,.1,1,1])
    fig.text(.5,.015,'Source: saved predictions + official labels; one frame-0 reference, evaluation excludes frame 0. No threshold tuning.',ha='center',fontsize=7)
    fig.savefig(out/'comparison.png',dpi=200)
    plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
