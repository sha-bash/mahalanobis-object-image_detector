"""Import human-validated tracker-assisted Drive CSVs; no independent scene claim."""
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from moid.evaluation import evaluate
from audit_evidence import sha


def main():
    import cv2
    folder=ROOT/'experiments/local/drive'
    def read(name,delimiter=','):
        return list(csv.DictReader((folder/name).open(encoding='utf-8'),delimiter=delimiter))
    manual=read('ground_truth_manual.csv');gazel=read('DAY_DASIAMRPN_GAZEL.csv',';');auto=read('DAY_DASIAMRPN_66.csv',';')
    gt=json.loads((ROOT/'annotations/instances.json').read_text())
    exact=sum([float(row[k]) for k in ['x','y','w','h']]==ann['bbox'] for row,ann in zip(gazel,gt['annotations']))
    cap=cv2.VideoCapture(str(ROOT/'data/search/video_test.mp4'));ok,first=cap.read();cap.release()
    if not ok:raise ValueError('Cannot read source video')
    height,width=first.shape[:2]
    def xyxy(row):
        x,y,w,h=[int(row[k]) for k in ['x','y','w','h']]
        return [x,y,x+w,y+h]
    frames=[]
    distractors={int(row['frame']):xyxy(row) for row in gazel}
    for row in manual:
        # visible=0 cannot distinguish fully occluded from absent without adjudication.
        status='present' if int(row['visible']) else 'ignore'
        frame=int(row['frame'])
        frames.append(dict(frame=frame,status=status,size=[width,height],bbox_xyxy=xyxy(row),
                           occluded=bool(int(row['occluded'])),distractors_complete=False,
                           distractor_boxes=[distractors[frame]] if frame in distractors else []))
    annotations=dict(provenance='Tracker-assisted, then human validated, confirmed by owner 2026-10-02',
                     source='ground_truth_manual.csv',frames=frames)
    (folder/'annotations_66.json').write_text(json.dumps(annotations,indent=2),encoding='utf-8')
    x1,y1,x2,y2=xyxy(manual[0]);cv2.imwrite(str(folder/'reference_66.png'),first[y1:y2,x1:x2])
    manifest=json.loads((ROOT/'experiments/protocol.json').read_text())
    manifest['version']='2026-10-02-drive-extension-before-new-query-runs'
    manifest['status']='second query, same development scene; not an independent test'
    manifest['sequences']=[dict(id='demo_66',split='debug',identities=['demo_van_1','demo_66_1'],
        video_id='demo_video',reference_video_id='demo_video',scene_ids=['demo_scene'],cluster_id='demo_scene',
        reference_protocol='same_scene',video='data/search/video_test.mp4',
        reference='experiments/local/drive/reference_66.png',ref_box=[0,0,x2-x1,y2-y1],
        annotations='experiments/local/drive/annotations_66.json',annotation_format='explicit_frames',
        rights='User-provided Drive data; no redistribution')]
    (ROOT/'experiments/drive_protocol.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    pred=dict(processed_frames=len(auto),output_size=[width,height],bbox_coordinates='output pixels',
        frames=[dict(frame=int(row['frame']),found=True,bbox_xyxy=xyxy(row)) for row in auto])
    metrics=evaluate(pred,frames)
    report=dict(gazel_coco_exact_matches=exact,gazel_csv_rows=len(gazel),manual_66_rows=len(manual),
                provenance=annotations['provenance'],independent_scenes=1,
                source_hashes={p.name:sha(p) for p in folder.glob('*.csv')},
                external_trajectory_66=dict(method='file named DAY_DASIAMRPN_66',
                  status='historical trajectory, not rerun; initialization and weights not recorded',
                  protocol='unknown initialization; do not combine with reference-only ranking',
                  metrics={k:v for k,v in metrics.items() if k!='frames'}))
    (ROOT/'experiments/evidence/drive_annotation_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
