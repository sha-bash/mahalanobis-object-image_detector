"""Fixed diagnostic candidate panel: two known objects at frames 0,180,360.

Uses human-validated boxes, bypasses candidate generation. Not an online benchmark.
"""
import argparse
import csv
import json
import importlib.metadata
import os
from pathlib import Path
import sys
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
os.environ['PYTHONPATH']=str(ROOT/'src')+os.pathsep+os.environ.get('PYTHONPATH','')


def main():
    import cv2
    from run import local_server,request_json
    from moid.semantic import SemanticVerifier
    from audit_evidence import sha
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    selections=[0,180,360]  # Fixed before scores; diagnostic, not independent trials.
    source=ROOT/'experiments/local/drive'
    boxes={}
    for identity,name,sep in [('van','DAY_DASIAMRPN_GAZEL.csv',';'),('66','ground_truth_manual.csv',',')]:
        boxes[identity]={int(r['frame']):[int(r[k]) for k in ['x','y','w','h']]
                         for r in csv.DictReader((source/name).open(),delimiter=sep)}
    cv2.setNumThreads(4);cv2.setRNGSeed(2027)
    ref=cv2.imread(str(ROOT/'data/refs/frame_000000.jpg'))[34:207,41:229]
    host='http://127.0.0.1:11435';records=[];start=time.perf_counter()
    with local_server(host,'qwen3.5:0.8b'):
        gate=SemanticVerifier(ref,host=host,threshold=6,cache_dir=args.out/'cache',
                              scene_context='Outdoor aerial drone footage of objects on the ground')
        metadata=dict(protocol='GT candidate oracle crops; query van; frame panel 0,180,360 fixed before inference',
                      independent_scenes=1,independent_target_ids=1,negative_ids=1,
                      threshold_source='pre-existing tau=6; not fitted on panel',
                      video_sha256=sha(ROOT/'data/search/video_test.mp4'),
                      model=request_json(host+'/api/tags'),semantic_profile=gate.metadata)
        metadata.update(commit=subprocess.check_output(['git','-c',f'safe.directory={ROOT.as_posix()}','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                        source_sha256=sha(Path(__file__)),python=sys.version,
                        versions={d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
                        frame_panel=selections,source_annotations={name:sha(source/name) for name in ['DAY_DASIAMRPN_GAZEL.csv','ground_truth_manual.csv']})
        cap=cv2.VideoCapture(str(ROOT/'data/search/video_test.mp4'))
        try:
            for f in selections:
                cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,frame=cap.read()
                if not ok:raise ValueError('Cannot decode selected frame')
                frame=cv2.resize(frame,(1920,1080),interpolation=cv2.INTER_AREA)
                for identity in ['van','66']:
                    x,y,w,h=boxes[identity][f]
                    import math
                    x1,y1,x2,y2=math.floor(x/2),math.floor(y/2),math.ceil((x+w)/2),math.ceil((y+h)/2)
                    crop=frame[y1:y2,x1:x2];crop_path=args.out/f'{identity}_{f}.png';cv2.imwrite(str(crop_path),crop)
                    started=time.perf_counter();decision=gate.verify(crop)
                    records.append(dict(frame=f,candidate_identity=identity,same_identity=identity=='van',
                                        bbox_xyxy=[x1,y1,x2,y2],crop_sha256=sha(crop_path),
                                        elapsed_sec=time.perf_counter()-started,**decision))
                    (args.out/'candidates.json').write_text(json.dumps(dict(metadata=metadata,records=records),indent=2),encoding='utf-8')
                    print(f,identity,decision['distance'],decision['accepted'],flush=True)
        finally:cap.release()
    summary=dict(scope=metadata['protocol'],positive_candidates=3,negative_candidates=3,
                 accepted_positives=sum(r['accepted'] and r['same_identity'] for r in records),
                 accepted_negatives=sum(r['accepted'] and not r['same_identity'] for r in records),
                 wall_sec=time.perf_counter()-start,independent_scenes=1,
                 note='No CI: repeated frames of one positive and one negative instance; not hard near-identical cars',
                 records=records)
    grid=json.loads((ROOT/'experiments/protocol.json').read_text())['threshold_grids']['semantic_distance']
    summary['threshold_sensitivity']=[dict(tau=t,
        positive_accepted=sum(r['distance'] is not None and r['distance']<=t and r['same_identity'] for r in records),
        negative_accepted=sum(r['distance'] is not None and r['distance']<=t and not r['same_identity'] for r in records)) for t in grid]
    (ROOT/'experiments/evidence/semantic_candidate_probe.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')


if __name__=='__main__':main()
