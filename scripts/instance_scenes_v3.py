"""Separate-capture scene transfer with visible-target absence and saved candidate diagnostics."""
import argparse
import io
import json
from pathlib import Path
import re
import sys
import tarfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from instance_data_v3 import save,sha,PROTOCOL
LOCAL=ROOT/'experiments/local/instance_scenes_v3'
OUT=ROOT/'experiments/evidence/instance_scenes_v3'
MAIN=ROOT/'experiments/evidence/instance_v3'
MODELS=ROOT/'experiments/local/instance_v3'


def annotations(archive):
    from scipy.io import loadmat
    m=next(m for m in archive.getmembers() if m.name.endswith('.mat'))
    source=loadmat(io.BytesIO(archive.extractfile(m).read()),squeeze_me=True,struct_as_record=False)['bboxes']
    result=[]
    for number,frame in enumerate(source,1):
        objects=[]
        for item in np.asarray(frame,dtype=object).reshape(-1):
            if hasattr(item,'_fieldnames'):
                objects.append(dict(identity=f'{item.category}_{int(item.instance)}',
                    bbox=[float(item.left)-1,float(item.top)-1,float(item.right),float(item.bottom)]))
        result.append(dict(frame=number,objects=objects))
    return result


def spaced(indices):
    return [indices[i] for i in np.unique(np.linspace(0,len(indices)-1,min(4,len(indices)),dtype=int))] if indices else []


def prepare():
    from rgbd_range_v3 import index,member
    LOCAL.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
    scenes={}
    for name in ['desk_1','meeting_small_1','desk_2','table_small_1']:
        with tarfile.open(ROOT/f'experiments/local/rgbd_v3/{name}.tar') as archive:
            scenes[name]=annotations(archive)
    ids={name:{obj['identity'] for f in frames for obj in f['objects']} for name,frames in scenes.items()}
    valid_ids=ids['desk_1']|ids['meeting_small_1']
    test_ids=(ids['desk_2']|ids['table_small_1'])-valid_ids
    assert not valid_ids & test_ids
    files={}
    for info in index():
        match=re.fullmatch(r'(.+_\d+)_(\d+)_(\d+)_crop.png',Path(info.filename).name)
        if match:
            identity,video,frame=match.groups()
            files.setdefault(identity,[]).append((int(video),int(frame),info))
    missing=sorted((valid_ids|test_ids)-set(files))
    refs={}
    for identity in sorted(test_ids):
        if identity not in files:
            continue
        _,_,info=min(files[identity],key=lambda r:(r[0],r[1]))
        path=LOCAL/f'ref_{identity}.png'
        if not path.exists(): path.write_bytes(member(info))
        refs[identity]=dict(key=path.stem,identity=identity,path=path.relative_to(ROOT).as_posix(),sha256=sha(path),source_member=info.filename)
    rows=[]
    # Validation IDs determine identity disjointness; thresholds are transferred from crop validation.
    for name in ['desk_2','table_small_1']:
        frames=scenes[name]
        with tarfile.open(ROOT/f'experiments/local/rgbd_v3/{name}.tar') as archive:
            members={Path(m.name).name:m for m in archive.getmembers() if m.name.endswith('.png') and '_depth' not in m.name}
            for identity in sorted(refs):
                positive=[f['frame'] for f in frames if any(o['identity']==identity for o in f['objects'])]
                negative=[f['frame'] for f in frames if not any(o['identity']==identity for o in f['objects'])]
                for frame in spaced(positive)+spaced(negative):
                    objects=frames[frame-1]['objects']; filename=f'{name}_{frame}.png'
                    path=LOCAL/filename
                    if not path.exists(): path.write_bytes(archive.extractfile(members[filename]).read())
                    target=next((o['bbox'] for o in objects if o['identity']==identity),None)
                    rows.append(dict(key=f'{name}_{frame}__{identity}',scene=name,frame_key=path.stem,frame=frame,
                        identity=identity,image=path.relative_to(ROOT).as_posix(),image_sha256=sha(path),
                        reference=refs[identity]['path'],reference_key=refs[identity]['key'],bbox=target,
                        present=target is not None,distractors=[o['bbox'] for o in objects if o['identity']!=identity]))
    save(OUT/'manifest.json',dict(protocol_sha256=sha(PROTOCOL),refs=list(refs.values()),frames=rows,
        validation_ids=sorted(valid_ids),test_ids=sorted(test_ids),missing_ids=missing,
        absence_definition='No visible target annotation; not proof of physical absence behind occluders'))
    print(dict(queries=len(refs),query_frames=len(rows),present=sum(r['present'] for r in rows),unique_frames=len({r['frame_key'] for r in rows})),flush=True)


def proposals():
    import cv2
    import torch
    from ultralytics import YOLO,settings
    from prepare_gmot40 import crop
    torch.set_num_threads(4); cv2.setNumThreads(4); settings.update({'sync':False})
    manifest=json.loads((OUT/'manifest.json').read_text())
    unique={r['frame_key']:r for r in manifest['frames']}
    model=YOLO(str(ROOT/'yolov8n.pt')); records=[]; crops=[]
    for key,row in unique.items():
        im=cv2.imread(str(ROOT/row['image']))
        result=model.predict(im,device='cpu',imgsz=640,conf=.25,iou=.45,max_det=20,verbose=False)[0]
        boxes=[]
        for j,bbox in enumerate(result.boxes.xyxy.cpu().numpy()):
            path=LOCAL/f'{key}_p{j}.png'; cv2.imwrite(str(path),crop(im,bbox))
            index=len(crops)
            crops.append(dict(key=path.stem,path=path.relative_to(ROOT).as_posix(),sha256=sha(path),bbox=bbox.tolist()))
            boxes.append(dict(crop_index=index,bbox=bbox.tolist()))
        records.append(dict(frame_key=key,boxes=boxes))
        save(OUT/'proposals.json',dict(records=records,crops=crops,weights_sha256=sha(ROOT/'yolov8n.pt')))
    print('Proposals:',len(crops),flush=True)


def visual():
    import torch
    from PIL import Image
    from transformers import AutoModel
    from transformers.models.bit.image_processing_pil_bit import BitImageProcessorPil
    from multimodal_data import DINO
    from mcd.projection.runtime import load_projector,project
    from moid.multimodal_metrics import normalize
    torch.set_num_threads(4)
    manifest=json.loads((OUT/'manifest.json').read_text()); proposals=json.loads((OUT/'proposals.json').read_text())
    frozen=json.loads((MAIN/'frozen_thresholds.json').read_text())
    samples=manifest['refs']+proposals['crops']
    processor=BitImageProcessorPil.from_pretrained(str(DINO),local_files_only=True)
    model=AutoModel.from_pretrained(str(DINO),local_files_only=True).eval(); vectors=[]
    for start in range(0,len(samples),8):
        images=[Image.open(ROOT/r['path']).convert('RGB') for r in samples[start:start+8]]
        with torch.inference_mode():
            vectors.extend(torch.nn.functional.normalize(model(**processor(images=images,return_tensors='pt')).last_hidden_state[:,0],dim=1).numpy())
    del model,processor
    x=np.array(vectors); head=load_projector(MODELS/'visual_hard.json'); z=normalize(project(head,x))
    from moid.multimodal_metrics import distance_matrix
    covariance=json.loads((MAIN/'covariance.json').read_text())['visual_hard']['full']
    matrices=dict(visual_cos=distance_matrix(x),visual_hard_cos=distance_matrix(z),visual_hard_full=distance_matrix(z,'mahalanobis',np.array(covariance)))
    method=frozen['selected_visual']; matrix=matrices[method]; refs={r['identity']:i for i,r in enumerate(manifest['refs'])}
    prop={r['frame_key']:r['boxes'] for r in proposals['records']}; offset=len(refs); output=[]; requested={}
    for row in manifest['frames']:
        boxes=prop[row['frame_key']]; a=refs[row['identity']]
        indices=[offset+b['crop_index'] for b in boxes]
        selected=int(np.argmin(matrix[a,indices])) if indices else None
        distance=float(matrix[a,indices[selected]]) if indices else 1e30
        raw=boxes[selected]['bbox'] if indices and distance<=frozen['gate']['visual_threshold'] else None
        request=raw is not None and frozen['gate']['enabled'] and distance>=frozen['gate']['ambiguity_threshold']
        cand_key=None
        if request:
            candidate=proposals['crops'][boxes[selected]['crop_index']]; ref=manifest['refs'][a]
            requested[candidate['key']]=candidate; requested[ref['key']]=ref; cand_key=candidate['key']
        output.append(dict(key=row['key'],bbox=raw,distance=distance,requested=bool(request),candidate_key=cand_key,reference_key=row['reference_key']))
    save(OUT/'selections.json',dict(method=method,records=output))
    captions=OUT/'captions'; captions.mkdir(exist_ok=True)
    save(captions/'manifest.json',dict(crops=list(requested.values())))
    np.savez(OUT/'visual_features.npz',visual=x,projected=z)
    print('Required text crops:',len(requested),flush=True)


def describe():
    import multimodal_data as shared
    samples=json.loads((OUT/'captions/manifest.json').read_text())['crops']
    if not samples:
        save(OUT/'captions/captions.json',dict(records=[],status='skipped_disabled_or_unrequested')); return
    shared.BASE=LOCAL; shared.EVIDENCE=OUT/'captions'; shared.PROTOCOL=PROTOCOL; shared.KEEP_ALIVE='5m'; shared.FINALIZE_FAILURES=False
    shared.describe()
    if json.loads((LOCAL/'captions.json').read_text())['status']!='completed':
        shared.FINALIZE_FAILURES=True; shared.describe()


def evaluate():
    from moid.evaluation import iou
    from moid.multimodal_metrics import normalize
    frozen=json.loads((MAIN/'frozen_thresholds.json').read_text())
    manifest=json.loads((OUT/'manifest.json').read_text()); selections=json.loads((OUT/'selections.json').read_text())['records']
    captions=json.loads((OUT/'captions/captions.json').read_text())['records']; vectors={}
    valid=[c for c in captions if c.get('valid_description',True)]
    if valid:
        import torch
        from sentence_transformers import SentenceTransformer
        from multimodal_data import SBERT
        from mcd.projection.runtime import load_projector,project
        torch.set_num_threads(4)
        model=SentenceTransformer(str(SBERT),device='cpu',local_files_only=True)
        x=model.encode([r['caption'] for r in valid],normalize_embeddings=True,batch_size=32)
        del model
        z=normalize(project(load_projector(MODELS/'text_hard.json'),x))
        vectors=dict(zip([r['key'] for r in valid],z))
    inverse=np.linalg.inv(np.array(json.loads((MAIN/'covariance.json').read_text())['text_hard']['full']))
    prop={r['frame_key']:r for r in json.loads((OUT/'proposals.json').read_text())['records']}
    output=[]
    for row,s in zip(manifest['frames'],selections):
        assert row['key']==s['key']; text_distance=None; gate_bbox=s['bbox']
        if s['requested'] and s['candidate_key'] in vectors and s['reference_key'] in vectors:
            d=vectors[s['reference_key']]-vectors[s['candidate_key']]
            text_distance=float(np.sqrt(max(0,d@inverse@d)))
            if text_distance>frozen['gate']['text_threshold']: gate_bbox=None
        for method,bbox in [('visual',s['bbox']),('conditional_text',gate_bbox)]:
            overlap=iou(bbox,row['bbox']) if bbox is not None and row['present'] else 0.
            tp=int(row['present'] and bbox is not None and overlap>=.5)
            fp=int(bbox is not None and not tp); fn=int(row['present'] and not tp)
            output.append(dict(key=row['key'],scene=row['scene'],identity=row['identity'],method=method,
                present=row['present'],bbox=bbox,iou=overlap,tp=tp,fp=fp,fn=fn,tn=int(not row['present'] and bbox is None),
                wrong_instance=int(fp and any(iou(bbox,b)>=.5 for b in row['distractors'])),
                proposal_available=bool(row['present'] and any(iou(b['bbox'],row['bbox'])>=.5 for b in prop[row['frame_key']]['boxes'])),
                text_requested=s['requested'],text_distance=text_distance))
    summary=[]
    for method in ['visual','conditional_text']:
        rs=[r for r in output if r['method']==method]; counts={k:sum(r[k] for r in rs) for k in ['tp','fp','fn','tn','wrong_instance']}
        present=[r for r in rs if r['present']]; absent=[r for r in rs if not r['present']]
        summary.append(dict(method=method,**counts,frames=len(rs),present_frames=len(present),absent_frames=len(absent),
            recall=counts['tp']/len(present),precision=counts['tp']/(counts['tp']+counts['fp']) if counts['tp']+counts['fp'] else 0.,
            absent_fpr=sum(r['fp'] for r in absent)/len(absent),mean_iou=float(np.mean([r['iou'] for r in present])),
            proposal_recall=sum(r['proposal_available'] for r in present)/len(present)))
    save(OUT/'predictions.json',output); save(OUT/'summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    import os
    os.environ['YOLO_CONFIG_DIR']=str(LOCAL/'ultralytics'); os.environ['YOLO_AUTOINSTALL']='False'; os.environ['HF_HUB_OFFLINE']='1'
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('stage',choices=['prepare','proposals','visual','describe','evaluate'])
    globals()[parser.parse_args().stage]()
