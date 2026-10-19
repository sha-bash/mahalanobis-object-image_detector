"""Full-frame YOLO/SIFT proposal and genuine learned semantic-gate ablation."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import zipfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from prepare_gmot40 import parse_labels,crop,sha
from multimodal_data import save

LOCAL=ROOT/'experiments/local/mlp_full_image'
EVIDENCE=ROOT/'experiments/evidence/mlp_full_image'
SOURCE=ROOT/'experiments/evidence/mlp_v2'


def prepare():
    import cv2
    cv2.setNumThreads(4)
    LOCAL.mkdir(parents=True,exist_ok=True)
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    source=json.loads((SOURCE/'manifest.json').read_text())['crops']
    rows=[]
    with zipfile.ZipFile(ROOT/'experiments/local/gmot40/track_label.zip') as labels,zipfile.ZipFile(ROOT/'experiments/local/gmot40/GenericMOT_JPEG_Sequence.zip') as images:
        files={n.split('/')[-3]+'/'+Path(n).stem:n for n in images.namelist() if '/img1/' in n and n.endswith('.jpg')}
        for clip in sorted({r['clip'] for r in source if r['split']!='train'}):
            items=[r for r in source if r['clip']==clip]
            identity=items[0]['identity']
            reference=next(r for r in items if r['identity']==identity and r['position']==0)
            anns=parse_labels(labels.read(f'track_label/{clip}.txt').decode())
            for item in [r for r in items if r['identity']==identity and r['role']=='candidate']:
                frame=item['source_frame']
                image=cv2.imdecode(np.frombuffer(images.read(files[f'{clip}/{frame:06d}']),np.uint8),cv2.IMREAD_COLOR)
                path=LOCAL/f'{clip}_{frame}.png'
                cv2.imwrite(str(path),image)
                rows.append(dict(key=path.stem,clip=clip,split=item['split'],frame=frame,target_id=identity,reference=reference['path'],reference_key=reference['key'],image=path.relative_to(ROOT).as_posix(),image_sha256=sha(path),bbox=item['bbox'],distractors=[r['bbox_xyxy'] for r in anns if r['frame']==frame and f'{clip}:{r["identity"]}'!=identity]))
    save(EVIDENCE/'manifest.json',dict(protocol_sha256=sha(ROOT/'experiments/FULL_IMAGE_MLP_PROTOCOL.md'),frames=rows))
    print('Full frames prepared:',len(rows),flush=True)


def proposals():
    import cv2
    import torch
    from ultralytics import YOLO,settings
    from moid.instance import InstanceLocator,clipped_box
    torch.set_num_threads(4); cv2.setNumThreads(4)
    settings.update({'sync':False})
    rows=json.loads((EVIDENCE/'manifest.json').read_text())['frames']
    model=YOLO(str(ROOT/'yolov8n.pt'))
    records=[]
    crops=[]
    for row in rows:
        image=cv2.imread(str(ROOT/row['image']))
        t=time.perf_counter()
        result=model.predict(image,device='cpu',imgsz=640,conf=.25,iou=.45,max_det=20,classes=None,verbose=False)[0]
        boxes=[]
        for i,box in enumerate(result.boxes.xyxy.cpu().numpy()):
            path=LOCAL/f'{row["key"]}_p{i}.png'
            patch=crop(image,box)
            if not patch.size:
                continue
            cv2.imwrite(str(path),patch)
            index=len(crops)
            crops.append(dict(key=path.stem,path=path.relative_to(ROOT).as_posix(),sha256=sha(path),frame_key=row['key'],bbox=box.tolist()))
            boxes.append(dict(bbox=box.tolist(),crop_index=index,confidence=float(result.boxes.conf[i]),class_id=int(result.boxes.cls[i])))
        yolo_sec=time.perf_counter()-t
        t=time.perf_counter()
        sift=None; error=None
        cv2.setRNGSeed(2027)
        try:
            locator=InstanceLocator(cv2.imread(str(ROOT/row['reference'])))
            matrix=locator.detect(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY))
            if matrix is not None:
                sift=clipped_box(locator.polygon(matrix),[image.shape[1],image.shape[0]])
        except ValueError as exc:
            error=str(exc)
        records.append(dict(key=row['key'],boxes=boxes,sift=sift,sift_failure=error,yolo_sec=yolo_sec,sift_sec=time.perf_counter()-t))
        save(EVIDENCE/'proposals.json',dict(records=records,crops=crops,weights_sha256=sha(ROOT/'yolov8n.pt'),class_names=model.names,source_sha256=sha(Path(__file__)),protocol_sha256=sha(ROOT/'experiments/FULL_IMAGE_MLP_PROTOCOL.md')))
        print(row['key'],len(boxes),'proposals',flush=True)


def visual():
    import torch
    from PIL import Image
    from transformers import AutoModel
    from transformers.models.bit.image_processing_pil_bit import BitImageProcessorPil
    from multimodal_data import DINO
    from mcd.projection.runtime import load_projector,project
    from moid.multimodal_metrics import normalize
    torch.set_num_threads(4)
    payload=json.loads((EVIDENCE/'proposals.json').read_text())
    rows=json.loads((EVIDENCE/'manifest.json').read_text())['frames']
    source=json.loads((SOURCE/'manifest.json').read_text())['crops']
    src_features=np.load(SOURCE/'features.npz')['visual']
    processor=BitImageProcessorPil.from_pretrained(str(DINO),local_files_only=True)
    model=AutoModel.from_pretrained(str(DINO),local_files_only=True).eval()
    x=[]
    t=time.perf_counter()
    for i in range(0,len(payload['crops']),8):
        images=[Image.open(ROOT/r['path']).convert('RGB') for r in payload['crops'][i:i+8]]
        with torch.inference_mode():
            features=model(**processor(images=images,return_tensors='pt')).last_hidden_state[:,0]
            x.extend(torch.nn.functional.normalize(features,dim=1).numpy())
    x=np.array(x)
    del model,processor
    np.savez(EVIDENCE/'visual_features.npz',visual=x)
    head=load_projector(ROOT/'experiments/local/mlp_v2/visual_mlp_s2027.json')
    z=normalize(project(head,x)); refs=normalize(project(head,src_features))
    selections=[]
    for row,p in zip(rows,payload['records']):
        assert row['key']==p['key']
        q=next(i for i,r in enumerate(source) if r['key']==row['reference_key'])
        indices=[b['crop_index'] for b in p['boxes']]
        distances={'visual_cos':np.clip(1-x[indices]@src_features[q],0,2).tolist(),'visual_mlp_s2027_cos':np.clip(1-z[indices]@refs[q],0,2).tolist()} if indices else {'visual_cos':[],'visual_mlp_s2027_cos':[]}
        selections.append(dict(key=row['key'],scores=distances,selected={name:p['boxes'][int(np.argmin(values))]['crop_index'] if values else None for name,values in distances.items()}))
    save(EVIDENCE/'selections.json',dict(records=selections,embedding_sec=time.perf_counter()-t))
    chosen=sorted({i for r in selections for i in r['selected'].values() if i is not None})
    save(EVIDENCE/'caption_manifest.json',dict(crops=[payload['crops'][i] for i in chosen]))
    print('Selected',len(chosen),'distinct proposal crops for independent VLM',flush=True)


def describe():
    import multimodal_data as shared
    # The common caption runner requires this filename; preserve full-frame manifest.
    caption_evidence=EVIDENCE/'captions'; caption_evidence.mkdir(exist_ok=True)
    save(caption_evidence/'manifest.json',json.loads((EVIDENCE/'caption_manifest.json').read_text()))
    shared.BASE=LOCAL; shared.EVIDENCE=caption_evidence
    shared.PROTOCOL=ROOT/'experiments/FULL_IMAGE_MLP_PROTOCOL.md'; shared.KEEP_ALIVE='5m'
    shared.FINALIZE_FAILURES=True
    shared.describe()


def text():
    import torch
    from sentence_transformers import SentenceTransformer
    from multimodal_data import SBERT
    from mcd.projection.runtime import load_projector,project
    from moid.multimodal_metrics import normalize
    torch.set_num_threads(4)
    records=json.loads((LOCAL/'captions.json').read_text())['records']
    records=[r for r in records if r.get('valid_description',True)]
    names=[r['key'] for r in records]
    model=SentenceTransformer(str(SBERT),device='cpu',local_files_only=True)
    t=time.perf_counter()
    x=model.encode([r['caption'] for r in records],normalize_embeddings=True,batch_size=32)
    del model
    head=load_projector(ROOT/'experiments/local/mlp_v2/text_mlp_s2027.json')
    z=normalize(project(head,x))
    np.savez(EVIDENCE/'text_features.npz',keys=np.array(names),text=x,projected=z)
    save(EVIDENCE/'text_timing.json',dict(sec=time.perf_counter()-t))


def evaluate():
    from moid.multimodal_metrics import normalize
    from moid.evaluation import iou
    frames=json.loads((EVIDENCE/'manifest.json').read_text())['frames']
    proposals=json.loads((EVIDENCE/'proposals.json').read_text())
    selections=json.loads((EVIDENCE/'selections.json').read_text())['records']
    source=json.loads((SOURCE/'manifest.json').read_text())['crops']
    ref_z=normalize(np.load(SOURCE/'projections.npz')['text_mlp_s2027'])
    ref_valid=np.load(SOURCE/'features.npz')['valid_text']
    text_features=np.load(EVIDENCE/'text_features.npz')
    text_by_name=dict(zip(text_features['keys'],text_features['projected']))
    inv=np.linalg.inv(np.array(json.loads((SOURCE/'covariance.json').read_text())['text_mlp_s2027']['full']))
    thresholds=json.loads((SOURCE/'frozen_thresholds.json').read_text())['thresholds']['1']
    output=[]
    for row,p,selected in zip(frames,proposals['records'],selections):
        q=next(i for i,r in enumerate(source) if r['key']==row['reference_key'])
        predictions={'SIFT':p['sift']}
        scores={}
        for method,index in selected['selected'].items():
            bbox=None; distance=None
            if index is not None:
                candidate=proposals['crops'][index]
                best=min(selected['scores'][method])
                if best<=thresholds[method]['threshold']:
                    bbox=candidate['bbox']
                z=text_by_name.get(candidate['key'])
                if z is not None and ref_valid[q]:
                    d=ref_z[q]-z
                    distance=float(np.sqrt(max(0,d@inv@d)))
            predictions[method]=bbox
            predictions[method+'+text']=bbox if distance is not None and distance<=thresholds['text_mlp_s2027_full']['threshold'] else None
            scores[method]=dict(visual=min(selected['scores'][method]) if index is not None else None,text=distance)
        for method,bbox in predictions.items():
            overlap=iou(bbox,row['bbox']) if bbox is not None else 0.
            tp=int(bbox is not None and overlap>=.5)
            fp=int(bbox is not None and not tp)
            wrong=int(fp and any(iou(bbox,b)>=.5 for b in row['distractors']))
            output.append(dict(key=row['key'],clip=row['clip'],split=row['split'],method=method,bbox=bbox,iou=overlap,tp=tp,fp=fp,fn=1-tp,wrong_instance=wrong,scores=scores))
    save(EVIDENCE/'predictions.json',output)
    summary=[]
    per_video=[]
    for split in ['validation','test']:
        for method in sorted({r['method'] for r in output}):
            selected=[r for r in output if r['method']==method and r['split']==split]
            for clip in sorted({r['clip'] for r in selected}):
                rs=[r for r in selected if r['clip']==clip]
                per_video.append(dict(split=split,method=method,clip=clip,**summarize(rs)))
            summary.append(dict(split=split,method=method,**summarize(selected)))
    save(EVIDENCE/'summary.json',summary); save(EVIDENCE/'per_video.json',per_video)
    print(json.dumps([r for r in summary if r['split']=='test'],indent=2),flush=True)


def summarize(rows):
    sums={key:sum(r[key] for r in rows) for key in ['tp','fp','fn','wrong_instance']}
    return dict(frames=len(rows),recall=sums['tp']/len(rows),precision=sums['tp']/(sums['tp']+sums['fp']) if sums['tp']+sums['fp'] else None,mean_iou=float(np.mean([r['iou'] for r in rows])),fp_per_frame=sums['fp']/len(rows),**sums)


if __name__=='__main__':
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['YOLO_CONFIG_DIR']=str(LOCAL/'ultralytics')
    os.environ['YOLO_AUTOINSTALL']='False'
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','proposals','visual','describe','text','evaluate'])
    globals()[parser.parse_args().stage]()
